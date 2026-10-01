"""Serve one model for measurement: pinned vLLM for a model directory, llama-server for a GGUF.

Every measured model -- the stock copy, a merged fine-tune, an AWQ export and
the deployed ``Q4_K_M`` GGUF -- is served the same way, on ``127.0.0.1`` only,
and the measure stage attaches to it (:mod:`jev_factory.measure.run`)::

    python -m jev_factory.measure.serve start MODEL PORT [RECORD_JSON]
    python -m jev_factory.measure.serve wait PORT [FULL_LOG]
    python -m jev_factory.measure.serve stop PORT

**vLLM** (MODEL is a directory): ``start`` runs a detached container named
``jev-measure-PORT`` from the serving image named by ``@sha256:`` digest (a
tag is refused), with MODEL bind-mounted read-only at ``/model``, the port
published on 127.0.0.1 only, and ``--max-logprobs`` at least the readout's
:data:`~jev_factory.backbones.causal_lm.readout.READOUT_TOP`, the number of
next-token log-probabilities every served scoring request asks for (fewer and
the readout would lose letters). A model directory without a greedy
``generation_config.json`` (temperature 0, ``do_sample`` false) is refused.

**llama-server** (MODEL is a ``.gguf`` file, the deployed ``Q4_K_M`` build):
``start`` launches the native binary (``llama_server``; no container),
detached in its own session, bound to 127.0.0.1, with exactly::

    --model MODEL --host 127.0.0.1 --port PORT --ctx-size CTX --jinja
    --n-gpu-layers 999 --temp 0 --top-k 1 --alias NAME

A GGUF has no ``generation_config.json``, so greedy decoding is applied by
flags. The launched server is "the one this helper started" only while its
pid still has the recorded start time (``/proc/<pid>/stat`` field 22) and its
``/proc/<pid>/cmdline`` is exactly the recorded argv (or that argv behind the
interpreter a script binary runs under). Every signal is sent only after that
check. ``wait`` is ready only when ``/v1/models`` lists the alias, the LISTEN
socket on the port is held by that child (or a descendant), and the child is
still that child; on a timeout or a dead server it prints the last log lines,
stops the server and exits 2.

**One port, one operation at a time (nvsh#58).** In nvsh a ``stop`` racing a
``start`` on the same port could kill the *new* server: stop's wait loop
re-read the pid file, which the concurrent start had just rewritten. Here
``start`` and ``stop`` each hold an exclusive ``flock`` on
``<run dir>/jev-measure-PORT.lock`` for their whole duration, and ``stop``
signals only the process identity it read under that lock. A start that
arrives while a stop is in flight waits for the lock (up to ``lock_seconds``)
and then starts cleanly; it is never signalled by that stop.

Nothing here touches a container not named ``jev-measure-*`` or a process
this helper did not launch.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from jev_factory.backbones.causal_lm import gen_config
from jev_factory.backbones.causal_lm.readout import READOUT_TOP

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/serve_for_measure.sh",
    "commit": "9debdc6",
    "adaptations": [
        "ported from bash to Python; start/wait/stop and every guard kept: image by"
        " @sha256 digest, 127.0.0.1-only publishing, read-only /model mount, greedy"
        " generation_config check, llama-server flags (lines 31-37, 366-367), identity by"
        " pid + /proc start time + exact argv (lines 107-143), listener ownership through"
        " /proc/net/tcp{,6} and /proc/<pid>/fd (lines 145-179), rollback of a failed start"
        " (lines 318-342), wait's alias/listener/identity readiness (lines 437-475)",
        "nvsh#58 fixed: one flock per port held across the whole of start and of stop"
        " (nvsh held its mkdir lock only around the pid-file claim), and stop signals only"
        " the identity it read under that lock instead of re-reading the pid file",
        "the pid and argv files become one JSON state file per port; names are"
        " jev-measure-PORT instead of q46-measure-PORT",
        "gen_config.py check (scripts/lfm-finetune/gen_config.py:check) is delegated to"
        " backbones/causal_lm/gen_config.check (greedy_problem only adds refusals for an"
        " unreadable or non-object file); uv run from the nvsh repo root is gone",
        "vLLM's --tool-call-parser is passed only when configured (the scorer path reads"
        " /completions log-probabilities, never tool calls); --max-logprobs below"
        " READOUT_TOP is refused",
        "MEASURE_* environment knobs become ServeSettings (from the run config or JEV_*"
        " variables); the run dir defaults to XDG_RUNTIME_DIR, else the system temp dir",
    ],
    "licence": "Apache-2.0",
}

#: The only container, state, log and lock names this helper creates or removes.
NAME_PREFIX = "jev-measure-"
#: vLLM's port inside the container.
CONTAINER_PORT = 8000
#: Where MODEL is mounted inside the container.
MODEL_MOUNT = "/model"
#: No image or video inputs: the model is measured on text alone.
NO_MULTIMODAL = '{"image": 0, "video": 0}'

LLAMA_BACKEND = "llama-server"
VLLM_BACKEND = "vllm"

_IMAGE_RE = re.compile(r"^[A-Za-z0-9][^@\s]*@sha256:[0-9a-f]{64}$")

#: ``run(argv, timeout) -> (exit code, combined output)``, a fixed argv and never a shell.
RunFn = Callable[[list[str], float], "tuple[int, str]"]

#: Popen objects of servers this process launched, so they are reaped, never left zombies.
_CHILDREN: dict[int, subprocess.Popen] = {}


class ServeError(Exception):
    """A refused or failed start/wait/stop: one line, exit code 1 (input) or 2 (machine)."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def default_run(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run *argv* (a fixed list, no shell) and return ``(exit code, output)``."""
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv list, no shell=True
            argv, check=False, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return (127, f"{argv[0]}: not found")
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, f"{type(exc).__name__}: {exc}")
    return (completed.returncode, (completed.stdout or "") + (completed.stderr or ""))


def _default_run_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())


@dataclass(frozen=True)
class ServeSettings:
    """Every knob of one serving helper; identical for every model a run measures."""

    run_dir: Path = field(default_factory=_default_run_dir)
    ctx: int = 2048
    image: str | None = None
    tool_call_parser: str | None = None
    gpu_fraction: float = 0.08
    max_logprobs: int = READOUT_TOP
    gpu_args: tuple[str, ...] = ("--gpus", "all")
    #: llama-server ``--n-gpu-layers``; 0 serves a GGUF on the CPU only (``--device none``).
    gpu_layers: int = 999
    llama_server: str | None = None
    model_name: str | None = None
    start_seconds: float = 10.0
    wait_seconds: float = 900.0
    poll_seconds: float = 2.0
    stop_seconds: float = 10.0
    lock_seconds: float = 60.0
    log_lines: int = 40

    @classmethod
    def from_config(cls, cfg, run_dir: str | Path, **overrides) -> ServeSettings:
        """Settings from a resolved :class:`~jev_factory.factory.config.RunConfig`."""
        values = {
            "run_dir": Path(run_dir),
            "ctx": int(cfg.get("measure_ctx") or 2048),
            "image": cfg.get("measure_image"),
            "tool_call_parser": cfg.get("tool_call_parser"),
            "gpu_fraction": float(cfg.get("measure_gpu_fraction") or 0.08),
            "max_logprobs": int(cfg.get("measure_max_logprobs") or READOUT_TOP),
            "gpu_layers": int(
                999 if cfg.get("measure_gpu_layers") is None else cfg.get("measure_gpu_layers")
            ),
            "llama_server": cfg.get("llama_server"),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServeSettings:
        """Settings from ``JEV_*`` variables (the run config's names, plus serve-only knobs)."""
        env = os.environ if environ is None else environ

        def get(name: str, cast: Callable, default):
            raw = env.get(name, "")
            if raw == "":
                return default
            try:
                return cast(raw)
            except ValueError:
                raise ServeError(f"{name}={raw!r} is not a valid value") from None

        gpu_args = env.get("JEV_MEASURE_GPU_ARGS")
        return cls(
            run_dir=Path(env.get("JEV_MEASURE_RUN_DIR") or _default_run_dir()),
            ctx=get("JEV_MEASURE_CTX", int, 2048),
            image=env.get("JEV_MEASURE_IMAGE") or None,
            tool_call_parser=env.get("JEV_TOOL_CALL_PARSER") or None,
            gpu_fraction=get("JEV_MEASURE_GPU_FRACTION", float, 0.08),
            max_logprobs=get("JEV_MEASURE_MAX_LOGPROBS", int, READOUT_TOP),
            gpu_layers=get("JEV_MEASURE_GPU_LAYERS", int, 999),
            gpu_args=tuple(gpu_args.split()) if gpu_args is not None else ("--gpus", "all"),
            llama_server=env.get("JEV_LLAMA_SERVER") or None,
            model_name=env.get("JEV_MEASURE_MODEL_NAME") or None,
            start_seconds=get("JEV_MEASURE_START_SECONDS", float, 10.0),
            wait_seconds=get("JEV_MEASURE_WAIT_SECONDS", float, 900.0),
            poll_seconds=get("JEV_MEASURE_POLL_SECONDS", float, 2.0),
            stop_seconds=get("JEV_MEASURE_STOP_SECONDS", float, 10.0),
            lock_seconds=get("JEV_MEASURE_LOCK_SECONDS", float, 60.0),
            log_lines=get("JEV_MEASURE_LOG_LINES", int, 40),
        )


# ---------------------------------------------------------------------------
# Names and files
# ---------------------------------------------------------------------------


def check_port(port: object) -> int:
    try:
        value = int(str(port))
    except ValueError:
        value = -1
    if not 1024 <= value <= 65535 or not str(port).isdigit():
        raise ServeError(f"PORT must be a number from 1024 to 65535 (got {port!r})")
    return value


def container_name(port: int) -> str:
    return f"{NAME_PREFIX}{port}"


def state_file(settings: ServeSettings, port: int) -> Path:
    return Path(settings.run_dir) / f"{NAME_PREFIX}{port}.json"


def log_file(settings: ServeSettings, port: int) -> Path:
    return Path(settings.run_dir) / f"{NAME_PREFIX}{port}.log"


def lock_file(settings: ServeSettings, port: int) -> Path:
    return Path(settings.run_dir) / f"{NAME_PREFIX}{port}.lock"


def read_state(settings: ServeSettings, port: int) -> dict | None:
    """The recorded llama-server state for *port*, or ``None`` (a torn file counts as none)."""
    try:
        raw = json.loads(state_file(settings, port).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("argv"), list):
        return None
    return raw


def _write_state(settings: ServeSettings, port: int, state: Mapping[str, object]) -> None:
    path = state_file(settings, port)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _remove_state(settings: ServeSettings, port: int) -> None:
    state_file(settings, port).unlink(missing_ok=True)


@contextlib.contextmanager
def port_lock(settings: ServeSettings, port: int) -> Iterator[None]:
    """Hold *port*'s exclusive lock for a whole start or stop (the nvsh#58 fix)."""
    Path(settings.run_dir).mkdir(parents=True, exist_ok=True)
    path = lock_file(settings, port)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + max(settings.lock_seconds, 0.0)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ServeError(
                        f"another start or stop holds port {port}'s lock ({path}); wait for it"
                        " to finish, then retry",
                        code=2,
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


# ---------------------------------------------------------------------------
# Process identity (Linux /proc)
# ---------------------------------------------------------------------------


def proc_starttime(pid: int) -> str | None:
    """``/proc/PID/stat`` field 22 (start time in clock ticks); ``None`` if gone or a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = stat.rsplit(") ", 1)[-1].split()
    if not fields or fields[0] == "Z" or len(fields) < 20:
        return None
    return fields[19]


def is_ours(pid: int, starttime: str | None, argv: Sequence[str]) -> bool:
    """Whether *pid* is still the process launched with *argv* at *starttime*."""
    if not starttime or proc_starttime(pid) != starttime:
        return False
    expected = b"".join(arg.encode("utf-8") + b"\0" for arg in argv)
    try:
        actual = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return bool(expected) and (actual == expected or actual.endswith(b"\0" + expected))


def own_pid(state: Mapping[str, object] | None) -> int | None:
    """The recorded pid while it is still the process the state names, else ``None``."""
    if not state:
        return None
    pid, starttime, argv = state.get("pid"), state.get("starttime"), state.get("argv")
    if not isinstance(pid, int) or not isinstance(starttime, str) or not isinstance(argv, list):
        return None
    _reap(pid)
    return pid if is_ours(pid, starttime, [str(a) for a in argv]) else None


def _reap(pid: int) -> None:
    child = _CHILDREN.get(pid)
    if child is not None and child.poll() is not None:
        _CHILDREN.pop(pid, None)


def _signal(pid: int, sig: int) -> None:
    """Send *sig* to *pid* (the one place a signal leaves this module)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)


def _children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            ppid = int(entry.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(entry.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return children


def process_tree(pid: int) -> list[int]:
    """*pid* and all its descendants."""
    children = _children_map()
    seen, stack = [pid], [pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.append(child)
                stack.append(child)
    return seen


def listener_inodes(port: int) -> set[str]:
    """Socket inodes of every LISTEN socket (state 0A) on *port*, IPv4 and IPv6."""
    wanted = f":{port:04X}"
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            cols = line.split()
            if len(cols) > 9 and cols[3] == "0A" and cols[1].endswith(wanted):
                inodes.add(cols[9])
    return inodes


def owns_listener(port: int, pid: int) -> bool:
    """Whether a LISTEN socket on *port* is open in *pid* or one of its descendants."""
    inodes = listener_inodes(port)
    if not inodes:
        return False
    for member in process_tree(pid):
        try:
            fds = os.listdir(f"/proc/{member}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"/proc/{member}/fd/{fd}")
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                return True
    return False


def port_busy(port: int) -> bool:
    """Whether anything accepts a connection on 127.0.0.1:PORT."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_record(record: Path, payload: Mapping[str, object]) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def greedy_problem(model_dir: Path) -> str | None:
    """``None`` when *model_dir*'s generation_config.json decodes greedily, else why not."""
    model_dir = Path(model_dir)
    path = model_dir / gen_config.GEN_CONFIG_FILE
    if path.is_file():
        # gen_config.check reads the file bare; keep the structured refusals for bad files.
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"unreadable generation_config.json: {exc}"
        if not isinstance(payload, dict):
            return "generation_config.json is not an object"
        if isinstance(payload.get("temperature"), bool):
            return f"temperature is {payload.get('temperature')!r}, not 0"
    return gen_config.check(model_dir)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


def llama_argv(binary: str, model: Path, port: int, settings: ServeSettings, name: str) -> list:
    if settings.gpu_layers < 0:
        raise ServeError(f"gpu_layers must be 0 or more, not {settings.gpu_layers}")
    cpu_only = ["--device", "none"] if settings.gpu_layers == 0 else []
    return [
        binary,
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(settings.ctx),
        "--jinja",
        "--n-gpu-layers",
        str(settings.gpu_layers),
        *cpu_only,
        "--temp",
        "0",
        "--top-k",
        "1",
        "--alias",
        name,
    ]


def _stop_launched(pid: int, launched: str | None, settings: ServeSettings) -> None:
    """Stop the child a failed start just launched, each signal only while it is still it."""
    if launched is None or proc_starttime(pid) != launched:
        _reap(pid)
        return
    _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + min(settings.stop_seconds, 2.0)
    while time.monotonic() < deadline:
        _reap(pid)
        if proc_starttime(pid) != launched:
            return
        time.sleep(0.05)
    if proc_starttime(pid) == launched:
        _signal(pid, signal.SIGKILL)
    _reap(pid)
    print(f"serve: stopped the llama-server it had just launched (pid {pid})", file=sys.stderr)


def start_llama(model: Path, port: int, settings: ServeSettings, record: Path | None = None) -> int:
    """Launch llama-server for *model* on *port*; returns its pid (see the module docstring)."""
    binary = settings.llama_server
    if not binary:
        raise ServeError(
            "a .gguf model is served by a native llama-server; set llama_server"
            " (JEV_LLAMA_SERVER) to llama.cpp's build/bin/llama-server",
            code=2,
        )
    path = Path(binary)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ServeError(f"llama_server={binary} is not an executable file", code=2)
    binary_path = str(path.resolve())
    model = Path(model).resolve()
    code, version = default_run([binary_path, "--version"], 60.0)
    if code != 0:
        raise ServeError(f"'{binary_path} --version' failed: {version.strip()}", code=2)
    name = settings.model_name or model.stem
    Path(settings.run_dir).mkdir(parents=True, exist_ok=True)
    argv = llama_argv(binary_path, model, port, settings, name)
    with port_lock(settings, port):
        state = read_state(settings, port)
        if state is not None or state_file(settings, port).exists():
            if own_pid(state) is not None:
                raise ServeError(
                    f"a llama-server this helper started runs on port {port}; stop it first",
                    code=2,
                )
            _remove_state(settings, port)  # stale: its process is gone or no longer ours
        if port_busy(port):
            raise ServeError(
                f"something already listens on 127.0.0.1:{port}; it would answer /v1/models in"
                f" place of {model.name} -- stop it or pick another measure_port",
                code=2,
            )
        if record is not None:
            _write_record(
                record,
                {
                    "backend": LLAMA_BACKEND,
                    "binary": binary_path,
                    "version": version.strip(),
                    "model_file": str(model),
                    "served_model_name": name,
                    "port": port,
                    "decoding": "greedy by flags (--temp 0 --top-k 1): a GGUF carries no"
                    " generation_config.json",
                    "state_file": str(state_file(settings, port)),
                    "log": str(log_file(settings, port)),
                    "started": _now(),
                    "argv": argv,
                },
            )
        with open(log_file(settings, port), "wb") as log:
            child = subprocess.Popen(  # nosec B603 - fixed argv list, no shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        _CHILDREN[child.pid] = child
        launched = proc_starttime(child.pid)
        try:
            deadline = time.monotonic() + settings.start_seconds
            while True:
                now = proc_starttime(child.pid)
                if now is None or now != launched:
                    raise ServeError(
                        f"llama-server exited at once; see {log_file(settings, port)}", code=2
                    )
                if is_ours(child.pid, now, argv):
                    break
                if time.monotonic() >= deadline:
                    raise ServeError(
                        f"the launched process never ran {binary_path} with the recorded argv",
                        code=2,
                    )
                time.sleep(0.05)
            _write_state(
                settings,
                port,
                {
                    "backend": LLAMA_BACKEND,
                    "pid": child.pid,
                    "starttime": now,
                    "argv": argv,
                    "alias": name,
                },
            )
        except BaseException:
            _stop_launched(child.pid, launched, settings)
            _remove_state(settings, port)
            raise
    print(
        f"serve: started llama-server (pid {child.pid}) serving {model.name} as '{name}'"
        f" on 127.0.0.1:{port}",
        file=sys.stderr,
    )
    return child.pid


def check_vllm_settings(settings: ServeSettings) -> None:
    if not settings.image or not _IMAGE_RE.match(settings.image):
        raise ServeError(
            "measure_image must name the serving image by @sha256: digest, not a tag"
            f" (got {settings.image!r})"
        )
    if int(settings.max_logprobs) < READOUT_TOP:
        raise ServeError(
            f"measure_max_logprobs {settings.max_logprobs} is below the readout's READOUT_TOP"
            f" ({READOUT_TOP}); every scoring request asks for {READOUT_TOP} log-probabilities"
        )
    if not 0.0 < float(settings.gpu_fraction) <= 1.0:
        raise ServeError(f"measure_gpu_fraction must be in (0, 1] (got {settings.gpu_fraction})")
    if int(settings.ctx) <= 0:
        raise ServeError(f"measure_ctx must be a positive whole number (got {settings.ctx})")


def vllm_argv(model_dir: Path, port: int, settings: ServeSettings, name: str) -> list[str]:
    parser = (
        ["--enable-auto-tool-choice", "--tool-call-parser", settings.tool_call_parser]
        if settings.tool_call_parser
        else []
    )
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name(port),
        "-p",
        f"127.0.0.1:{port}:{CONTAINER_PORT}",
        *settings.gpu_args,
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        f"{model_dir}:{MODEL_MOUNT}:ro",
        str(settings.image),
        "--model",
        MODEL_MOUNT,
        "--served-model-name",
        name,
        "--max-model-len",
        str(settings.ctx),
        "--gpu-memory-utilization",
        str(settings.gpu_fraction),
        *parser,
        "--max-logprobs",
        str(settings.max_logprobs),
        "--limit-mm-per-prompt",
        NO_MULTIMODAL,
    ]


def start_vllm(
    model_dir: Path,
    port: int,
    settings: ServeSettings,
    record: Path | None = None,
    *,
    run: RunFn = default_run,
) -> str:
    """Start the pinned vLLM container for *model_dir* on *port*; returns its name."""
    check_vllm_settings(settings)
    if not Path(model_dir).is_dir():
        raise ServeError(f"MODEL_DIR {model_dir} is not a directory")
    model_dir = Path(model_dir).resolve()
    if ":" in str(model_dir) or "," in str(model_dir):
        raise ServeError(f"MODEL_DIR {model_dir} contains ':' or ',', which docker -v cannot carry")
    problem = greedy_problem(model_dir)
    if problem is not None:
        raise ServeError(f"{problem}: every measured model decodes greedily (temperature 0)")
    name = settings.model_name or model_dir.name
    own = container_name(port)
    with port_lock(settings, port):
        if own_pid(read_state(settings, port)) is not None:
            raise ServeError(f"a llama-server this helper started runs on port {port}", code=2)
        _remove_state(settings, port)
        code, output = run(
            ["docker", "ps", "-a", "--filter", f"name=^{own}$", "--format", "{{.Names}}"], 30.0
        )
        if code != 0:
            raise ServeError(f"cannot list containers: docker ps exited {code}", code=2)
        if own in output.split():
            raise ServeError(f"a container named {own} already exists; stop port {port} first")
        if port_busy(port):
            raise ServeError(f"something already listens on 127.0.0.1:{port}", code=2)
        argv = vllm_argv(model_dir, port, settings, name)
        if record is not None:
            _write_record(
                record,
                {
                    "backend": VLLM_BACKEND,
                    "container": own,
                    "image": settings.image,
                    "model_dir": str(model_dir),
                    "served_model_name": name,
                    "port": port,
                    "started": _now(),
                    "argv": argv,
                },
            )
        code, output = run(argv, 600.0)
        if code != 0:
            raise ServeError(f"docker run exited {code}: {output.strip()[-400:]}", code=2)
    print(
        f"serve: started {own} serving {model_dir.name} as '{name}' on 127.0.0.1:{port}",
        file=sys.stderr,
    )
    return own


def start(
    model: str | Path,
    port: int,
    settings: ServeSettings,
    record: Path | None = None,
    *,
    run: RunFn = default_run,
) -> str:
    """Start the right backend for *model*; returns ``"llama-server"`` or ``"vllm"``."""
    port = check_port(port)
    if int(settings.ctx) <= 0:
        raise ServeError(f"measure_ctx must be a positive whole number (got {settings.ctx})")
    if str(model).endswith(".gguf"):
        if not Path(model).is_file():
            raise ServeError(f"MODEL {model} is not a file")
        start_llama(Path(model), port, settings, record)
        return LLAMA_BACKEND
    start_vllm(Path(model), port, settings, record, run=run)
    return VLLM_BACKEND


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


def fetch_models(url: str, timeout: float = 5.0) -> dict | None:
    """``GET url`` (localhost) as JSON, or ``None`` when it does not answer 200 with JSON."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310 - localhost
            if response.status != 200:
                return None
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def lists_model(payload: Mapping[str, object] | None, name: str) -> bool:
    data = payload.get("data") if isinstance(payload, Mapping) else None
    return isinstance(data, list) and any(
        isinstance(item, dict) and item.get("id") == name for item in data
    )


def _tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _wait_llama(port: int, settings: ServeSettings, full_log: Path | None) -> None:
    own = f"llama-server on port {port}"
    url = f"http://127.0.0.1:{port}/v1/models"
    state = read_state(settings, port)
    alias = str(state.get("alias") or "") if state else ""
    deadline = time.monotonic() + settings.wait_seconds
    timed_out = False
    while True:
        pid = own_pid(read_state(settings, port))
        if pid is None:
            print(f"serve: {own} stopped before it was ready; its last log lines:", file=sys.stderr)
            break
        # Ready only when the answer lists this child's alias, the listening socket is the
        # child's, and the child is still the one started: anything else is not it.
        if (
            alias
            and lists_model(fetch_models(url), alias)
            and owns_listener(port, pid)
            and own_pid(read_state(settings, port)) == pid
        ):
            print(f"serve: {own} is ready", file=sys.stderr)
            return
        if time.monotonic() >= deadline:
            print(
                f"serve: {own} not ready after {settings.wait_seconds:g}s; its last log lines:",
                file=sys.stderr,
            )
            timed_out = True
            break
        time.sleep(settings.poll_seconds)
    print(_tail(log_file(settings, port), settings.log_lines), file=sys.stderr)
    if full_log is not None:
        with contextlib.suppress(OSError):
            shutil.copyfile(log_file(settings, port), full_log)
            print(f"serve: the full server log is in {full_log}", file=sys.stderr)
    stop(port, settings)  # never leave a server behind a failed wait
    if timed_out:
        print(f"serve: stopped {own}", file=sys.stderr)
    raise ServeError(f"{own} did not become ready", code=2)


def _wait_vllm(port: int, settings: ServeSettings, full_log: Path | None, run: RunFn) -> None:
    own = container_name(port)
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + settings.wait_seconds
    while True:
        if fetch_models(url) is not None:
            print(f"serve: {own} is ready", file=sys.stderr)
            return
        code, output = run(["docker", "inspect", "-f", "{{.State.Running}}", own], 30.0)
        if code == 0 and output.strip() == "false":
            print(f"serve: {own} stopped before it was ready; its last log lines:", file=sys.stderr)
            break
        if time.monotonic() >= deadline:
            print(
                f"serve: {own} not ready after {settings.wait_seconds:g}s; its last log lines:",
                file=sys.stderr,
            )
            break
        time.sleep(settings.poll_seconds)
    _code, tail = run(["docker", "logs", "--tail", str(settings.log_lines), own], 60.0)
    print(tail.rstrip(), file=sys.stderr)
    if full_log is not None:
        _code, whole = run(["docker", "logs", own], 120.0)
        with contextlib.suppress(OSError):
            Path(full_log).write_text(whole, encoding="utf-8")
            print(f"serve: the full server log is in {full_log}", file=sys.stderr)
    raise ServeError(f"{own} did not become ready", code=2)


def wait(
    port: int,
    settings: ServeSettings,
    full_log: Path | None = None,
    *,
    run: RunFn = default_run,
) -> None:
    """Block until the server on *port* is ready; :class:`ServeError` (exit 2) otherwise."""
    port = check_port(port)
    if state_file(settings, port).exists():
        _wait_llama(port, settings, full_log)
    else:
        _wait_vllm(port, settings, full_log, run)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def _stop_llama_locked(port: int, settings: ServeSettings) -> int | None:
    """Stop the llama-server recorded for *port*; the caller holds the port lock."""
    state = read_state(settings, port)
    pid = own_pid(state)
    if pid is not None and state is not None:
        # The identity is the one read here, under the lock: never re-read (nvsh#58).
        starttime, argv = str(state["starttime"]), [str(a) for a in state["argv"]]
        _signal(pid, signal.SIGTERM)
        deadline = time.monotonic() + settings.stop_seconds
        while time.monotonic() < deadline:
            _reap(pid)
            if not is_ours(pid, starttime, argv):
                break
            time.sleep(0.1)
        _reap(pid)
        if is_ours(pid, starttime, argv):
            _signal(pid, signal.SIGKILL)
            time.sleep(0.1)
            _reap(pid)
    _remove_state(settings, port)
    return pid


def stop(port: int, settings: ServeSettings, *, run: RunFn = default_run) -> int | None:
    """Stop what this helper started on *port*; returns the llama-server pid it stopped."""
    port = check_port(port)
    with port_lock(settings, port):
        pid = _stop_llama_locked(port, settings)
        if shutil.which("docker"):
            run(["docker", "rm", "-f", container_name(port)], 60.0)
    return pid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, settings: ServeSettings | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.measure.serve",
        description="Serve one model for measurement (vLLM for a directory, llama-server for"
        " a .gguf); settings come from JEV_* variables.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    start_p = sub.add_parser("start", help="start MODEL on PORT")
    start_p.add_argument("model")
    start_p.add_argument("port")
    start_p.add_argument("record", nargs="?", default=None)
    wait_p = sub.add_parser("wait", help="wait until PORT's server is ready")
    wait_p.add_argument("port")
    wait_p.add_argument("full_log", nargs="?", default=None)
    stop_p = sub.add_parser("stop", help="stop what this helper started on PORT")
    stop_p.add_argument("port")
    args = parser.parse_args(argv)
    try:
        resolved = settings or ServeSettings.from_env()
        port = check_port(args.port)
        if args.command == "start":
            record = Path(args.record) if args.record else None
            start(args.model, port, resolved, record)
        elif args.command == "wait":
            wait(port, resolved, Path(args.full_log) if args.full_log else None)
        else:
            stop(port, resolved)
    except ServeError as exc:
        print(f"serve: {exc.message}", file=sys.stderr)
        return exc.code
    return 0


def with_name(settings: ServeSettings, name: str | None) -> ServeSettings:
    """*settings* serving under *name* (``--served-model-name`` / ``--alias``)."""
    return replace(settings, model_name=name) if name else settings


if __name__ == "__main__":
    raise SystemExit(main())
