"""The measure serving helper: jev_factory/measure/serve.py.

Stub ``llama-server`` and ``docker`` binaries (tests/fixtures/stub_servers.py)
stand in for the real ones: no container, GPU or model is ever started. The
stub llama-server is a real process listening on a real localhost port, so
identity checks (/proc start time and argv), listener ownership and the
per-port lock are exercised for real.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm.readout import READOUT_TOP
from jev_factory.measure import serve
from tests.fixtures.stub_servers import free_port, path_with, stub_bin, write_stub

IMAGE = "registry.example/vllm-openai@sha256:" + "a" * 64


@pytest.fixture
def stubs(tmp_path, monkeypatch) -> Path:
    directory = stub_bin(tmp_path / "bin")
    monkeypatch.setenv("PATH", path_with(directory))
    monkeypatch.setenv("STUB_DOCKER_LOG", str(tmp_path / "docker.log"))
    return directory


@pytest.fixture
def settings(tmp_path, stubs) -> serve.ServeSettings:
    return serve.ServeSettings(
        run_dir=tmp_path / "run",
        llama_server=str(stubs / "llama-server"),
        image=IMAGE,
        wait_seconds=20,
        poll_seconds=0.05,
        stop_seconds=5,
        lock_seconds=20,
    )


@pytest.fixture
def gguf(tmp_path) -> Path:
    path = tmp_path / "models" / "tool-jev.Q4_K_M.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF stub")
    return path


@pytest.fixture
def cleanup(settings):
    ports: list[int] = []
    yield ports
    for port in ports:
        serve.stop(port, replace(settings, lock_seconds=5))


def _docker_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "docker.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _alive(pid: int) -> bool:
    return serve.proc_starttime(pid) is not None


# -- llama-server: start, wait, stop -------------------------------------------


def test_llama_server_starts_with_the_exact_flags_and_stops(settings, gguf, tmp_path, cleanup):
    port = free_port()
    cleanup.append(port)
    record = tmp_path / "record.json"
    assert serve.start(gguf, port, settings, record) == serve.LLAMA_BACKEND
    state = serve.read_state(settings, port)
    assert state["argv"][1:] == [
        "--model",
        str(gguf.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--jinja",
        "--n-gpu-layers",
        "999",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--alias",
        "tool-jev.Q4_K_M",
    ]
    saved = json.loads(record.read_text())
    assert saved["backend"] == "llama-server"
    assert "9999 (stub llama-server)" in saved["version"]
    serve.wait(port, settings)
    pid = state["pid"]
    assert serve.owns_listener(port, pid)
    assert serve.stop(port, settings) == pid
    assert not _alive(pid)
    assert not serve.state_file(settings, port).exists()


def test_a_second_start_on_a_running_port_is_refused(settings, gguf, cleanup):
    port = free_port()
    cleanup.append(port)
    serve.start(gguf, port, settings)
    first = serve.read_state(settings, port)
    with pytest.raises(serve.ServeError, match="runs on port"):
        serve.start(gguf, port, settings)
    assert serve.read_state(settings, port) == first
    assert _alive(first["pid"])


def test_a_stale_state_file_is_replaced(settings, gguf, cleanup):
    port = free_port()
    cleanup.append(port)
    settings.run_dir.mkdir(parents=True)
    serve.state_file(settings, port).write_text(
        json.dumps({"pid": 999999, "starttime": "1", "argv": ["gone"], "alias": "x"})
    )
    serve.start(gguf, port, settings)
    assert serve.read_state(settings, port)["pid"] != 999999


def test_a_busy_port_is_refused(settings, gguf, cleanup):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        with pytest.raises(serve.ServeError, match="already listens"):
            serve.start(gguf, port, settings)
    assert not serve.state_file(settings, port).exists()


def test_a_server_that_exits_at_once_is_reported_and_leaves_no_claim(settings, gguf, monkeypatch):
    monkeypatch.setenv("STUB_LLAMA_FAIL", "1")
    port = free_port()
    with pytest.raises(serve.ServeError):
        serve.start(gguf, port, replace(settings, start_seconds=2))
        serve.wait(port, replace(settings, wait_seconds=2))
    assert not serve.state_file(settings, port).exists()


def test_wait_on_a_server_that_never_answers_stops_it(settings, gguf, monkeypatch, tmp_path):
    monkeypatch.setenv("STUB_LLAMA_DELAY", "30")
    port = free_port()
    serve.start(gguf, port, settings)
    pid = serve.read_state(settings, port)["pid"]
    full = tmp_path / "full.log"
    with pytest.raises(serve.ServeError) as excinfo:
        serve.wait(port, replace(settings, wait_seconds=0.5), full)
    assert excinfo.value.code == 2
    assert not _alive(pid)
    assert not serve.state_file(settings, port).exists()


def test_a_reused_pid_is_never_signalled(settings, gguf, monkeypatch, cleanup):
    """A recorded pid whose argv no longer matches is not ours: stop only removes the file."""
    port = free_port()
    cleanup.append(port)
    serve.start(gguf, port, settings)
    state = serve.read_state(settings, port)
    forged = {**state, "argv": [*state["argv"][:-1], "someone-else"]}
    serve.state_file(settings, port).write_text(json.dumps(forged))
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(serve, "_signal", lambda pid, sig: sent.append((pid, sig)))
    serve.stop(port, settings)
    assert sent == []
    assert _alive(state["pid"])
    serve.state_file(settings, port).write_text(json.dumps(state))  # let cleanup stop it


# -- nvsh#58: a stop racing a start on the same port never kills the new server --


def test_a_start_racing_a_stop_waits_for_the_lock_and_is_never_signalled(
    settings, gguf, monkeypatch, cleanup
):
    port = free_port()
    cleanup.append(port)
    serve.start(gguf, port, settings)
    serve.wait(port, settings)
    old = serve.read_state(settings, port)["pid"]

    sent: list[tuple[int, int]] = []
    real_signal = serve._signal
    started: dict[str, object] = {}
    racer: list[threading.Thread] = []

    def racing_start() -> None:
        try:
            serve.start(gguf, port, settings)
            started["pid"] = serve.read_state(settings, port)["pid"]
        except serve.ServeError as exc:  # pragma: no cover - would be the bug
            started["error"] = exc
        started["at"] = time.monotonic()

    def signal_and_race(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        real_signal(pid, sig)
        if not racer:
            # A start arrives while this stop is mid-flight (its old server is dying).
            thread = threading.Thread(target=racing_start)
            racer.append(thread)
            thread.start()
            time.sleep(0.5)

    monkeypatch.setattr(serve, "_signal", signal_and_race)
    serve.stop(port, settings)
    stopped_at = time.monotonic()
    racer[0].join(timeout=30)
    assert "error" not in started
    new = started["pid"]
    assert new != old
    assert started["at"] > stopped_at  # the start waited for the stop's lock
    assert {pid for pid, _sig in sent} == {old}  # the new server was never signalled
    assert _alive(new)
    assert serve.read_state(settings, port)["pid"] == new
    monkeypatch.setattr(serve, "_signal", real_signal)
    serve.wait(port, settings)


def test_a_start_that_cannot_get_the_lock_fails_cleanly(settings, gguf):
    port = free_port()
    with serve.port_lock(settings, port):
        with pytest.raises(serve.ServeError, match="lock"):
            serve.start(gguf, port, replace(settings, lock_seconds=0.2))
    assert not serve.state_file(settings, port).exists()


def test_stop_without_anything_started_is_a_no_op(settings, tmp_path):
    port = free_port()
    assert serve.stop(port, settings) is None
    assert ["rm", "-f", f"jev-measure-{port}"] in _docker_calls(tmp_path)


# -- vLLM by digest --------------------------------------------------------------


def _model_dir(tmp_path: Path, temperature=0, do_sample=False) -> Path:
    model = tmp_path / "merged"
    model.mkdir()
    (model / "generation_config.json").write_text(
        json.dumps({"temperature": temperature, "do_sample": do_sample})
    )
    return model


def test_vllm_is_started_by_digest_on_localhost_with_the_readout_logprobs(settings, tmp_path):
    model = _model_dir(tmp_path)
    port = free_port()
    record = tmp_path / "record.json"
    assert serve.start(model, port, settings, record) == serve.VLLM_BACKEND
    run = next(call for call in _docker_calls(tmp_path) if call[:1] == ["run"])
    assert IMAGE in run
    assert run[run.index("-p") + 1] == f"127.0.0.1:{port}:8000"
    assert f"{model.resolve()}:/model:ro" in run
    assert run[run.index("--max-logprobs") + 1] == str(READOUT_TOP)
    assert run[run.index("--max-model-len") + 1] == "2048"
    assert "--tool-call-parser" not in run
    assert json.loads(record.read_text())["backend"] == "vllm"


@pytest.mark.parametrize(
    "change, match",
    [
        ({"image": "vllm/vllm-openai:latest"}, "digest"),
        ({"max_logprobs": READOUT_TOP - 1}, "READOUT_TOP"),
        ({"gpu_fraction": 1.5}, "gpu_fraction"),
    ],
)
def test_vllm_settings_are_refused_before_docker_runs(settings, tmp_path, change, match):
    model = _model_dir(tmp_path)
    with pytest.raises(serve.ServeError, match=match):
        serve.start(model, free_port(), replace(settings, **change))
    assert not any(call[:1] == ["run"] for call in _docker_calls(tmp_path))


def test_vllm_refuses_a_model_that_does_not_decode_greedily(settings, tmp_path):
    model = _model_dir(tmp_path, temperature=0.7)
    with pytest.raises(serve.ServeError, match="temperature"):
        serve.start(model, free_port(), settings)


def test_vllm_refuses_an_existing_container(settings, tmp_path, monkeypatch):
    port = free_port()
    monkeypatch.setenv("STUB_DOCKER_EXISTING", f"jev-measure-{port}")
    with pytest.raises(serve.ServeError, match="already exists"):
        serve.start(_model_dir(tmp_path), port, settings)


def test_vllm_wait_reports_a_stopped_container(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_DOCKER_RUNNING", "false")
    full = tmp_path / "vllm.log"
    with pytest.raises(serve.ServeError) as excinfo:
        serve.wait(free_port(), settings, full)
    assert excinfo.value.code == 2
    assert "stub vllm log line" in full.read_text()


# -- settings and CLI ------------------------------------------------------------


def test_settings_come_from_the_run_config():
    from jev_factory.factory.config import load_config

    cfg = load_config(
        cli={
            "work": "/w",
            "base": "b",
            "base_rev": "r",
            "hub_prefix": "h",
            "licence": "Apache-2.0",
            "measure_ctx": 4096,
            "llama_server": "/opt/llama-server",
        },
        environ={},
    )
    got = serve.ServeSettings.from_config(cfg, "/runs/x")
    assert (got.ctx, got.llama_server, got.max_logprobs) == (4096, "/opt/llama-server", 20000)


def test_settings_from_the_environment(tmp_path):
    env = {
        "JEV_MEASURE_RUN_DIR": str(tmp_path),
        "JEV_MEASURE_CTX": "4096",
        "JEV_MEASURE_GPU_ARGS": "--runtime nvidia",
        "JEV_LLAMA_SERVER": "/opt/llama-server",
    }
    got = serve.ServeSettings.from_env(env)
    assert got.run_dir == tmp_path
    assert got.ctx == 4096
    assert got.gpu_args == ("--runtime", "nvidia")
    with pytest.raises(serve.ServeError):
        serve.ServeSettings.from_env({"JEV_MEASURE_CTX": "big"})


@pytest.mark.parametrize("port", ["80", "70000", "x1"])
def test_cli_refuses_a_bad_port(port, capsys):
    assert serve.main(["stop", port]) == 1
    assert "PORT" in capsys.readouterr().err


def test_cli_start_wait_stop_round_trip(settings, gguf, cleanup):
    port = free_port()
    cleanup.append(port)
    assert serve.main(["start", str(gguf), str(port)], settings=settings) == 0
    assert serve.main(["wait", str(port)], settings=settings) == 0
    pid = serve.read_state(settings, port)["pid"]
    assert serve.main(["stop", str(port)], settings=settings) == 0
    assert not _alive(pid)


def test_a_missing_llama_server_is_an_environment_error(settings, gguf, capsys):
    got = serve.main(
        ["start", str(gguf), str(free_port())], settings=replace(settings, llama_server=None)
    )
    assert got == 2
    assert "llama_server" in capsys.readouterr().err


def test_a_stub_process_dies_on_sigterm(tmp_path):
    """Sanity: a stub process dies on SIGTERM (the stop path relies on it)."""
    script = write_stub(tmp_path, "sleeper", "import time\ntime.sleep(30)\n")
    import subprocess

    child = subprocess.Popen([str(script)])
    try:
        os.kill(child.pid, signal.SIGTERM)
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()


def test_gpu_layers_zero_serves_the_gguf_on_the_cpu_only(tmp_path):
    gguf = Path("model-q4_k_m.gguf")
    default = serve.llama_argv("llama-server", gguf, 18061, serve.ServeSettings(), "m")
    assert default[default.index("--n-gpu-layers") + 1] == "999" and "--device" not in default
    cpu = serve.ServeSettings.from_env({"JEV_MEASURE_GPU_LAYERS": "0"})
    argv = serve.llama_argv("llama-server", gguf, 18061, cpu, "m")
    assert argv[argv.index("--n-gpu-layers") + 1] == "0"
    assert argv[argv.index("--device") + 1] == "none"
    assert serve.ServeSettings.from_config({"measure_gpu_layers": 0}, tmp_path).gpu_layers == 0
    assert serve.ServeSettings.from_config({}, tmp_path).gpu_layers == 999
    with pytest.raises(serve.ServeError):
        serve.llama_argv("llama-server", gguf, 1, serve.ServeSettings(gpu_layers=-1), "m")
