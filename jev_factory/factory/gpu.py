"""GPU residency guard and memory-floor watchdog for heavy stages.

Two protections around a GPU stage (train, heal, quantize, measure):

* **Residency guard.** One GPU serves one job at a time. Before a GPU stage
  starts, ``nvidia-smi --query-compute-apps`` is read; a compute process that
  was not started by this run refuses the stage, unless ``allow_foreign`` (the
  CLI's explicit override flag) is set.
* **Memory-floor watchdog.** ``capped.sh`` (vendored next to this module) runs
  a command under a hard RAM+swap cap *and* stops the command's whole process
  group when the machine's ``MemAvailable`` falls below ``TRAIN_MEMORY_FLOOR``.
  On GB10 the GPU and CPU share one physical pool and GPU allocations are not
  charged to the cgroup cap, so the floor is the only backstop for a GPU
  trainer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/capped.sh",
    "commit": "9debdc6",
    "adaptations": [
        "capped.sh is vendored as jev_factory/factory/capped.sh and run through bash; d17 changed"
        " only shell style ([[ ]] tests, explicit returns, named locals, a default case)",
        "run_capped, the residency guard and the stage wrapper are new Python seams",
        "tests ported from tests/test_lfm_finetune_pipeline.py (run_capped and watchdog tests)",
    ],
    "licence": "Apache-2.0",
}

#: Stages that occupy the GPU; each refuses to start next to a foreign process.
GPU_STAGES = ("train", "heal", "quantize", "measure")

#: run_capped's exit status when the watchdog stopped the command.
WATCHDOG_STATUS = 3

CAPPED_SH = Path(__file__).with_name("capped.sh")


class GpuBusyError(RuntimeError):
    """A compute process this run did not start holds the GPU."""

    def __init__(self, stage: str, foreign: Sequence[GpuProcess]) -> None:
        self.stage = stage
        self.foreign = list(foreign)
        listing = ", ".join(f"pid {p.pid} ({p.name})" for p in self.foreign)
        super().__init__(
            f"refusing to start GPU stage {stage!r}: compute process(es) not started by this "
            f"run are on the GPU: {listing}. Stop them, or pass the override flag "
            "(allow_foreign / --allow-foreign-gpu) to run anyway."
        )


class GpuQueryError(RuntimeError):
    """nvidia-smi is present but its compute-app query failed."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str


def _children_map() -> dict[int, list[int]]:
    """pid -> child pids, read from /proc (empty where /proc is unavailable)."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = entry.read_text(encoding="utf-8")
            pid = int(entry.parent.name)
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def own_pids(extra: Iterable[int] = ()) -> set[int]:
    """This process, its descendants, and any explicitly registered pids."""
    children = _children_map()
    seen = {os.getpid(), *extra}
    stack = list(seen)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def query_compute_apps(env: Mapping[str, str] | None = None) -> list[GpuProcess]:
    """Compute processes on the GPU, or ``[]`` when there is no nvidia-smi."""
    environ = os.environ if env is None else env
    exe = shutil.which("nvidia-smi", path=environ.get("PATH"))
    if exe is None:
        return []
    proc = subprocess.run(  # nosec B603 - fixed argv, resolved executable
        [exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=dict(environ),
    )
    if proc.returncode != 0:
        raise GpuQueryError(
            f"nvidia-smi --query-compute-apps exited {proc.returncode}: {proc.stderr.strip()}"
        )
    apps: list[GpuProcess] = []
    for line in proc.stdout.splitlines():
        pid_text, _, name = line.partition(",")
        pid_text = pid_text.strip()
        if pid_text.isdigit():
            apps.append(GpuProcess(int(pid_text), name.strip()))
    return apps


def foreign_gpu_processes(
    *, own: Iterable[int] = (), env: Mapping[str, str] | None = None
) -> list[GpuProcess]:
    """Compute processes on the GPU that this run did not start."""
    mine = own_pids(own)
    return [p for p in query_compute_apps(env) if p.pid not in mine]


def assert_gpu_free(
    stage: str,
    *,
    allow_foreign: bool = False,
    own: Iterable[int] = (),
    env: Mapping[str, str] | None = None,
) -> list[GpuProcess]:
    """Refuse to start GPU stage *stage* next to a foreign compute process.

    Returns the foreign processes that were tolerated (only ever non-empty when
    *allow_foreign* is set). Raises :class:`GpuBusyError` otherwise.
    """
    foreign = foreign_gpu_processes(own=own, env=env)
    if foreign and not allow_foreign:
        raise GpuBusyError(stage, foreign)
    return foreign


def run_capped(
    run_dir: Path | str,
    cmd: Sequence[str],
    *,
    memory_max: str,
    memory_floor: str = "8G",
    watchdog_seconds: int = 5,
    container_cap: bool = False,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run *cmd* under capped.sh's ``run_capped``; returns its CompletedProcess.

    ``returncode`` is the command's status, :data:`WATCHDOG_STATUS` when the
    memory-floor watchdog stopped it, or 2 when the settings were refused.
    """
    environ = dict(os.environ if env is None else env)
    environ["TRAIN_MEMORY_MAX"] = memory_max
    environ["TRAIN_MEMORY_FLOOR"] = memory_floor
    environ["TRAIN_WATCHDOG_SECONDS"] = str(watchdog_seconds)
    if container_cap:
        environ["TRAIN_MEMORY_CAP"] = "container"
    return subprocess.run(  # nosec B603 B607 - fixed script, caller-supplied argv
        ["bash", "-c", 'source "$0"; run_capped "$@"', str(CAPPED_SH), str(run_dir), *cmd],
        capture_output=True,
        text=True,
        env=environ,
        timeout=timeout,
        check=False,
    )


def run_gpu_stage(
    stage: str,
    run_dir: Path | str,
    cmd: Sequence[str],
    *,
    allow_foreign: bool = False,
    **capped: object,
) -> subprocess.CompletedProcess:
    """Guard, then run a GPU stage under the memory cap and floor watchdog."""
    if stage not in GPU_STAGES:
        raise ValueError(f"{stage!r} is not a GPU stage; expected one of {GPU_STAGES}")
    assert_gpu_free(stage, allow_foreign=allow_foreign)
    return run_capped(run_dir, cmd, **capped)  # type: ignore[arg-type]
