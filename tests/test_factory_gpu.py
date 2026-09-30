"""GPU residency guard (o38) and the capped.sh memory-floor watchdog (ported from nvsh)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from jev_factory.factory import gpu
from jev_factory.factory.gpu import GPU_STAGES, GpuBusyError, GpuProcess

_CAPPED = gpu.CAPPED_SH


# --------------------------------------------------------------------------
# Residency guard, with a stubbed nvidia-smi
# --------------------------------------------------------------------------


def _stub_smi(tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "smi-bin"
    bindir.mkdir()
    exe = bindir / "nvidia-smi"
    exe.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def _foreign_pid() -> int:
    """A pid that is certainly not this process or a descendant of it."""
    return 1  # init: never a descendant of the test process


@pytest.mark.behavioral("o38")
@pytest.mark.parametrize("stage", GPU_STAGES)
def test_each_gpu_stage_refuses_a_foreign_compute_process(stage, tmp_path, monkeypatch):
    _stub_smi(tmp_path, f'echo "{_foreign_pid()}, /usr/bin/other-trainer"', monkeypatch)
    with pytest.raises(GpuBusyError) as err:
        gpu.run_gpu_stage(stage, tmp_path / "run", ["true"], memory_max="1G")
    assert err.value.stage == stage
    assert err.value.foreign == [GpuProcess(_foreign_pid(), "/usr/bin/other-trainer")]
    assert "override" in str(err.value)
    assert not (tmp_path / "run").exists(), "the stage must not have started"


@pytest.mark.behavioral("o38")
@pytest.mark.parametrize("stage", GPU_STAGES)
def test_the_override_flag_lets_a_gpu_stage_start(stage, tmp_path, monkeypatch):
    _stub_smi(tmp_path, f'echo "{_foreign_pid()}, other"', monkeypatch)
    tolerated = gpu.assert_gpu_free(stage, allow_foreign=True)
    assert [p.pid for p in tolerated] == [_foreign_pid()]


def test_override_runs_the_stage_through_the_capped_runner(tmp_path, monkeypatch):
    _stub_smi(tmp_path, f'echo "{_foreign_pid()}, other"', monkeypatch)
    result = gpu.run_gpu_stage(
        "train",
        tmp_path / "run",
        ["bash", "-c", "exit 7"],
        allow_foreign=True,
        memory_max="200M",
        memory_floor="1K",
        watchdog_seconds=1,
        container_cap=True,
    )
    # No systemd-run on some hosts -> container cap mode; with it, 7 is the stage status.
    assert result.returncode in (7, 2)


def test_own_and_descendant_processes_are_not_foreign(tmp_path, monkeypatch):
    child = subprocess.Popen(["sleep", "30"])
    try:
        _stub_smi(tmp_path, f'echo "{os.getpid()}, me"; echo "{child.pid}, kid"', monkeypatch)
        assert gpu.foreign_gpu_processes() == []
        assert gpu.assert_gpu_free("train") == []
    finally:
        child.kill()
        child.wait()


def test_a_registered_pid_counts_as_this_runs(tmp_path, monkeypatch):
    _stub_smi(tmp_path, 'echo "1, earlier-stage"', monkeypatch)
    assert gpu.assert_gpu_free("measure", own=[1]) == []


def test_an_empty_gpu_passes(tmp_path, monkeypatch):
    _stub_smi(tmp_path, "true", monkeypatch)
    assert gpu.assert_gpu_free("quantize") == []


def test_no_nvidia_smi_means_nothing_to_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert gpu.query_compute_apps() == []


def test_a_failing_nvidia_smi_fails_closed(tmp_path, monkeypatch):
    _stub_smi(tmp_path, "echo boom >&2; exit 9", monkeypatch)
    with pytest.raises(gpu.GpuQueryError):
        gpu.assert_gpu_free("train")


def test_a_non_gpu_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        gpu.run_gpu_stage("split", tmp_path, ["true"], memory_max="1G")


def test_provenance_header():
    assert gpu.NVSH_PROVENANCE["commit"] == "9debdc6"
    assert gpu.NVSH_PROVENANCE["licence"] == "Apache-2.0"
    assert _CAPPED.is_file()


# --------------------------------------------------------------------------
# capped.sh tests ported from nvsh tests/test_lfm_finetune_pipeline.py
# --------------------------------------------------------------------------

_HOG = "b = bytearray(600 * 1024 * 1024)\nfor i in range(0, len(b), 4096):\n    b[i] = 1\n"
_HIDE_SYSTEMD_RUN = ' command() { [ "$2" = systemd-run ] && return 1; builtin command "$@"; };'


def _user_scope_works() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True, timeout=30
    )
    return probe.returncode == 0


needs_scope = pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")


def _mem_available_kb() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise AssertionError("no MemAvailable in /proc/meminfo")


def _run_watched(tmp_path: Path, floor: str, command: str, *, prefix: str = "") -> tuple:
    script = (
        f'source "{_CAPPED}";{prefix} TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR={floor}'
        f' TRAIN_WATCHDOG_SECONDS=1 run_capped "{tmp_path}" {command}'
    )
    started = time.monotonic()
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return result, time.monotonic() - started


@needs_scope
def test_run_capped_kills_a_process_over_the_cap_even_with_swap(tmp_path):
    hog = tmp_path / "hog.py"
    hog.write_text(_HOG, encoding="utf-8")
    script = f'source "{_CAPPED}"; TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" python3 "{hog}"'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert (tmp_path / "mem.log").read_text(encoding="utf-8").strip()


@needs_scope
def test_run_capped_lets_a_process_under_the_cap_finish(tmp_path):
    script = f'source "{_CAPPED}"; TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" true'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0


def test_run_capped_refuses_to_run_uncapped_without_systemd_run(tmp_path):
    script = (
        f'source "{_CAPPED}";{_HIDE_SYSTEMD_RUN} TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" true'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "TRAIN_MEMORY_CAP=container" in result.stderr


@needs_scope
def test_the_watchdog_stops_a_run_when_available_memory_falls_below_the_floor(tmp_path):
    floor = f"{_mem_available_kb() * 2}K"
    result, seconds = _run_watched(tmp_path, floor, "sleep 60")
    assert result.returncode == 3, result.stderr
    assert seconds < 20
    mem_log = (tmp_path / "mem.log").read_text(encoding="utf-8")
    assert "watchdog: MemAvailable" in mem_log
    assert f"below floor {floor}, stopping" in mem_log
    assert "watchdog: MemAvailable" in result.stderr


@needs_scope
def test_the_watchdog_stops_the_commands_children_too(tmp_path):
    floor = f"{_mem_available_kb() * 2}K"
    result, seconds = _run_watched(tmp_path, floor, "bash -c 'sleep 60 & wait'")
    assert result.returncode == 3, result.stderr
    assert seconds < 20


@needs_scope
def test_a_run_above_the_floor_finishes_normally(tmp_path):
    result, _ = _run_watched(tmp_path, "1K", "true")
    assert result.returncode == 0, result.stderr
    assert "watchdog" not in (tmp_path / "mem.log").read_text(encoding="utf-8")


def test_the_watchdog_also_guards_a_container_capped_run(tmp_path):
    floor = f"{_mem_available_kb() * 2}K"
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, seconds = _run_watched(tmp_path, floor, "sleep 60", prefix=prefix)
    assert result.returncode == 3, result.stderr
    assert seconds < 20
    assert "watchdog: MemAvailable" in (tmp_path / "mem.log").read_text(encoding="utf-8")


def test_a_container_capped_run_above_the_floor_finishes_normally(tmp_path):
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, _ = _run_watched(tmp_path, "1K", "true", prefix=prefix)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("floor", ["lots", "8X", "-1G", "0", "1.5G"])
def test_an_unreadable_memory_floor_is_refused(tmp_path, floor):
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, _ = _run_watched(tmp_path, floor, "true", prefix=prefix)
    assert result.returncode == 2
    assert "TRAIN_MEMORY_FLOOR" in result.stderr


def _alive(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return text.rsplit(")", 1)[1].split()[0] != "Z"


def _start_capped_sleep(tmp_path: Path, prefix: str):
    pid_file = tmp_path / "sleep.pid"
    script = (
        f'source "{_CAPPED}";{prefix} TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K'
        f' TRAIN_WATCHDOG_SECONDS=1 run_capped "{tmp_path}/run"'
        f" bash -c 'echo $$ > \"{pid_file}\"; exec sleep 60'"
    )
    shell = subprocess.Popen(
        ["bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        text = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
        if text:
            return shell, int(text)
        time.sleep(0.1)
    shell.kill()
    raise AssertionError("the capped command never started")


def _assert_sigterm_stops_the_command(tmp_path: Path, prefix: str) -> None:
    shell, sleep_pid = _start_capped_sleep(tmp_path, prefix)
    try:
        time.sleep(0.5)
        shell.terminate()
        deadline = time.monotonic() + 15
        while _alive(sleep_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(sleep_pid), "the capped command survived SIGTERM to run_capped"
        assert shell.wait(timeout=15) != 0
    finally:
        if _alive(sleep_pid):
            os.kill(sleep_pid, 9)
        if shell.poll() is None:
            shell.kill()


def test_sigterm_to_run_capped_stops_a_container_capped_command(tmp_path):
    _assert_sigterm_stops_the_command(tmp_path, _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container")


@needs_scope
def test_sigterm_to_run_capped_stops_a_scope_capped_command(tmp_path):
    _assert_sigterm_stops_the_command(tmp_path, "")


def test_run_capped_restores_the_callers_traps(tmp_path):
    script = (
        f'source "{_CAPPED}";{_HIDE_SYSTEMD_RUN} TRAIN_MEMORY_CAP=container;'
        " trap 'echo caller-term' TERM; trap 'echo caller-exit' EXIT;"
        f' TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K run_capped "{tmp_path}" true;'
        " trap -p TERM INT HUP"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "trap -- 'echo caller-term' SIGTERM" in result.stdout
    assert "SIGINT" not in result.stdout
    assert "SIGHUP" not in result.stdout
    assert result.stdout.rstrip().endswith("caller-exit")


def test_run_capped_still_returns_the_commands_status(tmp_path):
    script = (
        f'source "{_CAPPED}";{_HIDE_SYSTEMD_RUN} TRAIN_MEMORY_CAP=container;'
        f' TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K run_capped "{tmp_path}"'
        " bash -c 'echo out; exit 7'"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 7, result.stderr
    assert "out" in result.stdout
    assert (tmp_path / "train.log").read_text(encoding="utf-8").strip() == "out"


# Python wrapper over the same script


def test_python_run_capped_watchdog_kills_stage_and_children(tmp_path):
    floor = f"{_mem_available_kb() * 2}K"
    started = time.monotonic()
    result = gpu.run_capped(
        tmp_path,
        ["bash", "-c", "sleep 60 & wait"],
        memory_max="200M",
        memory_floor=floor,
        watchdog_seconds=1,
        container_cap=not _user_scope_works(),
        timeout=60,
    )
    assert result.returncode == gpu.WATCHDOG_STATUS, result.stderr
    assert time.monotonic() - started < 20
    assert "watchdog: MemAvailable" in (tmp_path / "mem.log").read_text(encoding="utf-8")
