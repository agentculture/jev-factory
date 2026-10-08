"""Stage engine tests: manifests, staleness (o23) and detached jobs (o37)."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from jev_factory import __version__
from jev_factory.factory import detach, stages
from jev_factory.factory.stages import Registry, Stage


def _counts(progress):
    """The items done/total of a progress record (it also carries timing)."""
    return {k: progress[k] for k in ("done", "total")}


def _registry(calls: list[str]) -> Registry:
    def make(name: str, src: str, dst: str):
        def fn(workdir: Path, knobs: dict) -> None:
            calls.append(name)
            (workdir / dst).write_text((workdir / src).read_text() + name)

        return fn

    reg = Registry()
    reg.register(Stage("a", make("a", "seed.txt", "a.out"), ("seed.txt",), ("a.out",)))
    reg.register(Stage("b", make("b", "a.out", "b.out"), ("a.out",), ("b.out",), ("a",)))
    reg.register(Stage("c", make("c", "b.out", "c.out"), ("b.out",), ("c.out",), ("b",)))
    return reg


def _setup(tmp_path: Path):
    (tmp_path / "seed.txt").write_text("s")
    calls: list[str] = []
    return tmp_path, calls, _registry(calls)


def _run_all(wd, reg):
    return [stages.run_stage(wd, n, registry=reg) for n in reg.names()]


def test_manifest_fields(tmp_path):
    wd, _, reg = _setup(tmp_path)
    man = stages.run_stage(wd, "a", {"k": 1}, registry=reg)
    on_disk = json.loads((wd / "manifests" / "a.json").read_text())
    assert on_disk == man
    assert man["stage"] == "a" and man["jev_factory_version"] == __version__
    assert man["status"] == "complete" and man["rc"] == 0 and man["knobs"] == {"k": 1}
    assert len(man["inputs"]["seed.txt"]) == 64 and len(man["outputs"]["a.out"]) == 64
    assert man["started"] <= man["finished"]


@pytest.mark.behavioral("o23")
def test_rerun_unchanged_is_noop(tmp_path):
    wd, calls, reg = _setup(tmp_path)
    _run_all(wd, reg)
    assert calls == ["a", "b", "c"]
    again = _run_all(wd, reg)
    assert calls == ["a", "b", "c"]
    assert all(m["skipped"] for m in again)
    assert stages.stale_stages(wd, registry=reg) == []


@pytest.mark.behavioral("o23")
def test_input_change_marks_stage_and_downstream_stale(tmp_path):
    wd, calls, reg = _setup(tmp_path)
    _run_all(wd, reg)
    (wd / "seed.txt").write_text("changed")
    assert stages.stale_stages(wd, registry=reg) == ["a", "b", "c"]
    assert "input sha256" in stages.staleness(wd, "a", registry=reg)
    assert "upstream stage" in stages.staleness(wd, "c", registry=reg)
    _run_all(wd, reg)
    assert calls == ["a", "b", "c"] * 2


@pytest.mark.behavioral("o23")
def test_midchain_change_leaves_upstream_fresh(tmp_path):
    wd, _, reg = _setup(tmp_path)
    _run_all(wd, reg)
    (wd / "a.out").write_text("tampered")
    assert stages.stale_stages(wd, registry=reg) == ["a", "b", "c"]
    reg2 = _registry([])
    _run_all(wd, reg2)
    (wd / "b.out").write_text("x")
    assert stages.stale_stages(wd, registry=reg2) == ["b", "c"]


def test_knob_change_is_stale_and_failure_recorded(tmp_path):
    wd, _, reg = _setup(tmp_path)
    stages.run_stage(wd, "a", {"k": 1}, registry=reg)
    assert stages.staleness(wd, "a", {"a": {"k": 2}}, registry=reg) == "knobs changed"
    reg.register(Stage("d", lambda w, k: 1 / 0, (), (), ()))
    man = stages.run_stage(wd, "d", registry=reg)
    assert man["status"] == "failed" and man["rc"] == 1 and "ZeroDivisionError" in man["error"]
    assert stages.staleness(wd, "d", registry=reg) == "last run failed"


def test_a_stage_never_skips_or_builds_on_an_upstream_whose_last_run_failed(tmp_path):
    wd, calls, reg = _setup(tmp_path)
    _run_all(wd, reg)
    failing = Registry()
    for name in reg.names():
        stage = reg.get(name)
        if name == "a":

            def broken(workdir: Path, knobs: dict) -> None:
                raise RuntimeError("teacher gateway down")

            stage = Stage("a", broken, stage.inputs, stage.outputs, stage.deps)
        failing.register(stage)
    assert stages.run_stage(wd, "a", {"retry": 1}, registry=failing)["status"] == "failed"
    assert (wd / "a.out").is_file()  # the old output is still there
    for name in ("b", "c"):  # fresh by their own inputs, but their producer failed
        assert stages.staleness(wd, name, registry=failing) is not None
        man = stages.run_stage(wd, name, registry=failing)
        assert not man.get("skipped") and man["status"] == "failed"
        assert "upstream stage a's last run failed" in man["error"]
    assert calls == ["a", "b", "c"]  # nothing ran on the failed producer's leftovers


def test_missing_input_fails_without_running(tmp_path):
    wd, calls, reg = _setup(tmp_path)
    man = stages.run_stage(wd, "b", registry=reg)
    assert man["status"] == "failed" and man["rc"] == 2 and calls == []


def test_registry_order_downstream_cycles_and_listing(tmp_path):
    _, _, reg = _setup(tmp_path)
    assert reg.downstream("a") == ["b", "c"] and reg.downstream("c") == []
    assert [s["stage"] for s in stages.list_stages(reg)] == ["a", "b", "c"]
    cyc = Registry()
    cyc.register(Stage("x", lambda w, k: None, deps=("y",)))
    cyc.register(Stage("y", lambda w, k: None, deps=("x",)))
    with pytest.raises(ValueError):
        cyc.names()
    already = reg.get("a")
    with pytest.raises(ValueError):
        reg.register(already)
    with pytest.raises(KeyError):
        reg.get("zz")


def test_directory_hash_tracks_content(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "f").write_text("1")
    h1 = stages.sha256_path(d)
    (d / "f").write_text("2")
    assert stages.sha256_path(d) != h1 and stages.sha256_path(tmp_path / "nope") is None


def _wait(jobdir, name, states, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        st = detach.job_status(jobdir, name)
        if st["state"] in states:
            return st
        time.sleep(0.05)
    raise AssertionError(f"timeout, last={st}")


def test_detached_records_real_rc_and_pid(tmp_path):
    pid = detach.start_detached(tmp_path, "ok", [sys.executable, "-c", "raise SystemExit(0)"])
    st = _wait(tmp_path, "ok", {"done"})
    assert st["rc"] == 0 and st["pid"] == pid
    detach.start_detached(tmp_path, "bad", [sys.executable, "-c", "raise SystemExit(7)"])
    assert _wait(tmp_path, "bad", {"done"})["rc"] == 7
    assert detach.job_status(tmp_path, "none")["state"] == "absent"


def test_detached_runs_in_new_session(tmp_path):
    script = "import os,sys;open(sys.argv[1],'w').write(str(os.getsid(0)==os.getpid()))"
    out = tmp_path / "sid"
    detach.start_detached(tmp_path, "s", [sys.executable, "-c", script, str(out)])
    _wait(tmp_path, "s", {"done"})
    assert out.read_text() == "False" or os.getsid(0) != int(
        detach.job_status(tmp_path, "s")["pid"]
    )


def test_killed_job_has_no_done_marker(tmp_path):

    pid = detach.start_detached(tmp_path, "k", [sys.executable, "-c", "import time;time.sleep(60)"])
    assert detach.job_status(tmp_path, "k")["state"] == "running"
    os.killpg(pid, signal.SIGKILL)
    st = _wait(tmp_path, "k", {"died", "done"})
    assert st["state"] == "died" and st["rc"] is None


@pytest.mark.behavioral("o37")
def test_killed_item_run_resumes_with_zero_resends_and_progress(tmp_path):
    items = [f"i{n}" for n in range(6)]
    sent: list[str] = []
    killed_once = []

    class Killed(BaseException):
        pass

    def fn(item):
        if item == "i3" and not killed_once:
            killed_once.append(1)
            raise Killed  # simulated kill mid-item
        sent.append(item)
        return item.upper()

    led = detach.ItemLedger(tmp_path, "draft", total=len(items))
    with pytest.raises(Killed):
        led.run(items, fn)
    assert _counts(detach.read_progress(tmp_path, "draft")) == {"done": 3, "total": 6}
    # torn trailing journal line from the kill must not break resume
    with (tmp_path / "draft.items.jsonl").open("a") as fh:
        fh.write('{"id": "i3", "res')
    sent.clear()
    led2 = detach.ItemLedger(tmp_path, "draft", total=len(items))
    led2.run(items, fn)
    assert sent == ["i3", "i4", "i5"]  # 0 re-sends of i0..i2, resumes at next item
    assert _counts(detach.read_progress(tmp_path, "draft")) == {"done": 6, "total": 6}
    assert _counts(detach.job_status(tmp_path, "draft")["progress"]) == {"done": 6, "total": 6}
