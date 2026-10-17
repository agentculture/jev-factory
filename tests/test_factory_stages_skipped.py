"""Declared skips: a ``skipped`` manifest satisfies dependents and shows in status."""

from __future__ import annotations

from pathlib import Path

import pytest

from jev_factory.cli._commands import status
from jev_factory.factory import stages
from jev_factory.factory.stages import Registry, Stage


def _registry(calls: list[str]) -> Registry:
    def fn(workdir: Path, knobs: dict) -> None:
        calls.append("b")
        (workdir / "b.out").write_text("b")

    reg = Registry()
    reg.register(Stage("a", lambda w, k: None, (), (), ()))
    reg.register(Stage("b", fn, (), ("b.out",), ("a",)))
    return reg


def test_skipped_satisfies_dependents(tmp_path):
    calls: list[str] = []
    reg = _registry(calls)
    man = stages.write_skipped(tmp_path, "a", "not needed here", {"x": 1})
    assert man["status"] == "skipped" and man["reason"] == "not needed here"
    assert man["inputs"] == {} and man["outputs"] == {}
    assert stages.staleness(tmp_path, "a", {"a": {"x": 1}}, registry=reg) is None
    out = stages.run_stage(tmp_path, "b", registry=reg)
    assert out["status"] == stages.COMPLETE and calls == ["b"]
    assert stages.stale_stages(tmp_path, {"a": {"x": 1}}, registry=reg) == []


def test_skipped_goes_stale_when_declaration_changes(tmp_path):
    reg = _registry([])
    stages.write_skipped(tmp_path, "a", "why", {"x": 1})
    assert stages.staleness(tmp_path, "a", {"a": {"x": 2}}, registry=reg) == "knobs changed"


def test_skipped_needs_a_reason(tmp_path):
    with pytest.raises(ValueError):
        stages.write_skipped(tmp_path, "a", " ", {})


def test_status_prints_skipped_with_reason(tmp_path):
    reg = _registry([])
    stages.write_skipped(tmp_path, "a", "teacher unavailable", {})
    rows = status._stage_entries(tmp_path, reg, lambda n: [])
    assert rows[0]["status"] == "skipped" and rows[0]["reason"] == "teacher unavailable"
    line = status._stage_line(rows[0])
    assert "skipped" in line and "teacher unavailable" in line


def test_jev_run_stages_never_write_skipped():
    import inspect

    from jev_factory.factory import pipeline

    assert "write_skipped" not in inspect.getsource(pipeline)
    assert "SKIPPED" not in inspect.getsource(pipeline)
