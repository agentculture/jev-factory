"""The r3b failure diagnostic (t31, obligation o13): gated, config-driven, no defaults."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.core.predictions import read_predictions
from jev_factory.decide import r3b_replay, records
from tests.fixtures.toy_domain import DOMAIN

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64


def _line(i: int, gold: str, pick: str, p: float, mutating_args=None) -> dict:
    rest = (1 - p) / 2
    candidates = {pick: p, "(explain)": rest, "(escalate)": rest}
    return {
        "id": f"e{i}",
        "expected": {"operation": gold, "args": mutating_args or {}},
        "outcome": "propose",
        "operation": pick,
        "arguments": mutating_args or {},
        "candidates": candidates,
        "tokens": 1,
        "ttfd_ms": 1.0,
        "latency_ms": 1.0,
    }


@pytest.fixture
def frozen(tmp_path):
    """A synthetic validation run, domain and config; nothing real."""
    lines = [_line(i, "lamp_status", "lamp_status", 0.9) for i in range(6)]
    lines += [_line(6 + i, "list_rooms", "list_rooms", 0.8) for i in range(4)]
    lines += [
        _line(10, "lamp_status", "list_rooms", 0.6),
        _line(11, "list_rooms", "lamp_status", 0.55),
    ]
    tree = tmp_path / "operator-tree"
    tree.mkdir()
    (tree / "val-predictions.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    folds = {"seed": 1, "fit_ids": [f"e{i}" for i in range(0, 12, 2)]}
    folds["selection_ids"] = [f"e{i}" for i in range(1, 12, 2)]
    (tree / "folds.json").write_text(json.dumps(folds))
    (tree / "domain.json").write_text(json.dumps(DOMAIN.to_dict()))
    config = {
        "domain": "domain.json",
        "predictions": "val-predictions.jsonl",
        "folds": "folds.json",
    }
    (tree / "replay.json").write_text(json.dumps(config))
    return tree


def _store(tmp_path, evaluation=None, verdict="stop_escalate"):
    store = tmp_path / records.DEFAULT_NAME
    details = {"evaluation": evaluation} if evaluation is not None else None
    records.append(
        store,
        run="jev-tool-1",
        verdict=verdict,
        reasons=["evaluation bar missed"],
        cited=[{"name": "m", "value": 0.1, "source": "s.json", "sha256": SHA}],
        rule_version="jev-decide-rule/1",
        details=details,
    )
    return store


FAILED = {"domain": "jev-tool", "outcome": "failed"}


@pytest.mark.behavioral("o13")
def test_replay_refuses_without_a_record_id(tmp_path, frozen):
    store = _store(tmp_path, FAILED)
    with pytest.raises(CliError, match="needs a failed jev-tool evaluation record id"):
        r3b_replay.replay(store, "", frozen / "replay.json")


@pytest.mark.behavioral("o13")
def test_replay_refuses_an_unknown_record_id(tmp_path, frozen):
    store = _store(tmp_path, FAILED)
    with pytest.raises(CliError, match="no decision record 'D9'"):
        r3b_replay.replay(store, "D9", frozen / "replay.json")


@pytest.mark.behavioral("o13")
@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        {"domain": "jev-tool", "outcome": "passed"},
        {"domain": "other-model", "outcome": "failed"},
    ],
)
def test_replay_refuses_a_record_that_is_not_a_failed_jev_tool_evaluation(
    tmp_path, frozen, evaluation
):
    store = _store(tmp_path, evaluation)
    with pytest.raises(CliError, match="does not mark a failed jev-tool evaluation"):
        r3b_replay.replay(store, "D1", frozen / "replay.json")


@pytest.mark.behavioral("o13")
def test_replay_refuses_a_ship_verdict_even_if_marked_failed(tmp_path, frozen):
    store = _store(tmp_path, FAILED, verdict="ship_candidate")
    with pytest.raises(CliError, match="not a failure"):
        r3b_replay.replay(store, "D1", frozen / "replay.json")


@pytest.mark.behavioral("o13")
def test_the_refusal_comes_before_any_config_or_data_is_opened(tmp_path, monkeypatch):
    store = _store(tmp_path)  # no evaluation marker
    opened = []
    monkeypatch.setattr(r3b_replay, "load_config", lambda *a: opened.append(a))
    monkeypatch.setattr(r3b_replay, "read_predictions", lambda *a: opened.append(a))
    with pytest.raises(CliError):
        r3b_replay.replay(store, "D1", tmp_path / "missing-config.json")
    assert opened == []


@pytest.mark.behavioral("o13")
def test_replay_has_no_default_location_it_needs_operator_config(tmp_path):
    store = _store(tmp_path, FAILED)
    with pytest.raises(CliError, match="missing: domain, predictions, folds"):
        r3b_replay.replay(store, "D1", {})
    with pytest.raises(CliError, match="cannot read replay config"):
        r3b_replay.replay(store, "D1", tmp_path / "nope.json")


@pytest.mark.behavioral("o13")
def test_a_test_or_held_out_predictions_file_is_refused(tmp_path, frozen):
    store = _store(tmp_path, FAILED)
    (frozen / "val-predictions.jsonl").rename(frozen / "held-out-predictions.jsonl")
    cfg = json.loads((frozen / "replay.json").read_text())
    cfg["predictions"] = "held-out-predictions.jsonl"
    (frozen / "replay.json").write_text(json.dumps(cfg))
    with pytest.raises(CliError, match="held-out"):
        r3b_replay.replay(store, "D1", frozen / "replay.json")


def test_replay_runs_the_select_stages_on_a_failed_record(tmp_path, frozen, capsys):
    store = _store(tmp_path, FAILED)
    report = r3b_replay.replay(store, "D1", frozen / "replay.json")
    assert report["evaluation_record"]["id"] == "D1"
    assert report["calibration"]["selection_n"] == 6
    assert report["selection"]["wrong_mutating"] == 0  # the toy ops here are read-only
    assert report["selection"]["right_proposals"] >= 1
    assert "no reference numbers" in report["localisation"]
    assert (
        r3b_replay.main(
            ["--decisions", str(store), "--record", "D1", "--config", str(frozen / "replay.json")]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation_record"]["run"] == "jev-tool-1"


def test_replay_localises_against_reference_numbers(tmp_path, frozen):
    store = _store(tmp_path, FAILED)
    base = r3b_replay.replay(store, "D1", frozen / "replay.json")["selection"]
    cfg = json.loads((frozen / "replay.json").read_text())
    cfg["reference"] = dict(base)
    same = r3b_replay.replay(store, "D1", {**cfg, **_abs(frozen, cfg)})
    assert "jev-CLI data or model" in same["localisation"]
    cfg["reference"] = {**base, "ece": (base["ece"] or 0) + 0.2}
    off = r3b_replay.replay(store, "D1", {**cfg, **_abs(frozen, cfg)})
    assert "calibration" in off["localisation"]
    assert "pipeline suspect" in off["localisation"]


def _abs(tree: Path, cfg: dict) -> dict:
    return {k: str(tree / cfg[k]) for k in ("domain", "predictions", "folds")}


def test_main_reports_a_refusal_on_stderr_with_exit_1(tmp_path, frozen, capsys):
    store = _store(tmp_path)
    code = r3b_replay.main(
        ["--decisions", str(store), "--record", "D1", "--config", str(frozen / "replay.json")]
    )
    assert code == 1
    assert "failed jev-tool evaluation" in capsys.readouterr().err
    assert len(read_predictions(frozen / "val-predictions.jsonl")) == 12


@pytest.mark.behavioral("o13")
def test_no_default_config_or_stage_names_the_frozen_artifacts():
    """No module or shipped config other than the replay itself mentions r3b or #53 artifacts."""
    needles = ("r3b", "#53", "scorer-r3b")
    hits = []
    for path in sorted((ROOT / "jev_factory").rglob("*")):
        if path.suffix not in {".py", ".json", ".toml", ".yaml", ".sh", ".env"}:
            continue
        if path.name == "r3b_replay.py" or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            tree = ast.parse(text)
            skip = set()  # docstrings and NVSH_PROVENANCE headers are documentation
            for n in ast.walk(tree):
                if isinstance(n, ast.Expr):
                    skip.update(id(c) for c in ast.walk(n))
                elif isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "NVSH_PROVENANCE" for t in n.targets
                ):
                    skip.update(id(c) for c in ast.walk(n))
            strings = [
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
            ]
            text = "\n".join(strings)
        hits += [f"{path.relative_to(ROOT)}: {n}" for n in needles if n in text]
    assert hits == [], "\n".join(hits)
    example = (ROOT / "docs" / "run-config.example.toml").read_text(encoding="utf-8")
    assert "r3b" not in example
    assert "frozen" not in example


def test_replay_module_commits_no_path_or_host():
    source = (ROOT / "jev_factory" / "decide" / "r3b_replay.py").read_text(encoding="utf-8")
    assert "/home/" not in source
    assert "localhost" not in source
    assert "http" not in source


def test_the_doc_describes_the_gated_procedure():
    doc = (ROOT / "docs" / "r3b-diagnostic.md").read_text(encoding="utf-8")
    for phrase in (
        "failed jev-tool evaluation record id",
        "r3b_replay",
        "operator",
        "never committed",
    ):
        assert phrase in doc
