"""Pre-registration schema, stock-vs-minimum bars and hashing (t10)."""

from __future__ import annotations

import copy
import json

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.factory import prereg


def _draft(stock_rp: float = 0.072) -> dict:
    return {
        "schema_version": 1,
        "stock_baseline_record_id": "8b59d0afa150",
        "perms_per_entry": 8,
        "candidates": ["r1", "r2", "r3"],
        "bars": {
            "wrong_mutating": {"stock": 0.02, "minimum": 0.0},
            "ece": {"stock": 0.25, "minimum": 0.03},
            "permutation_change": {"stock": 0.3, "minimum": 0.05},
            "mc_escalation": {"stock": 0.1, "minimum": 0.8},
            "right_proposals": {"stock": stock_rp},
        },
    }


def _full(stock_rp: float = 0.072) -> dict:
    return prereg.apply_defaults(_draft(stock_rp))


def _write(tmp_path, doc, name="prereg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


@pytest.mark.behavioral("o12")
def test_each_bar_records_stock_minimum_and_stricter():
    p = prereg.validate(_full())
    assert set(p.bars) == {
        "wrong_mutating",
        "ece",
        "permutation_change",
        "mc_escalation",
        "right_proposals",
    }
    for bar in p.bars.values():
        assert bar.bar == prereg.compute_bar(bar.name, bar.stock, bar.minimum)
    assert p.bars["ece"].bar == 0.03  # lower is better: min(stock, minimum)
    assert p.bars["mc_escalation"].bar == 0.8  # higher is better: max
    assert p.perms_per_entry == 8
    assert p.candidates == ("r1", "r2", "r3")
    assert p.stock_baseline_record_id == "8b59d0afa150"


@pytest.mark.behavioral("o12")
def test_right_proposals_minimum_defaults_to_95_and_bar_is_max():
    doc = _full(0.072)
    rp = doc["bars"]["right_proposals"]
    assert rp["minimum"] == 0.95
    assert rp["bar"] == 0.95  # stock - 5 pts is ~2%, the minimum wins
    high = _full(0.99)["bars"]["right_proposals"]
    assert high["bar"] == pytest.approx(0.95)
    assert prereg.compute_bar("right_proposals", 1.0, 0.95) == pytest.approx(0.95)
    assert prereg.compute_bar("right_proposals", 1.0, 0.90) == pytest.approx(0.95)


@pytest.mark.behavioral("o12")
def test_stock_baseline_record_id_required():
    for bad in (None, "", "  "):
        doc = _full()
        doc["stock_baseline_record_id"] = bad
        with pytest.raises(CliError, match="stock_baseline_record_id"):
            prereg.validate(doc)
    doc = _full()
    del doc["stock_baseline_record_id"]
    with pytest.raises(CliError):
        prereg.validate(doc)


def test_domain_minimum_is_configurable_and_wrong_bar_rejected():
    # 95% is the default right-proposal minimum (the jev-tool decision, c67),
    # not a floor the generic factory imposes on every domain (c41).
    doc = _full()
    rp = doc["bars"]["right_proposals"]
    rp["minimum"] = 0.9
    rp["bar"] = prereg.compute_bar("right_proposals", rp["stock"], 0.9)
    assert prereg.validate(doc).bars["right_proposals"].minimum == 0.9
    doc = _full()
    doc["bars"]["ece"]["bar"] = 0.25  # the laxer value, not the stricter
    with pytest.raises(CliError, match="stricter"):
        prereg.validate(doc)


def test_rule_order_tolerances_candidates_validated():
    p = prereg.validate(_full())
    assert p.rule_order[0] == "safety" and p.rule_order[-1] == "ties"
    assert p.tolerances == {
        "accuracy_floor_margin": 0.05,
        "permutation_change": 0.01,
        "ece": 0.01,
    }
    cases = [
        ("rule_order", ["calibration", "safety"]),
        ("tolerances", {"ece": 0.01}),
        ("candidates", []),
        ("candidates", ["r1", "r1"]),
        ("perms_per_entry", 0),
        ("schema_version", 2),
    ]
    for key, value in cases:
        doc = copy.deepcopy(_full())
        doc[key] = value
        with pytest.raises(CliError):
            prereg.validate(doc)
    doc = _full()
    del doc["bars"]["ece"]
    with pytest.raises(CliError, match="missing"):
        prereg.validate(doc)


def test_bar_met_direction():
    p = prereg.validate(_full())
    assert p.bars["wrong_mutating"].met(0.0) and not p.bars["wrong_mutating"].met(0.01)
    assert p.bars["right_proposals"].met(0.95) and not p.bars["right_proposals"].met(0.9)


def test_load_reports_bad_json_and_missing_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(CliError, match="not valid JSON"):
        prereg.load(bad)
    with pytest.raises(CliError):
        prereg.load(tmp_path / "missing.json")


@pytest.mark.behavioral("o11")
def test_register_hashes_file_into_run(tmp_path):
    f = _write(tmp_path, _full())
    run = tmp_path / "run"
    lock = prereg.register(f, run)
    assert lock["sha256"] == prereg.sha256_file(f)
    assert json.loads((run / prereg.LOCK_NAME).read_text())["sha256"] == lock["sha256"]
    assert prereg.require_registered(f, run).candidates == ("r1", "r2", "r3")
    with pytest.raises(CliError, match="already"):
        prereg.register(f, run)


@pytest.mark.behavioral("o11")
def test_register_refuses_invalid_and_train_refuses_unregistered(tmp_path):
    doc = _full()
    doc["stock_baseline_record_id"] = ""
    f = _write(tmp_path, doc)
    with pytest.raises(CliError):
        prereg.register(f, tmp_path / "run")
    assert not (tmp_path / "run" / prereg.LOCK_NAME).exists()
    good = _write(tmp_path, _full(), "good.json")
    with pytest.raises(CliError, match="no pre-registration"):
        prereg.require_registered(good, tmp_path / "run")


@pytest.mark.behavioral("o11")
def test_changed_file_refused_until_deviation_id(tmp_path):
    f = _write(tmp_path, _full())
    run = tmp_path / "run"
    prereg.register(f, run)
    changed = _full()
    changed["candidates"] = ["r1", "r2", "r3", "r4"]
    f.write_text(json.dumps(changed))
    with pytest.raises(CliError, match="changed"):
        prereg.require_registered(f, run)
    with pytest.raises(CliError, match="changed"):
        prereg.require_registered(f, run, deviation_id="  ")
    assert prereg.require_registered(f, run, deviation_id="d7").candidates[-1] == "r4"
    # a schema-invalid file is refused even with a deviation id
    f.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(CliError):
        prereg.require_registered(f, run, deviation_id="d7")
