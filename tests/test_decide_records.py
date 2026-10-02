"""Append-only decision records for jev decide (t15, obligation o21)."""

from __future__ import annotations

import hashlib
import json

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.decide import records

SHA = "a" * 64


def _cited(value: float = 0.0, name: str = "r1.wrong_mutating") -> list[dict]:
    return [{"name": name, "value": value, "source": "select/r1.json", "sha256": SHA}]


def _append(store, run="run-1", verdict="ship_candidate", **kw):
    kw.setdefault("reasons", ["r1 passes every rule step"])
    kw.setdefault("cited", _cited())
    kw.setdefault("rule_version", "jev-decide-rule/1")
    kw.setdefault("decider", "rule")
    return records.append(store, run=run, verdict=verdict, **kw)


@pytest.mark.behavioral("o21")
def test_append_writes_schema_valid_record_with_required_fields(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    rec = _append(store, params={"candidate": "r1"}, prereg_sha256=SHA)
    assert rec["id"] == "D1"
    for key in ("id", "run", "verdict", "cited", "rule_version", "decider", "reasons"):
        assert key in rec
    assert rec["decider"] == "rule"
    assert rec["cited"][0]["sha256"] == SHA
    assert records.validate_record(rec) == []
    on_disk = records.load(store)
    assert on_disk == [rec]


@pytest.mark.behavioral("o21")
def test_second_append_leaves_earlier_records_byte_identical(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    _append(store)
    before = store.read_bytes()
    rec2 = _append(store, verdict="recalibrate", reasons=["ECE 0.12 > 0.10"])
    after = store.read_bytes()
    assert after.startswith(before)
    assert rec2["id"] == "D2"
    assert rec2["prev_sha256"] == hashlib.sha256(before).hexdigest()
    assert [r["id"] for r in records.load(store)] == ["D1", "D2"]


@pytest.mark.behavioral("o21")
def test_human_override_is_a_new_record_and_never_an_edit(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    first = _append(store)
    before = store.read_bytes()
    over = records.override(
        store,
        overrides=first["id"],
        verdict="stop_escalate",
        reasons=["operator wants a second look"],
        deviation_id="v3",
        decided_by="operator",
    )
    assert store.read_bytes().startswith(before)
    assert over["id"] == "D2"
    assert over["decider"] == "human"
    assert over["overrides"] == "D1"
    assert over["deviation_id"] == "v3"
    assert over["run"] == first["run"]
    assert records.load(store)[0] == first


def test_override_requires_existing_target_and_deviation(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    _append(store)
    with pytest.raises(CliError, match="no record"):
        records.override(
            store, overrides="D9", verdict="stop_escalate", reasons=["x"], deviation_id="v1"
        )
    with pytest.raises(CliError, match="deviation"):
        records.override(
            store, overrides="D1", verdict="stop_escalate", reasons=["x"], deviation_id=" "
        )


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda r: r.pop("run"), "run"),
        (lambda r: r.update(verdict="ship it"), "verdict"),
        (lambda r: r.update(decider="oracle"), "decider"),
        (lambda r: r.update(cited=[]), "cited"),
        (lambda r: r["cited"][0].update(sha256="abc"), "sha256"),
        (lambda r: r["cited"][0].update(value="high"), "value"),
        (lambda r: r.update(reasons=[]), "reasons"),
        (lambda r: r.update(rule_version=""), "rule_version"),
        (lambda r: r.update(extra=1), "unknown"),
        (lambda r: r.update(decider="model"), "decider_ref"),
        (lambda r: r.update(decider="human"), "decider_ref"),
        (lambda r: r.update(overrides="D1"), "deviation_id"),
        (lambda r: r.update(prereg_sha256="xyz"), "prereg_sha256"),
    ],
)
def test_schema_rejects_bad_records(tmp_path, mutate, fragment):
    store = tmp_path / records.DEFAULT_NAME
    rec = _append(store)
    bad = json.loads(json.dumps(rec))
    mutate(bad)
    errs = records.validate_record(bad)
    assert errs and any(fragment in e for e in errs), errs


def test_append_refuses_invalid_record_and_writes_nothing(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    with pytest.raises(CliError, match="verdict"):
        _append(store, verdict="retrain everything")
    assert not store.exists()


def test_load_detects_a_rewritten_earlier_record(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    _append(store)
    _append(store, verdict="recalibrate")
    lines = store.read_text().splitlines()
    first = json.loads(lines[0])
    first["reasons"] = ["edited after the fact"]
    lines[0] = json.dumps(first, sort_keys=True)
    store.write_text("\n".join(lines) + "\n")
    with pytest.raises(CliError, match="chain"):
        records.load(store)


def test_model_decider_needs_a_ref(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    rec = _append(store, decider="model", decider_ref="decider-v1")
    assert rec["decider_ref"] == "decider-v1"


def test_cite_hashes_the_source_file(tmp_path):
    src = tmp_path / "metrics.json"
    src.write_text('{"ece": 0.02}')
    c = records.cite("r1.ece", 0.02, src)
    assert c == {
        "name": "r1.ece",
        "value": 0.02,
        "source": str(src),
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }
