"""t15: teacher audit against human gold, per source; assemble refuses without it."""

from __future__ import annotations

import json
import math

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import audit, prereg
from jev_factory.distill.records import Gold, Question, Record


def _record(rid, source, prov="human", gold="fix"):
    qs = {
        "a": Question("choice", (("fix", "f"), ("wait", "w"))),
        "b": Question("noul", (("false", ""), ("true", ""))),
    }
    g = {"a": Gold(label=gold), "b": Gold(label="true", probability_true=1.0)}
    return Record(rid, "s", qs, g, "train", source, "MIT", prov)


def _rows(rid, fix_p, p_true=0.9, build="clef-b1"):
    base = {"record_id": rid, "teacher_error": None, "teacher_build_id": build}
    return [
        {
            **base,
            "question_id": "a",
            "kind": "choice",
            "distribution": {"fix": fix_p, "wait": 1 - fix_p},
            "p_true": None,
        },
        {**base, "question_id": "b", "kind": "noul", "distribution": None, "p_true": p_true},
    ]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _prereg():
    return prereg.validate(prereg.example_doc())


def _fixture(tmp_path):
    recs = [
        _record("g1", "good"),
        _record("g2", "good"),
        _record("w1", "weak"),
        _record("w2", "weak"),
        _record("t1", "typed", prov="model-generated"),
        _record("t2", "typed", prov="model-generated"),
    ]
    rows = []
    for rid in ("g1", "g2"):
        rows += _rows(rid, 0.9)  # right on a, right on b
    for rid in ("w1", "w2"):
        rows += _rows(rid, 0.1, p_true=0.1)  # wrong on both
    for rid in ("t1", "t2"):
        rows += _rows(rid, 0.1, p_true=0.1)  # wrong, but gold is model-generated
    labels = tmp_path / "teacher_labels.jsonl"
    _write(labels, rows)
    return recs, labels


def _by_source(table):
    return {r["source"]: r for r in table["sources"]}


def test_per_source_metrics_and_weak_flag(tmp_path):
    recs, labels = _fixture(tmp_path)
    table = audit.build_table(recs, audit.load_labels(labels), floor=0.6)
    s = _by_source(table)
    assert s["good"]["n"] == 4
    assert s["good"]["accuracy"] == 1.0
    assert s["good"]["teacher_weak"] is False
    assert s["weak"]["accuracy"] == 0.0
    assert s["weak"]["teacher_weak"] is True
    assert math.isclose(s["good"]["log_loss"], -math.log(0.9), rel_tol=1e-9)
    assert math.isclose(s["good"]["ece"], 0.1, abs_tol=1e-9)
    assert table["floor"] == 0.6


def test_model_generated_gold_reported_never_flagged(tmp_path):
    recs, labels = _fixture(tmp_path)
    table = audit.build_table(recs, audit.load_labels(labels), floor=0.6)
    t = _by_source(table)["typed"]
    assert t["n"] == 4
    assert t["accuracy"] == 0.0  # reported
    assert t["gold_provenance"] == "model-generated"
    assert t["teacher_weak"] is None
    assert t["flaggable"] is False
    assert table["teacher_weak_sources"] == ["weak"]


def test_mixed_provenance_in_one_source_is_split(tmp_path):
    recs = [_record("h", "mix"), _record("m", "mix", prov="model-generated")]
    labels = tmp_path / "l.jsonl"
    _write(labels, _rows("h", 0.9) + _rows("m", 0.9))
    table = audit.build_table(recs, audit.load_labels(labels), floor=0.6)
    rows = sorted(
        (r for r in table["sources"] if r["source"] == "mix"),
        key=lambda r: r["gold_provenance"],
    )
    assert [r["gold_provenance"] for r in rows] == ["human", "model-generated"]
    assert [r["flaggable"] for r in rows] == [True, False]


def test_teacher_errors_and_unlabelled_are_counted_not_scored(tmp_path):
    recs = [_record("e", "s1"), _record("u", "s1")]
    rows = _rows("e", 0.9)
    rows[1] = {**rows[1], "teacher_error": "bad", "p_true": None}
    labels = tmp_path / "l.jsonl"
    _write(labels, rows)
    table = audit.build_table(recs, audit.load_labels(labels), floor=0.6)
    s = _by_source(table)["s1"]
    assert s["n"] == 1
    assert s["teacher_errors"] == 1
    assert s["unlabelled"] == 2


def test_source_with_no_scored_items_is_not_flagged(tmp_path):
    labels = tmp_path / "l.jsonl"
    _write(labels, [])
    table = audit.build_table([_record("x", "s")], audit.load_labels(labels), floor=0.6)
    s = _by_source(table)["s"]
    assert s["n"] == 0
    assert s["accuracy"] is None
    assert s["teacher_weak"] is None


def test_floor_must_be_preregistered(tmp_path):
    recs, labels = _fixture(tmp_path)
    reg = _prereg()
    with pytest.raises(CliError):
        audit.run_audit(recs, labels, tmp_path, reg, floor=0.123)
    assert not (tmp_path / audit.AUDIT_REL).exists()


def test_require_audit_refuses_until_written(tmp_path):
    with pytest.raises(CliError) as exc:
        audit.require_audit(tmp_path)
    assert "audit" in exc.value.message.lower()
    recs, labels = _fixture(tmp_path)
    audit.run_audit(recs, labels, tmp_path, _prereg(), floor=0.6)
    table = audit.require_audit(tmp_path)
    assert table["teacher_weak_sources"] == ["weak"]
    assert (tmp_path / audit.AUDIT_REL).is_file()


def test_require_audit_refuses_a_corrupt_table(tmp_path):
    p = tmp_path / audit.AUDIT_REL
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(CliError):
        audit.require_audit(tmp_path)
    p.write_text(json.dumps({"stage": "other"}), encoding="utf-8")
    with pytest.raises(CliError):
        audit.require_audit(tmp_path)


# ----------------------------------------------------------------- fx4: the audited build


def test_audit_records_the_teacher_build_id(tmp_path):
    recs, labels = _fixture(tmp_path)
    table = audit.run_audit(recs, labels, tmp_path, _prereg(), floor=0.6)
    assert table["teacher_build_id"] == "clef-b1"
    assert json.loads((tmp_path / audit.AUDIT_REL).read_text())["teacher_build_id"] == "clef-b1"
    assert audit.require_audit(tmp_path)["teacher_build_id"] == "clef-b1"


def test_audit_refuses_labels_with_more_than_one_build(tmp_path):
    recs = [_record("a1", "s"), _record("a2", "s")]
    labels = tmp_path / "l.jsonl"
    _write(labels, _rows("a1", 0.9) + _rows("a2", 0.9, build="clef-b2"))
    reg = _prereg()
    with pytest.raises(CliError, match="build"):
        audit.run_audit(recs, labels, tmp_path, reg, floor=0.6)
    assert not (tmp_path / audit.AUDIT_REL).exists()


def test_audit_refuses_labels_without_a_build_id(tmp_path):
    recs = [_record("a1", "s")]
    rows = _rows("a1", 0.9)
    del rows[0]["teacher_build_id"]
    labels = tmp_path / "l.jsonl"
    _write(labels, rows)
    reg = _prereg()
    with pytest.raises(CliError, match="build"):
        audit.run_audit(recs, labels, tmp_path, reg, floor=0.6)


def test_alternative_teacher_has_its_own_audit_table(tmp_path):
    recs, labels = _fixture(tmp_path)
    alt = tmp_path / "alt.jsonl"
    _write(alt, [{**r, "teacher_build_id": "alt-b7"} for r in audit.load_labels(labels)])
    audit.run_audit(recs, labels, tmp_path, _prereg(), floor=0.6)
    audit.run_audit(recs, alt, tmp_path, _prereg(), floor=0.6, role="alt")
    assert audit.require_audit(tmp_path)["teacher_build_id"] == "clef-b1"
    assert audit.require_audit(tmp_path, role="alt")["teacher_build_id"] == "alt-b7"


def test_require_audit_refuses_a_table_without_a_build(tmp_path):
    p = tmp_path / audit.AUDIT_REL
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stage": "teacher-audit", "sources": []}), encoding="utf-8")
    with pytest.raises(CliError, match="build"):
        audit.require_audit(tmp_path)


def test_variant_in_table_only_when_labels_carry_it(tmp_path):
    recs, labels = _fixture(tmp_path)
    rows = audit.load_labels(labels)
    assert "teacher_variant" not in audit.build_table(recs, rows, floor=0.6)
    for r in rows:
        r["teacher_variant"] = "BF16"
    assert audit.build_table(recs, rows, floor=0.6)["teacher_variant"] == "BF16"
    rows[0]["teacher_variant"] = "FP4"
    with pytest.raises(CliError, match="variants"):
        audit.build_table(recs, rows, floor=0.6)
