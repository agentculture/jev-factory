"""Correction records, leakage-checked intake and drafting (distillery t11)."""

from __future__ import annotations

import pytest

from jev_factory.distill import corrections as C
from jev_factory.distill.records import Record

TEXT = "please restart the nginx service on the staging web server right now"
PROTECTED = {
    "test": [{"id": "t1", "text": "what is the weather in paris today for my trip"}],
    "held_out": [{"id": "h1", "text": "list every open port on the database host please"}],
}


def _record(rid="r1", state=TEXT):
    return Record.from_json(
        {
            "id": rid,
            "state": state,
            "questions": {"q": {"type": "choice", "criteria": [["a", ""], ["b", ""]]}},
            "gold": {"q": {"label": "a"}},
            "split": "train",
            "source": "x",
            "licence": "MIT",
            "gold_provenance": "model-generated",
        }
    )


def _corr(**over):
    base = {
        "record_id": "r1",
        "question_id": "q",
        "author": "ori",
        "date": "2026-10-14",
        "reason": "weak",
        "label": "b",
        "text": TEXT,
    }
    base.update(over)
    return C.Correction(**base)


def test_valid_correction_roundtrips_through_jsonl(tmp_path):
    path = tmp_path / "run" / "corrections.jsonl"
    c = _corr(distribution={"a": 0.2, "b": 0.8})
    assert C.write_corrections(path, [c]) == 1
    assert C.read_corrections(path) == [c]
    assert c.status == "proposed"


@pytest.mark.parametrize("field", ["author", "date", "reason"])
def test_refuses_missing_author_date_or_reason(field):
    with pytest.raises(C.MissingCorrectionField):
        _corr(**{field: ""})
    with pytest.raises(C.MissingCorrectionField):
        C.Correction.from_json({k: v for k, v in _corr().to_json().items() if k != field})


def test_refuses_unknown_reason_status_and_empty_target():
    with pytest.raises(C.UnknownReason):
        _corr(reason="bad")
    with pytest.raises(C.UnknownStatus):
        _corr(status="done")
    with pytest.raises(C.BadTarget):
        _corr(label=None)


def test_refuses_exact_duplicate_of_protected_item():
    with pytest.raises(C.LeakageRefused, match="exact of test item 't1'"):
        C.intake(_corr(text=PROTECTED["test"][0]["text"].upper()), PROTECTED)


def test_refuses_near_duplicate_of_protected_item():
    near = "list every open port on the database host please now"
    with pytest.raises(C.LeakageRefused, match="near-duplicate of held_out item 'h1'"):
        C.intake(_corr(text=near), PROTECTED)


def test_clean_correction_passes_intake_and_empty_text_fails_closed():
    c = _corr()
    assert C.intake(c, PROTECTED) is c
    with pytest.raises(C.LeakageRefused):
        C.intake(_corr(text=""), PROTECTED)


def test_draft_yields_only_proposed_even_if_drafter_claims_approved():
    def drafter(record):
        return {"q": {"label": "b", "distribution": {"a": 0.1, "b": 0.9}, "status": "approved"}}

    res = C.draft_corrections([_record()], drafter, PROTECTED, author="clef-27b", date="d")
    assert [c.status for c in res.proposed] == ["proposed"]
    assert res.proposed[0].label == "b" and res.refused == []


def test_draft_refuses_unknown_options_and_failing_drafter():
    res = C.draft_corrections(
        [_record("r3")], lambda r: {"q": {"label": "zzz"}}, PROTECTED, author="m", date="d"
    )
    assert res.proposed == [] and len(res.refused) == 1

    def boom(record):
        raise RuntimeError("down")

    res = C.draft_corrections([_record()], boom, PROTECTED, author="m", date="d")
    assert res.refused == [("r1", "drafter failed: RuntimeError")]


def test_leaky_draft_is_refused_by_leakage():
    leaky = _record("r2", PROTECTED["test"][0]["text"])
    res = C.draft_corrections(
        [leaky], lambda r: {"q": {"label": "a"}}, PROTECTED, author="m", date="d"
    )
    assert res.proposed == [] and "exact" in res.refused[0][1]


def test_systemone_drafter_uses_client_distribution():
    from jev_factory.distill.systemone import CallResult, QuestionResult

    class Fake:
        def ask(self, record):
            qr = QuestionResult("q", "choice", distribution={"a": 0.3, "b": 0.7})
            return CallResult(record["id"], "m", {}, None, [qr])

    res = C.draft_corrections(
        [_record()], C.systemone_drafter(Fake()), PROTECTED, author="alt-teacher", date="d"
    )
    assert res.proposed[0].label == "b"
    assert res.proposed[0].distribution == {"a": 0.3, "b": 0.7}
    assert res.proposed[0].status == "proposed"
