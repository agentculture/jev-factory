"""The backbone-agnostic predictions record (jev_factory.core.predictions)."""

from __future__ import annotations

import pytest

from jev_factory.core import predictions as pr
from jev_factory.domain.model import ESCALATE_LABEL, EXPLAIN_LABEL


def _row(**over) -> dict:
    row = {
        "id": "e1",
        "expected": {"operation": "lamp_status", "args": {}},
        "outcome": "propose",
        "operation": "lamp_status",
        "arguments": {},
        "candidates": {"lamp_status": 0.7, EXPLAIN_LABEL: 0.2, ESCALATE_LABEL: 0.1},
        "tokens": 0,
        "ttfd_ms": 1.0,
        "latency_ms": 2.0,
    }
    row.update(over)
    return row


def test_required_row_parses_with_optional_fields_absent():
    p = pr.Prediction.from_dict(_row())
    assert (p.offered, p.raw_scores, p.raw_probabilities, p.grounded) == (None, None, None, None)
    assert p.offered_order() == ("lamp_status", EXPLAIN_LABEL, ESCALATE_LABEL)


def test_full_record_round_trips(tmp_path):
    full = _row(
        offered=[ESCALATE_LABEL, "lamp_status", EXPLAIN_LABEL],
        raw_scores={"lamp_status": -0.3, EXPLAIN_LABEL: -1.6, ESCALATE_LABEL: -2.3},
        raw_probabilities={"lamp_status": 0.6, EXPLAIN_LABEL: 0.25, ESCALATE_LABEL: 0.15},
        grounded=True,
    )
    p = pr.Prediction.from_dict(full)
    assert p.offered_order() == (ESCALATE_LABEL, "lamp_status", EXPLAIN_LABEL)
    assert p.to_dict() == full
    path = tmp_path / "p.jsonl"
    pr.write_predictions(path, [p])
    assert pr.read_predictions(path) == [p]


def test_to_dict_leaves_out_unset_optional_fields():
    assert set(pr.Prediction.from_dict(_row()).to_dict()) == set(pr.FIELDS)


@pytest.mark.parametrize(
    "over, fragment",
    [
        ({"offered": ["lamp_status"]}, "candidates labels differ"),
        ({"offered": ["a", "a"]}, "repeat"),
        ({"offered": []}, "offered"),
        ({"raw_scores": {"lamp_status": 1.0}}, "raw_scores labels differ"),
        ({"raw_scores": {"lamp_status": "x"}}, "raw_scores"),
        (
            {"raw_scores": {"lamp_status": float("inf"), EXPLAIN_LABEL: 0, ESCALATE_LABEL: 0}},
            "finite",
        ),
        ({"raw_probabilities": {"lamp_status": 0.5}}, "sum"),
        ({"grounded": "yes"}, "grounded"),
    ],
)
def test_optional_fields_are_validated(over, fragment):
    row = _row(**over)
    with pytest.raises(pr.PredictionError, match=fragment):
        pr.Prediction.from_dict(row)


def test_raw_scores_may_be_negative_log_probabilities():
    raw = {"lamp_status": -0.1, EXPLAIN_LABEL: -3.0, ESCALATE_LABEL: -4.0}
    assert pr.Prediction.from_dict(_row(raw_scores=raw)).raw_scores == raw


def test_unknown_keys_are_ignored():
    assert pr.Prediction.from_dict(_row(backbone_note="gliner")).id == "e1"


def test_from_rows_names_the_row_and_rejects_duplicates():
    bad_second = [_row(), _row(id="e2", tokens=-1)]
    with pytest.raises(pr.PredictionError, match="row 2"):
        pr.from_rows(bad_second)
    duplicated = [_row(), _row()]
    with pytest.raises(pr.PredictionError, match="duplicate"):
        pr.from_rows(duplicated)
    assert [p.id for p in pr.from_rows([_row(), _row(id="e2")])] == ["e1", "e2"]
