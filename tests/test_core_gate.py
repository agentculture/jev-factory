"""jev_factory.core.gate, ported from nvsh's tests/test_lfm_finetune_gate.py (gate part).

``gate.decide`` never names an operation in its own logic: it reads the
Domain's read-only flag, so the toy domain's operation names below appear
only as fixture data. The sweep_gate half of nvsh's test file belongs to the
sweep port (t16), not here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from jev_factory.core import gate
from jev_factory.domain.model import ESCALATE_LABEL, EXPLAIN_LABEL
from tests.fixtures.toy_domain import DOMAIN

# Real toy operation names, used only as fixture data.
READ_ONLY_OP = "lamp_status"
READ_ONLY_OP_2 = "list_rooms"
MUTATING_OP = "set_scene"

ESC = ESCALATE_LABEL
EXP = EXPLAIN_LABEL


def _decide(candidates, thresholds, offered=None, domain=DOMAIN):
    return gate.decide(
        candidates, list(candidates) if offered is None else offered, thresholds, domain
    )


# ---------------------------------------------------------------------------
# Thresholds / ThresholdSet
# ---------------------------------------------------------------------------


def test_threshold_set_json_round_trip():
    ts = gate.ThresholdSet(floor=0.6, margin=0.1, max_entropy=0.8)
    assert gate.ThresholdSet.from_json(ts.to_json()) == ts


def test_threshold_set_json_round_trip_all_none():
    ts = gate.ThresholdSet()
    assert ts.to_json() == {"floor": None, "margin": None, "max_entropy": None}
    assert gate.ThresholdSet.from_json(ts.to_json()) == ts


def test_thresholds_json_round_trip():
    thresholds = gate.Thresholds(
        escalate=0.4,
        read_only=gate.ThresholdSet(floor=0.6),
        mutating=gate.ThresholdSet(floor=0.8, margin=0.2, max_entropy=0.5),
    )
    assert gate.Thresholds.from_json(thresholds.to_json()) == thresholds


def test_thresholds_from_json_rejects_non_object():
    with pytest.raises(gate.GateError):
        gate.Thresholds.from_json("nope")


def test_threshold_set_from_json_rejects_bad_value():
    with pytest.raises(gate.GateError):
        gate.ThresholdSet.from_json({"floor": "high"})


def test_labels_are_the_domain_control_labels():
    assert (gate.ESCALATE_LABEL, gate.EXPLAIN_LABEL) == ("(escalate)", "(explain)")
    assert gate.OUTCOMES == ("propose", "explain", "escalate", "abstain_uncertain")


# ---------------------------------------------------------------------------
# decide(): the argmax path (every threshold disabled)
# ---------------------------------------------------------------------------


def test_decide_proposes_the_argmax_operation():
    decision = _decide({READ_ONLY_OP: 0.9, EXP: 0.05, ESC: 0.05}, gate.Thresholds())
    assert decision.outcome == "propose"
    assert decision.label == READ_ONLY_OP


def test_decide_explains_when_explain_is_argmax():
    decision = _decide({READ_ONLY_OP: 0.3, EXP: 0.6, ESC: 0.1}, gate.Thresholds())
    assert decision.outcome == "explain"
    assert decision.label == EXP


def test_decide_escalates_when_escalate_is_argmax():
    decision = _decide({READ_ONLY_OP: 0.3, EXP: 0.1, ESC: 0.6}, gate.Thresholds())
    assert decision.outcome == "escalate"
    assert decision.label == ESC
    assert decision.reason == "argmax"


def test_decide_rolls_up_escalate_reasons_for_argmax():
    candidates = {READ_ONLY_OP: 0.45, "escalate:multi_step": 0.3, "escalate:injection": 0.25}
    decision = _decide(candidates, gate.Thresholds())
    assert decision.outcome == "escalate"
    assert decision.label == ESC


def test_decide_ties_go_to_the_earlier_offered_label():
    candidates = {READ_ONLY_OP: 0.5, READ_ONLY_OP_2: 0.5}
    decision = _decide(candidates, gate.Thresholds(), offered=[READ_ONLY_OP_2, READ_ONLY_OP])
    assert decision.label == READ_ONLY_OP_2


def test_decide_rejects_empty_distribution():
    thresholds = gate.Thresholds()
    with pytest.raises(gate.GateError):
        gate.decide({}, [], thresholds, DOMAIN)


# ---------------------------------------------------------------------------
# decide(): semantic escalate by threshold
# ---------------------------------------------------------------------------


def test_decide_escalates_on_threshold_even_when_not_argmax():
    decision = _decide({READ_ONLY_OP: 0.55, ESC: 0.45}, gate.Thresholds(escalate=0.4))
    assert decision.outcome == "escalate"
    assert decision.reason == "threshold"


def test_decide_escalate_threshold_disabled_never_fires():
    decision = _decide({READ_ONLY_OP: 0.55, ESC: 0.45}, gate.Thresholds(escalate=None))
    assert decision.outcome == "propose"


# ---------------------------------------------------------------------------
# decide(): abstain_uncertain, per operation class
# ---------------------------------------------------------------------------


def test_decide_abstains_on_floor_for_read_only():
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.3, EXP: 0.3}
    decision = _decide(candidates, gate.Thresholds(read_only=gate.ThresholdSet(floor=0.5)))
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "floor"
    assert decision.label == READ_ONLY_OP


def test_decide_floor_disabled_never_fires():
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.3, EXP: 0.3}
    decision = _decide(candidates, gate.Thresholds(read_only=gate.ThresholdSet(floor=None)))
    assert decision.outcome == "propose"


def test_decide_abstains_on_margin():
    candidates = {READ_ONLY_OP: 0.45, READ_ONLY_OP_2: 0.40, EXP: 0.15}
    decision = _decide(candidates, gate.Thresholds(read_only=gate.ThresholdSet(margin=0.5)))
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "margin"


def test_decide_abstains_on_entropy():
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.35, EXP: 0.25}
    decision = _decide(candidates, gate.Thresholds(read_only=gate.ThresholdSet(max_entropy=0.5)))
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "entropy"


def test_decide_uses_mutating_thresholds_for_a_mutating_operation():
    thresholds = gate.Thresholds(
        read_only=gate.ThresholdSet(floor=0.99),
        mutating=gate.ThresholdSet(floor=0.5),
    )
    decision = _decide({MUTATING_OP: 0.6, EXP: 0.4}, thresholds)
    assert decision.outcome == "propose"
    assert decision.label == MUTATING_OP


def test_decide_mutating_floor_can_abstain_a_mutating_proposal():
    thresholds = gate.Thresholds(mutating=gate.ThresholdSet(floor=0.7))
    decision = _decide({MUTATING_OP: 0.6, EXP: 0.4}, thresholds)
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "floor"


def test_decide_abstain_uncertain_never_fires_for_a_control():
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.99))
    decision = _decide({EXP: 0.6, READ_ONLY_OP: 0.4}, thresholds)
    assert decision.outcome == "explain"


def test_an_unknown_operation_gets_the_mutating_thresholds():
    """An operation missing from the domain is gated the stricter, mutating way."""
    candidates = {"not_in_the_table": 0.6, EXP: 0.2, ESC: 0.2}
    thresholds = gate.Thresholds(mutating=gate.ThresholdSet(floor=0.9))
    assert _decide(candidates, thresholds).outcome == "abstain_uncertain"
    lenient_mutating = gate.Thresholds(
        read_only=gate.ThresholdSet(floor=0.9), mutating=gate.ThresholdSet(floor=0.1)
    )
    assert _decide(candidates, lenient_mutating).outcome == "propose"


def test_entropy_is_normalised_over_the_rolled_up_candidates():
    reasons = [f"escalate:r{i}" for i in range(8)]
    candidates = {READ_ONLY_OP: 1 / 3, EXP: 1 / 3}
    candidates.update({r: (1 / 3) / 8 for r in reasons})
    offered = [READ_ONLY_OP, EXP, *reasons]
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(max_entropy=0.95))
    assert _decide(candidates, thresholds, offered=offered).outcome == "abstain_uncertain"


def test_the_class_comes_from_the_domain_read_only_flag():
    """Flipping one operation's read_only flag in the Domain flips which set gates it."""
    candidates = {READ_ONLY_OP: 0.6, EXP: 0.4}
    thresholds = gate.Thresholds(
        read_only=gate.ThresholdSet(floor=0.5), mutating=gate.ThresholdSet(floor=0.9)
    )
    assert _decide(candidates, thresholds).outcome == "propose"
    flipped = dataclasses.replace(
        DOMAIN,
        operations=tuple(
            dataclasses.replace(op, read_only=False) if op.name == READ_ONLY_OP else op
            for op in DOMAIN.operations
        ),
    )
    assert _decide(candidates, thresholds, domain=flipped).outcome == "abstain_uncertain"


# ---------------------------------------------------------------------------
# normalized_entropy
# ---------------------------------------------------------------------------


def test_normalized_entropy_zero_for_one_offered():
    assert gate.normalized_entropy({READ_ONLY_OP: 1.0}, 1) == 0.0


def test_normalized_entropy_one_for_uniform_pair():
    entropy = gate.normalized_entropy({READ_ONLY_OP: 0.5, READ_ONLY_OP_2: 0.5}, 2)
    assert entropy == pytest.approx(1.0)


def test_normalized_entropy_zero_for_certain_answer():
    entropy = gate.normalized_entropy({READ_ONLY_OP: 1.0, READ_ONLY_OP_2: 0.0}, 2)
    assert entropy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# decide_prediction: the gate over a predictions record
# ---------------------------------------------------------------------------


def test_decide_prediction_uses_the_record_offered_order():
    from jev_factory.core.predictions import Prediction

    prediction = Prediction.from_dict(
        {
            "id": "x",
            "expected": {"operation": READ_ONLY_OP, "args": {}},
            "outcome": "propose",
            "operation": READ_ONLY_OP_2,
            "arguments": {},
            "candidates": {READ_ONLY_OP: 0.5, READ_ONLY_OP_2: 0.5},
            "offered": [READ_ONLY_OP_2, READ_ONLY_OP],
            "tokens": 0,
            "ttfd_ms": 0.0,
            "latency_ms": 0.0,
        }
    )
    decision = gate.decide_prediction(prediction, gate.Thresholds(), DOMAIN)
    assert decision is not None
    assert decision.label == READ_ONLY_OP_2


# ---------------------------------------------------------------------------
# No operation names in the gate source
# ---------------------------------------------------------------------------


def test_no_operation_names_in_source():
    text = Path(gate.__file__).read_text(encoding="utf-8")
    for name in DOMAIN.names():
        assert name not in text, f"gate.py names the operation {name!r}"
