"""ScorerPrediction -> Prediction: jev_factory/measure/predictions.py.

The rules are nvsh measure.py's scorer_line at 9debdc6; every line produced
must pass the shared record's own validation (Prediction.from_dict).
"""

from __future__ import annotations

import math

import pytest

from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.core.predictions import Prediction
from jev_factory.measure.predictions import to_prediction
from tests.fixtures.toy_domain import DOMAIN

WORLD = {"home": "toy-home", "rooms": ["kitchen", "bedroom"]}


def _scored(winner: str | None, text: str = "x", *, drop: str | None = None, reasons=False):
    labels = sc.labels_for(DOMAIN, DOMAIN.candidates(reasons), reasons=reasons)
    if winner is None:
        logprobs: dict[str, float] = {}
    else:
        rest = 0.1 / (len(labels) - 1)
        logprobs = {
            letter: math.log(0.9 if name == winner else rest)
            for name, letter in labels.items()
            if name != drop
        }
    return sc.score(
        DOMAIN, lambda _p, _t: logprobs, "p", text, world=WORLD, reasons=reasons, entry_id="e1"
    )


def _line(scored, expected=None, **kwargs) -> Prediction:
    line = to_prediction(
        scored,
        entry_id="e1",
        expected=expected or {"escalate": True},
        elapsed_ms=kwargs.pop("elapsed_ms", 12.5),
        **kwargs,
    )
    return Prediction.from_dict(line.to_dict())  # the shared schema accepts it


def test_a_grounded_proposal_is_a_propose_line():
    line = _line(
        _scored("lamp_on", "turn on the kitchen"),
        {"operation": "lamp_on", "args": {"room": "kitchen"}},
    )
    assert (line.outcome, line.operation, line.arguments, line.grounded) == (
        "propose",
        "lamp_on",
        {"room": "kitchen"},
        True,
    )
    assert line.candidates["lamp_on"] == pytest.approx(0.9)
    assert line.raw_probabilities == line.candidates
    assert line.offered == (*DOMAIN.names(), "(explain)", "(escalate)")
    assert set(line.raw_scores) == set(line.offered)
    assert (line.tokens, line.ttfd_ms, line.latency_ms) == (0, 12.5, 12.5)
    assert line.invalid_reason is None


def test_an_ungrounded_proposal_is_invalid_never_a_proposal():
    line = _line(_scored("lamp_on", "turn it on"))
    assert (line.outcome, line.operation, line.arguments) == ("invalid", None, None)
    assert line.invalid_reason == "not_grounded"
    assert line.grounded is False
    assert line.candidates is not None  # the distribution is still recorded


def test_explain_and_escalate():
    assert _line(_scored("explain")).outcome == "explain"
    line = _line(_scored("escalate"))
    assert line.outcome == "escalate"
    assert line.grounded is None
    assert "(escalate)" in line.candidates


@pytest.mark.parametrize("choice", ["escalate:outside_table", "escalate:injection"])
def test_every_escalate_reason_is_an_escalation(choice):
    line = _line(_scored(choice, reasons=True))
    assert line.outcome == "escalate"
    assert line.operation is None
    assert line.invalid_reason is None
    assert line.candidates[choice] == pytest.approx(0.9)


def test_no_choice_is_no_label_mass_or_a_tier_error():
    assert _line(_scored(None)).invalid_reason == "no_label_mass"
    line = _line(_scored(None), call_error=True)
    assert (line.outcome, line.invalid_reason) == ("invalid", "tier_error")
    assert line.candidates is None and line.raw_scores is None


def test_an_incomplete_readout_keeps_its_choice_but_no_distribution():
    line = _line(_scored("escalate", drop="lamp_status"))
    assert line.outcome == "escalate"
    assert line.candidates is None
    assert line.raw_probabilities is None
    assert line.raw_scores is None  # a label was never read: no partial raw scores


def test_negative_elapsed_time_is_clamped():
    assert _line(_scored("explain"), elapsed_ms=-3.0).latency_ms == 0.0
