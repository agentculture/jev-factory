"""One scored entry as a predictions line: :class:`ScorerPrediction` -> :class:`Prediction`.

The causal-LM adapter (:mod:`jev_factory.backbones.causal_lm.scorer`) returns
a pre-gate :class:`~jev_factory.backbones.causal_lm.scorer.ScorerPrediction`;
every distribution-only stage (metrics, calibration, the gate sweep, selection)
reads the backbone-agnostic :class:`~jev_factory.core.predictions.Prediction`.
This module is the one conversion between them, and it lives in the measure
package rather than in ``core`` because ``core`` never imports a backbone.

The rules (nvsh ``measure.py``'s ``scorer_line`` at 9debdc6):

* no choice at all -> ``"invalid"`` with ``invalid_reason`` ``"tier_error"``
  when the scorer's own call failed (a server that died), else
  ``"no_label_mass"`` (the model put no mass on any offered letter);
* ``explain`` -> ``"explain"``; ``escalate`` or any ``escalate:<reason>`` ->
  ``"escalate"`` (the reason's mass stays under its own label);
* an operation whose arguments were grounded -> ``"propose"`` with them;
* an operation whose arguments did **not** ground -> ``"invalid"`` with
  ``invalid_reason`` ``"not_grounded"`` (never a proposal the model did not
  ground, never arguments the model generated), ``grounded`` false.

``candidates`` is the calibration-labelled distribution, ``None`` when the
readout was incomplete (a label missing from the top log-probabilities) or had
no mass -- never renormalised over the labels that came back.
``raw_probabilities`` repeats the distribution as measured, so a later
calibration that rewrites ``candidates`` leaves the raw one on record;
``raw_scores`` carries each offered label's unnormalised score when every one
was read; ``offered`` is the offered labels in listing order.
"""

from __future__ import annotations

from typing import Mapping

from jev_factory.backbones.causal_lm.scorer import ScorerPrediction
from jev_factory.core.predictions import Prediction
from jev_factory.domain.model import ESCALATE, ESCALATE_PREFIX, EXPLAIN, Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/measure.py",
    "commit": "9debdc6",
    "adaptations": [
        "scorer_line (lines 1682-1709) and prediction_line (lines 1310-1333) become"
        " to_prediction over the adapter's ScorerPrediction; outcome None -> invalid"
        " (tier_error on a call error, else no_label_mass), explain, the escalate family,"
        " grounded propose, ungrounded -> invalid/not_grounded exactly as nvsh does",
        "DeclineReason.NOT_GROUNDED.value and TIER_ERROR_REASON (line 286) kept as literals",
        "adds offered, raw_scores, raw_probabilities and grounded (the shared record's"
        " optional fields) so raw and calibrated probabilities are each recorded",
    ],
    "licence": "Apache-2.0",
}

#: ``invalid_reason`` of a line whose scorer call failed (the server was unreachable).
TIER_ERROR_REASON = "tier_error"
#: ``invalid_reason`` of a line where no offered label had any mass.
NO_LABEL_MASS_REASON = "no_label_mass"
#: ``invalid_reason`` of a proposal whose arguments did not ground against the world.
NOT_GROUNDED_REASON = "not_grounded"


def _is_escalate(choice: str) -> bool:
    return choice == ESCALATE or choice.startswith(ESCALATE_PREFIX)


def to_prediction(
    scored: ScorerPrediction,
    *,
    entry_id: str,
    expected: Mapping[str, object],
    elapsed_ms: float,
    call_error: bool = False,
) -> Prediction:
    """The predictions line for one scored entry (see the module docstring)."""
    choice = scored.choice
    operation = arguments = grounded = invalid = None
    if choice is None:
        outcome, invalid = "invalid", (TIER_ERROR_REASON if call_error else NO_LABEL_MASS_REASON)
    elif choice == EXPLAIN:
        outcome = "explain"
    elif _is_escalate(choice):
        outcome = "escalate"
    elif scored.grounded and scored.arguments is not None:
        outcome, operation, arguments, grounded = "propose", choice, dict(scored.arguments), True
    else:
        outcome, invalid, grounded = "invalid", NOT_GROUNDED_REASON, False
    candidates = scored.candidates
    offered = tuple(Domain.calibration_label(name) for _label, name in scored.offered) or None
    raw_scores = None
    if scored.raw_scores and all(value is not None for value in scored.raw_scores.values()):
        raw_scores = {
            Domain.calibration_label(name): float(value)
            for name, value in scored.raw_scores.items()
        }
    elapsed = max(float(elapsed_ms), 0.0)
    return Prediction(
        id=entry_id,
        expected=dict(expected),
        outcome=outcome,
        operation=operation,
        arguments=arguments,
        candidates=None if candidates is None else dict(candidates),
        tokens=0,
        ttfd_ms=elapsed,
        latency_ms=elapsed,
        invalid_reason=invalid,
        offered=offered,
        raw_scores=raw_scores,
        raw_probabilities=None if candidates is None else dict(candidates),
        grounded=grounded,
    )
