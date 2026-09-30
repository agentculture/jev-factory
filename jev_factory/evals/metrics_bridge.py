"""Metrics bridge onto :mod:`jev_factory.core.metrics` and :mod:`jev_factory.core.gate`.

This module never reimplements a formula that already lives in the core
modules; it calls them. The only things new here are:

* **top-k accuracy** (k > 1), built on ``metrics.top_candidate``'s own
  ranking rule generalised to "is gold within the top *k* by probability";
* **log loss** (core reports Brier, not log loss), over the same gold label
  and rolled distribution ``metrics.brier_one`` uses;
* **per-row normalized entropy and top1-top2 margin**: ``gate.normalized_entropy``
  and the margin ``gate.decide`` computes from ``gate._argmax`` /
  ``gate._second_place``, reported per prediction.

Everything else (right proposals, ECE, Brier, abstain precision/recall,
missing-candidate, wrong-mutating, per-slice) is
:func:`jev_factory.core.metrics.compute`'s own output, passed through. Every
function takes the :class:`~jev_factory.domain.model.Domain`: read-only vs
mutating comes from it, and an operation it does not know is mutating.
"""

from __future__ import annotations

import math
from typing import Sequence

from jev_factory.core import gate as gate_mod
from jev_factory.core import metrics as metrics_mod
from jev_factory.core.predictions import Prediction
from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/metrics_bridge.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 33-78: path-loaded scripts/lfm-finetune/metrics.py and gate.py (and the"
        " metrics_mod/gate_mod parameters) are replaced by jev_factory.core.metrics/gate"
        " imports; every function takes the Domain, which the core functions need",
        "task_done (lines 262-294, deviation d9) is dropped: it credits Track A inspections,"
        " and Track A is deferred (c14); nvsh.ops.table goes with it",
        "kept: NOT_MEASURABLE, top-k accuracy, log loss, per-row entropy/margin, row_metrics,"
        " compute's shape (rows, metrics_compute, aliases)",
    ],
    "licence": "Apache-2.0",
}

#: The explicit sentinel every calibration figure reports for a row with no
#: candidate distribution: never a number, never an estimate.
NOT_MEASURABLE = "not_measurable"

#: Default top-k values this bridge reports accuracy for.
TOP_K_VALUES: tuple[int, ...] = (1, 3, 5)

#: log loss's floor probability, so an absent gold label gives a large finite number.
_LOG_LOSS_EPSILON = 1e-12


def _rolled_and_offered(prediction: Prediction):
    """(rolled candidates, offered labels) for *prediction*; (None, None) with no distribution."""
    if prediction.candidates is None:
        return None, None
    rolled = metrics_mod.rollup_escalate_candidates(prediction.candidates)
    return rolled, list(prediction.offered_order())


def gold_label(prediction: Prediction) -> str:
    """The calibration gold label for *prediction* (``metrics.expected_label``)."""
    return metrics_mod.expected_label(prediction.expected)


def top1(prediction: Prediction) -> tuple[str, float] | None:
    """(label, probability) of the rolled top candidate, or ``None`` with no distribution."""
    rolled, _ = _rolled_and_offered(prediction)
    if rolled is None:
        return None
    return metrics_mod.top_candidate(rolled)


def top1_correct(prediction: Prediction) -> bool | None:
    """Whether the rolled top candidate's label is the gold label, or ``None``."""
    candidate = top1(prediction)
    if candidate is None:
        return None
    return candidate[0] == gold_label(prediction)


def topk_correct(prediction: Prediction, k: int) -> bool | None:
    """Whether gold is among the top *k* rolled candidates (ties: the label sorting first)."""
    rolled, _ = _rolled_and_offered(prediction)
    if rolled is None:
        return None
    ranked = sorted(rolled.items(), key=lambda item: (-item[1], item[0]))
    return gold_label(prediction) in [label for label, _ in ranked[:k]]


def brier(prediction: Prediction) -> float | str:
    """``metrics.brier_one`` for this row, or :data:`NOT_MEASURABLE`."""
    rolled, _ = _rolled_and_offered(prediction)
    if rolled is None:
        return NOT_MEASURABLE
    return metrics_mod.brier_one(rolled, gold_label(prediction))


def log_loss_one(prediction: Prediction) -> float | str:
    """-log(p_gold) over the rolled distribution (floored), or :data:`NOT_MEASURABLE`."""
    rolled, _ = _rolled_and_offered(prediction)
    if rolled is None:
        return NOT_MEASURABLE
    p_gold = float(rolled.get(gold_label(prediction), 0.0))
    return -math.log(max(p_gold, _LOG_LOSS_EPSILON))


def entropy(prediction: Prediction) -> float | str:
    """``gate.normalized_entropy`` over the offered labels (after roll-up), or n/m."""
    rolled, offered = _rolled_and_offered(prediction)
    if rolled is None:
        return NOT_MEASURABLE
    n_offered = len({metrics_mod.canonical_label(label) for label in offered})
    return gate_mod.normalized_entropy(rolled, n_offered)


def margin(prediction: Prediction) -> float | str:
    """The top1-top2 margin ``gate.decide`` computes, or :data:`NOT_MEASURABLE`."""
    rolled, offered = _rolled_and_offered(prediction)
    if rolled is None:
        return NOT_MEASURABLE
    top1_label, p_top1 = gate_mod._argmax(rolled, offered)
    return p_top1 - gate_mod._second_place(rolled, top1_label)


def row_metrics(
    prediction: Prediction, domain: Domain, *, top_k: Sequence[int] = TOP_K_VALUES
) -> dict:
    """Every per-row bridge metric for one prediction, JSON-able.

    Distribution figures are :data:`NOT_MEASURABLE` when the row has none;
    ``missing_candidate``/``wrong_mutating``/``slice`` are always defined.
    """
    correct = top1_correct(prediction)
    return {
        "id": prediction.id,
        "slice": metrics_mod.slice_name(prediction, domain),
        "has_distribution": prediction.candidates is not None,
        "top1_correct": NOT_MEASURABLE if correct is None else correct,
        "topk_correct": {
            k: (NOT_MEASURABLE if (value := topk_correct(prediction, k)) is None else value)
            for k in top_k
        },
        "brier": brier(prediction),
        "log_loss": log_loss_one(prediction),
        "entropy": entropy(prediction),
        "margin": margin(prediction),
        "missing_candidate": metrics_mod.is_missing_candidate(prediction),
        "wrong_mutating": (
            metrics_mod._wrong_operation_mutating(prediction, domain)
            or metrics_mod._wrong_arguments_mutating(prediction, domain)
        ),
    }


def top_k_accuracy(
    predictions: Sequence[Prediction], *, top_k: Sequence[int] = TOP_K_VALUES
) -> dict[int, dict]:
    """Top-k accuracy over operation-expected rows that carry a distribution."""
    measured = [
        p
        for p in predictions
        if metrics_mod.expect_kind(p.expected) == "operation" and p.candidates is not None
    ]
    result: dict[int, dict] = {}
    for k in top_k:
        n = sum(1 for p in measured if topk_correct(p, k))
        result[k] = {"n": n, "N": len(measured), "rate": metrics_mod._ratio(n, len(measured))}
    return result


def mean_log_loss(predictions: Sequence[Prediction]) -> dict:
    """Mean log loss over rows that carry a distribution."""
    losses = [log_loss_one(p) for p in predictions if p.candidates is not None]
    return {
        "n": len(losses),
        "without_distribution": len(predictions) - len(losses),
        "mean": (math.fsum(losses) / len(losses)) if losses else None,
    }


def missing_candidate_summary(predictions: Sequence[Prediction]) -> dict:
    """Missing-candidate rate over every row (``metrics.is_missing_candidate``)."""
    n = sum(1 for p in predictions if metrics_mod.is_missing_candidate(p))
    return {"n": n, "N": len(predictions), "rate": metrics_mod._ratio(n, len(predictions))}


def compute(
    predictions: Sequence[Prediction],
    domain: Domain,
    *,
    top_k: Sequence[int] = TOP_K_VALUES,
    bootstrap_seed: int | None = None,
    bootstrap_resamples: int | None = None,
) -> dict:
    """Every bridge metric for *predictions*; ``metrics_compute`` is core's own output."""
    kwargs = {}
    if bootstrap_seed is not None:
        kwargs["bootstrap_seed"] = bootstrap_seed
    if bootstrap_resamples is not None:
        kwargs["bootstrap_resamples"] = bootstrap_resamples
    base = metrics_mod.compute(predictions, domain, **kwargs)
    return {
        "rows": [row_metrics(p, domain, top_k=top_k) for p in predictions],
        "metrics_compute": base,
        "top_k_accuracy": top_k_accuracy(predictions, top_k=top_k),
        "log_loss": mean_log_loss(predictions),
        "missing_candidate": missing_candidate_summary(predictions),
        # Aliases onto metrics_compute's own dicts, not a recomputation.
        "abstain": base["abstention"],
        "wrong_mutating": base["wrong_mutating"],
        "slices": base["slices"],
    }
