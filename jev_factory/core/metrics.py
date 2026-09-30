"""Decision and calibration metrics from a predictions record, the same for every model.

Every scorer (a causal-LM letter readout, an encoder that scores candidates
natively, a generative baseline) writes the same backbone-agnostic record
(:mod:`jev_factory.core.predictions`), and this module turns any one of them
into the same figures, so a comparison table compares models rather than
scorers. It needs no tokenizer, no model and no backbone adapter: only the
record and the :class:`~jev_factory.domain.model.Domain` the model was built
for.

Usage::

    python -m jev_factory.core.metrics --domain <module|file.json> predictions.jsonl

prints one JSON object with every metric plus the issue-46 mapping below.

The domain seam
---------------

Whether an operation exists, whether its arguments pass its schema, and
whether it is read-only or mutating are all read from the Domain
(:meth:`Domain.validate_args`, :meth:`Domain.is_mutating`), never from an
operation name: this module names no operation. An operation the Domain does
not know counts as **mutating** (the safe side). The control labels are the
Domain contract's ``(explain)`` / ``(escalate)``
(:data:`~jev_factory.domain.model.EXPLAIN_LABEL`,
:data:`~jev_factory.domain.model.ESCALATE_LABEL`).

The metrics
-----------

* **Right proposals**: operation-expected entries answered with that
  operation and exactly those arguments, n/N and %. A grounded argument
  compares as grounding matches it: case-insensitively, with any suffix the
  Domain's ground kind declares optional.
* **Escalation (abstain) recall and precision**: recall over escalate-expected
  entries; precision's false positives are escalations of operation-expected
  entries. Explain entries are left out of both, and escalations of them are
  reported separately as ``escalated_on_explain``. ``precision_strict`` also
  counts those as false escalations; the **abstention** precision is the
  strict one.
* **False-positive tool calls**: proposals on explain- or escalate-expected
  entries, over all of those entries.
* **Wrong mutating**: a mutating operation proposed where something else was
  expected, plus the expected mutating operation with wrong arguments; the
  bar is judged on the total. Mutating is the Domain's ``read_only`` flag.
* **Invalid outputs**: an ``"invalid"`` line, or a ``"propose"`` line whose
  operation the Domain does not know or whose arguments fail its schema
  (counted under the Domain's own validation code, never as a tool call).
* **ECE** (10 equal-width bins over the top candidate's probability, a
  probability of exactly 1.0 in the last bin) and **Brier** (multi-class: the
  sum over labels of (p - one-hot)^2, a gold label missing from the
  candidates counting as p = 0, averaged over lines). The top candidate is the
  highest probability, a tie going to the label that sorts first; it is right
  when its label is the expected one -- arguments are not part of calibration.
* **Per-slice calibration**: the same ECE/Brier/reliability bins by the
  *gold* label's slice -- ``read_only`` and ``mutating`` (the Domain's flag),
  with escalate/explain-expected entries forming ``escalate_or_explain``.
  Each slice also reports its offered candidate count and its
  missing-candidate rate (id ends ``-nocand``, or gold not offered).
* **Escalation reason roll-up**: an ``escalate:<reason>`` label is folded into
  the bare ``(escalate)`` label wherever metrics group by outcome; the reason
  itself is tallied in ``escalation_reasons``.
* **abstain_uncertain**: a confidence gate declining. It counts toward the
  escalation bars exactly like ``escalate`` but is tallied separately.
* **Bootstrap confidence intervals**: every rate and every ECE/Brier figure
  carries its ``n`` and a seeded percentile 95% bootstrap CI.
* **Tokens generated** per decision and **time to first decision** /
  **latency**: the first line cold, the rest warm (median, nearest-rank p95).

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jev_factory.core.predictions import (
    FIELDS,
    OUTCOMES,
    SUM_TOLERANCE,
    Prediction,
    PredictionError,
    read_predictions,
)
from jev_factory.domain.model import ESCALATE_LABEL, ESCALATE_PREFIX, EXPLAIN_LABEL, Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/metrics.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 125-128: sys.path hack and nvsh.ops.table / nvsh.tiers.bench imports removed;"
        " the Domain is passed in (compute, slice_name, invalid_reason, issue46_json ...)",
        "lines 130-300: record schema (FIELDS, OUTCOMES, Prediction, read_predictions) moved to"
        " jev_factory.core.predictions and re-exported; MetricsError aliases PredictionError",
        "lines 321-323, 423-428: tier_bench.ESCALATE_LABEL/EXPLAIN_LABEL become"
        " jev_factory.domain.model's; escalate: prefix is model.ESCALATE_PREFIX",
        "lines 331-334: ops_table.validate becomes Domain.validate_args (same error codes)",
        "lines 354-358, 643-656: ops_table.get(...).read_only becomes Domain.is_mutating;"
        " an unknown operation counts as mutating (nvsh's slice_name rule, now everywhere)",
        "lines 361-378: _canonical_argument's hard-coded service/container keys and"
        " '.service' suffix become the Domain ArgSpec.ground kind and its declared suffixes",
        "lines 759-772: tier_bench._median/_percentile_95 inlined as median/percentile_95",
        "main() takes --domain (module, dotted name or JSON file) and runs as"
        " python -m jev_factory.core.metrics",
    ],
    "licence": "Apache-2.0",
}

__all__ = [
    "ESCALATE_LABEL",
    "EXPLAIN_LABEL",
    "FIELDS",
    "OUTCOMES",
    "SUM_TOLERANCE",
    "MetricsError",
    "Prediction",
    "read_predictions",
    "compute",
]

#: A predictions record that does not follow the schema (the record module's error).
MetricsError = PredictionError

#: Equal-width confidence bins for ECE.
ECE_BINS = 10
#: ``invalid_reason`` when an invalid line gives none.
DEFAULT_INVALID_REASON = "unparseable"

#: A candidate label ``escalate:<reason>`` rolls up to the bare escalate label.
ESCALATE_REASON_PREFIX = ESCALATE_PREFIX
#: Bootstrap defaults for every rate/ECE/Brier confidence interval.
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
#: The per-slice breakdown of :func:`compute_slices`, in report order.
SLICE_NAMES = ("read_only", "mutating", "escalate_or_explain")

#: Outcome -> issue 46's decision JSON, for the report only.
ISSUE46_MAPPING = (
    {
        "nvsh": "propose",
        "issue46": '{"action": "tool", "tool": <operation>, "arguments": <arguments>}',
    },
    {"nvsh": "explain", "issue46": '{"action": "no_action"}'},
    {"nvsh": "escalate", "issue46": '{"action": "abstain"}'},
    {"nvsh": "abstain_uncertain", "issue46": '{"action": "abstain"}'},
    {"nvsh": "invalid", "issue46": '{"action": "invalid"}'},
)
ISSUE46_NOTE = (
    "Reporting only: every model is trained and scored on the domain's propose/explain/escalate "
    "candidates. Issue 46's abstain is escalate, so abstention recall is the escalation "
    "recall and abstention precision the strict escalation precision (an escalation on an "
    "explain entry counts against it). Explain (answer in words, no tool) has no counterpart in "
    "issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for "
    "Track B and is never counted as an abstention. abstain_uncertain (a confidence "
    "gate, not a semantic escalate decision) maps to the same issue-46 abstain action as "
    "escalate, and counts the same way in every escalation bar, but is tallied separately in "
    "escalation/outcome_counts. An invalid output is not a decision."
)


# ---------------------------------------------------------------------------
# Per-line classification
# ---------------------------------------------------------------------------


def expect_kind(expected: Mapping[str, object]) -> str:
    """``"escalate"``, ``"explain"`` or ``"operation"``."""
    if expected.get("escalate"):
        return "escalate"
    if expected.get("explain"):
        return "explain"
    return "operation"


def expected_label(expected: Mapping[str, object]) -> str:
    """The calibration label: the escalate/explain control label, or the operation name."""
    kind = expect_kind(expected)
    if kind == "escalate":
        return ESCALATE_LABEL
    if kind == "explain":
        return EXPLAIN_LABEL
    return str(expected["operation"])


def invalid_reason(prediction: Prediction, domain: Domain) -> str | None:
    """Why this line is an invalid output, or ``None`` when it is a valid decision."""
    if prediction.outcome == "invalid":
        return prediction.invalid_reason or DEFAULT_INVALID_REASON
    if prediction.outcome == "propose":
        error = domain.validate_args(prediction.operation, prediction.arguments)
        if error is not None:
            return error.code
    return None


def _proposed(prediction: Prediction, domain: Domain) -> bool:
    """A valid tool call: a proposal the Domain accepts."""
    return prediction.outcome == "propose" and invalid_reason(prediction, domain) is None


def _escalated(prediction: Prediction) -> bool:
    """True for either escalation outcome (``escalate`` or ``abstain_uncertain``)."""
    return prediction.outcome in ("escalate", "abstain_uncertain")


def _is_mutating_proposal(prediction: Prediction, domain: Domain) -> bool:
    if not _proposed(prediction, domain):
        return False
    return domain.is_mutating(str(prediction.operation))


def _canonical_argument(domain: Domain, operation: object, name: str, value: object) -> object:
    """*value* as grounding compares it (nvsh issue 53, deviation d5).

    An argument whose spec names a ground kind matches case-insensitively,
    with any of that kind's declared suffixes optional ('kitchen' and a
    grounded 'Kitchen.zone' name one room). Every other argument compares
    exactly.
    """
    if not isinstance(value, str) or not isinstance(operation, str):
        return value
    op = domain.get(operation)
    spec = None if op is None else next((a for a in op.args if a.name == name), None)
    if spec is None or spec.ground is None:
        return value
    kind = domain.ground_kind(spec.ground)
    value = value.casefold()
    for suffix in () if kind is None else kind.suffixes:
        folded = suffix.casefold()
        if folded and value.endswith(folded):
            return value[: -len(folded)]
    return value


def _same_arguments(domain: Domain, operation: object, got: object, expected: object) -> bool:
    if not isinstance(got, dict) or not isinstance(expected, dict):
        return got == expected
    if got.keys() != expected.keys():
        return False
    return all(
        _canonical_argument(domain, operation, key, got[key])
        == _canonical_argument(domain, operation, key, expected[key])
        for key in got
    )


def _right_proposal(prediction: Prediction, domain: Domain) -> bool:
    expected = prediction.expected
    return (
        _proposed(prediction, domain)
        and prediction.operation == expected.get("operation")
        and _same_arguments(
            domain, prediction.operation, prediction.arguments, expected.get("args", {})
        )
    )


def _wrong_operation_mutating(prediction: Prediction, domain: Domain) -> bool:
    """A mutating operation where something else was expected."""
    return _is_mutating_proposal(prediction, domain) and (
        prediction.operation != prediction.expected.get("operation")
    )


def _wrong_arguments_mutating(prediction: Prediction, domain: Domain) -> bool:
    """The expected mutating operation, wrong arguments."""
    return (
        _is_mutating_proposal(prediction, domain)
        and prediction.operation == prediction.expected.get("operation")
        and not _right_proposal(prediction, domain)
    )


# ---------------------------------------------------------------------------
# Escalation-reason roll-up
# ---------------------------------------------------------------------------


def _is_escalate_label(label: str) -> bool:
    """True for the bare ``(escalate)`` label or an ``escalate:<reason>`` one."""
    return label == ESCALATE_LABEL or label.startswith(ESCALATE_REASON_PREFIX)


def canonical_label(label: str) -> str:
    """*label*, or the bare escalate label when *label* is escalate-family."""
    return ESCALATE_LABEL if _is_escalate_label(label) else label


def escalate_reason(label: str) -> str | None:
    """The ``<reason>`` of an ``escalate:<reason>`` label, or ``None``."""
    if label.startswith(ESCALATE_REASON_PREFIX):
        return label[len(ESCALATE_REASON_PREFIX) :] or None
    return None


def rollup_escalate_candidates(candidates: Mapping[str, float]) -> dict[str, float]:
    """*candidates* with every escalate-family label's mass summed into one entry."""
    rolled: dict[str, float] = {}
    for label, value in candidates.items():
        key = canonical_label(label)
        rolled[key] = rolled.get(key, 0.0) + value
    return rolled


def escalation_reason_counts(predictions: Sequence[Prediction]) -> dict[str, int]:
    """How often each ``escalate:<reason>`` label was a line's (raw) top candidate."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction.candidates is None:
            continue
        label, _ = top_candidate(prediction.candidates)
        reason = escalate_reason(label)
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _resample(items: Sequence, rng: random.Random) -> list:
    n = len(items)
    return [items[rng.randrange(n)] for _ in range(n)]


def _mean_stat(values: Sequence[float]) -> float | None:
    return (math.fsum(values) / len(values)) if values else None


def bootstrap_ci(
    items: Sequence,
    stat_fn: Callable[[Sequence], float | None],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Percentile 95% CI of ``stat_fn(items)``, resampling *items* with replacement.

    A resample on which ``stat_fn`` returns ``None`` is skipped. Seeded, so a
    report is reproducible.
    """
    n = len(items)
    if n == 0:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    value = stat_fn(items)
    if n == 1:
        return {"n": 1, "value": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
    stats = []
    for _ in range(resamples):
        sample_stat = stat_fn(_resample(items, rng))
        if sample_stat is not None:
            stats.append(sample_stat)
    if not stats:
        return {"n": n, "value": value, "ci_low": None, "ci_high": None}
    stats.sort()
    lo = stats[int(round(0.025 * (len(stats) - 1)))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return {"n": n, "value": value, "ci_low": lo, "ci_high": hi}


def _stratified_bootstrap_ci(
    strata: Mapping[str, Sequence],
    stat_fn: Callable[[Mapping[str, Sequence]], float | None],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Like :func:`bootstrap_ci`, each named stratum resampled independently."""
    total_n = sum(len(items) for items in strata.values())
    value = stat_fn(strata)
    if total_n == 0:
        return {"n": 0, "value": value, "ci_low": None, "ci_high": None}
    if total_n == 1:
        return {"n": 1, "value": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
    stats = []
    for _ in range(resamples):
        sample = {name: _resample(items, rng) for name, items in strata.items() if items}
        for name in strata:
            sample.setdefault(name, [])
        sample_stat = stat_fn(sample)
        if sample_stat is not None:
            stats.append(sample_stat)
    if not stats:
        return {"n": total_n, "value": value, "ci_low": None, "ci_high": None}
    stats.sort()
    lo = stats[int(round(0.025 * (len(stats) - 1)))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return {"n": total_n, "value": value, "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def top_candidate(candidates: Mapping[str, float]) -> tuple[str, float]:
    """The highest-probability label; a tie goes to the label that sorts first."""
    label = min(candidates, key=lambda name: (-candidates[name], name))
    return label, float(candidates[label])


def bin_index(confidence: float, bins: int = ECE_BINS) -> int:
    """Equal-width bin of *confidence*: [0, 0.1) is 0, ..., [0.9, 1.0] is the last."""
    # Rounded first so a recorded 0.3 (0.30000000000000004 after * 10) cannot slip a bin.
    index = math.floor(round(confidence * bins, 9))
    return min(max(index, 0), bins - 1)


def ece_bins(pairs: Sequence[tuple[float, bool]], bins: int = ECE_BINS) -> list[dict]:
    """Per bin: count, mean confidence and accuracy (``None`` for an empty bin)."""
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in pairs:
        grouped[bin_index(confidence, bins)].append((confidence, correct))
    rows = []
    for index, group in enumerate(grouped):
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": len(group),
                "confidence": (math.fsum(c for c, _ in group) / len(group)) if group else None,
                "accuracy": (sum(1 for _, ok in group if ok) / len(group)) if group else None,
            }
        )
    return rows


def ece(pairs: Sequence[tuple[float, bool]], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error: sum over bins of (n_b / N) * |accuracy_b - confidence_b|."""
    if not pairs:
        return None
    total = len(pairs)
    return math.fsum(
        row["n"] / total * abs(row["accuracy"] - row["confidence"])
        for row in ece_bins(pairs, bins)
        if row["n"]
    )


def brier_one(candidates: Mapping[str, float], gold: str) -> float:
    """Multi-class Brier for one line; a gold label absent from *candidates* counts as p = 0."""
    labels = set(candidates) | {gold}
    return math.fsum(
        (float(candidates.get(label, 0.0)) - (1.0 if label == gold else 0.0)) ** 2
        for label in labels
    )


def compute_calibration(
    predictions: Sequence[Prediction],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """ECE/Brier/bins over *predictions*' rolled-up distributions, each with a bootstrap CI."""
    with_distribution = [p for p in predictions if p.candidates is not None]
    pairs = []
    briers = []
    for prediction in with_distribution:
        gold = expected_label(prediction.expected)
        rolled = rollup_escalate_candidates(prediction.candidates)  # type: ignore[arg-type]
        label, confidence = top_candidate(rolled)
        pairs.append((confidence, label == gold))
        briers.append(brier_one(rolled, gold))
    return {
        "n": len(with_distribution),
        "without_distribution": len(predictions) - len(with_distribution),
        "ece": ece(pairs),
        "ece_ci": bootstrap_ci(pairs, ece, seed, resamples),
        "brier": (math.fsum(briers) / len(briers)) if briers else None,
        "brier_ci": bootstrap_ci(briers, _mean_stat, seed, resamples),
        "bins": ece_bins(pairs),
    }


# ---------------------------------------------------------------------------
# Slices: read-only vs mutating vs escalate/explain
# ---------------------------------------------------------------------------


def slice_name(prediction: Prediction, domain: Domain) -> str:
    """Which of :data:`SLICE_NAMES` a prediction's GOLD label belongs to.

    Decided from the Domain's ``read_only`` flag, never a name; a gold
    operation the Domain does not know is bucketed as ``mutating``.
    """
    if expect_kind(prediction.expected) != "operation":
        return "escalate_or_explain"
    return "mutating" if domain.is_mutating(str(prediction.expected["operation"])) else "read_only"


def is_missing_candidate(prediction: Prediction) -> bool:
    """A missing-candidate line: id ends ``-nocand``, or gold isn't offered (after roll-up)."""
    if prediction.id.endswith("-nocand"):
        return True
    if prediction.candidates is None:
        return False
    gold = canonical_label(expected_label(prediction.expected))
    offered = {canonical_label(label) for label in prediction.candidates}
    return gold not in offered


def _candidate_count_stats(predictions: Sequence[Prediction]) -> dict:
    counts = [len(p.candidates) for p in predictions if p.candidates is not None]
    return {
        "n": len(counts),
        "mean": (sum(counts) / len(counts)) if counts else None,
        "median": _median(counts),
    }


def compute_slice(
    predictions: Sequence[Prediction],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """One slice's calibration, offered-candidate count and missing-candidate rate."""
    missing = [1 if is_missing_candidate(p) else 0 for p in predictions]
    return {
        "n": len(predictions),
        "calibration": compute_calibration(predictions, seed=seed, resamples=resamples),
        "candidate_count": _candidate_count_stats(predictions),
        "missing_candidate": {
            "n": sum(missing),
            "N": len(predictions),
            "rate": bootstrap_ci(missing, _mean_stat, seed, resamples),
        },
    }


def compute_slices(
    predictions: Sequence[Prediction],
    domain: Domain,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """:data:`SLICE_NAMES` -> :func:`compute_slice`, bucketed by :func:`slice_name`."""
    buckets: dict[str, list[Prediction]] = {name: [] for name in SLICE_NAMES}
    for prediction in predictions:
        buckets[slice_name(prediction, domain)].append(prediction)
    return {
        name: compute_slice(items, seed=seed, resamples=resamples)
        for name, items in buckets.items()
    }


def reliability_markdown(slices: Mapping[str, dict]) -> str:
    """One markdown reliability-bin table per slice, from :func:`compute_slices`' output."""
    tables = []
    for name in SLICE_NAMES:
        data = slices.get(name)
        if data is None:
            continue
        bins = data["calibration"]["bins"]
        lines = [
            f"### {name}",
            "",
            "| bin | n | confidence | accuracy |",
            "| --- | --- | --- | --- |",
        ]
        for row in bins:
            confidence = "-" if row["confidence"] is None else f"{row['confidence']:.3f}"
            accuracy = "-" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
            lines.append(
                f"| [{row['lower']:.1f}, {row['upper']:.1f}) | {row['n']} | "
                f"{confidence} | {accuracy} |"
            )
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


# ---------------------------------------------------------------------------
# The whole file
# ---------------------------------------------------------------------------


def _ratio(n: int, total: int) -> float | None:
    return (n / total) if total else None


def median(values: Sequence[float]) -> float:
    """The median of a non-empty sequence (nvsh bench's ``_median``)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile_95(sorted_values: Sequence[float]) -> float:
    """Nearest-rank p95 over an already-sorted non-empty sequence (nvsh bench's)."""
    n = len(sorted_values)
    rank = max(1, min(n, -(-95 * n // 100)))  # ceil(95/100 * n), clamped
    return float(sorted_values[rank - 1])


def _timing(values: Sequence[float]) -> dict:
    """The first value cold, the rest warm."""
    if not values:
        return {"cold_ms": None, "warm_median_ms": None, "warm_p95_ms": None}
    warm = list(values[1:])
    return {
        "cold_ms": values[0],
        "warm_median_ms": median(warm) if warm else None,
        "warm_p95_ms": percentile_95(sorted(warm)) if warm else None,
    }


def _median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def compute(
    predictions: Sequence[Prediction],
    domain: Domain,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Every metric for one predictions record against *domain*, plus the issue-46 mapping."""
    seed, resamples = bootstrap_seed, bootstrap_resamples
    kinds: dict[str, list[Prediction]] = {"operation": [], "escalate": [], "explain": []}
    for prediction in predictions:
        kinds[expect_kind(prediction.expected)].append(prediction)

    tp = sum(1 for p in kinds["escalate"] if _escalated(p))
    fn = len(kinds["escalate"]) - tp
    fp = sum(1 for p in kinds["operation"] if _escalated(p))
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    # An escalation on an explain entry is a false abstention too (strict precision).
    on_explain = sum(1 for p in kinds["explain"] if _escalated(p))
    precision_strict = _ratio(tp, tp + fp + on_explain)

    declines = kinds["explain"] + kinds["escalate"]
    right = sum(1 for p in kinds["operation"] if _right_proposal(p, domain))
    false_calls = sum(1 for p in declines if _proposed(p, domain))

    reasons: dict[str, int] = {}
    for prediction in predictions:
        reason = invalid_reason(prediction, domain)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    invalid = sum(reasons.values())

    wrong_operation = sum(1 for p in predictions if _wrong_operation_mutating(p, domain))
    wrong_arguments = sum(1 for p in predictions if _wrong_arguments_mutating(p, domain))

    tokens = [p.tokens for p in predictions]
    right_ratio = _ratio(right, len(kinds["operation"]))

    def _precision_stat(strata: Mapping[str, Sequence[Prediction]]) -> float | None:
        t = sum(1 for p in strata["escalate"] if _escalated(p))
        f = sum(1 for p in strata["operation"] if _escalated(p))
        denom = t + f
        return (t / denom) if denom else None

    def _precision_strict_stat(strata: Mapping[str, Sequence[Prediction]]) -> float | None:
        t = sum(1 for p in strata["escalate"] if _escalated(p))
        f = sum(1 for p in strata["operation"] if _escalated(p))
        e = sum(1 for p in strata["explain"] if _escalated(p))
        denom = t + f + e
        return (t / denom) if denom else None

    abstain_uncertain = {
        "tp": sum(1 for p in kinds["escalate"] if p.outcome == "abstain_uncertain"),
        "fp": sum(1 for p in kinds["operation"] if p.outcome == "abstain_uncertain"),
        "escalated_on_explain": sum(
            1 for p in kinds["explain"] if p.outcome == "abstain_uncertain"
        ),
    }
    outcome_counts = {
        outcome: sum(1 for p in predictions if p.outcome == outcome) for outcome in OUTCOMES
    }

    return {
        "right_proposals": {
            "n": right,
            "N": len(kinds["operation"]),
            "percent": None if right_ratio is None else 100 * right_ratio,
            "ci": bootstrap_ci(
                [1 if _right_proposal(p, domain) else 0 for p in kinds["operation"]],
                _mean_stat,
                seed,
                resamples,
            ),
        },
        "escalation": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": recall,
            "recall_ci": bootstrap_ci(
                [1 if _escalated(p) else 0 for p in kinds["escalate"]], _mean_stat, seed, resamples
            ),
            "precision": precision,
            "precision_ci": _stratified_bootstrap_ci(
                {"escalate": kinds["escalate"], "operation": kinds["operation"]},
                _precision_stat,
                seed,
                resamples,
            ),
            "precision_strict": precision_strict,
            "precision_strict_ci": _stratified_bootstrap_ci(
                {
                    "escalate": kinds["escalate"],
                    "operation": kinds["operation"],
                    "explain": kinds["explain"],
                },
                _precision_strict_stat,
                seed,
                resamples,
            ),
            "escalated_on_explain": on_explain,
            "abstain_uncertain": abstain_uncertain,
        },
        "abstention": {"recall": recall, "precision": precision_strict},
        "false_positive_tool_calls": {
            "n": false_calls,
            "N": len(declines),
            "rate": _ratio(false_calls, len(declines)),
            "ci": bootstrap_ci(
                [1 if _proposed(p, domain) else 0 for p in declines], _mean_stat, seed, resamples
            ),
        },
        "wrong_mutating": {
            "wrong_operation": wrong_operation,
            "wrong_arguments": wrong_arguments,
            "total": wrong_operation + wrong_arguments,
        },
        "invalid": {
            "n": invalid,
            "N": len(predictions),
            "rate": _ratio(invalid, len(predictions)),
            "ci": bootstrap_ci(
                [1 if invalid_reason(p, domain) is not None else 0 for p in predictions],
                _mean_stat,
                seed,
                resamples,
            ),
            "by_reason": dict(sorted(reasons.items())),
        },
        "calibration": compute_calibration(predictions, seed=seed, resamples=resamples),
        "outcome_counts": outcome_counts,
        "escalation_reasons": escalation_reason_counts(predictions),
        "slices": compute_slices(predictions, domain, seed=seed, resamples=resamples),
        "bootstrap": {"seed": seed, "resamples": resamples},
        "tokens": {
            "total": sum(tokens),
            "mean": (sum(tokens) / len(tokens)) if tokens else None,
            "median": _median(tokens),
        },
        "time_to_first_decision": _timing([p.ttfd_ms for p in predictions]),
        "latency": _timing([p.latency_ms for p in predictions]),
        "issue46_mapping": issue46_mapping(),
    }


# ---------------------------------------------------------------------------
# The issue-46 mapping
# ---------------------------------------------------------------------------


def issue46_json(prediction: Prediction, domain: Domain) -> dict:
    """One line as issue 46's decision JSON (see :data:`ISSUE46_MAPPING`)."""
    if invalid_reason(prediction, domain) is not None:
        return {"action": "invalid"}
    if prediction.outcome == "propose":
        return {
            "action": "tool",
            "tool": prediction.operation,
            "arguments": dict(prediction.arguments or {}),
        }
    if _escalated(prediction):
        return {"action": "abstain"}
    return {"action": "no_action"}


def issue46_mapping() -> dict:
    """The fixed mapping, as rows and as a markdown table for the report."""
    lines = ["| outcome | issue 46 JSON |", "|---|---|"]
    lines += [f"| {row['nvsh']} | `{row['issue46']}` |" for row in ISSUE46_MAPPING]
    return {
        "rows": [dict(row) for row in ISSUE46_MAPPING],
        "markdown": "\n".join(lines),
        "note": ISSUE46_NOTE,
    }


def main(argv: list[str] | None = None) -> int:
    from jev_factory.domain.validate import DomainError, load_domain

    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.core.metrics", description=__doc__.split("\n", 1)[0]
    )
    parser.add_argument(
        "--domain", required=True, help="domain module (dotted name) or JSON domain file"
    )
    parser.add_argument("predictions", type=Path, help="predictions JSONL file")
    args = parser.parse_args(argv)
    try:
        domain = load_domain(args.domain)
        result = compute(read_predictions(args.predictions), domain)
    except (OSError, MetricsError, DomainError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
