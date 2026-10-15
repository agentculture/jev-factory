"""Assessment metrics for typed-question predictions (choice / noul / score).

Every metric is reported per question kind and over all kinds, each with a seeded
percentile bootstrap CI from :func:`jev_factory.core.metrics.bootstrap_ci`. Reuses
``bootstrap_ci`` and ``brier_one`` from core; deliberately not routed through
``core.metrics.compute``, which is tool-scorer shaped (``expect_kind``).

Prediction shapes (all are plain distributions, so nothing reads a ``confidence`` field):

* choice: probabilities over the options, gold is the option index.
* score: probabilities over the ordered levels, gold is the level index; scored both on
  the argmax level index (accuracy, F1, confident-wrong) and on the full distribution
  (JSD, log-loss, Brier).
* noul: scored on P(true), held as the two-class distribution ``(P(false), P(true))``
  with class 1 = true.

Peak memory is a measurement field filled by the benchmark/final stages, not computed here.
Pure Python, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from jev_factory.core.metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_ci,
    brier_one,
)

KINDS = ("choice", "score", "noul")
SUM_TOLERANCE = 1e-6
LOG_EPSILON = 1e-12  # probability floor for log-loss, so a zero gold probability is finite


def _check_distribution(name: str, probs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(p) for p in probs)
    if len(values) < 2:
        raise ValueError(f"{name}: a distribution needs at least 2 entries")
    if any(p < 0.0 or not math.isfinite(p) for p in values):
        raise ValueError(f"{name}: probabilities must be finite and non-negative")
    if abs(math.fsum(values) - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"{name}: probabilities must sum to 1")
    return values


@dataclass(frozen=True)
class PredItem:
    """One question's prediction plus whatever references exist for it."""

    kind: str
    probs: tuple[float, ...]
    gold_index: int | None = None
    gold_probs: tuple[float, ...] | None = None
    teacher_probs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown question kind {self.kind!r}")
        object.__setattr__(self, "probs", _check_distribution("probs", self.probs))
        size = len(self.probs)
        if self.gold_index is not None and not 0 <= self.gold_index < size:
            raise ValueError(f"gold_index {self.gold_index} outside 0..{size - 1}")
        for name in ("gold_probs", "teacher_probs"):
            ref = getattr(self, name)
            if ref is None:
                continue
            ref = _check_distribution(name, ref)
            if len(ref) != size:
                raise ValueError(f"{name} has {len(ref)} entries, probs has {size}")
            object.__setattr__(self, name, ref)


def choice_item(
    probs: Sequence[float],
    *,
    gold_index: int | None = None,
    gold_probs: Sequence[float] | None = None,
    teacher_probs: Sequence[float] | None = None,
) -> PredItem:
    return _build("choice", probs, gold_index, gold_probs, teacher_probs)


def score_item(
    probs: Sequence[float],
    *,
    gold_level: int | None = None,
    gold_probs: Sequence[float] | None = None,
    teacher_probs: Sequence[float] | None = None,
) -> PredItem:
    return _build("score", probs, gold_level, gold_probs, teacher_probs)


def noul_item(
    p_true: float,
    *,
    gold_true: bool | None = None,
    gold_probability_true: float | None = None,
    teacher_probability_true: float | None = None,
) -> PredItem:
    def pair(name: str, value: float) -> tuple[float, float]:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return (1.0 - float(value), float(value))

    return PredItem(
        "noul",
        pair("p_true", p_true),
        None if gold_true is None else int(bool(gold_true)),
        (
            None
            if gold_probability_true is None
            else pair("gold_probability_true", gold_probability_true)
        ),
        (
            None
            if teacher_probability_true is None
            else pair("teacher_probability_true", teacher_probability_true)
        ),
    )


def _build(kind, probs, gold_index, gold_probs, teacher_probs) -> PredItem:
    return PredItem(
        kind,
        tuple(probs),
        gold_index,
        None if gold_probs is None else tuple(gold_probs),
        None if teacher_probs is None else tuple(teacher_probs),
    )


# ---------------------------------------------------------------------------
# Per-item and per-sample statistics
# ---------------------------------------------------------------------------


def argmax(probs: Sequence[float]) -> int:
    """Highest-probability index; a tie goes to the lowest index."""
    return min(range(len(probs)), key=lambda i: (-probs[i], i))


def _kl_bits(p: Sequence[float], q: Sequence[float]) -> float:
    return math.fsum(a * math.log2(a / b) for a, b in zip(p, q) if a > 0.0)


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, bounded to [0, 1]."""
    mid = [(a + b) / 2.0 for a, b in zip(p, q)]
    return 0.5 * _kl_bits(p, mid) + 0.5 * _kl_bits(q, mid)


def _gold_dist(item: PredItem) -> tuple[float, ...] | None:
    """Gold distribution: ``gold_probs`` if given, else one-hot on ``gold_index``."""
    if item.gold_probs is not None:
        return item.gold_probs
    if item.gold_index is None:
        return None
    return tuple(1.0 if i == item.gold_index else 0.0 for i in range(len(item.probs)))


def _correct(item: PredItem) -> bool:
    return argmax(item.probs) == item.gold_index


def _mean(values: Sequence[float]) -> float | None:
    return (math.fsum(values) / len(values)) if values else None


def _accuracy(sample: Sequence[PredItem]) -> float | None:
    return _mean([1.0 if _correct(i) else 0.0 for i in sample])


def _macro_f1(sample: Sequence[PredItem]) -> float | None:
    if not sample:
        return None
    gold = [i.gold_index for i in sample]
    pred = [argmax(i.probs) for i in sample]
    f1s = []
    for cls in sorted(set(gold) | set(pred)):
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return math.fsum(f1s) / len(f1s)


def _log_loss_one(item: PredItem) -> float:
    return -math.log(max(item.probs[item.gold_index], LOG_EPSILON))


def _brier(item: PredItem) -> float:
    return brier_one({str(i): p for i, p in enumerate(item.probs)}, str(item.gold_index))


def _jsd_gold(item: PredItem) -> float:
    return jensen_shannon(item.probs, _gold_dist(item))


def _jsd_teacher(item: PredItem) -> float:
    return jensen_shannon(item.probs, item.teacher_probs)


def _mean_of(fn: Callable[[PredItem], float]):
    return lambda sample: _mean([fn(i) for i in sample])


def _confident_wrong(threshold: float):
    def stat(sample: Sequence[PredItem]) -> float | None:
        flags = [1.0 if (max(i.probs) >= threshold and not _correct(i)) else 0.0 for i in sample]
        return _mean(flags)

    return stat


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def assess(
    items: Sequence[PredItem],
    *,
    confident_wrong_threshold: float,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Every metric per kind plus ``"all"``, each a ``bootstrap_ci`` dict.

    ``confident_wrong_threshold`` is required: it is a pre-registered value, never a default.
    An item lacking the reference a metric needs (gold, or a teacher distribution) is left out
    of that metric, and the reported ``n`` says how many were used.
    """
    if not 0.0 < confident_wrong_threshold <= 1.0:
        raise ValueError("confident_wrong_threshold must be within (0, 1]")

    def block(stat, needs: Callable[[PredItem], bool]) -> dict:
        out = {}
        for group in (*KINDS, "all"):
            pool = [i for i in items if (group == "all" or i.kind == group) and needs(i)]
            out[group] = bootstrap_ci(pool, stat, seed=seed, resamples=resamples)
        return out

    has_gold_index = lambda i: i.gold_index is not None  # noqa: E731
    has_gold_dist = lambda i: _gold_dist(i) is not None  # noqa: E731
    has_teacher = lambda i: i.teacher_probs is not None  # noqa: E731
    return {
        "accuracy": block(_accuracy, has_gold_index),
        "macro_f1": block(_macro_f1, has_gold_index),
        "jsd_gold": block(_mean_of(_jsd_gold), has_gold_dist),
        "jsd_teacher": block(_mean_of(_jsd_teacher), has_teacher),
        "confident_wrong": block(_confident_wrong(confident_wrong_threshold), has_gold_index),
        "confident_wrong_threshold": confident_wrong_threshold,
        "log_loss": block(_mean_of(_log_loss_one), has_gold_index),
        "brier": block(_mean_of(_brier), has_gold_index),
        "seed": seed,
        "resamples": resamples,
    }
