"""The uncertainty gate: turn a scorer's distribution into a decision.

Every scorer already produces, per request, a normalised probability
distribution over the offered candidates (operation names plus the
``(explain)`` / ``(escalate)`` controls or ``escalate:<reason>`` labels) and
an argmax choice. This module adds a confidence gate on top of that argmax:
a scorer that is *right on average* can still be wrong on a specific request,
and a mutating operation proposed with low confidence is a worse outcome than
asking the operator to look. :func:`decide` never generates a new choice --
it only decides whether to trust the argmax, override it with a semantic
escalation, or decline it as ``abstain_uncertain`` (a confidence decision, as
opposed to ``escalate``, a semantic one the scorer itself picked).

It needs only the distribution and the Domain: no tokenizer, no model and no
backbone adapter.

The decision, in order:

1. Every ``escalate:<reason>`` label is rolled into the bare ``(escalate)``
   label first (:func:`jev_factory.core.metrics.rollup_escalate_candidates`).
2. **Semantic escalate**: if escalate is the argmax, or the rolled escalate
   mass is at or above ``thresholds.escalate`` (when set), the decision is
   ``escalate``.
3. Otherwise take the argmax (``top1``). If it is the explain control, the
   decision is ``explain``.
4. Otherwise top1 names an operation. Which threshold set applies --
   :attr:`Thresholds.read_only` or :attr:`Thresholds.mutating` -- is decided
   from the Domain (:meth:`Domain.is_mutating`), **never from the operation's
   name**; an operation the Domain does not know is gated as mutating, the
   stricter set. The decision is ``abstain_uncertain`` when top1's probability
   is below the set's ``floor``, or its margin over the second place is below
   the set's ``margin``, or the rolled distribution's normalised entropy is
   above the set's ``max_entropy``; otherwise it is ``propose``. A threshold
   left as ``None`` never fires (an all-``None`` :class:`Thresholds`
   reproduces the bare argmax).

Normalised entropy is the rolled distribution's Shannon entropy divided by
``log(n)``, where ``n`` is the number of offered labels after the
escalate-reason roll-up -- 0 for a certain answer, 1 for a uniform one. With
at most one offered candidate it is 0.

Ties go to whichever candidate appears earlier in *offered*, matching a
scorer's own ``max(labels, key=...)`` tie-break, so :func:`decide` reproduces
a stored prediction's argmax exactly given the same distribution and order.

Stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from jev_factory.core import metrics
from jev_factory.core.predictions import Prediction
from jev_factory.domain.model import ESCALATE_LABEL, EXPLAIN_LABEL, Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/gate.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 62-85: sys.path hack, nvsh.ops.table / nvsh.tiers.bench imports and the"
        " _sibling importlib path-loader removed; metrics is a normal package import",
        "lines 88-89: ESCALATE_LABEL/EXPLAIN_LABEL come from jev_factory.domain.model",
        "lines 270-274: ops_table.get(top1).read_only becomes Domain.is_mutating(top1) (an"
        " unknown operation stays mutating); decide() takes the Domain as a 4th argument",
        "added decide_prediction(): the gate over a predictions record, in its offered order",
    ],
    "licence": "Apache-2.0",
}

#: Decision outcomes, in the order :func:`decide` can produce them.
OUTCOMES = ("propose", "explain", "escalate", "abstain_uncertain")

__all__ = [
    "ESCALATE_LABEL",
    "EXPLAIN_LABEL",
    "OUTCOMES",
    "GateError",
    "ThresholdSet",
    "Thresholds",
    "Decision",
    "decide",
    "decide_prediction",
    "normalized_entropy",
]


class GateError(ValueError):
    """A malformed threshold payload or an empty distribution."""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSet:
    """One operation class's abstention thresholds. A ``None`` field never fires.

    ``floor``: minimum top1 probability to propose.
    ``margin``: minimum top1-minus-top2 probability gap to propose.
    ``max_entropy``: maximum normalised entropy (0..1) to propose.
    """

    floor: float | None = None
    margin: float | None = None
    max_entropy: float | None = None

    def to_json(self) -> dict:
        return {"floor": self.floor, "margin": self.margin, "max_entropy": self.max_entropy}

    @classmethod
    def from_json(cls, data: Mapping) -> "ThresholdSet":
        if not isinstance(data, Mapping):
            raise GateError(f"a threshold set must be an object, got {data!r}")
        return cls(
            floor=_optional_float(data.get("floor"), "floor"),
            margin=_optional_float(data.get("margin"), "margin"),
            max_entropy=_optional_float(data.get("max_entropy"), "max_entropy"),
        )


@dataclass(frozen=True)
class Thresholds:
    """The gate's full configuration: one semantic-escalate floor, two class sets.

    ``read_only`` and ``mutating`` are chosen by the Domain's ``read_only`` flag
    for the argmax operation -- never by its name. ``Thresholds()`` (every
    field ``None``) disables every check.
    """

    escalate: float | None = None
    read_only: ThresholdSet = field(default_factory=ThresholdSet)
    mutating: ThresholdSet = field(default_factory=ThresholdSet)

    def to_json(self) -> dict:
        return {
            "escalate": self.escalate,
            "read_only": self.read_only.to_json(),
            "mutating": self.mutating.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping) -> "Thresholds":
        if not isinstance(data, Mapping):
            raise GateError(f"thresholds must be an object, got {data!r}")
        return cls(
            escalate=_optional_float(data.get("escalate"), "escalate"),
            read_only=ThresholdSet.from_json(data.get("read_only", {})),
            mutating=ThresholdSet.from_json(data.get("mutating", {})),
        )


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name} must be a number or null, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """One :func:`decide` result.

    ``outcome`` is one of :data:`OUTCOMES`. ``label`` is the candidate the
    decision is about: the argmax operation for ``propose`` and
    ``abstain_uncertain``, :data:`ESCALATE_LABEL` / :data:`EXPLAIN_LABEL` for
    the controls. ``reason`` names which check fired for ``abstain_uncertain``
    (``"floor"``, ``"margin"`` or ``"entropy"``), or ``"threshold"`` /
    ``"argmax"`` for ``escalate``; ``None`` otherwise.
    """

    outcome: str
    label: str
    reason: str | None = None


def _canonical_order(offered: Sequence[str], rolled: Mapping[str, float]) -> list[str]:
    """*offered*'s labels, canonicalised and de-duplicated, in first-seen order.

    Any label present only in *rolled* is appended at the end so it can
    still be found.
    """
    order: list[str] = []
    seen: set[str] = set()
    for name in offered:
        canon = metrics.canonical_label(name)
        if canon not in seen:
            seen.add(canon)
            order.append(canon)
    for name in rolled:
        if name not in seen:
            seen.add(name)
            order.append(name)
    return order


def _argmax(rolled: Mapping[str, float], offered: Sequence[str]) -> tuple[str, float]:
    """The highest-probability label in *rolled*; ties go to the earlier *offered* entry."""
    best_label: str | None = None
    best_p = -1.0
    for name in _canonical_order(offered, rolled):
        p = rolled.get(name, 0.0)
        if p > best_p:
            best_p = p
            best_label = name
    if best_label is None:  # pragma: no cover - rolled is non-empty (checked by decide)
        raise GateError("distribution must not be empty")
    return best_label, best_p


def _second_place(rolled: Mapping[str, float], top_label: str) -> float:
    """The highest probability among *rolled*'s labels other than *top_label*."""
    rest = [p for name, p in rolled.items() if name != top_label]
    return max(rest) if rest else 0.0


def normalized_entropy(rolled: Mapping[str, float], n_offered: int) -> float:
    """Shannon entropy of *rolled*, divided by ``log(n_offered)``; 0 when ``n_offered <= 1``."""
    if n_offered <= 1:
        return 0.0
    entropy = -math.fsum(p * math.log(p) for p in rolled.values() if p > 0.0)
    return entropy / math.log(n_offered)


def decide(
    distribution: Mapping[str, float],
    offered: Sequence[str],
    thresholds: Thresholds,
    domain: Domain,
) -> Decision:
    """Gate one request's scored distribution into a :class:`Decision`.

    *distribution* is a label -> probability mapping (a predictions record's
    ``candidates``: non-empty, summing to ~1). *offered* is the labels
    actually offered, in offered order. *domain* says which operations are
    read-only. See the module docstring for the decision order.
    """
    if not distribution:
        raise GateError("distribution must not be empty")
    rolled = metrics.rollup_escalate_candidates(distribution)
    top1_label, p_top1 = _argmax(rolled, offered)

    p_escalate = rolled.get(ESCALATE_LABEL, 0.0)
    if top1_label == ESCALATE_LABEL:
        return Decision("escalate", ESCALATE_LABEL, "argmax")
    if thresholds.escalate is not None and p_escalate >= thresholds.escalate:
        return Decision("escalate", ESCALATE_LABEL, "threshold")

    if top1_label == EXPLAIN_LABEL:
        return Decision("explain", EXPLAIN_LABEL)

    # An operation missing from the Domain is gated as mutating, the stricter set.
    mutating = domain.is_mutating(top1_label)
    thresholds_for_class = thresholds.mutating if mutating else thresholds.read_only

    margin = p_top1 - _second_place(rolled, top1_label)
    # Normalise over the offered candidates after the escalate:<reason> roll-up,
    # so a uniform rolled distribution reads 1.0 in reasons mode too.
    entropy = normalized_entropy(rolled, len({metrics.canonical_label(o) for o in offered}))

    if thresholds_for_class.floor is not None and p_top1 < thresholds_for_class.floor:
        return Decision("abstain_uncertain", top1_label, "floor")
    if thresholds_for_class.margin is not None and margin < thresholds_for_class.margin:
        return Decision("abstain_uncertain", top1_label, "margin")
    if thresholds_for_class.max_entropy is not None and entropy > thresholds_for_class.max_entropy:
        return Decision("abstain_uncertain", top1_label, "entropy")
    return Decision("propose", top1_label)


def decide_prediction(
    prediction: Prediction, thresholds: Thresholds, domain: Domain
) -> Decision | None:
    """:func:`decide` over a predictions record's ``candidates`` in its offered order.

    ``None`` when the record carries no distribution (nothing to gate).
    """
    if prediction.candidates is None:
        return None
    return decide(prediction.candidates, prediction.offered_order(), thresholds, domain)
