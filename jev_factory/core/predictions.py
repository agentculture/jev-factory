"""The backbone-agnostic predictions record: what every scorer emits and every
distribution-only stage reads.

One JSON object per line (JSONL), one line per corpus entry, in the order the
entries were run (the first line is the cold decision). A causal-LM letter
readout, an encoder that scores candidates natively (nvsh#67's GLiNER) or a
generative baseline all write this same record, so metrics, calibration, the
gate sweep, selection and ``jev decide`` compare models rather than scorers.
Nothing here knows about tokens, letters, tokenizers or models.

Required fields (nvsh ``metrics.py``'s schema, unchanged):

``id``
    The corpus entry's id (unique within a file). An id ending ``-nocand``
    marks a missing-candidate row (the gold candidate was withheld).
``expected``
    The entry's gold ``expect`` block: ``{"operation": name, "args": {...}}``,
    ``{"escalate": true}`` or ``{"explain": true}``.
``outcome``
    The final decision: ``"propose"``, ``"explain"``, ``"escalate"``,
    ``"abstain_uncertain"`` (a confidence gate declined), or ``"invalid"``
    when the output could not be read as a decision.
``operation`` / ``arguments``
    The proposed operation and its (grounded) arguments; ``null`` unless
    ``outcome`` is ``"propose"``.
``candidates``
    The probability distribution the decision was made from, over the labels
    offered: ``{label: p}`` with ``p`` in [0, 1] summing to 1, where a label
    is an operation name, :data:`~jev_factory.domain.model.EXPLAIN_LABEL`,
    :data:`~jev_factory.domain.model.ESCALATE_LABEL` or ``escalate:<reason>``.
    When a calibration was applied these are the *calibrated* probabilities.
    ``null`` when the run recorded none (left out of calibration, counted).
``tokens``
    Tokens generated for this decision (0 for a scorer).
``ttfd_ms`` / ``latency_ms``
    Time to first decision and total decision time, in milliseconds.

Optional fields (added by jev-factory; absent or ``null`` when not recorded):

``offered``
    The labels offered, in the order they were offered (the gate breaks
    probability ties by this order). Defaults to ``candidates``' key order.
``raw_scores``
    The backbone's own raw score per offered label before any normalisation
    (a letter's summed log-probability, an encoder's logit, ...). Any finite
    number.
``raw_probabilities``
    The distribution *before* calibration, same shape as ``candidates``.
    Raw and calibrated probabilities are each recorded, never conflated.
``grounded``
    Whether the proposed operation's arguments were grounded against the
    world outside the model. ``true``/``false`` for a proposal; ``null`` for
    other outcomes or when a re-made decision was never grounded.
``invalid_reason``
    Why an ``"invalid"`` line is invalid (``"unparseable"`` when absent).

Unknown keys are ignored on read, so a backbone may add its own diagnostics.
Stdlib only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/metrics.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 17-50 (schema docstring), 130-148 (FIELDS/OUTCOMES/SUM_TOLERANCE),"
        " 185-300 (MetricsError, Prediction.from_dict, _check_*, read_predictions)"
        " lifted into this module so every backbone emits one record",
        "MetricsError becomes PredictionError (metrics.MetricsError aliases it)",
        "added optional offered, raw_scores, raw_probabilities and grounded fields,"
        " validated against candidates; to_dict/write_predictions added",
        "control labels come from jev_factory.domain.model instead of nvsh.tiers.bench",
    ],
    "licence": "Apache-2.0",
}

#: The required fields, in documented order.
FIELDS = (
    "id",
    "expected",
    "outcome",
    "operation",
    "arguments",
    "candidates",
    "tokens",
    "ttfd_ms",
    "latency_ms",
)
#: The optional fields jev-factory adds, in documented order.
OPTIONAL_FIELDS = ("offered", "raw_scores", "raw_probabilities", "grounded", "invalid_reason")
OUTCOMES = ("propose", "explain", "escalate", "abstain_uncertain", "invalid")

#: How far a distribution's sum may stray from 1 (JSON float round trip).
SUM_TOLERANCE = 1e-4


class PredictionError(ValueError):
    """A predictions record that does not follow the schema above."""


@dataclass(frozen=True)
class Prediction:
    """One line of a predictions file (see the module docstring)."""

    id: str
    expected: dict
    outcome: str
    operation: str | None
    arguments: dict | None
    candidates: dict | None
    tokens: int
    ttfd_ms: float
    latency_ms: float
    invalid_reason: str | None = None
    offered: tuple[str, ...] | None = None
    raw_scores: dict | None = None
    raw_probabilities: dict | None = None
    grounded: bool | None = None

    @classmethod
    def from_dict(cls, row: object) -> "Prediction":
        """Validate one decoded line; raises :class:`PredictionError` naming the problem."""
        _check_row(row)
        _check_expected(row["expected"])
        outcome = row["outcome"]
        if outcome not in OUTCOMES:
            raise PredictionError(f"outcome {outcome!r} is not one of {', '.join(OUTCOMES)}")
        _check_proposal(outcome, row["operation"], row["arguments"])
        _check_distribution(row["candidates"], "candidates")
        _check_costs(row)
        reason = row.get("invalid_reason")
        if reason is not None and not isinstance(reason, str):
            raise PredictionError("invalid_reason must be a string")
        offered = _check_offered(row.get("offered"), row["candidates"])
        labels = _labels(offered, row["candidates"])
        raw_scores = _check_raw_scores(row.get("raw_scores"), labels)
        raw_probabilities = row.get("raw_probabilities")
        _check_distribution(raw_probabilities, "raw_probabilities")
        _check_same_labels(raw_probabilities, labels, "raw_probabilities")
        grounded = row.get("grounded")
        if grounded is not None and not isinstance(grounded, bool):
            raise PredictionError("grounded must be true, false or null")
        return cls(
            id=row["id"],
            expected=dict(row["expected"]),
            outcome=outcome,
            operation=row["operation"],
            arguments=_copy(row["arguments"]),
            candidates=_copy(row["candidates"]),
            tokens=row["tokens"],
            ttfd_ms=float(row["ttfd_ms"]),
            latency_ms=float(row["latency_ms"]),
            invalid_reason=reason,
            offered=offered,
            raw_scores=_copy(raw_scores),
            raw_probabilities=_copy(raw_probabilities),
            grounded=grounded,
        )

    def to_dict(self) -> dict[str, Any]:
        """The JSON line: every required field, plus each optional field that is set."""
        row: dict[str, Any] = {
            "id": self.id,
            "expected": dict(self.expected),
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "candidates": None if self.candidates is None else dict(self.candidates),
            "tokens": self.tokens,
            "ttfd_ms": self.ttfd_ms,
            "latency_ms": self.latency_ms,
        }
        optional = {
            "offered": None if self.offered is None else list(self.offered),
            "raw_scores": self.raw_scores,
            "raw_probabilities": self.raw_probabilities,
            "grounded": self.grounded,
            "invalid_reason": self.invalid_reason,
        }
        row.update({key: value for key, value in optional.items() if value is not None})
        return row

    def offered_order(self) -> tuple[str, ...]:
        """The labels in offered order: :attr:`offered`, else ``candidates``' key order."""
        if self.offered is not None:
            return self.offered
        return tuple(self.candidates or ())


def _check_row(row: object) -> None:
    if not isinstance(row, dict):
        raise PredictionError("a line must be a JSON object")
    missing = [name for name in FIELDS if name not in row]
    if missing:
        raise PredictionError(f"missing field(s): {', '.join(missing)}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise PredictionError("id must be a non-empty string")


def _check_costs(row: dict) -> None:
    tokens = row["tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise PredictionError("tokens must be a non-negative integer")
    for name in ("ttfd_ms", "latency_ms"):
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise PredictionError(f"{name} must be a non-negative number")


def _copy(mapping: Mapping | None) -> dict | None:
    return None if mapping is None else dict(mapping)


def _check_expected(expected: object) -> None:
    if not isinstance(expected, dict):
        raise PredictionError("expected must be an object")
    kinds = [key for key in ("operation", "escalate", "explain") if expected.get(key)]
    if len(kinds) != 1:
        raise PredictionError("expected must name exactly one of operation, escalate, explain")
    if kinds == ["operation"]:
        if not isinstance(expected["operation"], str):
            raise PredictionError("expected operation must be a string")
        if not isinstance(expected.get("args", {}), dict):
            raise PredictionError("expected args must be an object")


def _check_proposal(outcome: str, operation: object, arguments: object) -> None:
    if outcome == "propose":
        if not isinstance(operation, str) or not operation:
            raise PredictionError("a propose line needs an operation string")
        if not isinstance(arguments, dict):
            raise PredictionError("a propose line needs an arguments object")
    elif operation is not None or arguments is not None:
        raise PredictionError(f"operation and arguments must be null for outcome {outcome!r}")


def _check_distribution(candidates: object, name: str) -> None:
    if candidates is None:
        return
    if not isinstance(candidates, dict) or not candidates:
        raise PredictionError(f"{name} must be a non-empty object or null")
    for label, value in candidates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictionError(f"{name}[{label!r}] must be a number")
        if not 0.0 <= value <= 1.0:
            raise PredictionError(f"{name}[{label!r}] = {value} is outside [0, 1]")
    total = math.fsum(candidates.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise PredictionError(f"{name} sum to {total:.6f}, not 1")


def _check_offered(offered: object, candidates: object) -> tuple[str, ...] | None:
    if offered is None:
        return None
    if (
        not isinstance(offered, (list, tuple))
        or not offered
        or not all(isinstance(label, str) and label for label in offered)
    ):
        raise PredictionError("offered must be a non-empty list of label strings or null")
    if len(set(offered)) != len(offered):
        raise PredictionError("offered must not repeat a label")
    result = tuple(offered)
    _check_same_labels(candidates, result, "candidates")
    return result


def _labels(offered: tuple[str, ...] | None, candidates: object) -> tuple[str, ...] | None:
    if offered is not None:
        return offered
    if isinstance(candidates, dict):
        return tuple(candidates)
    return None


def _check_same_labels(values: object, labels: Sequence[str] | None, name: str) -> None:
    if values is None or labels is None or not isinstance(values, dict):
        return
    if set(values) != set(labels):
        raise PredictionError(f"{name} labels differ from the offered labels")


def _check_raw_scores(raw: object, labels: Sequence[str] | None) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise PredictionError("raw_scores must be a non-empty object or null")
    for label, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictionError(f"raw_scores[{label!r}] must be a number")
        if not math.isfinite(value):
            raise PredictionError(f"raw_scores[{label!r}] must be finite")
    _check_same_labels(raw, labels, "raw_scores")
    return raw


def read_predictions(path: str | Path) -> list[Prediction]:
    """Read and validate a predictions file; blank lines are skipped."""
    predictions: list[Prediction] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prediction = Prediction.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PredictionError(f"{path}: line {number}: not JSON ({exc.msg})") from None
            except PredictionError as exc:
                raise PredictionError(f"{path}: line {number}: {exc}") from None
            if prediction.id in seen:
                raise PredictionError(f"{path}: line {number}: duplicate id {prediction.id!r}")
            seen.add(prediction.id)
            predictions.append(prediction)
    return predictions


def from_rows(rows: Iterable[Mapping[str, Any]]) -> list[Prediction]:
    """Validate in-memory rows (e.g. a scorer's output) with read_predictions' checks."""
    predictions: list[Prediction] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=1):
        try:
            prediction = Prediction.from_dict(dict(row) if isinstance(row, Mapping) else row)
        except PredictionError as exc:
            raise PredictionError(f"row {number}: {exc}") from None
        if prediction.id in seen:
            raise PredictionError(f"row {number}: duplicate id {prediction.id!r}")
        seen.add(prediction.id)
        predictions.append(prediction)
    return predictions


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    """Write *predictions* as JSONL, one :meth:`Prediction.to_dict` per line."""
    with open(path, "w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict()) + "\n")
