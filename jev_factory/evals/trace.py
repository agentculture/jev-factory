"""Trace schema: raw model output and per-policy final decisions, side by side.

A :class:`Trace` is one evaluation case: the model's raw, unpolicied answer
(:class:`RawRecord`) plus zero or more policies' *final* decisions layered on
top of it. Policies never touch ``raw`` in place: each
:meth:`Trace.with_policy` call returns a new ``Trace`` (both are frozen), so
the raw record a report attributes to a model is provably the same one a
later policy comparison used.

``RawRecord`` is built from either of two sources the gate scores side by
side:

* a **predictions line** (:mod:`jev_factory.core.predictions`: ``outcome``,
  ``operation``, ``arguments``, ``candidates`` or ``None``, ``tokens``,
  ``ttfd_ms``/``latency_ms``, optional ``invalid_reason``/``offered``);
* a **provider answer**: a reference model's choice, parsed into the same
  vocabulary, plus which provider and model answered and the model id the
  provider actually returned.

Trace files are private run output: :func:`write_traces` refuses a path
inside a git worktree, judged by :func:`inside_git_worktree`
(``git rev-parse --is-inside-work-tree`` through
:func:`jev_factory.factory.workroot.resolve_work_root`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core.predictions import Prediction, PredictionError
from jev_factory.factory.workroot import resolve_work_root

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/trace.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 44-63: the path-loaded scripts/lfm-finetune/metrics.py is replaced by"
        " jev_factory.core.predictions (Prediction, PredictionError for MetricsError)",
        "lines 327-337: _is_inside_git_worktree walked up looking for a .git entry, which"
        " tripped on an empty .git directory (nvsh#71 item 2); inside_git_worktree now asks"
        " git rev-parse --is-inside-work-tree via jev_factory.factory.workroot",
        "RawRecord gains the optional offered order of a predictions line (the gate's"
        " tie-break order); the Track A-only inspections field (deviation d9) is dropped",
        "RawRecord.from_provider_answer accepts the choice interface only (Track A deferred)",
    ],
    "licence": "Apache-2.0",
}

__all__ = [
    "PredictionError",
    "RawRecord",
    "PolicyResult",
    "Trace",
    "TraceWriteError",
    "inside_git_worktree",
    "append_trace",
    "write_traces",
    "read_traces",
]


# ---------------------------------------------------------------------------
# RawRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    """The unpolicied model output for one case.

    ``tokens``/``ttfd_ms``/``latency_ms`` come from a predictions line and are
    ``None`` for a provider answer; ``provider``/``model``/``returned_model``/
    ``interface`` come from a provider answer and are ``None`` for a
    predictions line. ``offered`` is a predictions line's offered order, when
    it recorded one.
    """

    outcome: str
    operation: str | None
    arguments: dict | None
    candidates: dict | None
    invalid_reason: str | None = None
    tokens: int | None = None
    ttfd_ms: float | None = None
    latency_ms: float | None = None
    provider: str | None = None
    model: str | None = None
    returned_model: str | None = None
    interface: str | None = None
    offered: tuple[str, ...] | None = None

    @classmethod
    def from_prediction(cls, prediction: Prediction) -> "RawRecord":
        """Build a ``RawRecord`` from a validated :class:`Prediction`."""
        return cls(
            outcome=prediction.outcome,
            operation=prediction.operation,
            arguments=None if prediction.arguments is None else dict(prediction.arguments),
            candidates=None if prediction.candidates is None else dict(prediction.candidates),
            invalid_reason=prediction.invalid_reason,
            tokens=prediction.tokens,
            ttfd_ms=prediction.ttfd_ms,
            latency_ms=prediction.latency_ms,
            offered=None if prediction.offered is None else tuple(prediction.offered),
        )

    @classmethod
    def from_prediction_line(cls, row: Mapping[str, Any]) -> "RawRecord":
        """Validate and build a ``RawRecord`` straight from a decoded predictions line."""
        return cls.from_prediction(Prediction.from_dict(dict(row)))

    @classmethod
    def from_provider_answer(
        cls,
        *,
        provider: str,
        model: str,
        returned_model: str | None,
        interface: str,
        outcome: str,
        operation: str | None = None,
        arguments: dict | None = None,
        candidates: dict | None = None,
        invalid_reason: str | None = None,
    ) -> "RawRecord":
        """Build a ``RawRecord`` from a provider's parsed choice answer.

        ``candidates`` is ``None`` when the provider returned no logprobs.
        """
        if interface != "choice":
            raise ValueError(f"interface must be 'choice', got {interface!r}")
        return cls(
            outcome=outcome,
            operation=operation,
            arguments=None if arguments is None else dict(arguments),
            candidates=None if candidates is None else dict(candidates),
            invalid_reason=invalid_reason,
            provider=provider,
            model=model,
            returned_model=returned_model,
            interface=interface,
        )

    def offered_order(self) -> list[str]:
        """The offered labels in offered order: ``offered``, else ``candidates``' key order."""
        if self.offered is not None:
            return list(self.offered)
        return list(self.candidates or ())

    def to_dict(self) -> dict:
        """The full record, for JSONL serialisation (``offered`` only when recorded)."""
        out = {
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "candidates": None if self.candidates is None else dict(self.candidates),
            "invalid_reason": self.invalid_reason,
            "tokens": self.tokens,
            "ttfd_ms": self.ttfd_ms,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model": self.model,
            "returned_model": self.returned_model,
            "interface": self.interface,
        }
        if self.offered is not None:
            out["offered"] = list(self.offered)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawRecord":
        """Inverse of :meth:`to_dict`, for reading a written trace back."""
        fields = dict(data)
        if fields.get("offered") is not None:
            fields["offered"] = tuple(fields["offered"])
        return cls(**fields)

    def to_prediction_dict(self, case_id: str, expected: Mapping[str, Any]) -> dict:
        """A complete predictions line rebuilt from this record plus *case_id*/*expected*.

        A provider answer carries no timing figures; they become zero, which
        :class:`Prediction` accepts and no gate figure reads.
        """
        result: dict[str, Any] = {
            "id": case_id,
            "expected": dict(expected),
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "candidates": None if self.candidates is None else dict(self.candidates),
            "tokens": 0 if self.tokens is None else self.tokens,
            "ttfd_ms": 0.0 if self.ttfd_ms is None else self.ttfd_ms,
            "latency_ms": 0.0 if self.latency_ms is None else self.latency_ms,
        }
        if self.invalid_reason is not None:
            result["invalid_reason"] = self.invalid_reason
        if self.offered is not None and self.candidates is not None:
            result["offered"] = list(self.offered)
        return result


# ---------------------------------------------------------------------------
# Per-policy final decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    """One policy's final decision on a case."""

    decision: str
    reason: str

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyResult":
        return cls(decision=data["decision"], reason=data["reason"])


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """One evaluation case: raw model output plus each policy's final call on it."""

    case_id: str
    split: str
    raw: RawRecord
    final: Mapping[str, PolicyResult] = field(default_factory=dict)
    ground_truth: Mapping[str, Any] | None = None
    subject: str | None = None

    @classmethod
    def from_prediction_line(
        cls,
        row: Mapping[str, Any],
        *,
        split: str,
        subject: str | None = None,
    ) -> "Trace":
        """Build a ``Trace`` straight from a decoded predictions line."""
        prediction = Prediction.from_dict(dict(row))
        return cls(
            case_id=prediction.id,
            split=split,
            raw=RawRecord.from_prediction(prediction),
            ground_truth=dict(prediction.expected),
            subject=subject,
        )

    def with_policy(self, policy: str, decision: str, reason: str) -> "Trace":
        """A new ``Trace`` with one more policy's final decision; ``self`` is untouched."""
        new_final = dict(self.final)
        new_final[policy] = PolicyResult(decision=decision, reason=reason)
        return replace(self, final=new_final)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "raw": self.raw.to_dict(),
            "final": {name: result.to_dict() for name, result in self.final.items()},
            "ground_truth": None if self.ground_truth is None else dict(self.ground_truth),
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trace":
        return cls(
            case_id=data["case_id"],
            split=data["split"],
            raw=RawRecord.from_dict(data["raw"]),
            final={
                name: PolicyResult.from_dict(result)
                for name, result in data.get("final", {}).items()
            },
            ground_truth=data.get("ground_truth"),
            subject=data.get("subject"),
        )


# ---------------------------------------------------------------------------
# JSONL writer -- private run dir only, never inside a git worktree
# ---------------------------------------------------------------------------


class TraceWriteError(ValueError):
    """Raised when asked to write trace output to a path inside a git worktree."""


def inside_git_worktree(path: str | Path) -> bool:
    """True when *path* (or where it would be created) is inside a git worktree.

    Asked of git itself (``git rev-parse --is-inside-work-tree`` from the
    nearest existing ancestor, through
    :func:`~jev_factory.factory.workroot.resolve_work_root`), so a directory
    holding an empty ``.git`` (not a repository) is not refused. A missing
    git is an environment error and propagates as :class:`CliError`.
    """
    try:
        resolve_work_root(path)
    except CliError as exc:
        if exc.code == EXIT_USER_ERROR:
            return True
        raise
    return False


def _dump_trace(trace: Trace) -> str:
    """One JSON line, keys in insertion order.

    ``sort_keys=True`` would resort ``raw.candidates``, whose order is the
    offered order the gate breaks ties by; insertion order is deterministic
    for the same ``Trace`` and keeps it.
    """
    return json.dumps(trace.to_dict())


def append_trace(path: str | Path, trace: Trace) -> None:
    """Append one ``Trace`` as a JSONL line; refuses a path inside a git worktree."""
    target = Path(path)
    if inside_git_worktree(target):
        raise TraceWriteError(f"refusing to write a trace file inside a git worktree: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(_dump_trace(trace))
        handle.write("\n")


def write_traces(path: str | Path, traces: Iterable[Trace]) -> None:
    """Write ``traces`` to a fresh JSONL file; refuses a path inside a git worktree."""
    target = Path(path)
    if inside_git_worktree(target):
        raise TraceWriteError(f"refusing to write a trace file inside a git worktree: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(_dump_trace(trace))
            handle.write("\n")


def read_traces(path: str | Path) -> list[Trace]:
    """Read a JSONL trace file back into a list of :class:`Trace` objects."""
    traces: list[Trace] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            traces.append(Trace.from_dict(json.loads(line)))
    return traces
