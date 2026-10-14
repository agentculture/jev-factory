"""Correction records: a human-or-model proposed fix to one question's gold (distillery t11).

Corrections live in their own JSONL, apart from the records they correct. A
correction carries the record id, question id, a target (a label and/or a full
distribution over the question's options), who or what produced it (``author``),
when (``date``), why (``reason``: ``weak`` or ``outdated``), its ``status`` and
the record ``text`` it was drafted against (kept for the leakage check).

Intake is leakage-checked with :mod:`jev_factory.core.leakage`: a correction
whose text exactly or near-duplicates a fit/selection, test or held-out item is
refused. Status changes come only from the review flow (a later task); nothing
here marks a correction approved, and drafting always yields ``proposed``.
Stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.core.leakage import match
from jev_factory.distill.records import Record, RecordError

REASONS: frozenset[str] = frozenset({"weak", "outdated"})
STATUSES: frozenset[str] = frozenset({"proposed", "approved", "edited", "rejected"})
DRAFT_STATUS = "proposed"


class CorrectionError(RecordError):
    code = "correction_error"


class MissingCorrectionField(CorrectionError):
    code = "missing_correction_field"


class UnknownReason(CorrectionError):
    code = "unknown_reason"


class UnknownStatus(CorrectionError):
    code = "unknown_status"


class BadTarget(CorrectionError):
    code = "bad_target"


class LeakageRefused(CorrectionError):
    code = "correction_leakage"


@dataclass(frozen=True)
class Correction:
    """One correction; validated on construction."""

    record_id: str
    question_id: str
    author: str
    date: str
    reason: str
    status: str = DRAFT_STATUS
    label: str | None = None
    distribution: Mapping[str, float] | None = None
    text: str = ""

    def __post_init__(self) -> None:
        if self.distribution is not None:
            object.__setattr__(self, "distribution", dict(self.distribution))
        self.validate()

    def validate(self) -> None:
        for name in ("record_id", "question_id", "author", "date", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MissingCorrectionField(f"correction field {name!r} is empty")
        if self.reason not in REASONS:
            raise UnknownReason(f"{self.reason!r} not in {sorted(REASONS)}")
        if self.status not in STATUSES:
            raise UnknownStatus(f"{self.status!r} not in {sorted(STATUSES)}")
        if self.label is None and not self.distribution:
            raise BadTarget(
                f"correction for {self.record_id!r}/{self.question_id!r} has no label or"
                " distribution"
            )
        if self.distribution is not None and any(
            not 0.0 <= float(v) <= 1.0 for v in self.distribution.values()
        ):
            raise BadTarget("distribution probability outside [0, 1]")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_id": self.record_id,
            "question_id": self.question_id,
            "author": self.author,
            "date": self.date,
            "reason": self.reason,
            "status": self.status,
            "text": self.text,
        }
        if self.label is not None:
            out["label"] = self.label
        if self.distribution is not None:
            out["distribution"] = dict(self.distribution)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Correction:
        for key in ("record_id", "question_id", "author", "date", "reason"):
            if key not in data:
                raise MissingCorrectionField(f"correction json lacks {key!r}")
        return cls(
            record_id=str(data["record_id"]),
            question_id=str(data["question_id"]),
            author=str(data["author"]),
            date=str(data["date"]),
            reason=str(data["reason"]),
            status=str(data.get("status", DRAFT_STATUS)),
            label=None if data.get("label") is None else str(data["label"]),
            distribution=data.get("distribution"),
            text=str(data.get("text", "")),
        )


# ---------------------------------------------------------------------------
# Leakage-checked intake
# ---------------------------------------------------------------------------

#: Protected items by split name: ``{"test": [{"id": ..., "text": ...}], ...}``.
Protected = Mapping[str, Iterable[Mapping[str, Any]]]


def check_leakage(correction: Correction, protected: Protected) -> None:
    """Refuse *correction* if its text exactly/near-duplicates any protected item.

    Names the protected split, item id and kind only, never a text. A correction
    with no text to check is refused too: intake fails closed.
    """
    if not correction.text.strip():
        raise LeakageRefused(f"correction for {correction.record_id!r} has no text to check")
    for name, items in protected.items():
        for item in items:
            kind = match(correction.text, str(item.get("text", "")))
            if kind is not None:
                raise LeakageRefused(
                    f"correction for {correction.record_id!r} is a {kind} of {name} item"
                    f" {item.get('id')!r}"
                )


def intake(correction: Correction, protected: Protected) -> Correction:
    """Return *correction* once it passes the leakage check; raise otherwise."""
    check_leakage(correction, protected)
    return correction


def write_corrections(path: str | Path, corrections: Iterable[Correction]) -> int:
    """Append *corrections* to the JSONL at *path*; returns how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(c.to_json(), sort_keys=True, ensure_ascii=False) for c in corrections]
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")
    return len(rows)


def read_corrections(path: str | Path) -> list[Correction]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        Correction.from_json(json.loads(line))
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

#: A drafter maps a record to ``{qid: {"label": str} and/or {"distribution": {...}}}``.
Drafter = Callable[[Record], Mapping[str, Mapping[str, Any]]]


@dataclass
class DraftResult:
    proposed: list[Correction] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)  # (record id, reason)


def systemone_drafter(client: Any) -> Drafter:
    """The default drafter: an alternative teacher over ``/v1/systemone``.

    *client* is a :class:`~jev_factory.distill.systemone.SystemOneClient` (or any
    object with ``ask(record_dict) -> CallResult``). Questions the teacher failed
    on are left out, so no correction is drafted for them.
    """

    def draft(record: Record) -> dict[str, dict[str, Any]]:
        body = {
            "id": record.id,
            "state": record.state,
            "questions": {qid: q.to_json() for qid, q in record.questions.items()},
        }
        out: dict[str, dict[str, Any]] = {}
        for result in client.ask(body).questions:
            if not result.ok:
                continue
            if result.distribution:
                top = max(result.distribution, key=result.distribution.__getitem__)
                out[result.qid] = {"label": top, "distribution": result.distribution}
            elif result.p_true is not None:
                dist = {"false": 1.0 - result.p_true, "true": result.p_true}
                label = "true" if result.p_true >= 0.5 else "false"
                out[result.qid] = {"label": label, "distribution": dist}
        return out

    return draft


def _correction_from_draft(
    record: Record, qid: str, target: Mapping[str, Any], author: str, date: str, reason: str
) -> Correction:
    question = record.questions.get(qid)
    if question is None:
        raise BadTarget(f"record {record.id!r} has no question {qid!r}")
    names = question.names
    label = target.get("label")
    dist = target.get("distribution")
    stray = ([label] if label is not None and label not in names else []) + sorted(
        set(dist or {}) - set(names)
    )
    if stray:
        raise BadTarget(f"{stray} not in {list(names)} for {record.id!r}/{qid!r}")
    return Correction(
        record_id=record.id,
        question_id=qid,
        author=author,
        date=date,
        reason=reason,
        status=DRAFT_STATUS,  # never taken from the drafter
        label=None if label is None else str(label),
        distribution=dist,
        text=record.state,
    )


def draft_corrections(
    rows: Iterable[Record],
    drafter: Drafter,
    protected: Protected,
    *,
    author: str,
    date: str,
    reason: str = "weak",
) -> DraftResult:
    """Turn teacher-weak *rows* into ``proposed`` corrections via *drafter*.

    Every result is ``proposed``, whatever the drafter returns: approval is the
    review flow's job. A draft naming an unknown question or option, or failing
    the leakage check, is refused and reported rather than raised.
    """
    result = DraftResult()
    for record in rows:
        try:
            drafts = drafter(record)
        except Exception as exc:  # noqa: BLE001 - an injected drafter may fail any way
            result.refused.append((record.id, f"drafter failed: {type(exc).__name__}"))
            continue
        for qid, target in drafts.items():
            try:
                correction = _correction_from_draft(record, qid, target, author, date, reason)
                result.proposed.append(intake(correction, protected))
            except CorrectionError as exc:
                result.refused.append((record.id, str(exc)))
    return result
