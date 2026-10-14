"""Operator review of distillery corrections (``jev review --corrections``, t14).

A correction (:mod:`jev_factory.distill.corrections`) is born ``proposed`` and a
proposed correction is never trainable. The operator approves, edits or rejects each
one. Those decisions live in the **corrections review file**
(``<corrections stem>.review.jsonl`` beside the corrections), which is the record:
append-only, one JSON line per decision, the latest for a ``record_id/question_id``
key is current. The ``status`` field inside the corrections file is never trusted
for trainability; only the review file is.

The site stages decisions in a draft file (``<stem>.review-draft.jsonl``) and
``--apply`` appends them to the record. Each record holds ``before`` (the correction as
it stood) and its sha256, so a correction changed since the decision shows as a
conflict and is not exported. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_factory.distill.corrections import BadTarget, Correction, CorrectionError

REVIEW_SUFFIX = ".review.jsonl"
DRAFT_SUFFIX = ".review-draft.jsonl"
ACTIONS = ("approve", "edit", "reject")
#: The correction status each action yields.
STATUS_OF = {"approve": "approved", "edit": "edited", "reject": "rejected"}
DEFAULT_REVIEWER = "operator"


class CorrectionReviewError(ValueError):
    """A bad decision or a malformed review file."""


def review_path(corrections: Path) -> Path:
    return corrections.with_name(corrections.stem + REVIEW_SUFFIX)


def draft_path(corrections: Path) -> Path:
    return corrections.with_name(corrections.stem + DRAFT_SUFFIX)


def key_of(record_id: str, question_id: str) -> str:
    return f"{record_id}/{question_id}"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def correction_sha256(correction: Correction) -> str:
    blob = json.dumps(correction.to_json(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current(corrections: Iterable[Correction]) -> dict[str, Correction]:
    """The correction per key; a later line for the same key supersedes an earlier one."""
    return {key_of(c.record_id, c.question_id): c for c in corrections}


# -- the review file -----------------------------------------------------------------------


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorrectionReviewError(f"{path}:{number}: not JSON ({exc.msg})") from None
        if (
            not isinstance(record, dict)
            or record.get("action") not in ACTIONS
            or not isinstance(record.get("key"), str)
        ):
            raise CorrectionReviewError(f"{path}:{number}: not a correction review record")
        out.append(record)
    return out


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def latest(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        out[str(record["key"])] = record
    return out


# -- decisions -----------------------------------------------------------------------------


def _edited_target(correction: Correction, after: object) -> tuple[str | None, dict | None]:
    if not isinstance(after, Mapping):
        raise CorrectionReviewError("an edit needs an 'after' with a label and/or distribution")
    label = after.get("label", correction.label)
    dist = after.get("distribution", correction.distribution)
    label = None if label is None else str(label)
    dist = None if dist is None else dict(dist)
    try:
        Correction(
            record_id=correction.record_id,
            question_id=correction.question_id,
            author=correction.author,
            date=correction.date,
            reason=correction.reason,
            label=label,
            distribution=dist,
        )
    except CorrectionError as exc:
        raise CorrectionReviewError(str(exc)) from None
    if dist and label is not None and label not in dist:
        raise CorrectionReviewError(f"label {label!r} is not in the edited distribution")
    if label == correction.label and dist == correction.distribution:
        raise CorrectionReviewError("the edit changes nothing")
    return label, dist


def make_record(
    corrections: Iterable[Correction], decision: Mapping[str, Any], *, at: str | None = None
) -> dict[str, Any]:
    """Check one operator decision and return its record. ``before`` comes from the file."""
    action = decision.get("action")
    if action not in ACTIONS:
        raise CorrectionReviewError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    key = key_of(str(decision.get("record_id", "")), str(decision.get("question_id", "")))
    target = current(corrections).get(key)
    if target is None:
        raise CorrectionReviewError(f"no correction for {key}")
    after: dict[str, Any] | None = None
    if action == "edit":
        label, dist = _edited_target(target, decision.get("after"))
        after = {"label": label, "distribution": dist}
    reviewer = str(decision.get("reviewer") or DEFAULT_REVIEWER).strip() or DEFAULT_REVIEWER
    return {
        "at": at or now_utc(),
        "action": action,
        "key": key,
        "reviewer": reviewer,
        "note": str(decision.get("note") or "").strip(),
        "before": target.to_json(),
        "after": after,
        "correction_sha256": correction_sha256(target),
    }


# -- plan and export -----------------------------------------------------------------------


def status_of(record: Mapping[str, Any] | None) -> str:
    return "proposed" if record is None else STATUS_OF[str(record["action"])]


def is_stale(correction: Correction, record: Mapping[str, Any]) -> bool:
    return record.get("correction_sha256") != correction_sha256(correction)


@dataclass
class Plan:
    counts: dict[str, int] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts


def plan(corrections: Iterable[Correction], records: Sequence[Mapping[str, Any]]) -> Plan:
    """Status counts of the latest decisions, with a conflict per stale or orphan decision."""
    by_key = current(corrections)
    decided = latest(records)
    out = Plan(counts={s: 0 for s in ("proposed", "approved", "edited", "rejected")})
    for key, correction in by_key.items():
        record = decided.get(key)
        if record is not None and is_stale(correction, record):
            out.conflicts.append(f"{key}: the correction changed after it was reviewed")
            record = None
        out.counts[status_of(record)] += 1
    for key in decided:
        if key not in by_key:
            out.conflicts.append(f"{key}: reviewed but no longer in the corrections file")
    return out


@dataclass(frozen=True)
class Trainable:
    """A correction cleared for training, with the reviewer decision that cleared it."""

    record_id: str
    question_id: str
    status: str
    label: str | None
    distribution: Mapping[str, float] | None
    text: str
    author: str
    reason: str
    decision: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_id": self.record_id,
            "question_id": self.question_id,
            "status": self.status,
            "text": self.text,
            "author": self.author,
            "reason": self.reason,
            "reviewer_decision": dict(self.decision),
        }
        if self.label is not None:
            out["label"] = self.label
        if self.distribution is not None:
            out["distribution"] = dict(self.distribution)
        return out


def trainable(
    corrections: Iterable[Correction], records: Sequence[Mapping[str, Any]]
) -> list[Trainable]:
    """Corrections the review file approved or edited; nothing else is ever exported.

    A correction with no decision, a rejected one, one whose latest decision is stale, and
    a correction that merely *claims* ``approved`` in its own file are all left out.
    """
    decided = latest(records)
    out = []
    for key, correction in current(corrections).items():
        record = decided.get(key)
        if record is None or record["action"] == "reject" or is_stale(correction, record):
            continue
        label, dist = correction.label, correction.distribution
        if record["action"] == "edit":
            after = record.get("after") or {}
            label, dist = after.get("label"), after.get("distribution")
        if label is None and not dist:
            raise BadTarget(f"{key}: the reviewed target is empty")
        out.append(
            Trainable(
                record_id=correction.record_id,
                question_id=correction.question_id,
                status=STATUS_OF[record["action"]],
                label=label,
                distribution=dist,
                text=correction.text,
                author=correction.author,
                reason=correction.reason,
                decision={
                    k: record.get(k)
                    for k in ("at", "action", "reviewer", "note", "correction_sha256")
                },
            )
        )
    return out
