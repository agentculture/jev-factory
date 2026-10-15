"""Operator review of drafted records of every split (``jev review --drafts``, t25).

The agent drafts records (``jev distill dataset`` / the pilot drafter) with evidence
lines ``{id, group, quote, why}`` saying where each record came from. The operator goes
through them one at a time and decides: ``keep``, ``drop``, ``edit`` (correct a question's
gold label) or ``note`` (a remark that stays out of the records).

Decisions go to an append-only ledger (``<records stem>.draft-ledger.jsonl`` beside the
drafts), one JSON line each, binding ``record_id`` to the record's sha256 at decision time
(:func:`jev_factory.review.gold.record_sha256`). The drafted file is never touched by a
decision; :func:`apply_ledger` produces the reviewed draft file from it: dropped records
are removed, edits are applied, notes never reach a record.

Status of a record is its latest non-note entry for its *current* sha256 (``kept``,
``edited``, ``dropped``; no entry is ``unreviewed``). An edited test record has a new
sha256, so any earlier gold approval no longer matches: its approval still goes through
the t8 gold ledger (``jev review --gold``). A malformed ledger raises
:class:`DraftReviewError`; nothing is guessed. Stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.distill.records import Record
from jev_factory.review import gold

LEDGER_SUFFIX = ".draft-ledger.jsonl"
ACTIONS = ("keep", "drop", "edit", "note")
STATUSES = ("unreviewed", "kept", "edited", "dropped")
_STATUS_OF = {"keep": "kept", "edit": "edited", "drop": "dropped"}
DEFAULT_REVIEWER = gold.DEFAULT_REVIEWER


class DraftReviewError(gold.GoldLedgerError):
    """A bad decision, a malformed ledger or evidence file."""


def ledger_path(records: Path) -> Path:
    return records.with_name(records.stem + LEDGER_SUFFIX)


# -- evidence ------------------------------------------------------------------------------


def read_evidence(path: Path | None) -> dict[str, dict[str, str]]:
    """``{record id: {group, quote, why}}`` from the evidence JSONL; no path is empty."""
    if path is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DraftReviewError(f"{path}:{number}: not JSON ({exc.msg})") from None
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise DraftReviewError(f"{path}:{number}: an evidence line needs a string 'id'")
        out[item["id"]] = {k: str(item.get(k) or "") for k in ("group", "quote", "why")}
    return out


# -- the ledger ----------------------------------------------------------------------------


def _check_entry(entry: object, where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise DraftReviewError(f"{where}: not a draft ledger entry")
    sha = entry.get("record_sha256")
    ok = (
        entry.get("action") in ACTIONS
        and isinstance(entry.get("record_id"), str)
        and isinstance(sha, str)
        and len(sha) == 64
        and isinstance(entry.get("at"), str)
    )
    if not ok:
        raise DraftReviewError(f"{where}: not a draft ledger entry")
    if entry["action"] == "edit":
        edit = entry.get("edit")
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("question_id"), str)
            and isinstance(edit.get("label"), str)
        ):
            raise DraftReviewError(f"{where}: an edit entry needs edit.question_id and edit.label")
    if entry["action"] == "note" and not str(entry.get("note") or "").strip():
        raise DraftReviewError(f"{where}: a note entry needs a note")
    return entry


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """The ledger entries in file order; a missing file is empty, a bad line raises."""
    if not path.is_file():
        return []
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DraftReviewError(f"{where}: not JSON ({exc.msg})") from None
        out.append(_check_entry(entry, where))
    return out


append_entry = gold.append_entry


def make_entry(
    record: Record,
    action: str,
    *,
    reviewer: str | None = None,
    note: str = "",
    edit: tuple[str, str] | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """One ledger entry for *record*; ``edit`` is ``(question id, new gold label)``."""
    if action not in ACTIONS:
        raise DraftReviewError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    if (action == "edit") != (edit is not None):
        raise DraftReviewError(f"{record.id}: an edit goes with the edit action, and only with it")
    if action == "note" and not note.strip():
        raise DraftReviewError(f"{record.id}: a note needs text")
    entry: dict[str, Any] = {
        "at": at or gold.now_utc(),
        "action": action,
        "record_id": record.id,
        "record_sha256": gold.record_sha256(record),
        "reviewer": (reviewer or DEFAULT_REVIEWER).strip() or DEFAULT_REVIEWER,
        "note": note.strip(),
    }
    if edit is not None:
        _edit_gold(record, *edit)  # validates the question and the label
        entry["edit"] = {"question_id": edit[0], "label": edit[1]}
    return entry


def _edit_gold(record: Record, qid: str, label: str) -> Record:
    """*record* with the gold of *qid* set to *label* (validated by the gold module)."""
    return gold.make_entry(record, "edit", edit=(qid, label))[1]


# -- resolution ----------------------------------------------------------------------------


@dataclass
class Resolution:
    """What the ledger says about one record at its current sha256."""

    status: str = "unreviewed"
    edits: dict[str, str] = field(default_factory=dict)
    notes: list[dict[str, str]] = field(default_factory=list)


def resolve(
    ledger: Sequence[Mapping[str, Any]], records: Iterable[Record]
) -> dict[str, Resolution]:
    """Per record id: status (latest non-note entry), edits (latest per question), notes."""
    shas = {r.id: gold.record_sha256(r) for r in records}
    out = {rid: Resolution() for rid in shas}
    for number, raw in enumerate(ledger, start=1):
        entry = _check_entry(dict(raw), f"ledger entry {number}")
        res = out.get(entry["record_id"])
        if res is None or shas[entry["record_id"]] != entry["record_sha256"]:
            continue
        if entry["action"] == "note":
            res.notes.append({"at": entry["at"], "note": str(entry["note"])})
            continue
        res.status = _STATUS_OF[entry["action"]]
        if entry["action"] == "edit":
            res.edits[entry["edit"]["question_id"]] = entry["edit"]["label"]
    return out


def apply_ledger(
    records: Sequence[Record],
    ledger: Sequence[Mapping[str, Any]],
    *,
    include_unreviewed: bool = True,
) -> list[Record]:
    """The reviewed draft: dropped removed, edits applied, notes left out of the records.

    ``include_unreviewed=False`` also leaves out records nobody has decided on yet. An edit
    that restores a question's original gold changes nothing.
    """
    resolved = resolve(ledger, records)
    out: list[Record] = []
    for record in records:
        res = resolved[record.id]
        if res.status == "dropped" or (res.status == "unreviewed" and not include_unreviewed):
            continue
        after = record
        for qid, label in res.edits.items():
            if label != record.gold[qid].label:
                after = _edit_gold(after, qid, label)
        out.append(after)
    return out


def counts(resolved: Mapping[str, Resolution]) -> dict[str, int]:
    out = dict.fromkeys(STATUSES, 0)
    for res in resolved.values():
        out[res.status] += 1
    return out


def orphans(ledger: Sequence[Mapping[str, Any]], records: Iterable[Record]) -> list[str]:
    """Ids in the ledger that match no record, or only an older version of one."""
    shas = {(r.id, gold.record_sha256(r)) for r in records}
    return sorted(
        {e["record_id"] for e in ledger if (e["record_id"], e["record_sha256"]) not in shas}
    )
