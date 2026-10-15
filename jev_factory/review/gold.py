"""Operator approval ledger for human-gold test records (``jev review --gold``, t8).

The agent drafts held-out test records with *proposed* gold. A record counts as
human gold only when the operator approved **that exact record**: the ledger
(``<records stem>.gold-ledger.jsonl`` beside the records) is append-only, one JSON line
per decision, each binding ``record_id`` to the record's sha256 at decision time.

Rules the final stage relies on (:func:`approved_ids`):

* an id is approved only when the latest ledger entry for its *current* sha256 is an
  ``approve``; a ``reject`` (or an ``edit``, which is not an approval) leaves it out;
* editing a record changes its sha256, so an earlier approval no longer matches;
* a malformed ledger raises :class:`GoldLedgerError`; nothing is guessed (fails closed).

The record hash excludes ``gold_provenance``: the approval is what establishes it, so
flipping the label to ``human`` must not void the approval. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_factory.distill.records import Gold, Record

LEDGER_SUFFIX = ".gold-ledger.jsonl"
ACTIONS = ("approve", "edit", "reject")
DEFAULT_REVIEWER = "operator"
TEST_SPLIT = "test"


class GoldLedgerError(ValueError):
    """A bad decision, a malformed ledger or an inconsistent records file."""


def ledger_path(records: Path) -> Path:
    return records.with_name(records.stem + LEDGER_SUFFIX)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_sha256(record: Record | Mapping[str, Any]) -> str:
    """sha256 of the canonical record JSON, ``gold_provenance`` excluded."""
    data = dict(record.to_json() if isinstance(record, Record) else record)
    data.pop("gold_provenance", None)
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -- the records file ----------------------------------------------------------------------


def read_drafted(path: Path) -> list[Record]:
    """The drafted records; a bad line or a duplicate id is an error."""
    out: list[Record] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(Record.from_json(json.loads(line)))
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as exc:
            raise GoldLedgerError(f"{path}:{number}: not a record ({exc})") from None
    ids = [r.id for r in out]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise GoldLedgerError(f"{path}: duplicate record ids {dup[:3]}")
    return out


def write_drafted(path: Path, records: Iterable[Record]) -> None:
    lines = [json.dumps(r.to_json(), sort_keys=True, ensure_ascii=False) for r in records]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)


# -- the ledger ----------------------------------------------------------------------------


def _check_entry(entry: object, where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise GoldLedgerError(f"{where}: not a gold ledger entry")
    sha = entry.get("record_sha256")
    ok = (
        entry.get("action") in ACTIONS
        and isinstance(entry.get("record_id"), str)
        and isinstance(sha, str)
        and len(sha) == 64
        and isinstance(entry.get("at"), str)
    )
    if not ok:
        raise GoldLedgerError(f"{where}: not a gold ledger entry")
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
            raise GoldLedgerError(f"{where}: not JSON ({exc.msg})") from None
        out.append(_check_entry(entry, where))
    return out


def append_entry(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def approved_ids(ledger: Sequence[Mapping[str, Any]], records: Iterable[Record]) -> set[str]:
    """Ids whose current sha256 has an ``approve`` as its latest ledger entry.

    *ledger* is validated here too, so a caller that built it by hand cannot sneak a
    malformed entry past the check (:class:`GoldLedgerError`).
    """
    latest: dict[tuple[str, str], str] = {}
    for number, entry in enumerate(ledger, start=1):
        checked = _check_entry(dict(entry), f"ledger entry {number}")
        latest[(checked["record_id"], checked["record_sha256"])] = checked["action"]
    return {r.id for r in records if latest.get((r.id, record_sha256(r))) == "approve"}


# -- decisions -----------------------------------------------------------------------------


def _edited(record: Record, qid: str, label: str) -> Record:
    question = record.questions.get(qid)
    if question is None:
        raise GoldLedgerError(f"{record.id}: no question {qid!r}")
    if label not in question.names:
        raise GoldLedgerError(f"{record.id}/{qid}: {label!r} is not in {list(question.names)}")
    old = record.gold[qid]
    if question.type == "noul":
        new = Gold(label=label, probability_true=1.0 if label == "true" else 0.0)
    else:
        new = Gold(label=label)
    if new == old:
        raise GoldLedgerError(f"{record.id}/{qid}: the edit changes nothing")
    return Record(**{**_fields(record), "gold": {**record.gold, qid: new}})


def _fields(record: Record) -> dict[str, Any]:
    return {name: getattr(record, name) for name in record.__dataclass_fields__}


def make_entry(
    record: Record,
    action: str,
    *,
    reviewer: str | None = None,
    note: str = "",
    edit: tuple[str, str] | None = None,
    at: str | None = None,
) -> tuple[dict[str, Any], Record]:
    """One ledger entry for *record* and the record as it stands after the decision.

    ``edit`` is ``(question id, new gold label)``; it is required for ``edit`` only.
    """
    if action not in ACTIONS:
        raise GoldLedgerError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    if (action == "edit") != (edit is not None):
        raise GoldLedgerError(f"{record.id}: --edit goes with the edit action, and only with it")
    entry: dict[str, Any] = {
        "at": at or now_utc(),
        "action": action,
        "record_id": record.id,
        "record_sha256": record_sha256(record),
        "reviewer": (reviewer or DEFAULT_REVIEWER).strip() or DEFAULT_REVIEWER,
        "note": note.strip(),
    }
    after = record
    if edit is not None:
        after = _edited(record, *edit)
        entry["new_sha256"] = record_sha256(after)
        entry["after"] = {edit[0]: after.gold[edit[0]].to_json()}
    return entry, after


# -- plan ----------------------------------------------------------------------------------


@dataclass
class Plan:
    """Counts of the test records by status, plus ledger entries that match nothing."""

    counts: dict[str, int] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)


def status_of(
    record: Record, latest: Mapping[tuple[str, str], str]
) -> str:  # proposed | approved | rejected | edited
    action = latest.get((record.id, record_sha256(record)))
    return {"approve": "approved", "reject": "rejected", "edit": "edited"}.get(
        action or "", "proposed"
    )


def plan(ledger: Sequence[Mapping[str, Any]], records: Sequence[Record]) -> Plan:
    latest = {(e["record_id"], e["record_sha256"]): e["action"] for e in ledger}
    out = Plan(counts=dict.fromkeys(("proposed", "approved", "rejected", "edited"), 0))
    for record in records:
        out.counts[status_of(record, latest)] += 1
    known = {r.id for r in records}
    out.orphans = sorted({e["record_id"] for e in ledger} - known)
    return out
