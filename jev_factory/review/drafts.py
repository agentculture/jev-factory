"""Operator review of drafted records of every split (``jev review --drafts``, t25).

The agent drafts records (``jev distill dataset`` / the pilot drafter) with evidence
lines ``{id, group, quote, why}`` saying where each record came from. The operator goes
through them one at a time and decides: ``keep``, ``drop``, ``edit`` (correct a question's
gold, as a label or as probabilities over its options), ``add_option`` (an option the draft
missed, such as an escalation), ``restore`` (undo a question's edits and added options) or
``note`` (a remark that stays out of the records).

Decisions go to an append-only ledger (``<records stem>.draft-ledger.jsonl`` beside the
drafts), one JSON line each, binding ``record_id`` to the record's sha256 at decision time
(:func:`jev_factory.review.gold.record_sha256`). The drafted file is never touched by a
decision; :func:`apply_ledger` produces the reviewed draft file from it: dropped records
are removed, edits are applied, notes never reach a record.

Entries count only for the record's *current* sha256. Status is ``dropped`` after a drop,
``edited`` while the ledger changes the record, ``kept`` after a keep, else
``unreviewed``. An edited test record has a new sha256, so any earlier gold approval no
longer matches: its approval still goes through the t8 gold ledger (``jev review
--gold``). A malformed ledger raises :class:`DraftReviewError`; nothing is guessed.
Stdlib only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.distill.records import Gold, Question, Record
from jev_factory.review import gold

LEDGER_SUFFIX = ".draft-ledger.jsonl"
ACTIONS = ("keep", "drop", "edit", "note", "add_option", "restore")
STATUSES = ("unreviewed", "kept", "edited", "dropped")
DEFAULT_REVIEWER = gold.DEFAULT_REVIEWER
#: An added option's name: short snake_case, like a drafted one.
OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
MAX_OPTIONS = 24
#: How far a soft gold's probabilities may sum from 1 (the page sends percentages).
SUM_TOLERANCE = 0.011


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


def is_probabilities(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
        for k, v in value.items()
    )


def _check_payload(entry: dict[str, Any], where: str) -> None:
    action = entry["action"]
    if action == "edit":
        edit = entry.get("edit")
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("question_id"), str)
            and isinstance(edit.get("label"), str)
            and (edit.get("probabilities") is None or is_probabilities(edit["probabilities"]))
        ):
            raise DraftReviewError(f"{where}: an edit entry needs edit.question_id and edit.label")
    elif action == "add_option":
        option = entry.get("option")
        if not (
            isinstance(option, dict)
            and all(isinstance(option.get(k), str) for k in ("question_id", "name", "description"))
        ):
            raise DraftReviewError(
                f"{where}: an add_option entry needs option.question_id, .name and .description"
            )
    elif action == "restore":
        restore = entry.get("restore")
        if not (isinstance(restore, dict) and isinstance(restore.get("question_id", ""), str)):
            raise DraftReviewError(f"{where}: a restore entry needs restore.question_id")
    elif action == "note" and not str(entry.get("note") or "").strip():
        raise DraftReviewError(f"{where}: a note entry needs a note")


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
    _check_payload(entry, where)
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
    edit: tuple[str, str] | tuple[str, str, Mapping[str, float] | None] | None = None,
    option: tuple[str, str, str] | None = None,
    restore: str | None = None,
    current: Record | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """One ledger entry for *record*.

    ``edit`` is ``(question id, label)`` or ``(question id, label, probabilities)``: a soft
    gold over the question's names whose argmax is the label. ``option`` is ``(question id,
    name, description)``, an extra option on a choice question. ``restore`` is a question id
    (``""`` for every question): its edits and added options are undone. *current* is the
    record as the ledger already makes it (added options included); the edit or option is
    checked against it, while the entry binds the drafted record's sha256.
    """
    if action not in ACTIONS:
        raise DraftReviewError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    payloads = {"edit": edit, "add_option": option, "restore": restore}
    for name, value in payloads.items():
        if (action == name) != (value is not None):
            raise DraftReviewError(
                f"{record.id}: {name} goes with the {name} action, and only with it"
            )
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
    base = current or record
    if edit is not None:
        qid, label = edit[0], edit[1]
        probs = edit[2] if len(edit) > 2 else None  # type: ignore[misc]
        _gold_for(base, qid, label, probs)  # validates the question, label and probabilities
        entry["edit"] = {"question_id": qid, "label": label}
        if probs is not None:
            entry["edit"]["probabilities"] = {k: float(v) for k, v in probs.items()}
    if option is not None:
        _with_option(base, *option)  # validates
        entry["option"] = dict(zip(("question_id", "name", "description"), option))
    if restore is not None:
        if restore and restore not in record.questions:
            raise DraftReviewError(f"{record.id}: no question {restore!r}")
        entry["restore"] = {"question_id": restore}
    return entry


def _question(record: Record, qid: str) -> Question:
    question = record.questions.get(qid)
    if question is None:
        raise DraftReviewError(f"{record.id}: no question {qid!r}")
    return question


def _gold_for(record: Record, qid: str, label: str, probs: Mapping[str, float] | None) -> Gold:
    """The new gold of *qid*: a hard label, or a soft distribution whose argmax is *label*."""
    question = _question(record, qid)
    names = question.names
    if label not in names:
        raise DraftReviewError(f"{record.id}/{qid}: {label!r} is not in {list(names)}")
    if probs is None:
        if question.type == "noul":
            return Gold(label=label, probability_true=1.0 if label == "true" else 0.0)
        return Gold(label=label)
    stray = sorted(set(probs) - set(names))
    if stray:
        raise DraftReviewError(f"{record.id}/{qid}: probabilities for {stray} not in {list(names)}")
    full = {name: float(probs.get(name, 0.0)) for name in names}
    if any(not 0.0 <= v <= 1.0 for v in full.values()):
        raise DraftReviewError(f"{record.id}/{qid}: a probability is outside [0, 1]")
    if abs(sum(full.values()) - 1.0) > SUM_TOLERANCE:
        raise DraftReviewError(f"{record.id}/{qid}: probabilities sum to {sum(full.values()):.3f}")
    if full[label] < max(full.values()):
        raise DraftReviewError(f"{record.id}/{qid}: {label!r} is not the most likely answer")
    if question.type == "noul":
        return Gold(label=label, probability_true=round(full["true"], 4))
    return Gold(label=label, probabilities={k: round(v, 4) for k, v in full.items()})


def _fields(record: Record) -> dict[str, Any]:
    return {name: getattr(record, name) for name in record.__dataclass_fields__}


def _with_option(record: Record, qid: str, name: str, description: str) -> Record:
    question = _question(record, qid)
    if question.type != "choice":
        raise DraftReviewError(f"{record.id}/{qid}: options can be added to a choice question only")
    if not OPTION_NAME.match(name):
        raise DraftReviewError(f"{record.id}/{qid}: option name {name!r} is not short snake_case")
    if name in question.names:
        raise DraftReviewError(f"{record.id}/{qid}: {name!r} is already an option")
    if not description.strip() or len(description) > 200:
        raise DraftReviewError(f"{record.id}/{qid}: an option needs a one-line description")
    if len(question.names) >= MAX_OPTIONS:
        raise DraftReviewError(f"{record.id}/{qid}: a question has at most {MAX_OPTIONS} options")
    grown = Question(
        type=question.type,
        criteria=(*question.criteria, (name, description.strip())),
        instructions=question.instructions,
    )
    return Record(**{**_fields(record), "questions": {**record.questions, qid: grown}})


def _with_gold(record: Record, qid: str, new: Gold) -> Record:
    return Record(**{**_fields(record), "gold": {**record.gold, qid: new}})


# -- resolution ----------------------------------------------------------------------------


@dataclass
class Resolution:
    """What the ledger says about one record at its current sha256.

    ``decision`` is the latest keep or drop (an edit or added option implies keep);
    ``edits`` is the edited label per question, ``soft`` its probabilities when the
    operator gave them, ``added`` the options added per question. A restore clears a
    question's edits and added options.
    """

    status: str = "unreviewed"
    decision: str | None = None
    edits: dict[str, str] = field(default_factory=dict)
    soft: dict[str, dict[str, float]] = field(default_factory=dict)
    added: dict[str, list[list[str]]] = field(default_factory=dict)
    notes: list[dict[str, str]] = field(default_factory=list)


def _restore(res: Resolution, qid: str) -> None:
    for qids in (res.edits, res.soft, res.added):
        for key in [k for k in qids if not qid or k == qid]:
            del qids[key]


def _take(res: Resolution, entry: Mapping[str, Any]) -> None:
    action = entry["action"]
    if action == "note":
        res.notes.append({"at": entry["at"], "note": str(entry["note"])})
    elif action in ("keep", "drop"):
        res.decision = action
    elif action == "edit":
        qid = entry["edit"]["question_id"]
        res.decision = "keep"
        res.edits[qid] = entry["edit"]["label"]
        res.soft.pop(qid, None)
        if entry["edit"].get("probabilities") is not None:
            res.soft[qid] = dict(entry["edit"]["probabilities"])
    elif action == "add_option":
        option = entry["option"]
        res.decision = "keep"
        res.added.setdefault(option["question_id"], []).append(
            [option["name"], option["description"]]
        )
    else:  # restore
        _restore(res, entry["restore"]["question_id"])


def effective(record: Record, res: Resolution) -> Record:
    """*record* as the ledger makes it: added options, then edited gold; never dropped."""
    after = record
    for qid, options in res.added.items():
        for name, description in options:
            after = _with_option(after, qid, name, description)
    for qid, label in res.edits.items():
        new = _gold_for(after, qid, label, res.soft.get(qid))
        if new != after.gold[qid]:
            after = _with_gold(after, qid, new)
    return after


def resolve(
    ledger: Sequence[Mapping[str, Any]], records: Iterable[Record]
) -> dict[str, Resolution]:
    """Per record id: status, edits, added options and notes.

    Status is ``dropped`` after a drop, else ``edited`` while the record differs from the
    draft, else ``kept`` after a keep, else ``unreviewed``. An edit that restores the drafted
    gold, or a restore, leaves nothing edited.
    """
    by_id = {r.id: r for r in records}
    shas = {rid: gold.record_sha256(r) for rid, r in by_id.items()}
    out = {rid: Resolution() for rid in shas}
    for number, raw in enumerate(ledger, start=1):
        entry = _check_entry(dict(raw), f"ledger entry {number}")
        res = out.get(entry["record_id"])
        if res is None or shas[entry["record_id"]] != entry["record_sha256"]:
            continue
        _take(res, entry)
    for rid, res in out.items():
        changed = effective(by_id[rid], res) != by_id[rid]
        if not changed:
            res.edits.clear()
            res.soft.clear()
            res.added.clear()
        if res.decision == "drop":
            res.status = "dropped"
        elif changed:
            res.status = "edited"
        elif res.decision == "keep":
            res.status = "kept"
    return out


def apply_ledger(
    records: Sequence[Record],
    ledger: Sequence[Mapping[str, Any]],
    *,
    include_unreviewed: bool = True,
) -> list[Record]:
    """The reviewed draft: dropped removed, options added, edits applied, notes left out.

    ``include_unreviewed=False`` also leaves out records nobody has decided on yet. An edit
    that restores a question's original gold changes nothing.
    """
    resolved = resolve(ledger, records)
    out: list[Record] = []
    for record in records:
        res = resolved[record.id]
        if res.status == "dropped" or (res.status == "unreviewed" and not include_unreviewed):
            continue
        out.append(effective(record, res))
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
