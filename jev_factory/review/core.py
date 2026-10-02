"""The pure parts of a seed review: verb tree, review file and change plan.

The **verb tree** is the domain's operations arranged by their dotted names
(``jev`` -> ``jev.run`` -> ``jev.run.seed``), then the ``explain`` control and
the ``escalate`` control with one child per escalate reason. Every seed entry
attaches to one node: its operation, ``explain``, or the reason its
``decline:<reason>`` class names.

The **review file** (``<seed stem>.review.jsonl`` beside the seed by default)
is append-only: one JSON record per operator decision, never rewritten. The
latest record for an entry id is its current decision. ``before`` is always
the seed entry as it stood when the decision was recorded, so a change made to
the seed by other means since then shows as a conflict instead of being
overwritten.

The **change plan** turns the latest decisions into seed changes: ``reject``
removes the entry, ``edit`` replaces it, ``propose`` adds a new one and
``approve`` changes nothing in the seed (the review file is its record).
Planning is idempotent, so a decision already applied plans no change, and
the planned seed is validated against the domain before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_factory.domain.model import DECLINE_PREFIX, ESCALATE, ESCALATE_PREFIX, EXPLAIN, Domain
from jev_factory.measure.corpus import load_raw

REVIEW_SUFFIX = ".review.jsonl"
ACTIONS = ("approve", "reject", "edit", "propose")
#: The fields an entry needs; ``class`` and ``source`` are carried when present.
ENTRY_FIELDS = ("id", "kind", "text", "expect")
#: The source a proposed entry gets when the operator gives none.
OPERATOR_SOURCE_PREFIX = "operator-review-"
STATUS_PENDING = "pending"
_STATUS_OF_ACTION = {
    "approve": "approved",
    "reject": "rejected",
    "edit": "edited",
    "propose": "proposed",
}
_NUMBERED_ID = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


class ReviewError(ValueError):
    """A review file or decision that cannot be used."""


# -- the verb tree -----------------------------------------------------------------


def verb_tree(domain: Domain) -> list[dict[str, Any]]:
    """The domain's candidates as tree nodes, parents before children.

    Each node is ``{id, label, parent, kind, read_only, description}``; ``kind`` is
    ``domain`` (the root), ``group`` (a dotted prefix that is not an operation),
    ``operation``, ``control`` or ``reason``.
    """
    root = domain.name
    nodes: dict[str, dict[str, Any]] = {
        root: _node(root, root, None, "domain", None, domain.description)
    }
    names = set(domain.names())
    for op in domain.operations:
        parts = op.name.split(".")
        parent = root
        for depth in range(1, len(parts)):
            prefix = ".".join(parts[:depth])
            if prefix not in nodes:
                if prefix in names:
                    found = domain.get(prefix)
                    nodes[prefix] = _node(
                        prefix,
                        parts[depth - 1],
                        parent,
                        "operation",
                        bool(found.read_only) if found else None,
                        found.description if found else "",
                    )
                else:
                    nodes[prefix] = _node(prefix, parts[depth - 1], parent, "group", None, "")
            parent = prefix
        if op.name in nodes:  # already added as the prefix of an earlier operation
            continue
        nodes[op.name] = _node(
            op.name, parts[-1], parent, "operation", bool(op.read_only), op.description
        )
    nodes[EXPLAIN] = _node(
        EXPLAIN, EXPLAIN, root, "control", True, domain.control_description(EXPLAIN) or ""
    )
    nodes[ESCALATE] = _node(
        ESCALATE, ESCALATE, root, "control", True, domain.control_description(ESCALATE) or ""
    )
    for reason in domain.reasons:
        label = reason.escalate_label
        nodes[label] = _node(label, reason.name, ESCALATE, "reason", True, reason.description)
    return list(nodes.values())


def _node(
    node_id: str,
    label: str,
    parent: str | None,
    kind: str,
    read_only: bool | None,
    description: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "parent": parent,
        "kind": kind,
        "read_only": read_only,
        "description": description,
    }


def attach(entry: Mapping[str, Any], domain: Domain) -> str:
    """The tree node *entry* belongs to (the domain root when its expectation is unusable)."""
    expect = entry.get("expect")
    if not isinstance(expect, Mapping):
        return domain.name
    if expect.get("explain") is True:
        return EXPLAIN
    if expect.get("escalate") is True:
        reason = str(entry.get("class") or "").removeprefix(DECLINE_PREFIX)
        names = {r.name for r in domain.reasons}
        return ESCALATE_PREFIX + reason if reason in names else ESCALATE
    operation = expect.get("operation")
    if isinstance(operation, str) and domain.get(operation) is not None:
        return operation
    return domain.name


# -- the review file ---------------------------------------------------------------


def review_path(seed: Path) -> Path:
    """The default review file for *seed*: ``<stem>.review.jsonl`` beside it."""
    return seed.with_name(seed.stem + REVIEW_SUFFIX)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every record in the review file, in order; an absent file has none."""
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewError(f"{path}:{number}: not JSON ({exc.msg})") from None
        if not isinstance(record, dict) or record.get("action") not in ACTIONS:
            raise ReviewError(f"{path}:{number}: not a review record")
        if not isinstance(record.get("entry_id"), str):
            raise ReviewError(f"{path}:{number}: a review record needs an entry_id")
        records.append(record)
    return records


def latest(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The current decision per entry id: its latest record."""
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        out[str(record["entry_id"])] = record
    return out


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record as a JSON line and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- decisions ---------------------------------------------------------------------


def make_record(
    raw_seed: Mapping[str, Any],
    domain: Domain,
    decision: Mapping[str, Any],
    *,
    proposals: Mapping[str, Mapping[str, Any]] | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Check one decision from the operator and return the record to append.

    *decision* is ``{action, entry_id, after?, note?}``. ``before`` is taken from
    the seed, never from the caller. *proposals* are the entries already
    proposed in the review file: a proposal is revised by proposing it again
    and withdrawn by rejecting it. A bad decision raises :class:`ReviewError`.
    """
    action = decision.get("action")
    entry_id = decision.get("entry_id")
    if action not in ACTIONS:
        raise ReviewError(f"action must be one of {', '.join(ACTIONS)}")
    if not isinstance(entry_id, str) or not entry_id.strip():
        raise ReviewError("a decision needs an entry_id")
    note = decision.get("note", "")
    if not isinstance(note, str):
        raise ReviewError("note must be a string")
    current = _by_id(raw_seed).get(entry_id)
    proposed = (proposals or {}).get(entry_id)
    after: dict[str, Any] | None = None
    if action == "propose":
        if current is not None:
            raise ReviewError(f"{entry_id} is already in the seed; edit it instead")
        after = _checked_entry(decision.get("after"), entry_id, domain, propose=True, at=at)
    elif current is None:
        if proposed is None:
            raise ReviewError(f"{entry_id} is not in the seed")
        if action != "reject":
            raise ReviewError(
                f"{entry_id} is a proposal, not a seed entry; propose it again to revise it, "
                "or reject it to withdraw it"
            )
    elif action == "edit":
        after = _checked_entry(decision.get("after"), entry_id, domain, propose=False, at=at)
        if after == current:
            raise ReviewError(f"{entry_id}: the edit changes nothing")
    return {
        "at": at or now_utc(),
        "action": action,
        "entry_id": entry_id,
        "before": current,
        "after": after,
        "note": note.strip(),
        "seed_sha256": seed_sha256(raw_seed),
    }


def check_entry(entry: object, domain: Domain) -> list[str]:
    """Every problem with *entry* as a seed entry of *domain* (empty when it is usable)."""
    if not isinstance(entry, Mapping):
        return ["an entry must be an object"]
    problems = [f"missing field {name!r}" for name in ENTRY_FIELDS if name not in entry]
    if problems:
        return problems
    for name in ("id", "kind", "text"):
        if not isinstance(entry[name], str) or not entry[name].strip():
            problems.append(f"{name} must be a non-empty string")
    for name in ("class", "source"):
        if name in entry and not isinstance(entry[name], str):
            problems.append(f"{name} must be a string")
    if problems:
        return problems
    result = load_raw({"entries": [dict(entry)]}, domain)
    return [str(problem) for problem in result.problems]


def _checked_entry(
    after: object, entry_id: str, domain: Domain, *, propose: bool, at: str | None
) -> dict[str, Any]:
    if not isinstance(after, Mapping):
        raise ReviewError("an edit or a proposal needs the entry as 'after'")
    entry: dict[str, Any] = {str(key): value for key, value in after.items()}
    entry.setdefault("id", entry_id)
    if entry["id"] != entry_id:
        raise ReviewError(f"the entry's id {entry['id']!r} is not {entry_id!r}")
    if propose and not entry.get("source"):
        entry["source"] = OPERATOR_SOURCE_PREFIX + (at or now_utc())[:10]
    problems = check_entry(entry, domain)
    if not problems:
        return entry
    named = [p if p.startswith(f"{entry_id}:") else f"{entry_id}: {p}" for p in problems]
    raise ReviewError("; ".join(named))


# -- the change plan ---------------------------------------------------------------


@dataclass
class ChangePlan:
    """What applying the review file would do to the seed."""

    seed: dict[str, Any]
    changes: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.problems


def plan_changes(
    raw_seed: Mapping[str, Any], records: Sequence[Mapping[str, Any]], domain: Domain
) -> ChangePlan:
    """The seed the review file's latest decisions produce, and every change on the way."""
    entries = [dict(e) for e in raw_seed.get("entries", [])]
    index = {str(e.get("id")): i for i, e in enumerate(entries)}
    plan = ChangePlan(seed=dict(raw_seed))
    removed: set[int] = set()
    added: list[dict[str, Any]] = []
    for entry_id, record in latest(records).items():
        _plan_one(entry_id, record, entries, index, removed, added, plan)
    new_entries = [e for i, e in enumerate(entries) if i not in removed] + added
    plan.seed["entries"] = new_entries
    plan.problems.extend(_seed_problems(plan.seed, domain))
    decided = latest(records)
    plan.counts = dict(Counter(status(r) for r in decided.values()))
    plan.counts[STATUS_PENDING] = sum(1 for i in index if i not in decided)
    return plan


def _plan_one(
    entry_id: str,
    record: Mapping[str, Any],
    entries: list[dict[str, Any]],
    index: dict[str, int],
    removed: set[int],
    added: list[dict[str, Any]],
    plan: ChangePlan,
) -> None:
    action, before, after = record["action"], record.get("before"), record.get("after")
    position = index.get(entry_id)
    current = entries[position] if position is not None else None
    if action == "approve":
        if current is None:
            plan.conflicts.append(f"{entry_id}: approved, but it is no longer in the seed")
        elif before is not None and current != before:
            plan.conflicts.append(f"{entry_id}: approved, but the seed entry changed since")
        return
    if action == "reject":
        if current is None:
            return  # already removed (or a withdrawn proposal)
        if before is not None and current != before:
            plan.conflicts.append(f"{entry_id}: rejected, but the seed entry changed since")
            return
        removed.add(position)  # type: ignore[arg-type]
        plan.changes.append({"action": "remove", "entry_id": entry_id, "before": current})
        return
    if current == after:
        return  # already applied
    if action == "edit":
        if current is None:
            plan.conflicts.append(f"{entry_id}: edited, but it is no longer in the seed")
        elif current != before:
            plan.conflicts.append(f"{entry_id}: edited, but the seed entry changed since")
        else:
            entries[position] = dict(after)  # type: ignore[index, arg-type]
            plan.changes.append(
                {"action": "edit", "entry_id": entry_id, "before": current, "after": after}
            )
        return
    if current is not None:  # propose
        plan.conflicts.append(f"{entry_id}: proposed, but the id is already taken in the seed")
        return
    added.append(dict(after))  # type: ignore[arg-type]
    plan.changes.append({"action": "add", "entry_id": entry_id, "after": after})


def status(record: Mapping[str, Any] | None) -> str:
    """An entry's review status from its latest record; a rejected proposal is ``withdrawn``."""
    if record is None:
        return STATUS_PENDING
    if record["action"] == "reject" and record.get("before") is None:
        return "withdrawn"
    return _STATUS_OF_ACTION[str(record["action"])]


def _seed_problems(seed: Mapping[str, Any], domain: Domain) -> list[str]:
    ids = Counter(str(e.get("id")) for e in seed.get("entries", []))
    problems = [f"{i}: the id appears {n} times" for i, n in ids.items() if n > 1]
    problems.extend(load_raw(dict(seed), domain).problems)
    return problems


def proposals(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The entries whose latest decision is a proposal, by id."""
    return {
        i: r["after"]
        for i, r in latest(records).items()
        if r["action"] == "propose" and isinstance(r.get("after"), Mapping)
    }


# -- seed file helpers ---------------------------------------------------------------


def _by_id(raw_seed: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(e.get("id")): dict(e) for e in raw_seed.get("entries", [])}


def dump_seed(seed: Mapping[str, Any]) -> str:
    """The seed as written in the repo: two-space JSON, UTF-8 kept, trailing newline."""
    return json.dumps(seed, indent=2, ensure_ascii=False) + "\n"


def seed_sha256(raw_seed: Mapping[str, Any]) -> str:
    return hashlib.sha256(dump_seed(raw_seed).encode("utf-8")).hexdigest()


def next_ids(
    raw_seed: Mapping[str, Any],
    domain: Domain,
    taken: Iterable[str] = (),
) -> dict[str, str]:
    """A free id per kind of entry (operation, explain, escalate), in the seed's own style.

    The style is the most common ``<prefix><number>`` among the seed's entries of
    that kind; the number is one past the highest, zero-padded to the same width.
    """
    used = {str(e.get("id")) for e in raw_seed.get("entries", [])} | set(taken)
    styles: dict[str, Counter[tuple[str, int]]] = {}
    highest: dict[tuple[str, int], int] = {}
    for entry in raw_seed.get("entries", []):
        match = _NUMBERED_ID.match(str(entry.get("id", "")))
        if not match:
            continue
        style = (match["prefix"], len(match["num"]))
        styles.setdefault(_entry_kind(entry, domain), Counter())[style] += 1
        highest[style] = max(highest.get(style, 0), int(match["num"]))
    out = {}
    for kind in ("operation", "explain", "escalate"):
        counter = styles.get(kind)
        prefix, width = counter.most_common(1)[0][0] if counter else (f"{kind}-", 3)
        number = highest.get((prefix, width), 0) + 1
        while f"{prefix}{number:0{width}d}" in used:
            number += 1
        out[kind] = f"{prefix}{number:0{width}d}"
    return out


def _entry_kind(entry: Mapping[str, Any], domain: Domain) -> str:
    node = attach(entry, domain)
    if node == EXPLAIN:
        return "explain"
    if node == ESCALATE or node.startswith(ESCALATE_PREFIX):
        return "escalate"
    return "operation"
