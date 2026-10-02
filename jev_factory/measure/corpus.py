"""The measure stage's corpus loader: split and corpus files as validated entries.

A split, slice or corpus file is ``{"header": ..., "entries": [...], "world":
{...}?}``. Each entry needs ``id``, ``kind``, ``text`` and an ``expect`` block
naming exactly one answer: ``{"operation": name, "args": {...}}``,
``{"escalate": true}`` or ``{"explain": true}``. An operation expectation is
checked against the :class:`~jev_factory.domain.model.Domain`
(:meth:`Domain.validate_args`), so a corpus cannot silently rot when the
domain's operations change: a bad entry is reported in ``problems`` and
skipped, never raised.

The request text a scorer sees is the entry's ``text`` exactly -- the same
text :mod:`jev_factory.data.assemble` trains on. Stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "nvsh/tiers/bench.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 75-176 (CorpusEntry, CorpusLoadResult, load_corpus, _parse_entry,"
        " _validate_expect) kept; nvsh.ops.table.validate becomes Domain.validate_args",
        "an entry's kind is any non-empty string (nvsh accepted only explicit|failure);"
        " the request text is the entry's text as assemble.py trains on it, so nvsh's"
        " failure-kind request formatting (nvsh/tiers/lfm.py request_message) is dropped",
        "source_id and the file's own world object are carried on the result",
    ],
    "licence": "Apache-2.0",
}


@dataclass(frozen=True)
class CorpusEntry:
    """One corpus entry: a request and the one answer it expects."""

    id: str
    kind: str
    text: str
    expect: dict
    source: str = ""
    #: The entry's ``class`` (a phrasing class, or ``decline:<reason>``).
    phrasing: str = ""
    #: The original entry a variation came from (its own id when it has none).
    source_id: str = ""
    #: The offered operations of a missing-candidate slice entry, else ``None`` (all).
    candidates: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CorpusLoadResult:
    """What :func:`load_corpus` found: valid entries plus reported problems."""

    entries: tuple[CorpusEntry, ...]
    problems: tuple[str, ...]
    header: object = None
    world: Mapping[str, Any] | None = field(default=None)


def parse_entry(index: int, item: object) -> CorpusEntry | str:
    """One raw entry as a :class:`CorpusEntry`, or a one-line problem."""
    if not isinstance(item, dict):
        return f"entry[{index}]: not an object"
    try:
        entry_id = str(item["id"])
        kind = str(item["kind"])
        text = str(item["text"])
        expect = item["expect"]
    except KeyError as exc:
        return f"entry[{index}]: missing field {exc}"
    if not kind:
        return f"{entry_id}: kind must be a non-empty string"
    if not isinstance(expect, dict):
        return f"{entry_id}: expect must be an object"
    candidates = item.get("candidates")
    offered = tuple(str(name) for name in candidates) if isinstance(candidates, list) else None
    return CorpusEntry(
        id=entry_id,
        kind=kind,
        text=text,
        expect=dict(expect),
        source=str(item.get("source", "")),
        phrasing=str(item.get("class") or ""),
        source_id=str(item.get("source_id") or entry_id),
        candidates=offered,
    )


def validate_expect(entry: CorpusEntry, domain: Domain) -> str | None:
    """Why *entry*'s expectation is unusable against *domain*, or ``None``."""
    declines = [key for key in ("escalate", "explain") if entry.expect.get(key) is True]
    if len(declines) > 1 or (declines and "operation" in entry.expect):
        return (
            f"{entry.id}: expect mixes {', '.join(declines)} with another answer; give exactly one"
        )
    if declines:
        return None
    error = domain.validate_args(entry.expect.get("operation"), entry.expect.get("args", {}))
    if error is not None:
        return f"{entry.id}: expect {error.message}"
    return None


def load_raw(raw: object, domain: Domain) -> CorpusLoadResult:
    """Validate an already-decoded corpus object (see :func:`load_corpus`)."""
    header = raw.get("header") if isinstance(raw, dict) else None
    world = raw.get("world") if isinstance(raw, dict) else None
    raw_entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    if not isinstance(raw_entries, list):
        raw_entries = []
    entries: list[CorpusEntry] = []
    problems: list[str] = []
    for index, item in enumerate(raw_entries):
        parsed = parse_entry(index, item)
        if isinstance(parsed, str):
            problems.append(parsed)
            continue
        error = validate_expect(parsed, domain)
        if error is not None:
            problems.append(error)
            continue
        entries.append(parsed)
    return CorpusLoadResult(
        entries=tuple(entries),
        problems=tuple(problems),
        header=header,
        world=world if isinstance(world, dict) else None,
    )


def load_corpus(path: str | Path, domain: Domain) -> CorpusLoadResult:
    """Load and validate a corpus file. Never raises for a bad entry (only for bad JSON)."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return load_raw(raw, domain)
