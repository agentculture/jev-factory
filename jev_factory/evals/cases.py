"""Case model and case-set loader for the release gate.

A :class:`Case` is one evaluation prompt: its split tag, request text (never
populated for a held-out case), the operations it was offered as candidates,
its expected outcome, and whether that expectation is read-only or mutating.

Split files are the ``{"header": ..., "entries": [...]}`` JSON shape
:mod:`jev_factory.core.split` reads and writes: each entry has ``id``,
``text``, ``expect`` (``{"operation": name, "args": {...}}``,
``{"escalate": True}`` or ``{"explain": True}``), and optionally ``kind``,
``source``, ``class``, ``candidates`` (a missing-candidate slice's offered
operations) and ``source_id``.

Real split files live outside the repository, in the operator's private data
root; callers name them in a small ``{"splits": {tag: path}}`` mapping (or a
run manifest's ``[[case_set]]`` tables) that this module never ships.

``read_only``/mutating classification is never taken from a corpus entry or a
model's output: it is looked up in the :class:`~jev_factory.domain.model.Domain`
by the *expected* operation's name, and an operation the Domain does not know
is treated as mutating (:meth:`Domain.is_mutating`, the conservative side).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/cases.py",
    "commit": "9debdc6",
    "adaptations": [
        "line 49: nvsh.ops.table -> the Domain; _read_only_for (lines 160-178) uses"
        " Domain.is_mutating, so an unknown operation is mutating as before",
        "load_case_set/_entry_to_case take the Domain; an entry offering a candidate the"
        " Domain does not have is refused (ValueError) instead of failing later at labelling",
        "kept: Case, split tags, the held-out guard (no held-out text ever returned),"
        " HeldOutAccessError, TrainingOverlapError, training_overlap, case_sets_from_manifest",
    ],
    "licence": "Apache-2.0",
}

#: Split tags a Case may carry. ``val``/``val-mc`` let a gate run on the validation side
#: (a dry run before the final measurement, or a reference model's own validation set)
#: without relabelling it as test.
SPLIT_TAGS = ("val", "val-mc", "test", "test-mc", "heldout", "heldout-mc")

#: Split tags whose request text is never returned by the loader, whatever the
#: caller passes: the sealed held-out sets stay unread as text.
_HELDOUT_TAGS = ("heldout", "heldout-mc")


@dataclass(frozen=True)
class Case:
    """One evaluation prompt, resolved from a split-file entry.

    ``text`` is ``None`` for a held-out case, always. ``candidates`` is the
    tuple of operation names the case offered (``None`` means "every
    operation of the Domain", not "none offered"). ``expect`` is kept
    verbatim. ``read_only`` is ``None`` for a decline expectation (nothing to
    classify) and otherwise comes from the Domain: ``True``/``False`` for a
    known operation, ``False`` (mutating) for an unknown one.
    """

    id: str
    split: str
    text: str | None
    candidates: tuple[str, ...] | None
    expect: dict
    read_only: bool | None
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_id: str | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLIT_TAGS:
            raise ValueError(
                f"{self.id}: unknown split tag {self.split!r}, must be one of {SPLIT_TAGS}"
            )
        if self.split in _HELDOUT_TAGS and self.text is not None:
            raise ValueError(f"{self.id}: a held-out case must never carry request text")

    @property
    def expects_escalate(self) -> bool:
        return bool(self.expect.get("escalate"))

    @property
    def expects_explain(self) -> bool:
        return bool(self.expect.get("explain"))

    @property
    def expected_operation(self) -> str | None:
        return self.expect.get("operation")

    @property
    def expected_args(self) -> dict:
        return self.expect.get("args", {})


class HeldOutAccessError(RuntimeError):
    """Raised when a held-out case set is requested without ``include_heldout``."""


class TrainingOverlapError(RuntimeError):
    """Raised by :func:`training_overlap` when scored ids appear in a train split."""

    def __init__(self, overlapping_ids: tuple[str, ...]):
        self.overlapping_ids = overlapping_ids
        super().__init__(
            "refusing to score a checkpoint on case id(s) present in its own training "
            f"split: {list(overlapping_ids)}"
        )


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, str]:
    """A ``{"splits": {...}}`` mapping, a bare one, or a JSON file of either, as ``{tag: path}``."""
    raw: Mapping[str, Any]
    if isinstance(manifest, (str, Path)):
        raw = _read_json(manifest)
    else:
        raw = manifest
    splits = raw.get("splits", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(splits, Mapping):
        raise ValueError(f"manifest must map split tags to paths, got {splits!r}")
    return {str(tag): str(path) for tag, path in splits.items()}


def case_sets_from_manifest(manifest: Any, private_root: str | Path) -> dict[str, str]:
    """A run manifest's ``[[case_set]]`` entries as ``{split: path}`` under *private_root*.

    *manifest* is duck-typed (a ``case_sets`` iterable of objects with
    ``split``/``path``), so this module need not import the manifest module.
    When two case sets share a split tag, the later one wins.
    """
    root = Path(private_root)
    return {cs.split: str(root / cs.path) for cs in manifest.case_sets}


def _read_only_for(expect: dict, domain: Domain) -> bool | None:
    """The Domain's read-only flag for an expectation; ``None`` for a decline.

    An operation the Domain does not know is mutating (``False``), the same
    conservative default the gate uses.
    """
    if expect.get("escalate") or expect.get("explain"):
        return None
    name = expect.get("operation")
    if name is None:
        return None
    return not domain.is_mutating(str(name))


def _entry_to_case(entry: dict, split: str, domain: Domain) -> Case:
    entry_id = str(entry["id"])
    expect = entry["expect"]
    candidates = entry.get("candidates")
    if candidates is not None:
        unknown = [name for name in candidates if domain.get(name) is None]
        if unknown:
            raise ValueError(
                f"{entry_id}: offers candidate(s) domain {domain.name!r} does not have: "
                f"{', '.join(map(str, unknown))}"
            )
    tags: list[str] = []
    if entry.get("kind"):
        tags.append(str(entry["kind"]))
    if entry.get("class"):
        tags.append(str(entry["class"]))
    if entry.get("source"):
        tags.append(str(entry["source"]))
    if candidates is not None:
        tags.append("nocand")
    text = None if split in _HELDOUT_TAGS else str(entry.get("text", ""))
    return Case(
        id=entry_id,
        split=split,
        text=text,
        candidates=tuple(candidates) if candidates is not None else None,
        expect=dict(expect),
        read_only=_read_only_for(expect, domain),
        tags=tuple(tags),
        source_id=entry.get("source_id"),
    )


def load_case_set(
    manifest: Mapping[str, Any] | str | Path,
    split: str,
    *,
    domain: Domain,
    include_heldout: bool = False,
) -> tuple[Case, ...]:
    """Load every case on *split* from the split file the manifest names.

    Raises :class:`HeldOutAccessError` for a held-out split unless
    *include_heldout*; even then every returned case's ``text`` is ``None``.
    """
    if split not in SPLIT_TAGS:
        raise ValueError(f"unknown split tag {split!r}, must be one of {SPLIT_TAGS}")
    if split in _HELDOUT_TAGS and not include_heldout:
        raise HeldOutAccessError(
            f"split {split!r} is held-out; pass include_heldout=True to load it "
            "(and even then, no request text is returned)"
        )

    paths = load_manifest(manifest)
    if split not in paths:
        raise KeyError(f"manifest has no path for split {split!r}")

    raw = _read_json(paths[split])
    entries = raw.get("entries", []) if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ValueError(f"split file {paths[split]!r} has no usable 'entries' list")

    return tuple(_entry_to_case(entry, split, domain) for entry in entries)


def _load_ids(train_split_path: str | Path) -> frozenset[str]:
    raw = _read_json(train_split_path)
    entries = raw.get("entries", []) if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ValueError(f"train split file {train_split_path!r} has no usable 'entries' list")
    return frozenset(str(entry["id"]) for entry in entries)


def training_overlap(case_ids: Iterable[str], train_split_path: str | Path) -> tuple[str, ...]:
    """Refuse to score a checkpoint on any id present in its own train split.

    Raises :class:`TrainingOverlapError` naming every overlapping id, sorted;
    returns the (empty) overlap tuple otherwise.
    """
    train_ids = _load_ids(train_split_path)
    overlap = tuple(sorted(set(case_ids) & train_ids))
    if overlap:
        raise TrainingOverlapError(overlap)
    return overlap
