"""One fixed world snapshot for grounding every measured proposal.

A measurement grounds proposed arguments against a world. Grounding against
the live machine makes two runs (or the stock and tuned models) disagree for
reasons that have nothing to do with the models, so a measure stage grounds
against one fixed snapshot instead and records its sha256.

The snapshot's shape is the :class:`~jev_factory.domain.model.Domain`'s own
world schema (:attr:`Domain.world_schema`, checked by
:meth:`Domain.check_world`) plus two strings, ``source`` and ``created``.
:func:`build_snapshot` fills each groundable kind's world field from the
kind's live lookup (or, for a kind that declares none, the base world's own
field) plus every value the given split files expect for an argument of that
kind, spelled the way the world spells it. Every other schema field is copied
from the base world (by default the domain's seed-corpus world).

Usage::

    python -m jev_factory.measure.snapshot --domain <module|file.json> \\
        --out snapshot.json --from-split val.json --from-split test.json

prints counts only, never a value (a split's argument values are protected
data).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.domain.model import Domain, GroundKind

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/measure.py",
    "commit": "9debdc6",
    "adaptations": [
        "deviation d1's snapshot (lines 1784-1894: SNAPSHOT_KEYS, load_snapshot,"
        " snapshot_runner, _lookup_name, build_snapshot, run_snapshot) generalised from"
        " nvsh's services/containers to the Domain's world schema and ground kinds",
        "systemctl/docker lookup argv (nvsh.ops.ground._KINDS) becomes GroundKind.lookup;"
        " a kind with no lookup takes the base world's field",
        "_lookup_name's '.service' spelling becomes the kind's first declared suffix",
        "the grounding runner is gone: Domain.ground reads the snapshot dict directly",
        "the `measure.py snapshot` subcommand becomes python -m jev_factory.measure.snapshot",
    ],
    "licence": "Apache-2.0",
}

#: The two provenance strings every snapshot carries beside the world fields.
META_KEYS = ("source", "created")


class SnapshotError(ValueError):
    """A snapshot that cannot be read, fails the domain's world schema, or cannot be built."""

    def __init__(self, message: str, *, env: bool = False) -> None:
        super().__init__(message)
        #: True when the machine, not the input, is at fault (a lookup failed).
        self.env = env


def load_snapshot(path: Path, domain: Domain) -> tuple[dict, str]:
    """``(snapshot, sha256 of the file)``; a malformed snapshot raises :class:`SnapshotError`."""
    try:
        data = Path(path).read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"cannot read ground snapshot {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SnapshotError(f"{path} is not a ground snapshot object")
    problems = list(domain.check_world(raw))
    for kind in domain.ground_kinds:
        values = raw.get(kind.world_field)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            problems.append(f"{kind.world_field} must be a list of names")
    for key in META_KEYS:
        if not isinstance(raw.get(key), str):
            problems.append(f"{key} must be a string")
    if problems:
        raise SnapshotError(f"{path}: " + "; ".join(dict.fromkeys(problems)))
    return raw, hashlib.sha256(data).hexdigest()


def world_spelling(kind: GroundKind, value: str, known: Sequence[str]) -> str:
    """How the world spells *value*: a known spelling, else with the kind's first suffix."""
    wanted = {spelling.casefold() for spelling in kind.spellings(value)}
    for name in known:
        if name.casefold() in wanted:
            return name
    if kind.suffixes and not any(value.endswith(suffix) for suffix in kind.suffixes):
        return value + kind.suffixes[0]
    return value


def _grounded_args(domain: Domain) -> dict[str, dict[str, str]]:
    """operation -> {argument name: ground kind name} for every grounded argument."""
    return {
        op.name: {spec.name: spec.ground for spec in op.args if spec.ground is not None}
        for op in domain.operations
    }


def _split_entries(path: Path) -> list:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"cannot read split file {path}: {exc}") from exc
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise SnapshotError(f"{path} is not a split file ({{header, entries}})")
    return entries


def _kind_values(kind: GroundKind, base: Mapping[str, Any]) -> list[str]:
    """The kind's live lookup (or the base world's field), its string values only."""
    if kind.lookup is not None:
        try:
            values = kind.lookup()
        except Exception as exc:  # noqa: BLE001 -- any failed lookup refuses the build
            raise SnapshotError(
                f"cannot list this machine's {kind.plural}: {type(exc).__name__}", env=True
            ) from exc
    else:
        values = base.get(kind.world_field) or []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SnapshotError(f"the {kind.plural} lookup did not return a list", env=True)
    return [v for v in values if isinstance(v, str)]


def _expected_args(item: object) -> tuple[str, dict] | None:
    """``(operation, args)`` a split entry expects, or None when it names no args object."""
    expect = item.get("expect") if isinstance(item, dict) else None
    args = expect.get("args") if isinstance(expect, dict) else None
    if not isinstance(args, dict):
        return None
    return str(expect.get("operation")), args


def _entry_values(
    domain: Domain,
    grounded: Mapping[str, Mapping[str, str]],
    item: object,
    live: Mapping[str, list[str]],
) -> list[tuple[str, str]]:
    """``(ground kind name, world spelling)`` for each grounded value one split entry expects."""
    expected = _expected_args(item)
    if expected is None:
        return []
    operation, args = expected
    values: list[tuple[str, str]] = []
    for arg, kind_name in grounded.get(operation, {}).items():
        value = args.get(arg)
        kind = domain.ground_kind(kind_name)
        if isinstance(value, str) and value and kind is not None:
            values.append((kind.name, world_spelling(kind, value, live[kind.name])))
    return values


def _split_values(
    domain: Domain, splits: Sequence[Path], live: Mapping[str, list[str]]
) -> dict[str, set[str]]:
    """ground kind name -> every value the split files expect, spelled the world's way."""
    grounded = _grounded_args(domain)
    found: dict[str, set[str]] = {kind.name: set() for kind in domain.ground_kinds}
    for path in splits:
        for item in _split_entries(path):
            for kind_name, spelled in _entry_values(domain, grounded, item, live):
                found[kind_name].add(spelled)
    return found


def build_snapshot(
    domain: Domain,
    splits: Sequence[Path],
    *,
    source: str,
    created: str,
    base_world: Mapping[str, Any] | None = None,
) -> tuple[dict, dict[str, int]]:
    """This machine's (or the base world's) values plus every split value, and counts only."""
    base = dict(base_world or {})
    live = {kind.name: _kind_values(kind, base) for kind in domain.ground_kinds}
    found = _split_values(domain, splits, live)
    snapshot: dict[str, Any] = {
        field.name: base[field.name] for field in domain.world_schema if field.name in base
    }
    counts: dict[str, int] = {}
    for kind in domain.ground_kinds:
        key, mine = kind.world_field, set(live[kind.name])
        snapshot[key] = sorted(mine | found[kind.name])
        counts[key] = len(snapshot[key])
        counts[f"{key}_live"] = len(mine)
        counts[f"{key}_splits_only"] = len(found[kind.name] - mine)
    snapshot["source"] = source
    snapshot["created"] = created
    problems = domain.check_world(snapshot)
    if problems:
        raise SnapshotError(
            "the snapshot misses the domain's world schema: "
            + "; ".join(problems)
            + " (pass --base-world with a corpus whose world has them)"
        )
    return snapshot, counts


def _today() -> str:  # wall clock
    return datetime.date.today().isoformat()


def main(argv: list[str] | None = None, *, today: Callable[[], str] = _today) -> int:
    from jev_factory.domain.validate import load_domain

    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.measure.snapshot",
        description="Write a fixed grounding snapshot; prints counts only.",
    )
    parser.add_argument("--domain", required=True, help="domain module or JSON domain file")
    parser.add_argument("--out", required=True, help="snapshot JSON file to write")
    parser.add_argument(
        "--from-split",
        action="append",
        default=[],
        help="split file whose grounded argument values are added; repeat",
    )
    parser.add_argument(
        "--base-world",
        default=None,
        help="corpus file whose world fills the non-grounded fields (default: the seed corpus)",
    )
    parser.add_argument("--source", default=None, help="what the snapshot was taken from")
    parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        domain = load_domain(args.domain)
        if out.exists() and not args.force:
            raise SnapshotError(f"{out} already exists; pass --force to overwrite it")
        if args.base_world:
            raw = json.loads(Path(args.base_world).read_text(encoding="utf-8"))
            base = raw.get("world") if isinstance(raw, dict) else None
        elif domain.seed_corpus is not None:
            base = dict(domain.load_seed_corpus().world)
        else:
            base = None
        splits = [Path(path) for path in args.from_split]
        source = args.source or (
            "this machine's lookups plus the grounded argument values of"
            f" {len(splits)} split file(s)"
        )
        snapshot, counts = build_snapshot(
            domain, splits, source=source, created=today(), base_world=base
        )
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if exc.env else 1
    except (OSError, ValueError) as exc:  # DomainError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parts = [
        f"{counts[kind.world_field]} {kind.world_field} ({counts[kind.world_field + '_live']}"
        f" from this machine, {counts[kind.world_field + '_splits_only']} only from the"
        " split files)"
        for kind in domain.ground_kinds
    ]
    print(f"wrote {out}: " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
