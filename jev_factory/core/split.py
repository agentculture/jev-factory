"""Seeded train/val/test split of a domain's corpus (nvsh issues 39 and 53).

Splits a corpus's ``entries`` into three sides, train, val and test (test is
only measured on final runs), stratified by *expectation kind*
(``"operation"``, ``"escalate"`` or ``"explain"``, from each entry's ``expect``
block), and writes ``train.json``, ``val.json`` and ``test.json`` into the
out-dir in the ``{"header": ..., "entries": [...]}`` corpus shape. Splitting is
seeded (``random.Random(seed)``) and each group is sorted by id before it is
shuffled, so the same seed always yields the identical split regardless of
JSON object order or ``PYTHONHASHSEED``. Every output entry carries a
``source_id`` (its own ``id`` unless it already has one), so a variation of an
entry inherits its original's side and never appears on two sides.

The corpus is never hard-coded: pass ``--corpus`` (repeatable in v2), or call
:func:`main` with a :class:`~jev_factory.domain.model.Domain`, whose seed
corpus is used when no ``--corpus`` is given.

v2 mode (``--version``, more than one ``--corpus``, ``--val-size`` or
``--test-size`` or ``--train-only``) takes absolute val/test target sizes,
additionally stratifies by each entry's ``class`` field, and writes each side's
``header`` as the same plain string note (``Split '<side>' of corpus-<version>
(seed=N).``) that every downstream reader matches, with structured metadata
(version, seed, each input's path and sha256, side sizes) under a top-level
``split`` object. The val ids are then divided into fit and selection folds,
grouped by ``source_id`` (a source's variations never straddle the two folds),
recorded in ``val.json``'s ``split`` object and in ``folds.json``.
``--train-only CORPUS`` appends a corpus to the train side only.

The held-out split is refused as an input everywhere. The sealed held-out and
test sides are read back only through :func:`load_sealed`, which exposes ids,
counts and a sha256 and never an entry's text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/split.py",
    "commit": "9debdc6",
    "adaptations": [
        "the corpus comes from --corpus or the Domain's seed corpus; the hard-coded"
        " nvsh/tiers/corpus/dev.json default (:637-640) is removed",
        "refuse_if_under_nvsh (:216-227) becomes refuse_if_under_package: a v2 corpus is never"
        " written inside the installed jev_factory package",
        "_load_calibration_fit/calibration_fit.make_folds (:205-214, :492-497) become the"
        " local make_folds, taken from calibration_fit.py:165-200 (DEFAULT_FIT_FRACTION = 0.7,"
        " line 83; grouped by source_id via group_of; sorted ids, sorted group keys, seeded"
        " shuffle, n_fit = round(n_groups * fit_fraction)), because"
        " jev_factory.core.calibration does not exist yet; t16 may re-point it",
        "sys.path/_REPO_ROOT/importlib path loading removed in favour of package imports",
        "new load_sealed/SealedSide: read a sealed held-out or test side as ids, counts and"
        " sha256 only (never entry text)",
        "tests ported from tests/test_lfm_finetune_split.py (and _split_v2_readers' split.py"
        " parts) onto toy-domain fixtures",
    ],
    "licence": "Apache-2.0",
}

#: The directory a v2 corpus must never be written inside (the installed package).
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Mirrors build_dataset.py's own constant and refusal: the held-out split is
#: for judging a tuned model, never for building train/val/test splits from.
HELD_OUT_NAME = "held-out.json"

#: A committed default seed: identical input + this seed always yields the
#: identical split (pinned by tests/test_lfm_finetune_split.py).
DEFAULT_SEED = 39

#: train/val/test (decision c31: iterate on val, test only measured on final runs).
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

#: Every expectation kind the corpus schema knows about, in report order.
#: A corpus need not have every kind; a missing one is reported, not an error.
EXPECTATION_KINDS = ("operation", "escalate", "explain")

#: A v2 ``--version``: it lands in every side's header note, which has no spaces.
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def expectation_kind(expect: dict) -> str:
    """Classify one entry's ``expect`` block: "operation", "escalate" or "explain"."""
    if expect.get("escalate"):
        return "escalate"
    if expect.get("explain"):
        return "explain"
    if "operation" in expect:
        return "operation"
    raise ValueError(f"entry has an expect block the factory doesn't recognize: {expect!r}")


def _allocate(n: int, fractions: tuple[float, float, float]) -> list[int]:
    """How many of *n* items go to each side, by largest-remainder rounding.

    Once ``n`` reaches the number of sides, every side is guaranteed at
    least one item (borrowed from the largest side), so a kind present in
    the input with enough entries is present on every side.
    """
    raw = [f * n for f in fractions]
    counts = [int(r) for r in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(fractions)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(fractions)]] += 1
    if n >= len(fractions):
        for i, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(counts)), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] += 1
    return counts


def stratified_split(
    entries: list[dict],
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Split *entries* into train/val/test, stratified by expectation kind.

    Returns ``(sides, missing_kinds)``. *sides* maps each of
    :data:`SPLIT_NAMES` to a list of entries, each carrying its own
    ``source_id`` (c42). *missing_kinds* lists any of
    :data:`EXPECTATION_KINDS` entirely absent from *entries* -- reported,
    not an error, since a corpus may not yet have every kind.
    """
    if any(not math.isfinite(f) or not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError(f"each fraction must be between 0 and 1, got {fractions!r}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions!r}")

    id_counts = Counter(entry["id"] for entry in entries)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entry ids would split one source across sides: {duplicates}")

    # An entry that already carries a source_id (a variation, c42) is kept
    # with every other entry of that source: sources are what get split.
    sources: dict[str, list[dict]] = {}
    for entry in entries:
        sources.setdefault(entry.get("source_id", entry["id"]), []).append(entry)

    by_kind: dict[str, list[str]] = {kind: [] for kind in EXPECTATION_KINDS}
    for source_id, members in sources.items():
        kinds = {expectation_kind(member["expect"]) for member in members}
        if len(kinds) > 1:
            raise ValueError(f"source {source_id!r} mixes expectation kinds {sorted(kinds)}")
        by_kind[kinds.pop()].append(source_id)

    missing_kinds = [kind for kind in EXPECTATION_KINDS if not by_kind[kind]]

    sides: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for kind in EXPECTATION_KINDS:
        # Sort before shuffling: JSON array order is already deterministic,
        # but sorting makes that explicit and platform-independent rather
        # than relying on it.
        group = sorted(by_kind[kind])
        random.Random(seed).shuffle(group)  # nosec B311 - seeded, deterministic split
        counts = _allocate(len(group), fractions)
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for source_id in group[offset : offset + count]:
                for entry in sources[source_id]:
                    sides[name].append({**entry, "source_id": source_id})
            offset += count

    for name in SPLIT_NAMES:
        sides[name].sort(key=lambda entry: entry["id"])

    return sides, missing_kinds


def absent_from_sides(sides: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """``(kind, side)`` pairs for a kind present in the split but missing from a side.

    A kind with fewer entries than there are sides cannot reach every side;
    this names each gap so the caller reports it instead of passing silently.
    """
    present = {expectation_kind(e["expect"]) for side in sides.values() for e in side}
    return [
        (kind, name)
        for kind in EXPECTATION_KINDS
        if kind in present
        for name in SPLIT_NAMES
        if not any(expectation_kind(e["expect"]) == kind for e in sides[name])
    ]


def build_splits(
    corpus: Path,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> tuple[dict[str, list[dict]], list[str], str]:
    """Load *corpus* (as plain JSON, not via ``load_corpus``) and split it.

    Refuses the held-out split, exactly as ``build_dataset.py``'s ``build()``
    does. Returns ``(sides, missing_kinds, header)``.
    """
    if corpus.name == HELD_OUT_NAME:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    with open(corpus, encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    header = raw.get("header", "") if isinstance(raw, dict) else ""
    sides, missing_kinds = stratified_split(entries, seed, fractions)
    return sides, missing_kinds, header


def refuse_if_under_package(path: Path) -> None:
    """Raise :class:`ValueError` when *path* resolves inside the ``jev_factory`` package.

    Committed seed corpora live under the package; a v2 corpus is a run
    work-dir or private-data artifact and must never land there.
    """
    resolved = path.resolve()
    if resolved == _PACKAGE_DIR or _PACKAGE_DIR in resolved.parents:
        raise ValueError(f"refusing to write inside the jev_factory package: {path}")


DEFAULT_FIT_FRACTION = 0.7


def make_folds(
    ids: list[str],
    seed: int,
    fit_fraction: float = DEFAULT_FIT_FRACTION,
    group_of: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split *ids* into disjoint, sorted ``(fit_ids, selection_ids)``, seeded.

    Ids sharing a ``group_of`` key land on the same side (a source's variations
    never straddle the folds). Ids are sorted before shuffling, so the result
    never depends on input order or ``PYTHONHASHSEED``.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must be between 0 and 1 (exclusive), got {fit_fraction!r}")
    members: dict[str, list[str]] = {}
    for entry_id in sorted({str(i) for i in ids}):
        key = str((group_of or {}).get(entry_id, entry_id))
        members.setdefault(key, []).append(entry_id)
    keys = sorted(members)
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)  # nosec B311 - seeded, deterministic split
    n_fit = max(0, min(len(shuffled), round(len(shuffled) * fit_fraction)))
    fit_keys = set(shuffled[:n_fit])
    fit_ids = sorted(i for k in fit_keys for i in members[k])
    selection_ids = sorted(i for k in keys if k not in fit_keys for i in members[k])
    return fit_ids, selection_ids


def sha256_file(path: Path) -> str:
    """The hex sha256 digest of *path*'s bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_corpora(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Merge every *paths* corpus file's entries, de-duplicated by ``id``.

    Refuses the held-out split exactly like :func:`build_splits`. The first
    occurrence of a given ``id`` wins; a later duplicate (e.g. the same
    entry present in both ``dev.json`` and a drafted source file) is
    dropped silently -- the caller's ``merged=`` accounting in the printed
    summary is where that shows up. Returns ``(entries, sources)`` where
    *sources* is ``[{"path": str, "sha256": str}, ...]`` in input order, for
    the v2 header.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    sources: list[dict] = []
    for path in paths:
        if path.name == HELD_OUT_NAME:
            raise ValueError(
                "the held-out split is for judging a tuned model, never for training it"
            )
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        entries = raw.get("entries", []) if isinstance(raw, dict) else raw
        for entry in entries:
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            merged.append(entry)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    return merged, sources


def add_train_only(
    sides: dict[str, list[dict]], sources: list[dict], paths: list[Path]
) -> tuple[dict[str, list[dict]], list[dict]]:
    """*sides* with every *paths* corpus entry appended to train only (deviation d3).

    Issue 53's corpus v2 keeps ``dev.json`` -- scorer-b1's own training data and
    the issue-46 validation/test sides the lead has read -- off the fresh
    validation and test sides: those come only from the corpora that were
    split. A train-only entry whose ``id`` is already on any side is refused
    (it would put one source on two sides); each carries ``train_only: true``
    and its own ``source_id`` (default: its ``id``). Returns the new sides and
    *sources* extended with each train-only file, marked ``"train_only": true``.
    """
    taken = {entry["id"] for side in sides.values() for entry in side}
    extra, merged_sources = [], []
    for path in paths:
        more, more_sources = merge_corpora([path])
        for entry in more:
            if entry["id"] in taken:
                raise ValueError(
                    f"train-only entry {entry['id']!r} ({path}) is already on a split side"
                )
            taken.add(entry["id"])
            extra.append(
                {**entry, "source_id": entry.get("source_id", entry["id"]), "train_only": True}
            )
        merged_sources.extend({**src, "train_only": True} for src in more_sources)
    train = sorted([*sides["train"], *extra], key=lambda entry: entry["id"])
    return {**sides, "train": train}, [*sources, *merged_sources]


def _class_of(entry: dict) -> object:
    return entry.get("class")


def _stratify_by_class(
    source_ids: list[str],
    sources: dict[str, list[dict]],
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, list[str]]:
    """Assign *source_ids* to train/val/test, proportional to *fractions* per class.

    Groups ids by the first member's ``class`` field (``None`` when absent),
    shuffles each class group independently (seeded, so deterministic), then
    allocates *each class's own group* across the three sides with
    :func:`_allocate`'s largest-remainder rounding -- the same rule already
    used to divide entries across sides by expectation kind, now applied a
    level deeper, per class within a kind.

    A prior version instead built one contiguous order (round-robin across
    classes) and sliced it once for all three sides; with a rare class
    (e.g. 10 of 100 entries) and small val/test targets, the whole class
    could land entirely in train and never reach val or test (#53 review
    finding P2). Allocating per class avoids that: :func:`_allocate`
    guarantees every side gets at least one member of a class once that
    class has at least as many source ids as there are sides (>= 3 here).

    A corpus with no ``class`` field on any entry in the group behaves like
    a single class and reduces to the old per-kind allocation.
    """
    by_class: dict[object, list[str]] = {}
    for source_id in source_ids:
        cls = _class_of(sources[source_id][0])
        by_class.setdefault(cls, []).append(source_id)

    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for cls in sorted(by_class, key=lambda c: (c is None, str(c))):
        group = sorted(by_class[cls])
        random.Random(seed).shuffle(group)  # nosec B311 - seeded, deterministic split
        counts = _allocate(len(group), fractions)
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            assigned[name].extend(group[offset : offset + count])
            offset += count
    return assigned


def stratified_split_sized(
    entries: list[dict],
    seed: int,
    val_size: int,
    test_size: int,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Like :func:`stratified_split`, but val/test are absolute target counts.

    The rest goes to train. Internally this still uses the same
    largest-remainder allocation per expectation kind that
    :func:`stratified_split` uses (via fractions derived from the target
    sizes over the total entry count), so a kind's own proportional share
    of val/test is preserved; the difference is :func:`_stratify_by_class`,
    used here instead of a plain per-kind shuffle so entries sharing a
    ``class`` field (escalation entries) are themselves allocated across
    train/val/test proportionally to *fractions*, one class at a time --
    a rare class no longer risks being swept entirely into train (#53
    review finding P2). Because each kind (and each class within it) rounds
    its own share independently, the *total* val/test size lands close to,
    but is not always exactly, ``val_size``/``test_size`` -- the acceptance
    target is itself approximate ("test ~150").
    """
    total = len(entries)
    if total == 0:
        raise ValueError("cannot split an empty corpus")
    val_frac = val_size / total
    test_frac = test_size / total
    train_frac = 1.0 - val_frac - test_frac
    if train_frac < 0:
        raise ValueError(
            f"val_size + test_size ({val_size + test_size}) exceeds the corpus size ({total})"
        )
    fractions = (train_frac, val_frac, test_frac)

    id_counts = Counter(entry["id"] for entry in entries)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entry ids would split one source across sides: {duplicates}")

    sources: dict[str, list[dict]] = {}
    for entry in entries:
        sources.setdefault(entry.get("source_id", entry["id"]), []).append(entry)

    by_kind: dict[str, list[str]] = {kind: [] for kind in EXPECTATION_KINDS}
    for source_id, members in sources.items():
        kinds = {expectation_kind(member["expect"]) for member in members}
        if len(kinds) > 1:
            raise ValueError(f"source {source_id!r} mixes expectation kinds {sorted(kinds)}")
        by_kind[kinds.pop()].append(source_id)

    missing_kinds = [kind for kind in EXPECTATION_KINDS if not by_kind[kind]]

    sides: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for kind in EXPECTATION_KINDS:
        assigned = _stratify_by_class(sorted(by_kind[kind]), sources, seed, fractions)
        for name in SPLIT_NAMES:
            for source_id in assigned[name]:
                for entry in sources[source_id]:
                    sides[name].append({**entry, "source_id": source_id})

    for name in SPLIT_NAMES:
        sides[name].sort(key=lambda entry: entry["id"])

    return sides, missing_kinds


def _write_side(
    out_dir: Path,
    name: str,
    entries: list[dict],
    header: str,
    corpus_name: str,
    seed: int,
    world: dict | None = None,
) -> Path:
    note = f"Split '{name}' of {corpus_name} (seed={seed})."
    payload: dict = {
        "header": f"{header} {note}".strip(),
        "entries": entries,
    }
    # The corpus world (platform, fixture machine state) travels with every
    # side: without it a builder falls back to an unknown platform and the
    # system brief no longer matches what Tier 2 was measured with.
    if world is not None:
        payload["world"] = world
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def v2_corpus_name(version: str) -> str:
    """The corpus name a v2 side's header note uses: ``corpus-<version>``.

    A v2 split merges several ``--corpus`` inputs, so no one input's file
    name describes it; the version name is stable and deterministic, and the
    inputs themselves are listed (with sha256) under the ``split`` object.
    """
    return f"corpus-{version}"


def check_version(version: str) -> None:
    """Refuse a ``--version`` the side readers could misread.

    The version lands inside every side's ``Split '<side>' of
    corpus-<version> (seed=N).`` note, which readers match with ``\\S+``
    (no whitespace) and scan for the words ``test``/``val``/``train`` and
    ``held-out``: a version naming a side would make every side look like it.
    """
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(
            f"--version {version!r} must be letters, digits, '.', '_' or '-' (e.g. v2)"
        )
    words = {word for word in re.split(r"[^a-z0-9]+", version.lower()) if word}
    compact = re.sub(r"[^a-z0-9]", "", version.lower())
    if words & set(SPLIT_NAMES) or "heldout" in compact:
        raise ValueError(f"--version {version!r} must not name a split side")


def _write_side_v2(
    out_dir: Path,
    name: str,
    entries: list[dict],
    metadata: dict,
    world: dict | None = None,
) -> Path:
    """Like :func:`_write_side`, plus a structured ``split`` object (v2).

    ``header`` stays a string holding exactly :func:`_write_side`'s note
    (so every reader's ``Split '<side>' of ... (seed=N).`` match still
    works), followed by a sentence pointing at ``split``. *metadata* --
    ``version``, ``seed``, every input's ``sources`` (path + sha256) and
    the resulting ``sizes``, plus ``fold_seed``/``fit_ids``/``selection_ids``
    on the val side only -- goes under the top-level ``split`` key. Paths
    stay out of the header: a side reader scans it for ``test`` and
    ``held-out``, and an input path could innocently contain either.
    """
    corpus_name = v2_corpus_name(metadata["version"])
    note = f"Split '{name}' of {corpus_name} (seed={metadata['seed']})."
    header = (
        f"{note} Corpus {metadata['version']}, merged by split.py from "
        f"{len(metadata['sources'])} corpus file(s); its version, seed, sources "
        'and sizes are under "split".'
    )
    payload: dict = {"header": header, "split": {**metadata, "side": name}, "entries": entries}
    if world is not None:
        payload["world"] = world
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _main_v2(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    corpus_paths: list[Path],
    out_dir: Path,
) -> int:
    if args.val_size is None or args.test_size is None:
        parser.error(
            "v2 mode (--version, multiple --corpus, or --val-size/--test-size) "
            "requires both --val-size and --test-size"
        )
    if args.fold_seed is None:
        parser.error("v2 mode requires --fold-seed (seeds the fit/selection fold split)")
    version = args.version or "v2"

    train_only_paths = [Path(p) for p in (args.train_only or [])]
    try:
        check_version(version)
        entries, sources = merge_corpora(corpus_paths)
        sides, missing_kinds = stratified_split_sized(
            entries, args.seed, args.val_size, args.test_size
        )
        if train_only_paths:
            sides, sources = add_train_only(sides, sources, train_only_paths)
            entries = entries + [e for e in sides["train"] if e.get("train_only")]
    except ValueError as exc:
        parser.error(str(exc))
    gaps = absent_from_sides(sides)
    if gaps:
        described = ", ".join(f"{kind!r} on {name}" for kind, name in gaps)
        parser.error(f"too few entries to reach every side: missing {described}")

    world = None
    for path in corpus_paths:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict) and raw.get("world") is not None:
            world = raw["world"]
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {name: len(sides[name]) for name in SPLIT_NAMES}

    val_ids = [entry["id"] for entry in sides["val"]]
    # Group by source_id so a source's variations never split across the fit
    # and selection folds (#53 review finding P1).
    val_group_of = {entry["id"]: entry.get("source_id", entry["id"]) for entry in sides["val"]}
    fit_ids, selection_ids = make_folds(val_ids, args.fold_seed, group_of=val_group_of)

    for name in SPLIT_NAMES:
        metadata = {
            "version": version,
            "seed": args.seed,
            "sources": sources,
            "sizes": sizes,
        }
        if name == "val":
            metadata["fold_seed"] = args.fold_seed
            metadata["fit_ids"] = fit_ids
            metadata["selection_ids"] = selection_ids
        _write_side_v2(out_dir, name, sides[name], metadata, world)

    write_json(
        out_dir / "folds.json",
        {
            "seed": args.fold_seed,
            "source": str(out_dir / "val.json"),
            "fit_ids": fit_ids,
            "selection_ids": selection_ids,
        },
    )

    for kind in missing_kinds:
        print(f"note: no {kind!r} entries in the merged corpus; not present on any side")
    for name in SPLIT_NAMES:
        print(f"{name}={len(sides[name])}")
    plural = "y" if len(entries) == 1 else "ies"
    print(f"merged {len(entries)} unique entr{plural} from {len(corpus_paths)} corpus file(s)")
    return 0


def main(argv: list[str] | None = None, domain: Domain | None = None) -> int:
    """Run the splitter; *domain*'s seed corpus is the input when no ``--corpus`` is given."""
    parser = argparse.ArgumentParser(prog="jev-split", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="corpus file to read; repeatable in v2 mode (default: the domain's seed corpus)",
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_FRACTIONS[0])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_FRACTIONS[1])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_FRACTIONS[2])
    parser.add_argument(
        "--val-size", type=int, default=None, help="v2: absolute validation size (e.g. 150)"
    )
    parser.add_argument(
        "--test-size", type=int, default=None, help="v2: absolute test size (e.g. 150)"
    )
    parser.add_argument("--version", default=None, help="v2: corpus version name, e.g. v2")
    parser.add_argument(
        "--train-only",
        action="append",
        default=None,
        help=(
            "v2: corpus file whose entries go to the train side only, never val/test "
            "(repeatable; issue 53 deviation d3)"
        ),
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help="v2: seed for the fit/selection fold split of the written val ids",
    )
    args = parser.parse_args(argv)

    if args.corpus:
        corpus_paths = [Path(p) for p in args.corpus]
    elif domain is not None and domain.seed_corpus is not None:
        corpus_paths = [domain.seed_corpus]
    else:
        parser.error("no corpus: pass --corpus FILE (or run with a domain that has a seed corpus)")
    out_dir = Path(args.out_dir)
    try:
        refuse_if_under_package(out_dir)
    except ValueError as exc:
        parser.error(str(exc))

    v2_mode = (
        bool(args.version)
        or args.val_size is not None
        or args.test_size is not None
        or len(corpus_paths) > 1
        or bool(args.train_only)
    )
    if v2_mode:
        return _main_v2(parser, args, corpus_paths, out_dir)

    corpus = corpus_paths[0]
    fractions = (args.train_frac, args.val_frac, args.test_frac)
    try:
        sides, missing_kinds, header = build_splits(corpus, args.seed, fractions)
    except ValueError as exc:
        parser.error(str(exc))
    gaps = absent_from_sides(sides)
    if gaps:
        described = ", ".join(f"{kind!r} on {name}" for kind, name in gaps)
        parser.error(f"too few entries to reach every side: missing {described}")

    with open(corpus, encoding="utf-8") as handle:
        raw = json.load(handle)
    world = raw.get("world") if isinstance(raw, dict) else None

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_side(out_dir, name, sides[name], header, corpus.name, args.seed, world)

    for kind in missing_kinds:
        print(f"note: no {kind!r} entries in {corpus.name}; not present on any side")
    for name in SPLIT_NAMES:
        print(f"{name}={len(sides[name])}")
    return 0


@dataclass(frozen=True)
class SealedSide:
    """What a sealed held-out or test side discloses: ids, counts and a sha256, never text."""

    path: str
    sha256: str
    ids: tuple[str, ...]
    count: int
    by_kind: tuple[tuple[str, int], ...]

    def __repr__(self) -> str:
        return f"SealedSide(count={self.count}, sha256={self.sha256[:12]}...)"


def load_sealed(path: Path) -> SealedSide:
    """Read a sealed side (held-out or test) as ids, counts and sha256 only.

    The entries' text, expected answers and every other field are dropped
    before anything is returned, so a caller cannot print or log them by
    accident. A malformed file raises :class:`ValueError` naming the path only.
    """
    path = Path(path)
    try:
        digest = sha256_file(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw["entries"] if isinstance(raw, dict) else raw
        ids = tuple(str(entry["id"]) for entry in entries)
        kinds = Counter(expectation_kind(entry["expect"]) for entry in entries)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{path}: not a readable split file ({type(exc).__name__})") from None
    return SealedSide(
        path=str(path),
        sha256=digest,
        ids=ids,
        count=len(ids),
        by_kind=tuple(sorted(kinds.items())),
    )


if __name__ == "__main__":
    raise SystemExit(main())
