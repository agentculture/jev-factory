"""Run dataset build: licence allowlist, grouped folds, sealed sides, length filter.

Input is a list of :class:`~jev_factory.distill.records.Record` (typed-decisions
and the jev-bench configs). The build

1. refuses duplicate ids,
2. keeps only records whose licence is exactly ``apache-2.0`` or ``mit``
   (case-insensitive, whitespace-trimmed; any compound or ambiguous string such
   as ``cc-by-4.0 (PolyAI; mirror: mit)`` is NOT allowed) and writes a
   per-source licence table into the manifest for operator review,
3. drops every record whose rendered state + questions exceed ``max_tokens``
   student tokens (counted by an injectable counter; nothing is truncated) and
   counts the drops per source,
4. keeps each source's own ``test`` split as the test side, never trains on it,
5. carves a sealed held-out set, then grouped fit/selection folds, out of each
   source's ``train`` split with :func:`jev_factory.core.split.make_folds`
   (a group is every record sharing one state, so duplicates never straddle).

The real counter is :func:`subprocess_token_counter`, which runs the student
tokenizer inside the pinned student environment; tests inject a stdlib counter.
The sealed held-out and test sides are exposed as :class:`SealedSide`
(ids, count, sha256, never text). Stdlib only at import time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404 - runs the pinned student env's own python
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.core.split import make_folds
from jev_factory.distill.records import Record

STAGE = "distill-dataset"
MAX_TOKENS = 4096
DEFAULT_SEED = 27
DEFAULT_HELD_OUT_FRACTION = 0.10
DEFAULT_FIT_FRACTION = 0.70
#: Exact, normalised licence strings that pass; everything else is excluded.
ALLOWED_LICENCES: frozenset[str] = frozenset({"apache-2.0", "mit"})
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
SIDES = ("fit", "selection", "held_out", "test")

TokenCounter = Callable[[str], int]


def licence_allowed(licence: str) -> bool:
    """True only for exactly Apache-2.0 or MIT; compound/ambiguous strings are refused."""
    return str(licence).strip().lower() in ALLOWED_LICENCES


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_prompt(record: Record) -> str:
    """The text whose student-token length is bounded: the state plus the questions."""
    questions = {qid: q.to_json() for qid, q in sorted(record.questions.items())}
    return f"STATE:\n{record.state}\nQUESTIONS:\n" + json.dumps(
        questions, sort_keys=True, ensure_ascii=False
    )


_COUNT_SCRIPT = (
    "import json, sys\n"
    "from transformers import AutoTokenizer\n"
    "tok = AutoTokenizer.from_pretrained(sys.argv[1])\n"
    "texts = json.load(sys.stdin)\n"
    "print(json.dumps([len(tok(t, add_special_tokens=False)['input_ids']) for t in texts]))\n"
)


def subprocess_token_counter(python: str, tokenizer_dir: Path | str) -> TokenCounter:
    """A counter backed by the student tokenizer in the pinned student env (a subprocess).

    The returned callable counts one text; its ``many`` attribute counts a batch
    in a single subprocess. The tokenizer files must already be local.
    """

    def many(texts: Sequence[str]) -> list[int]:
        try:
            proc = subprocess.run(  # nosec B603 - argv list, no shell
                [python, "-I", "-c", _COUNT_SCRIPT, str(tokenizer_dir)],
                input=json.dumps(list(texts)),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"cannot run the student env python {python!r}: {exc}",
                remediation="point the counter at the pinned student environment's python",
            ) from None
        if proc.returncode != 0:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"student tokenizer counter exited {proc.returncode}",
                remediation=proc.stderr.strip()[-300:] or "check the student env",
            )
        try:
            counts = json.loads(proc.stdout)
        except ValueError:
            counts = None
        if not (
            isinstance(counts, list)
            and len(counts) == len(texts)
            and all(isinstance(c, int) and not isinstance(c, bool) for c in counts)
        ):
            raise CliError(code=EXIT_ENV_ERROR, message="student tokenizer counter output invalid")
        return counts

    def one(text: str) -> int:
        return many([text])[0]

    one.many = many  # type: ignore[attr-defined]
    return one


# ---------------------------------------------------------------------------
# Sealed sides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedSide:
    """What a sealed held-out or test side discloses: ids, counts and a sha256, never text."""

    path: str
    sha256: str
    ids: tuple[str, ...]
    count: int
    by_source: tuple[tuple[str, int], ...]

    def __repr__(self) -> str:
        return f"SealedSide(count={self.count}, sha256={self.sha256[:12]}...)"


def _sealed(path: str, records: Sequence[Record], digest: str) -> SealedSide:
    by_source = Counter(r.source for r in records)
    return SealedSide(
        path=path,
        sha256=digest,
        ids=tuple(sorted(r.id for r in records)),
        count=len(records),
        by_source=tuple(sorted(by_source.items())),
    )


def load_sealed(path: Path | str) -> SealedSide:
    """Read a sealed side (held-out or test) as ids, counts and sha256 only."""
    path = Path(path)
    try:
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        ids = [str(row["id"]) for row in rows]
        sources = Counter(str(row["source"]) for row in rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path}: not a readable sealed side ({type(exc).__name__})",
        ) from None
    return SealedSide(
        path=str(path),
        sha256=sha256_bytes(data),
        ids=tuple(sorted(ids)),
        count=len(ids),
        by_source=tuple(sorted(sources.items())),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """The built run dataset; sealed sides keep their records out of sight."""

    fit: tuple[Record, ...]
    selection: tuple[Record, ...]
    held_out: SealedSide
    test: SealedSide
    manifest: dict[str, Any]
    _held_out_records: tuple[Record, ...] = field(default=(), repr=False)
    _test_records: tuple[Record, ...] = field(default=(), repr=False)

    def all_records(self) -> tuple[Record, ...]:
        """The records that may be trained on or selected with (never held-out or test)."""
        return self.fit + self.selection


def _licence_table(records: Sequence[Record]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for r in records:
        row = table.setdefault(
            r.source, {"licences": set(), "total": 0, "excluded": 0, "allowed": True}
        )
        row["licences"].add(r.licence)
        row["total"] += 1
        if not licence_allowed(r.licence):
            row["excluded"] += 1
            row["allowed"] = False
    for row in table.values():
        row["licences"] = sorted(row["licences"])
    return dict(sorted(table.items()))


def _count_tokens(records: Sequence[Record], counter: TokenCounter) -> list[int]:
    texts = [render_prompt(r) for r in records]
    many = getattr(counter, "many", None)
    return list(many(texts)) if many else [counter(t) for t in texts]


def _carve(
    records: Sequence[Record], seed: int, held_out_fraction: float, fit_fraction: float
) -> tuple[list[Record], list[Record], list[Record]]:
    """Per-source (fit, selection, held_out) from train records, grouped by state."""
    by_id = {r.id: r for r in records}
    group_of = {r.id: sha256_bytes(r.state.encode("utf-8")) for r in records}
    pool, held = make_folds(list(by_id), seed, 1.0 - held_out_fraction, group_of)
    fit, selection = make_folds(pool, seed, fit_fraction, group_of)
    return [by_id[i] for i in fit], [by_id[i] for i in selection], [by_id[i] for i in held]


def build_dataset(
    records: Iterable[Record],
    *,
    token_counter: TokenCounter,
    seed: int = DEFAULT_SEED,
    max_tokens: int = MAX_TOKENS,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    fit_fraction: float = DEFAULT_FIT_FRACTION,
) -> Dataset:
    """Build the run dataset from *records*; see the module docstring."""
    recs = sorted(records, key=lambda r: r.id)
    ids = [r.id for r in recs]
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"duplicate record ids: {dupes[:5]}",
            remediation="record ids must be unique across sources",
        )
    licences = _licence_table(recs)
    licensed = [r for r in recs if licence_allowed(r.licence)]
    counts = _count_tokens(licensed, token_counter)
    kept = [r for r, n in zip(licensed, counts) if n <= max_tokens]
    dropped = Counter(r.source for r, n in zip(licensed, counts) if n > max_tokens)

    unused = Counter(r.split for r in kept if r.split not in (TRAIN_SPLIT, TEST_SPLIT))
    test_recs = [r for r in kept if r.split == TEST_SPLIT]
    fit: list[Record] = []
    selection: list[Record] = []
    held: list[Record] = []
    for source in sorted({r.source for r in kept}):
        train = [r for r in kept if r.source == source and r.split == TRAIN_SPLIT]
        if train:
            f, s, h = _carve(train, seed, held_out_fraction, fit_fraction)
            fit += f
            selection += s
            held += h
    fit.sort(key=lambda r: r.id)
    selection.sort(key=lambda r: r.id)
    held.sort(key=lambda r: r.id)
    manifest: dict[str, Any] = {
        "stage": STAGE,
        "status": "built",
        "knobs": {
            "seed": seed,
            "max_tokens": max_tokens,
            "held_out_fraction": held_out_fraction,
            "fit_fraction": fit_fraction,
            "licence_allowlist": sorted(ALLOWED_LICENCES),
        },
        "max_tokens": max_tokens,
        "licences": licences,
        "dropped_too_long": dict(sorted(dropped.items())),
        "unused_splits": dict(sorted(unused.items())),
        "counts": {
            "input": len(recs),
            "licence_excluded": len(recs) - len(licensed),
            "fit": len(fit),
            "selection": len(selection),
            "held_out": len(held),
            "test": len(test_recs),
        },
    }
    return Dataset(
        fit=tuple(fit),
        selection=tuple(selection),
        held_out=_sealed("", held, sha256_bytes(_dump(held))),
        test=_sealed("", test_recs, sha256_bytes(_dump(test_recs))),
        manifest=manifest,
        _held_out_records=tuple(held),
        _test_records=tuple(test_recs),
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _dump(records: Sequence[Record]) -> bytes:
    lines = [json.dumps(r.to_json(), sort_keys=True, ensure_ascii=False) for r in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def read_records(path: Path | str) -> list[Record]:
    """Read a training-side jsonl written by :func:`write_dataset` (not for sealed sides)."""
    text = Path(path).read_text(encoding="utf-8")
    return [Record.from_json(json.loads(line)) for line in text.splitlines() if line.strip()]


def write_dataset(ds: Dataset, out_dir: Path | str) -> dict[str, str]:
    """Write the four sides and ``manifest.json``; returns side name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sides = {
        "fit": ds.fit,
        "selection": ds.selection,
        "held_out": ds._held_out_records,
        "test": ds._test_records,
    }
    paths: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    for name, recs in sides.items():
        data = _dump(recs)
        path = out / f"{name}.jsonl"
        path.write_bytes(data)
        paths[name] = str(path)
        outputs[name] = {"path": path.name, "sha256": sha256_bytes(data), "count": len(recs)}
    manifest = dict(ds.manifest)
    manifest["outputs"] = outputs
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["manifest"] = str(mpath)
    return paths
