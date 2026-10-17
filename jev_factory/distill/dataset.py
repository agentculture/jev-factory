"""Run dataset build: licence allowlist, grouped folds, sealed sides, length filter.

Input is a list of :class:`~jev_factory.distill.records.Record` (typed-decisions
and the jev-bench configs). The build

1. refuses duplicate ids,
2. keeps only records whose licence is exactly ``apache-2.0``, ``mit`` or
   ``cc-by-4.0`` (case-insensitive, whitespace-trimmed; any compound or ambiguous
   string such as ``cc-by-4.0 (PolyAI; mirror: mit)`` is NOT allowed). A
   ``cc-by-4.0`` record also needs a non-empty ``attribution``. The manifest
   carries a per-source licence table and a per-licence table (counts and the
   attribution strings),
3. refuses every record with a question of more than ``max_slots`` options
   (``over_slots``; the student's ``num_slots``) or whose rendered state +
   questions exceed ``max_tokens`` student tokens (``over_length``; counted by an
   injectable counter). Nothing is truncated; refusals are counted by reason
   in ``refused`` and the length drops also per source,
4. keeps each source's own ``test`` split as the test side, never trains on it,
5. carves a sealed held-out set, then grouped train / fit / selection sides, out of
   each source's ``train`` split with :func:`jev_factory.core.split.make_folds`
   (a group is the record's ``group``, e.g. its workflow, when set, else every record
   sharing one state, so duplicates never straddle). Only the ``train`` side may
   become a training target; ``fit`` and ``selection`` are evaluation folds
   (deviation d2).

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
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile
from jev_factory.distill.records import ATTRIBUTION_LICENCES, Record

STAGE = "distill-dataset"
MAX_TOKENS = DEFAULT_2610.max_tokens
#: The student's ``num_slots`` (2610 strands_decider_config.json): most options per question.
DEFAULT_MAX_SLOTS = DEFAULT_2610.num_slots
DEFAULT_SEED = 27
DEFAULT_HELD_OUT_FRACTION = 0.10
#: Fractions of the pool left after the held-out carve; selection is the remainder.
#: Placeholders, exposed as ``build_dataset`` arguments and recorded in the manifest.
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_FIT_FRACTION = 0.15
#: Exact, normalised licence strings that pass; everything else is excluded.
ALLOWED_LICENCES: frozenset[str] = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
SIDES = ("train", "fit", "selection", "held_out", "test")

TokenCounter = Callable[[str], int]


def licence_allowed(licence: str) -> bool:
    """True only for exactly Apache-2.0, MIT or CC-BY-4.0; compound strings are refused."""
    return str(licence).strip().lower() in ALLOWED_LICENCES


def normalise_licence(licence: str) -> str:
    return str(licence).strip().lower()


def attribution_ok(record: Record) -> bool:
    """A cc-by-4.0 record needs a non-empty attribution string; other licences need none."""
    if normalise_licence(record.licence) not in ATTRIBUTION_LICENCES:
        return True
    return isinstance(record.attribution, str) and bool(record.attribution.strip())


def admitted(record: Record) -> bool:
    return licence_allowed(record.licence) and attribution_ok(record)


def option_count(record: Record) -> int:
    """The most options any one question of *record* offers."""
    return max((len(q.criteria) for q in record.questions.values()), default=0)


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

    train: tuple[Record, ...]
    fit: tuple[Record, ...]
    selection: tuple[Record, ...]
    held_out: SealedSide
    test: SealedSide
    manifest: dict[str, Any]
    _held_out_records: tuple[Record, ...] = field(default=(), repr=False)
    _test_records: tuple[Record, ...] = field(default=(), repr=False)

    def trainable_records(self) -> tuple[Record, ...]:
        """The ONLY records that may become training targets (teacher labels, losses)."""
        return self.train

    def eval_records(self) -> tuple[Record, ...]:
        """The evaluation folds (fit and selection); never training targets."""
        return self.fit + self.selection

    def all_records(self) -> tuple[Record, ...]:
        """Every non-sealed record (train + fit + selection); not a training set."""
        return self.train + self.fit + self.selection


def _licence_table(records: Sequence[Record]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for r in records:
        row = table.setdefault(
            r.source, {"licences": set(), "total": 0, "excluded": 0, "allowed": True}
        )
        row["licences"].add(r.licence)
        row["total"] += 1
        if not admitted(r):
            row["excluded"] += 1
            row["allowed"] = False
    for row in table.values():
        row["licences"] = sorted(row["licences"])
    return dict(sorted(table.items()))


def _per_licence_table(records: Sequence[Record]) -> dict[str, dict[str, Any]]:
    """Admitted records by normalised licence: count and the attribution strings seen."""
    table: dict[str, dict[str, Any]] = {}
    for r in records:
        row = table.setdefault(normalise_licence(r.licence), {"count": 0, "attributions": set()})
        row["count"] += 1
        if r.attribution:
            row["attributions"].add(r.attribution.strip())
    for row in table.values():
        row["attributions"] = sorted(row["attributions"])
    return dict(sorted(table.items()))


def _count_tokens(records: Sequence[Record], counter: TokenCounter) -> list[int]:
    texts = [render_prompt(r) for r in records]
    many = getattr(counter, "many", None)
    return list(many(texts)) if many else [counter(t) for t in texts]


def _carve(
    records: Sequence[Record],
    seed: int,
    held_out_fraction: float,
    train_fraction: float,
    fit_fraction: float,
) -> tuple[list[Record], list[Record], list[Record], list[Record]]:
    """Per-source (train, fit, selection, held_out) from train-split records, grouped."""
    by_id = {r.id: r for r in records}
    group_of = {
        r.id: (
            f"group:{r.source}:{r.group}"
            if r.group is not None
            else sha256_bytes(r.state.encode("utf-8"))
        )
        for r in records
    }
    pool, held = make_folds(list(by_id), seed, 1.0 - held_out_fraction, group_of)
    train, rest = make_folds(pool, seed, train_fraction, group_of)
    fit, selection = make_folds(rest, seed, fit_fraction / (1.0 - train_fraction), group_of)
    return (
        [by_id[i] for i in train],
        [by_id[i] for i in fit],
        [by_id[i] for i in selection],
        [by_id[i] for i in held],
    )


def _check_fractions(held_out: float, train: float, fit: float) -> None:
    ok = 0.0 <= held_out < 1.0 and 0.0 < train < 1.0 and 0.0 < fit and train + fit < 1.0
    if not ok:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"bad fractions held_out={held_out} train={train} fit={fit}: "
                "need 0<=held_out<1, train>0, fit>0 and train+fit<1 (selection is the remainder)"
            ),
            remediation="choose fractions that leave a non-empty selection side",
        )


def build_dataset(
    records: Iterable[Record],
    *,
    token_counter: TokenCounter,
    seed: int = DEFAULT_SEED,
    max_tokens: int | None = None,
    max_slots: int | None = None,
    profile: StudentProfile = DEFAULT_2610,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    fit_fraction: float = DEFAULT_FIT_FRACTION,
) -> Dataset:
    """Build the run dataset from *records*; see the module docstring.

    ``max_tokens`` and ``max_slots`` default to the *profile*'s limits; an explicit value wins.
    """
    max_tokens = profile.max_tokens if max_tokens is None else max_tokens
    max_slots = profile.num_slots if max_slots is None else max_slots
    _check_fractions(held_out_fraction, train_fraction, fit_fraction)
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
    licensed = [r for r in recs if admitted(r)]
    slotted = [r for r in licensed if option_count(r) <= max_slots]
    counts = _count_tokens(slotted, token_counter)
    kept = [r for r, n in zip(slotted, counts) if n <= max_tokens]
    dropped = Counter(r.source for r, n in zip(slotted, counts) if n > max_tokens)
    refused = {
        "licence": sum(1 for r in recs if not licence_allowed(r.licence)),
        "missing_attribution": sum(
            1 for r in recs if licence_allowed(r.licence) and not attribution_ok(r)
        ),
        "over_slots": len(licensed) - len(slotted),
        "over_length": sum(dropped.values()),
    }

    unused = Counter(r.split for r in kept if r.split not in (TRAIN_SPLIT, TEST_SPLIT))
    test_recs = [r for r in kept if r.split == TEST_SPLIT]
    train: list[Record] = []
    fit: list[Record] = []
    selection: list[Record] = []
    held: list[Record] = []
    for source in sorted({r.source for r in kept}):
        pool = [r for r in kept if r.source == source and r.split == TRAIN_SPLIT]
        if pool:
            t, f, s, h = _carve(pool, seed, held_out_fraction, train_fraction, fit_fraction)
            train += t
            fit += f
            selection += s
            held += h
    train.sort(key=lambda r: r.id)
    fit.sort(key=lambda r: r.id)
    selection.sort(key=lambda r: r.id)
    held.sort(key=lambda r: r.id)
    manifest: dict[str, Any] = {
        "stage": STAGE,
        "status": "built",
        "knobs": {
            "seed": seed,
            "max_tokens": max_tokens,
            "max_slots": max_slots,
            "held_out_fraction": held_out_fraction,
            "train_fraction": train_fraction,
            "fit_fraction": fit_fraction,
            "licence_allowlist": sorted(ALLOWED_LICENCES),
        },
        "max_tokens": max_tokens,
        "max_slots": max_slots,
        "licences": licences,
        "licence_table": _per_licence_table(kept),
        "refused": refused,
        "dropped_too_long": dict(sorted(dropped.items())),
        "unused_splits": dict(sorted(unused.items())),
        "counts": {
            "input": len(recs),
            "licence_excluded": len(recs) - len(licensed),
            "train": len(train),
            "fit": len(fit),
            "selection": len(selection),
            "held_out": len(held),
            "test": len(test_recs),
        },
    }
    return Dataset(
        train=tuple(train),
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
    """Write the five sides and ``manifest.json``; returns side name -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sides = {
        "train": ds.train,
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


def train_ids(source: "Dataset | Path | str") -> frozenset[str]:
    """Ids of the train side: from a built :class:`Dataset`, or a dataset directory.

    A directory is read from ``train.jsonl`` only, never from any other side.
    """
    if isinstance(source, Dataset):
        return frozenset(r.id for r in source.train)
    path = Path(source) / "train.jsonl"
    if not path.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path}: no train side found",
            remediation="point at a directory written by write_dataset",
        )
    return frozenset(r.id for r in read_records(path))
