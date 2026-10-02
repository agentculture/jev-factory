"""Assemble the scorer training set from a train split, then freeze it by sha256.

The assemble stage of the factory (nvsh ``build_dataset.py``, Track B half).
It takes the train side ``split`` wrote, folds in accepted augment variations
and a train-only supplement (:mod:`jev_factory.core.merge_variations`), runs the
leakage check against every protected side (fail closed), and writes
``scorer-train.json``: one entry per source row, each carrying its own
seeded, replayable candidate **subset, order and letter map** (``permutation``),
its ``gold`` candidate, its ``perm_seed`` and, where an escalate reason is
offered, its ``descriptions``. A configurable fraction of operation rows also
gets a derived ``<id>-nocand`` row: the same text with the gold operation
removed from the offer and the gold forced to escalate.

Then it **freezes** the data: ``freeze.json`` records the sha256 of every data
file that went in and of ``scorer-train.json``. Training selects its data by
that sha256 (:func:`select_frozen`), never by modification time, and any
change after the freeze needs a deviation id (:func:`verify_freeze`,
:func:`assemble` over an existing freeze).

Nothing here is nvsh-specific: the candidate pool, reasons, gold mapping and
prompt text come from the :class:`~jev_factory.domain.model.Domain`. The
chat/tool-call (Track A) examples of the nvsh script are not carried over:
Track A is not jev-like. Rendering a row with a real tokenizer's chat template
is an injectable check (:func:`verify_render`); transformers is only imported
by :func:`load_tokenizer`, lazily.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from jev_factory.backbones.causal_lm import scorer
from jev_factory.cli._errors import CliError
from jev_factory.core import leakage, merge_variations
from jev_factory.core.split import HELD_OUT_NAME, expectation_kind
from jev_factory.domain.model import ESCALATE, EXPLAIN, Domain
from jev_factory.factory.stages import sha256_path

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/build_dataset.py",
    "commit": "9debdc6",
    "adaptations": [
        "nvsh imports :57-66 (nvsh.platform, nvsh.tiers lfm/bench) and the _sibling path-loader"
        " :69-80 are replaced by package imports: scorer is jev_factory.backbones.causal_lm.scorer,"
        " the pool/reasons/gold mapping come from the Domain (REASON_CANDIDATES :87 and"
        " reason_for_entry :396 become Domain.candidates(with_reasons)/reason_for_class)",
        "seeding and shaping kept: example_seed :410, subset_size :425, build_permutation :466,"
        " enrich_example :489, _select_missing_candidate :519, scorer_entry :535,"
        " scorer_file :558, eval_side_markers :581, build_with_scorer_entries :626-774"
        " (held-out/side refusals, -nocand rows)",
        "defaults per issue #4: randomize labels on, perm seed, missing-candidate rate 0.3",
        "not carried over: the Track A chat/tool-call examples (answer_for, example_from_entry,"
        " --arguments-as, parse_rendered_call :98-176), the argparse main :777-961 and the"
        " Platform/bench request builders; the request is the entry's own text",
        "verify_build :262 (round trip through a base's chat template) becomes verify_render:"
        " the tokenizer is injectable and transformers is imported lazily in load_tokenizer",
        "new: the train-side merge of variations + supplement, the fail-closed leakage run,"
        " and the sha256 freeze (freeze.json) that training selects by, never by mtime",
    ],
    "licence": "Apache-2.0",
}

SCORER_TRAIN_NAME = "scorer-train.json"
FREEZE_NAME = "freeze.json"
FREEZE_SCHEMA = 1

#: How many candidates a randomised offer keeps at minimum, and the chance it keeps the full pool.
DEFAULT_MIN_SUBSET = 6
DEFAULT_FULL_SET_PROBABILITY = 0.3
#: Issue #4 default: fraction of operation rows that also get a ``-nocand`` row.
DEFAULT_MISSING_CANDIDATE_RATE = 0.3

_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")
_HELD_OUT_HEADER_MARKER = "Held-out split"
TRAIN_SIDE = "train"

#: What ``enrich`` adds to a row and the trainer reads back by entry id.
SCORER_CONTRACT_KEYS = ("permutation", "gold", "perm_seed", "descriptions")


@dataclass(frozen=True)
class AssembleConfig:
    """The assemble knobs; recorded in the freeze."""

    reasons: bool = False
    randomize_labels: bool = True
    perm_seed: int = 0
    missing_candidate_rate: float = DEFAULT_MISSING_CANDIDATE_RATE
    min_subset: int = DEFAULT_MIN_SUBSET
    full_set_probability: float = DEFAULT_FULL_SET_PROBABILITY


def _err(message: str, remediation: str = "", code: int = 1) -> CliError:
    return CliError(code, message, remediation)


# ---------------------------------------------------------------------------
# Seeding and permutations (deterministic, replayable)
# ---------------------------------------------------------------------------


def example_seed(perm_seed: int, example_id: str) -> int:
    """A per-example seed that is the same on every process and machine."""
    digest = hashlib.sha256(f"{perm_seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def subset_size(seed: int, pool_size: int, min_subset: int, full_probability: float) -> int:
    """A deterministic subset size in ``[min_subset, pool_size]``, drawn from *seed*."""
    rng = random.Random(seed)  # nosec B311 - dataset shaping, not security
    if pool_size <= min_subset or rng.random() < full_probability:
        return pool_size
    return rng.randint(min_subset, pool_size - 1)


def build_permutation(
    example_id: str,
    gold: str,
    offered: Sequence[str],
    full: Sequence[str],
    cfg: AssembleConfig,
) -> tuple[scorer.Permutation, int]:
    """This row's ``(Permutation, seed)``.

    Without ``randomize_labels`` the fixed order with positional letters; with
    it a seeded random subset (always keeping *gold*), order and letter map.
    """
    seed = example_seed(cfg.perm_seed, example_id)
    if not cfg.randomize_labels:
        labels = scorer.positional_labels(offered, full)
        return scorer.Permutation(order=tuple(offered), labels=labels), seed
    size = subset_size(seed, len(offered), cfg.min_subset, cfg.full_set_probability)
    return scorer.permute(seed, pool=list(offered), subset=size, keep=gold), seed


def gold_for(domain: Domain, entry: dict, reasons: bool) -> str:
    """The candidate *entry*'s expected answer names (what the scorer is trained to pick)."""
    expect = entry["expect"]
    kind = expectation_kind(expect)
    if kind == "explain":
        return EXPLAIN
    if kind == "escalate":
        return domain.reason_for_class(entry.get("class")) if reasons else ESCALATE
    return expect["operation"]


def contract_for(
    domain: Domain,
    example_id: str,
    gold: str,
    offered: Sequence[str],
    full: Sequence[str],
    cfg: AssembleConfig,
) -> dict[str, Any]:
    """The per-row scorer-training contract: permutation, gold, perm_seed, descriptions?."""
    permutation, seed = build_permutation(example_id, gold, offered, full, cfg)
    contract: dict[str, Any] = {
        "permutation": permutation.to_json(),
        "gold": gold,
        "perm_seed": seed,
    }
    reason_text = domain.reason_descriptions()
    descriptions = {name: reason_text[name] for name in permutation.order if name in reason_text}
    if descriptions:
        contract["descriptions"] = descriptions
    return contract


def select_missing_candidate(perm_seed: int, entry_id: str, rate: float) -> bool:
    """Deterministic, seeded: does *entry_id* also get a ``-nocand`` row."""
    if rate <= 0:
        return False
    seed = example_seed(perm_seed, f"{entry_id}:missing-candidate")
    return random.Random(seed).random() < rate  # nosec B311 - dataset shaping


def eval_side_markers(path: Path) -> set[str]:
    """Which of ``{"val", "test", "held-out"}`` *path*'s own file name claims."""
    stem_words = {word for word in re.split(r"[^a-z0-9]+", path.stem.lower()) if word}
    compact = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    markers: set[str] = set()
    if "test" in stem_words:
        markers.add("test")
    if "heldout" in compact:
        markers.add("held-out")
    if "val" in stem_words:
        markers.add("val")
    return markers


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def scorer_entry(raw: dict, contract: dict) -> dict:
    """*raw* (a corpus-format entry) plus *contract*, dropping any stale contract keys."""
    entry = {key: value for key, value in raw.items() if key not in SCORER_CONTRACT_KEYS}
    entry.update(contract)
    return entry


def build_rows(domain: Domain, entries: Sequence[dict], cfg: AssembleConfig) -> list[dict]:
    """Every scorer row: each entry with its contract, then its ``-nocand`` row when drawn."""
    pool = scorer.candidate_pool(domain, cfg.reasons)
    fallback = domain.default_reason_label() if cfg.reasons else ESCALATE
    taken = {str(entry["id"]) for entry in entries}
    rows: list[dict] = []
    for entry in entries:
        gold = gold_for(domain, entry, cfg.reasons)
        rows.append(scorer_entry(entry, contract_for(domain, entry["id"], gold, pool, pool, cfg)))
        operation = entry["expect"].get("operation")
        if operation is None or not select_missing_candidate(
            cfg.perm_seed, entry["id"], cfg.missing_candidate_rate
        ):
            continue
        derived_id = f"{entry['id']}-nocand"
        if derived_id in taken:
            raise _err(f"derived missing-candidate id {derived_id!r} is already an entry")
        derived_pool = tuple(name for name in pool if name != operation)
        contract = contract_for(domain, derived_id, fallback, derived_pool, pool, cfg)
        derived = {**entry, "id": derived_id, "expect": {"escalate": True}}
        derived["source_id"] = entry.get("source_id", entry["id"])
        if cfg.reasons:
            derived["class"] = "decline:" + fallback.split(":", 1)[1]
        else:
            derived.pop("class", None)
        rows.append(scorer_entry(derived, contract))
    return rows


# ---------------------------------------------------------------------------
# Rendering check (tokenizer injectable)
# ---------------------------------------------------------------------------


def load_tokenizer(base: str, revision: str | None = None):
    """*base*'s tokenizer from the local cache only (lazy transformers import)."""
    from transformers import AutoTokenizer  # heavy: only when actually loading

    return AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=True)


def render_row(domain: Domain, row: dict, tokenizer, reasons: bool = False) -> str:
    """The prompt text *tokenizer*'s chat template gives *row*'s stored offer."""
    permutation = scorer.Permutation.from_json(row["permutation"])
    messages = scorer.prompt_messages(
        domain,
        row["text"],
        labels=permutation.labels,
        order=permutation.order,
        descriptions=row.get("descriptions"),
        reasons=reasons,
    )
    return scorer.render_prompt(tokenizer, messages)


def verify_render(domain: Domain, rows: Sequence[dict], tokenizer, reasons: bool = False) -> None:
    """One row per outcome kind must render with every offered ``L) name:`` line and its request.

    A base whose chat template drops or rewrites the candidate lines cannot be
    trusted with this data, so it raises instead of shipping silently.
    """
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(expectation_kind(row["expect"]), row)
    for row in seen.values():
        rendered = render_row(domain, row, tokenizer, reasons)
        permutation = scorer.Permutation.from_json(row["permutation"])
        missing = [
            f"{permutation.labels[name]}) {name}:"
            for name in permutation.order
            if f"{permutation.labels[name]}) {name}:" not in rendered
        ]
        if missing or row["text"] not in rendered:
            raise _err(
                f"row {row.get('id')}: the chat template does not render the offer faithfully"
                f" (missing {missing or ['request text']})",
                "check the base tokenizer's chat template, or pass a different --base",
            )


# ---------------------------------------------------------------------------
# Inputs: train side only, merged, leak-checked
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CliError(
            2, f"{path}: not a readable JSON file ({type(exc).__name__})", "check the path"
        ) from None


def _read_jsonl(path: Path) -> list[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise CliError(
            2, f"{path}: not a readable JSONL file ({type(exc).__name__})", "check the path"
        ) from None


def check_train_source(path: Path, doc: Any, *, is_split: bool, cfg: AssembleConfig) -> None:
    """Refuse the held-out split, any non-train side, and (for -nocand) an eval-named corpus."""
    header = doc.get("header", "") if isinstance(doc, dict) else ""
    header = header if isinstance(header, str) else ""
    if path.name == HELD_OUT_NAME or _HELD_OUT_HEADER_MARKER in header:
        raise _err(
            f"{path}: the held-out split is for judging a tuned model, never for training it"
        )
    if is_split:
        match = _SPLIT_SIDE_RE.search(header)
        if match is None:
            raise _err(f"{path}: header names no side; only a train-side split may train")
        if match.group(1) != TRAIN_SIDE:
            raise _err(
                f"{path}: this is the {match.group(1)!r} side; only {TRAIN_SIDE!r} may train,"
                " val/test are held out for evaluation"
            )
    elif cfg.missing_candidate_rate > 0:
        markers = eval_side_markers(path)
        if markers:
            raise _err(
                f"{path}: looks like the {' and '.join(sorted(markers))} side; "
                "missing-candidate rows come only from the train side"
            )


def _protected_entries(path: Path) -> list[dict]:
    try:
        return leakage._load(Path(path))[1]
    except ValueError as exc:
        raise CliError(2, str(exc), "a protected side must be readable; fail closed") from None


def check_leakage(entries: Sequence[dict], protected: Sequence[Path]) -> None:
    """Fail closed when any training text matches a protected side (exact or near-duplicate)."""
    if not protected:
        return
    sides = {Path(p).name: _protected_entries(Path(p)) for p in protected}
    try:
        found = leakage.find_matches(list(entries), sides)
    except (KeyError, TypeError) as exc:
        raise CliError(2, f"leakage check failed on a malformed entry ({exc!r})", "") from None
    if found:
        kinds = sorted({m["kind"] for m in found})
        raise _err(
            f"{len(found)} training row(s) leak into a protected side ({', '.join(kinds)}):"
            f" {', '.join(m['train_id'] for m in found[:10])}",
            "remove or reword them; leakage never passes",
        )


def merged_entries(
    split_doc: dict,
    variations: Sequence[dict],
    supplement: dict | None,
    exclude_docs: Sequence[dict],
) -> tuple[dict, dict[str, Any]]:
    """The split with variations and supplement folded in, plus the counts."""
    exclude = merge_variations.excluded_texts(list(exclude_docs))
    try:
        merged, counts = merge_variations.merge(split_doc, list(variations), exclude)
        counts = dict(counts)
        if supplement is not None:
            merged, added = merge_variations.add_supplement(merged, supplement, exclude)
            counts["supplement"] = added
    except (ValueError, KeyError) as exc:
        raise _err(f"merge refused: {exc}", "fix the variations or supplement") from None
    return merged, counts


# ---------------------------------------------------------------------------
# Freeze: sha256 of every data file; training selects by it
# ---------------------------------------------------------------------------


def _file_record(path: Path) -> dict[str, str]:
    digest = sha256_path(path)
    if digest is None:
        raise CliError(2, f"{path}: missing, cannot be frozen", "check the path")
    return {"path": str(path), "sha256": digest}


def load_freeze(freeze_path: Path) -> dict[str, Any]:
    doc = _read_json(Path(freeze_path))
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), dict):
        raise _err(f"{freeze_path}: not a freeze record", "re-run the assemble stage")
    return doc


def _resolve(freeze_path: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else Path(freeze_path).parent / path


def verify_freeze(freeze_path: Path, deviation_id: str | None = None) -> list[str]:
    """Roles whose file changed since the freeze; raises unless a deviation id is given."""
    freeze = load_freeze(freeze_path)
    changed = [
        role
        for role, record in sorted(freeze["files"].items())
        if sha256_path(_resolve(freeze_path, record["path"])) != record["sha256"]
    ]
    if changed and not (deviation_id or "").strip():
        raise _err(
            f"data changed since the freeze (a deviation id is required): {', '.join(changed)}",
            "record a deviation and pass its id (--deviation <id>); never edit frozen data",
        )
    return changed


@dataclass(frozen=True)
class FrozenChoice:
    """The data file training uses: chosen by sha256, with the deviation it needed, if any."""

    path: Path
    sha256: str
    deviation_id: str | None = None


def select_frozen(
    freeze_path: Path,
    role: str = "scorer_train",
    search: Sequence[Path] = (),
    deviation_id: str | None = None,
) -> FrozenChoice:
    """The file whose sha256 is the frozen one for *role*; modification time is never consulted.

    The recorded path is tried first, then every ``*.json`` in the freeze's own
    directory and in *search*. If no file carries the frozen sha256, a file at
    the recorded path that changed is refused unless *deviation_id* is given
    (and then returned under that deviation).
    """
    freeze_path = Path(freeze_path)
    record = load_freeze(freeze_path)["files"].get(role)
    if record is None:
        raise _err(f"the freeze has no {role!r} file", "re-run the assemble stage")
    wanted = record["sha256"]
    recorded = _resolve(freeze_path, record["path"])
    candidates = [recorded]
    for directory in (freeze_path.parent, *map(Path, search)):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.json")))
    for candidate in candidates:
        if candidate.is_file() and sha256_path(candidate) == wanted:
            return FrozenChoice(candidate, wanted)
    if recorded.is_file() and (deviation_id or "").strip():
        return FrozenChoice(recorded, sha256_path(recorded) or "", deviation_id.strip())
    raise _err(
        f"no file with the frozen sha256 {wanted[:12]}... for {role!r}"
        + (" (changed after freeze: a deviation id is required)" if recorded.is_file() else ""),
        "restore the frozen file, or record a deviation and pass its id (--deviation <id>)",
    )


def write_freeze(
    out_dir: Path,
    files: dict[str, Path],
    cfg: AssembleConfig,
    counts: dict[str, Any],
    deviation_id: str | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "files": {role: _file_record(path) for role, path in sorted(files.items())},
        "config": asdict(cfg),
        "counts": counts,
    }
    if deviation_id:
        record["deviation_id"] = deviation_id
        if previous is not None:
            record["supersedes"] = {
                role: rec["sha256"] for role, rec in sorted(previous["files"].items())
            }
    (Path(out_dir) / FREEZE_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def assemble(
    out_dir: Path,
    *,
    domain: Domain,
    split: Path,
    variations: Sequence[Path] = (),
    supplement: Path | None = None,
    exclude: Sequence[Path] = (),
    protected: Sequence[Path] = (),
    cfg: AssembleConfig | None = None,
    tokenizer=None,
    deviation_id: str | None = None,
) -> dict[str, Any]:
    """Merge, leak-check, build ``scorer-train.json`` and freeze every data file's sha256.

    *exclude* are the sides whose wording training must not repeat (val/test);
    *protected* are every side the leakage check covers (val, test, held-out).
    Re-running over an existing freeze needs *deviation_id*. A *tokenizer*
    (injectable; see :func:`load_tokenizer`) turns on the render check.
    Returns the freeze record.
    """
    cfg = cfg or AssembleConfig()
    out_dir, split = Path(out_dir), Path(split)
    freeze_path = out_dir / FREEZE_NAME
    previous = load_freeze(freeze_path) if freeze_path.exists() else None
    if previous is not None and not (deviation_id or "").strip():
        raise _err(
            f"{freeze_path} already exists: the data is frozen",
            "a rerun after freeze is a recorded deviation; pass its id (--deviation <id>)",
        )
    if not 0.0 <= cfg.missing_candidate_rate <= 1.0:
        raise _err("missing_candidate_rate must be between 0 and 1")

    split_doc = _read_json(split)
    check_train_source(split, split_doc, is_split=True, cfg=cfg)
    variation_rows = [row for path in variations for row in _read_jsonl(Path(path))]
    supplement_doc = _read_json(Path(supplement)) if supplement else None
    exclude_docs = [_read_json(Path(p)) for p in exclude]
    merged, counts = merged_entries(split_doc, variation_rows, supplement_doc, exclude_docs)
    check_leakage(merged["entries"], protected)

    rows = build_rows(domain, merged["entries"], cfg)
    if tokenizer is not None:
        verify_render(domain, rows, tokenizer, cfg.reasons)
    derived = sum(1 for row in rows if str(row["id"]).endswith("-nocand"))
    counts.update(
        {"entries": len(rows), "original": len(rows) - derived, "missing_candidate": derived}
    )
    document: dict[str, Any] = {
        "header": f"{merged['header']} Scorer training rows written by assemble; see provenance.",
        "provenance": {"config": asdict(cfg), "counts": counts},
        "entries": rows,
    }
    if "world" in merged:
        document["world"] = merged["world"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / SCORER_TRAIN_NAME
    train_path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", "utf-8")

    files = {"split": split, "scorer_train": train_path}
    files.update({f"variations_{i}": Path(p) for i, p in enumerate(variations)})
    if supplement:
        files["supplement"] = Path(supplement)
    files.update({f"exclude_{i}": Path(p) for i, p in enumerate(exclude)})
    files.update({f"protected_{i}": Path(p) for i, p in enumerate(protected)})
    return write_freeze(out_dir, files, cfg, counts, deviation_id, previous)
