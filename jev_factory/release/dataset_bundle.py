"""Build the dataset folder of a bundle: the records a model was trained and measured on.

The folder holds ``data/train.jsonl``, ``data/validation.jsonl`` and
``data/test.jsonl`` (the train side as trained on, and the split's other two
sides), ``manifest.json`` (one row per record: split, origin, source, licence,
and for a variation the generator / corrector / reviewer models), ``LICENSE``,
``scorer-train.json`` (the candidate set the scorer actually trained on, with
private addresses redacted) and ``README.md``, the card, whose counts are all
computed here. ``held-out.json`` is refused by name, and so is a train record
that repeats a validation or test entry. This module never uploads; the
checks a whole bundle must pass (required files, symlinks, surface hash) live in
:mod:`jev_factory.release.bundle`, which calls :func:`build`.

Which model answered each of the augmentation roles is a fact about one run:
``role_models`` is the run's alias -> (name, licence) table, and
``apache_only`` refuses a bundle naming any teacher that is not Apache-2.0.
"""

from __future__ import annotations

import collections
import ipaddress
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/dataset_bundle.py",
    "commit": "9debdc6",
    "adaptations": [
        "teacher_summary/teacher_rows/decision_sentence/load_role_models/publishable_text kept",
        "CORPUS_FILE/SUPPLEMENT_FILE and the nvsh-specific card prose replaced by the"
        " Domain's name, card_text and the run config's licence and issue refs",
        "scorer-train.json ships at the bundle root (not data/) so every bundle kind has the"
        " same required files; scorer_train is now required",
        "the argparse main() and the per-issue RUN_LOGS/GROUNDING tables are dropped",
    ],
    "licence": "Apache-2.0",
}

HELD_OUT_NAME = "held-out.json"
APACHE_LICENCE = "Apache-2.0"
SCORER_TRAIN_FILE = "scorer-train.json"

#: The four functions the augmentation roles fill.
ROLES = (
    ("GENERATOR", "wrote the variation"),
    ("CORRECTOR", "copyedited it"),
    ("REVIEWER_A", "accepted it (reviewer A)"),
    ("REVIEWER_B", "accepted it (reviewer B)"),
)

DECIDED_BY_BOTH = "both"
DECIDED_BY_REVIEWER_B = "reviewer_b"

_SEED_RE = re.compile(r"\(seed=(\d+)\)")

DRAFT_SOURCE_PREFIX = "draft-"
TARGETED_SOURCE_PREFIX = "t15-"


class TeacherSummary:
    """Which models played each role over a run's accepted variations, and how
    their acceptance was decided."""

    def __init__(self) -> None:
        self.role_teachers: dict[str, dict[str, str]] = {}
        self.shared_corrector_reviewer_names: list[str] = []
        self.decisions: dict[str, int] = {}
        self.per_variation: dict[str, dict[str, str]] = {}


def decision_rule(row: dict[str, Any]) -> str:
    """``reviewer_b`` when reviewer B alone decided *row*'s acceptance, else ``both``."""
    if row.get("decided_by") == DECIDED_BY_REVIEWER_B:
        return DECIDED_BY_REVIEWER_B
    verdicts = row.get("verdicts")
    if isinstance(verdicts, dict) and "reviewer_b" in verdicts and "reviewer_a" not in verdicts:
        return DECIDED_BY_REVIEWER_B
    return DECIDED_BY_BOTH


def _teacher(role_models: dict[str, tuple[str, str]], alias: str) -> tuple[str, str]:
    if alias not in role_models:
        raise ValueError(f"{alias!r} is not in the run's teacher-models table")
    return role_models[alias]


#: How a jev-factory teacher record (``data.augment`` / ``data.targeted``: role name ->
#: ``TeacherRole.record()``) maps onto the card's roles. Reviewer B also copyedits.
TEACHER_RECORD_ROLES = {
    "generator": ("GENERATOR",),
    "reviewer_b": ("CORRECTOR", "REVIEWER_B"),
    "reviewer_a": ("REVIEWER_A",),
}


def _resolve_roles(
    entry_id: str, row: dict[str, Any], role_models: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """``{ROLE: (name, licence)}`` for one accepted variation record.

    Two record shapes are accepted: nvsh's ``models`` (role -> alias in *role_models*,
    every role required) and jev-factory's ``teachers`` (role name -> a record carrying
    ``model_name``/``model`` and ``licence``; reviewer A only when it was asked)."""
    models = row.get("models")
    if models:
        resolved = {}
        for role, _ in ROLES:
            if role not in models:
                raise ValueError(f"variation {entry_id}'s accepted record names no {role} teacher")
            resolved[role] = _teacher(role_models, models[role])
        return resolved
    teachers = row.get("teachers")
    if not isinstance(teachers, dict) or not teachers:
        raise ValueError(f"variation {entry_id} has no accepted record naming its models")
    resolved = {}
    for name, roles in TEACHER_RECORD_ROLES.items():
        info = teachers.get(name)
        if info is None:
            if name == "reviewer_a":
                continue
            raise ValueError(f"variation {entry_id}'s accepted record names no {name} teacher")
        if not isinstance(info, dict):
            raise ValueError(f"variation {entry_id}: teacher {name!r} is not a record")
        model = info.get("model_name") or info.get("model")
        licence = info.get("licence")
        if not model or not licence:
            raise ValueError(f"variation {entry_id}: teacher {name!r} needs a model and a licence")
        for role in roles:
            resolved[role] = (str(model), str(licence))
    return resolved


def teacher_summary(
    train: list[dict[str, Any]],
    accepted_rows: dict[str, dict[str, Any]],
    role_models: dict[str, tuple[str, str]],
    *,
    apache_only: bool = False,
    is_variation: Callable[[str], bool] | None = None,
) -> TeacherSummary:
    """The teachers of every variation in *train* (by default an id with ``~v``)."""
    summary = TeacherSummary()
    role_teachers: dict[str, dict[str, str]] = collections.defaultdict(dict)
    shared: dict[str, None] = {}
    decisions: collections.Counter[str] = collections.Counter()
    variation = is_variation or (lambda entry_id: "~v" in entry_id)
    for entry in train:
        if not variation(entry["id"]):
            continue
        row = accepted_rows.get(entry["id"], {})
        teachers: dict[str, str] = {}
        for role, (name, teacher_licence) in _resolve_roles(entry["id"], row, role_models).items():
            if apache_only and teacher_licence != APACHE_LICENCE:
                raise ValueError(
                    f"{entry['id']}: teacher {name!r} ({teacher_licence}) is not "
                    f"{APACHE_LICENCE}; refused by apache_only"
                )
            teachers[role] = name
            role_teachers[role][name] = teacher_licence
        if teachers.get("CORRECTOR") == teachers.get("REVIEWER_B"):
            shared.setdefault(teachers["CORRECTOR"], None)
        decisions[decision_rule(row)] += 1
        summary.per_variation[entry["id"]] = teachers
    summary.role_teachers = {
        role: dict(role_teachers[role]) for role, _ in ROLES if role in role_teachers
    }
    summary.shared_corrector_reviewer_names = list(shared)
    summary.decisions = dict(decisions)
    return summary


def role_description(role: str, decisions: dict[str, int]) -> str:
    """What *role* did in a run whose acceptances were decided as *decisions* says."""
    base = dict(ROLES)[role]
    by_b = decisions.get(DECIDED_BY_REVIEWER_B, 0)
    by_both = decisions.get(DECIDED_BY_BOTH, 0)
    if not by_b or role not in ("REVIEWER_A", "REVIEWER_B"):
        return base
    if role == "REVIEWER_A":
        if not by_both:
            return "reviewer A, advisory: asked and recorded, did not decide"
        return f"accepted it (reviewer A; deciding for {by_both} variations, advisory for {by_b})"
    return "accepted it (reviewer B, deciding)" if not by_both else base


def teacher_rows(summary: TeacherSummary) -> list[tuple[str, str, str]]:
    """``(name, licence, role description)`` per role and model, in ROLES order."""
    return [
        (name, licence, role_description(role, summary.decisions))
        for role, _ in ROLES
        for name, licence in summary.role_teachers.get(role, {}).items()
    ]


def decision_sentence(decisions: dict[str, int]) -> str:
    """How a rewrite came to be kept, for a card's prose."""
    by_b = decisions.get(DECIDED_BY_REVIEWER_B, 0)
    by_both = decisions.get(DECIDED_BY_BOTH, 0)
    asked = (
        "Two reviewer models were each asked whether the rewrite still calls for"
        " exactly that answer"
    )
    if not by_b:
        return f"{asked}, and a rewrite was kept only when both said yes."
    advisory = (
        "reviewer B's verdict alone decided whether a rewrite was kept; reviewer"
        " A's verdict, where it was asked, was recorded as advisory and did not decide"
    )
    if not by_both:
        return f"{asked}: {advisory}."
    return f"{asked}. For {by_both} kept variations both said yes; for {by_b}, {advisory}."


def split_seed(path: Path) -> int | None:
    """The seed the split wrote into *path*'s header, if it names one."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = raw.get("header", "") if isinstance(raw, dict) else ""
    match = _SEED_RE.search(str(header))
    return int(match.group(1)) if match else None


def load_role_models(path: Path) -> dict[str, tuple[str, str]]:
    """The alias/model-id -> (display name, licence) table for one run."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a JSON object of alias -> {{name, licence}}")
    table: dict[str, tuple[str, str]] = {}
    for alias, info in raw.items():
        if not isinstance(info, dict) or not info.get("name") or not info.get("licence"):
            raise ValueError(f"{path}: {alias!r} needs a non-empty 'name' and 'licence'")
        table[alias] = (str(info["name"]), str(info["licence"]))
    return table


def load_entries(path: Path) -> list[dict[str, Any]]:
    """The entries of a split / train file; ``held-out.json`` is refused by name."""
    if path.name == HELD_OUT_NAME:
        raise ValueError(f"{path}: the held-out split is never published with the training data")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["entries"] if isinstance(raw, dict) else raw)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _normal(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
DOCUMENTATION_NET = "192.0.2."
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def publishable_text(text: str) -> str:
    """*text* with each private (non-loopback) IPv4 address replaced by the RFC 5737
    documentation address with the same last octet (the bundle scan refuses private hosts)."""

    def swap(match: re.Match[str]) -> str:
        try:
            address = ipaddress.ip_address(match.group(1))
        except ValueError:
            return match.group(0)
        if address.is_loopback or address.is_unspecified:
            return match.group(0)
        if not (address.is_private or address in _CGNAT):
            return match.group(0)
        return DOCUMENTATION_NET + match.group(1).rsplit(".", 1)[1]

    return _IPV4.sub(swap, text)


def _record(entry: dict[str, Any], split: str) -> dict[str, Any]:
    keep = ("id", "text", "expect", "kind", "source_id", "class")
    record = {key: entry[key] for key in keep if key in entry}
    if "text" in record:
        record["text"] = publishable_text(record["text"])
    record.setdefault("source_id", entry["id"])
    record["split"] = split
    return record


def _origin(entry: dict[str, Any]) -> str:
    source = str(entry.get("source", ""))
    if "~v" in entry["id"]:
        return "variation"
    if source.startswith(DRAFT_SOURCE_PREFIX):
        return "draft"
    if source.startswith(("supplement", TARGETED_SOURCE_PREFIX)):
        return "supplement"
    return "corpus"


def _with_source(entry: dict[str, Any], default_source: str | None) -> dict[str, Any]:
    if entry.get("source"):
        return entry
    if not default_source:
        raise ValueError(f"{entry['id']}: every published record needs a 'source' field")
    return {**entry, "source": default_source}


def _answer_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    kinds: collections.Counter[str] = collections.Counter()
    for record in records:
        expect = record["expect"]
        kinds[
            (
                "explain"
                if expect.get("explain")
                else "escalate" if expect.get("escalate") else "propose"
            )
        ] += 1
    return dict(kinds)


def build(
    *,
    domain: Domain,
    splits: Path,
    train_augmented: Path,
    accepted: Path,
    rejected: Path | list[Path],
    licence: Path,
    role_models: dict[str, tuple[str, str]],
    scorer_train: Path,
    out: Path,
    apache_only: bool = False,
    issue_refs: str = "",
    model_repos: list[str] | None = None,
    default_source: str | None = None,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write the dataset folder to *out*; return the counts shown in the card.

    *scorer_train* is the candidate scorer's own training file (per-row label maps
    and missing-candidate rows); it ships as ``scorer-train.json`` with the same
    address redaction as every published record. *source_files* maps an origin
    (``corpus``, ``supplement``, ``draft``) to the file recorded as its source;
    by default the domain's seed corpus name is used for ``corpus``.
    """
    files = {
        "corpus": domain.seed_corpus.name if domain.seed_corpus else "seed corpus",
        "supplement": "train-supplement",
        "draft": "drafted requests",
        **(source_files or {}),
    }
    train = [_with_source(e, default_source) for e in load_entries(train_augmented)]
    sides = {
        side: [_with_source(e, default_source) for e in load_entries(splits / name)]
        for side, name in (("validation", "val.json"), ("test", "test.json"))
    }
    accepted_rows = {row["id"]: row for row in load_jsonl(accepted)}
    rejected_files = [rejected] if isinstance(rejected, Path) else list(rejected)
    rejected_count = sum(len(load_jsonl(path)) for path in rejected_files)
    if not licence.read_text(encoding="utf-8").strip():
        raise ValueError(f"{licence} is empty")

    held_apart = {_normal(e["text"]) for entries in sides.values() for e in entries}
    leaked = sum(1 for entry in train if _normal(entry["text"]) in held_apart)
    if leaked:
        raise ValueError(
            f"{leaked} train record(s) repeat a validation or test entry; re-run"
            " the merge with the held-apart sides excluded before building the dataset"
        )

    manifest: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    origins: collections.Counter[str] = collections.Counter()
    summary = teacher_summary(train, accepted_rows, role_models, apache_only=apache_only)
    for entry in train:
        if entry.get("side", "train") != "train":
            raise ValueError(f"{entry['id']} in {train_augmented} is not a train-side entry")
        origin = _origin(entry)
        manifest.append(
            {
                "id": entry["id"],
                "split": "train",
                "origin": origin,
                "source": entry["source"],
                "source_id": entry.get("source_id", entry["id"]),
                "source_file": files.get(origin, files["corpus"]),
                "licence": APACHE_LICENCE,
                "transformed": origin == "variation",
                "teachers": summary.per_variation.get(entry["id"], {}),
            }
        )
        origins[origin] += 1
        rows["train"].append(_record(entry, "train"))
    for split, entries in sides.items():
        for entry in entries:
            if "~v" in entry["id"]:
                raise ValueError(f"{entry['id']}: variations never leave the train side")
            origin = _origin(entry)
            manifest.append(
                {
                    "id": entry["id"],
                    "split": split,
                    "origin": origin,
                    "source": entry["source"],
                    "source_id": entry.get("source_id", entry["id"]),
                    "source_file": files.get(origin, files["corpus"]),
                    "licence": APACHE_LICENCE,
                    "transformed": False,
                    "teachers": {},
                }
            )
            rows[split].append(_record(entry, split))

    ids = [row["id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record ids across the splits")

    if out.exists():
        if any(out.iterdir()):
            raise ValueError(f"{out} is not empty")
        out.rmdir()
    (out / "data").mkdir(parents=True)
    for split, records in rows.items():
        with open(out / "data" / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(licence, out / "LICENSE")
    scorer_doc = json.loads(scorer_train.read_text(encoding="utf-8"))
    for entry in scorer_doc.get("entries", []):
        if isinstance(entry.get("text"), str):
            entry["text"] = publishable_text(entry["text"])
    (out / SCORER_TRAIN_FILE).write_text(
        json.dumps(scorer_doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_entries = [*train, *(e for entries in sides.values() for e in entries)]
    counts = {
        "train": len(rows["train"]),
        "validation": len(rows["validation"]),
        "test": len(rows["test"]),
        "corpus": origins["corpus"],
        "supplement": origins["supplement"],
        "variation": origins["variation"],
        "draft": origins["draft"],
        "redacted_hosts": sum(
            1 for e in all_entries if publishable_text(e.get("text", "")) != e.get("text", "")
        ),
        "accepted": len(accepted_rows),
        "rejected": rejected_count,
        "answers": _answer_counts(rows["train"]),
    }
    (out / "README.md").write_text(
        card(
            counts,
            summary,
            domain=domain,
            licence=APACHE_LICENCE if apache_only else "see LICENSE",
            issue_refs=issue_refs,
            model_repos=model_repos,
            seed=split_seed(splits / "val.json"),
        ),
        encoding="utf-8",
    )
    return counts


def card(
    counts: dict[str, Any],
    summary: TeacherSummary,
    *,
    domain: Domain,
    licence: str,
    issue_refs: str = "",
    model_repos: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """The dataset card: counts computed here, prose from the Domain's ``card_text``."""
    answers = counts["answers"]
    reviewed = counts["accepted"] + counts["rejected"]
    rate = f"{100 * counts['accepted'] / reviewed:.0f}%" if reviewed else "n/a"
    seeded = f"seed {seed}" if seed is not None else "seeded"
    if summary.role_teachers:
        teachers = "\n".join(
            f"| {name} | {lic} | {desc} |" for name, lic, desc in teacher_rows(summary)
        )
    else:
        teachers = "| (none) | (none) | this run produced no synthetic variations |"
    disclosure = ""
    if summary.shared_corrector_reviewer_names:
        names = ", ".join(summary.shared_corrector_reviewer_names)
        disclosure = (
            f"\n**Reviewer B is also the corrector** in this run ({names}): its"
            " accept/reject verdict is not independent of the copyedit it made.\n"
        )
    train_parts = (
        f"{counts['corpus']} corpus entries, {counts['supplement']} supplement entries,"
        f" {counts['variation']} synthetic variations, {counts['draft']} drafted requests"
    )
    repos = ", ".join(f"`{r}`" for r in model_repos or []) or "the models of this run"
    refs = f" ({issue_refs})" if issue_refs else ""
    redacted = counts.get("redacted_hosts", 0)
    redaction = (
        f"\n{redacted} record(s) named a private network address; it is published as the\n"
        f"documentation address {DOCUMENTATION_NET}x (RFC 5737) with the same last octet.\n"
        if redacted
        else ""
    )
    return f"""---
license: {licence.lower()}
language:
- en
tags:
- jev
- {domain.name}
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# {domain.name}

{domain.card_text}

Each record is a request labelled with the answer a jev-like scorer should give:
**propose** one operation from the domain's table, **explain** in words, or
**escalate**. It is the data {repos} was trained and measured on{refs}.

## Splits

| Split | Records | Contents |
|---|---|---|
| train | {counts['train']} | {train_parts} |
| validation | {counts['validation']} | the split's selection side ({seeded}) |
| test | {counts['test']} | never trained on or used to choose a run |

The train side's answers: {answers.get('propose', 0)} propose,
{answers.get('escalate', 0)} escalate, {answers.get('explain', 0)} explain.
{redaction}
**Do not train on validation or test** if you want comparable numbers. The sealed
held-out split is not in this dataset.

## Where the records come from

- **Variations**: rewrites of train entries made by a local teacher pipeline; neither
  teacher saw the expected answer. {decision_sentence(summary.decisions)} Of {reviewed}
  reviewed rewrites, {counts['accepted']} were accepted ({rate}).

| Model | Licence | Role |
|---|---|---|
{teachers}
{disclosure}
The teachers' licences do not carry over to their outputs. `manifest.json` maps every
record to its split, origin, source, source file, licence, whether it was transformed, and
a `teachers` map for a variation.

## Candidate-scorer training file

`scorer-train.json` is the file the candidate scorer trained on: the train records, each
with its own offered candidates, letter map and gold label, plus missing-candidate rows
(the right operation removed, answer: escalate).

## Licence

{licence} (see `LICENSE`).
"""
