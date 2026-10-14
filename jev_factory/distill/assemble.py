"""Assemble the distill training set: one target per (record, question), then freeze.

Inputs are the dataset's **train side only** (``train.jsonl`` of a dataset directory,
:func:`~jev_factory.distill.dataset.train_ids`), the teacher labels
(``teacher_labels.jsonl`` from :mod:`jev_factory.distill.teacher`: Clef's, and optionally an
alternative teacher's), the teacher audit (:func:`~jev_factory.distill.audit.require_audit`;
assemble refuses without it) and the operator-reviewed corrections
(:func:`jev_factory.review.corrections.trainable` only: a correction's own ``status`` field
is never read).

Precedence ``operator correction > human gold > alternative teacher > Clef``. The
interpretation below is recorded on every row (``precedence_rule``) and in the manifest
(:data:`PRECEDENCE`), pending the operator's confirmation:

1. ``correction``: an approved/edited correction replaces BOTH targets: gold is the
   correction's label/distribution, teacher is null (``target_source`` ``correction``).
2. ``teacher_weak``: else, a record from a source the audit flagged teacher-weak is
   gold-only (``human_gold``).
3. ``gold+teacher``: else gold is the record's dataset gold and teacher is the alternative
   teacher's label when it has a valid one, else Clef's (``alt_teacher+gold`` /
   ``clef+gold``). The ``teacher_filter`` knob ``agrees_with_gold`` nulls the teacher when
   its argmax disagrees with gold's (``gold_only:filtered``). A teacher error or a missing
   label nulls it too (``gold_only:teacher_error`` / ``gold_only:no_teacher_label``).

Refusals (fail closed, nothing written): no audit; an unregistered ``teacher_filter``; a
record whose split is not ``train``; a teacher row or trainable correction for a record off
the train side; more than one teacher build id among the teacher slots actually used
without a deviation id; any exact or near-duplicate (:func:`jev_factory.core.leakage.match`)
of a fit, selection, held-out or test state (reported by ids only); re-freezing a
different set over a frozen one without a deviation id.

Output (the trainer's input, :func:`jev_factory.backbones.pointer.train_distill.load_items`):
``<run_dir>/distill/assemble/train_set.jsonl`` plus ``assemble_manifest.json`` carrying its
sha256. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core.leakage import match
from jev_factory.core.textsim import (
    NEAR_DUP_JACCARD_THRESHOLD,
    WORD_JACCARD,
    normalize_text,
    shingles,
)
from jev_factory.distill.audit import require_audit
from jev_factory.distill.corrections import read_corrections
from jev_factory.distill.dataset import read_records, train_ids
from jev_factory.distill.prereg import DistillPrereg, require_knobs
from jev_factory.distill.records import Question, Record
from jev_factory.review import corrections as review

STAGE = "distill-assemble"
OUT_REL = "distill/assemble/train_set.jsonl"
MANIFEST_REL = "distill/assemble/assemble_manifest.json"
TRAIN_SPLIT = "train"
PROTECTED_SIDES = (("fit", "fit.jsonl"), ("selection", "selection.jsonl"))
SEALED_SIDES = (("held_out", "held_out.jsonl"), ("test", "test.jsonl"))
PRECEDENCE = {
    "order": ["operator_correction", "human_gold", "alternative_teacher", "clef"],
    "interpretation": [
        "correction: an approved/edited correction replaces both targets (teacher null)",
        "teacher_weak: a teacher-weak source is gold-only (teacher null)",
        "gold+teacher: dataset gold plus the alternative teacher's label, else Clef's;"
        " teacher_filter agrees_with_gold nulls a teacher whose argmax disagrees with gold",
    ],
    "status": "interpretation pending operator confirmation",
}
WEIGHT = 1.0


def _fail(message: str, remediation: str) -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- inputs


def _load_labels(path: Path | None, role: str) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    try:
        text = Path(path).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise _fail(
            f"{role} teacher labels unreadable ({type(exc).__name__})",
            "point at a teacher_labels.jsonl written by the teacher stage",
        ) from None
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("record_id")), str(row.get("question_id")))
        if key in out:
            raise _fail(f"{role} teacher labels repeat {key[0]}/{key[1]}", "one row per key")
        if not row.get("teacher_build_id"):
            raise _fail(
                f"{role} teacher row {key[0]}/{key[1]} has no teacher_build_id",
                "every teacher row must carry its build id",
            )
        out[key] = row
    return out


def _trainable_corrections(path: Path | None) -> dict[tuple[str, str], review.Trainable]:
    if path is None:
        return {}
    path = Path(path)
    items = review.trainable(read_corrections(path), review.read_records(review.review_path(path)))
    return {(t.record_id, t.question_id): t for t in items}


def _refuse_off_train(keys: Iterable[tuple[str, str]], allowed: frozenset[str], what: str) -> None:
    bad = sorted({rid for rid, _ in keys if rid not in allowed})
    if bad:
        raise _fail(
            f"refusing {what} for record(s) not on the dataset's train side: {bad[:5]}",
            "fit, selection, held-out and test records never become training targets",
        )


# --------------------------------------------------------------------------- targets


def _normalised(where: str, dist: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    stray = sorted(set(dist) - set(names))
    if stray:
        raise _fail(f"{where}: names {stray} are not options {list(names)}", "fix the source")
    probs = {n: float(dist.get(n, 0.0)) for n in names}
    total = math.fsum(probs.values())
    if any(not math.isfinite(p) or p < 0 for p in probs.values()) or total <= 0:
        raise _fail(f"{where}: not a probability distribution", "fix the source")
    return {n: p / total for n, p in probs.items()}


def _gold_target(record: Record, qid: str) -> dict[str, Any]:
    question, gold = record.questions[qid], record.gold[qid]
    where = f"gold {record.id}/{qid}"
    if question.type == "noul":
        return {"p_true": float(gold.probability_true)}
    if gold.probabilities:
        return {"probabilities": _normalised(where, gold.probabilities, question.names)}
    return {"label": gold.label}


def _correction_target(question: Question, item: review.Trainable) -> dict[str, Any]:
    where = f"correction {item.record_id}/{item.question_id}"
    names = question.names
    if item.label is not None and item.label not in names:
        raise _fail(f"{where}: label {item.label!r} not in {list(names)}", "fix the correction")
    if item.distribution:
        dist = _normalised(where, item.distribution, names)
        return {"p_true": dist["true"]} if question.type == "noul" else {"probabilities": dist}
    if question.type == "noul":
        return {"p_true": 1.0 if item.label == "true" else 0.0}
    return {"label": item.label}


def _teacher_dist(question: Question, row: Mapping[str, Any], where: str) -> dict[str, float]:
    if question.type == "noul":
        p = row.get("p_true")
        if p is None:
            raise _fail(f"{where}: noul teacher row has no p_true", "fix the teacher labels")
        return _normalised(where, {"false": 1.0 - float(p), "true": float(p)}, question.names)
    dist = row.get("distribution")
    if not isinstance(dist, Mapping):
        raise _fail(f"{where}: teacher row has no distribution", "fix the teacher labels")
    return _normalised(where, dist, question.names)


def _argmax(names: Sequence[str], probs: Mapping[str, float]) -> str:
    return max(names, key=lambda n: (probs.get(n, 0.0), -names.index(n)))


def _gold_argmax(question: Question, gold: Mapping[str, Any]) -> str:
    names = question.names
    if "p_true" in gold:
        return "true" if gold["p_true"] >= 0.5 else "false"
    if "probabilities" in gold:
        return _argmax(names, gold["probabilities"])
    return str(gold["label"])


def _usable(row: Mapping[str, Any] | None) -> bool:
    return row is not None and not row.get("teacher_error")


def _teacher_slot(
    key: tuple[str, str],
    question: Question,
    gold: Mapping[str, Any],
    alt: Mapping[tuple[str, str], Mapping[str, Any]],
    clef: Mapping[tuple[str, str], Mapping[str, Any]],
    teacher_filter: str,
) -> dict[str, Any]:
    """Rule 3: the teacher slot (alternative before Clef) and its filter outcome."""
    for role, labels in (("alt_teacher", alt), ("clef", clef)):
        row = labels.get(key)
        if _usable(row):
            break
    else:
        errored = any(labels.get(key) is not None for labels in (alt, clef))
        reason = "teacher_error" if errored else "no_teacher_label"
        return {
            "teacher": None,
            "target_source": f"gold_only:{reason}",
            "teacher_build_id": None,
            "teacher_filter": f"no_teacher:{reason}",
        }
    dist = _teacher_dist(question, row, f"{role} teacher {key[0]}/{key[1]}")
    out = {
        "teacher": dist,
        "target_source": f"{role}+gold",
        "teacher_build_id": str(row["teacher_build_id"]),
        "teacher_filter": "kept",
    }
    if teacher_filter == "agrees_with_gold":
        if _argmax(question.names, dist) != _gold_argmax(question, gold):
            out.update(
                teacher=None,
                target_source="gold_only:filtered",
                teacher_filter="dropped:disagrees_with_gold",
            )
    return out


def build_rows(
    records: Sequence[Record],
    *,
    weak_sources: frozenset[str],
    corrections: Mapping[tuple[str, str], review.Trainable],
    alt: Mapping[tuple[str, str], Mapping[str, Any]],
    clef: Mapping[tuple[str, str], Mapping[str, Any]],
    teacher_filter: str,
) -> list[dict[str, Any]]:
    """One trainer row per (record, question), by the precedence in the module docstring."""
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r.id):
        for qid in sorted(record.questions):
            question, key = record.questions[qid], (record.id, qid)
            item = corrections.get(key)
            row: dict[str, Any] = {
                "record_id": record.id,
                "question_id": qid,
                "kind": question.type,
                "state": record.state,
                "instructions": question.instructions or "",
                "options": [list(c) for c in question.criteria],
                "weight": WEIGHT,
                "source": record.source,
                "licence": record.licence,
                "gold_provenance": record.gold_provenance,
                "correction_id": None,
                "correction_status": None,
                "teacher_build_id": None,
            }
            if item is not None:
                row.update(
                    gold=_correction_target(question, item),
                    teacher=None,
                    target_source="correction",
                    precedence_rule="correction",
                    correction_id=review.key_of(*key),
                    correction_status=item.status,
                    teacher_filter="not_applied:correction",
                )
            elif record.source in weak_sources:
                row.update(
                    gold=_gold_target(record, qid),
                    teacher=None,
                    target_source="human_gold",
                    precedence_rule="teacher_weak",
                    teacher_filter="not_applied:teacher_weak",
                )
            else:
                gold = _gold_target(record, qid)
                row.update(gold=gold, precedence_rule="gold+teacher")
                row.update(_teacher_slot(key, question, gold, alt, clef, teacher_filter))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- leakage


def _prefixes(
    sets: Sequence[frozenset], rank: Mapping[Any, int], threshold: float
) -> list[list[Any]]:
    """Prefix filtering: two sets with Jaccard >= threshold share a token of their prefixes
    (each set sorted by the same global rarity order, prefix ``|s| - ceil(t*|s|) + 1``; the
    epsilon only ever lengthens a prefix, so no pair is lost to float rounding)."""
    out: list[list[Any]] = []
    for s in sets:
        ordered = sorted(s, key=lambda t: rank[t])
        keep = len(ordered) - math.ceil(threshold * len(ordered) - 1e-9) + 1
        out.append(ordered[: max(keep, 1)])
    return out


def _candidates(train: Sequence[frozenset], protected: Sequence[frozenset], threshold: float):
    counts = Counter(t for s in (*train, *protected) for t in s)
    rank = {t: i for i, (t, _) in enumerate(sorted(counts.items(), key=lambda kv: (kv[1], kv[0])))}
    index: dict[Any, list[int]] = {}
    for j, prefix in enumerate(_prefixes(protected, rank, threshold)):
        for tok in prefix:
            index.setdefault(tok, []).append(j)
    for i, prefix in enumerate(_prefixes(train, rank, threshold)):
        yield i, {j for tok in prefix for j in index.get(tok, ())}


def leakage_matches(
    train: Sequence[Mapping[str, Any]], protected: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Every (train, protected) pair :func:`jev_factory.core.leakage.match` flags; ids only.

    Equivalent to checking every pair, but candidates come from an exact-text index and
    prefix-filtered shingle and word-set indexes, and each candidate is confirmed by
    ``match`` itself, so the verdicts are exactly ``match``'s.
    """
    flat = [(name, e) for name, entries in protected.items() for e in entries]
    t_norm = [normalize_text(str(e["text"])) for e in train]
    p_norm = [normalize_text(str(e["text"])) for _, e in flat]
    for side, texts in (("training", t_norm), ("protected", p_norm)):
        if any(not t for t in texts):
            raise _fail(f"a {side} entry has no text to check", "leakage fails closed")
    cands: dict[int, set[int]] = {i: set() for i in range(len(train))}
    exact: dict[str, list[int]] = {}
    for j, t in enumerate(p_norm):
        exact.setdefault(t, []).append(j)
    for i, t in enumerate(t_norm):
        cands[i].update(exact.get(t, ()))
    for build, threshold in (
        (shingles, NEAR_DUP_JACCARD_THRESHOLD),
        (lambda t: frozenset(t.split()), WORD_JACCARD),
    ):
        for i, js in _candidates([build(t) for t in t_norm], [build(t) for t in p_norm], threshold):
            cands[i].update(js)
    found = []
    for i in range(len(train)):
        for j in sorted(cands[i]):
            name, other = flat[j]
            kind = match(str(train[i]["text"]), str(other["text"]))
            if kind:
                found.append(
                    {
                        "train_id": str(train[i]["id"]),
                        "protected": name,
                        "protected_id": str(other["id"]),
                        "kind": kind,
                    }
                )
    return found


def _protected_entries(dataset_dir: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for name, filename in PROTECTED_SIDES + SEALED_SIDES:
        path = dataset_dir / filename
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            out[name] = [
                {"id": str(row["id"]), "text": str(row["state"])}
                for row in map(json.loads, filter(str.strip, lines))
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise _fail(
                f"{path.name}: protected side unreadable ({type(exc).__name__})",
                "leakage fails closed; point at a directory written by write_dataset",
            ) from None
    return out


def _check_leakage(records: Sequence[Record], dataset_dir: Path) -> dict[str, Any]:
    protected = _protected_entries(dataset_dir)
    train = [{"id": r.id, "text": r.state} for r in records]
    hits = leakage_matches(train, protected)
    if hits:
        shown = ", ".join(
            f"{h['train_id']} is a {h['kind']} of {h['protected']} {h['protected_id']}"
            for h in hits[:5]
        )
        raise _fail(
            f"leakage: {len(hits)} training record(s) match a protected side: {shown}",
            "drop the matching training records upstream; assemble never freezes a leak",
        )
    return {
        "train": len(train),
        "protected": {name: len(entries) for name, entries in protected.items()},
        "hits": 0,
        "text": "record state",
    }


# --------------------------------------------------------------------------- assemble


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _file_ref(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": _sha256(Path(path).read_bytes())}


def assemble(
    run_dir: Path | str,
    dataset_dir: Path | str,
    *,
    prereg: DistillPrereg,
    teacher_filter: str,
    clef_labels: Path | str | None,
    alt_labels: Path | str | None = None,
    corrections_path: Path | str | None = None,
    deviation_id: str | None = None,
) -> dict[str, Any]:
    """Assemble, leakage-check and freeze the training set; return the manifest."""
    run_dir, dataset_dir = Path(run_dir), Path(dataset_dir)
    deviation = (deviation_id or "").strip() or None
    audit = require_audit(run_dir)
    require_knobs(prereg, {"teacher_filter": teacher_filter})
    allowed = train_ids(dataset_dir)
    records = read_records(dataset_dir / "train.jsonl")
    bad = [f"{r.id} ({r.split})" for r in records if r.split != TRAIN_SPLIT]
    if bad:
        raise _fail(
            f"refusing non-train record(s) on the train side: {bad[:5]}",
            "only train-split records may become training targets",
        )
    clef_path = None if clef_labels is None else Path(clef_labels)
    alt_path = None if alt_labels is None else Path(alt_labels)
    if clef_path is None and alt_path is None:
        raise _fail("no teacher labels given", "pass Clef's and/or an alternative teacher's")
    clef = _load_labels(clef_path, "clef")
    alt = _load_labels(alt_path, "alternative")
    corr_path = None if corrections_path is None else Path(corrections_path)
    corrections = _trainable_corrections(corr_path)
    _refuse_off_train(clef, allowed, "Clef teacher rows")
    _refuse_off_train(alt, allowed, "alternative teacher rows")
    _refuse_off_train(corrections, allowed, "trainable corrections")

    weak = frozenset(s["source"] for s in audit["sources"] if s.get("teacher_weak"))
    rows = build_rows(
        records,
        weak_sources=weak,
        corrections=corrections,
        alt=alt,
        clef=clef,
        teacher_filter=teacher_filter,
    )
    builds = sorted({r["teacher_build_id"] for r in rows if r["teacher_build_id"]})
    if len(builds) > 1 and deviation is None:
        raise _fail(
            f"the training set mixes teacher build ids {builds}",
            "use one teacher build, or record a deviation and pass its id",
        )
    leak = _check_leakage(records, dataset_dir)

    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    blob = data.encode("utf-8")
    digest = _sha256(blob)
    out_path, manifest_path = run_dir / OUT_REL, run_dir / MANIFEST_REL
    if manifest_path.is_file() and deviation is None:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        if frozen.get("output", {}).get("sha256") != digest:
            raise _fail(
                "a different training set is already frozen for this run",
                "a changed training set is a recorded deviation: pass its id",
            )
    manifest = {
        "stage": STAGE,
        "status": "frozen",
        "precedence": PRECEDENCE,
        "knobs": {"teacher_filter": teacher_filter},
        "teacher_build_ids": builds,
        "deviation_id": deviation,
        "teacher_weak_sources": sorted(weak),
        "inputs": {
            "dataset_train": _file_ref(dataset_dir / "train.jsonl"),
            "clef_labels": _file_ref(clef_path),
            "alt_labels": _file_ref(alt_path),
            "corrections": _file_ref(corr_path),
            "corrections_review": _file_ref(
                review.review_path(corr_path)
                if corr_path and review.review_path(corr_path).is_file()
                else None
            ),
            "audit_floor": audit.get("floor"),
        },
        "counts": {
            "records": len(records),
            "rows": len(rows),
            "trainable_corrections": len(corrections),
            "target_source": dict(sorted(Counter(r["target_source"] for r in rows).items())),
            "teacher_filter": dict(sorted(Counter(r["teacher_filter"] for r in rows).items())),
        },
        "leakage": leak,
        "output": {"path": str(out_path), "sha256": digest, "rows": len(rows)},
    }
    _write_atomic(out_path, blob)
    _write_atomic(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest
