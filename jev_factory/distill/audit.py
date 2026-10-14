"""Teacher audit: the teacher's train-side soft labels scored against gold, per source.

For each ``(source, gold_provenance)`` group of train records the audit reports ``n`` scored
questions, argmax accuracy, mean log-loss on the gold class and ECE (confidence = the
teacher's top probability). A human-gold source whose accuracy is below the pre-registered
``teacher_weak_floor`` knob is flagged ``teacher_weak``.

Sources whose gold is model-generated (typed-decisions) are **reported but never flagged**:
a model's gold cannot indict the teacher, so ``teacher_weak`` is ``None`` and ``flaggable``
is false for them. A source that mixes provenances is split into one row per provenance.

The table is written to ``<run_dir>/distill/teacher_audit.json``. :func:`require_audit` is the
guard the assemble stage calls first: it refuses until a valid table exists. Stdlib only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core.metrics import ece
from jev_factory.distill.metrics import LOG_EPSILON, argmax
from jev_factory.distill.prereg import DistillPrereg, require_knobs
from jev_factory.distill.records import Question, Record

STAGE = "teacher-audit"
AUDIT_REL = "distill/teacher_audit.json"
HUMAN = "human"


def _fail(message: str, remediation: str) -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def load_labels(path: Path) -> list[dict[str, Any]]:
    """Read ``teacher_labels.jsonl`` rows."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _teacher_probs(question: Question, row: Mapping[str, Any]) -> tuple[float, ...] | None:
    if question.type == "noul":
        p = row.get("p_true")
        return None if p is None else (1.0 - float(p), float(p))
    dist = row.get("distribution")
    if not dist:
        return None
    try:
        return tuple(float(dist[name]) for name in question.names)
    except KeyError:
        return None


def _gold_index(record: Record, qid: str) -> int:
    question, gold = record.questions[qid], record.gold[qid]
    if question.type == "noul":
        return 1 if float(gold.probability_true) >= 0.5 else 0
    return question.names.index(gold.label)


def _summary(
    source: str, provenance: str, scored: list[tuple[float, bool, float]], errs: int, unl: int
) -> dict[str, Any]:
    n = len(scored)
    return {
        "source": source,
        "gold_provenance": provenance,
        "n": n,
        "accuracy": (sum(1 for _, ok, _ in scored if ok) / n) if n else None,
        "log_loss": (math.fsum(ll for _, _, ll in scored) / n) if n else None,
        "ece": ece([(conf, ok) for conf, ok, _ in scored]) if n else None,
        "teacher_errors": errs,
        "unlabelled": unl,
        "flaggable": provenance == HUMAN,
    }


def build_table(
    records: Sequence[Record], label_rows: Sequence[Mapping[str, Any]], *, floor: float
) -> dict[str, Any]:
    """Score *label_rows* against the gold of *records*; flag human-gold sources below *floor*."""
    if not 0.0 <= floor <= 1.0:
        raise _fail(f"teacher_weak_floor {floor!r} is outside [0, 1]", "use the registered knob")
    by_key = {(r["record_id"], r["question_id"]): r for r in label_rows}
    groups: dict[tuple[str, str], list[Record]] = {}
    for rec in records:
        groups.setdefault((rec.source, rec.gold_provenance), []).append(rec)

    sources: list[dict[str, Any]] = []
    for (source, prov), recs in sorted(groups.items()):
        scored: list[tuple[float, bool, float]] = []
        errs = unl = 0
        for rec in recs:
            for qid, question in rec.questions.items():
                row = by_key.get((rec.id, qid))
                if row is None:
                    unl += 1
                    continue
                probs = None if row.get("teacher_error") else _teacher_probs(question, row)
                if probs is None:
                    errs += 1
                    continue
                gold = _gold_index(rec, qid)
                pred = argmax(probs)
                scored.append((probs[pred], pred == gold, -math.log(max(probs[gold], LOG_EPSILON))))
        entry = _summary(source, prov, scored, errs, unl)
        weak = entry["flaggable"] and entry["n"] > 0 and entry["accuracy"] < floor
        entry["teacher_weak"] = bool(weak) if entry["flaggable"] and entry["n"] else None
        sources.append(entry)

    return {
        "stage": STAGE,
        "floor": floor,
        "sources": sources,
        "teacher_weak_sources": sorted({s["source"] for s in sources if s["teacher_weak"]}),
    }


def run_audit(
    records: Sequence[Record],
    labels_path: Path,
    run_dir: Path,
    prereg: DistillPrereg,
    *,
    floor: float,
) -> dict[str, Any]:
    """Audit the teacher and write the table; *floor* must be a pre-registered grid value."""
    require_knobs(prereg, {"teacher_weak_floor": floor})
    table = build_table(records, load_labels(labels_path), floor=floor)
    path = Path(run_dir) / AUDIT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return table


def require_audit(run_dir: Path) -> dict[str, Any]:
    """Guard for assemble: return the audit table, or refuse if it is missing or invalid."""
    path = Path(run_dir) / AUDIT_REL
    remedy = "run the teacher audit (distill audit) before assemble"
    if not path.is_file():
        raise _fail("no teacher audit table for this run; assemble refuses", remedy)
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _fail(f"teacher audit table is not valid JSON: {exc}", remedy) from exc
    if (
        not isinstance(table, dict)
        or table.get("stage") != STAGE
        or not isinstance(table.get("sources"), list)
    ):
        raise _fail("teacher audit table is malformed; assemble refuses", remedy)
    return table
