"""Teacher-label stage: soft labels on the train split only, resumable, with provenance.

Every train record is sent to a localhost ``/v1/systemone`` teacher. Questions go
**jointly** by default (the teacher's native mode, one request per record); ``joint=False``
sends one request per question. The mode is recorded on every row and in the manifest.

The cache is keyed by the sha256 of the request body (as ``data/teachers.py`` ``cache_key``
does, but in its own ``cache/`` directory under the stage directory). Only fully valid
answers are cached, so a killed run re-sends exactly the unanswered items. An incomplete,
malformed or failed answer is stored as a teacher error row, never as a label. One teacher
build id is written on every row. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill.records import Record
from jev_factory.distill.systemone import CallResult, SystemOneClient, build_body

TRAIN_SPLIT = "train"
LABELS_FILE = "teacher_labels.jsonl"
MANIFEST_FILE = "teacher_manifest.json"
CACHE_DIR = "cache"


@dataclass(frozen=True)
class TeacherConfig:
    """Who the teacher is: written onto every row of one labelling run."""

    base_url: str
    model: str
    repo_id: str
    revision: str
    build_id: str
    engine: str
    joint: bool = True
    timeout: float = 120.0

    def __post_init__(self) -> None:
        for name in ("base_url", "model", "repo_id", "revision", "build_id", "engine"):
            if not getattr(self, name):
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=f"teacher config field {name!r} is empty",
                    remediation="every row carries the teacher's full provenance",
                )

    def provenance(self) -> dict[str, Any]:
        return {
            "teacher_repo_id": self.repo_id,
            "teacher_revision": self.revision,
            "teacher_build_id": self.build_id,
            "engine": self.engine,
            "mode": "joint" if self.joint else "per-question",
        }


def to_wire(record: Record, qids: Sequence[str] | None = None) -> dict[str, Any]:
    """The mapping the SystemOne client takes; gold is never included."""
    chosen = list(record.questions) if qids is None else list(qids)
    questions: dict[str, Any] = {}
    for qid in chosen:
        q = record.questions[qid]
        spec: dict[str, Any] = {"type": q.type, "criteria": {n: d for n, d in q.criteria}}
        if q.instructions is not None:
            spec["instructions"] = q.instructions
        questions[qid] = spec
    return {"id": record.id, "state": record.state, "questions": questions}


def _refuse_non_train(records: Sequence[Record]) -> None:
    bad = [f"{r.id} ({r.split})" for r in records if r.split != TRAIN_SPLIT]
    if bad:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"refusing to label non-train record(s): {bad[:5]}",
            remediation="the teacher labels the train split only; filter before calling",
        )


def _cache_path(cache: Path, key: str) -> Path:
    return cache / f"{key}.json"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _row(
    cfg: TeacherConfig, rid: str, result: Mapping[str, Any], call: Mapping[str, Any], cached: bool
) -> dict[str, Any]:
    error = result.get("error")
    return {
        "record_id": rid,
        "question_id": result["qid"],
        "kind": result["kind"],
        "distribution": None if error else result.get("distribution"),
        "p_true": None if error else result.get("p_true"),
        "teacher_error": error,
        **cfg.provenance(),
        "request_config": call["request_config"],
        "response_sha256": call["response_sha256"],
        "cached": cached,
    }


def _call_dict(call: CallResult) -> dict[str, Any]:
    return {
        "request_config": call.request_config,
        "response_sha256": call.response_sha256,
        "results": [q.to_dict() for q in call.questions],
    }


def label_records(
    records: Sequence[Record],
    stage_dir: Path,
    cfg: TeacherConfig,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Label *records* (train only); write rows and manifest to *stage_dir*; return the manifest."""
    _refuse_non_train(records)  # before any request
    stage_dir = Path(stage_dir)
    cache = stage_dir / CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    client = client or SystemOneClient(cfg.base_url, model=cfg.model, timeout=cfg.timeout)

    rows: list[dict[str, Any]] = []
    sent = hits = 0
    for record in records:
        groups = [None] if cfg.joint else [[qid] for qid in record.questions]
        for qids in groups:
            wire = to_wire(record, qids)
            key = hashlib.sha256(
                json.dumps(
                    build_body(wire, model=cfg.model), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            path = _cache_path(cache, key)
            cached = path.is_file()
            if cached:
                call = json.loads(path.read_text(encoding="utf-8"))
                hits += 1
            else:
                call = _call_dict(client.ask(wire))
                sent += 1
                if all(r.get("error") is None for r in call["results"]):
                    _write_atomic(path, json.dumps(call, sort_keys=True))
            rows.extend(_row(cfg, record.id, r, call, cached) for r in call["results"])

    errors = sum(1 for r in rows if r["teacher_error"])
    manifest = {
        "stage": "teacher-labels",
        "status": "complete",
        **cfg.provenance(),
        "joint": cfg.joint,
        "records": len(records),
        "rows": len(rows),
        "labelled": len(rows) - errors,
        "teacher_errors": errors,
        "requests_sent": sent,
        "cache_hits": hits,
        "labels_file": LABELS_FILE,
    }
    _write_atomic(
        stage_dir / LABELS_FILE, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    )
    _write_atomic(stage_dir / MANIFEST_FILE, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
