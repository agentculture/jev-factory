"""Distill edge: a non-test measurement of the served student on a board, through a tunnel (t15).

The board (AGX Orin or Orin Nano) serves the deployed bundle with the documented command under
``systemd-run --user -p MemoryMax=<cap>``; the operator opens a tunnel, and this module talks to
``http://127.0.0.1:<port>`` only. jev never ssh-executes on the board.

What is measured, on the **selection fold and the probe set only**: per-call latency (median and
p95), the server's resident memory and ``MemAvailable`` (sampled from ``GET /v1/stats`` around
every call), and prediction agreement with the final's student predictions. Nothing from the
sealed sides is read: a record whose id is in a sealed side (``test`` or ``held_out``), whose
split is ``test``, or a fold other than ``selection``/``probe`` exits non-zero before any request.

Refusals, all before the first question: ``/v1/bundle`` differs from the deployed artifact's
per-file sha256s (the final's bundle); ``/v1/stats`` reports no cgroup memory cap; the
teacher-label lock exists (one GPU job at a time, and a labelling run owns the box).

The manifest records the board model, the L4T release and the cap. The probe set is private: only
its sha256 and count are written, and probe record ids are hashed in the predictions file.

Assumptions about the final (t14 extends it later; consumed defensively): the student's answers
are ``<run>/distill/final/predictions/student.json``, a list of rows with ``record_id`` and
``questions: {qid: {kind, distribution, p_true, error}}``; agreement is over the
``(record_id, qid)`` keys both files hold (empty overlap is reported, not hidden). The expected
bundle is read from the deployed artifact on disk (``quant_manifest.json`` ``deployed_artifact``),
or from ``final.json`` ``deployed.bundle`` (``{"files": {...}, "manifest_sha256": ...}``) if the
final ever records one. Dry-run by default; stdlib only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as dquant
from jev_factory.distill.records import Record
from jev_factory.distill.systemone import SystemOneClient
from jev_factory.measure.preflight import _NoRedirect, require_localhost

STAGE = "distill-edge"
EDGE_REL = "distill/edge"
MANIFEST_FILE = "edge_manifest.json"
PREDICTIONS_FILE = "edge_predictions.json"
FINAL_REL = "distill/final"
FINAL_STUDENT_PREDICTIONS = f"{FINAL_REL}/predictions/student.json"
FINAL_FILE = "final.json"
#: The teacher-label stage holds this file while it runs (the wiring task makes it write it).
TEACHER_LOCK_REL = "distill/teacher/teacher-label.lock"
FOLD_SELECTION = "selection"
FOLD_PROBE = "probe"
FOLDS = (FOLD_SELECTION, FOLD_PROBE)
SEALED_SIDE_FILES = ("test", "held_out")
BUNDLE_PATH = "/v1/bundle"
STATS_PATH = "/v1/stats"
HTTP_TIMEOUT = 30.0
Clock = Callable[[], float]


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- teacher-label lock


def teacher_lock_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / TEACHER_LOCK_REL


def acquire_teacher_lock(run_dir: Path | str, owner: str = "teacher-label") -> Path:
    """Create the lock (for the teacher-label stage to call); refuse if it is already held."""
    path = teacher_lock_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise _refuse(f"{path}: the teacher-label lock is already held") from None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"owner": owner, "pid": os.getpid(), "date": _now()}) + "\n")
    return path


def release_teacher_lock(run_dir: Path | str) -> None:
    teacher_lock_path(run_dir).unlink(missing_ok=True)


def require_no_teacher_lock(run_dir: Path | str) -> None:
    path = teacher_lock_path(run_dir)
    if path.exists() or path.is_symlink():
        raise _refuse(
            f"{path}: the teacher-label lock exists; the box is busy labelling",
            "wait for teacher-label to finish (or remove a stale lock), then rerun",
        )


# --------------------------------------------------------------------- inputs


def _load_probe(path: Path) -> list[Record]:
    """The probe set: a JSON list, ``{"records": [...]}``, or JSONL of record mappings."""
    try:
        raw = path.read_text(encoding="utf-8")
        try:
            doc = json.loads(raw)
            items = doc if isinstance(doc, list) else doc.get("records", [doc])
        except ValueError:
            items = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        return [Record.from_json(item) for item in items]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _refuse(f"{path.name}: unreadable probe set ({type(exc).__name__})") from None


def sealed_ids(dataset_dir: Path | str) -> frozenset[str]:
    """Ids of the test and held-out sides (ids only; the sealed text is not used)."""
    ids: set[str] = set()
    for stem in SEALED_SIDE_FILES:
        path = Path(dataset_dir) / f"{stem}.jsonl"
        if not path.is_file():
            raise _refuse(f"{path}: no sealed side to guard against")
        ids.update(dataset.load_sealed(path).ids)
    return frozenset(ids)


def collect_inputs(
    *, selection: Path | str, probe: Path | str, dataset_dir: Path | str
) -> dict[str, list[Record]]:
    """Load selection and probe; refuse any test or held-out record (fail closed)."""
    blocked = sealed_ids(dataset_dir)
    selection_records = bench.load_fold(selection)
    if Path(selection).stem.lower() != FOLD_SELECTION:
        raise _refuse(f"{Path(selection).name}: edge takes the selection fold only")
    probe_records = _load_probe(Path(probe))
    out: dict[str, list[Record]] = {}
    for fold, records in ((FOLD_SELECTION, selection_records), (FOLD_PROBE, probe_records)):
        for rec in records:
            if rec.split in (dataset.TEST_SPLIT, "held_out", "held-out", "heldout"):
                raise _refuse(f"{fold}: a record has split {rec.split!r}; test/held-out refused")
        out[fold] = bench.check_records(records, sealed=[_IdsOnly(blocked)], origin=f"{fold} input")
    return out


class _IdsOnly:
    """Duck-typed sealed side: ``check_records`` only reads ``.ids``."""

    def __init__(self, ids: frozenset[str]) -> None:
        self.ids = ids


# --------------------------------------------------------------------- the server


def _get_json(base_url: str, path: str, timeout: float = HTTP_TIMEOUT) -> dict[str, Any]:
    try:
        require_localhost(base_url)
    except ValueError as exc:
        raise _refuse(str(exc), "open a tunnel and pass its http://127.0.0.1:<port> URL") from None
    opener = urllib.request.build_opener(_NoRedirect)
    url = base_url.rstrip("/") + path
    try:
        with opener.open(url, timeout=timeout) as response:  # nosec B310 - localhost only
            doc = json.loads(response.read())
    except (OSError, ValueError) as exc:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"GET {url} failed: {exc}",
            remediation="check the tunnel and that the board's server is up",
        ) from None
    if not isinstance(doc, dict):
        raise _refuse(f"GET {url}: expected a JSON object")
    return doc


def expected_bundle(run_dir: Path | str) -> dict[str, Any]:
    """The final's bundle: per-file sha256s and the manifest sha256 of the deployed artifact."""
    run_dir = Path(run_dir)
    final = _read_json(run_dir / FINAL_REL / FINAL_FILE, "final.json")
    recorded = (final.get("deployed") or {}).get("bundle")
    if isinstance(recorded, Mapping) and isinstance(recorded.get("files"), Mapping):
        return {
            "files": dict(recorded["files"]),
            "manifest_sha256": recorded.get("manifest_sha256"),
        }
    manifest = _read_json(run_dir / dquant.QUANT_REL / dquant.MANIFEST_FILE, "quant manifest")
    artifact = manifest.get("deployed_artifact")
    if not artifact:
        raise _refuse("the quant manifest names no deployed_artifact; cannot know the bundle")
    from jev_factory.backbones.pointer import quantized

    att = quantized.bundle_attestation(artifact)
    return {"files": dict(att["files"]), "manifest_sha256": att["manifest_sha256"]}


def check_bundle(served: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse unless the served ``/v1/bundle`` hashes equal the final's, file for file."""
    files = served.get("files")
    if not isinstance(files, Mapping) or dict(files) != dict(expected["files"]):
        differing = sorted(
            n
            for n in set(files or {}) | set(expected["files"])
            if (files or {}).get(n) != expected["files"].get(n)
        )
        raise _refuse(
            f"/v1/bundle differs from the final's bundle sha256s: {differing[:6]}",
            "serve the artifact the final measured (same files, same hashes)",
        )
    want = expected.get("manifest_sha256")
    if want is not None and served.get("manifest_sha256") != want:
        raise _refuse("/v1/bundle manifest_sha256 differs from the final's bundle")


def memory_cap(stats: Mapping[str, Any]) -> dict[str, Any]:
    """The cgroup cap from ``/v1/stats``; refuse when there is none."""
    cgroup = stats.get("cgroup") if isinstance(stats.get("cgroup"), Mapping) else {}
    effective = cgroup.get("effective_memory_max")
    if isinstance(effective, bool) or not isinstance(effective, int) or effective <= 0:
        raise _refuse(
            "/v1/stats reports no cgroup memory cap",
            "serve under `systemd-run --user -p MemoryMax=<cap>` so the figures mean something",
        )
    return {
        "effective_memory_max": effective,
        "memory_max": cgroup.get("memory_max"),
        "cgroup_path": cgroup.get("path"),
        "cgroup_version": cgroup.get("version"),
    }


# --------------------------------------------------------------------- agreement


def _decision(question: Mapping[str, Any]) -> Any:
    if question.get("error"):
        return None
    dist = question.get("distribution")
    if isinstance(dist, Mapping) and dist:
        return max(sorted(dist), key=lambda k: dist[k])
    p_true = question.get("p_true")
    return None if p_true is None else bool(p_true >= 0.5)


def _decisions(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for row in rows:
        for qid, question in (row.get("questions") or {}).items():
            out[(str(row.get("record_id")), qid)] = _decision(question)
    return out


def agreement(
    edge_rows: Sequence[Mapping[str, Any]], final_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Argmax agreement over the (record, question) keys both prediction sets answered."""
    edge, final = _decisions(edge_rows), _decisions(final_rows)
    shared = [k for k in edge if k in final and edge[k] is not None and final[k] is not None]
    agree = sum(1 for k in shared if edge[k] == final[k])
    return {
        "n_compared": len(shared),
        "n_agree": agree,
        "agreement": agree / len(shared) if shared else None,
        "n_edge_questions": len(edge),
        "n_final_questions": len(final),
    }


def _final_rows(run_dir: Path) -> list[Mapping[str, Any]]:
    path = run_dir / FINAL_STUDENT_PREDICTIONS
    if not path.is_file():
        return []
    doc = _read_json(path, "final student predictions")
    rows = doc if isinstance(doc, list) else doc.get("predictions", [])
    return [r for r in rows if isinstance(r, Mapping)]


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"{path}: cannot read the {what} ({type(exc).__name__})") from None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- the run


def _sample(base_url: str, peaks: dict[str, Any]) -> None:
    stats = _get_json(base_url, STATS_PATH)
    rss, avail = stats.get("rss_bytes"), stats.get("mem_available_bytes")
    if isinstance(rss, int):
        peaks["rss_bytes_peak"] = max(peaks["rss_bytes_peak"] or 0, rss)
    if isinstance(avail, int):
        low = peaks["mem_available_bytes_min"]
        peaks["mem_available_bytes_min"] = avail if low is None else min(low, avail)
    peaks["samples"] += 1


def _row_for(fold: str, rec: Record, result: Any, latency: float) -> dict[str, Any]:
    row = bench._prediction_row("student-edge", fold, rec, result, latency)
    if fold == FOLD_PROBE:  # the probe set is private: ids are hashed in what is written
        row["record_id"] = "sha256:" + _sha(rec.id)
    return row


def run_edge(
    *,
    run_dir: Path | str,
    dataset_dir: Path | str,
    selection: Path | str,
    probe: Path | str,
    base_url: str,
    served_model: str,
    label: str,
    apply: bool = False,
    client: Any = None,
    clock: Clock = time.perf_counter,
    now: Callable[[], str] = _now,
) -> dict[str, Any]:
    """Plan (dry-run) or run the edge measurement; write the manifest and predictions."""
    run_dir = Path(run_dir)
    try:
        require_localhost(base_url)
    except ValueError as exc:
        raise _refuse(str(exc), "open a tunnel and pass its http://127.0.0.1:<port> URL") from None
    folds = collect_inputs(selection=selection, probe=probe, dataset_dir=dataset_dir)
    require_no_teacher_lock(run_dir)
    base = {
        "stage": STAGE,
        "label": label,
        "base_url": base_url,
        "served_model": served_model,
        "folds": {
            FOLD_SELECTION: {"count": len(folds[FOLD_SELECTION]), "file": Path(selection).name},
            FOLD_PROBE: {
                "count": len(folds[FOLD_PROBE]),
                "sha256": dp.sha256_file(Path(probe)),
            },
        },
    }
    if not apply:
        return {**base, "dry_run": True}
    expected = expected_bundle(run_dir)
    served = _get_json(base_url, BUNDLE_PATH)
    check_bundle(served, expected)
    first = _get_json(base_url, STATS_PATH)
    cap = memory_cap(first)
    client = client or SystemOneClient(base_url, model=served_model)
    peaks: dict[str, Any] = {"rss_bytes_peak": None, "mem_available_bytes_min": None, "samples": 0}
    predictions: list[dict[str, Any]] = []
    raw_ids: list[str] = []
    latencies: list[float] = []
    for fold, records in folds.items():
        for rec in records:
            t0 = clock()
            result = client.ask(bench.wire_record(rec))
            elapsed = clock() - t0
            latencies.append(elapsed)
            predictions.append(_row_for(fold, rec, result, elapsed))
            raw_ids.append(rec.id)
            _sample(base_url, peaks)
    return _finish(base, run_dir, first, cap, served, peaks, latencies, predictions, raw_ids, now())


def _finish(
    base: dict[str, Any],
    run_dir: Path,
    stats: Mapping[str, Any],
    cap: Mapping[str, Any],
    served: Mapping[str, Any],
    peaks: Mapping[str, Any],
    latencies: Sequence[float],
    predictions: list[dict[str, Any]],
    raw_ids: Sequence[str],
    date: str,
) -> dict[str, Any]:
    # agreement is computed on raw ids; the written file hashes probe ids
    raw_rows = [{**row, "record_id": rid} for row, rid in zip(predictions, raw_ids)]
    out_dir = run_dir / EDGE_REL / base["label"]
    manifest = {
        **base,
        "dry_run": False,
        "date": date,
        "board": {
            "device_model": stats.get("device_model"),
            "nv_tegra_release": stats.get("nv_tegra_release"),
        },
        "cap": dict(cap),
        "bundle": {"files": served.get("files"), "manifest_sha256": served.get("manifest_sha256")},
        "latency": bench.latency_summary(latencies),
        "memory": dict(peaks),
        "n_records": len(predictions),
        "n_failed": sum(1 for p in predictions if p["error"] or bench._has_error(p)),
        "agreement": agreement(raw_rows, _final_rows(run_dir)),
        "predictions_file": PREDICTIONS_FILE,
    }
    _write_json(out_dir / PREDICTIONS_FILE, predictions)
    _write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest
