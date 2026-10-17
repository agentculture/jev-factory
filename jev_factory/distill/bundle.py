"""Distill bundle (t21): the deployed student as a model bundle plus a dataset bundle.

Two private upload folders are written under one output directory:

``model/``
    The deployed student exactly as ``distill/quant/format_decision.json`` chose it, verified
    by its own reader before and after: a bf16 ``hobson-hf-export/1`` bundle
    (:func:`jev_factory.backbones.pointer.layout.read_layout`) or a torchao weight-only
    artifact (:func:`jev_factory.backbones.pointer.quantized.read_artifact`). Its
    ``strands_decider_config.json`` carries **jev-factory's fitted** ``temperature_by_kind``
    (from ``distill/quant/calibration.json``), never Strands' upstream values; the scalar
    ``temperature`` is the identity ``1.0`` (a kind without a fitted value is left
    uncalibrated, not given upstream's). Beside the weights: ``calibration.json``,
    ``gate.json``, ``scorer-train.json`` (an index of the frozen training set with each row's
    target source, licence and teacher build, with no text), ``report.json`` (the single
    final comparison), ``assemble_manifest.json`` and ``distill_provenance.json``.

    Under a bf16 ship (``deployed_format bf16``, quantize skipped) the student is the train
    stage's winner, ``distill/train/<winner>/bundle`` (``hobson-hf-export/1``), with
    ``distill/select/calibration.json`` (:mod:`jev_factory.distill.deployed`), and
    ``distill_provenance.json`` additionally carries ``bundle_kind: bf16`` and
    ``deployed_student`` (where it came from). A quant bundle is unchanged.

``dataset/``
    ``train.jsonl``: the frozen training rows, **only those whose source licence is
    Apache-2.0 or MIT**, each row keeping its ``licence``, ``source`` and ``target_source``;
    the rest are counted by licence in ``dataset.json`` and left out.

Both folders carry a ``NOTICE`` listing every attribution string from the dataset manifest
(``licences`` table, ``attribution`` key at any depth, plus any per-record ``attribution``
field); a ``cc-by-4.0`` record in the dataset bundle without one refuses the build and the
upload. Before anything is sent, :func:`upload_bundles` runs the pre-upload scans
(:func:`pre_upload_scan`) and prints file and line only.

Neither folder ever holds the sealed held-out or test text; a training row whose record id is
on a sealed side refuses the build. A symlink anywhere in an input refuses it. Nothing here
uploads: :func:`upload_bundles` calls :mod:`jev_factory.release.hub`, which sends nothing
without ``apply``, always creates repositories private, fetches back and verifies sha256,
and has no public-visibility call.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # nosec B404 -- runs the repo's own scan-secrets.py, argv list, no shell
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jev_factory.backbones.pointer import layout, quantized
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import assemble as dassemble
from jev_factory.distill import audit as daudit
from jev_factory.distill import dataset as ddataset
from jev_factory.distill import deployed as ddeployed
from jev_factory.distill import final as dfinal
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dprereg
from jev_factory.release import hub, scan

MODEL_DIR = "model"
DATASET_DIR = "dataset"
ALLOWED_LICENCES = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
ATTRIBUTION_LICENCES = frozenset({"cc-by-4.0"})
NOTICE_FILE = "NOTICE"
PROBE_SET_KEY = "distill_probe_set"
SIDE_MODEL_LIST_KEY = "distill_side_model_list"
SECRETS_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan-secrets.py"
BUNDLE_FILE = "bundle.json"
CALIBRATION_FILE = "calibration.json"
GATE_FILE = "gate.json"
SCORER_TRAIN_FILE = "scorer-train.json"
PROVENANCE_FILE = "distill_provenance.json"
#: Set (to ``bf16``) in ``distill_provenance.json`` only for a bf16-kind bundle.
BUNDLE_KIND_KEY = "bundle_kind"
REPORT_FILE = "report.json"
MANIFEST_COPY = "assemble_manifest.json"
TRAIN_FILE = "train.jsonl"
DATASET_MANIFEST = "dataset.json"
BUNDLE_FORMAT = "jev-distill-bundle/1"
SURFACE_KIND = "frozen train set (sha256 of distill/assemble/train_set.jsonl)"
IDENTITY_TEMPERATURE = 1.0
_ASSEMBLE_MANIFEST = "assemble manifest"
_INDEX_KEYS = (
    "record_id",
    "question_id",
    "kind",
    "target_source",
    "precedence_rule",
    "teacher_filter",
    "teacher_build_id",
    "source",
    "licence",
    "gold_provenance",
)


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, what: str) -> bytes:
    if path.is_symlink():
        raise _refuse(f"{path}: {what} is a symlink", "a bundle never follows or holds a link")
    if not path.is_file():
        raise _refuse(f"{path}: no {what}", "run the earlier distill stage first")
    return path.read_bytes()


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(_read_bytes(path, what).decode("utf-8"))
    except ValueError as exc:
        raise _refuse(f"{path}: {what} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _refuse(f"{path}: {what} is not a JSON object")
    return data


def _dump(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _no_symlinks(folder: Path) -> None:
    links = scan.check_no_symlinks(folder)
    if links:
        raise _refuse(f"symlink(s) in {folder}: {links[:5]}", "a bundle never holds a link")


# --------------------------------------------------------------------- inputs


def _train_rows(run_dir: Path) -> tuple[bytes, list[dict[str, Any]]]:
    blob = _read_bytes(run_dir / dassemble.OUT_REL, "frozen training set")
    manifest = _read_json(run_dir / dassemble.MANIFEST_REL, _ASSEMBLE_MANIFEST)
    recorded = (manifest.get("output") or {}).get("sha256")
    if recorded != _sha256_bytes(blob):
        raise _refuse(
            "train_set.jsonl does not match the sha256 frozen in assemble_manifest.json",
            "the bundle ships the training set that was frozen, byte for byte",
        )
    try:
        rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    except ValueError as exc:
        raise _refuse(f"train_set.jsonl is not valid JSONL: {exc}") from exc
    if not rows or not all(isinstance(r, dict) and r.get("target_source") for r in rows):
        raise _refuse("train_set.jsonl is empty or has a row without a target_source")
    return blob, rows


def _sealed_ids(dataset_dir: Path | None) -> frozenset[str]:
    if dataset_dir is None:
        return frozenset()
    ids: set[str] = set()
    for _, name in dassemble.SEALED_SIDES:
        path = Path(dataset_dir) / name
        if path.is_file():
            ids.update(ddataset.load_sealed(path).ids)
    return frozenset(ids)


def _gate(
    prereg_path: Path, run_dir: Path, deviation_id: str | None, calibration_sha256: str
) -> dict[str, Any]:
    """The decision gate to ship: ``distill/gate/gate.json`` when the gate stage ran.

    A fitted gate must name the sha256 of the calibration the bundle ships (the one it was
    fitted on); any other is refused. Without the gate stage the bundle records that no gate
    was fitted: a pre-registered gate objective and grid is not a fitted gate.
    """
    reg = dprereg.require_locked(prereg_path, run_dir, deviation_id)
    del reg
    sha = dprereg.sha256_file(prereg_path)
    path = run_dir / dgate.GATE_REL / dgate.GATE_FILE
    if path.exists() or path.is_symlink():
        doc = _read_json(path, "fitted decision gate")
        fitted_on = doc.get("calibration_sha256")
        if fitted_on != calibration_sha256:
            raise _refuse(
                f"{path}: the gate was fitted on calibration {str(fitted_on)[:12]}, "
                f"but the bundle ships calibration {calibration_sha256[:12]}",
                "rerun the distill gate stage on the shipped calibration",
            )
        if not isinstance(doc.get("thresholds_by_kind"), dict) or doc.get("fitted") is not True:
            raise _refuse(f"{path}: not a fitted gate (no thresholds_by_kind)")
        return {
            **doc,
            "fitted": True,
            "source": f"{dgate.GATE_REL}/{dgate.GATE_FILE}",
            "gate_sha256": dprereg.sha256_file(path),
            "bundle_prereg_sha256": sha,
        }
    return {
        "fitted": False,
        "reason": "the distill gate stage has not run (no distill/gate/gate.json); none fitted",
        "thresholds_by_kind": {},
        "prereg_sha256": sha,
    }


def _artifact_path(run_dir: Path, recorded: Any) -> Path:
    if not recorded:
        raise _refuse("the quant manifest names no deployed artifact")
    path = Path(str(recorded))
    return path if path.is_absolute() else run_dir / path


# --------------------------------------------------------------------- the model bundle


def _config_with_calibration(config: Mapping[str, Any], calibration: Mapping[str, Any]):
    kinds = calibration.get("temperature_by_kind")
    if not isinstance(kinds, dict) or not kinds:
        raise _refuse("calibration.json has no fitted temperature_by_kind")
    if calibration.get("upstream_temperature_by_kind_used") is True:
        raise _refuse("calibration.json used upstream temperatures; refit on the fit fold")
    out = dict(config)
    out["temperature"] = IDENTITY_TEMPERATURE
    out["temperature_by_kind"] = {str(k): float(v) for k, v in sorted(kinds.items())}
    return out


def _copy_artifact(src: Path, out: Path) -> None:
    _no_symlinks(src)
    shutil.copytree(src, out, symlinks=False)


def _write_layout_manifest(out: Path) -> None:
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    names = [n for n in names if n != layout.MANIFEST]
    text = "".join(f"{layout.sha256_file(out / n)}  {n}\n" for n in names)
    (out / layout.MANIFEST).write_text(text)


def _stage_student(src: Path, out: Path, calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the verified student into *out*, install the fitted temperatures, re-verify."""
    if (src / quantized.MANIFEST_FILE).is_file():
        quantized.read_artifact(src)
        _copy_artifact(src, out)
        old = layout.sha256_file(out / layout.CONFIG)
        config = _config_with_calibration(
            json.loads((out / layout.CONFIG).read_text()), calibration
        )
        _write(out / layout.CONFIG, _dump(config))
        manifest = json.loads((out / quantized.MANIFEST_FILE).read_text())
        manifest["files"][layout.CONFIG] = layout.sha256_file(out / layout.CONFIG)
        manifest["config_rewritten_for_calibration"] = {"original_sha256": old}
        _write(out / quantized.MANIFEST_FILE, _dump(manifest))
        quantized.read_artifact(out)
        return {"layout": quantized.ARTIFACT_FORMAT, "bits": manifest["bits"]}
    layout.read_layout(src)
    _copy_artifact(src, out)
    old = layout.sha256_file(out / layout.CONFIG)
    config = _config_with_calibration(json.loads((out / layout.CONFIG).read_text()), calibration)
    _write(out / layout.CONFIG, _dump(config))
    prov = json.loads((out / layout.PROVENANCE).read_text())
    prov["strands_decider_config_sha256"] = layout.sha256_file(out / layout.CONFIG)
    prov["config_rewritten_for_calibration"] = {"original_sha256": old}
    _write(out / layout.PROVENANCE, _dump(prov))
    _write_layout_manifest(out)
    layout.read_layout(out)
    return {"layout": layout.FORMAT}


# --------------------------------------------------------------------- the dataset bundle


def _licence_ok(row: Mapping[str, Any]) -> bool:
    """Apache-2.0 and MIT rows pass; a cc-by-4.0 row passes only with a non-blank attribution."""
    licence = str(row.get("licence", "")).strip().lower()
    if licence in ATTRIBUTION_LICENCES:
        return bool(str(row.get("attribution") or "").strip())
    return licence in ALLOWED_LICENCES


def _index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in _INDEX_KEYS}


# --------------------------------------------------------------------- NOTICE


def _collect_attributions(node: Any, found: set[str]) -> None:
    """Every ``attribution`` string (or list of strings) at any depth of *node*."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("attribution", "attributions"):
                items = value if isinstance(value, list) else [value]
                found.update(i.strip() for i in items if isinstance(i, str) and i.strip())
            else:
                _collect_attributions(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_attributions(item, found)


def attribution_strings(manifest: Mapping[str, Any] | None, rows: Iterable[Mapping[str, Any]]):
    """Sorted, de-duplicated attribution strings from the dataset manifest's licence table
    (``licences``, key ``attribution``, a string or a list) and the rows' own field."""
    found: set[str] = set()
    for table in ("licences", "licence_table"):
        _collect_attributions((manifest or {}).get(table), found)
    _collect_attributions(list(rows), found)
    return sorted(found)


def notice_text(attributions: Iterable[str]) -> bytes:
    lines = ["NOTICE", "", "This bundle includes material under the attributions below.", ""]
    lines.extend(f"- {a}" for a in attributions)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _needs_notice(rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(str(r.get("licence", "")).strip().lower() in ATTRIBUTION_LICENCES for r in rows)


def _read_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not (path.is_file() or path.is_symlink()):
        return None
    return _read_json(path, "dataset manifest")


def check_notice(folder: Path) -> None:
    """Refuse a dataset bundle that holds a cc-by-4.0 record but no non-empty ``NOTICE``."""
    train = folder / TRAIN_FILE
    if not train.is_file():
        return
    rows = [json.loads(ln) for ln in train.read_text("utf-8").splitlines() if ln.strip()]
    if not _needs_notice(rows):
        return
    notice = folder / NOTICE_FILE
    if not notice.is_file() or not notice.read_text("utf-8").strip():
        raise _refuse(
            f"{folder}: a cc-by-4.0 record is present but there is no {NOTICE_FILE}",
            "write the attribution NOTICE from the dataset manifest (rebuild the bundle)",
        )


# --------------------------------------------------------------------- build


def _provenance(
    run_dir: Path,
    student: Mapping[str, Any],
    prereg_sha: str,
    extra: Mapping[str, Any],
    deviations: Iterable[str],
    layout_info: Mapping[str, Any],
) -> dict[str, Any]:
    audits = {}
    for role, rel in daudit.ROLES.items():
        path = run_dir / rel
        if path.is_file():
            audits[role] = {"path": rel, "sha256": _sha256_bytes(_read_bytes(path, "audit"))}
    assemble = _read_json(run_dir / dassemble.MANIFEST_REL, _ASSEMBLE_MANIFEST)
    manifest = student["manifest"] or {}
    prov = {
        "format": BUNDLE_FORMAT,
        "deployed_format": student["format"],
        "layout": dict(layout_info),
        "prereg_sha256": prereg_sha,
        "calibration_sha256": student["calibration_sha256"],
        "quant_manifest_prereg_sha256": manifest.get("prereg_sha256"),
        "teacher_build_ids": assemble.get("teacher_build_ids"),
        "audited_builds": assemble.get("audited_builds"),
        "audits": audits,
        "training_set_sha256": (assemble.get("output") or {}).get("sha256"),
        "deviations": sorted(set(deviations)),
        "pins": dict(extra.get("pins") or {}),
        "environments": dict(extra.get("environments") or {}),
        "held_out_published": False,
        "test_published": False,
    }
    if student.get("deployed_format") == ddeployed.BF16:  # additive: a quant bundle is unchanged
        prov[BUNDLE_KIND_KEY] = ddeployed.BF16
        prov["deployed_student"] = _bf16_origin(student)
    return prov


def _bf16_origin(student: Mapping[str, Any]) -> dict[str, Any]:
    """Where a bf16 bundle's student came from (the bf16 deployed contract, run-relative)."""
    winner = str(student["winner"])
    return {
        "winner": winner,
        "checkpoint": ddeployed.checkpoint_rel(winner),
        "calibration": student["calibration_file"],
        "predictions": ddeployed.predictions_rel(winner),
        "attestation": dict(student["bundle"]),
        "quantize": "skipped (deployed_format bf16)",
    }


def _require_fresh(*targets: Path) -> None:
    for target in targets:
        if target.exists() and any(target.iterdir()):
            raise _refuse(f"{target} exists and is not empty", "bundle into a fresh directory")


def _refuse_sealed_rows(rows: list[dict[str, Any]], dataset_dir: Path | str | None) -> None:
    sealed = _sealed_ids(Path(dataset_dir) if dataset_dir else None)
    leaked = sorted({r["record_id"] for r in rows if r.get("record_id") in sealed})
    if leaked:
        raise _refuse(
            f"training rows on a sealed side: {leaked[:5]}", "the sealed sets are never bundled"
        )


def _licensed_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """The rows the dataset bundle may carry, and the excluded rows' licence counts."""
    keep = [r for r in rows if _licence_ok(r)]
    if not keep:
        raise _refuse("no training row has an Apache-2.0 or MIT licence; no dataset to bundle")
    excluded = Counter(str(r.get("licence", "")) for r in rows if not _licence_ok(r))
    return keep, excluded


def _bundle_notice(
    keep: list[dict[str, Any]],
    dataset_dir: Path | str | None,
    dataset_manifest: Path | str | None,
) -> bytes:
    manifest_path = Path(dataset_manifest) if dataset_manifest else None
    if manifest_path is None and dataset_dir:
        manifest_path = Path(dataset_dir) / "manifest.json"
    attributions = attribution_strings(_read_manifest(manifest_path), keep)
    if _needs_notice(keep) and not attributions:
        raise _refuse(
            "a cc-by-4.0 record is in the dataset bundle but the dataset manifest holds no "
            "attribution string",
            "the dataset build must record an attribution per licence; no NOTICE, no bundle",
        )
    return notice_text(attributions)


def build_bundles(
    run_dir: Path | str,
    out_dir: Path | str,
    *,
    prereg_path: Path | str,
    dataset_dir: Path | str | None = None,
    deviation_id: str | None = None,
    deviations: Iterable[str] = (),
    provenance_extra: Mapping[str, Any] | None = None,
    dataset_manifest: Path | str | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/model`` and ``<out_dir>/dataset`` (scanned); return a summary."""
    run_dir, out_dir, prereg_path = Path(run_dir), Path(out_dir), Path(prereg_path)
    model_out, data_out = out_dir / MODEL_DIR, out_dir / DATASET_DIR
    _require_fresh(model_out, data_out)
    reg = dprereg.require_locked(prereg_path, run_dir, deviation_id)
    student = ddeployed.deployed_student(run_dir, reg)
    gate = _gate(prereg_path, run_dir, deviation_id, student["calibration_sha256"])
    blob, rows = _train_rows(run_dir)
    _refuse_sealed_rows(rows, dataset_dir)
    report = _read_bytes(run_dir / dfinal.FINAL_REL / dfinal.REPORT_FILE, "final report")
    assemble_bytes = _read_bytes(run_dir / dassemble.MANIFEST_REL, _ASSEMBLE_MANIFEST)
    calibration_bytes = _read_bytes(run_dir / student["calibration_file"], "calibration")
    calibration = json.loads(calibration_bytes)
    keep, excluded = _licensed_rows(rows)
    notice = _bundle_notice(keep, dataset_dir, dataset_manifest)

    src = _artifact_path(run_dir, student["artifact"])
    info = _stage_student(src, model_out, calibration)
    train_sha = _sha256_bytes(blob)
    prov = _provenance(
        run_dir, student, dprereg.sha256_file(prereg_path), provenance_extra or {}, deviations, info
    )
    identity = {"format": BUNDLE_FORMAT, "surface_sha256": train_sha, "surface_kind": SURFACE_KIND}
    index = {
        "file": dassemble.OUT_REL,
        "sha256": train_sha,
        "rows": len(rows),
        "target_sources": dict(sorted(Counter(r["target_source"] for r in rows).items())),
        "note": "index only: row text is in the dataset bundle (Apache-2.0/MIT rows)",
        "entries": [_index_row(r) for r in rows],
    }
    common = {
        CALIBRATION_FILE: calibration_bytes,
        GATE_FILE: _dump(gate),
        SCORER_TRAIN_FILE: _dump(index),
        BUNDLE_FILE: _dump({**identity, "kind": "model", "deployed_format": student["format"]}),
    }
    for name, data in {
        **common,
        PROVENANCE_FILE: _dump(prov),
        REPORT_FILE: report,
        MANIFEST_COPY: assemble_bytes,
    }.items():
        _write(model_out / name, data)
    _write(model_out / NOTICE_FILE, notice)

    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in keep)
    meta = {
        "licences_allowed": sorted(ALLOWED_LICENCES),
        "rows": len(keep),
        "rows_excluded_by_licence": dict(sorted(excluded.items())),
        "licences": dict(sorted(Counter(str(r["licence"]) for r in keep).items())),
        "target_sources": dict(sorted(Counter(r["target_source"] for r in keep).items())),
        "frozen_training_set_sha256": train_sha,
        "held_out_published": False,
    }
    for name, data in {
        **common,
        BUNDLE_FILE: _dump({**identity, "kind": "dataset"}),
        TRAIN_FILE: text.encode(),
        DATASET_MANIFEST: _dump(meta),
    }.items():
        _write(data_out / name, data)
    _write(data_out / NOTICE_FILE, notice)
    check_notice(data_out)
    for folder in (model_out, data_out):
        _no_symlinks(folder)
        scan.write_scan(folder)
    return {
        "model": str(model_out),
        "dataset": str(data_out),
        "deployed_format": student["format"],
        "layout": info,
        "gate_fitted": gate["fitted"],
        "rows": len(rows),
        "dataset_rows": len(keep),
        "rows_excluded_by_licence": meta["rows_excluded_by_licence"],
    }


# --------------------------------------------------------------------- pre-upload scans

Hit = tuple[str, int]  # (file relative to the bundle, 1-based line); never the content


def _config_path(config: Any, key: str) -> Path:
    value = None
    for source in (getattr(config, "values", None) or {}, getattr(config, "extra", None) or {}):
        value = value or source.get(key)
    if not value:
        raise _refuse(
            f"run config key {key!r} is not set; the pre-upload scan cannot run without it",
            f"set {key} to the operator-held path (never commit the file)",
        )
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise _refuse(f"run config key {key!r} names a file that does not exist")
    return path


def _text_files(folder: Path) -> list[tuple[str, str]]:
    out = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if rel == scan.SCAN_FILE:
            continue
        try:
            out.append((rel, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue  # weight binaries; hub.upload's own scan covers them
    return out


def _side_model_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name.lower())
    return names


def _side_model_hits(files: list[tuple[str, str]], names: list[str]) -> list[Hit]:
    hits = []
    for rel, text in files:
        for number, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if any(n in low for n in names):
                hits.append((rel, number))
    return hits


def _probe_keys(path: Path) -> tuple[set[str], set[str]]:
    """``(ids, state sha256s)`` of the Clef probe set (a JSON list/object or JSONL)."""
    raw = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(raw)
        records = doc if isinstance(doc, list) else doc.get("records", [doc])
    except ValueError:
        records = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    ids: set[str] = set()
    states: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ids.update(str(rec[k]) for k in ("id", "record_id") if rec.get(k))
        if isinstance(rec.get("state"), str) and rec["state"].strip():
            states.add(_sha256_bytes(rec["state"].encode("utf-8")))
    return ids, states


def _line_is_probe(line: str, ids: set[str], states: set[str]) -> bool:
    try:
        rec = json.loads(line)
    except ValueError:
        return False
    if not isinstance(rec, dict):
        return False
    if any(str(rec.get(k)) in ids for k in ("id", "record_id") if rec.get(k)):
        return True
    state = rec.get("state")
    return isinstance(state, str) and _sha256_bytes(state.encode("utf-8")) in states


def _probe_hits(files: list[tuple[str, str]], path: Path) -> list[Hit]:
    ids, states = _probe_keys(path)
    hits = []
    for rel, text in files:
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip() and _line_is_probe(line, ids, states):
                hits.append((rel, number))
    return hits


def _credential_hits(files: list[tuple[str, str]]) -> list[Hit]:
    hits = []
    for rel, text in files:
        found = (*scan.scan_credentials(rel, text), *scan.scan_endpoints(rel, text))
        hits.extend((rel, f.line) for f in found)
        hits.extend((rel, number) for number, _ in scan.private_hosts(text))
    return hits


def _script_flags(folder: Path, files: list[tuple[str, str]]) -> bool:
    """Whether scripts/scan-secrets.py exits non-zero on the bundle's text files (exit code
    only: its output carries matched values and is discarded)."""
    if not SECRETS_SCRIPT.is_file() or not files:
        return False
    argv = [sys.executable, str(SECRETS_SCRIPT), *(str(folder / rel) for rel, _ in files)]
    done = subprocess.run(  # nosec B603 -- fixed argv, repo script, no shell
        argv, capture_output=True, check=False, timeout=300
    )
    return done.returncode != 0


def pre_upload_scan(bundles: Mapping[str, Any], config: Any) -> None:
    """Refuse (file and line only) on a secret, hostname/private IP, side-model name or any
    record of the Clef probe set. The probe set and side-model list are read from the paths
    in run config (``distill_probe_set``, ``distill_side_model_list``); their content is
    never printed or stored."""
    probe_path = _config_path(config, PROBE_SET_KEY)
    names = _side_model_names(_config_path(config, SIDE_MODEL_LIST_KEY))
    lines: list[str] = []
    for key in ("model", "dataset"):
        folder = Path(bundles[key])
        files = _text_files(folder)
        found = {
            "secret/host": _credential_hits(files),
            "side-model name": _side_model_hits(files, names),
            "probe-set record": _probe_hits(files, probe_path),
        }
        for kind, hits in found.items():
            lines.extend(f"{kind}: {key}/{rel}:{n}" for rel, n in sorted(set(hits)))
        if _script_flags(folder, files) and not found["secret/host"]:
            lines.append(f"secret: {key}/ (scan-secrets.py hit; line unavailable)")
    if lines:
        raise _refuse(
            "pre-upload scan refused the bundles (file:line only):\n  " + "\n  ".join(lines),
            "remove the flagged lines and rebuild; nothing was sent",
        )


# --------------------------------------------------------------------- upload


def upload_bundles(
    bundles: Mapping[str, Any],
    *,
    model_repo: str,
    dataset_repo: str,
    config: Any,
    apply: bool = False,
    hub_client: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Upload both bundles through :func:`jev_factory.release.hub.upload`.

    Dry-run unless *apply*: nothing is sent. Repositories are created private, fetched back
    and compared by sha256; there is no public call."""
    check_notice(Path(bundles["dataset"]))
    pre_upload_scan(bundles, config)
    results = {}
    for key, repo, repo_type in (
        ("model", model_repo, "model"),
        ("dataset", dataset_repo, "dataset"),
    ):
        results[key] = hub.upload(
            bundle=Path(bundles[key]),
            repo=repo,
            repo_type=repo_type,
            config=config,
            apply=apply,
            hub=hub_client,
            environ=environ,
        )
    return {"applied": bool(apply), **results}
