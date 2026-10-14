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

``dataset/``
    ``train.jsonl``: the frozen training rows, **only those whose source licence is
    Apache-2.0 or MIT**, each row keeping its ``licence``, ``source`` and ``target_source``;
    the rest are counted by licence in ``dataset.json`` and left out.

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
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jev_factory.backbones.pointer import layout, quantized
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import assemble as dassemble
from jev_factory.distill import audit as daudit
from jev_factory.distill import dataset as ddataset
from jev_factory.distill import final as dfinal
from jev_factory.distill import prereg as dprereg
from jev_factory.distill import quant as dquant
from jev_factory.release import hub, scan

MODEL_DIR = "model"
DATASET_DIR = "dataset"
ALLOWED_LICENCES = frozenset({"apache-2.0", "mit"})
BUNDLE_FILE = "bundle.json"
CALIBRATION_FILE = "calibration.json"
GATE_FILE = "gate.json"
SCORER_TRAIN_FILE = "scorer-train.json"
PROVENANCE_FILE = "distill_provenance.json"
REPORT_FILE = "report.json"
MANIFEST_COPY = "assemble_manifest.json"
TRAIN_FILE = "train.jsonl"
DATASET_MANIFEST = "dataset.json"
BUNDLE_FORMAT = "jev-distill-bundle/1"
SURFACE_KIND = "frozen train set (sha256 of distill/assemble/train_set.jsonl)"
IDENTITY_TEMPERATURE = 1.0
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
    manifest = _read_json(run_dir / dassemble.MANIFEST_REL, "assemble manifest")
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


def _gate(prereg_path: Path, run_dir: Path, deviation_id: str | None) -> dict[str, Any]:
    reg = dprereg.require_locked(prereg_path, run_dir, deviation_id)
    del reg
    doc = json.loads(prereg_path.read_text(encoding="utf-8"))
    sha = dprereg.sha256_file(prereg_path)
    by_kind = doc.get("gate")
    if isinstance(by_kind, dict) and by_kind:
        return {"fitted": True, "by_kind": by_kind, "source": "locked prereg", "prereg_sha256": sha}
    return {
        "fitted": False,
        "reason": "the locked pre-registration registers no gate thresholds; none was fitted",
        "by_kind": {},
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
    return str(row.get("licence", "")).strip().lower() in ALLOWED_LICENCES


def _index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in _INDEX_KEYS}


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
    assemble = _read_json(run_dir / dassemble.MANIFEST_REL, "assemble manifest")
    manifest = student["manifest"]
    return {
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


def build_bundles(
    run_dir: Path | str,
    out_dir: Path | str,
    *,
    prereg_path: Path | str,
    dataset_dir: Path | str | None = None,
    deviation_id: str | None = None,
    deviations: Iterable[str] = (),
    provenance_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/model`` and ``<out_dir>/dataset`` (scanned); return a summary."""
    run_dir, out_dir, prereg_path = Path(run_dir), Path(out_dir), Path(prereg_path)
    model_out, data_out = out_dir / MODEL_DIR, out_dir / DATASET_DIR
    for target in (model_out, data_out):
        if target.exists() and any(target.iterdir()):
            raise _refuse(f"{target} exists and is not empty", "bundle into a fresh directory")
    gate = _gate(prereg_path, run_dir, deviation_id)
    student = dfinal.deployed_student(run_dir)
    blob, rows = _train_rows(run_dir)
    sealed = _sealed_ids(Path(dataset_dir) if dataset_dir else None)
    leaked = sorted({r["record_id"] for r in rows if r.get("record_id") in sealed})
    if leaked:
        raise _refuse(
            f"training rows on a sealed side: {leaked[:5]}", "the sealed sets are never bundled"
        )
    report = _read_bytes(run_dir / dfinal.FINAL_REL / dfinal.REPORT_FILE, "final report")
    assemble_bytes = _read_bytes(run_dir / dassemble.MANIFEST_REL, "assemble manifest")
    calibration_bytes = _read_bytes(
        run_dir / dquant.QUANT_REL / dquant.CALIBRATION_FILE, "calibration"
    )
    calibration = json.loads(calibration_bytes)
    keep = [r for r in rows if _licence_ok(r)]
    if not keep:
        raise _refuse("no training row has an Apache-2.0 or MIT licence; no dataset to bundle")
    excluded = Counter(str(r.get("licence", "")) for r in rows if not _licence_ok(r))

    src = _artifact_path(run_dir, student["artifact"])
    info = _stage_student(src, model_out, calibration)
    train_sha = _sha256_bytes(blob)
    prov = _provenance(
        run_dir, student, gate["prereg_sha256"], provenance_extra or {}, deviations, info
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
