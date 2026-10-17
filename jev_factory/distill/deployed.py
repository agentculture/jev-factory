"""The deployed student, resolved once for the gate, bundle and edge stages.

Under ``deployed_format quant`` (every v2 pre-registration) the deployed student is the quant
stage's shipped build, exactly as :func:`jev_factory.distill.final.deployed_student` reads and
cross-checks it; a quant stage that recorded no qualifying build is refused there, and there
is **no bf16 fallback** (risk r10).

Under ``deployed_format bf16`` the quantize and heal stages are skipped and the deployed
student is the train stage's selected candidate (the bf16 deployed contract):

* the candidate is ``distill/select/summary.json`` field ``winner``;
* its checkpoint is ``distill/train/<winner>/bundle`` (``hobson-hf-export/1``, the layout
  ``strands-decider serve`` loads; verified with
  :func:`jev_factory.backbones.pointer.layout.read_layout`);
* its calibration is ``distill/select/calibration.json`` (the fit-fold calibration the select
  step wrote for the winner);
* its fit- and selection-fold predictions are ``distill/train/<winner>/predictions.json``
  (what the served bundle answered when the candidate was scored);
* its attestation is the per-file sha256 of the bundle folder
  (:func:`jev_factory.release.hub.digests`), with no manifest sha256.

Both shapes return the keys the stages read (``format``, ``artifact``, ``calibration``,
``calibration_sha256``, ``calibration_file``, ``manifest``); the bf16 shape adds
``deployed_format``, ``winner``, ``predictions_file`` and ``bundle``. Stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_factory.backbones.pointer import layout
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as dtrain
from jev_factory.release import hub

BF16 = "bf16"
QUANT = "quant"
BUNDLE_DIR = "bundle"
SUMMARY_REL = f"{dtrain.SELECT_REL}/{dtrain.SUMMARY_FILE}"
CALIBRATION_REL = f"{dtrain.SELECT_REL}/{dtrain.CALIBRATION_FILE}"


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR, message, remediation or "run the distill train/select stage first"
    )


def _read_json(path: Path, what: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise _refuse(f"{path}: no {what} (or it is a symlink)")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"{path}: cannot read the {what} ({type(exc).__name__})") from None


def checkpoint_rel(winner: str) -> str:
    return f"{dtrain.TRAIN_REL}/{winner}/{BUNDLE_DIR}"


def predictions_rel(winner: str) -> str:
    return f"{dtrain.TRAIN_REL}/{winner}/{dtrain.PREDICTIONS_FILE}"


def attestation(bundle: Path | str) -> dict[str, Any]:
    """The bf16 build's attestation: per-file sha256 of the bundle folder, no manifest sha."""
    return {"files": hub.digests(Path(bundle)), "manifest_sha256": None}


def bf16_student(run_dir: Path | str) -> dict[str, Any]:
    """The bf16 deployed student per the contract above, verified; refused when incomplete."""
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / SUMMARY_REL, "train/select summary")
    winner = summary.get("winner") if isinstance(summary, dict) else None
    if not winner:
        raise _refuse(f"{run_dir / SUMMARY_REL} names no winner; there is no bf16 build to ship")
    bundle = run_dir / checkpoint_rel(str(winner))
    if bundle.is_symlink() or not bundle.is_dir():
        raise _refuse(f"{bundle}: the winner's hobson-hf-export/1 bundle does not exist")
    layout.read_layout(bundle)  # refuses a missing file or a hash mismatch
    cal_path = run_dir / CALIBRATION_REL
    calibration = _read_json(cal_path, "selected candidate's calibration")
    if not isinstance(calibration, dict):
        raise _refuse(f"{cal_path}: not a calibration object")
    named = calibration.get("candidate")
    if named is not None and named != winner:
        raise _refuse(
            f"{cal_path}: the calibration is candidate {named}'s, but the winner is {winner}",
            "rerun the distill train/select stage; the bf16 build ships its own calibration",
        )
    predictions = run_dir / predictions_rel(str(winner))
    if predictions.is_symlink() or not predictions.is_file():
        raise _refuse(f"{predictions}: no fit/selection predictions of the winner's bundle")
    return {
        "format": BF16,
        "deployed_format": BF16,
        "winner": str(winner),
        "artifact": bundle,
        "calibration": calibration,
        "calibration_sha256": dp.sha256_file(cal_path),
        "calibration_file": CALIBRATION_REL,
        "predictions_file": predictions,
        "bundle": attestation(bundle),
        "manifest": None,
    }


def read_predictions(student: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """``{fold: rows}`` of a bf16 student's ``predictions.json``."""
    path = Path(student["predictions_file"])
    doc = _read_json(path, "winner's predictions")
    if not isinstance(doc, dict):
        raise _refuse(f"{path}: expected an object of fold -> rows")
    return {str(fold): list(rows or []) for fold, rows in doc.items()}


def deployed_student(run_dir: Path | str, reg: dp.DistillPrereg) -> dict[str, Any]:
    """The deployed student under the registered ``deployed_format`` (quant: no bf16 fallback)."""
    if dp.deployed_format(reg) == BF16:
        return bf16_student(run_dir)
    from jev_factory.distill import final as dfinal  # lazy: final -> gate/edge -> here

    student = dfinal.deployed_student(Path(run_dir))
    return {**student, "calibration_file": dfinal.STUDENT_CALIBRATION_REL}
