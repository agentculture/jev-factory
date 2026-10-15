"""Distill final: score Clef, stock v21 and the deployed student once, on test and held-out.

The one place the sealed sides' text is read. One invocation scores **all three** models on
``test.jsonl`` and ``held_out.jsonl`` (the dataset's sealed sides), one model served at a time
(each :class:`~jev_factory.distill.benchmark.Target` is started, scored and stopped before the
next starts; one GPU job), under :class:`jev_factory.measure.once.OnceLedger`:

* the ledger is checked for both sides before anything starts, in a dry run too: a side
  already measured is refused without a deviation id, so a second final for **any** model
  (or for all of them) exits non-zero (``CliError``, exit 1);
* the ledger lock is held for the whole run, and one record per side is appended once
  something was scored (``measured``, or ``failed-mid-run`` when a later model failed); a run
  in which nothing was scored writes no record (nvsh P71), so it never blocks the real run;
* a partial invocation (not exactly Clef, stock v21 and the student) is refused: the three
  are measured together, never as separate measurements.

The sealed files are read as bytes and must hash to what :func:`dataset.load_sealed` reports.

Calibration at final, each recorded beside the row it produced: Clef raw (as served), plus a
``clef_calibrated`` row when a Clef calibration file is given; stock v21 as served (its
upstream temperatures); the student with ``distill/quant/calibration.json``, whose sha256 must
match the quant manifest. The deployed artifact is the one ``format_decision.json`` chose.

Outputs under ``<run>/distill/final/``: ``predictions/<model>.json`` (every raw answer with
its latency), ``final.json`` (sides, calibrations, runtime per model, ledger records) and
``report.json`` (:func:`jev_factory.distill.report.build_report`). Dry-run by default.
Stdlib only; servers are localhost only.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as dquant
from jev_factory.distill import report as dreport
from jev_factory.distill.records import Record
from jev_factory.distill.runspec import Bootstrap
from jev_factory.measure import once

STAGE = "distill-final"
LABEL = "distill-final"
FINAL_REL = "distill/final"
FINAL_FILE = "final.json"
REPORT_FILE = "report.json"
PREDICTIONS_DIR = "predictions"
MODELS = dreport.MODELS
#: dataset side file stem -> once-ledger side.
SIDES = {"test": once.TEST, "held_out": once.HELD_OUT}
STUDENT_CALIBRATION_REL = f"{dquant.QUANT_REL}/{dquant.CALIBRATION_FILE}"
CAL_RAW = "raw (as served)"
CAL_SERVED = "served (upstream temperatures, as reported)"


@dataclass(frozen=True)
class FinalModel:
    """One model of the final: how to serve it and the client that asks it."""

    target: bench.Target
    #: Anything with ``ask(record_mapping) -> CallResult`` (SystemOneClient, PointerClient).
    client: Any = None


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(
            f"{path}: cannot read the {what} ({type(exc).__name__})",
            "run the distill quant stage first; the student is measured as it was shipped",
        ) from None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- inputs


def deployed_student(run_dir: Path) -> dict[str, Any]:
    """The deployed format, its calibration and the quant manifest, cross-checked by sha256."""
    qdir = run_dir / dquant.QUANT_REL
    decision = _read_json(qdir / dquant.DECISION_FILE, "deployed format decision")
    manifest = _read_json(qdir / dquant.MANIFEST_FILE, "quant manifest")
    cal_path = qdir / dquant.CALIBRATION_FILE
    calibration = _read_json(cal_path, "student calibration")
    if manifest.get("status") != "done":
        raise _refuse(f"the quant manifest status is {manifest.get('status')!r}, not 'done'")
    if decision.get("format") != manifest.get("chosen_format"):
        raise _refuse("format_decision.json and quant_manifest.json disagree on the format")
    recorded = [
        sha
        for path, sha in (manifest.get("outputs") or {}).items()
        if Path(path).name == dquant.CALIBRATION_FILE
    ]
    actual = dp.sha256_file(cal_path)
    if recorded != [actual]:
        raise _refuse(
            f"{cal_path}: sha256 {actual[:12]} does not match the quant manifest {recorded}",
            "the student is measured with the calibration it was shipped with",
        )
    return {
        "format": decision.get("format"),
        "artifact": manifest.get("deployed_artifact"),
        "calibration": calibration,
        "calibration_sha256": actual,
        "manifest": manifest,
    }


def fitted_gate(run_dir: Path, calibration_sha256: str) -> dict[str, Any] | None:
    """The fitted ``distill/gate/gate.json`` the report gates with, or None if none was fitted.

    A gate fitted on any calibration other than the deployed student's (by sha256), or a file
    that is not a fitted gate, is refused: the report would gate with the wrong thresholds.
    """
    path = run_dir / dgate.GATE_REL / dgate.GATE_FILE
    if not (path.exists() or path.is_symlink()):
        return None
    doc = _read_json(path, "fitted decision gate")
    fitted_on = doc.get("calibration_sha256")
    if fitted_on != calibration_sha256:
        raise _refuse(
            f"{path}: the gate was fitted on calibration {str(fitted_on)[:12]}, but the "
            f"deployed student ships calibration {calibration_sha256[:12]}",
            "rerun the distill gate stage on the deployed calibration",
        )
    if doc.get("fitted") is not True or not isinstance(doc.get("thresholds_by_kind"), dict):
        raise _refuse(f"{path}: not a fitted gate (no thresholds_by_kind)")
    return {
        "file": f"{dgate.GATE_REL}/{dgate.GATE_FILE}",
        "sha256": dp.sha256_file(path),
        "doc": doc,
    }


def _check_models(models: Mapping[str, FinalModel]) -> None:
    missing = [m for m in MODELS if m not in models]
    extra = sorted(set(models) - set(MODELS))
    if missing or extra:
        raise _refuse(
            f"the final scores exactly {list(MODELS)} in one invocation;"
            f" missing {missing}, unexpected {extra}",
            "test and held-out are measured once: pass all three models together",
        )


def _check_ledger(ledger: once.OnceLedger, deviation: str | None) -> dict[str, int]:
    earlier = {}
    for side in SIDES.values():
        try:
            earlier[side] = ledger.check(side, deviation)
        except once.OnceError as exc:
            raise _refuse(
                str(exc), "record a deviation (/deviate) and pass its id (--deviation <id>)"
            ) from None
    return earlier


def _read_side(sealed: dataset.SealedSide) -> list[Record]:
    data = Path(sealed.path).read_bytes()
    if dataset.sha256_bytes(data) != sealed.sha256:
        raise _refuse(f"{sealed.path}: changed since it was sealed (sha256 mismatch)")
    lines = data.decode("utf-8").splitlines()
    return [Record.from_json(json.loads(line)) for line in lines if line.strip()]


class _Counted:
    """Wraps a client and counts the questions put to the server (did we measure anything?)."""

    def __init__(self, client: Any) -> None:
        self.client, self.asked = client, 0

    def ask(self, record: Mapping[str, Any]) -> Any:
        self.asked += 1
        return self.client.ask(record)


# --------------------------------------------------------------------- the run


def run_final(
    *,
    prereg_path: Path | str,
    run_dir: Path | str,
    dataset_dir: Path | str,
    models: Mapping[str, FinalModel],
    apply: bool = False,
    deviation_id: str | None = None,
    clef_calibration: Path | str | None = None,
    memory_probe: bench.MemoryProbe | None = bench.nvidia_smi_used_mib,
    proc_root: Path | str = bench.PROC_ROOT,
    host: str | None = None,
    now: Callable[[], str] = _now,
    contamination_manifest: Mapping[str, Any] | None = None,
    bootstrap: Bootstrap = Bootstrap(),
) -> dict[str, Any]:
    """Plan (dry-run) or run the single final measurement and write the report."""
    seed, resamples = bootstrap.seed, bootstrap.resamples
    prereg_path, run_dir, dataset_dir = Path(prereg_path), Path(run_dir), Path(dataset_dir)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    try:
        deviation = once.check_deviation_id(deviation_id)
    except ValueError as exc:
        raise _refuse(str(exc)) from None
    _check_models(models)
    sealed = {}
    for stem in SIDES:
        path = dataset_dir / f"{stem}.jsonl"
        if not path.is_file():
            raise _refuse(
                f"{path}: no sealed side", "point at a directory written by write_dataset"
            )
        sealed[stem] = dataset.load_sealed(path)
    student = deployed_student(run_dir)
    gate = fitted_gate(run_dir, student["calibration_sha256"])
    calibrations: dict[str, dict[str, Any]] = {
        dreport.CLEF: {"source": CAL_RAW, "sha256": None, "calibration": None},
        dreport.V21: {"source": CAL_SERVED, "sha256": None, "calibration": None},
        dreport.STUDENT: {
            "source": STUDENT_CALIBRATION_REL,
            "sha256": student["calibration_sha256"],
            "calibration": student["calibration"],
        },
    }
    if clef_calibration is not None:
        cpath = Path(clef_calibration)
        calibrations[dreport.CLEF_CALIBRATED] = {
            "source": str(cpath),
            "sha256": dp.sha256_file(cpath) if cpath.is_file() else None,
            "calibration": _read_json(cpath, "Clef calibration"),
        }
    ledger = once.OnceLedger(run_dir)
    earlier = _check_ledger(ledger, deviation)
    base = {
        "stage": STAGE,
        "prereg_sha256": dp.sha256_file(prereg_path),
        "deviation": deviation,
        "models": {
            n: {"served_model": models[n].target.model, "base_url": models[n].target.base_url}
            for n in MODELS
        },
        "sides": {s: {"count": v.count, "sha256": v.sha256} for s, v in sealed.items()},
        "earlier_measurements": earlier,
        "deployed": {"format": student["format"], "artifact": student["artifact"]},
        "calibrations": calibrations,
        "gate": None if gate is None else {k: gate[k] for k in ("file", "sha256")},
    }
    if not apply:
        return {**base, "dry_run": True}
    with ledger:
        _check_ledger(ledger, deviation)  # again, under the lock
        return _measure(
            base=base,
            ledger=ledger,
            sealed=sealed,
            models=models,
            run_dir=run_dir,
            memory_probe=memory_probe,
            proc_root=proc_root,
            host=host or platform.node(),
            now=now,
            report_inputs=_ReportInputs(
                prereg=reg,
                quant_manifest=student["manifest"],
                contamination_manifest=contamination_manifest,
                seed=seed,
                resamples=resamples,
                gate=None if gate is None else gate["doc"],
            ),
        )


@dataclass(frozen=True)
class _ReportInputs:
    """What the final report needs beyond the measurement itself."""

    prereg: dp.DistillPrereg
    quant_manifest: Mapping[str, Any]
    contamination_manifest: Mapping[str, Any] | None
    seed: int
    resamples: int
    gate: Mapping[str, Any] | None = None


def _measure(
    *,
    base: dict[str, Any],
    ledger: once.OnceLedger,
    sealed: Mapping[str, dataset.SealedSide],
    models: Mapping[str, FinalModel],
    run_dir: Path,
    memory_probe: bench.MemoryProbe | None,
    proc_root: Path | str,
    host: str,
    now: Callable[[], str],
    report_inputs: _ReportInputs,
) -> dict[str, Any]:
    records = {stem: _read_side(side) for stem, side in sealed.items()}
    started = now()
    counted: list[_Counted] = []
    results: dict[str, dict[str, Any]] = {}

    def record(status: str) -> list[dict[str, Any]]:
        written = []
        for stem, side in SIDES.items():
            entry = once.OnceRecord(
                side=side,
                split_sha256=sealed[stem].sha256,
                label=LABEL,
                date=started,
                status=status,
                deviation=base["deviation"],
            )
            ledger.append(entry)
            written.append(entry.to_dict())
        return written

    try:
        for name in MODELS:  # sequential: one served model at a time
            model = models[name]
            client = _Counted(
                model.client
                or bench.SystemOneClient(model.target.base_url, model=model.target.model)
            )
            counted.append(client)
            results[name] = bench.benchmark_model(
                model.target,
                records,
                memory_probe=memory_probe,
                client=client,
                proc_root=proc_root,
            )
    except BaseException:
        if any(c.asked for c in counted):
            record(once.FAILED_MID_RUN)
        raise
    ledger_records = record(once.MEASURED)
    out_dir = run_dir / FINAL_REL
    for name, result in results.items():
        _write_json(out_dir / PREDICTIONS_DIR / f"{name}.json", result["predictions"])
    runtime = {
        name: {
            "host": host,
            "n_records": r["n_records"],
            "n_failed": r["n_failed"],
            "latency": r["latency"],
            "memory": r["memory"],
        }
        for name, r in results.items()
    }
    final = {
        **base,
        "dry_run": False,
        "date": started,
        "runtime": runtime,
        "ledger": ledger_records,
    }
    report = dreport.build_report(
        records_by_side=records,
        predictions={name: r["predictions"] for name, r in results.items()},
        calibrations={k: v["calibration"] for k, v in base["calibrations"].items()},
        calibration_sources={k: v["source"] for k, v in base["calibrations"].items()},
        runtime=runtime,
        prereg=report_inputs.prereg,
        quant_manifest=report_inputs.quant_manifest,
        contamination_manifest=report_inputs.contamination_manifest,
        seed=report_inputs.seed,
        resamples=report_inputs.resamples,
        gate=report_inputs.gate,
    )
    _write_json(out_dir / FINAL_FILE, final)
    _write_json(out_dir / REPORT_FILE, report)
    return {"final": final, "report": report}
