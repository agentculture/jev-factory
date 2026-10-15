"""Distill final: Clef, stock, the bf16 student and the shipped quant, once, on the sealed sides.

The one place the sealed sides' text is read. One invocation scores **all four** models
(:data:`MODELS`: Clef W4A16, the stock model, the distilled bf16 student and the shipped quant)
on ``test.jsonl`` and ``held_out.jsonl`` (the dataset's sealed sides), one model at a time (each
:class:`~jev_factory.distill.benchmark.Target` is started, scored and stopped before the next
starts; a target with no ``start`` is **pre-served**, e.g. the quant on the AGX Orin behind an
SSH tunnel at ``http://127.0.0.1:<port>``), under :class:`jev_factory.measure.once.OnceLedger`:

* the ledger is checked for both sides before anything starts, in a dry run too: a side
  already measured is refused without a deviation id, so a second final for **any** model
  (or for all of them) exits non-zero (``CliError``, exit 1);
* the ledger lock is held for the whole run, and one record per side is appended once
  something was scored (``measured``, or ``failed-mid-run`` when a later model failed); a run
  in which nothing was scored writes no record (nvsh P71), so it never blocks the real run;
* a partial invocation (not exactly the four models) is refused: they are measured together,
  never as separate measurements.

Refusals before any request (dry run and apply): the run's teacher-label lock exists
(:func:`jev_factory.distill.edge.require_no_teacher_lock`; the box is busy labelling); a test
record has no ``approve`` entry for its current sha256 in the gold ledger
(:func:`jev_factory.review.gold.approved_ids`; the ids are printed, nothing else); the quant
manifest records no shipped format (a failed ``no qualifying format`` result: there is no bf16
fallback, risk r10, and bundle refuses it too). In the unverified-gold proving mode
(``FinalInputs.unverified_gold``, deviation d10) the ledger is not read, a deviation id is
required, and the report carries an ``unverified_c17`` block marked UNVERIFIED. Before any
request (apply): the shipped student's ``GET /v1/bundle`` (localhost only) differs from the
deployed artifact's attestation (:func:`jev_factory.backbones.pointer.quantized.bundle_attestation`:
per-file sha256 and manifest sha256), or its ``GET /v1/stats`` reports no cgroup memory cap. The
bundle is fetched again after the student is scored; a change fails the run. ``final.json``
records ``deployed.bundle`` (``files``, ``manifest_sha256``, ``weight_bytes``: what the edge
stage checks) and ``student_server`` (the attested bundle, the board, the memory cap that
``/v1/stats`` reported).

The sealed files are read as bytes and must hash to what :func:`dataset.load_sealed` reports.
An optional Clef **probe set** (private) is scored by every model in the same run; its
predictions are written with hashed record ids and it gates nothing.

Calibration at final, each recorded beside the row it produced: Clef raw (as served), plus a
``clef_calibrated`` row when a Clef calibration file is given; the stock model as served (its
upstream temperatures); the bf16 student raw unless a calibration file is given; the shipped
student with ``distill/quant/calibration.json``, whose sha256 must match the quant manifest.
The deployed artifact is the one ``format_decision.json`` chose.

Outputs under ``<run>/distill/final/``: ``predictions/<model>.json`` (every raw answer with
its latency), ``predictions/probe/<model>.json`` (probe ids hashed), ``final.json`` (sides,
calibrations, bundle, server, runtime per model, ledger records) and ``report.json``
(:func:`jev_factory.distill.report.build_report`). Dry-run by default. Stdlib only; servers
are localhost only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_factory.backbones.pointer import quantized as qz
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import edge as dedge
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as dquant
from jev_factory.distill import report as dreport
from jev_factory.distill.records import Record
from jev_factory.distill.runspec import Bootstrap
from jev_factory.measure import once
from jev_factory.measure.preflight import require_localhost
from jev_factory.review import gold

STAGE = "distill-final"
LABEL = "distill-final"
FINAL_REL = "distill/final"
FINAL_FILE = "final.json"
REPORT_FILE = "report.json"
PREDICTIONS_DIR = "predictions"
PROBE_DIR = "probe"
PROBE_FOLD = "probe"
MODELS = dreport.MODELS
#: The model whose served bundle is attested: the shipped quant.
ATTESTED = dreport.STUDENT
#: dataset side file stem -> once-ledger side.
SIDES = {"test": once.TEST, "held_out": once.HELD_OUT}
STUDENT_CALIBRATION_REL = f"{dquant.QUANT_REL}/{dquant.CALIBRATION_FILE}"
CAL_RAW = "raw (as served)"
CAL_SERVED = "served (upstream temperatures, as reported)"


@dataclass(frozen=True)
class FinalModel:
    """One model of the final: how to serve it and the client that asks it.

    A target without ``start``/``stop`` is pre-served (e.g. a tunnel to the AGX Orin).
    """

    target: bench.Target
    #: Anything with ``ask(record_mapping) -> CallResult`` (SystemOneClient, PointerClient).
    client: Any = None
    #: ``get(path) -> JSON object`` on the served model (``/v1/bundle``, ``/v1/stats``);
    #: by default an HTTP GET on the target's localhost ``base_url``.
    get: Callable[[str], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class FinalInputs:
    """Optional inputs of the final beyond the sealed sides and the models."""

    #: The human-gold approval ledger; default ``<dataset>/test.gold-ledger.jsonl``.
    gold_ledger: Path | str | None = None
    #: The (private) Clef probe set: scored by every model, reported, gates nothing.
    probe: Path | str | None = None
    #: A calibration for the bf16 student row; without one it is scored raw (as served).
    bf16_calibration: Path | str | None = None
    contamination_manifest: Mapping[str, Any] | None = None
    #: Proving mode (deviation d10): measure on the model-drafted, never-approved gold. Needs a
    #: deviation id; the gold ledger is not read and the report marks every result UNVERIFIED.
    unverified_gold: bool = False


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
    manifest = _read_json(qdir / dquant.MANIFEST_FILE, "quant manifest")
    if manifest.get("shippable") is False or manifest.get("status") == "failed":
        raise _refuse(
            f"the quant manifest records no shipped format: {manifest.get('reason') or ''}",
            "no quantized build qualified (risk r10: there is no bf16 fallback); a human "
            "decides the next step and records it as a deviation",
        )
    if manifest.get("status") != "done":
        raise _refuse(f"the quant manifest status is {manifest.get('status')!r}, not 'done'")
    shipped = manifest.get("shipped_format", manifest.get("chosen_format"))
    if not shipped or shipped != manifest.get("chosen_format"):
        raise _refuse("the quant manifest records no shipped format; nothing was shipped")
    decision = _read_json(qdir / dquant.DECISION_FILE, "deployed format decision")
    cal_path = qdir / dquant.CALIBRATION_FILE
    calibration = _read_json(cal_path, "student calibration")
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


def deployed_bundle(run_dir: Path, artifact: Any) -> dict[str, Any]:
    """The attestation of the deployed artifact on disk: what ``GET /v1/bundle`` must answer."""
    path = Path(str(artifact or ""))
    if not artifact:
        raise _refuse("the quant manifest names no deployed_artifact; cannot attest the bundle")
    if not path.is_absolute():
        path = run_dir / path
    try:
        att = qz.bundle_attestation(path)
    except OSError as exc:
        raise _refuse(
            f"{path}: cannot attest the deployed artifact ({type(exc).__name__})",
            "the final measures the shipped quantized bundle; run the distill quant stage",
        ) from None
    return {
        "files": dict(att["files"]),
        "manifest_sha256": att["manifest_sha256"],
        "weight_bytes": att.get("weight_bytes"),
        "format": att.get("format"),
    }


def _ledger_file(dataset_dir: Path, inputs: FinalInputs) -> Path:
    if inputs.gold_ledger is not None:
        return Path(inputs.gold_ledger)
    return gold.ledger_path(dataset_dir / f"{dataset.TEST_SPLIT}.jsonl")


def approved_records(
    ledger_file: Path, test: Sequence[Record], held_out: Sequence[Record]
) -> dict[str, Any]:
    """Refuse any test record without an approve entry; return the approved ids (all sides).

    The refusal prints record ids only, never record text. A malformed ledger fails closed.
    """
    try:
        entries = gold.read_ledger(ledger_file)
        approved = gold.approved_ids(entries, [*test, *held_out])
    except gold.GoldLedgerError as exc:
        raise _refuse(
            f"the gold ledger is not usable: {exc}",
            "fix or restore the ledger written by `jev review --gold`; nothing is guessed",
        ) from None
    missing = [r.id for r in test if r.id not in approved]
    if missing:
        raise _refuse(
            f"{len(missing)} test record(s) have no approve entry in the gold ledger "
            f"{ledger_file.name}: {', '.join(missing)}",
            "approve (or correct) each with `jev review --gold FILE --apply` before the final",
        )
    return {
        "file": str(ledger_file),
        "sha256": dp.sha256_file(ledger_file),
        "approved": approved,
        "approved_test": len(test),
        "approved_held_out": sum(1 for r in held_out if r.id in approved),
    }


def unverified_records(
    deviation_id: str | None, test: Sequence[Record], held_out: Sequence[Record]
) -> dict[str, Any]:
    """The approval block of the unverified-gold proving mode: nothing is approved.

    Refused without a deviation id: measuring on gold no human approved is an out-of-plan run.
    """
    if not deviation_id:
        raise _refuse(
            "the unverified-gold final needs a recorded deviation id",
            "record the proving run with /deviate and pass --deviation <id>",
        )
    return {
        "file": None,
        "verification": "unverified",
        "approved": frozenset(),
        "unverified_test": len(test),
        "unverified_held_out": len(held_out),
        "deviation": deviation_id,
    }


def load_probe(path: Path | str | None, sealed: Mapping[str, dataset.SealedSide]) -> list[Record]:
    """The private probe set (ids must not touch the sealed sides); empty when not given."""
    if path is None:
        return []
    records = dedge._load_probe(Path(path))  # noqa: SLF001 - one probe-set reader
    blocked = set().union(*(side.ids for side in sealed.values()))
    clash = sum(1 for r in records if r.id in blocked)
    if clash:
        raise _refuse(f"{clash} probe record(s) share an id with the sealed sides")
    return records


def _getter(model: FinalModel) -> Callable[[str], Mapping[str, Any]]:
    if model.get is not None:
        return model.get
    base_url = model.target.base_url
    return lambda path: dedge._get_json(base_url, path)  # noqa: SLF001 - localhost only


def attest_student(model: FinalModel, expected: Mapping[str, Any]) -> dict[str, Any]:
    """``GET /v1/bundle`` must equal the deployed bundle and ``/v1/stats`` must report a cap.

    Called before any request is sent to any model; a mismatch or no cap is refused.
    """
    try:
        require_localhost(model.target.base_url)
    except ValueError as exc:
        raise _refuse(
            str(exc), "open an SSH tunnel to the board and pass its http://127.0.0.1:<port> URL"
        ) from None
    get = _getter(model)
    served = get(dedge.BUNDLE_PATH)
    dedge.check_bundle(served, expected)
    stats = get(dedge.STATS_PATH)
    cap = dedge.memory_cap(stats)
    return {
        "base_url": model.target.base_url,
        "bundle": {
            "files": dict(served.get("files") or {}),
            "manifest_sha256": served.get("manifest_sha256"),
            "weight_bytes": served.get("weight_bytes"),
        },
        "board": {
            "device_model": stats.get("device_model"),
            "nv_tegra_release": stats.get("nv_tegra_release"),
        },
        "memory_cap": cap,
        "stats_at_attestation": {
            "rss_bytes": stats.get("rss_bytes"),
            "mem_available_bytes": stats.get("mem_available_bytes"),
        },
    }


def _check_models(models: Mapping[str, FinalModel]) -> None:
    missing = [m for m in MODELS if m not in models]
    extra = sorted(set(models) - set(MODELS))
    if missing or extra:
        raise _refuse(
            f"the final scores exactly {list(MODELS)} in one invocation;"
            f" missing {missing}, unexpected {extra}",
            "test and held-out are measured once: pass every model together",
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


def _calibrations(
    student: Mapping[str, Any], clef_calibration: Path | str | None, inputs: FinalInputs
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        dreport.CLEF: {"source": CAL_RAW, "sha256": None, "calibration": None},
        dreport.STOCK: {"source": CAL_SERVED, "sha256": None, "calibration": None},
        dreport.STUDENT_BF16: {"source": CAL_RAW, "sha256": None, "calibration": None},
        dreport.STUDENT: {
            "source": STUDENT_CALIBRATION_REL,
            "sha256": student["calibration_sha256"],
            "calibration": student["calibration"],
        },
    }
    for name, path, what in (
        (dreport.CLEF_CALIBRATED, clef_calibration, "Clef calibration"),
        (dreport.STUDENT_BF16, inputs.bf16_calibration, "bf16 student calibration"),
    ):
        if path is not None:
            cpath = Path(path)
            out[name] = {
                "source": str(cpath),
                "sha256": dp.sha256_file(cpath) if cpath.is_file() else None,
                "calibration": _read_json(cpath, what),
            }
    return out


def _sealed_sides(dataset_dir: Path) -> dict[str, dataset.SealedSide]:
    sealed = {}
    for stem in SIDES:
        path = dataset_dir / f"{stem}.jsonl"
        if not path.is_file():
            raise _refuse(
                f"{path}: no sealed side", "point at a directory written by write_dataset"
            )
        sealed[stem] = dataset.load_sealed(path)
    return sealed


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
    inputs: FinalInputs = FinalInputs(),
    bootstrap: Bootstrap = Bootstrap(),
) -> dict[str, Any]:
    """Plan (dry-run) or run the single final measurement and write the report."""
    prereg_path, run_dir, dataset_dir = Path(prereg_path), Path(run_dir), Path(dataset_dir)
    dedge.require_no_teacher_lock(run_dir)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    try:
        deviation = once.check_deviation_id(deviation_id)
    except ValueError as exc:
        raise _refuse(str(exc)) from None
    _check_models(models)
    sealed = _sealed_sides(dataset_dir)
    records = {stem: _read_side(side) for stem, side in sealed.items()}
    if inputs.unverified_gold:
        approval = unverified_records(deviation_id, records["test"], records["held_out"])
    else:
        approval = approved_records(
            _ledger_file(dataset_dir, inputs), records["test"], records["held_out"]
        )
    probe = load_probe(inputs.probe, sealed)
    student = deployed_student(run_dir)
    bundle = deployed_bundle(run_dir, student["artifact"])
    gate = fitted_gate(run_dir, student["calibration_sha256"])
    ledger = once.OnceLedger(run_dir)
    earlier = _check_ledger(ledger, deviation)
    prereg_sha256 = dp.sha256_file(prereg_path)
    base = {
        "stage": STAGE,
        "prereg_sha256": prereg_sha256,
        "deviation": deviation,
        "models": {
            n: {
                "served_model": models[n].target.model,
                "base_url": models[n].target.base_url,
                "pre_served": models[n].target.start is None,
            }
            for n in MODELS
        },
        "sides": {s: {"count": v.count, "sha256": v.sha256} for s, v in sealed.items()},
        "gold_ledger": {k: v for k, v in approval.items() if k != "approved"},
        "probe": (
            None
            if inputs.probe is None
            else {
                "count": len(probe),
                "sha256": dp.sha256_file(Path(inputs.probe)),
            }
        ),
        "earlier_measurements": earlier,
        "deployed": {
            "format": student["format"],
            "artifact": student["artifact"],
            "bundle": bundle,
        },
        "calibrations": _calibrations(student, clef_calibration, inputs),
        "gate": None if gate is None else {k: gate[k] for k in ("file", "sha256")},
    }
    if not apply:
        return {**base, "dry_run": True}
    with ledger:
        _check_ledger(ledger, deviation)  # again, under the lock
        base["student_server"] = attest_student(models[ATTESTED], bundle)  # before any request
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
                records=records,
                probe=probe,
                context=dreport.ReportContext(
                    prereg=reg,
                    prereg_sha256=prereg_sha256,
                    approved_ids=approval["approved"],
                    quant_manifest=student["manifest"],
                    bundle=bundle,
                    contamination_manifest=inputs.contamination_manifest,
                    gate=None if gate is None else gate["doc"],
                    probe_records=probe,
                    seed=bootstrap.seed,
                    resamples=bootstrap.resamples,
                    unverified_gold=inputs.unverified_gold,
                ),
            ),
        )


@dataclass(frozen=True)
class _ReportInputs:
    """The records the final reads and what the report cites beyond the measurement."""

    records: Mapping[str, Sequence[Record]]
    probe: Sequence[Record]
    context: dreport.ReportContext


def _hash_id(record_id: str) -> str:
    return "sha256:" + hashlib.sha256(record_id.encode("utf-8")).hexdigest()


def _write_predictions(out_dir: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    """Sealed-side rows per model; probe rows apart, with their record ids hashed."""
    for name, result in results.items():
        rows = result["predictions"]
        sealed = [r for r in rows if r["fold"] != PROBE_FOLD]
        _write_json(out_dir / PREDICTIONS_DIR / f"{name}.json", sealed)
        probe = [
            {**r, "record_id": _hash_id(r["record_id"])} for r in rows if r["fold"] == PROBE_FOLD
        ]
        if probe:
            _write_json(out_dir / PREDICTIONS_DIR / PROBE_DIR / f"{name}.json", probe)


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
    records = report_inputs.records
    folds: dict[str, Sequence[Record]] = dict(records)
    if report_inputs.probe:
        folds[PROBE_FOLD] = report_inputs.probe
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
                folds,
                memory_probe=memory_probe,
                client=client,
                proc_root=proc_root,
            )
            if name == ATTESTED:  # the bundle that answered is still the one attested
                dedge.check_bundle(_getter(model)(dedge.BUNDLE_PATH), base["deployed"]["bundle"])
    except BaseException:
        if any(c.asked for c in counted):
            record(once.FAILED_MID_RUN)
        raise
    ledger_records = record(once.MEASURED)
    out_dir = run_dir / FINAL_REL
    _write_predictions(out_dir, results)
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
        context=report_inputs.context,
    )
    _write_json(out_dir / FINAL_FILE, final)
    _write_json(out_dir / REPORT_FILE, report)
    return {"final": final, "report": report}
