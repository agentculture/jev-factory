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
deployed artifact's attestation (the student backbone's ``quantizer().bundle_attestation``:
per-file sha256 and manifest sha256), or its ``GET /v1/stats`` reports no cgroup memory cap. The
bundle is fetched again after the student is scored; a change fails the run. ``final.json``
records ``deployed.bundle`` (``files``, ``manifest_sha256``, ``weight_bytes``: what the edge
stage checks) and ``student_server`` (the attested bundle, the board, the memory cap that
``/v1/stats`` reported).

The model set follows the pre-registration (:func:`required_models`): under
``deployed_format`` bf16 the deployed student is the selected bf16 build
(:func:`bf16_deployed_student`: the ``select/summary.json`` winner's ``train/<winner>/bundle``,
calibrated with ``select/calibration.json``), scored once as ``student_bf16``, and attested by
the per-file sha256 of the folder its loopback server was started from
(``FinalModel.served_from``; strands-decider serve has no ``/v1/bundle``), before any request
and again after it was scored. No quant manifest is read then. The deployed quant may be any
attested loopback server (c9): with the ``deployed_server`` role absent it is not treated as a
board (its memory is this box's probe). :func:`deployed_url` reads the role-neutral
``distill_deployed_url`` run-config key, with ``distill_agx_url`` as its alias. ``final.json``
shows the Clef variant beside the teacher row (``teacher``, ``models.clef.teacher_variant``).

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

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import deployed as ddeployed
from jev_factory.distill import edge as dedge
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as dquant
from jev_factory.distill import report as dreport
from jev_factory.distill import train as dtrain
from jev_factory.distill.records import Record
from jev_factory.distill.runspec import Bootstrap
from jev_factory.distill.student import backbone as student_backbone
from jev_factory.measure import once
from jev_factory.measure.preflight import require_localhost
from jev_factory.release.hub import digests
from jev_factory.review import gold

#: The student's deployed-format facet, through the registry (deviation d2).
qz = student_backbone().quantizer()

STAGE = "distill-final"
LABEL = "distill-final"
FINAL_REL = "distill/final"
FINAL_FILE = "final.json"
REPORT_FILE = "report.json"
PREDICTIONS_DIR = "predictions"
PROBE_DIR = "probe"
PROBE_FOLD = "probe"
MODELS = dreport.MODELS
#: The model whose served bundle is attested: the shipped quant (``deployed_format`` quant).
ATTESTED = dreport.STUDENT
#: The run-config key naming the deployed student's loopback URL (any attested server, c9),
#: and the older, board-specific key still read as an alias.
DEPLOYED_URL_KEY = "distill_deployed_url"
DEPLOYED_URL_ALIASES: tuple[str, ...] = ("distill_agx_url",)
#: The bf16 deployed build (``deployed_format`` bf16): ``<run>/distill/train/<winner>/bundle``
#: (``hobson-hf-export/1``), calibrated with ``<run>/distill/select/calibration.json``.
BF16_BUNDLE_DIR = "bundle"
BF16_CALIBRATION_REL = f"{dtrain.SELECT_REL}/{dtrain.CALIBRATION_FILE}"
BF16_ATTESTATION = (
    "file sha256 of the bundle the local server was started from "
    "(strands-decider serve has no /v1/bundle)"
)
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
    #: The folder a locally started server loads; a bf16 deployed student is attested by the
    #: per-file sha256 of this folder (it has no ``/v1/bundle``).
    served_from: Path | str | None = None


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


def deployed_url(config: Any, *, required: bool = True) -> str | None:
    """The deployed student's URL from a run config: ``distill_deployed_url``, else an alias.

    *config* is anything with ``get`` (a mapping or a RunConfig). Two keys naming different
    URLs are refused rather than guessed; with neither set, ``None`` unless *required*.
    """
    found = {}
    for key in (DEPLOYED_URL_KEY, *DEPLOYED_URL_ALIASES):
        value = config.get(key)
        if value not in (None, ""):
            found[key] = str(value)
    if len(set(found.values())) > 1:
        raise _refuse(
            f"the run config names two deployed-student URLs: {found}",
            f"set only {DEPLOYED_URL_KEY} ({', '.join(DEPLOYED_URL_ALIASES)} is its old name)",
        )
    if found:
        return next(iter(found.values()))
    if required:
        raise _refuse(
            "the run config names no deployed-student URL",
            f"set {DEPLOYED_URL_KEY} to the attested loopback server, e.g. http://127.0.0.1:<port>",
        )
    return None


def required_models(deployed_format: str) -> tuple[str, ...]:
    """The models one final scores: the bf16 student is the deployed build under bf16."""
    if deployed_format == "bf16":
        return tuple(m for m in MODELS if m != dreport.STUDENT)
    return MODELS


def deployed_student(run_dir: Path, deployed_format: str = "quant") -> dict[str, Any]:
    """The deployed format, its calibration and the quant manifest, cross-checked by sha256.

    Under ``deployed_format`` bf16 it is the selected bf16 build instead
    (:func:`bf16_deployed_student`); there is no quant manifest and no fallback either way.
    """
    if deployed_format == "bf16":
        return bf16_deployed_student(run_dir)
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


def bf16_deployed_student(run_dir: Path) -> dict[str, Any]:
    """The bf16 deployed build: the ``select/summary.json`` winner's ``bundle`` folder.

    Resolved and verified by :func:`jev_factory.distill.deployed.bf16_student` (the one bf16
    resolver gate, bundle and edge use too); this adds the final's key names: ``artifact`` as
    text, ``selection_predictions`` and an empty quant ``manifest``.
    """
    student = ddeployed.bf16_student(run_dir)
    return {
        **student,
        "artifact": str(student["artifact"]),
        "selection_predictions": str(student["predictions_file"]),
        "manifest": {},
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


def bf16_attestation(path: Path) -> dict[str, Any]:
    """Per-file sha256 of a bf16 bundle folder, its manifest sha256 and weight bytes: the one
    attestation shape, :func:`jev_factory.distill.deployed.attestation`."""
    return ddeployed.attestation(path)


def deployed_bundle(run_dir: Path, artifact: Any, deployed_format: str = "quant") -> dict[str, Any]:
    """The attestation of the deployed artifact on disk: what ``GET /v1/bundle`` must answer
    (quant), or the folder hashes a bf16 server's ``served_from`` must match (bf16)."""
    path = Path(str(artifact or ""))
    if not artifact:
        raise _refuse("the quant manifest names no deployed_artifact; cannot attest the bundle")
    if not path.is_absolute():
        path = run_dir / path
    if deployed_format == "bf16":
        return bf16_attestation(path)
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


def _sealed_runtime(result: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and latency over the sealed sides only; probe-set calls are counted apart."""
    rows = [p for p in result["predictions"] if p.get("fold") != PROBE_FOLD]
    failed = sum(1 for p in rows if p.get("error") or bench._has_error(p))  # noqa: SLF001
    return {
        "n_records": len(rows),
        "n_failed": failed,
        "latency": bench.latency_summary([p["latency_s"] for p in rows]),
        "memory": result["memory"],
        "probe_records": len(result["predictions"]) - len(rows),
    }


def _served_memory(get: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
    """The served process's own memory, read from its ``/v1/stats`` after scoring."""
    try:
        stats = get(dedge.STATS_PATH)
    except (CliError, OSError, ValueError) as exc:
        return {"probe": "v1/stats", "measured": False, "reason": str(exc)}
    rss = stats.get("rss_bytes")
    return {
        "probe": "v1/stats",
        "measured": isinstance(rss, int),
        "rss_bytes_after": rss,
        "mem_available_bytes": stats.get("mem_available_bytes"),
    }


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


def attest_bf16_student(model: FinalModel, expected: Mapping[str, Any]) -> dict[str, Any]:
    """A bf16 deployed student: a loopback server started from a folder hashing to *expected*.

    Called before any request is sent to any model, and again after it was scored.
    """
    try:
        require_localhost(model.target.base_url)
    except ValueError as exc:
        raise _refuse(str(exc), "serve the bf16 student on http://127.0.0.1:<port>") from None
    if model.served_from is None:
        raise _refuse(
            "the bf16 student names no served_from folder; its server cannot be attested",
            "start the server from the deployed bundle and pass that folder as served_from",
        )
    folder = Path(model.served_from)
    served = digests(folder) if folder.is_dir() else {}
    if served != dict(expected["files"]):
        changed = sorted(
            k
            for k in set(served) | set(expected["files"])
            if served.get(k) != expected["files"].get(k)
        )
        raise _refuse(
            f"{folder}: the served bf16 bundle differs from the deployed bundle ({changed})",
            "start the server from distill/train/<winner>/bundle, unchanged",
        )
    return {
        "base_url": model.target.base_url,
        "attestation": BF16_ATTESTATION,
        "served_from": str(folder),
        "bundle": {
            "files": served,
            "manifest_sha256": expected.get("manifest_sha256"),
            "weight_bytes": expected.get("weight_bytes"),
        },
        "memory_cap": None,
    }


def _check_models(models: Mapping[str, FinalModel], required: Sequence[str] = MODELS) -> None:
    missing = [m for m in required if m not in models]
    extra = sorted(set(models) - set(required))
    if missing or extra:
        raise _refuse(
            f"the final scores exactly {list(required)} in one invocation;"
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
    student: Mapping[str, Any],
    clef_calibration: Path | str | None,
    inputs: FinalInputs,
    deployed_format: str = "quant",
) -> dict[str, dict[str, Any]]:
    bf16 = deployed_format == "bf16"
    deployed = {
        "source": BF16_CALIBRATION_REL if bf16 else STUDENT_CALIBRATION_REL,
        "sha256": student["calibration_sha256"],
        "calibration": student["calibration"],
    }
    out: dict[str, dict[str, Any]] = {
        dreport.CLEF: {"source": CAL_RAW, "sha256": None, "calibration": None},
        dreport.STOCK: {"source": CAL_SERVED, "sha256": None, "calibration": None},
        dreport.STUDENT_BF16: {"source": CAL_RAW, "sha256": None, "calibration": None},
        dreport.STUDENT: deployed,
    }
    if bf16:  # one build: the bf16 student is the deployed student, with its one calibration
        given = inputs.bf16_calibration
        if given is not None and (
            not Path(given).is_file() or dp.sha256_file(Path(given)) != deployed["sha256"]
        ):
            raise _refuse(
                f"{given}: a bf16 calibration other than the deployed {BF16_CALIBRATION_REL}",
                "under deployed_format bf16 the bf16 student is the deployed build; it is "
                "measured with the calibration it ships with",
            )
        out[dreport.STUDENT_BF16] = dict(deployed)
        clef_only = ((dreport.CLEF_CALIBRATED, clef_calibration, "Clef calibration"),)
    else:
        clef_only = (
            (dreport.CLEF_CALIBRATED, clef_calibration, "Clef calibration"),
            (dreport.STUDENT_BF16, inputs.bf16_calibration, "bf16 student calibration"),
        )
    for name, path, what in clef_only:
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
    view = dreport.prereg_view(reg)
    fmt = view.deployed_format
    plan = _Plan(
        names=required_models(fmt),
        attested=view.deployed_model,
        deployed_format=fmt,
        on_board=fmt == "quant" and dp.role_present(reg, "deployed_server"),
    )
    _check_models(models, plan.names)
    sealed = _sealed_sides(dataset_dir)
    records = {stem: _read_side(side) for stem, side in sealed.items()}
    if inputs.unverified_gold:
        approval = unverified_records(deviation_id, records["test"], records["held_out"])
    else:
        approval = approved_records(
            _ledger_file(dataset_dir, inputs), records["test"], records["held_out"]
        )
    probe = load_probe(inputs.probe, sealed)
    student = deployed_student(run_dir, fmt)
    bundle = deployed_bundle(run_dir, student["artifact"], fmt)
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
                **({"teacher_variant": view.teacher_variant} if n == dreport.CLEF else {}),
            }
            for n in plan.names
        },
        "teacher": {
            "row": dreport.CLEF,
            "variant": view.teacher_variant,
            "model": view.teacher_model,
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
            "model": plan.attested,
            **(
                {
                    "winner": student["winner"],
                    "selection_predictions": student["selection_predictions"],
                }
                if fmt == "bf16"
                else {}
            ),
        },
        "calibrations": _calibrations(student, clef_calibration, inputs, fmt),
        "gate": None if gate is None else {k: gate[k] for k in ("file", "sha256")},
    }
    if not apply:
        return {**base, "dry_run": True}
    with ledger:
        _check_ledger(ledger, deviation)  # again, under the lock
        base["student_server"] = plan.attest(models[plan.attested], bundle)  # before any request
        return _measure(
            plan=plan,
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
class _Plan:
    """Which models this final scores and how the deployed student is attested."""

    names: tuple[str, ...]
    #: The deployed student's row: the quant, or the bf16 student under bf16.
    attested: str
    deployed_format: str
    #: The deployed quant is pre-served on a board (role deployed_server present): its memory
    #: is read from its own ``/v1/stats``, never this box's GPU.
    on_board: bool

    def attest(self, model: FinalModel, expected: Mapping[str, Any]) -> dict[str, Any]:
        if self.deployed_format == "bf16":
            return attest_bf16_student(model, expected)
        return attest_student(model, expected)


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
    plan: _Plan,
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
        for name in plan.names:  # sequential: one served model at a time
            model = models[name]
            client = _Counted(
                model.client
                or bench.SystemOneClient(model.target.base_url, model=model.target.model)
            )
            counted.append(client)
            remote = model.target.start is None and name == plan.attested and plan.on_board
            results[name] = bench.benchmark_model(
                model.target,
                folds,
                memory_probe=None if remote else memory_probe,
                client=client,
                proc_root=proc_root,
            )
            if remote:  # this box's GPU says nothing about the board's memory
                results[name]["memory"] = _served_memory(_getter(model))
            if name == plan.attested:  # the bundle that answered is still the one attested
                if plan.deployed_format == "bf16":
                    attest_bf16_student(model, base["deployed"]["bundle"])
                else:
                    expected = base["deployed"]["bundle"]
                    dedge.check_bundle(_getter(model)(dedge.BUNDLE_PATH), expected)
    except BaseException:
        if any(c.asked for c in counted):
            record(once.FAILED_MID_RUN)
        raise
    ledger_records = record(once.MEASURED)
    out_dir = run_dir / FINAL_REL
    _write_predictions(out_dir, results)
    runtime = {name: {"host": host, **_sealed_runtime(r)} for name, r in results.items()}
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
