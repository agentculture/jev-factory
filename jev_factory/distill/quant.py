"""Distill quantize: spike the deployed formats, heal once, recalibrate the student.

One stage over the winner of :mod:`jev_factory.distill.train` (``distill/select/summary.json``
and its checkpoint under ``distill/train/<cid>/checkpoint``), reading the **locked** distill
pre-registration (:func:`jev_factory.distill.prereg.require_locked`).

1. **Spike.** Every deployed-format candidate in :data:`FORMATS` (W4A16 torso + fp32 pointer
   head, ONNX int8, ONNX int4, GGUF torso + side-loaded head) is built from the winner by an
   injected *builder* and measured, beside the bf16 winner itself, on the fit and selection
   folds by an injected *measurer* (:func:`served_measurer` serves a ``hobson-hf-export/1``
   bundle with ``strands-decider serve`` and asks it over /v1/systemone, recording latency
   and peak memory with the ``distill.benchmark`` probes: ``nvidia-smi``, else the served
   process tree's ``/proc`` RSS, which is what GB10 reports). Each format's predictions go
   through the same functions train.py uses (``question_items``, ``fit_calibration``,
   ``selection_metrics``), so its calibrated selection ECE is its **own** calibration. A
   format with no builder, or whose builder raises :class:`FormatUnavailable`, is recorded
   as unavailable with the reason; it is never silently dropped. The default builders
   (:func:`default_builders`) register bf16 and, given the student env's python and the
   strands-decider checkout, the two torchao weight-only formats (deviation d3):
   ``torch_int8_wo`` and ``torch_int4_wo`` merge the LoRA, quantize the torso's Linear
   weights with torchao and keep the pointer head fp32
   (:mod:`jev_factory.backbones.pointer.quantized`, which also drops the vision tower and
   MTP head and packs the full-vocabulary embedding per its ``QuantScheme``); a build that
   fails on the box is recorded unavailable with its error. **Size budget (#20):** every
   quantized candidate's weight files are summed from the built files on disk; a build at
   or over :data:`SIZE_BUDGET_BYTES` (10^9 bytes) is recorded unavailable with its byte
   count and is never served, measured or chosen. Their artifacts are served by that
   module's wrapper with the same /v1/systemone shapes (plus ``/v1/bundle`` and
   ``/v1/stats``), through the same benchmark path. strands-decider serve
   @ 8b6faa5 (``StrandsDeciderModel.load``) loads the torso from the config's base id at the
   checkpoint dtype, wraps it in ``PeftModel`` unmerged and runs a torch or MLX engine, so the
   other four formats have no serve path on this stack until a builder for one is supplied
   (:data:`UNSUPPORTED`).
2. **Choose.** :data:`FORMAT_RULE`, in order: built and served; answered every fit and
   selection question; calibrated selection ECE <= ``bars.quant.ece_max``; selection
   accuracy within :data:`HEAL_MARGIN_POINTS` of the bf16 winner (if none is, the closest
   one goes to the heal); then the lowest measured peak memory, the lower median latency and
   :data:`FORMATS` order. bf16 is a valid outcome when no quantized format qualifies; the
   decision says why. ``format_decision.json`` cites every measurement it used (value plus
   the sha256 of ``measurements.json``).
3. **Heal, one round.** When the chosen quant loses more than :data:`HEAL_MARGIN_POINTS`
   points against its own bf16 on the selection fold, the trigger is appended to
   ``heal_log.jsonl`` **first**, then one more training pass runs from the winner's
   checkpoint on the same frozen ``train_set.jsonl`` through
   :mod:`jev_factory.backbones.pointer.train_distill` (``run_train``, so ``run_gpu_stage``),
   with the winner's knobs and the pre-registered ``heal`` block (``epochs``, ``lr``,
   ``init_mode``). The healed checkpoint is re-quantized and both its bf16 and its quant are
   re-measured. A second trigger (the healed quant still loses too much, or a heal round is
   already in the log) raises ``stop_escalate`` and writes nothing further.
4. **Recalibrate.** The deployed build's per-kind temperature is refitted on its own
   fit-fold predictions with :func:`jev_factory.distill.train.fit_calibration` (vector kept
   only if it lowers selection-fold ECE) and written to ``distill/quant/calibration.json``.
   ``quant_manifest.json`` carries the chosen format, the heal count, the quant-vs-bf16
   delta and the sha256 of every output.

5. **Pruned-vocabulary comparison (decision c38, h9).** When a builder for a
   :data:`PRUNED_FORMATS` name is registered (:func:`default_builders` registers both), the
   run's train, fit and selection prompts are written to ``prompt_set.jsonl`` (never test
   or held-out) and the pruned build at the bits of the full-vocabulary torchao build it is
   compared with is built from the winner and measured the same way. Its weight bytes,
   selection-fold accuracy and ``out_of_vocabulary`` refusal count are reported beside the
   full-vocabulary build's in ``measurements.json`` and ``quant_manifest.json``
   (``pruned_vocabulary_comparison``). It is never a candidate: it is not in
   :data:`FORMATS`, the rule never sees it, and a choice naming it is refused.

The margin and round count have one owner (:mod:`jev_factory.decide.rules`); the
pre-registered ``bars.quant`` must agree with them or the stage refuses. Dry-run by default;
``apply`` builds, serves and trains. Stdlib only; servers are localhost only.
"""

from __future__ import annotations

import json
import math
import os
import subprocess  # nosec B404 - only its exception types are used here
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.backbones.pointer import serve as pserve
from jev_factory.backbones.pointer import train_distill as td
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.decide import records as drecords
from jev_factory.decide.rules import HEAL_MARGIN_POINTS, HEAL_ROUNDS
from jev_factory.distill import benchmark as bench
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as dtrain
from jev_factory.distill.assemble import OUT_REL as TRAIN_SET_REL
from jev_factory.distill.records import Record
from jev_factory.distill.runspec import Bootstrap, GpuLimits, TrainerEnv
from jev_factory.factory import gpu

QUANT_REL = "distill/quant"
MEASUREMENTS_FILE = "measurements.json"
DECISION_FILE = "format_decision.json"
CALIBRATION_FILE = "calibration.json"
MANIFEST_FILE = "quant_manifest.json"
HEAL_LOG_FILE = "heal_log.jsonl"
HEAL_DIR = "heal"
BUILDS_DIR = "builds"
BF16 = "bf16"
STOP_VERDICT = "stop_escalate"
HEAL_KEYS = ("epochs", "lr", "init_mode")
#: '< 1 GB' means < 10^9 bytes of weight files on disk; one owner (the quantized builder).
SIZE_BUDGET_BYTES = qz.SIZE_BUDGET_BYTES
#: Files counted as weights in an artifact that carries no ``jev_quant.json``.
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".gguf", ".onnx")

#: The deployed-format candidates (frame decision on v1), in tie-break order.
FORMATS: dict[str, str] = {
    "w4a16_fp32_head": "W4A16 torso + fp32 pointer head",
    "onnx_int8": "ONNX int8",
    "onnx_int4": "ONNX int4",
    "gguf_side_head": "GGUF torso + side-loaded pointer head",
    "torch_int8_wo": "torchao int8 weight-only torso (LoRA merged) + fp32 pointer head",
    "torch_int4_wo": "torchao int4 weight-only torso (LoRA merged) + fp32 pointer head",
}
#: The torchao weight-only formats (deviation d3) -> their weight bits.
TORCH_FORMATS: dict[str, int] = {fmt: bits for bits, fmt in qz.FORMAT_OF_BITS.items()}
#: The pruned-vocabulary comparison builds (c38) -> their weight bits. Never candidates.
PRUNED_FORMATS: dict[str, int] = {fmt: bits for bits, fmt in qz.PRUNED_FORMAT_OF_BITS.items()}
COMPARISON_FORMATS: dict[str, str] = {
    fmt: f"{FORMATS[qz.FORMAT_OF_BITS[bits]]}, embedding pruned to the ids the run's "
    "train/fit/selection prompts use (comparison only, never shipped)"
    for fmt, bits in PRUNED_FORMATS.items()
}
PROMPT_SET_FILE = "prompt_set.jsonl"
COMPARISON_DIR = "comparison"
COMPARISON_NOTE = (
    "c38: the pruned-vocabulary build is measured beside the full-vocabulary build and "
    "reported only; the full-vocabulary quant stays the shipped build"
)

_SERVE_PATH = (
    "strands-decider serve @ 8b6faa5 (StrandsDeciderModel.load) loads the torso from the "
    "config's base id at the checkpoint dtype, wraps it in PeftModel unmerged and runs a "
    "torch or MLX engine"
)
#: Why each quantized format is unavailable when no builder is supplied for it.
UNSUPPORTED: dict[str, str] = {
    "w4a16_fp32_head": f"no W4A16 builder registered: {_SERVE_PATH}; it has no "
    "compressed-tensors (W4A16) torso load path",
    "onnx_int8": f"no ONNX int8 builder registered: {_SERVE_PATH}; it has no ONNX "
    "export or ONNX runtime path",
    "onnx_int4": f"no ONNX int4 builder registered: {_SERVE_PATH}; it has no ONNX "
    "export or ONNX runtime path",
    "gguf_side_head": f"no GGUF builder registered: {_SERVE_PATH}; it has no GGUF torso "
    "path to side-load the pointer head onto",
    **{
        fmt: f"no torchao builder registered: default_builders(revision, python=..., "
        f"decider_repo=...) needs the student env python and the strands-decider checkout "
        f"(decider_repo) to build {fmt}"
        for fmt in TORCH_FORMATS
    },
}

FORMAT_RULE = {
    "available": "built by a registered builder and served by the measurer",
    "size": f"weight files on disk total < {SIZE_BUDGET_BYTES} bytes; a build at or over "
    "it is unavailable, never a candidate",
    "safety": "answered every fit and selection question without an error",
    "calibration": "calibrated (own fit) selection ECE <= bars.quant.ece_max",
    "accuracy": f"selection accuracy within {HEAL_MARGIN_POINTS} points of the bf16 winner; "
    "if none is, the closest goes to one heal round",
    "footprint": "lowest measured peak memory, then lower median latency, then FORMATS order",
    "fallback": "bf16 when no quantized format survives safety and calibration",
}

_EPS = 1e-9


class FormatUnavailable(Exception):
    """A builder cannot build (or the stack cannot serve) a format; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


#: format, source checkpoint, output directory -> the built artifact (a servable directory).
Builder = Callable[[str, Path, Path], Path]
#: format, artifact, ``{fold: records}`` -> ``{"predictions": {fold: rows}, "latency",
#: "memory"}`` (latency and memory as :func:`jev_factory.distill.benchmark.benchmark_model`).
Measurer = Callable[[str, Path, Mapping[str, Sequence[Record]]], Mapping[str, Any]]
#: a heal plan from ``train_distill.plan_train`` -> anything (it must write the checkpoint).
Trainer = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class QuantHooks:
    """Injectable seams for :func:`run_quant`: format builders, the measurer, the heal trainer.

    ``None`` keeps the defaults (no builders, apply refused without a measurer, the capped
    GPU trainer for the heal round).
    """

    builders: Mapping[str, Builder] | None = None
    measurer: Measurer | None = None
    trainer: Trainer | None = None


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation or "fix the distill run inputs; a change after the lock is a deviation",
    )


def _stop(message: str) -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        f"{STOP_VERDICT}: {message}",
        f"heal is {HEAL_ROUNDS} round only; a human decides the next step (record a deviation)",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------- inputs


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def quant_bars(reg: dp.DistillPrereg) -> dict[str, Any]:
    """``bars.quant``, refused unless it agrees with the heal rule's own margin and rounds."""
    bars = reg.bars["quant"]
    if not math.isclose(float(bars["max_accuracy_drop_points"]), HEAL_MARGIN_POINTS):
        raise _refuse(
            f"bars.quant.max_accuracy_drop_points is {bars['max_accuracy_drop_points']}, but the "
            f"heal margin is {HEAL_MARGIN_POINTS} (jev_factory.decide.rules)",
            "register the heal margin, or record a deviation",
        )
    if bars["heal_rounds_max"] != HEAL_ROUNDS:
        raise _refuse(
            f"bars.quant.heal_rounds_max is {bars['heal_rounds_max']}, but heal is "
            f"{HEAL_ROUNDS} round only (jev_factory.decide.rules)",
            "register heal_rounds_max = 1, or record a deviation",
        )
    return dict(bars)


def heal_knobs(reg: dp.DistillPrereg) -> dict[str, Any]:
    """The pre-registered ``heal`` block (``epochs``, ``lr``, ``init_mode``, ``source``)."""
    block = reg.raw.get("heal")
    if not isinstance(block, dict):
        raise _refuse(
            "the distill pre-registration has no heal block",
            "register heal {epochs, lr, init_mode, source} before the lock (see example_doc)",
        )
    missing = [k for k in (*HEAL_KEYS, "source") if k not in block]
    if missing:
        raise _refuse(f"heal is missing: {', '.join(missing)}")
    epochs, lr = block["epochs"], block["lr"]
    if not (isinstance(epochs, int) and not isinstance(epochs, bool) and epochs >= 1):
        raise _refuse("heal.epochs must be a positive integer")
    if not (_is_num(lr) and lr > 0):
        raise _refuse("heal.lr must be a positive number")
    if block["init_mode"] not in td.INIT_MODES:
        raise _refuse(f"heal.init_mode must be one of {list(td.INIT_MODES)}")
    if not (isinstance(block["source"], str) and block["source"].strip()):
        raise _refuse("heal.source must cite where the heal recipe comes from")
    return {k: block[k] for k in HEAL_KEYS}


def read_winner(run_dir: Path) -> dict[str, Any]:
    """The selected candidate from ``distill/select/summary.json`` (cid, knobs, checkpoint)."""
    path = run_dir / dtrain.SELECT_REL / dtrain.SUMMARY_FILE
    if not path.is_file():
        raise _refuse(f"{path} does not exist", "run the distill train/select stage first")
    summary = json.loads(path.read_text(encoding="utf-8"))
    cid = summary.get("winner")
    if not cid:
        raise _refuse(f"{path} names no winner; there is nothing to quantize")
    chosen = next((c for c in summary.get("candidates", []) if c.get("cid") == cid), None)
    if chosen is None:
        raise _refuse(f"{path}: winner {cid} is not among its candidates")
    checkpoint = Path(chosen["checkpoint"])
    if not checkpoint.is_dir():
        raise _refuse(f"the winner's checkpoint {checkpoint} does not exist")
    return {
        "cid": cid,
        "knobs": dict(chosen["knobs"]),
        "checkpoint": checkpoint,
        "summary": path,
        "summary_sha256": dp.sha256_file(path),
    }


# --------------------------------------------------------------------- builders


def bf16_builder(base_revision: str) -> Builder:
    """Export the checkpoint unchanged to a verified ``hobson-hf-export/1`` bundle."""

    def build(fmt: str, checkpoint: Path, out_dir: Path) -> Path:
        if not (out_dir / player.PROVENANCE).is_file():
            player.write_layout(
                checkpoint,
                out_dir,
                base_revision=base_revision,
                extra={"distill_format": fmt},
            )
        return out_dir

    return build


def _gpu_guard(stage: str) -> Any:
    return gpu.assert_gpu_free(stage)


def _reusable(
    out_dir: Path, checkpoint: Path, fmt: str, scheme: qz.QuantScheme, prompts: Path | None = None
) -> bool:
    """An artifact already built in *out_dir* for this exact checkpoint, format and scheme
    (and, for a pruned build, from the same prompt set)."""
    if not (out_dir / qz.MANIFEST_FILE).is_file():
        return False
    try:
        manifest = qz.read_artifact(out_dir)["manifest"]
    except CliError:
        return False
    if prompts is not None:
        built_from = ((manifest.get("vocabulary") or {}).get("prompt_set") or {}).get("sha256")
        if built_from != player.sha256_file(Path(prompts)):
            return False
    return (
        manifest["format"] == fmt
        and manifest["scheme"] == scheme.record()
        and manifest["source"] == qz.source_record(checkpoint)
    )


def torch_wo_builder(
    base_revision: str,
    *,
    python: str,
    decider_repo: Path | str,
    device: str = "cuda",
    runner: qz.Runner | None = None,
    guard: Callable[[str], Any] | None = None,
    timeout: float | None = None,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    allow_foreign: bool = False,
    scheme: qz.QuantScheme | None = None,
) -> Builder:
    """Build a quantized artifact in the student env; a failure is unavailable.

    The format names the block bits (``torch_int4_wo``: 4, ``torch_int8_wo``: 8); the rest
    of *scheme* (embedding bits, group sizes; default :class:`~qz.QuantScheme`) applies to
    both and is recorded in each artifact's manifest.

    The GPU residency guard runs before each build (one GPU job at a time). A build that
    exits non-zero (a memory-floor watchdog stop included), cannot start, or leaves no
    verified artifact raises :class:`FormatUnavailable` carrying the error, so the spike
    records it, never drops it. The build runs through ``run_gpu_stage("quantize", ...)``
    under *memory_max* / *memory_floor*; a missing *memory_max* is refused (a ``CliError``,
    not an unavailable format), since the box would otherwise be unprotected.
    """

    limits = {
        "runner": runner,
        "timeout": timeout,
        "memory_max": memory_max,
        "memory_floor": memory_floor,
        "allow_foreign": allow_foreign,
    }

    def build(fmt: str, checkpoint: Path, out_dir: Path, prompts: Path | None = None) -> Path:
        bits = TORCH_FORMATS.get(fmt) or PRUNED_FORMATS.get(fmt)
        if bits is None:
            raise FormatUnavailable(
                f"the torchao builder builds {[*TORCH_FORMATS, *PRUNED_FORMATS]}, not {fmt}"
            )
        if (fmt in PRUNED_FORMATS) != (prompts is not None):
            raise FormatUnavailable(f"{fmt}: a pruned build needs a prompt set, a full one none")
        out_dir = Path(out_dir)
        fmt_scheme = replace(scheme or qz.QuantScheme(), block_bits=bits)
        if _reusable(out_dir, Path(checkpoint), fmt, fmt_scheme, prompts):
            return out_dir
        if not memory_max:
            raise _refuse("a GPU quantize build needs a memory cap", "pass memory_max with apply")
        (guard or _gpu_guard)("quantize")
        common = {
            "apply": True,
            "python": python,
            "decider_repo": decider_repo,
            "base_revision": base_revision,
            "device": device,
            **limits,
        }
        try:
            if prompts is None:
                qz.build_weight_only(checkpoint, out_dir, fmt_scheme, **common)
            else:
                qz.build_pruned_vocabulary(checkpoint, out_dir, fmt_scheme, prompts, **common)
        except CliError as exc:
            raise FormatUnavailable(f"torchao build failed for {fmt}: {exc.message}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise FormatUnavailable(
                f"torchao build failed for {fmt}: {type(exc).__name__}: {exc}"
            ) from exc
        return out_dir

    return build


def default_builders(
    base_revision: str,
    *,
    python: str | None = None,
    decider_repo: Path | str | None = None,
    device: str = "cuda",
    runner: qz.Runner | None = None,
    guard: Callable[[str], Any] | None = None,
    timeout: float | None = None,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    allow_foreign: bool = False,
    scheme: qz.QuantScheme | None = None,
) -> dict[str, Builder]:
    """bf16, plus the torchao weight-only formats when the student env is given.

    Without *python* and *decider_repo* the torchao formats are not registered and are
    recorded unavailable with :data:`UNSUPPORTED`'s reason; the other four formats always are.
    With them, the pruned-vocabulary comparison builds (:data:`PRUNED_FORMATS`) are
    registered too; they are measured and reported, never candidates.
    """
    builders: dict[str, Builder] = {BF16: bf16_builder(base_revision)}
    if python and decider_repo:
        torch_build = torch_wo_builder(
            base_revision,
            python=python,
            decider_repo=decider_repo,
            device=device,
            runner=runner,
            guard=guard,
            timeout=timeout,
            memory_max=memory_max,
            memory_floor=memory_floor,
            allow_foreign=allow_foreign,
            scheme=scheme,
        )
        builders.update(dict.fromkeys((*TORCH_FORMATS, *PRUNED_FORMATS), torch_build))
    return builders


# --------------------------------------------------------------------- measuring


def _accuracy(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name, {}).get("all", {}).get("value")
    return float(value) if _is_num(value) else None


def assess(
    fmt: str,
    artifact: Path,
    fold_records: Mapping[str, Sequence[Record]],
    measured: Mapping[str, Any],
    *,
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """One format's measurement: its own calibration and its calibrated selection metrics."""
    predictions = measured.get("predictions") or {}
    items: dict[str, list[dtrain.QItem]] = {}
    problems = 0
    for fold in dtrain.FOLDS:
        items[fold], bad = dtrain.question_items(fold_records[fold], predictions.get(fold) or [])
        problems += bad
    calibration = dtrain.fit_calibration(
        dtrain.calibration_rows(items["fit"]), dtrain.calibration_rows(items["selection"])
    )
    calibrated = [dtrain.calibrate_item(i, calibration) for i in items["selection"]]
    metrics = dtrain.selection_metrics(
        calibrated, threshold=threshold, seed=seed, resamples=resamples
    )
    return {
        "format": fmt,
        "available": True,
        "reason": None,
        "artifact": str(artifact),
        "problems": problems,
        "safe": problems == 0,
        "accuracy": _accuracy(metrics, "accuracy"),
        "ece": _accuracy(metrics, "ece"),
        "confident_wrong": _accuracy(metrics, "confident_wrong"),
        "latency": dict(measured.get("latency") or {}),
        "memory": dict(measured.get("memory") or {}),
        "calibration": calibration,
        "selection_metrics": metrics,
    }


def _unavailable(fmt: str, reason: str, **extra: Any) -> dict[str, Any]:
    known = {k: v for k, v in extra.items() if v is not None}
    return {"format": fmt, "available": False, "reason": reason, **known}


def _weight_files(artifact: Path) -> dict[str, int]:
    """Each weight file's size: the manifest's own list, else every weight-suffixed file."""
    if (artifact / qz.MANIFEST_FILE).is_file():
        return dict(qz.read_artifact(artifact)["manifest"]["weight_files"])
    files = sorted(p for p in artifact.rglob("*") if p.suffix in WEIGHT_SUFFIXES)
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise FormatUnavailable(f"{artifact}: a weight file is a symlink or not a regular file")
    return {p.relative_to(artifact).as_posix(): p.stat().st_size for p in files}


def size_record(artifact: Path | str) -> dict[str, Any]:
    """The weight files' byte sum, measured from the files on disk, against the budget.

    For a ``jev_quant.json`` artifact the files are verified first (hashes and recorded
    sizes match). Raises :class:`FormatUnavailable` when the weight bytes cannot be known.
    """
    artifact = Path(artifact)
    if not artifact.is_dir():
        raise FormatUnavailable(f"{artifact} is not an artifact directory; its size is unknown")
    try:
        files = _weight_files(artifact)
    except CliError as exc:
        raise FormatUnavailable(f"{artifact} failed verification: {exc.message}") from exc
    if not files:
        raise FormatUnavailable(f"{artifact} has no weight files; its size is unknown")
    for name in files:  # the sizes are re-read here, never taken from a record
        files[name] = (artifact / name).stat().st_size
    total = sum(files.values())
    return {
        "weight_files": files,
        "weight_bytes": total,
        "size_budget_bytes": SIZE_BUDGET_BYTES,
        "within_size_budget": total < SIZE_BUDGET_BYTES,
    }


def _over_budget(fmt: str, size: Mapping[str, Any]) -> str:
    return (
        f"{fmt} weight files total {size['weight_bytes']} bytes, at or over the "
        f"{SIZE_BUDGET_BYTES}-byte budget; it is never a candidate"
    )


def _drop_points(bf16: Mapping[str, Any], quant: Mapping[str, Any]) -> float | None:
    if bf16.get("accuracy") is None or quant.get("accuracy") is None:
        return None
    return round(100.0 * (bf16["accuracy"] - quant["accuracy"]), 9)


def _measure_format(
    fmt: str,
    checkpoint: Path,
    out_dir: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    **kw: Any,
) -> dict[str, Any]:
    builder = builders.get(fmt)
    if builder is None:
        return _unavailable(fmt, UNSUPPORTED.get(fmt, f"no builder registered for {fmt}"))
    size = None
    try:
        artifact = Path(builder(fmt, checkpoint, out_dir))
        if fmt != BF16:  # the reference, never a candidate: no size gate
            size = size_record(artifact)
            if not size["within_size_budget"]:
                return _unavailable(fmt, _over_budget(fmt, size), size=size)
        measured = measurer(fmt, artifact, {f: list(fold_records[f]) for f in dtrain.FOLDS})
    except FormatUnavailable as exc:
        return _unavailable(fmt, exc.reason, size=size)
    return {
        **assess(fmt, artifact, fold_records, measured, **kw),
        "size": size,
        "oov_refusals": oov_refusals(measured),
    }


def oov_refusals(measured: Mapping[str, Any]) -> dict[str, Any]:
    """How many records the server refused as ``out_of_vocabulary``, per fold and in total."""
    by_fold = {
        fold: sum(1 for row in rows or () if qz.OOV_ERROR in str(row.get("error") or ""))
        for fold, rows in (measured.get("predictions") or {}).items()
    }
    return {"by_fold": by_fold, "total": sum(by_fold.values())}


def _measure_all(
    checkpoint: Path,
    builds: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    formats: Sequence[str],
    **kw: Any,
) -> dict[str, dict[str, Any]]:
    out = {}
    for fmt in (BF16, *formats):
        out[fmt] = _measure_format(
            fmt, checkpoint, builds / fmt, builders, measurer, fold_records, **kw
        )
    bf16 = out[BF16]
    if not bf16["available"]:
        raise _refuse(f"the bf16 winner could not be built or served: {bf16['reason']}")
    if not bf16["safe"] or bf16["accuracy"] is None:
        raise _refuse(
            f"the bf16 winner left {bf16['problems']} fit/selection question(s) unanswered",
            "every quant is compared against its own bf16; fix the serve path first",
        )
    for fmt in formats:
        if out[fmt]["available"]:
            out[fmt]["drop_points"] = _drop_points(bf16, out[fmt])
            out[fmt]["delta_points"] = (
                None if out[fmt]["drop_points"] is None else -out[fmt]["drop_points"]
            )
    return out


# --------------------------------------------------------------------- pruned comparison (c38)


def _wire_criteria(kind: str, options: Sequence[Sequence[str]]) -> Any:
    """Criteria as :func:`jev_factory.distill.benchmark.wire_record` sends them."""
    if kind == "choice":
        return {name: desc for name, desc in options}
    if kind == "score":
        return [desc for _name, desc in options]
    return None


def _train_bodies(train_file: Path) -> list[dict[str, Any]]:
    """The train side's ``/v1/systemone`` bodies, rebuilt from the frozen trainer rows."""
    bodies: dict[str, dict[str, Any]] = {}
    for line, text in enumerate(Path(train_file).read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        try:
            rid, qid, kind, options = (
                row["record_id"],
                row["question_id"],
                row["kind"],
                row["options"],
            )
            body = bodies.setdefault(rid, {"id": rid, "state": row["state"], "questions": {}})
        except (KeyError, TypeError) as exc:
            raise FormatUnavailable(f"{train_file} row {line} is not a trainer row: {exc}") from exc
        item: dict[str, Any] = {"type": kind, "criteria": _wire_criteria(kind, options)}
        if row.get("instructions"):
            item["instructions"] = row["instructions"]
        body["questions"][qid] = item
    return [bodies[rid] for rid in sorted(bodies)]


def write_prompt_set(
    path: Path, train_file: Path, fold_records: Mapping[str, Sequence[Record]]
) -> dict[str, Any]:
    """Write the train, fit and selection prompts (never test or held-out); return its record."""
    if not Path(train_file).is_file():
        raise FormatUnavailable(f"the frozen training set {train_file} does not exist")
    try:
        rows = [{"side": "train", "body": b} for b in _train_bodies(train_file)]
    except ValueError as exc:
        raise FormatUnavailable(f"{train_file} is not valid JSONL: {exc}") from exc
    for fold in dtrain.FOLDS:
        rows += [{"side": fold, "body": bench.wire_record(r)} for r in fold_records[fold]]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    try:
        return qz.prompt_set_record(path)
    except CliError as exc:
        raise FormatUnavailable(f"the prompt set is unusable: {exc.message}") from exc


def _reference_full(measured: Mapping[str, Mapping[str, Any]], chosen: str) -> str | None:
    """The full-vocabulary torchao build the pruned one is compared with."""
    if chosen in TORCH_FORMATS:
        return chosen
    return next((f for f in TORCH_FORMATS if measured.get(f, {}).get("available")), None)


def _pruned_artifact(builder: Builder, fmt: str, checkpoint: Path, out: Path, prompts: Path):
    artifact = Path(builder(fmt, checkpoint, out, prompts))  # type: ignore[call-arg]
    try:
        manifest = qz.read_artifact(artifact)["manifest"]
    except CliError as exc:
        raise FormatUnavailable(f"{artifact} failed verification: {exc.message}") from exc
    if not qz.is_pruned(manifest) or manifest["format"] != fmt:
        raise FormatUnavailable(f"{artifact} is not a {fmt} pruned-vocabulary build")
    return artifact, manifest


def _measure_pruned(
    fmt: str,
    checkpoint: Path,
    out_dir: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    train_file: Path,
    **kw: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build and measure the pruned comparison; return its measurement and prompt set."""
    if fmt not in builders:
        return _unavailable(fmt, f"no builder registered for {fmt}"), None
    prompt_set = size = None
    try:
        prompt_set = write_prompt_set(out_dir / PROMPT_SET_FILE, train_file, fold_records)
        artifact, manifest = _pruned_artifact(
            builders[fmt], fmt, checkpoint, out_dir / fmt, Path(prompt_set["path"])
        )
        size = size_record(artifact)
        measured = measurer(fmt, artifact, {f: list(fold_records[f]) for f in dtrain.FOLDS})
    except FormatUnavailable as exc:
        return _unavailable(fmt, exc.reason, size=size), prompt_set
    entry = {
        **assess(fmt, artifact, fold_records, measured, **kw),
        "size": size,
        "oov_refusals": oov_refusals(measured),
        "pruned_vocabulary": manifest["pruned_vocabulary"],
    }
    return entry, prompt_set


def _side_by_side(m: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if m is None:
        return None
    return {
        "format": m["format"],
        "available": m["available"],
        "reason": m.get("reason"),
        "weight_bytes": (m.get("size") or {}).get("weight_bytes"),
        "selection_accuracy": m.get("accuracy"),
        "oov_refusals": (m.get("oov_refusals") or {}).get("total"),
    }


def compare_pruned(
    measured: Mapping[str, Mapping[str, Any]],
    chosen: str,
    *,
    checkpoint: Path,
    out_dir: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    train_file: Path,
    **kw: Any,
) -> dict[str, Any] | None:
    """The c38 comparison: the pruned build beside its full-vocabulary build; ``None`` when
    no pruned builder is registered. Reported only; it never changes *chosen*."""
    registered = [f for f in PRUNED_FORMATS if f in builders]
    if not registered:
        return None
    reference = _reference_full(measured, chosen)
    fmt = qz.format_name(TORCH_FORMATS[reference], pruned=True) if reference else registered[0]
    entry, prompt_set = _measure_pruned(
        fmt, checkpoint, out_dir, builders, measurer, fold_records, train_file, **kw
    )
    return {
        "decision": COMPARISON_NOTE,
        "comparison_only": True,
        "shipped": False,
        "shipped_format": chosen,
        "checkpoint": str(checkpoint),
        "fold": "selection",
        "prompt_set": prompt_set,
        "pruned": _side_by_side(entry),
        "full_vocabulary": _side_by_side(measured.get(reference) if reference else None),
        "measurement": entry,
    }


# --------------------------------------------------------------------- the choice


def _footprint(m: Mapping[str, Any], order: Sequence[str]) -> tuple[float, float, int]:
    peak = (m.get("memory") or {}).get("peak_mib")
    median = (m.get("latency") or {}).get("median_s")
    return (
        float(peak) if _is_num(peak) else math.inf,
        float(median) if _is_num(median) else math.inf,
        list(order).index(m["format"]),
    )


def _refuse_comparison_candidates(formats: Sequence[str]) -> None:
    if comparison := [f for f in formats if f in COMPARISON_FORMATS]:
        raise _refuse(
            f"{comparison} are pruned-vocabulary comparison builds; they are never candidates",
            COMPARISON_NOTE,
        )


def choose_format(
    measured: Mapping[str, Mapping[str, Any]], *, ece_max: float, formats: Sequence[str]
) -> dict[str, Any]:
    """Apply :data:`FORMAT_RULE`; return the chosen format, the trail and whether to heal."""
    _refuse_comparison_candidates(formats)
    alive = [measured[f] for f in formats]
    trail = []

    def step(name: str, keep: Callable[[Mapping[str, Any]], bool]) -> None:
        nonlocal alive
        alive = [m for m in alive if keep(m)]
        trail.append({"step": name, "survivors": [m["format"] for m in alive]})

    step("available", lambda m: m["available"])
    step("safety", lambda m: m["safe"])
    step("calibration", lambda m: m["ece"] is not None and m["ece"] <= ece_max + _EPS)
    if not alive:
        return {
            "format": BF16,
            "heal": False,
            "trail": trail,
            "reason": "no quantized format survived the "
            + next(t["step"] for t in trail if not t["survivors"])
            + " step; the deployed build stays bf16",
        }
    within = [
        m
        for m in alive
        if m.get("drop_points") is not None and m["drop_points"] <= HEAL_MARGIN_POINTS + _EPS
    ]
    if within:
        alive = within
        heal = False
    else:
        closest = min(m["drop_points"] for m in alive if m.get("drop_points") is not None)
        alive = [m for m in alive if m.get("drop_points") == closest]
        heal = True
    trail.append({"step": "accuracy", "survivors": [m["format"] for m in alive], "heal": heal})
    alive = sorted(alive, key=lambda m: _footprint(m, formats))[:1]
    trail.append({"step": "footprint", "survivors": [m["format"] for m in alive]})
    chosen = alive[0]
    reason = (
        f"{chosen['format']} loses {chosen['drop_points']} points against its own bf16 "
        f"(> {HEAL_MARGIN_POINTS}); no format is within the margin, so it is healed once"
        if heal
        else f"{chosen['format']} is within {HEAL_MARGIN_POINTS} points of its own bf16 "
        f"({chosen['drop_points']} lost) with calibrated ECE {chosen['ece']} <= {ece_max}, "
        "and has the smallest measured footprint among those that qualify"
    )
    return {"format": chosen["format"], "heal": heal, "trail": trail, "reason": reason}


def _cited(measured: Mapping[str, Mapping[str, Any]], source: Path) -> list[dict[str, Any]]:
    digest = dp.sha256_file(source)
    out = []
    for fmt, m in measured.items():
        if not m["available"]:
            continue
        values = {
            "selection_accuracy": m["accuracy"],
            "selection_ece_calibrated": m["ece"],
            "selection_confident_wrong": m["confident_wrong"],
            "accuracy_delta_points_vs_bf16": m.get("delta_points", 0.0 if fmt == BF16 else None),
            "latency_median_s": (m["latency"] or {}).get("median_s"),
            "peak_memory_mib": (m["memory"] or {}).get("peak_mib"),
            "weight_bytes": (m.get("size") or {}).get("weight_bytes"),
        }
        for name, value in values.items():
            value = float(value) if _is_num(value) else None
            out.append(drecords.cite(f"{fmt}.{name}", value, source, digest))
    return out


# --------------------------------------------------------------------- heal


def _heal_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def heal_rounds_spent(log: Path) -> int:
    return sum(1 for e in _heal_log(log) if e.get("event") == "heal_trigger")


def _append(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def plan_heal(
    *,
    reg: dp.DistillPrereg,
    prereg_path: Path,
    run_dir: Path,
    winner: Mapping[str, Any],
    train_file: Path,
    trainer_python: str,
    decider_repo: Path | str,
    deviation_id: str | None,
) -> dict[str, Any]:
    """One more training pass from the winner's checkpoint, with the heal knobs."""
    knobs = {**winner["knobs"], **heal_knobs(reg)}
    return td.plan_train(
        prereg_path=prereg_path,
        run_dir=run_dir,
        run_knobs=knobs,
        train_file=train_file,
        init_from=winner["checkpoint"],
        output_dir=run_dir / QUANT_REL / HEAL_DIR / dtrain.CHECKPOINT_DIR,
        trainer_python=trainer_python,
        decider_repo=decider_repo,
        deviation_id=deviation_id,
    )


# --------------------------------------------------------------------- the stage


def _outputs(paths: Sequence[Path]) -> dict[str, str]:
    return {str(p): dp.sha256_file(p) for p in paths if p.is_file()}


def _dry_run(
    base: Mapping[str, Any],
    formats: Sequence[str],
    builders: Mapping[str, Builder],
    heal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """The dry-run plan: each candidate format, whether it has a builder, and the heal plan."""
    return {
        **base,
        "dry_run": True,
        "formats": [
            {
                "format": f,
                "description": FORMATS[f],
                "builder": f in builders,
                "unavailable_reason": None if f in builders else UNSUPPORTED[f],
            }
            for f in formats
        ],
        "comparison": [
            {"format": f, "description": COMPARISON_FORMATS[f], "comparison_only": True}
            for f in PRUNED_FORMATS
            if f in builders
        ],
        "comparison_note": COMPARISON_NOTE,
        "heal_plan": heal_plan,
    }


def _check_apply_inputs(measurer: Measurer | None, builders: Mapping[str, Builder]) -> None:
    if measurer is None:
        raise _refuse(
            "apply needs a measurer for the fit and selection folds",
            "pass measurer=served_measurer(...) (or a test double)",
        )
    if BF16 not in builders:
        raise _refuse("apply needs a bf16 builder", "pass builders=default_builders(revision)")


def _check_heal_allowed(
    fmt: str, measured: Mapping[str, Mapping[str, Any]], heal_log: Path, train_file: Path
) -> None:
    """The one heal round must still be unspent, and the frozen training set must exist."""
    if heal_rounds_spent(heal_log) >= HEAL_ROUNDS:
        raise _stop(
            f"{fmt} loses {measured[fmt]['drop_points']} points against its own bf16 "
            f"(> {HEAL_MARGIN_POINTS}) and the heal round is already spent ({heal_log})"
        )
    if not train_file.is_file():
        raise _refuse(
            f"the frozen training set {train_file} does not exist; heal trains on it",
            "run the distill assemble stage first",
        )


def _heal_trigger(
    reg: dp.DistillPrereg, fmt: str, measured: Mapping[str, Mapping[str, Any]], heal_log: Path
) -> dict[str, Any]:
    return {
        "event": "heal_trigger",
        "format": fmt,
        "heal_round": heal_rounds_spent(heal_log) + 1,
        "drop_points": measured[fmt]["drop_points"],
        "bf16_accuracy": measured[BF16]["accuracy"],
        "quant_accuracy": measured[fmt]["accuracy"],
        "margin_points": HEAL_MARGIN_POINTS,
        "heal_knobs": heal_knobs(reg),
        "heal_source": reg.raw["heal"]["source"],
    }


def _check_healed(fmt: str, again: Mapping[str, Any], bars: Mapping[str, Any]) -> None:
    """The healed build must measure cleanly, within the heal margin and the ECE bar."""
    if not again["available"] or not again["safe"]:
        raise _stop(f"the healed {fmt} could not be measured cleanly: {again.get('reason')}")
    if again["drop_points"] is None or again["drop_points"] > HEAL_MARGIN_POINTS + _EPS:
        raise _stop(
            f"second heal trigger: the healed {fmt} still loses {again['drop_points']} "
            f"points against its own bf16 (> {HEAL_MARGIN_POINTS})"
        )
    if again["ece"] is None or again["ece"] > float(bars["ece_max"]) + _EPS:
        raise _stop(f"the healed {fmt} has calibrated ECE {again['ece']} > {bars['ece_max']}")


def run_quant(
    *,
    prereg_path: Path | str,
    run_dir: Path | str,
    folds: Mapping[str, Any],
    trainer_env: TrainerEnv,
    apply: bool = False,
    hooks: QuantHooks = QuantHooks(),
    sealed: Sequence[Any] = (),
    train_file: Path | str | None = None,
    deviation_id: str | None = None,
    limits: GpuLimits = GpuLimits(),
    formats: Sequence[str] = tuple(FORMATS),
    bootstrap: Bootstrap = Bootstrap(),
) -> dict[str, Any]:
    """Plan (dry-run) or run the format spike, the one heal round and the recalibration."""
    builders, measurer, trainer = hooks.builders, hooks.measurer, hooks.trainer
    trainer_python, decider_repo = trainer_env.python, trainer_env.decider_repo
    memory_max, memory_floor = limits.memory_max, limits.memory_floor
    allow_foreign, timeout = limits.allow_foreign, limits.timeout
    seed, resamples = bootstrap.seed, bootstrap.resamples
    prereg_path, run_dir = Path(prereg_path), Path(run_dir)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    bars = quant_bars(reg)
    threshold = dtrain._threshold(reg)
    _refuse_comparison_candidates(formats)
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise _refuse(f"unknown deployed format(s): {unknown}; candidates are {list(FORMATS)}")
    winner = read_winner(run_dir)
    dtrain.check_raw_readout(winner["checkpoint"])
    fold_records = dtrain._folds(folds, sealed)
    train_file = Path(train_file) if train_file else run_dir / TRAIN_SET_REL
    builders = dict(builders or {})
    out_dir = run_dir / QUANT_REL
    heal_log = out_dir / HEAL_LOG_FILE
    heal_plan = plan_heal(
        reg=reg,
        prereg_path=prereg_path,
        run_dir=run_dir,
        winner=winner,
        train_file=train_file,
        trainer_python=trainer_python,
        decider_repo=decider_repo,
        deviation_id=deviation_id,
    )
    base = {
        "stage": "distill-quant",
        "prereg_sha256": dp.sha256_file(prereg_path),
        "winner": {k: str(v) if isinstance(v, Path) else v for k, v in winner.items()},
        "folds": {f: len(fold_records[f]) for f in dtrain.FOLDS},
        "heal_margin_points": HEAL_MARGIN_POINTS,
        "heal_rounds_max": HEAL_ROUNDS,
        "size_budget_bytes": SIZE_BUDGET_BYTES,
        "format_rule": FORMAT_RULE,
    }
    if not apply:
        return _dry_run(base, formats, builders, heal_plan)
    _check_apply_inputs(measurer, builders)
    kw = {"threshold": threshold, "seed": seed, "resamples": resamples}
    measured = _measure_all(
        winner["checkpoint"], out_dir / BUILDS_DIR, builders, measurer, fold_records, formats, **kw
    )
    choice = choose_format(measured, ece_max=float(bars["ece_max"]), formats=formats)
    deployed_checkpoint = winner["checkpoint"]
    heal_record: dict[str, Any] | None = None
    healed: dict[str, dict[str, Any]] | None = None
    if choice["heal"]:
        fmt = choice["format"]
        _check_heal_allowed(fmt, measured, heal_log, train_file)
        trigger = _heal_trigger(reg, fmt, measured, heal_log)
        _append(heal_log, trigger)  # logged before anything heals
        if trainer is None:
            trainer = _default_trainer(
                allow_foreign=allow_foreign,
                memory_max=memory_max,
                memory_floor=memory_floor,
                timeout=timeout,
            )
        trainer(heal_plan)
        healed_checkpoint = Path(heal_plan["output_dir"])
        dtrain.check_raw_readout(healed_checkpoint)
        healed = _measure_all(
            healed_checkpoint,
            out_dir / HEAL_DIR / BUILDS_DIR,
            builders,
            measurer,
            fold_records,
            (fmt,),
            **kw,
        )
        again = healed[fmt]
        _check_healed(fmt, again, bars)
        deployed_checkpoint = healed_checkpoint
        heal_record = {**trigger, "healed_checkpoint": str(healed_checkpoint)}
        _append(
            heal_log,
            {"event": "heal_result", "format": fmt, "drop_points": again["drop_points"]},
        )
    comparison = compare_pruned(
        measured,
        choice["format"],
        checkpoint=winner["checkpoint"],
        out_dir=out_dir / COMPARISON_DIR,
        builders=builders,
        measurer=measurer,
        fold_records=fold_records,
        train_file=train_file,
        **kw,
    )
    return _finish(
        base=base,
        out_dir=out_dir,
        measured=measured,
        healed=healed,
        choice=choice,
        heal_record=heal_record,
        heal_log=heal_log,
        deployed_checkpoint=deployed_checkpoint,
        comparison=comparison,
    )


def _finish(
    *,
    base: Mapping[str, Any],
    out_dir: Path,
    measured: Mapping[str, Mapping[str, Any]],
    healed: Mapping[str, Mapping[str, Any]] | None,
    choice: Mapping[str, Any],
    heal_record: Mapping[str, Any] | None,
    heal_log: Path,
    deployed_checkpoint: Path,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fmt = choice["format"]
    _refuse_comparison_candidates([fmt])
    final = (healed or measured)[fmt]
    final_bf16 = (healed or measured)[BF16]
    measurements_path = out_dir / MEASUREMENTS_FILE
    _write_json(
        measurements_path,
        {
            "prereg_sha256": base["prereg_sha256"],
            "fold": "selection",
            "formats": measured,
            "healed": healed,
            "pruned_vocabulary_comparison": comparison,
        },
    )
    cited = _cited(measured, measurements_path)
    if healed:
        cited += [{**c, "name": "healed." + c["name"]} for c in _cited(healed, measurements_path)]
    decision = {
        "decision": "deployed_format",
        "format": fmt,
        "description": FORMATS.get(fmt, "bf16 (unquantized)"),
        "reason": choice["reason"],
        "rule": FORMAT_RULE,
        "trail": choice["trail"],
        "unavailable": {f: m["reason"] for f, m in measured.items() if not m["available"]},
        "cited": cited,
        "prereg_sha256": base["prereg_sha256"],
        "rule_owner": "jev_factory.decide.rules (HEAL_MARGIN_POINTS, HEAL_ROUNDS)",
        "never_candidates": {f: COMPARISON_NOTE for f in COMPARISON_FORMATS},
    }
    decision_path = out_dir / DECISION_FILE
    _write_json(decision_path, decision)
    calibration_path = out_dir / CALIBRATION_FILE
    _write_json(
        calibration_path,
        {
            **final["calibration"],
            "format": fmt,
            "checkpoint": str(deployed_checkpoint),
            "fitted_on_predictions_of": fmt,
            "prereg_sha256": base["prereg_sha256"],
        },
    )
    delta = None if fmt == BF16 else final.get("delta_points")
    manifest = {
        **base,
        "dry_run": False,
        "status": "done",
        "chosen_format": fmt,
        "deployed_checkpoint": str(deployed_checkpoint),
        "deployed_artifact": final["artifact"],
        "deployed_size": final.get("size"),
        "heal_count": 1 if heal_record else 0,
        "heal": heal_record,
        "quant_vs_bf16_delta_points": 0.0 if fmt == BF16 else delta,
        "quant_vs_bf16_initial_delta_points": (
            None if fmt == BF16 else measured[fmt].get("delta_points")
        ),
        "bf16_selection_accuracy": final_bf16["accuracy"],
        "deployed_selection_accuracy": final["accuracy"],
        "deployed_selection_ece": final["ece"],
        "pruned_vocabulary_comparison": (
            None
            if comparison is None
            else {k: comparison[k] for k in comparison if k != "measurement"}
        ),
        "outputs": _outputs([measurements_path, decision_path, calibration_path, heal_log]),
    }
    _write_json(out_dir / MANIFEST_FILE, manifest)
    return {**manifest, "decision": decision, "calibration": final["calibration"]}


def _default_trainer(**kw: Any) -> Trainer:
    def train(plan: Mapping[str, Any]) -> Any:
        if not kw.get("memory_max"):
            raise _refuse("a GPU heal run needs a memory cap", "pass memory_max with apply")
        return td.run_train(plan, apply=True, **kw)

    return train


# --------------------------------------------------------------------- served measurer


def _record_uncapped(jobdir: Path, name: str) -> None:
    """Add ``capped: false`` (and why) to ``<name>.serve.json``: servers run without capped.sh."""
    path = Path(jobdir) / f"{name}.serve.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["capped"] = False
    record["capped_note"] = qz.UNCAPPED_NOTE
    _write_json(path, record)


def _torchao_starter(
    jobdir: Path,
    name: str,
    env: Any,
    artifact: Path,
    port: int,
    device: str | None,
    decider_repo: Path | str | None,
) -> Callable[[], Any]:
    """Verify a torchao artifact; return what starts its wrapper server."""
    try:
        qz.read_artifact(artifact)
    except CliError as exc:
        raise FormatUnavailable(
            f"{artifact} is not a verified torchao artifact: {exc.message}"
        ) from exc

    def start() -> Any:
        return qz.launch(
            jobdir,
            name,
            env,
            artifact,
            port=port,
            decider_repo=decider_repo,
            device=device,
        )

    return start


def _bundle_starter(
    jobdir: Path, name: str, env: Any, artifact: Path, port: int, device: str | None
) -> Callable[[], Any]:
    """Verify a ``hobson-hf-export/1`` bundle; return what starts ``strands-decider serve``."""
    try:
        player.read_layout(artifact)
    except CliError as exc:
        raise FormatUnavailable(
            f"{artifact} is not a hobson-hf-export/1 bundle strands-decider serve "
            f"can load: {exc.message}"
        ) from exc

    def start() -> Any:
        pid = pserve.launch(jobdir, name, env, artifact, port=port, device=device)
        _record_uncapped(jobdir, name)
        return pid

    return start


def served_measurer(
    *,
    env: Any,
    jobdir: Path | str,
    port: int,
    device: str | None = None,
    ready_timeout: float = 600.0,
    memory_probe: bench.MemoryProbe | None = bench.nvidia_smi_used_mib,
    decider_repo: Path | str | None = None,
) -> Measurer:
    """Serve an artifact and measure it over /v1/systemone.

    A ``hobson-hf-export/1`` bundle is served by ``strands-decider serve``; a torchao
    weight-only artifact (:mod:`jev_factory.backbones.pointer.quantized`, recognised by its
    ``jev_quant.json``) by that module's wrapper, with *decider_repo* on its path. The GPU
    residency guard runs first; latency and peak memory come from
    :func:`jev_factory.distill.benchmark.benchmark_model` (``nvidia-smi``, else the served
    process tree's ``/proc`` RSS, read from the job's pid file). The server is always
    stopped. An artifact the server cannot load is :class:`FormatUnavailable`; so is a
    torchao artifact that fails its hash check or whose server dies or never gets ready.
    """
    jobdir = Path(jobdir)
    base_url = f"http://{pserve.LOOPBACK}:{int(port)}"

    def pid_of(name: str) -> Callable[[], int | None]:
        def read() -> int | None:
            try:
                return int((jobdir / f"{name}.pid").read_text().strip())
            except (OSError, ValueError):
                return None

        return read

    def measure(
        fmt: str, artifact: Path, fold_records: Mapping[str, Sequence[Record]]
    ) -> dict[str, Any]:
        name = f"distill-quant-{fmt}"
        quantized = (Path(artifact) / qz.MANIFEST_FILE).is_file()
        if quantized:
            start = _torchao_starter(jobdir, name, env, artifact, port, device, decider_repo)
        else:
            start = _bundle_starter(jobdir, name, env, artifact, port, device)
        gpu.assert_gpu_free("measure")
        target = bench.Target(
            name=name,
            base_url=base_url,
            model=name,
            start=start,
            ready=lambda: pserve.wait_ready(
                base_url, timeout=ready_timeout, jobdir=jobdir, name=name
            ),
            stop=lambda: dtrain._stop_job(jobdir, name),
            pid=pid_of(name),
        )
        started = time.monotonic()
        try:
            out = bench.benchmark_model(
                target,
                fold_records,
                memory_probe=memory_probe,
                client=pserve.PointerClient(base_url, model=name),
            )
        except CliError as exc:
            if not quantized:
                raise
            raise FormatUnavailable(
                f"the torchao {fmt} artifact could not be served: {exc.message}"
            ) from exc
        return {
            "predictions": {
                f: [p for p in out["predictions"] if p["fold"] == f] for f in fold_records
            },
            "latency": out["latency"],
            "memory": out["memory"],
            "wall_s": time.monotonic() - started,
        }

    return measure
