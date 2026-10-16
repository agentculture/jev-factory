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
   :data:`FORMATS` order. When no quantized format qualifies (none built under the 10^9-byte
   budget, or none survives safety and calibration), there is **no bf16 fallback** (operator
   decision on risk r10): the stage records a failed ``no qualifying format`` result
   (``quant_manifest.json`` ``status: failed``, ``shipped_format: null``; no
   ``calibration.json``), ships nothing and exits non-zero; ``final`` and ``bundle`` refuse
   such a manifest. ``format_decision.json`` cites every measurement it used (value plus the
   sha256 of ``measurements.json``).
3. **Agreement (c17, h10).** Every quantized build's argmax agreement with the bf16
   student is computed on the **fit and selection** predictions only (never test or
   held-out; any other fold in a measurement is ignored) and must reach the pre-registered
   ``bars.c17.quant.quant_vs_bf16_agreement_min`` (0.97). A chosen quant below it is not
   shippable as is and goes to the heal.
4. **Heal, one round per format (c10).** When the chosen quant loses more than
   :data:`HEAL_MARGIN_POINTS` points against its own bf16 on the selection fold, or agrees
   with the bf16 student below the bar, a heal round runs. The heal set
   (``heal/<format>/round-<n>/heal_set.jsonl``) is the frozen ``train_set.jsonl``'s rows
   with Clef's distribution (rows whose teacher is Clef) and no gold; a row whose record id
   is a test, held-out, fit or selection id is refused (exit non-zero) before anything is
   logged, and the trainer re-checks the same ids (as sha256) before it loads a model. The
   trigger is then appended to ``heal_log.jsonl`` and one training pass runs from the
   winner's checkpoint through :mod:`jev_factory.backbones.pointer.train_distill` in heal
   mode (``run_train``, so ``run_gpu_stage``), with the winner's knobs and the
   pre-registered ``heal`` block (``epochs``, ``lr``, ``init_mode``, and ``mix``: the
   weights on the bf16 student's and Clef's per-option distributions in each soft
   target). The trainer trains the quantized student (its LoRA and pointer head, through a
   fake-quantized forward at the built format's scheme, read from the artifact's
   ``jev_quant.json``) and falls back to nvsh d18's bf16 continuation when QAT is
   unavailable on the student env; the healed checkpoint records which ran
   (``heal.method``), and the manifest carries it. The healed checkpoint is re-quantized
   and re-measured; its agreement is against the **original** bf16 student's predictions.
   A healed build that still loses too much, misses the ECE bar or agrees below the bar is
   not shippable: a ``heal_result`` with ``shippable: false`` is appended, the same failed
   ``no qualifying format`` result is recorded (nothing ships) and the stage raises
   ``stop_escalate``. A format whose heal round is spent is refused another without
   a deviation id (each further round needs a deviation id not used before for it).
5. **Recalibrate.** The deployed build's per-kind temperature is refitted on its own
   fit-fold predictions with :func:`jev_factory.distill.train.fit_calibration` (vector kept
   only if it lowers selection-fold ECE) and written to ``distill/quant/calibration.json``.
   ``quant_manifest.json`` carries the chosen format, the heal count, the quant-vs-bf16
   delta and the sha256 of every output. ``deployed_predictions.json`` keeps the shipped
   build's own **selection-fold** predictions with the sha256s of the bundle that produced
   them (``/v1/bundle``'s attestation): the edge stage compares a board's answers with them.

6. **Pruned-vocabulary comparison (decision c38, h9).** When a builder for a
   :data:`PRUNED_FORMATS` name is registered (:func:`default_builders` registers both), the
   run's train, fit and selection prompts are written to ``prompt_set.jsonl`` (never test
   or held-out) and the pruned build at the bits of the full-vocabulary torchao build it is
   compared with is built from the winner and measured the same way, plus on the operator's
   **probe set** when one is given (risk r14: probe records are disjoint from the kept-id
   prompts, so out-of-vocabulary refusals show there; a probe record whose id is in the prompt
   set or a sealed side is refused). Its weight bytes, selection-fold accuracy and
   ``out_of_vocabulary`` refusal counts (per fold, the probe's apart) are reported beside the
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
from jev_factory.factory import gpu, safepath

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
HEAL_SET_FILE = "heal_set.jsonl"
HEAL_SPEC_FILE = "heal_spec.json"
DEPLOYED_PREDICTIONS_FILE = "deployed_predictions.json"
#: The outcome recorded when no quantized build qualifies; nothing ships (risk r10).
NO_QUALIFYING = "no qualifying format"
PROBE_FOLD = "probe"
#: The folds agreement with the bf16 student is computed on (never test or held-out).
AGREEMENT_FOLDS: tuple[str, ...] = dtrain.FOLDS
POOLED = "fit+selection"
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
    "guard": "with the gating knob only (deviation d12): accuracy, calibration and agreement "
    "read the clean sources, and a format whose guard (every other source) selection "
    "accuracy is more than 5 points below stock's fails",
    "accuracy": f"selection accuracy within {HEAL_MARGIN_POINTS} points of the bf16 winner; "
    "if none is, the closest goes to one heal round",
    "footprint": "lowest measured peak memory, then lower median latency, then FORMATS order",
    "no_fallback": f"{NO_QUALIFYING} when no quantized format survives: the stage records a "
    "failed result and ships nothing (never bf16)",
}

_EPS = 1e-9
#: Deviation d12: how far (points) the guard may fall below the stock model before it fails.
GUARD_TOLERANCE_POINTS = 5.0


@dataclass(frozen=True)
class Gating:
    """Deviation d12: the rules read only *sources*; every other source is the guard.

    Accuracy, calibration, the heal trigger and agreement with the bf16 student are computed
    on records from *sources* (the contamination stage's clean sources). The guard's raw
    selection accuracy is reported per format and only fails a format below *guard_floor*;
    with no floor (``gating=clean-report``, d14) it is reported beside *guard_stock* only.
    """

    sources: frozenset[str]
    guard_floor: float | None = None
    #: stock's guard selection accuracy, for the report (d14)
    guard_stock: float | None = None

    def split(
        self, fold_records: Mapping[str, Sequence[Record]]
    ) -> tuple[dict[str, list[Record]], dict[str, list[Record]]]:
        domain = {f: [r for r in rs if r.source in self.sources] for f, rs in fold_records.items()}
        guard = {
            f: [r for r in rs if r.source not in self.sources] for f, rs in fold_records.items()
        }
        return domain, guard


def _only(rows: Sequence[Mapping[str, Any]] | None, ids: set[str]) -> list[Mapping[str, Any]]:
    return [r for r in rows or () if str(r.get("record_id")) in ids]


def _restrict(predictions: Mapping[str, Any], records: Mapping[str, Sequence[Record]]) -> dict:
    return {f: _only(predictions.get(f), {r.id for r in rs}) for f, rs in records.items()}


def raw_accuracy(records: Sequence[Record], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Argmax accuracy of *rows* on *records*' gold-bearing questions (no calibration)."""
    items, problems = dtrain.question_items(records, _only(rows, {r.id for r in records}))
    gold = [i for i in items if i.gold is not None]
    right = sum(1 for i in gold if max(sorted(i.probs), key=lambda k: i.probs[k]) == i.gold)
    return {
        "n": len(gold),
        "n_correct": right,
        "accuracy": right / len(gold) if gold else None,
        "problems": problems,
    }


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


@dataclass(frozen=True)
class QuantKnobs:
    """The quant stage's operator knobs for :func:`run_quant`.

    *formats* are the deployed-format candidates. The other three are deviation-gated
    overrides of the locked rule, each refused without a deviation id: *heal_mix* replaces
    the registered ``heal.mix`` (both mixes are recorded in the heal log), *gating* (d12)
    reads the quant rules on the gating sources only, and *heal_repeat* (d13) repeats the
    heal set's domain and guard rows.
    """

    formats: Sequence[str] = tuple(FORMATS)
    heal_mix: Mapping[str, float] | None = None
    gating: Gating | None = None
    heal_repeat: Mapping[str, int] | None = None


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
    path = safepath.contained(path, path.parent)  # a symlink out of the folder is refused
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


def heal_mix(reg: dp.DistillPrereg) -> dict[str, float]:
    """The pre-registered ``heal.mix``: weights on the bf16 student's and Clef's targets."""
    heal_knobs(reg)  # the block itself first
    block = reg.raw["heal"]
    if "mix" not in block:
        raise _refuse(
            f"the heal block names no mix {list(td.HEAL_MIX_KEYS)}",
            "register heal.mix before the lock (see example_doc), or record a deviation",
        )
    return td.check_mix(block["mix"])


def agreement_min(reg: dp.DistillPrereg) -> float:
    """``bars.c17.quant.quant_vs_bf16_agreement_min`` (validated >= 0.97 by the prereg)."""
    return float(reg.bars["c17"]["quant"]["quant_vs_bf16_agreement_min"])


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


def _gpu_guard(stage: str, allow_foreign: bool = False) -> Any:
    return gpu.assert_gpu_free(stage, allow_foreign=allow_foreign)


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
        if guard is None:
            _gpu_guard("quantize", allow_foreign=allow_foreign)
        else:
            guard("quantize")
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


def _answer(question: Mapping[str, Any]) -> Any:
    """A question's argmax (option name, or ``p_true >= 0.5`` for noul); None on an error."""
    if question.get("error"):
        return None
    dist = question.get("distribution")
    if isinstance(dist, Mapping) and dist:
        return max(sorted(dist), key=lambda k: dist[k])
    p_true = question.get("p_true")
    return None if p_true is None else bool(p_true >= 0.5)


def _answers(rows: Sequence[Mapping[str, Any]] | None) -> dict[tuple[str, str], Any]:
    return {
        (str(row.get("record_id")), str(qid)): _answer(q)
        for row in rows or ()
        if not row.get("error")
        for qid, q in (row.get("questions") or {}).items()
    }


def _agree(ref: Mapping[Any, Any], other: Mapping[Any, Any]) -> dict[str, Any]:
    keys = [k for k, v in ref.items() if v is not None]
    compared = [k for k in keys if other.get(k) is not None]
    agree = sum(1 for k in compared if ref[k] == other[k])
    return {
        "n_reference": len(keys),
        "n_compared": len(compared),
        "n_agree": agree,
        # a reference answer the quant left unanswered counts against it
        "agreement": agree / len(keys) if keys else None,
    }


def agreement_vs_bf16(
    bf16: Mapping[str, Sequence[Mapping[str, Any]]],
    quant: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Argmax agreement of a quant's predictions with the bf16 student's, per fold and pooled.

    Only :data:`AGREEMENT_FOLDS` (fit and selection) are read; any other fold in either
    prediction set is ignored, so test and held-out never enter it.
    """
    out: dict[str, Any] = {"folds": list(AGREEMENT_FOLDS)}
    pooled_ref: dict[Any, Any] = {}
    pooled_q: dict[Any, Any] = {}
    for fold in AGREEMENT_FOLDS:
        ref, other = _answers(bf16.get(fold)), _answers(quant.get(fold))
        out[fold] = _agree(ref, other)
        pooled_ref.update({(fold, *k): v for k, v in ref.items()})
        pooled_q.update({(fold, *k): v for k, v in other.items()})
    out[POOLED] = _agree(pooled_ref, pooled_q)
    return out


def _pooled_agreement(m: Mapping[str, Any]) -> float | None:
    value = ((m.get("agreement_vs_bf16") or {}).get(POOLED) or {}).get("agreement")
    return float(value) if _is_num(value) else None


def _measure_format(
    fmt: str,
    checkpoint: Path,
    out_dir: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    gating: Gating | None = None,
    **kw: Any,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """One format's measurement and its raw predictions (``{}`` when unavailable).

    With *gating* (d12) every record is still asked, but the assessment reads the gating
    sources only and the guard's raw selection accuracy is recorded beside it."""
    builder = builders.get(fmt)
    if builder is None:
        return _unavailable(fmt, UNSUPPORTED.get(fmt, f"no builder registered for {fmt}")), {}
    size = None
    try:
        artifact = Path(builder(fmt, checkpoint, out_dir))
        if fmt != BF16:  # the reference, never a candidate: no size gate
            size = size_record(artifact)
            if not size["within_size_budget"]:
                return _unavailable(fmt, _over_budget(fmt, size), size=size), {}
        measured = measurer(fmt, artifact, {f: list(fold_records[f]) for f in dtrain.FOLDS})
    except FormatUnavailable as exc:
        return _unavailable(fmt, exc.reason, size=size), {}
    if gating is None:
        assessed = assess(fmt, artifact, fold_records, measured, **kw)
    else:
        domain, guard = gating.split(fold_records)
        raw = measured.get("predictions") or {}
        assessed = assess(
            fmt, artifact, domain, {**measured, "predictions": _restrict(raw, domain)}, **kw
        )
        assessed["guard"] = {
            "sources": sorted({r.source for rs in guard.values() for r in rs}),
            "selection": raw_accuracy(guard["selection"], raw.get("selection") or []),
            "floor": gating.guard_floor,
            "stock": gating.guard_stock,
        }
    entry = {**assessed, "size": size, "oov_refusals": oov_refusals(measured)}
    return entry, measured.get("predictions") or {}


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
    reference: Mapping[str, Any] | None = None,
    gating: Gating | None = None,
    **kw: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Measure bf16 and *formats*; return the measurements and each format's raw predictions.

    Each available quant's ``agreement_vs_bf16`` is against *reference* (the bf16
    student's predictions; default: this round's bf16).
    """
    out, predictions = {}, {}
    for fmt in (BF16, *formats):
        out[fmt], predictions[fmt] = _measure_format(
            fmt, checkpoint, builds / fmt, builders, measurer, fold_records, gating, **kw
        )
    reference = predictions[BF16] if reference is None else reference
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
            ref, got = reference, predictions[fmt]
            if gating is not None:
                domain, _guard = gating.split(fold_records)
                ref, got = _restrict(reference, domain), _restrict(got, domain)
            out[fmt]["agreement_vs_bf16"] = agreement_vs_bf16(ref, got)
    return out, predictions


# --------------------------------------------------------------------- pruned comparison (c38)


def _wire_criteria(kind: str, options: Sequence[Sequence[str]]) -> Any:
    """Criteria as :func:`jev_factory.distill.benchmark.wire_record` sends them."""
    if kind == "choice":
        return dict(options)
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


def _reference_full(measured: Mapping[str, Mapping[str, Any]], chosen: str | None) -> str | None:
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


def _refuse_probe_in_prompts(
    probe: Sequence[Record], train_file: Path, fold_records: Mapping[str, Sequence[Record]]
) -> None:
    """The probe set must be disjoint from the kept-id prompts (train, fit, selection)."""
    if not probe:
        return
    prompt_ids = {r.id for f in dtrain.FOLDS for r in fold_records[f]}
    if Path(train_file).is_file():
        prompt_ids |= {b["id"] for b in _train_bodies(train_file)}
    clash = sorted({r.id for r in probe} & prompt_ids)
    if clash:
        raise _refuse(
            f"{len(clash)} probe record(s) are in the pruned build's prompt set",
            "the probe set measures out-of-vocabulary refusals on prompts the kept ids "
            "were not taken from; remove the overlapping records from the probe set",
        )


def _measure_pruned(
    fmt: str,
    checkpoint: Path,
    out_dir: Path,
    builders: Mapping[str, Builder],
    measurer: Measurer,
    fold_records: Mapping[str, Sequence[Record]],
    train_file: Path,
    probe: Sequence[Record] = (),
    **kw: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build and measure the pruned comparison (and on the probe set); return its measurement
    and prompt set."""
    if fmt not in builders:
        return _unavailable(fmt, f"no builder registered for {fmt}"), None
    _refuse_probe_in_prompts(probe, train_file, fold_records)
    prompt_set = size = None
    sets: dict[str, list[Record]] = {f: list(fold_records[f]) for f in dtrain.FOLDS}
    if probe:
        sets[PROBE_FOLD] = list(probe)
    try:
        prompt_set = write_prompt_set(out_dir / PROMPT_SET_FILE, train_file, fold_records)
        artifact, manifest = _pruned_artifact(
            builders[fmt], fmt, checkpoint, out_dir / fmt, Path(prompt_set["path"])
        )
        size = size_record(artifact)
        measured = measurer(fmt, artifact, sets)
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
    oov = m.get("oov_refusals") or {}
    by_fold = dict(oov.get("by_fold") or {})
    return {
        "format": m["format"],
        "available": m["available"],
        "reason": m.get("reason"),
        "weight_bytes": (m.get("size") or {}).get("weight_bytes"),
        "selection_accuracy": m.get("accuracy"),
        "oov_refusals": oov.get("total"),
        "oov_refusals_by_fold": by_fold,
        # None: the build was not asked the probe set (no probe given, or not this build)
        "probe_oov_refusals": by_fold.get(PROBE_FOLD),
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
    probe: Sequence[Record] = (),
    **kw: Any,
) -> dict[str, Any] | None:
    """The c38 comparison: the pruned build beside its full-vocabulary build; ``None`` when
    no pruned builder is registered. Reported only; it never changes *chosen*.

    The pruned build is also asked *probe* (records disjoint from the kept-id prompts), so its
    out-of-vocabulary refusals are counted where they can happen (risk r14)."""
    registered = [f for f in PRUNED_FORMATS if f in builders]
    if not registered:
        return None
    reference = _reference_full(measured, chosen)
    fmt = qz.format_name(TORCH_FORMATS[reference], pruned=True) if reference else registered[0]
    entry, prompt_set = _measure_pruned(
        fmt, checkpoint, out_dir, builders, measurer, fold_records, train_file, probe, **kw
    )
    return {
        "decision": COMPARISON_NOTE,
        "comparison_only": True,
        "shipped": False,
        "shipped_format": chosen,
        "checkpoint": str(checkpoint),
        "fold": "selection",
        "probe": {
            "count": len(probe),
            "measured": bool(probe),
            "note": (
                "the pruned build only: the full-vocabulary build has no out-of-vocabulary path"
                if probe
                else "no probe set given: out-of-vocabulary refusals are counted on "
                "the fit and selection folds only"
            ),
        },
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


def _guard_holds(m: Mapping[str, Any]) -> bool:
    """d12: a format fails only when its guard accuracy falls below the guard floor."""
    guard = m.get("guard") or {}
    floor, acc = guard.get("floor"), (guard.get("selection") or {}).get("accuracy")
    return floor is None or (acc is not None and acc >= floor - _EPS)


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
            "format": None,
            "heal": False,
            "trail": trail,
            "reason": f"{NO_QUALIFYING}: no quantized format survived the "
            + next(t["step"] for t in trail if not t["survivors"])
            + " step; nothing ships (there is no bf16 fallback)",
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
            "agreement_vs_bf16": None if fmt == BF16 else _pooled_agreement(m),
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


def _triggers(log: Path, fmt: str | None) -> list[dict[str, Any]]:
    """Heal triggers in *log* for *fmt* (all of them for ``None``; an entry naming no
    format counts against every format)."""
    return [
        e
        for e in _heal_log(log)
        if e.get("event") == "heal_trigger" and (fmt is None or e.get("format") in (fmt, None))
    ]


def heal_rounds_spent(log: Path, fmt: str | None = None) -> int:
    return len(_triggers(log, fmt))


def _append(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def heal_root(out_dir: Path, fmt: str, round_: int) -> Path:
    """``<quant>/heal/<format>/round-<n>``: the heal set, spec, checkpoint and builds."""
    return Path(out_dir) / HEAL_DIR / fmt / f"round-{int(round_)}"


def protected_digests(
    fold_records: Mapping[str, Sequence[Record]], sealed: Sequence[Any]
) -> list[str]:
    """sha256 of every id the heal must never train on: test, held-out, fit and selection."""
    ids = {str(i) for side in sealed for i in getattr(side, "ids", ())}
    ids |= {r.id for recs in fold_records.values() for r in recs}
    return sorted(td.id_digest(i) for i in ids)


def _clef_of(row: Mapping[str, Any]) -> Any:
    """Clef's distribution for a trainer row: its teacher when the teacher slot is Clef's."""
    return row.get("teacher") if str(row.get("target_source", "")).startswith("clef") else None


_HEAL_ROW_KEYS = ("record_id", "question_id", "kind", "state", "options")


def _repeats(
    row: Mapping[str, Any], repeat: Mapping[str, int] | None, sources: frozenset[str]
) -> int:
    if not repeat:
        return 1
    return int(repeat["domain"] if row.get("source") in sources else repeat["guard"])


def write_heal_set(
    train_file: Path,
    path: Path,
    protected: Sequence[str],
    repeat: Mapping[str, int] | None = None,
    sources: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The heal input from the frozen train rows: Clef's distribution, no gold or teacher.

    Refuses (exit non-zero, nothing written) a row whose record id is protected.
    """
    rows = []
    for line, text in enumerate(Path(train_file).read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        missing = [k for k in _HEAL_ROW_KEYS if not isinstance(row, dict) or k not in row]
        if missing:
            raise _refuse(f"{train_file} row {line} is not a trainer row (no {missing})")
        rows.append(row)
    td.check_heal_ids([r["record_id"] for r in rows], protected)
    if not rows:
        raise _refuse(f"the frozen training set {train_file} has no rows to heal on")
    out = [
        {
            **{k: r[k] for k in _HEAL_ROW_KEYS},
            "instructions": r.get("instructions") or "",
            "weight": r.get("weight", 1.0),
            "clef": _clef_of(r),
            "clef_source": r.get("target_source"),
        }
        for r in rows
        for _ in range(_repeats(r, repeat, sources))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": dp.sha256_file(path),
        "rows": len(out),
        "repeat": dict(repeat) if repeat else None,
        "domain_sources": sorted(sources) if repeat else [],
        "rows_with_clef": sum(r["clef"] is not None for r in out),
        "train_file_sha256": dp.sha256_file(Path(train_file)),
    }


def fake_quant_scheme(fmt: str, artifact: Path | str | None) -> dict[str, Any] | None:
    """The fake-quant scheme a QAT heal mirrors: the built torchao artifact's own scheme.

    ``None`` (the heal then falls back to the bf16 continuation) for a format that is not a
    torchao weight-only build or whose artifact carries no verified ``jev_quant.json``.
    """
    if fmt not in TORCH_FORMATS or artifact is None:
        return None
    if not (Path(artifact) / qz.MANIFEST_FILE).is_file():
        return None
    try:
        scheme = qz.read_artifact(Path(artifact))["manifest"]["scheme"]
    except CliError:
        return None
    bits = int(scheme["block_bits"])
    return {
        "block_bits": bits,
        # torchao int4 weight-only: grouped, scale and zero; int8: per channel, symmetric
        "block_group_size": int(scheme["block_group_size"]) if bits == 4 else None,
        "block_symmetric": bits == 8,
        "embedding_bits": int(scheme["embedding_bits"]),
        "embedding_group_size": int(scheme["embedding_group_size"]),
    }


def plan_heal(
    *,
    reg: dp.DistillPrereg,
    prereg_path: Path,
    run_dir: Path,
    winner: Mapping[str, Any],
    root: Path,
    trainer_python: str,
    decider_repo: Path | str,
    deviation_id: str | None,
) -> dict[str, Any]:
    """One heal-mode training pass from the winner's checkpoint, with the heal knobs."""
    knobs = {**winner["knobs"], **heal_knobs(reg)}
    return td.plan_train(
        prereg_path=prereg_path,
        run_dir=run_dir,
        run_knobs=knobs,
        train_file=root / HEAL_SET_FILE,
        init_from=winner["checkpoint"],
        output_dir=root / dtrain.CHECKPOINT_DIR,
        trainer_python=trainer_python,
        decider_repo=decider_repo,
        deviation_id=deviation_id,
        heal_spec=root / HEAL_SPEC_FILE,
    )


# --------------------------------------------------------------------- the stage


def _outputs(paths: Sequence[Path]) -> dict[str, str]:
    return {str(p): dp.sha256_file(p) for p in paths if p.is_file()}


def _dry_run(
    base: Mapping[str, Any],
    formats: Sequence[str],
    builders: Mapping[str, Builder],
    heal_plans: Mapping[str, Any],
) -> dict[str, Any]:
    """The dry-run plan: each candidate format, whether it has a builder, and its heal plan."""
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
        "heal_plans": dict(heal_plans),
    }


def _check_apply_inputs(measurer: Measurer | None, builders: Mapping[str, Builder]) -> None:
    if measurer is None:
        raise _refuse(
            "apply needs a measurer for the fit and selection folds",
            "pass measurer=served_measurer(...) (or a test double)",
        )
    if BF16 not in builders:
        raise _refuse("apply needs a bf16 builder", "pass builders=default_builders(revision)")


def heal_reasons(
    fmt: str, measured: Mapping[str, Mapping[str, Any]], choice: Mapping[str, Any], minimum: float
) -> list[str]:
    """Why the chosen quant needs a heal (empty: it ships as is). bf16 never heals."""
    if fmt == BF16:
        return []
    reasons = []
    if choice["heal"]:
        reasons.append(
            f"accuracy: {fmt} loses {measured[fmt]['drop_points']} points against its own "
            f"bf16 (> {HEAL_MARGIN_POINTS})"
        )
    agreement = _pooled_agreement(measured[fmt])
    if agreement is None or agreement < minimum - _EPS:
        reasons.append(
            f"agreement: {fmt} agrees {agreement} with the bf16 student on fit+selection "
            f"(< {minimum})"
        )
    if not _guard_holds(measured[fmt]):
        reasons.append(f"guard: {_guard_text(measured[fmt])}")
    return reasons


def _guard_text(m: Mapping[str, Any]) -> str:
    guard = m.get("guard") or {}
    acc = (guard.get("selection") or {}).get("accuracy")
    return f"{m['format']} guard selection accuracy {acc} is below its floor {guard.get('floor')}"


@dataclass(frozen=True)
class _HealRun:
    """What one heal round needs from the stage."""

    reg: dp.DistillPrereg
    prereg_path: Path
    run_dir: Path
    out_dir: Path
    winner: Mapping[str, Any]
    train_file: Path
    trainer_env: TrainerEnv
    deviation_id: str | None
    fold_records: Mapping[str, Sequence[Record]]
    sealed: Sequence[Any]
    builders: Mapping[str, Builder]
    measurer: Measurer
    kw: Mapping[str, Any]
    #: A deviation's heal mix in place of the registered one (``run_quant(heal_mix=...)``).
    mix_override: Mapping[str, float] | None = None
    gating: Gating | None = None
    #: A deviation's repeat counts for the heal set's domain and guard rows (d13).
    repeat: Mapping[str, int] | None = None

    @property
    def mix(self) -> dict[str, float]:
        if self.mix_override is not None:
            return dict(self.mix_override)
        return heal_mix(self.reg)

    @property
    def heal_log(self) -> Path:
        return self.out_dir / HEAL_LOG_FILE

    def plan(self, fmt: str, round_: int) -> dict[str, Any]:
        return plan_heal(
            reg=self.reg,
            prereg_path=self.prereg_path,
            run_dir=self.run_dir,
            winner=self.winner,
            root=heal_root(self.out_dir, fmt, round_),
            trainer_python=self.trainer_env.python,
            decider_repo=self.trainer_env.decider_repo,
            deviation_id=self.deviation_id,
        )


def _check_heal_allowed(
    fmt: str, reasons: Sequence[str], heal_log: Path, train_file: Path, deviation_id: str | None
) -> int:
    """This format's heal round number, refused past :data:`HEAL_ROUNDS` without a fresh
    deviation id; the frozen training set must exist."""
    spent = _triggers(heal_log, fmt)
    if len(spent) >= HEAL_ROUNDS:
        why = "; ".join(reasons)
        if not (deviation_id or "").strip():
            raise _stop(
                f"{why}; the heal round for {fmt} is already spent ({heal_log}) and a "
                "second heal of the same format needs a deviation id"
            )
        if any(e.get("deviation_id") == deviation_id for e in spent):
            raise _stop(
                f"{why}; deviation {deviation_id} already paid for a heal round of {fmt} "
                f"({heal_log}); a further round needs its own deviation"
            )
    if not train_file.is_file():
        raise _refuse(
            f"the frozen training set {train_file} does not exist; heal trains on it",
            "run the distill assemble stage first",
        )
    return len(spent) + 1


def _heal_trigger(
    hr: _HealRun,
    fmt: str,
    round_: int,
    reasons: Sequence[str],
    measured: Mapping[str, Mapping[str, Any]],
    heal_set: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "event": "heal_trigger",
        "format": fmt,
        "heal_round": round_,
        "deviation_id": hr.deviation_id if round_ > HEAL_ROUNDS else None,
        "reasons": list(reasons),
        "drop_points": measured[fmt]["drop_points"],
        "agreement_vs_bf16": _pooled_agreement(measured[fmt]),
        "agreement_min": agreement_min(hr.reg),
        "bf16_accuracy": measured[BF16]["accuracy"],
        "quant_accuracy": measured[fmt]["accuracy"],
        "margin_points": HEAL_MARGIN_POINTS,
        "heal_knobs": heal_knobs(hr.reg),
        "heal_repeat": dict(hr.repeat) if hr.repeat else None,
        "heal_mix": hr.mix,
        "heal_mix_registered": heal_mix(hr.reg),
        "heal_source": hr.reg.raw["heal"]["source"],
        "heal_set": dict(heal_set),
    }


def healed_method(checkpoint: Path) -> dict[str, Any]:
    """Which heal ran (``qat`` or the ``bf16_continuation`` fallback), from the checkpoint."""
    path = Path(checkpoint) / td.CONFIG_FILE
    heal = None
    if path.is_file():
        heal = json.loads(path.read_text(encoding="utf-8")).get("heal")
    if not isinstance(heal, dict) or heal.get("method") not in (td.HEAL_QAT, td.HEAL_BF16):
        raise _refuse(
            f"{path} does not record which heal method ran (qat or bf16_continuation)",
            "heal with the distill trainer's heal mode, which records it",
        )
    return {
        "method": heal["method"],
        "fallback_reason": heal.get("fallback_reason"),
        "fake_quant": heal.get("fake_quant"),
        "target_sources": heal.get("target_sources"),
    }


def _healed_problem(
    fmt: str, again: Mapping[str, Any], bars: Mapping[str, Any], minimum: float
) -> str | None:
    """Why the healed build is not shippable (``None``: it is)."""
    if not again["available"] or not again["safe"]:
        return f"the healed {fmt} could not be measured cleanly: {again.get('reason')}"
    if again["drop_points"] is None or again["drop_points"] > HEAL_MARGIN_POINTS + _EPS:
        return (
            f"second heal trigger: the healed {fmt} still loses {again['drop_points']} "
            f"points against its own bf16 (> {HEAL_MARGIN_POINTS})"
        )
    if again["ece"] is None or again["ece"] > float(bars["ece_max"]) + _EPS:
        return f"the healed {fmt} has calibrated ECE {again['ece']} > {bars['ece_max']}"
    agreement = _pooled_agreement(again)
    if agreement is None or agreement < minimum - _EPS:
        return (
            f"the healed {fmt} agrees {agreement} with the bf16 student on fit+selection, "
            f"below the {minimum} bar; it is not shippable"
        )
    if not _guard_holds(again):
        return f"the healed {_guard_text(again)}; it is not shippable"
    return None


def _heal_round(
    hr: _HealRun,
    fmt: str,
    reasons: Sequence[str],
    measured: Mapping[str, Mapping[str, Any]],
    bf16_predictions: Mapping[str, Any],
    trainer: Trainer,
) -> _Healed:
    """One heal round of *fmt*: heal set, trigger, train, re-quantize, re-measure, verdict.

    A healed build that is not shippable is returned with its ``problem`` (the caller records
    the failed ``no qualifying format`` result); nothing past the heal log is written here."""
    round_ = _check_heal_allowed(fmt, reasons, hr.heal_log, hr.train_file, hr.deviation_id)
    root = heal_root(hr.out_dir, fmt, round_)
    protected = protected_digests(hr.fold_records, hr.sealed)
    heal_set = write_heal_set(  # refuses first
        hr.train_file,
        root / HEAL_SET_FILE,
        protected,
        repeat=hr.repeat,
        sources=hr.gating.sources if hr.gating else frozenset(),
    )
    spec = {
        "format": fmt,
        "mix": hr.mix,
        "fake_quant": fake_quant_scheme(fmt, measured[fmt].get("artifact")),
        "protected_id_sha256": protected,
        "bf16_student": str(hr.winner["checkpoint"]),
        "heal_set_sha256": heal_set["sha256"],
    }
    _write_json(root / HEAL_SPEC_FILE, spec)
    plan = hr.plan(fmt, round_)
    trigger = _heal_trigger(hr, fmt, round_, reasons, measured, heal_set)
    _append(hr.heal_log, trigger)  # logged before anything heals
    trainer(plan)
    healed_checkpoint = Path(plan["output_dir"])
    dtrain.check_raw_readout(healed_checkpoint)
    method = healed_method(healed_checkpoint)
    healed, healed_predictions = _measure_all(
        healed_checkpoint,
        root / BUILDS_DIR,
        hr.builders,
        hr.measurer,
        hr.fold_records,
        (fmt,),
        reference=bf16_predictions,
        gating=hr.gating,
        **hr.kw,
    )
    again = healed[fmt]
    problem = _healed_problem(fmt, again, quant_bars(hr.reg), agreement_min(hr.reg))
    _append(
        hr.heal_log,
        {
            "event": "heal_result",
            "format": fmt,
            "heal_round": round_,
            "method": method["method"],
            "fallback_reason": method["fallback_reason"],
            "drop_points": again.get("drop_points"),
            "agreement_vs_bf16": _pooled_agreement(again),
            "shippable": problem is None,
            "problem": problem,
        },
    )
    record = {**trigger, "healed_checkpoint": str(healed_checkpoint), "method": method}
    return _Healed(record, healed, healed_predictions, healed_checkpoint, problem)


@dataclass(frozen=True)
class _Healed:
    """What a heal round produced: its record, measurements, predictions and its verdict."""

    record: dict[str, Any]
    measured: dict[str, dict[str, Any]]
    predictions: Mapping[str, Mapping[str, Any]]
    checkpoint: Path
    #: why the healed build is not shippable (``None``: it is)
    problem: str | None = None


def _heal_plans(hr: _HealRun, formats: Sequence[str]) -> dict[str, Any]:
    """The dry run's heal plan per buildable quantized format (its next round)."""
    return {
        f: hr.plan(f, heal_rounds_spent(hr.heal_log, f) + 1)
        for f in formats
        if f in hr.builders and f != BF16
    }


def _check_repeat(
    repeat: Mapping[str, Any] | None, gating: Gating | None, deviation_id: str | None
) -> dict[str, int] | None:
    """A deviation's heal-set repeat counts ``{"domain": n, "guard": m}``, validated.

    Repeating rows changes how much each side weighs in the heal without touching the
    pre-registered heal epochs or lr; it needs the gating knob (which names the domain) and a
    deviation id."""
    if repeat is None:
        return None
    if not (deviation_id or "").strip():
        raise _refuse(
            "a heal repeat needs a deviation id",
            "record a deviation and pass --deviation-id with the heal_repeat knob",
        )
    if gating is None:
        raise _refuse("heal_repeat needs the gating knob (it names the domain sources)")
    if not isinstance(repeat, Mapping) or sorted(repeat) != ["domain", "guard"]:
        raise _refuse('knob heal_repeat must be {"domain": n, "guard": m}')
    for side, n in repeat.items():
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 10):
            raise _refuse(f"heal_repeat.{side} must be an integer in [1, 10], not {n!r}")
    return {"domain": int(repeat["domain"]), "guard": int(repeat["guard"])}


def _checked_knobs(knobs: QuantKnobs, deviation_id: str | None) -> QuantKnobs:
    """*knobs* with the heal mix and repeat validated; a deviation-gated knob needs an id."""
    heal_mix_override = knobs.heal_mix
    if heal_mix_override is not None:
        if not (deviation_id or "").strip():
            raise _refuse(
                "a heal mix other than the registered heal.mix needs a deviation id",
                "record a deviation and pass --deviation-id with the heal_mix knob",
            )
        heal_mix_override = td.check_mix(heal_mix_override, where="knob heal_mix")
    heal_repeat = _check_repeat(knobs.heal_repeat, knobs.gating, deviation_id)
    if knobs.gating is not None and not (deviation_id or "").strip():
        raise _refuse(
            "rules read from the gating sources only change the locked rule; they need a "
            "deviation id",
            "record a deviation and pass --deviation-id with the gating knob",
        )
    _refuse_comparison_candidates(knobs.formats)
    unknown = [f for f in knobs.formats if f not in FORMATS]
    if unknown:
        raise _refuse(f"unknown deployed format(s): {unknown}; candidates are {list(FORMATS)}")
    return replace(knobs, heal_mix=heal_mix_override, heal_repeat=heal_repeat)


def _gating_record(gating: Gating | None) -> dict[str, Any] | None:
    if gating is None:
        return None
    return {
        "sources": sorted(gating.sources),
        "guard_floor": gating.guard_floor,
        "guard_stock": gating.guard_stock,
    }


@dataclass(frozen=True)
class _Deployed:
    """What the deployed build is after the (optional) heal round."""

    checkpoint: Any
    predictions: Any
    heal_record: dict[str, Any] | None = None
    healed: dict[str, dict[str, Any]] | None = None


def _heal_if_needed(
    hr: _HealRun,
    *,
    base: Mapping[str, Any],
    choice: Mapping[str, Any],
    reasons: Sequence[str],
    measured: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Any],
    hooks: QuantHooks,
    limits: GpuLimits,
) -> _Deployed:
    """Run the one heal round when *reasons* call for it; a failed heal records and stops."""
    fmt = choice["format"]
    if not reasons:
        return _Deployed(checkpoint=hr.winner["checkpoint"], predictions=predictions[fmt])
    trainer = hooks.trainer or _default_trainer(
        allow_foreign=limits.allow_foreign,
        memory_max=limits.memory_max,
        memory_floor=limits.memory_floor,
        timeout=limits.timeout,
    )
    heal = _heal_round(hr, fmt, reasons, measured, predictions[BF16], trainer)
    if heal.problem:
        failed = {**choice, "reason": f"{NO_QUALIFYING}: {heal.problem}"}
        _fail(base=base, out_dir=hr.out_dir, measured=measured, healed=heal.measured, choice=failed)
    return _Deployed(
        checkpoint=heal.checkpoint,
        predictions=heal.predictions[fmt],
        heal_record=heal.record,
        healed=heal.measured,
    )


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
    bootstrap: Bootstrap = Bootstrap(),
    probe: Sequence[Record] = (),
    knobs: QuantKnobs = QuantKnobs(),
) -> dict[str, Any]:
    """Plan (dry-run) or run the format spike, the heal round and the recalibration.

    *probe* (the operator's private probe set, never a sealed record) is asked of the pruned
    comparison build only, to count its out-of-vocabulary refusals (risk r14).
    *knobs* carries the format candidates and the deviation-gated overrides
    (:class:`QuantKnobs`); a ``heal_mix`` replaces the registered ``heal.mix`` and needs a
    deviation id; both mixes are recorded in the heal log."""
    prereg_path, run_dir = Path(prereg_path), Path(run_dir)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    bars = quant_bars(reg)
    minimum = agreement_min(reg)
    heal_mix(reg)  # refused before anything runs when the heal block names no mix
    knobs = _checked_knobs(knobs, deviation_id)
    formats, gating = knobs.formats, knobs.gating
    winner = read_winner(run_dir)
    dtrain.check_raw_readout(winner["checkpoint"])
    fold_records = dtrain._folds(folds, sealed)
    probe = bench.check_records(probe, sealed=sealed, origin="probe set") if probe else []
    out_dir = run_dir / QUANT_REL
    kw = {
        "threshold": dtrain._threshold(reg),
        "seed": bootstrap.seed,
        "resamples": bootstrap.resamples,
    }
    hr = _HealRun(
        reg=reg,
        prereg_path=prereg_path,
        run_dir=run_dir,
        out_dir=out_dir,
        winner=winner,
        train_file=Path(train_file) if train_file else run_dir / TRAIN_SET_REL,
        trainer_env=trainer_env,
        deviation_id=deviation_id,
        fold_records=fold_records,
        sealed=tuple(sealed),
        builders=dict(hooks.builders or {}),
        measurer=hooks.measurer,  # type: ignore[arg-type]
        kw=kw,
        mix_override=knobs.heal_mix,
        gating=gating,
        repeat=knobs.heal_repeat,
    )
    base = {
        "stage": "distill-quant",
        "prereg_sha256": dp.sha256_file(prereg_path),
        "winner": {k: str(v) if isinstance(v, Path) else v for k, v in winner.items()},
        "folds": {f: len(fold_records[f]) for f in dtrain.FOLDS},
        "heal_margin_points": HEAL_MARGIN_POINTS,
        "heal_rounds_max": HEAL_ROUNDS,
        "heal_mix": hr.mix,
        "heal_mix_registered": heal_mix(reg),
        "gating": _gating_record(gating),
        "agreement_min": minimum,
        "agreement_folds": list(AGREEMENT_FOLDS),
        "size_budget_bytes": SIZE_BUDGET_BYTES,
        "format_rule": FORMAT_RULE,
    }
    if not apply:
        return _dry_run(base, formats, hr.builders, _heal_plans(hr, formats))
    _check_apply_inputs(hooks.measurer, hr.builders)
    measured, predictions = _measure_all(
        winner["checkpoint"],
        out_dir / BUILDS_DIR,
        hr.builders,
        hr.measurer,
        fold_records,
        formats,
        gating=gating,
        **kw,
    )
    choice = choose_format(measured, ece_max=float(bars["ece_max"]), formats=formats)
    if choice["format"] is None:
        _fail(base=base, out_dir=out_dir, measured=measured, healed=None, choice=choice)
    reasons = heal_reasons(choice["format"], measured, choice, minimum)
    deployed = _heal_if_needed(
        hr,
        base=base,
        choice=choice,
        reasons=reasons,
        measured=measured,
        predictions=predictions,
        hooks=hooks,
        limits=limits,
    )
    comparison = compare_pruned(
        measured,
        choice["format"],
        checkpoint=winner["checkpoint"],
        out_dir=out_dir / COMPARISON_DIR,
        builders=hr.builders,
        measurer=hr.measurer,
        fold_records=fold_records,
        train_file=hr.train_file,
        probe=probe,
        **kw,
    )
    return _finish(
        base=base,
        out_dir=out_dir,
        measured=measured,
        healed=deployed.healed,
        choice={**choice, "heal_reasons": reasons},
        heal_record=deployed.heal_record,
        heal_log=hr.heal_log,
        deployed_checkpoint=deployed.checkpoint,
        comparison=comparison,
        deployed_predictions=deployed.predictions,
    )


def _write_measurements(
    base: Mapping[str, Any],
    out_dir: Path,
    measured: Mapping[str, Mapping[str, Any]],
    healed: Mapping[str, Mapping[str, Any]] | None,
    comparison: Mapping[str, Any] | None,
) -> Path:
    path = out_dir / MEASUREMENTS_FILE
    _write_json(
        path,
        {
            "prereg_sha256": base["prereg_sha256"],
            "fold": "selection",
            "agreement_folds": list(AGREEMENT_FOLDS),
            "gating": base.get("gating"),
            "formats": measured,
            "healed": healed,
            "pruned_vocabulary_comparison": comparison,
        },
    )
    return path


def _fail(
    *,
    base: Mapping[str, Any],
    out_dir: Path,
    measured: Mapping[str, Mapping[str, Any]],
    healed: Mapping[str, Mapping[str, Any]] | None,
    choice: Mapping[str, Any],
) -> None:
    """Record the failed ``no qualifying format`` result, ship nothing and stop (risk r10).

    ``measurements.json``, ``format_decision.json`` (``format: null``) and
    ``quant_manifest.json`` (``status: failed``, ``shipped_format: null``) are written so the
    failure is on the record; no ``calibration.json`` and no deployed predictions are."""
    stale = [out_dir / CALIBRATION_FILE, out_dir / DEPLOYED_PREDICTIONS_FILE]
    for path in stale:  # a deployable output of an earlier run must not outlive this verdict
        path.unlink(missing_ok=True)
    measurements_path = _write_measurements(base, out_dir, measured, healed, None)
    decision = {
        "decision": "deployed_format",
        "outcome": NO_QUALIFYING,
        "format": None,
        "reason": choice["reason"],
        "rule": FORMAT_RULE,
        "trail": choice["trail"],
        "unavailable": {f: m["reason"] for f, m in measured.items() if not m["available"]},
        "cited": _cited(measured, measurements_path),
        "prereg_sha256": base["prereg_sha256"],
    }
    decision_path = out_dir / DECISION_FILE
    _write_json(decision_path, decision)
    _write_json(
        out_dir / MANIFEST_FILE,
        {
            **base,
            "dry_run": False,
            "status": "failed",
            "outcome": NO_QUALIFYING,
            "reason": choice["reason"],
            "chosen_format": None,
            "shipped_format": None,
            "shippable": False,
            "deployed_artifact": None,
            "outputs": _outputs([measurements_path, decision_path, out_dir / HEAL_LOG_FILE]),
        },
    )
    raise _stop(choice["reason"])


def _deployed_predictions_doc(
    fmt: str, artifact: Any, predictions: Mapping[str, Any]
) -> dict[str, Any]:
    """The shipped build's selection-fold predictions and the bundle that produced them."""
    try:
        att = qz.bundle_attestation(artifact)
        bundle = {"files": dict(att["files"]), "manifest_sha256": att["manifest_sha256"]}
    except (OSError, ValueError, KeyError, CliError):
        bundle = None
    return {
        "format": fmt,
        "artifact": str(artifact),
        "bundle": bundle,
        "fold": "selection",
        "predictions": list(predictions.get("selection") or []),
    }


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
    deployed_predictions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fmt = choice["format"]
    _refuse_comparison_candidates([fmt])
    final = (healed or measured)[fmt]
    final_bf16 = (healed or measured)[BF16]
    measurements_path = _write_measurements(base, out_dir, measured, healed, comparison)
    predictions_path = out_dir / DEPLOYED_PREDICTIONS_FILE
    _write_json(
        predictions_path,
        _deployed_predictions_doc(fmt, final["artifact"], deployed_predictions or {}),
    )
    cited = _cited(measured, measurements_path)
    if healed:
        cited += [{**c, "name": "healed." + c["name"]} for c in _cited(healed, measurements_path)]
    decision = {
        "decision": "deployed_format",
        "format": fmt,
        "description": FORMATS.get(fmt, "bf16 (unquantized)"),
        "reason": choice["reason"],
        "heal_reasons": list(choice.get("heal_reasons") or []),
        "rule": FORMAT_RULE,
        "trail": choice["trail"],
        "unavailable": {f: m["reason"] for f, m in measured.items() if not m["available"]},
        "cited": cited,
        "prereg_sha256": base["prereg_sha256"],
        "rule_owner": "jev_factory.decide.rules (HEAL_MARGIN_POINTS, HEAL_ROUNDS)",
        "never_candidates": dict.fromkeys(COMPARISON_FORMATS, COMPARISON_NOTE),
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
        "shipped_format": fmt,
        "deployed_checkpoint": str(deployed_checkpoint),
        "deployed_artifact": final["artifact"],
        "deployed_size": final.get("size"),
        "heal_count": 1 if heal_record else 0,
        "heal": heal_record,
        "heal_method": (heal_record or {}).get("method"),
        "quant_vs_bf16_delta_points": 0.0 if fmt == BF16 else delta,
        "quant_vs_bf16_initial_delta_points": (
            None if fmt == BF16 else measured[fmt].get("delta_points")
        ),
        "quant_vs_bf16_agreement": None if fmt == BF16 else final.get("agreement_vs_bf16"),
        # a quant reaches here only at or above the bar (else it healed or stopped)
        "shippable": True,
        "bf16_selection_accuracy": final_bf16["accuracy"],
        "deployed_selection_accuracy": final["accuracy"],
        "deployed_selection_ece": final["ece"],
        "pruned_vocabulary_comparison": (
            None
            if comparison is None
            else {k: comparison[k] for k in comparison if k != "measurement"}
        ),
        "outputs": _outputs(
            [measurements_path, decision_path, calibration_path, predictions_path, heal_log]
        ),
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
    allow_foreign: bool = False,
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
        gpu.assert_gpu_free("measure", allow_foreign=allow_foreign)
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
