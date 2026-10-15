"""Distill trainer for Strands Decider: continue from the stock model on teacher labels and gold.

Two halves in one module:

* **Planning, here (stdlib only).** :func:`plan_train` reads every hyperparameter from the
  locked distill pre-registration (:func:`jev_factory.distill.prereg.require_locked`) and
  returns the trainer's subprocess command without running anything. :func:`run_train` is
  dry-run unless ``apply``; with ``apply`` the command runs through
  :func:`jev_factory.factory.gpu.run_gpu_stage` (residency guard, memory cap, memory-floor
  watchdog) and nowhere else.
* **Training, in the trainer env.** ``python -m jev_factory.backbones.pointer.train_distill``
  (:func:`main`) re-checks the lock itself, loads the stock checkpoint with upstream's
  ``StrandsDeciderModel.load``, renders rows with upstream's ``SystemOneCollator`` (the
  serving prompt and option-position readout) and minimises
  ``alpha * KL(teacher || student at T_kind) + (1 - alpha) * CE(gold)``
  (:mod:`.distill_loss`) on LoRA plus the pointer head. The heavy packages (torch, peft,
  safetensors, strands_decider and the checkout's ``training`` package) are loaded with
  :func:`importlib.import_module` inside the training functions only: the jev process never
  loads them, and this module imports cleanly with none of them installed.

Heal mode (``--heal SPEC``, written by :mod:`jev_factory.distill.quant`; spec c10, h10): the
rows are the heal set (train records only; any row whose record id hashes into the spec's
protected ids, the test, held-out, fit and selection sides, exits non-zero before a model is
loaded). The bf16 distilled student (``--init-from``) first labels every row with its own
per-option distribution; each row's soft target is the pre-registered ``mix`` of that and
Clef's distribution (:func:`mix_targets`), and the loss is the KL to it alone. The student is
then trained **quantization-aware**: every text-block linear computes its output from a
fake-quantized ``W + scale * B A`` at the deployed format's bits and group size (straight-
through gradients, :func:`fake_quant`), and the embedding output is fake-quantized the same
way, so the LoRA and the pointer head learn to absorb the quantization error. When that
cannot be set up on the student env (no fake-quant scheme for the format, no LoRA-wrapped
block linear, or a probe forward fails) the heal falls back to nvsh d18's plain bf16
continuation on the same targets; which one ran, and why, is recorded in
``distill_train_config.json`` (``heal.method``). The checkpoint format carries the LoRA
adapter, the head and the config only, so norms are not trained (they would not be saved).

Init modes (a pre-registered knob, ``knob_grid.init_mode``; frame park v6):

* ``continue``: train the stock model's own LoRA adapter (made trainable again; upstream
  ``load`` opens it frozen) and the stock model's own pointer head. The adapter's shape is the
  stock model's, so no ``lora_*`` knob is read.
* ``merge_fresh``: merge the stock model's adapter into the torso (``merge_and_unload``),
  attach a fresh LoRA of the pre-registered ``lora_r``/``lora_alpha``/``lora_dropout``/
  ``lora_targets``, and train it with the stock model's head. ``strands-decider serve`` loads
  one adapter over the *unmerged* base, so at save time the stock model's adapter and the
  fresh one are stacked into one adapter of rank ``r_stock + r_fresh`` (:func:`stack_lora`,
  exact: ``B A`` of the stack is the sum of both scaled deltas). The fresh adapter was trained
  against the bf16 merge, so the served model can differ from the trained one by that merge's
  rounding; the final measurement is taken on the served artifact.

Unlike upstream ``init_from`` (which discards the loaded head for a fresh ``SlotHead``),
both modes keep the stock model's pointer head. Upstream's ``kl_frozen_weight`` (KL to the
untouched torso's option-number readout) and ``teacher_weight`` (an additive teacher term) are not
applied: the distill objective is the pre-registered alpha mixture only, and the plan says
so under ``not_applied``. Training rows are the JSONL items :func:`load_items` reads (one
per record and question, canonical option order).

Upstream reuse (strands-labs/strands-decider @ 8b6faa553038a59e73fa7f16c4081f8ceb21d27e,
Copyright the strands-decider authors, Apache License 2.0; see :data:`UPSTREAM_PROVENANCE`):
``training/train.py`` ``_build_optimizer`` (head / adapter parameter groups, no decay on
norms) and ``_lr_lambda`` (warm-up then cosine) are called as-is; its micro-batch, gradient
accumulation and clipping loop and its upstream knob names are mirrored here; the teacher-KL
masking is re-expressed in :mod:`.distill_loss`. ``training/data/collate.py``
``SystemOneCollator`` and ``CollatorConfig`` render and tokenise every batch unchanged (a
subclass only records each row's option order, so targets follow the same permutation as
upstream's ``teacher`` tensor); ordinal smoothing is re-expressed in
:func:`.distill_loss.gold_target`. ``src/strands_decider/modeling.py``
``StrandsDeciderModel.load``/``attach_lora``/``save_pretrained`` load and save the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.pointer import distill_loss as dl
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill import prereg as dp
from jev_factory.factory import gpu, safepath

MODULE = "jev_factory.backbones.pointer.train_distill"
STAGE = "train"
INIT_MODES = ("continue", "merge_fresh")
QUESTION_KINDS = ("choice", "noul", "score")
CONFIG_FILE = "distill_train_config.json"
HISTORY_FILE = "distill_history.json"
TEACHER_SUM_TOLERANCE = 0.01

UPSTREAM_PROVENANCE: dict[str, Any] = {
    "repo": "strands-labs/strands-decider",
    "commit": "8b6faa553038a59e73fa7f16c4081f8ceb21d27e",
    "licence": "Apache-2.0",
    "files": {
        "training/train.py": (
            "_build_optimizer and _lr_lambda called as-is; micro-batch / grad_accum / clip loop "
            "and knob names mirrored; _kl/_teacher_kl forward masking re-expressed in "
            "distill_loss"
        ),
        "training/data/collate.py": (
            "SystemOneCollator and CollatorConfig used unchanged (subclass records option "
            "order); _target_distribution ordinal smoothing re-expressed in "
            "distill_loss.gold_target"
        ),
        "src/strands_decider/modeling.py": (
            "StrandsDeciderModel.load / attach_lora / save_pretrained used unchanged"
        ),
    },
}

#: Searched knobs the trainer needs: chosen by the run from the grid (or a one-value grid).
SEARCHED = ("alpha", "temperature", "epochs", "lr", "init_mode")
#: Fixed hyperparameters every run needs, from ``upstream_knobs`` (or a grid, if registered).
REQUIRED_UPSTREAM = (
    "head_lr",
    "ordinal_smoothing",
    "micro_batch_size",
    "grad_accum",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "max_length",
    "seed",
    "shuffle_options",
    "reverse_score_prob",
)
#: The fresh adapter's shape; read only for ``merge_fresh``.
LORA_UPSTREAM = ("lora_r", "lora_alpha", "lora_dropout", "lora_targets")
#: Upstream terms the distill objective replaces; reported, never silently dropped.
SUPERSEDED = {
    "kl_frozen_weight": (
        "KL to the untouched torso's option-number readout is not part of the pre-registered "
        "distill loss (alpha * KL(teacher) + (1 - alpha) * CE(gold))"
    ),
    "teacher_weight": "replaced by alpha, which mixes teacher KL and gold CE",
}
_POS_INT = ("epochs", "micro_batch_size", "grad_accum", "max_length", "lora_r")
_NON_NEG = (
    "lr",
    "head_lr",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "lora_alpha",
    "lora_dropout",
)
_UNIT = ("alpha", "ordinal_smoothing", "reverse_score_prob")

#: The heal's soft-target sources, as the pre-registered ``heal.mix`` names them.
HEAL_MIX_KEYS = ("bf16_student", "clef")
HEAL_SPEC_KEYS = ("format", "mix", "fake_quant", "protected_id_sha256", "bf16_student")
HEAL_TARGETS_FILE = "heal_targets.jsonl"
#: The heal methods: quantization-aware training, else nvsh d18's bf16 continuation.
HEAL_QAT, HEAL_BF16 = "qat", "bf16_continuation"
#: The heal trains on the KL to its mixed soft targets only (no gold CE term).
HEAL_ALPHA = 1.0


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation
        or "fix the distill pre-registration or the run knobs; a change after "
        "the lock is a recorded deviation",
    )


# --------------------------------------------------------------------- hyperparameters


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pick(
    name: str, reg: dp.DistillPrereg, knobs: Mapping[str, Any], upstream: Mapping[str, Any]
) -> tuple[Any, str]:
    grid = reg.knob_grid
    if name in grid:
        if name in knobs:
            return knobs[name], "knob_grid"
        values = grid[name]
        if name == "temperature" and isinstance(values, dict):
            if all(isinstance(v, list) and len(v) == 1 for v in values.values()):
                return {k: v[0] for k, v in values.items()}, "knob_grid"
        elif isinstance(values, list) and len(values) == 1:
            return values[0], "knob_grid"
        raise _refuse(
            f"knob {name} has several pre-registered values and the run chose none",
            f"pass {name} in the run knobs (one of the registered values)",
        )
    if name in upstream:
        knob = upstream[name]
        return knob.value, f"upstream:{knob.source_file}#{knob.source_key}"
    raise _refuse(f"hyperparameter {name} is not in the distill pre-registration")


def _check_temperature(temp: Any) -> None:
    if not isinstance(temp, dict) or sorted(temp) != sorted(QUESTION_KINDS):
        raise _refuse(f"temperature must give one value per question kind {list(QUESTION_KINDS)}")
    if not all(_is_num(v) and v > 0 for v in temp.values()):
        raise _refuse("temperature values must be positive numbers")


def _check_numeric_ranges(hp: Mapping[str, Any]) -> None:
    for name in _POS_INT:
        if name in hp and not (_is_int(hp[name]) and hp[name] >= 1):
            raise _refuse(f"{name} must be a positive integer, got {hp[name]!r}")
    for name in _NON_NEG:
        if name in hp and not (_is_num(hp[name]) and hp[name] >= 0):
            raise _refuse(f"{name} must be a non-negative number, got {hp[name]!r}")
    for name in _UNIT:
        if not (_is_num(hp[name]) and 0.0 <= hp[name] <= 1.0):
            raise _refuse(f"{name} must be a number in [0, 1], got {hp[name]!r}")


def _check_lora_targets(targets: Any) -> None:
    if targets is not None and not (
        isinstance(targets, list) and targets and all(isinstance(t, str) and t for t in targets)
    ):
        raise _refuse("lora_targets must be a non-empty list of module names")


def _check_values(hp: Mapping[str, Any]) -> None:
    if hp["init_mode"] not in INIT_MODES:
        raise _refuse(f"init_mode {hp['init_mode']!r} is not one of {list(INIT_MODES)}")
    _check_temperature(hp["temperature"])
    _check_numeric_ranges(hp)
    if not _is_int(hp["seed"]):
        raise _refuse(f"seed must be an integer, got {hp['seed']!r}")
    if not isinstance(hp["shuffle_options"], bool):
        raise _refuse("shuffle_options must be true or false")
    _check_lora_targets(hp.get("lora_targets"))


def resolve_hyperparameters(reg: dp.DistillPrereg, run_knobs: Mapping[str, Any]) -> dict[str, Any]:
    """Every trainer hyperparameter, each from the pre-registration, with its source.

    Run knobs are checked against the registration first (name and value). A grid knob is
    the run's choice, or the grid's only value; any other name comes from ``upstream_knobs``.
    Anything missing is refused, never defaulted.
    """
    dp.require_knobs(reg, dict(run_knobs))
    upstream = {k.name: k for k in reg.upstream_knobs}
    hp: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in SEARCHED + REQUIRED_UPSTREAM:
        hp[name], sources[name] = _pick(name, reg, run_knobs, upstream)
    if hp["init_mode"] == "merge_fresh":
        for name in LORA_UPSTREAM:
            hp[name], sources[name] = _pick(name, reg, run_knobs, upstream)
    _check_values(hp)
    hp["sources"] = sources
    return hp


def resolve_from_locked(
    prereg_path: Path | str,
    run_dir: Path | str,
    run_knobs: Mapping[str, Any],
    deviation_id: str | None = None,
) -> dict[str, Any]:
    """:func:`resolve_hyperparameters` on the locked registration (refused without a lock)."""
    reg = dp.require_locked(Path(prereg_path), Path(run_dir), deviation_id)
    return resolve_hyperparameters(reg, run_knobs)


def _not_applied(reg: dp.DistillPrereg) -> dict[str, str]:
    present = {k.name for k in reg.upstream_knobs} | set(reg.knob_grid)
    return {name: why for name, why in SUPERSEDED.items() if name in present}


# --------------------------------------------------------------------- plan / run


def _jev_root() -> Path:
    return Path(dl.__file__).resolve().parents[3]


def plan_train(
    *,
    prereg_path: Path | str,
    run_dir: Path | str,
    run_knobs: Mapping[str, Any],
    train_file: Path | str,
    init_from: Path | str,
    output_dir: Path | str,
    trainer_python: str,
    decider_repo: Path | str,
    deviation_id: str | None = None,
    heal_spec: Path | str | None = None,
) -> dict[str, Any]:
    """The trainer's subprocess command and resolved hyperparameters; runs nothing.

    With *heal_spec* (a heal spec written by :mod:`jev_factory.distill.quant`) the trainer
    runs in heal mode on *train_file*, the heal set (see the module docstring).

    *init_from* is the verified stock checkpoint directory (``lora/``, ``head.safetensors``,
    ``strands_decider_config.json``); *decider_repo* is the pinned strands-decider checkout
    whose ``training`` package and ``src`` go on the trainer's ``PYTHONPATH``.
    """
    reg = dp.require_locked(Path(prereg_path), Path(run_dir), deviation_id)
    hp = resolve_hyperparameters(reg, run_knobs)
    knobs_json = json.dumps(dict(run_knobs), sort_keys=True)
    argv = [
        str(trainer_python),
        "-m",
        MODULE,
        "--prereg",
        str(prereg_path),
        "--run-dir",
        str(run_dir),
        "--knobs",
        knobs_json,
        "--train-file",
        str(train_file),
        "--init-from",
        str(init_from),
        "--output-dir",
        str(output_dir),
    ]
    if deviation_id:
        argv += ["--deviation", str(deviation_id)]
    if heal_spec is not None:
        argv += ["--heal", str(heal_spec)]
    repo = Path(decider_repo)
    pythonpath = os.pathsep.join([str(repo), str(repo / "src"), str(_jev_root())])
    return {
        "backbone": "pointer",
        "stage": STAGE,
        "dry_run": True,
        "argv": argv,
        "env": {"PYTHONPATH": pythonpath},
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "init_from": str(init_from),
        "train_file": str(train_file),
        "prereg_sha256": dp.sha256_file(Path(prereg_path)),
        "hyperparameters": hp,
        "not_applied": _not_applied(reg),
        "upstream": {k: UPSTREAM_PROVENANCE[k] for k in ("repo", "commit", "licence")},
        "heal_spec": None if heal_spec is None else str(heal_spec),
    }


def run_train(
    plan: Mapping[str, Any],
    *,
    apply: bool = False,
    allow_foreign: bool = False,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Dry-run by default; with *apply*, run the planned command via ``run_gpu_stage``."""
    if not apply:
        return {**plan, "dry_run": True}
    if not memory_max:
        raise _refuse(
            "a GPU training run needs a memory cap", "pass memory_max (e.g. '100G') with apply"
        )
    env = {**os.environ, **plan["env"]}
    done = gpu.run_gpu_stage(
        STAGE,
        plan["run_dir"],
        list(plan["argv"]),
        allow_foreign=allow_foreign,
        memory_max=memory_max,
        memory_floor=memory_floor,
        env=env,
        timeout=timeout,
    )
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-5:]
        raise CliError(
            EXIT_ENV_ERROR,
            f"distill trainer exited {done.returncode}: {' | '.join(tail)}",
            "read the trainer log under the run directory; status 3 is the memory watchdog",
        )
    return {
        "dry_run": False,
        "returncode": done.returncode,
        "output_dir": plan["output_dir"],
        "stdout_tail": (done.stdout or "").strip().splitlines()[-5:],
    }


# --------------------------------------------------------------------- training rows


@dataclass(frozen=True)
class TrainItem:
    """One (record, question) training row, targets in canonical option order."""

    record_id: str
    question_id: str
    kind: str
    state: Any
    instructions: str
    options: list[list[str]]
    target: list[float] | None
    teacher: list[float] | None
    temperature: float
    weight: float = 1.0


def _bad(where: str, message: str) -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        f"{where}: {message}",
        "rebuild the distill training set; the trainer never repairs a row",
    )


def _distribution(where: str, what: str, value: Any, names: Sequence[str]) -> list[float]:
    if not isinstance(value, Mapping):
        raise _bad(where, f"{what} must map option names to probabilities")
    stray = sorted(set(value) - set(names))
    if stray:
        raise _bad(where, f"{what} names {stray} are not options {list(names)}")
    probs = [value.get(n, 0.0) for n in names]
    if not all(_is_num(p) and p >= 0 for p in probs):
        raise _bad(where, f"{what} has a negative or non-numeric probability")
    total = sum(probs)
    if abs(total - 1.0) > TEACHER_SUM_TOLERANCE:
        raise _bad(where, f"{what} distribution sums to {total:.4f}, not 1")
    return [p / total for p in probs]


def _gold(where: str, kind: str, gold: Any, names: Sequence[str], eps: float) -> list[float]:
    if not isinstance(gold, Mapping):
        raise _bad(where, "gold must be an object or null")
    if gold.get("probabilities") is not None:
        return _distribution(where, "gold", gold["probabilities"], names)
    if kind == "noul" and gold.get("p_true") is not None:
        p = gold["p_true"]
        if not (_is_num(p) and 0.0 <= p <= 1.0):
            raise _bad(where, f"gold p_true {p!r} outside [0, 1]")
        return dl.noul_target(float(p))
    label = gold.get("label")
    if label not in names:
        raise _bad(where, f"gold label {label!r} is not one of the options {list(names)}")
    return dl.gold_target(kind, len(names), list(names).index(label), ordinal_smoothing=eps)


def _shape(where: str, raw: Any) -> tuple[str, list[Any], list[str], float]:
    """A row's kind, options, option names and weight, each checked."""
    if not isinstance(raw, Mapping):
        raise _bad(where, "not a JSON object")
    kind = raw.get("kind")
    if kind not in QUESTION_KINDS:
        raise _bad(where, f"unknown question kind {kind!r}; expected one of {QUESTION_KINDS}")
    options = raw.get("options")
    if not (
        isinstance(options, list)
        and len(options) >= 2
        and all(isinstance(o, list) and len(o) == 2 for o in options)
    ):
        raise _bad(where, "options must be a list of >= 2 [name, description] pairs")
    names = [str(o[0]) for o in options]
    if len(set(names)) != len(names):
        raise _bad(where, f"duplicate option names {names}")
    weight = raw.get("weight", 1.0)
    if not (_is_num(weight) and weight > 0):
        raise _bad(where, f"weight must be a positive number, got {weight!r}")
    return kind, options, names, float(weight)


def _train_item(
    raw: Mapping[str, Any],
    shape: tuple[str, list[Any], list[str], float],
    hp: Mapping[str, Any],
    target: list[float] | None,
    teacher: list[float] | None,
) -> TrainItem:
    kind, options, _names, weight = shape
    return TrainItem(
        record_id=str(raw.get("record_id", "")),
        question_id=str(raw.get("question_id", "")),
        kind=kind,
        state=raw.get("state", ""),
        instructions=str(raw.get("instructions") or ""),
        options=[[str(a), str(b)] for a, b in options],
        target=target,
        teacher=teacher,
        temperature=float(hp["temperature"][kind]),
        weight=weight,
    )


def _item(raw: Any, line: int, hp: Mapping[str, Any]) -> TrainItem:
    where = f"row {line}"
    shape = _shape(where, raw)
    kind, _options, names, _weight = shape
    gold, teacher = raw.get("gold"), raw.get("teacher")
    if gold is None and teacher is None:
        raise _bad(where, "the row has neither a gold target nor a teacher target")
    target = None if gold is None else _gold(where, kind, gold, names, hp["ordinal_smoothing"])
    tea = None if teacher is None else _distribution(where, "teacher", teacher, names)
    return _train_item(raw, shape, hp, target, tea)


def load_items(path: Path | str, hp: Mapping[str, Any]) -> list[TrainItem]:
    """Read and validate the training rows; the per-kind temperature is attached per row."""
    out: list[TrainItem] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line, text in enumerate(fh, 1):
            if not text.strip():
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise _bad(f"row {line}", f"not valid JSON: {exc}") from exc
            out.append(_item(raw, line, hp))
    if not out:
        raise _bad(str(path), "no training rows")
    return out


# --------------------------------------------------------------------- heal rows (pure)


@dataclass(frozen=True)
class HealRow:
    """One heal row: the row (no target yet) and Clef's distribution, canonical order."""

    item: TrainItem
    clef: list[float] | None


def id_digest(record_id: Any) -> str:
    """The sha256 of a record id: how a heal spec names its protected ids."""
    return hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()


def check_mix(mix: Any, where: str = "heal.mix") -> dict[str, float]:
    """The heal's soft-target mix: weights on :data:`HEAL_MIX_KEYS` in [0, 1] summing to 1."""
    if not isinstance(mix, Mapping) or sorted(mix) != sorted(HEAL_MIX_KEYS):
        raise _refuse(f"{where} must weigh exactly {list(HEAL_MIX_KEYS)}")
    if not all(_is_num(v) and 0.0 <= v <= 1.0 for v in mix.values()):
        raise _refuse(f"{where} weights must be numbers in [0, 1]")
    if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-9):
        raise _refuse(f"{where} weights must sum to 1, not {sum(mix.values())}")
    return {k: float(mix[k]) for k in HEAL_MIX_KEYS}


def mix_targets(
    student: Sequence[float], clef: Sequence[float] | None, mix: Mapping[str, float]
) -> tuple[list[float] | None, str]:
    """One heal row's soft target and its source: ``mix`` of the student's and Clef's.

    A row with no Clef distribution (an alternative-teacher, gold-only or corrected row)
    takes the bf16 student's alone; with a zero student weight such a row has no target.
    """
    w_student, w_clef = float(mix["bf16_student"]), float(mix["clef"])
    if clef is None:
        if w_student <= 0.0:
            return None, "dropped:no_clef"
        return [float(p) for p in student], "bf16_student_only:no_clef"
    if len(clef) != len(student):
        raise ValueError(f"student has {len(student)} options, Clef {len(clef)}")
    mixed = [w_student * float(s) + w_clef * float(c) for s, c in zip(student, clef)]
    total = sum(mixed)
    if total <= 0.0:
        raise ValueError("the mixed heal target has no mass")
    return [p / total for p in mixed], "bf16_student+clef"


def unpermute(slot: Sequence[float], order: Sequence[int] | None) -> list[float]:
    """Slot-order probabilities back to canonical order (``order[k]`` = canonical index)."""
    if order is None:
        return [float(p) for p in slot]
    if sorted(order) != list(range(len(slot))):
        raise ValueError(f"order {list(order)} is not a permutation of {len(slot)} slots")
    out = [0.0] * len(slot)
    for k, canonical in enumerate(order):
        out[canonical] = float(slot[k])
    return out


def _protected_refusal(hits: Sequence[str]) -> CliError:
    shown = ", ".join(sorted(hits)[:5]) + (" ..." if len(hits) > 5 else "")
    return CliError(
        EXIT_USER_ERROR,
        f"the heal input holds {len(hits)} protected record id(s) (test, held-out, fit or "
        f"selection): {shown}",
        "the heal trains on train records only; rebuild the heal set from the train side",
    )


def check_heal_ids(record_ids: Sequence[Any], protected: Sequence[str] | set[str]) -> None:
    """Refuse (exit non-zero) any record id whose sha256 is among *protected*."""
    digests = set(protected)
    hits = sorted({str(r) for r in record_ids if id_digest(r) in digests})
    if hits:
        raise _protected_refusal(hits)


def load_heal_rows(
    path: Path | str, hp: Mapping[str, Any], protected: Sequence[str] | set[str]
) -> list[HealRow]:
    """Read the heal set; refuse a protected record id before anything else is read."""
    raws: list[tuple[int, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line, text in enumerate(fh, 1):
            if not text.strip():
                continue
            try:
                raws.append((line, json.loads(text)))
            except json.JSONDecodeError as exc:
                raise _bad(f"heal row {line}", f"not valid JSON: {exc}") from exc
    check_heal_ids([r.get("record_id") for _, r in raws if isinstance(r, Mapping)], protected)
    rows: list[HealRow] = []
    for line, raw in raws:
        where = f"heal row {line}"
        shape = _shape(where, raw)
        clef = raw.get("clef")
        dist = None if clef is None else _distribution(where, "clef", clef, shape[2])
        rows.append(HealRow(_train_item(raw, shape, hp, None, None), dist))
    if not rows:
        raise _bad(str(path), "no heal rows")
    return rows


def heal_items(
    rows: Sequence[HealRow], student: Sequence[Sequence[float]], mix: Mapping[str, float]
) -> tuple[list[TrainItem], list[dict[str, Any]], dict[str, int]]:
    """Each row with its mixed soft target as the teacher (no gold); the per-row log; counts."""
    if len(rows) != len(student):
        raise ValueError("one student distribution per heal row is required")
    items: list[TrainItem] = []
    log: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row, dist in zip(rows, student):
        target, source = mix_targets(dist, row.clef, mix)
        counts[source] = counts.get(source, 0) + 1
        log.append(
            {
                "record_id": row.item.record_id,
                "question_id": row.item.question_id,
                "bf16_student": list(dist),
                "clef": row.clef,
                "target": target,
                "source": source,
            }
        )
        if target is not None:
            items.append(replace(row.item, target=None, teacher=target))
    return items, log, counts


def load_heal_spec(path: Path | str, run_dir: Path | str) -> dict[str, Any]:
    """The heal spec quant wrote: format, mix, fake-quant scheme, protected id hashes."""
    spec_path = safepath.regular_file(path, Path(run_dir))
    try:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _refuse(f"{spec_path} is not valid JSON: {exc}") from exc
    missing = [k for k in HEAL_SPEC_KEYS if not isinstance(spec, dict) or k not in spec]
    if missing:
        raise _refuse(f"the heal spec {spec_path} is missing: {', '.join(missing)}")
    spec["mix"] = check_mix(spec["mix"], "heal spec mix")
    ids = spec["protected_id_sha256"]
    if not (isinstance(ids, list) and all(isinstance(i, str) and len(i) == 64 for i in ids)):
        raise _refuse("the heal spec's protected_id_sha256 must be a list of sha256 hex digests")
    if spec["fake_quant"] is not None and not isinstance(spec["fake_quant"], dict):
        raise _refuse("the heal spec's fake_quant must be an object or null")
    return spec


# --------------------------------------------------------------------- LoRA stacking


def _modules(state: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    out: dict[str, tuple[Any, Any]] = {}
    for key in state:
        if key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            out[prefix] = (state[key], state[f"{prefix}.lora_B.weight"])
    return out


def stack_lora(
    first: Mapping[str, Any], first_scale: float, second: Mapping[str, Any], second_scale: float
) -> tuple[dict[str, Any], int, int]:
    """One adapter equal to two: ``A = [A1; A2]``, ``B = [s1 B1, s2 B2]``, scaling 1.

    Returns ``(state_dict, rank, lora_alpha)`` with ``lora_alpha == rank``. A module only
    one adapter targets gets zero rows/columns for the other, so every module has the same
    rank ``r1 + r2`` (a uniform PEFT config).
    """
    torch = importlib.import_module("torch")
    m1, m2 = _modules(first), _modules(second)
    if not m1 or not m2:
        raise ValueError("both adapters need lora_A/lora_B weights")
    r1 = next(iter(m1.values()))[0].shape[0]
    r2 = next(iter(m2.values()))[0].shape[0]
    state: dict[str, Any] = {}
    for prefix in sorted(set(m1) | set(m2)):
        ref_a, ref_b = m1.get(prefix, m2.get(prefix))  # type: ignore[misc]
        d_in, d_out = ref_a.shape[1], ref_b.shape[0]
        a1, b1 = m1.get(prefix, (torch.zeros(r1, d_in), torch.zeros(d_out, r1)))
        a2, b2 = m2.get(prefix, (torch.zeros(r2, d_in), torch.zeros(d_out, r2)))
        if a1.shape[0] != r1 or a2.shape[0] != r2:
            raise ValueError(f"{prefix}: adapters must have one rank per adapter")
        dtype = ref_a.dtype
        state[f"{prefix}.lora_A.weight"] = torch.cat([a1.float(), a2.float()], dim=0).to(dtype)
        state[f"{prefix}.lora_B.weight"] = torch.cat(
            [b1.float() * first_scale, b2.float() * second_scale], dim=1
        ).to(dtype)
    rank = int(r1 + r2)
    return state, rank, rank


# --------------------------------------------------------------------- trainer env


def _load(name: str) -> Any:
    return importlib.import_module(name)


def _lora_scale(torso: Any) -> float:
    cfg = torso.peft_config["default"]
    return float(cfg.lora_alpha) / float(cfg.r)


def prepare_init(model: Any, hp: Mapping[str, Any]) -> dict[str, Any]:
    """Make the stock model trainable per ``init_mode``; its pointer head is kept and trained."""
    info: dict[str, Any] = {"init_mode": hp["init_mode"]}
    if hp["init_mode"] == "continue":
        n = 0
        for name, p in model.torso.named_parameters():
            trainable = "lora_" in name
            p.requires_grad_(trainable)
            n += int(trainable)
        if not n:
            raise CliError(EXIT_ENV_ERROR, "the stock checkpoint has no LoRA parameters")
    else:
        peft = _load("peft")
        info["stock_state"] = {
            k: v.detach().clone() for k, v in peft.get_peft_model_state_dict(model.torso).items()
        }
        info["stock_scale"] = _lora_scale(model.torso)
        info["stock_targets"] = sorted(model.torso.peft_config["default"].target_modules)
        model.torso = model.torso.merge_and_unload()
        model.config.lora_r = int(hp["lora_r"])
        model.config.lora_alpha = int(hp["lora_alpha"])
        model.config.lora_dropout = float(hp["lora_dropout"])
        model.config.lora_targets = list(hp["lora_targets"])
        model.attach_lora()
    for p in model.head.parameters():
        p.requires_grad_(True)
    # forward() applies the config temperatures; distillation trains raw logits, and the
    # deployed calibration is fitted later by jev-factory (upstream init_from does the same).
    model.config.temperature = 1.0
    model.config.temperature_by_kind = {}
    return info


def _recording_collator(collate: Any, tokenizer: Any, config: Any, train: bool = True) -> Any:
    base = collate.SystemOneCollator

    class Recording(base):  # type: ignore[misc, valid-type]
        """Upstream's collator, unchanged, that also records each row's option order."""

        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            self.orders: list[Any] = []

        def _option_order(self, ex: Any) -> Any:
            order = super()._option_order(ex)
            self.orders.append(order)
            return order

        def __call__(self, batch: Any) -> Any:
            self.orders = []
            return super().__call__(batch)

    return Recording(tokenizer, config, train=train)


def _example(fmt: Any, item: TrainItem) -> Any:
    ref = item.target if item.target is not None else item.teacher
    # a heal row is labelled by the student before it has a target: the label is unused
    label = 0 if ref is None else max(range(len(ref)), key=ref.__getitem__)
    return fmt.Example(
        kind=item.kind,
        state=item.state,
        instructions=item.instructions,
        options=[list(o) for o in item.options],
        label=label,
        task=item.question_id or "distill",
        weight=item.weight,
    )


def slot_rows(items: Sequence[TrainItem], orders: Sequence[Any]) -> list[dl.LossRow]:
    """Targets in the slot order the collator rendered (``order[k]`` = canonical index)."""
    if len(items) != len(orders):
        raise ValueError("one option order per row is required")
    return [
        dl.LossRow(
            logits=[0.0] * len(it.options),
            target=None if it.target is None else dl.permute(it.target, order),
            teacher=None if it.teacher is None else dl.permute(it.teacher, order),
            temperature=it.temperature,
            weight=it.weight,
        )
        for it, order in zip(items, orders)
    ]


def train_steps(
    model: Any,
    make_batch: Callable[[Sequence[TrainItem]], tuple[dict[str, Any], list[dl.LossRow]]],
    items: Sequence[TrainItem],
    hp: Mapping[str, Any],
    *,
    optimizer: Any,
    scheduler: Any,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """The loop: micro-batches, gradient accumulation, clipping, one scheduler step per step.

    ``make_batch(rows)`` returns the model inputs and the slot-order loss rows; the model
    returns ``{"logits": [B, K]}``.
    """
    torch = _load("torch")
    mb, accum = int(hp["micro_batch_size"]), int(hp["grad_accum"])
    if len(items) < mb:
        raise CliError(EXIT_USER_ERROR, f"{len(items)} rows is fewer than one micro-batch ({mb})")
    rng = random.Random(int(hp["seed"]))  # nosec B311 - data order, not security
    params = [p for p in model.parameters() if p.requires_grad]
    history: list[dict[str, float]] = []
    step = micro = 0
    run_loss = run_kl = run_ce = 0.0
    model.train()
    for epoch in range(int(hp["epochs"])):
        order = list(range(len(items)))
        rng.shuffle(order)
        for start in range(0, len(order) - len(order) % mb, mb):  # drop_last, as upstream
            rows = [items[i] for i in order[start : start + mb]]
            inputs, loss_rows = make_batch(rows)
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            target = {k: v.to(device) for k, v in dl.to_tensors(loss_rows).items()}
            res = dl.distill_loss_torch(out["logits"], target, alpha=float(hp["alpha"]))
            (res["loss"] / accum).backward()
            run_loss += float(res["loss"].detach())
            run_kl += float(res["kl"].detach())
            run_ce += float(res["ce"].detach())
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, float(hp["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            entry = {
                "epoch": epoch,
                "step": step,
                "loss": run_loss / accum,
                "kl": run_kl / accum,
                "ce": run_ce / accum,
            }
            history.append(entry)
            log(f"[jev distill] epoch {epoch} step {step} loss {entry['loss']:.4f}")
            run_loss = run_kl = run_ce = 0.0
    if not history:
        raise CliError(EXIT_USER_ERROR, "no optimizer step ran; add rows or lower grad_accum")
    return history


def _save_stacked(model: Any, info: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """merge_fresh: replace the saved fresh adapter by stock + fresh stacked over the base."""
    peft = _load("peft")
    st = _load("safetensors.torch")
    fresh = {k: v.detach().cpu() for k, v in peft.get_peft_model_state_dict(model.torso).items()}
    stock = {k: v.cpu() for k, v in info["stock_state"].items()}
    state, rank, alpha = stack_lora(stock, info["stock_scale"], fresh, _lora_scale(model.torso))
    lora = out / "lora"
    st.save_file(state, str(safepath.contained(lora / "adapter_model.safetensors", out)))
    cfg_path = safepath.regular_file(lora / "adapter_config.json", out)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    targets = sorted(set(info["stock_targets"]) | set(model.config.lora_targets))
    cfg.update(r=rank, lora_alpha=alpha, target_modules=targets, rank_pattern={}, alpha_pattern={})
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sd_path = safepath.regular_file(out / "strands_decider_config.json", out)
    sd = json.loads(sd_path.read_text(encoding="utf-8"))
    sd.update(lora_r=rank, lora_alpha=alpha, lora_targets=targets)
    sd_path.write_text(json.dumps(sd, indent=2) + "\n", encoding="utf-8")
    return {"stacked_rank": rank, "stacked_alpha": alpha, "targets": targets}


@dataclass
class _Session:
    """The trainer env's loaded model and the upstream modules a run needs."""

    torch: Any
    collate: Any
    fmt: Any
    upstream: Any
    model: Any
    raw: Any
    info: dict[str, Any]
    device: str


def _raw_module(torch: Any, inner: Any) -> Any:
    class _Raw(torch.nn.Module):  # forward at temperature 1, logits only
        def __init__(self) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, **kw: Any) -> Any:
            return self.inner(**kw, temperature=1.0)

    return _Raw()


def _session(args: argparse.Namespace, hp: Mapping[str, Any]) -> _Session:
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    seed = int(hp["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = modeling.StrandsDeciderModel.load(str(args.init_from))
    if model.config.head_type != "pointer":
        raise CliError(EXIT_USER_ERROR, f"{args.init_from} is not a pointer-head checkpoint")
    info = prepare_init(model, hp)
    model.to(device)
    return _Session(
        torch=torch,
        collate=_load("training.data.collate"),
        fmt=_load("training.data.format"),
        upstream=_load("training.train"),
        model=model,
        raw=_raw_module(torch, model),
        info=info,
        device=device,
    )


def _collator(sess: _Session, hp: Mapping[str, Any], train: bool = True) -> Any:
    return _recording_collator(
        sess.collate,
        sess.model.tokenizer,
        sess.collate.CollatorConfig(
            max_length=int(hp["max_length"]),
            num_slots=sess.model.config.num_slots,
            head_type="pointer",
            shuffle_options=bool(hp["shuffle_options"]),
            reverse_score_prob=float(hp["reverse_score_prob"]),
            ordinal_smoothing=float(hp["ordinal_smoothing"]),
            seed=int(hp["seed"]),
        ),
        train=train,
    )


_INPUTS = ("input_ids", "attention_mask", "n_slots", "opt_idx")


def _batch_fn(
    sess: _Session, coll: Any
) -> Callable[[Sequence[TrainItem]], tuple[dict[str, Any], list[dl.LossRow]]]:
    def make_batch(rows: Sequence[TrainItem]) -> tuple[dict[str, Any], list[dl.LossRow]]:
        b = coll([_example(sess.fmt, r) for r in rows])
        return {k: b[k] for k in _INPUTS}, slot_rows(rows, coll.orders)

    return make_batch


def _fit(
    sess: _Session, hp: Mapping[str, Any], items: Sequence[TrainItem]
) -> list[dict[str, float]]:
    """Upstream's optimizer and schedule, then :func:`train_steps` over *items*."""
    torch, upstream = sess.torch, sess.upstream
    cfg = argparse.Namespace(lr=hp["lr"], head_lr=hp["head_lr"], weight_decay=hp["weight_decay"])
    optimizer = upstream._build_optimizer(sess.model, cfg)
    per_epoch = max(1, (len(items) // int(hp["micro_batch_size"])) // int(hp["grad_accum"]))
    total = per_epoch * int(hp["epochs"])
    warmup = max(1, int(total * float(hp["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: upstream._lr_lambda(s, warmup, total)
    )
    make_batch = _batch_fn(sess, _collator(sess, hp))
    return train_steps(
        sess.raw,
        make_batch,
        items,
        hp,
        optimizer=optimizer,
        scheduler=scheduler,
        device=sess.device,
    )


def _save(
    args: argparse.Namespace,
    hp: Mapping[str, Any],
    sess: _Session,
    items: Sequence[TrainItem],
    history: Sequence[Mapping[str, float]],
    extra: Mapping[str, Any] | None = None,
) -> int:
    out = Path(args.output_dir)
    sess.model.save_pretrained(str(out))
    merge = hp["init_mode"] == "merge_fresh"
    stacked = _save_stacked(sess.model, sess.info, out) if merge else None
    record = {
        "hyperparameters": dict(hp),
        "prereg_sha256": dp.sha256_file(Path(args.prereg)),
        "deviation_id": args.deviation,
        "init_from": str(args.init_from),
        "train_file": str(args.train_file),
        "rows": len(items),
        "rows_with_teacher": sum(i.teacher is not None for i in items),
        "rows_with_gold": sum(i.target is not None for i in items),
        "stacked_lora": stacked,
        "upstream": UPSTREAM_PROVENANCE,
        **dict(extra or {}),
    }
    safepath.contained(out / CONFIG_FILE, out).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    safepath.contained(out / HISTORY_FILE, out).write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "steps": len(history)}))
    return 0


def _train(args: argparse.Namespace, hp: dict[str, Any], items: list[TrainItem]) -> int:
    sess = _session(args, hp)
    history = _fit(sess, hp, items)
    return _save(args, hp, sess, items, history)


# --------------------------------------------------------------------- heal (trainer env)


class QatUnavailable(Exception):
    """Quantization-aware training cannot be set up; the heal falls back, recording why."""


def fake_quant(w: Any, bits: int, group_size: int | None, symmetric: bool) -> Any:
    """Group-wise round-to-nearest fake quantization with a straight-through gradient.

    Groups run along the last dimension (*group_size* ``None``: one group per row, i.e.
    per output channel). Symmetric: one scale per group (torchao int8 weight-only, the
    packed embedding); asymmetric: a scale and a zero point per group (torchao int4
    weight-only). The forward sees the dequantized values; the gradient passes through.
    """
    shape = w.shape
    cols = int(shape[-1])
    group = int(group_size or cols)
    if cols % group:
        raise QatUnavailable(f"last dimension {cols} is not a multiple of group size {group}")
    x = w.float().reshape(-1, group)
    if symmetric:
        qmax = 2 ** (int(bits) - 1) - 1
        scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        deq = (x / scale).round().clamp(-qmax - 1, qmax) * scale
    else:
        qmax = 2 ** int(bits) - 1
        lo = x.amin(dim=1, keepdim=True)
        scale = (x.amax(dim=1, keepdim=True) - lo).clamp_min(1e-8) / qmax
        zero = (-lo / scale).round()
        deq = ((x / scale).round() + zero).clamp(0, qmax).sub(zero) * scale
    return (x + (deq - x).detach()).reshape(shape).to(w.dtype)


def _lora_qat_forward(torch: Any, fq: Mapping[str, Any]) -> Callable[..., Any]:
    """A PEFT LoRA linear's forward on the fake-quantized merged weight ``W + s B A``."""

    def forward(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        base = self.base_layer
        w = base.weight.float()
        for name in self.active_adapters:
            if name in self.lora_A:
                delta = self.lora_B[name].weight.float() @ self.lora_A[name].weight.float()
                w = w + delta * float(self.scaling[name])
        wq = fake_quant(w, fq["block_bits"], fq.get("block_group_size"), fq["block_symmetric"])
        wq = wq.to(base.weight.dtype)
        return torch.nn.functional.linear(x.to(wq.dtype), wq, base.bias)

    return forward


def _plain_qat_forward(torch: Any, fq: Mapping[str, Any]) -> Callable[..., Any]:
    def forward(self: Any, x: Any) -> Any:
        wq = fake_quant(
            self.weight, fq["block_bits"], fq.get("block_group_size"), fq["block_symmetric"]
        )
        return torch.nn.functional.linear(x.to(wq.dtype), wq, self.bias)

    return forward


def _embedding_hook(fq: Mapping[str, Any]) -> Callable[..., Any]:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return fake_quant(output, fq["embedding_bits"], fq["embedding_group_size"], True)

    return hook


def enable_fake_quant(model: Any, fq: Mapping[str, Any]) -> tuple[Callable[[], None], dict]:
    """Fake-quantize the torso's text-block linears and embedding output; return the undo.

    Which modules are touched is :func:`jev_factory.backbones.pointer.quantized.module_class`,
    the rule the deployed build quantizes by. Raises :class:`QatUnavailable` when no
    LoRA-wrapped text-block linear is found.
    """
    from jev_factory.backbones.pointer import quantized as qz

    torch = _load("torch")
    patched: list[Any] = []
    hooks: list[Any] = []
    counts = {"lora_linears": 0, "plain_linears": 0, "embeddings": 0}
    for name, module in model.torso.named_modules():
        cls = qz.module_class(name)
        if cls == qz.EMBEDDING and fq.get("embedding_bits"):
            hooks.append(module.register_forward_hook(_embedding_hook(fq)))
            counts["embeddings"] += 1
        elif cls == qz.BLOCK_LINEAR and hasattr(module, "base_layer") and hasattr(module, "lora_A"):
            module.forward = types.MethodType(_lora_qat_forward(torch, fq), module)
            patched.append(module)
            counts["lora_linears"] += 1
        elif cls == qz.BLOCK_LINEAR and isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(_plain_qat_forward(torch, fq), module)
            patched.append(module)
            counts["plain_linears"] += 1

    def restore() -> None:
        for module in patched:
            module.__dict__.pop("forward", None)
        for handle in hooks:
            handle.remove()

    if not counts["lora_linears"]:
        restore()
        raise QatUnavailable("no LoRA-wrapped text-block linear to train quantization-aware")
    return restore, counts


def _noop() -> None:
    return None


def enable_heal_method(
    model: Any, spec: Mapping[str, Any], probe: Callable[[], Any]
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Set up quantization-aware training, else fall back to the bf16 continuation.

    Returns the method record (``method``, ``fallback_reason``, the fake-quant scheme and
    what it touched) and the undo. The fallback is taken when the spec has no fake-quant
    scheme for the format, no LoRA-wrapped block linear exists, or a probe forward fails.
    """
    fq = spec.get("fake_quant")
    fallback = {"method": HEAL_BF16, "fake_quant": None, "touched": None}
    if not fq:
        reason = f"no fake-quant scheme for {spec.get('format')} (not a torchao build)"
        return {**fallback, "fallback_reason": reason}, _noop
    try:
        restore, touched = enable_fake_quant(model, fq)
    except QatUnavailable as exc:
        return {**fallback, "fallback_reason": str(exc)}, _noop
    try:
        probe()
    except Exception as exc:  # any failure of the probe forward means QAT is unavailable here
        restore()
        reason = f"the QAT probe forward failed: {type(exc).__name__}: {exc}"
        return {**fallback, "fallback_reason": reason}, _noop
    record = {"method": HEAL_QAT, "fallback_reason": None, "fake_quant": dict(fq)}
    return {**record, "touched": touched}, restore


def student_distributions(
    sess: _Session, coll: Any, items: Sequence[TrainItem], batch_size: int
) -> list[list[float]]:
    """The bf16 student's own per-option distribution per row (temperature 1, canonical)."""
    torch = sess.torch
    out: list[list[float]] = []
    sess.raw.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            rows = list(items[start : start + batch_size])
            b = coll([_example(sess.fmt, r) for r in rows])
            logits = sess.raw(**{k: b[k].to(sess.device) for k in _INPUTS})["logits"]
            logits = logits.float().cpu()
            orders = coll.orders if len(coll.orders) == len(rows) else [None] * len(rows)
            for i, (row, order) in enumerate(zip(rows, orders)):
                n = len(row.options)
                out.append(unpermute(torch.softmax(logits[i, :n], dim=-1).tolist(), order))
    sess.raw.train()
    return out


def _write_heal_targets(out: Path, log: Sequence[Mapping[str, Any]]) -> str:
    out.mkdir(parents=True, exist_ok=True)
    path = safepath.contained(out / HEAL_TARGETS_FILE, out)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in log)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _heal(
    args: argparse.Namespace, hp: dict[str, Any], rows: list[HealRow], spec: Mapping[str, Any]
) -> int:
    """Label with the bf16 student, mix with Clef, then train quantization-aware (or d18)."""
    sess = _session(args, hp)
    mb = int(hp["micro_batch_size"])
    student = student_distributions(
        sess, _collator(sess, hp, train=False), [r.item for r in rows], mb
    )
    items, log, counts = heal_items(rows, student, spec["mix"])
    targets_sha = _write_heal_targets(Path(args.output_dir), log)
    heal_hp = {**hp, "alpha": HEAL_ALPHA}
    probe_batch = _batch_fn(sess, _collator(sess, heal_hp))

    def probe() -> None:
        inputs, _rows = probe_batch(items[:mb])
        with sess.torch.no_grad():
            sess.raw(**{k: v.to(sess.device) for k, v in inputs.items()})

    method, restore = enable_heal_method(sess.model, spec, probe)
    print(f"[jev distill] heal method {method['method']}: {method['fallback_reason'] or 'ok'}")
    try:
        history = _fit(sess, heal_hp, items)
    finally:
        restore()
    heal = {
        **method,
        "format": spec["format"],
        "mix": dict(spec["mix"]),
        "alpha": HEAL_ALPHA,
        "loss": "KL to the mixed soft targets only (no gold CE term)",
        "bf16_student": str(args.init_from),
        "target_sources": counts,
        "heal_targets_sha256": targets_sha,
        "trained": "LoRA adapter and pointer head (norms are not in the checkpoint format)",
        "lora_dropout": "not applied in the QAT forward" if method["method"] == HEAL_QAT else None,
    }
    return _save(args, heal_hp, sess, items, history, {"heal": heal})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=MODULE, description="Distill trainer (trainer env).")
    p.add_argument("--prereg", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--knobs", required=True, help="run knobs as a JSON object")
    p.add_argument("--train-file", required=True)
    p.add_argument("--init-from", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deviation", default=None)
    p.add_argument("--heal", default=None, help="heal spec (heal mode; see the module docstring)")
    return p


def _contain(args: argparse.Namespace) -> None:
    """Hold every path the trainer reads or writes to ``--run-dir``, in place.

    The pipeline puts the registration (``distill/prereg.json``), the assembled rows and every
    checkpoint it trains under the run directory, so a path that resolves outside it (``..``,
    an absolute path elsewhere, a symlink out) is refused before anything is read. Only
    ``--init-from`` (the verified stock checkpoint in the pin cache) may live elsewhere.
    """
    run_dir = safepath.resolved(args.run_dir)
    args.run_dir = str(run_dir)
    args.prereg = str(safepath.regular_file(args.prereg, run_dir))
    args.train_file = str(safepath.contained(args.train_file, run_dir))
    args.output_dir = str(safepath.contained(args.output_dir, run_dir))
    args.init_from = str(safepath.resolved(args.init_from))
    if args.heal is not None:
        args.heal = str(safepath.regular_file(args.heal, run_dir))


def main(argv: Sequence[str] | None = None) -> int:
    """Trainer entry: re-check the lock, resolve hyperparameters, read rows, then train."""
    args = _parser().parse_args(argv)
    try:
        _contain(args)
        knobs = json.loads(args.knobs)
        if not isinstance(knobs, dict):
            raise _refuse("--knobs must be a JSON object")
        hp = resolve_from_locked(args.prereg, args.run_dir, knobs, args.deviation)
        train_file = safepath.regular_file(args.train_file, args.run_dir)
        if args.heal is not None:
            spec = load_heal_spec(args.heal, args.run_dir)
            rows = load_heal_rows(train_file, hp, spec["protected_id_sha256"])
            if str(safepath.resolved(spec["bf16_student"])) != args.init_from:
                raise _refuse(
                    f"the heal spec's bf16 student {spec['bf16_student']} is not --init-from "
                    f"{args.init_from}",
                    "the heal starts from, and is labelled by, the bf16 distilled student",
                )
            return _heal(args, hp, rows, spec)
        items = load_items(train_file, hp)
        return _train(args, hp, items)
    except json.JSONDecodeError as exc:
        print(f"error: --knobs is not valid JSON: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except CliError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.remediation:
            print(f"hint: {exc.remediation}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":  # runs in the trainer env
    sys.exit(main())
