"""Distill trainer for Strands Decider: continue from v21 on teacher soft labels and gold.

Two halves in one module:

* **Planning, here (stdlib only).** :func:`plan_train` reads every hyperparameter from the
  locked distill pre-registration (:func:`jev_factory.distill.prereg.require_locked`) and
  returns the trainer's subprocess command without running anything. :func:`run_train` is
  dry-run unless ``apply``; with ``apply`` the command runs through
  :func:`jev_factory.factory.gpu.run_gpu_stage` (residency guard, memory cap, memory-floor
  watchdog) and nowhere else.
* **Training, in the trainer env.** ``python -m jev_factory.backbones.pointer.train_distill``
  (:func:`main`) re-checks the lock itself, loads the v21 checkpoint with upstream's
  ``StrandsDeciderModel.load``, renders rows with upstream's ``SystemOneCollator`` (the
  serving prompt and option-position readout) and minimises
  ``alpha * KL(teacher || student at T_kind) + (1 - alpha) * CE(gold)``
  (:mod:`.distill_loss`) on LoRA plus the pointer head. The heavy packages (torch, peft,
  safetensors, strands_decider and the checkout's ``training`` package) are loaded with
  :func:`importlib.import_module` inside the training functions only: the jev process never
  loads them, and this module imports cleanly with none of them installed.

Init modes (a pre-registered knob, ``knob_grid.init_mode``; frame park v6):

* ``continue``: train v21's own LoRA adapter (made trainable again; upstream ``load`` opens
  it frozen) and v21's own pointer head. The adapter's shape is v21's, so no ``lora_*``
  knob is read.
* ``merge_fresh``: merge v21's adapter into the torso (``merge_and_unload``), attach a fresh
  LoRA of the pre-registered ``lora_r``/``lora_alpha``/``lora_dropout``/``lora_targets``,
  and train it with v21's head. ``strands-decider serve`` loads one adapter over the
  *unmerged* base, so at save time v21's adapter and the fresh one are stacked into one
  adapter of rank ``r_v21 + r_fresh`` (:func:`stack_lora`, exact: ``B A`` of the stack is
  the sum of both scaled deltas). The fresh adapter was trained against the bf16 merge, so
  the served model can differ from the trained one by that merge's rounding; the final
  measurement is taken on the served artifact.

Unlike upstream ``init_from`` (which discards the loaded head for a fresh ``SlotHead``),
both modes keep v21's pointer head. Upstream's ``kl_frozen_weight`` (KL to the untouched
torso's option-number readout) and ``teacher_weight`` (an additive teacher term) are not
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
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.pointer import distill_loss as dl
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill import prereg as dp
from jev_factory.factory import gpu

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


def _check_values(hp: Mapping[str, Any]) -> None:
    if hp["init_mode"] not in INIT_MODES:
        raise _refuse(f"init_mode {hp['init_mode']!r} is not one of {list(INIT_MODES)}")
    temp = hp["temperature"]
    if not isinstance(temp, dict) or sorted(temp) != sorted(QUESTION_KINDS):
        raise _refuse(f"temperature must give one value per question kind {list(QUESTION_KINDS)}")
    if not all(_is_num(v) and v > 0 for v in temp.values()):
        raise _refuse("temperature values must be positive numbers")
    for name in _POS_INT:
        if name in hp and not (_is_int(hp[name]) and hp[name] >= 1):
            raise _refuse(f"{name} must be a positive integer, got {hp[name]!r}")
    for name in _NON_NEG:
        if name in hp and not (_is_num(hp[name]) and hp[name] >= 0):
            raise _refuse(f"{name} must be a non-negative number, got {hp[name]!r}")
    for name in _UNIT:
        if not (_is_num(hp[name]) and 0.0 <= hp[name] <= 1.0):
            raise _refuse(f"{name} must be a number in [0, 1], got {hp[name]!r}")
    if not _is_int(hp["seed"]):
        raise _refuse(f"seed must be an integer, got {hp['seed']!r}")
    if not isinstance(hp["shuffle_options"], bool):
        raise _refuse("shuffle_options must be true or false")
    targets = hp.get("lora_targets")
    if targets is not None and not (
        isinstance(targets, list) and targets and all(isinstance(t, str) and t for t in targets)
    ):
        raise _refuse("lora_targets must be a non-empty list of module names")


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
) -> dict[str, Any]:
    """The trainer's subprocess command and resolved hyperparameters; runs nothing.

    *init_from* is the verified v21 checkpoint directory (``lora/``, ``head.safetensors``,
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


def _item(raw: Any, line: int, hp: Mapping[str, Any]) -> TrainItem:
    where = f"row {line}"
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
    gold, teacher = raw.get("gold"), raw.get("teacher")
    if gold is None and teacher is None:
        raise _bad(where, "the row has neither a gold target nor a teacher target")
    target = None if gold is None else _gold(where, kind, gold, names, hp["ordinal_smoothing"])
    tea = None if teacher is None else _distribution(where, "teacher", teacher, names)
    weight = raw.get("weight", 1.0)
    if not (_is_num(weight) and weight > 0):
        raise _bad(where, f"weight must be a positive number, got {weight!r}")
    return TrainItem(
        record_id=str(raw.get("record_id", "")),
        question_id=str(raw.get("question_id", "")),
        kind=kind,
        state=raw.get("state", ""),
        instructions=str(raw.get("instructions") or ""),
        options=[[str(a), str(b)] for a, b in options],
        target=target,
        teacher=tea,
        temperature=float(hp["temperature"][kind]),
        weight=float(weight),
    )


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
    """Make v21 trainable per ``init_mode``; v21's pointer head is kept and trained."""
    info: dict[str, Any] = {"init_mode": hp["init_mode"]}
    if hp["init_mode"] == "continue":
        n = 0
        for name, p in model.torso.named_parameters():
            trainable = "lora_" in name
            p.requires_grad_(trainable)
            n += int(trainable)
        if not n:
            raise CliError(EXIT_ENV_ERROR, "the v21 checkpoint has no LoRA parameters")
    else:
        peft = _load("peft")
        info["v21_state"] = {
            k: v.detach().clone() for k, v in peft.get_peft_model_state_dict(model.torso).items()
        }
        info["v21_scale"] = _lora_scale(model.torso)
        info["v21_targets"] = sorted(model.torso.peft_config["default"].target_modules)
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


def _recording_collator(collate: Any, tokenizer: Any, config: Any) -> Any:
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

    return Recording(tokenizer, config, train=True)


def _example(fmt: Any, item: TrainItem) -> Any:
    ref = item.target if item.target is not None else item.teacher
    label = max(range(len(ref)), key=ref.__getitem__)  # type: ignore[arg-type]
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
    """merge_fresh: replace the saved fresh adapter by v21 + fresh stacked over the base."""
    peft = _load("peft")
    st = _load("safetensors.torch")
    fresh = {k: v.detach().cpu() for k, v in peft.get_peft_model_state_dict(model.torso).items()}
    v21 = {k: v.cpu() for k, v in info["v21_state"].items()}
    state, rank, alpha = stack_lora(v21, info["v21_scale"], fresh, _lora_scale(model.torso))
    lora = out / "lora"
    st.save_file(state, str(lora / "adapter_model.safetensors"))
    cfg_path = lora / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    targets = sorted(set(info["v21_targets"]) | set(model.config.lora_targets))
    cfg.update(r=rank, lora_alpha=alpha, target_modules=targets, rank_pattern={}, alpha_pattern={})
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sd_path = out / "strands_decider_config.json"
    sd = json.loads(sd_path.read_text(encoding="utf-8"))
    sd.update(lora_r=rank, lora_alpha=alpha, lora_targets=targets)
    sd_path.write_text(json.dumps(sd, indent=2) + "\n", encoding="utf-8")
    return {"stacked_rank": rank, "stacked_alpha": alpha, "targets": targets}


def _train(args: argparse.Namespace, hp: dict[str, Any], items: list[TrainItem]) -> int:
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    collate = _load("training.data.collate")
    fmt = _load("training.data.format")
    upstream = _load("training.train")
    seed = int(hp["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = modeling.StrandsDeciderModel.load(str(args.init_from))
    if model.config.head_type != "pointer":
        raise CliError(EXIT_USER_ERROR, f"{args.init_from} is not a pointer-head checkpoint")
    info = prepare_init(model, hp)
    model.to(device)
    coll = _recording_collator(
        collate,
        model.tokenizer,
        collate.CollatorConfig(
            max_length=int(hp["max_length"]),
            num_slots=model.config.num_slots,
            head_type="pointer",
            shuffle_options=bool(hp["shuffle_options"]),
            reverse_score_prob=float(hp["reverse_score_prob"]),
            ordinal_smoothing=float(hp["ordinal_smoothing"]),
            seed=seed,
        ),
    )

    def make_batch(rows: Sequence[TrainItem]) -> tuple[dict[str, Any], list[dl.LossRow]]:
        b = coll([_example(fmt, r) for r in rows])
        inputs = {
            "input_ids": b["input_ids"],
            "attention_mask": b["attention_mask"],
            "n_slots": b["n_slots"],
            "opt_idx": b["opt_idx"],
        }
        return inputs, slot_rows(rows, coll.orders)

    class _Raw(torch.nn.Module):  # forward at temperature 1, logits only
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, **kw: Any) -> Any:
            return self.inner(**kw, temperature=1.0)

    cfg = argparse.Namespace(lr=hp["lr"], head_lr=hp["head_lr"], weight_decay=hp["weight_decay"])
    optimizer = upstream._build_optimizer(model, cfg)
    per_epoch = max(1, (len(items) // int(hp["micro_batch_size"])) // int(hp["grad_accum"]))
    total = per_epoch * int(hp["epochs"])
    warmup = max(1, int(total * float(hp["warmup_ratio"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: upstream._lr_lambda(s, warmup, total)
    )
    history = train_steps(
        _Raw(model), make_batch, items, hp, optimizer=optimizer, scheduler=scheduler, device=device
    )

    out = Path(args.output_dir)
    model.save_pretrained(str(out))
    stacked = _save_stacked(model, info, out) if hp["init_mode"] == "merge_fresh" else None
    record = {
        "hyperparameters": hp,
        "prereg_sha256": dp.sha256_file(Path(args.prereg)),
        "deviation_id": args.deviation,
        "init_from": str(args.init_from),
        "train_file": str(args.train_file),
        "rows": len(items),
        "rows_with_teacher": sum(i.teacher is not None for i in items),
        "rows_with_gold": sum(i.target is not None for i in items),
        "stacked_lora": stacked,
        "upstream": UPSTREAM_PROVENANCE,
    }
    (out / CONFIG_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (out / HISTORY_FILE).write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "steps": len(history)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=MODULE, description="Distill trainer (trainer env).")
    p.add_argument("--prereg", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--knobs", required=True, help="run knobs as a JSON object")
    p.add_argument("--train-file", required=True)
    p.add_argument("--init-from", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deviation", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Trainer entry: re-check the lock, resolve hyperparameters, read rows, then train."""
    args = _parser().parse_args(argv)
    try:
        knobs = json.loads(args.knobs)
        if not isinstance(knobs, dict):
            raise _refuse("--knobs must be a JSON object")
        hp = resolve_from_locked(args.prereg, args.run_dir, knobs, args.deviation)
        items = load_items(args.train_file, hp)
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
