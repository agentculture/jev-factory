"""Merge a scorer's LoRA adapter, verified, and drive the train and heal GPU stages.

The causal-LM half of nvsh's ``train.py`` that the scorer path uses (nvsh
issue 46): the GPU memory cap, the Qwen3.5 specifics, and the merge. Track A
(generative tool-call training) is not imported: it is not jev-like.

**The merge is verified, never trusted** (nvsh #46, lapse l4: a silent PEFT
key mismatch produced "tuned" models bit-identical to stock, while PEFT only
*warned* "Found missing adapter keys"). :func:`verify_merge` refuses a merge

* when **any** missing-adapter-key warning was emitted while the adapter
  loaded (:func:`capture_load_warnings` records both ``warnings.warn`` and
  log records; :func:`check_load_warnings` judges them);
* when any tensor in the adapter file did not load (:func:`check_adapter_loaded`);
* when **any** sampled merged weight equals its base weight
  (:func:`sample_names` picks a deterministic sample of the adapted modules,
  always the first and last; :func:`check_weights_changed` compares them);
* when the merged model is not the text-only causal LM, or its state dict
  carries a vision-language ``language_model`` prefix (:func:`check_text_only`).

Each check is a pure function over names, messages and state-dict-like
mappings, so the logic is tested without torch or peft; :func:`merge_adapter`
only gathers those inputs from a real model (heavy imports inside it).

**Qwen3.5 specifics** (issue #4 stage 8): a loader may return the
multimodal processor, so rendering uses its ``.tokenizer``
(:func:`text_tokenizer`); a vision-language adapter's keys are mapped onto
the text-only ``Qwen3_5ForCausalLM`` (:func:`text_adapter_key`) and the
merge is into that class, never the vision-language one whose save doubled
the ``language_model`` prefix vLLM rejects.

**Stage drivers.** :func:`plan_train` and :func:`plan_heal` build the train
and heal stages: each refuses without a registered pre-registration
(:func:`jev_factory.factory.prereg.require_registered`), selects the frozen
training set by sha256 (:func:`jev_factory.data.assemble.select_frozen`, never
by mtime), and lists the two commands it runs, the scorer trainer then this
module's verified merge. :func:`run_plan` is a dry run unless ``apply``; when
applied every command goes through the guarded GPU stage runner
(:func:`jev_factory.factory.gpu.run_gpu_stage`: residency guard, then the
memory cap and floor watchdog), then the greedy ``generation_config.json``
is written into ``<run>/merged``. **Heal** is a short bf16 continuation from
the run's own merged checkpoint: 1 epoch at lr 5e-5 on the same frozen set,
one round only; its merge is verified exactly like a training run's.

    python -m jev_factory.backbones.causal_lm.train --merge-only <run>/adapter --out <run> \\
        --base <base> --revision <commit>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from jev_factory.backbones.causal_lm import gen_config
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.data.assemble import select_frozen
from jev_factory.factory.gpu import WATCHDOG_STATUS, run_gpu_stage
from jev_factory.factory.prereg import require_registered

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/train.py",
    "commit": "9debdc6",
    "adaptations": [
        "kept for the scorer path: gpu_memory_fraction/cap_gpu_memory (env var renamed to the"
        " run config's JEV_TRAIN_GPU_MEMORY_GB), save_valid_generation_config, text_tokenizer,"
        " LORA_TARGETS/lora_targets, text_adapter_key, adapter_leaf_modules,"
        " check_adapter_loaded and merge_adapter",
        "Track A generative training is not imported (read_examples, labels_from_mask,"
        " tokenize_example, answer_span, the unsloth Trainer path in main :392-508): Track A"
        " is not jev-like; main keeps only --merge-only",
        "merge verification made total and testable: capture_load_warnings +"
        " check_load_warnings fail on any missing-adapter-key warning; sample_names +"
        " check_weights_changed compare a deterministic sample of adapted weights (not one"
        " probe) and fail when any equals the base; check_text_only refuses a vision-language"
        " class or language_model prefix; verify_merge runs them all and merge_adapter writes"
        " merge-report.json",
        "new: plan_train/plan_heal/run_plan replace pipeline.sh's train-scorer and heal"
        " stages (:640-664, :762-780): prereg gate, frozen-set selection by sha256, dry-run by"
        " default, commands run through factory.gpu.run_gpu_stage; heal is 1 epoch lr 5e-5"
        " bf16 on the same frozen set via train_scorer --heal, one round only",
        "tests ported from tests/test_lfm_finetune_train.py (scorer-path helpers)",
    ],
    "licence": "Apache-2.0",
}

#: The environment variable that caps the trainer's GPU memory, in GiB (run config key
#: ``train_gpu_memory_gb``).
GPU_MEMORY_ENV = "JEV_TRAIN_GPU_MEMORY_GB"

#: The text-only causal-LM class a Qwen3.5 adapter is merged into (issue #4 stage 8).
TEXT_ONLY_ARCHITECTURE = "Qwen3_5ForCausalLM"

#: What the verified merge writes next to ``merged/``.
MERGE_REPORT = "merge-report.json"

#: The heal recipe (nvsh #46 d18; issue #4 stage 12): a short bf16 continuation.
HEAL_EPOCHS = 1
HEAL_LR = 5e-5
HEAL_PRECISION = "bf16"

#: The scorer training defaults (issue #4 stage 8: r16/alpha32, 3 epochs, lr 2e-4).
TRAIN_DEFAULTS: dict[str, Any] = {
    "epochs": 3,
    "lr": 2e-4,
    "rank": 16,
    "alpha": 32,
    "batch": 4,
    "seed": 46,
    "max_length": 2048,
}

TRAIN_SCORER_MODULE = "jev_factory.backbones.causal_lm.train_scorer"
TRAIN_MODULE = "jev_factory.backbones.causal_lm.train"

#: How many adapted modules' weights the merge compares against the base.
DEFAULT_WEIGHT_SAMPLE = 8

#: A load warning that says adapter tensors were not found in the model.
_MISSING_KEYS_RE = re.compile(r"missing (?:adapter )?keys", re.IGNORECASE)

#: A vision-language checkpoint's language-model prefix.
_VL_PREFIX = "language_model."


class MergeError(ValueError):
    """The merge cannot be trusted; nothing is saved."""


# ---------------------------------------------------------------------------
# GPU memory cap (unified memory: the cgroup cap does not see CUDA allocations)
# ---------------------------------------------------------------------------


def gpu_memory_fraction(gb: str, total_bytes: int) -> float:
    """The share of a device of *total_bytes* that a budget of *gb* GiB is.

    The OS memory cap does not see CUDA allocations, and on unified memory
    (GB10, Jetson) they come out of the same pool, so the trainer caps itself.
    Refuses a budget that is not a positive number or is larger than the device.
    """
    try:
        value = float(gb)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{GPU_MEMORY_ENV}={gb!r} is not a positive number of GiB")
    budget = value * 2**30
    if budget > total_bytes:
        raise ValueError(
            f"{GPU_MEMORY_ENV}={gb!r} exceeds the device's {total_bytes / 2**30:.1f} GiB"
        )
    return budget / total_bytes


def cap_gpu_memory(torch, environ: Mapping[str, str] = os.environ) -> float | None:
    """Apply ``$JEV_TRAIN_GPU_MEMORY_GB`` to CUDA device 0 before a model loads.

    Returns the fraction set, or ``None`` when the variable is unset, empty or
    whitespace-only (all mean "no cap"), or there is no CUDA device. Raises
    ``ValueError`` for a budget :func:`gpu_memory_fraction` refuses.
    """
    gb = environ.get(GPU_MEMORY_ENV)
    if gb is None or not gb.strip():
        return None
    if not torch.cuda.is_available():
        print(f"{GPU_MEMORY_ENV} set but no CUDA device; no GPU cap applied", file=sys.stderr)
        return None
    fraction = gpu_memory_fraction(gb, torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"{GPU_MEMORY_ENV}={gb.strip()}: GPU memory fraction {fraction:.4f}", file=sys.stderr)
    return fraction


def save_valid_generation_config(model) -> None:
    """Clear a greedy temperature so transformers will save the model.

    A served checkpoint's generation config says temperature 0 with
    ``do_sample`` False (:mod:`.gen_config`), a combination transformers
    refuses to save. A heal merges from a checkpoint that carries it, so the
    temperature is cleared before the save; the stage writes the greedy file
    back into the merged dir afterwards.
    """
    config = getattr(model, "generation_config", None)
    if config is None:
        return
    if getattr(config, "do_sample", None) is False and getattr(config, "temperature", None) == 0:
        config.temperature = None


# ---------------------------------------------------------------------------
# Qwen3.5 specifics and LoRA targets
# ---------------------------------------------------------------------------


def text_tokenizer(loaded):
    """The text tokenizer inside whatever the loader returned.

    For Qwen3.5 (a vision-language model) a loader may return the multimodal
    processor, whose chat template expects content as a list of parts and
    crashes on plain strings. Its ``tokenizer`` attribute is the text
    tokenizer, with the same chat template.
    """
    inner = getattr(loaded, "tokenizer", None)
    return inner if inner is not None else loaded


def text_adapter_key(key: str) -> str:
    """*key* as the text-only causal LM names it.

    A vision-language training class keeps adapter keys under
    ``model.language_model.``; the text-only model has the same modules under
    ``model.``. Merging into the vision-language class instead saved doubled
    prefixes (``model.language_model.language_model.*``) that vLLM cannot
    load (nvsh #46, lapse l4).
    """
    return key.replace(".model.language_model.", ".model.", 1)


def adapter_leaf_modules(keys: Iterable[str]) -> list[str]:
    """The adapted modules' own names (``q_proj``, ``in_proj_qkv``, ...), for an
    explicit ``target_modules`` that matches the text-only model's paths."""
    leaves = set()
    for key in keys:
        head = key.split(".lora_", 1)[0]
        leaves.add(head.rsplit(".", 1)[-1])
    return sorted(leaves)


#: LoRA target module sets. ``attn-mlp`` is the attention and MLP projections;
#: ``attn-mlp-gdn`` adds Qwen3.5's Gated-DeltaNet projections (18 of its 24
#: layers are linear attention, which the default list never adapts; nvsh #46 r9).
LORA_TARGETS = {
    "attn-mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn-mlp-gdn": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    ],
}


def lora_targets(name: str) -> list[str]:
    if name not in LORA_TARGETS:
        raise ValueError(f"unknown LoRA targets {name!r}; one of {', '.join(LORA_TARGETS)}")
    return list(LORA_TARGETS[name])


# ---------------------------------------------------------------------------
# Merge verification (pure: names, messages and state-dict-like mappings)
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Collects each WARNING-or-worse record once, however many watched loggers it passes."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.sink = sink
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if any(seen is record for seen in self.records):
            return  # propagated from a child logger this handler also watches
        self.records.append(record)
        self.sink.append(record.getMessage())


#: Loggers a load warning can arrive on (``transformers`` does not propagate to root).
_LOAD_LOGGERS = ("", "peft", "transformers")


@contextmanager
def capture_load_warnings(loggers: Sequence[str] = _LOAD_LOGGERS) -> Iterator[list[str]]:
    """Collect every warning raised or logged inside the block.

    Yields a list that holds, once the block exits, the text of every
    ``warnings.warn`` (all of them, even ones a filter would hide) and of
    every WARNING-or-worse record on *loggers*. Handlers are removed again.
    """
    messages: list[str] = []
    handler = _ListHandler(messages)
    attached = [logging.getLogger(name) for name in dict.fromkeys(loggers)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for logger in attached:
            logger.addHandler(handler)
        try:
            yield messages
        finally:
            for logger in attached:
                logger.removeHandler(handler)
            messages.extend(str(item.message) for item in caught)


def check_load_warnings(messages: Iterable[str]) -> None:
    """Refuse the merge when any message is a missing-adapter-key warning."""
    missing = [message for message in messages if _MISSING_KEYS_RE.search(message)]
    if missing:
        first = missing[0] if len(missing[0]) <= 200 else missing[0][:200] + "..."
        raise MergeError(
            f"{len(missing)} missing adapter key warning(s) while loading the adapter"
            f" (first: {first}); the merge would not carry the trained weights"
        )


def check_adapter_loaded(file_keys: Sequence[str], loaded_keys: Iterable[str]) -> None:
    """Refuse a merge unless every tensor in the adapter file was loaded.

    PEFT only warns about missing adapter keys and then merges zero-initialised
    LoRA weights, which silently returns the base model (nvsh #46, lapse l4).
    *loaded_keys* are the model's LoRA parameter names (with PEFT's adapter
    name, ``.default``, which the file's names do not carry).
    """
    loaded = {key.replace(".default.", ".") for key in loaded_keys}
    missing = [key for key in file_keys if key not in loaded]
    if not missing:
        return
    found = len(file_keys) - len(missing)
    if found == 0:
        raise MergeError(
            f"none of the adapter's {len(file_keys)} tensors loaded into the base "
            f"(first missing: {missing[0]}); the merge would return the base unchanged"
        )
    raise MergeError(
        f"only {found} of {len(file_keys)} adapter tensors loaded (first missing: {missing[0]})"
    )


def sample_names(names: Iterable[str], k: int = DEFAULT_WEIGHT_SAMPLE, seed: int = 0) -> list[str]:
    """A deterministic, sorted sample of *k* of *names*: always the first and last, the rest
    drawn with *seed*. Every name when there are at most *k*."""
    ordered = sorted(set(names))
    if len(ordered) <= k:
        return ordered
    ends = {ordered[0], ordered[-1]}
    middle = ordered[1:-1]
    rng = random.Random(seed)  # nosec B311 -- which weights to compare, not security
    return sorted(ends | set(rng.sample(middle, max(k - len(ends), 0))))


def weights_equal(a: Any, b: Any) -> bool:
    """Whether two weights are identical: ``a.equal(b)`` for tensors, ``==`` otherwise."""
    equal = getattr(a, "equal", None)
    if callable(equal):
        return bool(equal(b))
    return bool(a == b)


def check_weights_changed(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    equal: Callable[[Any, Any], bool] = weights_equal,
) -> list[str]:
    """The sampled weight names, once each merged weight is shown to differ from its base.

    Refuses when nothing was sampled, when a sampled weight is missing from
    the merged model, or when **any** sampled merged weight equals the base.
    """
    if not before:
        raise MergeError("no adapted weight was sampled; the merge cannot be verified")
    absent = [name for name in before if name not in after]
    if absent:
        raise MergeError(f"sampled weight {absent[0]} is missing after the merge")
    unchanged = [name for name in sorted(before) if equal(before[name], after[name])]
    if unchanged:
        raise MergeError(
            f"{len(unchanged)} of {len(before)} sampled merged weight(s) equals the base"
            f" (first: {unchanged[0]}); refusing to save what may be the base model"
        )
    return sorted(before)


def check_text_only(architectures: Iterable[str], keys: Iterable[str]) -> None:
    """Refuse a merge that is not the text-only causal LM or carries a VL key prefix."""
    names = [name for name in architectures if name]
    if not names or not all(name.endswith("ForCausalLM") for name in names):
        raise MergeError(
            f"the merged model is {names or 'unnamed'}, not a text-only causal LM"
            f" (for Qwen3.5: {TEXT_ONLY_ARCHITECTURE}); merge into the text-only class"
        )
    for key in keys:
        if _VL_PREFIX in key:
            raise MergeError(
                f"merged state dict key {key!r} carries a vision-language language_model"
                " prefix a text-only server cannot load"
            )


def verify_merge(
    *,
    file_keys: Sequence[str],
    loaded_keys: Iterable[str],
    load_warnings: Iterable[str],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    architectures: Iterable[str] | None = None,
    saved_keys: Iterable[str] | None = None,
    equal: Callable[[Any, Any], bool] = weights_equal,
) -> dict[str, Any]:
    """Every merge check, in order; the report for ``merge-report.json``, or :class:`MergeError`."""
    messages = list(load_warnings)
    check_load_warnings(messages)
    check_adapter_loaded(file_keys, loaded_keys)
    sampled = check_weights_changed(before, after, equal=equal)
    if architectures is not None or saved_keys is not None:
        check_text_only(architectures or [], saved_keys or [])
    return {
        "adapter_tensors": len(file_keys),
        "missing_adapter_keys": 0,
        "load_warnings": len(messages),
        "sampled_weights": sampled,
        "changed_weights": len(sampled),
        "architectures": list(architectures or []),
    }


def merge_adapter(
    base: str,
    revision: str | None,
    adapter: Path,
    out: Path,
    *,
    sample: int = DEFAULT_WEIGHT_SAMPLE,
    seed: int = 0,
) -> dict[str, Any]:  # needs torch, peft and a model
    """Merge a saved LoRA adapter into a fresh text-only copy of *base*, verified, into *out*.

    Plain transformers + peft (``merge_and_unload``), never a trainer's merged
    saver that copies read-only cache files. The adapter's keys are mapped
    onto the text-only causal LM first. The tokenizer (and so the chat
    template) is saved from the base unchanged; :mod:`.stage_cache` checks
    that byte for byte. Writes ``<out>/../merge-report.json``.
    """
    import tempfile

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with safe_open(str(adapter / "adapter_model.safetensors"), "pt") as handle:
        tensors = {text_adapter_key(key): handle.get_tensor(key) for key in handle.keys()}
    file_keys = list(tensors)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = adapter_leaf_modules(file_keys)
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmp:
        save_file(tensors, str(Path(tmp) / "adapter_model.safetensors"))
        (Path(tmp) / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        with capture_load_warnings() as messages:
            peft_model = PeftModel.from_pretrained(model, tmp)
    loaded_keys = [key for key in peft_model.state_dict() if "lora_" in key]
    adapted = {
        name: module
        for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "base_layer")
    }
    names = sample_names(adapted, sample, seed)
    before = {name: adapted[name].base_layer.weight.detach().clone() for name in names}
    merged = peft_model.merge_and_unload()
    after = {
        name: merged.get_submodule(name.replace("base_model.model.", "", 1)).weight.detach()
        for name in names
    }
    report = verify_merge(
        file_keys=file_keys,
        loaded_keys=loaded_keys,
        load_warnings=messages,
        before=before,
        after=after,
        architectures=[type(merged).__name__],
        saved_keys=list(merged.state_dict()),
        equal=torch.equal,
    )
    out.mkdir(parents=True, exist_ok=True)
    save_valid_generation_config(merged)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(str(out))
    report.update({"base": base, "revision": revision, "adapter": adapter.name})
    (out.parent / MERGE_REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"merged {len(file_keys)} adapter tensors; {len(names)} sampled weights changed")
    return report


# ---------------------------------------------------------------------------
# Stage drivers: train and heal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainPlan:
    """What a train or heal stage will run: nothing is written until :func:`run_plan` applies it."""

    stage: str
    run_dir: Path
    data: Path
    data_sha256: str
    base: str
    revision: str | None
    commands: tuple[tuple[str, ...], ...]
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    deviation_id: str | None = None
    heal_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_dir": str(self.run_dir),
            "data": str(self.data),
            "data_sha256": self.data_sha256,
            "base": self.base,
            "revision": self.revision,
            "commands": [list(cmd) for cmd in self.commands],
            "hyperparameters": dict(self.hyperparameters),
            "deviation_id": self.deviation_id,
            "heal_of": self.heal_of,
        }


def _gates(prereg_path: Path, lock_dir: Path, freeze: Path, deviation_id: str | None):
    """The pre-registration gate, then the frozen training set chosen by sha256."""
    require_registered(Path(prereg_path), Path(lock_dir), deviation_id)
    return select_frozen(Path(freeze), deviation_id=deviation_id)


def _hyper_flags(hyper: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in ("epochs", "lr", "rank", "alpha", "batch", "seed", "max_length"):
        flags += [f"--{key.replace('_', '-')}", f"{hyper[key]}"]
    return flags


def _commands(
    *,
    python: str,
    domain: str,
    data: Path,
    sha256: str,
    run_dir: Path,
    base: str,
    revision: str | None,
    prereg_path: Path,
    lock_dir: Path,
    hyper: Mapping[str, Any],
    val: Path | None,
    deviation_id: str | None,
    heal: bool,
    extra_args: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    train = [python, "-m", TRAIN_SCORER_MODULE, "--domain", domain]
    train += ["--train", str(data), "--expect-sha256", sha256, "--out", str(run_dir)]
    train += ["--base", base]
    train += ["--revision", revision] if revision else []
    train += ["--prereg", str(prereg_path), "--lock-dir", str(lock_dir)]
    train += ["--val", str(val)] if val is not None else []
    train += ["--deviation", deviation_id] if deviation_id else []
    train += _hyper_flags(hyper)
    train += ["--heal"] if heal else []
    train += list(extra_args)
    merge = [python, "-m", TRAIN_MODULE, "--merge-only", str(run_dir / "adapter")]
    merge += ["--out", str(run_dir), "--base", base]
    merge += ["--revision", revision] if revision else []
    return (tuple(train), tuple(merge))


def plan_train(
    *,
    run_dir: Path,
    domain: str,
    prereg_path: Path,
    lock_dir: Path,
    freeze: Path,
    base: str,
    revision: str | None,
    python: str,
    val: Path | None = None,
    deviation_id: str | None = None,
    extra_args: Sequence[str] = (),
    **hyper: Any,
) -> TrainPlan:
    """The train stage: prereg gate, frozen set by sha256, the trainer then the verified merge.

    *domain* is what :func:`jev_factory.domain.validate.load_domain` accepts
    (a dotted module or a JSON file); *hyper* overrides :data:`TRAIN_DEFAULTS`.
    """
    unknown = sorted(set(hyper) - set(TRAIN_DEFAULTS))
    if unknown:
        raise CliError(EXIT_USER_ERROR, f"unknown training knob(s): {', '.join(unknown)}")
    choice = _gates(prereg_path, lock_dir, freeze, deviation_id)
    knobs = {**TRAIN_DEFAULTS, **hyper}
    run_dir = Path(run_dir)
    commands = _commands(
        python=python,
        domain=domain,
        data=choice.path,
        sha256=choice.sha256,
        run_dir=run_dir,
        base=base,
        revision=revision,
        prereg_path=Path(prereg_path),
        lock_dir=Path(lock_dir),
        hyper=knobs,
        val=val,
        deviation_id=choice.deviation_id or deviation_id,
        heal=False,
        extra_args=extra_args,
    )
    return TrainPlan(
        stage="train",
        run_dir=run_dir,
        data=choice.path,
        data_sha256=choice.sha256,
        base=base,
        revision=revision,
        commands=commands,
        hyperparameters={**knobs, "precision": HEAL_PRECISION},
        deviation_id=choice.deviation_id or deviation_id,
    )


def _read_train_log(run: Path) -> dict[str, Any]:
    path = run / "train-log.json"
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CliError(
            EXIT_USER_ERROR,
            f"{path}: no readable train-log.json; heal continues a finished training run",
            "train the base run first",
        ) from None
    return log if isinstance(log, dict) else {}


def plan_heal(
    *,
    run_dir: Path,
    base_run: Path,
    domain: str,
    prereg_path: Path,
    lock_dir: Path,
    freeze: Path,
    python: str,
    val: Path | None = None,
    deviation_id: str | None = None,
    extra_args: Sequence[str] = (),
    **hyper: Any,
) -> TrainPlan:
    """The heal stage: 1 epoch, lr 5e-5, bf16, from *base_run*'s own merged checkpoint.

    Refuses a base run with no verified merge, a base run that is itself a
    heal (one round only), any epochs/lr other than the recipe, and a frozen
    set whose sha256 is not the one the base run trained on.
    """
    base_run = Path(base_run)
    merged = base_run / "merged"
    if not merged.is_dir() or not (base_run / MERGE_REPORT).is_file():
        raise CliError(
            EXIT_USER_ERROR,
            f"{base_run}: no verified merged checkpoint (merged/ and {MERGE_REPORT})",
            "heal continues a run whose merge was verified; train it first",
        )
    base_log = _read_train_log(base_run)
    if base_log.get("heal"):
        raise CliError(
            EXIT_USER_ERROR,
            f"{base_run.name} is itself a heal; healing is one round only",
            "drop the build that still fails rather than heal it again",
        )
    if hyper.get("epochs", HEAL_EPOCHS) != HEAL_EPOCHS:
        raise CliError(EXIT_USER_ERROR, f"heal is {HEAL_EPOCHS} epoch, not {hyper['epochs']}")
    if hyper.get("lr", HEAL_LR) != HEAL_LR:
        raise CliError(EXIT_USER_ERROR, f"heal is lr {HEAL_LR}, not {hyper['lr']}")
    unknown = sorted(set(hyper) - set(TRAIN_DEFAULTS))
    if unknown:
        raise CliError(EXIT_USER_ERROR, f"unknown training knob(s): {', '.join(unknown)}")
    choice = _gates(prereg_path, lock_dir, freeze, deviation_id)
    if base_log.get("train_sha256") != choice.sha256:
        raise CliError(
            EXIT_USER_ERROR,
            f"{base_run.name} trained on {str(base_log.get('train_sha256'))[:12]}..., not the"
            f" frozen set {choice.sha256[:12]}...; heal continues on the same frozen set",
            "heal the run that trained on this freeze, or record a deviation",
        )
    knobs = {**TRAIN_DEFAULTS, **hyper, "epochs": HEAL_EPOCHS, "lr": HEAL_LR}
    run_dir = Path(run_dir)
    commands = _commands(
        python=python,
        domain=domain,
        data=choice.path,
        sha256=choice.sha256,
        run_dir=run_dir,
        base=str(merged),
        revision=None,
        prereg_path=Path(prereg_path),
        lock_dir=Path(lock_dir),
        hyper=knobs,
        val=val,
        deviation_id=choice.deviation_id or deviation_id,
        heal=True,
        extra_args=extra_args,
    )
    return TrainPlan(
        stage="heal",
        run_dir=run_dir,
        data=choice.path,
        data_sha256=choice.sha256,
        base=str(merged),
        revision=None,
        commands=commands,
        hyperparameters={**knobs, "precision": HEAL_PRECISION},
        deviation_id=choice.deviation_id or deviation_id,
        heal_of=base_run.name,
    )


#: The files a finished train or heal run must leave in its run dir.
RUN_OUTPUTS = ("train-log.json", "row-maps.json", MERGE_REPORT)

#: The guarded GPU stage runner every applied command goes through.
DEFAULT_RUNNER: Callable[..., Any] = run_gpu_stage


def run_plan(
    plan: TrainPlan,
    *,
    apply: bool = False,
    allow_foreign: bool = False,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    watchdog_seconds: int = 5,
    gpu_memory_gb: float | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Dry run (the plan, nothing written) unless *apply*; then each command through the
    guarded GPU stage runner, then the greedy generation config in ``<run>/merged``."""
    if not apply:
        return {"applied": False, "plan": plan.to_dict()}
    if not memory_max:
        raise CliError(
            EXIT_ENV_ERROR,
            "a GPU stage needs a hard memory cap (run config key train_memory_max)",
            "set train_memory_max (for example 24G) in the run file or JEV_TRAIN_MEMORY_MAX",
        )
    environ = dict(os.environ if env is None else env)
    if gpu_memory_gb is not None:
        environ[GPU_MEMORY_ENV] = f"{gpu_memory_gb:g}"
    run = runner or DEFAULT_RUNNER
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    for cmd in plan.commands:
        done = run(
            plan.stage,
            plan.run_dir,
            list(cmd),
            allow_foreign=allow_foreign,
            memory_max=memory_max,
            memory_floor=memory_floor,
            watchdog_seconds=watchdog_seconds,
            env=environ,
        )
        if done.returncode == WATCHDOG_STATUS:
            raise CliError(
                EXIT_ENV_ERROR,
                f"{plan.stage}: the memory-floor watchdog stopped {cmd[2]}",
                "free memory on the machine or lower the batch/length, then rerun",
            )
        if done.returncode != 0:
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            raise CliError(
                EXIT_ENV_ERROR, f"{plan.stage}: {cmd[2]} exited {done.returncode}: {tail[0]}"
            )
    missing = [name for name in RUN_OUTPUTS if not (plan.run_dir / name).is_file()]
    if missing or not (plan.run_dir / "merged").is_dir():
        raise CliError(
            EXIT_ENV_ERROR,
            f"{plan.stage}: the run left no {', '.join(missing or ['merged/'])}",
            "check the trainer's log in the run dir",
        )
    gen_config.write(plan.run_dir / "merged")
    return {"applied": True, "plan": plan.to_dict(), "outputs": sorted(RUN_OUTPUTS)}


# ---------------------------------------------------------------------------
# CLI: the verified merge
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a scorer's LoRA adapter, verified.")
    parser.add_argument(
        "--merge-only",
        required=True,
        type=Path,
        metavar="ADAPTER",
        help="merge this saved adapter into the base and write --out/merged",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample", type=int, default=DEFAULT_WEIGHT_SAMPLE)
    return parser


def main(argv: list[str] | None = None) -> int:  # needs a GPU stack
    args = _parser().parse_args(argv)
    try:
        merge_adapter(
            args.base, args.revision, args.merge_only, args.out / "merged", sample=args.sample
        )
    except MergeError as exc:
        print(f"train: merge refused: {exc}", file=sys.stderr)
        return 1
    print(f"merged {args.merge_only} into {args.out / 'merged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
