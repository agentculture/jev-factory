"""jev_factory.backbones.causal_lm.train: merge verification, heal, the GPU cap (t22).

Ported from nvsh's tests/test_lfm_finetune_train.py (the scorer-path helpers;
the Track A tokenisation tests are not carried over, Track A is not jev-like)
plus the merge-verification and heal/stage-driver tests this task adds.
Nothing here needs torch, peft, a GPU or a model download: the merge checks
run on fake state dicts and fake warning streams.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import gen_config
from jev_factory.backbones.causal_lm import train as tr
from jev_factory.cli._errors import CliError
from jev_factory.data import assemble as asm
from jev_factory.factory import prereg
from tests.fixtures.toy_domain import SEED_CORPUS

SEED = json.loads(SEED_CORPUS.read_text(encoding="utf-8"))
DOMAIN_SOURCE = "tests.fixtures.toy_domain"


# ---------------------------------------------------------------------------
# merge verification (behavioral o15): missing-adapter-key warnings, unchanged weights
# ---------------------------------------------------------------------------

_FILE_KEYS = [
    "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
    "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight",
]
_LOADED = [key.replace(".weight", ".default.weight") for key in _FILE_KEYS]
_MODULES = [
    "base_model.model.model.layers.0.mlp.down_proj",
    "base_model.model.model.layers.1.self_attn.q_proj",
]
_BASE = {_MODULES[0]: [[0.5, 0.25], [1.0, -1.0]], _MODULES[1]: [[2.0, 0.0], [0.0, 2.0]]}
_MERGED = {_MODULES[0]: [[0.5, 0.26], [1.0, -1.0]], _MODULES[1]: [[2.0, 0.1], [0.0, 2.0]]}


def _verify(**overrides):
    kwargs = {
        "file_keys": _FILE_KEYS,
        "loaded_keys": _LOADED,
        "load_warnings": [],
        "before": _BASE,
        "after": _MERGED,
        "architectures": ["Qwen3_5ForCausalLM"],
        "saved_keys": ["model.layers.0.mlp.down_proj.weight", "lm_head.weight"],
    }
    kwargs.update(overrides)
    return tr.verify_merge(**kwargs)


@pytest.mark.behavioral("o15")
def test_a_clean_merge_passes_and_reports_zero_missing_keys() -> None:
    report = _verify()
    assert report["adapter_tensors"] == 4
    assert report["missing_adapter_keys"] == 0
    assert report["load_warnings"] == 0
    assert report["sampled_weights"] == sorted(_MODULES)
    assert report["changed_weights"] == 2


@pytest.mark.behavioral("o15")
def test_a_missing_adapter_key_warning_fails_the_merge() -> None:
    warning = f"Found missing adapter keys while loading the checkpoint: {_FILE_KEYS[:1]}"
    with pytest.raises(tr.MergeError, match="missing adapter key"):
        _verify(load_warnings=[warning])


@pytest.mark.behavioral("o15")
@pytest.mark.parametrize(
    "message",
    [
        "Found missing adapter keys while loading the checkpoint: ['x.lora_A.weight']",
        "found MISSING ADAPTER KEYS: ['y']",
        "Missing keys when loading adapter: ['z']",
    ],
)
def test_every_spelling_of_the_missing_key_warning_is_caught(message: str) -> None:
    with pytest.raises(tr.MergeError):
        tr.check_load_warnings([message])


@pytest.mark.behavioral("o15")
def test_unrelated_load_warnings_do_not_fail_the_merge() -> None:
    tr.check_load_warnings(["`torch_dtype` is deprecated! Use `dtype` instead!"])


@pytest.mark.behavioral("o15")
def test_a_sampled_merged_weight_equal_to_the_base_fails_the_merge() -> None:
    merged = {**_MERGED, _MODULES[1]: [row[:] for row in _BASE[_MODULES[1]]]}
    with pytest.raises(tr.MergeError, match="equals the base"):
        _verify(after=merged)


@pytest.mark.behavioral("o15")
def test_a_merge_that_changed_nothing_at_all_fails() -> None:
    unchanged = {name: [row[:] for row in value] for name, value in _BASE.items()}
    with pytest.raises(tr.MergeError, match="equals the base"):
        _verify(after=unchanged)


@pytest.mark.behavioral("o15")
def test_an_adapter_tensor_that_did_not_load_fails_even_without_a_warning() -> None:
    with pytest.raises(tr.MergeError, match="3 of 4"):
        _verify(loaded_keys=_LOADED[:3])
    with pytest.raises(tr.MergeError, match="none"):
        _verify(loaded_keys=[])


@pytest.mark.behavioral("o15")
def test_no_sampled_weights_or_a_weight_missing_after_merge_fails() -> None:
    with pytest.raises(tr.MergeError, match="no adapted weight"):
        _verify(before={}, after={})
    with pytest.raises(tr.MergeError, match="missing after the merge"):
        _verify(after={_MODULES[0]: _MERGED[_MODULES[0]]})


@pytest.mark.behavioral("o15")
def test_the_warning_capture_sees_warnings_and_log_records() -> None:
    with tr.capture_load_warnings() as messages:
        warnings.warn("Found missing adapter keys while loading the checkpoint: ['a']")
        logging.getLogger("peft").warning("Found missing adapter keys: ['b']")
        logging.getLogger("transformers.modeling_utils").warning("Missing keys: ['c']")
    assert len(messages) == 3
    with pytest.raises(tr.MergeError, match="3 missing adapter key warning"):
        tr.check_load_warnings(messages)


def test_the_warning_capture_leaves_the_loggers_as_it_found_them() -> None:
    before = {name: list(logging.getLogger(name).handlers) for name in ("", "peft", "transformers")}
    with tr.capture_load_warnings():
        pass
    after = {name: list(logging.getLogger(name).handlers) for name in ("", "peft", "transformers")}
    assert before == after


class _Tensor:
    """A torch-like tensor: ``equal`` is how torch compares two whole tensors."""

    def __init__(self, values) -> None:
        self.values = list(values)

    def equal(self, other: "_Tensor") -> bool:
        return self.values == other.values


@pytest.mark.behavioral("o15")
def test_weights_are_compared_with_the_tensor_equal_method_when_there_is_one() -> None:
    assert tr.weights_equal(_Tensor([1, 2]), _Tensor([1, 2]))
    assert not tr.weights_equal(_Tensor([1, 2]), _Tensor([1, 3]))
    same = {"m": _Tensor([1.0, 2.0])}
    equal = {"m": _Tensor([1.0, 2.0])}
    with pytest.raises(tr.MergeError, match="equals the base"):
        tr.check_weights_changed(same, equal)
    assert tr.check_weights_changed(same, {"m": _Tensor([1.0, 2.5])}) == ["m"]


@pytest.mark.behavioral("o15")
def test_real_tensors_equal_to_the_base_fail_the_merge() -> None:
    torch = pytest.importorskip("torch")
    base = {"m": torch.ones(2, 2, dtype=torch.bfloat16)}
    copy = {"m": base["m"].clone()}
    with pytest.raises(tr.MergeError, match="equals the base"):
        tr.check_weights_changed(base, copy)
    nudged = base["m"].clone()
    nudged[0, 0] += 0.5
    assert tr.check_weights_changed(base, {"m": nudged}) == ["m"]


def test_the_weight_sample_is_deterministic_and_keeps_the_first_and_last() -> None:
    names = [f"layers.{i}.q_proj" for i in range(40)]
    sample = tr.sample_names(names, k=6, seed=3)
    assert sample == tr.sample_names(list(reversed(names)), k=6, seed=3)
    assert len(sample) == 6
    assert sorted(names)[0] in sample and sorted(names)[-1] in sample
    assert tr.sample_names(names[:3], k=8) == sorted(names[:3])
    assert tr.sample_names([], k=8) == []


@pytest.mark.behavioral("o15")
def test_the_merge_must_be_the_text_only_causal_lm() -> None:
    with pytest.raises(tr.MergeError, match="text-only"):
        _verify(architectures=["Qwen3_5ForConditionalGeneration"])
    with pytest.raises(tr.MergeError, match="language_model"):
        _verify(saved_keys=["model.language_model.language_model.layers.0.mlp.down_proj.weight"])
    tr.check_text_only(["Qwen3_5ForCausalLM"], ["model.layers.0.mlp.down_proj.weight"])


# ---------------------------------------------------------------------------
# Qwen3.5 specifics (ported): VL keys map onto the text-only model; processor.tokenizer
# ---------------------------------------------------------------------------


def test_vision_language_adapter_keys_map_onto_the_text_model() -> None:
    vl = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    text = "base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight"
    assert tr.text_adapter_key(vl) == "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    assert tr.text_adapter_key(text) == text


def test_adapter_leaf_modules_come_from_the_keys() -> None:
    keys = [
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
        "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight",
    ]
    assert tr.adapter_leaf_modules(keys) == ["down_proj", "in_proj_qkv"]


def test_check_adapter_loaded_matches_peft_default_adapter_names() -> None:
    file_keys = _FILE_KEYS[:2]
    tr.check_adapter_loaded(file_keys, _LOADED[:2])  # does not raise
    with pytest.raises(ValueError, match="1 of 2"):
        tr.check_adapter_loaded(file_keys, _LOADED[:1])
    with pytest.raises(ValueError, match="none"):
        tr.check_adapter_loaded(file_keys, [])


def test_text_tokenizer_unwraps_a_multimodal_processor() -> None:
    class Tok:
        pass

    class Processor:
        tokenizer = Tok()

    tok = Tok()
    assert tr.text_tokenizer(tok) is tok
    assert tr.text_tokenizer(Processor()) is Processor.tokenizer


# ---------------------------------------------------------------------------
# LoRA targets, generation config, GPU memory cap (ported)
# ---------------------------------------------------------------------------


def test_default_targets_are_attention_and_mlp_and_gdn_adds_linear_attention() -> None:
    assert tr.lora_targets("attn-mlp") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    gdn = tr.lora_targets("attn-mlp-gdn")
    assert gdn[:7] == tr.lora_targets("attn-mlp")
    assert set(gdn[7:]) == {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    with pytest.raises(ValueError, match="targets"):
        tr.lora_targets("everything")


class _GenConfig:
    def __init__(self, temperature, do_sample):
        self.temperature = temperature
        self.do_sample = do_sample


class _Model:
    def __init__(self, generation_config):
        self.generation_config = generation_config


def test_a_greedy_generation_config_is_made_save_valid() -> None:
    model = _Model(_GenConfig(0.0, False))
    tr.save_valid_generation_config(model)
    assert model.generation_config.temperature is None
    assert model.generation_config.do_sample is False
    sampling = _Model(_GenConfig(0.7, True))
    tr.save_valid_generation_config(sampling)
    assert sampling.generation_config.temperature == 0.7
    bare = _Model(None)
    tr.save_valid_generation_config(bare)
    assert bare.generation_config is None


def test_the_gpu_memory_fraction_is_the_budget_over_the_device_total() -> None:
    assert tr.gpu_memory_fraction("8", 128 * 2**30) == 0.0625
    assert tr.gpu_memory_fraction(" 12.5 ", 100 * 2**30) == 0.125


@pytest.mark.parametrize("value", ["0", "-4", "lots", "", "nan", "inf", "129"])
def test_a_gpu_memory_budget_that_is_not_a_positive_fit_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match=tr.GPU_MEMORY_ENV):
        tr.gpu_memory_fraction(value, 128 * 2**30)


class _FakeCuda:
    def __init__(self, total: int | None) -> None:
        self.total = total
        self.fractions: list[float] = []

    def is_available(self) -> bool:
        return self.total is not None

    def get_device_properties(self, device: int):
        return type("Props", (), {"total_memory": self.total})()

    def set_per_process_memory_fraction(self, fraction: float, device: int = 0) -> None:
        self.fractions.append(fraction)


class _FakeTorch:
    def __init__(self, total: int | None) -> None:
        self.cuda = _FakeCuda(total)


def test_the_gpu_memory_cap_is_set_from_the_environment(capsys) -> None:
    torch = _FakeTorch(128 * 2**30)
    assert tr.cap_gpu_memory(torch, {tr.GPU_MEMORY_ENV: "8"}) == 0.0625
    assert torch.cuda.fractions == [0.0625]
    assert "0.0625" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, "", "   ", "\t"])
def test_no_or_an_empty_gpu_memory_budget_leaves_torch_alone(value) -> None:
    torch = _FakeTorch(128 * 2**30)
    environ = {} if value is None else {tr.GPU_MEMORY_ENV: value}
    assert tr.cap_gpu_memory(torch, environ) is None
    assert torch.cuda.fractions == []


def test_a_gpu_memory_budget_over_the_device_is_refused_before_any_cap() -> None:
    torch = _FakeTorch(128 * 2**30)
    with pytest.raises(ValueError, match="exceeds"):
        tr.cap_gpu_memory(torch, {tr.GPU_MEMORY_ENV: "200"})
    assert torch.cuda.fractions == []


def test_the_gpu_memory_variable_is_the_run_configs() -> None:
    from jev_factory.factory import config

    assert tr.GPU_MEMORY_ENV == config._BY_NAME["train_gpu_memory_gb"].env_var


def test_the_merge_cli_needs_merge_only(capsys) -> None:
    args = tr._parser().parse_args(["--merge-only", "adapter", "--out", "o", "--base", "b"])
    assert str(args.merge_only) == "adapter"
    parser = tr._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "o"])


# ---------------------------------------------------------------------------
# stage drivers: train and heal (prereg, frozen data by sha256, GPU guard, dry-run)
# ---------------------------------------------------------------------------


def _prereg_doc() -> dict:
    return prereg.apply_defaults(_prereg_draft())


def _prereg_draft() -> dict:
    return {
        "schema_version": 1,
        "stock_baseline_record_id": "8b59d0afa150",
        "perms_per_entry": 8,
        "candidates": ["r1"],
        "bars": {
            "wrong_mutating": {"stock": 0.02, "minimum": 0.0},
            "ece": {"stock": 0.25, "minimum": 0.03},
            "permutation_change": {"stock": 0.3, "minimum": 0.05},
            "mc_escalation": {"stock": 0.1, "minimum": 0.8},
            "right_proposals": {"stock": 0.072},
        },
    }


@pytest.fixture
def work(tmp_path) -> dict:
    """A work dir with a registered pre-registration and an assembled, frozen training set."""
    root = tmp_path / "work"
    split = root / "splits" / "train.json"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "header": "Toy. Split 'train' of toy (seed=1).",
                "entries": SEED["entries"],
                "world": SEED["world"],
            }
        )
    )
    data = root / "data"
    asm.assemble(data, domain=__import__(DOMAIN_SOURCE, fromlist=["DOMAIN"]).DOMAIN, split=split)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(_prereg_doc()))
    prereg.register(prereg_path, root)
    return {
        "root": root,
        "prereg": prereg_path,
        "freeze": data / asm.FREEZE_NAME,
        "scorer_train": data / asm.SCORER_TRAIN_NAME,
    }


def _plan(work, run_dir=None, **kwargs):
    return tr.plan_train(
        run_dir=run_dir or work["root"] / "runs" / "scorer-r1",
        domain=DOMAIN_SOURCE,
        prereg_path=work["prereg"],
        lock_dir=work["root"],
        freeze=work["freeze"],
        base="example-org/base-model",
        revision="0123abc",
        python="/opt/train/bin/python",
        **kwargs,
    )


def test_training_refuses_without_a_registered_preregistration(tmp_path, work) -> None:
    with pytest.raises(CliError, match="no pre-registration"):
        _plan({**work, "root": tmp_path / "elsewhere"}, run_dir=tmp_path / "run")


def test_training_selects_the_frozen_set_by_sha256_not_by_path(work) -> None:
    plan = _plan(work)
    assert plan.data == work["scorer_train"]
    assert plan.data_sha256 == asm.sha256_path(work["scorer_train"])
    # A copy renamed next to the freeze is still found by its bytes.
    moved = work["scorer_train"].with_name("renamed.json")
    work["scorer_train"].rename(moved)
    assert _plan(work).data == moved


def test_training_on_changed_frozen_data_needs_a_deviation(work) -> None:
    work["scorer_train"].write_text(work["scorer_train"].read_text() + " ")
    with pytest.raises(CliError, match="deviation"):
        _plan(work)
    plan = _plan(work, deviation_id="dv-7")
    assert plan.deviation_id == "dv-7"


def test_the_train_plan_runs_the_scorer_trainer_then_the_verified_merge(work) -> None:
    plan = _plan(work)
    train_cmd, merge_cmd = plan.commands
    assert train_cmd[:3] == ("/opt/train/bin/python", "-m", tr.TRAIN_SCORER_MODULE)
    assert train_cmd[train_cmd.index("--expect-sha256") + 1] == plan.data_sha256
    assert train_cmd[train_cmd.index("--prereg") + 1] == str(work["prereg"])
    assert "--heal" not in train_cmd
    assert merge_cmd[:3] == ("/opt/train/bin/python", "-m", tr.TRAIN_MODULE)
    assert merge_cmd[merge_cmd.index("--merge-only") + 1] == str(plan.run_dir / "adapter")
    assert plan.hyperparameters["epochs"] == 3 and plan.hyperparameters["lr"] == 2e-4


def _fake_runner(calls: list, *, status: int = 0, write: bool = True):
    def runner(stage, run_dir, cmd, *, allow_foreign=False, **capped):
        calls.append({"stage": stage, "cmd": list(cmd), "allow": allow_foreign, **capped})
        run = Path(run_dir)
        if write and "--merge-only" in cmd:
            merged = run / "merged"
            merged.mkdir(parents=True, exist_ok=True)
            (merged / "config.json").write_text(json.dumps({"eos_token_id": 7}))
            (run / tr.MERGE_REPORT).write_text(json.dumps({"missing_adapter_keys": 0}))
        elif write:
            run.mkdir(parents=True, exist_ok=True)
            (run / "train-log.json").write_text("{}")
            (run / "row-maps.json").write_text("[]")
        return type("Done", (), {"returncode": status, "stdout": "", "stderr": "boom"})()

    return runner


def test_a_train_plan_is_a_dry_run_unless_applied(work) -> None:
    plan = _plan(work)
    calls: list = []
    result = tr.run_plan(plan, memory_max="24G", runner=_fake_runner(calls))
    assert result["applied"] is False
    assert calls == []
    assert not plan.run_dir.exists()


def test_an_applied_train_goes_through_the_gpu_stage_runner(work) -> None:
    plan = _plan(work)
    calls: list = []
    result = tr.run_plan(
        plan, apply=True, memory_max="24G", gpu_memory_gb=8, runner=_fake_runner(calls)
    )
    assert result["applied"] is True
    assert [call["stage"] for call in calls] == ["train", "train"]
    assert all(call["memory_max"] == "24G" for call in calls)
    assert calls[0]["env"][tr.GPU_MEMORY_ENV] == "8"
    assert gen_config.check(plan.run_dir / "merged") is None  # greedy config written
    assert result["outputs"] == sorted(["train-log.json", "row-maps.json", tr.MERGE_REPORT])


def test_the_default_runner_is_the_guarded_gpu_stage() -> None:
    from jev_factory.factory import gpu

    assert tr.DEFAULT_RUNNER is gpu.run_gpu_stage


def test_a_failed_or_watchdog_stopped_stage_is_an_error(work) -> None:
    plan = _plan(work)
    failed = _fake_runner([], status=1)
    with pytest.raises(CliError, match="exited 1"):
        tr.run_plan(plan, apply=True, memory_max="24G", runner=failed)
    stopped = _fake_runner([], status=3)
    with pytest.raises(CliError, match="watchdog"):
        tr.run_plan(plan, apply=True, memory_max="24G", runner=stopped)


def test_a_run_that_leaves_no_train_log_or_row_maps_is_an_error(work) -> None:
    plan = _plan(work)
    runner = _fake_runner([], write=False)
    with pytest.raises(CliError, match="train-log.json"):
        tr.run_plan(plan, apply=True, memory_max="24G", runner=runner)


def test_applying_needs_a_memory_cap(work) -> None:
    plan, runner = _plan(work), _fake_runner([])
    with pytest.raises(CliError, match="train_memory_max"):
        tr.run_plan(plan, apply=True, memory_max=None, runner=runner)


def _trained_run(work, name="scorer-r1", **log) -> Path:
    run = work["root"] / "runs" / name
    (run / "merged").mkdir(parents=True)
    (run / tr.MERGE_REPORT).write_text(json.dumps({"missing_adapter_keys": 0}))
    record = {"train_sha256": asm.sha256_path(work["scorer_train"]), "heal": None, **log}
    (run / "train-log.json").write_text(json.dumps(record))
    return run


def _heal(work, base_run, **kwargs):
    return tr.plan_heal(
        run_dir=work["root"] / "runs" / "scorer-r1-heal",
        base_run=base_run,
        domain=DOMAIN_SOURCE,
        prereg_path=work["prereg"],
        lock_dir=work["root"],
        freeze=work["freeze"],
        python="/opt/train/bin/python",
        **kwargs,
    )


def test_heal_is_one_epoch_lr_5e_5_bf16_from_the_runs_own_merged_checkpoint(work) -> None:
    base_run = _trained_run(work)
    plan = _heal(work, base_run)
    assert plan.stage == "heal"
    assert (tr.HEAL_EPOCHS, tr.HEAL_LR, tr.HEAL_PRECISION) == (1, 5e-5, "bf16")
    assert plan.hyperparameters["epochs"] == 1
    assert plan.hyperparameters["lr"] == 5e-5
    assert plan.hyperparameters["precision"] == "bf16"
    assert plan.base == str(base_run / "merged")
    assert plan.revision is None
    train_cmd, merge_cmd = plan.commands
    assert "--heal" in train_cmd
    assert train_cmd[train_cmd.index("--base") + 1] == str(base_run / "merged")
    assert merge_cmd[merge_cmd.index("--base") + 1] == str(base_run / "merged")
    assert "--revision" not in merge_cmd


def test_heal_trains_on_the_same_frozen_set_as_its_base_run(work) -> None:
    plan = _heal(work, _trained_run(work))
    assert plan.data_sha256 == asm.sha256_path(work["scorer_train"])
    other = _trained_run(work, name="scorer-r2", train_sha256="0" * 64)
    with pytest.raises(CliError, match="same frozen"):
        _heal(work, other)


def test_heal_runs_through_the_heal_gpu_stage(work) -> None:
    plan = _heal(work, _trained_run(work))
    calls: list = []
    tr.run_plan(plan, apply=True, memory_max="24G", runner=_fake_runner(calls))
    assert [call["stage"] for call in calls] == ["heal", "heal"]


def test_heal_refuses_a_run_that_is_itself_a_heal_or_was_never_merged(work) -> None:
    healed = _trained_run(work, name="scorer-r1-heal0", heal={"of": "x"})
    with pytest.raises(CliError, match="one round"):
        _heal(work, healed)
    unmerged = work["root"] / "runs" / "scorer-r3"
    unmerged.mkdir(parents=True)
    with pytest.raises(CliError, match="merged"):
        _heal(work, unmerged)


def test_heal_refuses_hyperparameters_other_than_the_recipe(work) -> None:
    base_run = _trained_run(work)
    with pytest.raises(CliError, match="1 epoch"):
        _heal(work, base_run, epochs=2)
    other_run = _trained_run(work, name="scorer-r4")
    with pytest.raises(CliError, match="5e-05"):
        _heal(work, other_run, lr=1e-4)


def test_plan_to_dict_is_json_safe(work) -> None:
    doc = _plan(work).to_dict()
    assert json.loads(json.dumps(doc))["stage"] == "train"
