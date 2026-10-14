"""Distill pre-registration: schema, lock, deviation refusal, knob/source checks."""

from __future__ import annotations

import copy
import json

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import prereg as dp


def good_doc() -> dict:
    return dp.example_doc()


def write(tmp_path, doc, name="prereg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_example_doc_validates_and_seeds_upstream_knobs():
    reg = dp.validate(good_doc())
    seeded = {k.name: k for k in reg.upstream_knobs}
    assert seeded["lr"].value == 1e-4
    assert seeded["head_lr"].value == 1e-3
    assert seeded["epochs"].value == 1
    assert seeded["kl_frozen_weight"].value == 0.3
    assert seeded["teacher_weight"].value == 1.0
    assert seeded["ordinal_smoothing"].value == 0.1
    assert all(
        k.source_file in ("train_config.json", "strands_decider_config.json") and k.source_key
        for k in seeded.values()
    )


#: The values read once from StrandsAgents/strands-decider-2B-hobson-v21 (raw/main).
TRAINER_UPSTREAM = {
    "micro_batch_size": (8, "train_config.json"),
    "grad_accum": (4, "train_config.json"),
    "weight_decay": (0.01, "train_config.json"),
    "warmup_ratio": (0.03, "train_config.json"),
    "max_grad_norm": (1.0, "train_config.json"),
    "max_length": (4096, "train_config.json"),
    "seed": (5, "train_config.json"),
    "shuffle_options": (True, "train_config.json"),
    "reverse_score_prob": (0.5, "train_config.json"),
    "lora_r": (16, "train_config.json"),
    "lora_alpha": (32, "train_config.json"),
    "lora_dropout": (0.05, "strands_decider_config.json"),
}


def test_example_doc_carries_every_knob_the_trainer_needs_with_cited_sources():
    from jev_factory.backbones.pointer import train_distill as td

    reg = dp.validate(good_doc())
    seeded = {k.name: k for k in reg.upstream_knobs}
    for name, (value, source) in TRAINER_UPSTREAM.items():
        assert seeded[name].value == value, name
        assert seeded[name].source_file == source, name
        assert seeded[name].source_key == name, name
    targets = seeded["lora_targets"]
    assert targets.source_file == "train_config.json" and targets.source_key == "lora_targets"
    assert "q_proj" in targets.value and "out_proj" in targets.value
    assert reg.knob_grid["init_mode"] == list(td.INIT_MODES)
    needed = set(td.REQUIRED_UPSTREAM) | set(td.LORA_UPSTREAM)
    assert needed <= set(seeded) | set(reg.knob_grid)


def test_example_doc_preregisters_the_confident_wrong_threshold():
    reg = dp.validate(good_doc())
    assert 0.0 < reg.selection_rule["confident_wrong_threshold"] <= 1.0


@pytest.mark.parametrize("init_mode", ["continue", "merge_fresh"])
def test_trainer_plan_accepts_the_example_doc(tmp_path, init_mode):
    from jev_factory.backbones.pointer import train_distill as td

    path = write(tmp_path, good_doc())
    run_dir = tmp_path / "run"
    dp.lock(path, run_dir)
    knobs = {
        "alpha": 0.7,
        "temperature": {"choice": 2.0, "score": 1.0, "noul": 1.0},
        "epochs": 1,
        "lr": 1e-4,
        "init_mode": init_mode,
    }
    plan = td.plan_train(
        prereg_path=path,
        run_dir=run_dir,
        run_knobs=knobs,
        train_file=tmp_path / "train.jsonl",
        init_from=tmp_path / "v21",
        output_dir=tmp_path / "out",
        trainer_python="python3",
        decider_repo=tmp_path / "strands-decider",
    )
    hp = plan["hyperparameters"]
    assert hp["init_mode"] == init_mode
    assert hp["seed"] == 5 and hp["micro_batch_size"] == 8
    if init_mode == "merge_fresh":
        assert hp["lora_r"] == 16 and hp["lora_dropout"] == 0.05
        assert hp["sources"]["lora_dropout"] == "upstream:strands_decider_config.json#lora_dropout"


def test_schema_covers_bars_rule_grid_and_references():
    reg = dp.validate(good_doc())
    assert reg.bars["test"]["ece_max"] == 0.10
    assert reg.bars["test"]["gap_closure_min"] == 0.5
    assert reg.bars["test"]["latency_ratio_max"] == 0.2
    assert reg.bars["quant"]["max_accuracy_drop_points"] == 3.0
    assert reg.bars["quant"]["ece_max"] == 0.10
    assert set(reg.knob_grid) >= set(dp.KNOB_NAMES)
    assert reg.knob_grid["teacher_filter"] == ["all", "agrees_with_gold"]
    assert set(reg.reference_rows) == {"clef", "stock_v21"}
    assert reg.selection_rule["order"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("bars"),
        lambda d: d["bars"]["test"].pop("ece_max"),
        lambda d: d["bars"]["test"].__setitem__("ece_max", 1.5),
        lambda d: d["bars"]["quant"].__setitem__("max_accuracy_drop_points", -1),
        lambda d: d["knob_grid"].pop("alpha"),
        lambda d: d["knob_grid"].__setitem__("teacher_filter", ["all", "bogus"]),
        lambda d: d["knob_grid"].__setitem__("alpha", [0.5, 1.5]),
        lambda d: d["knob_grid"].__setitem__("epochs", []),
        lambda d: d["reference_rows"].pop("clef"),
        lambda d: d.__setitem__("schema_version", 99),
        lambda d: d["selection_rule"].pop("order"),
    ],
)
def test_invalid_docs_are_refused(mutate):
    doc = good_doc()
    mutate(doc)
    with pytest.raises(CliError):
        dp.validate(doc)


def test_upstream_value_without_cited_source_is_refused_by_name():
    doc = good_doc()
    del doc["upstream_knobs"][0]["source_key"]
    with pytest.raises(dp.UpstreamSourceMissingError) as exc:
        dp.validate(doc)
    assert doc["upstream_knobs"][0]["name"] in exc.value.message
    doc = good_doc()
    doc["upstream_knobs"][1]["source_file"] = "  "
    with pytest.raises(dp.UpstreamSourceMissingError):
        dp.validate(doc)


def test_lock_then_require_locked_roundtrip(tmp_path):
    path = write(tmp_path, good_doc())
    run_dir = tmp_path / "run"
    lock = dp.lock(path, run_dir)
    assert (run_dir / "distill" / "prereg.lock.json").exists()
    assert lock["sha256"] == dp.sha256_file(path)
    assert dp.require_locked(path, run_dir).bars["quant"]["ece_max"] == 0.10


def test_second_lock_is_refused(tmp_path):
    path = write(tmp_path, good_doc())
    dp.lock(path, tmp_path / "run")
    with pytest.raises(CliError):
        dp.lock(path, tmp_path / "run")


def test_does_not_touch_factory_prereg_lock(tmp_path):
    path = write(tmp_path, good_doc())
    run_dir = tmp_path / "run"
    dp.lock(path, run_dir)
    assert not (run_dir / "prereg.lock.json").exists()


def test_require_locked_without_lock_is_refused(tmp_path):
    path = write(tmp_path, good_doc())
    with pytest.raises(CliError):
        dp.require_locked(path, tmp_path / "run")


def test_changed_file_refused_without_deviation_id(tmp_path):
    path = write(tmp_path, good_doc())
    run_dir = tmp_path / "run"
    dp.lock(path, run_dir)
    changed = copy.deepcopy(good_doc())
    changed["bars"]["test"]["ece_max"] = 0.2
    write(tmp_path, changed)
    with pytest.raises(CliError) as exc:
        dp.require_locked(path, run_dir)
    assert "changed" in exc.value.message
    with pytest.raises(CliError):
        dp.require_locked(path, run_dir, deviation_id="  ")
    assert dp.require_locked(path, run_dir, deviation_id="d17").bars["test"]["ece_max"] == 0.2


def test_run_knob_missing_from_prereg_is_refused_by_name():
    reg = dp.validate(good_doc())
    dp.require_knobs(reg, {"alpha": 0.5, "epochs": 1})
    with pytest.raises(dp.UnregisteredKnobError) as exc:
        dp.require_knobs(reg, {"alpha": 0.5, "warmup_ratio": 0.1})
    assert "warmup_ratio" in exc.value.message


def test_run_knob_value_outside_grid_is_refused():
    reg = dp.validate(good_doc())
    with pytest.raises(dp.UnregisteredKnobError) as exc:
        dp.require_knobs(reg, {"alpha": 0.123})
    assert "alpha" in exc.value.message


def test_per_kind_temperature_is_checked():
    reg = dp.validate(good_doc())
    kind = next(iter(reg.knob_grid["temperature"]))
    value = reg.knob_grid["temperature"][kind][0]
    dp.require_knobs(reg, {"temperature": {kind: value}})
    with pytest.raises(dp.UnregisteredKnobError):
        dp.require_knobs(reg, {"temperature": {kind: 99.0}})
    with pytest.raises(dp.UnregisteredKnobError):
        dp.require_knobs(reg, {"temperature": {"no_such_kind": 1.0}})


def test_upstream_knob_names_are_registered_run_knobs():
    reg = dp.validate(good_doc())
    dp.require_knobs(reg, {"head_lr": 1e-3, "kl_frozen_weight": 0.3})


def test_example_doc_registers_one_heal_round_with_cited_knobs_in_the_grid():
    from jev_factory.distill import quant

    doc = dp.example_doc()
    reg = dp.validate(doc)
    knobs = quant.heal_knobs(reg)
    assert knobs == {"epochs": 1, "lr": 5e-5, "init_mode": "continue"}
    assert "d18" in doc["heal"]["source"] and "5e-5" in doc["heal"]["source"]
    for name in ("epochs", "lr", "init_mode"):
        assert knobs[name] in doc["knob_grid"][name]
    assert doc["bars"]["quant"]["heal_rounds_max"] == 1
