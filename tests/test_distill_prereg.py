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
        k.source_file
        in ("train_config.json", "strands_decider_config.json", "lora/adapter_config.json")
        and k.source_key
        for k in seeded.values()
    )


#: The values read once from StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 @ 216d1ccb.
TRAINER_UPSTREAM = {
    "micro_batch_size": (8, "train_config.json"),
    "grad_accum": (4, "train_config.json"),
    "weight_decay": (0.01, "train_config.json"),
    "warmup_ratio": (0.03, "train_config.json"),
    "max_grad_norm": (1.0, "train_config.json"),
    "max_length": (4096, "train_config.json"),
    "seed": (0, "train_config.json"),
    "shuffle_options": (True, "train_config.json"),
    "reverse_score_prob": (0.5, "train_config.json"),
}


def test_example_doc_carries_every_knob_the_trainer_needs_with_cited_sources():
    from jev_factory.backbones.pointer import train_distill as td

    reg = dp.validate(good_doc())
    seeded = {k.name: k for k in reg.upstream_knobs}
    for name, (value, source) in TRAINER_UPSTREAM.items():
        assert seeded[name].value == value, name
        assert seeded[name].source_file == source, name
        assert seeded[name].source_key == name, name
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
    assert hp["seed"] == 0
    assert hp["micro_batch_size"] == 8
    if init_mode == "merge_fresh":
        assert hp["lora_r"] == 48
        assert hp["lora_dropout"] == 0.05
        assert hp["sources"]["lora_dropout"] == "upstream:lora/adapter_config.json#lora_dropout"


def test_schema_covers_bars_rule_grid_and_references():
    reg = dp.validate(good_doc())
    assert reg.bars["test"]["ece_max"] == 0.10
    assert reg.bars["test"]["gap_closure_min"] == 0.5
    assert reg.bars["test"]["latency_ratio_max"] == 0.2
    assert reg.bars["quant"]["max_accuracy_drop_points"] == 3.0
    assert reg.bars["quant"]["ece_max"] == 0.10
    assert set(reg.knob_grid) >= set(dp.KNOB_NAMES)
    assert reg.knob_grid["teacher_filter"] == ["all", "agrees_with_gold"]
    assert set(reg.reference_rows) == {"clef", "stock"}
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
        lambda d: d["bars"].pop("c17"),
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
    assert "d18" in doc["heal"]["source"]
    assert "5e-5" in doc["heal"]["source"]
    for name in ("epochs", "lr", "init_mode"):
        assert knobs[name] in doc["knob_grid"][name]
    assert doc["bars"]["quant"]["heal_rounds_max"] == 1


# --------------------------------------------------------------------- the decision gate (d4)


def test_example_doc_registers_a_per_kind_gate_objective_and_grid_as_placeholders():
    reg = dp.validate(good_doc())
    gate = reg.gate
    assert gate is not None
    assert gate["objective"] == "max_selective_accuracy"
    assert set(gate["min_coverage"]) == set(dp.GATE_KINDS) == {"choice", "noul", "score"}
    assert all(0.0 < v <= 1.0 for v in gate["min_coverage"].values())
    assert set(gate["grid"]) == {"floor", "margin", "max_entropy"}
    for values in gate["grid"].values():
        assert None in values  # a disabled threshold is always on the grid
    assert "placeholder" in gate["status"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: g.update(objective="max_accuracy"),
        lambda g: g["min_coverage"].pop("noul"),
        lambda g: g["min_coverage"].update(choice=1.5),
        lambda g: g["grid"].pop("margin"),
        lambda g: g["grid"].update(floor=[]),
        lambda g: g["grid"].update(floor=[None, 1.2]),
        lambda g: g["grid"].update(max_entropy=["x"]),
    ],
)
def test_an_invalid_gate_block_is_refused(mutate):
    doc = good_doc()
    mutate(doc["gate"])
    with pytest.raises(CliError):
        dp.validate(doc)


def test_a_prereg_without_a_gate_block_still_validates():
    doc = good_doc()
    del doc["gate"]
    assert dp.validate(doc).gate is None


def test_require_locked_refuses_a_lock_symlinked_from_another_run(tmp_path):
    """A lock file borrowed from another run by symlink must not unlock this one."""
    path = write(tmp_path, good_doc())
    other = tmp_path / "other"
    dp.lock(path, other)
    run_dir = tmp_path / "run"
    (run_dir / "distill").mkdir(parents=True)
    (run_dir / "distill" / "prereg.lock.json").symlink_to(other / "distill" / "prereg.lock.json")
    with pytest.raises(CliError, match="escapes"):
        dp.require_locked(path, run_dir)


def test_lock_refuses_a_lock_directory_symlinked_out_of_the_run(tmp_path):
    path = write(tmp_path, good_doc())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (run_dir / "distill").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(CliError, match="escapes"):
        dp.lock(path, run_dir)
    assert not (tmp_path / "elsewhere" / "prereg.lock.json").exists()


# --------------------------------------------------------------------- 2610 seed and c17 bars

TARGETS_2610 = [
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
]


def seeded():
    return {k.name: k for k in dp.validate(good_doc()).upstream_knobs}


def test_example_doc_seeds_from_2610_at_the_pinned_revision():
    """prereg.example_doc seeds from 2610: source URL at 216d1ccb."""
    assert "strands-decider-2B-qwen3.5-v1-2610" in dp.UPSTREAM_SOURCE
    assert dp.UPSTREAM_SOURCE.endswith("/raw/216d1ccb46a9e9b036eb9187b248db98480f77d4")
    assert "hobson" not in dp.UPSTREAM_SOURCE
    stock = good_doc()["reference_rows"]["stock"]["model"]
    assert stock == "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610@" + dp.UPSTREAM_REVISION


def test_lora_hyperparameters_come_from_the_2610_adapter_config():
    """lora_r 48, lora_alpha 96, lora_dropout 0.05, each citing lora/adapter_config.json."""
    knobs = seeded()
    for name, value, key in (
        ("lora_r", 48, "r"),
        ("lora_alpha", 96, "lora_alpha"),
        ("lora_dropout", 0.05, "lora_dropout"),
    ):
        assert knobs[name].value == value
        assert knobs[name].source_file == "lora/adapter_config.json"
        assert knobs[name].source_key == key


def test_lora_targets_are_exactly_the_12_modules_of_2610():
    """lora_targets exactly the 12 modules in 2610's lora/adapter_config.json."""
    targets = seeded()["lora_targets"]
    assert targets.source_file == "lora/adapter_config.json"
    assert targets.source_key == "target_modules"
    assert len(targets.value) == 12
    assert sorted(targets.value) == sorted(TARGETS_2610)
    assert tuple(targets.value) == dp.STOCK_LORA_TARGETS


def test_temperature_by_kind_is_seeded_from_the_2610_decider_config():
    """temperature_by_kind from its strands_decider_config.json."""
    knob = seeded()["temperature_by_kind"]
    assert knob.source_file == "strands_decider_config.json"
    assert knob.source_key == "temperature_by_kind"
    assert knob.value["choice"] == 0.8629556148274228
    assert knob.value["noul"] == 1.0704394404346953
    assert knob.value["score"] == 1.834409163182587


def test_the_size_budget_is_imported_not_duplicated():
    from jev_factory.backbones.pointer import quantized

    assert dp.SIZE_BUDGET_BYTES is quantized.SIZE_BUDGET_BYTES
    assert dp.SIZE_BUDGET_BYTES == 10**9


def test_c17_bars_are_registered_for_both_builds():
    """Both the bf16 student and the shipped quant carry the top-1 bars; latency has no bar."""
    c17 = dp.validate(good_doc()).bars["c17"]
    assert set(c17) == {"bf16", "quant"}
    for build in ("bf16", "quant"):
        assert c17[build]["top1_over_stock_min_points"] == 10.0
        assert c17[build]["top1_under_clef_w4a16_max_points"] == 5.0
        assert c17[build]["latency_reported"] == ["median", "p95"]
        assert not any("latency" in n and n != "latency_reported" for n in c17[build])
    assert c17["quant"]["quant_vs_bf16_agreement_min"] == 0.97
    assert c17["quant"]["weight_bytes_max"] == 10**9
    assert "weight_bytes_max" not in c17["bf16"]


def _drop(build, name):
    return lambda d: d["bars"]["c17"][build].pop(name)


def _set(build, name, value):
    return lambda d: d["bars"]["c17"][build].__setitem__(name, value)


C17_NAMES = [
    ("bf16", "top1_over_stock_min_points"),
    ("bf16", "top1_under_clef_w4a16_max_points"),
    ("quant", "top1_over_stock_min_points"),
    ("quant", "top1_under_clef_w4a16_max_points"),
    ("quant", "quant_vs_bf16_agreement_min"),
    ("quant", "weight_bytes_max"),
    ("quant", "latency_reported"),
    ("bf16", "latency_reported"),
]


@pytest.mark.parametrize(("build", "name"), C17_NAMES)
def test_a_prereg_missing_any_c17_bar_is_refused(build, name):
    doc = good_doc()
    _drop(build, name)(doc)
    with pytest.raises(CliError, match=name):
        dp.validate(doc)


@pytest.mark.parametrize(("build", "name"), C17_NAMES)
@pytest.mark.parametrize("bad", ["10", None, True, [1]])
def test_a_c17_bar_of_the_wrong_type_is_refused(build, name, bad):
    doc = good_doc()
    _set(build, name, bad)(doc)
    with pytest.raises(CliError, match=name):
        dp.validate(doc)


@pytest.mark.parametrize(
    ("build", "name", "bad"),
    [
        ("bf16", "top1_over_stock_min_points", 9.9),
        ("quant", "top1_over_stock_min_points", 0),
        ("bf16", "top1_under_clef_w4a16_max_points", 5.1),
        ("quant", "top1_under_clef_w4a16_max_points", 20),
        ("quant", "quant_vs_bf16_agreement_min", 0.96),
        ("quant", "weight_bytes_max", 10**9 + 1),
        ("quant", "weight_bytes_max", 1.0e9),
        ("quant", "latency_reported", ["median"]),
        ("bf16", "latency_reported", []),
    ],
)
def test_a_c17_bar_weaker_than_the_bar_is_refused(build, name, bad):
    doc = good_doc()
    _set(build, name, bad)(doc)
    with pytest.raises(CliError, match=name):
        dp.validate(doc)


def test_a_stricter_c17_bar_is_accepted():
    doc = good_doc()
    doc["bars"]["c17"]["quant"]["weight_bytes_max"] = 9 * 10**8
    doc["bars"]["c17"]["quant"]["quant_vs_bf16_agreement_min"] = 0.99
    doc["bars"]["c17"]["bf16"]["top1_over_stock_min_points"] = 12
    assert dp.validate(doc).bars["c17"]["quant"]["weight_bytes_max"] == 9 * 10**8


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["bars"]["c17"].pop("quant"),
        lambda d: d["bars"]["c17"].pop("bf16"),
        lambda d: d["bars"]["c17"].__setitem__("int4", {}),
        lambda d: d["bars"]["c17"]["bf16"].__setitem__("latency_ratio_max", 0.2),
        lambda d: d["bars"]["c17"]["bf16"].__setitem__("weight_bytes_max", 10**9),
        lambda d: d["bars"].__setitem__("c17", []),
    ],
)
def test_c17_structure_is_checked(mutate):
    doc = good_doc()
    mutate(doc)
    with pytest.raises(CliError):
        dp.validate(doc)


def test_a_number_is_not_a_bytes_count_for_weight_bytes():
    doc = good_doc()
    doc["bars"]["c17"]["quant"]["weight_bytes_max"] = 999999999.5
    with pytest.raises(CliError, match="integer"):
        dp.validate(doc)


def test_an_18_shape_prereg_is_refused_with_a_clear_message(tmp_path):
    """A schema_version 1 (#18-shape) file is refused, not reinterpreted."""
    doc = good_doc()
    doc["schema_version"] = 1
    del doc["bars"]["c17"]
    with pytest.raises(CliError, match="#18-shape") as exc:
        dp.validate(doc)
    assert "bars.c17" in exc.value.message
    path = write(tmp_path, doc)
    with pytest.raises(CliError, match="#18-shape"):
        dp.load(path)
    with pytest.raises(CliError, match="#18-shape"):
        dp.lock(path, tmp_path / "run")
