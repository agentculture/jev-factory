"""Distill pre-registration: schema, lock, deviation refusal, knob/source checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import prereg as dp
from jev_factory.distill import profile as dprofile


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
        prereg=dp,
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


# --- schema v3 (t7): roles, deployed format, a variant-neutral teacher bar ------------------

#: A byte-for-byte copy of the locked jetson-ops-20 (#20 proving run) pre-registration and its
#: lock: the shape a locked v2 file has in the field. Only the pre-registration, no record text.
V2_FIXTURE = Path(__file__).parent / "fixtures" / "distill_prereg_v2"
V2_FIXTURE_FILE = V2_FIXTURE / "distill" / "prereg.json"
V2_FIXTURE_SHA256 = "f8a7954f055be18ca1c4ccd1af3e46a56a12ca9735967274f26a0f7d2dd27f71"


def v3_doc(**kw) -> dict:
    return dp.example_doc_v3(**kw)


def test_v3_example_validates_with_roles_and_deployed_format():
    reg = dp.validate(v3_doc())
    assert reg.schema_version == 3
    assert set(dp.roles(reg)) == set(dp.ROLE_NAMES)
    for name in ("teacher", "stock", "student"):
        assert dp.role_present(reg, name)
    assert dp.deployed_format(reg) == "quant"
    assert dp.teacher_variant(reg) == "W4A16"


@pytest.mark.parametrize("role", ["teacher", "stock", "student"])
def test_v3_refuses_a_mandatory_role_declared_absent(role):
    doc = v3_doc()
    doc["roles"][role] = {"present": False}
    with pytest.raises(CliError, match=role):
        dp.validate(doc)


@pytest.mark.parametrize("role", ["teacher", "stock", "student", "alt_teacher", "edge"])
def test_v3_refuses_an_undeclared_role(role):
    doc = v3_doc()
    del doc["roles"][role]
    with pytest.raises(CliError, match=role):
        dp.validate(doc)


@pytest.mark.parametrize("role", ["alt_teacher", "deployed_server", "edge"])
@pytest.mark.parametrize("present", [True, False])
def test_v3_optional_roles_may_be_present_or_absent(role, present):
    doc = v3_doc()
    # the alt teacher's model comes from run config (distill_alt_teacher_model), not here
    doc["roles"][role] = {"present": present}
    assert dp.role_present(dp.validate(doc), role) is present


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["roles"].__setitem__("oracle", {"present": True}),
        lambda d: d["roles"]["edge"].__setitem__("present", "yes"),
        lambda d: d["roles"].__setitem__("edge", True),
        lambda d: d["roles"]["teacher"].pop("variant"),
        lambda d: d["roles"]["teacher"].__setitem__("variant", " "),
        lambda d: d["roles"]["teacher"].pop("model"),
        lambda d: d["roles"]["stock"].pop("model"),
        lambda d: d["roles"]["student"].__setitem__("present", 1),
        lambda d: d.pop("roles"),
        lambda d: d.pop("deployed_format"),
        lambda d: d.__setitem__("deployed_format", "gguf"),
        lambda d: d.__setitem__("reference_rows", {}),
    ],
)
def test_v3_roles_and_format_are_checked(mutate):
    doc = v3_doc()
    mutate(doc)
    with pytest.raises(CliError):
        dp.validate(doc)


def test_v3_teacher_bar_is_variant_neutral():
    doc = v3_doc()
    for build in ("bf16", "quant"):
        assert "top1_under_teacher_max_points" in doc["bars"]["c17"][build]
        assert not any("clef" in n or "w4a16" in n for n in doc["bars"]["c17"][build])
    reg = dp.validate(doc)
    assert dp.teacher_bar(reg, "bf16") == 5.0
    assert dp.teacher_bar(reg, "quant") == 5.0


def test_v3_records_the_preregistered_teacher_variant():
    doc = v3_doc(teacher_variant="BF16")
    assert doc["roles"]["teacher"]["variant"] == "BF16"
    assert dp.teacher_variant(dp.validate(doc)) == "BF16"


def test_v3_refuses_the_v2_teacher_bar_name():
    doc = v3_doc()
    bars = doc["bars"]["c17"]["bf16"]
    bars["top1_under_clef_w4a16_max_points"] = bars.pop("top1_under_teacher_max_points")
    with pytest.raises(CliError, match="top1_under_teacher_max_points"):
        dp.validate(doc)


@pytest.mark.parametrize(
    ("build", "name", "bad"),
    [
        ("bf16", "top1_under_teacher_max_points", 5.1),
        ("quant", "top1_under_teacher_max_points", 20),
        ("bf16", "top1_over_stock_min_points", 9.9),
        ("quant", "quant_vs_bf16_agreement_min", 0.96),
        ("quant", "weight_bytes_max", 10**9 + 1),
    ],
)
def test_v3_floors_hold_tighten_never_loosen(build, name, bad):
    doc = v3_doc()
    doc["bars"]["c17"][build][name] = bad
    with pytest.raises(CliError, match="weaker than the c17 bar") as exc:
        dp.validate(doc)
    assert name in exc.value.message
    assert "never loosened" in exc.value.remediation


def test_v3_a_stricter_teacher_bar_is_accepted():
    doc = v3_doc()
    doc["bars"]["c17"]["bf16"]["top1_under_teacher_max_points"] = 3.0
    assert dp.teacher_bar(dp.validate(doc), "bf16") == 3.0


@pytest.mark.parametrize("name", ["top1_over_stock_min_points", "top1_under_teacher_max_points"])
@pytest.mark.parametrize("marker", [None, "not_applicable", "n/a"])
def test_v3_relative_c17_bars_cannot_be_not_applicable_while_teacher_and_stock_present(
    name, marker
):
    doc = v3_doc()
    doc["bars"]["c17"]["bf16"][name] = marker
    with pytest.raises(CliError, match="not applicable"):
        dp.validate(doc)


def test_v3_quant_format_requires_the_quant_only_bars_exactly_as_today():
    for drop in (
        lambda d: d["bars"]["c17"].pop("quant"),
        lambda d: d["bars"]["c17"]["quant"].pop("quant_vs_bf16_agreement_min"),
        lambda d: d["bars"]["c17"]["quant"].pop("weight_bytes_max"),
        lambda d: d["bars"].pop("quant"),
        lambda d: d["bars"]["quant"].pop("heal_rounds_max"),
    ):
        doc = v3_doc()
        drop(doc)
        with pytest.raises(CliError):
            dp.validate(doc)


def test_v3_bf16_format_does_not_require_the_quant_only_bars():
    doc = v3_doc(deployed_format="bf16")
    assert doc["deployed_format"] == "bf16"
    assert "quant" not in doc["bars"]["c17"]
    assert "quant" not in doc["bars"]
    reg = dp.validate(doc)
    assert dp.deployed_format(reg) == "bf16"
    assert set(dp.c17_bars(reg)) == {"bf16"}


def test_v3_bf16_format_still_checks_quant_bars_when_given():
    doc = v3_doc(deployed_format="bf16")
    doc["bars"]["quant"] = {"max_accuracy_drop_points": -1, "ece_max": 0.1, "heal_rounds_max": 1}
    with pytest.raises(CliError, match="max_accuracy_drop_points"):
        dp.validate(doc)


def test_v1_is_still_refused_by_name():
    doc = v3_doc()
    doc["schema_version"] = 1
    with pytest.raises(CliError, match="#18-shape"):
        dp.validate(doc)


def test_v3_lock_and_require_locked_roundtrip(tmp_path):
    path = write(tmp_path, v3_doc())
    run_dir = tmp_path / "run"
    rec = dp.lock(path, run_dir)
    assert rec["schema_version"] == 3
    assert dp.deployed_format(dp.require_locked(path, run_dir)) == "quant"


# --- the locked v2 file: read through an explicit mapping, never rewritten ------------------


def test_the_v2_fixture_is_the_locked_jetson_ops_20_shape():
    assert dp.sha256_file(V2_FIXTURE_FILE) == V2_FIXTURE_SHA256
    lock = json.loads((V2_FIXTURE / dp.LOCK_REL).read_text(encoding="utf-8"))
    assert lock == {"file": "prereg.json", "schema_version": 2, "sha256": V2_FIXTURE_SHA256}


def test_a_locked_v2_file_loads_unchanged_and_is_never_rewritten():
    before = V2_FIXTURE_FILE.read_bytes()
    lock_before = (V2_FIXTURE / dp.LOCK_REL).read_bytes()
    reg = dp.require_locked(V2_FIXTURE_FILE, V2_FIXTURE)
    assert V2_FIXTURE_FILE.read_bytes() == before
    assert (V2_FIXTURE / dp.LOCK_REL).read_bytes() == lock_before
    assert dp.sha256_file(V2_FIXTURE_FILE) == V2_FIXTURE_SHA256
    assert reg.schema_version == 2
    # the v2-shaped view is unchanged: today's readers keep working
    assert reg.bars["c17"]["quant"]["top1_under_clef_w4a16_max_points"] == 5.0
    assert reg.bars == json.loads(before)["bars"]
    assert set(reg.reference_rows) == {"clef", "stock"}


def test_a_v2_file_maps_its_w4a16_bar_to_the_teacher_bar_with_variant_w4a16():
    reg = dp.load(V2_FIXTURE_FILE)
    assert dp.teacher_variant(reg) == "W4A16"
    assert dp.teacher_bar(reg, "bf16") == 5.0
    assert dp.teacher_bar(reg, "quant") == 5.0
    c17 = dp.c17_bars(reg)
    assert "top1_under_teacher_max_points" in c17["quant"]
    assert "top1_under_clef_w4a16_max_points" not in c17["quant"]
    assert c17["quant"]["quant_vs_bf16_agreement_min"] == 0.97
    assert dp.deployed_format(reg) == "quant"
    roles = dp.roles(reg)
    assert all(roles[name]["present"] for name in dp.ROLE_NAMES)
    assert roles["teacher"]["model"] == "Clef 27B"
    assert roles["stock"]["model"].startswith("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610@")


def test_the_v2_mapping_is_explicit_and_yields_a_valid_v3_doc():
    raw = json.loads(V2_FIXTURE_FILE.read_text(encoding="utf-8"))
    snapshot = copy.deepcopy(raw)
    mapped = dp.v2_to_v3(raw)
    assert raw == snapshot  # the mapping never mutates its input
    assert mapped["schema_version"] == 3
    assert mapped["roles"]["teacher"]["variant"] == "W4A16"
    assert dp.V2_BAR_MAP == {"top1_under_clef_w4a16_max_points": "top1_under_teacher_max_points"}
    assert dp.validate(mapped).schema_version == 3
    with pytest.raises(CliError, match="schema_version 2"):
        dp.v2_to_v3(dp.example_doc_v3())


def test_a_v2_lock_still_records_schema_version_2(tmp_path):
    path = write(tmp_path, good_doc())
    assert dp.lock(path, tmp_path / "run")["schema_version"] == 2


def test_a_tighter_v2_w4a16_bar_maps_through():
    doc = good_doc()
    doc["bars"]["c17"]["bf16"]["top1_under_clef_w4a16_max_points"] = 2.0
    assert dp.teacher_bar(dp.validate(doc), "bf16") == 2.0


# --- example_doc seeds from the active profile --------------------------------------------


def test_example_doc_with_default_2610_is_todays_draft():
    assert dp.example_doc() == dp.example_doc(profile=dprofile.DEFAULT_2610)
    assert json.dumps(dp.example_doc(), sort_keys=True) == json.dumps(
        _todays_example_doc(), sort_keys=True
    )


def test_example_doc_v3_keeps_every_v2_field_value_with_default_2610():
    v2, v3 = dp.example_doc(), dp.example_doc_v3()
    for key, value in v2.items():
        if key in ("schema_version", "bars", "reference_rows"):
            continue
        assert v3[key] == value, key
    assert v3["bars"]["test"] == v2["bars"]["test"]
    assert v3["bars"]["quant"] == v2["bars"]["quant"]
    for build in ("bf16", "quant"):
        expect = dict(v2["bars"]["c17"][build])
        expect["top1_under_teacher_max_points"] = expect.pop("top1_under_clef_w4a16_max_points")
        assert v3["bars"]["c17"][build] == expect
    assert v3["roles"]["teacher"]["model"] == v2["reference_rows"]["clef"]["model"]
    assert v3["roles"]["stock"]["model"] == v2["reference_rows"]["stock"]["model"]


def test_example_doc_seeds_from_another_profile():
    import dataclasses

    other = dataclasses.replace(
        dprofile.DEFAULT_2610,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_targets=("q_proj", "v_proj"),
        temperature_by_kind={"choice": 1.5, "noul": 1.0, "score": 2.0},
        upstream_model="Org/other-decider",
        upstream_revision="c0ffee11",
        weight_budget_bytes=5 * 10**8,
    )
    for doc in (dp.example_doc(profile=other), dp.example_doc_v3(profile=other)):
        knobs = {k["name"]: k["value"] for k in doc["upstream_knobs"]}
        assert knobs["lora_r"] == 16 and knobs["lora_alpha"] == 32
        assert knobs["lora_dropout"] == 0.1
        assert knobs["lora_targets"] == ["q_proj", "v_proj"]
        assert knobs["temperature_by_kind"]["choice"] == 1.5
        assert doc["bars"]["c17"]["quant"]["weight_bytes_max"] == 5 * 10**8
        dp.validate(doc)
    assert dp.example_doc(profile=other)["reference_rows"]["stock"]["model"] == (
        "Org/other-decider@c0ffee11"
    )
    assert dp.example_doc_v3(profile=other)["roles"]["stock"]["model"] == (
        "Org/other-decider@c0ffee11"
    )


def _todays_example_doc() -> dict:
    """example_doc() as it was before t7, rebuilt from the fixture's literal constants."""
    upstream = [
        ("lr", 1e-4, "train_config.json", "lr"),
        ("head_lr", 1e-3, "train_config.json", "head_lr"),
        ("epochs", 1, "train_config.json", "epochs"),
        ("kl_frozen_weight", 0.3, "train_config.json", "kl_frozen_weight"),
        ("teacher_weight", 1.0, "train_config.json", "teacher_weight"),
        ("ordinal_smoothing", 0.1, "train_config.json", "ordinal_smoothing"),
        ("micro_batch_size", 8, "train_config.json", "micro_batch_size"),
        ("grad_accum", 4, "train_config.json", "grad_accum"),
        ("weight_decay", 0.01, "train_config.json", "weight_decay"),
        ("warmup_ratio", 0.03, "train_config.json", "warmup_ratio"),
        ("max_grad_norm", 1.0, "train_config.json", "max_grad_norm"),
        ("max_length", 4096, "train_config.json", "max_length"),
        ("seed", 0, "train_config.json", "seed"),
        ("shuffle_options", True, "train_config.json", "shuffle_options"),
        ("reverse_score_prob", 0.5, "train_config.json", "reverse_score_prob"),
        ("lora_r", 48, "lora/adapter_config.json", "r"),
        ("lora_alpha", 96, "lora/adapter_config.json", "lora_alpha"),
        ("lora_dropout", 0.05, "lora/adapter_config.json", "lora_dropout"),
        ("lora_targets", TARGETS_2610, "lora/adapter_config.json", "target_modules"),
        (
            "temperature_by_kind",
            {"noul": 1.0704394404346953, "choice": 0.8629556148274228, "score": 1.834409163182587},
            "strands_decider_config.json",
            "temperature_by_kind",
        ),
    ]
    raw = json.loads(V2_FIXTURE_FILE.read_text(encoding="utf-8"))
    c17_build = {
        "top1_over_stock_min_points": 10.0,
        "top1_under_clef_w4a16_max_points": 5.0,
        "latency_reported": ["median", "p95"],
    }
    return {
        "schema_version": 2,
        "bars": {
            "c17": {
                "bf16": dict(c17_build),
                "quant": {
                    **c17_build,
                    "quant_vs_bf16_agreement_min": 0.97,
                    "weight_bytes_max": 10**9,
                },
            },
            "test": raw["bars"]["test"],
            "quant": raw["bars"]["quant"],
        },
        "selection_rule": raw["selection_rule"],
        "knob_grid": {
            "alpha": [0.5, 0.7, 0.9],
            "temperature": {"choice": [1.0, 2.0], "score": [1.0, 2.0], "noul": [1.0]},
            "teacher_filter": ["all", "agrees_with_gold"],
            "teacher_weak_floor": [0.6, 0.7],
            "data_mix": ["teacher_only", "teacher_plus_corrections"],
            "epochs": [1, 2],
            "lr": [5e-5, 1e-4],
            "init_mode": ["continue", "merge_fresh"],
        },
        "heal": {
            "epochs": 1,
            "lr": 5e-5,
            "init_mode": "continue",
            "mix": {"bf16_student": 0.5, "clef": 0.5},
            "mix_source": raw["heal"]["mix_source"],
            "source": raw["heal"]["source"],
        },
        "gate": {
            "objective": "max_selective_accuracy",
            "min_coverage": {"choice": 0.8, "noul": 0.8, "score": 0.8},
            "grid": raw["gate"]["grid"],
            "status": dp.GATE_PLACEHOLDER,
        },
        "reference_rows": raw["reference_rows"],
        "upstream_knobs": [
            {"name": n, "value": v, "source_file": f, "source_key": k} for n, v, f, k in upstream
        ],
    }
