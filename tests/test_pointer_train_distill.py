"""Distill trainer for the pointer backbone: the loss, the plan, and the GPU-guarded run.

The pure-Python loss reference runs everywhere (CI has no torch), so alpha = 0, the
no-teacher row and the student-equals-teacher row are proven on lists. The torch loss
and the LoRA stacking are checked against that reference only where torch is present.
No GPU, no network, no model download.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from jev_factory.backbones.pointer import distill_loss as dl
from jev_factory.backbones.pointer import train_distill as td
from jev_factory.cli._errors import CliError
from jev_factory.distill import prereg as dp
from jev_factory.factory import gpu

# --------------------------------------------------------------------- helpers


def _ce(logits, gold):
    m = max(logits)
    z = math.log(sum(math.exp(v - m) for v in logits)) + m
    return z - logits[gold]


def _row(logits, kind="choice", gold=None, teacher=None, temperature=1.0, weight=1.0):
    n = len(logits)
    target = None if gold is None else dl.gold_target(kind, n, gold, ordinal_smoothing=0.1)
    return dl.LossRow(
        logits=list(logits),
        target=target,
        teacher=teacher,
        temperature=temperature,
        weight=weight,
    )


def prereg_doc(**grid_overrides):
    doc = dp.example_doc()
    doc["knob_grid"]["init_mode"] = list(td.INIT_MODES)
    doc["knob_grid"].update(grid_overrides)
    extra = [
        ("micro_batch_size", 8, "train_config.json"),
        ("grad_accum", 4, "train_config.json"),
        ("weight_decay", 0.01, "train_config.json"),
        ("warmup_ratio", 0.03, "train_config.json"),
        ("max_grad_norm", 1.0, "train_config.json"),
        ("max_length", 4096, "train_config.json"),
        ("seed", 0, "train_config.json"),
        ("shuffle_options", True, "train_config.json"),
        ("reverse_score_prob", 0.5, "train_config.json"),
        ("lora_r", 16, "train_config.json"),
        ("lora_alpha", 32, "train_config.json"),
        ("lora_dropout", 0.05, "strands_decider_config.json"),
        ("lora_targets", ["q_proj", "v_proj"], "train_config.json"),
    ]
    doc["upstream_knobs"] += [
        {"name": n, "value": v, "source_file": f, "source_key": n} for n, v, f in extra
    ]
    return doc


RUN_KNOBS = {
    "alpha": 0.7,
    "temperature": {"choice": 2.0, "score": 1.0, "noul": 1.0},
    "epochs": 1,
    "lr": 1e-4,
    "init_mode": "continue",
}


def locked(tmp_path, doc=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(doc or prereg_doc()), encoding="utf-8")
    run_dir = tmp_path / "run"
    dp.lock(path, run_dir)
    return path, run_dir


def plan(tmp_path, knobs=None, doc=None, **kw):
    path, run_dir = locked(tmp_path, doc)
    return td.plan_train(
        prereg_path=path,
        run_dir=run_dir,
        run_knobs=dict(RUN_KNOBS if knobs is None else knobs),
        train_file=tmp_path / "train.jsonl",
        init_from=tmp_path / "v21",
        output_dir=tmp_path / "out",
        trainer_python="/opt/trainer/bin/python",
        decider_repo=tmp_path / "strands-decider",
        **kw,
    )


# --------------------------------------------------------------------- pure loss reference


def test_alpha_zero_equals_gold_ce_within_1e_6():
    rng = random.Random(3)
    rows, ces = [], []
    for _ in range(6):
        n = rng.randint(2, 6)
        logits = [rng.uniform(-3, 3) for _ in range(n)]
        gold = rng.randrange(n)
        raw = [rng.random() + 1e-3 for _ in range(n)]
        teacher = [v / sum(raw) for v in raw]
        rows.append(_row(logits, gold=gold, teacher=teacher, temperature=2.0))
        ces.append(_ce(logits, gold))
    out = dl.distill_loss(rows, alpha=0.0)
    assert abs(out.loss - sum(ces) / len(ces)) < 1e-6
    assert abs(out.ce - out.loss) < 1e-6


def test_alpha_zero_equals_ordinal_smoothed_ce_on_a_score_row():
    logits = [0.2, 1.5, -0.4]
    target = dl.gold_target("score", 3, 1, ordinal_smoothing=0.1)
    assert target == pytest.approx([0.05, 0.9, 0.05])
    row = dl.LossRow(logits=logits, target=target, teacher=None, temperature=1.0)
    q = dl.log_softmax(logits)
    want = -sum(t * lq for t, lq in zip(target, q))
    assert abs(dl.distill_loss([row], alpha=0.0).loss - want) < 1e-6


def test_row_without_teacher_has_zero_kl():
    row = _row([1.0, -2.0, 0.5], gold=0, teacher=None, temperature=2.0)
    assert dl.row_kl(row) == 0.0
    out = dl.distill_loss([row], alpha=1.0)
    assert out.per_row_kl == [0.0]
    assert out.loss == 0.0
    assert out.n_teacher == 0


def test_kl_is_zero_when_student_equals_teacher_at_any_temperature():
    teacher = [0.6, 0.3, 0.1]
    logits = [math.log(p) + 4.0 for p in teacher]  # same distribution, shifted logits
    for temperature in (1.0, 2.0, 0.5):
        row = _row(logits, gold=None, teacher=teacher, temperature=temperature)
        assert abs(dl.row_kl(row)) < 1e-12


def test_kl_matches_hand_value_and_is_positive():
    teacher = [0.5, 0.5]
    row = _row([0.0, math.log(3.0)], teacher=teacher)  # student [0.25, 0.75]
    want = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert dl.row_kl(row) == pytest.approx(want, abs=1e-12)
    assert dl.row_kl(row) > 0


def test_temperature_softens_teacher_and_student_and_scales_by_t_squared():
    teacher = [0.8, 0.2]
    logits = [0.0, 0.0]
    t = 2.0
    soft = dl.soften(teacher, t)
    assert soft[0] == pytest.approx(math.sqrt(0.8) / (math.sqrt(0.8) + math.sqrt(0.2)))
    q = [0.5, 0.5]
    want = t * t * sum(p * math.log(p / qq) for p, qq in zip(soft, q))
    assert dl.row_kl(_row(logits, teacher=teacher, temperature=t)) == pytest.approx(want)


def test_teacher_zeros_carry_no_weight_like_upstream_forward_kl():
    row = _row([0.0, 1.0, 2.0], teacher=[0.5, 0.5, 0.0])
    q = dl.log_softmax([0.0, 1.0, 2.0])
    want = 0.5 * (math.log(0.5) - q[0]) + 0.5 * (math.log(0.5) - q[1])
    assert dl.row_kl(row) == pytest.approx(want)


def test_loss_mixes_mean_kl_and_mean_ce_by_alpha():
    a = _row([1.0, 0.0], gold=0, teacher=[0.3, 0.7])
    b = _row([0.0, 2.0, 1.0], gold=2, teacher=None)
    c = _row([0.5, 0.5], gold=None, teacher=[0.9, 0.1])
    out = dl.distill_loss([a, b, c], alpha=0.7)
    kl = (dl.row_kl(a) + dl.row_kl(c)) / 2
    ce = (_ce([1.0, 0.0], 0) + _ce([0.0, 2.0, 1.0], 2)) / 2
    assert out.n_teacher == 2 and out.n_gold == 2
    assert out.loss == pytest.approx(0.7 * kl + 0.3 * ce)


def test_gold_ce_is_a_weighted_mean_like_upstream():
    a = _row([1.0, 0.0], gold=0, weight=3.0)
    b = _row([0.0, 1.0], gold=0, weight=1.0)
    want = (3 * _ce([1.0, 0.0], 0) + _ce([0.0, 1.0], 0)) / 4
    assert dl.distill_loss([a, b], alpha=0.0).loss == pytest.approx(want)


def test_a_row_with_neither_target_is_refused():
    with pytest.raises(ValueError, match="neither"):
        dl.distill_loss([_row([0.0, 1.0])], alpha=0.5)


def test_alpha_outside_unit_interval_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        dl.distill_loss([_row([0.0, 1.0], gold=0)], alpha=1.5)


def test_gold_targets_mirror_the_upstream_collator():
    assert dl.gold_target("choice", 3, 2, ordinal_smoothing=0.1) == [0.0, 0.0, 1.0]
    assert dl.gold_target("score", 3, 0, ordinal_smoothing=0.1) == pytest.approx([0.9, 0.1, 0.0])
    assert dl.gold_target("score", 4, 3, ordinal_smoothing=0.2) == pytest.approx(
        [0.0, 0.0, 0.2, 0.8]
    )
    assert dl.gold_target("score", 3, 1, ordinal_smoothing=0.0) == [0.0, 1.0, 0.0]
    assert dl.noul_target(0.8) == pytest.approx([0.2, 0.8])


def test_permute_maps_canonical_targets_to_slot_order():
    # order[k] = canonical index shown at slot k (upstream collator convention)
    assert dl.permute([0.1, 0.2, 0.7], [2, 0, 1]) == [0.7, 0.1, 0.2]
    assert dl.permute([0.1, 0.9], None) == [0.1, 0.9]


# --------------------------------------------------------------------- torch loss (optional)


def test_torch_loss_matches_the_reference():
    torch = pytest.importorskip("torch")
    rng = random.Random(11)
    rows = []
    for i in range(7):
        n = rng.randint(2, 5)
        logits = [rng.uniform(-2, 2) for _ in range(n)]
        gold = None if i == 3 else rng.randrange(n)
        raw = [rng.random() + 1e-3 for _ in range(n)]
        teacher = None if i in (1, 4) else [v / sum(raw) for v in raw]
        kind = "score" if i == 5 else "choice"
        target = None if gold is None else dl.gold_target(kind, n, gold, ordinal_smoothing=0.1)
        rows.append(dl.LossRow(logits, target, teacher, rng.choice([1.0, 2.0]), 1.0 + i % 2))
    batch = dl.to_tensors(rows)
    for alpha in (0.0, 0.3, 1.0):
        got = dl.distill_loss_torch(batch["logits"], batch, alpha=alpha)
        want = dl.distill_loss(rows, alpha=alpha)
        assert float(got["loss"]) == pytest.approx(want.loss, abs=1e-6)
        assert got["per_row_kl"].tolist() == pytest.approx(want.per_row_kl, abs=1e-6)
    logits = batch["logits"].clone().requires_grad_(True)
    dl.distill_loss_torch(logits, batch, alpha=0.5)["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_torch_alpha_zero_equals_cross_entropy_and_equal_kl_is_zero():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, -0.5, 0.3], [0.2, 0.9, -1e9]])
    rows = [
        dl.LossRow([1.0, -0.5, 0.3], [0.0, 1.0, 0.0], [0.2, 0.5, 0.3], 2.0),
        dl.LossRow([0.2, 0.9], [1.0, 0.0], None, 1.0),
    ]
    batch = dl.to_tensors(rows)
    got = dl.distill_loss_torch(logits, batch, alpha=0.0)["loss"]
    want = torch.nn.functional.cross_entropy(logits[:, :3][:1], torch.tensor([1]))
    want = (want + torch.nn.functional.cross_entropy(logits[1:, :2], torch.tensor([0]))) / 2
    assert float(got) == pytest.approx(float(want), abs=1e-6)
    teacher = [0.6, 0.3, 0.1]
    same = dl.to_tensors([dl.LossRow([math.log(p) for p in teacher], None, teacher, 2.0)])
    kl = dl.distill_loss_torch(same["logits"], same, alpha=1.0)
    assert float(kl["per_row_kl"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(dl.distill_loss_torch(batch["logits"], batch, alpha=1.0)["per_row_kl"][1]) == 0


def test_stacked_lora_equals_the_sum_of_both_deltas():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    a1, b1 = torch.randn(4, 6, generator=g), torch.randn(5, 4, generator=g)
    a2, b2 = torch.randn(2, 6, generator=g), torch.randn(5, 2, generator=g)
    key = "base_model.model.layers.0.q_proj"
    first = {f"{key}.lora_A.weight": a1, f"{key}.lora_B.weight": b1}
    second = {f"{key}.lora_A.weight": a2, f"{key}.lora_B.weight": b2}
    state, rank, alpha = td.stack_lora(first, 8 / 4, second, 4 / 2)
    assert rank == 6 and alpha == 6
    a, b = state[f"{key}.lora_A.weight"], state[f"{key}.lora_B.weight"]
    want = b1 @ a1 * 2.0 + b2 @ a2 * 2.0
    assert torch.allclose(b @ a * (alpha / rank), want, atol=1e-5)


# --------------------------------------------------------------------- hyperparameters / plan


def test_plan_refuses_without_a_locked_prereg(tmp_path):
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg_doc()), encoding="utf-8")
    with pytest.raises(CliError, match="locked"):
        td.plan_train(
            prereg_path=path,
            run_dir=tmp_path / "run",
            run_knobs=dict(RUN_KNOBS),
            train_file=tmp_path / "t.jsonl",
            init_from=tmp_path / "v21",
            output_dir=tmp_path / "out",
            trainer_python="python",
            decider_repo=tmp_path / "sd",
        )


def test_plan_refuses_a_prereg_changed_after_the_lock(tmp_path):
    path, run_dir = locked(tmp_path)
    doc = prereg_doc()
    doc["knob_grid"]["alpha"] = [0.5, 0.7, 0.9, 0.95]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CliError, match="changed"):
        td.resolve_from_locked(path, run_dir, dict(RUN_KNOBS))


def test_plan_returns_the_command_without_running_it(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("plan_train must not run anything")

    monkeypatch.setattr(gpu, "run_gpu_stage", boom)
    monkeypatch.setattr(gpu, "run_capped", boom)
    p = plan(tmp_path)
    assert p["dry_run"] is True and p["stage"] == "train"
    argv = p["argv"]
    assert argv[:3] == ["/opt/trainer/bin/python", "-m", td.MODULE]
    assert argv[argv.index("--prereg") + 1] == str(tmp_path / "prereg.json")
    assert argv[argv.index("--run-dir") + 1] == str(tmp_path / "run")
    assert json.loads(argv[argv.index("--knobs") + 1]) == RUN_KNOBS
    assert argv[argv.index("--init-from") + 1] == str(tmp_path / "v21")
    assert str(tmp_path / "strands-decider") in p["env"]["PYTHONPATH"].split(":")
    assert not (tmp_path / "out").exists()


def test_every_hyperparameter_comes_from_the_locked_prereg(tmp_path):
    hp = plan(tmp_path)["hyperparameters"]
    assert hp["alpha"] == 0.7 and hp["lr"] == 1e-4 and hp["epochs"] == 1
    assert hp["temperature"] == {"choice": 2.0, "score": 1.0, "noul": 1.0}
    assert hp["head_lr"] == 1e-3 and hp["ordinal_smoothing"] == 0.1
    assert hp["micro_batch_size"] == 8 and hp["grad_accum"] == 4
    assert hp["init_mode"] == "continue"
    assert set(hp["sources"]) >= set(td.REQUIRED_UPSTREAM)
    assert hp["sources"]["alpha"] == "knob_grid" and hp["sources"]["head_lr"].startswith(
        "upstream:train_config.json"
    )


def test_unregistered_knob_values_are_refused(tmp_path):
    with pytest.raises(CliError, match="alpha"):
        plan(tmp_path, knobs={**RUN_KNOBS, "alpha": 0.33})


def test_a_missing_upstream_hyperparameter_is_refused(tmp_path):
    doc = prereg_doc()
    doc["upstream_knobs"] = [k for k in doc["upstream_knobs"] if k["name"] != "grad_accum"]
    with pytest.raises(CliError, match="grad_accum"):
        plan(tmp_path, doc=doc)


def test_an_unchosen_multi_valued_grid_knob_is_refused(tmp_path):
    knobs = {k: v for k, v in RUN_KNOBS.items() if k != "alpha"}
    with pytest.raises(CliError, match="alpha"):
        plan(tmp_path, knobs=knobs)


def test_temperature_must_cover_every_question_kind(tmp_path):
    knobs = {**RUN_KNOBS, "temperature": {"choice": 2.0}}
    with pytest.raises(CliError, match="temperature"):
        plan(tmp_path, knobs=knobs)


def test_init_mode_must_be_pre_registered(tmp_path):
    doc = prereg_doc()
    del doc["knob_grid"]["init_mode"]
    with pytest.raises(CliError, match="init_mode"):
        plan(tmp_path, doc=doc)


@pytest.mark.parametrize("mode", td.INIT_MODES)
def test_both_init_modes_plan(tmp_path, mode):
    hp = plan(tmp_path, knobs={**RUN_KNOBS, "init_mode": mode})["hyperparameters"]
    assert hp["init_mode"] == mode
    if mode == "merge_fresh":
        assert hp["lora_r"] == 16 and hp["lora_targets"] == ["q_proj", "v_proj"]
    else:
        assert "lora_r" not in hp  # continue trains v21's own adapter as configured


def test_merge_fresh_needs_the_lora_shape_pre_registered(tmp_path):
    doc = prereg_doc()
    doc["upstream_knobs"] = [k for k in doc["upstream_knobs"] if k["name"] != "lora_r"]
    with pytest.raises(CliError, match="lora_r"):
        plan(tmp_path, knobs={**RUN_KNOBS, "init_mode": "merge_fresh"}, doc=doc)
    assert plan(tmp_path / "b", doc=doc)["hyperparameters"]["init_mode"] == "continue"


def test_superseded_upstream_terms_are_reported_not_silently_dropped(tmp_path):
    p = plan(tmp_path)
    assert set(p["not_applied"]) == {"kl_frozen_weight", "teacher_weight"}
    assert all(p["not_applied"].values())


# --------------------------------------------------------------------- execution


def test_run_train_is_dry_run_by_default(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: called.append(a))
    out = td.run_train(plan(tmp_path))
    assert out["dry_run"] is True and not called


def test_run_train_apply_goes_through_run_gpu_stage(tmp_path, monkeypatch):
    seen = {}

    class Done:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake(stage, run_dir, cmd, **kw):
        seen.update(stage=stage, run_dir=run_dir, cmd=list(cmd), kw=kw)
        return Done()

    monkeypatch.setattr(gpu, "run_gpu_stage", fake)
    p = plan(tmp_path)
    out = td.run_train(p, apply=True, memory_max="100G")
    assert seen["stage"] == "train"
    assert seen["cmd"] == p["argv"]
    assert seen["kw"]["memory_max"] == "100G"
    assert seen["kw"]["env"]["PYTHONPATH"] == p["env"]["PYTHONPATH"]
    assert seen["kw"]["allow_foreign"] is False
    assert out["returncode"] == 0 and out["dry_run"] is False


def test_run_train_apply_refuses_a_foreign_gpu_process(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "query_compute_apps", lambda env=None: [gpu.GpuProcess(1, "x")])
    monkeypatch.setattr(gpu, "own_pids", lambda extra=(): {2})
    monkeypatch.setattr(gpu, "run_capped", lambda *a, **k: pytest.fail("ran past the guard"))
    with pytest.raises(gpu.GpuBusyError):
        td.run_train(plan(tmp_path), apply=True, memory_max="100G")


def test_run_train_apply_reports_a_failed_trainer(tmp_path, monkeypatch):
    class Failed:
        returncode = 3
        stdout = ""
        stderr = "watchdog"

    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: Failed())
    with pytest.raises(CliError, match="exited 3"):
        td.run_train(plan(tmp_path), apply=True, memory_max="100G")


def test_trainer_main_refuses_without_a_lock_before_loading_torch(tmp_path, capsys):
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg_doc()), encoding="utf-8")
    argv = [
        "--prereg",
        str(path),
        "--run-dir",
        str(tmp_path / "run"),
        "--knobs",
        json.dumps(RUN_KNOBS),
        "--train-file",
        str(tmp_path / "t.jsonl"),
        "--init-from",
        str(tmp_path / "v21"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert td.main(argv) == 1
    assert "locked" in capsys.readouterr().err


# --------------------------------------------------------------------- training rows


def _item(**over):
    item = {
        "record_id": "r1",
        "question_id": "decide",
        "kind": "choice",
        "state": "is the lamp on?",
        "instructions": "Which action?",
        "options": [["lamp_status", "read the lamp"], ["explain", "answer"]],
        "gold": {"label": "lamp_status"},
        "teacher": {"lamp_status": 0.8, "explain": 0.2},
    }
    item.update(over)
    return item


def write_items(tmp_path, items):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    return path


HP = {"ordinal_smoothing": 0.1, "temperature": {"choice": 2.0, "score": 1.0, "noul": 1.0}}


def test_load_items_builds_canonical_targets(tmp_path):
    items = [
        _item(),
        _item(record_id="r2", teacher=None),
        _item(
            record_id="r3",
            kind="score",
            options=[["0", "low"], ["1", "mid"], ["2", "high"]],
            gold={"label": "2"},
            teacher=None,
        ),
        _item(
            record_id="r4",
            kind="noul",
            options=[["false", ""], ["true", ""]],
            gold={"p_true": 0.75},
            teacher={"false": 0.1, "true": 0.9},
        ),
        _item(record_id="r5", gold=None),
    ]
    out = td.load_items(write_items(tmp_path, items), HP)
    assert [i.record_id for i in out] == ["r1", "r2", "r3", "r4", "r5"]
    assert out[0].target == [1.0, 0.0] and out[0].teacher == [0.8, 0.2]
    assert out[0].temperature == 2.0
    assert out[1].teacher is None
    assert out[2].target == pytest.approx([0.0, 0.1, 0.9])
    assert out[3].target == pytest.approx([0.25, 0.75]) and out[3].temperature == 1.0
    assert out[4].target is None and out[4].teacher == [0.8, 0.2]


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"teacher": {"lamp_status": 0.8, "other": 0.2}}, "teacher"),
        ({"teacher": {"lamp_status": 0.5, "explain": 0.1}}, "sum"),
        ({"gold": {"label": "nope"}}, "gold"),
        ({"gold": None, "teacher": None}, "neither"),
        ({"kind": "rank"}, "kind"),
    ],
)
def test_load_items_refuses_malformed_rows(tmp_path, bad, match):
    with pytest.raises(CliError, match=match):
        td.load_items(write_items(tmp_path, [_item(**bad)]), HP)


def test_soft_gold_probabilities_are_used_as_given(tmp_path):
    item = _item(gold={"probabilities": {"lamp_status": 0.6, "explain": 0.4}})
    (out,) = td.load_items(write_items(tmp_path, [item]), HP)
    assert out.target == pytest.approx([0.6, 0.4])


def test_upstream_provenance_is_cited():
    prov = td.UPSTREAM_PROVENANCE
    assert prov["repo"] == "strands-labs/strands-decider"
    assert prov["commit"] == "8b6faa553038a59e73fa7f16c4081f8ceb21d27e"
    assert prov["licence"] == "Apache-2.0"
    assert {"training/train.py", "training/data/collate.py"} <= set(prov["files"])


def test_slot_rows_follow_the_collator_order(tmp_path):
    items = td.load_items(write_items(tmp_path, [_item()]), HP)
    (row,) = td.slot_rows(items, [[1, 0]])
    assert row.target == [0.0, 1.0] and row.teacher == [0.2, 0.8]
    assert row.temperature == 2.0
    with pytest.raises(ValueError):
        td.slot_rows(items, [])


def test_train_steps_lowers_the_loss_on_a_toy_model(tmp_path):
    torch = pytest.importorskip("torch")
    names = {"lamp_status": 0, "explain": 1}
    items = td.load_items(write_items(tmp_path, [_item(record_id=f"r{i}") for i in range(8)]), HP)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

        def forward(self, ids):
            return {"logits": self.w[ids]}

    def make_batch(rows):
        ids = torch.tensor([[names[o[0]] for o in r.options] for r in rows])
        return {"ids": ids}, td.slot_rows(rows, [None] * len(rows))

    model = Toy()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    hp = {**HP, "micro_batch_size": 2, "grad_accum": 2, "epochs": 5, "seed": 0}
    hp.update(alpha=0.5, max_grad_norm=1.0)
    hist = td.train_steps(model, make_batch, items, hp, optimizer=opt, scheduler=sched, log=str)
    assert len(hist) == 10
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert float(model.w[0].detach()) > float(model.w[1].detach())
