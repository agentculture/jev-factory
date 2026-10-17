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
import types

import pytest

from jev_factory.backbones.pointer import distill_loss as dl
from jev_factory.backbones.pointer import train_distill as td
from jev_factory.cli._errors import CliError
from jev_factory.distill import executor as dex
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
    names = {n for n, _, _ in extra}  # these fixture values replace example_doc's own
    doc["upstream_knobs"] = [k for k in doc["upstream_knobs"] if k["name"] not in names]
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
        prereg=kw.pop("prereg", dp),
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
    assert out.n_teacher == 2
    assert out.n_gold == 2
    assert out.loss == pytest.approx(0.7 * kl + 0.3 * ce)


def test_gold_ce_is_a_weighted_mean_like_upstream():
    a = _row([1.0, 0.0], gold=0, weight=3.0)
    b = _row([0.0, 1.0], gold=0, weight=1.0)
    want = (3 * _ce([1.0, 0.0], 0) + _ce([0.0, 1.0], 0)) / 4
    assert dl.distill_loss([a, b], alpha=0.0).loss == pytest.approx(want)


def test_a_row_with_neither_target_is_refused():
    rows = [_row([0.0, 1.0])]
    with pytest.raises(ValueError, match="neither"):
        dl.distill_loss(rows, alpha=0.5)


def test_alpha_outside_unit_interval_is_refused():
    rows = [_row([0.0, 1.0], gold=0)]
    with pytest.raises(ValueError, match="alpha"):
        dl.distill_loss(rows, alpha=1.5)


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
    assert rank == 6
    assert alpha == 6
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
            prereg=dp,
        )


def test_plan_refuses_a_prereg_changed_after_the_lock(tmp_path):
    path, run_dir = locked(tmp_path)
    doc = prereg_doc()
    doc["knob_grid"]["alpha"] = [0.5, 0.7, 0.9, 0.95]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CliError, match="changed"):
        td.resolve_from_locked(path, run_dir, dict(RUN_KNOBS), prereg=dp)


def test_plan_returns_the_command_without_running_it(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("plan_train must not run anything")

    monkeypatch.setattr(gpu, "run_gpu_stage", boom)
    monkeypatch.setattr(gpu, "run_capped", boom)
    p = plan(tmp_path)
    assert p["dry_run"] is True
    assert p["stage"] == "train"
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
    assert hp["alpha"] == 0.7
    assert hp["lr"] == 1e-4
    assert hp["epochs"] == 1
    assert hp["temperature"] == {"choice": 2.0, "score": 1.0, "noul": 1.0}
    assert hp["head_lr"] == 1e-3
    assert hp["ordinal_smoothing"] == 0.1
    assert hp["micro_batch_size"] == 8
    assert hp["grad_accum"] == 4
    assert hp["init_mode"] == "continue"
    assert set(hp["sources"]) >= set(td.REQUIRED_UPSTREAM)
    assert hp["sources"]["alpha"] == "knob_grid"
    assert hp["sources"]["head_lr"].startswith("upstream:train_config.json")


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
        assert hp["lora_r"] == 16
        assert hp["lora_targets"] == ["q_proj", "v_proj"]
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
    out = td.run_train(plan(tmp_path), jobs=dex)
    assert out["dry_run"] is True
    assert not called


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
    out = td.run_train(p, apply=True, memory_max="100G", jobs=dex)
    assert seen["stage"] == "train"
    assert seen["cmd"] == p["argv"]
    assert seen["kw"]["memory_max"] == "100G"
    assert seen["kw"]["env"]["PYTHONPATH"] == p["env"]["PYTHONPATH"]
    assert seen["kw"]["allow_foreign"] is False
    assert out["returncode"] == 0
    assert out["dry_run"] is False


def test_run_train_apply_refuses_a_foreign_gpu_process(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "query_compute_apps", lambda env=None: [gpu.GpuProcess(1, "x")])
    monkeypatch.setattr(gpu, "own_pids", lambda extra=(): {2})
    monkeypatch.setattr(gpu, "run_capped", lambda *a, **k: pytest.fail("ran past the guard"))
    train_plan = plan(tmp_path)
    with pytest.raises(gpu.GpuBusyError):
        td.run_train(train_plan, apply=True, memory_max="100G", jobs=dex)


def test_run_train_apply_reports_a_failed_trainer(tmp_path, monkeypatch):
    class Failed:
        returncode = 3
        stdout = ""
        stderr = "watchdog"

    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: Failed())
    train_plan = plan(tmp_path)
    with pytest.raises(CliError, match="exited 3"):
        td.run_train(train_plan, apply=True, memory_max="100G", jobs=dex)


def _main_argv(tmp_path, **paths):
    """The trainer's argv in the pipeline's layout: prereg, rows and output under the run."""
    run = tmp_path / "run"
    (run / "distill").mkdir(parents=True, exist_ok=True)
    prereg = run / "distill" / "prereg.json"
    prereg.write_text(json.dumps(prereg_doc()), encoding="utf-8")
    values = {
        "prereg": prereg,
        "run_dir": run,
        "train_file": run / "t.jsonl",
        "init_from": tmp_path / "v21",
        "output_dir": run / "out",
    }
    values.update(paths)
    hp = td.resolve_hyperparameters(dp.validate(prereg_doc()), dict(RUN_KNOBS), prereg=dp)
    argv = [
        "--knobs",
        json.dumps(RUN_KNOBS),
        "--hyperparameters",
        json.dumps(hp, sort_keys=True),
        "--prereg-sha256",
        dp.sha256_file(prereg),
    ]
    for key, value in values.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return argv


def test_trainer_main_refuses_without_a_lock_before_loading_torch(tmp_path, capsys):
    assert td.main(_main_argv(tmp_path)) == 1
    assert "locked" in capsys.readouterr().err


@pytest.mark.parametrize(
    "key, rel",
    [
        ("output_dir", "../out"),
        ("output_dir", "../../elsewhere"),
        ("train_file", "../t.jsonl"),
        ("prereg", "../prereg.json"),
    ],
)
def test_trainer_main_refuses_a_path_outside_the_run_directory(tmp_path, capsys, key, rel):
    outside = tmp_path / "run" / rel
    if key == "prereg":
        (tmp_path / "prereg.json").write_text(json.dumps(prereg_doc()), encoding="utf-8")
    assert td.main(_main_argv(tmp_path, **{key: outside})) == 1
    assert "escapes" in capsys.readouterr().err


def test_trainer_main_refuses_a_symlinked_prereg(tmp_path, capsys):
    real = tmp_path / "run" / "distill" / "real.json"
    argv = _main_argv(tmp_path)
    real.write_text(json.dumps(prereg_doc()), encoding="utf-8")
    link = tmp_path / "run" / "distill" / "link.json"
    link.symlink_to(real)
    argv[argv.index("--prereg") + 1] = str(link)
    assert td.main(argv) == 1
    assert "symlink" in capsys.readouterr().err


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
    assert out[0].target == [1.0, 0.0]
    assert out[0].teacher == [0.8, 0.2]
    assert out[0].temperature == 2.0
    assert out[1].teacher is None
    assert out[2].target == pytest.approx([0.0, 0.1, 0.9])
    assert out[3].target == pytest.approx([0.25, 0.75])
    assert out[3].temperature == 1.0
    assert out[4].target is None
    assert out[4].teacher == [0.8, 0.2]


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
    path = write_items(tmp_path, [_item(**bad)])
    with pytest.raises(CliError, match=match):
        td.load_items(path, HP)


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
    assert row.target == [0.0, 1.0]
    assert row.teacher == [0.2, 0.8]
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


# --------------------------------------------------------------------- heal mode (t12)

MIX = {"bf16_student": 0.5, "clef": 0.5}


def test_mix_targets_weigh_the_bf16_student_and_clef_per_the_registered_mix():
    """AC1: each heal target mixes the bf16 student's and Clef's per-option distributions."""
    target, source = td.mix_targets([0.8, 0.2], [0.2, 0.8], {"bf16_student": 0.75, "clef": 0.25})
    assert target == pytest.approx([0.65, 0.35])
    assert source == "bf16_student+clef"
    target, source = td.mix_targets([0.8, 0.2], None, MIX)
    assert (target, source) == ([0.8, 0.2], "bf16_student_only:no_clef")
    assert td.mix_targets([0.8, 0.2], None, {"bf16_student": 0.0, "clef": 1.0}) == (
        None,
        "dropped:no_clef",
    )


@pytest.mark.parametrize(
    "mix, match",
    [
        ({"bf16_student": 1.0}, "exactly"),
        ({"bf16_student": 0.6, "clef": 0.6}, "sum to 1"),
        ({"bf16_student": "a", "clef": 0.5}, "numbers"),
    ],
)
def test_check_mix_refuses_a_malformed_mix(mix, match):
    with pytest.raises(CliError, match=match):
        td.check_mix(mix)


def test_unpermute_inverts_the_slot_permutation():
    canonical = [0.1, 0.2, 0.7]
    order = [2, 0, 1]
    assert td.unpermute(dl.permute(canonical, order), order) == pytest.approx(canonical)
    assert td.unpermute([0.3, 0.7], None) == [0.3, 0.7]
    with pytest.raises(ValueError):
        td.unpermute([0.3, 0.7], [0, 0])


def _heal_row(**over):
    row = _item()
    row["clef"] = row.pop("teacher")
    row.pop("gold")
    row.update(over)
    return row


def test_heal_items_train_on_the_mixed_target_alone(tmp_path):
    """AC1: heal rows carry the mixed soft target as the teacher and no gold."""
    path = write_items(tmp_path, [_heal_row(), _heal_row(record_id="r2", clef=None)])
    rows = td.load_heal_rows(path, HP, [])
    items, log, counts = td.heal_items(rows, [[0.6, 0.4], [0.9, 0.1]], MIX)
    assert [i.target for i in items] == [None, None]
    assert items[0].teacher == pytest.approx([0.7, 0.3])
    assert items[1].teacher == pytest.approx([0.9, 0.1])
    assert counts == {"bf16_student+clef": 1, "bf16_student_only:no_clef": 1}
    assert log[0]["bf16_student"] == [0.6, 0.4]
    assert log[0]["clef"] == [0.8, 0.2]


def test_a_protected_id_in_the_heal_rows_is_refused_with_a_user_error(tmp_path):
    """AC2: a test or held-out id in the heal input is refused (exit 1)."""
    path = write_items(tmp_path, [_heal_row(), _heal_row(record_id="held-1")])
    protected = [td.id_digest("held-1")]
    with pytest.raises(CliError, match="protected record id") as err:
        td.load_heal_rows(path, HP, protected)
    assert err.value.code == 1


def test_plan_train_passes_the_heal_spec(tmp_path):
    train_plan = plan(tmp_path, heal_spec=tmp_path / "run" / "heal_spec.json")
    argv = train_plan["argv"]
    assert argv[argv.index("--heal") + 1] == str(tmp_path / "run" / "heal_spec.json")
    assert train_plan["heal_spec"] == str(tmp_path / "run" / "heal_spec.json")
    assert "--heal" not in plan(tmp_path / "plain")["argv"]


def _heal_argv(tmp_path, rows, protected_ids):
    argv = _main_argv(tmp_path)
    run = tmp_path / "run"
    dp.lock(run / "distill" / "prereg.json", run)
    (run / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "v21").mkdir(exist_ok=True)
    spec = {
        "format": "torch_int4_wo",
        "mix": MIX,
        "fake_quant": None,
        "protected_id_sha256": [td.id_digest(i) for i in protected_ids],
        "bf16_student": str(tmp_path / "v21"),
    }
    (run / "heal_spec.json").write_text(json.dumps(spec))
    return argv + ["--heal", str(run / "heal_spec.json")]


def test_trainer_main_exits_non_zero_on_a_protected_heal_id_before_loading_torch(
    tmp_path, capsys, monkeypatch
):
    """AC2: the trainer itself re-checks the heal input and exits non-zero."""
    monkeypatch.setattr(td, "_heal", lambda *a: pytest.fail("the heal must not start"))
    argv = _heal_argv(tmp_path, [_heal_row(), _heal_row(record_id="test-9")], ["test-9"])
    assert td.main(argv) == 1
    assert "protected record id" in capsys.readouterr().err


def test_trainer_main_starts_the_heal_from_the_specs_bf16_student(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(td, "_heal", lambda args, hp, rows, spec: seen.update(rows=rows) or 0)
    assert td.main(_heal_argv(tmp_path, [_heal_row()], ["test-9"])) == 0
    assert [r.item.record_id for r in seen["rows"]] == ["r1"]


def test_trainer_main_refuses_a_heal_spec_naming_another_student(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(td, "_heal", lambda *a: pytest.fail("the heal must not start"))
    argv = _heal_argv(tmp_path, [_heal_row()], [])
    spec_path = tmp_path / "run" / "heal_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["bf16_student"] = str(tmp_path / "other")
    spec_path.write_text(json.dumps(spec))
    assert td.main(argv) == 1
    assert "bf16 student" in capsys.readouterr().err


def test_the_heal_falls_back_to_the_bf16_continuation_without_a_fake_quant_scheme():
    """Instruction: the d18 bf16 continuation is the fallback, with the reason recorded."""
    spec = {"format": "onnx_int8", "fake_quant": None}
    method, restore = td.enable_heal_method(object(), spec, probe=lambda: None)
    assert method["method"] == td.HEAL_BF16
    assert "onnx_int8" in method["fallback_reason"]
    assert restore() is None


# ------------------------------------------------- QAT forward (torch only; skipped in CI)


def test_fake_quant_matches_round_to_nearest_with_a_straight_through_gradient():
    torch = pytest.importorskip("torch")
    w = torch.tensor([[0.0, 0.1, 0.2, 0.7], [-1.0, -0.5, 0.25, 1.0]], requires_grad=True)
    q = td.fake_quant(w, 8, None, True)
    assert torch.allclose(q, w, atol=1.0 / 127)
    q4 = td.fake_quant(w, 4, 2, False)
    assert len(set(q4.detach().reshape(-1, 2)[0].tolist())) <= 2
    q4.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))  # straight-through
    with pytest.raises(td.QatUnavailable):
        td.fake_quant(w, 4, 3, False)


def _toy_lora_model(torch):
    class Lora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
            self.scaling = {"default": 2.0}
            self.active_adapters = ["default"]

        def forward(self, x):
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x)) * 2.0

    class Torso(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].q_proj = Lora()
            self.embed_tokens = torch.nn.Embedding(10, 4)

    class Model:
        torso = Torso()

    return Model()


def test_qat_forward_fake_quantizes_the_merged_lora_weight_and_restores():
    torch = pytest.importorskip("torch")
    model = _toy_lora_model(torch)
    layer = model.torso.layers[0].q_proj
    x = torch.randn(3, 4)
    plain = layer(x)
    fq = {
        "block_bits": 4,
        "block_group_size": 2,
        "block_symmetric": False,
        "embedding_bits": 4,
        "embedding_group_size": 2,
    }
    method, restore = td.enable_heal_method(
        model, {"format": "torch_int4_wo", "fake_quant": fq}, probe=lambda: None
    )
    assert method["method"] == td.HEAL_QAT
    assert method["touched"] == {"lora_linears": 1, "plain_linears": 0, "embeddings": 1}
    merged = (
        layer.base_layer.weight
        + 2.0 * layer.lora_B["default"].weight @ layer.lora_A["default"].weight
    )
    expected = x @ td.fake_quant(merged, 4, 2, False).T
    assert torch.allclose(layer(x), expected, atol=1e-5)
    layer(x).sum().backward()
    assert layer.lora_A["default"].weight.grad is not None  # the LoRA trains through it
    restore()
    assert torch.allclose(layer(x), plain, atol=1e-6)


def test_a_failing_qat_probe_falls_back_and_restores():
    torch = pytest.importorskip("torch")
    model = _toy_lora_model(torch)
    fq = {"block_bits": 4, "block_group_size": 2, "block_symmetric": False}

    def probe():
        raise RuntimeError("no kernel")

    method, _restore = td.enable_heal_method(
        model, {"format": "torch_int4_wo", "fake_quant": fq}, probe
    )
    assert method["method"] == td.HEAL_BF16
    assert "no kernel" in method["fallback_reason"]
    assert "forward" not in model.torso.layers[0].q_proj.__dict__


def test_student_distributions_are_the_students_own_softmax_in_canonical_order(tmp_path):
    """AC1: the bf16 student labels each heal row (temperature 1), back in canonical order."""
    torch = pytest.importorskip("torch")
    items = td.load_items(write_items(tmp_path, [_item(), _item(record_id="r2")]), HP)

    class Fmt:
        @staticmethod
        def Example(**kw):
            return kw

    class Coll:
        orders = []

        def __call__(self, examples):
            self.orders = [[1, 0] for _ in examples]  # the collator swapped the two options
            n = len(examples)
            ones = torch.ones(n, 2, dtype=torch.long)
            return {"input_ids": ones, "attention_mask": ones, "n_slots": ones, "opt_idx": ones}

    class Raw(torch.nn.Module):
        def forward(self, **kw):
            n = kw["input_ids"].shape[0]
            return {"logits": torch.tensor([[0.0, math.log(3.0)]] * n)}  # slot 1: p = 0.75

    sess = td._Session(torch, None, Fmt, None, None, Raw(), {}, "cpu")
    out = td.student_distributions(sess, Coll(), items, batch_size=1)
    assert out == [pytest.approx([0.75, 0.25])] * 2  # slot 1 is canonical option 0


def test_head_to_safetensors_replaces_the_pickled_head(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    state = {"proj.weight": torch.randn(3, 4), "proj.bias": torch.randn(3)}
    torch.save(state, str(tmp_path / "slot_head.pt"))
    td.head_to_safetensors(torch, tmp_path)
    assert not (tmp_path / "slot_head.pt").exists()
    back = st.load_file(str(tmp_path / "head.safetensors"))
    assert set(back) == set(state)
    assert all(torch.equal(back[k], state[k]) for k in state)


# --------------------------------------------------------------------- d1: injected seams


class _FakeJobs:
    """A job seam in place of jev_factory.distill.executor: records what it was given."""

    class _Limits:
        def __init__(self, **kw):
            self.kw = kw

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def job_limits(self, **kw):
        return self._Limits(**kw)

    def job_spec_from_plan(self, plan, *, expected_outputs, environment, limits):
        self.calls.append(("spec", limits.kw, tuple(expected_outputs)))
        return types.SimpleNamespace(to_json=lambda: {"fake": True})

    def run_job(self, spec, executor, *, protected):
        self.calls.append(("run", executor, tuple(protected)))
        return {"returncode": self.returncode, "stdout_tail": [], "stderr_tail": ["x"]}


def test_run_train_takes_its_job_seam_from_the_caller(tmp_path):
    jobs = _FakeJobs()
    p = plan(tmp_path)
    assert td.run_train(p, jobs=jobs)["job_spec"] == {"fake": True}
    out = td.run_train(p, apply=True, memory_max="100G", jobs=jobs, executor="ex", protected=["a"])
    assert out["job"]["returncode"] == 0
    assert ("run", "ex", ("a",)) in jobs.calls
    assert jobs.calls[0][1]["memory_max"] is None
    assert jobs.calls[1][1]["memory_max"] == "100G"


def test_run_train_needs_a_job_seam(tmp_path):
    p = plan(tmp_path)
    with pytest.raises(TypeError):
        td.run_train(p)  # pylint: disable=missing-kwoa


def test_plan_train_takes_its_pre_registration_seam_from_the_caller(tmp_path):
    calls = []

    class Prereg:
        @staticmethod
        def require_locked(path, run_dir, deviation_id=None):
            calls.append(("locked", path, run_dir, deviation_id))
            return dp.require_locked(path, run_dir, deviation_id)

        @staticmethod
        def require_knobs(reg, knobs):
            calls.append(("knobs", dict(knobs)))
            return dp.require_knobs(reg, knobs)

    p = plan(tmp_path, prereg=Prereg)
    assert [c[0] for c in calls] == ["locked", "knobs"]
    assert p["hyperparameters"]["alpha"] == RUN_KNOBS["alpha"]


def test_the_plan_hands_the_trainer_its_resolved_hyperparameters_and_prereg_hash(tmp_path):
    p = plan(tmp_path)
    argv = p["argv"]
    assert json.loads(argv[argv.index("--hyperparameters") + 1]) == p["hyperparameters"]
    assert argv[argv.index("--prereg-sha256") + 1] == p["prereg_sha256"]


def _locked_main_argv(tmp_path):
    argv = _main_argv(tmp_path)
    run = tmp_path / "run"
    dp.lock(run / "distill" / "prereg.json", run)
    (run / "t.jsonl").write_text(json.dumps(_item()) + "\n")
    return argv


def test_trainer_main_refuses_without_the_plans_hyperparameters(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(td, "_train", lambda *a: pytest.fail("the train must not start"))
    argv = _locked_main_argv(tmp_path)
    i = argv.index("--hyperparameters")
    del argv[i : i + 2]
    assert td.main(argv) == 1
    assert "hyperparameters" in capsys.readouterr().err


def test_trainer_main_refuses_a_prereg_other_than_the_planned_one(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(td, "_train", lambda *a: pytest.fail("the train must not start"))
    argv = _locked_main_argv(tmp_path)
    argv[argv.index("--prereg-sha256") + 1] = "0" * 64
    assert td.main(argv) == 1
    assert "planned" in capsys.readouterr().err


def test_trainer_main_refuses_hyperparameters_that_disagree_with_the_knobs(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(td, "_train", lambda *a: pytest.fail("the train must not start"))
    argv = _locked_main_argv(tmp_path)
    i = argv.index("--hyperparameters") + 1
    hp = json.loads(argv[i])
    hp["alpha"] = 0.9
    argv[i] = json.dumps(hp)
    assert td.main(argv) == 1
    assert "alpha" in capsys.readouterr().err


def test_trainer_main_trains_with_the_planned_hyperparameters(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(td, "_train", lambda args, hp, items: seen.update(hp=hp) or 0)
    argv = _locked_main_argv(tmp_path)
    assert td.main(argv) == 0
    want = json.loads(argv[argv.index("--hyperparameters") + 1])
    assert seen["hp"] == want
