"""Distill quantize, heal once and recalibrate (t19).

Every deployed-format candidate is planned, built by an injected builder and measured on the
fit and selection folds by an injected measurer (faked here: no GPU, no torch, no server);
the chosen format is a decision citing the measurements; heal runs at most once, only past
the 3.0-point margin; recalibration fits on the deployed build's own fit-fold predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.decide.rules import HEAL_MARGIN_POINTS, HEAL_ROUNDS
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as Q
from jev_factory.distill import train as T
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.distill.runspec import Bootstrap, TrainerEnv

NEUTRAL = {"head_type": "pointer", "temperature": 1.0, "temperature_by_kind": {}}
WINNER_KNOBS = {
    "alpha": 0.9,
    "temperature": {"choice": 1.0, "score": 1.0, "noul": 1.0},
    "epochs": 2,
    "lr": 1e-4,
    "init_mode": "merge_fresh",
}

# --------------------------------------------------------------------- fixtures


def _rec(i: int) -> Record:
    return Record(
        id=f"r{i}",
        state=f"state {i}",
        questions={
            "c": Question("choice", [("fix", "fix it"), ("wait", "wait")]),
            "s": Question("score", [("0", "low"), ("1", "mid"), ("2", "high")]),
            "n": Question("noul", [("false", ""), ("true", "")]),
        },
        gold={
            "c": Gold(label="fix" if i % 3 else "wait"),
            "s": Gold(label=str(i % 3)),
            "n": Gold(label="true", probability_true=1.0 if i % 2 else 0.0),
        },
        split="train",
        source="src",
        licence="mit",
        gold_provenance="human",
    )


def folds():
    return {"fit": [_rec(i) for i in range(0, 20)], "selection": [_rec(i) for i in range(20, 40)]}


def _answer(rec: Record, p_right: float, right: bool) -> dict:
    qs = {}
    for qid, q in rec.questions.items():
        names, gold = q.names, rec.gold[qid]
        if q.type == "noul":
            truth = gold.probability_true >= 0.5
            p_true = p_right if truth == right else 1.0 - p_right
            qs[qid] = {"kind": "noul", "distribution": None, "p_true": p_true, "error": None}
            continue
        g = names.index(gold.label)
        target = g if right else (g + 1) % len(names)
        rest = (1.0 - p_right) / (len(names) - 1)
        dist = {n: (p_right if k == target else rest) for k, n in enumerate(names)}
        qs[qid] = {"kind": q.type, "distribution": dist, "p_true": None, "error": None}
    return qs


#: format -> (selection accuracy, peak MiB, median s); "healed" keys apply to heal builds.
PROFILE = {
    "bf16": (0.9, 5000.0, 0.05),
    "w4a16_fp32_head": (0.9, 2000.0, 0.04),
    "onnx_int8": (0.85, 2500.0, 0.03),
    "onnx_int4": (0.9, 1500.0, 0.03),
    "gguf_side_head": (0.6, 1000.0, 0.02),
    "torch_int8_wo": (0.9, 3000.0, 0.04),
    "torch_int4_wo": (0.9, 2200.0, 0.05),
}


class FakeMeasurer:
    """Answers each record right on ``accuracy`` of them, at a fixed confidence."""

    def __init__(self, profile=None, healed=None, p_right=0.9, fit_p_right=None):
        self.profile = dict(profile or PROFILE)
        self.healed = dict(healed or {})
        self.p_right = p_right
        self.fit_p_right = fit_p_right or {}
        self.calls = []

    def __call__(self, fmt, artifact, fold_records):
        is_heal = Q.HEAL_DIR in Path(artifact).parts
        self.calls.append((fmt, is_heal, sorted(fold_records)))
        acc, peak, median = (self.healed if is_heal and fmt in self.healed else self.profile)[fmt]
        preds = {}
        for fold, recs in fold_records.items():
            p = self.fit_p_right.get(fmt, self.p_right) if fold == "fit" else self.p_right
            rows = []
            for k, rec in enumerate(recs):
                right = (k % 20) < round(acc * 20)
                rows.append(
                    {
                        "fold": fold,
                        "record_id": rec.id,
                        "error": None,
                        "questions": _answer(rec, p, right),
                    }
                )
            preds[fold] = rows
        return {
            "predictions": preds,
            "latency": {"n": 40, "median_s": median, "p95_s": median, "max_s": median},
            "memory": {"probe": "proc-rss", "peak_mib": peak, "measured": True},
        }


class FakeBuilders(dict):
    def __init__(self, formats=("bf16", *Q.FORMATS), unavailable=None, weight_bytes=None):
        super().__init__()
        self.calls = []
        self.weight_bytes = dict(weight_bytes or {})
        unavailable = unavailable or {}
        for fmt in formats:
            self[fmt] = self._make(unavailable.get(fmt))

    def _make(self, reason):
        def build(fmt, checkpoint, out_dir):
            self.calls.append((fmt, Path(checkpoint)))
            if reason:
                raise Q.FormatUnavailable(reason)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "model.safetensors").open("wb") as fh:  # sparse: no disk used
                fh.truncate(self.weight_bytes.get(fmt, 64))
            return out_dir

        return build


class FakeTrainer:
    def __init__(self, log_path):
        self.plans = []
        self.log_path = log_path
        self.log_at_call = None

    def __call__(self, plan):
        self.log_at_call = (
            self.log_path.read_text(encoding="utf-8") if self.log_path.is_file() else None
        )
        self.plans.append(plan)
        out = Path(plan["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "strands_decider_config.json").write_text(json.dumps(NEUTRAL))
        return {"dry_run": False, "returncode": 0}


def setup(tmp_path, doc=None, lock=True):
    doc = doc if doc is not None else dp.example_doc()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    run_dir = tmp_path / "run"
    if lock:
        dp.lock(path, run_dir)
    ckpt = run_dir / T.TRAIN_REL / "c007" / T.CHECKPOINT_DIR
    ckpt.mkdir(parents=True)
    (ckpt / "strands_decider_config.json").write_text(json.dumps(NEUTRAL))
    summary = {
        "winner": "c007",
        "candidates": [{"cid": "c007", "knobs": WINNER_KNOBS, "checkpoint": str(ckpt)}],
    }
    sel = run_dir / T.SELECT_REL
    sel.mkdir(parents=True)
    (sel / T.SUMMARY_FILE).write_text(json.dumps(summary))
    train_set = run_dir / "distill" / "assemble" / "train_set.jsonl"
    train_set.parent.mkdir(parents=True, exist_ok=True)
    train_set.write_text("{}\n")
    return path, run_dir, ckpt


def run(tmp_path, *, doc=None, apply=True, builders=None, measurer=None, trainer=None, **kw):
    path, run_dir, ckpt = setup(tmp_path, doc)
    trainer = trainer or FakeTrainer(run_dir / Q.QUANT_REL / Q.HEAL_LOG_FILE)
    args = dict(
        prereg_path=path,
        run_dir=run_dir,
        folds=folds(),
        trainer_env=TrainerEnv(python="python3", decider_repo=tmp_path / "strands-decider"),
        apply=apply,
        hooks=Q.QuantHooks(
            builders=FakeBuilders() if builders is None else builders,
            measurer=measurer or FakeMeasurer(),
            trainer=trainer,
        ),
        bootstrap=Bootstrap(resamples=50),
    )
    args.update(kw)
    return Q.run_quant(**args), run_dir, trainer


def outputs(run_dir):
    out = run_dir / Q.QUANT_REL
    return {
        name: (out / name).exists()
        for name in (Q.MEASUREMENTS_FILE, Q.DECISION_FILE, Q.CALIBRATION_FILE, Q.MANIFEST_FILE)
    }


def _read(run_dir, name):
    return json.loads((run_dir / Q.QUANT_REL / name).read_text())


# --------------------------------------------------------------------- refusals


def test_refuses_without_a_locked_prereg(tmp_path):
    path, run_dir, _ = setup(tmp_path, lock=False)
    all_folds = folds()
    env = TrainerEnv(python="python3", decider_repo=tmp_path)
    with pytest.raises(CliError, match="no distill pre-registration is locked"):
        Q.run_quant(prereg_path=path, run_dir=run_dir, folds=all_folds, trainer_env=env)


def test_refuses_a_prereg_without_a_heal_block(tmp_path):
    doc = dp.example_doc()
    del doc["heal"]
    with pytest.raises(CliError, match="no heal block"):
        run(tmp_path, doc=doc, apply=False)


def test_refuses_bars_that_disagree_with_the_heal_rule(tmp_path):
    doc = dp.example_doc()
    doc["bars"]["quant"]["max_accuracy_drop_points"] = 2.0
    with pytest.raises(CliError, match="heal margin"):
        run(tmp_path, doc=doc, apply=False)
    doc = dp.example_doc()
    doc["bars"]["quant"]["heal_rounds_max"] = 2
    with pytest.raises(CliError, match="round only"):
        run(tmp_path / "b", doc=doc, apply=False)


def test_refuses_a_sealed_fold(tmp_path):
    sealed = {**folds(), "test": [_rec(99)]}
    with pytest.raises(CliError, match="refused"):
        run(tmp_path, folds=sealed)


def test_the_margin_and_rounds_are_the_decide_rule_constants():
    assert Q.HEAL_MARGIN_POINTS is HEAL_MARGIN_POINTS
    assert HEAL_MARGIN_POINTS == 3.0
    assert Q.HEAL_ROUNDS is HEAL_ROUNDS
    assert HEAL_ROUNDS == 1


# --------------------------------------------------------------------- the spike


def test_dry_run_plans_every_format_and_the_heal_and_writes_nothing(tmp_path):
    out, run_dir, trainer = run(tmp_path, apply=False, builders={})
    assert out["dry_run"] is True
    assert [f["format"] for f in out["formats"]] == list(Q.FORMATS)
    assert set(Q.FORMATS) == {
        "w4a16_fp32_head",
        "onnx_int8",
        "onnx_int4",
        "gguf_side_head",
        "torch_int8_wo",
        "torch_int4_wo",
    }
    for f in out["formats"]:
        assert f["builder"] is False
        assert f["unavailable_reason"] == Q.UNSUPPORTED[f["format"]]
    for fmt in ("w4a16_fp32_head", "onnx_int8", "onnx_int4", "gguf_side_head"):
        assert "8b6faa5" in Q.UNSUPPORTED[fmt]
    hp = out["heal_plan"]["hyperparameters"]
    assert (hp["epochs"], hp["lr"], hp["init_mode"]) == (1, 5e-5, "continue")
    assert hp["alpha"] == 0.9  # the winner's other knobs are kept
    assert out["heal_plan"]["init_from"].endswith("c007/checkpoint")
    assert out["heal_plan"]["train_file"].endswith("train_set.jsonl")
    assert not (run_dir / Q.QUANT_REL).exists()
    assert trainer.plans == []


def test_apply_measures_every_format_and_records_the_choice_citing_measurements(tmp_path):
    builders, measurer = FakeBuilders(), FakeMeasurer()
    out, run_dir, trainer = run(tmp_path, builders=builders, measurer=measurer)
    # bf16 plus each candidate, built from the winner, measured on fit + selection only
    assert [c[0] for c in builders.calls] == ["bf16", *Q.FORMATS]
    assert all(c[1].name == "checkpoint" for c in builders.calls)
    assert all(c[2] == ["fit", "selection"] for c in measurer.calls)
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]
    for fmt in Q.FORMATS:
        m = meas[fmt]
        assert m["available"] is True
        assert m["latency"]["median_s"] == PROFILE[fmt][2]
        assert m["memory"]["peak_mib"] == PROFILE[fmt][1]
        assert m["ece"] is not None
        assert m["delta_points"] == pytest.approx(100 * (PROFILE[fmt][0] - 0.9))
    # within margin: w4a16 (0) and onnx_int4 (0) and onnx_int8 (5 > 3 is out); lowest memory
    assert out["chosen_format"] == "onnx_int4"
    decision = _read(run_dir, Q.DECISION_FILE)
    assert decision["format"] == "onnx_int4"
    digest = dp.sha256_file(run_dir / Q.QUANT_REL / Q.MEASUREMENTS_FILE)
    names = {c["name"] for c in decision["cited"]}
    for fmt in ("bf16", *Q.FORMATS):
        for metric in (
            "selection_accuracy",
            "selection_ece_calibrated",
            "accuracy_delta_points_vs_bf16",
            "latency_median_s",
            "peak_memory_mib",
        ):
            assert f"{fmt}.{metric}" in names
    assert all(c["sha256"] == digest for c in decision["cited"])
    assert [t["step"] for t in decision["trail"]] == [
        "available",
        "safety",
        "calibration",
        "accuracy",
        "footprint",
    ]
    assert out["heal_count"] == 0
    assert trainer.plans == []


def test_an_unbuildable_format_is_recorded_unavailable_never_dropped(tmp_path):
    builders = FakeBuilders(
        formats=("bf16", "w4a16_fp32_head", "onnx_int8"),
        unavailable={"onnx_int8": "onnxruntime has no Qwen3.5 linear-attention op"},
    )
    out, run_dir, _ = run(tmp_path, builders=builders)
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]
    assert set(meas) == {"bf16", *Q.FORMATS}
    assert meas["onnx_int8"] == {
        "format": "onnx_int8",
        "available": False,
        "reason": "onnxruntime has no Qwen3.5 linear-attention op",
    }
    no_builder = ("onnx_int4", "gguf_side_head", "torch_int8_wo", "torch_int4_wo")
    for fmt in no_builder:  # no builder registered at all
        assert meas[fmt]["available"] is False
        assert Q.UNSUPPORTED[fmt] == meas[fmt]["reason"]
    unavailable = _read(run_dir, Q.DECISION_FILE)["unavailable"]
    assert set(unavailable) == {"onnx_int8", *no_builder}
    assert out["chosen_format"] == "w4a16_fp32_head"


def test_bf16_is_the_outcome_when_no_quantized_format_qualifies(tmp_path):
    builders = FakeBuilders(formats=("bf16",))
    out, run_dir, trainer = run(tmp_path, builders=builders)
    assert out["chosen_format"] == "bf16"
    assert out["heal_count"] == 0
    decision = _read(run_dir, Q.DECISION_FILE)
    assert "stays bf16" in decision["reason"]
    assert "available" in decision["reason"]
    assert out["quant_vs_bf16_delta_points"] == 0.0
    assert trainer.plans == []


def test_a_quant_over_the_ece_bar_does_not_qualify(tmp_path):
    doc = dp.example_doc()
    doc["bars"]["quant"]["ece_max"] = 0.0
    out, run_dir, _ = run(tmp_path, doc=doc)
    assert out["chosen_format"] == "bf16"
    assert "calibration" in _read(run_dir, Q.DECISION_FILE)["reason"]


# --------------------------------------------------------------------- heal


def only(fmt, acc, healed_acc=None):
    profile = {"bf16": PROFILE["bf16"], fmt: (acc, 1000.0, 0.02)}
    healed = {"bf16": PROFILE["bf16"]}
    if healed_acc is not None:
        healed[fmt] = (healed_acc, 1000.0, 0.02)
    return FakeBuilders(formats=("bf16", fmt)), FakeMeasurer(profile, healed)


def test_heal_triggers_only_strictly_past_the_margin():
    measured = {
        "bf16": {"format": "bf16", "available": True, "safe": True, "accuracy": 0.9},
        "onnx_int8": {
            "format": "onnx_int8",
            "available": True,
            "safe": True,
            "ece": 0.01,
            "drop_points": HEAL_MARGIN_POINTS,
            "memory": {},
            "latency": {},
        },
    }
    choice = Q.choose_format(measured, ece_max=0.1, formats=("onnx_int8",))
    assert (choice["format"], choice["heal"]) == ("onnx_int8", False)
    measured["onnx_int8"]["drop_points"] = HEAL_MARGIN_POINTS + 0.01
    choice = Q.choose_format(measured, ece_max=0.1, formats=("onnx_int8",))
    assert (choice["format"], choice["heal"]) == ("onnx_int8", True)


def test_a_quant_within_the_margin_ships_without_a_heal(tmp_path):
    builders, measurer = only("onnx_int8", 0.9)
    out, run_dir, trainer = run(tmp_path, builders=builders, measurer=measurer)
    assert out["chosen_format"] == "onnx_int8"
    assert out["heal_count"] == 0
    assert out["heal"] is None
    assert trainer.plans == []
    assert not (run_dir / Q.QUANT_REL / Q.HEAL_LOG_FILE).exists()


def test_heal_runs_once_from_the_winner_and_lands_in_the_manifest(tmp_path):
    builders, measurer = only("onnx_int8", 0.75, healed_acc=0.9)
    out, run_dir, trainer = run(tmp_path, builders=builders, measurer=measurer)
    assert len(trainer.plans) == 1
    plan = trainer.plans[0]
    hp = plan["hyperparameters"]
    assert (hp["epochs"], hp["lr"], hp["init_mode"]) == (1, 5e-5, "continue")
    assert plan["init_from"].endswith("c007/checkpoint")
    assert plan["train_file"].endswith("distill/assemble/train_set.jsonl")
    # the trigger was logged before the trainer ran
    assert trainer.log_at_call
    assert '"heal_trigger"' in trainer.log_at_call
    # healed checkpoint re-quantized and re-measured (bf16 and the quant)
    heal_builds = [c for c in builders.calls if Q.HEAL_DIR in c[1].parts]
    assert [c[0] for c in heal_builds] == ["bf16", "onnx_int8"]
    manifest = _read(run_dir, Q.MANIFEST_FILE)
    assert manifest["chosen_format"] == "onnx_int8"
    assert manifest["heal_count"] == 1
    assert manifest["quant_vs_bf16_initial_delta_points"] == pytest.approx(-15.0)
    assert manifest["quant_vs_bf16_delta_points"] == pytest.approx(0.0)
    assert manifest["heal"]["drop_points"] == pytest.approx(15.0)
    assert manifest["deployed_checkpoint"].endswith("quant/heal/checkpoint")
    assert out == {**out, **manifest}


def test_a_second_trigger_stops_and_escalates_and_writes_nothing_further(tmp_path):
    builders, measurer = only("onnx_int8", 0.75, healed_acc=0.8)
    with pytest.raises(CliError, match="stop_escalate") as err:
        run(tmp_path, builders=builders, measurer=measurer)
    assert "second heal trigger" in err.value.message
    run_dir = tmp_path / "run"
    assert outputs(run_dir) == dict.fromkeys(outputs(run_dir), False)
    log = (run_dir / Q.QUANT_REL / Q.HEAL_LOG_FILE).read_text().splitlines()
    assert [json.loads(line)["event"] for line in log] == ["heal_trigger"]


def test_a_spent_heal_round_refuses_a_new_trigger_without_training(tmp_path):
    path, run_dir, _ = setup(tmp_path)
    log = run_dir / Q.QUANT_REL / Q.HEAL_LOG_FILE
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"event": "heal_trigger", "format": "onnx_int8"}) + "\n")
    builders, measurer = only("onnx_int8", 0.75, healed_acc=0.9)
    trainer = FakeTrainer(log)
    all_folds = folds()
    env = TrainerEnv(python="python3", decider_repo=tmp_path)
    hooks = Q.QuantHooks(builders=builders, measurer=measurer, trainer=trainer)
    bootstrap = Bootstrap(resamples=50)
    with pytest.raises(CliError, match="stop_escalate"):
        Q.run_quant(
            prereg_path=path,
            run_dir=run_dir,
            folds=all_folds,
            trainer_env=env,
            apply=True,
            hooks=hooks,
            bootstrap=bootstrap,
        )
    assert trainer.plans == []
    assert not (run_dir / Q.QUANT_REL / Q.MANIFEST_FILE).exists()


# --------------------------------------------------------------------- recalibration


def test_recalibration_fits_on_the_deployed_quants_own_fit_fold(tmp_path):
    # the quant is more overconfident on the fit fold than bf16, so its temperature differs
    measurer = FakeMeasurer(fit_p_right={"onnx_int4": 0.99})
    builders = FakeBuilders(formats=("bf16", "onnx_int4"))
    doc = dp.example_doc()
    doc["bars"]["quant"]["ece_max"] = 1.0  # this test is about the fit, not the ECE bar
    out, run_dir, _ = run(tmp_path, doc=doc, builders=builders, measurer=measurer)
    assert out["chosen_format"] == "onnx_int4"
    written = _read(run_dir, Q.CALIBRATION_FILE)
    assert written["format"] == written["fitted_on_predictions_of"] == "onnx_int4"
    assert written["fitted_on"] == "fit"

    recs = folds()
    preds = measurer("onnx_int4", tmp_path / "x", recs)["predictions"]
    items = {f: T.question_items(recs[f], preds[f])[0] for f in T.FOLDS}
    expected = T.fit_calibration(
        T.calibration_rows(items["fit"]), T.calibration_rows(items["selection"])
    )
    assert written["temperature_by_kind"] == expected["temperature_by_kind"]
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]
    assert meas["bf16"]["calibration"]["temperature_by_kind"] != written["temperature_by_kind"]
    manifest = _read(run_dir, Q.MANIFEST_FILE)
    assert manifest["outputs"][str(run_dir / Q.QUANT_REL / Q.CALIBRATION_FILE)] == (
        dp.sha256_file(run_dir / Q.QUANT_REL / Q.CALIBRATION_FILE)
    )


# --------------------------------------------------------------------- served measurer


def test_served_measurer_records_a_non_bundle_artifact_as_unavailable(tmp_path):
    measure = Q.served_measurer(env=None, jobdir=tmp_path / "jobs", port=8123)
    all_folds = folds()
    with pytest.raises(Q.FormatUnavailable, match="hobson-hf-export/1"):
        measure("onnx_int8", tmp_path / "model.onnx", all_folds)


def test_default_builders_register_bf16_and_the_torchao_formats():
    assert set(Q.default_builders("a" * 40)) == {"bf16"}  # no student env: not buildable
    builders = Q.default_builders("a" * 40, python="py", decider_repo="/sd")
    assert set(builders) == {"bf16", "torch_int8_wo", "torch_int4_wo"}
    assert set(Q.UNSUPPORTED) == set(Q.FORMATS)
    for fmt in ("torch_int8_wo", "torch_int4_wo"):
        assert "fp32 pointer head" in Q.FORMATS[fmt]
        assert "python" in Q.UNSUPPORTED[fmt]
        assert "decider_repo" in Q.UNSUPPORTED[fmt]


# --------------------------------------------------------------------- torchao builders (d3)


def _ckpt(root: Path) -> Path:
    from jev_factory.backbones.pointer import layout as player

    root.mkdir(parents=True, exist_ok=True)
    (root / "lora").mkdir(exist_ok=True)
    (root / player.ADAPTER_CONFIG).write_text("{}")
    (root / player.ADAPTER).write_bytes(b"a")
    (root / player.HEAD).write_bytes(b"h")
    (root / player.CONFIG).write_text(json.dumps(NEUTRAL))
    for name in player.TOKENIZER_FILES:
        (root / name).write_text("{}")
    return root


def _write_artifact(out: Path, bits: int, source=None, scheme=None) -> None:
    from jev_factory.backbones.pointer import layout as player
    from jev_factory.backbones.pointer import quantized as qz

    out.mkdir(parents=True, exist_ok=True)
    (out / qz.TORSO_FILE).write_bytes(b"t")
    (out / player.HEAD).write_bytes(b"h")
    (out / player.CONFIG).write_text(json.dumps(NEUTRAL))
    for name in player.TOKENIZER_FILES:
        (out / name).write_text("{}")
    qz.write_manifest(
        out,
        bits=bits,
        torchao_api=qz.TORCHAO_APIS[bits][0],
        torchao_version="0.18.0",
        torch_version="2.12.1",
        source=source or {"checkpoint": "c", "sha256": {"head.safetensors": "0" * 64}},
        base_model="Qwen/x",
        base_model_revision="a" * 40,
        scheme=scheme,
    )


def _argv_scheme(argv):
    """The QuantScheme a build argv asks for (what the in-env build would record)."""
    from jev_factory.backbones.pointer import quantized as qz

    flag = {argv[i]: argv[i + 1] for i in range(6, len(argv) - 1, 2)}
    return qz.QuantScheme(
        block_bits=int(flag["--bits"]),
        embedding_bits=int(flag["--embedding-bits"]),
        block_group_size=int(flag["--block-group-size"]),
        embedding_group_size=int(flag["--embedding-group-size"]),
    )


class Runner:
    def __init__(self, rc=0, stderr=""):
        self.rc, self.stderr, self.calls = rc, stderr, []

    def __call__(self, argv, env):
        import subprocess  # nosec B404 - a completed-process value only

        self.calls.append(list(argv))
        if self.rc == 0:
            from jev_factory.backbones.pointer import quantized as qz

            _write_artifact(
                Path(argv[5]),
                int(argv[7]),
                qz.source_record(Path(argv[4])),
                scheme=_argv_scheme(argv),
            )
        return subprocess.CompletedProcess(argv, self.rc, stdout="", stderr=self.stderr)


def test_the_torchao_builder_builds_a_verified_artifact_in_the_student_env(tmp_path):
    runner, guarded = Runner(), []
    builders = Q.default_builders(
        "a" * 40,
        python="py",
        decider_repo=tmp_path / "sd",
        runner=runner,
        guard=guarded.append,
        memory_max="100G",
    )
    ckpt = _ckpt(tmp_path / "ckpt")
    art = builders["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "builds" / "q4")
    assert art == tmp_path / "builds" / "q4"
    assert runner.calls[0][:4] == ["py", "-m", "jev_factory.backbones.pointer.quantized", "build"]
    assert runner.calls[0][7] == "4"
    assert guarded == ["quantize"]
    # a second call on a built artifact for the same checkpoint reuses it
    builders["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "builds" / "q4")
    assert len(runner.calls) == 1


def test_a_torchao_build_failure_is_recorded_unavailable_with_the_error(tmp_path):
    runner = Runner(rc=1, stderr="NotImplementedError: Int4Tensor needs fbgemm-gpu-genai")
    builders = Q.default_builders(
        "a" * 40,
        python="py",
        decider_repo="/sd",
        runner=runner,
        guard=lambda stage: None,
        memory_max="100G",
    )
    torch_build = builders["torch_int8_wo"]
    builders["bf16"] = FakeBuilders(formats=("bf16",))["bf16"]
    for fmt in Q.TORCH_FORMATS:  # the winner's checkpoint gets real-looking files first
        builders[fmt] = lambda f, ck, out: torch_build(f, _ckpt(Path(ck)), out)
    ckpt = _ckpt(tmp_path / "ckpt")
    with pytest.raises(Q.FormatUnavailable, match="fbgemm-gpu-genai"):
        builders["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "q4")
    out, run_dir, _ = run(
        tmp_path / "r", builders=builders, formats=("torch_int8_wo", "torch_int4_wo")
    )
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]
    for fmt in ("torch_int8_wo", "torch_int4_wo"):
        assert meas[fmt]["available"] is False
        assert "torchao build failed" in meas[fmt]["reason"]
        assert "fbgemm-gpu-genai" in meas[fmt]["reason"]
    assert out["chosen_format"] == "bf16"


def test_a_torchao_build_without_a_memory_cap_is_refused_not_recorded_unavailable(tmp_path):
    runner = Runner()
    builders = Q.default_builders(
        "a" * 40, python="py", decider_repo="/sd", runner=runner, guard=lambda stage: None
    )
    ckpt = _ckpt(tmp_path / "ckpt")
    with pytest.raises(CliError, match="memory cap"):
        builders["torch_int8_wo"]("torch_int8_wo", ckpt, tmp_path / "q8")
    assert runner.calls == []


def test_the_torchao_builder_builds_through_run_gpu_stage(tmp_path, monkeypatch):
    import subprocess  # nosec B404 - a completed-process value only

    from jev_factory.backbones.pointer import quantized as qz

    seen = []

    def stage(name, run_dir, cmd, **kw):
        seen.append((name, kw))
        _write_artifact(
            Path(cmd[5]), int(cmd[7]), qz.source_record(Path(cmd[4])), scheme=_argv_scheme(cmd)
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(qz.gpu, "run_gpu_stage", stage)
    builders = Q.default_builders(
        "a" * 40,
        python="py",
        decider_repo="/sd",
        memory_max="90G",
        memory_floor="10G",
        guard=lambda s: None,
    )
    builders["torch_int8_wo"]("torch_int8_wo", _ckpt(tmp_path / "ckpt"), tmp_path / "q8")
    assert seen[0][0] == "quantize"
    assert seen[0][1]["memory_max"] == "90G"
    assert seen[0][1]["memory_floor"] == "10G"


def test_a_watchdog_stop_is_recorded_unavailable_with_the_reason(tmp_path, monkeypatch):
    import subprocess  # nosec B404 - a completed-process value only

    from jev_factory.backbones.pointer import quantized as qz

    monkeypatch.setattr(
        qz.gpu,
        "run_gpu_stage",
        lambda name, run_dir, cmd, **kw: subprocess.CompletedProcess(
            cmd, 3, "", "MemAvailable low"
        ),
    )
    builders = Q.default_builders(
        "a" * 40, python="py", decider_repo="/sd", guard=lambda s: None, memory_max="90G"
    )
    ckpt = _ckpt(tmp_path / "ckpt")
    with pytest.raises(Q.FormatUnavailable, match="memory-floor watchdog"):
        builders["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "q4")


def test_the_bf16_server_launch_records_that_it_is_not_capped(tmp_path, monkeypatch):
    def fake_launch(jobdir, name, env, artifact, *, port, device=None):
        (Path(jobdir) / f"{name}.serve.json").write_text(json.dumps({"argv": ["x"]}))
        return 1

    monkeypatch.setattr(Q.pserve, "launch", fake_launch)
    monkeypatch.setattr(Q.pserve, "wait_ready", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(Q.bench, "benchmark_model", FakeBench())
    monkeypatch.setattr(Q.gpu, "assert_gpu_free", lambda stage: [])
    monkeypatch.setattr(Q.player, "read_layout", lambda p: {})
    (tmp_path / "jobs").mkdir()
    measure = Q.served_measurer(env="e", jobdir=tmp_path / "jobs", port=8125)
    measure("bf16", tmp_path / "bundle", folds())
    record = json.loads((tmp_path / "jobs" / "distill-quant-bf16.serve.json").read_text())
    assert record["argv"] == ["x"]
    assert record["capped"] is False
    assert "long-lived" in record["capped_note"]


class FakeBench:
    """Stands in for distill.benchmark.benchmark_model: starts the target, answers nothing."""

    def __init__(self, fail_start=None):
        self.fail_start, self.targets = fail_start, []

    def __call__(self, target, fold_records, memory_probe=None, client=None):
        self.targets.append(target)
        target.start()
        target.ready()
        return {
            "predictions": [],
            "latency": {"median_s": 0.01},
            "memory": {"probe": "proc-rss", "peak_mib": 900.0, "measured": True},
        }


def _measurer(tmp_path, monkeypatch, *, launch=None, ready=None):
    from jev_factory.backbones.pointer import quantized as qz

    calls = {"launch": [], "ready": []}

    def fake_launch(jobdir, name, env, artifact, *, port, decider_repo=None, device=None):
        calls["launch"].append((name, Path(artifact), port, decider_repo))
        if launch:
            launch()
        return 1

    def fake_ready(base_url, **kw):
        calls["ready"].append(base_url)
        if ready:
            ready()
        return {"status": "ok"}

    bench = FakeBench()
    monkeypatch.setattr(qz, "launch", fake_launch)
    monkeypatch.setattr(Q.pserve, "wait_ready", fake_ready)
    monkeypatch.setattr(Q.bench, "benchmark_model", bench)
    monkeypatch.setattr(Q.gpu, "assert_gpu_free", lambda stage: [])
    measure = Q.served_measurer(
        env="student-env", jobdir=tmp_path / "jobs", port=8124, decider_repo=tmp_path / "sd"
    )
    return measure, calls, bench


def test_the_served_measurer_serves_a_torchao_artifact_through_the_wrapper(tmp_path, monkeypatch):
    measure, calls, bench = _measurer(tmp_path, monkeypatch)
    art = tmp_path / "q8"
    _write_artifact(art, 8)
    out = measure("torch_int8_wo", art, folds())
    assert calls["launch"] == [("distill-quant-torch_int8_wo", art, 8124, tmp_path / "sd")]
    assert calls["ready"] == ["http://127.0.0.1:8124"]
    assert out["memory"]["probe"] == "proc-rss"
    assert out["memory"]["peak_mib"] == 900.0
    assert set(out["predictions"]) == {"fit", "selection"}
    assert callable(bench.targets[0].pid)


def test_a_torchao_artifact_that_cannot_be_served_is_unavailable(tmp_path, monkeypatch):
    def die():
        raise CliError(2, "strands-decider server 'q' exited before it was ready")

    measure, _calls, _ = _measurer(tmp_path, monkeypatch, ready=die)
    art = tmp_path / "q4"
    _write_artifact(art, 4)
    all_folds = folds()
    with pytest.raises(Q.FormatUnavailable, match="exited before it was ready"):
        measure("torch_int4_wo", art, all_folds)


def test_a_tampered_torchao_artifact_is_unavailable_not_served(tmp_path, monkeypatch):
    from jev_factory.backbones.pointer import quantized as qz

    measure, calls, _ = _measurer(tmp_path, monkeypatch)
    art = tmp_path / "q8"
    _write_artifact(art, 8)
    (art / qz.TORSO_FILE).write_bytes(b"tampered")
    all_folds = folds()
    with pytest.raises(Q.FormatUnavailable, match="sha256"):
        measure("torch_int8_wo", art, all_folds)
    assert calls["launch"] == []


# --------------------------------------------------------------------- size budget (#20, t10)


def test_the_size_budget_is_ten_to_the_ninth_bytes_with_one_owner():
    """c9/h21: '< 1 GB' means < 10^9 bytes of weight files; quant.py uses the builder's value."""
    from jev_factory.backbones.pointer import quantized as qz

    assert Q.SIZE_BUDGET_BYTES == 10**9
    assert Q.SIZE_BUDGET_BYTES == qz.SIZE_BUDGET_BYTES
    assert "size" in Q.FORMAT_RULE


def test_a_build_at_or_over_the_size_budget_is_unavailable_never_a_candidate(tmp_path):
    """c9/h21: quant.py records a build at or over 10^9 bytes as unavailable, never a candidate.

    torch_int4_wo is the better candidate on every measured axis but weighs exactly 10^9
    bytes (a sparse file), so it is never measured; torch_int8_wo at 10^9 - 1 ships.
    """
    builders = FakeBuilders(
        formats=("bf16", "torch_int8_wo", "torch_int4_wo"),
        weight_bytes={"torch_int4_wo": 10**9, "torch_int8_wo": 10**9 - 1},
    )
    measurer = FakeMeasurer()
    out, run_dir, _ = run(
        tmp_path,
        builders=builders,
        measurer=measurer,
        formats=("torch_int8_wo", "torch_int4_wo"),
    )
    measured = {fmt for fmt, _heal, _folds in measurer.calls}
    assert measured == {"bf16", "torch_int8_wo"}
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]
    over = meas["torch_int4_wo"]
    assert over["available"] is False
    assert over["size"]["weight_bytes"] == 10**9
    assert over["size"]["within_size_budget"] is False
    assert "1000000000 bytes" in over["reason"]
    assert "never a candidate" in over["reason"]
    assert meas["torch_int8_wo"]["size"]["weight_bytes"] == 10**9 - 1
    assert meas["torch_int8_wo"]["size"]["within_size_budget"] is True
    decision = _read(run_dir, Q.DECISION_FILE)
    assert "torch_int4_wo" in decision["unavailable"]
    assert decision["trail"][0] == {"step": "available", "survivors": ["torch_int8_wo"]}
    assert out["chosen_format"] == "torch_int8_wo"
    assert out["deployed_size"]["weight_bytes"] == 10**9 - 1
    cited = {c["name"]: c["value"] for c in decision["cited"]}
    assert cited["torch_int8_wo.weight_bytes"] == 10**9 - 1


def test_the_size_is_measured_from_the_files_on_disk(tmp_path):
    """c9/h21: the byte sum comes from the files; a missing or unverifiable size is unavailable."""
    from jev_factory.backbones.pointer import quantized as qz

    art = tmp_path / "q4"
    _write_artifact(art, 4)
    size = Q.size_record(art)
    want = {n: (art / n).stat().st_size for n in qz.WEIGHT_FILES}
    assert size["weight_files"] == want
    assert size["weight_bytes"] == sum(want.values())
    manifest = json.loads((art / qz.MANIFEST_FILE).read_text())
    manifest["weight_bytes"] = 1
    (art / qz.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(Q.FormatUnavailable, match="weight_bytes"):
        Q.size_record(art)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Q.FormatUnavailable, match="no weight files"):
        Q.size_record(empty)


def test_an_unverifiable_quant_artifact_is_unavailable_in_the_stage(tmp_path):
    """c9/h21: a candidate whose size cannot be known is unavailable, not measured."""
    builders = FakeBuilders(formats=("bf16",))
    builders["torch_int4_wo"] = lambda fmt, ck, out: out.mkdir(parents=True) or out
    measurer = FakeMeasurer()
    _out, run_dir, _ = run(
        tmp_path, builders=builders, measurer=measurer, formats=("torch_int4_wo",)
    )
    meas = _read(run_dir, Q.MEASUREMENTS_FILE)["formats"]["torch_int4_wo"]
    assert meas["available"] is False
    assert "size is unknown" in meas["reason"]
    assert [c[0] for c in measurer.calls] == ["bf16"]


def test_the_dry_run_names_the_size_budget(tmp_path):
    plan, _run_dir, _ = run(tmp_path, apply=False)
    assert plan["size_budget_bytes"] == 10**9
    assert plan["format_rule"]["size"].startswith("weight files on disk")


def test_the_torchao_builder_passes_the_scheme_and_the_format_sets_the_block_bits(tmp_path):
    """c9/h21: block bits follow the format; embedding bits and group sizes are recorded."""
    from jev_factory.backbones.pointer import quantized as qz

    runner = Runner()
    scheme = qz.QuantScheme(embedding_bits=8, block_group_size=128, embedding_group_size=64)
    builders = Q.default_builders(
        "a" * 40,
        python="py",
        decider_repo="/sd",
        runner=runner,
        guard=lambda s: None,
        memory_max="90G",
        scheme=scheme,
    )
    ckpt = _ckpt(tmp_path / "ckpt")
    art = builders["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "q4")
    recorded = qz.read_artifact(art)["manifest"]["scheme"]
    assert recorded == {
        "block_bits": 4,
        "embedding_bits": 8,
        "block_group_size": 128,
        "embedding_group_size": 64,
    }
    assert _argv_scheme(runner.calls[0]) == qz.QuantScheme(4, 8, 128, 64)
    builders["torch_int8_wo"]("torch_int8_wo", ckpt, tmp_path / "q8")
    assert _argv_scheme(runner.calls[1]).block_bits == 8


def test_a_built_artifact_with_another_scheme_is_not_reused(tmp_path):
    from jev_factory.backbones.pointer import quantized as qz

    runner = Runner()
    common = dict(python="py", decider_repo="/sd", runner=runner, guard=lambda s: None)
    ckpt = _ckpt(tmp_path / "ckpt")
    first = Q.default_builders("a" * 40, memory_max="90G", **common)["torch_int4_wo"]
    first("torch_int4_wo", ckpt, tmp_path / "q4")
    other = qz.QuantScheme(embedding_group_size=32)
    second = Q.default_builders("a" * 40, memory_max="90G", scheme=other, **common)
    with pytest.raises(Q.FormatUnavailable, match="not empty"):
        second["torch_int4_wo"]("torch_int4_wo", ckpt, tmp_path / "q4")
    assert len(runner.calls) == 1
