"""The stage registry (jev_factory/factory/pipeline.py): order, sub-steps and guards.

Re-expresses the guards of nvsh's ``pipeline.sh`` (its tests in
tests/test_lfm_finetune_pipeline.py) as stage tests: an unknown stage lists
every stage, a dry run writes nothing, upload refuses unless the operator named
the repository, heal refuses without a base run, quantize refuses without a
trained run, a build that was never quantized is not measured, the frozen
training set is chosen by sha256, measurement defaults cover the readout. The
GPU, llama.cpp, the teachers and the hub are stubs (tests/pipeline_support.py);
the measurement and probe code itself runs for real, in process, over a
synthetic model.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm.readout import READOUT_TOP
from jev_factory.cli._errors import CliError
from jev_factory.core import calibration as calib
from jev_factory.core import sweep_gate
from jev_factory.core.gate import Thresholds
from jev_factory.decide import records
from jev_factory.factory import config as config_mod
from jev_factory.factory import pipeline
from jev_factory.factory.stages import DEFAULT_REGISTRY, list_stages
from tests import pipeline_support as ps
from tests.fixtures.fake_teacher import FakeGateway, roles
from tests.release_support import TOKEN, FakeHub

EXPECTED_ORDER = [
    "config",
    "preregister",
    "seed",
    "teachers-pilot",
    "draft-heldout",
    "draft-eval",
    "split",
    "snapshot",
    "baseline",
    "augment",
    "targeted",
    "assemble",
    "train",
    "select",
    "quantize",
    "heal",
    "recalibrate",
    "measure-final",
    "edge-check",
    "bundle",
    "upload",
    "release-gate",
]

CHAIN = ("config", "preregister", "seed", "draft-heldout", "split", "snapshot")
DATA = ("augment", "targeted", "assemble")


# --- fixtures -------------------------------------------------------------------


def _apply(workdir, ctx, name, knobs=None, **kw):
    man = pipeline.run(workdir, name, ctx, knobs, apply=True, **kw)
    assert man["status"] == "complete", man.get("error")
    return man


def _failed(workdir, ctx, name, knobs=None, **kw) -> str:
    man = pipeline.run(workdir, name, ctx, knobs, apply=True, **kw)
    assert man["status"] == "failed"
    return man["error"]


class Run:
    """A toy run with every GPU/llama.cpp/hub seam stubbed."""

    def __init__(self, tmp_path: Path, **config):
        self.tmp = tmp_path
        self.workdir = tmp_path / "run"
        self.workdir.mkdir()
        self.trainer = ps.FakeTrainRunner()
        self.llama = ps.FakeLlamaCpp()
        self.hub = FakeHub(tmp_path / "remote")
        self.runner = ps.ModuleRunner(
            self.workdir, {"jev_factory.measure.run": ps.measure_handler(self.workdir)}
        )
        self.ctx = ps.toy_context(
            self.workdir,
            self.runner,
            config={**ps.gpu_config(tmp_path), **config},
            train_runner=self.trainer,
            quant_run=self.llama,
            serve=ps.fake_serve,
            hub=self.hub,
            today=lambda: "2026-09-30",
        )
        self.ctx.services.environ = {"HF_TOKEN": TOKEN}
        (self.workdir / "prereg.json").write_text(json.dumps(ps.toy_prereg()), encoding="utf-8")
        ps.write_pool(self.workdir)
        sealed, sha = ps.write_sealed_heldout(tmp_path / "operator" / "held-out.json")
        self.knobs = {
            "draft-heldout": {"import_from": str(sealed), "import_sha256": sha},
            "split": {"val_size": 30, "test_size": 20, "fold_seed": 3},
            "augment": {"per_source": 0},
            "select": {"bootstrap_resamples": 0},
            "recalibrate": {"bootstrap_resamples": 0},
            "measure-final": {"bootstrap_resamples": 0},
            "train": {"candidates": {"r2": {"epochs": 2}}},
        }

    def apply(self, name, knobs=None, **kw):
        return _apply(self.workdir, self.ctx, name, knobs or self.knobs.get(name), **kw)

    def fail(self, name, knobs=None, **kw):
        return _failed(self.workdir, self.ctx, name, knobs or self.knobs.get(name), **kw)

    def through(self, *names):
        for name in names:
            self.apply(name)
        return self

    def json(self, rel):
        return json.loads((self.workdir / rel).read_text())


@pytest.fixture
def toy(tmp_path):
    return Run(tmp_path).through(*CHAIN, *DATA)


@pytest.fixture
def selected(toy):
    return toy.through("train", "select")


@pytest.fixture
def deployed(selected):
    return selected.through("quantize", "heal", "recalibrate")


# --- the stage list (acceptance criterion 1, o8) --------------------------------


@pytest.mark.behavioral("o8")
def test_the_pipeline_registers_every_stage_in_order():
    assert list(pipeline.STAGE_ORDER) == EXPECTED_ORDER
    assert pipeline.stage_names() == EXPECTED_ORDER
    assert [s["stage"] for s in list_stages(DEFAULT_REGISTRY)][: len(EXPECTED_ORDER)] == (
        EXPECTED_ORDER
    )
    position = {name: i for i, name in enumerate(EXPECTED_ORDER)}
    for name in EXPECTED_ORDER:
        stage = pipeline.get_stage(name)
        assert all(position[d] < position[name] for d in stage.deps), name
        assert stage.outputs and stage.summary


def test_select_runs_calibrate_then_gate_fit_then_probe_then_the_rule():
    assert pipeline.SUBSTEPS["select"] == ("calibrate", "gate-fit", "probe", "rule")
    assert "calibrate -> gate-fit -> probe -> rule" in pipeline.get_stage("select").summary


#: Issue #4's scorer-path steps (obligation o8) -> the stage, or stage and sub-step.
ISSUE4_STEPS = {
    "config": ("config", None),
    "pre-registration": ("preregister", None),
    "seed": ("seed", None),
    "teachers and pilot": ("teachers-pilot", "pilot-draft"),
    "sealed held-out": ("draft-heldout", "seal"),
    "eval pool and folds": ("split", "fit-selection-folds"),
    "eval pool drafting": ("draft-eval", "draft-eval-pool"),
    "snapshot": ("snapshot", "grounding-snapshot"),
    "baseline": ("baseline", "measure-stock"),
    "augment": ("augment", None),
    "targeted recipes": ("targeted", "targeted-recipes"),
    "assemble and freeze": ("assemble", "freeze"),
    "train with merge check": ("train", "merge-check"),
    "select": ("select", "rule"),
    "logprob calibration": ("select", "calibrate"),
    "gate fit": ("select", "gate-fit"),
    "permutation probe": ("select", "probe"),
    "quantize": ("quantize", "gguf-q4_k_m"),
    "heal": ("heal", "heal-once"),
    "recalibrate": ("recalibrate", "calibrate"),
    "final measurement": ("measure-final", "measure-test"),
    "edge check": ("edge-check", "record-operator-results"),
    "bundle": ("bundle", "model-bundle"),
    "upload": ("upload", "private-upload"),
    "release gate": ("release-gate", "evals-run"),
}


@pytest.mark.behavioral("o8")
@pytest.mark.parametrize("step", sorted(ISSUE4_STEPS))
def test_every_scorer_path_step_is_a_stage_or_documented_sub_step(step):
    stage, sub = ISSUE4_STEPS[step]
    assert pipeline.get_stage(stage).name == stage
    if sub is not None:
        assert sub in pipeline.SUBSTEPS[stage]
        assert sub in pipeline.get_stage(stage).summary


def test_every_stage_has_documented_knobs_and_sub_steps():
    assert set(pipeline.DEFAULT_KNOBS) == set(EXPECTED_ORDER) == set(pipeline.SUBSTEPS)


# --- engine guards (pipeline.sh's stage contract) -------------------------------


def test_an_unknown_stage_lists_every_stage(tmp_path):
    with pytest.raises(CliError) as exc:
        pipeline.run(tmp_path, "train-scorer", ps.toy_context(tmp_path), apply=True)
    assert "unknown stage" in exc.value.message
    assert exc.value.remediation.endswith(", ".join(EXPECTED_ORDER))
    assert not (tmp_path / "manifests").exists()


def test_an_unknown_knob_is_refused(tmp_path):
    with pytest.raises(CliError, match="unknown knob"):
        pipeline.run(tmp_path, "split", ps.toy_context(tmp_path), {"vall_size": 3}, apply=True)


def _tree(root: Path) -> dict[str, float]:
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}


def test_a_dry_run_writes_nothing(toy):
    before = _tree(toy.tmp)
    for name in EXPECTED_ORDER:
        doc = pipeline.run(toy.workdir, name, toy.ctx)
        assert doc["applied"] is False and doc["stage"] == name
    assert _tree(toy.tmp) == before
    plan = pipeline.plan(toy.workdir, "train", toy.ctx)
    assert plan["would_run"] and plan["stale"] == "no manifest"
    assert pipeline.plan(toy.workdir, "seed", toy.ctx)["stale"] is None
    # the pool was written by hand here, not by draft-eval: staleness says so
    split_plan = pipeline.plan(toy.workdir, "split", toy.ctx, toy.knobs["split"])
    assert split_plan["stale"] == "upstream stage draft-eval is stale"


def test_a_fresh_stage_is_a_no_op_and_a_changed_knob_makes_it_stale(toy):
    again = pipeline.run(toy.workdir, "split", toy.ctx, toy.knobs["split"], apply=True)
    assert again["skipped"] is True
    plan = pipeline.plan(toy.workdir, "split", toy.ctx, {**toy.knobs["split"], "fold_seed": 4})
    assert plan["stale"] == "knobs changed"


def test_a_work_root_inside_a_git_worktree_is_refused(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # nosec B603 B607
    workdir = tmp_path / "run"
    with pytest.raises(CliError, match="inside a git worktree"):
        pipeline.run(workdir, "config", ps.toy_context(workdir), apply=True)


def test_a_stage_holds_the_run_lock(toy):
    lock = toy.workdir / ".jev-run.lock"
    lock.write_text(json.dumps({"pid": os.getppid()}))
    with pytest.raises(CliError, match="locked by PID"):
        pipeline.run(toy.workdir, "split", toy.ctx, {"fold_seed": 9}, apply=True)
    lock.unlink()
    seen = []

    def spy(module, argv, **kw):
        seen.append(lock.exists())
        return toy.runner(module, argv, **kw)

    toy.ctx.services.run_module = spy
    toy.through("train")
    assert seen and all(seen)
    assert not lock.exists()


def test_a_missing_input_fails_the_stage_without_running_it(tmp_path):
    run = Run(tmp_path).through("config")
    error = run.fail("split")
    assert "missing inputs" in error and "seed/seed.json" in error


def test_the_measure_defaults_cover_the_readout():
    key = {k.name: k for k in config_mod.KEYS}["measure_max_logprobs"]
    assert key.default >= READOUT_TOP


# --- data stages ----------------------------------------------------------------


def _generator(user: str) -> str:
    import re

    found = re.search(r"operation (\w+)", user)
    if found and "Write" in user:
        op = found.group(1)
        args = {"room_status": {"room": "kitchen"}, "lamp_on": {"room": "study"}}
        args["set_scene"] = {"scene": "reading"}
        return json.dumps(
            [{"text": f"please do {op} now {i}", "args": args.get(op, {})} for i in range(2)]
        )
    return json.dumps([{"text": f"a question number {i}", "answer": "words"} for i in range(2)])


def test_the_teacher_pilot_records_a_yield_per_class(tmp_path):
    from jev_factory.data.teachers import TeacherClient

    gateway = FakeGateway(
        generator=_generator, reject=lambda role, user: "please do lamp_on" in user
    )
    client = TeacherClient(roles(), tmp_path / "cache", caller=gateway, sleep=lambda s: None)
    run = Run(tmp_path)
    run.ctx.services.teacher_client = client
    run.through("config", "seed")
    run.apply("teachers-pilot", {"per_op": 1, "per_reason": 1, "explain": 1})
    classes = run.json("pilot/yields.json")["classes"]
    assert classes["op-lamp_on"]["accepted"] == 0 and classes["op-lamp_on"]["reviewed"] > 0
    assert classes["op-lamp_status"]["rate"] == 1.0
    ids = [y.name for y in pipeline._yields(run.workdir)]
    assert "op-lamp_on" in ids


def test_the_sealed_held_out_is_drafted_read_only_and_never_redrafted(tmp_path):
    def drafter(seed):
        return (
            lambda system, user: json.dumps(
                [{"text": f"held request {len(user)} {seed}", "args": {}, "answer": "w"}]
            ),
            "rev-1",
        )

    run = Run(tmp_path)
    run.ctx.services.drafter = drafter
    run.through("config", "seed")
    run.apply("draft-heldout", {"review": False, "per_op": 1, "escalate": 1, "explain": 1})
    sealed = run.workdir / "heldout" / "held-out.json"
    assert not sealed.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    summary = run.json("heldout/summary.json")
    assert set(summary) == {"path", "entries", "by_kind", "sha256"}
    error = run.fail("draft-heldout", {"review": False}, force=True)
    assert "sealed" in error


def test_an_imported_held_out_must_match_its_sha256(tmp_path):
    run = Run(tmp_path).through("config", "seed")
    knobs = {**run.knobs["draft-heldout"], "import_sha256": "0" * 64}
    assert "import_sha256" in run.fail("draft-heldout", knobs)


def test_assemble_fails_closed_on_a_leak_into_training(tmp_path):
    run = Run(tmp_path).through(*CHAIN, "augment", "targeted")
    val = run.json("splits/val.json")["entries"][0]
    leak = {"id": "leak~v1", "source_id": val["id"], "text": val["text"], "expect": val["expect"]}
    (run.workdir / "aug" / "accepted.jsonl").write_text(json.dumps(leak) + "\n")
    error = run.fail("assemble")
    assert "leak" in error.lower() or "near-duplicate" in error or "exact" in error


def test_baseline_measures_the_greedy_stock_copy_and_writes_a_record_id(tmp_path):
    run = Run(tmp_path).through(*CHAIN)
    run.apply("baseline")
    summary = run.json("baseline/summary.json")
    assert summary["record_id"].startswith("baseline-") and summary["fold"] == "selection"
    assert set(summary["stock"]) >= {"right_proposals", "ece", "mc_escalation"}
    assert "permutation_change" in summary["stock"]
    assert (run.workdir / "stock" / "generation_config.json").is_file()


# --- train and select (acceptance criterion 2) -----------------------------------


def test_train_refuses_without_a_registered_pre_registration(toy):
    (toy.workdir / "prereg.lock.json").unlink()
    error = toy.fail("train")
    assert "missing inputs" in error and "prereg.lock.json" in error
    (toy.workdir / "prereg.lock.json").write_text("{}")
    assert "pre-registration" in toy.fail("train")


def test_train_runs_every_candidate_under_train_py_with_the_package_importable(selected):
    trainer = selected.trainer
    modules = [cmd[2] for _, cmd, _ in trainer.calls]
    assert (
        modules
        == [
            "jev_factory.backbones.causal_lm.train_scorer",
            "jev_factory.backbones.causal_lm.train",
        ]
        * 2
    )
    root = str(Path(pipeline.jev_factory.__file__).resolve().parent.parent)
    for _stage, cmd, kw in trainer.calls:
        assert kw["env"]["PYTHONPATH"].split(os.pathsep)[0] == root
        assert cmd[0] == selected.ctx.config["train_py"]
    summary = selected.json("train/summary.json")["candidates"]
    assert set(summary) == {"r1", "r2"}
    freeze = selected.json("data/freeze.json")
    assert summary["r1"]["train_sha256"] == freeze["files"]["scorer_train"]["sha256"]
    # the merged checkpoint is staged into the HF cache, as the shell pipeline did
    cache = Path(selected.ctx.config["hf_cache"]) / "hub"
    assert any(cache.glob("models--example-org--toy-lamps-jev-r1/snapshots/*"))
    epochs = [cmd[cmd.index("--epochs") + 1] for _, cmd, _ in trainer.calls if "--epochs" in cmd]
    assert epochs[1] == "2"


def test_train_picks_the_frozen_set_by_sha256_never_a_changed_file(toy):
    path = toy.workdir / "data" / "scorer-train.json"
    path.write_text(path.read_text().replace('"header"', '"header" ', 1))
    assert "frozen sha256" in toy.fail("train")


@pytest.mark.behavioral("o6")
def test_select_fits_calibration_and_the_gate_on_the_fit_fold_only(toy, monkeypatch):
    toy.through("train")
    fit_ids = set(toy.json("splits/folds.json")["fit_ids"])
    fit_seen: list[set] = []
    swept: list[set] = []
    real_rows, real_sweep = calib.fit_rows, sweep_gate.run_sweep

    def rows(predictions, ids):
        fit_seen.append(set(ids))
        return real_rows(predictions, ids)

    def sweep(originals, grid, domain):
        swept.append({p.id for p in originals})
        return real_sweep(originals, grid, domain)

    monkeypatch.setattr(calib, "fit_rows", rows)
    monkeypatch.setattr(sweep_gate, "run_sweep", sweep)
    toy.apply("select")
    assert fit_seen and all(ids == fit_ids for ids in fit_seen)
    allowed = fit_ids | {f"{i}-nocand" for i in fit_ids}
    assert swept and all(ids <= allowed for ids in swept)
    gate_doc = toy.json("select/r1/gate.json")
    assert gate_doc["fit"]["fold"] == "fit"
    summary = toy.json("select/r1/summary.json")
    assert summary["gate_fold"] == "fit"


def test_select_applies_the_pre_registered_rule_through_decide_rules(selected):
    selection = selected.json("select/selection.json")
    assert selection["rule_version"] == "jev-decide-rule/1"
    assert [s["step"] for s in selection["trail"]] == list(ps.toy_prereg()["rule_order"])
    assert selection["prereg_sha256"] == selected.json("prereg.lock.json")["sha256"]
    assert not (selected.workdir / "decisions.jsonl").exists()  # select never records


def test_select_refuses_a_changed_pre_registration(selected):
    path = selected.workdir / "prereg.json"
    doc = json.loads(path.read_text())
    doc["perms_per_entry"] = 3
    path.write_text(json.dumps(doc))
    assert "changed since it was registered" in selected.fail("select", force=True)


def test_decide_appends_records_and_never_rewrites(selected):
    first = pipeline.decide(selected.workdir, selected.ctx)
    second = pipeline.decide(selected.workdir, selected.ctx)
    assert (first["id"], second["id"]) == ("D1", "D2")
    assert records.load(selected.workdir / "decisions.jsonl") == [first, second]


# --- quantize, heal, recalibrate --------------------------------------------------


def test_quantize_refuses_unless_select_chose_a_candidate(toy):
    toy.through("train")
    (toy.workdir / "select").mkdir()
    (toy.workdir / "select" / "selection.json").write_text(
        json.dumps({"winner": None, "verdict": "stop_escalate"})
    )
    assert "not ship_candidate" in toy.fail("quantize")


def test_naming_the_winner_does_not_bypass_a_non_ship_verdict(toy):
    toy.through("train")
    (toy.workdir / "select").mkdir()
    (toy.workdir / "select" / "selection.json").write_text(
        json.dumps({"winner": "r1", "verdict": "recalibrate"})
    )
    assert "not ship_candidate" in toy.fail("quantize", {"candidate": "r1", "awq": False})
    toy.ctx.deviation_id = "d7"  # an operator choice against the verdict is a deviation
    toy.apply("quantize", {"candidate": "r1", "awq": False})


def test_quantize_refuses_a_candidate_that_was_never_trained(selected):
    import shutil

    shutil.rmtree(selected.workdir / "runs" / "r1")
    assert "no trained run" in selected.fail("quantize")


def test_quantize_builds_q4_k_m_measures_it_and_checks_the_heal_trigger(selected):
    selected.apply("quantize")
    summary = selected.json("quant/summary.json")
    assert summary["build"]["candidate"] == "r1" and summary["heal_needed"] is False
    assert summary["build"]["gguf"].endswith("model-q4_k_m.gguf")
    measured = [argv for module, argv in selected.runner.calls if module.endswith("measure.run")]
    served = [argv for argv in measured if "--serve" in argv]
    assert served and all(
        "--llama-server" in argv and "--ground-snapshot" in argv for argv in served
    )
    log = (selected.workdir / "quant" / "heal-trigger-log.jsonl").read_text()
    assert '"needed": false' in log


@pytest.mark.behavioral("o6")
def test_recalibrate_refits_on_the_deployed_quants_own_predictions(deployed):
    build = deployed.json("deployed/build.json")
    cal = deployed.json("deployed/calibration.json")
    gate_doc = deployed.json("deployed/gate.json")
    assert cal["predictions_source"].endswith(build["val"])
    assert "quant/" in build["val"] and cal["fold"] == "fit"
    assert gate_doc["fit"]["build_sha256"] == build["gguf_sha256"]
    assert gate_doc["fit"]["predictions"] == build["val"]
    select_cal = deployed.json("select/r1/calibration.json")
    assert select_cal["predictions_source"] != cal["predictions_source"]
    assert Thresholds.from_json(gate_doc)


def test_heal_without_a_trigger_deploys_the_quantized_build(deployed):
    heal = deployed.json("heal/summary.json")
    assert heal["healed"] is False and heal["deployed"]["candidate"] == "r1"
    assert not any("--heal" in cmd for _, cmd, _ in deployed.trainer.calls)


def _force_heal(run: Run) -> None:
    path = run.workdir / "quant" / "summary.json"
    doc = json.loads(path.read_text())
    doc["heal_needed"] = True
    path.write_text(json.dumps(doc))


def test_heal_refuses_without_a_base_run(selected):
    import shutil

    selected.apply("quantize")
    _force_heal(selected)
    shutil.rmtree(selected.workdir / "runs" / "r1" / "merged")
    assert "no verified merged checkpoint" in selected.fail("heal")


def test_heal_is_one_epoch_on_the_same_frozen_set_and_one_round_only(selected):
    selected.apply("quantize")
    _force_heal(selected)
    selected.apply("heal")
    heal_cmds = [cmd for _, cmd, _ in selected.trainer.calls if "--heal" in cmd]
    assert len(heal_cmds) == 1
    assert heal_cmds[0][heal_cmds[0].index("--epochs") + 1] == "1"
    assert heal_cmds[0][heal_cmds[0].index("--lr") + 1] == "5e-05"
    heal = selected.json("heal/summary.json")
    assert heal["healed"] is True and heal["deployed"]["run"] == "runs/r1-heal"
    # the healed run is itself a heal: a second heal round is refused
    (selected.workdir / "quant" / "summary.json").write_text(
        json.dumps({**selected.json("quant/summary.json"), "build": heal["deployed"]})
    )
    assert "one round only" in selected.fail("heal", force=True)


# --- the final measurement and the release stages ---------------------------------


def test_measure_final_refuses_a_build_that_was_never_quantized(deployed):
    (deployed.workdir / "quant" / "r1" / "model-q4_k_m.gguf").unlink()
    assert "missing or changed" in deployed.fail("measure-final")


def test_measure_final_measures_once_applies_the_gate_and_reports_every_bar(deployed):
    deployed.apply("measure-final")
    report = deployed.json("final/report.json")
    assert set(report["sides"]) == {"test", "held-out"}
    test = report["sides"]["test"]
    assert set(test["bars"]) == set(ps.toy_prereg()["bars"])
    assert test["bars"]["permutation_change"]["value"] == 0.0
    assert test["mc_n"] > 0 and test["bars"]["mc_escalation"]["met"] is True
    assert report["sides"]["held-out"]["mc_n"] > 0
    ledger = [
        json.loads(line)
        for line in (deployed.workdir / "measure" / "once-ledger.jsonl").read_text().splitlines()
    ]
    assert sorted((r["side"], r["slice"]) for r in ledger) == sorted(
        [(s, sl) for s in ("test", "held-out") for sl in ("full", "missing-candidate")]
    )
    assert all(r["deviation"] is None for r in ledger)
    assert "right_proposals_ci" in test and "ece_ci" in test
    finals = [a for m, a in deployed.runner.calls if m.endswith("measure.run") and "--final" in a]
    assert finals and all("--calibration" in a for a in finals)
    assert (deployed.workdir / "final" / "test.gated.predictions.jsonl").is_file()
    # a second final measurement is a recorded deviation
    error = deployed.fail("measure-final", force=True)
    assert "measure final-test exited 1" in error


def test_a_probe_failure_keeps_the_sealed_numbers_and_the_retry_resumes(deployed):
    import contextlib

    @contextlib.contextmanager
    def broken_serve(ctx, model, name, run_dir):
        raise RuntimeError("llama-server did not start")
        yield  # pragma: no cover

    def sealed_runs():
        return [a for m, a in deployed.runner.calls if m.endswith("measure.run") and "--final" in a]

    working = deployed.ctx.services.serve
    deployed.ctx.services.serve = broken_serve
    assert "llama-server did not start" in deployed.fail("measure-final")
    measured = len(sealed_runs())
    assert measured and (deployed.workdir / "final" / "report.partial.json").is_file()
    deployed.ctx.services.serve = working
    deployed.apply("measure-final")  # no deviation id: resumes, never re-measures
    assert len(sealed_runs()) == measured
    report = deployed.json("final/report.json")
    assert set(report["sides"]) == {"test", "held-out"}
    assert report["sides"]["test"]["permutation_change"] == 0.0
    assert not (deployed.workdir / "final" / "report.partial.json").exists()


def test_measure_final_with_a_deviation_measures_again(deployed):
    deployed.apply("measure-final")
    deployed.ctx.deviation_id = "d9"
    deployed.apply("measure-final", force=True)


def _edge(run: Run, **overrides) -> None:
    build = run.json("deployed/build.json")
    doc = {
        "device": "AGX Orin (operator's bench unit)",
        "build_sha256": build["gguf_sha256"],
        "decisions": {"propose": 10, "escalate": 3},
        "latency_ms": {"median": 41.0},
        "operator_approval": True,
        **overrides,
    }
    path = run.workdir / "edge" / "operator-results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_edge_check_records_only_what_the_operator_ran(deployed):
    deployed.apply("measure-final")
    assert "missing inputs" in deployed.fail("edge-check")
    _edge(deployed, build_sha256="f" * 64)
    assert "another build" in deployed.fail("edge-check")
    _edge(deployed, operator_approval=False)
    assert "approval" in deployed.fail("edge-check")
    _edge(deployed)
    deployed.apply("edge-check")
    assert deployed.json("edge/edge-check.json")["results"]["device"].startswith("AGX Orin")


def test_bundle_upload_and_release_gate(deployed, tmp_path):
    deployed.through("measure-final")
    _edge(deployed)
    deployed.apply("edge-check")
    assert "repo_suffix" in deployed.fail("bundle")
    deployed.apply("bundle", {"repo_suffix": "scorer"})
    record = deployed.json("bundle/record.json")
    folder = deployed.workdir / record["folder"]
    assert record["repo"] == "example-org/toy-lamps-jev-scorer"
    for name in ("calibration.json", "gate.json", "scorer-train.json"):
        assert (folder / name).is_file()
    assert (folder / "gate.json").read_bytes() == (
        deployed.workdir / "deployed/gate.json"
    ).read_bytes()
    assert "repo_suffix" in deployed.fail("upload")
    assert "built for" in deployed.fail("upload", {"repo_suffix": "other"})
    deployed.apply("upload", {"repo_suffix": "scorer"})
    calls = [c[0] for c in deployed.hub.calls]
    assert "create_repo" in calls and "snapshot_download" in calls
    creates = [c for c in deployed.hub.calls if c[0] == "create_repo"]
    assert all(c[2]["private"] is True for c in creates)
    assert all(
        c[2].get("private") is True for c in deployed.hub.calls if c[0] == "update_repo_visibility"
    )
    # the release gate runs python -m jev_factory.evals over the operator's manifest
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    assert "manifest" in deployed.fail("release-gate")
    seen = []
    deployed.runner.handlers["jev_factory.evals"] = lambda argv, env: seen.append(argv) or 0
    deployed.apply("release-gate", {"manifest": str(manifest), "no_deepeval": True})
    assert seen[0][:3] == ["run", "--manifest", str(manifest)] and "--no-deepeval" in seen[0]
    assert deployed.json("release-gate/result.json")["exit_code"] == 0
    deployed.runner.handlers["jev_factory.evals"] = lambda argv, env: 4
    assert "exited 4" in deployed.fail(
        "release-gate", {"manifest": str(manifest), "no_deepeval": False}
    )


def test_bundle_refuses_without_the_deployed_calibration(deployed):
    deployed.through("measure-final")
    (deployed.workdir / "deployed" / "calibration.json").unlink()
    error = deployed.fail("bundle", {"repo_suffix": "scorer"})
    assert "missing inputs" in error and "deployed/calibration.json" in error


def test_decide_after_quantize_cites_the_heal_check(deployed):
    record = pipeline.decide(deployed.workdir, deployed.ctx)
    assert record["verdict"] == "ship_candidate"
    assert any(c["name"] == "r1.heal_rounds" for c in record["cited"])


def _client(tmp_path, **gateway):
    from jev_factory.data.teachers import TeacherClient

    return TeacherClient(
        roles(), tmp_path / "cache", caller=FakeGateway(**gateway), sleep=lambda s: None
    )


def test_draft_eval_drafts_the_pool_with_the_teachers_resumably(tmp_path):
    run = Run(tmp_path)
    run.ctx.services.teacher_client = _client(tmp_path, generator=_generator)
    run.through("config", "seed")
    (run.workdir / "pool" / "draft.json").unlink()
    run.apply("draft-eval", {"per_op": 1, "per_reason": 1, "explain": 1})
    pool = run.json("pool/draft.json")
    assert pool["header"]["domain"] == "toy-lamps" and pool["entries"]
    progress = run.workdir / "jobs" / "draft-eval" / "draft-review.progress.json"
    assert json.loads(progress.read_text())["done"] >= len(pool["entries"])


def test_augment_and_targeted_feed_assemble_through_the_teachers(tmp_path):
    rewrite = "could you please brighten things up in there for me today"
    run = Run(tmp_path)
    run.ctx.services.teacher_client = _client(tmp_path, generator=lambda user: rewrite)
    run.through(*CHAIN)
    run.apply("augment", {"per_source": 1})
    counts = run.json("aug/counts.json")
    assert counts["per_source"] == 1 and counts["generated"] > 0
    progress = json.loads((run.workdir / "jobs" / "augment.progress.json").read_text())
    assert progress["total"] > 0
    run.apply("targeted", {"recipes": ["missing-argument"], "per_recipe": 1})
    summary = run.json("aug/targeted.json")
    assert summary["recipes"] == ["missing-argument"]
    run.apply("assemble")
    freeze = run.json("data/freeze.json")["files"]
    assert summary["supplement"] == "aug/supplement.json"
    assert {"variations_0", "supplement", "protected_2"} <= set(freeze)


def test_the_bundle_card_names_the_teachers_of_the_rows_it_trained_on(tmp_path):
    from jev_factory.data.assemble import select_frozen

    rewrite = "could you please brighten things up in there for me today"
    run = Run(tmp_path)
    run.ctx.services.teacher_client = _client(tmp_path, generator=lambda user: rewrite)
    run.through(*CHAIN)
    run.apply("augment", {"per_source": 1})
    run.apply("targeted", {"recipes": ["missing-argument"], "per_recipe": 1})
    run.apply("assemble")
    frozen = select_frozen(run.workdir / "data" / "freeze.json")
    teachers = pipeline._bundle_teachers(run.workdir, frozen.path, apache_only=True)
    assert teachers is not None and teachers.decisions
    assert set(teachers.role_teachers) >= {"GENERATOR", "CORRECTOR", "REVIEWER_B"}
    trained = {str(e["id"]) for e in json.loads(frozen.path.read_text())["entries"]}
    assert set(teachers.per_variation) <= trained


def test_a_bundle_without_synthetic_rows_says_so(deployed):
    deployed.through("measure-final")
    _edge(deployed)
    deployed.apply("edge-check")
    deployed.apply("bundle", {"repo_suffix": "scorer"})
    folder = deployed.workdir / deployed.json("bundle/record.json")["folder"]
    card = (folder / "README.md").read_text()
    assert "No synthetic variations were used" in card
    assert "data set bundle" not in card and "`scorer-train.json` in this repository" in card


def _epochs_trained(run: Run) -> list[str]:
    return [cmd[cmd.index("--epochs") + 1] for _, cmd, _ in run.trainer.calls if "--epochs" in cmd]


def test_a_changed_recipe_retrains_and_an_unchanged_one_reuses_the_run(toy):
    toy.through("train")
    first = len(toy.trainer.calls)
    toy.apply("train", force=True)  # same request: every finished run is reused
    assert len(toy.trainer.calls) == first
    toy.apply("train", {"candidates": {"r2": {"epochs": 4}}})  # more epochs: retrain r2
    assert _epochs_trained(toy)[-1] == "4"
    assert toy.json("runs/r2/train-request.json")["hyperparameters"]["epochs"] == 4
