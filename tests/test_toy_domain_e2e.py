"""The toy domain end to end through the pipeline's own stage code, on CPU (o6).

A domain that is not the jev CLI (tests/fixtures/toy_domain: five operations,
two of them mutating) runs validate (the config stage), split, assemble,
calibrate, gate sweep, select and decide through ``jev_factory.factory.pipeline``,
the same stages the jev-tool model runs. What needs teachers, a GPU or a model
is replaced by its output: the eval pool the draft-eval stage would write, a
hash-verified sealed held-out import, and each candidate's validation
predictions scored by a synthetic model through the causal-LM adapter's own
readout (``tests/pipeline_support.py``). No stage below is stubbed.
"""

from __future__ import annotations

import json

import pytest

from jev_factory.core.gate import Thresholds
from jev_factory.decide import records
from jev_factory.factory import pipeline
from tests import pipeline_support as ps

CPU_CHAIN = (
    "config",
    "preregister",
    "seed",
    "draft-heldout",
    "split",
    "augment",
    "targeted",
    "assemble",
)


def _apply(workdir, ctx, name, knobs=None):
    man = pipeline.run(workdir, name, ctx, knobs, apply=True)
    assert man["status"] == "complete", man.get("error")
    return man


@pytest.fixture
def toy_run(tmp_path):
    workdir = tmp_path / "toy-run"
    workdir.mkdir()
    runner = ps.ModuleRunner(workdir)
    ctx = ps.toy_context(workdir, runner)
    (workdir / "prereg.json").write_text(json.dumps(ps.toy_prereg()), encoding="utf-8")
    ps.write_pool(workdir)
    sealed, sealed_sha = ps.write_sealed_heldout(tmp_path / "operator" / "held-out.json")
    knobs = {
        "draft-heldout": {"import_from": str(sealed), "import_sha256": sealed_sha},
        "split": {"val_size": 30, "test_size": 20, "fold_seed": 3},
        "augment": {"per_source": 0},
    }
    for name in CPU_CHAIN:
        _apply(workdir, ctx, name, knobs.get(name))
    for name, model in ps.MODELS.items():
        ps.synth_candidate_predictions(workdir, name, model)
    _apply(workdir, ctx, "select", {"bootstrap_resamples": 0})
    return workdir, ctx, runner


@pytest.mark.behavioral("o6")
def test_the_toy_domain_is_a_valid_non_jev_domain_with_a_mutating_half():
    ops = ps.DOMAIN.operations
    assert ps.DOMAIN.name != "jev-cli"
    assert len(ops) >= 3
    assert any(not op.read_only for op in ops)


@pytest.mark.behavioral("o6")
def test_validate_split_assemble_select_and_decide_run_end_to_end_on_cpu(toy_run):
    workdir, ctx, runner = toy_run
    # validate: the config stage validated the domain and recorded its surface.
    config = json.loads((workdir / "config.json").read_text())
    assert config["domain"] == "toy-lamps"
    assert config["mutating"] == 2
    # split: grouped sides plus seeded fit/selection folds.
    folds = json.loads((workdir / "splits" / "folds.json").read_text())
    assert folds["fit_ids"]
    assert folds["selection_ids"]
    assert not set(folds["fit_ids"]) & set(folds["selection_ids"])
    # assemble: frozen by sha256, scorer rows with -nocand rows.
    freeze = json.loads((workdir / "data" / "freeze.json").read_text())
    assert "scorer_train" in freeze["files"]
    rows = json.loads((workdir / "data" / "scorer-train.json").read_text())["entries"]
    assert any(str(r["id"]).endswith("-nocand") for r in rows)
    # calibrate and gate sweep, per candidate, then the rule.
    for name in ("r1", "r2"):
        here = workdir / "select" / name
        assert json.loads((here / "calibration.json").read_text())["fold"] == "fit"
        assert Thresholds.from_json(json.loads((here / "gate.json").read_text()))
        assert len(json.loads((here / "sweep.json").read_text())) > 1
        assert "pooled_change_rate" in json.loads((here / "probe.json").read_text())
    selection = json.loads((workdir / "select" / "selection.json").read_text())
    assert selection["winner"] == "r1"
    assert selection["verdict"] == "ship_candidate"
    # the probe ran through the real probe module with the synthetic model.
    assert [m for m, _ in runner.calls] == ["jev_factory.measure.probe"] * 2
    # decide: one record, the rule's verdict, every metric cited with a sha256.
    record = pipeline.decide(workdir, ctx)
    assert record["verdict"] == "ship_candidate"
    assert record["params"] == {"candidate": "r1"}
    assert record["decider"] == "rule"
    assert record["id"] == "D1"
    assert all(len(c["sha256"]) == 64 for c in record["cited"])
    assert records.load(workdir / "decisions.jsonl") == [record]


@pytest.mark.behavioral("o6")
def test_the_bad_candidate_is_filtered_by_the_rule_not_by_hand(toy_run):
    workdir, _ctx, _runner = toy_run
    good = json.loads((workdir / "select" / "r1" / "summary.json").read_text())
    bad = json.loads((workdir / "select" / "r2" / "summary.json").read_text())
    assert good["wrong_mutating"] == 0
    assert good["permutation_change"] == 0.0
    assert bad["permutation_change"] > good["permutation_change"]
    trail = json.loads((workdir / "select" / "selection.json").read_text())["trail"]
    survivors = {step["step"]: step["survivors"] for step in trail}
    assert "r2" not in survivors["permutation_robustness"]


@pytest.mark.behavioral("o6")
def test_every_stage_of_the_chain_left_a_complete_manifest(toy_run):
    workdir, _ctx, _runner = toy_run
    for name in CPU_CHAIN + ("select",):
        man = json.loads((workdir / "manifests" / f"{name}.json").read_text())
        assert man["status"] == "complete"
        assert man["stage"] == name
        assert man["knobs"]["_domain"]["name"] == "toy-lamps"


@pytest.mark.behavioral("o6")
def test_rerunning_the_chain_is_a_no_op(toy_run):
    workdir, ctx, _runner = toy_run
    for name in CPU_CHAIN:
        knobs = {
            "draft-heldout": json.loads((workdir / "manifests" / "draft-heldout.json").read_text())[
                "knobs"
            ],
        }.get(name)
        if knobs is not None:
            knobs = {k: v for k, v in knobs.items() if not k.startswith("_")}
        if name == "split":
            knobs = {"val_size": 30, "test_size": 20, "fold_seed": 3}
        if name == "augment":
            knobs = {"per_source": 0}
        man = pipeline.run(workdir, name, ctx, knobs, apply=True)
        assert man.get("skipped") is True, name
