"""Distill decision gate (deviation d4): per-kind answer/abstain thresholds, fit on fit only.

Pure-Python tests over hand-built calibrated distributions and a fake measurer: no server,
no GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import gate as G
from jev_factory.distill import metrics as dm
from jev_factory.distill import prereg as dp
from jev_factory.distill.records import Gold, Question, Record

RESAMPLES = 200
OFF = G.KindThresholds()

# --------------------------------------------------------------------- decide()


@pytest.mark.parametrize(
    "dist, thresholds, outcome, reason",
    [
        # every threshold disabled: the bare argmax is answered
        ([0.4, 0.35, 0.25], OFF, "answer", None),
        # floor: top below the floor abstains; at the floor answers
        ([0.4, 0.35, 0.25], G.KindThresholds(floor=0.5), "abstain", "floor"),
        ([0.5, 0.3, 0.2], G.KindThresholds(floor=0.5), "answer", None),
        # margin: top minus second below the margin abstains
        ([0.4, 0.35, 0.25], G.KindThresholds(margin=0.1), "abstain", "margin"),
        ([0.6, 0.3, 0.1], G.KindThresholds(margin=0.1), "answer", None),
        # max_entropy: normalised entropy above the bound abstains
        ([1 / 3, 1 / 3, 1 / 3], G.KindThresholds(max_entropy=0.99), "abstain", "entropy"),
        ([0.98, 0.01, 0.01], G.KindThresholds(max_entropy=0.5), "answer", None),
        # order: floor fires before margin before entropy
        ([0.4, 0.35, 0.25], G.KindThresholds(0.5, 0.1, 0.1), "abstain", "floor"),
        ([0.6, 0.35, 0.05], G.KindThresholds(0.5, 0.3, 0.1), "abstain", "margin"),
        ([0.6, 0.2, 0.2], G.KindThresholds(0.5, 0.3, 0.1), "abstain", "entropy"),
        # noul as its (P(false), P(true)) pair
        ([0.2, 0.8], G.KindThresholds(floor=0.7, margin=0.5), "answer", None),
        ([0.45, 0.55], G.KindThresholds(floor=0.7), "abstain", "floor"),
    ],
)
def test_decide_table(dist, thresholds, outcome, reason):
    d = G.decide(dist, thresholds)
    assert (d.outcome, d.reason) == (outcome, reason)
    assert d.index == dm.argmax(dist)


def test_decide_on_a_label_mapping_names_the_top_label_and_ties_go_first():
    d = G.decide({"fix": 0.5, "wait": 0.5}, OFF)
    assert d.label == "fix"
    assert d.outcome == "answer"
    assert d.margin == 0.0
    assert G.decide({"fix": 0.2, "wait": 0.8}, OFF).label == "wait"


def test_decide_is_deterministic_and_refuses_a_non_distribution():
    dist = [0.7, 0.2, 0.1]
    th = G.KindThresholds(0.5, 0.2, 0.9)
    assert G.decide(dist, th) == G.decide(list(dist), th)
    with pytest.raises(ValueError):
        G.decide([0.7, 0.7], th)
    with pytest.raises(ValueError):
        G.decide([1.0], th)


def test_thresholds_out_of_range_are_refused():
    with pytest.raises(ValueError):
        G.KindThresholds(floor=1.5)
    with pytest.raises(ValueError):
        G.KindThresholds(margin=-0.1)


# --------------------------------------------------------------------- fit_gate()


def _items(kind, n_sure, n_unsure, *, sure=0.9, unsure=0.55):
    """*n_sure* confident right answers, *n_unsure* hesitant wrong ones (gold index 0)."""
    out = []
    for _ in range(n_sure):
        out.append(dm.PredItem(kind, (sure, 1.0 - sure), 0))
    for _ in range(n_unsure):
        out.append(dm.PredItem(kind, (1.0 - unsure, unsure), 0))
    return out


def _objective(min_cov):
    return {"objective": "max_selective_accuracy", "min_coverage": min_cov}


GRID = {"floor": [None, 0.6, 0.95], "margin": [None], "max_entropy": [None]}


def _toy():
    fit = _items("choice", 8, 4) + _items("noul", 6, 4) + _items("score", 10, 0)
    sel = _items("choice", 6, 2) + _items("noul", 5, 5) + _items("score", 5, 0)
    return fit, sel


def test_fit_gate_picks_the_best_selective_accuracy_meeting_the_coverage_minimum():
    fit, sel = _toy()
    cov = {"choice": 0.6, "noul": 0.6, "score": 0.6}
    out = G.fit_gate(fit, sel, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    choice = out["by_kind"]["choice"]
    # floor 0.6 keeps the 8 confident answers: coverage 8/12 >= 0.6, selective accuracy 1.0
    assert choice["thresholds"] == {"floor": 0.6, "margin": None, "max_entropy": None}
    assert choice["fit"]["coverage"] == pytest.approx(8 / 12)
    assert choice["fit"]["selective_accuracy"] == 1.0
    assert choice["fit"]["coverage"] >= cov["choice"]
    # noul: floor 0.6 would keep only 6/10 = 0.6 >= 0.6, accuracy 1.0
    assert out["by_kind"]["noul"]["thresholds"]["floor"] == 0.6
    # score: everything already right, so the widest coverage (the disabled gate) wins the tie
    assert out["by_kind"]["score"]["thresholds"] == {
        "floor": None,
        "margin": None,
        "max_entropy": None,
    }


def test_fit_gate_respects_a_tighter_coverage_minimum():
    fit, sel = _toy()
    cov = {"choice": 0.8, "noul": 0.8, "score": 0.8}
    out = G.fit_gate(fit, sel, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    for kind in ("choice", "noul", "score"):
        assert out["by_kind"][kind]["fit"]["coverage"] >= 0.8
    assert out["by_kind"]["choice"]["thresholds"]["floor"] is None


def test_fit_gate_never_reads_the_selection_fold_to_choose():
    fit, sel = _toy()
    cov = {"choice": 0.6, "noul": 0.6, "score": 0.6}
    a = G.fit_gate(fit, sel, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    flipped = [dm.PredItem(i.kind, i.probs, 1) for i in sel]  # every selection gold flipped
    b = G.fit_gate(fit, flipped, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    assert G.thresholds_by_kind(a) == G.thresholds_by_kind(b)


def test_fit_gate_reports_selection_coverage_and_selective_accuracy_with_seeded_cis():
    fit, sel = _toy()
    cov = {"choice": 0.6, "noul": 0.6, "score": 0.6}
    out = G.fit_gate(fit, sel, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    choice = out["by_kind"]["choice"]["selection"]
    assert choice["coverage"]["value"] == pytest.approx(6 / 8)
    assert choice["selective_accuracy"]["value"] == 1.0
    for stat in (choice["coverage"], choice["selective_accuracy"]):
        assert stat["n"] == 8
        assert stat["ci_low"] is not None
        assert stat["ci_high"] is not None
    again = G.fit_gate(fit, sel, objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    assert again == out


def test_fit_gate_refuses_when_no_grid_point_meets_the_minimum():
    fit = _items("choice", 2, 8)
    grid = {"floor": [0.6], "margin": [None], "max_entropy": [None]}
    cov = {"choice": 0.5, "noul": 0.5, "score": 0.5}
    objective = _objective(cov)
    with pytest.raises(CliError, match="coverage"):
        G.fit_gate(fit, [], objective=objective, grid=grid, resamples=RESAMPLES)


def test_a_kind_with_no_fit_rows_is_left_ungated_and_says_so():
    fit = _items("choice", 8, 4)
    cov = {"choice": 0.6, "noul": 0.6, "score": 0.6}
    out = G.fit_gate(fit, [], objective=_objective(cov), grid=GRID, resamples=RESAMPLES)
    noul = out["by_kind"]["noul"]
    assert noul["fitted"] is False
    assert noul["reason"]
    assert noul["thresholds"] == {"floor": None, "margin": None, "max_entropy": None}


# --------------------------------------------------------------------- run_gate()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rec(i: int) -> Record:
    return Record(
        id=f"r{i}",
        state=f"state {i}",
        questions={
            "c": Question("choice", [("fix", "fix it"), ("wait", "wait")]),
            "n": Question("noul", [("false", ""), ("true", "")]),
        },
        gold={"c": Gold(label="fix"), "n": Gold(label="true", probability_true=1.0)},
        split="train",
        source="src",
        licence="mit",
        gold_provenance="human",
    )


def folds():
    return {"fit": [_rec(i) for i in range(0, 20)], "selection": [_rec(i) for i in range(20, 32)]}


def _answer(i: int) -> dict:
    sure = i % 3 != 0  # one in three is a hesitant wrong answer
    p = 0.9 if sure else 0.45
    return {
        "c": {"kind": "choice", "distribution": {"fix": p, "wait": 1 - p}, "error": None},
        "n": {"kind": "noul", "distribution": None, "p_true": p, "error": None},
    }


class FakeMeasurer:
    def __init__(self):
        self.calls = []

    def __call__(self, fmt, artifact, fold_records):
        self.calls.append((fmt, str(artifact), sorted(fold_records)))
        preds = {}
        for fold, recs in fold_records.items():
            preds[fold] = [
                {"record_id": r.id, "error": None, "questions": _answer(int(r.id[1:]))}
                for r in recs
            ]
        return {"predictions": preds}


CAL = {
    "temperature_by_kind": {"choice": 1.0, "score": 1.0, "noul": 1.0},
    "vector_by_kind": {},
    "upstream_temperature_by_kind_used": False,
}


def setup(tmp_path, doc=None):
    run = tmp_path / "run"
    prereg = tmp_path / "prereg.json"
    doc = doc or dp.example_doc()
    doc["gate"]["min_coverage"] = {"choice": 0.5, "noul": 0.5, "score": 0.5}
    prereg.write_text(json.dumps(doc), encoding="utf-8")
    dp.lock(prereg, run)
    q = run / "distill" / "quant"
    q.mkdir(parents=True)
    cal = q / "calibration.json"
    cal.write_text(json.dumps(CAL), encoding="utf-8")
    (q / "format_decision.json").write_text(json.dumps({"format": "bf16"}), encoding="utf-8")
    (q / "quant_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "chosen_format": "bf16",
                "deployed_artifact": str(tmp_path / "artifact"),
                "outputs": {str(cal): _sha(cal)},
            }
        ),
        encoding="utf-8",
    )
    return run, prereg


def test_dry_run_plans_the_fit_and_writes_nothing(tmp_path):
    run, prereg = setup(tmp_path)
    before = sorted(p for p in run.rglob("*"))
    measurer = FakeMeasurer()
    out = G.run_gate(run, prereg_path=prereg, folds=folds(), measurer=measurer)
    assert out["dry_run"] is True
    assert out["deployed_format"] == "bf16"
    assert out["calibration_sha256"] == _sha(run / "distill/quant/calibration.json")
    assert sorted(p for p in run.rglob("*")) == before
    assert not (run / G.GATE_REL).exists()
    assert measurer.calls == []


def test_apply_fits_on_the_deployed_build_and_writes_gate_json(tmp_path):
    run, prereg = setup(tmp_path)
    measurer = FakeMeasurer()
    out = G.run_gate(
        run, prereg_path=prereg, folds=folds(), measurer=measurer, apply=True, resamples=RESAMPLES
    )
    assert measurer.calls == [("bf16", str(tmp_path / "artifact"), ["fit", "selection"])]
    doc = json.loads((run / G.GATE_REL / G.GATE_FILE).read_text())
    assert doc == out
    assert set(doc["thresholds_by_kind"]) == {"choice", "noul", "score"}
    choice = doc["by_kind"]["choice"]
    assert choice["fit"]["selective_accuracy"] == 1.0
    assert choice["fit"]["coverage"] == 0.65
    assert doc["by_kind"]["score"]["fitted"] is False  # no score questions in the folds
    assert doc["calibration_sha256"] == _sha(run / "distill/quant/calibration.json")
    assert doc["prereg_sha256"] == _sha(prereg)
    assert doc["objective"]["objective"] == "max_selective_accuracy"
    sel = doc["by_kind"]["choice"]["selection"]
    assert sel["coverage"]["ci_low"] is not None
    assert sel["selective_accuracy"]["value"] == 1.0
    assert doc["fitted_on"] == "fit"
    assert doc["reported_on"] == "selection"


def test_apply_without_a_measurer_is_refused(tmp_path):
    run, prereg = setup(tmp_path)
    fold_set = folds()
    with pytest.raises(CliError, match="measurer"):
        G.run_gate(run, prereg_path=prereg, folds=fold_set, apply=True)


def test_a_held_out_or_test_fold_is_refused(tmp_path):
    run, prereg = setup(tmp_path)
    for side in ("held_out", "test"):
        bad_folds = {**folds(), side: [_rec(99)]}
        with pytest.raises(CliError):
            G.run_gate(run, prereg_path=prereg, folds=bad_folds)


def test_a_prereg_without_a_gate_block_is_refused(tmp_path):
    run, prereg = setup(tmp_path)
    doc = json.loads(prereg.read_text())
    del doc["gate"]
    other = tmp_path / "other"
    other.mkdir()
    p2 = other / "prereg.json"
    p2.write_text(json.dumps(doc))
    dp.lock(p2, other / "run")
    q = run / "distill" / "quant"
    (other / "run" / "distill" / "quant").mkdir(parents=True)
    for name in ("calibration.json", "format_decision.json", "quant_manifest.json"):
        (other / "run" / "distill" / "quant" / name).write_bytes((q / name).read_bytes())
    other_run = other / "run"
    fold_set = folds()
    with pytest.raises(CliError, match="gate"):
        G.run_gate(other_run, prereg_path=p2, folds=fold_set)


def test_a_calibration_that_does_not_match_the_quant_manifest_is_refused(tmp_path):
    run, prereg = setup(tmp_path)
    (run / "distill/quant/calibration.json").write_text(json.dumps({**CAL, "x": 1}))
    fold_set = folds()
    with pytest.raises(CliError):
        G.run_gate(run, prereg_path=prereg, folds=fold_set)
