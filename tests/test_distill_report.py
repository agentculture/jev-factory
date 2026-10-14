"""Distill final report (t20): paired gate over Clef, stock v21 and the deployed student.

Pure-function tests over hand-built predictions: no server, no GPU.
"""

from __future__ import annotations

import pytest

from jev_factory.distill import prereg as dp
from jev_factory.distill import report as R
from jev_factory.distill.records import Gold, Question, Record

CLEAN = "Praveenrajus/jev-bench:scifact"
#: v21 trained on banking77; the jev-bench record source must still be matched to it.
OVERLAP = "Praveenrajus/jev-bench:banking77"
EVAL_ONLY = "jev-bench:emotion"
TYPED = "LocalLLaMA/typed-decisions"
RESAMPLES = 200


def _rec(i: int, source: str = CLEAN, provenance: str = "human") -> Record:
    return Record(
        id=f"{source[-6:]}-{i}",
        state=f"state {i}",
        questions={"c": Question("choice", [("a", "A"), ("b", "B")])},
        gold={"c": Gold(label="a")},
        split="test",
        source=source,
        licence="mit",
        gold_provenance=provenance,
    )


def _row(rec: Record, p_a: float | None) -> dict:
    if p_a is None:
        return {"record_id": rec.id, "error": "boom", "questions": {}}
    return {
        "record_id": rec.id,
        "error": None,
        "questions": {
            "c": {
                "kind": "choice",
                "distribution": {"a": p_a, "b": round(1.0 - p_a, 9)},
                "p_true": None,
                "error": None,
            }
        },
    }


def _rows(recs, right: int, p_right: float, p_wrong_a: float) -> list[dict]:
    """The first *right* records answered ``a`` with p_right; the rest ``b``."""
    return [_row(r, p_right if i < right else p_wrong_a) for i, r in enumerate(recs)]


def _good(recs):
    """Clef 38/40, v21 20/40 at 0.6 (ECE 0.1), student 36/40 well calibrated."""
    return {
        "clef": _rows(recs, 38, 0.9, 0.2),
        "stock_v21": _rows(recs, 20, 0.6, 0.4),
        "student": _rows(recs, 36, 0.97, 0.45),
    }


def _runtime(student_s=0.1, clef_s=1.0, hosts=("box", "box", "box")):
    out = {}
    for (name, median), host in zip(
        (("clef", clef_s), ("stock_v21", 0.2), ("student", student_s)), hosts
    ):
        out[name] = {
            "host": host,
            "latency": {"n": 40, "median_s": median, "p95_s": median * 2},
            "memory": {"peak_mib": 1000.0, "probe": "proc-rss", "measured": True},
        }
    return out


QUANT_OK = {"quant_vs_bf16_delta_points": 0.0, "heal_count": 0, "deployed_selection_ece": 0.05}


def _build(test, preds, *, prereg=None, runtime=None, quant=None, cal=None, held_out=(), gate=None):
    return R.build_report(
        records_by_side={"test": test, "held_out": list(held_out)},
        predictions=preds,
        calibrations=cal or {"clef": None, "stock_v21": None, "student": None},
        calibration_sources={"clef": "raw", "stock_v21": "served", "student": "quant"},
        runtime=runtime or _runtime(),
        prereg=prereg or dp.validate(dp.example_doc()),
        quant_manifest=quant or QUANT_OK,
        resamples=RESAMPLES,
        gate=gate,
    )


def _bar(rep, name, section="c42"):
    return next(b for b in rep[section]["bars"] if b["bar"] == name)


def test_a_good_run_passes_every_c42_and_c43_bar_each_with_n_and_ci():
    recs = [_rec(i) for i in range(40)]
    rep = _build(recs, _good(recs))
    assert rep["result"] == R.PASSED, rep["c42"]["missed"] + rep["c43"]["missed"]
    names = [b["bar"] for b in rep["c42"]["bars"]]
    assert names == list(dp.TEST_BARS)
    assert [b["bar"] for b in rep["c43"]["bars"]] == list(dp.QUANT_BARS)
    acc = _bar(rep, "accuracy_diff_ci_lower_min")
    assert acc["n"] == 40 and acc["ci_low"] is not None and acc["ci_high"] is not None
    assert acc["point"] == pytest.approx(0.9 - 0.5)
    assert _bar(rep, "gap_closure_min")["value"] == pytest.approx(0.4 / 0.45)
    assert rep["c42"]["side"] == "test" and rep["c42"]["pool"] == R.HEADLINE


def test_every_c42_bar_follows_the_locked_prereg():
    recs = [_rec(i) for i in range(40)]
    doc = dp.example_doc()
    doc["bars"]["test"]["gap_closure_min"] = 0.95
    doc["bars"]["test"]["ece_max"] = 0.01
    rep = _build(recs, _good(recs), prereg=dp.validate(doc))
    assert rep["c42"]["result"] == R.MISSED
    assert set(rep["c42"]["missed"]) == {"gap_closure_min", "ece_max"}
    assert _bar(rep, "gap_closure_min")["threshold"] == 0.95


def test_headline_is_human_gold_from_clean_sources_and_overlap_is_reported_apart():
    clean = [_rec(i) for i in range(40)]
    overlap = [_rec(i, OVERLAP) for i in range(10)]
    evaluated = [_rec(i, EVAL_ONLY) for i in range(6)]
    preds = _good(clean)
    for recs in (overlap, evaluated):
        preds["clef"] += _rows(recs, len(recs), 0.9, 0.2)
        preds["stock_v21"] += _rows(recs, len(recs), 0.9, 0.2)
        preds["student"] += _rows(recs, 0, 0.9, 0.1)  # all wrong: must not touch the gate
    rep = _build(clean + overlap + evaluated, preds)
    assert rep["sources"][OVERLAP]["status"] == "overlapping"
    assert rep["sources"][OVERLAP]["checked_as"] == "jev-bench:banking77"
    assert rep["sources"][EVAL_ONLY]["status"] == "v21-evaluated"
    assert rep["sources"][CLEAN]["headline"] is True
    pools = rep["sides"]["test"]["pools"]
    assert pools[R.HEADLINE]["items"] == 40
    assert pools[R.OVERLAPPING]["items"] == 10
    assert pools[R.V21_EVALUATED]["items"] == 6
    assert pools[R.OVERLAPPING]["stats"]["student.accuracy"]["value"] == 0.0
    assert rep["c42"]["result"] == R.PASSED
    assert _bar(rep, "accuracy_diff_ci_lower_min")["n"] == 40


def test_typed_decisions_agreement_is_reported_not_gated():
    clean = [_rec(i) for i in range(40)]
    typed = [_rec(i, TYPED, "model-generated") for i in range(12)]
    preds = _good(clean)
    preds["clef"] += _rows(typed, 12, 0.9, 0.2)
    preds["stock_v21"] += _rows(typed, 12, 0.9, 0.2)
    preds["student"] += _rows(typed, 0, 0.9, 0.1)
    rep = _build(clean + typed, preds)
    block = rep["sides"]["test"]["pools"][R.MODEL_GENERATED]
    assert block["items"] == 12
    assert block["stats"]["student.accuracy"]["value"] == 0.0
    assert block["stats"]["clef.accuracy"]["value"] == 1.0
    assert rep["sources"][TYPED]["headline"] is False
    assert rep["c42"]["result"] == R.PASSED


def test_the_three_models_are_resampled_jointly():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student"] = [dict(r) for r in preds["stock_v21"]]  # identical to v21, item by item
    rep = _build(recs, preds)
    diff = rep["sides"]["test"]["pools"][R.HEADLINE]["stats"]["accuracy_diff_student_v21"]
    # Unpaired resampling would give a spread; paired draws give exactly zero every time.
    assert diff["ci_low"] == 0.0 and diff["ci_high"] == 0.0
    assert rep["c42"]["result"] == R.MISSED
    assert "accuracy_diff_ci_lower_min" in rep["c42"]["missed"]


def test_paired_bootstrap_hands_every_statistic_one_shared_draw():
    seen = []

    def stat(indices):
        seen.append(list(indices))
        return {"x": float(sum(indices)), "y": float(sum(indices))}

    out = R.paired_bootstrap(5, stat, seed=3, resamples=50)
    assert len(seen) == 51  # the point estimate plus one call per draw
    assert out["x"] == out["y"]
    assert out["x"]["n"] == 5


def test_higher_accuracy_with_higher_ece_than_v21_is_a_regression():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student"] = _rows(recs, 30, 1.0, 0.0)  # 0.75 accuracy, every answer at p = 1
    doc = dp.example_doc()
    doc["bars"]["test"]["ece_max"] = 1.0
    doc["bars"]["test"]["ece_no_worse_than_v21"] = False
    rep = _build(recs, preds, prereg=dp.validate(doc))
    assert rep["regression"] is True
    assert rep["sides"]["test"]["pools"][R.HEADLINE]["regression"]["regression"] is True
    assert rep["c42"]["result"] == R.MISSED and "regression" in rep["c42"]["missed"]
    assert rep["result"] == R.MISSED


def test_a_miss_is_recorded_as_a_miss():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student"] = _rows(recs, 10, 0.9, 0.2)
    rep = _build(recs, preds)
    assert rep["result"] == R.MISSED
    assert "accuracy_diff_ci_lower_min" in rep["c42"]["missed"]
    assert _bar(rep, "accuracy_diff_ci_lower_min")["result"] == R.MISSED


def test_no_headline_items_is_a_miss_with_a_reason_not_a_crash():
    typed = [_rec(i, TYPED, "model-generated") for i in range(5)]
    preds = {m: _rows(typed, 5, 0.9, 0.1) for m in R.MODELS}
    rep = _build(typed, preds)
    assert rep["c42"]["result"] == R.MISSED
    assert _bar(rep, "gap_closure_min")["reason"]


def test_unanswered_questions_count_as_wrong():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student"] = [_row(r, None) for r in recs]
    rep = _build(recs, preds)
    block = rep["sides"]["test"]["pools"][R.HEADLINE]
    assert block["unanswered"]["student"] == 40
    assert block["stats"]["student.accuracy"]["value"] == 0.0
    assert rep["c42"]["result"] == R.MISSED


def test_c43_is_read_from_the_quant_manifest():
    recs = [_rec(i) for i in range(40)]
    quant = {"quant_vs_bf16_delta_points": -4.0, "heal_count": 2, "deployed_selection_ece": 0.2}
    rep = _build(recs, _good(recs), quant=quant)
    assert rep["c43"]["result"] == R.MISSED
    assert set(rep["c43"]["missed"]) == {"max_accuracy_drop_points", "heal_rounds_max", "ece_max"}
    drop = _bar(rep, "max_accuracy_drop_points", "c43")
    assert drop["value"] == 4.0
    assert drop["source"].endswith("quant_manifest.json:quant_vs_bf16_delta_points")
    assert rep["result"] == R.MISSED


def test_latency_and_memory_are_reported_for_all_three_on_the_same_box():
    recs = [_rec(i) for i in range(40)]
    rep = _build(recs, _good(recs))
    table = rep["runtime"]
    assert set(table["models"]) == set(R.MODELS)
    for row in table["models"].values():
        assert row["latency_median_s"] is not None and row["latency_p95_s"] is not None
        assert row["peak_memory_mib"] == 1000.0 and row["memory_probe"] == "proc-rss"
    assert table["same_box"] is True
    assert _bar(rep, "latency_ratio_max")["value"] == pytest.approx(0.1)
    split = _build(recs, _good(recs), runtime=_runtime(hosts=("a", "b", "a")))
    assert split["runtime"]["same_box"] is False
    slow = _build(recs, _good(recs), runtime=_runtime(student_s=0.5))
    assert "latency_ratio_max" in slow["c42"]["missed"]


def test_each_models_calibration_is_applied_and_recorded():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    cal = {"temperature_by_kind": {"choice": 5.0}, "vector_by_kind": {}}
    raw = _build(recs, preds)
    hot = _build(recs, preds, cal={"clef": None, "stock_v21": None, "student": cal})
    key = "student.ece"
    pool = R.HEADLINE
    assert (
        hot["sides"]["test"]["pools"][pool]["stats"][key]["value"]
        != raw["sides"]["test"]["pools"][pool]["stats"][key]["value"]
    )
    assert hot["calibration_used"]["student"] == "quant"


def test_clef_calibrated_row_is_reported_beside_raw_clef():
    recs = [_rec(i) for i in range(40)]
    cal = {"temperature_by_kind": {"choice": 2.0}, "vector_by_kind": {}}
    rep = _build(
        recs,
        _good(recs),
        cal={"clef": None, "stock_v21": None, "student": None, "clef_calibrated": cal},
    )
    stats = rep["sides"]["test"]["pools"][R.HEADLINE]["stats"]
    assert "clef_calibrated.ece" in stats and "clef.ece" in stats
    assert stats["clef_calibrated.accuracy"]["value"] == stats["clef.accuracy"]["value"]


def test_held_out_is_reported_but_not_gated():
    test = [_rec(i) for i in range(40)]
    held = [_rec(100 + i) for i in range(20)]
    preds = _good(test)
    preds["clef"] += _rows(held, 20, 0.9, 0.2)
    preds["stock_v21"] += _rows(held, 20, 0.9, 0.2)
    preds["student"] += _rows(held, 0, 0.9, 0.1)
    rep = _build(test, preds, held_out=held)
    assert rep["sides"]["held_out"]["c42"]["gated"] is False
    assert rep["sides"]["held_out"]["c42"]["result"] == R.MISSED
    assert rep["c42"]["result"] == R.PASSED


# --------------------------------------------------------------------- the decision gate (d4)


GATE = {
    "fitted": True,
    "thresholds_by_kind": {
        "choice": {"floor": 0.9, "margin": None, "max_entropy": None},
        "noul": {"floor": None, "margin": None, "max_entropy": None},
        "score": {"floor": None, "margin": None, "max_entropy": None},
    },
    "calibration_sha256": "c" * 64,
}


def test_report_shows_per_kind_coverage_and_selective_accuracy_of_the_gated_student():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)  # student: 36 right at 0.97, 4 wrong with p(a)=0.45 -> top 0.55
    preds["student"][-1] = _row(recs[-1], None)  # one unanswered question
    rep = _build(recs, preds, gate=GATE)
    dg = rep["decision_gate"]
    assert dg["fitted"] is True and dg["gated"] is False and dg["reason"]
    assert dg["thresholds_by_kind"] == GATE["thresholds_by_kind"]
    choice = dg["sides"]["test"]["choice"]
    # floor 0.9 answers the 36 confident right ones; the 3 hesitant and 1 unanswered abstain
    assert choice["coverage"]["value"] == pytest.approx(36 / 40)
    assert choice["selective_accuracy"]["value"] == 1.0
    assert choice["coverage"]["n"] == 40 and choice["coverage"]["ci_low"] is not None
    assert choice["abstained"] == 4 and choice["unanswered"] == 1
    assert dg["sides"]["test"]["all"]["coverage"]["value"] == pytest.approx(36 / 40)
    assert dg["sides"]["test"]["noul"]["coverage"]["n"] == 0
    # reported only: the gate never changes the c42 verdict
    assert rep["c42"] == _build(recs, preds)["c42"]
    assert "decision_gate" not in rep["c42"]["missed"]


def test_report_without_a_fitted_gate_says_so():
    recs = [_rec(i) for i in range(40)]
    dg = _build(recs, _good(recs))["decision_gate"]
    assert dg["fitted"] is False and dg["sides"] == {} and dg["reason"]
