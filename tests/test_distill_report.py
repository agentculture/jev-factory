"""Distill final report (t20, t14): c17 bars over Clef, stock, the bf16 student and the quant.

Pure-function tests over hand-built predictions: no server, no GPU.
"""

from __future__ import annotations

import pytest

from jev_factory.distill import prereg as dp
from jev_factory.distill import report as R
from jev_factory.distill.records import Gold, Question, Record

CLEAN = "Praveenrajus/jev-bench:scifact"
#: the stock model trained on banking77; the jev-bench record source must still match it.
OVERLAP = "Praveenrajus/jev-bench:banking77"
EVAL_ONLY = "jev-bench:emotion"
TYPED = "LocalLLaMA/typed-decisions"
RESAMPLES = 200
PREREG_SHA = "e" * 64
BUNDLE = {"files": {"torso": "a" * 64}, "manifest_sha256": "b" * 64, "weight_bytes": 900_000_000}


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
    """Clef 38/40, stock 20/40 at 0.6 (ECE 0.1), both students 36/40 well calibrated."""
    return {
        "clef": _rows(recs, 38, 0.9, 0.2),
        "stock": _rows(recs, 20, 0.6, 0.4),
        "student_bf16": _rows(recs, 36, 0.97, 0.45),
        "student": _rows(recs, 36, 0.97, 0.45),
    }


def _runtime(student_s=0.1, clef_s=1.0, hosts=("box", "box", "box", "box")):
    out = {}
    for (name, median), host in zip(
        (("clef", clef_s), ("stock", 0.2), ("student_bf16", 0.3), ("student", student_s)), hosts
    ):
        out[name] = {
            "host": host,
            "latency": {"n": 40, "median_s": median, "p95_s": median * 2},
            "memory": {"peak_mib": 1000.0, "probe": "proc-rss", "measured": True},
        }
    return out


QUANT_OK = {"quant_vs_bf16_delta_points": 0.0, "heal_count": 0, "deployed_selection_ece": 0.05}


def _build(
    test,
    preds,
    *,
    prereg=None,
    runtime=None,
    quant=None,
    cal=None,
    held_out=(),
    gate=None,
    approved=None,
    bundle=BUNDLE,
    probe=(),
    unverified=False,
):
    every = [*test, *held_out]
    return R.build_report(
        records_by_side={"test": test, "held_out": list(held_out)},
        predictions=preds,
        calibrations=cal or {"clef": None, "stock": None, "student": None},
        calibration_sources={"clef": "raw", "stock": "served", "student": "quant"},
        runtime=runtime or _runtime(),
        context=R.ReportContext(
            prereg=prereg or dp.validate(dp.example_doc()),
            prereg_sha256=PREREG_SHA,
            approved_ids={r.id for r in every} if approved is None else approved,
            quant_manifest=quant or QUANT_OK,
            bundle=bundle,
            gate=gate,
            probe_records=list(probe),
            resamples=RESAMPLES,
            unverified_gold=unverified,
        ),
    )


def test_unverified_mode_adds_a_marked_c17_on_model_gold_and_leaves_the_headline_alone():
    """d10: model-drafted clean test gold gets its own c17, marked UNVERIFIED."""
    recs = [_rec(i, provenance="model-generated") for i in range(40)]
    rep = _build(recs, _good(recs), unverified=True)
    assert rep["headline_n"]["records"] == 0
    assert rep["result"] == R.MISSED  # no human gold: the real c17 is an honest miss
    block = rep["unverified_c17"]
    assert block["verified"] is False
    assert block["pool"] == R.UNVERIFIED
    assert block["unverified_share"] == 1.0
    assert block["n_records"] == 40
    assert block["note"].startswith("UNVERIFIED")
    assert rep["unverified_result"] == block["result"] == R.PASSED, block["missed"]
    assert rep["verification"].startswith("UNVERIFIED: 100.0%")


def test_unverified_share_counts_human_headline_records_and_skips_unclean_sources():
    model = [_rec(i, provenance="model-generated") for i in range(30)]
    human = [_rec(100 + i) for i in range(10)]
    overlap = [_rec(i, OVERLAP, provenance="model-generated") for i in range(5)]
    recs = [*model, *human, *overlap]
    rep = _build(recs, _good(recs), unverified=True)
    block = rep["unverified_c17"]
    assert block["n_records"] == 30
    assert block["unverified_share"] == pytest.approx(30 / 40)
    assert block["sources"] == [CLEAN]


def test_without_unverified_mode_the_report_has_no_unverified_block():
    recs = [_rec(i, provenance="model-generated") for i in range(10)]
    rep = _build(recs, _good(recs))
    assert "unverified_c17" not in rep
    assert "unverified_result" not in rep


def _bar(rep, name, section="c42"):
    return next(b for b in rep[section]["bars"] if b["bar"] == name)


def test_a_good_run_passes_every_c42_and_c43_bar_each_with_n_and_ci():
    recs = [_rec(i) for i in range(40)]
    rep = _build(recs, _good(recs))
    assert rep["result"] == R.PASSED, rep["c17"]["missed"]
    assert rep["c42"]["result"] == R.PASSED, rep["c42"]["missed"]
    assert rep["c43"]["result"] == R.PASSED, rep["c43"]["missed"]
    names = [b["bar"] for b in rep["c42"]["bars"]]
    assert names == list(dp.TEST_BARS)
    assert [b["bar"] for b in rep["c43"]["bars"]] == list(dp.QUANT_BARS)
    acc = _bar(rep, "accuracy_diff_ci_lower_min")
    assert acc["n"] == 40
    assert acc["ci_low"] is not None
    assert acc["ci_high"] is not None
    assert acc["point"] == pytest.approx(0.9 - 0.5)
    assert _bar(rep, "gap_closure_min")["value"] == pytest.approx(0.4 / 0.45)
    assert rep["c42"]["side"] == "test"
    assert rep["c42"]["pool"] == R.HEADLINE


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
        preds["stock"] += _rows(recs, len(recs), 0.9, 0.2)
        preds["student"] += _rows(recs, 0, 0.9, 0.1)  # all wrong: must not touch the gate
    rep = _build(clean + overlap + evaluated, preds)
    assert rep["sources"][OVERLAP]["status"] == "overlapping"
    assert rep["sources"][OVERLAP]["checked_as"] == "jev-bench:banking77"
    assert rep["sources"][EVAL_ONLY]["status"] == "stock-evaluated"
    assert rep["sources"][CLEAN]["headline"] is True
    pools = rep["sides"]["test"]["pools"]
    assert pools[R.HEADLINE]["items"] == 40
    assert pools[R.OVERLAPPING]["items"] == 10
    assert pools[R.STOCK_EVALUATED]["items"] == 6
    assert pools[R.OVERLAPPING]["stats"]["student.accuracy"]["value"] == 0.0
    assert rep["c42"]["result"] == R.PASSED
    assert _bar(rep, "accuracy_diff_ci_lower_min")["n"] == 40
    assert rep["c17"]["n_items"] == 40


def test_typed_decisions_agreement_is_reported_not_gated():
    clean = [_rec(i) for i in range(40)]
    typed = [_rec(i, TYPED, "model-generated") for i in range(12)]
    preds = _good(clean)
    preds["clef"] += _rows(typed, 12, 0.9, 0.2)
    preds["stock"] += _rows(typed, 12, 0.9, 0.2)
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
    preds["student"] = [dict(r) for r in preds["stock"]]  # identical to stock, item by item
    rep = _build(recs, preds)
    diff = rep["sides"]["test"]["pools"][R.HEADLINE]["stats"]["accuracy_diff_student_stock"]
    # Unpaired resampling would give a spread; paired draws give exactly zero every time.
    assert diff["ci_low"] == 0.0
    assert diff["ci_high"] == 0.0
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


def test_higher_accuracy_with_higher_ece_than_stock_is_a_regression():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student"] = _rows(recs, 30, 1.0, 0.0)  # 0.75 accuracy, every answer at p = 1
    doc = dp.example_doc()
    doc["bars"]["test"]["ece_max"] = 1.0
    doc["bars"]["test"]["ece_no_worse_than_stock"] = False
    rep = _build(recs, preds, prereg=dp.validate(doc))
    assert rep["regression"] is True
    assert rep["sides"]["test"]["pools"][R.HEADLINE]["regression"]["regression"] is True
    assert rep["c42"]["result"] == R.MISSED
    assert "regression" in rep["c42"]["missed"]
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
    assert rep["c43"]["gated"] is False  # the #18-shape table; c17 is the run's verdict
    assert rep["result"] == rep["c17"]["result"]


def test_latency_and_memory_are_reported_for_all_three_on_the_same_box():
    recs = [_rec(i) for i in range(40)]
    rep = _build(recs, _good(recs))
    table = rep["runtime"]
    assert set(table["models"]) == set(R.MODELS)
    for row in table["models"].values():
        assert row["latency_median_s"] is not None
        assert row["latency_p95_s"] is not None
        assert row["peak_memory_mib"] == 1000.0
        assert row["memory_probe"] == "proc-rss"
    assert table["same_box"] is True
    assert _bar(rep, "latency_ratio_max")["value"] == pytest.approx(0.1)
    split = _build(recs, _good(recs), runtime=_runtime(hosts=("a", "b", "a", "a")))
    assert split["runtime"]["same_box"] is False
    slow = _build(recs, _good(recs), runtime=_runtime(student_s=0.5))
    assert "latency_ratio_max" not in slow["c42"]["missed"]  # latency is never pass/fail
    assert _bar(slow, "latency_ratio_max")["result"] == R.REPORTED
    assert slow["result"] == R.PASSED


def test_each_models_calibration_is_applied_and_recorded():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    cal = {"temperature_by_kind": {"choice": 5.0}, "vector_by_kind": {}}
    raw = _build(recs, preds)
    hot = _build(recs, preds, cal={"clef": None, "stock": None, "student": cal})
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
        cal={"clef": None, "stock": None, "student": None, "clef_calibrated": cal},
    )
    stats = rep["sides"]["test"]["pools"][R.HEADLINE]["stats"]
    assert "clef_calibrated.ece" in stats
    assert "clef.ece" in stats
    assert stats["clef_calibrated.accuracy"]["value"] == stats["clef.accuracy"]["value"]


def test_held_out_is_reported_but_not_gated():
    test = [_rec(i) for i in range(40)]
    held = [_rec(100 + i) for i in range(20)]
    preds = _good(test)
    preds["clef"] += _rows(held, 20, 0.9, 0.2)
    preds["stock"] += _rows(held, 20, 0.9, 0.2)
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
    assert dg["fitted"] is True
    assert dg["gated"] is False
    assert dg["reason"]
    assert dg["thresholds_by_kind"] == GATE["thresholds_by_kind"]
    choice = dg["sides"]["test"]["choice"]
    # floor 0.9 answers the 36 confident right ones; the 3 hesitant and 1 unanswered abstain
    assert choice["coverage"]["value"] == pytest.approx(36 / 40)
    assert choice["selective_accuracy"]["value"] == 1.0
    assert choice["coverage"]["n"] == 40
    assert choice["coverage"]["ci_low"] is not None
    assert choice["abstained"] == 4
    assert choice["unanswered"] == 1
    assert dg["sides"]["test"]["all"]["coverage"]["value"] == pytest.approx(36 / 40)
    assert dg["sides"]["test"]["noul"]["coverage"]["n"] == 0
    # reported only: the gate never changes the c42 verdict
    assert rep["c42"] == _build(recs, preds)["c42"]
    assert "decision_gate" not in rep["c42"]["missed"]


def test_report_without_a_fitted_gate_says_so():
    recs = [_rec(i) for i in range(40)]
    dg = _build(recs, _good(recs))["decision_gate"]
    assert dg["fitted"] is False
    assert dg["sides"] == {}
    assert dg["reason"]


# --------------------------------------------------------------------- c17 (t14)


def _c17(rep, build, name):
    return next(b for b in rep["c17"]["builds"][build]["bars"] if b["bar"] == name)


def test_every_c17_bar_is_computed_for_bf16_and_quant_with_paired_ci_and_n():
    """AC3: report.py computes each c17 bar for both the bf16 student and the shipped quant
    with paired bootstrap CIs and cites the hashed prereg."""
    recs = [_rec(i) for i in range(40)]
    rep = _build(recs, _good(recs))
    c17 = rep["c17"]
    assert c17["result"] == R.PASSED, c17["missed"]
    assert set(c17["builds"]) == {"bf16", "quant"}
    assert c17["builds"]["bf16"]["model"] == R.STUDENT_BF16
    assert c17["builds"]["quant"]["model"] == R.STUDENT
    assert [b["bar"] for b in c17["builds"]["bf16"]["bars"]] == list(dp.C17_TOP1_BARS)
    assert [b["bar"] for b in c17["builds"]["quant"]["bars"]] == [
        *dp.C17_TOP1_BARS,
        *dp.C17_QUANT_ONLY_BARS,
    ]
    for build in ("bf16", "quant"):
        over = _c17(rep, build, "top1_over_stock_min_points")
        assert over["value"] == pytest.approx(100 * (0.9 - 0.5))
        assert over["n"] == 40
        assert over["ci_low"] is not None
        assert over["ci_high"] is not None
        assert over["threshold"] == dp.C17_STOCK_MARGIN_POINTS
        under = _c17(rep, build, "top1_under_clef_w4a16_max_points")
        assert under["value"] == pytest.approx(100 * (0.95 - 0.9))
        assert under["ci_low"] is not None
    agree = _c17(rep, "quant", "quant_vs_bf16_agreement_min")
    assert agree["value"] == 1.0
    assert agree["n"] == 40
    weight = _c17(rep, "quant", "weight_bytes_max")
    assert weight["value"] == BUNDLE["weight_bytes"]
    assert weight["result"] == R.PASSED
    assert c17["prereg_sha256"] == PREREG_SHA
    assert PREREG_SHA in c17["cites"]
    assert rep["prereg"]["sha256"] == PREREG_SHA
    assert c17["side"] == "test"
    assert c17["pool"] == R.HEADLINE
    assert c17["n_items"] == 40
    assert rep["headline_n"] == {"records": 40, "items": 40}


def test_c17_follows_the_locked_prereg_and_a_tightened_bar_misses():
    """AC3: each c17 bar is read from bars.c17 of the locked pre-registration."""
    recs = [_rec(i) for i in range(40)]
    doc = dp.example_doc()
    doc["bars"]["c17"]["bf16"]["top1_over_stock_min_points"] = 45.0
    doc["bars"]["c17"]["quant"]["top1_under_clef_w4a16_max_points"] = 1.0
    rep = _build(recs, _good(recs), prereg=dp.validate(doc))
    assert rep["c17"]["missed"] == [
        "bf16.top1_over_stock_min_points",
        "quant.top1_under_clef_w4a16_max_points",
    ]
    assert rep["result"] == R.MISSED
    assert _c17(rep, "bf16", "top1_over_stock_min_points")["threshold"] == 45.0


def test_a_quant_that_disagrees_with_its_bf16_misses_agreement_on_paired_draws():
    """AC3: quant-vs-bf16 agreement is a c17 bar with a paired CI (same draws as the rest)."""
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student_bf16"] = _rows(recs, 38, 0.97, 0.45)  # 2 more right than the quant
    rep = _build(recs, preds)
    agree = _c17(rep, "quant", "quant_vs_bf16_agreement_min")
    assert agree["value"] == pytest.approx(38 / 40)
    assert agree["result"] == R.MISSED
    assert agree["ci_low"] <= agree["value"] <= agree["ci_high"]
    assert "quant.quant_vs_bf16_agreement_min" in rep["c17"]["missed"]
    assert rep["c17"]["builds"]["bf16"]["result"] == R.PASSED


def test_weight_bytes_must_be_strictly_below_the_bar_and_attested():
    """AC3/c17: '< 1 GB' means < 10^9 bytes of attested weight files."""
    recs = [_rec(i) for i in range(40)]
    over = _build(recs, _good(recs), bundle={**BUNDLE, "weight_bytes": 10**9})
    assert _c17(over, "quant", "weight_bytes_max")["result"] == R.MISSED
    missing = _build(recs, _good(recs), bundle=None)
    row = _c17(missing, "quant", "weight_bytes_max")
    assert row["result"] == R.MISSED
    assert row["reason"]


def test_a_bf16_regression_fails_c17_even_when_the_quant_is_fine():
    recs = [_rec(i) for i in range(40)]
    preds = _good(recs)
    preds["student_bf16"] = _rows(recs, 30, 1.0, 0.0)  # above stock, ECE 0.25 > 0.1
    rep = _build(recs, preds)
    assert rep["c17"]["builds"]["bf16"]["regression"]["regression"] is True
    assert rep["c17"]["builds"]["quant"]["regression"]["regression"] is False
    assert "bf16.regression" in rep["c17"]["missed"]
    assert rep["regression"] is True
    assert rep["result"] == R.MISSED


def test_the_headline_counts_only_approved_human_gold_test_records():
    """AC2: an unapproved human-gold record never counts toward the bars (reported apart)."""
    recs = [_rec(i) for i in range(40)]
    unapproved = [_rec(100 + i) for i in range(5)]
    preds = _good(recs)
    for name in R.MODELS:
        preds[name] += _rows(unapproved, 0, 0.9, 0.1)  # all wrong: must not move the bars
    rep = _build(recs + unapproved, preds, approved={r.id for r in recs})
    pools = rep["sides"]["test"]["pools"]
    assert pools[R.HEADLINE]["records"] == 40
    assert pools[R.UNAPPROVED]["records"] == 5
    assert rep["c17"]["n_records"] == 40
    assert rep["c17"]["result"] == R.PASSED


def test_probe_table_and_latency_carry_no_pass_fail():
    """AC3: the Clef probe-set table and latency median/p95 carry no pass/fail."""
    recs = [_rec(i) for i in range(40)]
    probe = [_rec(i, "private/clef-probe") for i in range(8)]
    preds = _good(recs)
    for name in R.MODELS:
        preds[name] += _rows(probe, 4 if name == "student" else 8, 0.9, 0.1)
    rep = _build(recs, preds, probe=probe)
    table = rep["probe"]
    assert table["measured"] is True
    assert table["gated"] is False
    assert table["items"] == 8
    assert table["stats"]["student.accuracy"]["value"] == 0.5
    assert table["stats"]["clef.accuracy"]["value"] == 1.0
    assert "sources" not in table  # the probe set is private
    assert "result" not in table
    assert rep["result"] == R.PASSED  # a worse probe score gates nothing
    assert "private/clef-probe" not in rep["sources"]  # never mixed into the sealed tables
    for build in ("bf16", "quant"):
        lat = rep["c17"]["builds"][build]["latency"]
        assert lat["median_s"] is not None
        assert lat["p95_s"] is not None
        assert lat["gated"] is False
        assert "result" not in lat
    assert rep["runtime"]["gated"] is False
    assert all("result" not in row for row in rep["runtime"]["models"].values())
    assert _build(recs, _good(recs))["probe"]["measured"] is False
