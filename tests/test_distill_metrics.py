"""Distill assessment metrics: hand-computed fixtures and seeded-CI determinism."""

from __future__ import annotations

import math

import pytest

from jev_factory.core import metrics as core_metrics
from jev_factory.distill import metrics as dm

THRESHOLD = 0.65


def _items():
    return [
        dm.choice_item([0.8, 0.2], gold_index=0),  # right
        dm.choice_item([0.4, 0.6], gold_index=0),  # wrong, conf 0.6 (< threshold)
        dm.score_item([0.1, 0.2, 0.7], gold_level=2),  # right
        dm.score_item([0.1, 0.8, 0.1], gold_level=0),  # wrong, confident
        dm.noul_item(0.9, gold_true=True),  # right
        dm.noul_item(0.3, gold_true=True),  # wrong, confident (P(false)=0.7)
    ]


def _assess(items=None, **kw):
    kw.setdefault("confident_wrong_threshold", THRESHOLD)
    kw.setdefault("resamples", 50)
    return dm.assess(items if items is not None else _items(), **kw)


def _value(block, key):
    return block[key]["value"]


def test_accuracy_per_kind():
    acc = _assess()["accuracy"]
    assert _value(acc, "choice") == 0.5
    assert _value(acc, "score") == 0.5
    assert _value(acc, "noul") == 0.5
    assert _value(acc, "all") == 0.5
    assert acc["choice"]["n"] == 2


def test_score_scored_on_level_index_and_distribution():
    item = dm.score_item([0.1, 0.2, 0.7], gold_level=2, gold_probs=[0.0, 0.3, 0.7])
    result = _assess([item])
    assert _value(result["accuracy"], "score") == 1.0  # argmax level == gold level index
    assert _value(result["log_loss"], "score") == pytest.approx(-math.log(0.7))
    # full distribution: JSD against the 3-level gold distribution, not just the index
    assert _value(result["jsd_gold"], "score") == pytest.approx(
        dm.jensen_shannon([0.1, 0.2, 0.7], [0.0, 0.3, 0.7])
    )
    assert _value(result["jsd_gold"], "score") > 0.0


def test_noul_scored_on_p_true():
    item = dm.noul_item(0.9, gold_true=True)
    assert item.probs == pytest.approx((0.1, 0.9))
    assert item.gold_index == 1
    wrong = dm.noul_item(0.3, gold_true=True)
    assert dm.argmax(wrong.probs) == 0
    assert _value(_assess([item, wrong])["accuracy"], "noul") == 0.5
    low = dm.noul_item(0.2, gold_true=False)
    assert low.gold_index == 0


def test_noul_gold_probability_true_feeds_jsd():
    item = dm.noul_item(0.9, gold_true=True, gold_probability_true=0.5)
    assert item.gold_probs == pytest.approx((0.5, 0.5))
    got = _value(_assess([item])["jsd_gold"], "noul")
    assert got == pytest.approx(dm.jensen_shannon([0.1, 0.9], [0.5, 0.5]))


def test_macro_f1_hand_computed():
    # choice: preds [0, 1], gold [0, 0] -> F1(class0)=2/3, F1(class1)=0 -> macro 1/3
    f1 = _assess()["macro_f1"]
    assert _value(f1, "choice") == pytest.approx(1 / 3)
    # score: preds [2, 1], gold [2, 0] -> class2 F1=1, class1 F1=0, class0 F1=0 -> 1/3
    assert _value(f1, "score") == pytest.approx(1 / 3)
    # noul: preds [1, 0], gold [1, 1] -> class1 F1=2/3, class0 F1=0 -> 1/3
    assert _value(f1, "noul") == pytest.approx(1 / 3)


def test_jsd_known_values():
    assert dm.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)
    assert dm.jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    # 0.5*log2(1/0.75) + 0.5*(0.5*log2(0.5/0.75) + 0.5*log2(2)) = 0.3112778
    assert dm.jensen_shannon([1, 0], [0.5, 0.5]) == pytest.approx(0.3112778, abs=1e-6)


def test_jsd_to_gold_uses_one_hot_when_no_gold_probs():
    item = dm.choice_item([0.5, 0.5], gold_index=0)
    assert _value(_assess([item])["jsd_gold"], "choice") == pytest.approx(0.3112778, abs=1e-6)


def test_jsd_to_teacher_only_over_items_with_teacher():
    items = [
        dm.choice_item([1.0, 0.0], gold_index=0, teacher_probs=[0.0, 1.0]),
        dm.choice_item([0.5, 0.5], gold_index=0, teacher_probs=[0.5, 0.5]),
        dm.choice_item([0.5, 0.5], gold_index=0),  # no teacher: excluded
    ]
    block = _assess(items)["jsd_teacher"]["choice"]
    assert block["n"] == 2
    assert block["value"] == pytest.approx(0.5)


def test_confident_wrong_rate_uses_preregistered_threshold():
    cw = _assess()["confident_wrong"]
    assert _value(cw, "all") == pytest.approx(2 / 6)
    assert _value(cw, "choice") == 0.0
    assert _value(cw, "score") == 0.5
    assert _value(cw, "noul") == 0.5
    lower = _assess(confident_wrong_threshold=0.55)["confident_wrong"]
    assert _value(lower, "choice") == 0.5  # the 0.6-confident wrong choice now counts
    assert _assess()["confident_wrong_threshold"] == THRESHOLD


def test_threshold_is_required():
    items = _items()
    with pytest.raises(TypeError):
        dm.assess(items)  # type: ignore[call-arg]


def test_log_loss_hand_computed():
    ll = _assess()["log_loss"]
    assert _value(ll, "choice") == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert _value(ll, "noul") == pytest.approx(-(math.log(0.9) + math.log(0.3)) / 2)
    assert _value(ll, "score") == pytest.approx(-(math.log(0.7) + math.log(0.1)) / 2)


def test_log_loss_clips_zero_probability():
    item = dm.choice_item([1.0, 0.0], gold_index=1)
    value = _value(_assess([item])["log_loss"], "choice")
    assert math.isfinite(value)
    assert value > 20


def test_brier_hand_computed():
    br = _assess()["brier"]
    # A: .2^2+.2^2=.08 ; B: .6^2+.6^2=.72
    assert _value(br, "choice") == pytest.approx(0.4)
    # [.1,.2,.7] gold 2: .01+.04+.09=.14 ; [.1,.8,.1] gold 0: .81+.64+.01=1.46
    assert _value(br, "score") == pytest.approx(0.8)
    # [.1,.9] gold 1: .02 ; [.7,.3] gold 1: .49+.49=.98
    assert _value(br, "noul") == pytest.approx(0.5)


def test_ci_identical_across_runs_with_same_seed():
    first = _assess(seed=7, resamples=200)
    second = _assess(seed=7, resamples=200)
    assert first == second
    assert first["accuracy"]["all"]["ci_low"] is not None


def test_ci_matches_core_bootstrap_ci():
    items = _items()
    expected = core_metrics.bootstrap_ci(
        items,
        lambda sample: sum(1 for i in sample if dm.argmax(i.probs) == i.gold_index) / len(sample),
        seed=3,
        resamples=100,
    )
    got = _assess(items, seed=3, resamples=100)["accuracy"]["all"]
    assert got == expected


def test_ci_differs_with_seed():
    a = _assess(seed=1, resamples=300)["log_loss"]["all"]
    b = _assess(seed=2, resamples=300)["log_loss"]["all"]
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


def test_empty_kind_reports_null_not_crash():
    result = _assess([dm.choice_item([0.5, 0.5], gold_index=0)])
    assert result["accuracy"]["noul"]["n"] == 0
    assert result["accuracy"]["noul"]["value"] is None


def test_invalid_distribution_rejected():
    with pytest.raises(ValueError):
        dm.choice_item([0.5, 0.4], gold_index=0)
    with pytest.raises(ValueError):
        dm.choice_item([1.2, -0.2], gold_index=0)
    with pytest.raises(ValueError):
        dm.choice_item([0.5, 0.5], gold_index=2)
    with pytest.raises(ValueError):
        dm.choice_item([0.5, 0.5], gold_index=0, teacher_probs=[1.0])
    with pytest.raises(ValueError):
        dm.noul_item(1.5, gold_true=True)


def test_no_peak_memory_computed_here():
    assert "peak_memory" not in _assess()
