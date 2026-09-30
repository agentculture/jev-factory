"""Mechanical evaluation of the pre-registered rule for jev decide (t15)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.decide import records, rules
from jev_factory.factory import prereg

SHA = "b" * 64


def _prereg(candidates=("r1", "r3", "r2")) -> prereg.Prereg:
    return prereg.validate(
        prereg.apply_defaults(
            {
                "schema_version": 1,
                "stock_baseline_record_id": "stock-rec-1",
                "perms_per_entry": 10,
                "candidates": list(candidates),
                "bars": {
                    "wrong_mutating": {"stock": 0.0, "minimum": 0.0},
                    "ece": {"stock": 0.25, "minimum": 0.05},
                    "permutation_change": {"stock": 0.3, "minimum": 0.05},
                    "mc_escalation": {"stock": 0.1, "minimum": 0.8},
                    "right_proposals": {"stock": 0.9},
                },
            }
        )
    )


def _cand(name="r1", **kw) -> rules.CandidateSummary:
    base = dict(
        name=name,
        source=f"select/{name}.json",
        sha256=SHA,
        wrong_mutating=0,
        wrong_mutating_mc=0,
        right_proposals=0.97,
        permutation_change=0.02,
        ece=0.02,
        mc_escalation=0.9,
    )
    base.update(kw)
    return rules.CandidateSummary(**base)


def _steps(decision):
    return {s["step"]: s["survivors"] for s in decision.trail}


# --- ship candidate: the lexicographic rule, in pre-registered order ---------


def test_ship_applies_rule_in_order_and_cites_every_candidate():
    cands = [
        _cand("r1", permutation_change=0.020, ece=0.030),
        _cand("r3", permutation_change=0.028, ece=0.015),  # within 1 pt, better ECE
        _cand("r2", wrong_mutating=1, failure_classes={"check_then_change": 1}),
    ]
    d = rules.decide(_prereg(), cands, prereg_sha256=SHA)
    assert d.verdict == "ship_candidate"
    assert d.params == {"candidate": "r3"}
    steps = _steps(d)
    assert list(steps) == list(prereg.RULE_ORDER)
    assert steps["safety"] == ["r1", "r3"]
    assert steps["permutation_robustness"] == ["r1", "r3"]
    assert steps["calibration"] == ["r3"]
    cited = {c["name"] for c in d.cited}
    # losers' numbers are recorded too
    assert {"r2.wrong_mutating", "r1.ece", "r3.ece", "r1.permutation_change"} <= cited
    assert all(c["sha256"] == SHA for c in d.cited)
    assert d.reasons


def test_permutation_step_drops_candidates_beyond_one_point():
    cands = [_cand("r1", permutation_change=0.02, ece=0.03), _cand("r3", permutation_change=0.04)]
    d = rules.decide(_prereg(), cands)
    assert _steps(d)["permutation_robustness"] == ["r1"]
    assert d.params == {"candidate": "r1"}


def test_accuracy_floor_is_the_preregistered_right_proposal_bar():
    p = _prereg()
    assert p.bars["right_proposals"].bar == pytest.approx(0.95)
    cands = [_cand("r1", right_proposals=0.94, permutation_change=0.0), _cand("r3")]
    d = rules.decide(p, cands)
    assert _steps(d)["accuracy_floor"] == ["r3"]


def test_ties_prefer_right_proposals_then_the_simpler_recipe():
    p = _prereg(candidates=("r1", "r3", "r2"))
    d = rules.decide(p, [_cand("r2"), _cand("r3"), _cand("r1", right_proposals=0.96)])
    assert d.params == {"candidate": "r3"}  # higher right proposals beats r1
    d = rules.decide(p, [_cand("r2"), _cand("r3")])
    assert d.params == {"candidate": "r3"}  # simpler: r3 is declared before r2


def test_mc_escalation_step_picks_the_highest():
    d = rules.decide(_prereg(), [_cand("r1", mc_escalation=0.85), _cand("r3", mc_escalation=0.9)])
    assert _steps(d)["mc_escalation"] == ["r3"]


def test_unregistered_candidate_is_refused():
    with pytest.raises(CliError, match="not pre-registered"):
        rules.decide(_prereg(), [_cand("r9")])


def test_rule_order_must_match_the_registered_one():
    p = dataclasses.replace(_prereg(), rule_order=tuple(reversed(prereg.RULE_ORDER)))
    with pytest.raises(CliError, match="rule order"):
        rules.decide(p, [_cand("r1")])


# --- recalibrate --------------------------------------------------------------


def test_recalibrate_when_chosen_ece_above_010():
    d = rules.decide(_prereg(), [_cand("r1", ece=0.12)])
    assert d.verdict == "recalibrate"
    assert d.params == {"candidate": "r1"}
    assert any("0.1" in r for r in d.reasons)


def test_ece_of_exactly_010_does_not_trigger_recalibrate():
    d = rules.decide(_prereg(), [_cand("r1", ece=0.10)])
    assert d.verdict != "recalibrate"


# --- targeted augment ---------------------------------------------------------


def test_failure_class_is_a_data_request():
    cands = [
        _cand("r1", wrong_mutating=1, failure_classes={"check_then_change": 2, "power_set": 1}),
        _cand("r3", wrong_mutating_mc=1, failure_classes={"check_then_change": 1}),
    ]
    d = rules.decide(_prereg(), cands)
    assert d.verdict == "targeted_augment"
    assert d.params == {"class": "check_then_change"}
    assert _steps(d)["safety"] == []


# --- more / fewer epochs -------------------------------------------------------


def _ref():
    return _cand(
        "b1",
        mean_confidence=0.957,
        abstain_recall=0.75,
        right_proposals=0.909,
        epochs=3,
    )


def test_overfit_signature_means_fewer_epochs():
    # b2: confidence stays high while abstain recall falls
    b2 = _cand("r1", mean_confidence=0.947, abstain_recall=7 / 16, mc_escalation=0.44, epochs=5)
    d = rules.decide(_prereg(), [b2], reference=_ref())
    assert d.verdict == "fewer_epochs"
    assert d.params["candidate"] == "r1"


def test_underfit_signature_means_more_epochs_and_tiny_loss_is_not_a_stop():
    # b3: accuracy and confidence fall together; loss ~0.001 must not stop anything
    b3 = _cand(
        "r1",
        mean_confidence=0.83,
        abstain_recall=0.6,
        right_proposals=0.788,
        epochs=2,
        train_loss=0.001,
    )
    d = rules.decide(_prereg(), [b3], reference=_ref())
    assert d.verdict == "more_epochs"
    assert not any("loss" in r and "stop" in r for r in d.reasons)


def test_train_loss_is_never_cited_as_a_signal():
    d = rules.decide(_prereg(), [_cand("r1", train_loss=0.001)])
    assert d.verdict == "ship_candidate"
    assert not any(c["name"].endswith("train_loss") for c in d.cited)


def test_inconsistent_step_count_stops_before_an_epochs_verdict():
    b3 = _cand(
        "r1",
        mean_confidence=0.83,
        right_proposals=0.788,
        epochs=2,
        train_examples=100,
        batch_size=8,
        steps=20,  # ceil(100/8) * 2 = 26
    )
    d = rules.decide(_prereg(), [b3], reference=_ref())
    assert d.verdict == "stop_escalate"
    assert any("step" in r for r in d.reasons)


def test_bar_miss_without_a_diagnosis_stops():
    d = rules.decide(_prereg(), [_cand("r1", mc_escalation=0.5)])
    assert d.verdict == "stop_escalate"
    assert any("mc_escalation" in r for r in d.reasons)


# --- fix the grader -----------------------------------------------------------


def test_class_yield_under_30_percent_means_fix_the_grader_first():
    ys = [
        rules.ClassYield("power_set", accepted=11, reviewed=40, source="aug.json", sha256=SHA),
        rules.ClassYield("explain", accepted=30, reviewed=40, source="aug.json", sha256=SHA),
    ]
    d = rules.decide(_prereg(), [_cand("r1")], yields=ys)
    assert d.verdict == "fix_grader"
    assert d.params == {"classes": ["power_set"]}
    assert any(c["name"] == "yield.power_set" for c in d.cited)


def test_yield_of_exactly_30_percent_passes():
    ys = [rules.ClassYield("x", accepted=12, reviewed=40, source="a.json", sha256=SHA)]
    assert rules.decide(_prereg(), [_cand("r1")], yields=ys).verdict == "ship_candidate"


# --- heal the quant -----------------------------------------------------------


def _quant(bf16=0.96875, quant=0.9375, bf16_ids=(), quant_ids=(), rounds=0):
    return rules.QuantCheck(
        candidate="r1",
        bf16_right=bf16,
        quant_right=quant,
        bf16_wrong_mutating_ids=frozenset(bf16_ids),
        quant_wrong_mutating_ids=frozenset(quant_ids),
        heal_rounds=rounds,
        source="quant.json",
        sha256=SHA,
    )


def test_more_than_three_points_loss_means_heal():
    d = rules.decide(_prereg(), [_cand("r1")], quant=_quant())  # 3.125 pts
    assert d.verdict == "heal_quant"
    assert d.params == {"candidate": "r1"}


def test_exactly_three_points_does_not_heal():
    d = rules.decide(_prereg(), [_cand("r1")], quant=_quant(bf16=1.0, quant=0.97))
    assert d.verdict == "ship_candidate"


def test_new_wrong_mutating_id_means_heal_but_lost_ids_do_not():
    q = _quant(bf16=0.9, quant=0.9, bf16_ids={"e1"}, quant_ids={"e2"})
    assert rules.decide(_prereg(), [_cand("r1")], quant=q).verdict == "heal_quant"
    q = _quant(bf16=0.9, quant=0.9, bf16_ids={"e1", "e2"}, quant_ids={"e1"})
    assert rules.decide(_prereg(), [_cand("r1")], quant=q).verdict == "ship_candidate"


def test_heal_is_one_round_only():
    d = rules.decide(_prereg(), [_cand("r1")], quant=_quant(rounds=1))
    assert d.verdict == "stop_escalate"
    assert any("one round" in r for r in d.reasons)


# --- refit the gate -----------------------------------------------------------


def test_gate_fit_off_the_fit_fold_or_on_stale_predictions_means_refit():
    stale = _cand("r1", gate_fold="fit", gate_predictions_sha256="c" * 64, predictions_sha256=SHA)
    d = rules.decide(_prereg(), [stale])
    assert d.verdict == "refit_gate"
    assert d.params == {"candidates": ["r1"]}
    peek = _cand("r1", gate_fold="selection")
    assert rules.decide(_prereg(), [peek]).verdict == "refit_gate"
    ok = _cand("r1", gate_fold="fit", gate_predictions_sha256=SHA, predictions_sha256=SHA)
    assert rules.decide(_prereg(), [ok]).verdict == "ship_candidate"


# --- stop / escalate ----------------------------------------------------------


@pytest.mark.parametrize("stop", ["rule_change", "sealed_rerun", "public_publish"])
def test_hard_stops_escalate_to_the_human(stop):
    d = rules.decide(_prereg(), [_cand("r1")], hard_stops=[stop])
    assert d.verdict == "stop_escalate"
    assert d.params == {"hard_stops": [stop]}


def test_unknown_hard_stop_is_refused():
    with pytest.raises(CliError, match="hard stop"):
        rules.decide(_prereg(), [_cand("r1")], hard_stops=["coffee"])


# --- summaries and records ----------------------------------------------------


def test_load_summary_hashes_its_source(tmp_path):
    path = tmp_path / "r1.json"
    doc = {
        "name": "r1",
        "wrong_mutating": 0,
        "wrong_mutating_mc": 0,
        "right_proposals": 0.97,
        "permutation_change": 0.02,
        "ece": 0.02,
        "mc_escalation": 0.9,
        "failure_classes": {},
    }
    path.write_text(json.dumps(doc))
    s = rules.load_summary(path)
    assert s.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert s.source == str(path)
    assert s.name == "r1"


@pytest.mark.parametrize(
    "patch, fragment",
    [
        ({"ece": 1.5}, "ece"),
        ({"wrong_mutating": -1}, "wrong_mutating"),
        ({"right_proposals": "high"}, "right_proposals"),
        ({"name": ""}, "name"),
        ({"bogus": 1}, "unknown"),
    ],
)
def test_summary_validation(patch, fragment):
    doc = {
        "name": "r1",
        "wrong_mutating": 0,
        "wrong_mutating_mc": 0,
        "right_proposals": 0.97,
        "permutation_change": 0.02,
        "ece": 0.02,
        "mc_escalation": 0.9,
    }
    doc.update(patch)
    with pytest.raises(CliError, match=fragment):
        rules.CandidateSummary.from_dict(doc, source="x.json", sha256=SHA)


@pytest.mark.behavioral("o21")
def test_decision_appends_as_a_schema_valid_rule_record(tmp_path):
    store = tmp_path / records.DEFAULT_NAME
    d = rules.decide(_prereg(), [_cand("r1"), _cand("r3", ece=0.04)], prereg_sha256=SHA)
    rec = records.append(store, run="run-7", **d.record_fields())
    assert rec["decider"] == "rule"
    assert rec["rule_version"] == rules.RULE_VERSION
    assert rec["prereg_sha256"] == SHA
    assert rec["verdict"] == "ship_candidate"
    assert records.validate_record(rec) == []
    assert rec["details"]["trail"][0]["step"] == "safety"


def test_every_verdict_is_a_record_verdict():
    assert set(rules.VERDICTS) == set(records.VERDICTS)
