"""jev_factory.core.metrics, ported from nvsh's tests/test_lfm_finetune_metrics.py.

Every figure is checked against the same hand-computed thirteen-line fixture
nvsh uses, with nvsh's operations swapped for the toy lamp domain's (the
structure, and so every number, is unchanged). Operation names appear only
as fixture data; the module under test reads read-only/mutating from the
Domain. nvsh's ``test_escalation_agrees_with_bench_compute_escalation`` is
not ported: it compared against ``nvsh.tiers.bench``, which jev-factory does
not have; the tp/fn/fp it checked are asserted directly below.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from jev_factory.core import metrics
from jev_factory.domain.model import (
    ESCALATE_LABEL,
    EXPLAIN_LABEL,
    ArgSpec,
    GroundKind,
    Operation,
)
from tests.fixtures.toy_domain import DOMAIN

ESC = ESCALATE_LABEL
EXP = EXPLAIN_LABEL

# The toy domain's operations, as fixture data only.
RO = "lamp_status"  # read-only, no args (nvsh memory_stats)
RO_ARG = "room_status"  # read-only, room (nvsh service_status)
RO_2 = "list_rooms"  # read-only (nvsh service_logs)
MUT_ROOM = "lamp_on"  # mutating, room (nvsh service_restart)
MUT_SCENE = "set_scene"  # mutating, scene choice (nvsh container_restart / power_set)


def _op(name: str, **args) -> dict:
    return {"operation": name, "args": dict(args)}


def _rec(
    entry_id: str,
    expected: dict,
    outcome: str,
    operation: str | None = None,
    arguments: dict | None = None,
    candidates: dict | None = None,
    tokens: int = 0,
    ttfd_ms: float = 0.0,
    latency_ms: float = 0.0,
    **extra,
) -> dict:
    return {
        "id": entry_id,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": tokens,
        "ttfd_ms": ttfd_ms,
        "latency_ms": latency_ms,
        **extra,
    }


TOKENS = [10, 12, 8, 9, 30, 4, 11, 6, 20, 5, 7, 3, 15]
TTFD = [500.0] + [40.0 + 2 * i for i in range(12)]  # cold 500, warm 40..62


def _fixture_rows() -> list[dict]:
    """Thirteen predictions; the expected figures are worked out in each test."""
    rows = [
        # p1: right proposal
        _rec("p1", _op(RO), "propose", RO, {}, {RO: 0.9, ESC: 0.1}),
        # p2: right read-only operation, wrong arguments
        _rec(
            "p2",
            _op(RO_ARG, room="kitchen"),
            "propose",
            RO_ARG,
            {"room": "hall"},
            {RO_ARG: 0.6, RO_2: 0.4},
        ),
        # p3: a different mutating operation -> wrong mutating (wrong operation)
        _rec(
            "p3",
            _op(MUT_ROOM, room="kitchen"),
            "propose",
            MUT_SCENE,
            {"scene": "bright"},
            {MUT_SCENE: 0.7, MUT_ROOM: 0.3},
        ),
        # p4: the expected mutating operation with wrong arguments
        _rec(
            "p4",
            _op(MUT_SCENE, scene="reading"),
            "propose",
            MUT_SCENE,
            {"scene": "night_light"},
            {MUT_SCENE: 0.8, ESC: 0.2},
        ),
        # p5: the model's own output was unparseable
        _rec(
            "p5",
            _op(RO),
            "invalid",
            candidates={RO: 0.55, ESC: 0.45},
            invalid_reason="malformed",
        ),
        # p6: escalate expected, escalated (TP)
        _rec("p6", {"escalate": True}, "escalate", candidates={ESC: 1.0}),
        # p7: escalate expected, a mutating proposal (FN, FP tool call, wrong mutating)
        _rec(
            "p7",
            {"escalate": True},
            "propose",
            MUT_ROOM,
            {"room": "x"},
            {MUT_ROOM: 0.5, ESC: 0.3, EXP: 0.2},
        ),
        # p8: escalate expected, explained (FN)
        _rec("p8", {"escalate": True}, "explain", candidates={EXP: 0.6, ESC: 0.4}),
        # p9: explain expected, explained
        _rec("p9", {"explain": True}, "explain", candidates={EXP: 0.9, ESC: 0.1}),
        # p10: explain expected, escalated (left out of precision, like bench)
        _rec("p10", {"explain": True}, "escalate", candidates={ESC: 0.7, EXP: 0.3}),
        # p11: explain expected, a read-only proposal (FP tool call, not mutating)
        _rec("p11", {"explain": True}, "propose", RO, {}, {RO: 0.95, EXP: 0.05}),
        # p12: operation expected, escalated (FP escalation)
        _rec("p12", _op(RO), "escalate", candidates={ESC: 0.6, RO: 0.4}),
        # p13: escalate expected, an operation not in the table -> invalid; no distribution
        _rec("p13", {"escalate": True}, "propose", "not_an_operation", {}, None),
    ]
    for row, tokens, ttfd in zip(rows, TOKENS, TTFD):
        row["tokens"] = tokens
        row["ttfd_ms"] = ttfd
        row["latency_ms"] = ttfd + 10.0
    return rows


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def result(tmp_path):
    return metrics.compute(metrics.read_predictions(_write(tmp_path, _fixture_rows())), DOMAIN)


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_schema_fields_are_the_documented_ones():
    assert metrics.FIELDS == (
        "id",
        "expected",
        "outcome",
        "operation",
        "arguments",
        "candidates",
        "tokens",
        "ttfd_ms",
        "latency_ms",
    )
    assert metrics.OUTCOMES == ("propose", "explain", "escalate", "abstain_uncertain", "invalid")


def test_read_predictions_round_trips_the_fixture(tmp_path):
    predictions = metrics.read_predictions(_write(tmp_path, _fixture_rows()))
    assert [p.id for p in predictions] == [f"p{i}" for i in range(1, 14)]
    assert predictions[1].arguments == {"room": "hall"}
    assert predictions[12].candidates is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda row: row.pop("tokens"), "tokens"),
        (lambda row: row.update(outcome="abstain"), "outcome"),
        (lambda row: row.update(expected={"escalate": True, "explain": True}), "expected"),
        (lambda row: row.update(candidates={RO: 0.5, ESC: 0.2}), "sum"),
        (lambda row: row.update(candidates={RO: 1.2, ESC: -0.2}), "candidates"),
        (lambda row: row.update(tokens=-1), "tokens"),
        (lambda row: row.update(operation=None), "operation"),
    ],
)
def test_read_predictions_rejects_bad_records_with_the_line(tmp_path, mutate, fragment):
    rows = _fixture_rows()
    mutate(rows[0])
    path = _write(tmp_path, rows)
    with pytest.raises(metrics.MetricsError) as info:
        metrics.read_predictions(path)
    assert "line 1" in str(info.value)
    assert fragment in str(info.value)


def test_read_predictions_rejects_unparseable_json_and_duplicate_ids(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_fixture_rows()[0]) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(metrics.MetricsError, match="line 2"):
        metrics.read_predictions(path)
    rows = _fixture_rows()
    rows[1]["id"] = "p1"
    duplicated = _write(tmp_path, rows)
    with pytest.raises(metrics.MetricsError, match="duplicate"):
        metrics.read_predictions(duplicated)


# ---------------------------------------------------------------------------
# Decision metrics on the fixture
# ---------------------------------------------------------------------------


def test_right_proposals_n_of_n_and_percent(result):
    # operation-expected: p1 p2 p3 p4 p5 p12 -> N=6; only p1 is right
    right_proposals = dict(result["right_proposals"])
    ci = right_proposals.pop("ci")
    assert right_proposals == {"n": 1, "N": 6, "percent": pytest.approx(100 / 6)}
    assert ci["n"] == 6
    assert ci["value"] == pytest.approx(1 / 6)


def test_escalation_recall_and_precision(result):
    # escalate-expected: p6 p7 p8 p13 -> TP=1 (p6), FN=3; FP=1 (p12, operation-expected);
    # p10 escalated on an explain entry and is reported separately, as in bench.
    escalation = result["escalation"]
    assert (escalation["tp"], escalation["fn"], escalation["fp"]) == (1, 3, 1)
    assert escalation["recall"] == pytest.approx(0.25)
    assert escalation["precision"] == pytest.approx(0.5)
    assert escalation["escalated_on_explain"] == 1


def test_abstain_is_the_escalate_outcome_with_strict_precision(result):
    # abstention precision is strict (an escalation on an explain entry is a false
    # abstention too): p6 / (p6 + p12 + p10) = 1/3; bench's figure stays in "escalation".
    assert result["abstention"]["recall"] == result["escalation"]["recall"]
    assert result["abstention"]["precision"] == pytest.approx(1 / 3)
    assert result["escalation"]["precision_strict"] == pytest.approx(1 / 3)
    assert result["escalation"]["precision"] == pytest.approx(0.5)


def test_false_positive_tool_calls_over_explain_and_escalate_items(result):
    # explain/escalate-expected: p6 p7 p8 p13 p9 p10 p11 -> N=7; proposals: p7, p11
    # (p13's unknown operation is an invalid output, counted there instead)
    false_calls = dict(result["false_positive_tool_calls"])
    ci = false_calls.pop("ci")
    assert false_calls == {"n": 2, "N": 7, "rate": pytest.approx(2 / 7)}
    assert ci["n"] == 7
    assert ci["value"] == pytest.approx(2 / 7)


def test_wrong_mutating_splits_operation_and_arguments(result):
    # wrong operation (bench's definition): p3 (other mutating op), p7 (on escalate);
    # wrong arguments of the expected mutating operation: p4.  p2/p11 are read-only.
    assert result["wrong_mutating"] == {"wrong_operation": 2, "wrong_arguments": 1, "total": 3}


def test_invalid_outputs_count_malformed_and_out_of_table(result):
    # p5 recorded invalid (malformed), p13 proposed an operation not in the table
    invalid = dict(result["invalid"])
    ci = invalid.pop("ci")
    assert invalid == {
        "n": 2,
        "N": 13,
        "rate": pytest.approx(2 / 13),
        "by_reason": {"malformed": 1, "unknown_operation": 1},
    }
    assert ci["n"] == 13
    assert ci["value"] == pytest.approx(2 / 13)


def test_proposal_with_arguments_failing_the_schema_is_invalid():
    predictions = [
        metrics.Prediction.from_dict(
            _rec(
                "x",
                _op(MUT_SCENE, scene="bright"),
                "propose",
                MUT_SCENE,
                {"scene": "turbo"},
            )
        )
    ]
    got = metrics.compute(predictions, DOMAIN)
    assert got["invalid"]["n"] == 1
    assert list(got["invalid"]["by_reason"]) == ["bad_choice"]
    assert got["wrong_mutating"]["total"] == 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_ece_ten_equal_width_bins_on_the_fixture(result):
    # Top candidate vs the expected label, 12 records with a distribution (p13 has none):
    #   bin 5: p5 .55 right, p7 .50 wrong            -> |0.5  - 0.525 | * 2/12
    #   bin 6: p2 .6 right, p8 .6 wrong, p12 .6 wrong -> |1/3  - 0.6   | * 3/12
    #   bin 7: p3 .7 wrong, p10 .7 wrong             -> |0    - 0.7   | * 2/12
    #   bin 8: p4 .8 right                           -> |1    - 0.8   | * 1/12
    #   bin 9: p1 .9 p6 1.0 p9 .9 right, p11 .95 wrong -> |0.75 - 0.9375| * 4/12
    #   sum = (0.05 + 0.8 + 1.4 + 0.2 + 0.75) / 12 = 3.2 / 12
    calibration = result["calibration"]
    assert calibration["n"] == 12
    assert calibration["without_distribution"] == 1
    assert calibration["ece"] == pytest.approx(3.2 / 12)
    assert [b["n"] for b in calibration["bins"]] == [0, 0, 0, 0, 0, 2, 3, 2, 1, 4]


def test_brier_on_the_fixture(result):
    # Multi-class Brier per record, sum over labels of (p - onehot)^2, then the mean:
    #   .02 .32 .98 .08 .405 0 .78 .72 .02 .98 1.805 .72 -> 6.83 / 12
    assert result["calibration"]["brier"] == pytest.approx(6.83 / 12)


def test_ece_is_zero_when_confidence_matches_accuracy():
    pairs = [(0.75, True), (0.75, True), (0.75, True), (0.75, False)]
    assert metrics.ece(pairs) == pytest.approx(0.0)


def test_ece_puts_confidence_one_in_the_last_bin_and_edges_up():
    assert metrics.bin_index(1.0) == 9
    assert metrics.bin_index(0.0) == 0
    assert metrics.bin_index(0.1) == 1
    assert metrics.bin_index(0.3) == 3
    assert metrics.ece([(1.0, False)]) == pytest.approx(1.0)


def test_brier_counts_a_gold_label_missing_from_the_candidates():
    assert metrics.brier_one({"a": 1.0}, "b") == pytest.approx(2.0)
    assert metrics.brier_one({"a": 0.5, "b": 0.5}, "b") == pytest.approx(0.5)


def test_top_candidate_tie_is_broken_by_label_order():
    assert metrics.top_candidate({"b": 0.5, "a": 0.5}) == ("a", 0.5)


def test_calibration_empty_when_no_record_has_a_distribution():
    predictions = [metrics.Prediction.from_dict(_rec("x", {"escalate": True}, "escalate"))]
    calibration = metrics.compute(predictions, DOMAIN)["calibration"]
    assert (calibration["n"], calibration["ece"], calibration["brier"]) == (0, None, None)


# ---------------------------------------------------------------------------
# Tokens and timing
# ---------------------------------------------------------------------------


def test_tokens_generated_per_decision(result):
    # total 140 over 13; sorted 3 4 5 6 7 8 [9] 10 11 12 15 20 30
    assert result["tokens"] == {"total": 140, "mean": pytest.approx(140 / 13), "median": 9}


def test_time_to_first_decision_and_latency_cold_and_warm(result):
    # cold = the first record; warm 40..62 step 2 -> median (50+52)/2, nearest-rank p95 = 62
    assert result["time_to_first_decision"] == {
        "cold_ms": 500.0,
        "warm_median_ms": 51.0,
        "warm_p95_ms": 62.0,
    }
    assert result["latency"] == {"cold_ms": 510.0, "warm_median_ms": 61.0, "warm_p95_ms": 72.0}


def test_median_and_percentile_95_match_bench():
    assert metrics.median([3, 1, 2]) == 2
    assert metrics.median([4, 1, 2, 3]) == 2.5
    assert metrics.percentile_95([float(i) for i in range(1, 21)]) == 19.0
    assert metrics.percentile_95([7.0]) == 7.0


# ---------------------------------------------------------------------------
# The issue-46 mapping (reporting only)
# ---------------------------------------------------------------------------


def test_issue46_json_maps_each_outcome():
    def as46(row):
        return metrics.issue46_json(metrics.Prediction.from_dict(row), DOMAIN)

    assert as46(_rec("a", _op(RO_ARG, room="d"), "propose", RO_ARG, {"room": "d"})) == {
        "action": "tool",
        "tool": RO_ARG,
        "arguments": {"room": "d"},
    }
    assert as46(_rec("b", {"escalate": True}, "escalate")) == {"action": "abstain"}
    assert as46(_rec("c", {"explain": True}, "explain")) == {"action": "no_action"}
    assert as46(_rec("d", {"explain": True}, "invalid")) == {"action": "invalid"}
    assert as46(_rec("e", {"escalate": True}, "abstain_uncertain")) == {"action": "abstain"}


def test_mapping_is_emitted_with_the_metrics(result):
    mapping = result["issue46_mapping"]
    assert [row["nvsh"] for row in mapping["rows"]] == list(metrics.OUTCOMES)
    assert '{"action": "abstain"}' in mapping["markdown"]
    assert "escalate" in mapping["markdown"]
    assert mapping["note"]


# ---------------------------------------------------------------------------
# abstain_uncertain: counted separately, rolled into escalation bars
# ---------------------------------------------------------------------------


def _abstain_uncertain_rows() -> list[dict]:
    return [
        _rec("q1", {"escalate": True}, "abstain_uncertain", candidates={ESC: 1.0}),
        _rec("q2", {"escalate": True}, "explain", candidates={EXP: 0.6, ESC: 0.4}),
        _rec("q3", _op(RO), "abstain_uncertain", candidates={ESC: 0.7, RO: 0.3}),
        _rec("q4", {"explain": True}, "abstain_uncertain", candidates={ESC: 0.55, EXP: 0.45}),
    ]


def test_abstain_uncertain_counts_toward_escalation_bars_but_separately(tmp_path):
    predictions = metrics.read_predictions(_write(tmp_path, _abstain_uncertain_rows()))
    got = metrics.compute(predictions, DOMAIN)
    escalation = got["escalation"]
    assert (escalation["tp"], escalation["fn"], escalation["fp"]) == (1, 1, 1)
    assert escalation["escalated_on_explain"] == 1
    assert escalation["abstain_uncertain"] == {"tp": 1, "fp": 1, "escalated_on_explain": 1}
    assert got["outcome_counts"]["escalate"] == 0
    assert got["outcome_counts"]["abstain_uncertain"] == 3


def test_abstain_uncertain_and_escalate_both_map_to_issue46_abstain():
    rows = [row["nvsh"] for row in metrics.issue46_mapping()["rows"]]
    assert rows == ["propose", "explain", "escalate", "abstain_uncertain", "invalid"]


# ---------------------------------------------------------------------------
# Escalation-reason roll-up
# ---------------------------------------------------------------------------


def test_escalate_reason_label_rolls_up_to_bare_escalate_for_calibration():
    rolled = metrics.rollup_escalate_candidates({"escalate:missing_argument": 0.7, EXP: 0.3})
    assert rolled == {ESC: pytest.approx(0.7), EXP: pytest.approx(0.3)}
    label, confidence = metrics.top_candidate(rolled)
    assert (label, confidence) == (ESC, pytest.approx(0.7))


def test_escalate_reason_is_extracted_and_tallied(tmp_path):
    rows = [
        _rec("r1", {"escalate": True}, "escalate", candidates={"escalate:injection": 1.0}),
        _rec("r2", _op(RO), "propose", RO, {}, {"escalate:multi_step": 0.6, RO: 0.4}),
        _rec("r3", {"escalate": True}, "escalate", candidates={ESC: 1.0}),
    ]
    predictions = metrics.read_predictions(_write(tmp_path, rows))
    got = metrics.compute(predictions, DOMAIN)
    assert got["escalation_reasons"] == {"injection": 1, "multi_step": 1}
    assert metrics.escalate_reason(ESC) is None
    assert metrics.escalate_reason("escalate:") is None
    assert metrics.canonical_label("escalate:multi_step") == ESC
    assert metrics.canonical_label(RO) == RO


def test_missing_candidate_rolls_up_escalate_reason_before_checking_gold():
    prediction = metrics.Prediction.from_dict(
        _rec("s1", {"escalate": True}, "escalate", candidates={"escalate:injection": 1.0})
    )
    assert metrics.is_missing_candidate(prediction) is False


# ---------------------------------------------------------------------------
# Per-slice ECE/Brier/bins, candidate count, missing-candidate
# ---------------------------------------------------------------------------


def test_slice_name_is_the_gold_operations_read_only_flag():
    read_only = metrics.Prediction.from_dict(_rec("t1", _op(RO), "explain"))
    mutating = metrics.Prediction.from_dict(_rec("t2", _op(MUT_ROOM, room="x"), "explain"))
    unknown = metrics.Prediction.from_dict(_rec("t5", _op("not_in_the_table"), "explain"))
    escalate = metrics.Prediction.from_dict(_rec("t3", {"escalate": True}, "escalate"))
    explain = metrics.Prediction.from_dict(_rec("t4", {"explain": True}, "explain"))
    assert metrics.slice_name(read_only, DOMAIN) == "read_only"
    assert metrics.slice_name(mutating, DOMAIN) == "mutating"
    assert metrics.slice_name(unknown, DOMAIN) == "mutating"
    assert metrics.slice_name(escalate, DOMAIN) == "escalate_or_explain"
    assert metrics.slice_name(explain, DOMAIN) == "escalate_or_explain"


def test_slices_partition_the_fixture_and_report_calibration(result):
    slices = result["slices"]
    assert set(slices) == {"read_only", "mutating", "escalate_or_explain"}
    assert sum(s["n"] for s in slices.values()) == 13
    # p1 p2 p5 p12 are read-only gold; p3 p4 are mutating gold.
    assert slices["read_only"]["n"] == 4
    assert slices["mutating"]["n"] == 2
    for name in ("read_only", "mutating", "escalate_or_explain"):
        calibration = slices[name]["calibration"]
        for key in ("ece", "ece_ci", "brier", "brier_ci", "bins"):
            assert key in calibration
        assert len(calibration["bins"]) == metrics.ECE_BINS


def test_slice_candidate_count_and_missing_candidate(tmp_path):
    rows = [
        _rec("u1-nocand", _op(RO), "escalate", candidates={RO_ARG: 0.5, ESC: 0.5}),
        _rec("u2", _op(RO), "propose", RO, {}, {RO: 1.0}),
    ]
    predictions = metrics.read_predictions(_write(tmp_path, rows))
    got = metrics.compute(predictions, DOMAIN)
    read_only = got["slices"]["read_only"]
    assert read_only["n"] == 2
    assert read_only["candidate_count"] == {"n": 2, "mean": 1.5, "median": 1.5}
    assert read_only["missing_candidate"]["n"] == 1
    assert read_only["missing_candidate"]["N"] == 2
    assert read_only["missing_candidate"]["rate"]["value"] == pytest.approx(0.5)


def test_is_missing_candidate_by_id_suffix_or_absent_gold():
    by_suffix = metrics.Prediction.from_dict(
        _rec("v-nocand", _op(RO), "escalate", candidates={ESC: 1.0})
    )
    by_absent_gold = metrics.Prediction.from_dict(
        _rec("v2", _op(RO), "propose", RO, {}, {RO_ARG: 1.0})
    )
    present = metrics.Prediction.from_dict(_rec("v3", _op(RO), "propose", RO, {}, {RO: 1.0}))
    assert metrics.is_missing_candidate(by_suffix) is True
    assert metrics.is_missing_candidate(by_absent_gold) is True
    assert metrics.is_missing_candidate(present) is False


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_seeded_and_reproducible():
    items = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    first = metrics.bootstrap_ci(items, metrics._mean_stat, seed=7, resamples=200)
    second = metrics.bootstrap_ci(items, metrics._mean_stat, seed=7, resamples=200)
    assert first == second
    assert first["n"] == 10
    assert first["value"] == pytest.approx(0.6)
    assert 0.0 <= first["ci_low"] <= first["value"] <= first["ci_high"] <= 1.0


def test_bootstrap_ci_degenerate_cases():
    assert metrics.bootstrap_ci([], metrics._mean_stat) == {
        "n": 0,
        "value": None,
        "ci_low": None,
        "ci_high": None,
    }
    one = metrics.bootstrap_ci([1], metrics._mean_stat)
    assert one == {"n": 1, "value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_compute_bootstrap_seed_and_resamples_are_reported_and_parameterised(tmp_path):
    predictions = metrics.read_predictions(_write(tmp_path, _fixture_rows()))
    default = metrics.compute(predictions, DOMAIN)
    assert default["bootstrap"] == {
        "seed": metrics.DEFAULT_BOOTSTRAP_SEED,
        "resamples": metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
    }
    custom = metrics.compute(predictions, DOMAIN, bootstrap_seed=99, bootstrap_resamples=50)
    assert custom["bootstrap"] == {"seed": 99, "resamples": 50}
    assert custom["right_proposals"]["percent"] == default["right_proposals"]["percent"]


def test_every_rate_and_ece_carries_n_and_a_bootstrap_ci(result):
    for ci in (
        result["right_proposals"]["ci"],
        result["escalation"]["recall_ci"],
        result["escalation"]["precision_ci"],
        result["escalation"]["precision_strict_ci"],
        result["false_positive_tool_calls"]["ci"],
        result["invalid"]["ci"],
        result["calibration"]["ece_ci"],
        result["calibration"]["brier_ci"],
    ):
        assert set(ci) == {"n", "value", "ci_low", "ci_high"}
        assert isinstance(ci["n"], int)
        assert ci["n"] > 0


# ---------------------------------------------------------------------------
# The reliability markdown renderer
# ---------------------------------------------------------------------------


def test_reliability_markdown_has_one_table_per_slice_and_no_operation_names(result):
    markdown = metrics.reliability_markdown(result["slices"])
    for name in metrics.SLICE_NAMES:
        assert f"### {name}" in markdown
    assert "| bin | n | confidence | accuracy |" in markdown
    for op in DOMAIN.names():
        assert op not in markdown


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_prints_json(tmp_path, capsys):
    path = _write(tmp_path, _fixture_rows())
    assert metrics.main(["--domain", "tests.fixtures.toy_domain", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["right_proposals"]["N"] == 6
    assert out["issue46_mapping"]["rows"]


def test_main_reports_a_bad_file_on_stderr(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{nope\n", encoding="utf-8")
    assert metrics.main(["--domain", "tests.fixtures.toy_domain", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 1" in captured.err


def test_main_reports_a_bad_domain_on_stderr(tmp_path, capsys):
    path = _write(tmp_path, _fixture_rows())
    assert metrics.main(["--domain", "no_such_domain_module", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


# ---------------------------------------------------------------------------
# Grounded arguments compare as grounding matches them (nvsh issue 53, d5)
# ---------------------------------------------------------------------------

#: The toy domain with a suffix on its room kind ('kitchen' grounds to 'kitchen.zone')
#: and one free-text, ungrounded argument.
SUFFIXED = dataclasses.replace(
    DOMAIN,
    ground_kinds=(
        GroundKind(
            name="room", noun="room", plural="rooms", world_field="rooms", suffixes=(".zone",)
        ),
    ),
    operations=DOMAIN.operations
    + (Operation("note_add", "Add a note.", False, (ArgSpec("text", "str"),)),),
)


def test_a_grounded_argument_compares_as_grounding_matches_it(tmp_path):
    """Gold 'kitchen' and the grounded 'kitchen.zone' name one room, case-insensitively;
    a different room is still wrong arguments."""
    rows = [
        _rec(
            "s1",
            _op(MUT_ROOM, room="kitchen"),
            "propose",
            MUT_ROOM,
            {"room": "kitchen.zone"},
        ),
        _rec("s2", _op(RO_ARG, room="Hall.zone"), "propose", RO_ARG, {"room": "hall"}),
        _rec("s3", _op(MUT_ROOM, room="attic"), "propose", MUT_ROOM, {"room": "cellar.zone"}),
        _rec("c1", _op(MUT_ROOM, room="Kitchen"), "propose", MUT_ROOM, {"room": "kitchen"}),
    ]
    result = metrics.compute(metrics.read_predictions(_write(tmp_path, rows)), SUFFIXED)
    assert result["right_proposals"]["n"] == 3
    assert result["wrong_mutating"]["wrong_arguments"] == 1


def test_a_ground_suffix_is_not_dropped_from_other_arguments(tmp_path):
    rows = [
        _rec("m1", _op("note_add", text="Hi"), "propose", "note_add", {"text": "hi.zone"}),
        _rec("m2", _op("note_add", text="Hi"), "propose", "note_add", {"text": "hi"}),
    ]
    result = metrics.compute(metrics.read_predictions(_write(tmp_path, rows)), SUFFIXED)
    assert result["right_proposals"]["n"] == 0
    assert result["wrong_mutating"]["wrong_arguments"] == 2


# ---------------------------------------------------------------------------
# The Domain seam: read-only comes from the domain, never an operation name
# ---------------------------------------------------------------------------


def test_flipping_read_only_in_the_domain_moves_the_mutating_counts():
    row = _rec(
        "w1",
        {"explain": True},
        "propose",
        RO,
        {},
        {RO: 0.9, EXP: 0.1},
    )
    predictions = [metrics.Prediction.from_dict(row)]
    assert metrics.compute(predictions, DOMAIN)["wrong_mutating"]["wrong_operation"] == 0
    flipped = dataclasses.replace(
        DOMAIN,
        operations=tuple(
            dataclasses.replace(op, read_only=False) if op.name == RO else op
            for op in DOMAIN.operations
        ),
    )
    assert metrics.compute(predictions, flipped)["wrong_mutating"]["wrong_operation"] == 1


def test_metrics_source_names_no_operation():
    text = Path(metrics.__file__).read_text(encoding="utf-8")
    for name in DOMAIN.names():
        assert name not in text
