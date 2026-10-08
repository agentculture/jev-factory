"""Release-gate scoring: traces, the worktree guard, policies, the metrics bridge,
the DeepEval layer's exact verdicts and the report's model-only vs model+harness rows.

Ported from nvsh evals/tool_jev/tests/test_trace.py, test_policies.py,
test_metrics_bridge.py, test_deepeval_layer.py and test_report.py onto the
toy domain. The DeepEval wrappers themselves are exercised only when the
``evals`` dependency group is installed (``pytest.importorskip``).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from jev_factory.core.predictions import Prediction
from jev_factory.evals import deepeval_layer, metrics_bridge, policies, report
from jev_factory.evals.trace import (
    RawRecord,
    Trace,
    TraceWriteError,
    append_trace,
    inside_git_worktree,
    read_traces,
    write_traces,
)
from tests.evals_support import CASES, candidate_predictions
from tests.fixtures.toy_domain import DOMAIN

STRICT = "mutating-strict-example"


def _traces(subject="cand.toy-test") -> list[Trace]:
    return [
        Trace.from_prediction_line(row, split="test", subject=subject)
        for row in candidate_predictions()
    ]


# ---------------------------------------------------------------------------
# trace + the worktree guard
# ---------------------------------------------------------------------------


def test_trace_round_trips_and_keeps_candidate_order(tmp_path):
    traces = _traces()
    decided = [t.with_policy("raw", "propose", "") for t in traces]
    assert traces[0].final == {} and decided[0].raw is traces[0].raw
    path = tmp_path / "t" / "cand.jsonl"
    write_traces(path, decided)
    back = read_traces(path)
    assert [t.to_dict() for t in back] == [t.to_dict() for t in decided]
    assert list(back[0].raw.candidates) == list(traces[0].raw.candidates)
    append_trace(path, traces[0])
    assert len(read_traces(path)) == len(decided) + 1


def test_raw_record_from_a_provider_answer_and_back_to_a_prediction():
    record = RawRecord.from_provider_answer(
        provider="p",
        model="m",
        returned_model="m-1",
        interface="choice",
        outcome="propose",
        operation="lamp_status",
        arguments={},
        candidates={"lamp_status": 0.9, "(explain)": 0.1},
    )
    row = record.to_prediction_dict("c-2", {"operation": "lamp_status"})
    assert Prediction.from_dict(row).tokens == 0
    assert RawRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError):
        RawRecord.from_provider_answer(
            provider="p", model="m", returned_model=None, interface="tool_call", outcome="explain"
        )


@pytest.mark.behavioral("o32")
def test_guard_passes_in_a_temp_dir_holding_an_empty_git_dir(tmp_path):
    (tmp_path / ".git").mkdir()  # nvsh#71 item 2: not a repository, so not refused
    assert inside_git_worktree(tmp_path / "run") is False
    write_traces(tmp_path / "run" / "traces" / "x.jsonl", _traces()[:1])


@pytest.mark.behavioral("o32")
def test_guard_uses_git_rev_parse_and_refuses_a_real_worktree(tmp_path, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)
    assert inside_git_worktree(tmp_path / "deep" / "run") is True
    traces = _traces()[:1]
    with pytest.raises(TraceWriteError):
        write_traces(tmp_path / "t.jsonl", traces)
    first = _traces()[0]
    with pytest.raises(TraceWriteError):
        append_trace(tmp_path / "t.jsonl", first)

    calls = []
    real_run = subprocess.run

    def spy(argv, *args, **kwargs):
        calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr("jev_factory.factory.workroot.subprocess.run", spy)
    inside_git_worktree(tmp_path)
    assert calls and calls[0][-2:] == ["rev-parse", "--is-inside-work-tree"]


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def test_raw_policy_is_the_bare_argmax_and_never_abstains():
    raw = policies.load_policy(policies.builtin_policy_path("raw"))
    row = candidate_predictions()[1]  # c-2: lamp_on at 0.3, the argmax
    decision, reason, name, version = policies.apply(raw, row, list(row["candidates"]), DOMAIN)
    assert (decision, reason, name, version) == ("propose", None, "raw", "1")
    escalate = candidate_predictions()[3]
    assert policies.apply(raw, escalate, list(escalate["candidates"]), DOMAIN)[0] == "escalate"


def test_strict_policy_gates_a_weak_mutating_pick_on_the_mutating_set():
    strict = policies.load_policy(policies.builtin_policy_path(STRICT))
    row = candidate_predictions()[1]
    decision, reason, *_ = policies.apply(strict, row, list(row["candidates"]), DOMAIN)
    assert decision == "abstain_uncertain" and reason == "floor"
    confident = candidate_predictions()[0]
    assert policies.apply(strict, confident, list(confident["candidates"]), DOMAIN)[0] == (
        "propose"
    )


def test_an_unknown_operation_is_gated_as_mutating():
    strict = policies.load_policy(policies.builtin_policy_path(STRICT))
    # 0.55 clears no mutating floor (0.6) but would pass the read-only set (margin only).
    candidates = {"warp_drive": 0.55, "lamp_status": 0.05, "(explain)": 0.4}
    no_temperature = {**strict, "calibration": None}
    decision, reason, *_ = policies.apply(
        no_temperature, {"candidates": candidates}, list(candidates), DOMAIN
    )
    assert (decision, reason) == ("abstain_uncertain", "floor")


def test_policy_validation_and_the_not_gateable_record(tmp_path):
    with pytest.raises(policies.PolicyError):
        policies.validate_policy({"version": "1"})
    with pytest.raises(policies.PolicyError):
        policies.validate_policy({"name": "x"})
    with pytest.raises(policies.PolicyError):
        policies.validate_policy(["x"])
    raw = policies.load_policy(policies.builtin_policy_path("raw"))
    assert policies.apply(raw, {"candidates": None}, [], DOMAIN)[:2] == (
        "not_gateable",
        "no_distribution",
    )


# ---------------------------------------------------------------------------
# metrics bridge
# ---------------------------------------------------------------------------


def test_bridge_passes_core_metrics_through_and_adds_its_own():
    predictions = [Prediction.from_dict(row) for row in candidate_predictions()]
    out = metrics_bridge.compute(predictions, DOMAIN, bootstrap_resamples=10)
    assert out["wrong_mutating"]["total"] == 1
    assert out["metrics_compute"]["right_proposals"]["n"] == 1
    top_k = out["top_k_accuracy"]
    assert top_k[1]["N"] == 2 and top_k[3]["n"] == 1 and top_k[5]["n"] == 2
    assert out["missing_candidate"]["n"] == 1
    assert out["log_loss"]["n"] == 5 and out["log_loss"]["mean"] > 0
    row = out["rows"][1]
    assert row["slice"] == "read_only" and row["wrong_mutating"] is True
    assert row["top1_correct"] is False and 0 <= row["entropy"] <= 1 and row["margin"] >= 0


def test_bridge_reports_not_measurable_without_a_distribution():
    row = dict(candidate_predictions()[3], candidates=None)
    prediction = Prediction.from_dict(row)
    figures = metrics_bridge.row_metrics(prediction, DOMAIN)
    for name in ("top1_correct", "brier", "log_loss", "entropy", "margin"):
        assert figures[name] == metrics_bridge.NOT_MEASURABLE
    assert figures["topk_correct"][3] == metrics_bridge.NOT_MEASURABLE


# ---------------------------------------------------------------------------
# DeepEval layer (verdicts need no deepeval)
# ---------------------------------------------------------------------------


def test_policy_applied_prediction_is_what_the_verdicts_grade():
    traces = _traces()
    raw_metadata = deepeval_layer.case_metadata(traces[1], "raw", DOMAIN)
    strict_metadata = deepeval_layer.case_metadata(traces[1], STRICT, DOMAIN)
    assert raw_metadata["final_decision"] == "propose"
    assert strict_metadata["final_decision"] == "abstain_uncertain"
    assert deepeval_layer.wrong_mutating(raw_metadata, DOMAIN)[0] is False
    assert deepeval_layer.wrong_mutating(strict_metadata, DOMAIN)[0] is True
    assert deepeval_layer.right_action(raw_metadata, DOMAIN)[0] is False
    assert "not applicable" in deepeval_layer.correct_abstain_escalate(raw_metadata, DOMAIN)[1]
    first = deepeval_layer.case_metadata(traces[0], "raw", DOMAIN)
    assert deepeval_layer.right_action(first, DOMAIN)[0] is True
    explain = deepeval_layer.case_metadata(traces[2], "raw", DOMAIN)
    assert deepeval_layer.correct_abstain_escalate(explain, DOMAIN)[0] is True
    assert "not applicable" in deepeval_layer.right_action(explain, DOMAIN)[1]
    escalate = deepeval_layer.case_metadata(traces[3], "raw", DOMAIN)
    assert deepeval_layer.correct_abstain_escalate(escalate, DOMAIN)[0] is True
    nocand = deepeval_layer.case_metadata(traces[4], "raw", DOMAIN)
    assert deepeval_layer.missing_candidate_handled(nocand, DOMAIN)[0] is True
    assert "not applicable" in deepeval_layer.missing_candidate_handled(first, DOMAIN)[1]
    assert json.dumps(raw_metadata)  # JSON-able, and it never carries the case text
    assert CASES[1]["text"] not in json.dumps(raw_metadata)


def test_a_truncated_or_distribution_free_row_is_never_regated():
    trace = Trace(
        case_id="c-1",
        split="test",
        raw=RawRecord(
            outcome="invalid",
            operation=None,
            arguments=None,
            candidates={"lamp_on": 0.9, "(escalate)": 0.1},
            invalid_reason="truncated",
        ),
        ground_truth=CASES[0]["expect"],
    )
    prediction = deepeval_layer.prediction_from_trace(trace)
    assert deepeval_layer.apply_policy_to_prediction("raw", prediction, DOMAIN) is prediction
    unknown_case = Trace("x", "test", trace.raw)
    with pytest.raises(deepeval_layer.DeepevalLayerError):
        deepeval_layer.prediction_from_trace(unknown_case)


def test_corpus_metrics_separate_the_model_from_the_harness():
    traces = _traces()
    raw = deepeval_layer.corpus_metrics(traces, "raw", DOMAIN)
    strict = deepeval_layer.corpus_metrics(traces, STRICT, DOMAIN)
    assert raw["wrong_mutating"]["total"] == 1
    assert strict["wrong_mutating"]["total"] == 0


def test_deepeval_wrappers_when_the_evals_group_is_installed(tmp_path):
    pytest.importorskip("deepeval")
    metrics = deepeval_layer.build_metrics(DOMAIN)
    assert [m.__name__ for m in metrics] == list(deepeval_layer.METRIC_NAMES)
    case = deepeval_layer.build_test_case(_traces()[1], STRICT, DOMAIN)
    assert case.input == "case:c-2" and case.actual_output == "abstain_uncertain"
    outcome = deepeval_layer.evaluate_traces(
        _traces(), STRICT, DOMAIN, results_folder=tmp_path / "de"
    )
    assert outcome.corpus_metrics["wrong_mutating"]["total"] == 0
    assert list((tmp_path / "de").glob("test_run_*.json"))


# ---------------------------------------------------------------------------
# report: model-only vs model+harness rows
# ---------------------------------------------------------------------------


def _run_dir(tmp_path):
    traces = _traces()
    write_traces(report.traces_path(tmp_path, "cand.toy-test"), traces)
    for policy in ("raw", STRICT):
        path = report.metrics_path(tmp_path, "cand.toy-test", policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(deepeval_layer.corpus_metrics(traces, policy, DOMAIN)), encoding="utf-8"
        )
    (tmp_path / report.MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "run_id": "r1",
                "date": "2026-09-30",
                "deepeval": False,
                "subjects": [
                    {
                        "name": "cand.toy-test",
                        "kind": "candidate",
                        "policies": ["raw", STRICT],
                        "artifact": {"predictions_sha256": "abc"},
                    }
                ],
            }
        )
    )
    return tmp_path


def test_report_rows_answer_model_and_harness_separately_and_deterministically(tmp_path):
    run_dir = _run_dir(tmp_path)
    result, page = report.generate(run_dir)
    model_only, harness = result["rows"]
    assert (model_only["variant"], model_only["harness_policy"]) == ("model-only", None)
    assert (harness["variant"], harness["harness_policy"]) == ("model+harness", STRICT)
    assert model_only["wrong_mutations"] == 1 and harness["wrong_mutations"] == 0
    assert result["deepeval"] is False and result["reference_rows"] == []
    assert harness["slices"]["permutation"] == "not_run"
    assert "| cand.toy-test (model+harness) | mutating-strict-example |" in page
    assert "DeepEval per-case layer: did not run" in page
    for case in CASES:
        assert case["text"] not in page and case["id"] not in json.dumps(result)
    assert report.generate(run_dir) == (result, page)


def test_report_manifest_is_validated(tmp_path):
    (tmp_path / report.MANIFEST_FILENAME).write_text(
        json.dumps({"run_id": "r", "date": "d", "subjects": [{"name": "s", "kind": "judge"}]})
    )
    with pytest.raises(report.ReportError):
        report.load_manifest(tmp_path)
    (tmp_path / report.MANIFEST_FILENAME).write_text(
        json.dumps({"run_id": "r", "date": "d", "deepeval": "yes", "subjects": []})
    )
    with pytest.raises(report.ReportError):
        report.load_manifest(tmp_path)


def test_missing_candidate_rows_report_their_escalation_rate_beside_their_share():
    def row(rid, outcome, offered):
        proposed = outcome == "propose"
        return Prediction.from_dict(
            {
                "id": rid,
                "expected": {"operation": "lamp_on", "args": {}},
                "outcome": outcome,
                "operation": "lamp_on" if proposed else None,
                "arguments": {} if proposed else None,
                "candidates": {label: 1.0 / len(offered) for label in offered},
                "tokens": 1,
                "ttfd_ms": 1.0,
                "latency_ms": 1.0,
            }
        )

    gone = ["lamp_status", "(explain)", "(escalate)"]  # lamp_on is not offered
    rows = [
        row("a-nocand", "escalate", gone),
        row("b-nocand", "abstain_uncertain", gone),
        row("c-nocand", "explain", gone),
        row("d", "propose", ["lamp_on", "(explain)", "(escalate)"]),
    ]
    out = metrics_bridge.missing_candidate_summary(rows)
    assert (out["n"], out["N"], out["escalated"]) == (3, 4, 2)
    assert out["rate"] == 0.75 and abs(out["escalation_rate"] - 2 / 3) < 1e-9
