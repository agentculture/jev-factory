"""DeepEval layer over release-gate traces.

One ``deepeval.test_case.LLMTestCase`` per ``(subject, case, policy)``: the
case's raw model output *replayed through one harness policy*, exactly as
:func:`apply_policy_to_prediction` builds it. Four exact metrics grade each
test case, with no LLM judge, no network and no randomness:

- **right action**: :func:`jev_factory.core.metrics._right_proposal` on the
  **policy-applied** prediction. A policy that gates a correct raw pick away
  into an abstention loses right-action credit for it;
- **wrong mutating**: the negation of ``_wrong_operation_mutating`` OR
  ``_wrong_arguments_mutating`` on the policy-applied prediction. A raw
  wrong-mutating pick the policy gates away is no longer a wrong mutating call
  *under that policy*: "the model improved" versus "the harness prevented the
  model's mistake";
- **correct abstain/escalate**: for a decline-expected case, whether the
  policy-applied outcome declined correctly;
- **missing-candidate handled**: for a case whose gold operation was not
  offered, whether the policy-applied prediction escalated.

The verdicts are plain functions (:func:`right_action`, :func:`wrong_mutating`,
:func:`correct_abstain_escalate`, :func:`missing_candidate_handled`) of a test
case's ``additional_metadata`` and the Domain, so they run and are tested
without deepeval; the ``deepeval.metrics.BaseMetric`` wrappers are built on
first use by :func:`build_metrics`.

Corpus-level figures (ECE, Brier, coverage, abstain precision/recall,
per-slice) are never computed by DeepEval: :func:`corpus_metrics` returns
:func:`jev_factory.evals.metrics_bridge.compute`'s output over the same
policy-applied predictions, and :func:`evaluate_traces` returns it next to
DeepEval's own per-case ``EvaluationResult``.

**deepeval is imported lazily**, inside :func:`build_metrics` and
:func:`evaluate_traces` only: it lives in the ``evals`` dependency group,
never the base install, and this package's ``__init__`` (the env guard) has
always run first. :func:`deepeval_available` says whether it is installed
without importing it.

``evaluate_traces`` keeps DeepEval's own local state directory (``.deepeval/``,
resolved relative to the process's working directory) out of any repository
by running ``evaluate()`` inside ``contextlib.chdir()`` into a directory under
``results_folder``; ``os.chdir`` is process-global, so this is safe only
because a gate run evaluates one ``(subject, policy)`` at a time.
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_factory.core import metrics as metrics_mod
from jev_factory.core.predictions import Prediction
from jev_factory.domain.model import Domain

from . import metrics_bridge
from . import policies as policy_module
from .trace import Trace, inside_git_worktree

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/deepeval_layer.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 83-88: deepeval is imported lazily (build_metrics/evaluate_traces) instead of"
        " at module import, so the module imports without the evals dependency group",
        "the four metric classes (lines 395-555) are built on first use around module-level"
        " verdict functions (right_action, wrong_mutating, correct_abstain_escalate,"
        " missing_candidate_handled) that need no deepeval",
        "path-loaded metrics/gate modules -> jev_factory.core; every function takes the"
        " Domain; policy application reuses policies.scaled_and_decided",
        "corpus_metrics is split out of evaluate_traces so a run without deepeval computes"
        " the same corpus figures; the git-worktree refusal uses trace.inside_git_worktree"
        " (git rev-parse via factory.workroot)",
        "a policy-applied proposal keeps the recorded arguments only when the gate's label is"
        " the recorded operation, as before",
    ],
    "licence": "Apache-2.0",
}

#: ``invalid_reason`` of a reply the provider cut at the output budget.
TRUNCATED = "truncated"

#: The metric names, in the order :func:`build_metrics` returns them.
METRIC_NAMES = (
    "right_action",
    "wrong_mutating",
    "correct_abstain_escalate",
    "missing_candidate_handled",
)


class DeepevalLayerError(ValueError):
    """A trace this module cannot build a scorable test case from."""


def deepeval_available() -> bool:
    """Whether deepeval is installed (checked without importing it)."""
    return importlib.util.find_spec("deepeval") is not None


# ---------------------------------------------------------------------------
# Trace -> Prediction, and the policy-applied prediction
# ---------------------------------------------------------------------------


def _prediction_row(
    *,
    case_id: str,
    expected: Mapping[str, Any],
    outcome: str,
    operation: str | None,
    arguments: dict | None,
    candidates: dict | None,
    tokens: int | None,
    ttfd_ms: float | None,
    latency_ms: float | None,
    invalid_reason: str | None,
    offered: Sequence[str] | None = None,
) -> dict:
    """A ``Prediction.from_dict``-ready row; missing timing figures default to zero."""
    row = {
        "id": case_id,
        "expected": dict(expected),
        "outcome": outcome,
        "operation": operation,
        "arguments": None if arguments is None else dict(arguments),
        "candidates": None if candidates is None else dict(candidates),
        "tokens": 0 if tokens is None else tokens,
        "ttfd_ms": 0.0 if ttfd_ms is None else ttfd_ms,
        "latency_ms": 0.0 if latency_ms is None else latency_ms,
        "invalid_reason": invalid_reason,
    }
    if offered is not None and candidates is not None:
        row["offered"] = list(offered)
    return row


def prediction_from_trace(trace: Trace) -> Prediction:
    """The raw :class:`Prediction` of *trace*; :class:`DeepevalLayerError` without ground truth."""
    if trace.ground_truth is None:
        raise DeepevalLayerError(
            f"trace {trace.case_id!r} carries no ground_truth to score against"
        )
    raw = trace.raw
    return Prediction.from_dict(
        _prediction_row(
            case_id=trace.case_id,
            expected=trace.ground_truth,
            outcome=raw.outcome,
            operation=raw.operation,
            arguments=raw.arguments,
            candidates=raw.candidates,
            tokens=raw.tokens,
            ttfd_ms=raw.ttfd_ms,
            latency_ms=raw.latency_ms,
            invalid_reason=raw.invalid_reason,
            offered=raw.offered,
        )
    )


def resolve_policy(policy: Mapping[str, Any] | str) -> dict:
    """A builtin policy name, or an in-memory policy dict, as a validated policy dict."""
    if isinstance(policy, str):
        return policy_module.load_policy(policy_module.builtin_policy_path(policy))
    return policy_module.validate_policy(policy)


def apply_policy_to_prediction(
    policy: Mapping[str, Any] | str, prediction: Prediction, domain: Domain
) -> Prediction:
    """The :class:`Prediction` *policy* actually produces on *prediction*.

    The one place a policy-applied prediction is built, shared by the four
    metrics and the corpus figures so both always agree. The outcome becomes
    the gate's decision; ``operation``/``arguments`` are kept only for a
    ``propose`` (the recorded arguments when the gate's label is the recorded
    operation, else ``{}``); ``candidates`` is the calibrated distribution.
    A row with no distribution, or a reply cut at the output budget, is
    returned unchanged: there is nothing for a policy to gate.
    """
    validated = resolve_policy(policy)
    if prediction.outcome == "invalid" and prediction.invalid_reason == TRUNCATED:
        return prediction
    if not prediction.candidates:
        return prediction
    offered = list(prediction.offered_order())
    scaled, decision = policy_module.scaled_and_decided(
        validated, prediction.candidates, offered, domain
    )
    if decision.outcome == "propose":
        operation = decision.label
        arguments = (
            dict(prediction.arguments)
            if (prediction.operation == operation and prediction.arguments is not None)
            else {}
        )
    else:
        operation = None
        arguments = None
    return Prediction.from_dict(
        _prediction_row(
            case_id=prediction.id,
            expected=prediction.expected,
            outcome=decision.outcome,
            operation=operation,
            arguments=arguments,
            candidates=scaled,
            tokens=prediction.tokens,
            ttfd_ms=prediction.ttfd_ms,
            latency_ms=prediction.latency_ms,
            invalid_reason=prediction.invalid_reason,
            offered=prediction.offered,
        )
    )


def case_metadata(trace: Trace, policy: Mapping[str, Any] | str, domain: Domain) -> dict:
    """Everything a metric needs to rebuild the policy-applied prediction, JSON-able.

    Never the case text: only ids, expectations, raw fields and the resolved
    policy JSON, so a verdict is reproducible from the metadata alone.
    """
    policy_dict = resolve_policy(policy)
    raw_prediction = prediction_from_trace(trace)
    applied = apply_policy_to_prediction(policy_dict, raw_prediction, domain)
    raw = trace.raw
    return {
        "case_id": trace.case_id,
        "split": trace.split,
        "subject": trace.subject,
        "policy": policy_dict["name"],
        "policy_json": policy_dict,
        "expected": dict(trace.ground_truth or {}),
        "raw_outcome": raw.outcome,
        "raw_operation": raw.operation,
        "raw_arguments": raw.arguments,
        "raw_candidates": raw.candidates,
        "raw_offered": None if raw.offered is None else list(raw.offered),
        "raw_tokens": raw.tokens,
        "raw_ttfd_ms": raw.ttfd_ms,
        "raw_latency_ms": raw.latency_ms,
        "raw_invalid_reason": raw.invalid_reason,
        "final_decision": applied.outcome,
        "final_operation": applied.operation,
        "final_arguments": applied.arguments,
        "final_candidates": applied.candidates,
    }


def _applied_from_metadata(metadata: Mapping[str, Any], domain: Domain) -> Prediction:
    raw_prediction = Prediction.from_dict(
        _prediction_row(
            case_id=metadata["case_id"],
            expected=metadata["expected"],
            outcome=metadata["raw_outcome"],
            operation=metadata["raw_operation"],
            arguments=metadata["raw_arguments"],
            candidates=metadata["raw_candidates"],
            tokens=metadata["raw_tokens"],
            ttfd_ms=metadata["raw_ttfd_ms"],
            latency_ms=metadata["raw_latency_ms"],
            invalid_reason=metadata["raw_invalid_reason"],
            offered=metadata.get("raw_offered"),
        )
    )
    return apply_policy_to_prediction(metadata["policy_json"], raw_prediction, domain)


# ---------------------------------------------------------------------------
# The four exact verdicts (no deepeval needed)
# ---------------------------------------------------------------------------


def right_action(metadata: Mapping[str, Any], domain: Domain) -> tuple[bool, str]:
    """Right action on the policy-applied prediction; passes when no operation is expected."""
    prediction = _applied_from_metadata(metadata, domain)
    if metrics_mod.expect_kind(prediction.expected) != "operation":
        return True, "not applicable: case does not expect an operation"
    gold = metrics_mod.expected_label(prediction.expected)
    correct = metrics_mod._right_proposal(prediction, domain)
    verb = "proposes" if correct else "does not propose"
    return bool(correct), f"policy-applied outcome {prediction.outcome!r} {verb} gold {gold!r}"


def wrong_mutating(metadata: Mapping[str, Any], domain: Domain) -> tuple[bool, str]:
    """Fails when the policy-applied prediction is a wrong mutating call."""
    prediction = _applied_from_metadata(metadata, domain)
    wrong = metrics_mod._wrong_operation_mutating(
        prediction, domain
    ) or metrics_mod._wrong_arguments_mutating(prediction, domain)
    if wrong:
        return False, "policy-applied outcome is a wrong mutating call"
    return True, "policy-applied outcome is not a wrong mutating call"


def correct_abstain_escalate(metadata: Mapping[str, Any], domain: Domain) -> tuple[bool, str]:
    """On a decline-expected case, whether the policy-applied outcome declined correctly."""
    prediction = _applied_from_metadata(metadata, domain)
    kind = metrics_mod.expect_kind(prediction.expected)
    if kind == "escalate":
        escalated = metrics_mod._escalated(prediction)
        verb = "correctly escalates" if escalated else "fails to escalate"
        return escalated, (
            f"policy-applied outcome {prediction.outcome!r} {verb} an escalate-expected case"
        )
    if kind == "explain":
        explained = prediction.outcome == "explain"
        verb = "correctly explains" if explained else "fails to explain"
        return explained, (
            f"policy-applied outcome {prediction.outcome!r} {verb} an explain-expected case"
        )
    return True, "not applicable: case expects an operation, not a decline"


def missing_candidate_handled(metadata: Mapping[str, Any], domain: Domain) -> tuple[bool, str]:
    """On a missing-candidate case, whether the policy-applied prediction escalated."""
    prediction = _applied_from_metadata(metadata, domain)
    if not metrics_mod.is_missing_candidate(prediction):
        return True, "not applicable: gold operation was offered"
    escalated = metrics_mod._escalated(prediction)
    verb = "correctly escalated" if escalated else "did not escalate"
    return escalated, f"missing-candidate case {verb} under this policy"


VERDICTS = {
    "right_action": right_action,
    "wrong_mutating": wrong_mutating,
    "correct_abstain_escalate": correct_abstain_escalate,
    "missing_candidate_handled": missing_candidate_handled,
}


# ---------------------------------------------------------------------------
# deepeval wrappers (lazy)
# ---------------------------------------------------------------------------

_METRIC_CLASSES: dict[str, type] = {}


def _metric_classes() -> dict[str, type]:
    """One ``BaseMetric`` subclass per verdict, built on first use (imports deepeval)."""
    if _METRIC_CLASSES:
        return _METRIC_CLASSES
    from deepeval.metrics import BaseMetric

    class _ExactMetric(BaseMetric):
        """An exact, deterministic check: score 1.0 passes, 0.0 fails (threshold 1.0)."""

        verdict_name = ""

        def __init__(self, domain: Domain):
            self.domain = domain
            self.threshold = 1.0
            self.async_mode = False
            self.score = None
            self.success = None
            self.reason = None
            self.error = None

        @property
        def __name__(self) -> str:  # noqa: N802 - DeepEval's metric-name protocol
            return self.verdict_name

        def measure(self, test_case, *args, **kwargs) -> float:
            self.error = None
            success, reason = VERDICTS[self.verdict_name](test_case.metadata or {}, self.domain)
            self.success = success
            self.reason = reason
            self.score = 1.0 if success else 0.0
            return self.score

        async def a_measure(self, test_case, *args, **kwargs) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool | None:
            if self.error is not None:
                self.success = False
                return False
            return self.success

    for name in METRIC_NAMES:
        _METRIC_CLASSES[name] = type(
            "".join(part.title() for part in name.split("_")) + "Metric",
            (_ExactMetric,),
            {"verdict_name": name},
        )
    return _METRIC_CLASSES


def build_metrics(domain: Domain) -> list:
    """Fresh instances of every exact metric, in :data:`METRIC_NAMES` order (imports deepeval)."""
    classes = _metric_classes()
    return [classes[name](domain) for name in METRIC_NAMES]


def build_test_case(trace: Trace, policy: Mapping[str, Any] | str, domain: Domain):
    """One ``LLMTestCase`` for *trace* replayed through *policy* (imports deepeval).

    ``input`` is a case-id placeholder, never the case text.
    """
    from deepeval.test_case import LLMTestCase

    metadata = case_metadata(trace, policy, domain)
    return LLMTestCase(
        input=f"case:{trace.case_id}",
        actual_output=metadata["final_decision"],
        expected_output=metrics_mod.expected_label(metadata["expected"]),
        additional_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Corpus metrics and the evaluate() wrapper
# ---------------------------------------------------------------------------


def policy_applied_predictions(
    traces: Sequence[Trace], policy: Mapping[str, Any] | str, domain: Domain
) -> list[Prediction]:
    """Every trace's policy-applied prediction, in trace order."""
    policy_dict = resolve_policy(policy)
    return [
        apply_policy_to_prediction(policy_dict, prediction_from_trace(trace), domain)
        for trace in traces
    ]


def corpus_metrics(
    traces: Sequence[Trace], policy: Mapping[str, Any] | str, domain: Domain
) -> dict:
    """:func:`metrics_bridge.compute` over the policy-applied predictions (no deepeval)."""
    return metrics_bridge.compute(policy_applied_predictions(traces, policy, domain), domain)


@dataclass(frozen=True)
class EvaluationOutcome:
    """DeepEval's own per-case result, alongside the corpus figures."""

    deepeval_result: Any
    corpus_metrics: dict


def _refuse_inside_repo(path: Path, what: str) -> None:
    if inside_git_worktree(path):
        raise DeepevalLayerError(
            f"{what} {path} is inside a git worktree; deepeval output must go to a "
            f"private run directory outside the repository"
        )


def evaluate_traces(
    traces: Sequence[Trace],
    policy: Mapping[str, Any] | str,
    domain: Domain,
    *,
    results_folder: str | Path,
    metrics: Sequence[Any] | None = None,
) -> EvaluationOutcome:
    """Score *traces* replayed through *policy* with DeepEval, and report corpus metrics.

    Runs synchronously and never calls a model or the network: every metric
    is a pure function of recorded fields. ``results_folder`` (outside any git
    worktree) receives DeepEval's ``test_run_*.json`` and its local state.
    """
    from deepeval.evaluate import evaluate
    from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig

    policy_dict = resolve_policy(policy)
    metric_list = list(metrics) if metrics is not None else build_metrics(domain)
    test_cases = [build_test_case(trace, policy_dict, domain) for trace in traces]

    results_folder = Path(results_folder).resolve()
    _refuse_inside_repo(results_folder, "results_folder")
    run_dir = results_folder / ".deepeval-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(run_dir):
        deepeval_result = evaluate(
            test_cases,
            metric_list,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(
                results_folder=str(results_folder), print_results=False, show_indicator=False
            ),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
    return EvaluationOutcome(
        deepeval_result=deepeval_result,
        corpus_metrics=corpus_metrics(traces, policy_dict, domain),
    )
