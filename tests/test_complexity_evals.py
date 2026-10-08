"""Characterization tests for the release-gate functions refactored under d17 (Sonar S3776).

Each test pins the current behaviour (outputs, exceptions and their message
text, printed lines, files written) of a function whose cognitive complexity
was paid down, so the refactor is provably behaviour-neutral. They cover the
branches the end-to-end tests in ``test_evals_*.py`` leave unexercised.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from jev_factory.evals import __main__ as evals_main
from jev_factory.evals import deepeval_layer, report
from jev_factory.evals import run as runner
from jev_factory.evals import runplan
from jev_factory.evals.cases import Case
from jev_factory.evals.ledger import DONE, INVALID, PENDING, CachedResponse, Entry
from jev_factory.evals.manifest import (
    ENV_MANIFEST_PATH,
    Budget,
    ManifestError,
    Reference,
    _parse_budgets,
    load_manifest,
)
from jev_factory.evals.providers import openai_compat
from jev_factory.evals.providers.base import HeldoutSplitRefused, ProviderCapabilities
from jev_factory.evals.request import build_choice_request
from tests.evals_support import EXTRA_LOCAL, Fakes, setup_private
from tests.fixtures.toy_domain import DOMAIN

REF = "openrouter/vendor/fake-sync"

# ---------------------------------------------------------------------------
# __main__.main
# ---------------------------------------------------------------------------


def _call_main(argv, monkeypatch, **kwargs):
    monkeypatch.delenv(ENV_MANIFEST_PATH, raising=False)
    monkeypatch.delenv(evals_main.ENV_RUN_DIR, raising=False)
    out: list[str] = []
    code = evals_main.main(argv, out=out.append, **kwargs)
    return code, out


def _record_start(monkeypatch, name, outcome=None, error=None):
    calls: list[tuple[tuple, dict]] = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["out"]("a message")
        if error is not None:
            raise error
        return outcome or runner.StepOutcome(runner.STATUS_COMPLETE, [], exit_code=0)

    monkeypatch.setattr(runner, name, fake)
    return calls


def test_main_status_prints_rendered_lines(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "status", lambda run_dir: {"run_dir": str(run_dir)})
    monkeypatch.setattr(runner, "render_status", lambda doc: ["line one", doc["run_dir"]])
    code, out = _call_main(["status", "--run-dir", str(tmp_path)], monkeypatch)
    assert code == runner.EXIT_OK
    assert out == ["line one", str(tmp_path)]


def test_main_status_json_prints_the_document(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "status", lambda run_dir: {"b": 1, "a": 2})
    code, out = _call_main(["status", "--json", "--run-dir", str(tmp_path)], monkeypatch)
    assert code == runner.EXIT_OK
    assert out == [json.dumps({"b": 1, "a": 2}, indent=2, sort_keys=True)]


def test_main_needs_a_run_dir_and_a_manifest(monkeypatch, tmp_path):
    code, out = _call_main(["run"], monkeypatch)
    assert code == runner.EXIT_USER
    assert out == ["error: --run-dir is required (or set JEV_EVALS_RUN_DIR)"]
    code, out = _call_main(["run", "--run-dir", str(tmp_path)], monkeypatch)
    assert code == runner.EXIT_USER
    assert out == ["error: --manifest is required (or set JEV_EVALS_MANIFEST)"]


def test_main_run_passes_every_flag_and_the_clock(monkeypatch, tmp_path):
    calls = _record_start(
        monkeypatch, "start", runner.StepOutcome(runner.STATUS_STOPPED, [], exit_code=4)
    )
    clock = lambda: 1.0  # noqa: E731
    argv = ["run", "--run-dir", str(tmp_path), "--manifest", "m.toml", "--run-id", "r"]
    argv += ["--date", "2026-01-01", "--expand", "--no-deepeval"]
    code, out = _call_main(argv, monkeypatch, factory="F", env={"E": "1"}, clock=clock)
    assert code == 4
    assert out == ["a message"]
    args, kwargs = calls[0]
    assert args == (tmp_path, Path("m.toml"))
    assert kwargs == {
        "run_id": "r",
        "date": "2026-01-01",
        "expand": True,
        "use_deepeval": False,
        "env": {"E": "1"},
        "factory": "F",
        "out": kwargs["out"],
        "clock": clock,
    }


def test_main_run_without_a_clock_does_not_pass_one(monkeypatch, tmp_path):
    calls = _record_start(monkeypatch, "start")
    code, _out = _call_main(["run", "--run-dir", str(tmp_path), "--manifest", "m"], monkeypatch)
    assert code == 0
    assert "clock" not in calls[0][1]
    assert calls[0][1]["use_deepeval"] is True
    assert calls[0][1]["expand"] is False


def test_main_continue_passes_retry_and_deepeval(monkeypatch, tmp_path):
    calls = _record_start(monkeypatch, "step")
    argv = ["continue", "--run-dir", str(tmp_path), "--manifest", "m", "--retry-rejected"]
    code, out = _call_main(argv, monkeypatch)
    assert code == 0
    assert out == ["a message"]
    args, kwargs = calls[0]
    assert args == (tmp_path, Path("m"))
    assert kwargs["retry_rejected"] is True
    assert kwargs["use_deepeval"] is True
    assert set(kwargs) == {"retry_rejected", "use_deepeval", "env", "factory", "out"}


def test_main_smoke_passes_the_case_count_and_refuses_zero(monkeypatch, tmp_path):
    calls = _record_start(monkeypatch, "start")
    argv = ["smoke", "--run-dir", str(tmp_path), "--manifest", "m"]
    code, _out = _call_main([*argv, "--cases", "3"], monkeypatch)
    assert code == 0
    assert calls[0][1]["smoke_cases"] == 3
    assert set(calls[0][1]) == {"smoke_cases", "env", "factory", "out"}
    code, out = _call_main([*argv, "--cases", "0"], monkeypatch)
    assert code == runner.EXIT_USER
    assert out == ["error: --cases must be at least 1"]
    assert len(calls) == 1


def test_main_json_wraps_the_messages_in_one_document(monkeypatch, tmp_path):
    _record_start(monkeypatch, "start", runner.StepOutcome("stopped", ["x"], exit_code=4))
    argv = ["run", "--json", "--run-dir", str(tmp_path), "--manifest", "m"]
    code, out = _call_main(argv, monkeypatch)
    assert code == 4
    assert out == [
        json.dumps({"status": "stopped", "exit_code": 4, "messages": ["a message"]}, indent=2)
    ]


@pytest.mark.parametrize(
    "error,code,line",
    [
        (runner.StopAndAsk("decide"), runner.EXIT_ASK, "stop and ask: decide"),
        (runner.EnvError("no env"), runner.EXIT_ENV, "error: no env"),
        (runner.RunError("bad"), runner.EXIT_USER, "error: bad"),
        (
            KeyboardInterrupt(),
            runner.EXIT_INTERRUPTED,
            "interrupted: the ledger is consistent; `continue` resumes where this stopped",
        ),
    ],
)
def test_main_maps_each_error_to_its_exit_code(monkeypatch, tmp_path, error, code, line):
    _record_start(monkeypatch, "start", error=error)
    argv = ["run", "--run-dir", str(tmp_path), "--manifest", "m"]
    got, out = _call_main(argv, monkeypatch)
    assert got == code
    assert out == ["a message", line]
    got, out = _call_main([*argv, "--json"], monkeypatch)
    assert got == code
    assert out == [line]


def test_main_lets_an_unexpected_error_through(monkeypatch, tmp_path):
    _record_start(monkeypatch, "start", error=ValueError("boom"))
    argv = ["run", "--run-dir", str(tmp_path), "--manifest", "m"]
    with pytest.raises(ValueError, match="boom"):
        _call_main(argv, monkeypatch)


# ---------------------------------------------------------------------------
# manifest._parse_budgets
# ---------------------------------------------------------------------------


def _budget(**overrides):
    table = {"usd_cap": 1, "concurrency_cap": 2}
    table.update(overrides)
    return table


def test_parse_budgets_builds_one_budget_per_provider_in_order():
    got = _parse_budgets(
        {
            "nvidia": _budget(requests_per_minute=30, timeout_seconds=5),
            "local": _budget(usd_cap=0.5),
        },
        "budget",
    )
    assert got == (
        Budget("nvidia", 1.0, 2, 30.0, 5.0),
        Budget("local", 0.5, 2, None, 60.0),
    )
    assert isinstance(got[0].usd_cap, float)
    assert isinstance(got[0].requests_per_minute, float)
    assert _parse_budgets({}, "budget") == ()


@pytest.mark.parametrize(
    "raw,message",
    [
        ([], "budget must be a table"),
        ({"openai": _budget()}, "budget.openai: unknown provider 'openai' (must be one of"),
        ({"local": _budget(usd_cap=None)}, "budget.local: usd_cap must be a non-negative number"),
        ({"local": _budget(usd_cap=True)}, "budget.local: usd_cap must be a non-negative"),
        ({"local": _budget(usd_cap=-1)}, "budget.local: usd_cap must be a non-negative"),
        ({"local": _budget(usd_cap="1")}, "budget.local: usd_cap must be a non-negative"),
        ({"local": _budget(concurrency_cap=None)}, "concurrency_cap must be a positive integer"),
        ({"local": _budget(concurrency_cap=True)}, "concurrency_cap must be a positive integer"),
        ({"local": _budget(concurrency_cap=0)}, "concurrency_cap must be a positive integer"),
        ({"local": _budget(concurrency_cap=2.0)}, "concurrency_cap must be a positive integer"),
        (
            {"local": _budget(batch_discount=0.5)},
            "budget.local: batch_discount is not supported (batch APIs are deferred)",
        ),
        ({"local": _budget(requests_per_minute=0)}, "requests_per_minute must be a positive"),
        ({"local": _budget(requests_per_minute=True)}, "requests_per_minute must be a positive"),
        ({"local": _budget(requests_per_minute="5")}, "requests_per_minute must be a positive"),
        ({"local": _budget(timeout_seconds=-1)}, "timeout_seconds must be a non-negative number"),
    ],
)
def test_parse_budgets_refuses_bad_tables(raw, message):
    with pytest.raises(ManifestError) as caught:
        _parse_budgets(raw, "budget")
    assert message in str(caught.value)


def test_parse_budgets_unknown_provider_message_lists_the_allowed_ones():
    with pytest.raises(ManifestError) as caught:
        _parse_budgets({"openai": {}}, "budget")
    assert str(caught.value) == (
        "budget.openai: unknown provider 'openai' "
        "(must be one of ['local', 'nvidia', 'openrouter'])"
    )


def test_parse_budgets_checks_in_order_and_stops_at_the_first_bad_provider():
    raw = {"local": _budget(usd_cap=-1, concurrency_cap=0), "openai": {}}
    with pytest.raises(ManifestError, match="usd_cap"):
        _parse_budgets(raw, "b")


def test_parse_budgets_non_table_entry_is_a_manifest_error():
    # Fixed after the d17 report: it used to escape as AttributeError (a CLI traceback).
    with pytest.raises(ManifestError, match=r"budget\.local must be a table"):
        _parse_budgets({"local": 3}, "budget")


@pytest.mark.parametrize("field", ["usd_cap", "requests_per_minute"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_parse_budgets_refuses_a_non_finite_number(field, value):
    table = {"usd_cap": 1.0, "concurrency_cap": 1, field: value}
    with pytest.raises(ManifestError, match=field):
        _parse_budgets({"local": table}, "budget")


# ---------------------------------------------------------------------------
# openai_compat.OpenAICompatProvider._classify_and_answer
# ---------------------------------------------------------------------------


def _compat(logprobs=True):
    return openai_compat.OpenAICompatProvider(
        "local",
        "vendor/model",
        capabilities=ProviderCapabilities(logprobs=logprobs, batch=False, reasoning=False),
    )


def _choice_request(labels=True):
    case = Case(
        id="c-1",
        split="test",
        text="turn on the lamps in the kitchen",
        candidates=None,
        expect={"operation": "lamp_on"},
        read_only=False,
    )
    request = build_choice_request(case, DOMAIN)
    if not labels:
        request = dataclasses.replace(request, params={})
    return request


def _top(labels, skip=()):
    return [{"token": letter, "logprob": -1.0} for letter in labels.values() if letter not in skip]


@pytest.mark.parametrize(
    "response,expected",
    [
        ({}, (None, None, True, False)),
        ({"choices": []}, (None, None, True, False)),
        ({"choices": None}, (None, None, True, False)),
        ({"choices": [{}]}, (None, None, True, False)),
        ({"choices": [{"message": None}]}, (None, None, True, False)),
        ({"choices": [{"message": {"refusal": "no", "content": "A"}}]}, (None, None, False, True)),
        ({"choices": [{"message": {"refusal": "", "content": " A "}}]}, ("A", None, False, False)),
        ({"choices": [{"message": {"content": "   "}}]}, (None, None, True, False)),
        ({"choices": [{"message": {"content": ["A"]}}]}, (None, None, True, False)),
        (
            {"choices": [{"message": {"content": "A"}, "logprobs": None}]},
            ("A", None, False, False),
        ),
        (
            {"choices": [{"message": {"content": "A"}, "logprobs": {"content": []}}]},
            ("A", None, False, False),
        ),
        (
            {"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{}]}}]},
            ("A", None, False, False),
        ),
    ],
)
def test_classify_and_answer_without_a_distribution(response, expected):
    assert _compat()._classify_and_answer(_choice_request(), response) == expected


def test_classify_and_answer_reads_the_first_token_distribution():
    request = _choice_request()
    labels = request.params["labels"]
    tokens = [{"token": 7, "logprob": 0.0}, {"token": None, "logprob": 0.0}]
    tokens += _top(labels)
    first_letter = next(iter(labels.values()))
    tokens.append({"token": first_letter, "logprob": 0.0})  # a duplicate: the first one wins
    response = {
        "choices": [
            {
                "message": {"content": first_letter},
                "logprobs": {"content": [{"top_logprobs": tokens}, {"top_logprobs": []}]},
            }
        ]
    }
    answer, candidates, malformed, refused = _compat()._classify_and_answer(request, response)
    assert answer == first_letter
    assert malformed is False
    assert refused is False
    assert list(candidates) == list(labels)
    assert all(value == pytest.approx(1 / len(labels)) for value in candidates.values())


def test_classify_and_answer_partial_labels_give_no_distribution():
    request = _choice_request()
    labels = request.params["labels"]
    skip = {next(iter(labels.values()))}
    response = {
        "choices": [
            {
                "message": {"content": "B"},
                "logprobs": {"content": [{"top_logprobs": _top(labels, skip)}]},
            }
        ]
    }
    assert _compat()._classify_and_answer(request, response) == ("B", None, False, False)


def test_classify_and_answer_ignores_logprobs_without_the_capability_or_labels():
    request = _choice_request()
    labels = request.params["labels"]
    response = {
        "choices": [
            {"message": {"content": "B"}, "logprobs": {"content": [{"top_logprobs": _top(labels)}]}}
        ]
    }
    expected = ("B", None, False, False)
    assert _compat(logprobs=False)._classify_and_answer(request, response) == expected
    assert _compat()._classify_and_answer(_choice_request(labels=False), response) == expected


# ---------------------------------------------------------------------------
# report._render_slices
# ---------------------------------------------------------------------------


def _slices(permutation="not_run"):
    return {
        "read_only_vs_mutating": {
            "read_only": {"n": 3, "ece": 0.1234, "brier": 0.2, "missing_candidate": 0.5},
            "mutating": {"n": 0, "ece": None, "brier": None, "missing_candidate": None},
        },
        "candidate_count": {
            "2": {"n_rows": 4, "n_measurable": 3, "top1_rate": 0.75},
            "7": {"n_rows": 1, "n_measurable": 0, "top1_rate": None},
        },
        "confidence_bucket": [
            {"lower": 0.0, "upper": 0.5, "n": 2, "confidence": 0.31, "accuracy": 0.5},
            {"lower": None, "upper": 1.0, "n": 0, "confidence": None, "accuracy": None},
            {"lower": 0.5, "upper": None, "n": 1, "confidence": 0.9, "accuracy": None},
        ],
        "missing_candidate_split": {
            "missing": {"n_rows": 2, "n_measurable": 2, "top1_rate": 1.0},
            "complete": {"n_rows": 3, "n_measurable": 1, "top1_rate": None},
        },
        "semantic_vs_epistemic": {
            "semantic_escalation": {"n": 1, "N": 2, "rate": 0.5},
            "epistemic_abstention": {"n": 0, "N": 0, "rate": None},
        },
        "permutation": permutation,
    }


_SLICES_PAGE = """\
### cand (raw)

Read-only vs. mutating:

| slice | n | ECE | Brier | missing-candidate |
| --- | --- | --- | --- | --- |
| read_only | 3 | 0.123 | 0.200 | 50.0% |
| mutating | 0 | NM | NM | - |

Candidate count (offered candidates per case):

| candidates offered | rows | measurable | top-1 rate |
| --- | --- | --- | --- |
| 2 | 4 | 3 | 75.0% |
| 7 | 1 | 0 | - |

Confidence bucket:

| bucket | n | mean confidence | accuracy |
| --- | --- | --- | --- |
| 0.0 to 0.5 | 2 | 0.310 | 0.500 |
| n/a | 0 | - | - |
| n/a | 1 | 0.900 | - |

Missing-candidate split:

| bucket | rows | measurable | top-1 rate |
| --- | --- | --- | --- |
| missing | 2 | 2 | 100.0% |
| complete | 3 | 1 | - |

Semantic escalation vs. uncertainty abstention:

| kind | n | N | rate |
| --- | --- | --- | --- |
| semantic escalation | 1 | 2 | 50.0% |
| epistemic abstention | 0 | 0 | - |

Permutation slice: not run.

### base (harness)

Read-only vs. mutating:
"""


def test_render_slices_renders_every_table_per_row():
    rows = [
        {"subject": "cand", "variant": "raw", "slices": _slices()},
        {"subject": "base", "variant": "harness", "slices": _slices({"b": 1, "a": [0.5]})},
    ]
    lines = report._render_slices(rows)
    text = "\n".join(lines)
    assert text.startswith(_SLICES_PAGE.replace("NM", report.NM_MARKER))
    assert lines[-2] == 'Permutation slice: `{"a": [0.5], "b": 1}`'
    assert lines[-1] == ""
    assert len(lines) == 2 * len(report._render_slices(rows[:1]))


def test_render_slices_of_no_rows_is_empty():
    assert report._render_slices([]) == []


# ---------------------------------------------------------------------------
# run.Runner.run_sync (a Runner without __init__: the pool logic in isolation)
# ---------------------------------------------------------------------------


class _SyncRunner(runner.Runner):
    def __init__(self, budgets, *, blocked=(), unclaimable=()):  # no super(): no run dir
        self.plan = SimpleNamespace(manifest=SimpleNamespace(budget_for=budgets.get))
        self._blocked = set(blocked)
        self._unclaimable = set(unclaimable)
        self.events: list[tuple] = []
        self._events_lock = threading.Lock()

    def _event(self, *event):
        with self._events_lock:
            self.events.append(event)

    def blocked(self, model):
        self._event("blocked?", model.label)
        return model.label in self._blocked

    def claim(self, model, key, request):
        self._event("claim", key)
        return key not in self._unclaimable

    def _note_host(self, model):
        self._event("host", model.label)

    def _uncertain(self, key, why):
        self._event("uncertain", key, why)

    def _release(self, *keys):
        self._event("release", *keys)

    def apply_stop(self, model, classification, params):
        self._event("stop", model.label, classification.reason, dict(params))

    def record(self, model, key, result):
        self._event("record", model.label, key, result)


def _sync_model(label, kind, answer):
    def submit_sync(request):
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(request)
        return answer

    return SimpleNamespace(
        label=label, kind=kind, provider=SimpleNamespace(submit_sync=submit_sync)
    )


def _item(key):
    return runplan.WorkItem(key, SimpleNamespace(params={"max_output_tokens": 8}))


def test_run_sync_with_no_work_makes_no_progress():
    sync = _SyncRunner({})
    assert sync.run_sync([]) is False
    assert sync.events == []


def test_run_sync_sends_and_records_every_item_per_provider_pool():
    budgets = {"a": SimpleNamespace(concurrency_cap=2)}
    a = _sync_model("a/m", "a", "answer-a")
    b = _sync_model("b/m", "b", "answer-b")  # no budget: a pool of one
    sync = _SyncRunner(budgets)
    work = [(a, _item("k1")), (b, _item("k2")), (a, _item("k3"))]
    assert sync.run_sync(work) is True
    records = sorted(e for e in sync.events if e[0] == "record")
    assert records == [
        ("record", "a/m", "k1", "answer-a"),
        ("record", "a/m", "k3", "answer-a"),
        ("record", "b/m", "k2", "answer-b"),
    ]
    assert sorted(e for e in sync.events if e[0] == "host") == [
        ("host", "a/m"),
        ("host", "a/m"),
        ("host", "b/m"),
    ]


def test_run_sync_skips_blocked_and_unclaimable_items():
    a = _sync_model("a/m", "a", "x")
    b = _sync_model("b/m", "b", "y")
    sync = _SyncRunner({}, blocked={"b/m"}, unclaimable={"k1"})
    assert sync.run_sync([(a, _item("k1")), (b, _item("k2"))]) is False
    assert sorted(sync.events) == [
        ("blocked?", "a/m"),
        ("blocked?", "b/m"),
        ("claim", "k1"),
    ]


def test_run_sync_claims_nothing_past_the_round_deadline(monkeypatch):
    monkeypatch.setattr(runner, "ROUND_SECONDS", -1.0)
    sync = _SyncRunner({})
    assert sync.run_sync([(_sync_model("a/m", "a", "x"), _item("k1"))]) is False
    assert sync.events == []


@pytest.mark.parametrize(
    "error,events",
    [
        (
            TimeoutError("slow"),
            [
                ("uncertain", "k1", "sent but not answered (network_loss)"),
                ("stop", "a/m", "network_loss", {"max_output_tokens": 8}),
            ],
        ),
        (
            ConnectionRefusedError("down"),
            [("release", "k1"), ("stop", "a/m", "network_loss", {"max_output_tokens": 8})],
        ),
        (
            ValueError("odd"),
            [
                ("release", "k1"),
                ("stop", "a/m", "unexpected_error:ValueError", {"max_output_tokens": 8}),
            ],
        ),
    ],
)
def test_run_sync_classifies_a_failed_send_into_a_stop(error, events):
    sync = _SyncRunner({})
    assert sync.run_sync([(_sync_model("a/m", "a", error), _item("k1"))]) is False
    assert sync.events == [("blocked?", "a/m"), ("claim", "k1"), ("host", "a/m"), *events]


def test_run_sync_progress_survives_a_failed_send_beside_a_good_one():
    a = _sync_model("a/m", "a", "x")
    b = _sync_model("b/m", "b", ValueError("odd"))
    sync = _SyncRunner({})
    assert sync.run_sync([(a, _item("k1")), (b, _item("k2"))]) is True


@pytest.mark.parametrize("error", [HeldoutSplitRefused("sealed"), KeyboardInterrupt()])
def test_run_sync_reraises_what_must_not_be_classified(error):
    sync = _SyncRunner({})
    model = _sync_model("a/m", "a", error)
    item = _item("k1")
    with pytest.raises(type(error)):
        sync.run_sync([(model, item)])
    assert ("release", "k1") not in sync.events


def test_run_sync_raises_the_first_failure_after_draining_the_pools():
    barrier = threading.Barrier(2, timeout=10)

    def fail(request):
        barrier.wait()  # both sends are in flight before either fails
        raise HeldoutSplitRefused("sealed")

    a = _sync_model("a/m", "a", fail)
    sync = _SyncRunner({"a": SimpleNamespace(concurrency_cap=2)})
    pairs = [(a, _item("k1")), (a, _item("k2"))]
    with pytest.raises(HeldoutSplitRefused):
        sync.run_sync(pairs)
    assert sorted(e for e in sync.events if e[0] == "claim") == [("claim", "k1"), ("claim", "k2")]


def test_run_sync_cancels_what_has_not_started_after_a_failure():
    started = []

    def answer(request):
        started.append(request)
        raise HeldoutSplitRefused("sealed")

    a = _sync_model("a/m", "a", answer)
    sync = _SyncRunner({})
    pairs = [(a, _item(f"k{i}")) for i in range(20)]
    with pytest.raises(HeldoutSplitRefused):
        sync.run_sync(pairs)
    assert 1 <= len(started) < 20


# ---------------------------------------------------------------------------
# run.Runner.finalize (the DeepEval branch, with the layer stubbed)
# ---------------------------------------------------------------------------


def _full_run(tmp_path, args, fakes=None):
    manifest, run_dir, env = setup_private(tmp_path)
    out: list[str] = []
    code = evals_main.main(
        [*args, "--manifest", str(manifest), "--run-dir", str(run_dir)],
        factory=fakes or Fakes(),
        env=env,
        out=out.append,
    )
    return code, out, run_dir


def test_finalize_with_deepeval_writes_its_log_and_the_same_figures(tmp_path, monkeypatch, capsys):
    calls = []

    def evaluate_traces(traces, policy_json, domain, *, results_folder):
        print("deepeval chatter")
        calls.append(results_folder.name)
        figures = deepeval_layer.corpus_metrics(traces, policy_json, domain)
        return SimpleNamespace(corpus_metrics=figures)

    monkeypatch.setattr(deepeval_layer, "deepeval_available", lambda: True)
    monkeypatch.setattr(deepeval_layer, "evaluate_traces", evaluate_traces)
    code, out, run_dir = _full_run(tmp_path / "with", ["run", "--run-id", "r", "--date", "d"])
    assert code == runner.EXIT_OK, out
    assert "deepeval chatter" not in capsys.readouterr().out
    folders = sorted(p.name for p in (run_dir / runner.DEEPEVAL_DIR).iterdir())
    assert sorted(calls) == folders
    assert len(calls) == 5
    for folder in (run_dir / runner.DEEPEVAL_DIR).iterdir():
        assert (folder / "deepeval.log").read_text() == "deepeval chatter\n"
    code, _out, plain_dir = _full_run(
        tmp_path / "without", ["run", "--no-deepeval", "--run-id", "r", "--date", "d"]
    )
    assert code == runner.EXIT_OK
    with_deepeval = json.loads((run_dir / runner.RESULT_FILE).read_text())
    without = json.loads((plain_dir / runner.RESULT_FILE).read_text())
    assert with_deepeval.pop("deepeval") is True
    assert without.pop("deepeval") is False
    assert with_deepeval == without
    assert out[-1] == f"complete: {run_dir / runner.RESULT_FILE} and {run_dir / runner.PAGE_FILE}"
    manifest_doc = json.loads((run_dir / report.MANIFEST_FILENAME).read_text())
    assert manifest_doc["deepeval"] is True
    assert [s["kind"] for s in manifest_doc["subjects"]] == ["candidate", "candidate", "reference"]
    assert manifest_doc["subjects"][0]["artifact"]["repo_id"] == "example-org/cand"
    assert "artifact" not in manifest_doc["subjects"][2]


# ---------------------------------------------------------------------------
# run.Runner.write_smoke (a Runner without __init__ over an in-memory ledger)
# ---------------------------------------------------------------------------


class _Reply:
    name = "stub"

    def __init__(self, cut_raws=(), broken_raws=()):
        self.cut_raws = set(cut_raws)
        self.broken_raws = set(broken_raws)

    def reply_text(self, raw):
        if raw in self.broken_raws:
            raise ValueError("unreadable")
        return SimpleNamespace(truncated=raw in self.cut_raws)


class _SmokeRunner(runner.Runner):
    def __init__(self, run_dir, models, ledger, billed, stops, cases, full):  # no super()
        entries, cached = ledger
        self.run_dir = run_dir
        self.plan = SimpleNamespace(
            cases=cases, full_case_count=full, models={m.label: m for m in models}
        )
        self._spec_model = {(m.ref.provider, m.ref.model): m for m in models}
        self.ledger = SimpleNamespace(entries=lambda: entries, cached=cached.get)
        self.billing = SimpleNamespace(entries=lambda: billed)
        self.model_stops = stops
        self.state = {}
        self.persisted = 0
        self.lines: list[str] = []
        self.out = self.lines.append

    def _persist(self):
        self.persisted += 1


def _smoke_model(provider, model, reply, **ref):
    reference = Reference(
        provider=provider, model=model, usd_per_mtok_in=2.0, usd_per_mtok_out=10.0, **ref
    )
    return runplan.Model(ref=reference, provider=reply, host="h")


def _spec(model, case_id, **params):
    return {
        "provider": model.ref.provider,
        "model": model.ref.model,
        "case_id": case_id,
        "target": runplan.CHOICE_TARGET,
        "params": params or dict(model.knobs),
    }


def _cache(raw, **usage):
    return CachedResponse(raw=raw, model_id="m", response_id="r", usage=usage)


def test_write_smoke_counts_tokens_cost_truncation_and_projects_the_full_run(tmp_path):
    reply = _Reply(cut_raws={b"cut"}, broken_raws={b"broken"})
    a = _smoke_model("openrouter", "vendor/a", reply)
    b = _smoke_model("openrouter", "vendor/b", reply, max_output_tokens=64)
    c = _smoke_model("local", "c", reply, reasoning="high")
    entries = [
        Entry("k01", _spec(a, "c-1"), DONE),
        Entry("k02", _spec(a, "c-2"), INVALID, reason=runner.TRUNCATED),
        Entry("k03", _spec(a, "c-9"), DONE),  # not a smoke case
        Entry("k04", {"provider": "nvidia", "model": "gone", "case_id": "c-1"}, DONE),
        Entry("k05", _spec(a, "c-1"), PENDING),
        Entry("k06", _spec(a, "c-1", max_output_tokens=1), DONE),  # stale knobs
        Entry("k07", _spec(b, "c-1"), DONE),
        Entry("k08", _spec(b, "c-2"), INVALID, reason="unparsable"),
        Entry("k09", _spec(c, "c-1"), DONE),
        Entry("k10", _spec(c, "c-2"), DONE),
    ]
    cached = {
        "k01": _cache(b"ok", prompt_tokens=1000, completion_tokens=100, reasoning_tokens=7),
        "k02": _cache(b"ok", input_tokens=500, output_tokens=50),
        "k07": _cache(b"cut", prompt_tokens=10, completion_tokens=5),
        "k08": _cache(b"broken", prompt_tokens=20),
        "k10": _cache(b"ok", prompt_tokens=1, completion_tokens=1),
    }
    billed = [
        {"key": "k01", "kind": "answer", "cost_usd": 0.5},
        {"key": "k07", "kind": "reservation", "cost_usd": 99.0},
    ]
    stops = {"openrouter/vendor/b": {"kind": "truncation"}}
    cases = {"toy": (SimpleNamespace(id="c-1"), SimpleNamespace(id="c-2"))}
    smoke_runner = _SmokeRunner(tmp_path, [a, b, c], (entries, cached), billed, stops, cases, 7)
    smoke_runner.write_smoke()
    smoke = json.loads((tmp_path / runner.SMOKE_FILE).read_text())
    assert smoke == _SMOKE
    assert smoke_runner.state == {"smoke": smoke, "status": runner.STATUS_COMPLETE}
    assert smoke_runner.persisted == 1
    assert smoke_runner.lines == _SMOKE_LINES


def test_write_smoke_with_no_smoke_cases_projects_nothing(tmp_path):
    smoke_runner = _SmokeRunner(tmp_path, [], ([], {}), [], {}, {}, 5)
    smoke_runner.write_smoke()
    assert json.loads((tmp_path / runner.SMOKE_FILE).read_text()) == {
        "cases": 0,
        "full_run_cases": 5,
        "models": {},
        "providers": {},
    }
    assert smoke_runner.lines == []


def _smoke_row(provider, calls, invalid, truncated, tokens, cost, projected, **extra):
    row = {
        "provider": provider,
        "calls": calls,
        "invalid": invalid,
        "truncated": truncated,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "reasoning_tokens": tokens[2],
        "cost_usd": cost,
        "projected_full_run_usd": projected,
        "max_output_tokens": 512,
        "reasoning": "medium",
        "flag": "CAPPED" if truncated else "OK",
    }
    row.update(extra)
    return row


_SMOKE = {
    "cases": 2,
    "full_run_cases": 7,
    "models": {
        "local/c": _smoke_row("local", 2, 0, 0, (1, 1, 0), 1.2e-05, 0.0, reasoning="high"),
        "openrouter/vendor/a": _smoke_row("openrouter", 2, 1, 1, (1500, 150, 7), 0.5015, 1.7552),
        "openrouter/vendor/b": _smoke_row(
            "openrouter", 2, 1, 1, (30, 5, 0), 0.00011, 0.0004, max_output_tokens=64
        )
        | {"stop": "truncation"},
    },
    "providers": {
        "local": {"cost_usd": 1.2e-05, "projected_full_run_usd": 0.0},
        "openrouter": {"cost_usd": 0.50161, "projected_full_run_usd": 1.7556},
    },
}
_SMOKE_LINES = [
    "smoke local/c: OK calls=2 truncated=0 invalid=0 tokens in/out/reasoning=1/1/0 "
    "cost=$0.0000 projected full run=$0.00",
    "smoke openrouter/vendor/a: CAPPED calls=2 truncated=1 invalid=1 "
    "tokens in/out/reasoning=1500/150/7 cost=$0.5015 projected full run=$1.76",
    "smoke openrouter/vendor/b: CAPPED calls=2 truncated=1 invalid=1 "
    "tokens in/out/reasoning=30/5/0 cost=$0.0001 projected full run=$0.00",
]


def test_a_smoke_run_after_a_rejected_model_is_retried_reports_every_model(tmp_path):
    manifest, run_dir, env = setup_private(tmp_path, extra_refs=EXTRA_LOCAL)
    fakes = Fakes()
    out: list[str] = []
    fakes(type("R", (), {"provider": "local", "model": "fake-cut"}), None, env)
    fakes.get("local/fake-cut").fail = {1: "400"}
    paths = ["--manifest", str(manifest), "--run-dir", str(run_dir)]
    code = evals_main.main(
        ["smoke", "--cases", "2", *paths], factory=fakes, env=env, out=out.append
    )
    assert code == runner.EXIT_STOPPED, out
    fakes.get("local/fake-cut").fail = {}
    code = evals_main.main(
        ["continue", "--retry-rejected", *paths], factory=fakes, env=env, out=out.append
    )
    assert code == runner.EXIT_OK, out
    smoke = json.loads((run_dir / runner.SMOKE_FILE).read_text())
    assert sorted(smoke["models"]) == ["local/fake-cut", REF]
    assert smoke["cases"] == 2


# ---------------------------------------------------------------------------
# runplan._load_saved and runplan.build_plan
# ---------------------------------------------------------------------------


def _plan_inputs(tmp_path, **setup):
    manifest_path, _run_dir, env = setup_private(tmp_path, **setup)
    manifest = load_manifest(manifest_path)
    return manifest_path, manifest, env


def _saved_inputs(tmp_path):
    _path, manifest, env = _plan_inputs(tmp_path)
    entry = manifest.candidates[0]
    cs = manifest.case_sets[0]
    cases = runplan._load_cases(cs, runplan.load_run_domain(manifest, env), env)
    predictions = Path(env["JEV_EVALS_PRIVATE_ROOT"]) / "predictions" / "cand.jsonl"
    return entry, cs, cases, env, predictions


def _load_saved(entry, cs, cases, env):
    return runplan._load_saved(entry, cs, cases, "cand.toy-test", env)


def test_load_saved_reads_the_rows_in_case_order_with_the_artifact(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    lines = predictions.read_text().splitlines()
    data = "\n".join(["", "   ", *reversed(lines), json.dumps([1]), '"text"']) + "\n"
    predictions.write_text(data)
    saved = _load_saved(entry, cs, cases, env)
    assert [t.case_id for t in saved.traces] == [c.id for c in cases]
    assert {t.subject for t in saved.traces} == {"cand.toy-test"}
    assert {t.split for t in saved.traces} == {"test"}
    import hashlib

    assert saved.artifact == {
        "predictions_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "repo_id": "example-org/cand",
        "revision": "rev-1",
    }
    bare = dataclasses.replace(entry, repo_id=None, revision=None)
    assert _load_saved(bare, cs, cases, env).artifact == {
        "predictions_sha256": hashlib.sha256(data.encode()).hexdigest()
    }


def test_load_saved_with_no_row_for_the_set_is_none(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    predictions.write_text('{"id": "other"}\n')
    assert _load_saved(entry, cs, cases, env) is None


def test_load_saved_skips_a_row_whose_id_is_not_a_string(tmp_path):
    # Fixed after the d17 report: an unhashable id raised TypeError from the set lookup.
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    good = predictions.read_text().splitlines()
    predictions.write_text("\n".join(['{"id": [1]}', '{"id": {"a": 1}}', *good]) + "\n")
    saved = _load_saved(entry, cs, cases, env)
    assert [t.case_id for t in saved.traces] == [c.id for c in cases]


def test_load_saved_refuses_an_unreadable_file(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    predictions.unlink()
    with pytest.raises(runner.RunError) as caught:
        _load_saved(entry, cs, cases, env)
    assert str(caught.value).startswith(f"cand: cannot read saved predictions {predictions}: ")


def test_load_saved_refuses_a_line_that_is_not_json(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    predictions.write_text('\n{"id": "c-1"}\nnot json\n')
    with pytest.raises(runner.RunError) as caught:
        _load_saved(entry, cs, cases, env)
    assert str(caught.value) == f"cand: {predictions} line 3 is not JSON"


def test_load_saved_refuses_a_case_twice(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    predictions.write_text('{"id": "c-2"}\n{"id": "c-2"}\n')
    with pytest.raises(runner.RunError) as caught:
        _load_saved(entry, cs, cases, env)
    assert str(caught.value) == f"cand: {predictions} has case 'c-2' twice"


def test_load_saved_refuses_a_partial_replay(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    predictions.write_text(predictions.read_text().splitlines()[0] + "\n")
    with pytest.raises(runner.RunError) as caught:
        _load_saved(entry, cs, cases, env)
    assert str(caught.value) == (
        "cand: saved predictions for case set 'toy-test' miss 4 of 5 cases; "
        "a partial replay would bias every figure"
    )


def test_load_saved_checks_the_train_split(tmp_path):
    entry, cs, cases, env, _predictions = _saved_inputs(tmp_path)
    root = Path(env["JEV_EVALS_PRIVATE_ROOT"])
    missing = dataclasses.replace(entry, train_split="nope.json")
    with pytest.raises(runner.RunError) as caught:
        _load_saved(missing, cs, cases, env)
    assert str(caught.value).startswith("cand: cannot read its train split: ")
    (root / "train.json").write_text(json.dumps({"entries": [{"id": "c-3"}]}))
    overlapping = dataclasses.replace(entry, train_split="train.json")
    with pytest.raises(runner.RunError) as caught:
        _load_saved(overlapping, cs, cases, env)
    assert str(caught.value).startswith("cand: ")
    assert "c-3" in str(caught.value)
    (root / "train.json").write_text(json.dumps({"entries": [{"id": "t-1"}]}))
    clean = dataclasses.replace(entry, train_split="train.json")
    assert len(_load_saved(clean, cs, cases, env).traces) == 5


def test_load_saved_refuses_a_bad_prediction_row(tmp_path):
    entry, cs, cases, env, predictions = _saved_inputs(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[1]["outcome"] = "nonsense"
    predictions.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(runner.RunError) as caught:
        _load_saved(entry, cs, cases, env)
    assert str(caught.value).startswith("cand: case 'c-2': ")


def _subjects(plan):
    return [(s.name, s.kind, s.case_set.name, s.policies, s.model) for s in plan.subjects]


def test_build_plan_full_lists_saved_and_reference_subjects(tmp_path):
    _path, manifest, env = _plan_inputs(tmp_path, extra_refs=EXTRA_LOCAL)
    plan = runplan.build_plan(manifest, env, Fakes())
    policies = ("raw", "mutating-strict-example")
    assert _subjects(plan) == [
        ("cand.toy-test", "candidate", "toy-test", policies, None),
        ("cand.toy-heldout", "candidate", "toy-heldout", policies, None),
        ("openrouter.vendor-fake-sync.toy-test", "reference", "toy-test", ("raw",), REF),
        ("local.fake-cut.toy-test", "reference", "toy-test", ("raw",), "local/fake-cut"),
    ]
    assert sorted(plan.saved) == ["cand.toy-heldout", "cand.toy-test"]
    assert list(plan.models) == [REF, "local/fake-cut"]
    assert plan.models[REF].host == "openrouter.invalid"
    assert plan.smoke is False
    assert plan.full_case_count == 5
    assert plan.scope == {"mode": "full"}
    assert {name: len(cases) for name, cases in plan.cases.items()} == {
        "toy-test": 5,
        "toy-heldout": 2,
    }
    assert plan.world == {"home": "toy-home", "rooms": ["kitchen", "bedroom", "study", "hallway"]}
    assert runplan.build_plan(manifest, env, Fakes(), scope={"mode": "full"}).scope == {
        "mode": "full"
    }


def test_build_plan_skips_a_case_set_without_saved_rows(tmp_path):
    _path, manifest, env = _plan_inputs(tmp_path)
    predictions = Path(env["JEV_EVALS_PRIVATE_ROOT"]) / "predictions" / "cand.jsonl"
    rows = predictions.read_text().splitlines()
    predictions.write_text("\n".join(row for row in rows if '"h-' not in row))
    plan = runplan.build_plan(manifest, env, Fakes())
    assert [s.name for s in plan.subjects] == [
        "cand.toy-test",
        "openrouter.vendor-fake-sync.toy-test",
    ]


def test_build_plan_smoke_keeps_only_the_selection(tmp_path):
    _path, manifest, env = _plan_inputs(tmp_path)
    scope = {"mode": "smoke", "case_set": "toy-test", "case_ids": ["c-3", "c-1"]}
    plan = runplan.build_plan(manifest, env, Fakes(), scope=scope)
    assert plan.smoke is True
    assert plan.saved == {}
    assert plan.scope == scope
    assert plan.scope is not scope
    assert plan.full_case_count == 5
    assert {name: [c.id for c in cases] for name, cases in plan.cases.items()} == {
        "toy-test": ["c-3", "c-1"]
    }
    assert _subjects(plan) == [
        ("openrouter.vendor-fake-sync.toy-test", "reference", "toy-test", ("raw",), REF)
    ]
    empty = runplan.build_plan(
        manifest, env, Fakes(), scope={"mode": "smoke", "case_set": "toy-test"}
    )
    assert empty.cases == {"toy-test": ()}


@pytest.mark.parametrize(
    "scope,message",
    [
        (
            {"mode": "smoke", "case_set": "toy-heldout", "case_ids": []},
            "the smoke case set 'toy-heldout' is not in the manifest",
        ),
        ({"mode": "smoke"}, "the smoke case set None is not in the manifest"),
        (
            {"mode": "smoke", "case_set": "toy-test", "case_ids": ["c-1", "x", "y"]},
            "smoke cases ['x', 'y'] are no longer in 'toy-test'",
        ),
    ],
)
def test_build_plan_refuses_a_stale_smoke_selection(tmp_path, scope, message):
    _path, manifest, env = _plan_inputs(tmp_path)
    fakes = Fakes()
    with pytest.raises(runner.RunError) as caught:
        runplan.build_plan(manifest, env, fakes, scope=scope)
    assert str(caught.value) == message


def test_build_plan_refuses_an_unknown_policy_before_loading_predictions(tmp_path):
    manifest_path, _manifest, env = _plan_inputs(tmp_path)
    manifest_path.write_text(
        manifest_path.read_text().replace('"mutating-strict-example"', '"nope"')
    )
    (Path(env["JEV_EVALS_PRIVATE_ROOT"]) / "predictions" / "cand.jsonl").unlink()
    manifest = load_manifest(manifest_path)
    fakes = Fakes()
    with pytest.raises(runner.RunError) as caught:
        runplan.build_plan(manifest, env, fakes)
    assert str(caught.value) == "cand: unknown policy 'nope'"


def test_build_plan_refuses_a_reference_without_a_budget(tmp_path):
    manifest_path, _manifest, env = _plan_inputs(tmp_path)
    text = manifest_path.read_text().replace("[budget.openrouter]", "[budget.nvidia]")
    manifest_path.write_text(text)
    manifest = load_manifest(manifest_path)
    fakes = Fakes()
    with pytest.raises(runner.RunError) as caught:
        runplan.build_plan(manifest, env, fakes)
    assert str(caught.value) == "openrouter/vendor/fake-sync: no [budget.openrouter] table"


def test_build_plan_refuses_a_batch_adapter(tmp_path):
    _path, manifest, env = _plan_inputs(tmp_path)
    fakes = Fakes()
    fakes(type("R", (), {"provider": "openrouter", "model": "vendor/fake-sync"}), None, env)
    fakes.get(REF).capabilities = ProviderCapabilities(logprobs=True, batch=True, reasoning=True)
    with pytest.raises(runner.RunError) as caught:
        runplan.build_plan(manifest, env, fakes)
    assert str(caught.value) == "openrouter/vendor/fake-sync: batch adapters are not supported"


def test_build_plan_refuses_two_subjects_with_one_file_name(tmp_path):
    manifest_path, _manifest, env = _plan_inputs(tmp_path)
    text = manifest_path.read_text()
    extra = (
        '\n[[reference]]\nprovider = "openrouter"\nmodel = "vendor:fake-sync"\n'
        "usd_per_mtok_in = 1.0\n"
    )
    manifest_path.write_text(text.replace("\n[[case_set]]", extra + "\n[[case_set]]", 1))
    manifest = load_manifest(manifest_path)
    fakes = Fakes()
    with pytest.raises(runner.RunError) as caught:
        runplan.build_plan(manifest, env, fakes)
    assert str(caught.value) == "two subjects map to the same file name; rename a checkpoint"
