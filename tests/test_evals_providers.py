"""Release-gate providers: error taxonomy, the base-class guards, the fake, OpenAI-compatible.

Ported from nvsh evals/tool_jev/tests/test_providers_base.py and
test_provider_openai_compat.py (choice interface only; batch and Track A
cases are deferred with those features).
"""

from __future__ import annotations

import json

import pytest

from jev_factory.evals.cases import Case
from jev_factory.evals.providers import base, errors, fake, openai_compat
from jev_factory.evals.providers.base import CallRequest, ProviderCapabilities
from jev_factory.evals.providers.errors import Outcome
from jev_factory.evals.request import build_choice_request
from tests.fixtures.toy_domain import DOMAIN


def _case(case_id="c-1", split="test", text="turn on the lamps in the kitchen", candidates=None):
    return Case(
        id=case_id,
        split=split,
        text=text,
        candidates=candidates,
        expect={"operation": "lamp_on", "args": {"room": "kitchen"}},
        read_only=False,
    )


def _request(**kwargs):
    return build_choice_request(_case(**kwargs), DOMAIN)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,reason,retryable",
    [
        (402, "insufficient_credit", True),
        (429, "rate_limited", True),
        (408, "timeout", True),
        (503, "network_or_server_error", True),
        (400, "request_rejected:bad_request", False),
        (401, "request_rejected:auth_failed", False),
    ],
)
def test_status_codes_classify_as_pending_stops(status, reason, retryable):
    c = errors.classify_transport("openai_compat", status_code=status)
    assert c.outcome is Outcome.PENDING
    assert c.stop
    assert c.reason == reason
    assert c.retryable is retryable
    assert c.rejected is (not retryable)


def test_named_error_type_wins_over_status_and_unknown_stays_pending():
    c = errors.classify_transport(
        "openai_compat", status_code=400, error_type="unsupported_parameter"
    )
    assert c.reason == "request_rejected:unsupported_parameter"
    unknown = errors.classify_transport("openai_compat", status_code=418)
    assert unknown.outcome is Outcome.PENDING
    assert unknown.retryable
    assert unknown.reason == "unrecognized_infra_condition:http_418"
    with pytest.raises(KeyError):
        errors.classify_transport("nope", status_code=400)


def test_answer_classification_is_structural():
    assert errors.classify_answer(None, malformed=True).reason == "malformed"
    assert errors.classify_answer("I cannot", refused=True).reason == "refusal"
    assert errors.classify_answer("  ").reason == "empty_answer"
    assert errors.classify_answer("x", ("a", "b")).reason == "outside_offered_set"
    ok = errors.classify_answer("I cannot determine that; run the status check")
    assert ok.outcome is Outcome.OK


def test_stop_messages_name_the_kind_of_stop():
    rejected = errors.stop_message("m", errors.rejected("bad"), 3)
    assert "request rejected" in rejected
    assert "3 call(s)" in rejected
    transient = errors.stop_message(
        "p", errors.classify_transport("openai_compat", status_code=429), 2
    )
    assert "transient" in transient
    assert "2 call(s)" in transient


# ---------------------------------------------------------------------------
# base: held-out guard, redaction, keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["heldout", "heldout-mc"])
def test_heldout_split_is_refused_before_any_send(split):
    provider = fake.FakeProvider(script=[("answer", "D")])
    request = CallRequest(case_id="h", split=split, case_text="sealed")
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_sync(request)
    assert provider.received == []
    assert provider.remaining_script() == 1


def test_every_outgoing_text_passes_the_jev_redactor():
    provider = fake.FakeProvider(script=[("answer", "A")])
    secret = "sk-" + "a" * 30
    request = CallRequest(
        case_id="c",
        split="test",
        case_text=f"use key {secret} please",
        prompt=f"Authorization: Bearer {'b' * 20}",
        offered_candidates=("A",),
    )
    provider.submit_sync(request)
    sent = provider.received[0]
    assert secret not in sent.case_text
    assert "<REDACTED:openai_key>" in sent.case_text
    assert "<REDACTED:authorization_header>" in sent.prompt


def test_read_api_key_takes_a_name_never_a_literal(monkeypatch):
    monkeypatch.delenv("EVALS_TEST_KEY", raising=False)
    with pytest.raises(base.MissingProviderKey):
        base.read_api_key("EVALS_TEST_KEY")
    monkeypatch.setenv("EVALS_TEST_KEY", "value")
    assert base.read_api_key("EVALS_TEST_KEY") == "value"


def test_call_request_and_result_validate_their_shape():
    with pytest.raises(ValueError):
        CallRequest(case_id="c", split="test", case_text="x", interface="tool_call")
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=Outcome.OK, raw="text")
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=Outcome.OK, usage={"x": 1.5})
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=Outcome.OK, candidates={"a": "high"})


def test_visible_text_drops_inline_reasoning():
    assert base.visible_text("<think>hmm</think> D ") == "D"
    assert base.visible_text("<think>never closed") == ""
    assert base.visible_text(None) == ""


# ---------------------------------------------------------------------------
# fake
# ---------------------------------------------------------------------------


def test_fake_answers_failures_and_infra_stops():
    provider = fake.FakeProvider(
        script=[("answer", "D"), ("answer", "Z"), "refusal", "malformed", "402", "400", "network"]
    )
    request = _request()
    assert provider.submit_sync(request).outcome is Outcome.OK
    assert provider.submit_sync(request).reason == "outside_offered_set"
    assert provider.submit_sync(request).reason == "refusal"
    assert provider.submit_sync(request).reason == "malformed"
    for reason in ("insufficient_credit", "request_rejected:bad_request", "network_loss"):
        with pytest.raises(fake.FakeProviderError) as info:
            provider.submit_sync(request)
        assert info.value.classification.reason == reason
    with pytest.raises(fake.ScriptExhausted):
        provider.submit_sync(request)


def test_fake_rereads_its_raw_answer_and_reply_text():
    provider = fake.FakeProvider(
        script=[fake.ScriptedOutcome("answer", answer="D", text="D", truncated=True)]
    )
    request = _request()
    result = provider.submit_sync(request)
    again = provider.result_from_raw(request, result.raw)
    assert again.answer == "D"
    assert again.response_id == result.response_id
    assert provider.reply_text(result.raw).truncated
    not_an_answer = json.dumps({"kind": "402"}).encode()
    with pytest.raises(ValueError):
        provider.result_from_raw(request, not_an_answer)


def test_fake_without_logprobs_never_returns_a_distribution():
    provider = fake.FakeProvider(
        capabilities=ProviderCapabilities(logprobs=False, batch=False, reasoning=False),
        script=[fake.ScriptedOutcome("answer", answer="D", candidates={"lamp_on": 1.0})],
    )
    request = _request()
    with pytest.raises(ValueError):
        provider.submit_sync(request)
    nonsense = fake.FakeProvider(script=["nonsense"])
    request = _request()
    with pytest.raises(ValueError):
        nonsense.submit_sync(request)


# ---------------------------------------------------------------------------
# openai_compat
# ---------------------------------------------------------------------------


def _reply(content="D", *, top=None, finish="stop", refusal=None, usage=None):
    choice = {"message": {"content": content, "refusal": refusal}, "finish_reason": finish}
    if top is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": content,
                    "logprob": 0.0,
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top.items()],
                }
            ]
        }
    body = {
        "id": "resp-1",
        "model": "vendor/model-2026",
        "choices": [choice],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
    return json.dumps(body).encode()


class Transport:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, json.loads(body)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _provider(kind="local", transport=None, logprobs=False, reasoning=False, **kwargs):
    return openai_compat.OpenAICompatProvider(
        kind,
        "vendor/model",
        api_key_env=kwargs.pop("api_key_env", None),
        capabilities=ProviderCapabilities(logprobs=logprobs, batch=False, reasoning=reasoning),
        transport=transport,
        **kwargs,
    )


def test_payload_is_the_canonical_content_and_answer_is_parsed():
    transport = Transport((200, _reply("D")))
    provider = _provider(transport=transport)
    request = _request()
    result = provider.submit_sync(request)
    url, headers, payload = transport.calls[0]
    assert url == "http://localhost:8001/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["messages"][0] == {"role": "system", "content": request.prompt}
    assert payload["messages"][1] == {"role": "user", "content": request.case_text}
    assert payload["max_tokens"] == 512
    assert "logprobs" not in payload
    assert result.outcome is Outcome.OK
    assert result.answer == "D"
    assert result.candidates is None
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 2, "reasoning_tokens": 1}
    assert result.returned_model == "vendor/model-2026"
    assert provider.host == "localhost"


def test_logprobs_give_a_distribution_only_when_every_label_came_back():
    request = _request()
    labels = request.params["labels"]
    full = {label: -1.0 for label in labels.values()}
    partial = dict(list(full.items())[:3])
    transport = Transport((200, _reply("D", top=full)), (200, _reply("D", top=partial)))
    provider = _provider(transport=transport, logprobs=True)
    first = provider.submit_sync(request)
    assert set(first.candidates) == set(labels)
    assert abs(sum(first.candidates.values()) - 1.0) < 1e-9
    assert transport.calls[0][2]["top_logprobs"] == 20
    assert provider.submit_sync(request).candidates is None


def test_openrouter_sends_its_reasoning_shape_and_denies_data_collection(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "k")
    transport = Transport((200, _reply("D")))
    provider = openai_compat.OpenAICompatProvider("openrouter", "vendor/model", transport=transport)
    provider.submit_sync(_request())
    _url, headers, payload = transport.calls[0]
    assert headers["Authorization"] == "Bearer k"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["provider"] == {"data_collection": "deny"}


def test_reasoning_is_sent_to_a_local_host_only_when_capable():
    transport = Transport((200, _reply("D")), (200, _reply("D")))
    _provider(transport=transport, reasoning=True).submit_sync(_request())
    _provider(transport=transport).submit_sync(_request())
    assert transport.calls[0][2]["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in transport.calls[1][2]


def test_refusal_malformed_and_http_errors():
    transport = Transport(
        (200, _reply(None, refusal="no")),
        (200, _reply("")),
        (200, b"not json"),
        (200, _reply("Q")),
        (402, json.dumps({"error": {"code": "insufficient_quota"}}).encode()),
        (400, json.dumps({"error": {"type": "unsupported_parameter"}}).encode()),
        openai_compat.TransportError("timeout"),
    )
    provider = _provider(transport=transport)
    request = _request()
    assert provider.submit_sync(request).reason == "refusal"
    assert provider.submit_sync(request).reason == "malformed"
    assert provider.submit_sync(request).reason == "malformed"
    assert provider.submit_sync(request).reason == "malformed"  # not an offered letter
    for reason in ("insufficient_credit", "request_rejected:unsupported_parameter", "timeout"):
        with pytest.raises(openai_compat.OpenAICompatInfraError) as info:
            provider.submit_sync(request)
        assert info.value.classification.reason == reason


def test_reply_text_reports_truncation_and_never_reasoning():
    provider = _provider()
    assert provider.reply_text(_reply("D", finish="length")).truncated
    assert provider.reply_text(_reply("<think>x</think>D")).text == "D"
    assert provider.reply_text(b"junk").text == ""
    assert provider.reply_text(json.dumps({"choices": []}).encode()).text == ""


@pytest.mark.parametrize(
    "kind,url",
    [
        ("local", "https://api.example.com/v1"),
        ("openrouter", "http://localhost:9000/v1"),
        ("nvidia", "ftp://integrate.api.nvidia.com/v1"),
    ],
)
def test_base_url_rules(kind, url):
    with pytest.raises(ValueError):
        openai_compat.OpenAICompatProvider(kind, "m", base_url=url)


def test_constructor_refuses_unknown_kinds_and_batch():
    with pytest.raises(ValueError):
        openai_compat.OpenAICompatProvider("openai", "m")
    with_batch = ProviderCapabilities(True, True, False)
    with pytest.raises(ValueError):
        openai_compat.OpenAICompatProvider("local", "m", capabilities=with_batch)


def test_rate_limiter_spaces_calls_without_sleeping():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = openai_compat.RateLimiter(60, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert slept == [1.0, 1.0]
    with pytest.raises(ValueError):
        openai_compat.RateLimiter(0)


def test_error_type_from_body_reads_code_then_type():
    assert openai_compat._error_type_from_body(b'{"error": {"code": "rate_limited"}}') == (
        "rate_limited"
    )
    assert openai_compat._error_type_from_body(b'{"error": {"type": "timeout"}}') == "timeout"
    assert openai_compat._error_type_from_body(b'{"error": "flat"}') is None
    assert openai_compat._error_type_from_body(b"junk") is None
