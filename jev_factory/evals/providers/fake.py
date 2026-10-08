"""Scriptable fake provider for release-gate tests and no-secrets smoke runs.

``FakeProvider`` implements :class:`~jev_factory.evals.providers.base.Provider`
entirely in-process: no network, no subprocess, nothing executed. Every call
consumes the next entry from a script the caller supplies (or, in a subclass,
answers from the request itself), so a test can drive a provider through the
exact sequence of outcomes it needs (an answer, a refusal, a malformed reply,
a 402, a 429, a timeout) without touching a real API.

Every request the base class hands ``FakeProvider`` has already passed the
held-out guard and the redactor, and ``FakeProvider`` records each one in
``received`` so a test can assert on exactly what an adapter would have sent.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from dataclasses import dataclass, field
from typing import cast

from .base import (
    BaseProvider,
    CallRequest,
    CallResult,
    ProviderCapabilities,
    ReplyText,
    visible_text,
)
from .errors import Classification, classify_answer, classify_transport

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/providers/fake.py",
    "commit": "9debdc6",
    "adaptations": [
        "batch half removed (_send_batch/_find_batch/_check_batch/_collect_batch,"
        " DuplicateSubmitRef, mark_unresolved, submitted_refs): batch APIs deferred (c14)",
        "the 'batch_expired' scripted kind is dropped with the batch API",
        "default capabilities are logprobs=True, batch=False (no batch API exists here)",
        "kept: ScriptedOutcome, FakeProviderError, ScriptExhausted, _resolve,"
        " result_from_raw and reply_text",
    ],
    "licence": "Apache-2.0",
}

#: Scripted outcome kinds FakeProvider understands.
ANSWER_KINDS = frozenset({"answer"})
ANSWER_FAILURE_KINDS = frozenset({"refusal", "malformed"})
_STATUS_BY_KIND = {
    "402": 402,
    "429": 429,
    "timeout": 408,
    "400": 400,
    "401": 401,
    "403": 403,
    "404": 404,
    "422": 422,
}
_ERROR_TYPE_BY_KIND = {
    "network": "network_error",
    "reset": "connection_reset",
    "invalid_request": "invalid_request",
    "auth_failed": "auth_failed",
    "model_not_found": "model_not_found",
    "unsupported_parameter": "unsupported_parameter",
}
#: Transient stops (retryable) and permanent request rejections (not retryable).
TRANSIENT_KINDS = frozenset({"402", "429", "timeout", "network", "reset"})
REJECTED_KINDS = (frozenset(_STATUS_BY_KIND) | frozenset(_ERROR_TYPE_BY_KIND)) - TRANSIENT_KINDS
INFRA_KINDS = TRANSIENT_KINDS | REJECTED_KINDS
ALL_KINDS = ANSWER_KINDS | ANSWER_FAILURE_KINDS | INFRA_KINDS


@dataclass(frozen=True)
class ScriptedOutcome:
    """One scripted outcome. ``kind`` is one of :data:`ALL_KINDS`.

    ``candidates`` scripts a logprob distribution (only allowed when the
    fake's capabilities say ``logprobs=True``). ``usage`` scripts reported
    token counts. ``raw`` overrides the synthetic response bytes. ``text``
    scripts the reply's visible words and ``truncated`` the provider's
    cut-at-budget signal; :meth:`FakeProvider.reply_text` reads both back.
    """

    kind: str
    answer: str | None = None
    candidates: dict[str, float] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: bytes | None = None
    text: str | None = None
    truncated: bool = False


class FakeProviderError(Exception):
    """Raised when a scripted infrastructure outcome (402/429/timeout/...) fires.

    Carries the :class:`~jev_factory.evals.providers.errors.Classification`
    (always ``outcome=PENDING, stop=True``) and the provider name.
    """

    def __init__(self, classification: Classification, provider: str, case_id: str) -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        super().__init__(f"{provider}: {classification.reason} on case {case_id!r}")


class ScriptExhausted(AssertionError):
    """Raised when a FakeProvider call runs out of scripted outcomes."""


def _coerce(entry) -> ScriptedOutcome:
    if isinstance(entry, ScriptedOutcome):
        return entry
    if isinstance(entry, tuple):
        kind, answer = entry
        return ScriptedOutcome(kind=kind, answer=answer)
    return ScriptedOutcome(kind=entry)


class FakeProvider(BaseProvider):
    """In-process, fully scriptable :class:`Provider` double.

    *script* entries are consumed one per call, in order: a
    :class:`ScriptedOutcome`, a ``(kind, answer)`` tuple, or a bare kind
    string (for answer-less kinds such as ``"402"``). :meth:`queue` appends
    more after construction (e.g. to model a top-up mid-test).
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        capabilities: ProviderCapabilities | None = None,
        script: list | None = None,
        provider_kind: str = "openai_compat",
        model: str = "fake-model",
    ) -> None:
        self.name = name
        self.model = model
        self.capabilities = capabilities or ProviderCapabilities(
            logprobs=True, batch=False, reasoning=False
        )
        self.provider_kind = provider_kind
        self._script: deque = deque(_coerce(entry) for entry in (script or []))
        #: Every request handed to this provider, post-guard, post-redaction.
        self.received: list[CallRequest] = []
        self._response_counter = 0

    def queue(self, kind: str, answer: str | None = None, **extra) -> None:
        """Append one more scripted outcome (e.g. to simulate a top-up)."""
        self._script.append(ScriptedOutcome(kind=kind, answer=answer, **extra))

    def remaining_script(self) -> int:
        return len(self._script)

    def _next_outcome(self, case_id: str) -> ScriptedOutcome:
        if not self._script:
            raise ScriptExhausted(
                f"{self.name}: fake provider script exhausted before case {case_id!r}"
            )
        return self._script.popleft()

    def _result(
        self, request: CallRequest, outcome: ScriptedOutcome, classification: Classification
    ) -> CallResult:
        if outcome.candidates is not None and not self.capabilities.logprobs:
            raise ValueError(
                f"{self.name}: scripted a candidate distribution but capabilities say "
                "logprobs=False; a provider without logprobs never returns one"
            )
        self._response_counter += 1
        response_id = f"{self.name}-resp-{self._response_counter}"
        raw = outcome.raw
        if raw is None:
            raw = json.dumps(
                {
                    "id": response_id,
                    "model": self.model,
                    "answer": outcome.answer,
                    "kind": outcome.kind,
                    "candidates": outcome.candidates,
                    "text": outcome.text,
                    "truncated": outcome.truncated,
                },
                sort_keys=True,
            ).encode("utf-8")
        return CallResult(
            case_id=request.case_id,
            outcome=classification.outcome,
            answer=outcome.answer,
            reason=classification.reason,
            provider=self.name,
            candidates=dict(outcome.candidates) if outcome.candidates is not None else None,
            raw=raw,
            response_id=response_id,
            returned_model=self.model,
            usage=dict(outcome.usage),
            interface=request.interface,
        )

    def _resolve(self, request: CallRequest, outcome: ScriptedOutcome) -> CallResult:
        if outcome.kind in ANSWER_KINDS:
            classification = classify_answer(outcome.answer, request.offered_candidates or None)
            return self._result(request, outcome, classification)
        if outcome.kind == "refusal":
            # Structural refusal (the provider's own refusal signal), whatever the text says.
            classification = classify_answer(outcome.answer, refused=True)
            return self._result(request, outcome, classification)
        if outcome.kind == "malformed":
            classification = classify_answer(outcome.answer, malformed=True)
            return self._result(request, outcome, classification)
        if outcome.kind in _STATUS_BY_KIND:
            classification = classify_transport(
                self.provider_kind, status_code=_STATUS_BY_KIND[outcome.kind]
            )
        elif outcome.kind in _ERROR_TYPE_BY_KIND:
            classification = classify_transport(
                self.provider_kind, error_type=_ERROR_TYPE_BY_KIND[outcome.kind]
            )
        else:
            raise ValueError(f"unknown scripted outcome kind: {outcome.kind!r}")
        raise FakeProviderError(classification, self.name, request.case_id)

    # -- BaseProvider hooks --------------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        self.received.append(request)
        outcome = self._next_outcome(request.case_id)
        return self._resolve(request, outcome)

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read the fake's own synthetic raw answer (a scripted ``raw`` override cannot be)."""
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("kind") not in ANSWER_KINDS | ANSWER_FAILURE_KINDS:
            raise ValueError(f"{self.name}: not a raw answer this fake wrote")
        outcome = ScriptedOutcome(
            kind=doc["kind"],
            answer=doc.get("answer"),
            candidates=doc.get("candidates"),
            raw=raw,
            text=doc.get("text"),
            truncated=bool(doc.get("truncated")),
        )
        result = self._resolve(request, outcome)
        return cast(
            CallResult,
            dataclasses.replace(
                result, response_id=doc.get("id", ""), returned_model=doc.get("model")
            ),
        )

    def reply_text(self, raw: bytes) -> ReplyText:
        """The scripted ``text``/``truncated`` of a raw answer this fake wrote."""
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            return ReplyText(text="")
        return ReplyText(text=visible_text(doc.get("text")), truncated=bool(doc.get("truncated")))
