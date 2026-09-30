"""OpenAI-compatible chat-completions adapter (OpenRouter, NVIDIA, local).

One :class:`OpenAICompatProvider`, parameterized by ``kind`` (one of
``"openrouter"``, ``"nvidia"``, ``"local"``), covers three hosts that all
speak the same ``POST /chat/completions`` shape:

========== ===================================== =====================
kind       default base URL                      default key env var
========== ===================================== =====================
openrouter https://openrouter.ai/api/v1          OPEN_ROUTER_API_KEY
nvidia     https://integrate.api.nvidia.com/v1   NGC_API_KEY
local      http://localhost:8001/v1              LOCAL_LLM_API_KEY
========== ===================================== =====================

These defaults are **code**, never manifest data. A caller may override
``base_url``/``api_key_env`` at construction time subject to one rule
(:func:`_validate_base_url`): ``kind="local"`` must resolve to a localhost
host, and the two hosted kinds must not.

No batch API: ``capabilities.batch`` is always ``False``.

The payload is built only from
:func:`jev_factory.evals.request.canonical_content`, so the system/user text
is byte-identical to what every other adapter sends. The model answers with
one label; the answer is its text, stripped, read by
:func:`jev_factory.evals.request.parse_choice`. When ``capabilities.logprobs``
is true the adapter asks for ``logprobs=true, top_logprobs=20`` and builds a
candidate distribution from the **first content token's** ``top_logprobs``
(never a reasoning token's) with
:func:`jev_factory.evals.request.distribution_from_logprobs`.
"""

from __future__ import annotations

import functools
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urlsplit

from .. import request as contract
from .base import (
    BaseProvider,
    CallRequest,
    CallResult,
    ProviderCapabilities,
    ReplyText,
    read_api_key,
    visible_text,
)
from .errors import Classification, classify_answer, classify_transport

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/providers/openai_compat.py",
    "commit": "9debdc6",
    "adaptations": [
        "choice interface only: the tool_call payload/answer path (tool_choice, tool_calls,"
        " history rendering in _chat_messages) is deferred with Track A (c14)",
        "line 92: classify_result is inlined from providers/openai.py (lines 333-351) for the"
        " choice interface; the OpenAI Responses and Anthropic adapters are not ported",
        "the four batch hooks raising NotImplementedError are dropped with the batch API",
        "the local kind's default key env var is a neutral LOCAL_LLM_API_KEY",
        "kept: base-URL validation, the injectable transport, RateLimiter, error-body"
        " classification, logprob distribution, usage parsing, reply_text truncation",
    ],
    "licence": "Apache-2.0",
}

#: The three OpenAI-compatible hosts this adapter knows how to talk to.
KINDS = ("openrouter", "nvidia", "local")

#: Code defaults, overridable per instance for the operator's private setup.
DEFAULT_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "local": "http://localhost:8001/v1",
}

#: Env var *names* (never literal keys) read via ``base.read_api_key``.
DEFAULT_API_KEY_ENV: dict[str, str | None] = {
    "openrouter": "OPEN_ROUTER_API_KEY",
    "nvidia": "NGC_API_KEY",
    "local": "LOCAL_LLM_API_KEY",
}

_LOCALHOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})

#: The classification table this adapter's transport errors are read against.
_PROVIDER_KIND = "openai_compat"

#: Error-body ``error.code``/``error.type`` strings forwarded to
#: ``errors.classify_transport`` as the *named* error type.
_RECOGNISED_ERROR_TYPES = frozenset(
    {
        "insufficient_quota",
        "budget_cap_reached",
        "rate_limited",
        "timeout",
        "network_error",
        "connection_reset",
        "invalid_request",
        "auth_failed",
        "model_not_found",
        "unsupported_parameter",
    }
)


def _validate_base_url(kind: str, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"base_url {base_url!r} must use http:// or https:// (got scheme {parsed.scheme!r})"
        )
    hostname = (parsed.hostname or "").lower()
    is_localhost = hostname in _LOCALHOST_NAMES
    if kind == "local" and not is_localhost:
        raise ValueError(
            f"kind='local' base_url {base_url!r} must resolve to a localhost host "
            f"(one of {sorted(_LOCALHOST_NAMES)}); non-localhost is only for "
            "kind='openrouter'/'nvidia'"
        )
    if kind != "local" and is_localhost:
        raise ValueError(
            f"kind={kind!r} base_url {base_url!r} resolves to localhost; the two hosted "
            "kinds (openrouter, nvidia) must use a non-localhost URL -- only kind='local' "
            "may point at localhost"
        )


# ---------------------------------------------------------------------------
# Transport: a small, injectable seam. Tests never make a real HTTP call.
# ---------------------------------------------------------------------------


class TransportError(Exception):
    """A connection-level failure; ``error_type`` is one of ``errors.py``'s names."""

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type
        super().__init__(error_type)


Transport = Callable[[str, dict, bytes], tuple[int, bytes]]


def _default_transport(
    url: str, headers: dict, body: bytes, *, timeout: float = 60.0
) -> tuple[int, bytes]:
    """Real HTTP transport used outside tests (stdlib only)."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        # The scheme is restricted to http/https by _validate_base_url.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except TimeoutError as exc:
        raise TransportError("timeout") from exc
    except urllib.error.URLError as exc:
        raise TransportError("network_error") from exc


class RateLimiter:
    """Spaces calls to at most ``requests_per_minute``; injectable clock and sleep."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: float | None = None
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the next call is allowed; each caller claims its own slot."""
        with self._lock:
            now = self._clock()
            slot = now if self._next_allowed is None else max(now, self._next_allowed)
            self._next_allowed = slot + self._interval
        if slot > now:
            self._sleep(slot - now)


class OpenAICompatInfraError(Exception):
    """A transport-level / non-2xx infrastructure condition, with its Classification."""

    def __init__(self, classification: Classification, provider: str, case_id: str) -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        super().__init__(f"{provider}: {classification.reason} on case {case_id!r}")


def _error_type_from_body(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, str) and code in _RECOGNISED_ERROR_TYPES:
        return code
    error_type = error.get("type")
    if isinstance(error_type, str) and error_type in _RECOGNISED_ERROR_TYPES:
        return error_type
    return None


def classify_result(
    answer: str | None,
    offered: tuple[str, ...] | None,
    labels: dict[str, str] | None,
    *,
    malformed: bool = False,
    refused: bool = False,
) -> Classification:
    """Outcome for one choice answer, via the request contract's single parser."""
    if malformed or refused or answer is None:
        return classify_answer(answer, None, malformed=malformed, refused=refused)
    if labels:
        return contract.parse_choice(answer, labels)[0]
    return classify_answer(answer, offered or None)


class OpenAICompatProvider(BaseProvider):
    """One OpenAI-compatible chat-completions adapter for three hosts.

    ``api_key_env`` is an environment variable *name*, read at call time;
    pass ``None`` for a local server that needs no credential.
    ``capabilities`` is data from the manifest, never hard-coded per model;
    ``capabilities.batch`` must be ``False``. ``transport`` is injectable
    (tests always inject a canned one). ``reasoning_param_name`` is the body
    field that carries ``params["reasoning"]`` to an nvidia/local host whose
    capabilities say it reasons (OpenRouter uses its own
    ``{"reasoning": {"effort": ...}}``); that field name is an unverified
    assumption carried over from nvsh.
    """

    def __init__(
        self,
        kind: str,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = ...,  # sentinel: "not given" differs from explicit None
        capabilities: ProviderCapabilities | None = None,
        transport: Transport | None = None,
        timeout_seconds: float = 60.0,
        rate_limiter: RateLimiter | None = None,
        reasoning_param_name: str = "reasoning_effort",
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.model = model
        self.name = f"{kind}:{model}"
        resolved_base_url = base_url or DEFAULT_BASE_URLS[kind]
        _validate_base_url(kind, resolved_base_url)
        self.base_url = resolved_base_url.rstrip("/")
        self._api_key_env = DEFAULT_API_KEY_ENV.get(kind) if api_key_env is ... else api_key_env
        self.capabilities = capabilities or ProviderCapabilities(
            logprobs=False, batch=False, reasoning=False
        )
        if self.capabilities.batch:
            raise ValueError(
                f"{self.name}: openai_compat has no batch API; capabilities.batch must be False"
            )
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport or functools.partial(
            _default_transport, timeout=self.timeout_seconds
        )
        self._rate_limiter = rate_limiter
        self._reasoning_param_name = reasoning_param_name
        self.provider_kind = _PROVIDER_KIND

    @property
    def host(self) -> str:
        """The network host this adapter sends case text to."""
        return urlsplit(self.base_url).hostname or self.base_url

    # -- payload -------------------------------------------------------------

    def _build_payload(self, request: CallRequest) -> dict:
        params = request.params or {}
        system_text, messages, _labels = contract.canonical_content(request)
        chat = [{"role": "system", "content": system_text}] if system_text else []
        chat.extend({"role": m["role"], "content": m["content"]} for m in messages)
        payload: dict = {"model": self.model, "messages": chat}
        max_output_tokens = params.get("max_output_tokens")
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if self.capabilities.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 20
        reasoning = params.get("reasoning")
        if reasoning:
            if self.kind == "openrouter":
                payload["reasoning"] = {"effort": reasoning}
            elif self.capabilities.reasoning:
                payload[self._reasoning_param_name] = reasoning
            # nvidia/local without capabilities.reasoning: omitted, never sent blind.
        if self.kind == "openrouter":
            payload["provider"] = {"data_collection": "deny"}
        return payload

    # -- response parsing ----------------------------------------------------

    def _classify_and_answer(
        self, request: CallRequest, response: dict
    ) -> tuple[str | None, dict[str, float] | None, bool, bool]:
        """Return (answer, candidates, malformed, refused)."""
        choices = response.get("choices") or []
        if not choices:
            return None, None, True, False
        message = choices[0].get("message") or {}
        if message.get("refusal"):
            return None, None, False, True
        content = message.get("content")
        answer = content.strip() if isinstance(content, str) else ""
        labels = contract.canonical_content(request)[2]
        candidates = None
        if self.capabilities.logprobs and labels:
            logprobs_block = choices[0].get("logprobs") or {}
            content_tokens = logprobs_block.get("content") or []
            if content_tokens:
                top: dict[str, float] = {}
                for entry in content_tokens[0].get("top_logprobs") or []:
                    token, logprob = entry.get("token"), entry.get("logprob")
                    if isinstance(token, str) and token not in top:
                        top[token] = logprob
                candidates = contract.distribution_from_logprobs(top, labels, tuple(labels))
        return (answer or None), candidates, not answer, False

    # -- BaseProvider hooks --------------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        body = json.dumps(self._build_payload(request)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            headers["Authorization"] = f"Bearer {read_api_key(self._api_key_env)}"
        try:
            status, raw = self._transport(f"{self.base_url}/chat/completions", headers, body)
        except TransportError as exc:
            classification = classify_transport(self.provider_kind, error_type=exc.error_type)
            raise OpenAICompatInfraError(classification, self.name, request.case_id) from exc
        if status >= 400:
            classification = classify_transport(
                self.provider_kind, status_code=status, error_type=_error_type_from_body(raw)
            )
            raise OpenAICompatInfraError(classification, self.name, request.case_id)
        return self.result_from_raw(request, raw)

    def reply_text(self, raw: bytes) -> ReplyText:
        """``message.content`` (never ``reasoning_content``) and ``finish_reason == "length"``."""
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ReplyText(text="")
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ReplyText(text="")
        message = choices[0].get("message") or {}
        return ReplyText(
            text=visible_text(message.get("content")),
            truncated=choices[0].get("finish_reason") == "length",
        )

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Parse one chat-completions response body (fresh, or cached by the ledger)."""
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response = None
        if not isinstance(response, dict):
            classification = classify_answer(None, malformed=True)
            return CallResult(
                case_id=request.case_id,
                outcome=classification.outcome,
                reason=classification.reason,
                provider=self.name,
                raw=raw,
                interface=request.interface,
            )
        answer, candidates, malformed, refused = self._classify_and_answer(request, response)
        classification = classify_result(
            answer,
            request.offered_candidates or None,
            contract.canonical_content(request)[2],
            malformed=malformed,
            refused=refused,
        )
        usage_raw = response.get("usage") or {}
        usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage_raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
        reasoning_tokens = (usage_raw.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            usage["reasoning_tokens"] = reasoning_tokens
        return CallResult(
            case_id=request.case_id,
            outcome=classification.outcome,
            answer=answer,
            reason=classification.reason,
            provider=self.name,
            model_id=self.model,
            candidates=candidates,
            raw=raw,
            response_id=str(response.get("id", "")),
            returned_model=response.get("model"),
            usage=usage,
            interface=request.interface,
        )
