"""Provider protocol and base class for release-gate reference-model calls.

Every concrete provider adapter (an OpenAI-compatible host, the scriptable
fake, ...) implements the :class:`Provider` protocol. Adapters subclass
:class:`BaseProvider` rather than implementing the protocol from scratch so
two rules cannot be skipped by an individual adapter:

1. **Redaction is not optional.** Every outgoing case text and prompt passes
   through :func:`jev_factory.release.scan.redact` before an adapter's own
   transport code ever sees it.
2. **The held-out guard runs before any network call.** A request whose split
   tag is ``heldout`` or ``heldout-mc`` is refused in this base class, so no
   adapter can accidentally send a sealed case.

``BaseProvider`` enforces both by making :meth:`BaseProvider.submit_sync` a
non-overridable entry point that calls the ``_send_sync`` the subclass
implements; the public method does the guard and the redaction, then hands a
*redacted* request to the subclass.

Reading provider API keys is deliberately *not* this module's job for literal
values: :func:`read_api_key` only ever accepts an **environment variable
name** and reads it from ``os.environ`` at call time, so an adapter cannot be
constructed with a hard-coded key.
"""

from __future__ import annotations

import dataclasses
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jev_factory.release.scan import redact

from .errors import Outcome

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/providers/base.py",
    "commit": "9debdc6",
    "adaptations": [
        "line 50: nvsh.redact.redact -> jev_factory.release.scan.redact (same bytes-in,"
        " bytes-out contract)",
        "INTERFACES narrowed to {'choice'}: the Track A tool-call interface and the judge"
        " panel's free-text interface are deferred (c14)",
        "batch API removed (submit_batch/find_batch/poll_batch/fetch_batch, BatchHandle,"
        " BatchStatus, BatchLookupUnresolved): deferred per c14",
        "CallRequest.history, tool_choice_forced, native_turn and the per-field JSON"
        " redaction helpers (lines 125-160, 368-470) removed: they served only the Track A"
        " multi-round loop; case_text and prompt are redacted as text, as before",
        "kept verbatim: read_api_key, ProviderCapabilities, CallResult, ReplyText,"
        " visible_text, the held-out guard",
    ],
    "licence": "Apache-2.0",
}

#: The answer interfaces a request can use: one constrained choice among lettered
#: candidates (the jev-like scorer interface).
INTERFACES = frozenset({"choice"})

#: Split tags that must never reach a network call: the sealed held-out set and
#: its missing-candidate slice.
HELDOUT_SPLIT_TAGS = frozenset({"heldout", "heldout-mc"})


class HeldoutSplitRefused(Exception):
    """Raised when a request's split tag is heldout/heldout-mc, before any network call."""


class MissingProviderKey(Exception):
    """Raised by :func:`read_api_key` when the named env var is unset/empty."""


def read_api_key(env_var_name: str) -> str:
    """Read a provider API key from the environment variable named ``env_var_name``.

    Adapters call this instead of accepting a literal key argument, so a key
    can only ever come from the process environment, never from a committed
    literal or a manifest value itself.
    """
    value = os.environ.get(env_var_name)
    if not value:
        raise MissingProviderKey(
            f"environment variable {env_var_name!r} is not set; provider keys "
            "are read only from environment variable names, never literals"
        )
    return value


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do, so callers don't have to special-case names.

    - ``logprobs``: the provider can return per-candidate token probabilities
      (needed for the calibration metrics);
    - ``batch``: an async batch API. Always ``False`` here: batch APIs are not
      ported (deferred), and the runner refuses a manifest that asks for one;
    - ``reasoning``: the provider accepts a reasoning-effort parameter.
    """

    logprobs: bool
    batch: bool
    reasoning: bool


@dataclass(frozen=True)
class CallRequest:
    """One planned call: a case routed at a specific provider/model.

    ``split`` is the case's split tag and is checked by the held-out guard
    before anything else happens. ``case_text`` and ``prompt`` carry case
    content and are redacted before a subclass ever sees them.
    """

    case_id: str
    split: str
    case_text: str
    prompt: str = ""
    offered_candidates: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    interface: str = "choice"

    def __post_init__(self) -> None:
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {sorted(INTERFACES)}: {self.interface!r}")


@dataclass(frozen=True)
class CallResult:
    """The outcome of one call.

    - ``candidates``: the full candidate -> probability distribution **only**
      when the provider actually returned logprobs; ``None`` otherwise. Never
      estimated.
    - ``raw``: the exact provider response bytes (what the ledger caches).
    - ``response_id``: the provider's own id for this response.
    - ``returned_model``: the model id the provider *reported*.
    - ``usage``: token counts where reported, ints only.
    """

    case_id: str
    outcome: Outcome
    answer: str | None = None
    reason: str = ""
    provider: str = ""
    model_id: str | None = None
    candidates: dict[str, float] | None = None
    raw: bytes = b""
    response_id: str = ""
    returned_model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    interface: str = "choice"

    def __post_init__(self) -> None:
        if not isinstance(self.raw, bytes):
            raise TypeError("CallResult.raw must be bytes (the exact provider response)")
        for name, value in self.usage.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"usage[{name!r}] must be an int, got {value!r}")
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {sorted(INTERFACES)}: {self.interface!r}")
        if self.candidates is not None:
            for name, value in self.candidates.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"candidates[{name!r}] must be a probability, got {value!r}")


@dataclass(frozen=True)
class ReplyText:
    """The visible text of one cached answer and whether the provider cut it short.

    ``truncated`` is the provider's own signal that the answer was cut at the
    output budget (chat-completions ``finish_reason: length``); a truncated
    reply is never read as an answer.
    """

    text: str
    truncated: bool = False


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def visible_text(text: object) -> str:
    """*text* without inline ``<think>...</think>`` reasoning, stripped."""
    if not isinstance(text, str):
        return ""
    return _THINK_BLOCK.sub("", text).strip()


@runtime_checkable
class Provider(Protocol):
    """The provider-agnostic surface every adapter (and the fake) presents."""

    name: str
    capabilities: ProviderCapabilities

    def submit_sync(self, request: CallRequest) -> CallResult: ...

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read a cached ``CallResult.raw`` for *request* exactly as a fresh answer."""

    def reply_text(self, raw: bytes) -> ReplyText:
        """The visible text of a cached answer and whether it was cut."""


def _redact_text(text: str) -> str:
    """Round-trip ``text`` through the jev redactor (bytes in/out)."""
    return redact(text.encode("utf-8", errors="surrogateescape")).decode(
        "utf-8", errors="surrogateescape"
    )


def _redacted_request(request: CallRequest) -> CallRequest:
    return dataclasses.replace(
        request,
        case_text=_redact_text(request.case_text),
        prompt=_redact_text(request.prompt),
    )


class BaseProvider(ABC):
    """Common enforcement every concrete provider adapter inherits.

    Subclasses implement ``_send_sync`` (the real transport). They never
    override :meth:`submit_sync`, which keeps the held-out guard and the
    redaction choke point in one place.
    """

    name: str
    capabilities: ProviderCapabilities

    def _refuse_heldout(self, request: CallRequest) -> None:
        if request.split in HELDOUT_SPLIT_TAGS:
            raise HeldoutSplitRefused(
                f"{self.name}: refusing case {request.case_id!r} "
                f"(split={request.split!r}) before any network call"
            )

    def submit_sync(self, request: CallRequest) -> CallResult:
        self._refuse_heldout(request)
        return self._send_sync(_redacted_request(request))

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read a cached answer exactly as the adapter first parsed it. Sends nothing."""
        raise NotImplementedError(f"{self.name}: this provider cannot re-read a cached answer")

    def reply_text(self, raw: bytes) -> ReplyText:
        """The visible text of a cached answer and whether the provider cut it. Sends nothing."""
        raise NotImplementedError(f"{self.name}: this provider cannot read a reply's text")

    @abstractmethod
    def _send_sync(self, request: CallRequest) -> CallResult:
        """Send one already-guarded, already-redacted request."""
