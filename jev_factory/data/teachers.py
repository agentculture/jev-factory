"""Teacher client: roles, JSON verdicts, content-hash response cache, Apache-only.

Three roles call one OpenAI-compatible gateway (stdlib ``urllib`` only):

* ``generator``  (worker,  Qwen3.6-35B-A3B)  rewrites and drafts requests;
* ``reviewer_a`` (senses,  Gemma-4-26B-A4B)  judges a candidate;
* ``reviewer_b`` (cortex,  Qwen3.8-27B)      judges, decides and corrects.

Replaces nvsh's free-text ``augment.parse_verdict`` blacklist of hedge words
(which rejected ``no-argument``, ``ambiguous`` and a mid-sentence ``but``)
with a JSON verdict validated against a schema. An empty or malformed reply
is retried twice, then recorded as an **error**: never as a reject, because
a reject throws a candidate away for good. Every request/response is cached
by content hash, so a killed run restarted re-sends nothing it already has.
Teachers must be listed Apache-2.0 and may not be a codex/agy/kiro endpoint.
The gateway key is read by env-var name through ``factory.secrets``.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory.config import RunConfig
from jev_factory.factory.secrets import Secret, read_secret, scrub

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/augment.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 177-194 RoleConfig and 702-740 chat_payload/_post_chat_completion kept in shape",
        "lines 834-858 retry-with-backoff reduced to a bounded attempt budget",
        "lines 925-975 parse_verdict hedge-word blacklist replaced by a JSON verdict schema",
        "roles configured by RunConfig keys and a secret env-var name, not NVSH_AUG_* env",
        "added the content-hash response cache, the Apache-2.0 list and the endpoint guard",
    ],
    "licence": "Apache-2.0",
}

ROLES = ("generator", "reviewer_a", "reviewer_b")
APACHE = "Apache-2.0"
RETRIES = 2
"""Empty or malformed replies are retried this many times (3 attempts total)."""

#: Teachers known to be Apache-2.0, by gateway alias (display name, licence).
DEFAULT_TEACHERS: dict[str, dict[str, str]] = {
    "worker": {"name": "Qwen3.6-35B-A3B", "licence": APACHE},
    "senses": {"name": "Gemma-4-26B-A4B", "licence": APACHE},
    "cortex": {"name": "Qwen3.8-27B", "licence": APACHE},
}
_BANNED_ENDPOINT = ("codex", "agy", "kiro")
_TOKEN = re.compile(r"[a-z0-9]+")
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

JSON_INSTRUCTION = (
    ' Reply with ONLY a JSON object: {"verdict": "yes" or "no", "reason": "<short reason>"}.'
)


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _banned(text: str) -> str | None:
    for token in _TOKEN.findall(text.lower()):
        for word in _BANNED_ENDPOINT:
            if token.startswith(word):
                return word
    return None


def check_endpoint(url: str) -> None:
    """Reject an endpoint whose host or path names codex, agy or kiro."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise _err("teacher endpoint must be an http(s) URL", "set aug_url to the gateway URL")
    hit = _banned(f"{parts.hostname} {parts.path}")
    if hit:
        raise _err(
            f"teacher endpoint refused: it names {hit!r}",
            "codex, agy and kiro endpoints are not allowed as teachers",
        )


def load_teacher_list(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, str]]:
    """Built-in teachers plus an optional ``{alias: {name, licence}}`` JSON file."""
    listed = {k: dict(v) for k, v in DEFAULT_TEACHERS.items()}
    if path is not None:
        try:
            extra = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise _err("teacher_models file is unreadable or not JSON") from None
        if not isinstance(extra, dict) or not all(
            isinstance(v, dict) and "licence" in v for v in extra.values()
        ):
            raise _err("teacher_models must map alias to {name, licence}")
        listed.update({str(k): {str(a): str(b) for a, b in v.items()} for k, v in extra.items()})
    return listed


def check_licence(model: str, listed: dict[str, dict[str, str]]) -> dict[str, str]:
    """Return the listing for ``model`` or refuse: only listed Apache-2.0 teachers."""
    entry = listed.get(model)
    if entry is None:
        raise _err(
            f"teacher model {model!r} is not in the teacher list",
            "list it in teacher_models with its licence; only Apache-2.0 is accepted",
        )
    if entry.get("licence") != APACHE:
        raise _err(
            f"teacher model {model!r} is listed as {entry.get('licence')!r}, not {APACHE}",
            "remove it: every teacher must be listed Apache-2.0",
        )
    hit = _banned(f"{model} {entry.get('name', '')}")
    if hit:
        raise _err(f"teacher model {model!r} names {hit!r}, which is not allowed")
    return entry


@dataclass(frozen=True)
class RoleConfig:
    """One role's gateway settings, all of which are recorded with its output."""

    role: str
    url: str
    model: str
    model_name: str
    licence: str
    max_tokens: int
    timeout: float
    temperature: float = 0.7
    reasoning_effort: str | None = None
    key: Secret | None = field(default=None, repr=False)

    def record(self) -> dict[str, Any]:
        """Provenance of this role: what an accepted entry names (never the key)."""
        return {
            "role": self.role,
            "model": self.model,
            "model_name": self.model_name,
            "licence": self.licence,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


def load_roles(
    config: RunConfig, *, environ: dict[str, str] | None = None
) -> dict[str, RoleConfig]:
    """Build the three roles from a run config, refusing non-Apache or banned ones."""
    url = config.require("aug_url")
    check_endpoint(url)
    listed = load_teacher_list(config.get("teacher_models"))
    key = read_secret(config, "aug_key_env", environ=environ)
    roles: dict[str, RoleConfig] = {}
    for role, model_key, budget_key in (
        ("generator", "teacher_generator_model", "teacher_generator_max_tokens"),
        ("reviewer_a", "teacher_reviewer_a_model", "teacher_reviewer_max_tokens"),
        ("reviewer_b", "teacher_reviewer_b_model", "teacher_reviewer_max_tokens"),
    ):
        model = config[model_key]
        entry = check_licence(model, listed)
        roles[role] = RoleConfig(
            role=role,
            url=url,
            model=model,
            model_name=entry.get("name", model),
            licence=entry["licence"],
            max_tokens=int(config[budget_key]),
            timeout=float(config["teacher_timeout"]),
            reasoning_effort=config.get("teacher_reasoning_effort"),
            key=key,
        )
    return roles


def chat_payload(role: RoleConfig, system: str, user: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": role.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": role.temperature,
        "max_tokens": role.max_tokens,
    }
    if role.reasoning_effort is not None:
        payload["chat_template_kwargs"] = {"reasoning_effort": role.reasoning_effort}
    return payload


def cache_key(role: RoleConfig, system: str, user: str) -> str:
    """Content hash of everything that shapes the reply (never the key or the URL)."""
    body = json.dumps(
        {"role": role.role, "payload": chat_payload(role, system, user)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class TeacherError(Exception):
    """A reply that cannot be used (empty, malformed, transport); retried, then recorded."""


def _post(role: RoleConfig, system: str, user: str) -> str:
    headers = {"Content-Type": "application/json"}
    if role.key:
        headers["Authorization"] = f"Bearer {role.key.reveal()}"
    request = urllib.request.Request(  # noqa: S310 - scheme checked by check_endpoint
        role.url,
        data=json.dumps(chat_payload(role, system, user)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=role.timeout) as response:  # nosec B310
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices or "message" not in choices[0]:
        raise TeacherError("malformed gateway reply: no choices")
    # Only message.content is the answer; a reasoning trace never is.
    return (choices[0]["message"].get("content") or "").strip()


# --------------------------------------------------------------------------
# verdicts: JSON against a schema
# --------------------------------------------------------------------------

_FENCE = "```"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def strip_fence(text: str) -> str:
    """*text* without a leading ```` ```lang ```` fence and a trailing ```` ``` ````.

    String operations, not a regex: a reply is model output of any length, and the
    regex this replaces backtracked quadratically on a long run of whitespace.
    """
    body = text.strip()
    if body.startswith(_FENCE):
        body = body[len(_FENCE) :].lstrip(_LETTERS)
    if body.endswith(_FENCE):
        body = body[: -len(_FENCE)]
    return body.strip()


_DEPTH_STEP = {"{": 1, "}": -1}


def _object_end(text: str, start: int) -> int | None:
    """The index of the ``}`` closing the ``{`` at *start* (string-aware), or None."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            # Inside a JSON string: the character after a backslash is escaped, so an
            # escaped quote never ends the string.
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        depth += _DEPTH_STEP.get(ch, 0)
        if ch == "}" and depth == 0:
            return i
    return None


def _first_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text`` (string-aware), or None."""
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end is not None:
            return text[start : end + 1]
        start = text.find("{", start + 1)
    return None


def parse_verdict(text: str) -> tuple[bool, str]:
    """Parse a reviewer reply into ``(accepted, reason)`` from JSON.

    Schema: an object with ``verdict`` equal to ``"yes"`` or ``"no"``
    (case-insensitive) and an optional string ``reason``. Words inside the
    reason (``no-argument``, ``ambiguous``, a mid-sentence ``but``) carry no
    weight. Empty, non-JSON or off-schema text raises :class:`TeacherError`:
    it is not a judgement, so it is never a reject.
    """
    stripped = strip_fence(text)
    if not stripped:
        raise TeacherError("empty reply")
    blob = _first_object(stripped)
    if blob is None:
        raise TeacherError("reply has no JSON object")
    try:
        obj = json.loads(blob)
    except ValueError:
        raise TeacherError("reply JSON is malformed") from None
    verdict = obj.get("verdict") if isinstance(obj, dict) else None
    if not isinstance(verdict, str) or verdict.strip().lower() not in ("yes", "no"):
        raise TeacherError("verdict must be 'yes' or 'no'")
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise TeacherError("reason must be a string")
    return verdict.strip().lower() == "yes", reason.strip()


@dataclass
class Outcome:
    """One teacher call: ``status`` is ``ok`` or ``error`` (never a reject)."""

    status: str
    role: dict[str, Any]
    text: str = ""
    accepted: bool | None = None
    reason: str = ""
    error: str = ""
    attempts: int = 0
    cached: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "teacher": self.role,
            "text": self.text,
            "accepted": self.accepted,
            "reason": self.reason,
            "error": self.error,
            "attempts": self.attempts,
            "cached": self.cached,
        }


Caller = Callable[[RoleConfig, str, str], str]


class TeacherClient:
    """Calls roles through the gateway with a content-hash cache on disk."""

    def __init__(
        self,
        roles: dict[str, RoleConfig],
        cache_dir: str | os.PathLike[str],
        *,
        caller: Caller | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: float = 1.0,
    ) -> None:
        self.roles = roles
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._caller = caller or _post
        self._sleep = sleep
        self._backoff = backoff
        self.sent = 0  # requests actually sent to the gateway by this client
        self._secrets = [r.key for r in roles.values() if r.key is not None]

    @classmethod
    def from_config(cls, config: RunConfig, **kw: Any) -> TeacherClient:
        cache = config.get("teacher_cache") or str(Path(config.require("work")) / "teacher-cache")
        return cls(load_roles(config), cache, **kw)

    # -- cache ------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> str | None:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))["response"]
        except (OSError, ValueError, KeyError, TypeError):
            return None  # absent or torn by a kill: treated as a miss

    def _cache_put(self, key: str, role: RoleConfig, response: str) -> None:
        # A unique temp name per writer: two threads caching one key must not share a file.
        tmp = self._path(key).with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"role": role.record(), "response": response}), encoding="utf-8")
        os.replace(tmp, self._path(key))

    # -- calls ------------------------------------------------------------
    def complete(
        self, role_name: str, system: str, user: str, parse: Callable[[str], Any] | None = None
    ) -> Outcome:
        """One role call. Invalid replies are retried twice, then an error outcome."""
        role = self.roles[role_name]
        key = cache_key(role, system, user)
        cached = self._cached_outcome(role, key, parse)
        if cached is not None:
            return cached
        last = ""
        for attempt in range(1, RETRIES + 2):
            outcome, last, retryable = self._attempt(role, system, user, parse, attempt)
            if outcome is not None:
                self._cache_put(key, role, outcome.text)
                return outcome
            if not retryable:
                break  # the request itself is wrong: retrying repeats it
            if attempt <= RETRIES:
                self._sleep(self._backoff * attempt)
        return Outcome("error", role.record(), error=last, attempts=min(attempt, RETRIES + 1))

    def _cached_outcome(
        self, role: RoleConfig, key: str, parse: Callable[[str], Any] | None
    ) -> Outcome | None:
        hit = self._cache_get(key)
        if hit is None:
            return None
        try:
            return self._ok(role, hit, parse, 0, True)
        except TeacherError:
            return None  # a cached reply the current schema refuses: ask again

    def _attempt(
        self,
        role: RoleConfig,
        system: str,
        user: str,
        parse: Callable[[str], Any] | None,
        attempt: int,
    ) -> tuple[Outcome | None, str, bool]:
        """``(outcome, "", True)`` on success, else ``(None, error, retryable)``."""
        try:
            self.sent += 1
            text = self._caller(role, system, user)
            return self._ok(role, text, parse, attempt, False), "", True
        except urllib.error.HTTPError as exc:
            return None, f"HTTP {exc.code}", exc.code in _TRANSIENT_STATUS
        except (
            TeacherError,
            OSError,
            http.client.HTTPException,
        ) as exc:  # URLError is an OSError
            return None, scrub(str(exc) or type(exc).__name__, self._secrets), True
        except ValueError:
            return None, "gateway reply is not JSON", True

    @staticmethod
    def _ok(role: RoleConfig, text: str, parse: Callable | None, attempts: int, cached: bool):
        if parse is None:
            if not text.strip():
                raise TeacherError("empty reply")
            return Outcome("ok", role.record(), text=text, attempts=attempts, cached=cached)
        accepted, reason = parse(text)
        return Outcome(
            "ok",
            role.record(),
            text=text,
            accepted=accepted,
            reason=reason,
            attempts=attempts,
            cached=cached,
        )

    def generate(self, system: str, user: str) -> Outcome:
        return self.complete("generator", system, user)

    def review(self, role_name: str, system: str, user: str) -> Outcome:
        """A reviewer verdict; ``accepted`` is None when the outcome is an error."""
        if role_name not in ("reviewer_a", "reviewer_b"):
            raise ValueError(f"not a reviewer role: {role_name!r}")
        return self.complete(role_name, system + JSON_INSTRUCTION, user, parse_verdict)
