"""SystemOne client and probability-only response parser (distillery, issue #18).

``POST /v1/systemone`` takes ``{model, state, questions: {qid: {type, instructions?,
criteria}}}`` and answers each question with per-option ``probabilities`` (choice, score)
or ``noul`` = P(true). This module reads **only** those numbers. The vendor's scalar
certainty field is defined differently per model and is not a probability, so it is
never read; nor is the reported argmax (``choice``) or expected ``score``, which the
distribution already determines.

A question's answer is a distribution only if it covers exactly the offered options and
sums to 1 within ``SUM_TOLERANCE``; anything else is an error result with a message,
never a distribution. Records are plain mappings ``{id, state, questions}``; gold is never
put on the wire. Stdlib only; base URLs must be localhost (``measure.preflight``).

Score answers may key ``probabilities`` by criteria text, by level index ("0".."n-1", what
strands-decider sends) or as a list; the ``legend`` is never read.

The response envelope is not formally specified upstream. The answers are looked up under
``answers`` or ``results`` when present, else at the top level, keyed by question id.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from jev_factory.cli._errors import EXIT_ENV_ERROR, CliError
from jev_factory.measure.preflight import _NoRedirect, require_localhost

SYSTEMONE_PATH = "/v1/systemone"
SUM_TOLERANCE = 0.01
DEFAULT_TIMEOUT = 120.0
MODEL_HEADER = "X-Lobes-Model"


@dataclass(frozen=True)
class QuestionResult:
    """One question's parsed answer: a distribution / P(true), or an error."""

    qid: str
    kind: str
    distribution: dict[str, float] | None = None
    p_true: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "kind": self.kind,
            "distribution": self.distribution,
            "p_true": self.p_true,
            "error": self.error,
        }


@dataclass(frozen=True)
class CallResult:
    """Parsed answers plus provenance (model id, request config, response body sha256)."""

    record_id: str
    model: str
    request_config: dict[str, Any]
    response_sha256: str | None
    questions: list[QuestionResult] = field(default_factory=list)
    error: str | None = None


def build_body(record: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    """The /v1/systemone request body for *record*; gold and metadata are not sent."""
    questions: dict[str, Any] = {}
    for qid, spec in record["questions"].items():
        item: dict[str, Any] = {"type": spec["type"]}
        if spec.get("instructions") is not None:
            item["instructions"] = spec["instructions"]
        item["criteria"] = spec.get("criteria")  # noul questions may carry none: send null
        questions[qid] = item
    return {"model": model, "state": record["state"], "questions": questions}


def _offered(spec: Mapping[str, Any]) -> list[str]:
    criteria = spec.get("criteria")
    if isinstance(criteria, Mapping):
        return [str(k) for k in criteria]
    if isinstance(criteria, (list, tuple)):
        return [str(k) for k in criteria]
    return []


def _prob(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _raw_probabilities(
    spec: Mapping[str, Any], answer: Any, offered: list[str]
) -> Mapping[Any, Any]:
    """The answer's probabilities keyed by option name (score lists and level indices mapped)."""
    raw = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if isinstance(raw, (list, tuple)) and spec["type"] == "score":
        if len(raw) != len(offered):
            raise ValueError(f"expected {len(offered)} probabilities, got {len(raw)}")
        raw = dict(zip(offered, raw))
    if not isinstance(raw, Mapping):
        raise ValueError("no 'probabilities' in the answer")
    keys = {str(k) for k in raw}
    index = [str(i) for i in range(len(offered))]
    if spec["type"] == "score" and keys != set(offered) and keys & set(index):
        # strands_decider keys score probabilities by level index ("0".."n-1"); the i-th key
        # is the record's i-th level. Missing, extra or out-of-range indices fall through to
        # the same fail-closed checks in _distribution.
        raw = {(offered[int(k)] if k in index else str(k)): v for k, v in raw.items()}
    return raw


def _distribution(spec: Mapping[str, Any], answer: Any) -> dict[str, float]:
    """Per-option probabilities over exactly the offered options, or ``ValueError``."""
    offered = _offered(spec)
    if len(offered) < 2:
        raise ValueError("fewer than 2 offered options")
    raw = _raw_probabilities(spec, answer, offered)
    keys = {str(k) for k in raw}
    if missing := [o for o in offered if o not in keys]:
        raise ValueError(f"missing option(s): {missing}")
    if extra := sorted(keys - set(offered)):
        raise ValueError(f"extra option(s): {extra}")
    by_key = {str(k): v for k, v in raw.items()}
    out: dict[str, float] = {}
    for option in offered:
        p = _prob(by_key[option])
        if p is None:
            raise ValueError(f"probability for {option!r} is not a number in [0, 1]")
        out[option] = p
    total = sum(out.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"probabilities sum to {total:.4f}, not 1 +/- {SUM_TOLERANCE}")
    return out


def _parse_question(qid: str, spec: Mapping[str, Any], answer: Any) -> QuestionResult:
    kind = str(spec.get("type"))
    try:
        if answer is None:
            raise ValueError("question missing from the response")
        if kind == "noul":
            p = _prob(answer.get("noul")) if isinstance(answer, Mapping) else None
            if p is None:
                raise ValueError("'noul' is not a number in [0, 1]")
            return QuestionResult(qid, kind, p_true=p)
        if kind not in ("choice", "score"):
            raise ValueError(f"unknown question type {kind!r}")
        return QuestionResult(qid, kind, distribution=_distribution(spec, answer))
    except ValueError as exc:
        return QuestionResult(qid, kind, error=str(exc))


def parse_response(record: Mapping[str, Any], response: Any) -> list[QuestionResult]:
    """One result per question of *record*, in record order."""
    answers: Any = response
    if isinstance(response, Mapping):
        for key in ("answers", "results"):
            if isinstance(response.get(key), Mapping):
                answers = response[key]
                break
    if not isinstance(answers, Mapping):
        answers = {}
        why = "response is not a JSON object"
    else:
        why = None
    results = [
        _parse_question(qid, spec, answers.get(qid)) for qid, spec in record["questions"].items()
    ]
    if why:
        results = [QuestionResult(r.qid, r.kind, error=why) for r in results]
    return results


class SystemOneClient:
    """Stdlib ``urllib`` client for one localhost /v1/systemone endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        auth_env: str | None = None,
        model_header: str | None = None,
    ):
        """*auth_env* is the NAME of an env var holding a bearer key; it is read here and
        only the name is ever recorded. *model_header* is sent as ``X-Lobes-Model``."""
        require_localhost(base_url)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.auth_env = auth_env or None
        self.model_header = model_header or None
        self._key = self._read_key(self.auth_env)

    @staticmethod
    def _read_key(auth_env: str | None) -> str | None:
        if auth_env is None:
            return None
        key = os.environ.get(auth_env)
        if not key:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"environment variable {auth_env} is not set (teacher bearer key)",
                remediation=f"export {auth_env} before running; only its name is recorded",
            )
        return key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key}"
        if self.model_header is not None:
            headers[MODEL_HEADER] = self.model_header
        return headers

    def ask(self, record: Mapping[str, Any]) -> CallResult:
        """POST *record*; transport and parse problems become error results, not raises."""
        body = build_body(record, model=self.model)
        data = json.dumps(body, sort_keys=True).encode()
        config = {
            "url": self.base_url + SYSTEMONE_PATH,
            "timeout": self.timeout,
            "body_sha256": hashlib.sha256(data).hexdigest(),
        }
        if self.auth_env is not None:
            config["auth_env"] = self.auth_env  # the variable's NAME, never its value
        if self.model_header is not None:
            config["model_header"] = self.model_header
        failed = f"{self.base_url}{SYSTEMONE_PATH}"
        request = urllib.request.Request(
            config["url"], data=data, method="POST", headers=self._headers()
        )
        opener = urllib.request.build_opener(_NoRedirect)  # a redirect could leave localhost
        try:
            with opener.open(request, timeout=self.timeout) as response:  # nosec B310 - localhost
                raw = response.read()
        except (OSError, ValueError) as exc:  # URLError, HTTPError, timeouts
            failed = f"request to {failed} failed: {exc}"
            return self._failure(record, config, None, failed)
        digest = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            return self._failure(record, config, digest, f"response is not JSON: {exc}")
        return CallResult(
            str(record["id"]), self.model, config, digest, parse_response(record, payload)
        )

    def _failure(self, record, config, digest, message) -> CallResult:
        questions = [
            QuestionResult(qid, str(spec.get("type")), error=message)
            for qid, spec in record["questions"].items()
        ]
        return CallResult(str(record["id"]), self.model, config, digest, questions, message)
