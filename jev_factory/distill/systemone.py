"""The /v1/systemone client, at its old distill path (issue #18).

The client and the probability-only parser live in :mod:`jev_factory.wire.systemone`, outside
the distillery, so the pointer backbone can use them without importing distill (deviation d1).
Every name is re-exported here as the very same object.
"""

from __future__ import annotations

from jev_factory.wire.systemone import (  # noqa: F401  (re-exported)
    DEFAULT_TIMEOUT,
    MODEL_HEADER,
    SUM_TOLERANCE,
    SYSTEMONE_PATH,
    CallResult,
    QuestionResult,
    SystemOneClient,
    build_body,
    parse_response,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "MODEL_HEADER",
    "SUM_TOLERANCE",
    "SYSTEMONE_PATH",
    "CallResult",
    "QuestionResult",
    "SystemOneClient",
    "build_body",
    "parse_response",
]
