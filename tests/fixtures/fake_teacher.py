"""A scripted stand-in for the teacher gateway: no network, no model.

``FakeGateway`` is the ``caller`` a real :class:`TeacherClient` takes, so the
tests still run the client's JSON-verdict parsing, retries and cache. Replies
are routed by role and by phrases the prompts carry; ``calls`` records every
request as ``(role, system, user)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from jev_factory.data.teachers import RoleConfig, TeacherClient

ROLE_NAMES = ("generator", "reviewer_a", "reviewer_b")


def roles() -> dict[str, RoleConfig]:
    return {
        name: RoleConfig(
            role=name,
            url="http://127.0.0.1:1/v1/chat/completions",
            model=f"{name}-model",
            model_name=f"{name}-name",
            licence="Apache-2.0",
            max_tokens=256,
            timeout=1.0,
        )
        for name in ROLE_NAMES
    }


def verdict(accept: bool, reason: str = "fine") -> str:
    return json.dumps({"verdict": "yes" if accept else "no", "reason": reason})


class FakeGateway:
    """Generator replies come from ``generator(user)``; the corrector (a free-text
    ``reviewer_b`` call) echoes the request unless ``corrector(text)`` is given;
    verdicts are yes unless ``reject(role, user)`` returns true."""

    def __init__(
        self,
        generator: Callable[[str], str] | None = None,
        corrector: Callable[[str], str] | None = None,
        reject: Callable[[str, str], bool] | None = None,
        reviewer_reply: Callable[[str, str], str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.generator = generator or (lambda user: "[]")
        self.corrector = corrector
        self.reject = reject or (lambda role, user: False)
        self.reviewer_reply = reviewer_reply

    def __call__(self, role: RoleConfig, system: str, user: str) -> str:
        self.calls.append((role.role, system, user))
        if role.role == "generator":
            return self.generator(user)
        if "copyedit" in system:
            text = user.split("Request to copyedit:\n", 1)[1]
            return self.corrector(text) if self.corrector else text
        if self.reviewer_reply is not None:
            return self.reviewer_reply(role.role, user)
        return verdict(not self.reject(role.role, user))

    def roles_called(self) -> set[str]:
        return {role for role, _, _ in self.calls}

    def users(self, role: str | None = None) -> list[str]:
        return [u for r, _, u in self.calls if role is None or r == role]


def make_client(tmp_path: Path, gateway: FakeGateway) -> TeacherClient:
    return TeacherClient(
        roles(), tmp_path / "teacher-cache", caller=gateway, sleep=lambda _s: None, backoff=0.0
    )
