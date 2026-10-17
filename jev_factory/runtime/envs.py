"""One pinned environment: a python path xor an image digest, plus a lockfile hash.

Shared by the distillery (:mod:`jev_factory.distill.envs`, which re-exports :class:`EnvSpec`)
and the pointer backbone's server launchers, so neither needs the other. Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

ROLES = ("teacher", "student", "trainer")

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCK = re.compile(r"^[0-9a-f]{64}$")


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


@dataclass(frozen=True)
class EnvSpec:
    """One pinned environment: a python path xor an image digest, plus a lockfile hash."""

    role: str
    lockfile_hash: str
    python: str | None = None
    image_digest: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise _refuse(f"unknown environment role {self.role!r}", f"use one of {ROLES}")
        if bool(self.python) == bool(self.image_digest):
            raise _refuse(
                f"{self.role} environment needs exactly one of python or image_digest",
                "give a python path or an image digest, not both and not neither",
            )
        if self.image_digest and not _DIGEST.match(self.image_digest):
            raise _refuse(f"{self.role} image_digest must be sha256:<64 hex>")
        if not _LOCK.match(self.lockfile_hash or ""):
            raise _refuse(f"{self.role} lockfile_hash must be 64 hex characters (sha256)")

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "lockfile_hash": self.lockfile_hash}
        if self.python:
            out["python"] = self.python
        if self.image_digest:
            out["image_digest"] = self.image_digest
        return out
