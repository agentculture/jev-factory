"""The pre-registration lock check, without the pre-registration schema.

A run's pre-registration is hashed into ``<run_dir>/distill/prereg.lock.json`` before any
training command (:func:`jev_factory.distill.prereg.lock`). This module checks a file against
that lock: the lock must exist as the run's own regular file, and the file's sha256 must equal
the recorded one unless a deviation id is given. It reads no schema, so a trainer process can
re-check the lock without importing the distillery (deviation d1); the distillery's
:func:`~jev_factory.distill.prereg.require_locked` runs the same checks around its validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory import safepath

#: Where a run's lock lives, relative to the run directory.
LOCK_REL = "distill/prereg.lock.json"


def _err(message: str, remediation: str) -> CliError:
    return CliError(EXIT_USER_ERROR, message, remediation)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lock(run_dir: Path) -> dict[str, Any]:
    """The run's lock record, or refuse when there is none (or it is not the run's own file)."""
    lock_path = Path(run_dir) / LOCK_REL
    if not lock_path.exists():
        raise _err(
            "no distill pre-registration is locked for this run",
            "lock the distill pre-registration before any training command",
        )
    # The lock must be this run's own file: a symlink (e.g. to another run's lock) is refused.
    lock_path = safepath.regular_file(lock_path, Path(run_dir))
    return json.loads(lock_path.read_text(encoding="utf-8"))


def check_unchanged(
    prereg_path: Path, recorded: dict[str, Any], deviation_id: str | None = None
) -> str:
    """The file's sha256, refused when it differs from the lock and no deviation id is given."""
    digest = sha256_file(prereg_path)
    if digest != recorded.get("sha256") and not (deviation_id or "").strip():
        raise _err(
            "the distill pre-registration file changed since it was locked",
            "record a deviation and pass its id (--deviation <id>)",
        )
    return digest


def require_locked_file(prereg_path: Path, run_dir: Path, deviation_id: str | None = None) -> str:
    """:func:`read_lock` then :func:`check_unchanged`; returns the file's sha256."""
    return check_unchanged(Path(prereg_path), read_lock(Path(run_dir)), deviation_id)
