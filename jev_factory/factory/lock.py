"""Run lock: one jev process per run directory, stale locks reported, never taken."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

LOCK_NAME = ".jev-run.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Context manager holding ``<run_dir>/.jev-run.lock`` for the current PID."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / LOCK_NAME
        self._held = False

    def _refuse(self) -> CliError:
        remedy = f"remove {self.path} once you have confirmed no jev process uses this run"
        try:
            pid = int(json.loads(self.path.read_text())["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return CliError(
                EXIT_USER_ERROR, f"run lock {self.path} exists and is unreadable", remedy
            )
        if _pid_alive(pid):
            return CliError(
                EXIT_USER_ERROR,
                f"run is locked by PID {pid} ({self.path})",
                "wait for that process to finish, or stop it",
            )
        return CliError(
            EXIT_USER_ERROR,
            f"stale run lock left by dead PID {pid} ({self.path})",
            remedy,
        )

    def acquire(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise self._refuse() from None
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid()}, handle)
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
