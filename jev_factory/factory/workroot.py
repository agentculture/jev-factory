"""Work-root guard: a run's work directory must live outside any git worktree."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return Path(path.anchor or ".")


def resolve_work_root(path: str | Path) -> Path:
    """Return the absolute work root, or raise CliError if it is inside a git worktree.

    Detection is ``git rev-parse --is-inside-work-tree`` run from the nearest
    existing ancestor, so a root that does not exist yet is judged by where it
    would be created. A directory holding an empty ``.git`` is not a repository
    to git, so it is accepted.
    """
    root = Path(path).expanduser().resolve()
    probe = _nearest_existing(root)
    git = shutil.which("git")
    if git is None:
        raise CliError(
            EXIT_ENV_ERROR,
            "git is not on PATH; cannot check the work root",
            "install git and make sure it is on PATH",
        )
    try:
        proc = subprocess.run(
            [git, "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise CliError(
            EXIT_ENV_ERROR,
            f"cannot run git to check the work root: {exc}",
            "install git and make sure it is on PATH",
        ) from exc
    if proc.returncode == 0 and proc.stdout.strip() == "true":
        raise CliError(
            EXIT_USER_ERROR,
            f"work root {root} is inside a git worktree",
            "choose a work root outside any git worktree (run artifacts must not land in a repo)",
        )
    return root
