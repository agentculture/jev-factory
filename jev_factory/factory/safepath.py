"""Containment for paths that arrive from CLI arguments, run configs or manifests.

A path is resolved (``..`` collapsed, every existing symlink followed) and refused unless it
stays inside its *root* (the run directory, a bundle or artifact directory, a checkpoint
directory). :func:`regular_file` additionally requires an existing regular file that is not
itself a symlink, for the files a stage hashes, parses or copies. Refusals are
:class:`~jev_factory.cli._errors.CliError` with exit code 1 (a user error). Stdlib only.
"""

from __future__ import annotations

import os
from pathlib import Path

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

__all__ = ["contained", "regular_file", "resolved"]


def _text(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path)
    if "\x00" in text:
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing a path that contains a NUL byte: {text!r}",
            "pass a plain filesystem path",
        )
    return text


def resolved(path: str | os.PathLike[str]) -> Path:
    """*path* resolved, refusing a NUL byte; for a path with no root to hold it to."""
    return Path(os.path.realpath(_text(path)))


def contained(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    """Return *path* resolved, refusing it unless it lies inside *root* (also resolved).

    Relative paths resolve against the working directory, as a CLI argument would. The
    target need not exist (a write target); any existing symlink on the way is followed,
    so a link that points out of *root* is refused.
    """
    base = Path(os.path.realpath(_text(root)))
    target = resolved(path)
    if not target.is_relative_to(base):
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing {os.fspath(path)!r}: it escapes {str(base)!r} "
            f"(resolves to {str(target)!r})",
            f"keep the path inside {base}; no '..', absolute path or symlink may lead out",
        )
    return target


def regular_file(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    """:func:`contained`, and *path* must be an existing regular file, not a symlink."""
    target = contained(path, root)
    if Path(path).is_symlink():
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing {os.fspath(path)!r}: it is a symlink where a regular file is required",
            "copy the real file into place; inputs are never followed through links",
        )
    if not target.is_file():
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing {os.fspath(path)!r}: it is not a regular file",
            "check the path",
        )
    return target
