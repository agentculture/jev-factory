"""Work-root guard (o40) and run lock (o39)."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.factory.lock import LOCK_NAME, RunLock
from jev_factory.factory.workroot import resolve_work_root


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.mark.behavioral("o40")
def test_workroot_inside_git_worktree_refused(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    inside = tmp_path / "sub" / "deeper"
    inside.mkdir(parents=True)
    for target in (tmp_path, inside, tmp_path / "not-yet-created"):
        with pytest.raises(CliError) as exc:
            resolve_work_root(target)
        assert exc.value.code == 1
        assert "git worktree" in exc.value.message


@pytest.mark.behavioral("o40")
def test_workroot_with_empty_dot_git_accepted(tmp_path):
    (tmp_path / ".git").mkdir()
    assert resolve_work_root(tmp_path) == tmp_path.resolve()


@pytest.mark.behavioral("o40")
def test_workroot_plain_dir_and_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_work_root("new/run") == (tmp_path / "new" / "run").resolve()


@pytest.mark.behavioral("o39")
def test_second_acquire_names_holder_pid(tmp_path):
    with RunLock(tmp_path):
        second = RunLock(tmp_path)
        with contextlib.ExitStack() as stack, pytest.raises(CliError) as exc:
            stack.enter_context(second)  # must refuse to enter
        assert exc.value.code == 1
        assert str(os.getpid()) in exc.value.message
    # released on exit: acquirable again
    with RunLock(tmp_path):
        pass
    assert not (tmp_path / LOCK_NAME).exists()


@pytest.mark.behavioral("o39")
def test_stale_lock_reported_not_taken(tmp_path):
    pid = _dead_pid()
    lock_file = tmp_path / LOCK_NAME
    lock_file.write_text(f'{{"pid": {pid}}}')
    lock = RunLock(tmp_path)
    with contextlib.ExitStack() as stack, pytest.raises(CliError) as exc:
        stack.enter_context(lock)  # a stale lock must not be silently taken
    assert "stale" in exc.value.message
    assert str(pid) in exc.value.message
    assert lock_file.read_text() == f'{{"pid": {pid}}}'


@pytest.mark.behavioral("o39")
def test_unreadable_lock_is_refused(tmp_path):
    (tmp_path / LOCK_NAME).write_text("garbage")
    lock = RunLock(tmp_path)
    with contextlib.ExitStack() as stack, pytest.raises(CliError):
        stack.enter_context(lock)  # must not enter
