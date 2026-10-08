"""Detached long jobs: pid file, real-rc done marker and items-done/total.

A job started here survives the invoking shell (new session via setsid). A
wrapper records the command's real exit code in ``<name>.done`` only once it
ends, so a killed or crashed job leaves a pid file but no done marker.
``ItemLedger`` gives item-level resume: finished items are journaled, so a
restarted job skips them and re-sends nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

_WRAPPER = 'rc=0; "$@" || rc=$?; printf "%s\\n" "$rc" > "$DONE.tmp" && mv "$DONE.tmp" "$DONE"'


def _paths(jobdir: Path, name: str) -> tuple[Path, Path, Path]:
    return jobdir / f"{name}.pid", jobdir / f"{name}.done", jobdir / f"{name}.log"


def start_detached(jobdir: Path, name: str, argv: list[str]) -> int:
    """Start ``argv`` detached from this shell; return the pid."""
    jobdir = Path(jobdir)
    jobdir.mkdir(parents=True, exist_ok=True)
    pid_f, done_f, log_f = _paths(jobdir, name)
    done_f.unlink(missing_ok=True)
    env = {**os.environ, "DONE": str(done_f)}
    with log_f.open("ab") as log:
        proc = subprocess.Popen(  # nosec B603
            ["/bin/sh", "-c", _WRAPPER, "jev-detach", *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # setsid
        )
    pid_f.write_text(f"{proc.pid}\n")
    return proc.pid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def job_status(jobdir: Path, name: str) -> dict[str, Any]:
    """state: running | done | died (pid gone, no done marker) | absent.

    A job with no ``.pid`` file (an in-process stage's ledger, a measurement, a probe)
    takes its state from its progress file: ``complete`` when every item is done, else
    ``running`` while the writing process lives and ``stopped`` once it is gone."""
    pid_f, done_f, _ = _paths(Path(jobdir), name)
    rc = _read_rc(done_f)
    pid = int(pid_f.read_text()) if pid_f.is_file() else None
    progress = read_progress(jobdir, name)
    state = _job_state(rc, pid, progress)
    return {"state": state, "pid": pid, "rc": rc, "progress": progress}


def _read_rc(done_f: Path) -> int | None:
    """The real exit code from a done marker, or None (absent or garbled)."""
    if not done_f.is_file():
        return None
    try:
        return int(done_f.read_text().strip())
    except ValueError:
        return None


def _job_state(rc: int | None, pid: int | None, progress: dict[str, Any] | None) -> str:
    if rc is not None:
        return "done"
    if pid is not None:
        return "running" if _alive(pid) else "died"
    if progress is not None and progress.get("pid"):
        return _progress_state(progress)
    return "absent"


def _progress_state(progress: dict[str, Any]) -> str:
    """State of a pid-less job from its progress file (complete, running or stopped)."""
    if progress["done"] >= progress["total"]:
        return "complete"
    return "running" if _alive(int(progress["pid"])) else "stopped"


class Progress:
    """A job's ``<name>.progress.json``: items done of total, when this session started
    (and how many were done then, so a resumed job's rate counts only this session's
    items), when it was last updated, and the writing process's pid. Written atomically
    after every item; :func:`read_progress` and ``jev status`` read it."""

    def __init__(self, jobdir: Path, name: str, total: int, done: int = 0):
        self.path = Path(jobdir) / f"{name}.progress.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = int(total)
        self.done = int(done)
        self.start_done = int(done)
        self.started = time.time()
        self._write()

    def update(self, done: int) -> None:
        self.done = int(done)
        self._write()

    def advance(self, n: int = 1) -> None:
        self.update(self.done + n)

    def _write(self) -> None:
        doc = {
            "done": self.done,
            "total": self.total,
            "started": self.started,
            "start_done": self.start_done,
            "updated": time.time(),
            "pid": os.getpid(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        os.replace(tmp, self.path)


class ItemLedger:
    """Append-only journal of finished items plus a progress file."""

    def __init__(self, jobdir: Path, name: str, total: int):
        self.jobdir = Path(jobdir)
        self.jobdir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.total = total
        self.ledger = self.jobdir / f"{name}.items.jsonl"
        self.done: dict[str, Any] = {}
        if self.ledger.is_file():
            for line in self.ledger.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a kill
                self.done[rec["id"]] = rec.get("result")
        self._progress = Progress(self.jobdir, name, total, len(self.done))
        self.progress = self._progress.path

    def record(self, item_id: str, result: Any = None) -> None:
        with self.ledger.open("a") as fh:
            fh.write(json.dumps({"id": item_id, "result": result}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done[item_id] = result
        self._progress.update(len(self.done))

    def run(self, items: Iterable[str], fn: Callable[[str], Any]) -> dict[str, Any]:
        """Call ``fn`` only for items not yet journaled; resume at the next one."""
        for item in items:
            if item not in self.done:
                self.record(item, fn(item))
        return self.done


def read_progress(jobdir: Path, name: str) -> dict[str, Any] | None:
    """``done``/``total``, plus ``started``/``start_done``/``updated``/``pid`` when the
    writer recorded them (older progress files carry only the counts)."""
    p = Path(jobdir) / f"{name}.progress.json"
    try:
        data = json.loads(p.read_text())
        out: dict[str, Any] = {"done": int(data["done"]), "total": int(data["total"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for key, cast in (("started", float), ("start_done", int), ("updated", float), ("pid", int)):
        if data.get(key) is not None:
            try:
                out[key] = cast(data[key])
            except (TypeError, ValueError):
                pass
    return out


def rate_and_eta(progress: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Items per second over this session, and seconds left at that rate (``None`` when
    the progress file has no timing or no item finished yet)."""
    if not progress or "started" not in progress or "updated" not in progress:
        return None, None
    elapsed = progress["updated"] - progress["started"]
    finished = progress["done"] - progress.get("start_done", 0)
    if elapsed <= 0 or finished <= 0:
        return None, None
    rate = finished / elapsed
    return rate, max(progress["total"] - progress["done"], 0) / rate
