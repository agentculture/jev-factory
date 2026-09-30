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
        proc = subprocess.Popen(  # nosec B603 - argv list, no shell interpolation
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
    """state: running | done | died (pid gone, no done marker) | absent."""
    pid_f, done_f, _ = _paths(Path(jobdir), name)
    rc = None
    if done_f.is_file():
        try:
            rc = int(done_f.read_text().strip())
        except ValueError:
            rc = None
    pid = int(pid_f.read_text()) if pid_f.is_file() else None
    if rc is not None:
        state = "done"
    elif pid is None:
        state = "absent"
    else:
        state = "running" if _alive(pid) else "died"
    return {"state": state, "pid": pid, "rc": rc, "progress": read_progress(jobdir, name)}


class ItemLedger:
    """Append-only journal of finished items plus a progress file."""

    def __init__(self, jobdir: Path, name: str, total: int):
        self.jobdir = Path(jobdir)
        self.jobdir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.total = total
        self.ledger = self.jobdir / f"{name}.items.jsonl"
        self.progress = self.jobdir / f"{name}.progress.json"
        self.done: dict[str, Any] = {}
        if self.ledger.is_file():
            for line in self.ledger.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a kill
                self.done[rec["id"]] = rec.get("result")
        self._write_progress()

    def _write_progress(self) -> None:
        tmp = self.progress.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": len(self.done), "total": self.total}))
        os.replace(tmp, self.progress)

    def record(self, item_id: str, result: Any = None) -> None:
        with self.ledger.open("a") as fh:
            fh.write(json.dumps({"id": item_id, "result": result}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done[item_id] = result
        self._write_progress()

    def run(self, items: Iterable[str], fn: Callable[[str], Any]) -> dict[str, Any]:
        """Call ``fn`` only for items not yet journaled; resume at the next one."""
        for item in items:
            if item not in self.done:
                self.record(item, fn(item))
        return self.done


def read_progress(jobdir: Path, name: str) -> dict[str, int] | None:
    p = Path(jobdir) / f"{name}.progress.json"
    try:
        data = json.loads(p.read_text())
        return {"done": int(data["done"]), "total": int(data["total"])}
    except (OSError, ValueError, KeyError):
        return None
