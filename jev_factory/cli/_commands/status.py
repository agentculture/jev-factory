"""``jev status <run>`` -- read-only view of a run's stages.

Reads each stage's manifest (``manifests/<stage>.json``), whether the stage is
stale and why, the decision-record count, and the detached jobs under
``jobs/`` with their items-done/total progress. Writes nothing and needs no
domain: staleness is judged against the knobs each stage last ran with.

``--watch`` is the standard progress cadence for long-running work (deviation d15):
it prints a timestamped update every ``--every`` (default 30 minutes) -- each job's
items done/total, its rate and ETA, and what changed since the last update -- and
stops with a final update once nothing is running. With ``--json`` each update is
one JSON line.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_result
from jev_factory.decide import records
from jev_factory.factory import detach, pipeline
from jev_factory.factory.stages import COMPLETE, read_manifest, staleness

#: Stages whose in-process jobs live under a differently named jobs/ entry.
_JOB_ALIASES = {"teachers-pilot": "pilot"}


def _job_names(jobs: Path) -> list[tuple[str, Path, str]]:
    """Every detached job or item ledger under jobs/: ``(label, jobdir, name)``."""
    found: list[tuple[str, Path, str]] = []
    if not jobs.is_dir():
        return found
    for directory in [jobs, *sorted(p for p in jobs.iterdir() if p.is_dir())]:
        names = {
            f.name.split(".", 1)[0]
            for f in sorted(directory.iterdir())
            if f.is_file() and f.name.endswith((".pid", ".progress.json"))
        }
        for name in sorted(names):
            label = name if directory == jobs else f"{directory.name}/{name}"
            found.append((label, directory, name))
    return found


def _stage_jobs(stage: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alias = _JOB_ALIASES.get(stage, stage)
    return [j for j in jobs if j["job"].split("/", 1)[0] in (stage, alias)]


def collect(workdir: Path) -> dict[str, Any]:
    """The status document for a work dir."""
    workdir = Path(workdir)
    if not workdir.is_dir():
        raise CliError(
            EXIT_USER_ERROR,
            f"{workdir} is not a run directory",
            "pass the run's work directory (jev init creates it with --apply)",
        )
    recorded = {
        n: (read_manifest(workdir, n) or {}).get("knobs", {}) for n in pipeline.stage_names()
    }
    jobs = [
        {"job": label, **detach.job_status(jobdir, name)}
        for label, jobdir, name in _job_names(workdir / pipeline.JOBS_DIR)
    ]
    stages = []
    for name in pipeline.stage_names():
        man = read_manifest(workdir, name)
        mine = _stage_jobs(name, jobs)
        entry: dict[str, Any] = {
            "stage": name,
            "status": man.get("status") if man else "not-run",
            "stale": staleness(workdir, name, recorded, registry=pipeline.REGISTRY),
            "finished": man.get("finished") if man else None,
            "rc": man.get("rc") if man else None,
            "jobs": mine,
        }
        if man and man.get("error"):
            entry["error"] = man["error"]
        if any(j["state"] == "running" for j in mine):
            entry["status"] = "running"
        stages.append(entry)
    decisions = workdir / pipeline.DECISIONS_FILE
    return {
        "work": str(workdir),
        "stages": stages,
        "jobs": jobs,
        "decisions": len(records.load(decisions)) if decisions.is_file() else 0,
    }


#: The standard update interval for long-running work (operator decision d15).
DEFAULT_EVERY_SECONDS = 30 * 60

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$")


def parse_every(text: str) -> float:
    """``90``/``90s``, ``30m``, ``1h`` -> seconds (at least one second)."""
    match = _DURATION_RE.match(str(text))
    if not match:
        raise CliError(
            EXIT_USER_ERROR, f"--every {text!r} is not a duration", "use e.g. 30m, 1h or 90s"
        )
    value = float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    if value < 1:
        raise CliError(EXIT_USER_ERROR, "--every must be at least 1s", "use e.g. 30m")
    return value


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _progress(job: dict[str, Any]) -> str:
    p = job.get("progress")
    if not p:
        return "no progress file"
    text = f"{p['done']}/{p['total']} items"
    rate, eta = detach.rate_and_eta(p)
    if rate is not None and job.get("state") == "running":
        text += f", {rate * 60:.1f}/min, ETA {_duration(eta)}"
    return text


def running(doc: dict[str, Any]) -> bool:
    """Whether any stage or job of the run is still running."""
    return any(j["state"] == "running" for j in doc["jobs"]) or any(
        s["status"] == "running" for s in doc["stages"]
    )


def _changes(doc: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    if previous is None:
        return []
    before_stage = {s["stage"]: s["status"] for s in previous["stages"]}
    before_job = {j["job"]: j for j in previous["jobs"]}
    changes = [
        f"{s['stage']}: {before_stage.get(s['stage'], 'not-run')} -> {s['status']}"
        for s in doc["stages"]
        if before_stage.get(s["stage"], "not-run") != s["status"]
    ]
    for job in doc["jobs"]:
        old = before_job.get(job["job"])
        done = (job.get("progress") or {}).get("done")
        old_done = ((old or {}).get("progress") or {}).get("done")
        if old is None:
            changes.append(f"{job['job']}: new ({job['state']})")
        elif old["state"] != job["state"]:
            changes.append(f"{job['job']}: {old['state']} -> {job['state']}")
        if old is not None and done is not None and old_done is not None and done != old_done:
            changes.append(f"{job['job']}: +{done - old_done} items")
    return changes


def update(doc: dict[str, Any], previous: dict[str, Any] | None, now: float) -> dict[str, Any]:
    """One progress update: when, whether work is still running, every job, the changes."""
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "work": doc["work"],
        "running": running(doc),
        "jobs": [
            {
                **job,
                "rate_per_min": (
                    None if (r := detach.rate_and_eta(job.get("progress"))[0]) is None else r * 60
                ),
                "eta_seconds": detach.rate_and_eta(job.get("progress"))[1],
            }
            for job in doc["jobs"]
        ],
        "stages": {s["stage"]: s["status"] for s in doc["stages"] if s["status"] != "not-run"},
        "changes": _changes(doc, previous),
    }


def render_update(upd: dict[str, Any], *, final: bool = False) -> str:
    head = "final update" if final else ("update" if upd["running"] else "nothing running")
    lines = [f"[{upd['at']}] {upd['work']}: {head}"]
    for job in upd["jobs"]:
        lines.append(f"  {job['job']}: {job['state']}, {_progress(job)}")
    failed = [name for name, status in upd["stages"].items() if status == "failed"]
    if failed:
        lines.append(f"  failed stages: {', '.join(failed)}")
    if upd["changes"]:
        lines.append("  since last update: " + "; ".join(upd["changes"]))
    return "\n".join(lines)


def watch(
    workdir: Path,
    every: float,
    *,
    json_mode: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    stream: Any = None,
) -> int:
    """Print an update now and every *every* seconds until nothing is running."""
    out = stream if stream is not None else sys.stdout
    previous: dict[str, Any] | None = None
    while True:
        doc = collect(workdir)
        upd = update(doc, previous, clock())
        final = previous is not None and not upd["running"]
        emit_result(
            upd if json_mode else render_update(upd, final=final), json_mode=json_mode, stream=out
        )
        out.flush()
        if not upd["running"]:
            return 0
        previous = doc
        sleep(every)


def render(doc: dict[str, Any]) -> str:
    lines = [f"run: {doc['work']}", ""]
    for s in doc["stages"]:
        mark = (
            "-" if s["status"] == "not-run" else ("ok" if s["status"] == COMPLETE else s["status"])
        )
        line = f"  {s['stage']:<16} {mark:<8}"
        if s["status"] == COMPLETE and s["stale"]:
            line += f" stale: {s['stale']}"
        if s.get("error"):
            line += f" error: {s['error']}"
        for job in s["jobs"]:
            line += f" [{job['job']}: {job['state']}, {_progress(job)}]"
        lines.append(line.rstrip())
    staged = {j["job"] for s in doc["stages"] for j in s["jobs"]}
    others = [j for j in doc["jobs"] if j["job"] not in staged]
    if others:
        lines += ["", "jobs:"]
        lines += [f"  {j['job']}: {j['state']}, {_progress(j)}" for j in others]
    lines += ["", f"decision records: {doc['decisions']}"]
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    if args.every is not None and not args.watch:
        raise CliError(EXIT_USER_ERROR, "--every needs --watch", "add --watch")
    if args.watch:
        every = parse_every(args.every) if args.every is not None else DEFAULT_EVERY_SECONDS
        return watch(Path(args.run).expanduser(), every, json_mode=bool(args.json))
    doc = collect(Path(args.run).expanduser())
    json_mode = bool(args.json)
    emit_result(doc if json_mode else render(doc), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "status",
        help="Show a run's stage manifests, staleness and detached-job progress (read-only).",
    )
    p.add_argument("run", help="The run's work directory.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--watch",
        action="store_true",
        help="Print a progress update every --every until nothing is running (read-only).",
    )
    p.add_argument(
        "--every",
        default=None,
        help="Update interval for --watch, e.g. 30m, 1h, 90s (default 30m, the standard).",
    )
    p.set_defaults(func=cmd_status)
