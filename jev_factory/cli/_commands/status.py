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

A run that has distill stages (``<work>/distill``, :mod:`jev_factory.distill.pipeline`) also
shows them, from ``<work>/distill/manifests/``, with their detached jobs (``jobs/distill/``).
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
from jev_factory.distill import pipeline as distill
from jev_factory.factory import detach, pipeline
from jev_factory.factory.stages import (
    COMPLETE,
    SATISFIED,
    SKIPPED,
    Registry,
    read_manifest,
    staleness,
)

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
    jobs = [
        {"job": label, **detach.job_status(jobdir, name)}
        for label, jobdir, name in _job_names(workdir / pipeline.JOBS_DIR)
    ]
    stages = _stage_entries(workdir, pipeline.REGISTRY, lambda n: _stage_jobs(n, jobs))
    distill_dir = distill.stage_dir(workdir)
    distill_stages = (
        _stage_entries(
            distill_dir,
            distill.REGISTRY,
            lambda n: [j for j in jobs if j["job"] == f"{distill.DISTILL_DIR}/{n}"],
        )
        if distill_dir.is_dir()
        else []
    )
    decisions = workdir / pipeline.DECISIONS_FILE
    return {
        "work": str(workdir),
        "stages": stages,
        "distill_stages": distill_stages,
        "jobs": jobs,
        "decisions": len(records.load(decisions)) if decisions.is_file() else 0,
    }


def _stage_entries(
    workdir: Path, registry: Registry, jobs_of: Callable[[str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One status row per stage of *registry*, manifests read from *workdir*."""
    names = registry.names()
    recorded = {n: (read_manifest(workdir, n) or {}).get("knobs", {}) for n in names}
    stages = []
    for name in names:
        man = read_manifest(workdir, name)
        mine = jobs_of(name)
        entry: dict[str, Any] = {
            "stage": name,
            "status": man.get("status") if man else "not-run",
            "stale": staleness(workdir, name, recorded, registry=registry),
            "finished": man.get("finished") if man else None,
            "rc": man.get("rc") if man else None,
            "jobs": mine,
        }
        if man and man.get("error"):
            entry["error"] = man["error"]
        if man and man.get("status") == SKIPPED:
            entry["reason"] = man.get("reason")
        if any(j["state"] == "running" for j in mine):
            entry["status"] = "running"
        stages.append(entry)
    return stages


def _all_stages(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The build's stages, then the distill stages as ``distill/<stage>``."""
    prefix = distill.DISTILL_DIR + "/"
    return list(doc["stages"]) + [
        {**s, "stage": prefix + s["stage"]} for s in doc.get("distill_stages") or ()
    ]


#: The standard update interval for long-running work (operator decision d15).
DEFAULT_EVERY_SECONDS = 30 * 60

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?) *([smh]?)")


def parse_every(text: str) -> float:
    """``90``/``90s``, ``30m``, ``1h`` -> seconds (at least one second)."""
    match = _DURATION_RE.fullmatch(str(text).strip())
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
        s["status"] == "running" for s in _all_stages(doc)
    )


def _changes(doc: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    if previous is None:
        return []
    before_stage = {s["stage"]: s["status"] for s in _all_stages(previous)}
    before_job = {j["job"]: j for j in previous["jobs"]}
    changes = [
        f"{s['stage']}: {before_stage.get(s['stage'], 'not-run')} -> {s['status']}"
        for s in _all_stages(doc)
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
        "stages": {s["stage"]: s["status"] for s in _all_stages(doc) if s["status"] != "not-run"},
        "changes": _changes(doc, previous),
    }


def _update_head(upd: dict[str, Any], final: bool) -> str:
    if final:
        return "final update"
    return "update" if upd["running"] else "nothing running"


def render_update(upd: dict[str, Any], *, final: bool = False) -> str:
    head = _update_head(upd, final)
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


def _stage_mark(status: str) -> str:
    if status == "not-run":
        return "-"
    return "ok" if status == COMPLETE else status


def _stage_line(s: dict[str, Any]) -> str:
    line = f"  {s['stage']:<16} {_stage_mark(s['status']):<8}"
    if s["status"] == SKIPPED and s.get("reason"):
        line += f" reason: {s['reason']}"
    if s["status"] in SATISFIED and s["stale"]:
        line += f" stale: {s['stale']}"
    if s.get("error"):
        line += f" error: {s['error']}"
    for job in s["jobs"]:
        line += f" [{job['job']}: {job['state']}, {_progress(job)}]"
    return line.rstrip()


def render(doc: dict[str, Any]) -> str:
    lines = [f"run: {doc['work']}", ""]
    lines += [_stage_line(s) for s in doc["stages"]]
    if doc.get("distill_stages"):
        lines += ["", "distill stages:"]
        lines += [_stage_line(s) for s in doc["distill_stages"]]
    staged = {j["job"] for s in _all_stages(doc) for j in s["jobs"]}
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
