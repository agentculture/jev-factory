"""``jev status <run>`` -- read-only view of a run's stages.

Reads each stage's manifest (``manifests/<stage>.json``), whether the stage is
stale and why, the decision-record count, and the detached jobs under
``jobs/`` with their items-done/total progress. Writes nothing and needs no
domain: staleness is judged against the knobs each stage last ran with.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def _progress(job: dict[str, Any]) -> str:
    p = job.get("progress")
    return f"{p['done']}/{p['total']} items" if p else "no progress file"


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
    lines += ["", f"decision records: {doc['decisions']}"]
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
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
    p.set_defaults(func=cmd_status)
