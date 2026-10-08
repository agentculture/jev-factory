"""``jev run <stage>`` -- one subcommand per registered build stage.

The subcommands are generated from the stage registry
(:data:`jev_factory.factory.pipeline.REGISTRY`), so ``jev run --help`` lists
every stage and a new stage needs no edit here. Every stage is a write verb:
dry-run by default (it prints what it would read and write and whether it is
stale, and changes nothing), ``--apply`` runs it. ``--apply --detach`` starts
the stage in its own session so a long stage outlives the shell; follow it with
``jev status``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jev_factory.cli._commands._runctx import add_context_options, load_context, parse_knobs
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_result
from jev_factory.factory import detach, pipeline
from jev_factory.factory.workroot import resolve_work_root


def _passthrough(args: argparse.Namespace, workdir: Path) -> list[str]:
    argv = ["--work", str(workdir)]
    for flag, value in (
        ("--config", args.config),
        ("--domain", args.domain),
        ("--deviation-id", args.deviation_id),
    ):
        if value:
            argv += [flag, str(value)]
    for knob in args.knob or []:
        argv += ["--knob", knob]
    for flag, on in (("--force", args.force), ("--allow-foreign-gpu", args.allow_foreign_gpu)):
        if on:
            argv.append(flag)
    return argv


def _render(result: dict) -> str:
    if result.get("detached"):
        return (
            f"stage {result['stage']} started detached (pid {result['pid']})\n"
            f"log: {result['log']}\nfollow it with: jev status {result['work']}"
        )
    if result.get("applied", True) is False:
        lines = [f"dry run: stage {result['stage']} ({result['summary']})"]
        lines.extend(
            [
                f"  would run: {result['would_run']}"
                + (f" ({result['stale']})" if result["stale"] else ""),
                "  substeps: " + ", ".join(result["substeps"]),
                "  reads: " + (", ".join(result["inputs"]) or "nothing"),
            ]
        )
        if result["missing_inputs"]:
            lines.append("  missing inputs: " + ", ".join(result["missing_inputs"]))
        lines.extend(
            [
                "  writes: " + (", ".join(result["outputs"]) or "nothing"),
                f"  {result['note']}",
            ]
        )
        return "\n".join(lines)
    state = "skipped (fresh)" if result.get("skipped") else result.get("status", "done")
    return f"stage {result['stage']}: {state}"


def _run_stage(args: argparse.Namespace) -> int:
    stage = args.stage_name
    workdir, ctx = load_context(args, args.work)
    knobs = parse_knobs(args.knob)
    if args.detach and not args.apply:
        raise CliError(
            EXIT_USER_ERROR,
            "--detach needs --apply: a dry run is instant and changes nothing",
            f"run: jev run {stage} --apply --detach",
        )
    json_mode = bool(args.json)
    if args.detach:
        pipeline.get_stage(stage)
        pipeline.effective_knobs(stage, knobs, ctx)  # refuse unknown knobs before detaching
        workdir = resolve_work_root(workdir)
        jobs = workdir / pipeline.JOBS_DIR
        current = detach.job_status(jobs, stage)
        if current["state"] == "running":
            raise CliError(
                EXIT_USER_ERROR,
                f"stage {stage} is already running (pid {current['pid']})",
                f"follow it with: jev status {workdir}",
            )
        argv = [sys.executable, "-m", "jev_factory", "run", stage, "--apply", "--json"]
        argv += _passthrough(args, workdir)
        pid = detach.start_detached(jobs, stage, argv)
        result = {
            "stage": stage,
            "detached": True,
            "pid": pid,
            "work": str(workdir),
            "log": str(jobs / f"{stage}.log"),
        }
        emit_result(result if json_mode else _render(result), json_mode=json_mode)
        return 0
    result = pipeline.run(workdir, stage, ctx, knobs, apply=args.apply, force=args.force)
    if result.get("status") == "failed":
        raise CliError(
            int(result.get("rc") or EXIT_USER_ERROR),
            f"stage {stage} failed: {result.get('error', 'unknown error')}",
            f"see manifests/{stage}.json in the work dir, fix the cause and re-run",
        )
    emit_result(result if json_mode else _render(result), json_mode=json_mode)
    return 0


def _list_stages(args: argparse.Namespace) -> int:
    rows = [
        {
            "stage": n,
            "summary": pipeline.get_stage(n).summary,
            "deps": list(pipeline.get_stage(n).deps),
        }
        for n in pipeline.stage_names()
    ]
    if getattr(args, "json", False):
        emit_result({"stages": rows}, json_mode=True)
    else:
        emit_result(
            "\n".join(
                ["stages, in order (jev run <stage> --help):"]
                + [f"  {r['stage']} -- {r['summary']}" for r in rows]
            ),
            json_mode=False,
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="Run one build stage (dry-run by default; --apply commits).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=_list_stages, json=False)
    stages = p.add_subparsers(dest="stage_name", title="stages", parser_class=type(p))
    for name in pipeline.stage_names():
        st = pipeline.get_stage(name)
        sp = stages.add_parser(name, help=st.summary, description=st.summary)
        add_context_options(sp, work=True)
        sp.add_argument(
            "--knob",
            action="append",
            metavar="KEY=VALUE",
            help="Stage knob (VALUE is JSON when it parses); repeatable.",
        )
        sp.add_argument("--apply", action="store_true", help="Run the stage (default: dry run).")
        sp.add_argument(
            "--detach",
            action="store_true",
            help="With --apply: start the stage detached from this shell (see 'jev status').",
        )
        sp.add_argument("--force", action="store_true", help="Re-run even when the stage is fresh.")
        sp.add_argument(
            "--allow-foreign-gpu",
            action="store_true",
            help="Allow GPU stages while another process holds the GPU.",
        )
        sp.set_defaults(func=_run_stage, stage_name=name)
