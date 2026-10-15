"""``jev distill <stage>`` -- the distillery: one stage of a teacher -> student distill run.

The stages come from the distillery's own registry (:data:`jev_factory.distill.pipeline.REGISTRY`),
separate from ``jev run``'s. ``jev distill`` alone lists them in order. Every stage is dry-run by
default: it prints what it would read and write, whether it is stale, the run-config keys it needs
that are unset and, where the owning module has its own dry run, that plan; it changes nothing.
``--apply`` runs it under the run lock; ``--apply --detach`` starts it in its own session (pid,
log and done marker under ``<work>/jobs/distill/``) so a long stage outlives the shell; follow it
with ``jev status <work>``. ``--force`` re-runs a fresh stage.

The stage is a positional argument (not one subcommand per stage) so the jev-CLI domain gains
one ``distill`` operation, not fifteen: the jev-tool model's candidate set stays within its
52-letter bound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jev_factory.cli._commands._runctx import CONFIG_FILE, parse_knobs
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_result
from jev_factory.distill import pipeline
from jev_factory.factory import detach
from jev_factory.factory.config import load_config
from jev_factory.factory.workroot import resolve_work_root


def _context(args: argparse.Namespace) -> pipeline.DistillContext:
    work = Path(args.work).expanduser() if args.work else None
    config = args.config
    if config is None and work is not None and (work / CONFIG_FILE).is_file():
        config = str(work / CONFIG_FILE)
    cfg = load_config(config, cli={"work": str(work) if work is not None else None})
    return pipeline.DistillContext(
        work=Path(cfg["work"]),
        config=cfg,
        deviation_id=args.deviation_id,
        allow_foreign_gpu=bool(args.allow_foreign_gpu),
    )


def _passthrough(args: argparse.Namespace, work: Path) -> list[str]:
    argv = ["--work", str(work)]
    for flag, value in (("--config", args.config), ("--deviation-id", args.deviation_id)):
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
            f"distill stage {result['stage']} started detached (pid {result['pid']})\n"
            f"log: {result['log']}\nfollow it with: jev status {result['work']}"
        )
    if result.get("applied", True) is False:
        lines = [
            f"dry run: distill stage {result['stage']} ({result['summary']})",
            f"  would run: {result['would_run']}"
            + (f" ({result['stale']})" if result["stale"] else ""),
            "  upstream: " + (", ".join(result["deps"]) or "none"),
            "  reads: " + (", ".join(result["inputs"]) or "nothing"),
        ]
        if result["missing_inputs"]:
            lines.append("  missing inputs: " + ", ".join(result["missing_inputs"]))
        lines.append("  writes: " + (", ".join(result["outputs"]) or "nothing"))
        if result["unset_config"]:
            lines.append("  unset run-config keys: " + ", ".join(result["unset_config"]))
        blocked = result.get("module_plan_blocked")
        if blocked:
            lines.append(f"  module plan blocked: {blocked['message']}")
        elif "module_plan" in result:
            lines.append("  module plan: available with --json")
        lines.append(f"  {result['note']}")
        return "\n".join(lines)
    state = "skipped (fresh)" if result.get("skipped") else result.get("status", "done")
    return f"distill stage {result['stage']}: {state}"


def _detach(args: argparse.Namespace, ctx: pipeline.DistillContext, knobs: dict) -> dict:
    stage = args.stage
    pipeline.effective_knobs(stage, knobs, ctx)  # refuse unknown knobs before detaching
    work = resolve_work_root(ctx.work)
    jobs = pipeline.jobs_dir(work)
    current = detach.job_status(jobs, stage)
    if current["state"] == "running":
        raise CliError(
            EXIT_USER_ERROR,
            f"distill stage {stage} is already running (pid {current['pid']})",
            f"follow it with: jev status {work}",
        )
    argv = [sys.executable, "-m", "jev_factory", "distill", stage, "--apply", "--json"]
    argv += _passthrough(args, work)
    pid = detach.start_detached(jobs, stage, argv)
    return {
        "stage": stage,
        "verb": "distill",
        "detached": True,
        "pid": pid,
        "work": str(work),
        "log": str(jobs / f"{stage}.log"),
    }


def _list(json_mode: bool) -> int:
    rows = [
        {
            "stage": n,
            "summary": pipeline.get_stage(n).summary,
            "deps": list(pipeline.get_stage(n).deps),
        }
        for n in pipeline.stage_names()
    ]
    if json_mode:
        emit_result({"stages": rows}, json_mode=True)
    else:
        lines = ["distill stages, in order (jev distill <stage> --work <dir>):"]
        lines += [f"  {r['stage']} -- {r['summary']}" for r in rows]
        emit_result("\n".join(lines), json_mode=False)
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    if args.stage is None:
        if args.apply or args.detach or args.force:
            raise CliError(
                EXIT_USER_ERROR,
                "name the distill stage to run",
                "list them with: jev distill; then: jev distill <stage> --work <dir>",
            )
        return _list(json_mode)
    pipeline.get_stage(args.stage)
    if args.detach and not args.apply:
        raise CliError(
            EXIT_USER_ERROR,
            "--detach needs --apply: a dry run is instant and changes nothing",
            f"run: jev distill {args.stage} --apply --detach",
        )
    knobs = parse_knobs(args.knob)
    ctx = _context(args)
    if args.detach:
        result = _detach(args, ctx, knobs)
    else:
        result = pipeline.run(ctx.work, args.stage, ctx, knobs, apply=args.apply, force=args.force)
        if result.get("status") == "failed":
            raise CliError(
                int(result.get("rc") or EXIT_USER_ERROR),
                f"distill stage {args.stage} failed: {result.get('error', 'unknown error')}",
                f"see distill/manifests/{args.stage}.json in the work dir, fix the cause, re-run",
            )
    emit_result(result if json_mode else _render(result), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "distill",
        help="Run one distill stage, teacher to student, through the edge-board check "
        "(dry-run by default; --apply commits).",
        description="Run one stage of a distill run; with no stage, list the stages in order.",
    )
    p.add_argument(
        "stage",
        nargs="?",
        choices=pipeline.stage_names(),
        metavar="stage",
        help="The distill stage: " + ", ".join(pipeline.stage_names()) + ".",
    )
    p.add_argument(
        "--work",
        help="The run's work directory (outside any git worktree); "
        "default: the 'work' key of the run config.",
    )
    p.add_argument("--config", help="Run config TOML (default: <work>/run.toml, if present).")
    p.add_argument("--deviation-id", help="Recorded deviation that authorises a changed plan.")
    p.add_argument(
        "--knob",
        action="append",
        metavar="KEY=VALUE",
        help="Stage knob (VALUE is JSON when it parses); repeatable.",
    )
    p.add_argument("--apply", action="store_true", help="Run the stage (default: dry run).")
    p.add_argument(
        "--detach",
        action="store_true",
        help="With --apply: start the stage detached from this shell (see 'jev status').",
    )
    p.add_argument("--force", action="store_true", help="Re-run even when the stage is fresh.")
    p.add_argument(
        "--allow-foreign-gpu",
        action="store_true",
        help="Allow GPU stages while another process holds the GPU.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_distill)
