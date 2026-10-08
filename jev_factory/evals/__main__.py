"""``python -m jev_factory.evals run|continue|status|smoke``: the release-gate runner.

Usage::

    uv run --group evals python -m jev_factory.evals run --manifest M --run-dir D
        [--run-id ID] [--date YYYY-MM-DD] [--expand] [--no-deepeval]
    uv run --group evals python -m jev_factory.evals continue --manifest M --run-dir D
        [--retry-rejected] [--no-deepeval]
    uv run python -m jev_factory.evals status --run-dir D [--json]
    uv run python -m jev_factory.evals smoke --manifest M --run-dir D [--cases 10]

A smoke run sends the first N cases of the first sendable case set to every
reference and reports tokens, cost, a projected full-run cost and OK/CAPPED;
it needs no deepeval. A smoke run dir stays a smoke run under ``continue``;
``run --expand`` turns it into the full run (the smoke answers are reused).
A full run needs deepeval (the ``evals`` dependency group) unless
``--no-deepeval`` is given, which is recorded in ``result.json``.

``--manifest`` defaults to ``$JEV_EVALS_MANIFEST`` and ``--run-dir`` to
``$JEV_EVALS_RUN_DIR``; the run dir must be outside every git worktree. Case
sets, the world snapshot and relative prediction paths resolve under
``$JEV_EVALS_PRIVATE_ROOT``. Provider keys are read from the environment
variables the manifest names, at call time.

Exit codes: 0 complete; 1 configuration error; 2 environment error; 3 stop
and ask; 4 stopped with calls pending (money, rejected, truncation,
transient); 130 interrupted (Ctrl+C; ``continue`` resumes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import run as runner
from .manifest import ENV_MANIFEST_PATH

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/__main__.py",
    "commit": "9debdc6",
    "adaptations": [
        "the drive command (deviation d2: the unattended loop, its signals and Discord"
        " alerts) is deferred per c14; run/continue/status/smoke are kept",
        "--no-deepeval runs a full pass with the exact corpus metrics only (recorded in"
        " result.json), so a no-secrets, no-deepeval environment can finish a run",
        "NVSH_EVALS_* env vars -> JEV_EVALS_*",
        "d17 (2026-10-08): restructured for SonarCloud code quality (cognitive "
        "complexity split into private helpers, plus lint-level cleanups); behaviour "
        "unchanged, pinned by tests/test_complexity_*.py and differential checks against "
        "the imported version",
    ],
    "licence": "Apache-2.0",
}

ENV_RUN_DIR = "JEV_EVALS_RUN_DIR"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.evals", description="jev release-gate runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(name: str, help_text: str, *, manifest: bool = True) -> argparse.ArgumentParser:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--run-dir", default=os.environ.get(ENV_RUN_DIR))
        if manifest:
            cmd.add_argument("--manifest", default=os.environ.get(ENV_MANIFEST_PATH))
        cmd.add_argument("--json", action="store_true", help="print a JSON document")
        return cmd

    begin = common("run", "start a gate run and take the first pass")
    begin.add_argument("--run-id")
    begin.add_argument("--date")
    begin.add_argument(
        "--expand", action="store_true", help="expand a smoke run dir into the full run"
    )
    resume = common("continue", "resume a run: send what is pending, finalize when done")
    resume.add_argument(
        "--retry-rejected",
        action="store_true",
        help="retry models stopped by a rejected request (e.g. after fixing a key)",
    )
    for cmd in (begin, resume):
        cmd.add_argument(
            "--no-deepeval",
            action="store_true",
            help="score with the exact corpus metrics only (no DeepEval per-case layer)",
        )
    common("status", "per-provider counts and spend so far", manifest=False)
    smoke = common("smoke", "N cases per reference, cost projection")
    smoke.add_argument("--cases", type=int, default=10)
    return parser


def _require(value: str | None, flag: str, env_name: str) -> Path:
    if not value:
        raise runner.RunError(f"{flag} is required (or set {env_name})")
    return Path(value)


def _print_status(run_dir: Path, as_json: bool, emit: Callable[[str], None]) -> None:
    doc = runner.status(run_dir)
    if as_json:
        emit(json.dumps(doc, indent=2, sort_keys=True))
        return
    for line in runner.render_status(doc):
        emit(line)


def _send(args: argparse.Namespace, run_dir: Path, kwargs: dict) -> runner.StepOutcome:
    """Run the ``run``, ``continue`` or ``smoke`` command."""
    manifest = _require(args.manifest, "--manifest", ENV_MANIFEST_PATH)
    if args.command == "run":
        return runner.start(
            run_dir,
            manifest,
            run_id=args.run_id,
            date=args.date,
            expand=args.expand,
            use_deepeval=not args.no_deepeval,
            **kwargs,
        )
    if args.command == "continue":
        return runner.step(
            run_dir,
            manifest,
            retry_rejected=args.retry_rejected,
            use_deepeval=not args.no_deepeval,
            **kwargs,
        )
    if args.cases < 1:
        raise runner.RunError("--cases must be at least 1")
    return runner.start(run_dir, manifest, smoke_cases=args.cases, **kwargs)


def main(
    argv: Sequence[str] | None = None,
    *,
    factory: runner.ProviderFactory | None = None,
    env=None,
    out: Callable[[str], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    lines: list[str] = []
    emit = out or (lambda text: print(text, flush=True))
    say = lines.append if args.json else emit
    try:
        run_dir = _require(args.run_dir, "--run-dir", ENV_RUN_DIR)
        if args.command == "status":
            _print_status(run_dir, args.json, emit)
            return runner.EXIT_OK
        kwargs = {"env": env, "factory": factory, "out": say}
        if clock is not None:
            kwargs["clock"] = clock
        outcome = _send(args, run_dir, kwargs)
        if args.json:
            emit(
                json.dumps(
                    {"status": outcome.status, "exit_code": outcome.exit_code, "messages": lines},
                    indent=2,
                )
            )
        return outcome.exit_code
    except runner.StopAndAsk as exc:
        emit(f"stop and ask: {exc}")
        return runner.EXIT_ASK
    except runner.EnvError as exc:
        emit(f"error: {exc}")
        return runner.EXIT_ENV
    except runner.RunError as exc:
        emit(f"error: {exc}")
        return runner.EXIT_USER
    except KeyboardInterrupt:
        emit("interrupted: the ledger is consistent; `continue` resumes where this stopped")
        return runner.EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
