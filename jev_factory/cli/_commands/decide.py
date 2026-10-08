"""``jev decide <run>`` -- apply the pre-registered rule and record the verdict.

Calls :func:`jev_factory.factory.pipeline.decide`, which refuses an
unregistered or changed pre-registration and appends one record to
``decisions.jsonl`` (append-only; :mod:`jev_factory.decide.records`). The verdict
is printed with every metric it cited. The record is the only thing written:
this verb never runs a stage and never passes ``--apply`` to anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jev_factory.cli._commands._runctx import add_context_options, load_context
from jev_factory.cli._output import emit_result
from jev_factory.factory import pipeline


def render(rec: dict[str, Any]) -> str:
    lines = [
        f"{rec['id']}: {rec['verdict']}  (rule {rec['rule_version']}, decider {rec['decider']})"
    ]
    if rec.get("params"):
        lines.append("  params: " + ", ".join(f"{k}={v}" for k, v in sorted(rec["params"].items())))
    lines += [f"  reason: {r}" for r in rec["reasons"]]
    lines.append("  cited metrics:")
    for c in rec["cited"]:
        sha = (c.get("sha256") or "")[:12]
        lines.append(f"    {c['name']} = {c['value']}  ({c['source']} sha256:{sha})")
    return "\n".join(lines)


def cmd_decide(args: argparse.Namespace) -> int:
    workdir, ctx = load_context(args, args.run)
    reference = Path(args.reference).expanduser() if args.reference else None
    rec = pipeline.decide(workdir, ctx, reference=reference, hard_stops=tuple(args.hard_stop or ()))
    json_mode = bool(args.json)
    emit_result(rec if json_mode else render(rec), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "decide",
        help="Apply the pre-registered rule to a run and append a decision record.",
    )
    p.add_argument("run", help="The run's work directory.")
    add_context_options(p, work=False)
    p.add_argument("--reference", help="Reference candidate summary JSON to compare against.")
    p.add_argument(
        "--hard-stop",
        action="append",
        metavar="KIND",
        help="A hard stop that forces stop_escalate (rule_change, sealed_rerun, "
        "public_publish); repeatable.",
    )
    p.set_defaults(func=cmd_decide)
