"""``jev-factory learn`` — the learnability affordance.

Prints a structured self-teaching prompt. Must satisfy the agent-first rubric:
>=200 chars and mention purpose, command map, exit codes, --json, and explain.
"""

from __future__ import annotations

import argparse

from jev_factory import __version__
from jev_factory.cli._output import emit_result

_TEXT = """\
jev-factory — a factory for jev-like candidate scorers, and an AgentCulture mesh agent.

Purpose
-------
Builds jev-like models: calibrated scorers that read one request plus a bounded
set of lettered candidate actions, answer in one token, and pass a
deterministic read-only/mutating gate. The build is a pipeline of registered
stages (`run`), each resumable with a manifest; every between-runs decision is
a recorded, rule-driven step (`decide`). Every write verb is dry-run by
default: nothing changes until you pass --apply.

Commands
--------
  jev-factory init <domain>      Scaffold a run directory and run config.
  jev-factory run <stage>        Run one build stage (--apply commits, --detach
                                 keeps a long stage alive past the shell).
  jev-factory status <run>       Stage manifests, staleness, job progress.
  jev-factory decide <run>       Apply the pre-registered rule; append a record.
  jev-factory whoami             Identity from culture.yaml.
  jev-factory learn              This self-teaching prompt.
  jev-factory explain <path>...  Markdown docs for any noun/verb path.
  jev-factory overview           Descriptive snapshot of the agent.
  jev-factory doctor             Check the agent-identity invariants.
  jev-factory cli overview       Describe the CLI surface itself.

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr never mix.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error
  3+ reserved

More detail
-----------
  jev-factory explain jev-factory
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "jev-factory",
        "version": __version__,
        "purpose": "Factory for jev-like calibrated candidate scorers (a mesh agent).",
        "commands": [
            {"path": ["init"], "summary": "Scaffold a run directory (dry-run by default)."},
            {"path": ["run", "<stage>"], "summary": "Run one build stage (dry-run by default)."},
            {"path": ["status"], "summary": "Stage manifests, staleness and job progress."},
            {"path": ["decide"], "summary": "Apply the pre-registered rule; record it."},
            {"path": ["whoami"], "summary": "Identity probe from culture.yaml."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by path."},
            {"path": ["overview"], "summary": "Descriptive snapshot of the agent."},
            {"path": ["doctor"], "summary": "Check the agent-identity invariants."},
            {"path": ["cli", "overview"], "summary": "Describe the CLI surface."},
        ],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "jev-factory explain <path>",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
