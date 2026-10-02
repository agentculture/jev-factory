"""``jev init <domain>`` -- scaffold a run directory for a domain.

Validates the domain, then writes ``run.json`` (domain reference and config
file name) and ``run.toml`` (every documented run-config key, the required ones
commented out unless given by flag) into the work directory. A write verb:
a dry run by default that lists the files and writes nothing; ``--apply``
writes them. It never overwrites an existing run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_factory.cli._commands._runctx import CONFIG_FILE, RUN_FILE
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_result
from jev_factory.domain.validate import DomainError, load_domain
from jev_factory.factory.config import KEYS, REQUIRED
from jev_factory.factory.workroot import resolve_work_root

_FLAG_KEYS = ("base", "base_rev", "hub_prefix", "licence", "seed")


def _toml_value(key, value) -> str:
    if key.kind in ("int", "float"):
        return str(value)
    if key.kind == "bool":
        return "true" if value else "false"
    return json.dumps(str(value))


def render_config(work: Path, given: dict[str, object]) -> str:
    lines = [
        "# Run config written by `jev init`. Precedence for every key:",
        "#   CLI flag > this file > environment (JEV_<KEY>) > default",
        "# No secret belongs here; *_env keys name the env var that holds one.",
        "",
    ]
    for key in KEYS:
        lines.append(f"# {key.doc}")
        if key.name == "work":
            lines.append(f"work = {json.dumps(str(work))}")
        elif key.name in given and given[key.name] is not None:
            lines.append(f"{key.name} = {_toml_value(key, given[key.name])}")
        elif key.default is REQUIRED:
            lines.append(f'# {key.name} = ""  # REQUIRED: set before running a stage')
        elif key.default is None:
            lines.append(f'# {key.name} = ""')
        else:
            lines.append(f"# {key.name} = {_toml_value(key, key.default)}")
        lines.append("")
    return "\n".join(lines)


def cmd_init(args: argparse.Namespace) -> int:
    try:
        domain = load_domain(args.domain)
    except DomainError as exc:
        raise CliError(
            EXIT_USER_ERROR,
            f"invalid domain {args.domain!r}: {exc}",
            "pass a dotted module that exposes DOMAIN, or a domain JSON file",
        ) from exc
    work = resolve_work_root(args.work)
    ref = str(Path(args.domain).resolve()) if Path(args.domain).is_file() else args.domain
    targets = {work / RUN_FILE: None, work / CONFIG_FILE: None}
    clash = [str(p) for p in targets if p.exists()]
    if clash:
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing to overwrite an existing run: {', '.join(clash)}",
            "choose a new --work directory",
        )
    given = {k: getattr(args, k) for k in _FLAG_KEYS}
    run_doc = json.dumps({"domain": ref, "config": CONFIG_FILE}, indent=2) + "\n"
    contents = {work / RUN_FILE: run_doc, work / CONFIG_FILE: render_config(work, given)}
    result = {
        "domain": domain.name,
        "work": str(work),
        "applied": bool(args.apply),
        "files": [str(p) for p in contents],
        "missing_required": [k for k in _FLAG_KEYS[:4] if given[k] is None],
    }
    if args.apply:
        work.mkdir(parents=True, exist_ok=True)
        for path, text in contents.items():
            path.write_text(text, encoding="utf-8")
    else:
        result["note"] = "dry run: nothing was written; pass --apply to create the run"
    json_mode = bool(args.json)
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        verb = "wrote" if args.apply else "would write"
        lines = [f"{verb}: {f}" for f in result["files"]]
        if result["missing_required"]:
            lines.append("still to set in run.toml: " + ", ".join(result["missing_required"]))
        if not args.apply:
            lines.append(result["note"])
        emit_result("\n".join(lines), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "init",
        help="Scaffold a run directory and run config for a domain (dry-run by default).",
    )
    p.add_argument("domain", help="Domain: a dotted module exposing DOMAIN, or a JSON file.")
    p.add_argument("--work", required=True, help="Work directory to create (outside any git tree).")
    p.add_argument("--base", help="Base model id (org/name).")
    p.add_argument("--base-rev", dest="base_rev", help="Base model revision (commit).")
    p.add_argument("--hub-prefix", dest="hub_prefix", help="Hub repo prefix for private bundles.")
    p.add_argument("--licence", help="Licence of the shipped bundle (SPDX id).")
    p.add_argument("--seed", type=int, help="Seed for the grouped split and assembly.")
    p.add_argument("--apply", action="store_true", help="Write the files (default: dry run).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
