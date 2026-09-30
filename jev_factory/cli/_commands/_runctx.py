"""Shared plumbing for the factory verbs (``init``, ``run``, ``status``, ``decide``).

A *run* is a work directory. ``jev init`` writes two small files into it,
``run.json`` (the domain reference and the config file name) and ``run.toml``
(the run config); every other verb finds the domain and config from there
unless a flag or ``JEV_DOMAIN`` says otherwise. Nothing here writes anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory.config import load_config
from jev_factory.factory.pipeline import RunContext

RUN_FILE = "run.json"
CONFIG_FILE = "run.toml"
DOMAIN_ENV = "JEV_DOMAIN"


def add_context_options(p: argparse.ArgumentParser, *, work: bool) -> None:
    """``--config``, ``--domain`` (and ``--work`` when the verb has no run positional)."""
    if work:
        p.add_argument(
            "--work",
            help="The run's work directory (outside any git worktree); "
            "default: the 'work' key of the run config.",
        )
    p.add_argument("--config", help="Run config TOML (default: <work>/run.toml, if present).")
    p.add_argument(
        "--domain",
        help="Domain: a dotted module or a JSON file (default: $JEV_DOMAIN, else <work>/run.json).",
    )
    p.add_argument("--deviation-id", help="Recorded deviation that authorises a changed plan.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")


def _read_run_file(work: Path) -> dict[str, Any]:
    path = work / RUN_FILE
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CliError(
            EXIT_USER_ERROR, f"{path} is unreadable: {exc}", "re-run 'jev init' or fix the file"
        ) from None
    return doc if isinstance(doc, dict) else {}


def parse_knobs(pairs: list[str] | None) -> dict[str, Any]:
    """``--knob key=value`` pairs; a value that is valid JSON is decoded, else kept as text."""
    knobs: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise CliError(
                EXIT_USER_ERROR,
                f"--knob expects key=value, got {pair!r}",
                "for example --knob per_op=3 or --knob review=false",
            )
        try:
            knobs[key] = json.loads(raw)
        except ValueError:
            knobs[key] = raw
    return knobs


def load_context(args: argparse.Namespace, work: str | None) -> tuple[Path | None, RunContext]:
    """The validated run context for a verb; raises :class:`CliError` (writes nothing)."""
    wd = Path(work).expanduser() if work else None
    run_doc = _read_run_file(wd) if wd is not None else {}
    config = getattr(args, "config", None)
    if config is None and wd is not None and (wd / CONFIG_FILE).is_file():
        config = str(wd / CONFIG_FILE)
    domain = getattr(args, "domain", None) or os.environ.get(DOMAIN_ENV) or run_doc.get("domain")
    if not domain:
        raise CliError(
            EXIT_USER_ERROR,
            "no domain for this run",
            f"pass --domain <module|file.json>, export {DOMAIN_ENV}, or run 'jev init'",
        )
    cfg = load_config(config, cli={"work": str(wd) if wd is not None else None})
    ctx = RunContext.load(
        str(domain),
        cfg,
        deviation_id=getattr(args, "deviation_id", None),
        allow_foreign_gpu=bool(getattr(args, "allow_foreign_gpu", False)),
    )
    return Path(cfg["work"]), ctx
