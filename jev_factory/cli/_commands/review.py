"""``jev review <domain>`` -- operator review of a domain's seed corpus (deviation d16).

Three modes over the same append-only review file (``<seed stem>.review.jsonl``
beside the seed unless ``--review-file`` says otherwise):

* default, a dry run: what the recorded decisions would change in the seed,
  with conflicts and problems; nothing is written;
* ``--serve``: the local review site (React Flow over the domain's verb tree),
  where each decision is appended to the review file;
* ``--apply``: write the planned seed. It refuses when a decision conflicts
  with the seed as it stands now or the result does not validate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.cli._output import emit_diagnostic, emit_result
from jev_factory.domain.model import Domain
from jev_factory.domain.validate import DomainError, load_domain
from jev_factory.review import core

DEFAULT_PORT = 18765


def _domain(ref: str) -> Domain:
    try:
        domain = load_domain(ref)
    except DomainError as exc:
        raise CliError(
            EXIT_USER_ERROR,
            f"invalid domain {ref!r}: {exc}",
            "pass a dotted module that exposes DOMAIN, or a domain JSON file",
        ) from exc
    if domain.seed_corpus is None:
        raise CliError(
            EXIT_USER_ERROR,
            f"domain {domain.name!r} declares no seed corpus",
            "set the domain's seed_corpus to the seed JSON file to review",
        )
    if not Path(domain.seed_corpus).is_file():
        raise CliError(
            EXIT_ENV_ERROR,
            f"seed corpus not found: {domain.seed_corpus}",
            "check the domain's seed_corpus path",
        )
    return domain


def _load(domain: Domain, review_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = Path(domain.seed_corpus)  # type: ignore[arg-type]
    try:
        raw = json.loads(seed.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(
            EXIT_USER_ERROR, f"{seed}: not JSON ({exc.msg})", "fix the seed file"
        ) from None
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise CliError(EXIT_USER_ERROR, f"{seed}: a seed needs an 'entries' list", "fix the seed")
    try:
        records = core.read_records(review_file)
    except core.ReviewError as exc:
        raise CliError(
            EXIT_USER_ERROR, str(exc), "the review file is append-only; restore it from git"
        ) from None
    return raw, records


def _describe(change: dict[str, Any]) -> str:
    entry_id = change["entry_id"]
    if change["action"] == "remove":
        return f"remove {entry_id}: {change['before'].get('text', '')!r}"
    if change["action"] == "add":
        return f"add {entry_id}: {change['after'].get('text', '')!r}"
    before, after = change["before"], change["after"]
    diffs = [
        f"{k}: {json.dumps(before.get(k), ensure_ascii=False)} -> "
        f"{json.dumps(after.get(k), ensure_ascii=False)}"
        for k in sorted(set(before) | set(after))
        if before.get(k) != after.get(k)
    ]
    return f"edit {entry_id}: " + "; ".join(diffs)


def _plan_result(domain: Domain, seed: Path, review_file: Path, plan: core.ChangePlan) -> dict:
    return {
        "domain": domain.name,
        "seed": str(seed),
        "review_file": str(review_file),
        "counts": plan.counts,
        "changes": plan.changes,
        "conflicts": plan.conflicts,
        "problems": plan.problems,
    }


def _render(result: dict[str, Any], *, applied: bool) -> str:
    counts = result["counts"]
    lines = [
        f"seed: {result['seed']}",
        f"review file: {result['review_file']}",
        "decisions: "
        + ", ".join(
            f"{counts.get(k, 0)} {k}"
            for k in ("pending", "approved", "rejected", "edited", "proposed", "withdrawn")
        ),
    ]
    verb = "changed" if applied else "would change"
    if result["changes"]:
        lines.append(f"{verb} {len(result['changes'])} entr(y/ies):")
        lines.extend("  " + _describe(c) for c in result["changes"])
    else:
        lines.append("no seed changes to apply")
    lines.extend(f"conflict: {c}" for c in result["conflicts"])
    lines.extend(f"problem: {p}" for p in result["problems"])
    if not applied and result["changes"]:
        lines.append("dry run: nothing was written; pass --apply to write the seed")
    return "\n".join(lines)


def _serve(domain: Domain, review_file: Path, port: int, json_mode: bool) -> int:
    from jev_factory.review import server

    seed = Path(domain.seed_corpus)  # type: ignore[arg-type]
    app = server.ReviewApp(domain, seed, review_file)
    try:
        httpd = server.make_server(app, port)
    except OSError as exc:
        raise CliError(
            EXIT_ENV_ERROR, f"cannot listen on {server.HOST}:{port}: {exc}", "pass a free --port"
        ) from None
    url = server.local_url(httpd.server_address[1])
    emit_result(
        {"url": url, "review_file": str(review_file)} if json_mode else url, json_mode=json_mode
    )
    sys.stdout.flush()  # the server blocks next; a piped caller must see the URL now
    emit_diagnostic(f"serving the {domain.name} seed review at {url} (Ctrl-C to stop)")
    emit_diagnostic(f"decisions are appended to {review_file}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        emit_diagnostic("stopped")
    finally:
        httpd.server_close()
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    if args.serve and args.apply:
        raise CliError(EXIT_USER_ERROR, "--serve and --apply are separate steps", "pick one")
    domain = _domain(args.domain)
    seed = Path(domain.seed_corpus)  # type: ignore[arg-type]
    review_file = Path(args.review_file) if args.review_file else core.review_path(seed)
    if args.serve:
        return _serve(domain, review_file, args.port, json_mode)
    raw, records = _load(domain, review_file)
    plan = core.plan_changes(raw, records, domain)
    result = _plan_result(domain, seed, review_file, plan)
    if args.apply:
        if not plan.ok:
            raise CliError(
                EXIT_USER_ERROR,
                f"refusing to write the seed: {len(plan.conflicts)} conflict(s), "
                f"{len(plan.problems)} problem(s); first: {(plan.conflicts + plan.problems)[0]}",
                "record a new decision for each listed entry in the review site, then retry",
            )
        if plan.changes:
            seed.write_text(core.dump_seed(plan.seed), encoding="utf-8")
    result["applied"] = bool(args.apply and plan.changes)
    emit_result(
        result if json_mode else _render(result, applied=result["applied"]), json_mode=json_mode
    )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "review",
        help="Review a domain's seed corpus: dry-run plan, --serve the site, --apply to write.",
    )
    p.add_argument("domain", help="Domain: a dotted module exposing DOMAIN, or a JSON file.")
    p.add_argument(
        "--review-file",
        dest="review_file",
        help="Append-only review file (default: <seed stem>.review.jsonl beside the seed).",
    )
    p.add_argument("--serve", action="store_true", help="Serve the local review site.")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Port for --serve (default {DEFAULT_PORT})."
    )
    p.add_argument(
        "--apply", action="store_true", help="Write the reviewed seed (default: dry run)."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_review)
