"""``jev review <domain>`` -- operator review of a domain's seed corpus (deviation d16).

Three modes over the same append-only review file (``<seed stem>.review.jsonl``
beside the seed unless ``--review-file`` says otherwise):

* default, a dry run: what the recorded decisions would change in the seed,
  with conflicts and problems; nothing is written;
* ``--serve``: the local review site (React Flow over the domain's verb tree),
  where each decision is appended to the review file;
* ``--apply``: write the planned seed. It refuses when a decision conflicts
  with the seed as it stands now or the result does not validate.

``--corrections FILE`` switches to a corrections review (distillery t14): the same three
modes over a corrections JSONL, whose review file is the record. Only approved and edited
corrections are trainable (:mod:`jev_factory.review.corrections`).
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
_FREE_PORT = "pass a free --port"
_CHECK_PATH = "check the path"
_FIX_FILE = "fix or restore the file"


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


def _serve(
    domain: Domain, review_file: Path, port: int, json_mode: bool, allow_hosts: tuple[str, ...] = ()
) -> int:
    from jev_factory.review import server

    seed = Path(domain.seed_corpus)  # type: ignore[arg-type]
    app = server.ReviewApp(domain, seed, review_file)
    httpd = _listen(server, app, port, allow_hosts)
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


def _corrections_mode(args: argparse.Namespace, json_mode: bool) -> int:
    """Review distillery corrections: ``--serve`` stages decisions, ``--apply`` records them."""
    from jev_factory.review import corrections as cr

    path = Path(args.corrections)
    if not path.is_file():
        raise CliError(EXIT_ENV_ERROR, f"corrections file not found: {path}", _CHECK_PATH)
    review_file = Path(args.review_file) if args.review_file else cr.review_path(path)
    draft = cr.draft_path(path)
    if args.serve:
        return _serve_corrections(
            path, review_file, args.port, json_mode, tuple(args.allow_host or ())
        )
    corrections, recorded, staged = _read_corrections_review(path, review_file, draft)
    by_key = cr.current(corrections)
    new = _unrecorded(staged, cr.latest(recorded))
    stale = [r["key"] for r in new if r["key"] not in by_key or cr.is_stale(by_key[r["key"]], r)]
    plan = cr.plan(corrections, [*recorded, *new])
    result = {
        "corrections": str(path),
        "review_file": str(review_file),
        "counts": plan.counts,
        "staged": len(new),
        "stale": stale,
        "conflicts": plan.conflicts,
        "applied": False,
        "recorded": 0,
    }
    if args.apply:
        if stale:
            raise CliError(
                EXIT_USER_ERROR,
                f"refusing to record: {len(stale)} staged decision(s) are stale; first: {stale[0]}",
                "re-decide those corrections in the review site, then retry",
            )
        for record in new:
            cr.append_record(review_file, record)
        draft.unlink(missing_ok=True)
        result.update(applied=bool(new), recorded=len(new))
    lines = [
        f"corrections: {path}",
        f"review file: {review_file}",
        "decisions: " + ", ".join(f"{n} {k}" for k, n in plan.counts.items()),
        f"{'recorded' if args.apply else 'would record'} {len(new)} staged decision(s)",
    ]
    lines.extend(f"stale: {k}" for k in stale)
    if new and not args.apply:
        lines.append("dry run: nothing was written; pass --apply to record the decisions")
    emit_result(result if json_mode else "\n".join(lines), json_mode=json_mode)
    return 0


def _read_corrections_review(path: Path, review_file: Path, draft: Path) -> tuple[Any, Any, Any]:
    """The corrections, the recorded decisions and the latest staged decision per key."""
    from jev_factory.distill.corrections import read_corrections
    from jev_factory.review import corrections as cr

    try:
        corrections = read_corrections(path)
        recorded = cr.read_records(review_file)
        staged = list(cr.latest(cr.read_records(draft)).values())
    except ValueError as exc:  # CorrectionError and CorrectionReviewError are ValueErrors
        raise CliError(EXIT_USER_ERROR, str(exc), _FIX_FILE) from None
    return corrections, recorded, staged


def _unrecorded(staged: list[Any], last: Any) -> list[Any]:
    """The staged decisions whose action or edit differs from the last recorded one."""
    return [
        r
        for r in staged
        if (last.get(r["key"]) or {}).get("action") != r["action"]
        or (last.get(r["key"]) or {}).get("after") != r.get("after")
    ]


def _serve_corrections(
    path: Path, review_file: Path, port: int, json_mode: bool, allow_hosts: tuple[str, ...] = ()
) -> int:
    from jev_factory.review import server

    app = server.CorrectionsApp(path, review_file)
    httpd = _listen(server, app, port, allow_hosts)
    url = server.local_url(httpd.server_address[1])
    emit_result(
        {"url": url, "draft": str(app.draft_file)} if json_mode else url, json_mode=json_mode
    )
    sys.stdout.flush()
    emit_diagnostic(f"serving the corrections review at {url} (Ctrl-C to stop)")
    emit_diagnostic(f"decisions are staged in {app.draft_file}; record them with --apply")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        emit_diagnostic("stopped")
    finally:
        httpd.server_close()
    return 0


def _serve_gold(
    path: Path, ledger_file: Path, port: int, json_mode: bool, allow_hosts: tuple[str, ...] = ()
) -> int:
    from jev_factory.review import server

    app = server.GoldApp(path, ledger_file)
    httpd = _listen(server, app, port, allow_hosts)
    url = server.local_url(httpd.server_address[1])
    emit_result({"url": url, "ledger": str(ledger_file)} if json_mode else url, json_mode=json_mode)
    sys.stdout.flush()
    emit_diagnostic(f"serving the gold review at {url} (Ctrl-C to stop)")
    emit_diagnostic(f"decisions are appended to {ledger_file}; edits rewrite {path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        emit_diagnostic("stopped")
    finally:
        httpd.server_close()
    return 0


def _gold_decisions(args: argparse.Namespace) -> list[tuple[str, str, tuple[str, str] | None]]:
    """``(record id, action, edit)`` from ``--decide ID=ACTION`` and ``--edit ID:QID=LABEL``."""
    out: list[tuple[str, str, tuple[str, str] | None]] = []
    for item in args.decide or []:
        rid, sep, action = item.rpartition("=")
        if not sep or not rid:
            raise CliError(EXIT_USER_ERROR, f"bad --decide {item!r}", "use ID=approve|reject")
        if action == "edit":
            raise CliError(
                EXIT_USER_ERROR, f"bad --decide {item!r}", "edit with --edit ID:QID=LABEL"
            )
        out.append((rid, action, None))
    for item in args.edit or []:
        target, sep, label = item.rpartition("=")
        rid, colon, qid = target.rpartition(":")
        if not (sep and colon and rid and qid and label):
            raise CliError(EXIT_USER_ERROR, f"bad --edit {item!r}", "use ID:QID=LABEL")
        out.append((rid, "edit", (qid, label)))
    return out


def _gold_entries(records: list[Any], decisions: list[Any], args: argparse.Namespace) -> tuple:
    """Ledger entries for the decisions, and the records after the edits among them."""
    from jev_factory.review import gold

    working = {r.id: r for r in records}
    entries = []
    for rid, action, edit in decisions:
        if rid not in working:
            raise CliError(EXIT_USER_ERROR, f"no test record {rid!r}", "check the record id")
        try:
            entry, after = gold.make_entry(
                working[rid], action, reviewer=args.reviewer, note=args.note or "", edit=edit
            )
        except ValueError as exc:
            raise CliError(EXIT_USER_ERROR, str(exc), "fix the decision") from None
        working[rid] = after
        entries.append(entry)
    return entries, working


def _gold_mode(args: argparse.Namespace, json_mode: bool) -> int:
    """Review drafted human-gold test records; ``--apply`` appends to the gold ledger."""
    from jev_factory.review import gold

    path = Path(args.gold)
    if not path.is_file():
        raise CliError(EXIT_ENV_ERROR, f"gold records file not found: {path}", _CHECK_PATH)
    ledger_file = Path(args.review_file) if args.review_file else gold.ledger_path(path)
    if args.serve:
        return _serve_gold(path, ledger_file, args.port, json_mode, tuple(args.allow_host or ()))
    try:
        drafted = gold.read_drafted(path)
        ledger = gold.read_ledger(ledger_file)
    except ValueError as exc:
        raise CliError(EXIT_USER_ERROR, str(exc), _FIX_FILE) from None
    test = [r for r in drafted if r.split == gold.TEST_SPLIT]
    entries, working = _gold_entries(test, _gold_decisions(args), args)
    edited = any(e["action"] == "edit" for e in entries)
    if args.apply:
        for entry in entries:
            gold.append_entry(ledger_file, entry)
        if edited:
            gold.write_drafted(path, [working.get(r.id, r) for r in drafted])
    shown = [working[r.id] for r in test]
    after_plan = gold.plan([*ledger, *entries], shown)
    latest = {(e["record_id"], e["record_sha256"]): e["action"] for e in [*ledger, *entries]}
    rows = [
        {
            "id": r.id,
            "status": gold.status_of(r, latest),
            "source": r.source,
            "gold": {q: g.to_json() for q, g in r.gold.items()},
        }
        for r in shown
    ]
    result = {
        "records": str(path),
        "ledger": str(ledger_file),
        "test_records": len(test),
        "skipped_non_test": len(drafted) - len(test),
        "counts": after_plan.counts,
        "orphans": after_plan.orphans,
        "decisions": len(entries),
        "applied": bool(args.apply and entries),
        "rows": rows,
    }
    emit_result(
        result if json_mode else _render_gold(result, rows, args.apply), json_mode=json_mode
    )
    return 0


def _render_gold(result: dict[str, Any], rows: list[dict[str, Any]], applied: bool) -> str:
    lines = [f"records: {result['records']}", f"ledger: {result['ledger']}"]
    lines.append(f"{result['test_records']} test record(s), {result['skipped_non_test']} other")
    lines.append("status: " + ", ".join(f"{n} {k}" for k, n in result["counts"].items()))
    for row in rows:
        gold_text = "; ".join(f"{q}={g.get('label')}" for q, g in row["gold"].items())
        lines.append(f"  {row['id']} [{row['status']}] {gold_text} ({row['source']})")
    lines.extend(f"orphan ledger entry: {o}" for o in result["orphans"])
    verb = "recorded" if applied else "would record"
    lines.append(f"{verb} {result['decisions']} decision(s)")
    if result["decisions"] and not applied:
        lines.append("dry run: nothing was written; pass --apply to record the decisions")
    return "\n".join(lines)


def _serve_drafts(app: Any, port: int, json_mode: bool, allow_hosts: tuple[str, ...] = ()) -> int:
    from jev_factory.review import server

    httpd = _listen(server, app, port, allow_hosts)
    url = server.local_url(httpd.server_address[1])
    ledger = app.ledger_file
    emit_result({"url": url, "ledger": str(ledger)} if json_mode else url, json_mode=json_mode)
    sys.stdout.flush()
    emit_diagnostic(f"serving the drafts review at {url} (Ctrl-C to stop)")
    emit_diagnostic(f"decisions are appended to {ledger}; {app.records} is not changed")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        emit_diagnostic("stopped")
    finally:
        httpd.server_close()
    return 0


def _listen(server: Any, app: Any, port: int, allow_hosts: tuple[str, ...]) -> Any:
    """The review server on 127.0.0.1; a busy port or a bad ``--allow-host`` is a CliError."""
    try:
        return server.make_server(app, port, allow_hosts=allow_hosts)
    except ValueError as exc:
        raise CliError(
            EXIT_USER_ERROR, str(exc), "pass a plain host name, e.g. --allow-host spark"
        ) from None
    except OSError as exc:
        raise CliError(
            EXIT_ENV_ERROR, f"cannot listen on {server.HOST}:{port}: {exc}", _FREE_PORT
        ) from None


def _drafts_entries(
    records: list[Any], ledger: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    from jev_factory.review import drafts

    by_id = {r.id: r for r in records}
    resolved = drafts.resolve(ledger, records)
    entries = []
    for rid, action, edit in _gold_decisions(args):
        if rid not in by_id:
            raise CliError(EXIT_USER_ERROR, f"no record {rid!r}", "check the record id")
        # checked against the record as the ledger leaves it (added options count), as the
        # page does
        current = drafts.effective(by_id[rid], resolved[rid])
        try:
            entries.append(
                drafts.make_entry(
                    by_id[rid],
                    action,
                    reviewer=args.reviewer,
                    note=args.note or "",
                    edit=edit,
                    current=current,
                )
            )
        except ValueError as exc:
            raise CliError(EXIT_USER_ERROR, str(exc), "fix the decision") from None
    return entries


def _drafts_inputs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """The ``--drafts`` file and the optional ``--evidence`` file, each checked to exist."""
    path = Path(args.drafts)
    if not path.is_file():
        raise CliError(EXIT_ENV_ERROR, f"drafted records file not found: {path}", _CHECK_PATH)
    evidence = Path(args.evidence) if args.evidence else None
    if evidence is not None and not evidence.is_file():
        raise CliError(EXIT_ENV_ERROR, f"evidence file not found: {evidence}", _CHECK_PATH)
    return path, evidence


def _drafts_mode(args: argparse.Namespace, json_mode: bool) -> int:
    """Review drafted records of every split; ``--apply`` appends to the draft ledger."""
    from jev_factory.review import drafts, gold

    path, evidence = _drafts_inputs(args)
    ledger_file = Path(args.review_file) if args.review_file else drafts.ledger_path(path)
    if args.serve:
        from jev_factory.review import server

        app = server.DraftsApp(path, evidence, ledger_file)
        return _serve_drafts(app, args.port, json_mode, tuple(args.allow_host or ()))
    try:
        records = gold.read_drafted(path)
        ledger = drafts.read_ledger(ledger_file)
        drafts.read_evidence(evidence)
        entries = _drafts_entries(records, ledger, args)
        after = [*ledger, *entries]
        resolved = drafts.resolve(after, records)
        reviewed = drafts.apply_ledger(records, after)
    except ValueError as exc:
        raise CliError(EXIT_USER_ERROR, str(exc), _FIX_FILE) from None
    out = Path(args.reviewed_out) if args.reviewed_out else None
    if args.apply:
        for entry in entries:
            drafts.append_entry(ledger_file, entry)
        if out is not None:
            gold.write_drafted(out, reviewed)
    result = {
        "records": str(path),
        "ledger": str(ledger_file),
        "total": len(records),
        "counts": drafts.counts(resolved),
        "decisions": len(entries),
        "applied": bool(args.apply and entries),
        "reviewed_records": len(reviewed),
        "reviewed_out": str(out) if out and args.apply else None,
        "orphans": drafts.orphans(after, records),
    }
    lines = _drafts_lines(result, apply=bool(args.apply), dry_run=bool(entries or out))
    emit_result(result if json_mode else "\n".join(lines), json_mode=json_mode)
    return 0


def _drafts_lines(result: dict[str, Any], *, apply: bool, dry_run: bool) -> list[str]:
    """The human rendering of a ``--drafts`` result."""
    lines = [
        f"records: {result['records']}",
        f"ledger: {result['ledger']}",
        f"{result['total']} record(s): "
        + ", ".join(f"{n} {k}" for k, n in result["counts"].items()),
        f"the reviewed file would hold {result['reviewed_records']} record(s)",
    ]
    lines.extend(f"orphan ledger entry: {o}" for o in result["orphans"])
    lines.append(f"{'recorded' if apply else 'would record'} {result['decisions']} decision(s)")
    if result["reviewed_out"]:
        lines.append(f"wrote the reviewed drafts to {result['reviewed_out']}")
    if dry_run and not apply:
        lines.append("dry run: nothing was written; pass --apply to record and write")
    return lines


def cmd_review(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    if args.serve and args.apply:
        raise CliError(EXIT_USER_ERROR, "--serve and --apply are separate steps", "pick one")
    if args.drafts:
        return _drafts_mode(args, json_mode)
    if args.gold:
        return _gold_mode(args, json_mode)
    if args.corrections:
        return _corrections_mode(args, json_mode)
    if not args.domain:
        raise CliError(
            EXIT_USER_ERROR,
            "a domain (or --corrections FILE) is required",
            "pass a domain, or --corrections <corrections.jsonl>",
        )
    return _seed_mode(args, json_mode)


def _seed_mode(args: argparse.Namespace, json_mode: bool) -> int:
    """Review a domain's seed corpus: dry-run plan, ``--serve`` the site, ``--apply`` to write."""
    domain = _domain(args.domain)
    seed = Path(domain.seed_corpus)  # type: ignore[arg-type]
    review_file = Path(args.review_file) if args.review_file else core.review_path(seed)
    if args.serve:
        return _serve(domain, review_file, args.port, json_mode, tuple(args.allow_host or ()))
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
    p.add_argument(
        "domain",
        nargs="?",
        help="Domain: a dotted module exposing DOMAIN, or a JSON file (not with --corrections).",
    )
    p.add_argument(
        "--corrections",
        metavar="FILE",
        help="Review a corrections JSONL instead of a seed: decisions are staged by --serve and "
        "recorded in <stem>.review.jsonl by --apply.",
    )
    p.add_argument(
        "--gold",
        metavar="FILE",
        help="Review drafted human-gold test records (JSONL): list them with proposed gold; "
        "--decide/--edit with --apply append to <stem>.gold-ledger.jsonl.",
    )
    p.add_argument(
        "--drafts",
        metavar="FILE",
        help="Review drafted records of every split (JSONL) one at a time: --serve for the "
        "site, or --decide ID=keep|drop|note / --edit with --apply to append to "
        "<stem>.draft-ledger.jsonl.",
    )
    p.add_argument(
        "--evidence", metavar="FILE", help="With --drafts: evidence JSONL {id, group, quote, why}."
    )
    p.add_argument(
        "--reviewed-out",
        dest="reviewed_out",
        metavar="FILE",
        help="With --drafts --apply: write the reviewed drafts (dropped removed, edits applied).",
    )
    p.add_argument(
        "--decide",
        action="append",
        metavar="ID=ACTION",
        help="With --gold: approve or reject. With --drafts: keep, drop or note.",
    )
    p.add_argument(
        "--edit",
        action="append",
        metavar="ID:QID=LABEL",
        help="With --gold or --drafts: correct the gold.",
    )
    p.add_argument("--reviewer", help="With --gold: who decides (default operator).")
    p.add_argument("--note", help="With --gold: a note stored with each decision.")
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
        "--allow-host",
        action="append",
        metavar="NAME",
        help="With --serve: also accept this exact Host name (e.g. the tailnet name that "
        "'tailscale serve' forwards); the site still listens on 127.0.0.1 only. Repeatable.",
    )
    p.add_argument(
        "--apply", action="store_true", help="Write the reviewed seed (default: dry run)."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_review)
