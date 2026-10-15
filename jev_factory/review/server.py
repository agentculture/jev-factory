"""The local review site behind ``jev review <domain> --serve``.

A stdlib HTTP server bound to the loopback interface only. It serves one page
(``page.html``: React and React Flow from pinned CDN URLs with integrity hashes)
and a small JSON API:

* ``GET /api/state``: the verb tree, every seed entry with the node it attaches
  to and its current decision, the open proposals and free ids;
* ``POST /api/check``: check an entry against the domain without recording it;
* ``POST /api/decision``: record one decision in the review file.

Every API call must carry the per-run token the page was served with
(``X-Review-Key``), and the ``Host`` header must name the loopback address
the server is bound to, so another web page in the same browser cannot read or
write the review. The server appends to the review file and never writes the
seed: that is ``jev review --apply``.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlunsplit

from jev_factory.distill.corrections import read_corrections
from jev_factory.domain.model import Domain
from jev_factory.review import core
from jev_factory.review import corrections as corr_review
from jev_factory.review import drafts as drafts_review
from jev_factory.review import gold as gold_review

HOST = "127.0.0.1"
AUTH_HEADER = "X-Review-Key"
MAX_BODY = 1 << 20
PAGE = Path(__file__).with_name("page.html")
GOLD_PAGE = Path(__file__).with_name("gold_page.html")
DRAFTS_PAGE = Path(__file__).with_name("drafts_page.html")
_KEY_SLOT = "__JEV_REVIEW_KEY__"
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline' https://unpkg.com; "
    "style-src 'unsafe-inline' https://unpkg.com; connect-src 'self'; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class ReviewApp:
    """The review's state and decisions, independent of HTTP (tests drive it directly)."""

    def __init__(self, domain: Domain, seed: Path, review_file: Path) -> None:
        self.domain = domain
        self.seed = seed
        self.review_file = review_file
        self.token = secrets.token_hex(16)
        #: The page, rendered once from the packaged file and this run's key.
        self.page = page(self.token)
        self._lock = threading.Lock()

    def _raw_seed(self) -> dict[str, Any]:
        return json.loads(self.seed.read_text(encoding="utf-8"))

    def state(self) -> dict[str, Any]:
        raw = self._raw_seed()
        records = core.read_records(self.review_file)
        decided = core.latest(records)
        proposed = core.proposals(records)
        entries = []
        for entry in raw.get("entries", []):
            record = decided.get(str(entry.get("id")))
            entries.append(
                {
                    "entry": entry,
                    "node": core.attach(entry, self.domain),
                    "status": core.status(record),
                    "decision": _public(record),
                }
            )
        for entry_id, entry in proposed.items():
            entries.append(
                {
                    "entry": entry,
                    "node": core.attach(entry, self.domain),
                    "status": "proposed",
                    "decision": _public(decided[entry_id]),
                }
            )
        plan = core.plan_changes(raw, records, self.domain)
        return {
            "domain": self.domain.name,
            "seed": {"path": str(self.seed), "entries": len(raw.get("entries", []))},
            "review_file": str(self.review_file),
            "nodes": core.verb_tree(self.domain),
            "entries": entries,
            "classes": sorted(
                {str(e.get("class")) for e in raw.get("entries", []) if "class" in e}
            ),
            "kinds": sorted({str(e.get("kind")) for e in raw.get("entries", []) if "kind" in e}),
            "sources": sorted(
                {str(e.get("source")) for e in raw.get("entries", []) if "source" in e}
                | {str(e.get("source")) for e in proposed.values() if e.get("source")}
            ),
            "next_ids": core.next_ids(raw, self.domain, taken=proposed),
            "pending_changes": len(plan.changes),
            "conflicts": plan.conflicts,
            "problems": plan.problems,
            "counts": plan.counts,
        }

    def check(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"problems": core.check_entry(payload.get("entry"), self.domain)}

    def decide(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            raw = self._raw_seed()
            records = core.read_records(self.review_file)
            record = core.make_record(raw, self.domain, payload, proposals=core.proposals(records))
            core.append_record(self.review_file, record)
        return {"recorded": True, "action": record["action"]}


class CorrectionsApp:
    """The corrections-mode review (``jev review --corrections``), independent of HTTP.

    Decisions are staged in the draft file, never in the record: ``jev review
    --corrections --apply`` moves them to the corrections review file. The page shows the
    draft laid over the record, so the operator sees what applying would make current.
    """

    def __init__(self, corrections: Path, review_file: Path | None = None) -> None:
        self.corrections = corrections
        self.review_file = review_file or corr_review.review_path(corrections)
        self.draft_file = corr_review.draft_path(corrections)
        self.token = secrets.token_hex(16)
        self.page = page(self.token)
        self._lock = threading.Lock()

    def state(self) -> dict[str, Any]:
        corrections = read_corrections(self.corrections)
        recorded = corr_review.latest(corr_review.read_records(self.review_file))
        staged = corr_review.latest(corr_review.read_records(self.draft_file))
        items = []
        for key, correction in corr_review.current(corrections).items():
            record = staged.get(key) or recorded.get(key)
            if record is not None and corr_review.is_stale(correction, record):
                record = None
            items.append(
                {
                    "key": key,
                    "correction": correction.to_json(),
                    "status": corr_review.status_of(record),
                    "staged": key in staged,
                    "decision": _public_correction(record),
                }
            )
        overlay = list(corr_review.latest([*recorded.values(), *staged.values()]).values())
        return {
            "mode": "corrections",
            "corrections_file": str(self.corrections),
            "review_file": str(self.review_file),
            "items": items,
            "staged": len(staged),
            "counts": corr_review.plan(corrections, overlay).counts,
        }

    def check(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            corr_review.make_record(read_corrections(self.corrections), payload)
        except corr_review.CorrectionReviewError as exc:
            return {"problems": [str(exc)]}
        return {"problems": []}

    def decide(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = corr_review.make_record(read_corrections(self.corrections), payload)
            corr_review.append_record(self.draft_file, record)
        return {"recorded": True, "staged": True, "action": record["action"]}


class GoldApp:
    """The gold-mode review (``jev review --gold --serve``), independent of HTTP.

    Every read and write goes through :mod:`jev_factory.review.gold`: a decision is one
    append-only ledger entry bound to the record's current sha256; an edit also rewrites
    the records file, exactly as ``--gold --apply`` does. A malformed ledger or records
    file raises :class:`GoldLedgerError` before anything is written.
    """

    def __init__(self, records: Path, ledger_file: Path | None = None) -> None:
        self.records = records
        self.ledger_file = ledger_file or gold_review.ledger_path(records)
        self.token = secrets.token_hex(16)
        self.page = GOLD_PAGE.read_text(encoding="utf-8").replace(_KEY_SLOT, self.token).encode()
        self._lock = threading.Lock()

    def _load(self) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
        drafted = gold_review.read_drafted(self.records)
        ledger = gold_review.read_ledger(self.ledger_file)
        test = [r for r in drafted if r.split == gold_review.TEST_SPLIT]
        return drafted, test, ledger

    def state(self) -> dict[str, Any]:
        drafted, test, ledger = self._load()
        latest = {(e["record_id"], e["record_sha256"]): e["action"] for e in ledger}
        items = []
        for record in test:
            sha = gold_review.record_sha256(record)
            items.append(
                {
                    "id": record.id,
                    "sha256": sha,
                    "status": gold_review.status_of(record, latest),
                    "source": record.source,
                    "state": record.state,
                    "gold_provenance": record.gold_provenance,
                    "questions": {
                        q: question.to_json() for q, question in record.questions.items()
                    },
                    "gold": {q: g.to_json() for q, g in record.gold.items()},
                }
            )
        plan = gold_review.plan(ledger, test)
        return {
            "mode": "gold",
            "records_file": str(self.records),
            "ledger_file": str(self.ledger_file),
            "test_records": len(test),
            "skipped_non_test": len(drafted) - len(test),
            "items": items,
            "counts": plan.counts,
            "orphans": plan.orphans,
        }

    def check(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a decision without recording it."""
        try:
            _drafted, test, _ledger = self._load()
            record = next((r for r in test if r.id == payload.get("record_id")), None)
            if record is None:
                raise gold_review.GoldLedgerError(f"no test record {payload.get('record_id')!r}")
            edit = None
            if payload.get("action") == "edit":
                edit = (str(payload.get("question_id") or ""), str(payload.get("label") or ""))
            gold_review.make_entry(record, str(payload.get("action")), edit=edit)
        except gold_review.GoldLedgerError as exc:
            return {"problems": [str(exc)]}
        return {"problems": []}

    def decide(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            drafted, test, _ledger = self._load()
            record = next((r for r in test if r.id == payload.get("record_id")), None)
            if record is None:
                raise gold_review.GoldLedgerError(f"no test record {payload.get('record_id')!r}")
            seen = payload.get("record_sha256")
            if seen is not None and seen != gold_review.record_sha256(record):
                raise gold_review.GoldLedgerError(
                    f"{record.id}: the record changed since the page loaded; reload"
                )
            edit = None
            if payload.get("action") == "edit":
                edit = (str(payload.get("question_id") or ""), str(payload.get("label") or ""))
            entry, after = gold_review.make_entry(
                record,
                str(payload.get("action")),
                reviewer=_text(payload.get("reviewer")),
                note=_text(payload.get("note")) or "",
                edit=edit,
            )
            gold_review.append_entry(self.ledger_file, entry)
            if edit is not None:
                gold_review.write_drafted(
                    self.records, [after if r.id == after.id else r for r in drafted]
                )
        return {
            "recorded": True,
            "action": entry["action"],
            "record_sha256": entry["record_sha256"],
        }


class DraftsApp:
    """The drafts-mode review (``jev review --drafts --serve``), independent of HTTP.

    Every decision is one append-only entry in the draft-review ledger bound to the
    record's current sha256; the drafted file is never written (``apply_ledger`` makes the
    reviewed file). A malformed ledger, records or evidence file raises
    :class:`DraftReviewError` before anything is written.
    """

    def __init__(
        self, records: Path, evidence: Path | None = None, ledger_file: Path | None = None
    ) -> None:
        self.records = records
        self.evidence = evidence
        self.ledger_file = ledger_file or drafts_review.ledger_path(records)
        self.token = secrets.token_hex(16)
        self.page = DRAFTS_PAGE.read_text(encoding="utf-8").replace(_KEY_SLOT, self.token).encode()
        self._lock = threading.Lock()

    def _load(self) -> tuple[list[Any], list[dict[str, Any]], dict[str, dict[str, str]]]:
        try:
            drafted = gold_review.read_drafted(self.records)
        except gold_review.GoldLedgerError as exc:
            raise drafts_review.DraftReviewError(str(exc)) from None
        ledger = drafts_review.read_ledger(self.ledger_file)
        return drafted, ledger, drafts_review.read_evidence(self.evidence)

    def state(self) -> dict[str, Any]:
        drafted, ledger, evidence = self._load()
        resolved = drafts_review.resolve(ledger, drafted)
        items = []
        for record in drafted:
            res = resolved[record.id]
            effective = drafts_review.apply_ledger([record], _only(ledger, record.id))
            final = effective[0] if effective else record
            items.append(
                {
                    "id": record.id,
                    "sha256": gold_review.record_sha256(record),
                    "status": res.status,
                    "edits": res.edits,
                    "notes": res.notes,
                    "split": record.split,
                    "source": record.source,
                    "group": record.group,
                    "gold_provenance": record.gold_provenance,
                    "state": record.state,
                    "questions": {q: x.to_json() for q, x in record.questions.items()},
                    "original_gold": {q: g.to_json() for q, g in record.gold.items()},
                    "gold": {q: g.to_json() for q, g in final.gold.items()},
                    "evidence": evidence.get(record.id),
                }
            )
        return {
            "mode": "drafts",
            "records_file": str(self.records),
            "evidence_file": str(self.evidence) if self.evidence else None,
            "ledger_file": str(self.ledger_file),
            "items": items,
            "splits": sorted({i["split"] for i in items}),
            "sources": sorted({i["source"] for i in items}),
            "counts": drafts_review.counts(resolved),
            "orphans": drafts_review.orphans(ledger, drafted),
        }

    def _entry(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        drafted, _ledger, _evidence = self._load()
        record = next((r for r in drafted if r.id == payload.get("record_id")), None)
        if record is None:
            raise drafts_review.DraftReviewError(f"no record {payload.get('record_id')!r}")
        seen = payload.get("record_sha256")
        if seen is not None and seen != gold_review.record_sha256(record):
            raise drafts_review.DraftReviewError(
                f"{record.id}: the record changed since the page loaded; reload"
            )
        edit = None
        if payload.get("action") == "edit":
            edit = (str(payload.get("question_id") or ""), str(payload.get("label") or ""))
        entry = drafts_review.make_entry(
            record,
            str(payload.get("action")),
            reviewer=_text(payload.get("reviewer")),
            note=_text(payload.get("note")) or "",
            edit=edit,
        )
        return entry, record

    def check(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a decision without recording it."""
        try:
            self._entry(payload)
        except gold_review.GoldLedgerError as exc:
            return {"problems": [str(exc)]}
        return {"problems": []}

    def decide(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry, _record = self._entry(payload)
            drafts_review.append_entry(self.ledger_file, entry)
        return {
            "recorded": True,
            "action": entry["action"],
            "record_sha256": entry["record_sha256"],
        }


def _only(ledger: list[dict[str, Any]], record_id: str) -> list[dict[str, Any]]:
    return [e for e in ledger if e["record_id"] == record_id]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _public_correction(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: record.get(k) for k in ("at", "action", "after", "note", "reviewer")}


def _public(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: record.get(k) for k in ("at", "action", "after", "note")}


def page(token: str) -> bytes:
    return PAGE.read_text(encoding="utf-8").replace(_KEY_SLOT, token).encode("utf-8")


class _ReviewHandler(BaseHTTPRequestHandler):
    """The review site's request handler; :func:`make_handler` binds it to one app."""

    server_version = "jev-review"
    review_app: ReviewApp | CorrectionsApp | GoldApp | DraftsApp

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib name)
        """Quiet: the review site logs nothing per request."""

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"{HOST}:{port}", f"localhost:{port}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _authorised(self) -> bool:
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
            return False
        given = self.headers.get(AUTH_HEADER, "")
        if not hmac.compare_digest(given.encode(), self.review_app.token.encode()):
            self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong review token"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        if self.path in ("/", "/index.html"):
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
                return
            self._send(HTTPStatus.OK, self.review_app.page, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            if self._authorised():
                self._respond(self.review_app.state)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        routes = {"/api/decision": self.review_app.decide, "/api/check": self.review_app.check}
        handler = routes.get(self.path)
        if handler is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorised():
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "a JSON body up to 1 MiB is needed"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "the body is not JSON"})
            return
        if not isinstance(payload, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "the body must be a JSON object"})
            return
        self._respond(lambda: handler(payload))

    def _respond(self, call: Any) -> None:
        try:
            self._json(HTTPStatus.OK, call())
        except (core.ReviewError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def make_handler(
    app: ReviewApp | CorrectionsApp | GoldApp | DraftsApp,
) -> type[BaseHTTPRequestHandler]:
    class Handler(_ReviewHandler):
        review_app = app

    return Handler


def local_url(port: int) -> str:
    """The site's address. Plain HTTP is right here: the server only listens on loopback."""
    return urlunsplit(("http", f"{HOST}:{port}", "/", "", ""))


def make_server(
    app: ReviewApp | CorrectionsApp | GoldApp | DraftsApp, port: int
) -> ThreadingHTTPServer:
    """A server for *app* on the loopback interface (*port* 0 picks a free one)."""
    return ThreadingHTTPServer((HOST, port), make_handler(app))
