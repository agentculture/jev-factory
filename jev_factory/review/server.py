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

from jev_factory.domain.model import Domain
from jev_factory.review import core

HOST = "127.0.0.1"
AUTH_HEADER = "X-Review-Key"
MAX_BODY = 1 << 20
PAGE = Path(__file__).with_name("page.html")
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
                    "status": _status(record),
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
        return {"recorded": _public(record)}


def _status(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return core.STATUS_PENDING
    return {"approve": "approved", "reject": "rejected", "edit": "edited"}.get(
        str(record["action"]), "proposed"
    )


def _public(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: record.get(k) for k in ("at", "action", "after", "note")}


def page(token: str) -> bytes:
    return PAGE.read_text(encoding="utf-8").replace(_KEY_SLOT, token).encode("utf-8")


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "jev-review"

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
            if not hmac.compare_digest(given.encode(), app.token.encode()):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong review token"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802 (stdlib name)
            if self.path in ("/", "/index.html"):
                if not self._host_ok():
                    self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
                    return
                self._send(HTTPStatus.OK, page(app.token), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                if self._authorised():
                    self._respond(app.state)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (stdlib name)
            routes = {"/api/decision": app.decide, "/api/check": app.check}
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

    return Handler


def make_server(app: ReviewApp, port: int) -> ThreadingHTTPServer:
    """A server for *app* on the loopback interface (*port* 0 picks a free one)."""
    return ThreadingHTTPServer((HOST, port), make_handler(app))
