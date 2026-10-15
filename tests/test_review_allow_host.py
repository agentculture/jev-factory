"""`jev review --serve --allow-host NAME`: a tailnet proxy's exact name, socket still loopback."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from jev_factory.review import server
from tests.test_review_drafts import paths  # noqa: F401 (fixture)

TAILNET = "spark.tail0be7e0.ts.net"


def _get(port, path, host, key=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    headers = {"Host": host}
    if key:
        headers["X-Review-Key"] = key
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


@pytest.fixture
def allowed(paths):  # noqa: F811
    app = server.DraftsApp(*paths)
    httpd = server.make_server(app, 0, allow_hosts=(TAILNET + ".",))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, httpd
    httpd.shutdown()
    httpd.server_close()


def test_the_socket_stays_on_loopback(allowed):
    _app, httpd = allowed
    assert httpd.server_address[0] == "127.0.0.1"


@pytest.mark.parametrize("host", [TAILNET, f"{TAILNET}:8443", f"{TAILNET}:443"])
def test_the_allowed_tailnet_name_reaches_the_page_and_the_api(allowed, host):
    app, httpd = allowed
    port = httpd.server_address[1]
    assert _get(port, "/", host)[0] == 200
    status, body = _get(port, "/api/state", host, key=app.token)
    assert status == 200
    assert json.loads(body)["mode"] == "drafts"


@pytest.mark.parametrize("host", ["evil.example", f"x.{TAILNET}", "spark", "100.127.105.72:8443"])
def test_any_other_host_is_still_refused(allowed, host):
    app, httpd = allowed
    status, body = _get(httpd.server_address[1], "/api/state", host, key=app.token)
    assert status == 403
    assert json.loads(body)["error"] == "unexpected Host header"


def test_the_token_is_still_required_through_the_tailnet_name(allowed):
    _app, httpd = allowed
    assert _get(httpd.server_address[1], "/api/state", TAILNET, key="wrong")[0] == 403


def test_without_allow_host_the_tailnet_name_is_refused(paths):  # noqa: F811
    app = server.DraftsApp(*paths)
    httpd = server.make_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status = _get(httpd.server_address[1], "/api/state", TAILNET, key=app.token)[0]
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert status == 403


@pytest.mark.parametrize("bad", ["*.ts.net", "http://spark", "spark:8443", "", "a b", "-x.net"])
def test_a_malformed_allow_host_is_refused(bad):
    with pytest.raises(ValueError, match="not a plain host name"):
        server.check_allow_hosts((bad,))
