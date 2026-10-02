"""Served-model preflight: jev_factory/measure/preflight.py (ported from nvsh's
tests/test_lfm_finetune_measure.py preflight and served-context tests)."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from jev_factory.measure import preflight as pf


class _ModelsHandler(http.server.BaseHTTPRequestHandler):
    status = 200
    ids: list[str] = ["good-model"]
    max_model_len: int | None = None
    owned_by: str | None = None
    props_n_ctx: int | None = None
    props_redirect = False

    def _send(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        path = self.path.rstrip("/")
        cls = _ModelsHandler
        if path == "/props" and cls.props_redirect:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2:9/props")
            self.end_headers()
            return
        if path == "/props" and cls.props_n_ctx is not None:
            self._send({"default_generation_settings": {"n_ctx": cls.props_n_ctx}})
            return
        if path not in ("/models", "/v1/models"):
            self._send({}, 404)
            return
        entries = []
        for model_id in cls.ids:
            entry: dict = {"id": model_id}
            if cls.max_model_len is not None:
                entry["max_model_len"] = cls.max_model_len
            if cls.owned_by is not None:
                entry["owned_by"] = cls.owned_by
            entries.append(entry)
        self._send({"data": entries}, cls.status)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def url():
    cls = _ModelsHandler
    cls.status, cls.ids, cls.max_model_len = 200, ["good-model"], None
    cls.owned_by, cls.props_n_ctx, cls.props_redirect = None, None, False
    server = http.server.HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_accepts_a_server_serving_the_model(url):
    pf.preflight_models(url, "good-model")


def test_refuses_the_wrong_model_name(url):
    with pytest.raises(pf.PreflightError, match="other-model"):
        pf.preflight_models(url, "other-model")


def test_refuses_a_wrong_http_status(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "status", 500)
    with pytest.raises(pf.PreflightError, match="500"):
        pf.preflight_models(url, "good-model")


def test_refuses_a_refused_connection():
    with pytest.raises(pf.PreflightError):
        pf.preflight_models("http://127.0.0.1:1", "good-model")


@pytest.mark.parametrize(
    "bad", ["https://127.0.0.1:1", "http://example.invalid:1", "http://user:pw@127.0.0.1:1"]
)
def test_refuses_anything_but_plain_localhost(bad):
    with pytest.raises(pf.PreflightError):
        pf.preflight_models(bad, "good-model")


def test_accepts_a_matching_served_context(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "max_model_len", 4096)
    pf.preflight_models(url, "good-model", ctx=4096)


def test_refuses_a_different_served_context(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "max_model_len", 2048)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.preflight_models(url, "good-model", ctx=4096)
    assert "2048" in str(excinfo.value) and "4096" in str(excinfo.value)


def test_refuses_when_the_served_context_cannot_be_read(url):
    with pytest.raises(pf.PreflightError, match="max_model_len"):
        pf.preflight_models(url, "good-model", ctx=4096)


@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_reads_a_llama_server_context_from_props(url, monkeypatch, suffix):
    monkeypatch.setattr(_ModelsHandler, "owned_by", "llamacpp")
    monkeypatch.setattr(_ModelsHandler, "props_n_ctx", 2048)
    pf.preflight_models(url + suffix, "good-model", ctx=2048)


def test_refuses_a_llama_server_with_another_context(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "owned_by", "llamacpp")
    monkeypatch.setattr(_ModelsHandler, "props_n_ctx", 4096)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.preflight_models(url + "/v1", "good-model", ctx=2048)
    assert "4096" in str(excinfo.value)


def test_refuses_a_llama_server_without_props(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "owned_by", "llamacpp")
    with pytest.raises(pf.PreflightError, match="n_ctx"):
        pf.preflight_models(url + "/v1", "good-model", ctx=2048)


def test_a_redirected_props_answer_is_refused(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "owned_by", "llamacpp")
    monkeypatch.setattr(_ModelsHandler, "props_redirect", True)
    with pytest.raises(pf.PreflightError):
        pf.preflight_models(url + "/v1", "good-model", ctx=2048)


def test_props_are_not_consulted_for_a_server_that_is_not_llama_cpp(url, monkeypatch):
    monkeypatch.setattr(_ModelsHandler, "props_n_ctx", 2048)
    with pytest.raises(pf.PreflightError):
        pf.preflight_models(url, "good-model", ctx=2048)
