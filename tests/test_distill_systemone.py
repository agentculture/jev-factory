"""t4: the SystemOne client builds /v1/systemone bodies and reads only probabilities."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev_factory.distill import systemone as s1

RECORD = {
    "id": "r1",
    "state": "the build is red",
    "questions": {
        "q_choice": {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {"fix": "fix it", "wait": "wait", "page": "page oncall"},
        },
        "q_noul": {"type": "noul", "criteria": "is it urgent?"},
        "q_score": {"type": "score", "criteria": ["low", "mid", "high"]},
    },
}


def _answers(**over):
    out = {
        "q_choice": {
            "choice": "fix",
            "confidence": 0.99,
            "probabilities": {"fix": 0.6, "wait": 0.3, "page": 0.1},
        },
        "q_noul": {"noul": 0.25, "confidence": 0.01},
        "q_score": {
            "score": 1.1,
            "legend": ["low", "mid", "high"],
            "probabilities": {"low": 0.2, "mid": 0.5, "high": 0.3},
            "confidence": 0.01,
        },
    }
    out.update(over)
    return {"answers": out}


class _Server:
    def __init__(self, payload):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append((self.path, json.loads(body)))
                raw = json.dumps(outer.payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_a):
                pass

        self.payload = payload
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(payload):
        srv = _Server(payload)
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.close()


def _parsed(result):
    return json.dumps([q.to_dict() for q in result.questions], sort_keys=True)


def test_build_body_is_systemone_shape():
    body = s1.build_body(RECORD, model="clef")
    assert body["model"] == "clef"
    assert body["state"] == "the build is red"
    assert body["questions"]["q_choice"] == {
        "type": "choice",
        "instructions": "pick one",
        "criteria": {"fix": "fix it", "wait": "wait", "page": "page oncall"},
    }
    assert "instructions" not in body["questions"]["q_noul"]
    assert "gold" not in json.dumps(body)


def test_parse_distributions_over_exactly_offered_options():
    parsed = {q.qid: q for q in s1.parse_response(RECORD, _answers())}
    assert all(q.ok for q in parsed.values())
    assert list(parsed["q_choice"].distribution) == ["fix", "wait", "page"]
    assert parsed["q_choice"].distribution["fix"] == 0.6
    assert list(parsed["q_score"].distribution) == ["low", "mid", "high"]
    assert parsed["q_noul"].p_true == 0.25 and parsed["q_noul"].distribution is None


def test_score_probabilities_as_list_align_to_levels():
    ans = _answers(q_score={"score": 1, "probabilities": [0.2, 0.5, 0.3]})
    parsed = {q.qid: q for q in s1.parse_response(RECORD, ans)}
    assert parsed["q_score"].distribution == {"low": 0.2, "mid": 0.5, "high": 0.3}


def test_wrong_confidence_gives_byte_identical_parse(server):
    good = server(_answers())
    bad_payload = _answers()
    for ans in bad_payload["answers"].values():
        ans["confidence"] = -123.0
    bad_payload["answers"]["q_choice"]["choice"] = "page"  # also a lying argmax
    bad = server(bad_payload)
    r1 = s1.SystemOneClient(good.url, model="m").ask(RECORD)
    r2 = s1.SystemOneClient(bad.url, model="m").ask(RECORD)
    assert _parsed(r1) == _parsed(r2)
    assert "confidence" not in _parsed(r1)
    assert r1.response_sha256 != r2.response_sha256


@pytest.mark.parametrize(
    "probs",
    [
        {"fix": 0.6, "wait": 0.4},  # missing option
        {"fix": 0.5, "wait": 0.3, "page": 0.1, "other": 0.1},  # extra option
        {"fix": 0.6, "wait": 0.3, "page": 0.2},  # sum 1.1
        {"fix": 0.5, "wait": 0.3, "page": 0.1},  # sum 0.9
        {"fix": 1.2, "wait": -0.2, "page": 0.0},  # out of range
        {"fix": "x", "wait": 0.5, "page": 0.5},  # not a number
        {"fix": True, "wait": 0.0, "page": 0.0},  # bool is not a probability
    ],
)
def test_bad_choice_probabilities_are_errors_never_distributions(probs):
    ans = _answers(q_choice={"choice": "fix", "probabilities": probs})
    parsed = {q.qid: q for q in s1.parse_response(RECORD, ans)}
    bad = parsed["q_choice"]
    assert not bad.ok and bad.distribution is None and bad.error
    assert parsed["q_noul"].ok  # other questions are unaffected


def test_sum_tolerance_edges():
    ok = _answers(q_choice={"probabilities": {"fix": 0.605, "wait": 0.3, "page": 0.1}})
    assert {q.qid: q for q in s1.parse_response(RECORD, ok)}["q_choice"].ok
    no = _answers(q_choice={"probabilities": {"fix": 0.612, "wait": 0.3, "page": 0.1}})
    assert not {q.qid: q for q in s1.parse_response(RECORD, no)}["q_choice"].ok


def test_missing_question_and_bad_noul_are_errors():
    ans = _answers(q_noul={"noul": 1.5})
    del ans["answers"]["q_score"]
    parsed = {q.qid: q for q in s1.parse_response(RECORD, ans)}
    assert not parsed["q_noul"].ok and parsed["q_noul"].p_true is None
    assert not parsed["q_score"].ok
    assert not s1.parse_response(RECORD, "junk")[0].ok


def test_provenance_on_every_result(server):
    srv = server(_answers())
    client = s1.SystemOneClient(srv.url, model="clef-q4", timeout=3.0)
    result = client.ask(RECORD)
    path, sent = srv.requests[0]
    assert path == "/v1/systemone" and sent == s1.build_body(RECORD, model="clef-q4")
    assert result.model == "clef-q4" and result.record_id == "r1"
    assert result.request_config["timeout"] == 3.0
    assert result.request_config["body_sha256"]
    import hashlib

    raw = json.dumps(srv.payload).encode()
    assert result.response_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.error is None


def test_transport_failure_is_an_error_result():
    client = s1.SystemOneClient("http://127.0.0.1:9", model="m", timeout=1.0)
    result = client.ask(RECORD)
    assert result.error and not any(q.ok for q in result.questions)
    assert result.response_sha256 is None


@pytest.mark.parametrize(
    "url", ["http://example.com:8000", "https://127.0.0.1:8000", "http://u:p@127.0.0.1:1"]
)
def test_non_localhost_refused(url):
    with pytest.raises(ValueError):
        s1.SystemOneClient(url, model="m")


def test_stdlib_only():
    import inspect

    src = inspect.getsource(s1)
    for banned in ("requests", "httpx", "aiohttp", "confidence"):
        assert f"import {banned}" not in src
    # the field is never read: it appears nowhere as a key lookup
    assert '["confidence"]' not in src and '"confidence"' not in src
