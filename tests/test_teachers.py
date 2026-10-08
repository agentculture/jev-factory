"""Teacher client: JSON verdicts (o20), Apache-only (o22), cache + resume (o37)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.data import teachers as T
from jev_factory.factory.config import load_config
from jev_factory.factory.detach import ItemLedger, read_progress

KEYVAR = "TEST_TEACHER_KEY"
SECRET = "-".join(("canary", "value", "for", "tests"))  # built at runtime: not a real secret
REQUIRED = {
    "work": "w",
    "base": "org/base",
    "base_rev": "abc",
    "hub_prefix": "org/fam",
    "licence": "Apache-2.0",
}


def _counts(progress):
    """The items done/total of a progress record (it also carries timing)."""
    return {k: progress[k] for k in ("done", "total")}


class Gateway:
    """A local fake OpenAI-compatible gateway; ``replies`` is consumed in order."""

    def __init__(self, replies=None, default="ok"):
        self.requests: list[dict] = []
        self.auth: list[str] = []
        self.replies = list(replies or [])
        self.default = default
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.auth.append(self.headers.get("Authorization", ""))
                reply = outer.replies.pop(0) if outer.replies else outer.default
                if isinstance(reply, int):
                    self.send_response(reply)
                    self.end_headers()
                    return
                data = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway():
    made = []

    def make(*a, **k):
        g = Gateway(*a, **k)
        made.append(g)
        return g

    yield make
    for g in made:
        g.close()


def _config(tmp_path, url, **extra):
    values = {**REQUIRED, "work": str(tmp_path), "aug_url": url, "aug_key_env": KEYVAR, **extra}
    path = tmp_path / "run.toml"
    path.write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items()))
    return load_config(path, environ={})


def _client(tmp_path, url, **extra):
    cfg = _config(tmp_path, url, **extra)
    roles = T.load_roles(cfg, environ={KEYVAR: SECRET})
    return T.TeacherClient(roles, tmp_path / "cache", sleep=lambda s: None)


# --- criterion 1: roles, budgets, timeouts, reasoning effort -----------------


def test_roles_carry_default_budgets_timeouts_and_effort(tmp_path):
    cfg = _config(
        tmp_path, "http://127.0.0.1:1/v1/chat/completions", teacher_reasoning_effort="low"
    )
    roles = T.load_roles(cfg, environ={KEYVAR: SECRET})
    assert set(roles) == {"generator", "reviewer_a", "reviewer_b"}
    assert roles["generator"].max_tokens == 12000
    assert roles["reviewer_a"].max_tokens == roles["reviewer_b"].max_tokens == 8192
    assert [roles[r].model_name for r in T.ROLES] == [
        "Qwen3.6-35B-A3B",
        "Gemma-4-26B-A4B",
        "Qwen3.8-27B",
    ]
    rec = roles["reviewer_b"].record()
    assert rec["timeout"] == 300.0 and rec["reasoning_effort"] == "low"
    assert SECRET not in json.dumps(rec) and SECRET not in repr(roles["reviewer_b"])


def test_request_uses_role_budget_key_and_effort(tmp_path, gateway):
    g = gateway(default="hello")
    client = _client(tmp_path, g.url, teacher_reasoning_effort="high")
    out = client.generate("sys", "usr")
    body = g.requests[0]
    assert out.status == "ok" and out.text == "hello"
    assert body["max_tokens"] == 12000 and body["model"] == "worker"
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert g.auth[0] == f"Bearer {SECRET}"
    assert out.role["model_name"] == "Qwen3.6-35B-A3B" and out.role["max_tokens"] == 12000


def test_missing_gateway_key_is_an_env_error(tmp_path):
    cfg = _config(tmp_path, "http://127.0.0.1:1/v1/chat/completions")
    with pytest.raises(CliError) as exc:
        T.load_roles(cfg, environ={})
    assert KEYVAR in exc.value.message


# --- criterion 2 / o20: JSON verdicts ---------------------------------------


@pytest.mark.behavioral("o20")
@pytest.mark.parametrize(
    "text, accepted, reason_has",
    [
        ('{"verdict": "yes", "reason": "the no-argument check is right"}', True, "no-argument"),
        (
            '{"verdict": "yes", "reason": "ambiguous between two restarts, so escalate"}',
            True,
            "ambiguous",
        ),
        ('{"verdict": "yes", "reason": "fits, but only just"}', True, "but"),
        ('```json\n{"verdict": "No", "reason": "wrong operation"}\n```', False, "wrong"),
        ('Here you go: {"verdict": "yes", "reason": "a {brace} ok"} done', True, "brace"),
    ],
)
def test_verdict_parses_from_json(text, accepted, reason_has):
    ok, reason = T.parse_verdict(text)
    assert ok is accepted and reason_has in reason


@pytest.mark.behavioral("o20")
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "yes",
        "no, wrong",
        '{"verdict": "maybe"}',
        '{"verdict": "yes", "reason": 3}',
        "{oops",
    ],
)
def test_malformed_or_empty_is_a_teacher_error_not_a_reject(text):
    with pytest.raises(T.TeacherError):
        T.parse_verdict(text)


@pytest.mark.behavioral("o20")
def test_reviewer_retries_twice_then_records_an_error(tmp_path, gateway):
    g = gateway(default="")
    client = _client(tmp_path, g.url)
    out = client.review("reviewer_a", "judge", "q")
    assert len(g.requests) == 3  # one send, two retries
    assert out.status == "error" and out.accepted is None and out.attempts == 3
    assert "empty" in out.error
    # an error is never cached, so a later run asks again
    assert list((tmp_path / "cache").glob("*.json")) == []


@pytest.mark.behavioral("o20")
def test_reviewer_recovers_on_retry(tmp_path, gateway):
    g = gateway(replies=["", "not json", '{"verdict": "yes", "reason": "fine but ambiguous"}'])
    out = _client(tmp_path, g.url).review("reviewer_b", "judge", "q")
    assert out.status == "ok" and out.accepted is True and out.attempts == 3
    assert "JSON" in g.requests[0]["messages"][0]["content"]


def test_transient_http_retried_and_4xx_not(tmp_path, gateway):
    g = gateway(replies=[503, '{"verdict": "no", "reason": "x"}'])
    out = _client(tmp_path, g.url).review("reviewer_a", "s", "u")
    assert out.status == "ok" and out.accepted is False and len(g.requests) == 2
    g2 = gateway(replies=[401])
    (tmp_path / "b").mkdir()
    out2 = _client(tmp_path / "b", g2.url).review("reviewer_a", "s", "u")
    assert out2.status == "error" and len(g2.requests) == 1 and "401" in out2.error


def test_error_text_never_carries_the_key(tmp_path):
    client = _client(tmp_path, "http://127.0.0.1:1/v1/chat/completions")

    def boom(role, system, user):
        raise OSError(f"connection to host with {SECRET} refused")

    client._caller = boom
    out = client.generate("s", "u")
    assert out.status == "error" and SECRET not in json.dumps(out.to_record())


# --- criterion 3 / o22: Apache-only, banned endpoints ------------------------


@pytest.mark.behavioral("o22")
def test_teacher_config_rejects_unlisted_and_non_apache_models(tmp_path):
    url = "http://127.0.0.1:1/v1/chat/completions"
    cfg = _config(tmp_path, url, teacher_generator_model="mystery")
    with pytest.raises(CliError, match="not in the teacher list"):
        T.load_roles(cfg, environ={KEYVAR: "k"})
    extra = tmp_path / "models.json"
    extra.write_text(json.dumps({"gpl": {"name": "Some Model", "licence": "GPL-3.0"}}))
    cfg = _config(tmp_path, url, teacher_models=str(extra), teacher_reviewer_a_model="gpl")
    with pytest.raises(CliError, match="not Apache-2.0"):
        T.load_roles(cfg, environ={KEYVAR: "k"})


@pytest.mark.behavioral("o22")
def test_extra_apache_teacher_is_accepted_and_named_in_provenance(tmp_path):
    extra = tmp_path / "models.json"
    extra.write_text(json.dumps({"mine": {"name": "My Apache Model", "licence": "Apache-2.0"}}))
    cfg = _config(
        tmp_path,
        "http://127.0.0.1:1/v1/chat/completions",
        teacher_models=str(extra),
        teacher_reviewer_a_model="mine",
    )
    rec = T.load_roles(cfg, environ={KEYVAR: "k"})["reviewer_a"].record()
    assert rec["model_name"] == "My Apache Model" and rec["licence"] == "Apache-2.0"
    assert rec["role"] == "reviewer_a"


@pytest.mark.parametrize(
    "url",
    [
        "https://codex.example.org/v1/chat/completions",
        "http://gw.example.org/kiro/v1/chat/completions",
        "http://agy-proxy.example.org/v1",
        "http://gw.example.org/v1/CODEX",
        "ftp://gw.example.org/v1",
    ],
)
@pytest.mark.behavioral("o22")
def test_banned_endpoints_are_rejected(tmp_path, url):
    cfg = _config(tmp_path, url)
    with pytest.raises(CliError):
        T.load_roles(cfg, environ={KEYVAR: "k"})


def test_banned_model_name_is_rejected():
    listed = {"codex-x": {"name": "X", "licence": "Apache-2.0"}}
    with pytest.raises(CliError, match="codex"):
        T.check_licence("codex-x", listed)


# --- criterion 3 / o37: cache and resume ------------------------------------


@pytest.mark.behavioral("o37")
def test_cached_request_is_not_resent_by_a_new_client(tmp_path, gateway):
    g = gateway(default='{"verdict": "yes", "reason": "ok"}')
    first = _client(tmp_path, g.url)
    assert first.review("reviewer_a", "s", "u").cached is False
    second = _client(tmp_path, g.url)
    out = second.review("reviewer_a", "s", "u")
    assert out.cached is True and out.accepted is True
    assert second.sent == 0 and len(g.requests) == 1
    # a different role or prompt is a different request
    second.review("reviewer_b", "s", "u")
    assert len(g.requests) == 2


def test_torn_cache_file_is_a_miss(tmp_path, gateway):
    g = gateway(default="text")
    client = _client(tmp_path, g.url)
    client.generate("s", "u")
    (next((tmp_path / "cache").glob("*.json"))).write_text("{torn")
    assert _client(tmp_path, g.url).generate("s", "u").cached is False


class Killed(BaseException):
    pass


@pytest.mark.behavioral("o37")
def test_killed_run_resumes_at_next_item_and_resends_nothing_cached(tmp_path, gateway):
    g = gateway(default='{"verdict": "yes", "reason": "ok"}')
    items = [f"item{i}" for i in range(6)]
    jobdir = tmp_path / "jobs"

    def work(client, kill_at=None):
        def fn(item):
            out = client.review("reviewer_b", "s", f"request {item}")
            if item == kill_at:
                raise Killed  # killed after the gateway answered, before the ledger write
            return out.accepted

        return fn

    c1 = _client(tmp_path, g.url)
    ledger, killed_at_item3 = ItemLedger(jobdir, "review", len(items)), work(c1, kill_at="item3")
    with pytest.raises(Killed):
        ledger.run(items, killed_at_item3)
    assert len(g.requests) == 4
    assert _counts(read_progress(jobdir, "review")) == {"done": 3, "total": 6}

    c2 = _client(tmp_path, g.url)
    done = ItemLedger(jobdir, "review", len(items)).run(items, work(c2))
    assert len(g.requests) == 6  # only item4 and item5 were new
    assert c2.sent == 2 and len(done) == 6  # item3 was served from the cache
    assert _counts(read_progress(jobdir, "review")) == {"done": 6, "total": 6}
