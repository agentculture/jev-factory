"""``jev review --gold --serve`` (distillery t24): the gold page over the t8 ledger.

The tests drive the HTTP endpoints directly (no browser)."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.review import gold, server


def _rec(rid, label="a", split="test"):
    return Record(
        id=rid,
        state=f"state of {rid}",
        questions={"q1": Question(type="choice", criteria=(("a", ""), ("b", "")))},
        gold={"q1": Gold(label=label)},
        split=split,
        source="evidence",
        licence="apache-2.0",
        gold_provenance="model-generated",
    )


@pytest.fixture
def records(tmp_path):
    path = tmp_path / "test.jsonl"
    gold.write_drafted(path, [_rec("r1"), _rec("r2"), _rec("t1", split="train")])
    return path


class Client:
    def __init__(self, app):
        self.app = app
        self.httpd = server.make_server(app, 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, method, path, body=None, key=None):
        headers = {"X-Review-Key": key or self.app.token, "Host": f"127.0.0.1:{self.port}"}
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request(method, path, body=json.dumps(body) if body else None, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site(records):
    client = Client(server.GoldApp(records))
    yield client
    client.close()


def _decision(rid, action, **extra):
    return {"record_id": rid, "action": action, **extra}


def test_the_page_lists_test_records_with_proposed_gold_on_loopback_only(site):
    """Criterion 1: lists test records with proposed gold; binds 127.0.0.1 only."""
    status, state = site.call("GET", "/api/state")
    assert status == 200
    assert [i["id"] for i in state["items"]] == ["r1", "r2"]
    assert state["items"][0]["gold"]["q1"]["label"] == "a"
    assert state["counts"]["proposed"] == 2
    assert state["skipped_non_test"] == 1
    assert site.httpd.server_address[0] == "127.0.0.1"
    assert server.HOST == "127.0.0.1"


def test_a_wrong_token_is_refused(site):
    """Criterion 1: the page's API needs the per-run token."""
    status, _ = site.call("GET", "/api/state", key="bad")
    assert status == 403


def test_approve_and_reject_write_the_same_entries_as_the_cli(site, records):
    """Criterion 1: approve and reject append ledger entries via review.gold."""
    assert site.call("POST", "/api/decision", _decision("r1", "approve"))[0] == 200
    assert site.call("POST", "/api/decision", _decision("r2", "reject", note="unclear"))[0] == 200
    ledger = gold.read_ledger(gold.ledger_path(records))
    drafted = {r.id: r for r in gold.read_drafted(records)}
    assert [e["action"] for e in ledger] == ["approve", "reject"]
    assert ledger[0]["record_sha256"] == gold.record_sha256(drafted["r1"])
    assert ledger[1]["note"] == "unclear"
    assert gold.approved_ids(ledger, drafted.values()) == {"r1"}


def test_the_ledger_entries_match_the_cli_path(records, capsys):
    """Criterion 1: the page's entry equals the CLI's (apart from the timestamp)."""
    other = records.with_name("other.jsonl")
    other.write_text(records.read_text(encoding="utf-8"), encoding="utf-8")
    client = Client(server.GoldApp(records))
    try:
        client.call("POST", "/api/decision", _decision("r1", "approve", reviewer="me", note="ok"))
    finally:
        client.close()
    args = ["review", "--gold", str(other), "--decide", "r1=approve", "--reviewer", "me"]
    assert cli_main([*args, "--note", "ok", "--apply"]) == 0
    capsys.readouterr()
    page = gold.read_ledger(gold.ledger_path(records))[0]
    cli = gold.read_ledger(gold.ledger_path(other))[0]
    page.pop("at")
    cli.pop("at")
    assert page == cli


def test_an_edit_rewrites_the_record_and_voids_an_earlier_approval(site, records):
    """Criterion 2: approve then edit voids the approval, as the CLI path does."""
    site.call("POST", "/api/decision", _decision("r1", "approve"))
    status, _ = site.call(
        "POST", "/api/decision", _decision("r1", "edit", question_id="q1", label="b")
    )
    assert status == 200
    ledger = gold.read_ledger(gold.ledger_path(records))
    drafted = gold.read_drafted(records)
    edited = next(r for r in drafted if r.id == "r1")
    assert edited.gold["q1"].label == "b"
    assert [e["action"] for e in ledger] == ["approve", "edit"]
    assert gold.approved_ids(ledger, drafted) == set()
    _, state = site.call("GET", "/api/state")
    assert state["items"][0]["status"] == "proposed"
    site.call("POST", "/api/decision", _decision("r1", "approve"))
    ledger = gold.read_ledger(gold.ledger_path(records))
    assert gold.approved_ids(ledger, gold.read_drafted(records)) == {"r1"}


def test_a_bad_decision_is_refused_and_writes_nothing(site, records):
    """Criterion 1: an edit to a label outside the options is a 400, not a ledger line."""
    body = _decision("r1", "edit", question_id="q1", label="zzz")
    status, data = site.call("POST", "/api/decision", body)
    assert status == 400
    assert "zzz" in data["error"]
    assert not gold.ledger_path(records).exists()


def test_a_stale_page_cannot_decide_on_a_changed_record(site, records):
    """Criterion 1: a decision cites the sha256 the page showed."""
    body = _decision("r1", "approve", record_sha256="0" * 64)
    status, _ = site.call("POST", "/api/decision", body)
    assert status == 400
    assert not gold.ledger_path(records).exists()


def test_a_corrupted_ledger_renders_an_error_and_writes_nothing(records):
    """Criterion 2: a corrupted ledger gives an error on state and on decisions."""
    ledger_file = gold.ledger_path(records)
    ledger_file.write_text("{not json\n", encoding="utf-8")
    before_ledger = ledger_file.read_bytes()
    before_records = records.read_bytes()
    client = Client(server.GoldApp(records))
    try:
        status, data = client.call("GET", "/api/state")
        assert status == 400
        assert "not JSON" in data["error"]
        status, _ = client.call("POST", "/api/decision", _decision("r1", "approve"))
        assert status == 400
        edit = _decision("r1", "edit", question_id="q1", label="b")
        assert client.call("POST", "/api/decision", edit)[0] == 400
    finally:
        client.close()
    assert ledger_file.read_bytes() == before_ledger
    assert records.read_bytes() == before_records


def test_the_page_shows_one_record_at_a_time_with_keyboard_shortcuts(site):
    """Criterion 1: the served page is the gold page (one record, shortcuts, progress)."""
    html = site.app.page.decode()
    assert "__JEV_REVIEW_KEY__" not in html
    assert site.app.token in html
    assert 'ev.key === "a"' in html
    assert "progress" in html


def test_serve_gold_missing_file_is_an_env_error(tmp_path):
    """Criterion 1: --gold --serve no longer refuses; a missing file still does."""
    assert cli_main(["review", "--gold", str(tmp_path / "none.jsonl"), "--serve"]) == 2
