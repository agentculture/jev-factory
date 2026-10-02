"""``jev review`` (deviation d16): the verb tree, the append-only review file, the change
plan, ``--apply``, and the local review site's API and its guards.

Most tests run on the toy domain to show the review is not jev-CLI specific; the tree
test runs on the jev-CLI domain, whose dotted verb names make the deep tree."""

from __future__ import annotations

import dataclasses
import http.client
import io
import json
import re
import shutil
import sys
import threading
from pathlib import Path

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.cli._commands import review as review_cmd
from jev_factory.domains.jev_cli.generate import generate_domain
from jev_factory.review import core, server
from tests.fixtures.toy_domain import DOMAIN as TOY

AT = "2026-10-02T12:00:00Z"


@pytest.fixture
def toy(tmp_path):
    seed = tmp_path / "seed.json"
    shutil.copyfile(TOY.seed_corpus, seed)
    return dataclasses.replace(TOY, seed_corpus=seed)


def _raw(domain) -> dict:
    return json.loads(Path(domain.seed_corpus).read_text(encoding="utf-8"))


def _entry(domain, entry_id) -> dict:
    return next(e for e in _raw(domain)["entries"] if e["id"] == entry_id)


def _decide(domain, decision, review_file=None) -> dict:
    path = review_file or core.review_path(Path(domain.seed_corpus))
    records = core.read_records(path)
    record = core.make_record(
        _raw(domain), domain, decision, proposals=core.proposals(records), at=AT
    )
    core.append_record(path, record)
    return record


def _plan(domain) -> core.ChangePlan:
    records = core.read_records(core.review_path(Path(domain.seed_corpus)))
    return core.plan_changes(_raw(domain), records, domain)


# -- the verb tree ---------------------------------------------------------------------


def test_the_tree_follows_dotted_verb_names_then_the_controls_and_reasons():
    domain = generate_domain()
    nodes = {n["id"]: n for n in core.verb_tree(domain)}
    assert nodes[domain.name]["parent"] is None
    assert nodes["jev.run.dataset-bundle"]["parent"] == "jev.run"
    assert nodes["jev.run"]["kind"] == "operation" and nodes["jev.run"]["read_only"] is True
    assert nodes["jev.run.train"]["read_only"] is False
    assert nodes["jev.cli.overview"]["parent"] == "jev.cli"
    assert nodes["explain"]["parent"] == domain.name
    assert nodes["escalate:repair"]["parent"] == "escalate"
    for name in domain.names():
        assert name in nodes
    order = [n["id"] for n in core.verb_tree(domain)]
    for node in nodes.values():  # parents come first
        if node["parent"] is not None:
            assert order.index(node["parent"]) < order.index(node["id"])


def test_entries_attach_to_their_operation_explain_or_reason(toy):
    raw = _raw(toy)
    by_id = {e["id"]: e for e in raw["entries"]}
    assert core.attach(by_id["toy-01"], toy) == "lamp_status"
    assert core.attach(by_id["toy-11"], toy) == "explain"
    decline = next(e for e in raw["entries"] if e["class"] == "decline:injection")
    assert core.attach(decline, toy) == "escalate:injection"
    assert core.attach({"expect": {"operation": "no_such"}}, toy) == toy.name


# -- decisions and the review file -----------------------------------------------------


def test_a_decision_takes_before_from_the_seed_and_appends_one_line(toy):
    path = core.review_path(Path(toy.seed_corpus))
    assert path.name == "seed.review.jsonl"
    record = _decide(toy, {"action": "approve", "entry_id": "toy-01", "note": " fine "})
    assert record["before"] == _entry(toy, "toy-01") and record["after"] is None
    assert record["note"] == "fine" and record["at"] == AT
    _decide(toy, {"action": "reject", "entry_id": "toy-02"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["action"] == "approve"


@pytest.mark.parametrize(
    "decision, message",
    [
        ({"action": "bless", "entry_id": "toy-01"}, "action must be one of"),
        ({"action": "approve"}, "needs an entry_id"),
        ({"action": "approve", "entry_id": "toy-99"}, "not in the seed"),
        ({"action": "propose", "entry_id": "toy-01", "after": {}}, "already in the seed"),
        ({"action": "edit", "entry_id": "toy-01"}, "needs the entry as 'after'"),
        ({"action": "approve", "entry_id": "toy-01", "note": 3}, "note must be a string"),
    ],
)
def test_a_bad_decision_is_refused(toy, decision, message):
    with pytest.raises(core.ReviewError, match=re.escape(message)):
        core.make_record(_raw(toy), toy, decision, at=AT)


def test_an_edit_must_change_something_and_validate_against_the_domain(toy):
    same = _entry(toy, "toy-01")
    with pytest.raises(core.ReviewError, match="changes nothing"):
        core.make_record(_raw(toy), toy, {"action": "edit", "entry_id": "toy-01", "after": same})
    bad = dict(same, expect={"operation": "no_such_op", "args": {}})
    with pytest.raises(core.ReviewError, match="toy-01") as raised:
        core.make_record(_raw(toy), toy, {"action": "edit", "entry_id": "toy-01", "after": bad})
    assert str(raised.value).count("toy-01") == 1  # named once, not "toy-01: toy-01: ..."
    other_id = dict(same, id="toy-77", text="x")
    with pytest.raises(core.ReviewError, match="is not 'toy-01'"):
        core.make_record(
            _raw(toy), toy, {"action": "edit", "entry_id": "toy-01", "after": other_id}
        )


def test_a_proposal_gets_an_operator_source_and_can_be_revised_or_withdrawn(toy):
    after = {
        "kind": "explicit",
        "text": "is the hall lamp on",
        "expect": {"operation": "lamp_status"},
    }
    record = _decide(toy, {"action": "propose", "entry_id": "toy-90", "after": after})
    assert record["after"]["source"] == "operator-review-2026-10-02"
    assert record["after"]["id"] == "toy-90" and record["before"] is None
    with pytest.raises(core.ReviewError, match="propose it again"):
        _decide(toy, {"action": "approve", "entry_id": "toy-90"})
    revised = dict(after, text="is the hall lamp on right now")
    _decide(toy, {"action": "propose", "entry_id": "toy-90", "after": revised})
    assert _plan(toy).changes[0]["after"]["text"] == "is the hall lamp on right now"
    _decide(toy, {"action": "reject", "entry_id": "toy-90"})
    plan = _plan(toy)
    assert plan.changes == [] and plan.counts.get("withdrawn") == 1
    assert "rejected" not in plan.counts


def test_a_malformed_review_file_is_refused_with_its_line(tmp_path):
    path = tmp_path / "seed.review.jsonl"
    path.write_text('{"action": "approve", "entry_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(core.ReviewError, match=r"seed.review.jsonl:2"):
        core.read_records(path)
    path.write_text('{"action": "delete", "entry_id": "a"}\n', encoding="utf-8")
    with pytest.raises(core.ReviewError, match="not a review record"):
        core.read_records(path)


# -- the change plan -------------------------------------------------------------------


def test_the_plan_removes_edits_and_adds_and_approval_changes_nothing(toy):
    _decide(toy, {"action": "approve", "entry_id": "toy-01"})
    _decide(toy, {"action": "reject", "entry_id": "toy-02"})
    edited = dict(_entry(toy, "toy-03"), text="Which rooms do you know about?")
    _decide(toy, {"action": "edit", "entry_id": "toy-03", "after": edited})
    new = {"kind": "explicit", "text": "turn on the lamp please", "expect": {"escalate": True}}
    _decide(toy, {"action": "propose", "entry_id": "toy-91", "after": new})
    plan = _plan(toy)
    assert plan.ok, (plan.conflicts, plan.problems)
    assert [(c["action"], c["entry_id"]) for c in plan.changes] == [
        ("remove", "toy-02"),
        ("edit", "toy-03"),
        ("add", "toy-91"),
    ]
    ids = [e["id"] for e in plan.seed["entries"]]
    assert "toy-02" not in ids and ids[-1] == "toy-91"
    assert ids.index("toy-03") == 1  # an edit keeps the entry's place
    assert plan.counts == {"approved": 1, "rejected": 1, "edited": 1, "proposed": 1, "pending": 13}


def test_the_latest_decision_wins(toy):
    _decide(toy, {"action": "reject", "entry_id": "toy-02"})
    _decide(toy, {"action": "approve", "entry_id": "toy-02", "note": "on second look"})
    assert _plan(toy).changes == []


def test_applying_is_idempotent(toy):
    _decide(toy, {"action": "reject", "entry_id": "toy-02"})
    edited = dict(_entry(toy, "toy-03"), text="Which rooms do you know about?")
    _decide(toy, {"action": "edit", "entry_id": "toy-03", "after": edited})
    Path(toy.seed_corpus).write_text(core.dump_seed(_plan(toy).seed), encoding="utf-8")
    again = _plan(toy)
    assert again.ok and again.changes == []


def test_a_seed_changed_since_the_decision_is_a_conflict_not_an_overwrite(toy):
    edited = dict(_entry(toy, "toy-03"), text="Which rooms do you know about?")
    _decide(toy, {"action": "edit", "entry_id": "toy-03", "after": edited})
    _decide(toy, {"action": "approve", "entry_id": "toy-04"})
    raw = _raw(toy)
    for entry in raw["entries"]:
        if entry["id"] in ("toy-03", "toy-04"):
            entry["text"] += " (changed by hand)"
    Path(toy.seed_corpus).write_text(core.dump_seed(raw), encoding="utf-8")
    plan = _plan(toy)
    assert not plan.ok and plan.changes == []
    assert any("toy-03: edited, but the seed entry changed" in c for c in plan.conflicts)
    assert any("toy-04: approved, but the seed entry changed" in c for c in plan.conflicts)


def test_next_ids_follow_the_seed_style_and_skip_taken_ids(toy):
    ids = core.next_ids(_raw(toy), toy, taken={"toy-17"})
    assert ids == {"operation": "toy-18", "explain": "toy-18", "escalate": "toy-18"}
    jev = generate_domain()
    raw = json.loads(Path(jev.seed_corpus).read_text(encoding="utf-8"))
    jev_ids = core.next_ids(raw, jev)
    assert re.fullmatch(r"jcs-op-\d{3}", jev_ids["operation"])
    assert jev_ids["explain"].startswith("jcs-ex-") and jev_ids["escalate"].startswith("jcs-dc-")


def test_dump_seed_matches_the_committed_seed_format():
    path = Path(generate_domain().seed_corpus)
    text = path.read_text(encoding="utf-8")
    assert core.dump_seed(json.loads(text)) == text


# -- the CLI ---------------------------------------------------------------------------


@pytest.fixture
def cli_toy(toy, monkeypatch):
    monkeypatch.setattr(review_cmd, "load_domain", lambda ref: toy)
    return toy


def test_the_default_is_a_dry_run_that_writes_nothing(cli_toy, capsys):
    _decide(cli_toy, {"action": "reject", "entry_id": "toy-02", "note": "duplicate"})
    before = Path(cli_toy.seed_corpus).read_bytes()
    assert cli_main(["review", "toy", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["applied"] is False and result["changes"][0]["entry_id"] == "toy-02"
    assert Path(cli_toy.seed_corpus).read_bytes() == before
    assert cli_main(["review", "toy"]) == 0
    text = capsys.readouterr().out
    assert "remove toy-02" in text and "dry run: nothing was written" in text


def test_apply_writes_the_seed_and_a_rerun_changes_nothing(cli_toy, capsys):
    _decide(cli_toy, {"action": "reject", "entry_id": "toy-02"})
    assert cli_main(["review", "toy", "--apply", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is True
    assert "toy-02" not in {e["id"] for e in _raw(cli_toy)["entries"]}
    assert cli_main(["review", "toy", "--apply", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["changes"] == []


def test_apply_refuses_on_a_conflict_and_writes_nothing(cli_toy, capsys):
    _decide(cli_toy, {"action": "reject", "entry_id": "toy-02"})
    raw = _raw(cli_toy)
    raw["entries"][1]["text"] += "!"
    Path(cli_toy.seed_corpus).write_text(core.dump_seed(raw), encoding="utf-8")
    before = Path(cli_toy.seed_corpus).read_bytes()
    assert cli_main(["review", "toy", "--apply"]) == 1
    assert "refusing to write the seed" in capsys.readouterr().err
    assert Path(cli_toy.seed_corpus).read_bytes() == before


def test_serve_and_apply_are_separate_steps(cli_toy, capsys):
    assert cli_main(["review", "toy", "--serve", "--apply"]) == 1
    assert "separate steps" in capsys.readouterr().err


def test_a_domain_without_a_seed_is_a_user_error(toy, monkeypatch, capsys):
    monkeypatch.setattr(
        review_cmd, "load_domain", lambda ref: dataclasses.replace(toy, seed_corpus=None)
    )
    assert cli_main(["review", "toy"]) == 1
    assert "declares no seed corpus" in capsys.readouterr().err


def test_review_has_a_catalog_entry(capsys):
    assert cli_main(["explain", "review"]) == 0
    assert "--apply" in capsys.readouterr().out


# -- the review site -------------------------------------------------------------------


@pytest.fixture
def site(toy):
    app = server.ReviewApp(toy, Path(toy.seed_corpus), core.review_path(Path(toy.seed_corpus)))
    httpd = server.make_server(app, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield app, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _call(port, method, path, body=None, token=None, host=None):
    conn = http.client.HTTPConnection(server.HOST, port, timeout=10)
    headers = {"Host": host or f"{server.HOST}:{port}"}
    if token is not None:
        headers[server.AUTH_HEADER] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    res = conn.getresponse()
    payload = res.read()
    conn.close()
    return res.status, res.getheader("Content-Type", ""), payload, res


def test_the_site_serves_the_page_with_its_token_and_a_strict_csp(site):
    app, port = site
    status, ctype, body, res = _call(port, "GET", "/")
    assert status == 200 and ctype.startswith("text/html")
    assert app.token.encode() in body and b"__JEV_REVIEW_KEY__" not in body
    assert "connect-src 'self'" in res.getheader("Content-Security-Policy")


def test_the_api_needs_the_token_and_the_loopback_host(site):
    app, port = site
    assert _call(port, "GET", "/api/state")[0] == 403
    assert _call(port, "GET", "/api/state", token="wrong")[0] == 403
    assert _call(port, "GET", "/api/state", token=app.token, host="evil.example:80")[0] == 403
    assert _call(port, "GET", "/", host="evil.example:80")[0] == 403
    assert (
        _call(port, "POST", "/api/decision", {"action": "approve", "entry_id": "toy-01"})[0] == 403
    )
    assert not app.review_file.exists()


def test_the_state_carries_the_tree_entries_and_statuses(site):
    app, port = site
    status, _, body, _ = _call(port, "GET", "/api/state", token=app.token)
    state = json.loads(body)
    assert status == 200 and state["domain"] == app.domain.name
    assert {n["id"] for n in state["nodes"]} >= {"lamp_status", "explain", "escalate:injection"}
    assert len(state["entries"]) == 16
    assert {e["status"] for e in state["entries"]} == {"pending"}
    assert state["next_ids"]["operation"] == "toy-17"


def test_a_decision_through_the_api_is_appended_and_shows_in_the_state(site):
    app, port = site
    status, _, body, _ = _call(
        port,
        "POST",
        "/api/decision",
        {"action": "reject", "entry_id": "toy-02", "note": "dup"},
        token=app.token,
    )
    assert status == 200 and json.loads(body) == {"recorded": True, "action": "reject"}
    state = json.loads(_call(port, "GET", "/api/state", token=app.token)[2])
    rejected = [e for e in state["entries"] if e["status"] == "rejected"]
    assert [e["entry"]["id"] for e in rejected] == ["toy-02"]
    assert state["pending_changes"] == 1
    assert len(core.read_records(app.review_file)) == 1


def test_a_bad_decision_or_body_through_the_api_is_a_400_and_records_nothing(site):
    app, port = site
    bad = {"action": "edit", "entry_id": "toy-01", "after": {"text": "x"}}
    status, _, body, _ = _call(port, "POST", "/api/decision", bad, token=app.token)
    assert status == 400 and "missing field" in json.loads(body)["error"]
    assert _call(port, "POST", "/api/decision", [1], token=app.token)[0] == 400
    assert _call(port, "POST", "/api/nowhere", {}, token=app.token)[0] == 404
    assert not app.review_file.exists()


def test_check_reports_problems_without_recording(site):
    app, port = site
    entry = {"id": "toy-50", "kind": "explicit", "text": "x", "expect": {"operation": "nope"}}
    status, _, body, _ = _call(port, "POST", "/api/check", {"entry": entry}, token=app.token)
    assert status == 200 and json.loads(body)["problems"]
    assert not app.review_file.exists()


def test_the_page_pins_every_cdn_asset_with_an_integrity_hash():
    html = server.PAGE.read_text(encoding="utf-8")
    tags = re.findall(r"<(?:script|link)\b[^>]*https://[^>]*>", html)
    assert len(tags) == 4
    for tag in tags:
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/=]+"', tag), tag
        assert re.search(r"@\d+\.\d+\.\d+/", tag), tag  # an exact version, never a range


def test_serve_flushes_the_url_before_it_blocks(cli_toy, monkeypatch):
    class Out(io.StringIO):
        flushed_with = None

        def flush(self):
            Out.flushed_with = self.getvalue()

    class Httpd:
        server_address = (server.HOST, 4321)

        def serve_forever(self):
            assert Out.flushed_with and "http://127.0.0.1:4321/" in Out.flushed_with
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(server, "make_server", lambda app, port: Httpd())
    monkeypatch.setattr(sys, "stdout", Out())
    assert cli_main(["review", "toy", "--serve", "--json"]) == 0
    assert json.loads(sys.stdout.getvalue())["url"] == "http://127.0.0.1:4321/"


def test_the_state_offers_the_sources_of_proposals_too(site):
    app, port = site
    after = {
        "kind": "explicit",
        "text": "is the hall lamp on",
        "expect": {"operation": "lamp_status"},
    }
    body = {"action": "propose", "entry_id": "toy-90", "after": after}
    assert _call(port, "POST", "/api/decision", body, token=app.token)[0] == 200
    state = json.loads(_call(port, "GET", "/api/state", token=app.token)[2])
    assert any(s.startswith(core.OPERATOR_SOURCE_PREFIX) for s in state["sources"])
