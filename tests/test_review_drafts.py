"""``jev review --drafts`` (t25): the draft-review ledger, library and local site.

The tests drive the HTTP endpoints directly (no browser)."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.review import drafts, gold, server


def _rec(rid, label="a", split="train", source="src-a"):
    return Record(
        id=rid,
        state=f"state of {rid}\n$ tegrastats\nRAM 1/2MB",
        questions={"q1": Question(type="choice", criteria=(("a", "first"), ("b", "second")))},
        gold={"q1": Gold(label=label)},
        split=split,
        source=source,
        group=f"docs/{rid}.md",
        licence="apache-2.0",
        gold_provenance="model-generated",
    )


@pytest.fixture
def paths(tmp_path):
    records = tmp_path / "pilot.jsonl"
    gold.write_drafted(
        records,
        [_rec("r1"), _rec("r2", split="test"), _rec("r3", split="dev", source="src-b")],
    )
    evidence = tmp_path / "pilot.evidence.jsonl"
    evidence.write_text(
        "\n".join(
            json.dumps({"id": i, "group": f"docs/{i}.md", "quote": f"quote {i}", "why": f"why {i}"})
            for i in ("r1", "r2")
        )
        + "\n",
        encoding="utf-8",
    )
    return records, evidence


class Client:
    def __init__(self, app):
        self.app = app
        self.httpd = server.make_server(app, 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, method, path, body=None, key=None, host=None):
        headers = {
            "X-Review-Key": key or self.app.token,
            "Host": host or f"127.0.0.1:{self.port}",
        }
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request(method, path, body=json.dumps(body) if body else None, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site(paths):
    client = Client(server.DraftsApp(*paths))
    yield client
    client.close()


def _decision(rid, action, **extra):
    return {"record_id": rid, "action": action, **extra}


def test_the_site_shows_every_split_with_evidence_on_loopback_only(site):
    """Criterion 1: all splits, state, questions, gold, evidence quote/why, group, split."""
    status, state = site.call("GET", "/api/state")
    assert status == 200
    assert [i["split"] for i in state["items"]] == ["train", "test", "dev"]
    first = state["items"][0]
    assert first["evidence"] == {"group": "docs/r1.md", "quote": "quote r1", "why": "why r1"}
    assert first["group"] == "docs/r1.md"
    assert first["gold"]["q1"]["label"] == "a"
    assert first["state"].startswith("state of r1")
    assert state["items"][2]["evidence"] is None
    assert state["splits"] == ["dev", "test", "train"]
    assert state["sources"] == ["src-a", "src-b"]
    assert state["counts"]["unreviewed"] == 3
    assert site.httpd.server_address[0] == "127.0.0.1"


def test_the_page_has_filters_shortcuts_and_progress(site):
    """Criterion 1: filters by split, source, status; keyboard shortcuts; progress."""
    html = site.app.page.decode()
    assert "__JEV_REVIEW_KEY__" not in html
    assert site.app.token in html
    for ident in ("f-split", "f-source", "f-status", "progress"):
        assert ident in html
    for key in ('"k"', '"d"', '"e"', '"n"', '"j"', '"u"', "ArrowRight", "ArrowLeft"):
        assert key in html


def test_a_wrong_token_or_host_is_refused(site):
    """Criterion 1: token and Host checks as in GoldApp."""
    assert site.call("GET", "/api/state", key="bad")[0] == 403
    assert site.call("GET", "/api/state", host="evil.example:80")[0] == 403
    assert site.call("GET", "/", host="evil.example:80")[0] == 403


def test_keep_drop_note_append_to_the_ledger(site, paths):
    """Criterion 2: keep, drop and note decisions are append-only ledger lines."""
    records, _ = paths
    before = records.read_bytes()
    assert site.call("POST", "/api/decision", _decision("r1", "keep"))[0] == 200
    assert site.call("POST", "/api/decision", _decision("r2", "drop", note="bad"))[0] == 200
    assert site.call("POST", "/api/decision", _decision("r1", "note", note="check units"))[0] == 200
    ledger = drafts.read_ledger(drafts.ledger_path(records))
    drafted = {r.id: r for r in gold.read_drafted(records)}
    assert [e["action"] for e in ledger] == ["keep", "drop", "note"]
    assert ledger[0]["record_sha256"] == gold.record_sha256(drafted["r1"])
    assert ledger[1]["note"] == "bad"
    assert records.read_bytes() == before
    _, state = site.call("GET", "/api/state")
    by_id = {i["id"]: i for i in state["items"]}
    assert by_id["r1"]["status"] == "kept"
    assert by_id["r1"]["notes"][0]["note"] == "check units"
    assert by_id["r2"]["status"] == "dropped"
    assert state["counts"] == {"unreviewed": 1, "kept": 1, "edited": 0, "dropped": 1}


def test_an_edit_is_a_ledger_entry_and_shows_the_effective_gold(site, paths):
    """Criterion 2: edit-gold is appended; the records file is not rewritten."""
    records, _ = paths
    before = records.read_bytes()
    body = _decision("r1", "edit", question_id="q1", label="b")
    assert site.call("POST", "/api/decision", body)[0] == 200
    assert records.read_bytes() == before
    _, state = site.call("GET", "/api/state")
    item = state["items"][0]
    assert item["status"] == "edited"
    assert item["gold"]["q1"]["label"] == "b"
    assert item["original_gold"]["q1"]["label"] == "a"


def test_a_bad_or_stale_decision_is_refused_and_writes_nothing(site, paths):
    """Criterion 2: bad label, unknown id, empty note, stale sha: 400 and no ledger."""
    records, _ = paths
    bodies = [
        _decision("r1", "edit", question_id="q1", label="zzz"),
        _decision("nope", "keep"),
        _decision("r1", "note", note="  "),
        _decision("r1", "keep", record_sha256="0" * 64),
        _decision("r1", "dance"),
    ]
    for body in bodies:
        assert site.call("POST", "/api/decision", body)[0] == 400
    assert not drafts.ledger_path(records).exists()


def test_a_corrupted_ledger_renders_an_error_and_writes_nothing(paths):
    """Criterion 3: a corrupted ledger gives an error and no writes."""
    records, evidence = paths
    ledger_file = drafts.ledger_path(records)
    ledger_file.write_text("{not json\n", encoding="utf-8")
    before_ledger = ledger_file.read_bytes()
    before_records = records.read_bytes()
    client = Client(server.DraftsApp(records, evidence))
    try:
        status, data = client.call("GET", "/api/state")
        assert status == 400
        assert "not JSON" in data["error"]
        for body in (
            _decision("r1", "keep"),
            _decision("r1", "edit", question_id="q1", label="b"),
            _decision("r1", "note", note="x"),
        ):
            assert client.call("POST", "/api/decision", body)[0] == 400
    finally:
        client.close()
    assert ledger_file.read_bytes() == before_ledger
    assert records.read_bytes() == before_records
    assert "Cannot read" in client.app.page.decode()


def test_apply_ledger_drops_edits_and_keeps_notes_out(paths):
    """Criterion 2: dropped removed, edits applied, notes never in records."""
    records, _ = paths
    recs = gold.read_drafted(records)
    entries = [
        drafts.make_entry(recs[0], "edit", edit=("q1", "b")),
        drafts.make_entry(recs[0], "note", note="why b"),
        drafts.make_entry(recs[1], "drop"),
        drafts.make_entry(recs[2], "keep"),
    ]
    reviewed = drafts.apply_ledger(recs, entries)
    assert [r.id for r in reviewed] == ["r1", "r3"]
    assert reviewed[0].gold["q1"].label == "b"
    assert "why b" not in json.dumps([r.to_json() for r in reviewed])
    only = drafts.apply_ledger(recs, entries, include_unreviewed=False)
    assert [r.id for r in only] == ["r1", "r3"]
    assert drafts.apply_ledger(recs, [], include_unreviewed=False) == []


def test_a_later_decision_wins_and_a_changed_record_voids_entries(paths):
    """Criterion 2: latest non-note entry decides; entries bind the record's sha256."""
    records, _ = paths
    recs = gold.read_drafted(records)
    ledger = [drafts.make_entry(recs[0], "drop"), drafts.make_entry(recs[0], "keep")]
    assert drafts.resolve(ledger, recs)["r1"].status == "kept"
    changed = gold.read_drafted(records)
    changed[0] = Record(
        **{**{n: getattr(changed[0], n) for n in changed[0].__dataclass_fields__}, "state": "new"}
    )
    assert drafts.resolve(ledger, changed)["r1"].status == "unreviewed"
    assert drafts.orphans(ledger, changed) == ["r1"]


def test_an_edited_test_record_still_needs_the_gold_ledger_approval(paths):
    """Criterion 2: edited test record: earlier gold approval is void; re-approve via t8."""
    records, _ = paths
    recs = gold.read_drafted(records)
    test = next(r for r in recs if r.split == "test")
    approval, _ = gold.make_entry(test, "approve")
    assert gold.approved_ids([approval], [test]) == {test.id}
    edit = drafts.make_entry(test, "edit", edit=("q1", "b"))
    reviewed = drafts.apply_ledger(recs, [edit])
    edited = next(r for r in reviewed if r.id == test.id)
    assert gold.approved_ids([approval], [edited]) == set()
    fresh, _ = gold.make_entry(edited, "approve")
    assert gold.approved_ids([approval, fresh], [edited]) == {test.id}


def test_ledger_validation_fails_closed(tmp_path, paths):
    """Criterion 3: malformed entries raise; the ledger is never guessed."""
    records, _ = paths
    recs = gold.read_drafted(records)
    good = drafts.make_entry(recs[0], "keep")
    for bad in (
        {**good, "action": "approve"},
        {**good, "record_sha256": "x"},
        {**good, "action": "edit"},
        {**good, "action": "note", "note": ""},
        "[]",
    ):
        path = tmp_path / "l.jsonl"
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(drafts.DraftReviewError):
            drafts.read_ledger(path)
    assert drafts.read_ledger(tmp_path / "missing.jsonl") == []


def test_evidence_errors_are_reported(tmp_path):
    """Criterion 3: bad evidence lines are an error, not silently skipped."""
    bad = tmp_path / "e.jsonl"
    bad.write_text('{"id": 3}\n', encoding="utf-8")
    with pytest.raises(drafts.DraftReviewError):
        drafts.read_evidence(bad)
    assert drafts.read_evidence(None) == {}


def test_cli_dry_run_writes_nothing_and_apply_writes(paths, capsys, tmp_path):
    """Criterion 2: the CLI path is dry-run by default; --apply writes ledger and reviewed file."""
    records, evidence = paths
    out = tmp_path / "reviewed.jsonl"
    args = ["review", "--drafts", str(records), "--evidence", str(evidence), "--json"]
    decisions = ["--decide", "r2=drop", "--edit", "r1:q1=b", "--reviewed-out", str(out)]
    assert cli_main([*args, *decisions]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["decisions"] == 2
    assert dry["applied"] is False
    assert not drafts.ledger_path(records).exists()
    assert not out.exists()
    assert cli_main([*args, *decisions, "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    ledger = drafts.read_ledger(drafts.ledger_path(records))
    assert sorted(e["action"] for e in ledger) == ["drop", "edit"]
    reviewed = gold.read_drafted(out)
    assert [r.id for r in reviewed] == ["r1", "r3"]
    assert reviewed[0].gold["q1"].label == "b"


def test_cli_errors(paths, tmp_path):
    """Criterion 3: missing file is exit 2; unknown id and bad ledger are exit 1."""
    records, _ = paths
    assert cli_main(["review", "--drafts", str(tmp_path / "none.jsonl")]) == 2
    assert cli_main(["review", "--drafts", str(records), "--decide", "zz=keep"]) == 1
    assert cli_main(["review", "--drafts", str(records), "--decide", "r1=note"]) == 1
    drafts.ledger_path(records).write_text("garbage\n", encoding="utf-8")
    assert cli_main(["review", "--drafts", str(records)]) == 1


def test_explain_review_drafts_exists(capsys):
    """Criterion 3: the catalog documents the verb mode."""
    assert cli_main(["explain", "review", "drafts"]) == 0
    assert "--drafts" in capsys.readouterr().out


def test_soft_gold_added_options_and_restore_through_the_site(site, paths):
    """d7: probabilities, an added escalation option and a restore are ledger entries."""
    records, _ = paths
    before = records.read_bytes()
    add = _decision(
        "r1", "add_option", question_id="q1", name="escalate", description="hand it to the operator"
    )
    assert site.call("POST", "/api/decision", add)[0] == 200
    soft = _decision(
        "r1", "edit", question_id="q1", label="escalate",
        probabilities={"a": 0.2, "b": 0.0, "escalate": 0.8},
    )  # fmt: skip
    assert site.call("POST", "/api/decision", soft)[0] == 200
    _, state = site.call("GET", "/api/state")
    item = state["items"][0]
    assert item["status"] == "edited"
    assert [c[0] for c in item["questions"]["q1"]["criteria"]] == ["a", "b", "escalate"]
    assert item["gold"]["q1"]["probabilities"] == {"a": 0.2, "b": 0.0, "escalate": 0.8}
    assert item["added"] == {"q1": [["escalate", "hand it to the operator"]]}
    assert (
        site.call("POST", "/api/decision", _decision("r1", "restore", question_id="q1"))[0] == 200
    )
    _, state = site.call("GET", "/api/state")
    item = state["items"][0]
    assert item["status"] == "kept"  # the edit implied keep; the restore undid the change only
    assert item["gold"]["q1"] == item["original_gold"]["q1"]
    assert [c[0] for c in item["questions"]["q1"]["criteria"]] == ["a", "b"]
    assert records.read_bytes() == before


@pytest.mark.parametrize(
    "extra",
    [
        {"label": "a", "probabilities": {"a": 0.4, "b": 0.6}},  # label is not the argmax
        {"label": "a", "probabilities": {"a": 0.5, "b": 0.2}},  # does not sum to 1
        {"label": "a", "probabilities": {"a": 1.0, "zzz": 0.0}},  # unknown option
        {"label": "a", "probabilities": {"a": "high"}},  # not a number
    ],
)
def test_a_bad_soft_gold_is_refused(site, paths, extra):
    records, _ = paths
    body = _decision("r1", "edit", question_id="q1", **extra)
    assert site.call("POST", "/api/decision", body)[0] == 400
    assert not drafts.ledger_path(records).exists()


@pytest.mark.parametrize(
    "name, description",
    [("a", "dup"), ("Escalate", "caps"), ("escalate", ""), ("two words", "x")],
)
def test_a_bad_added_option_is_refused(site, paths, name, description):
    records, _ = paths
    body = _decision("r1", "add_option", question_id="q1", name=name, description=description)
    assert site.call("POST", "/api/decision", body)[0] == 400
    assert not drafts.ledger_path(records).exists()


def test_status_rules_after_edits_keeps_drops_and_restores(paths):
    """Edited while it differs; keep after an edit stays edited; drop wins; restore all."""
    records, _ = paths
    recs = gold.read_drafted(records)
    r1 = recs[0]

    def status(ledger):
        return drafts.resolve(ledger, recs)["r1"].status

    edit = drafts.make_entry(r1, "edit", edit=("q1", "b"))
    keep = drafts.make_entry(r1, "keep")
    drop = drafts.make_entry(r1, "drop")
    back = drafts.make_entry(r1, "edit", edit=("q1", "a"))
    everything = drafts.make_entry(r1, "restore", restore="")
    assert status([edit, keep]) == "edited"
    assert status([edit, drop]) == "dropped"
    assert status([drop, edit]) == "edited"  # an edit implies keep
    assert status([edit, back]) == "kept"  # editing back to the draft is no change
    assert status([edit, everything]) == "kept"
    assert status([keep, everything]) == "kept"
    assert status([everything]) == "unreviewed"
    reviewed = drafts.apply_ledger(recs, [edit, everything])
    assert reviewed[0] == r1


def test_soft_gold_reaches_the_reviewed_records(paths):
    records, _ = paths
    recs = gold.read_drafted(records)
    r1 = recs[0]
    option = drafts.make_entry(r1, "add_option", option=("q1", "escalate", "hand it over"))
    grown = drafts.effective(r1, drafts.resolve([option], recs)["r1"])
    soft = drafts.make_entry(
        r1, "edit", edit=("q1", "escalate", {"escalate": 0.7, "a": 0.3}), current=grown
    )
    with pytest.raises(drafts.DraftReviewError, match="not in"):
        drafts.make_entry(r1, "edit", edit=("q1", "escalate", {"escalate": 1.0}))
    out = drafts.apply_ledger(recs, [option, soft])[0]
    assert out.questions["q1"].names == ("a", "b", "escalate")
    assert out.gold["q1"] == Gold(
        label="escalate", probabilities={"a": 0.3, "b": 0.0, "escalate": 0.7}
    )
    path = records.with_name("l.jsonl")  # the entries round-trip through the file format
    for entry in (option, soft):
        drafts.append_entry(path, entry)
    assert drafts.apply_ledger(recs, drafts.read_ledger(path))[0] == out
