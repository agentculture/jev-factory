"""``jev review --corrections`` (distillery t14): operator review of corrections.

A proposed correction is never trainable. Approved and edited ones are, each citing the
reviewer decision. The corrections review file is the record; ``--apply`` writes to it,
the dry run and ``--serve`` staging do not."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.distill.corrections import Correction, read_corrections, write_corrections
from jev_factory.review import corrections as rc
from jev_factory.review import server

AT = "2026-10-14T09:00:00Z"


def _c(rid, qid="q1", label="a", status="proposed", dist=None):
    return Correction(
        record_id=rid,
        question_id=qid,
        author="teacher-x",
        date="2026-10-13",
        reason="weak",
        status=status,
        label=label,
        distribution=dist,
        text=f"state of {rid}",
    )


@pytest.fixture
def corr(tmp_path):
    path = tmp_path / "corrections.jsonl"
    write_corrections(path, [_c("r1"), _c("r2"), _c("r3"), _c("r4")])
    return path


def _decide(path, rid, action, **extra):
    corrections = read_corrections(path)
    decision = {"action": action, "record_id": rid, "question_id": "q1", **extra}
    record = rc.make_record(corrections, decision, at=AT)
    rc.append_record(rc.review_path(path), record)
    return record


def test_review_path_sits_beside_the_corrections_file(tmp_path):
    assert rc.review_path(tmp_path / "c.jsonl") == tmp_path / "c.review.jsonl"


def test_a_proposed_correction_is_never_exported_as_trainable(corr):
    assert rc.trainable(read_corrections(corr), rc.read_records(rc.review_path(corr))) == []
    # not even when the correction file itself claims it is approved
    write_corrections(corr.with_name("self.jsonl"), [_c("r9", status="approved")])
    assert rc.trainable(read_corrections(corr.with_name("self.jsonl")), []) == []


def test_rejected_is_not_trainable_and_latest_decision_wins(corr):
    _decide(corr, "r1", "approve")
    _decide(corr, "r1", "reject", note="wrong after all")
    out = rc.trainable(read_corrections(corr), rc.read_records(rc.review_path(corr)))
    assert out == []


def test_approved_and_edited_are_trainable_citing_the_reviewer_decision(corr):
    _decide(corr, "r1", "approve", note="agree", reviewer="ori")
    _decide(corr, "r2", "edit", after={"label": "b"}, note="b is right")
    _decide(corr, "r3", "reject")
    records = rc.read_records(rc.review_path(corr))
    out = {t.record_id: t for t in rc.trainable(read_corrections(corr), records)}
    assert set(out) == {"r1", "r2"}  # r3 rejected, r4 still proposed
    assert out["r1"].label == "a" and out["r1"].decision["action"] == "approve"
    assert out["r1"].decision["reviewer"] == "ori" and out["r1"].decision["at"] == AT
    assert out["r2"].label == "b"  # the operator's edit, not the proposal
    assert out["r2"].decision["action"] == "edit" and out["r2"].decision["note"] == "b is right"
    row = out["r2"].to_json()
    assert row["status"] == "edited" and row["reviewer_decision"]["action"] == "edit"
    assert out["r1"].to_json()["status"] == "approved"


def test_an_edited_distribution_replaces_the_proposed_target(corr):
    _decide(corr, "r1", "edit", after={"label": "b", "distribution": {"a": 0.2, "b": 0.8}})
    (t,) = rc.trainable(read_corrections(corr), rc.read_records(rc.review_path(corr)))
    assert t.distribution == {"a": 0.2, "b": 0.8}


def test_decisions_are_validated(corr):
    corrections = read_corrections(corr)
    with pytest.raises(rc.CorrectionReviewError):
        rc.make_record(corrections, {"action": "approve", "record_id": "nope", "question_id": "q1"})
    with pytest.raises(rc.CorrectionReviewError):
        rc.make_record(corrections, {"action": "maybe", "record_id": "r1", "question_id": "q1"})
    with pytest.raises(rc.CorrectionReviewError):  # an edit must change something
        rc.make_record(
            corrections,
            {"action": "edit", "record_id": "r1", "question_id": "q1", "after": {"label": "a"}},
        )
    with pytest.raises(rc.CorrectionReviewError):  # label outside the edited distribution
        rc.make_record(
            corrections,
            {
                "action": "edit",
                "record_id": "r1",
                "question_id": "q1",
                "after": {"label": "z", "distribution": {"a": 1.0}},
            },
        )


def test_a_decision_goes_stale_when_the_correction_changes(corr):
    _decide(corr, "r1", "approve")
    write_corrections(corr, [_c("r1", label="c")])  # appended: now the last r1 differs
    corrections = read_corrections(corr)
    records = rc.read_records(rc.review_path(corr))
    assert [t for t in rc.trainable(corrections, records) if t.record_id == "r1"] == []
    plan = rc.plan(corrections, records)
    assert any("r1" in c for c in plan.conflicts)


def test_the_review_file_is_append_only_and_cites_its_corrections(corr):
    first = _decide(corr, "r1", "approve")
    _decide(corr, "r1", "reject")
    lines = rc.review_path(corr).read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == first
    assert first["correction_sha256"] and first["before"]["label"] == "a"


# -- the CLI -----------------------------------------------------------------------------


def _stage(path, rid, action, **extra):
    app = server.CorrectionsApp(path)
    app.decide({"action": action, "record_id": rid, "question_id": "q1", **extra})
    return app


def test_the_dry_run_writes_nothing(corr, capsys):
    _stage(corr, "r1", "approve")
    assert cli_main(["review", "--corrections", str(corr), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["staged"] == 1 and out["applied"] is False
    assert not rc.review_path(corr).exists()


def test_apply_writes_the_decisions_to_the_review_file(corr, capsys):
    _stage(corr, "r1", "approve")
    _stage(corr, "r2", "edit", after={"label": "b"})
    _stage(corr, "r3", "reject")
    assert cli_main(["review", "--corrections", str(corr), "--apply", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is True and out["recorded"] == 3
    records = rc.read_records(rc.review_path(corr))
    assert [r["action"] for r in records] == ["approve", "edit", "reject"]
    assert {t.record_id for t in rc.trainable(read_corrections(corr), records)} == {"r1", "r2"}
    # applying again records nothing more
    assert cli_main(["review", "--corrections", str(corr), "--apply", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] == 0
    assert len(rc.read_records(rc.review_path(corr))) == 3


def test_apply_refuses_a_stale_staged_decision(corr, capsys):
    _stage(corr, "r1", "approve")
    corrections = read_corrections(corr)
    write_corrections(corr, [_c("r1", label="c")])
    assert len(corrections) == 4
    assert cli_main(["review", "--corrections", str(corr), "--apply"]) == 1
    assert not rc.review_path(corr).exists()


def test_corrections_mode_needs_no_domain_and_conflicts_with_serve_plus_apply(corr):
    assert cli_main(["review", "--corrections", str(corr), "--serve", "--apply"]) == 1
    assert cli_main(["review"]) == 1


# -- the site ----------------------------------------------------------------------------


def test_the_site_serves_corrections_state_and_stages_decisions(corr):
    app = server.CorrectionsApp(corr)
    httpd = server.make_server(app, 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        headers = {"X-Review-Key": app.token, "Host": f"127.0.0.1:{port}"}
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/api/state", headers=headers)
        state = json.loads(conn.getresponse().read())
        assert state["mode"] == "corrections"
        assert [i["status"] for i in state["items"]] == ["proposed"] * 4
        body = json.dumps({"action": "approve", "record_id": "r1", "question_id": "q1"})
        conn.request("POST", "/api/decision", body=body, headers=headers)
        assert conn.getresponse().status == 200
        conn.request("GET", "/api/state", headers=headers)
        state = json.loads(conn.getresponse().read())
        assert state["items"][0]["status"] == "approved"
        assert state["staged"] == 1
        conn.request("GET", "/api/state", headers={**headers, "X-Review-Key": "bad"})
        assert conn.getresponse().status == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert not rc.review_path(corr).exists()  # staging is not the record


def test_the_page_has_a_corrections_view():
    page = server.page("k").decode()
    assert "corrections" in page and 'mode === "corrections"' in page
