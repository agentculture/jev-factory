"""``jev review --gold`` (distillery t8): the human-gold approval ledger.

A drafted test record is human gold only with an approve entry bound to its current
sha256; edits change the hash, rejects and corrupt ledgers never approve."""

from __future__ import annotations

import json

import pytest

from jev_factory.cli import main as cli_main
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.review import gold

AT = "2026-10-15T09:00:00Z"


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


def _entry(record, action, **kw):
    return gold.make_entry(record, action, at=AT, **kw)[0]


@pytest.fixture
def drafted(tmp_path):
    path = tmp_path / "test.jsonl"
    gold.write_drafted(path, [_rec("r1"), _rec("r2"), _rec("r3"), _rec("t1", split="train")])
    return path


def test_ledger_path_sits_beside_the_records(tmp_path):
    """Criterion 1: the ledger is a file beside the records."""
    assert gold.ledger_path(tmp_path / "t.jsonl") == tmp_path / "t.gold-ledger.jsonl"


def test_an_approve_entry_carries_sha_decision_and_iso_time():
    """Criterion 1: record sha256, decision, ISO time."""
    record = _rec("r1")
    entry = _entry(record, "approve")
    assert entry["record_sha256"] == gold.record_sha256(record)
    assert entry["action"] == "approve"
    assert entry["at"] == AT


def test_approved_ids_returns_only_approved_records():
    """Criterion 2: approve counts; reject and no decision do not."""
    r1, r2, r3 = _rec("r1"), _rec("r2"), _rec("r3")
    ledger = [_entry(r1, "approve"), _entry(r2, "reject")]
    assert gold.approved_ids(ledger, [r1, r2, r3]) == {"r1"}


def test_a_later_reject_withdraws_an_approve():
    """Criterion 2: the latest entry for the current sha wins."""
    r1 = _rec("r1")
    ledger = [_entry(r1, "approve"), _entry(r1, "reject")]
    assert gold.approved_ids(ledger, [r1]) == set()


def test_edit_after_approve_voids_the_approval():
    """Criteria 1 and 2: an edited record has a new sha and its old approval no longer matches."""
    r1 = _rec("r1")
    ledger = [_entry(r1, "approve")]
    edit_entry, edited = gold.make_entry(r1, "edit", edit=("q1", "b"), at=AT)
    ledger.append(edit_entry)
    assert gold.record_sha256(edited) != gold.record_sha256(r1)
    assert edit_entry["new_sha256"] == gold.record_sha256(edited)
    assert gold.approved_ids(ledger, [edited]) == set()
    ledger.append(_entry(edited, "approve"))
    assert gold.approved_ids(ledger, [edited]) == {"r1"}


def test_a_changed_record_is_not_approved_by_its_old_entry():
    """Criterion 2: a record edited outside the tool is not approved either."""
    ledger = [_entry(_rec("r1"), "approve")]
    assert gold.approved_ids(ledger, [_rec("r1", label="b")]) == set()


def test_provenance_flip_keeps_the_approval():
    """Criterion 2: the hash excludes gold_provenance, which the approval establishes."""
    r1 = _rec("r1")
    human = Record.from_json({**r1.to_json(), "gold_provenance": "human"})
    assert gold.approved_ids([_entry(r1, "approve")], [human]) == {"r1"}


def test_a_corrupted_ledger_file_fails_closed(tmp_path):
    """Criterion 2: a bad line raises; nothing is silently skipped."""
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps(_entry(_rec("r1"), "approve")) + "\n{not json\n")
    with pytest.raises(gold.GoldLedgerError):
        gold.read_ledger(path)


def test_a_malformed_entry_fails_closed(tmp_path):
    """Criterion 2: a well-formed JSON line that is not an entry raises."""
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps({"action": "approve", "record_id": "r1"}) + "\n")
    with pytest.raises(gold.GoldLedgerError):
        gold.read_ledger(path)


def test_approved_ids_rejects_a_hand_built_malformed_ledger():
    """Criterion 2: approved_ids validates the entries it is handed."""
    r1 = _rec("r1")
    bad = [{"action": "approve", "record_id": "r1", "record_sha256": "short", "at": AT}]
    with pytest.raises(gold.GoldLedgerError):
        gold.approved_ids(bad, [r1])


def test_a_missing_ledger_is_empty_and_approves_nothing(tmp_path):
    """Criterion 2: no ledger means no approvals."""
    assert gold.read_ledger(tmp_path / "none.jsonl") == []
    assert gold.approved_ids([], [_rec("r1")]) == set()


def test_cli_dry_run_lists_records_and_writes_nothing(drafted, capsys):
    """Criterion 3: dry-run by default; test records listed with proposed gold."""
    assert cli_main(["review", "--gold", str(drafted), "--decide", "r1=approve"]) == 0
    out = capsys.readouterr().out
    assert "r1 [approved] q1=a" in out  # status after the planned decision
    assert "r2 [proposed] q1=a" in out
    assert "t1" not in out
    assert "dry run" in out
    assert not gold.ledger_path(drafted).exists()


def test_cli_apply_appends_to_the_ledger(drafted, capsys):
    """Criterion 3: --apply writes the append-only ledger; approved_ids sees it."""
    args = ["review", "--gold", str(drafted), "--decide", "r1=approve", "--decide", "r2=reject"]
    assert cli_main([*args, "--apply", "--reviewer", "ori"]) == 0
    capsys.readouterr()
    ledger = gold.read_ledger(gold.ledger_path(drafted))
    assert [e["action"] for e in ledger] == ["approve", "reject"]
    assert ledger[0]["reviewer"] == "ori"
    assert gold.approved_ids(ledger, gold.read_drafted(drafted)) == {"r1"}
    assert cli_main(["review", "--gold", str(drafted), "--decide", "r3=approve", "--apply"]) == 0
    assert len(gold.read_ledger(gold.ledger_path(drafted))) == 3


def test_cli_edit_apply_rewrites_the_record_and_voids_the_approval(drafted, capsys):
    """Criterion 1 via the CLI: edit changes the record, so the old approval stops matching."""
    assert cli_main(["review", "--gold", str(drafted), "--decide", "r1=approve", "--apply"]) == 0
    assert cli_main(["review", "--gold", str(drafted), "--edit", "r1:q1=b", "--apply"]) == 0
    capsys.readouterr()
    records = gold.read_drafted(drafted)
    assert {r.id: r.gold["q1"].label for r in records}["r1"] == "b"
    ledger = gold.read_ledger(gold.ledger_path(drafted))
    assert gold.approved_ids(ledger, records) == set()


def test_cli_refuses_a_corrupted_ledger_with_a_user_error(drafted, capsys):
    """Criterion 2 via the CLI: a corrupted ledger is exit 1."""
    gold.ledger_path(drafted).write_text("garbage\n")
    assert cli_main(["review", "--gold", str(drafted)]) == 1


def test_cli_unknown_record_and_missing_file(drafted, tmp_path):
    """Criterion 3: bad id is a user error, missing file an environment error."""
    assert cli_main(["review", "--gold", str(drafted), "--decide", "zz=approve"]) == 1
    assert cli_main(["review", "--gold", str(tmp_path / "nope.jsonl")]) == 2


def test_cli_edit_to_an_unknown_label_is_refused(drafted):
    """Criterion 1: an edit must stay inside the question's options."""
    assert cli_main(["review", "--gold", str(drafted), "--edit", "r1:q1=zzz", "--apply"]) == 1


def test_explain_review_gold_documents_the_flag(capsys):
    """Criterion 3: jev explain review documents --gold."""
    assert cli_main(["explain", "review", "gold"]) == 0
    out = capsys.readouterr().out
    for word in ("--gold", "--apply", "dry run", "approved_ids"):
        assert word in out
    assert cli_main(["explain", "review"]) == 0
    assert "--gold" in capsys.readouterr().out
