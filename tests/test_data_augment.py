"""augment.py ported onto the toy domain: generate, correct, double review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.data import augment as aug
from jev_factory.factory.detach import ItemLedger
from tests.fixtures.fake_teacher import FakeGateway, make_client, verdict
from tests.fixtures.toy_domain import DOMAIN

ENTRIES = [
    {
        "id": "s1",
        "source_id": "s1",
        "kind": "explicit",
        "text": "Turn on the kitchen lights",
        "expect": {"operation": "lamp_on", "args": {"room": "kitchen"}},
        "class": "operation",
        "source": "toy",
    },
    {
        "id": "s2",
        "source_id": "s2",
        "kind": "explicit",
        "text": "Which lights are on?",
        "expect": {"operation": "lamp_status", "args": {}},
    },
    {
        "id": "s3",
        "source_id": "s3",
        "kind": "explicit",
        "text": "Order me a new bulb",
        "expect": {"escalate": True},
        "class": "decline:outside_table",
    },
]


def seed_file(tmp_path: Path, name: str = "train.json", entries=None, header=None) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "header": header or "Split 'train' of toy (seed=1).",
                "entries": ENTRIES if entries is None else entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paraphrase(user: str) -> str:
    return "rewrite of: " + user.split("Original request: ", 1)[1].split("\n", 1)[0]


def run(tmp_path, gateway=None, **kw):
    gateway = gateway or FakeGateway(generator=paraphrase)
    client = make_client(tmp_path, gateway)
    acc, rej = tmp_path / "acc.jsonl", tmp_path / "rej.jsonl"
    counts = aug.run_augment(
        [seed_file(tmp_path)], DOMAIN, client, acc, rej, per_source=kw.pop("per_source", 1), **kw
    )
    return counts, gateway, acc, rej


# -- seeds ---------------------------------------------------------------------------


def test_load_seeds_carries_corpus_fields_and_needs_no_skills_fields(tmp_path):
    seeds = aug.load_seeds(seed_file(tmp_path), DOMAIN)
    assert [s.source_id for s in seeds] == ["s1", "s2", "s3"]
    assert {s.side for s in seeds} == {"train"}
    assert seeds[0].corpus_fields == {"kind": "explicit", "source": "toy", "class": "operation"}
    assert not hasattr(seeds[0], "skill_names") and not hasattr(seeds[0], "seed_format")


def test_change_check_for_read_only_escalate_and_explain_not_for_mutating(tmp_path):
    lamp_on, status, escalate = aug.load_seeds(seed_file(tmp_path), DOMAIN)
    assert not lamp_on.needs_change_check
    assert status.needs_change_check and escalate.needs_change_check
    assert aug.needs_change_check({"explain": True, "answer": "x"}, DOMAIN)
    assert aug.needs_change_check({"operation": "nope", "args": {}}, DOMAIN)


def test_the_held_out_file_is_refused_by_name_and_by_header(tmp_path):
    held_out = seed_file(tmp_path, "held-out.json")
    with pytest.raises(aug.SeedRefused):
        aug.load_seeds(held_out, DOMAIN)
    renamed = seed_file(tmp_path, "x.json", header="Held-out split of toy.")
    with pytest.raises(aug.SeedRefused):
        aug.load_seeds(renamed, DOMAIN, side="test")


def test_side_is_required_when_not_inferable_and_must_agree_when_inferable(tmp_path):
    odd = seed_file(tmp_path, "x.json", header="toy corpus")
    with pytest.raises(aug.ConfigError, match="cannot infer"):
        aug.load_seeds(odd, DOMAIN)
    assert aug.load_seeds(odd, DOMAIN, side="val")[0].side == "val"
    train = seed_file(tmp_path)
    with pytest.raises(aug.ConfigError, match="conflicts"):
        aug.load_seeds(train, DOMAIN, side="val")
    renamed = seed_file(tmp_path, "y.json", header="Split 'val' of toy (seed=1).")
    assert aug.load_seeds(renamed, DOMAIN)[0].side == "val"


def test_conflicting_source_ids_are_refused_before_any_call(tmp_path):
    clash = [ENTRIES[0], {**ENTRIES[1], "source_id": "s1"}]
    path = seed_file(tmp_path, entries=clash)
    gateway = FakeGateway(generator=paraphrase)
    client = make_client(tmp_path, gateway)
    with pytest.raises(ValueError, match="more than one expect"):
        aug.run_augment(
            [path],
            DOMAIN,
            client,
            tmp_path / "a",
            tmp_path / "r",
            per_source=1,
        )
    assert gateway.calls == []


# -- prompts come from the Domain ------------------------------------------------------


def test_prompts_are_built_from_the_domain(tmp_path):
    lamp_on, status, _ = aug.load_seeds(seed_file(tmp_path), DOMAIN)
    system, user = aug.generator_prompt(lamp_on, DOMAIN, 0)
    assert DOMAIN.persona in system
    assert DOMAIN.phrasing_styles[0] in user
    rev_system, rev_user = aug.reviewer_prompt(status, "which lamps are lit", DOMAIN)
    assert "Turn on the lamps in one named room" in rev_system  # capabilities, from the table
    assert DOMAIN.answer_policy in rev_system
    assert "could reasonably be carried out" in rev_system  # read-only: change check
    assert "could reasonably be carried out" not in aug.reviewer_prompt(lamp_on, "x", DOMAIN)[0]
    assert "show which lamps are on" in rev_user


def test_each_variation_number_asks_for_a_different_phrasing_style(tmp_path):
    seed = aug.load_seeds(seed_file(tmp_path), DOMAIN)[0]
    styles = DOMAIN.phrasing_styles
    users = [aug.generator_prompt(seed, DOMAIN, n)[1] for n in range(len(styles) + 1)]
    for n, style in enumerate(styles):
        assert style in users[n]
    assert users[len(styles)] == users[0]
    bare = replace(DOMAIN, phrasing_styles=())
    assert aug.generator_prompt(seed, bare, 0)[1]  # falls back to the default styles


def test_the_expected_answer_is_described_in_words_not_json(tmp_path):
    text = aug.answer_in_words({"operation": "lamp_on", "args": {"room": "kitchen"}}, DOMAIN)
    assert "propose this change" in text and "room kitchen" in text
    assert "lamp_on" not in text and "{" not in text
    assert "read-only check" in aug.answer_in_words(
        {"operation": "lamp_status", "args": {}}, DOMAIN
    )
    assert "more capable assistant" in aug.answer_in_words({"escalate": True}, DOMAIN)
    assert "along the lines of: hi" in aug.answer_in_words(
        {"explain": True, "answer": "hi"}, DOMAIN
    )


def test_the_generator_never_sees_the_expected_answer(tmp_path):
    seed = aug.load_seeds(seed_file(tmp_path), DOMAIN)[0]
    system, user = aug.generator_prompt(seed, DOMAIN, 0)
    for leaked in ("lamp_on", "propose", "escalate", "approve"):
        assert leaked not in (system + user)


# -- guards ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,leak",
    [
        ("please run lamp_on for me", "lamp_on"),
        ("what does room_status show", "room_status"),
        ("turn the lamp on", ""),
        ("lamp_onward", ""),
    ],
)
def test_a_variation_naming_an_operation_identifier_is_caught(text, leak):
    assert aug.names_internal_operation(text, DOMAIN) == leak


@pytest.mark.parametrize(
    "text,found",
    [
        ("propose this change for me", True),
        ("please run a read-only check", True),
        ("room = kitchen", True),
        ("turn on the kitchen", False),
    ],
)
def test_a_variation_copying_the_answer_template_is_caught(text, found):
    assert bool(aug.copies_answer_template(text)) is found


@pytest.mark.parametrize(
    "text,found",
    [
        ("escalate this to a human", True),
        ("hand this off", True),
        ("escalating temperatures", False),
    ],
)
def test_asks_for_handoff(text, found):
    assert bool(aug.asks_for_handoff(text)) is found


# -- the pipeline ---------------------------------------------------------------------


def test_accepted_variation_inherits_side_answer_source_and_records_teachers(tmp_path):
    counts, gateway, acc, rej = run(tmp_path)
    accepted = read(acc)
    assert counts.accepted == 3 and counts.errors == 0 and not rej.exists()
    first = next(r for r in accepted if r["source_id"] == "s1")
    assert first["id"] == "s1~v1" and first["side"] == "train"
    assert first["expect"] == ENTRIES[0]["expect"]
    assert (first["kind"], first["source"], first["class"]) == ("explicit", "toy", "operation")
    assert first["text"] == "rewrite of: Turn on the kitchen lights"
    assert first["decided_by"] == "reviewer_b"
    assert set(first["verdicts"]) == {"reviewer_a", "reviewer_b"}
    assert first["teachers"]["reviewer_b"]["model"] == "reviewer_b-model"
    assert "seed_format" not in first


def test_correction_is_one_reviewer_b_free_text_call_before_the_verdicts(tmp_path):
    gateway = FakeGateway(generator=paraphrase, corrector=lambda t: t.upper())
    counts, gateway, acc, _ = run(tmp_path, gateway, workers=1)
    assert read(acc)[0]["text"].startswith("REWRITE OF")
    order = [(role, "copyedit" in system) for role, system, _ in gateway.calls[:4]]
    assert order == [
        ("generator", False),
        ("reviewer_b", True),  # the corrector
        ("reviewer_a", False),
        ("reviewer_b", False),
    ]
    assert counts.generated == counts.corrected == 3


def test_reviewer_b_decides_by_default_and_reviewer_a_is_recorded(tmp_path):
    gateway = FakeGateway(generator=paraphrase, reject=lambda role, user: role == "reviewer_a")
    counts, _, acc, rej = run(tmp_path, gateway)
    assert counts.accepted == 3 and counts.rejected_by_a == 3 and not rej.exists()
    assert read(acc)[0]["verdicts"]["reviewer_a"]["accept"] is False


def test_decide_by_both_rejects_when_either_reviewer_says_no(tmp_path):
    gateway = FakeGateway(generator=paraphrase, reject=lambda role, user: role == "reviewer_a")
    counts, _, acc, rej = run(tmp_path, gateway, decide_by="both")
    assert counts.accepted == 0 and not acc.exists()
    assert len(read(rej)) == 3
    only_b = FakeGateway(generator=paraphrase, reject=lambda role, user: role == "reviewer_b")
    (tmp_path / "b").mkdir()
    counts, _, _, _ = run(tmp_path / "b", only_b, decide_by="both")
    assert counts.accepted == 0 and counts.rejected_by_b == 3


def test_a_reject_keeps_the_reason_and_every_verdict(tmp_path):
    gateway = FakeGateway(
        generator=paraphrase,
        reviewer_reply=lambda role, user: verdict(False, "it asks for something else"),
    )
    _, _, _, rej = run(tmp_path, gateway)
    rejected = read(rej)[0]
    assert rejected["verdicts"]["reviewer_b"] == {
        "accept": False,
        "reason": "it asks for something else",
    }


def test_no_argument_ambiguous_and_but_in_a_reason_do_not_reject(tmp_path):
    gateway = FakeGateway(
        generator=paraphrase,
        reviewer_reply=lambda role, user: verdict(
            True, "no-argument case, ambiguous wording but the answer is right"
        ),
    )
    counts, _, acc, _ = run(tmp_path, gateway)
    assert counts.accepted == 3 and len(read(acc)) == 3


def test_deterministic_guards_reject_whatever_the_reviewers_said(tmp_path):
    gateway = FakeGateway(generator=lambda user: "please run lamp_on now")
    counts, _, acc, rej = run(tmp_path, gateway)
    assert counts.accepted == 0 and not acc.exists()
    verdicts = read(rej)[0]["verdicts"]
    assert verdicts["identifier_check"]["accept"] is False
    assert verdicts["reviewer_b"]["accept"] is True


def test_handoff_wording_is_rejected(tmp_path):
    gateway = FakeGateway(generator=lambda user: "escalate this to a human please")
    _, _, _, rej = run(tmp_path, gateway)
    assert "handoff_check" in read(rej)[0]["verdicts"]


def test_an_empty_generator_reply_is_an_error_and_retried_on_resume(tmp_path):
    empty = FakeGateway(generator=lambda user: "")
    counts, _, acc, rej = run(tmp_path, empty)
    assert counts.errors == 3 and not acc.exists() and not rej.exists()
    counts, _, acc, _ = run(tmp_path, FakeGateway(generator=paraphrase))
    assert counts.accepted == 3  # nothing was written, so all three were attempted again


def test_a_malformed_verdict_is_an_error_not_a_reject(tmp_path):
    gateway = FakeGateway(generator=paraphrase, reviewer_reply=lambda role, user: "no way")
    counts, _, acc, rej = run(tmp_path, gateway)
    assert counts.errors == 3 and counts.rejected_by_b == 0
    assert not acc.exists() and not rej.exists()


def test_resume_skips_ids_already_written_and_limit_caps_new_ones(tmp_path):
    counts, _, acc, rej = run(tmp_path, limit=2)
    assert counts.accepted == 2
    gateway = FakeGateway(generator=paraphrase)
    counts, _, acc, _ = run(tmp_path, gateway)
    assert counts.accepted == 1  # only the third seed is left
    assert sorted(r["id"] for r in read(acc)) == ["s1~v1", "s2~v1", "s3~v1"]


def test_resume_reads_the_ledger_too(tmp_path):
    ledger = ItemLedger(tmp_path / "job", "augment", 3)
    ledger.record("s1~v1", "accepted")
    counts, _, acc, _ = run(tmp_path, ledger=ledger)
    assert counts.accepted == 2
    assert set(ledger.done) == {"s1~v1", "s2~v1", "s3~v1"}


def test_killed_run_restarts_without_resending_cached_teacher_requests(tmp_path):
    first = FakeGateway(generator=paraphrase)
    counts, gateway, acc, rej = run(tmp_path, first)
    sent = len(gateway.calls)
    acc.unlink()
    again = FakeGateway(generator=paraphrase)
    client = make_client(tmp_path, again)
    aug.run_augment([seed_file(tmp_path)], DOMAIN, client, acc, rej, per_source=1)
    assert len(again.calls) == 0 < sent  # every request came from the content-hash cache


def test_tasks_are_planned_round_robin_across_seeds(tmp_path):
    seeds = aug.load_seeds(seed_file(tmp_path), DOMAIN)
    tasks = aug.plan_tasks(seeds, 2, None, set())
    assert [v for _, v in tasks] == ["s1~v1", "s2~v1", "s3~v1", "s1~v2", "s2~v2", "s3~v2"]
    assert len(aug.plan_tasks(seeds, 2, 4, {"s1~v1"})) == 4


def test_dry_run_counts_attempts_and_calls_nothing(tmp_path):
    gateway = FakeGateway(generator=paraphrase)
    counts, _, acc, rej = run(tmp_path, gateway, per_source=2, dry_run=True)
    assert counts.generated == 6 and gateway.calls == [] and not acc.exists()


def test_workers_write_whole_records_exactly_once(tmp_path):
    counts, _, acc, _ = run(tmp_path, per_source=4, workers=4)
    ids = [r["id"] for r in read(acc)]
    assert len(ids) == len(set(ids)) == 12 and counts.accepted == 12


def test_an_unknown_decide_by_rule_is_refused(tmp_path):
    with pytest.raises(ValueError, match="decide_by"):
        run(tmp_path, decide_by="either")


def test_accepted_output_is_accepted_by_merge(tmp_path):
    from jev_factory.core.merge_variations import merge

    _, _, acc, _ = run(tmp_path)
    split = json.loads(seed_file(tmp_path).read_text(encoding="utf-8"))
    merged, counts = merge(split, read(acc))
    assert counts["kept"] == 3
    assert {e["source_id"] for e in merged["entries"]} == {"s1", "s2", "s3"}


# -- re-review --------------------------------------------------------------------------


def stored(tmp_path) -> Path:
    path = tmp_path / "stored.jsonl"
    rows = [
        {  # stored as accepted: no verdicts
            "id": "s1~v1",
            "source_id": "s1",
            "side": "train",
            "kind": "explicit",
            "text": "lights on in the kitchen",
            "expect": ENTRIES[0]["expect"],
        },
        {  # stored as rejected by reviewer b
            "id": "s2~v1",
            "source_id": "s2",
            "side": "train",
            "kind": "explicit",
            "text": "which lamps are lit",
            "expect": ENTRIES[1]["expect"],
            "verdicts": {"reviewer_b": {"accept": False, "reason": "old"}},
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_rereview_calls_only_reviewer_b_and_is_a_clean_slate(tmp_path):
    gateway = FakeGateway()
    client = make_client(tmp_path, gateway)
    acc, rej = tmp_path / "a.jsonl", tmp_path / "r.jsonl"
    counts = aug.run_rereview([stored(tmp_path)], DOMAIN, client, acc, rej)
    assert gateway.roles_called() == {"reviewer_b"}
    assert counts.processed == counts.accepted == 2 and counts.errors == 0
    flipped = next(r for r in read(acc) if r["id"] == "s2~v1")
    assert flipped["prior_verdicts"]["reviewer_b"]["accept"] is False
    assert flipped["verdicts"]["reviewer_b"]["accept"] is True
    assert read(acc)[0]["prior_verdicts"] == {"stored_as": "accepted"}
    assert (counts.compared, counts.agreed) == (2, 1)


def test_rereview_guard_failure_stays_rejected_and_limit_and_resume_apply(tmp_path):
    path = stored(tmp_path)
    rows = read(path)
    rows[0]["text"] = "run lamp_on in the kitchen"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    client = make_client(tmp_path, FakeGateway())
    acc, rej = tmp_path / "a.jsonl", tmp_path / "r.jsonl"
    counts = aug.run_rereview([path], DOMAIN, client, acc, rej, limit=1)
    assert (counts.processed, counts.rejected) == (1, 1)
    assert "identifier_check" in read(rej)[0]["verdicts"]
    counts = aug.run_rereview([path], DOMAIN, client, acc, rej)
    assert counts.processed == 1  # s1 was already written; only s2 is left
