"""targeted.py ported onto the toy domain: one recipe per missing shape."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.core.merge_variations import add_supplement
from jev_factory.data import targeted as T
from jev_factory.domain.model import ArgSpec, Operation, Reason
from tests.fixtures.fake_teacher import FakeGateway, make_client
from tests.fixtures.toy_domain import DOMAIN as TOY

DIAGNOSIS = Reason(
    name="diagnosis",
    description="Hand this off: it needs investigation of this home.",
    definition="the user asks why something is happening in their home or asks to fix it",
)
DOMAIN = replace(TOY, reasons=TOY.reasons + (DIAGNOSIS,))
SEED = json.loads(Path(TOY.seed_corpus).read_text(encoding="utf-8"))["entries"]
TRAIN = [
    {**e, "source_id": e["id"]}
    for e in SEED
    if e["id"] in {"toy-01", "toy-05", "toy-07", "toy-08", "toy-09", "toy-10", "toy-03", "toy-11"}
]


def train_file(tmp_path: Path, entries=None, name="train.json", header=None) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "header": header or "Split 'train' of toy (seed=1).",
                "entries": TRAIN if entries is None else entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def asks_for(user: str, name: str) -> bool:
    return re.search(rf"the operation {name}\b", user) is not None


def generator(user: str) -> str:
    if T.DX_MARKER in user:
        return json.dumps(
            [
                {
                    "topic": "dimming",
                    "explain": "What does dimming a lamp mean?",
                    "answer": "It lowers the power the lamp draws.",
                    "diagnose": "Why does my lamp flicker when dimmed?",
                }
            ]
        )
    if T.CHOICE_MARKER in user:
        value = re.search(r"to '(\w+)'\.", user).group(1)
        return json.dumps(
            [{"text": f"go to {value.replace('_', ' ')} please", "args": {"scene": value}}]
        )
    if T.DISAMBIGUATION_MARKER in user:
        if asks_for(user, "room_status"):
            return json.dumps(
                [{"text": "kitchen lights?", "args": {"room": "kitchen"}, "confusable": "lamp_on"}]
            )
        return "[]"
    if T.HARD_NEGATIVE_MARKER in user:
        if asks_for(user, "lamp_on"):
            return json.dumps(
                [{"text": "what makes a lamp warm white?", "answer": "Its colour temperature."}]
            )
        return "[]"
    if T.CTC_MARKER in user:
        if asks_for(user, "lamp_on"):
            return json.dumps(
                [
                    {
                        "check": "Is the study lit?",
                        "operation": "room_status",
                        "args": {"room": "study"},
                        "conditional": "check if the study is dark and if so light it up",
                    },
                    {  # the check half must be read-only
                        "check": "light up the study",
                        "operation": "lamp_on",
                        "args": {"room": "study"},
                        "conditional": "check the study then light it",
                    },
                    {  # the check half must validate
                        "check": "status of that",
                        "operation": "room_status",
                        "args": {},
                        "conditional": "check the room then light it",
                    },
                ]
            )
        return "[]"
    return "[]"


def run(tmp_path, recipes, gateway=None, per_recipe=1, domain=DOMAIN, **kw):
    gateway = gateway or FakeGateway(generator=generator)
    client = make_client(tmp_path, gateway)
    out = tmp_path / "supplement.json"
    summary = T.run(
        train_file(tmp_path),
        out,
        tuple(recipes),
        per_recipe,
        53,
        domain,
        client,
        **kw,
    )
    return summary, json.loads(out.read_text(encoding="utf-8")), gateway


# -- missing-argument: rule-based stripping ---------------------------------------------


def arg(op_name: str) -> ArgSpec:
    return TOY.get(op_name).args[0]


def test_strip_removes_the_argument_value_and_its_noun() -> None:
    for pick in range(4):
        stripped, reason = T.strip_argument(
            "Turn on the kitchen lights", arg("lamp_on"), "kitchen", pick, TOY
        )
        assert reason == ""
        assert stripped is not None
        assert "kitchen" not in stripped.lower()
    stripped, _ = T.strip_argument("is the kitchen room lit", arg("room_status"), "kitchen", 0, TOY)
    assert stripped == "is the room lit"  # "the kitchen room" is one span, vague ref is "the room"


def test_strip_handles_a_choice_value_with_underscores() -> None:
    stripped, _ = T.strip_argument("night light mode", arg("set_scene"), "night_light", 0, TOY)
    assert stripped is not None
    assert "night light" not in stripped.lower()
    assert "a different scene" in stripped


def test_strip_reports_no_span_too_short_and_value_remains() -> None:
    assert T.strip_argument("max perf", arg("set_scene"), "bright", 0, TOY) == (
        None,
        "no_argument_span",
    )
    assert T.strip_argument("kitchen", arg("lamp_on"), "kitchen", 2, TOY) == (None, "too_short")
    # one spelling is stripped, another survives
    stripped, reason = T.strip_argument(
        "night light or night-light", arg("set_scene"), "night_light", 0, TOY
    )
    assert reason in {"", "value_remains"}


def test_a_dotted_value_matches_its_bare_stem() -> None:
    svc = ArgSpec(name="service", kind="str")
    stripped, _ = T.strip_argument("Is vllm.service running?", svc, "vllm.service", 0, TOY)
    assert stripped is not None
    assert "vllm" not in stripped.lower()
    stripped, _ = T.strip_argument("restart docker", svc, "docker.service", 0, TOY)
    assert stripped is not None
    assert "docker" not in stripped.lower()


def test_span_nouns_and_vague_refs_derive_from_the_domain() -> None:
    nouns = T.span_nouns(TOY, arg("lamp_on"))
    assert "room" in nouns
    assert "rooms" in nouns
    assert "scene" in nouns
    assert "service" not in nouns  # nothing nvsh-specific
    assert "container" not in nouns
    assert set(T.VAGUE_REFS) == {"str", "choice"}
    grounded = ArgSpec(name="place", kind="str", ground="room")
    assert T.arg_noun(grounded, TOY) == "room"  # the ground kind's noun beats the arg name
    assert T.arg_noun(arg("set_scene"), TOY) == "scene"
    # another domain, other nouns
    other = replace(
        TOY,
        operations=(
            Operation(
                "pump_on", "Start a pump.", False, (ArgSpec(name="pump", kind="str", ground="p"),)
            ),
        ),
        ground_kinds=(replace(TOY.ground_kinds[0], name="p", noun="pump", plural="pumps"),),
    )
    assert "pump" in T.span_nouns(other, other.operations[0].args[0])
    assert "room" not in T.span_nouns(other, other.operations[0].args[0])


def test_missing_argument_units_are_deterministic_and_link_their_original() -> None:
    a, b = {}, {}
    first = T.missing_argument_units(TRAIN, 5, 7, a, TOY)
    second = T.missing_argument_units(TRAIN, 5, 7, b, TOY)
    assert [u.items[0].text for u in first] == [u.items[0].text for u in second]
    assert first
    originals = {e["id"]: e for e in TRAIN}
    for unit in first:
        (item,) = unit.items
        assert item.expect == {"escalate": True}
        assert item.cls == "decline:missing_argument"
        assert unit.group == originals[item.extra["pair_of"]]["source_id"]
        value = next(iter(originals[item.extra["pair_of"]]["expect"]["args"].values()))
        assert value.replace("_", " ") not in item.text.lower()
    assert "toy-01" not in {u.items[0].extra["pair_of"] for u in first}  # no-argument op


def test_missing_argument_caps_per_operation() -> None:
    units = T.missing_argument_units(TRAIN, 1, 3, {}, TOY)
    by_id = {e["id"]: e for e in TRAIN}
    ops = [by_id[u.items[0].extra["pair_of"]]["expect"]["operation"] for u in units]
    assert len(ops) == len(set(ops))


# -- end to end, one recipe at a time ---------------------------------------------------------


def test_missing_argument_recipe_writes_escalate_entries_and_needs_no_generator(tmp_path) -> None:
    summary, doc, gateway = run(tmp_path, ["missing-argument"], per_recipe=5)
    assert doc["entries"]
    for entry in doc["entries"]:
        assert entry["id"].startswith("tgt-marg-")
        assert entry["source"] == "tgt-missing-argument"
        assert entry["expect"] == {"escalate": True}
        assert entry["class"] == "decline:missing_argument"
        assert entry["source_id"] == entry["pair_of"]
    assert summary["recipes"]["missing-argument"]["kept"] == len(doc["entries"])
    assert gateway.roles_called() == {"reviewer_b"}
    users = gateway.users()
    assert any(T.MARG_UNSPECIFIED_MARKER in u for u in users)
    assert any(T.MARG_NATURAL_MARKER in u for u in users)


def test_a_reviewer_b_rejection_drops_the_item_and_is_recorded(tmp_path) -> None:
    gateway = FakeGateway(reject=lambda role, user: T.MARG_NATURAL_MARKER in user)
    summary, doc, _ = run(tmp_path, ["missing-argument"], gateway, per_recipe=5)
    assert doc["entries"] == []
    counts = summary["recipes"]["missing-argument"]
    assert counts["kept"] == 0
    assert counts["rejected"]["reviewer_b"] >= 1


def test_decide_by_both_also_asks_reviewer_a(tmp_path) -> None:
    _, _, gateway = run(tmp_path, ["missing-argument"], per_recipe=5, decide_by="both")
    assert gateway.roles_called() == {"reviewer_a", "reviewer_b"}


def test_a_malformed_verdict_is_an_error_reason_never_a_reject(tmp_path) -> None:
    gateway = FakeGateway(reviewer_reply=lambda role, user: "it seems fine")
    summary, doc, _ = run(tmp_path, ["missing-argument"], gateway, per_recipe=5)
    rejected = summary["recipes"]["missing-argument"]["rejected"]
    assert doc["entries"] == []
    assert rejected["error"] >= 1
    assert "reviewer_b" not in rejected
    assert "reviewer" not in rejected


def test_diagnosis_explain_pairs_share_a_source_id(tmp_path) -> None:
    summary, doc, _ = run(tmp_path, ["diagnosis-explain"])
    explain, diagnose = doc["entries"]
    assert explain["expect"] == {
        "explain": True,
        "answer": "It lowers the power the lamp draws.",
    }
    assert explain["class"] == "explain:question"
    assert diagnose["expect"] == {"escalate": True}
    assert diagnose["class"] == "decline:diagnosis"
    assert explain["source_id"] == diagnose["source_id"]
    assert explain["source_id"].startswith("tgt-dx-p")
    assert {e["source"] for e in doc["entries"]} == {"tgt-diagnosis-explain"}


def test_a_rejected_half_drops_the_whole_pair(tmp_path) -> None:
    gateway = FakeGateway(
        generator=generator, reject=lambda role, user: "flicker when dimmed" in user
    )
    summary, doc, _ = run(tmp_path, ["diagnosis-explain"], gateway)
    assert doc["entries"] == []
    assert summary["recipes"]["diagnosis-explain"]["rejected"]["reviewer_b"] == 1


def test_a_recipe_whose_reason_the_domain_lacks_is_unavailable(tmp_path) -> None:
    with pytest.raises(T.RecipeUnavailable, match="diagnosis"):
        run(tmp_path, ["diagnosis-explain"], domain=TOY)
    assert T.decline_class(TOY, "check-then-change") == "decline:multi_step"


def test_power_set_covers_every_choice_value_in_the_domain(tmp_path) -> None:
    summary, doc, _ = run(tmp_path, ["power-set"])
    targets = T.choice_targets(DOMAIN)
    assert [t[2] for t in targets] == ["bright", "reading", "night_light"]
    got = {(e["expect"]["operation"], json.dumps(e["expect"]["args"])) for e in doc["entries"]}
    assert got == {(op.name, json.dumps({a.name: v})) for op, a, v in targets}
    for entry in doc["entries"]:
        assert DOMAIN.validate_args(entry["expect"]["operation"], entry["expect"]["args"]) is None
        assert entry["id"].startswith("tgt-pset-")


def test_power_set_generalises_to_any_choice_argument() -> None:
    two = replace(
        TOY,
        operations=TOY.operations
        + (Operation("set_dim", "Dim.", False, (ArgSpec("level", "choice", ("low", "high")),)),),
    )
    assert len(T.choice_targets(two)) == 3 + 2


def test_power_set_rejects_invalid_or_off_target_args(tmp_path) -> None:
    def bad(user: str) -> str:
        value = re.search(r"to '(\w+)'\.", user).group(1)
        other = next(c for c in ("bright", "reading") if c != value)
        return json.dumps(
            [
                {"text": "set it to turbo", "args": {"scene": "turbo"}},
                {"text": "go to the other one", "args": {"scene": other}},
                {"text": "go there", "args": {}},
            ]
        )

    summary, doc, _ = run(tmp_path, ["power-set"], FakeGateway(generator=bad), per_recipe=3)
    rejected = summary["recipes"]["power-set"]["rejected"]
    n = len(T.choice_targets(DOMAIN))
    assert doc["entries"] == []
    assert rejected["invalid_args"] >= 2 * n
    assert rejected["wrong_value"] >= 1


def test_disambiguation_keeps_a_validated_gold_with_its_confusable(tmp_path) -> None:
    _, doc, gateway = run(tmp_path, ["disambiguation"])
    (entry,) = doc["entries"]
    assert entry["expect"] == {"operation": "room_status", "args": {"room": "kitchen"}}
    assert entry["confusable"] == "lamp_on"
    assert entry["id"].startswith("tgt-disamb-")
    assert any(T.MOST_NATURAL_MARKER in u for u in gateway.users("reviewer_b"))


def test_disambiguation_rejects_invalid_args_and_a_bad_confusable(tmp_path) -> None:
    def bad(user: str) -> str:
        if not asks_for(user, "room_status"):
            return "[]"
        return json.dumps(
            [
                {"text": "kitchen status", "args": {}, "confusable": "lamp_on"},
                {"text": "study state", "args": {"room": "study"}, "confusable": "x"},
                {"text": "hall health", "args": {"room": "hallway"}, "confusable": "room_status"},
            ]
        )

    summary, doc, _ = run(tmp_path, ["disambiguation"], FakeGateway(generator=bad), per_recipe=3)
    rejected = summary["recipes"]["disambiguation"]["rejected"]
    assert doc["entries"] == []
    assert rejected["invalid_args"] == 1
    assert rejected["invalid_confusable"] == 2


def test_hard_negative_is_an_explain_entry(tmp_path) -> None:
    _, doc, _ = run(tmp_path, ["hard-negative"])
    (entry,) = doc["entries"]
    assert entry["expect"]["explain"] is True
    assert entry["expect"]["answer"]
    assert entry["class"] == "explain:question"
    assert entry["mentions"] == "lamp_on"
    assert entry["id"].startswith("tgt-hneg-")


def test_the_name_the_subject_rule_is_in_the_hard_negative_prompt() -> None:
    system, user = T.hard_negative_prompt(TOY, "lamp_on", 8)
    assert "name the subject the way a user would" in user
    assert "Never write lamp_on or any other operation identifier" in user
    assert TOY.persona in system
    assert "name the subject the way a user would" in T.dx_prompt(TOY, 4)[1]
    assert "Never write an operation identifier" in T.dx_prompt(TOY, 4)[1]
    assert (
        "name the subject the way a user would" in T.check_then_change_prompt(TOY, "lamp_on", 3)[1]
    )


def test_prompts_carry_the_domain_table_and_topics() -> None:
    _, user = T.dx_prompt(TOY, 4)
    assert "lamp_on: Turn on the lamps in one named room." in user
    assert "what a lighting scene is" in user
    assert TOY.description in user
    assert "Jetson" not in user


def test_a_text_naming_an_operation_identifier_is_rejected(tmp_path) -> None:
    def leak(user: str) -> str:
        if not asks_for(user, "lamp_on"):
            return "[]"
        return json.dumps([{"text": "what does lamp_on do?", "answer": "It lights lamps."}])

    summary, doc, _ = run(tmp_path, ["hard-negative"], FakeGateway(generator=leak))
    assert doc["entries"] == []
    assert summary["recipes"]["hard-negative"]["rejected"]["identifier"] == 1


def test_a_repeat_of_a_train_text_is_dropped(tmp_path) -> None:
    def repeat(user: str) -> str:
        if not asks_for(user, "lamp_on"):
            return "[]"
        return json.dumps([{"text": "Which lights are on?", "answer": "Look."}])

    summary, doc, _ = run(tmp_path, ["hard-negative"], FakeGateway(generator=repeat))
    assert summary["recipes"]["hard-negative"]["rejected"]["train_exact"] == 1


def test_exclude_drops_exact_and_near_duplicates_of_a_protected_text(tmp_path) -> None:
    protected = tmp_path / "val.json"
    protected.write_text(
        json.dumps(
            {
                "header": "Split 'val' of toy (seed=1).",
                "entries": [
                    {"id": "v1", "text": "What does dimming a lamp mean?"},
                    {"id": "v2", "text": "Why does my lamp flicker when it is dimmed a lot?"},
                ],
            }
        ),
        encoding="utf-8",
    )
    summary, doc, _ = run(tmp_path, ["diagnosis-explain"], exclude=[protected])
    assert doc["entries"] == []
    assert summary["recipes"]["diagnosis-explain"]["rejected"]["protected_exact"] == 1
    assert doc["targeted"]["exclude"][0]["name"] == "val.json"


def test_a_generator_failure_is_counted_not_raised(tmp_path) -> None:
    def boom(user: str) -> str:
        raise ValueError("boom")

    summary, doc, _ = run(tmp_path, ["hard-negative"], FakeGateway(generator=boom))
    rejected = summary["recipes"]["hard-negative"]["rejected"]
    assert rejected["error"] == len(DOMAIN.operations)
    assert doc["entries"] == []


def test_an_unparseable_generator_reply_is_retried_then_an_error(tmp_path) -> None:
    gateway = FakeGateway(generator=lambda user: "sorry, I cannot do that")
    summary, _, gateway = run(tmp_path, ["diagnosis-explain"], gateway)
    assert summary["recipes"]["diagnosis-explain"]["rejected"]["error"] == 1
    assert len(gateway.users("generator")) == 3  # the client's bounded retries


def test_more_than_one_round_carries_its_batch_number(tmp_path) -> None:
    _, _, gateway = run(tmp_path, ["diagnosis-explain"], per_recipe=7)
    users = gateway.users("generator")
    assert len(users) == 2
    assert len(set(users)) == 2  # never the same request twice
    assert "Batch 2 of 2" in users[1]


def test_check_then_change_pairs_a_read_only_check_with_an_escalation(tmp_path) -> None:
    summary, doc, _ = run(tmp_path, ["check-then-change"], per_recipe=3)
    check, conditional = doc["entries"]
    assert check["expect"] == {"operation": "room_status", "args": {"room": "study"}}
    assert conditional["expect"] == {"escalate": True}
    assert conditional["class"] == "decline:multi_step"
    assert conditional["changes"] == "lamp_on"
    assert check["source_id"] == conditional["source_id"]
    assert summary["recipes"]["check-then-change"]["rejected"]["invalid_item"] == 2


def test_planned_calls_and_candidates_follow_the_domain() -> None:
    mutating = [op for op in DOMAIN.operations if not op.read_only]
    assert T.planned_teacher_calls("check-then-change", 3, DOMAIN) == len(mutating)
    assert T.planned_candidates("check-then-change", 3, DOMAIN) == 3 * len(mutating) * 2
    assert T.planned_teacher_calls("power-set", 3, DOMAIN) == len(T.choice_targets(DOMAIN))
    assert T.planned_teacher_calls("missing-argument", 3, DOMAIN) == 0


def test_plan_reports_without_any_call(tmp_path) -> None:
    planned = T.plan(train_file(tmp_path), T.RECIPES, 3, 5, DOMAIN)
    assert set(planned) == set(T.RECIPES)
    assert planned["missing-argument"]["teacher_calls"] == 0
    assert planned["missing-argument"]["candidates"] >= 1


# -- inputs ---------------------------------------------------------------------------------


def test_the_held_out_file_and_a_non_train_side_are_refused(tmp_path) -> None:
    by_name = train_file(tmp_path, name="held-out.json")
    with pytest.raises(ValueError, match="held-out"):
        T.load_train(by_name)
    by_header = train_file(tmp_path, name="x.json", header="Held-out split of toy")
    with pytest.raises(ValueError, match="held-out"):
        T.load_train(by_header)
    val_side = train_file(tmp_path, name="val.json", header="Split 'val' of toy (seed=1).")
    with pytest.raises(ValueError, match="train side"):
        T.load_train(val_side)


def test_unknown_recipe_and_decide_by_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="recipe"):
        run(tmp_path, ["nope"])
    with pytest.raises(ValueError, match="decide_by"):
        run(tmp_path, ["missing-argument"], decide_by="either")


# -- the output -----------------------------------------------------------------------------


def test_output_is_accepted_by_merge_with_pairs_kept_grouped(tmp_path) -> None:
    _, doc, _ = run(tmp_path, [r for r in T.RECIPES], per_recipe=2)
    assert doc["entries"]
    split = json.loads(train_file(tmp_path).read_text(encoding="utf-8"))
    merged, added = add_supplement(split, doc)
    assert added == len(doc["entries"])
    by_id = {e["id"]: e for e in merged["entries"]}
    for entry in doc["entries"]:
        assert by_id[entry["id"]]["source_id"] == entry["source_id"]
        assert entry["kind"] == "explicit"
        assert entry["source"].startswith("tgt-")


def test_output_carries_counts_sha256_and_a_train_header(tmp_path) -> None:
    summary, doc, _ = run(tmp_path, ["diagnosis-explain"])
    meta = doc["targeted"]
    assert meta["sha256"] == summary["sha256"] == T.sha256_of_entries(doc["entries"])
    assert meta["counts"] == summary["recipes"]
    assert meta["seed"] == 53
    assert meta["domain"] == DOMAIN.name
    assert meta["teachers"]["reviewer_b"]["role"] == "reviewer_b"
    assert "Split 'train' of " in doc["header"]
    assert not {"test", "held-out", "val"} & set(doc["header"].lower().split())


def test_the_summary_never_carries_an_entry_text(tmp_path) -> None:
    summary, doc, _ = run(tmp_path, ["diagnosis-explain", "hard-negative"])
    blob = json.dumps(summary)
    for entry in doc["entries"]:
        assert entry["text"] not in blob


def test_every_recipe_records_its_reject_reasons_per_unit(tmp_path) -> None:
    review = tmp_path / "review.jsonl"
    gateway = FakeGateway(generator=generator, reject=lambda role, user: "flicker" in user)
    summary, _, _ = run(
        tmp_path, ["diagnosis-explain", "hard-negative"], gateway, review_out=review
    )
    rows = [json.loads(line) for line in review.read_text(encoding="utf-8").splitlines()]
    dx = next(r for r in rows if r["recipe"] == "diagnosis-explain")
    assert dx["accepted"] is False
    assert dx["reason"] == "reviewer"
    assert dx["votes"]
    hneg = next(r for r in rows if r["recipe"] == "hard-negative")
    assert hneg["accepted"] is True
    assert hneg["reason"] is None
    assert summary["recipes"]["diagnosis-explain"]["rejected"] == {"reviewer_b": 1}
    assert summary["recipes"]["hard-negative"]["rejected"] == {}
