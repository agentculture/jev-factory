"""Characterization tests pinned before the d17 complexity refactor of ``jev_factory.data``.

They cover the branches the existing suites left unexercised in ``run_augment``,
``_op_candidates``, ``draft_heldout``, ``missing_argument_units``, ``_first_object``
and ``TeacherClient.complete``, and pin current behaviour exactly (outputs, counts
and messages), so the refactor can be shown not to change it.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.data import augment as aug
from jev_factory.data import draft as D
from jev_factory.data import targeted as TG
from jev_factory.data import teachers as T
from jev_factory.domain.model import ArgSpec, Operation
from tests.fixtures.fake_teacher import FakeGateway, make_client, roles
from tests.fixtures.toy_domain import DOMAIN

# -- teachers._first_object ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", None),
        ("no braces here", None),
        ("{", None),
        ("}{", None),
        ("{}", "{}"),
        ('x {"a": 1} y', '{"a": 1}'),
        ('{"a": {"b": 2}} tail', '{"a": {"b": 2}}'),
        ('{"a": "}"}', '{"a": "}"}'),
        ('{"a": "{"}', '{"a": "{"}'),
        # an unbalanced first brace: the scan moves on to the next one
        ('{ {"a": 1}', '{"a": 1}'),
        ('{"open" {"b": 1}', '{"b": 1}'),
        ('{"a": 1', None),
        # a backslash pair inside a string
        ('{"a": "x\\\\"}', '{"a": "x\\\\"}'),
        # an escaped quote does not end the string (fixed after the d17 report)
        ('{"a": "\\"}"} rest', '{"a": "\\"}"}'),
        ('{"a": "\\"x"}', '{"a": "\\"x"}'),
        ("{} {}", "{}"),
        ('"{"}', None),
    ],
)
def test_first_object_pins_the_balanced_scan(text, expected) -> None:
    assert T._first_object(text) == expected


# -- teachers.TeacherClient.complete ------------------------------------------------------


class _Scripted:
    """A caller replaying *replies*; an exception instance is raised instead of returned."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, role, system, user):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _client(tmp_path: Path, caller, sleeps: list[float]) -> T.TeacherClient:
    return T.TeacherClient(
        roles(), tmp_path / "cache", caller=caller, sleep=sleeps.append, backoff=0.5
    )


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:1", code, "x", {}, io.BytesIO(b""))


def test_a_cached_reply_the_parser_refuses_is_asked_again_and_recached(tmp_path) -> None:
    sleeps: list[float] = []
    first = _client(tmp_path, _Scripted("not a verdict"), sleeps)
    assert first.complete("reviewer_b", "s", "u").status == "ok"  # cached as free text
    caller = _Scripted(json.dumps({"verdict": "yes", "reason": " r "}))
    second = _client(tmp_path, caller, sleeps)
    outcome = second.complete("reviewer_b", "s", "u", T.parse_verdict)
    assert outcome.to_record() == {
        "status": "ok",
        "teacher": roles()["reviewer_b"].record(),
        "text": json.dumps({"verdict": "yes", "reason": " r "}),
        "accepted": True,
        "reason": "r",
        "error": "",
        "attempts": 1,
        "cached": False,
    }
    assert caller.calls == 1
    assert second.sent == 1
    third = _client(tmp_path, _Scripted(), sleeps)
    again = third.complete("reviewer_b", "s", "u", T.parse_verdict)
    assert again.cached is True
    assert again.attempts == 0
    assert third.sent == 0
    assert sleeps == []


def test_a_cached_free_text_reply_is_returned_without_a_call(tmp_path) -> None:
    sleeps: list[float] = []
    _client(tmp_path, _Scripted("hello"), sleeps).complete("generator", "s", "u")
    client = _client(tmp_path, _Scripted(), sleeps)
    outcome = client.complete("generator", "s", "u")
    assert (outcome.status, outcome.text, outcome.cached, outcome.attempts) == (
        "ok",
        "hello",
        True,
        0,
    )
    assert client.sent == 0


@pytest.mark.parametrize(
    "replies, error, attempts, sleeps_expected, sent",
    [
        ([_http(400)], "HTTP 400", 1, [], 1),
        ([_http(503), _http(404)], "HTTP 404", 2, [0.5], 2),
        ([_http(503), _http(429), _http(500)], "HTTP 500", 3, [0.5, 1.0], 3),
        (["", "  ", ""], "empty reply", 3, [0.5, 1.0], 3),
        (
            [ValueError("x"), ValueError("y"), ValueError("z")],
            "gateway reply is not JSON",
            3,
            [0.5, 1.0],
            3,
        ),
        ([OSError(), OSError(), OSError()], "OSError", 3, [0.5, 1.0], 3),
        ([OSError("boom"), "", ValueError("v")], "gateway reply is not JSON", 3, [0.5, 1.0], 3),
    ],
)
def test_complete_error_paths_pin_attempts_sleeps_and_message(
    tmp_path, replies, error, attempts, sleeps_expected, sent
) -> None:
    sleeps: list[float] = []
    client = _client(tmp_path, _Scripted(*replies), sleeps)
    outcome = client.complete("generator", "s", "u")
    assert outcome.status == "error"
    assert outcome.error == error
    assert outcome.attempts == attempts
    assert outcome.text == ""
    assert sleeps == sleeps_expected
    assert client.sent == sent
    assert list((tmp_path / "cache").iterdir()) == []


def test_complete_recovers_after_errors_and_caches_the_good_reply(tmp_path) -> None:
    sleeps: list[float] = []
    client = _client(tmp_path, _Scripted(_http(502), "", "fine"), sleeps)
    outcome = client.complete("generator", "s", "u")
    assert (outcome.status, outcome.text, outcome.attempts, outcome.cached) == (
        "ok",
        "fine",
        3,
        False,
    )
    assert sleeps == [0.5, 1.0]
    assert len(list((tmp_path / "cache").iterdir())) == 1


# -- augment.run_augment -------------------------------------------------------------------

AUG_ENTRIES = [
    {
        "id": "s1",
        "source_id": "s1",
        "kind": "explicit",
        "text": "Turn on the kitchen lights",
        "expect": {"operation": "lamp_on", "args": {"room": "kitchen"}},
    },
    {
        "id": "s2",
        "source_id": "s2",
        "kind": "explicit",
        "text": "Which lights are on?",
        "expect": {"operation": "lamp_status", "args": {}},
    },
]


def _aug_seed(tmp_path: Path) -> Path:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps({"header": "Split 'train' of toy (seed=1).", "entries": AUG_ENTRIES}),
        encoding="utf-8",
    )
    return path


def _paraphrase(user: str) -> str:
    return "rewrite of: " + user.split("Original request: ", 1)[1].split("\n", 1)[0]


def _augment(tmp_path: Path, gateway: FakeGateway, **kw) -> aug.PipelineCounts:
    return aug.run_augment(
        [_aug_seed(tmp_path)],
        DOMAIN,
        make_client(tmp_path, gateway),
        tmp_path / "acc.jsonl",
        tmp_path / "rej.jsonl",
        per_source=kw.pop("per_source", 1),
        **kw,
    )


def test_run_augment_with_every_task_done_calls_nothing(tmp_path) -> None:
    first = _augment(tmp_path, FakeGateway(generator=_paraphrase))
    assert first.as_dict()["accepted"] == 2
    gateway = FakeGateway(generator=_paraphrase)
    counts = _augment(tmp_path, gateway)
    assert counts.as_dict() == aug.PipelineCounts().as_dict()
    assert gateway.calls == []


def test_run_augment_counts_errors_rejects_and_accepts_together(tmp_path) -> None:
    def generator(user: str) -> str:
        return "" if "Which lights" in user else _paraphrase(user)

    gateway = FakeGateway(generator=generator, reject=lambda role, user: role == "reviewer_a")
    counts = _augment(tmp_path, gateway, per_source=2, decide_by="both", workers=1)
    assert counts.as_dict() == {
        "generated": 2,
        "corrected": 2,
        "accepted": 0,
        "rejected_by_a": 2,
        "rejected_by_b": 0,
        "errors": 2,
    }
    rejected = [
        json.loads(line) for line in (tmp_path / "rej.jsonl").read_text().splitlines() if line
    ]
    assert sorted(r["id"] for r in rejected) == ["s1~v1", "s1~v2"]
    assert not (tmp_path / "acc.jsonl").exists()


def test_run_augment_dry_run_counts_only(tmp_path) -> None:
    gateway = FakeGateway(generator=_paraphrase)
    counts = _augment(tmp_path, gateway, per_source=3, limit=5, dry_run=True)
    assert counts.as_dict() == {**aug.PipelineCounts().as_dict(), "generated": 5}
    assert gateway.calls == []


# -- draft._op_candidates -------------------------------------------------------------------


def _draft_client(tmp_path: Path, reply: str) -> T.TeacherClient:
    return make_client(tmp_path, FakeGateway(generator=lambda user: reply))


def test_op_candidates_pin_full_output(tmp_path) -> None:
    reply = json.dumps(
        [
            {"text": "show the kitchen", "args": {"room": "kitchen"}},
            {"text": "", "args": {}},
            {"text": "x", "args": "kitchen"},
            {"text": "go reading", "args": {"scene": "reading"}},
            {"text": "plain"},
        ]
    )
    rejects: dict[str, int] = {}
    out = D._op_candidates(
        DOMAIN, _draft_client(tmp_path, reply), 2, rejects, lambda k, fn: fn(), max_batch=1
    )
    assert [(c.kind_tag, c.text, c.expect["args"], c.cls) for c in out] == [
        ("op-lamp_status", "plain", {}, None),
        ("op-lamp_status", "plain", {}, None),
        ("op-list_rooms", "plain", {}, None),
        ("op-list_rooms", "plain", {}, None),
        ("op-room_status", "show the kitchen", {"room": "kitchen"}, None),
        ("op-room_status", "show the kitchen", {"room": "kitchen"}, None),
        ("op-lamp_on", "show the kitchen", {"room": "kitchen"}, None),
        ("op-lamp_on", "show the kitchen", {"room": "kitchen"}, None),
        ("op-set_scene", "go reading", {"scene": "reading"}, None),
        ("op-set_scene", "go reading", {"scene": "reading"}, None),
    ]
    assert rejects == {"invalid_args": 40}


# -- draft.draft_heldout --------------------------------------------------------------------


def test_draft_heldout_pins_every_entry_and_reject(tmp_path) -> None:
    def drafter(system: str, user: str) -> str:
        if "operation lamp_on." in user:
            return json.dumps(
                [
                    {"text": "Light the study", "args": {"room": "study"}},
                    {"text": "light   THE study", "args": {"room": "study"}},  # dup of the first
                    {"text": "light it", "args": {"room": 3}},
                    {"text": "light the den"},
                    {"text": " ", "args": {"room": "den"}},
                ]
            )
        if "operation set_scene." in user:
            return "not json at all"
        if "hand back to a human" in user:
            return json.dumps(["call my landlord", {"text": "Light the study"}])
        if "answered in words" in user:
            return json.dumps(
                [{"text": "what is a scene?", "answer": " a preset "}, "what is a lamp?"]
            )
        return "[]"

    result = D.draft_heldout(
        DOMAIN,
        tmp_path / "ho",
        5,
        drafter,
        teachers={},
        dev_texts=["what is a lamp?"],
        per_op=1,
        escalate=1,
        explain=1,
    )
    doc = json.loads((tmp_path / "ho" / D.DRAFT_FILE).read_text(encoding="utf-8"))
    pinned = [(e["id"], e["text"], e["expect"], e["source"], e["kind"]) for e in doc["entries"]]
    prefix = D.held_out_id_prefix(5)
    source = f"heldout-draft-{D._slug(D.HELD_OUT_MODEL)}"
    assert pinned == [
        (
            f"{prefix}001",
            "Light the study",
            {"operation": "lamp_on", "args": {"room": "study"}},
            source,
            "explicit",
        ),
        (f"{prefix}002", "call my landlord", {"escalate": True}, source, "explicit"),
        (
            f"{prefix}003",
            "what is a scene?",
            {"explain": True, "answer": "a preset"},
            source,
            "explicit",
        ),
    ]
    assert result["rejects"] == {"parse": 1, "invalid_op_args": 2, "duplicate": 4}
    assert result["entries"] == 3


# -- targeted.missing_argument_units --------------------------------------------------------


def _marg_entry(entry_id: str, text: str, op: str, args, kind: str = "explicit", **extra):
    return {
        "id": entry_id,
        "kind": kind,
        "text": text,
        "expect": {"operation": op, "args": args},
        **extra,
    }


def _unit_view(unit):
    (item,) = unit.items
    return (
        unit.recipe,
        unit.group,
        item.text,
        item.expect,
        item.cls,
        item.extra,
        [r[1] for r in item.reviews],
    )


def test_missing_argument_skips_ineligible_entries_and_counts_strip_failures() -> None:
    entries = [
        _marg_entry("e1", "turn on the kitchen lights", "lamp_on", {"room": "kitchen"}),
        _marg_entry("e2", "turn on the hall lights", "lamp_on", {"room": "kitchen"}),  # no span
        _marg_entry("e3", "kitchen", "lamp_on", {"room": "kitchen"}),  # too short
        _marg_entry("e4", "turn on the den lights", "lamp_on", {}),  # invalid args
        _marg_entry("e5", "turn on the den lights", "lamp_on", {"room": "den"}, kind="paraphrase"),
        _marg_entry("e6", "which lamps are on", "lamp_status", {}),  # no argument
        _marg_entry("e7", "do the thing", "no_such_op", {}),
        {"id": "e8", "kind": "explicit", "text": "hm", "expect": None},
        _marg_entry(
            "e9", "set the reading scene", "set_scene", {"scene": "reading"}, source_id="src9"
        ),
    ]
    rejects: dict[str, int] = {}
    units = TG.missing_argument_units(entries, 5, 11, rejects, DOMAIN)
    assert rejects == {"no_argument_span": 1, "too_short": 1}
    views = [_unit_view(u) for u in units]
    assert [(v[1], v[5]) for v in views] == [
        ("e1", {"pair_of": "e1", "stripped_arg": "room"}),
        ("src9", {"pair_of": "e9", "stripped_arg": "scene"}),
    ]
    for view in views:
        assert view[0] == "missing-argument"
        assert view[3] == {"escalate": True}
        assert view[4] == "decline:missing_argument"
        assert "kitchen" not in view[2]
        assert "reading" not in view[2]
        assert TG.MARG_UNSPECIFIED_MARKER in view[6][0]
        assert TG.MARG_NATURAL_MARKER in view[6][1]


TWO_ARG = Operation(
    name="dim_room",
    description="Dim the lamps in one named room to a preset level.",
    read_only=False,
    args=(
        ArgSpec(name="room", kind="str", ground="room"),
        ArgSpec(name="level", kind="choice", choices=("low", "medium")),
    ),
)
TWO_ARG_DOMAIN = replace(DOMAIN, operations=DOMAIN.operations + (TWO_ARG,))


def test_missing_argument_falls_through_to_the_next_argument() -> None:
    entries = [
        # room not spelled, level is: the second argument is stripped
        _marg_entry(
            "d1", "dim things to medium please", "dim_room", {"room": "kitchen", "level": "medium"}
        ),
        # neither spelled: the last argument's reason is counted
        _marg_entry("d2", "dim things please", "dim_room", {"room": "kitchen", "level": "low"}),
        # the first argument strips: the second is never tried
        _marg_entry(
            "d3", "dim the kitchen lights low", "dim_room", {"room": "kitchen", "level": "low"}
        ),
    ]
    rejects: dict[str, int] = {}
    units = TG.missing_argument_units(entries, 5, 2, rejects, TWO_ARG_DOMAIN)
    assert rejects == {"no_argument_span": 1}
    assert sorted((u.group, u.items[0].extra["stripped_arg"]) for u in units) == [
        ("d1", "level"),
        ("d3", "room"),
    ]
    by_group = {u.group: u.items[0] for u in units}
    assert "medium" not in by_group["d1"].text
    assert "kitchen" not in by_group["d3"].text
    assert "level" in by_group["d1"].reviews[0][1]


def test_missing_argument_caps_and_is_deterministic_per_seed() -> None:
    entries = [
        _marg_entry(f"k{i}", f"turn on the {room} lights", "lamp_on", {"room": room})
        for i, room in enumerate(("kitchen", "study", "hall", "den", "attic"))
    ]
    first = [_unit_view(u) for u in TG.missing_argument_units(entries, 2, 9, {}, DOMAIN)]
    second = [_unit_view(u) for u in TG.missing_argument_units(entries, 2, 9, {}, DOMAIN)]
    assert first == second
    assert len(first) == 2
    zero = TG.missing_argument_units(entries, 0, 9, {}, DOMAIN)
    assert zero == []


@pytest.mark.parametrize(
    "reason",
    ['a "}" brace', 'quote " then } then {', "back\\slash", 'mixed \\" and "}"'],
)
def test_a_reason_with_escaped_quotes_and_braces_round_trips(reason) -> None:
    """Regression: an escaped quote inside the reason used to end the string early, so a
    later ``}`` cut the object short and a valid verdict was refused as malformed JSON."""
    reply = "here you go: " + json.dumps({"reason": reason, "verdict": "no"}) + " done"
    assert T.parse_verdict(reply) == (False, reason)
