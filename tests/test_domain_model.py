"""The declarative Domain contract (jev_factory.domain.model) on the toy domain."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from jev_factory.domain import model
from jev_factory.domain.model import (
    ESCALATE,
    EXPLAIN,
    MAX_CANDIDATES,
    ArgSpec,
    Domain,
    GroundDecline,
    Grounded,
    GroundKind,
    Operation,
    Reason,
    UnknownOperation,
    UnknownReason,
    WorldField,
)
from tests.fixtures.toy_domain import DOMAIN

# --- declarative fields ------------------------------------------------------


def test_toy_domain_has_at_least_three_operations_and_one_mutating():
    assert len(DOMAIN.operations) >= 3
    assert any(not op.read_only for op in DOMAIN.operations)
    assert any(op.read_only for op in DOMAIN.operations)


def test_every_declared_field_is_present():
    fields = {f.name for f in dataclasses.fields(Domain)}
    assert {
        "operations",
        "ground_kinds",
        "world_schema",
        "reasons",
        "answer_policy",
        "persona",
        "explain_topics",
        "phrasing_styles",
        "paraphrases",
        "seed_corpus",
        "hub_prefix",
        "card_text",
    } <= fields


def test_domain_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DOMAIN.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        DOMAIN.operations[0].read_only = False  # type: ignore[misc]
    assert isinstance(DOMAIN.operations, tuple)
    assert isinstance(DOMAIN.reasons, tuple)


def test_lists_are_coerced_to_tuples():
    op = Operation("x", "Do x.", True, args=[ArgSpec("a", "choice", choices=["p", "q"])])
    assert isinstance(op.args, tuple)
    assert op.args[0].choices == ("p", "q")
    dom = Domain(
        name="d",
        operations=[op],
        reasons=[Reason("r", "Hand off.", "def")],
        paraphrases={"x": ["Do the x thing."]},
    )
    assert isinstance(dom.operations, tuple)
    assert dom.paraphrases == (("x", ("Do the x thing.",)),)
    hash(dom)  # hashable: every field is immutable


# --- operation queries -------------------------------------------------------


def test_names_keep_table_order():
    assert DOMAIN.names() == (
        "lamp_status",
        "list_rooms",
        "room_status",
        "lamp_on",
        "set_scene",
    )


def test_get_and_read_only():
    assert DOMAIN.get("lamp_on").description.startswith("Turn on")
    assert DOMAIN.get("nope") is None
    assert DOMAIN.read_only("lamp_status") is True
    assert DOMAIN.read_only("set_scene") is False
    with pytest.raises(UnknownOperation):
        DOMAIN.read_only("nope")


def test_unknown_is_mutating_and_controls_are_not():
    assert DOMAIN.is_mutating("lamp_on") is True
    assert DOMAIN.is_mutating("lamp_status") is False
    assert DOMAIN.is_mutating("never_heard_of_it") is True
    assert DOMAIN.is_mutating(EXPLAIN) is False
    assert DOMAIN.is_mutating(ESCALATE) is False
    assert DOMAIN.is_mutating("escalate:injection") is False


def test_candidates_plain_and_with_reasons():
    assert DOMAIN.candidates() == DOMAIN.names() + (EXPLAIN, ESCALATE)
    with_reasons = DOMAIN.candidates(with_reasons=True)
    assert with_reasons == DOMAIN.names() + (
        EXPLAIN,
        "escalate:outside_table",
        "escalate:missing_argument",
        "escalate:multi_step",
        "escalate:injection",
    )
    assert ESCALATE not in with_reasons
    assert MAX_CANDIDATES == 52 == len(model.LABEL_ALPHABET)


def test_calibration_label():
    assert DOMAIN.calibration_label(EXPLAIN) == "(explain)"
    assert DOMAIN.calibration_label(ESCALATE) == "(escalate)"
    assert DOMAIN.calibration_label("lamp_on") == "lamp_on"


# --- reasons: one source, both labels derived --------------------------------


def test_reason_labels_are_derived_from_the_one_list():
    names = [r.name for r in DOMAIN.reasons]
    assert DOMAIN.escalate_labels() == tuple(f"escalate:{n}" for n in names)
    assert DOMAIN.decline_labels() == tuple(f"decline:{n}" for n in names)
    r = DOMAIN.reasons[0]
    assert r.escalate_label == "escalate:outside_table"
    assert r.decline_label == "decline:outside_table"


def test_reason_lookup_accepts_every_spelling():
    for key in ("injection", "escalate:injection", "decline:injection"):
        assert DOMAIN.reason(key).name == "injection"
    with pytest.raises(UnknownReason):
        DOMAIN.reason("escalate:nope")
    with pytest.raises(UnknownReason):
        DOMAIN.reason("other:injection")


def test_reason_descriptions_and_definitions_views():
    descriptions = DOMAIN.reason_descriptions()
    assert list(descriptions) == list(DOMAIN.escalate_labels())
    assert descriptions["escalate:multi_step"].startswith("Hand this off")
    definitions = DOMAIN.reason_definitions()
    assert list(definitions) == [r.name for r in DOMAIN.reasons]
    assert "room" in definitions["missing_argument"]


def test_reason_for_class_rolls_unknown_up_to_default():
    assert DOMAIN.reason_for_class("decline:injection") == "escalate:injection"
    assert DOMAIN.reason_for_class("decline:unheard") == "escalate:outside_table"
    assert DOMAIN.reason_for_class(None) == "escalate:outside_table"
    assert DOMAIN.reason_for_class("operation") == "escalate:outside_table"


# --- argument validation (ported from nvsh/ops/table.py validate) -----------


@pytest.mark.parametrize(
    "name,args,code",
    [
        (42, {}, "unknown_operation"),
        ("nope", {}, "unknown_operation"),
        ("lamp_on", ["kitchen"], "wrong_type"),
        ("lamp_on", {}, "missing_argument"),
        ("lamp_status", {"room": "kitchen"}, "unexpected_argument"),
        ("lamp_on", {"room": 3}, "wrong_type"),
        ("lamp_on", {"room": "  "}, "wrong_type"),
        ("set_scene", {"scene": "disco"}, "bad_choice"),
    ],
)
def test_validate_args_codes(name, args, code):
    problem = DOMAIN.validate_args(name, args)
    assert problem is not None and problem.code == code


def test_validate_args_accepts_valid_calls():
    assert DOMAIN.validate_args("lamp_status", {}) is None
    assert DOMAIN.validate_args("lamp_on", {"room": "kitchen"}) is None
    assert DOMAIN.validate_args("set_scene", {"scene": "night_light"}) is None


def test_choice_spellings():
    spec = DOMAIN.get("set_scene").args[0]
    assert spec.spellings("night_light") == ("night_light", "night light", "night-light")


# --- grounding hook per arg kind (ported from nvsh/ops/ground.py) -----------

WORLD = {"home": "toy-home", "rooms": ["kitchen", "Bedroom", "study", "STUDY"]}


def test_ground_against_world_snapshot_uses_world_spelling():
    got = DOMAIN.ground("lamp_on", {"room": "KITCHEN"}, world=WORLD)
    assert got == Grounded(args={"room": "kitchen"})
    got = DOMAIN.ground("lamp_on", {"room": "bedroom"}, world=WORLD)
    assert got == Grounded(args={"room": "Bedroom"})


def test_ground_declines_unknown_and_ambiguous():
    missing = DOMAIN.ground("lamp_on", {"room": "garage"}, world=WORLD)
    assert isinstance(missing, GroundDecline) and missing.code == "no_such_room"
    ambiguous = DOMAIN.ground("lamp_on", {"room": "study"}, world=WORLD)
    assert isinstance(ambiguous, GroundDecline) and ambiguous.code == "ambiguous"


def test_ground_without_grounded_args_needs_no_world():
    assert DOMAIN.ground("set_scene", {"scene": "bright"}) == Grounded(args={"scene": "bright"})
    assert DOMAIN.ground("lamp_status", {}) == Grounded(args={})


def test_ground_lookup_failures_never_raise():
    no_world = DOMAIN.ground("lamp_on", {"room": "kitchen"})
    assert isinstance(no_world, GroundDecline) and no_world.code == "lookup_failed"
    bad_type = DOMAIN.ground("lamp_on", {"room": 7}, world=WORLD)
    assert isinstance(bad_type, GroundDecline) and bad_type.code == "lookup_failed"
    unknown = DOMAIN.ground("nope", {"room": "kitchen"}, world=WORLD)
    assert isinstance(unknown, GroundDecline) and unknown.code == "unknown_operation"


def test_ground_uses_live_lookup_and_suffix_canonicalisation():
    calls = []

    def lookup():
        calls.append(1)
        return ["vllm.service", "docker.service"]

    kind = GroundKind(
        name="service",
        noun="service",
        plural="services",
        world_field="services",
        suffixes=(".service",),
        lookup=lookup,
    )
    dom = Domain(
        name="svc",
        operations=(
            Operation(
                "service_restart",
                "Restart a named service.",
                False,
                args=(ArgSpec("service", "str", ground="service"),),
            ),
        ),
        ground_kinds=(kind,),
        world_schema=(WorldField("services", "list"),),
        reasons=(Reason("outside_table", "Hand off.", "no op covers it"),),
    )
    assert kind.spellings("vllm") == ("vllm", "vllm.service")
    got = dom.ground("service_restart", {"service": "vllm"})
    assert got == Grounded(args={"service": "vllm.service"})
    assert calls == [1]
    # a snapshot, when given, wins over the live lookup
    got = dom.ground("service_restart", {"service": "nginx"}, world={"services": ["nginx.service"]})
    assert got == Grounded(args={"service": "nginx.service"})
    assert calls == [1]


def test_ground_live_lookup_that_raises_is_a_failed_lookup():
    def broken():
        raise OSError("no systemctl")

    kind = GroundKind("service", "service", "services", "services", lookup=broken)
    dom = dataclasses.replace(
        DOMAIN,
        operations=(
            Operation("s", "Check one service.", True, (ArgSpec("service", "str", "", "service"),)),
        ),
        ground_kinds=(kind,),
        world_schema=(WorldField("services", "list"),),
    )
    got = dom.ground("s", {"service": "x"})
    assert isinstance(got, GroundDecline) and got.code == "lookup_failed"


def test_ground_message_is_scrubbed():
    got = DOMAIN.ground("lamp_on", {"room": "gar\nage\x00" + "x" * 80}, world=WORLD)
    assert isinstance(got, GroundDecline)
    assert "\n" not in got.message and "\x00" not in got.message
    assert len(got.message) < 80


# --- world snapshot schema and seed corpus ----------------------------------


def test_check_world_accepts_the_seed_world_and_flags_problems():
    corpus = DOMAIN.load_seed_corpus()
    assert DOMAIN.check_world(corpus.world) == ()
    problems = DOMAIN.check_world({"rooms": "kitchen"})
    assert any("rooms" in p for p in problems)
    assert any("home" in p for p in problems)


def test_load_seed_corpus_shape():
    corpus = DOMAIN.load_seed_corpus()
    assert corpus.header
    assert corpus.entries
    ops = {e["expect"]["operation"] for e in corpus.entries if "operation" in e["expect"]}
    assert ops == set(DOMAIN.names())
    for entry in corpus.entries:
        cls = entry.get("class", "")
        if cls.startswith("decline:"):
            DOMAIN.reason(cls)  # every decline class names a declared reason


def test_load_seed_corpus_rejects_bad_shape(tmp_path: Path):
    bad = tmp_path / "seed.json"
    bad.write_text(json.dumps({"header": "h", "entries": []}))
    dom = dataclasses.replace(DOMAIN, seed_corpus=bad)
    with pytest.raises(ValueError, match="world"):
        dom.load_seed_corpus()
    dom = dataclasses.replace(DOMAIN, seed_corpus=None)
    with pytest.raises(ValueError, match="no seed corpus"):
        dom.load_seed_corpus()


# --- paraphrases, serialisation, surface hash --------------------------------


def test_paraphrases_for():
    assert DOMAIN.paraphrases_for("lamp_on") == ("Switch the lights on in a given room.",)
    assert DOMAIN.paraphrases_for("list_rooms") == ()


def test_to_dict_round_trips_through_from_dict():
    data = DOMAIN.to_dict()
    json.dumps(data)  # JSON-serialisable
    again = Domain.from_dict(data)
    assert again.to_dict() == data
    assert again.candidates(with_reasons=True) == DOMAIN.candidates(with_reasons=True)


def test_surface_sha256_is_stable_and_tracks_the_candidate_surface():
    digest = DOMAIN.surface_sha256()
    assert len(digest) == 64
    assert Domain.from_dict(DOMAIN.to_dict()).surface_sha256() == digest
    # prose that is not part of the surface does not move the hash
    assert dataclasses.replace(DOMAIN, card_text="other").surface_sha256() == digest
    changed = dataclasses.replace(
        DOMAIN, operations=DOMAIN.operations + (Operation("x", "Do x.", True),)
    )
    assert changed.surface_sha256() != digest
