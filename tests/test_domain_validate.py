"""The domain-module validator (jev_factory.domain.validate) and loader."""

from __future__ import annotations

import dataclasses
import json
import string
import sys
import types
from pathlib import Path

import pytest

from jev_factory.domain import validate as v
from jev_factory.domain.model import ArgSpec, Domain, Operation, WorldField
from tests.fixtures.toy_domain import DOMAIN


def _with(**changes) -> Domain:
    return dataclasses.replace(DOMAIN, **changes)


def _ops(n: int) -> tuple[Operation, ...]:
    return tuple(Operation(f"op_{i}", f"Do thing {i}.", True) for i in range(n))


# --- o25: each rejection carries a named error --------------------------------


@pytest.mark.behavioral("o25")
def test_toy_domain_is_valid():
    assert v.problems(DOMAIN) == ()
    assert v.validate(DOMAIN) is DOMAIN


@pytest.mark.behavioral("o25")
def test_rejects_duplicate_operation():
    dup = _with(operations=DOMAIN.operations + (DOMAIN.operations[0],))
    with pytest.raises(v.DuplicateOperation) as exc:
        v.validate(dup)
    assert exc.value.code == "duplicate_operation"
    assert "lamp_status" in str(exc.value)


@pytest.mark.behavioral("o25")
@pytest.mark.parametrize("text", ["", "   "])
def test_rejects_empty_description(text):
    ops = (dataclasses.replace(DOMAIN.operations[0], description=text),) + DOMAIN.operations[1:]
    domain = _with(operations=ops)
    with pytest.raises(v.EmptyDescription) as exc:
        v.validate(domain)
    assert exc.value.code == "empty_description"
    assert "lamp_status" in str(exc.value)


@pytest.mark.behavioral("o25")
@pytest.mark.parametrize("missing", ["description", "definition"])
def test_rejects_reason_missing_either_text(missing):
    broken = dataclasses.replace(DOMAIN.reasons[1], **{missing: ""})
    reasons = (DOMAIN.reasons[0], broken) + DOMAIN.reasons[2:]
    domain = _with(reasons=reasons)
    with pytest.raises(v.IncompleteReason) as exc:
        v.validate(domain)
    assert exc.value.code == "incomplete_reason"
    assert "missing_argument" in str(exc.value)
    assert missing in str(exc.value)


@pytest.mark.behavioral("o25")
def test_rejects_more_than_52_candidates_including_controls():
    # 50 operations + explain + escalate = 52: allowed
    ok = _with(
        operations=_ops(50),
        reasons=DOMAIN.reasons[:1],
        default_reason=DOMAIN.reasons[0].name,
        paraphrases=(),
    )
    assert v.problems(ok) == ()
    # 51 operations + 2 controls = 53: rejected
    too_many = dataclasses.replace(ok, operations=_ops(51))
    with pytest.raises(v.TooManyCandidates) as exc:
        v.validate(too_many)
    assert exc.value.code == "too_many_candidates"
    assert "53" in str(exc.value)


@pytest.mark.behavioral("o25")
def test_rejects_too_many_candidates_in_reasons_mode():
    # 48 operations + explain + 4 reasons = 53, though plain mode has only 50
    dom = _with(operations=_ops(48), paraphrases=())
    assert len(dom.candidates()) == 50
    with pytest.raises(v.TooManyCandidates) as exc:
        v.validate(dom)
    assert "reasons" in str(exc.value)


@pytest.mark.behavioral("o25")
def test_rejects_unknown_arg_kind():
    op = Operation("dim", "Dim one room.", False, args=(ArgSpec("level", "int"),))
    domain = _with(operations=DOMAIN.operations + (op,))
    with pytest.raises(v.UnknownArgKind) as exc:
        v.validate(domain)
    assert exc.value.code == "unknown_arg_kind"
    assert "'int'" in str(exc.value) and "dim" in str(exc.value)


@pytest.mark.behavioral("o25")
def test_error_classes_are_named_and_share_a_base():
    for cls in (
        v.DuplicateOperation,
        v.EmptyDescription,
        v.IncompleteReason,
        v.TooManyCandidates,
        v.UnknownArgKind,
    ):
        assert issubclass(cls, v.DomainError)
        assert issubclass(cls, ValueError)
        assert cls.code and cls.code == cls.code.lower()


# --- further structural checks ------------------------------------------------


@pytest.mark.parametrize(
    "domain,error",
    [
        (
            _with(operations=(Operation("explain", "Say it.", True),) + DOMAIN.operations),
            v.ReservedName,
        ),
        (
            _with(operations=(Operation("bad name!", "Do it.", True),) + DOMAIN.operations),
            v.InvalidName,
        ),
        (
            _with(
                operations=DOMAIN.operations
                + (Operation("a", "Do a.", True, (ArgSpec("x", "str"), ArgSpec("x", "str"))),)
            ),
            v.DuplicateArgument,
        ),
        (
            _with(
                operations=DOMAIN.operations
                + (Operation("a", "Do a.", True, (ArgSpec("x", "choice"),)),)
            ),
            v.BadChoices,
        ),
        (
            _with(
                operations=DOMAIN.operations
                + (Operation("a", "Do a.", True, (ArgSpec("x", "str", ground="nope"),)),)
            ),
            v.UnknownGroundKind,
        ),
        (
            _with(
                operations=DOMAIN.operations
                + (Operation("a", "Do a.", True, (ArgSpec("x", "choice", ("p",), ground="room"),)),)
            ),
            v.UnknownGroundKind,
        ),
        (_with(reasons=DOMAIN.reasons + (DOMAIN.reasons[0],)), v.DuplicateReason),
        (_with(reasons=()), v.NoReasons),
        (_with(default_reason="nope"), v.UnknownReasonRef),
        (_with(world_schema=(WorldField("home", "str"),)), v.UnknownWorldField),
        (_with(world_schema=DOMAIN.world_schema + (WorldField("x", "dict"),)), v.UnknownWorldField),
        (_with(answer_policy=" "), v.EmptyText),
        (_with(instruction=""), v.EmptyText),
        (_with(paraphrases=(("ghost", ("Boo.",)),)), v.UnknownOperationRef),
        (_with(operations=()), v.NoOperations),
    ],
)
def test_structural_errors(domain, error):
    assert any(isinstance(p, error) for p in v.problems(domain))
    with pytest.raises(v.DomainError):
        v.validate(domain)


def test_problems_collects_every_error():
    bad = _with(
        operations=DOMAIN.operations + (DOMAIN.operations[0], Operation("z", "", True)),
        answer_policy="",
    )
    kinds = {type(p) for p in v.problems(bad)}
    assert {v.DuplicateOperation, v.EmptyDescription, v.EmptyText} <= kinds


# --- loader -----------------------------------------------------------------


def test_load_domain_from_instance_module_and_dotted_name():
    assert v.load_domain(DOMAIN) is DOMAIN
    mod = sys.modules["tests.fixtures.toy_domain"]
    assert v.load_domain(mod) is DOMAIN
    assert v.load_domain("tests.fixtures.toy_domain") is DOMAIN


def test_load_domain_from_mapping_and_json(tmp_path: Path):
    data = DOMAIN.to_dict()
    assert v.load_domain(data).to_dict() == data
    path = tmp_path / "domain.json"
    path.write_text(json.dumps(data))
    assert v.load_domain(path).to_dict() == data


def test_load_domain_is_loud():
    empty = types.ModuleType("empty_domain")
    with pytest.raises(v.NotADomain):
        v.load_domain(empty)
    bad = DOMAIN.to_dict()
    bad["operations"].append(dict(bad["operations"][0]))
    with pytest.raises(v.DuplicateOperation):
        v.load_domain(bad)
    with pytest.raises(v.NotADomain):
        v.load_domain({"name": "x", "operations": "nope"})
    with pytest.raises(v.NotADomain):
        v.load_domain(12)


def test_letters_cover_the_limit():
    assert v.MAX_CANDIDATES == len(string.ascii_letters)
