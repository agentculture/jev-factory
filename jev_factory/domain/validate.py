"""Loud validation and loading of domain modules.

:func:`problems` lists every way a :class:`~jev_factory.domain.model.Domain`
is invalid, each as an instance of a *named* :class:`DomainError` subclass
(``DuplicateOperation``, ``EmptyDescription``, ``IncompleteReason``,
``TooManyCandidates``, ``UnknownArgKind`` and the structural rest).
:func:`validate` raises the first of them; :func:`load_domain` accepts a
``Domain``, a module exposing ``DOMAIN``, a dotted module name, a mapping in
:meth:`Domain.to_dict` shape, or a path to such a JSON file, and returns the
validated domain. Nothing here is a knob: an invalid domain never loads.
"""

from __future__ import annotations

import importlib
import json
import re
import types
from collections.abc import Mapping
from pathlib import Path

from jev_factory.domain.model import (
    ARG_KINDS,
    CONTROLS,
    MAX_CANDIDATES,
    WORLD_KINDS,
    Domain,
)

__all__ = [
    "MAX_CANDIDATES",
    "BadChoices",
    "DomainError",
    "DuplicateArgument",
    "DuplicateOperation",
    "DuplicateReason",
    "EmptyDescription",
    "EmptyText",
    "IncompleteReason",
    "InvalidName",
    "NoOperations",
    "NoReasons",
    "NotADomain",
    "ReservedName",
    "TooManyCandidates",
    "UnknownArgKind",
    "UnknownGroundKind",
    "UnknownOperationRef",
    "UnknownReasonRef",
    "UnknownWorldField",
    "load_domain",
    "problems",
    "validate",
]

#: Operation, argument, reason and ground-kind names: no spaces, no ':'
#: (reserved for ``escalate:<r>`` / ``decline:<r>``), so a candidate line
#: ``"<L>) <name>: <description>"`` always parses.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DomainError(ValueError):
    """An invalid domain module. ``code`` names the rule it broke."""

    code = "invalid_domain"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DuplicateOperation(DomainError):
    code = "duplicate_operation"


class EmptyDescription(DomainError):
    code = "empty_description"


class IncompleteReason(DomainError):
    code = "incomplete_reason"


class TooManyCandidates(DomainError):
    code = "too_many_candidates"


class UnknownArgKind(DomainError):
    code = "unknown_arg_kind"


class NoOperations(DomainError):
    code = "no_operations"


class InvalidName(DomainError):
    code = "invalid_name"


class ReservedName(DomainError):
    code = "reserved_name"


class DuplicateArgument(DomainError):
    code = "duplicate_argument"


class BadChoices(DomainError):
    code = "bad_choices"


class UnknownGroundKind(DomainError):
    code = "unknown_ground_kind"


class UnknownWorldField(DomainError):
    code = "unknown_world_field"


class NoReasons(DomainError):
    code = "no_reasons"


class DuplicateReason(DomainError):
    code = "duplicate_reason"


class UnknownReasonRef(DomainError):
    code = "unknown_reason"


class UnknownOperationRef(DomainError):
    code = "unknown_operation"


class EmptyText(DomainError):
    code = "empty_text"


class NotADomain(DomainError):
    code = "not_a_domain"


def _blank(text: object) -> bool:
    return not isinstance(text, str) or not text.strip()


def _name_problem(what: str, name: object) -> DomainError | None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return InvalidName(f"{what} name {name!r} must match {_NAME_RE.pattern}")
    return None


def _operation_problems(domain: Domain) -> list[DomainError]:
    found: list[DomainError] = []
    if not domain.operations:
        found.append(NoOperations(f"domain {domain.name!r} declares no operations"))
    seen: set[str] = set()
    ground_names = {k.name for k in domain.ground_kinds}
    for op in domain.operations:
        bad_name = _name_problem("operation", op.name)
        if bad_name:
            found.append(bad_name)
        if op.name in CONTROLS:
            found.append(ReservedName(f"operation {op.name!r} collides with a control"))
        if op.name in seen:
            found.append(DuplicateOperation(f"operation {op.name!r} is declared more than once"))
        seen.add(op.name)
        if _blank(op.description):
            found.append(EmptyDescription(f"operation {op.name!r} has an empty description"))
        found.extend(_arg_problems(op.name, op.args, ground_names))
    return found


def _arg_problems(op_name: str, args, ground_names: set[str]) -> list[DomainError]:
    found: list[DomainError] = []
    arg_names: set[str] = set()
    for spec in args:
        bad_name = _name_problem(f"argument of {op_name!r}", spec.name)
        if bad_name:
            found.append(bad_name)
        if spec.name in arg_names:
            found.append(DuplicateArgument(f"{op_name!r} declares argument {spec.name!r} twice"))
        arg_names.add(spec.name)
        if spec.kind not in ARG_KINDS:
            found.append(
                UnknownArgKind(
                    f"{op_name!r} argument {spec.name!r} has unknown kind {spec.kind!r}"
                    f" (known: {', '.join(sorted(ARG_KINDS))})"
                )
            )
        if spec.kind == "choice":
            if not spec.choices or any(_blank(c) for c in spec.choices):
                found.append(
                    BadChoices(f"{op_name!r} argument {spec.name!r} needs non-empty choices")
                )
            elif len(set(spec.choices)) != len(spec.choices):
                found.append(BadChoices(f"{op_name!r} argument {spec.name!r} repeats a choice"))
        elif spec.choices:
            found.append(
                BadChoices(f"{op_name!r} argument {spec.name!r} is not a choice but lists choices")
            )
        if spec.ground is not None:
            if spec.kind != "str":
                found.append(
                    UnknownGroundKind(
                        f"{op_name!r} argument {spec.name!r}: only a str argument is grounded"
                    )
                )
            elif spec.ground not in ground_names:
                found.append(
                    UnknownGroundKind(
                        f"{op_name!r} argument {spec.name!r} grounds on undeclared kind"
                        f" {spec.ground!r}"
                    )
                )
    return found


def _world_problems(domain: Domain) -> list[DomainError]:
    found: list[DomainError] = []
    fields = {}
    for wf in domain.world_schema:
        bad_name = _name_problem("world field", wf.name)
        if bad_name:
            found.append(bad_name)
        if wf.kind not in WORLD_KINDS:
            found.append(UnknownWorldField(f"world field {wf.name!r} has unknown kind {wf.kind!r}"))
        fields[wf.name] = wf.kind
    seen: set[str] = set()
    for kind in domain.ground_kinds:
        bad_name = _name_problem("ground kind", kind.name)
        if bad_name:
            found.append(bad_name)
        if kind.name in seen:
            found.append(InvalidName(f"ground kind {kind.name!r} is declared more than once"))
        seen.add(kind.name)
        if fields.get(kind.world_field) != "list":
            found.append(
                UnknownWorldField(
                    f"ground kind {kind.name!r} reads {kind.world_field!r}, which is not a"
                    " list field of the world schema"
                )
            )
        if _blank(kind.noun) or _blank(kind.plural):
            found.append(EmptyText(f"ground kind {kind.name!r} needs a noun and a plural"))
    return found


def _reason_problems(domain: Domain) -> list[DomainError]:
    found: list[DomainError] = []
    if not domain.reasons:
        found.append(NoReasons(f"domain {domain.name!r} declares no escalate reasons"))
    seen: set[str] = set()
    for reason in domain.reasons:
        bad_name = _name_problem("reason", reason.name)
        if bad_name:
            found.append(bad_name)
        if reason.name in seen:
            found.append(DuplicateReason(f"reason {reason.name!r} is declared more than once"))
        seen.add(reason.name)
        missing = [
            what
            for what, text in (
                ("description", reason.description),
                ("definition", reason.definition),
            )
            if _blank(text)
        ]
        if missing:
            found.append(
                IncompleteReason(
                    f"reason {reason.name!r} is missing its {' and '.join(missing)}"
                    " (prompt description and generator definition are both required)"
                )
            )
    if domain.default_reason and domain.default_reason not in seen:
        found.append(
            UnknownReasonRef(f"default_reason {domain.default_reason!r} is not a declared reason")
        )
    return found


def _candidate_problems(domain: Domain) -> list[DomainError]:
    found: list[DomainError] = []
    for mode, with_reasons in (("plain", False), ("reasons", True)):
        count = len(domain.candidates(with_reasons=with_reasons))
        if count > MAX_CANDIDATES:
            found.append(
                TooManyCandidates(
                    f"{count} candidates including controls in {mode} mode; at most"
                    f" {MAX_CANDIDATES} letters (A-Z, a-z) exist"
                )
            )
    return found


def _text_problems(domain: Domain) -> list[DomainError]:
    found: list[DomainError] = []
    if _blank(domain.name):
        found.append(EmptyText("the domain needs a name"))
    for what in ("instruction", "answer_policy"):
        if _blank(getattr(domain, what)):
            found.append(EmptyText(f"domain {domain.name!r} has an empty {what}"))
    for what in ("explain_topics", "phrasing_styles"):
        if any(_blank(t) for t in getattr(domain, what)):
            found.append(EmptyText(f"domain {domain.name!r} has an empty entry in {what}"))
    names = set(domain.names())
    for op_name, texts in domain.paraphrases:
        if op_name not in names:
            found.append(UnknownOperationRef(f"paraphrases name unknown operation {op_name!r}"))
        if not texts or any(_blank(t) for t in texts):
            found.append(EmptyText(f"paraphrases for {op_name!r} include an empty text"))
    return found


def problems(domain: Domain) -> tuple[DomainError, ...]:
    """Every rule *domain* breaks, as named errors (empty when it is valid)."""
    return tuple(
        _operation_problems(domain)
        + _world_problems(domain)
        + _reason_problems(domain)
        + _candidate_problems(domain)
        + _text_problems(domain)
    )


def validate(domain: Domain) -> Domain:
    """Return *domain* unchanged, or raise its first :class:`DomainError`.

    The raised error carries every problem found in ``.problems``.
    """
    if not isinstance(domain, Domain):
        raise NotADomain(f"expected a Domain, got {type(domain).__name__}")
    found = problems(domain)
    if found:
        first = found[0]
        first.problems = found  # type: ignore[attr-defined]
        raise first
    return domain


def _from_mapping(data: Mapping) -> Domain:
    try:
        return Domain.from_dict(data)
    except (TypeError, KeyError, AttributeError, ValueError) as exc:
        raise NotADomain(f"not a domain definition: {exc}") from exc


def load_domain(source: object) -> Domain:
    """Load and validate a domain from a ``Domain``, a module with ``DOMAIN``, a dotted
    module name, a mapping, or a JSON file path. Raises a :class:`DomainError`."""
    if isinstance(source, Domain):
        return validate(source)
    if isinstance(source, types.ModuleType):
        domain = getattr(source, "DOMAIN", None)
        if not isinstance(domain, Domain):
            raise NotADomain(f"module {source.__name__!r} defines no DOMAIN")
        return validate(domain)
    if isinstance(source, Mapping):
        return validate(_from_mapping(source))
    if isinstance(source, Path) or (isinstance(source, str) and source.endswith(".json")):
        path = Path(source)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NotADomain(f"cannot read domain file {path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise NotADomain(f"{path}: a domain file must hold a JSON object")
        return validate(_from_mapping(data))
    if isinstance(source, str):
        try:
            module = importlib.import_module(source)
        except ImportError as exc:
            raise NotADomain(f"cannot import domain module {source!r}: {exc}") from exc
        return load_domain(module)
    raise NotADomain(f"cannot load a domain from {type(source).__name__}")
