"""The declarative domain-module contract: one immutable :class:`Domain` per domain.

A *domain module* is everything that makes one jev-like model about one
domain: the bounded candidate set (operations plus the ``explain`` and
``escalate`` controls), which of those candidates mutate, how their arguments
are grounded outside the model, the world snapshot grounding reads, the one
list of escalate reasons, and the prose every generator, reviewer and bundle
card needs. In nvsh these facts were spread over a dozen files
(``nvsh/ops/table.py``, ``ground.py``, ``scorer.py``, ``data/reasons.json``,
``draft_sources.py``, ``augment.py`` ...); here they are one value.

Every stage reads the domain only through this module's small query API:

* operations: :meth:`Domain.get`, :meth:`Domain.names`,
  :meth:`Domain.read_only`, :meth:`Domain.is_mutating` (unknown names count
  as mutating), :meth:`Domain.validate_args`;
* candidates: :meth:`Domain.candidates` (plain, or with the bare ``escalate``
  replaced by one ``escalate:<reason>`` per reason), :meth:`Domain.calibration_label`;
* reasons, from the one list: :meth:`Domain.reason`, :meth:`Domain.escalate_labels`,
  :meth:`Domain.decline_labels`, :meth:`Domain.reason_descriptions` (prompt text),
  :meth:`Domain.reason_definitions` (generator text), :meth:`Domain.reason_for_class`;
* grounding, one hook per groundable arg kind: :meth:`Domain.ground`,
  against a world snapshot or the kind's live lookup, never raising;
* world and seed: :meth:`Domain.check_world`, :meth:`Domain.load_seed_corpus`;
* identity: :meth:`Domain.to_dict` / :meth:`Domain.from_dict` and
  :meth:`Domain.surface_sha256` (the candidate surface a bundle was trained on).

The dataclasses here only *hold* a domain; they do not judge it. Loud
validation, with one named error per problem, lives in
:mod:`jev_factory.domain.validate`. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NVSH_PROVENANCE = {
    "upstream": "nvsh/ops/_model.py",
    "commit": "9debdc6",
    "adaptations": [
        "ArgSpec/Operation frozen dataclasses kept; ArgSpec gains `ground` naming a"
        " domain-declared groundable kind instead of nvsh's hard-coded service/container keys",
        "nvsh/ops/table.py get/names/validate become Domain.get/names/validate_args over the"
        " domain's own operation tuple; ValidationError codes and messages unchanged",
        "nvsh/ops/ground.py _Kind/_match/_scrub become GroundKind + Domain.ground: world"
        " lookup is a snapshot field or an injected live-lookup callable instead of a"
        " systemctl/docker argv; exact case-insensitive match and ambiguity decline unchanged",
        "scripts/lfm-finetune/scorer.py candidates/candidate_pool/reason_for_class/"
        "calibration_label/LABEL_ALPHABET become Domain.candidates(with_reasons=...) etc.",
        "scripts/lfm-finetune/data/reasons.json and draft_sources.REASON_DEFINITIONS merge"
        " into one Reason list; escalate:<r> and decline:<r> are derived from it",
        "scripts/lfm-finetune/scorer.py _spellings becomes ArgSpec.spellings",
    ],
    "licence": "Apache-2.0",
}

#: The two controls every request is offered, after the operations.
EXPLAIN = "explain"
ESCALATE = "escalate"
CONTROLS: tuple[str, ...] = (EXPLAIN, ESCALATE)

#: The controls' labels in a predictions file (metrics.py's convention).
EXPLAIN_LABEL = "(explain)"
ESCALATE_LABEL = "(escalate)"
CALIBRATION_LABELS: Mapping[str, str] = {EXPLAIN: EXPLAIN_LABEL, ESCALATE: ESCALATE_LABEL}

#: Prefixes of the two labels derived from each reason.
ESCALATE_PREFIX = "escalate:"
DECLINE_PREFIX = "decline:"

#: Candidate letters, in candidate order: A-Z then a-z.
LABEL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
MAX_CANDIDATES = len(LABEL_ALPHABET)

#: The argument kinds an operation may declare.
ARG_KINDS: frozenset[str] = frozenset({"str", "choice"})

#: The value kinds a world-snapshot field may declare.
WORLD_KINDS: frozenset[str] = frozenset({"list", "str"})

DEFAULT_INSTRUCTION = (
    "You pick the one action that handles the user's request."
    " Answer with the action's letter only."
)

#: A live world lookup: returns every current value of one groundable kind.
Lookup = Callable[[], Sequence[str]]


class UnknownOperation(KeyError):
    """A name that is not one of the domain's operations."""


class UnknownReason(KeyError):
    """A name or label that is not one of the domain's escalate reasons."""


def _tuple(value: Iterable[Any] | None) -> tuple[Any, ...]:
    return () if value is None else tuple(value)


def _set(obj: object, name: str, value: object) -> None:
    object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgSpec:
    """One argument an operation declares.

    ``kind`` is ``"str"`` (free text, optionally grounded) or ``"choice"``
    (one of ``choices``). ``ground`` names the domain's :class:`GroundKind`
    that grounds a ``str`` argument, or ``None`` when the value is taken as given.
    """

    name: str
    kind: str
    choices: tuple[str, ...] = ()
    ground: str | None = None

    def __post_init__(self) -> None:
        _set(self, "choices", _tuple(self.choices))

    @staticmethod
    def spellings(choice: str) -> tuple[str, ...]:
        """The ways a user writes *choice*: as is, with spaces, with hyphens."""
        return tuple(dict.fromkeys((choice, choice.replace("_", " "), choice.replace("_", "-"))))


@dataclass(frozen=True)
class Operation:
    """One candidate action: its name, its one-line description and whether it is read-only."""

    name: str
    description: str
    read_only: bool
    args: tuple[ArgSpec, ...] = ()

    def __post_init__(self) -> None:
        _set(self, "args", _tuple(self.args))


@dataclass(frozen=True)
class GroundKind:
    """How one groundable argument kind is looked up and canonicalised.

    Values come from the world snapshot's ``world_field`` or, with no
    snapshot, from ``lookup()``. A user value matches a world value when one
    of :meth:`spellings` equals it case-insensitively (``suffixes`` lets
    ``vllm`` match ``vllm.service``). ``lookup`` is behaviour, not data: it
    is left out of equality, hashing and :meth:`Domain.to_dict`.
    """

    name: str
    noun: str
    plural: str
    world_field: str
    suffixes: tuple[str, ...] = ()
    lookup: Lookup | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _set(self, "suffixes", _tuple(self.suffixes))

    def spellings(self, value: str) -> tuple[str, ...]:
        """*value* as given, then with each declared suffix added."""
        return tuple(dict.fromkeys((value, *(value + suffix for suffix in self.suffixes))))


@dataclass(frozen=True)
class WorldField:
    """One field of the world snapshot grounding reads: a ``list`` of names or a ``str``."""

    name: str
    kind: str = "list"
    description: str = ""


@dataclass(frozen=True)
class Reason:
    """One escalate reason, the single source of both of its texts and both labels.

    ``description`` is the prompt text offered beside the ``escalate:<name>``
    candidate; ``definition`` is the one-line shape the generator and
    reviewer prompts use.
    """

    name: str
    description: str
    definition: str

    @property
    def escalate_label(self) -> str:
        return ESCALATE_PREFIX + self.name

    @property
    def decline_label(self) -> str:
        return DECLINE_PREFIX + self.name


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """An argument problem from :meth:`Domain.validate_args` (returned, never raised)."""

    code: str
    message: str


@dataclass(frozen=True)
class Grounded:
    """Every grounded argument replaced by the world's own spelling of it."""

    args: dict[str, str]


@dataclass(frozen=True)
class GroundDecline:
    """Grounding refused: ``no_such_<noun>``, ``ambiguous``, ``lookup_failed`` or
    ``unknown_operation``, with a one-line message."""

    code: str
    message: str


@dataclass(frozen=True)
class SeedCorpus:
    """A domain's seed corpus: ``{header, world, entries}`` (nvsh's dev.json shape)."""

    header: str
    world: Mapping[str, Any]
    entries: tuple[Mapping[str, Any], ...]


def _scrub(value: str) -> str:
    """Every non-printable-ASCII character as ``?``, truncated to 40."""
    return "".join(c if c.isascii() and c.isprintable() else "?" for c in value)[:40]


def _match(value: str, candidates: Sequence[str], kind: GroundKind) -> str | GroundDecline:
    """The world's one spelling of *value*: exact case-insensitive match, never fuzzy.

    Every candidate is examined, so two that differ only by case are
    ambiguous rather than the first listed winning.
    """
    wanted = {spelling.casefold() for spelling in kind.spellings(value)}
    matches = sorted({c for c in candidates if isinstance(c, str) and c.casefold() in wanted})
    if len(matches) == 1:
        return matches[0]
    if matches:
        return GroundDecline("ambiguous", f"{_scrub(value)} matches: {', '.join(matches)}")
    return GroundDecline(f"no_such_{kind.noun}", f"no such {kind.noun}: {_scrub(value)}")


def _paraphrases(value: Mapping[str, Iterable[str]] | Iterable[Any] | None) -> tuple:
    pairs = value.items() if isinstance(value, Mapping) else _tuple(value)
    return tuple((str(name), _tuple(texts)) for name, texts in pairs)


# ---------------------------------------------------------------------------
# The domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """One declarative domain module. Hold it; validate it with
    :func:`jev_factory.domain.validate.validate`.

    ``operations`` are in candidate order (it fixes each one's letter).
    ``paraphrases`` pairs an operation name with alternative descriptions
    for the permutation probe. ``seed_corpus`` is the path of a
    ``{header, world, entries}`` JSON file (train-only). ``control_descriptions``
    overrides the prompt text of the ``explain``/``escalate`` controls (a model trained
    elsewhere is measured with the exact text it was trained on); unset, the backbone's
    default control text is used.
    """

    name: str
    operations: tuple[Operation, ...] = ()
    description: str = ""
    ground_kinds: tuple[GroundKind, ...] = ()
    world_schema: tuple[WorldField, ...] = ()
    reasons: tuple[Reason, ...] = ()
    default_reason: str = ""
    instruction: str = DEFAULT_INSTRUCTION
    answer_policy: str = ""
    persona: str = ""
    explain_topics: tuple[str, ...] = ()
    phrasing_styles: tuple[str, ...] = ()
    paraphrases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    seed_corpus: Path | None = None
    hub_prefix: str = ""
    card_text: str = ""
    licence: str = "Apache-2.0"
    control_descriptions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "operations",
            "ground_kinds",
            "world_schema",
            "reasons",
            "explain_topics",
            "phrasing_styles",
        ):
            _set(self, name, _tuple(getattr(self, name)))
        _set(self, "paraphrases", _paraphrases(self.paraphrases))
        controls = self.control_descriptions
        pairs = controls.items() if isinstance(controls, Mapping) else _tuple(controls)
        _set(self, "control_descriptions", tuple((str(k), str(v)) for k, v in pairs))
        if self.seed_corpus is not None and not isinstance(self.seed_corpus, Path):
            _set(self, "seed_corpus", Path(self.seed_corpus))

    # -- operations --------------------------------------------------------

    def get(self, name: str) -> Operation | None:
        """The operation called *name*, or ``None``."""
        for op in self.operations:
            if op.name == name:
                return op
        return None

    def names(self) -> tuple[str, ...]:
        """Every operation name, in table order."""
        return tuple(op.name for op in self.operations)

    def read_only(self, name: str) -> bool:
        """Whether operation *name* is read-only; :class:`UnknownOperation` otherwise."""
        op = self.get(name)
        if op is None:
            raise UnknownOperation(name)
        return op.read_only

    def is_mutating(self, name: str) -> bool:
        """``True`` unless *name* is a read-only operation, a control or a reason label.

        An unknown name counts as mutating: the safe side for a gate.
        """
        if name in CONTROLS or name in self.escalate_labels():
            return False
        op = self.get(name)
        return True if op is None else not op.read_only

    def validate_args(self, name: object, args: object) -> ValidationError | None:
        """The first problem with calling *name* with *args*, or ``None``. Never raises."""
        if not isinstance(name, str):
            return ValidationError("unknown_operation", f"{name!r} is not a valid operation name")
        op = self.get(name)
        if op is None:
            return ValidationError("unknown_operation", f"unknown operation {name!r}")
        if not isinstance(args, dict):
            return ValidationError(
                "wrong_type", f"arguments to {name!r} must be a dict, not {type(args).__name__}"
            )
        declared = {spec.name for spec in op.args}
        for spec in op.args:
            if spec.name not in args:
                return ValidationError(
                    "missing_argument", f"{name!r} requires argument {spec.name!r}"
                )
        for key in args:
            if key not in declared:
                return ValidationError(
                    "unexpected_argument", f"{name!r} does not accept argument {key!r}"
                )
        for spec in op.args:
            problem = _check_value(name, spec, args[spec.name])
            if problem is not None:
                return problem
        return None

    # -- candidates --------------------------------------------------------

    def candidates(self, with_reasons: bool = False) -> tuple[str, ...]:
        """Every operation, then ``explain`` and ``escalate``; with reasons, the bare
        ``escalate`` is replaced by one ``escalate:<reason>`` per reason."""
        if with_reasons:
            return self.names() + (EXPLAIN,) + self.escalate_labels()
        return self.names() + CONTROLS

    @staticmethod
    def calibration_label(name: str) -> str:
        """*name*'s label in a predictions file: the control label, else the name."""
        return CALIBRATION_LABELS.get(name, name)

    # -- reasons -----------------------------------------------------------

    def reason(self, key: str) -> Reason:
        """The reason *key* names: ``<r>``, ``escalate:<r>`` or ``decline:<r>``."""
        name = key
        for prefix in (ESCALATE_PREFIX, DECLINE_PREFIX):
            if key.startswith(prefix):
                name = key[len(prefix) :]
                break
        for reason in self.reasons:
            if reason.name == name:
                return reason
        raise UnknownReason(key)

    def escalate_labels(self) -> tuple[str, ...]:
        return tuple(r.escalate_label for r in self.reasons)

    def decline_labels(self) -> tuple[str, ...]:
        return tuple(r.decline_label for r in self.reasons)

    def control_description(self, name: str) -> str | None:
        """The domain's own prompt text for control *name*, or ``None`` for the default."""
        return dict(self.control_descriptions).get(name)

    def reason_descriptions(self) -> dict[str, str]:
        """``escalate:<r>`` -> prompt description (nvsh's data/reasons.json shape)."""
        return {r.escalate_label: r.description for r in self.reasons}

    def reason_definitions(self) -> dict[str, str]:
        """``<r>`` -> generator definition (nvsh's draft_sources.REASON_DEFINITIONS shape)."""
        return {r.name: r.definition for r in self.reasons}

    def default_reason_label(self) -> str:
        name = self.default_reason or (self.reasons[0].name if self.reasons else "")
        return ESCALATE_PREFIX + name

    def reason_for_class(self, cls: str | None) -> str:
        """The ``escalate:<r>`` a corpus ``class`` (``decline:<r>``) names.

        A missing or unknown class rolls up to the default reason, never a
        fabricated one.
        """
        if cls and cls.startswith(DECLINE_PREFIX):
            label = ESCALATE_PREFIX + cls[len(DECLINE_PREFIX) :]
            if label in self.escalate_labels():
                return label
        return self.default_reason_label()

    # -- grounding ---------------------------------------------------------

    def ground_kind(self, name: str) -> GroundKind | None:
        for kind in self.ground_kinds:
            if kind.name == name:
                return kind
        return None

    def ground(
        self,
        operation: str,
        args: Mapping[str, object],
        world: Mapping[str, Any] | None = None,
    ) -> Grounded | GroundDecline:
        """Ground every argument of *operation* that declares a ground kind. Never raises.

        Values come from *world* (a snapshot) when given, else from the
        kind's live ``lookup``. The untrusted value is only ever compared
        with those values. Other arguments are copied unchanged, and an
        operation with nothing to ground needs no world at all.
        """
        op = self.get(operation)
        if op is None:
            return GroundDecline("unknown_operation", f"unknown operation {_scrub(str(operation))}")
        result: dict[str, Any] = dict(args)
        seen: dict[str, list[str] | GroundDecline] = {}
        for spec in op.args:
            if spec.ground is None or spec.name not in args:
                continue
            kind = self.ground_kind(spec.ground)
            if kind is None:
                return GroundDecline("lookup_failed", f"no ground kind {spec.ground!r}")
            value = args[spec.name]
            if not isinstance(value, str):
                return GroundDecline(
                    "lookup_failed", f"{spec.name} must be a string to be grounded"
                )
            if kind.name not in seen:
                seen[kind.name] = _world_values(kind, world)
            values = seen[kind.name]
            if isinstance(values, GroundDecline):
                return values
            matched = _match(value, values, kind)
            if isinstance(matched, GroundDecline):
                return matched
            result[spec.name] = matched
        return Grounded(args=result)

    # -- world and seed ----------------------------------------------------

    def check_world(self, snapshot: object) -> tuple[str, ...]:
        """Every way *snapshot* misses :attr:`world_schema` (empty when it fits)."""
        if not isinstance(snapshot, Mapping):
            return (f"world snapshot must be an object, not {type(snapshot).__name__}",)
        problems: list[str] = []
        for wf in self.world_schema:
            if wf.name not in snapshot:
                problems.append(f"world snapshot lacks {wf.name!r}")
                continue
            value = snapshot[wf.name]
            if wf.kind == "list":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    problems.append(f"world field {wf.name!r} must be a list of strings")
            elif not isinstance(value, str):
                problems.append(f"world field {wf.name!r} must be a string")
        return tuple(problems)

    def load_seed_corpus(self) -> SeedCorpus:
        """Read :attr:`seed_corpus`; :class:`ValueError` when it is absent or misshapen."""
        if self.seed_corpus is None:
            raise ValueError(f"domain {self.name!r} declares no seed corpus")
        data = json.loads(self.seed_corpus.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{self.seed_corpus}: seed corpus must be a JSON object")
        header, world, entries = data.get("header"), data.get("world"), data.get("entries")
        if not isinstance(header, str):
            raise ValueError(f"{self.seed_corpus}: seed corpus needs a string 'header'")
        if not isinstance(world, dict):
            raise ValueError(f"{self.seed_corpus}: seed corpus needs a 'world' object")
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ValueError(f"{self.seed_corpus}: seed corpus needs an 'entries' list")
        problems = self.check_world(world)
        if problems:
            raise ValueError(f"{self.seed_corpus}: " + "; ".join(problems))
        return SeedCorpus(header=header, world=world, entries=tuple(entries))

    # -- prose -------------------------------------------------------------

    def paraphrases_for(self, name: str) -> tuple[str, ...]:
        for op_name, texts in self.paraphrases:
            if op_name == name:
                return texts
        return ()

    # -- identity ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form (live lookups left out); :meth:`from_dict` reverses it."""
        return {
            "name": self.name,
            "description": self.description,
            "operations": [
                {
                    "name": op.name,
                    "description": op.description,
                    "read_only": op.read_only,
                    "args": [
                        {
                            "name": a.name,
                            "kind": a.kind,
                            "choices": list(a.choices),
                            "ground": a.ground,
                        }
                        for a in op.args
                    ],
                }
                for op in self.operations
            ],
            "ground_kinds": [
                {
                    "name": k.name,
                    "noun": k.noun,
                    "plural": k.plural,
                    "world_field": k.world_field,
                    "suffixes": list(k.suffixes),
                }
                for k in self.ground_kinds
            ],
            "world_schema": [
                {"name": w.name, "kind": w.kind, "description": w.description}
                for w in self.world_schema
            ],
            "reasons": [
                {"name": r.name, "description": r.description, "definition": r.definition}
                for r in self.reasons
            ],
            "default_reason": self.default_reason,
            "instruction": self.instruction,
            "answer_policy": self.answer_policy,
            "persona": self.persona,
            "explain_topics": list(self.explain_topics),
            "phrasing_styles": list(self.phrasing_styles),
            "paraphrases": {name: list(texts) for name, texts in self.paraphrases},
            "seed_corpus": None if self.seed_corpus is None else str(self.seed_corpus),
            "hub_prefix": self.hub_prefix,
            "card_text": self.card_text,
            "licence": self.licence,
            "control_descriptions": dict(self.control_descriptions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Domain:
        """Build a domain from :meth:`to_dict`'s shape. Unknown keys raise ``TypeError``."""
        fields = dict(data)
        fields["operations"] = tuple(
            Operation(
                name=op["name"],
                description=op["description"],
                read_only=op["read_only"],
                args=tuple(ArgSpec(**a) for a in op.get("args", ())),
            )
            for op in fields.get("operations", ())
        )
        fields["ground_kinds"] = tuple(GroundKind(**k) for k in fields.get("ground_kinds", ()))
        fields["world_schema"] = tuple(WorldField(**w) for w in fields.get("world_schema", ()))
        fields["reasons"] = tuple(Reason(**r) for r in fields.get("reasons", ()))
        return cls(**fields)

    def surface(self) -> dict[str, Any]:
        """What a model trained on this domain sees and is gated by: candidates,
        their prompt text, read-only flags, argument schemas and grounding."""
        data = self.to_dict()
        return {
            "operations": data["operations"],
            "controls": list(CONTROLS),
            "reasons": [
                {"name": r["name"], "description": r["description"]} for r in data["reasons"]
            ],
            "ground_kinds": [
                {k: g[k] for k in ("name", "world_field", "suffixes")} for g in data["ground_kinds"]
            ],
            "instruction": self.instruction,
            "control_descriptions": dict(self.control_descriptions),
        }

    def surface_sha256(self) -> str:
        """sha256 of :meth:`surface` as canonical JSON."""
        blob = json.dumps(self.surface(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check_value(name: str, spec: ArgSpec, value: object) -> ValidationError | None:
    if not isinstance(value, str):
        return ValidationError(
            "wrong_type",
            f"{name!r} argument {spec.name!r} must be a string, not {type(value).__name__}",
        )
    if not value.strip():
        return ValidationError("wrong_type", f"{name!r} argument {spec.name!r} must not be empty")
    if spec.kind == "choice" and value not in spec.choices:
        return ValidationError(
            "bad_choice",
            f"{name!r} argument {spec.name!r} must be one of {spec.choices}, not {value!r}",
        )
    return None


def _world_values(kind: GroundKind, world: Mapping[str, Any] | None) -> list[str] | GroundDecline:
    failed = GroundDecline("lookup_failed", f"could not list {kind.plural}")
    if world is not None:
        values = world.get(kind.world_field)
    elif kind.lookup is not None:
        try:
            values = kind.lookup()
        except Exception:  # noqa: BLE001 -- an injected lookup that fails is a failed lookup
            return failed
    else:
        return failed
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return failed
    return [v for v in values if isinstance(v, str)]
