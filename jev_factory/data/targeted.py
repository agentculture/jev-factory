"""Targeted train-side augmentation: one recipe per shape the training set lacks.

A failure class found between runs (a wrong-way confusion, a missing
hand-off, a thin slice) is a data request. Each recipe here is one such
request, run over a train-side split file and written as a train-only
supplement that :func:`jev_factory.core.merge_variations.add_supplement`
folds in:

``missing-argument``
    Rule-based, no teacher. From the train side's own explicit operation
    entries whose operation takes an argument and whose request names that
    argument's value, the value's span is replaced by a vague reference ("the
    room", "a different scene", "it") picked deterministically from the seed
    and the entry id. Expected answer: escalate, class
    ``decline:missing_argument``. The new entry keeps ``pair_of`` and the
    original's ``source_id``, so the complete request and its stripped twin
    stay one group. Mechanical stripping can yield broken English, so each
    stripped text is put to the reviewer twice: "does it leave the argument
    unspecified?" and "is it a natural request?"; both must be yes.
``diagnosis-explain``
    Teacher-drafted contrastive pairs about one topic: "what does X mean / how
    does X work" (explain, with a one-sentence answer) against "why is X
    happening to me / fix X" (escalate, ``decline:diagnosis``). A pair is kept
    or dropped whole.
``power-set``
    Explicit-mode positives for **every choice value** of every choice
    argument of every operation in the domain, requested by synonym. Nothing
    here names an operation: the targets come from the domain.
``disambiguation``
    Per operation, requests whose wording could plausibly be mistaken for
    another listed operation, each with its single correct operation and
    arguments (validated by the domain) and the ``confusable`` operation
    recorded; the reviewer must agree the gold is the most natural reading.
``hard-negative``
    Per operation, knowledge questions that mention what the operation deals
    with only in passing but want an answer in words (explain + answer).
``check-then-change``
    Per operation that changes state, pairs: a read-only check alone against
    the same check followed by a conditional change, which escalates as
    ``decline:multi_step``. Kept or dropped whole.

The escalate recipes need the domain to declare the reason they label with
(``missing_argument``, ``diagnosis``, ``multi_step``); one that does not is a
loud :class:`RecipeUnavailable`, never a silently mislabelled row.

Every entry is in corpus format (``id``, ``kind``, ``text``, ``expect``,
``class`` where it has one, ``source: "tgt-<recipe>"``, ``source_id``), ids
prefixed per recipe (``tgt-marg-0001``, ``tgt-dx-...``). Before any review
call a candidate is dropped when it names an operation identifier, copies
answer-template wording or asks for a hand-off in so many words (the
:mod:`~jev_factory.data.augment` guards), when it exactly repeats a train
text or an earlier kept text, or, exactly or as a near-duplicate
(:func:`jev_factory.core.leakage.match`), a text of an ``exclude`` file.
Each recipe **records its reject reasons** (counts per reason, and with
``review_out`` one row per candidate with its reason and every verdict). Only
counts and a sha256 are returned, never an entry's text.

Verdicts and drafts go through :mod:`jev_factory.data.teachers`: a failed
call or unparseable reply is an ``error`` reason, never a reviewer reject.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jev_factory.core.merge_variations import _TRAIN_HEADER, _normal
from jev_factory.data.augment import (
    DECIDE_BY_RULES,
    HELD_OUT_MARKER,
    HELD_OUT_NAME,
    CandidateError,
    asks_for_handoff,
    copies_answer_template,
    names_internal_operation,
    reviewer_verdict,
)
from jev_factory.data.common import (
    bump,
    duplicates_against,
    expect_words,
    list_check,
    operation_table_text,
    parse_json_list,
    reviewer_intro,
    sha256_of_entries,
)
from jev_factory.data.teachers import TeacherClient
from jev_factory.domain.model import ARG_KINDS, DECLINE_PREFIX, ArgSpec, Domain, Operation

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/targeted_augment.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 1-1087 imported; nvsh.ops.table becomes a Domain (operations, validate_args,"
        " choice targets) and every prompt's table, persona and topics come from it",
        "_SPAN_NOUNS (:200-211) derive from the domain's ground-kind nouns and argument names;"
        " VAGUE_REFS (:224-227) are keyed by the domain's ARG_KINDS and use the argument's"
        " ground-kind noun",
        "the Jetson-specific diagnosis-explain topic list and the 'GPU usage' example in the"
        " hard-negative, diagnosis and check-then-change prompts become the domain's"
        " explain_topics and a generic 'name the subject the way a user would' rule",
        "recipe reasons (decline:missing_argument, decline:diagnosis, decline:multi_step)"
        " must be declared by the domain; ids and sources use the tgt- prefix, not t15-",
        "draft_sources.py helpers the recipes used (reviewer prompt and expectation words,"
        " the strict reviewer system, JSON-list parsing, exact/near-duplicate check,"
        " entry sha256) are rebuilt here from the Domain; the near-duplicate check is"
        " core.leakage.match",
        "roles, the seeded HTTP caller, retry policy and free-text verdict parsing (and the"
        " allowed-hedge lists they needed) are replaced by TeacherClient: JSON verdicts, JSON-list"
        " replies validated by the client so a bad one is retried uncached",
        "a multi-round ask (k above the batch size) carries its batch number so the client cache"
        " never answers two rounds with one reply",
        "the argparse main and --roles-from are not ported; the stage engine is the CLI",
    ],
    "licence": "Apache-2.0",
}

RECIPES: tuple[str, ...] = (
    "missing-argument",
    "diagnosis-explain",
    "power-set",
    "disambiguation",
    "hard-negative",
    "check-then-change",
)
ID_PREFIX = "tgt"
ID_TAGS: dict[str, str] = {
    "missing-argument": "marg",
    "diagnosis-explain": "dx",
    "power-set": "pset",
    "disambiguation": "disamb",
    "hard-negative": "hneg",
    "check-then-change": "ctc",
}
#: The recipes a teacher drafts (``missing-argument`` is rule-based).
GENERATED = frozenset(RECIPES) - {"missing-argument"}

#: The escalate reason each recipe labels its escalate rows with.
RECIPE_REASONS: dict[str, str] = {
    "missing-argument": "missing_argument",
    "diagnosis-explain": "diagnosis",
    "check-then-change": "multi_step",
}

#: Most items asked of the generator in one call; more are asked in rounds.
BATCH = 5

EXPLAIN_CLASS = "explain:question"

# Phrases the prompts below carry verbatim (tests route a fake teacher on them).
MARG_UNSPECIFIED_MARKER = "leave the"
MARG_NATURAL_MARKER = "natural, grammatical request"
DX_MARKER = "contrastive pairs"
CHOICE_MARKER = "explicitly ask to set"
DISAMBIGUATION_MARKER = "could plausibly be mistaken"
HARD_NEGATIVE_MARKER = "only in passing"
MOST_NATURAL_MARKER = "most natural reading"
CTC_MARKER = "check-then-change pairs"

#: The rule that keeps an operation identifier out of a drafted request: users
#: describe what they want, they never name the table's identifiers.
NAME_THE_SUBJECT = (
    "name the subject the way a user would (describe what it is in plain words), "
    "not with the operation's identifier"
)


class RecipeUnavailable(ValueError):
    """A recipe whose escalate reason the domain does not declare."""


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One would-be entry: its text, answer, class, the review prompts that
    must all be accepted, and any extra fields it carries into the output."""

    text: str
    expect: dict[str, Any]
    cls: str | None
    reviews: list[tuple[str, str]]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Unit:
    """Items kept or dropped together (a contrastive pair is one unit).

    ``group`` is the ``source_id`` every item shares; ``None`` means a fresh
    one (the entry's own id for a single item, ``tgt-<tag>-pNNNN`` for a pair).
    """

    recipe: str
    items: list[Item]
    group: str | None = None


def decline_class(domain: Domain, recipe: str) -> str:
    """The corpus ``class`` (``decline:<reason>``) *recipe* labels escalations with."""
    name = RECIPE_REASONS[recipe]
    try:
        domain.reason(name)
    except KeyError:
        raise RecipeUnavailable(
            f"recipe {recipe!r} labels its escalations with reason {name!r}, "
            f"which domain {domain.name!r} does not declare"
        ) from None
    return DECLINE_PREFIX + name


def _table_head(domain: Domain) -> str:
    return f"Operations:\n{operation_table_text(domain)}\n\n"


def generator_system(domain: Domain) -> str:
    persona = domain.persona or "a user of the assistant"
    return (
        f"You write realistic test requests, as typed by {persona}, for an assistant that can "
        "only act through the operations listed below. Write varied wording: sometimes informal, "
        "sometimes terse, sometimes a full sentence. Never copy an operation's name or "
        "description word for word. Reply with JSON only."
    )


def strict_reviewer_system(domain: Domain) -> str:
    topics = ", ".join(domain.explain_topics)
    scope = f" (topics such as: {topics})" if topics else ""
    return (
        f"{reviewer_intro(domain)} Answering in words is only for knowledge "
        f"questions in this domain{scope}; any other text that is not a request in this domain "
        "(greetings, thanks, jokes, farewells, questions or remarks about the assistant itself) "
        "is handed off, not answered. Given the operation table and a request, judge whether the "
        "stated handling is exactly correct. Answer strictly 'yes' or 'no', then a short reason."
    )


def review_prompt(
    text: str, expect: dict[str, Any], cls: str | None, domain: Domain
) -> tuple[str, str]:
    user = (
        f"Operation table:\n{operation_table_text(domain)}\n\nRequest: {text}\n\n"
        f"Is this the correct handling: {expect_words(expect, cls, domain)}? "
        "Answer yes or no, then a short reason."
    )
    return strict_reviewer_system(domain), user


# ---------------------------------------------------------------------------
# missing-argument (rule-based)
# ---------------------------------------------------------------------------

_SPAN_ARTICLES = ("the", "a", "an", "my", "this", "that", "our")

#: Vague references by argument kind; ``{noun}`` is the argument's noun. A free-text
#: argument loses its value to a definite-but-unnamed reference, a choice
#: argument to an unnamed alternative. Keyed by the domain contract's ARG_KINDS.
VAGUE_REFS: dict[str, tuple[str, ...]] = {
    "str": ("the {noun}", "that {noun}", "it"),
    "choice": ("a different {noun}", "another {noun}", "the other {noun}"),
}
if set(VAGUE_REFS) != set(ARG_KINDS):  # guards a contract change
    raise RuntimeError("VAGUE_REFS must have one entry per argument kind in ARG_KINDS")


def arg_noun(arg: ArgSpec, domain: Domain) -> str:
    """What a request calls *arg*: its ground kind's noun, else its own name."""
    kind = domain.ground_kind(arg.ground) if arg.ground else None
    return (kind.noun if kind is not None else arg.name).replace("_", " ").lower()


def span_nouns(domain: Domain, arg: ArgSpec) -> tuple[str, ...]:
    """Words that may follow an argument value and belong to its span ("the
    kitchen room", "reading scene"): every ground kind's noun and plural and
    every argument's name in the domain, plus this argument's own noun.
    Longest first, so the widest span wins."""
    nouns = {arg_noun(arg, domain), arg.name.replace("_", " ").lower()}
    for kind in domain.ground_kinds:
        nouns.update((kind.noun.lower(), kind.plural.lower()))
    for op in domain.operations:
        nouns.update(a.name.replace("_", " ").lower() for a in op.args)
    return tuple(sorted((n for n in nouns if n), key=lambda n: (-len(n), n)))


def value_forms(value: str) -> list[str]:
    """How a request may spell *value*: verbatim, ``_``/``.``/``-`` as spaces
    or hyphens, and a dotted name's bare stem ("vllm" for "vllm.service").
    Longest first, so the widest span wins."""
    low = value.lower()
    forms = {low, low.replace("_", " "), low.replace("_", "-"), low.replace(".", " ")}
    if "." in low:
        stem = low.split(".", 1)[0]
        if len(stem) >= 3:
            forms.add(stem)
            forms.add(stem.replace("_", " "))
    return sorted((f for f in forms if f.strip()), key=lambda f: (-len(f), f))


def _span_re(arg: ArgSpec, value: str, domain: Domain) -> re.Pattern[str]:
    forms = "|".join(re.escape(f) for f in value_forms(value))
    nouns = "|".join(re.escape(n) for n in span_nouns(domain, arg))
    articles = "|".join(_SPAN_ARTICLES)
    noun_tail = rf"(?:\s+(?:{nouns}))?" if nouns else ""
    return re.compile(
        rf"(?<![\w.-])(?:(?:{articles})\s+)?(?:{forms}){noun_tail}(?![\w.-])",
        re.IGNORECASE,
    )


def _names_value(text: str, value: str) -> bool:
    return any(
        re.search(rf"(?<![\w.-]){re.escape(form)}(?![\w.-])", text, re.IGNORECASE)
        for form in value_forms(value)
    )


def strip_argument(
    text: str, arg: ArgSpec, value: str, pick: int, domain: Domain
) -> tuple[str | None, str]:
    """``(stripped text, "")``, or ``(None, reason)`` when *text* cannot be
    stripped: ``no_argument_span`` (the value is not spelled in it),
    ``too_short`` (nothing but the value), ``value_remains`` (a spelling
    survived). *pick* selects the vague reference, so the result is a pure
    function of its inputs."""
    pattern = _span_re(arg, value, domain)
    if not pattern.search(text):
        return None, "no_argument_span"
    if not re.search(r"\w", pattern.sub(" ", text)):
        return None, "too_short"
    refs = VAGUE_REFS["choice" if arg.kind == "choice" else "str"]
    ref = refs[pick % len(refs)].format(noun=arg_noun(arg, domain))

    def _replace(found: re.Match[str]) -> str:
        if found.start() == 0 and text[:1].isupper():
            return ref[:1].upper() + ref[1:]
        return ref

    stripped = " ".join(pattern.sub(_replace, text).split())
    if _names_value(stripped, value):
        return None, "value_remains"
    return stripped, ""


def marg_unspecified_prompt(text: str, noun: str, domain: Domain) -> tuple[str, str]:
    user = (
        f"{_table_head(domain)}Request: {text}\n\n"
        f"The action this request asks for needs a {noun} to act on. Does the request "
        f"{MARG_UNSPECIFIED_MARKER} {noun} unspecified, so that the assistant cannot tell "
        f"from the request alone which {noun} is meant? Answer yes or no, then a short reason."
    )
    return strict_reviewer_system(domain), user


def marg_natural_prompt(text: str, domain: Domain) -> tuple[str, str]:
    user = (
        f"Request: {text}\n\n"
        f"Is this a {MARG_NATURAL_MARKER} that a real user might type to the assistant? "
        "Answer yes or no, then a short reason."
    )
    return strict_reviewer_system(domain), user


def _rng(key: str) -> random.Random:
    return random.Random(key)  # nosec B311 - seeded and deterministic, not security


def _strippable_by_op(
    train_entries: list[dict[str, Any]], domain: Domain
) -> dict[str, list[dict[str, Any]]]:
    """Valid explicit train entries of every operation that takes an argument, by op name."""
    by_op: dict[str, list[dict[str, Any]]] = {}
    for entry in train_entries:
        expect = entry.get("expect") or {}
        op = domain.get(str(expect.get("operation")))
        if entry.get("kind") != "explicit" or op is None or not op.args:
            continue
        if domain.validate_args(op.name, expect.get("args", {})) is not None:
            continue
        by_op.setdefault(op.name, []).append(entry)
    return by_op


def _strip_first_argument(
    entry: dict[str, Any], op: Operation, pick: int, domain: Domain
) -> tuple[str | None, str, ArgSpec | None]:
    """``(stripped, "", arg)`` for the first of *op*'s arguments that strips from the
    entry text, else ``(None, reason, None)`` with the last argument's reason."""
    reason = "no_argument_span"
    for arg in op.args:
        stripped, reason = strip_argument(
            entry["text"], arg, entry["expect"]["args"][arg.name], pick, domain
        )
        if stripped is not None:
            return stripped, reason, arg
    return None, reason, None


def _missing_argument_unit(
    entry: dict[str, Any], stripped: str, arg: ArgSpec, cls: str, domain: Domain
) -> Unit:
    noun = arg_noun(arg, domain)
    item = Item(
        text=stripped,
        expect={"escalate": True},
        cls=cls,
        reviews=[
            marg_unspecified_prompt(stripped, noun, domain),
            marg_natural_prompt(stripped, domain),
        ],
        extra={"pair_of": str(entry["id"]), "stripped_arg": arg.name},
    )
    group = str(entry.get("source_id") or entry["id"])
    return Unit("missing-argument", [item], group=group)


def missing_argument_units(
    train_entries: list[dict[str, Any]],
    per_op: int,
    seed: int,
    rejects: dict[str, int],
    domain: Domain,
) -> list[Unit]:
    """Up to *per_op* stripped requests per operation that takes an argument,
    drawn from *train_entries* in an order shuffled by *seed*; deterministic."""
    cls = decline_class(domain, "missing-argument")
    by_op = _strippable_by_op(train_entries, domain)
    units: list[Unit] = []
    for op in domain.operations:
        pool = sorted(by_op.get(op.name, []), key=lambda e: str(e["id"]))
        _rng(f"{seed}:missing-argument:{op.name}").shuffle(pool)
        made = 0
        for entry in pool:
            if made >= per_op:
                break
            pick = _rng(f"{seed}:missing-argument:{entry['id']}").randrange(1 << 16)
            stripped, reason, arg = _strip_first_argument(entry, op, pick, domain)
            if stripped is None or arg is None:
                bump(rejects, reason)
                continue
            made += 1
            units.append(_missing_argument_unit(entry, stripped, arg, cls, domain))
    return units


# ---------------------------------------------------------------------------
# teacher-drafted recipes: prompts
# ---------------------------------------------------------------------------

_DX_ASK = (
    "Write {k} {marker} of user messages about {subject}. Both messages of a pair are about "
    'the same topic X and worded closely. "explain" is a definitional question -- what X means '
    "or how X works in general -- that is answered in one sentence without looking at the "
    'user\'s own situation; "answer" is that one-sentence correct answer. "diagnose" asks why '
    "X is happening to the user right now, or asks to fix it -- that needs investigation, so the "
    "assistant must hand it back to a human instead of answering. Do not ask for the hand-off in "
    "so many words. Never write an operation identifier from the list: {rule}. Reply as a "
    'JSON list of objects with keys "topic", "explain", "answer" and "diagnose".'
)

_CHOICE_ASK = (
    "Write {k} different user requests that {marker} the {arg} of the operation {name} to "
    "{value!r}. Each request names that {arg} clearly, but in its own way: a synonym, an "
    "abbreviation, a vendor or tool term, or a plain description -- vary them. Each is a clear, "
    "direct request, never a question about what the {arg} is. Reply as a JSON list of "
    'objects with keys "text" and "args" (an object).'
)

_DISAMBIGUATION_ASK = (
    "Write {k} different user requests that should be handled by the operation {name}, but "
    "whose wording {marker} for another operation in the list (for example because it shares "
    "a keyword with it). For each, give the exact arguments {name} needs (use only allowed "
    "choices; for a free-text argument use a concrete value that appears in the request) and "
    "the other operation it could be confused with. Reply as a JSON list of objects with keys "
    '"text", "args" (an object) and "confusable" (the other operation\'s name).'
)

_HARD_NEGATIVE_ASK = (
    "Write {k} different user questions that mention what the operation {name} deals with "
    "{marker}, but want an explanation in words -- what something means or how it works -- "
    "and not for the operation to be run or anything to be checked or changed. "
    "Never write {name} or any other operation identifier from the list: {rule}. "
    'Reply as a JSON list of objects with keys "text" and "answer" (a one-sentence correct '
    "answer)."
)

_CTC_ASK = (
    "Write {k} {marker} of user messages about the operation {name}, which changes state. "
    '"check" asks only to look at one thing that a read-only operation from the list '
    'answers; give that operation\'s name as "operation" and its arguments as "args". '
    '"conditional" asks to look at that same thing and then, depending on what it shows, to do '
    "what {name} does (for example: check X, and if it is over some limit, change Y). A "
    "conditional needs several steps with a decision between them, so the assistant must hand "
    "it back to a human instead of making the change; do not ask for the hand-off in so many "
    "words. Never write an operation identifier from the list in either message: {rule}. Reply "
    'as a JSON list with ONE object per pair, each carrying all four keys, like {{"check": "<the '
    'check alone>", "operation": "<a read-only operation from the list>", "args": {{}}, '
    '"conditional": "<the same check, then the change>"}}.'
)


def _subject(domain: Domain) -> str:
    if domain.explain_topics:
        return f"{domain.description or 'this domain'}, such as: {'; '.join(domain.explain_topics)}"
    return domain.description or domain.name


def check_then_change_prompt(domain: Domain, op_name: str, k: int) -> tuple[str, str]:
    ask = _CTC_ASK.format(k=k, marker=CTC_MARKER, name=op_name, rule=NAME_THE_SUBJECT)
    return generator_system(domain), _table_head(domain) + ask


def dx_prompt(domain: Domain, k: int) -> tuple[str, str]:
    ask = _DX_ASK.format(k=k, marker=DX_MARKER, subject=_subject(domain), rule=NAME_THE_SUBJECT)
    return generator_system(domain), _table_head(domain) + ask


def choice_prompt(
    domain: Domain, op_name: str, arg_name: str, value: str, k: int
) -> tuple[str, str]:
    ask = _CHOICE_ASK.format(k=k, marker=CHOICE_MARKER, arg=arg_name, name=op_name, value=value)
    return generator_system(domain), _table_head(domain) + ask


def disambiguation_prompt(domain: Domain, op_name: str, k: int) -> tuple[str, str]:
    ask = _DISAMBIGUATION_ASK.format(k=k, marker=DISAMBIGUATION_MARKER, name=op_name)
    return generator_system(domain), _table_head(domain) + ask


def hard_negative_prompt(domain: Domain, op_name: str, k: int) -> tuple[str, str]:
    ask = _HARD_NEGATIVE_ASK.format(
        k=k, marker=HARD_NEGATIVE_MARKER, name=op_name, rule=NAME_THE_SUBJECT
    )
    return generator_system(domain), _table_head(domain) + ask


def most_natural_prompt(
    text: str, expect: dict[str, Any], confusable: str, domain: Domain
) -> tuple[str, str]:
    other = expect_words({"operation": confusable, "args": {}}, None, domain)
    user = (
        f"{_table_head(domain)}Request: {text}\n\n"
        f"Is this the correct handling, and the {MOST_NATURAL_MARKER} of the request rather "
        f"than '{other}': {expect_words(expect, None, domain)}? Answer yes or no, then a "
        "short reason."
    )
    return strict_reviewer_system(domain), user


def choice_targets(domain: Domain) -> list[tuple[Operation, ArgSpec, str]]:
    """``(operation, argument, choice)`` for every choice value of every choice
    argument of every operation in the domain."""
    return [
        (op, arg, value)
        for op in domain.operations
        for arg in op.args
        if arg.kind == "choice"
        for value in arg.choices
    ]


def _rounds(k: int) -> list[int]:
    """Batch sizes that ask for *k* items in total, at most :data:`BATCH` each."""
    return [min(BATCH, k - start) for start in range(0, max(k, 0), BATCH)]


def _ask(
    client: TeacherClient,
    prompt: Callable[[int], tuple[str, str]],
    k: int,
    rejects: dict[str, int],
) -> list[dict[str, Any]]:
    """Every object the generator returns over :func:`_rounds` of *k*, capped at
    *k*; a failed call (after the client's retries) counts ``error`` and is
    skipped."""
    items: list[dict[str, Any]] = []
    rounds = _rounds(k)
    for index, size in enumerate(rounds):
        system, user = prompt(size)
        if len(rounds) > 1:
            user += (
                f"\n\n(Batch {index + 1} of {len(rounds)}: write requests different from "
                "those of an earlier batch.)"
            )
        outcome = client.complete("generator", system, user, list_check)
        if outcome.status != "ok":
            bump(rejects, "error")
            continue
        items.extend(i for i in parse_json_list(outcome.text) if isinstance(i, dict))
    return items[:k]


def _text(item: dict[str, Any], key: str = "text") -> str:
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# teacher-drafted recipes: units
# ---------------------------------------------------------------------------


def dx_units(domain: Domain, client: TeacherClient, k: int, rejects: dict[str, int]) -> list[Unit]:
    diagnosis = decline_class(domain, "diagnosis-explain")
    units = []
    for item in _ask(client, lambda size: dx_prompt(domain, size), k, rejects):
        explain, answer, diagnose = (
            _text(item, "explain"),
            _text(item, "answer"),
            _text(item, "diagnose"),
        )
        if not (explain and answer and diagnose):
            bump(rejects, "invalid_item")
            continue
        explain_expect = {"explain": True, "answer": answer}
        escalate_expect = {"escalate": True}
        units.append(
            Unit(
                "diagnosis-explain",
                [
                    Item(
                        explain,
                        explain_expect,
                        EXPLAIN_CLASS,
                        [review_prompt(explain, explain_expect, None, domain)],
                    ),
                    Item(
                        diagnose,
                        escalate_expect,
                        diagnosis,
                        [review_prompt(diagnose, escalate_expect, diagnosis, domain)],
                    ),
                ],
            )
        )
    return units


def choice_units(
    domain: Domain, client: TeacherClient, k: int, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    for op, arg, value in choice_targets(domain):

        def prompt(size: int, op=op, arg=arg, value=value) -> tuple[str, str]:
            return choice_prompt(domain, op.name, arg.name, value, size)

        for item in _ask(client, prompt, k, rejects):
            text, args = _text(item), item.get("args")
            if not text or domain.validate_args(op.name, args) is not None:
                bump(rejects, "invalid_args")
                continue
            if args.get(arg.name) != value:
                bump(rejects, "wrong_value")
                continue
            expect = {"operation": op.name, "args": dict(args)}
            review = review_prompt(text, expect, None, domain)
            units.append(Unit("power-set", [Item(text, expect, None, [review])]))
    return units


def disambiguation_units(
    domain: Domain, client: TeacherClient, k: int, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    known = set(domain.names())
    for op in domain.operations:

        def prompt(size: int, op=op) -> tuple[str, str]:
            return disambiguation_prompt(domain, op.name, size)

        for item in _ask(client, prompt, k, rejects):
            text, args = _text(item), item.get("args")
            if not text or domain.validate_args(op.name, args) is not None:
                bump(rejects, "invalid_args")
                continue
            confusable = item.get("confusable")
            if confusable not in known or confusable == op.name:
                bump(rejects, "invalid_confusable")
                continue
            expect = {"operation": op.name, "args": dict(args)}
            review = most_natural_prompt(text, expect, confusable, domain)
            units.append(
                Unit(
                    "disambiguation",
                    [Item(text, expect, None, [review], extra={"confusable": confusable})],
                )
            )
    return units


def hard_negative_units(
    domain: Domain, client: TeacherClient, k: int, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    for op in domain.operations:

        def prompt(size: int, op=op) -> tuple[str, str]:
            return hard_negative_prompt(domain, op.name, size)

        for item in _ask(client, prompt, k, rejects):
            text, answer = _text(item), _text(item, "answer")
            if not text or not answer:
                bump(rejects, "invalid_item")
                continue
            expect = {"explain": True, "answer": answer}
            review = review_prompt(text, expect, None, domain)
            units.append(
                Unit(
                    "hard-negative",
                    [Item(text, expect, EXPLAIN_CLASS, [review], extra={"mentions": op.name})],
                )
            )
    return units


def check_then_change_units(
    domain: Domain, client: TeacherClient, k: int, rejects: dict[str, int]
) -> list[Unit]:
    """A read-only check paired with the same check followed by a conditional
    change, which escalates as ``decline:multi_step``."""
    multi_step = decline_class(domain, "check-then-change")
    units = []
    for op in (op for op in domain.operations if not op.read_only):

        def prompt(size: int, op=op) -> tuple[str, str]:
            return check_then_change_prompt(domain, op.name, size)

        for item in _ask(client, prompt, k, rejects):
            check, conditional = _text(item, "check"), _text(item, "conditional")
            name, args = item.get("operation"), item.get("args")
            checked = domain.get(name) if isinstance(name, str) else None
            if (
                not (check and conditional)
                or checked is None
                or not checked.read_only
                or domain.validate_args(name, args) is not None
            ):
                bump(rejects, "invalid_item")
                continue
            check_expect = {"operation": name, "args": dict(args)}
            escalate_expect = {"escalate": True}
            units.append(
                Unit(
                    "check-then-change",
                    [
                        Item(
                            check,
                            check_expect,
                            None,
                            [review_prompt(check, check_expect, None, domain)],
                        ),
                        Item(
                            conditional,
                            escalate_expect,
                            multi_step,
                            [review_prompt(conditional, escalate_expect, multi_step, domain)],
                            extra={"changes": op.name},
                        ),
                    ],
                )
            )
    return units


_DRAFTERS = {
    "check-then-change": check_then_change_units,
    "diagnosis-explain": dx_units,
    "power-set": choice_units,
    "disambiguation": disambiguation_units,
    "hard-negative": hard_negative_units,
}


def _units_per_round(recipe: str, domain: Domain) -> int:
    """Generator prompts one round of a recipe makes."""
    if recipe == "diagnosis-explain":
        return 1
    if recipe == "power-set":
        return len(choice_targets(domain))
    if recipe == "check-then-change":
        return sum(1 for op in domain.operations if not op.read_only)
    return len(domain.operations)


def planned_teacher_calls(recipe: str, k: int, domain: Domain) -> int:
    """Generator calls a recipe makes for ``per_recipe`` *k* (no parse retries)."""
    if recipe == "missing-argument":
        return 0
    return len(_rounds(k)) * _units_per_round(recipe, domain)


def planned_candidates(recipe: str, k: int, domain: Domain) -> int:
    """Items a recipe asks the generator for, at most (a pair counts two)."""
    if recipe == "missing-argument":
        return 0
    per_item = 2 if recipe in ("diagnosis-explain", "check-then-change") else 1
    return max(k, 0) * _units_per_round(recipe, domain) * per_item


# ---------------------------------------------------------------------------
# guards, dedupe, review
# ---------------------------------------------------------------------------


def guard_reason(text: str, domain: Domain) -> str | None:
    """The augment guards, reused: the first that fires."""
    if names_internal_operation(text, domain):
        return "identifier"
    if copies_answer_template(text):
        return "template"
    if asks_for_handoff(text):
        return "handoff"
    return None


class _Dedupe:
    """Exact repeats of train texts and of earlier kept texts; exact or
    near-duplicate repeats of a protected (``exclude``) text."""

    def __init__(self, train_texts: list[str], protected_texts: list[str]):
        self.train = {_normal(t) for t in train_texts}
        self.kept: set[str] = set()
        self.protected = protected_texts

    def reason(self, unit: Unit) -> str | None:
        keys = [_normal(item.text) for item in unit.items]
        if len(set(keys)) != len(keys):
            return "duplicate"
        for item, key in zip(unit.items, keys):
            if key in self.train:
                return "train_exact"
            if key in self.kept:
                return "duplicate"
            kind = duplicates_against(item.text, self.protected) if self.protected else None
            if kind == "exact":
                return "protected_exact"
            if kind == "near-duplicate":
                return "protected_near"
        return None

    def keep(self, unit: Unit) -> None:
        self.kept.update(_normal(item.text) for item in unit.items)


def review_unit(
    unit: Unit, client: TeacherClient, decide_by: str
) -> tuple[bool, list[dict[str, Any]]]:
    """Ask every review prompt of every item, clean slate; stop at the first
    reject. Reviewer B always decides; with *decide_by* ``both`` reviewer A must
    also say yes. A failed call raises :class:`CandidateError`."""
    votes: list[dict[str, Any]] = []
    for item in unit.items:
        for system, user in item.reviews:
            accept_b, reason_b = reviewer_verdict(client, "reviewer_b", system, user)
            vote: dict[str, Any] = {"reviewer_b": {"accept": accept_b, "reason": reason_b}}
            ok = accept_b
            if decide_by == "both":
                accept_a, reason_a = reviewer_verdict(client, "reviewer_a", system, user)
                vote["reviewer_a"] = {"accept": accept_a, "reason": reason_a}
                ok = ok and accept_a
            votes.append(vote)
            if not ok:
                return False, votes
    return True, votes


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_train(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A train-side split file; the held-out split and any other side are refused."""
    path = Path(path)
    if path.name == HELD_OUT_NAME:
        raise ValueError(f"{path}: the held-out split is never a training input")
    doc = json.loads(path.read_text(encoding="utf-8"))
    header = doc.get("header") if isinstance(doc, dict) else None
    header = header if isinstance(header, str) else ""
    if header.casefold().startswith(HELD_OUT_MARKER):
        raise ValueError(f"{path}: the held-out split is never a training input")
    if not _TRAIN_HEADER.search(header):
        raise ValueError(f"{path}: its header does not name the train side")
    return doc, list(doc.get("entries", []))


def load_texts(path: Path) -> list[str]:
    """Every entry text of a split or corpus file (``{entries}``, a list, or JSONL)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        doc = json.loads(raw)
        records = doc.get("entries", []) if isinstance(doc, dict) else doc
    return [r["text"] for r in records if isinstance(r, dict) and isinstance(r.get("text"), str)]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def check_recipes(recipes: tuple[str, ...]) -> None:
    unknown = [r for r in recipes if r not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s) {unknown}; choose from {', '.join(RECIPES)}")


def plan(
    train_path: Path, recipes: tuple[str, ...], per_recipe: int, seed: int, domain: Domain
) -> dict[str, dict[str, int]]:
    """What a run would attempt, calling nothing: generator calls and the most
    candidates per recipe (``missing-argument``: the stripped requests, exactly)."""
    check_recipes(recipes)
    _, entries = load_train(train_path)
    out: dict[str, dict[str, int]] = {}
    for recipe in recipes:
        if recipe in RECIPE_REASONS:
            decline_class(domain, recipe)
        if recipe == "missing-argument":
            rejects: dict[str, int] = {}
            units = missing_argument_units(entries, per_recipe, seed, rejects, domain)
            out[recipe] = {"teacher_calls": 0, "candidates": len(units), **rejects}
        else:
            out[recipe] = {
                "teacher_calls": planned_teacher_calls(recipe, per_recipe, domain),
                "candidates": planned_candidates(recipe, per_recipe, domain),
            }
    return out


def _check_run_inputs(recipes: tuple[str, ...], decide_by: str, domain: Domain) -> None:
    check_recipes(recipes)
    if decide_by not in DECIDE_BY_RULES:
        raise ValueError(f"decide_by must be one of {DECIDE_BY_RULES}, got {decide_by!r}")
    for recipe in recipes:
        if recipe in RECIPE_REASONS:
            decline_class(domain, recipe)  # fail before any call


@dataclass
class _RunContext:
    """What every recipe of one :func:`run` shares."""

    domain: Domain
    client: TeacherClient
    decide_by: str
    dedupe: _Dedupe


@dataclass
class _RecipeTally:
    """One recipe's running entry, pair and kept-unit numbers."""

    recipe: str
    tag: str
    n_entry: int = 0
    n_pair: int = 0
    units_kept: int = 0

    def keep(self, unit: Unit) -> tuple[list[str | None], list[dict[str, Any]]]:
        """Number a kept *unit*: its entry ids and the entries themselves."""
        self.units_kept += 1
        group = unit.group
        if group is None and len(unit.items) > 1:
            self.n_pair += 1
            group = f"{ID_PREFIX}-{self.tag}-p{self.n_pair:04d}"
        ids: list[str | None] = []
        entries: list[dict[str, Any]] = []
        for item in unit.items:
            self.n_entry += 1
            entry_id = f"{ID_PREFIX}-{self.tag}-{self.n_entry:04d}"
            ids.append(entry_id)
            entries.append(self._entry(item, entry_id, group))
        return ids, entries

    def _entry(self, item: Item, entry_id: str, group: str | None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": entry_id,
            "kind": "explicit",
            "text": item.text,
            "expect": item.expect,
            "source": f"{ID_PREFIX}-{self.recipe}",
            "source_id": group or entry_id,
        }
        if item.cls:
            entry["class"] = item.cls
        entry.update(item.extra)
        return entry


def _recipe_units(
    recipe: str,
    train_entries: list[dict[str, Any]],
    per_recipe: int,
    seed: int,
    rejects: dict[str, int],
    ctx: _RunContext,
) -> list[Unit]:
    if recipe == "missing-argument":
        return missing_argument_units(train_entries, per_recipe, seed, rejects, ctx.domain)
    return _DRAFTERS[recipe](ctx.domain, ctx.client, per_recipe, rejects)


def _guard_or_dedupe_reason(unit: Unit, ctx: _RunContext) -> str | None:
    guarded = (guard_reason(item.text, ctx.domain) for item in unit.items)
    return next((r for r in guarded if r), None) or ctx.dedupe.reason(unit)


def _review_reason(
    unit: Unit, ctx: _RunContext, rejects: dict[str, int]
) -> tuple[str | None, list[dict[str, Any]]]:
    """``(None, votes)`` when the reviewers accept *unit*, else ``(reason, votes)``;
    a reviewer that said no is counted in *rejects*."""
    try:
        accepted, votes = review_unit(unit, ctx.client, ctx.decide_by)
    except CandidateError:
        return "error", []
    if accepted:
        return None, votes
    for who in ("reviewer_a", "reviewer_b"):
        if who in votes[-1] and not votes[-1][who]["accept"]:
            bump(rejects, who)
    return "reviewer", votes


def _judge_unit(
    unit: Unit, ctx: _RunContext, rejects: dict[str, int]
) -> tuple[str | None, list[dict[str, Any]]]:
    """Guard, dedupe, then review *unit*: ``(None, votes)`` if kept, else the reason."""
    reason = _guard_or_dedupe_reason(unit, ctx)
    votes: list[dict[str, Any]] = []
    if reason is None:
        reason, votes = _review_reason(unit, ctx, rejects)
    if reason is not None and reason != "reviewer":
        bump(rejects, reason)
    return reason, votes


def _run_recipe(
    tally: _RecipeTally,
    units: list[Unit],
    ctx: _RunContext,
    rejects: dict[str, int],
    entries: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Judge every unit of the tally's recipe, appending kept entries and every
    review row; returns the recipe's counts."""
    for unit in units:
        reason, votes = _judge_unit(unit, ctx, rejects)
        ids: list[str | None] = [None] * len(unit.items)
        if reason is None:
            ctx.dedupe.keep(unit)
            ids, kept = tally.keep(unit)
            entries.extend(kept)
        review_rows.append(
            {
                "recipe": tally.recipe,
                "ids": ids,
                "texts": [item.text for item in unit.items],
                "accepted": reason is None,
                "reason": reason,
                "votes": votes,
            }
        )
    return {"kept": tally.n_entry, "units_kept": tally.units_kept, "rejected": rejects}


def _write_outputs(
    out: Path,
    doc: dict[str, Any],
    review_out: Path | None,
    review_rows: list[dict[str, Any]],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if review_out is None:
        return
    review_out = Path(review_out)
    review_out.parent.mkdir(parents=True, exist_ok=True)
    review_out.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )


def run(
    train_path: Path,
    out: Path,
    recipes: tuple[str, ...],
    per_recipe: int,
    seed: int,
    domain: Domain,
    client: TeacherClient,
    exclude: list[Path] | tuple[Path, ...] = (),
    decide_by: str = "reviewer_b",
    review_out: Path | None = None,
) -> dict[str, Any]:
    """Draft, guard, dedupe and review every recipe; write the supplement to *out*
    (and every verdict to *review_out*); return counts and the sha256."""
    _check_run_inputs(recipes, decide_by, domain)
    train_path, out = Path(train_path), Path(out)
    _, train_entries = load_train(train_path)
    protected = [text for path in exclude for text in load_texts(Path(path))]
    dedupe = _Dedupe([str(e.get("text", "")) for e in train_entries], protected)
    ctx = _RunContext(domain, client, decide_by, dedupe)
    entries: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        tally = _RecipeTally(recipe, ID_TAGS[recipe])
        rejects: dict[str, int] = {}
        units = _recipe_units(recipe, train_entries, per_recipe, seed, rejects, ctx)
        counts[recipe] = _run_recipe(tally, units, ctx, rejects, entries, review_rows)

    sha256 = sha256_of_entries(entries)
    meta = {
        "tool": "jev_factory.data.targeted",
        "train": {"name": train_path.name, "sha256": _sha256_file(train_path)},
        "exclude": [{"name": Path(p).name, "sha256": _sha256_file(Path(p))} for p in exclude],
        "domain": domain.name,
        "recipes": list(recipes),
        "per_recipe": per_recipe,
        "seed": seed,
        "decide_by": decide_by,
        "teachers": {name: role.record() for name, role in sorted(client.roles.items())},
        "counts": counts,
        "sha256": sha256,
    }
    header = (
        f"Split 'train' of {train_path.name}: a train-only targeted supplement written by "
        'jev_factory.data.targeted; its recipes, seed, counts and sha256 are under "targeted".'
    )
    doc = {"header": header, "targeted": meta, "entries": entries}
    _write_outputs(out, doc, review_out, review_rows)
    return {"out": str(out), "kept": len(entries), "recipes": counts, "sha256": sha256}
