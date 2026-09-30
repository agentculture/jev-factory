"""Augmentation: generate, correct, double review (seed requests -> variations).

Given seed requests whose *answer is already fixed* (a split-side file of
``{"header", "entries"}``; each entry carries ``id``, ``text``, ``expect`` and
``source_id``) this module asks the teachers for ``per_source`` paraphrased
variations per seed without ever letting a model choose or drift the answer:

1. the **generator** is told nothing of the answer and asked only to rephrase
   the seed request in a given phrasing style;
2. the **corrector** fixes grammar and spelling only. nvsh had a fourth
   model role for this; :mod:`jev_factory.data.teachers` has three, so the
   correction is one free-text call to ``reviewer_b`` (the role that "judges,
   decides and corrects");
3. ``reviewer_a`` and ``reviewer_b`` each answer a JSON yes/no verdict to
   "does this request still mean exactly this answer?". With
   ``decide_by="both"`` both must say yes; with ``"reviewer_b"`` (the default)
   reviewer B alone decides and reviewer A is still asked and recorded. When
   the fixed answer is read-only, an explanation or an escalation, the
   reviewer is also asked whether the request could be read as asking for a
   change: a "could" is a reject, since a read-only or escalate answer must
   never drift into a mutating one.

Three deterministic guards run after the reviewers and reject whatever they
said: a request that names an operation identifier, one that copies the
answer-template wording, and one that asks for a hand-off in so many words.

Every prompt is built from the :class:`~jev_factory.domain.model.Domain`: its
operation table and read-only flags, persona, phrasing styles and answer
policy. A teacher call that fails or returns nothing usable is an **error**
(the variation is written nowhere, so a resume tries it again), never a
reject, because a reject throws a candidate away for good. The run is
resumable: an id already in either output file (or journaled in the optional
:class:`~jev_factory.factory.detach.ItemLedger`) is skipped.

This module is the library half; the ``jev run augment`` stage drives it.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.data.teachers import TeacherClient
from jev_factory.domain.model import Domain
from jev_factory.factory.detach import ItemLedger

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/augment.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 1-1925 imported; every nvsh.ops.table call becomes a Domain query"
        " (get/names/read_only) and the persona, phrasing styles and answer policy in the"
        " prompts (:232-296, :316-420) come from the Domain",
        "lines 103-701 NVSH_AUG_* role config, :702-870 chat client and retry/backoff and"
        " :925-975 parse_verdict dropped: calls and JSON verdicts go through"
        " jev_factory.data.teachers.TeacherClient",
        "the corrector (a fourth nvsh role) is one free-text reviewer_b call, since"
        " teachers.py has three roles",
        "the Track-A skills seed shape (Seed.seed_format/context/skill_names, _refuse_if_eval,"
        " REVIEWER_SYSTEM_SKILL, DETAILED_STYLES, names_skill_identifier) is dropped",
        "an empty or unparseable teacher reply is an error via TeacherClient (retried, then"
        " recorded), never a reject; the old _reviewer_verdict raise is the same rule",
        "decide_by defaults to reviewer_b (issue 46 d11), as teachers.py documents;"
        " the reviewer system text loses its 'start with yes or no' line because the client"
        " appends the JSON-verdict instruction",
        "rederive_clean_slate (:1437-1543, a one-off offline migration of outputs written under a"
        " superseded rule) and the argparse main are not ported; the stage engine is the CLI",
        "resume also reads an ItemLedger when one is given; progress lines are the ledger's",
    ],
    "licence": "Apache-2.0",
}

DECIDE_BY_RULES = ("both", "reviewer_b")
DEFAULT_DECIDE_BY = "reviewer_b"

#: split.py's own held-out guard, mirrored: the held-out split is for judging a
#: tuned model, never for generating training data from.
HELD_OUT_NAME = "held-out.json"
HELD_OUT_MARKER = "held-out split"

#: Used when a domain declares no phrasing styles of its own.
DEFAULT_PHRASING_STYLES = (
    "a short imperative",
    "a polite question",
    "a description of the symptom or situation, without saying what to do",
    "terse, a few words, like a note to self",
    "casual and conversational",
    "as part of a longer sentence that gives a reason",
)


class SeedRefused(ValueError):
    """A held-out file passed as a seed."""


class ConfigError(ValueError):
    """A run setting that cannot be used (an unknown rule, a missing side)."""


class CandidateError(Exception):
    """A teacher call for one candidate failed; the candidate is retried on resume."""


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    source_id: str
    side: str | None
    seed_text: str  # the request to rephrase
    expect: dict[str, Any]  # the entry's own, fixed expect block
    needs_change_check: bool  # ask the extra "could this be a change?" question
    #: The source entry's own corpus fields (``kind``, and ``source``/``class``
    #: when present), carried through unchanged so an accepted or rejected
    #: record is still a corpus entry.
    corpus_fields: dict[str, Any] = field(default_factory=dict)


#: split.py writes "Split 'train' of <file> (seed=N)." onto every side file;
#: matching it recognises a seed file renamed away from train/val/test.json.
_HEADER_SIDE_RE = re.compile(r"split\s+['\"](train|val|test)['\"]", re.IGNORECASE)


def _infer_side(path: Path, header: str | None = None) -> str | None:
    if path.stem in ("train", "val", "test"):
        return path.stem
    if header:
        match = _HEADER_SIDE_RE.search(header)
        if match:
            return match.group(1).lower()
    return None


def _resolve_side(path: Path, header: str | None, side: str | None) -> str:
    """Reconcile an explicit ``side`` with the side inferred from the file itself.

    ``side`` exists for a file whose own side cannot be told; when it *can* be
    inferred the two must agree, or the run refuses rather than relabel
    entries onto the wrong side.
    """
    inferred = _infer_side(path, header)
    if side is not None:
        if inferred is not None and side != inferred:
            raise ConfigError(
                f"{path}: side {side!r} conflicts with this file's own side {inferred!r}; "
                "side is only for files whose side cannot be inferred"
            )
        return side
    if inferred is None:
        raise ConfigError(
            f"{path}: cannot infer the split side from the filename or header; pass side"
        )
    return inferred


def needs_change_check(expect: dict[str, Any], domain: Domain) -> bool:
    """True when *expect*'s fixed answer is read-only, an explanation or an escalation.

    The reviewer must then also refuse any variation that could be read as
    asking for a change. An unrecognised operation counts as needing the
    guard rather than silently skipping it.
    """
    if expect.get("escalate") or "explain" in expect:
        return True
    operation = domain.get(str(expect.get("operation")))
    return True if operation is None else bool(operation.read_only)


def seed_from_entry(entry: dict[str, Any], side: str, domain: Domain) -> Seed:
    expect = entry["expect"]
    if "kind" not in entry:
        raise ValueError(
            f"{entry.get('id', entry.get('source_id', '?'))}: seed entry is missing "
            "its own corpus 'kind' (explicit/failure)"
        )
    corpus_fields: dict[str, Any] = {"kind": entry["kind"]}
    for key in ("source", "class"):
        if key in entry:
            corpus_fields[key] = entry[key]
    return Seed(
        source_id=entry.get("source_id", entry["id"]),
        side=side,
        seed_text=entry["text"],
        expect=expect,
        needs_change_check=needs_change_check(expect, domain),
        corpus_fields=corpus_fields,
    )


def load_seeds(path: Path, domain: Domain, side: str | None = None) -> list[Seed]:
    """Seeds from a split-side file (``{"header", "entries"}``, a bare list or JSONL).

    The held-out split is refused by name and by its header.
    """
    path = Path(path)
    if path.name == HELD_OUT_NAME:
        raise SeedRefused(f"{path}: the held-out split is never a seed")
    header: str | None = None
    if path.suffix == ".jsonl":
        records: list[Any] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "entries" in raw:
            records, header = raw["entries"], raw.get("header")
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(f"{path}: unrecognized seed file shape")
    if isinstance(header, str) and header.casefold().startswith(HELD_OUT_MARKER):
        raise SeedRefused(f"{path}: the held-out split is never a seed")
    resolved = _resolve_side(path, header, side)
    return [seed_from_entry(r, resolved, domain) for r in records]


# ---------------------------------------------------------------------------
# prompts (all built from the Domain)
# ---------------------------------------------------------------------------


def operation_table_text(domain: Domain) -> str:
    """The operation table as prompt text: name, description, effect, arguments."""
    lines = []
    for op in domain.operations:
        args = ", ".join(_arg_text(a) for a in op.args) or "no arguments"
        effect = "read-only" if op.read_only else "changes state"
        lines.append(f"- {op.name}: {op.description} [{effect}; arguments: {args}]")
    return "\n".join(lines)


def _arg_text(arg: Any) -> str:
    if arg.kind == "choice":
        return f"{arg.name} (one of: {'/'.join(arg.choices)})"
    return f"{arg.name} (free text)"


def capabilities(domain: Domain) -> str:
    """What the small assistant can do, from the operation table, in words.

    Without it both reviewers judged escalations against a capable general
    assistant ("that is a common task, no need to hand it off") and rejected
    every one. Built from the domain, never hard-coded.
    """
    checks = [op.description.rstrip(".") for op in domain.operations if op.read_only]
    changes = [op.description.rstrip(".") for op in domain.operations if not op.read_only]
    text = (
        "This assistant is small and can only do these things. Checks it can run "
        f"and report on: {'; '.join(checks) or 'none'}. Changes it can propose for the user "
        f"to approve: {'; '.join(changes) or 'none'}. It can also answer a general question "
        "in words. Anything else -- including anything that needs investigation, "
        "several steps, or a change not in that list -- must be passed on to a "
        "more capable assistant."
    )
    return f"{text} {domain.answer_policy}".rstrip() if domain.answer_policy else text


def answer_in_words(expect: dict[str, Any], domain: Domain) -> str:
    """The expected answer as a person would state it, not as a JSON block.

    Reviewers shown the raw block rejected every request that did not spell
    out the operation's identifier (a real user never says it), so the
    operation is named with its own description and arguments. It is
    described by what it does, never by identifier: a generator shown one
    copied it into the request.
    """
    if expect.get("escalate"):
        return (
            "pass it on to a more capable assistant, because it needs investigation"
            " or changes beyond a small fixed set of actions"
        )
    if expect.get("explain"):
        return f"reply in words, along the lines of: {expect.get('answer', '')}"
    operation = domain.get(str(expect.get("operation")))
    what = operation.description if operation is not None else "the expected action"
    what = what.rstrip(".").lower()
    values = ", ".join(
        f"{key} {str(value).replace('_', ' ')}"
        for key, value in sorted(expect.get("args", {}).items())
    )
    detail = f" -- {values}" if values else ""
    if operation is not None and operation.read_only:
        # A read-only operation answers the request by being run.
        return (
            f"run this read-only check (its output is shown to the user as the answer): "
            f"{what}{detail}"
        )
    return f"propose this change for the user to approve: {what}{detail}"


def phrasing_styles(domain: Domain) -> tuple[str, ...]:
    return domain.phrasing_styles or DEFAULT_PHRASING_STYLES


def generator_system(domain: Domain) -> str:
    persona = domain.persona or "a user of the assistant"
    return (
        "You rewrite user requests for a training dataset. You are given one "
        f"request typed by {persona}. Rewrite it in different words as that user "
        "might have typed it. Do not change what is being asked for. Reply with "
        "only the rewritten request, nothing else."
    )


CORRECTOR_SYSTEM = (
    "You copyedit short user requests for a training dataset. Fix grammar and "
    "spelling only; keep the user's wording, tone and meaning, and do not "
    "change what the request is asking for. Reply with only the corrected "
    "request, nothing else."
)


def reviewer_system(domain: Domain, change_check: bool = False) -> str:
    text = (
        "You are a strict reviewer for a training dataset. You are given a user "
        "request and the response an assistant should give to it. Answer "
        "strictly 'yes' or 'no' to whether that response is exactly the right "
        "one for the request -- not a different operation, different arguments, "
        "or a different kind of response -- then a short reason. Users never "
        "name internal operations or their argument identifiers, and never ask "
        "for a hand-off in so many words: judge what the request needs. "
        + capabilities(domain)
        + " Asking the user to approve a change before making it is always the "
        "right way to carry out a request for one of the listed changes. Running "
        "one of the listed checks is a complete answer to a request for what that "
        "check reports: its output is what the user then sees, so never reject a "
        "check for not already containing its results."
    )
    if change_check:
        text += (
            " The expected response here is not one of the listed changes. If the "
            "request could reasonably be carried out by one of the changes listed "
            "above, answer 'no' even if it otherwise matches."
        )
    return text


def generator_prompt(seed: Seed, domain: Domain, variation: int = 0) -> tuple[str, str]:
    """``(system, user)`` for the generator. *variation* picks the phrasing style,
    so the Nth variation of a request is asked for in a different register from
    the (N+1)th. The generator never sees the expected answer: shown it, it
    copied its wording into the request."""
    styles = phrasing_styles(domain)
    style = styles[variation % len(styles)]
    user = (
        f"Original request: {seed.seed_text}\n\n"
        "Rewrite the request above in different words, keeping exactly the same meaning"
        " and asking for exactly the same thing, no more and no less."
        f" Write it {style}. Do not name internal operations or their identifiers."
    )
    return generator_system(domain), user


def corrector_prompt(text: str) -> tuple[str, str]:
    # Like the generator, the corrector never sees the expected answer.
    return CORRECTOR_SYSTEM, f"Request to copyedit:\n{text}"


def reviewer_prompt(seed: Seed, text: str, domain: Domain) -> tuple[str, str]:
    system = reviewer_system(domain, change_check=seed.needs_change_check)
    user = (
        f"Response the assistant should give: {answer_in_words(seed.expect, domain)}\n\n"
        f"User request: {text}\n\n"
        "Is that response exactly the right one for this request? Answer 'yes' "
        "or 'no' and then a short reason."
    )
    return system, user


# ---------------------------------------------------------------------------
# deterministic guards
# ---------------------------------------------------------------------------

#: Wording that only appears in the expected-answer descriptions shown to the
#: corrector and reviewers. A request that carries it was copied from the
#: answer, not written as a user would, and is rejected whatever the reviewers say.
_ANSWER_TEMPLATE_RE = re.compile(
    r"take this action|propose this (?:action|change)|read-only check|report what it shows"
    r"|user to approve|more capable assistant|full agent"
    r"|hand (?:the|this) request|reply in words|\b\w+ = \S+",
    re.IGNORECASE,
)


def copies_answer_template(text: str) -> str:
    """The answer-template wording *text* copies, or "" if none."""
    match = _ANSWER_TEMPLATE_RE.search(text)
    return match.group(0) if match else ""


#: A request that asks for the hand-off in so many words: users never do.
#: Words only -- "escalating temperatures" is not a hand-off request.
_HANDOFF_RE = re.compile(
    r"\b(escalate (?:this|it|that|the (?:issue|problem|request|ticket))|"
    r"escalate to (?:a|an|the) (?:human|senior|support|more capable|person|expert)|"
    r"hand (?:it|this|that) (?:off|over)|hand-?off to (?:a|an|someone|another)|"
    r"(?:to|for) (?:a|an|the) (?:human|senior|more capable) "
    r"(?:agent|assistant|operator|engineer))\b",
    re.IGNORECASE,
)


def asks_for_handoff(text: str) -> str:
    """The hand-off wording *text* uses, or "" if none."""
    match = _HANDOFF_RE.search(text)
    return match.group(0) if match else ""


def names_internal_operation(text: str, domain: Domain) -> str:
    """The first operation identifier *text* names, or "" if none."""
    lowered = text.lower()
    for name in domain.names():
        if re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", lowered):
            return name
    return ""


def guard_verdicts(text: str, domain: Domain) -> dict[str, dict[str, Any]]:
    """Every deterministic guard that fires on *text*, as verdict entries."""
    found: dict[str, dict[str, Any]] = {}
    leaked = names_internal_operation(text, domain)
    if leaked:
        found["identifier_check"] = {"accept": False, "reason": f"names {leaked!r}"}
    copied = "" if leaked else copies_answer_template(text)
    if copied:
        found["template_check"] = {"accept": False, "reason": f"copies {copied!r}"}
    handoff = asks_for_handoff(text)
    if handoff:
        found["handoff_check"] = {"accept": False, "reason": f"asks for {handoff!r}"}
    return found


# ---------------------------------------------------------------------------
# teacher calls: errors, never rejects
# ---------------------------------------------------------------------------


def _generate_text(client: TeacherClient, role: str, system: str, user: str) -> str:
    outcome = client.complete(role, system, user)
    if outcome.status != "ok":
        raise CandidateError(f"{role}: {outcome.error or 'no usable reply'}")
    text = outcome.text.strip()
    if not text:
        raise CandidateError(f"empty reply from {role}")
    return text


def reviewer_verdict(client: TeacherClient, role: str, system: str, user: str) -> tuple[bool, str]:
    """One reviewer's JSON verdict; a failed call raises :class:`CandidateError`."""
    outcome = client.review(role, system, user)
    if outcome.status != "ok" or outcome.accepted is None:
        raise CandidateError(f"{role}: {outcome.error or 'no usable verdict'}")
    return bool(outcome.accepted), outcome.reason


def _roles_record(client: TeacherClient) -> dict[str, Any]:
    return {name: role.record() for name, role in client.roles.items()}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineCounts:
    generated: int = 0
    corrected: int = 0
    accepted: int = 0
    rejected_by_a: int = 0
    rejected_by_b: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "corrected": self.corrected,
            "accepted": self.accepted,
            "rejected_by_a": self.rejected_by_a,
            "rejected_by_b": self.rejected_by_b,
            "errors": self.errors,
        }


def _check_decide_by(decide_by: str) -> None:
    if decide_by not in DECIDE_BY_RULES:
        raise ValueError(
            f"decide_by must be one of {', '.join(DECIDE_BY_RULES)}, got {decide_by!r}"
        )


def _variation_number(variation_id: str) -> int:
    """The N in ``<source_id>~vN``; 0 when the id has no such suffix."""
    _, _, tail = variation_id.rpartition("~v")
    return int(tail) if tail.isdigit() else 0


def process_variation(
    seed: Seed,
    variation_id: str,
    domain: Domain,
    client: TeacherClient,
    counts: PipelineCounts,
    decide_by: str = DEFAULT_DECIDE_BY,
) -> dict[str, Any]:
    """Run one seed through generator -> corrector -> reviewers.

    Returns ``{"accepted": bool, "record": {...}}``; raises
    :class:`CandidateError` when a teacher call fails.
    """
    gen_system, gen_user = generator_prompt(seed, domain, _variation_number(variation_id))
    generated = _generate_text(client, "generator", gen_system, gen_user)
    counts.generated += 1

    cor_system, cor_user = corrector_prompt(generated)
    corrected = _generate_text(client, "reviewer_b", cor_system, cor_user)
    counts.corrected += 1

    rev_system, rev_user = reviewer_prompt(seed, corrected, domain)
    accept_a = reason_a = None
    if decide_by == "both" or "reviewer_a" in client.roles:
        accept_a, reason_a = reviewer_verdict(client, "reviewer_a", rev_system, rev_user)
    accept_b, reason_b = reviewer_verdict(client, "reviewer_b", rev_system, rev_user)
    if accept_a is False:
        counts.rejected_by_a += 1
    if not accept_b:
        counts.rejected_by_b += 1

    verdicts: dict[str, Any] = {"reviewer_b": {"accept": accept_b, "reason": reason_b}}
    if accept_a is not None:
        verdicts = {"reviewer_a": {"accept": accept_a, "reason": reason_a}, **verdicts}
    guards = guard_verdicts(corrected, domain)
    verdicts.update(guards)
    if decide_by == "reviewer_b":
        accepted = bool(accept_b) and not guards
    else:
        accepted = bool(accept_a) and bool(accept_b) and not guards

    record: dict[str, Any] = {
        "id": variation_id,
        "source_id": seed.source_id,
        "side": seed.side,
        "text": corrected,
        "expect": seed.expect,
        "teachers": _roles_record(client),
        "decided_by": decide_by,
    }
    record.update(seed.corpus_fields)
    # Every verdict is kept, so an overruled opinion stays auditable on an
    # accepted record too.
    record["verdicts"] = verdicts
    if accepted:
        counts.accepted += 1
    return {"accepted": accepted, "record": record}


def existing_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "id" in record:
                    ids.add(record["id"])
    return ids


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_seed_consistency(seeds: list[Seed]) -> None:
    """Refuse before any call if one ``source_id`` shows up with more than one
    side or expect block: two seeds that disagree about what their shared id
    means must never both be accepted under the same variation id."""
    seen: dict[str, Seed] = {}
    for seed in seeds:
        prior = seen.get(seed.source_id)
        if prior is None:
            seen[seed.source_id] = seed
            continue
        if prior.side != seed.side:
            raise ValueError(
                f"source_id {seed.source_id!r} appears with more than one side "
                f"({prior.side!r} and {seed.side!r})"
            )
        if prior.expect != seed.expect:
            raise ValueError(
                f"source_id {seed.source_id!r} appears with more than one expect block "
                f"({prior.expect!r} and {seed.expect!r})"
            )


def plan_tasks(
    seeds: list[Seed], per_source: int, limit: int | None, done: Iterable[str]
) -> list[tuple[Seed, str]]:
    """The ordered ``(seed, variation_id)`` work list, round-robin: variation 1 of
    every seed, then variation 2, and so on, so a run stopped part-way (or a
    *limit*) leaves every seed equally covered. Ids in *done* are skipped."""
    reserved = set(done)
    tasks: list[tuple[Seed, str]] = []
    for n in range(1, per_source + 1):
        for seed in seeds:
            if limit is not None and len(tasks) >= limit:
                return tasks
            variation_id = f"{seed.source_id}~v{n}"
            if variation_id in reserved:
                continue
            reserved.add(variation_id)
            tasks.append((seed, variation_id))
    return tasks


def run_augment(
    seed_files: list[Path],
    domain: Domain,
    client: TeacherClient,
    accepted_out: Path,
    rejected_out: Path,
    per_source: int,
    side: str | None = None,
    limit: int | None = None,
    workers: int = 2,
    decide_by: str = DEFAULT_DECIDE_BY,
    ledger: ItemLedger | None = None,
    dry_run: bool = False,
) -> PipelineCounts:
    """Augment every seed ``per_source`` times; resumable.

    Accepted variations go to *accepted_out* (one JSON object per line),
    rejected ones to *rejected_out* with every verdict and reason. A variation
    whose teacher call failed is written to neither, so the next run retries
    it. *dry_run* plans only: ``counts.generated`` is then the number of
    variations that would be attempted and no teacher is called.
    """
    _check_decide_by(decide_by)
    seeds: list[Seed] = []
    for seed_file in seed_files:
        seeds.extend(load_seeds(Path(seed_file), domain, side))
    validate_seed_consistency(seeds)
    accepted_out, rejected_out = Path(accepted_out), Path(rejected_out)
    done = existing_ids(accepted_out) | existing_ids(rejected_out)
    if ledger is not None:
        done |= set(ledger.done)
    tasks = plan_tasks(seeds, per_source, limit, done)
    counts = PipelineCounts()
    if dry_run:
        counts.generated = len(tasks)
        return counts
    if not tasks:
        return counts
    lock = threading.Lock()

    def run_one(seed: Seed, variation_id: str) -> None:
        local = PipelineCounts()
        try:
            outcome = process_variation(seed, variation_id, domain, client, local, decide_by)
        except CandidateError:
            with lock:
                counts.generated += local.generated
                counts.corrected += local.corrected
                counts.errors += 1
            return
        with lock:
            counts.generated += local.generated
            counts.corrected += local.corrected
            counts.accepted += local.accepted
            counts.rejected_by_a += local.rejected_by_a
            counts.rejected_by_b += local.rejected_by_b
            if variation_id in done:  # pragma: no cover - plan_tasks already dedupes
                return
            append_jsonl(accepted_out if outcome["accepted"] else rejected_out, outcome["record"])
            done.add(variation_id)
            if ledger is not None:
                ledger.record(variation_id, "accepted" if outcome["accepted"] else "rejected")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for future in [executor.submit(run_one, s, v) for s, v in tasks]:
            future.result()
    return counts


# ---------------------------------------------------------------------------
# re-review: reviewer B alone, clean slate
# ---------------------------------------------------------------------------


@dataclass
class RereviewCounts:
    """Counts for a re-review: only ``reviewer_b`` is ever called; ``compared`` and
    ``agreed`` track how the fresh verdict lines up with the stored one."""

    processed: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    compared: int = 0
    agreed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.errors,
            "compared": self.compared,
            "agreed": self.agreed,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prior_verdicts(record: dict[str, Any]) -> dict[str, Any]:
    """The verdicts a candidate carried before this re-review: history only, never
    an input to the decision. A record with none was stored as accepted."""
    verdicts = record.get("verdicts")
    return {"stored_as": "accepted"} if verdicts is None else dict(verdicts)


def _stored_reviewer_b_accept(record: dict[str, Any]) -> bool | None:
    verdicts = record.get("verdicts")
    if verdicts is None:
        return True
    reviewer_b = verdicts.get("reviewer_b") if isinstance(verdicts, dict) else None
    if not isinstance(reviewer_b, dict) or "accept" not in reviewer_b:
        return None
    return bool(reviewer_b["accept"])


def seed_from_stored_record(record: dict[str, Any], domain: Domain) -> Seed:
    """Enough of a :class:`Seed` from a stored record to build the reviewer prompt
    again. The generator and corrector are never re-run: the record's own
    ``text`` is the request under review."""
    expect = record["expect"]
    return Seed(
        source_id=str(record.get("source_id", record.get("id", ""))),
        side=record.get("side"),
        seed_text=record.get("text", ""),
        expect=expect,
        needs_change_check=needs_change_check(expect, domain),
        corpus_fields={k: record[k] for k in ("kind", "source", "class") if k in record},
    )


def rereview_candidate(
    record: dict[str, Any], domain: Domain, client: TeacherClient
) -> dict[str, Any]:
    """Re-review one stored candidate as a clean slate: only ``reviewer_b`` is called,
    on a prompt built from the text and the expected answer alone, and that fresh
    verdict plus the deterministic guards decide. Whatever was stored before moves
    to ``prior_verdicts`` as history."""
    old_accept_b = _stored_reviewer_b_accept(record)
    prior = _prior_verdicts(record)
    seed = seed_from_stored_record(record, domain)
    system, user = reviewer_prompt(seed, record["text"], domain)
    accept_b, reason_b = reviewer_verdict(client, "reviewer_b", system, user)
    guards = guard_verdicts(record["text"], domain)
    new_record = dict(record)
    new_record["teachers"] = {
        **dict(record.get("teachers", {})),
        "reviewer_b": client.roles["reviewer_b"].record(),
    }
    new_record["verdicts"] = {"reviewer_b": {"accept": accept_b, "reason": reason_b}, **guards}
    new_record["prior_verdicts"] = prior
    return {
        "accepted": accept_b and not guards,
        "record": new_record,
        "old_accept_b": old_accept_b,
        "new_accept_b": accept_b,
    }


def run_rereview(
    candidate_files: list[Path],
    domain: Domain,
    client: TeacherClient,
    accepted_out: Path,
    rejected_out: Path,
    limit: int | None = None,
    workers: int = 1,
) -> RereviewCounts:
    """Re-review stored accepted and rejected candidates with ``reviewer_b`` only.

    Resumable like :func:`run_augment`: an id already in an output file is
    skipped. *limit* caps how many candidates are attempted.
    """
    accepted_out, rejected_out = Path(accepted_out), Path(rejected_out)
    candidates = [r for path in candidate_files for r in load_jsonl(Path(path))]
    done = existing_ids(accepted_out) | existing_ids(rejected_out)
    tasks = [r for r in candidates if r.get("id") not in done]
    if limit is not None:
        tasks = tasks[:limit]
    counts = RereviewCounts()
    lock = threading.Lock()

    def run_one(record: dict[str, Any]) -> None:
        record_id = record.get("id", "?")
        try:
            outcome = rereview_candidate(record, domain, client)
        except (CandidateError, KeyError, ValueError):
            with lock:
                counts.errors += 1
            return
        with lock:
            if record_id in done:  # pragma: no cover - tasks already dedupes
                return
            counts.processed += 1
            if outcome["accepted"]:
                counts.accepted += 1
                append_jsonl(accepted_out, outcome["record"])
            else:
                counts.rejected += 1
                append_jsonl(rejected_out, outcome["record"])
            done.add(record_id)
            if outcome["old_accept_b"] is not None:
                counts.compared += 1
                counts.agreed += int(outcome["old_accept_b"] == outcome["new_accept_b"])

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for future in [executor.submit(run_one, r) for r in tasks]:
                future.result()
    return counts
