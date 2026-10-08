"""Draft fresh evaluation pools and the sealed held-out (teachers and a non-teacher).

Two drafters share one prompt vocabulary, every word of which comes from the
:class:`~jev_factory.domain.model.Domain` (persona, phrasing styles, explain
topics, the escalate reasons and their definitions, the answer policy, each
operation's ``read_only`` flag):

``run_draft``
    The **eval pool**. The ``generator`` teacher drafts requests from the
    operation table only (never the seed corpus or any split): ``per_op`` per
    operation with exact arguments checked by ``Domain.validate_args``,
    ``per_reason`` per escalate reason, and ``explain`` knowledge questions.
    Every candidate that survives the exact and near-duplicate check is judged
    by ``reviewer_a`` and ``reviewer_b`` (JSON yes/no verdicts through the
    :class:`~jev_factory.data.teachers.TeacherClient`); only entries both
    accept are kept. A reviewer that cannot give a verdict is an **error**,
    counted and never kept, but never counted as a reject either.

``draft_heldout`` and ``run_review``
    The **sealed held-out**. A model that is *not* one of the teachers, run in
    this process (Qwen3.5-4B in nvsh; loaded lazily, and injected as a plain
    callable so tests use a fake), sees only the operation table. The draft is
    written with a ``Held-out split`` header, a unique ``ho<seed>-`` id prefix
    and mode 0o444, is never overwritten, and only counts and a sha256 are
    ever returned or reported. ``run_review`` then applies the same two
    reviewers and the dedupe check to such a file without a human reading it.

Requests are asked for in **small batches** (``MAX_BATCH`` items): a
60-item explain request never parsed in nvsh issue 53. Nothing here prints
entry text; the sealed paths never put it in a message, an error or a report.
Stdlib only at import time; ``transformers`` is imported inside
:func:`load_qwen_drafter`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core.split import HELD_OUT_NAME, expectation_kind, load_sealed, sha256_file
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
from jev_factory.data.teachers import (
    RoleConfig,
    TeacherClient,
    TeacherError,
    chat_payload,
    load_teacher_list,
)
from jev_factory.domain.model import Domain
from jev_factory.factory.detach import ItemLedger

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/draft_sources.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 1-120 ROLES/NVSH_DRAFT_* env loading and ConfigError replaced by the"
        " TeacherClient roles (generator, reviewer_a, reviewer_b) built from the run config",
        "lines 79-99 REASON_DEFINITIONS become Domain.reason_definitions(); the operation"
        " table, persona, phrasing styles, explain topics and answer policy are read from"
        " the Domain instead of nvsh.ops.table and hard-coded Jetson prose",
        "lines 120-205 call_seed, make_seeded_caller and the seeded HTTP post kept, but"
        " reusing teachers.chat_payload and the TeacherClient caller shape",
        "lines 240-275 _generate_list retry loop replaced by TeacherClient.complete with a"
        " parse check, so a reply that is not a JSON list is retried and never cached",
        "generator asks are split into batches of at most MAX_BATCH items (a 60-item"
        " explain request never parsed in issue 53)",
        "lines 380-470 dedupe uses core.leakage.match (shingles and word Jaccard) instead"
        " of importing leakage_check by sys.path",
        "lines 505-590 reviewer hedge-word allowance and empty-reply retry dropped: verdicts"
        " are JSON and an unusable reply is an error outcome, counted apart from rejects",
        "lines 600-660 ids carry the seed (s<seed>-eval-<kind>-<n>) so pools drafted with"
        " different seeds never collide; the heldout pool is no longer teacher-drafted",
        "lines 700-800 run_review reads the domain seed corpus for dedupe and writes a"
        " read-only Held-out split file; the argparse CLI is left to the jev run verbs",
        "added ItemLedger item-level resume for generator batches and judgments",
        "also ports scripts/lfm-finetune/draft_heldout.py (the second upstream, same commit):"
        " lines 27-60 table_text and the SYSTEM/_OP/_ESCALATE/_EXPLAIN asks are built from"
        " the Domain (persona, reasons, explain topics, read_only flags)",
        "draft_heldout.py lines 96-110 as_item and parse_json_list kept; --seed/--issue argv"
        " parsing became function arguments",
        "draft_heldout.py lines 136-218 model load and generation moved into load_qwen_drafter"
        " (lazy transformers import) behind an injected drafter callable; asks are batched",
        "draft_heldout.py lines 192-218 output is a Held-out split header, a per-seed id"
        " prefix, mode 0o444, only counts and sha256 are returned; raw-generations log dropped",
    ],
    "licence": "Apache-2.0",
}

POOL = "eval"
MAX_BATCH = 8
"""Most items asked for in one generator request."""
HELD_OUT_HEADER = "Held-out split"
"""A sealed draft's header opens with this phrase, as nvsh's held-out corpus does."""
HELD_OUT_MODEL = "Qwen/Qwen3.5-4B"
HELD_OUT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HELD_OUT_TEMPERATURE = 0.7
READ_ONLY_LABEL = "read-only"
MUTATING_LABEL = "changes state"

Drafter = Callable[[str, str], str]
"""``(system, user) -> raw reply``: the in-process non-teacher held-out drafter."""
RawSeededCaller = Callable[[RoleConfig, str, str, int], str]


class HeldOutRefused(ValueError):
    """Raised when a path named like the sealed held-out corpus is opened."""


#: The operation table as a prompt (kept under draft's historical name).
table_text = operation_table_text


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


# ---------------------------------------------------------------------------
# per-call request seeding (a run seed reaches the HTTP request body)
# ---------------------------------------------------------------------------


def call_seed(run_seed: int, role: str, prompt_key: str, attempt: int) -> int:
    """A deterministic non-negative per-call seed for ``(run_seed, role, prompt, attempt)``.

    Reproducible only on an endpoint that honours the request ``"seed"``.
    """
    digest = hashlib.sha256(f"{run_seed}:{role}:{prompt_key}:{attempt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _prompt_key(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()


def post_seeded(role: RoleConfig, system: str, user: str, seed_value: int) -> str:
    """The gateway call ``teachers`` makes, plus a top-level ``"seed"`` in the body."""
    payload = chat_payload(role, system, user)
    payload["seed"] = seed_value
    headers = {"Content-Type": "application/json"}
    if role.key:
        headers["Authorization"] = f"Bearer {role.key.reveal()}"
    request = urllib.request.Request(  # noqa: S310 - scheme checked by check_endpoint
        role.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=role.timeout) as response:  # nosec B310
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices or "message" not in choices[0]:
        raise TeacherError("malformed gateway reply: no choices")
    return (choices[0]["message"].get("content") or "").strip()


def make_seeded_caller(
    run_seed: int, raw_caller: RawSeededCaller = post_seeded
) -> Callable[[RoleConfig, str, str], str]:
    """A ``TeacherClient`` caller that derives each call's seed from *run_seed*.

    A prompt repeated in one run (a parse retry, two reviewer votes) is tracked
    by an attempt counter, so repeats get distinct but reproducible seeds.
    """
    attempts: dict[tuple[str, str], int] = {}

    def _caller(role: RoleConfig, system: str, user: str) -> str:
        prompt_key = _prompt_key(system, user)
        key = (role.role, prompt_key)
        attempt = attempts.get(key, 0)
        attempts[key] = attempt + 1
        return raw_caller(role, system, user, call_seed(run_seed, role.role, prompt_key, attempt))

    return _caller


# ---------------------------------------------------------------------------
# prompts, all read from the Domain
# ---------------------------------------------------------------------------


def generator_system(domain: Domain) -> str:
    """The generator's system prompt: the domain's persona and phrasing styles."""
    who = domain.persona or "a user of the assistant"
    styles = (
        f" Vary the wording: {'; '.join(domain.phrasing_styles)}." if domain.phrasing_styles else ""
    )
    return (
        f"You write realistic test requests as {who} would type them to an assistant. The "
        "assistant can only act through the operations listed below."
        f"{styles} Sometimes informal, sometimes with typos. Never copy an operation's name "
        "or description word for word. Reply with JSON only."
    )


_OP_ASK = (
    "Write {k} different user requests that should be handled by the operation {name}. "
    "For each, give the exact arguments it needs (use only allowed choices; for a free-text "
    "argument use a concrete realistic value that appears in the request). Reply as a JSON "
    'list of objects with keys "text" and "args" (an object).'
)
_DECLINE_ASK = (
    "Write {k} different user requests that the assistant must NOT handle with any of these "
    "operations and must hand back to a human instead, because {definition}. Do not ask for the "
    "hand-off in so many words -- write it the way a real user would phrase the request itself. "
    'Reply as a JSON list of objects with a single key "text".'
)
_EXPLAIN_ASK = (
    "Write {k} different user questions that should be answered in words, without running "
    "anything: {topics}. Reply as a JSON list of objects with keys "
    '"text" and "answer" (a one-sentence correct answer).'
)
_BATCH_NOTE = " (Batch {i} of {n}: make these clearly different from the other batches.)"


def _explain_topics(domain: Domain) -> str:
    if not domain.explain_topics:
        return "general knowledge questions about this domain"
    return "questions about " + ", ".join(domain.explain_topics)


def _head(domain: Domain) -> str:
    return f"Operations:\n{operation_table_text(domain)}\n\n"


def _with_batch(ask: str, index: int, total: int) -> str:
    return ask + (_BATCH_NOTE.format(i=index, n=total) if total > 1 else "")


def batch_sizes(k: int, max_batch: int = MAX_BATCH) -> list[int]:
    """Split a request for *k* items into batches of at most *max_batch*."""
    if k <= 0:
        return []
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    full, rest = divmod(k, max_batch)
    return [max_batch] * full + ([rest] if rest else [])


def op_prompt(domain: Domain, op_name: str, k: int, index: int = 1, total: int = 1):
    ask = _OP_ASK.format(k=k, name=op_name)
    return generator_system(domain), _head(domain) + _with_batch(ask, index, total)


def decline_prompt(domain: Domain, reason: str, k: int, index: int = 1, total: int = 1):
    definition = domain.reason_definitions()[reason]
    ask = _DECLINE_ASK.format(k=k, definition=definition)
    return generator_system(domain), _head(domain) + _with_batch(ask, index, total)


def explain_prompt(domain: Domain, k: int, index: int = 1, total: int = 1):
    ask = _EXPLAIN_ASK.format(k=k, topics=_explain_topics(domain))
    return generator_system(domain), _head(domain) + _with_batch(ask, index, total)


def as_item(item: Any) -> dict[str, Any]:
    """A reply item as a dict: a bare string is a request with no other fields."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"text": item}
    return {}


# ---------------------------------------------------------------------------
# reasons
# ---------------------------------------------------------------------------


def split_reasons(text: str | None) -> list[str] | None:
    """A comma-separated ``--only-reasons`` value as a list, or ``None`` when unset."""
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def check_reasons(domain: Domain, only: Sequence[str] | None) -> tuple[str, ...]:
    """The escalate reasons a draft covers: *only* (validated, domain order) or all."""
    every = tuple(r.name for r in domain.reasons)
    if only is None:
        return every
    unknown = sorted(set(only) - set(every))
    if unknown or not only:
        raise ValueError(f"unknown decline reason(s) {unknown}; choose from {', '.join(every)}")
    return tuple(name for name in every if name in set(only))


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class Candidate:
    """One drafted entry before ids are assigned, plus the ``kind_tag`` its id is built from."""

    __slots__ = ("kind_tag", "text", "expect", "cls")

    def __init__(self, kind_tag: str, text: str, expect: dict[str, Any], cls: str | None = None):
        self.kind_tag = kind_tag
        self.text = text
        self.expect = expect
        self.cls = cls


Step = Callable[[str, Callable[[], Any]], Any]


def _generate_items(
    client: TeacherClient,
    system: str,
    user: str,
    rejects: dict[str, int],
    step: Step,
    key: str,
) -> list[dict[str, Any]]:
    """One generator batch as a list of item dicts; an unusable reply is counted, not raised."""

    def ask() -> list[dict[str, Any]]:
        outcome = client.complete("generator", system, user, parse=list_check)
        if outcome.status != "ok":
            return [{"__error__": True}]
        return [as_item(item) for item in parse_json_list(outcome.text)]

    items = step(key, ask)
    if items == [{"__error__": True}]:
        bump(rejects, "generator_error")
        return []
    return items


def _op_candidates(
    domain: Domain,
    client: TeacherClient,
    k: int,
    rejects: dict[str, int],
    step: Step,
    max_batch: int = MAX_BATCH,
) -> list[Candidate]:
    out: list[Candidate] = []
    sizes = batch_sizes(k, max_batch)
    for op in domain.operations:
        for n, size in enumerate(sizes, 1):
            system, user = op_prompt(domain, op.name, size, n, len(sizes))
            for item in _generate_items(
                client, system, user, rejects, step, f"gen:op:{op.name}:{n}"
            ):
                text = str(item.get("text", "")).strip()
                args = item.get("args") or {}
                if not text or not isinstance(args, dict):
                    bump(rejects, "invalid_args")
                    continue
                if domain.validate_args(op.name, args) is not None:
                    bump(rejects, "invalid_args")
                    continue
                out.append(Candidate(f"op-{op.name}", text, {"operation": op.name, "args": args}))
    return out


def _decline_candidates(
    domain: Domain,
    client: TeacherClient,
    k: int,
    rejects: dict[str, int],
    step: Step,
    reasons: Sequence[str],
    max_batch: int = MAX_BATCH,
) -> list[Candidate]:
    out: list[Candidate] = []
    sizes = batch_sizes(k, max_batch)
    for reason in reasons:
        for n, size in enumerate(sizes, 1):
            system, user = decline_prompt(domain, reason, size, n, len(sizes))
            key = f"gen:decline:{reason}:{n}"
            for item in _generate_items(client, system, user, rejects, step, key):
                text = str(item.get("text", "")).strip()
                if not text:
                    bump(rejects, "invalid_args")
                    continue
                cand = Candidate(f"decline-{reason}", text, {"escalate": True}, f"decline:{reason}")
                out.append(cand)
    return out


def _explain_candidates(
    domain: Domain,
    client: TeacherClient,
    k: int,
    rejects: dict[str, int],
    step: Step,
    max_batch: int = MAX_BATCH,
) -> list[Candidate]:
    out: list[Candidate] = []
    sizes = batch_sizes(k, max_batch)
    for n, size in enumerate(sizes, 1):
        system, user = explain_prompt(domain, size, n, len(sizes))
        for item in _generate_items(client, system, user, rejects, step, f"gen:explain:{n}"):
            text = str(item.get("text", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not text or not answer:
                bump(rejects, "invalid_args")
                continue
            out.append(Candidate("explain", text, {"explain": True, "answer": answer}))
    return out


def planned_batches(
    domain: Domain,
    per_op: int,
    per_reason: int,
    explain: int,
    reasons: Sequence[str],
    max_batch: int,
) -> int:
    """How many generator requests a draft makes (the ledger's total)."""
    return (
        len(domain.operations) * len(batch_sizes(per_op, max_batch))
        + len(reasons) * len(batch_sizes(per_reason, max_batch))
        + len(batch_sizes(explain, max_batch))
    )


# ---------------------------------------------------------------------------
# dedupe (exact + near-duplicate, via core.leakage.match)
# ---------------------------------------------------------------------------


def load_dev_texts(domain: Domain) -> list[str]:
    """The domain's seed-corpus request texts, what a draft must not repeat."""
    return [
        e["text"]
        for e in domain.load_seed_corpus().entries
        if isinstance(e.get("text"), str) and e["text"]
    ]


def dedupe_candidates(
    candidates: list[Candidate], dev_texts: Sequence[str], rejects: dict[str, int]
) -> list[Candidate]:
    """Drop a candidate that exact-or-near-duplicates the seed corpus or an earlier one."""
    kept: list[Candidate] = []
    kept_texts: list[str] = []
    for candidate in candidates:
        kind = duplicates_against(candidate.text, dev_texts) or duplicates_against(
            candidate.text, kept_texts
        )
        if kind == "exact":
            bump(rejects, "dedupe_exact")
            continue
        if kind == "near-duplicate":
            bump(rejects, "dedupe_near")
            continue
        kept.append(candidate)
        kept_texts.append(candidate.text)
    return kept


# ---------------------------------------------------------------------------
# reviewer (strict yes/no, both must accept)
# ---------------------------------------------------------------------------


def reviewer_system(domain: Domain) -> str:
    """The reviewers' system prompt: the domain's answer policy, then a strict yes/no ask."""
    return (
        f"{reviewer_intro(domain)} Given the operation table and a request, "
        "judge whether the stated handling is exactly correct."
    )


def reviewer_prompt(
    domain: Domain, text: str, expect: dict[str, Any], cls: str | None = None
) -> tuple[str, str]:
    user = (
        f"Operation table:\n{operation_table_text(domain)}\n\n"
        f"Request: {text}\n\n"
        f"Is this the correct handling: {expect_words(expect, cls, domain)}?"
    )
    return reviewer_system(domain), user


def judge(
    domain: Domain, text: str, expect: dict[str, Any], cls: str | None, client: TeacherClient
) -> dict[str, Any]:
    """Ask both reviewers independently.

    ``accepted`` is true only when both said yes; ``errored`` is true when a
    reviewer gave no usable verdict (never a reject); ``votes`` records both.
    """
    system, user = reviewer_prompt(domain, text, expect, cls)
    votes: dict[str, dict[str, Any]] = {}
    for role in ("reviewer_a", "reviewer_b"):
        outcome = client.review(role, system, user)
        votes[role] = {"accept": outcome.accepted, "reason": outcome.reason or outcome.error}
    accepts = [v["accept"] for v in votes.values()]
    return {
        "accepted": all(a is True for a in accepts),
        "errored": any(a is None for a in accepts),
        "votes": votes,
    }


def _tally_rejects(outcome: dict[str, Any], rejects: dict[str, int]) -> None:
    votes = outcome["votes"]
    for role in ("reviewer_a", "reviewer_b"):
        if votes[role]["accept"] is False:
            bump(rejects, role)
        elif votes[role]["accept"] is None:
            bump(rejects, f"{role}_error")


# ---------------------------------------------------------------------------
# ids, output
# ---------------------------------------------------------------------------


def _make_id(seed: int, kind_tag: str, n: int) -> str:
    return f"s{seed}-{POOL}-{kind_tag}-{n:03d}"


def _entry_from_candidate(entry_id: str, candidate: Candidate) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": entry_id,
        "kind": "explicit",
        "text": candidate.text,
        "expect": candidate.expect,
        "source": f"draft-{POOL}",
        "source_id": entry_id,
    }
    if candidate.cls:
        entry["class"] = candidate.cls
    return entry


def _by_kind(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        expect = entry.get("expect") or {}
        try:
            kind = expectation_kind(expect)
        except (AttributeError, TypeError):
            kind = "unknown"
        bump(counts, kind or "unknown")
    return counts


def _review_lines(rows: list[dict[str, Any]], strip_text: bool) -> str:
    lines = []
    for row in rows:
        row = dict(row)
        if strip_text:
            row.pop("text", None)
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)


def write_sealed(path: Path, doc: dict[str, Any]) -> str:
    """Write a sealed draft read-only (0o444) and return its file sha256.

    Refuses to overwrite an existing file: a sealed draft is written once.
    """
    path = Path(path)
    if path.exists():
        raise _err(
            f"refusing to overwrite an existing sealed draft: {path.name}",
            "a sealed file is written once; choose a new output directory",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.chmod(0o444)
    os.replace(tmp, path)
    return sha256_file(path)


def refuse_held_out(path: Path) -> None:
    """The sealed corpus file itself is never an input here (by its name)."""
    if Path(path).name == HELD_OUT_NAME:
        raise HeldOutRefused(f"{path}: the sealed held-out corpus is never read by this tool")


def _stepper(jobdir: Path | None, name: str, total: int) -> Step:
    """A resumable step runner: journaled through an ``ItemLedger`` when a jobdir is given."""
    if jobdir is None:
        return lambda key, fn: fn()
    ledger = ItemLedger(Path(jobdir), name, total)

    def step(key: str, fn: Callable[[], Any]) -> Any:
        if key not in ledger.done:
            ledger.record(key, fn())
        return ledger.done[key]

    return step


# ---------------------------------------------------------------------------
# eval pool
# ---------------------------------------------------------------------------


DRAFT_FILE = "draft.json"


def run_draft(
    domain: Domain,
    out_dir: Path,
    seed: int,
    per_op: int,
    per_reason: int,
    explain: int,
    client: TeacherClient,
    *,
    dev_texts: Sequence[str] | None = None,
    only_reasons: Sequence[str] | None = None,
    max_batch: int = MAX_BATCH,
    jobdir: Path | None = None,
) -> dict[str, Any]:
    """Draft, dedupe and two-reviewer-check a fresh eval pool; returns counts and a sha256."""
    reasons = check_reasons(domain, only_reasons)
    random.seed(seed)
    texts = list(dev_texts) if dev_texts is not None else load_dev_texts(domain)

    rejects: dict[str, int] = {}
    gen_step = _stepper(
        jobdir,
        "draft-generate",
        planned_batches(domain, per_op, per_reason, explain, reasons, max_batch),
    )
    candidates: list[Candidate] = []
    candidates += _op_candidates(domain, client, per_op, rejects, gen_step, max_batch)
    candidates += _decline_candidates(
        domain, client, per_reason, rejects, gen_step, reasons, max_batch
    )
    candidates += _explain_candidates(domain, client, explain, rejects, gen_step, max_batch)
    candidates = dedupe_candidates(candidates, texts, rejects)

    judge_step = _stepper(jobdir, "draft-review", len(candidates))
    counters: dict[str, int] = {}
    rejected: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        outcome = judge_step(
            f"judge:{index}",
            lambda c=candidate: judge(domain, c.text, c.expect, c.cls, client),
        )
        if outcome["accepted"]:
            bump(counters, candidate.kind_tag)
            entry_id = _make_id(seed, candidate.kind_tag, counters[candidate.kind_tag])
            entries.append(_entry_from_candidate(entry_id, candidate))
        else:
            _tally_rejects(outcome, rejects)
            label = "error" if outcome["errored"] else "rejected"
            bump(rejected, f"{candidate.kind_tag}-{label}")
            entry_id = (
                f"s{seed}-{POOL}-{candidate.kind_tag}-{label}-"
                f"{rejected[f'{candidate.kind_tag}-{label}']:03d}"
            )
        rows.append({"id": entry_id, "text": candidate.text, "votes": outcome["votes"]})

    by_reason: dict[str, int] = {}
    for entry in entries:
        if entry.get("class", "").startswith("decline:"):
            bump(by_reason, entry["class"].removeprefix("decline:"))
    by_kind = _by_kind(entries)
    sha256 = sha256_of_entries(entries)
    header = {
        "tool": "draft.run_draft",
        "domain": domain.name,
        "surface_sha256": domain.surface_sha256(),
        "pool": POOL,
        "seed": seed,
        "per_op": per_op,
        "per_reason": per_reason,
        "reasons": list(reasons),
        "explain": explain,
        "max_batch": max_batch,
        "models": {role: cfg.model for role, cfg in client.roles.items()},
        "sampling": {
            "per_call_seed": True,
            "note": (
                "reproducible only on endpoints that honour the request seed; a "
                "gateway/engine change can still alter outputs"
            ),
        },
        "counts": {"kept": len(entries), "by_kind": by_kind, "by_reason": by_reason, **rejects},
        "sha256": sha256,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"header": header, "entries": entries}
    (out_dir / DRAFT_FILE).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "review.jsonl").write_text(_review_lines(rows, strip_text=False), encoding="utf-8")
    return {
        "pool": POOL,
        "seed": seed,
        "kept": len(entries),
        "by_kind": by_kind,
        "by_reason": by_reason,
        "rejects": rejects,
        "sha256": sha256,
    }


# ---------------------------------------------------------------------------
# sealed held-out: a non-teacher model, in process
# ---------------------------------------------------------------------------


def check_non_teacher(model: str, listed: dict[str, dict[str, str]] | None = None) -> None:
    """Refuse a held-out drafter that is one of the teachers (by alias, name or id tail)."""
    listed = listed if listed is not None else load_teacher_list()

    def norm(value: str) -> str:
        return value.strip().lower().rsplit("/", 1)[-1]

    names = set()
    for alias, info in listed.items():
        names.add(norm(alias))
        names.add(norm(info.get("name", alias)))
    if norm(model) in names:
        raise _err(
            f"held-out drafter {model!r} is one of the teachers",
            "the sealed held-out must be drafted by a model that is not a pipeline teacher",
        )


def held_out_header(model: str, snapshot: str, seed: int, domain: Domain) -> str:
    """The sealed draft's header; it opens with ``Held-out split`` and names no entry."""
    return (
        f"{HELD_OUT_HEADER} drafted by {model} (snapshot {snapshot}, not a pipeline teacher) "
        f"from the {domain.name} operation table only, seed {seed}, temperature "
        f"{HELD_OUT_TEMPERATURE}, thinking off. Awaiting operator review; the lead agent has "
        "not read these entries."
    )


def held_out_id_prefix(seed: int) -> str:
    """Ids of a sealed draft start with this, unique per seed."""
    return f"ho{seed}-"


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower().rsplit("/", 1)[-1]).strip("-")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _held_out_asks(domain: Domain, per_op: int, escalate: int, explain: int, max_batch: int):
    """``(key, system, user)`` for every held-out request, in small batches."""
    system = generator_system(domain)
    head = _head(domain)
    out: list[tuple[str, str, str]] = []
    definitions = "; ".join(f"{d}" for d in domain.reason_definitions().values())
    escalate_ask = (
        "Write {k} different user requests that the assistant must NOT handle with any of "
        "these operations and must hand back to a human instead, such as requests where "
        '{defs}. Reply as a JSON list of objects with a single key "text".'
    )
    plans = (
        *((f"op:{op.name}", per_op, _OP_ASK, {"name": op.name}) for op in domain.operations),
        ("escalate", escalate, escalate_ask, {"defs": definitions}),
        ("explain", explain, _EXPLAIN_ASK, {"topics": _explain_topics(domain)}),
    )
    for key, count, template, fields in plans:
        sizes = batch_sizes(count, max_batch)
        for n, size in enumerate(sizes, 1):
            ask = _with_batch(template.format(k=size, **fields), n, len(sizes))
            out.append((f"{key}:{n}", system, head + ask))
    return out


def draft_heldout(
    domain: Domain,
    out_dir: Path,
    seed: int,
    drafter: Drafter,
    *,
    model: str = HELD_OUT_MODEL,
    snapshot: str = HELD_OUT_REVISION,
    per_op: int = 3,
    escalate: int = 16,
    explain: int = 16,
    max_batch: int = MAX_BATCH,
    dev_texts: Sequence[str] | None = None,
    teachers: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Draft the sealed held-out with a non-teacher *drafter* and write ``draft.json`` read-only.

    Returns counts and sha256 only (``entries``, ``by_kind``, ``rejects``,
    ``sha256``, ``path``): no entry text ever leaves this function.
    """
    check_non_teacher(model, teachers)
    random.seed(seed)
    texts = list(dev_texts) if dev_texts is not None else load_dev_texts(domain)
    dev_norm = {_normalize(t) for t in texts}
    seen: set[str] = set()
    rejects = {"parse": 0, "invalid_op_args": 0, "duplicate": 0}
    entries: list[dict[str, Any]] = []
    prefix = held_out_id_prefix(seed)
    source = f"heldout-draft-{_slug(model)}"
    for key, system, user in _held_out_asks(domain, per_op, escalate, explain, max_batch):
        kind = key.split(":")[0]
        try:
            items = parse_json_list(drafter(system, user))
        except Exception:  # noqa: BLE001 - any unusable reply is a counted parse failure
            rejects["parse"] += 1
            continue
        for item in map(as_item, items):
            text = str(item.get("text", "")).strip()
            norm = _normalize(text)
            if not text or norm in seen or norm in dev_norm:
                rejects["duplicate"] += 1
                continue
            if kind == "op":
                op, args = key.split(":")[1], item.get("args") or {}
                if domain.validate_args(op, args) is not None:
                    rejects["invalid_op_args"] += 1
                    continue
                expect: dict[str, Any] = {"operation": op, "args": args}
            elif kind == "escalate":
                expect = {"escalate": True}
            else:
                expect = {"explain": True, "answer": str(item.get("answer", "")).strip()}
            seen.add(norm)
            entries.append(
                {
                    "id": f"{prefix}{len(entries) + 1:03d}",
                    "kind": "explicit",
                    "text": text,
                    "expect": expect,
                    "source": source,
                }
            )
    doc = {"header": held_out_header(model, snapshot, seed, domain), "entries": entries}
    path = Path(out_dir) / DRAFT_FILE
    write_sealed(path, doc)
    return {**summarize_sealed(path), "snapshot": snapshot, "rejects": rejects}


def summarize_sealed(path: Path) -> dict[str, Any]:
    """Counts and the file sha256 of a sealed draft: the only thing ever reported."""
    side = load_sealed(Path(path))
    return {
        "path": side.path,
        "entries": side.count,
        "by_kind": dict(side.by_kind),
        "sha256": side.sha256,
    }


def load_qwen_drafter(
    seed: int,
    *,
    model: str = HELD_OUT_MODEL,
    revision: str = HELD_OUT_REVISION,
    device: str = "cuda",
    max_new_tokens: int = 2048,
) -> tuple[Drafter, str]:
    """Load the non-teacher drafter in this process; returns ``(drafter, snapshot)``.

    The heavy imports live here so the base install stays light. The model
    must already be in the local Hugging Face cache (no network is used).
    """
    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    torch.manual_seed(seed)
    snap = Path(snapshot_download(model, revision=revision, local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snap, revision=revision)  # nosec B615
    lm = AutoModelForCausalLM.from_pretrained(  # nosec B615
        snap, revision=revision, dtype=torch.bfloat16, device_map=device
    )

    def drafter(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        out = lm.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=HELD_OUT_TEMPERATURE,
            top_p=0.9,
        )
        return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)

    return drafter, snap.name


# ---------------------------------------------------------------------------
# review (post-hoc two-reviewer check of an existing draft file)
# ---------------------------------------------------------------------------


def run_review(
    domain: Domain,
    in_path: Path,
    out_dir: Path,
    client: TeacherClient,
    *,
    dev_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Dedupe and two-reviewer-check a ``{header, entries}`` draft; write the kept entries.

    The output is a sealed file: a ``Held-out split`` header, mode 0o444. The
    review log never carries text, and the returned summary is counts and a
    sha256 only.
    """
    in_path = Path(in_path)
    refuse_held_out(in_path)
    texts = list(dev_texts) if dev_texts is not None else load_dev_texts(domain)
    try:
        doc = json.loads(in_path.read_text(encoding="utf-8"))
        source_entries = list(doc.get("entries", []))
    except (OSError, ValueError, AttributeError):
        raise _err(f"{in_path.name}: not a readable draft file") from None

    rejects: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    kept_texts: list[str] = []
    rows: list[dict[str, Any]] = []
    for entry in source_entries:
        text = entry.get("text", "")
        kind = duplicates_against(text, texts) or duplicates_against(text, kept_texts)
        if kind == "exact":
            bump(rejects, "dedupe_exact")
            continue
        if kind == "near-duplicate":
            bump(rejects, "dedupe_near")
            continue
        outcome = judge(domain, text, entry.get("expect", {}), entry.get("class"), client)
        rows.append({"id": entry.get("id", ""), "votes": outcome["votes"]})
        if outcome["accepted"]:
            kept.append(entry)
            kept_texts.append(text)
        else:
            _tally_rejects(outcome, rejects)

    header = (
        f"{HELD_OUT_HEADER} reviewed by two teacher reviewers "
        f"({', '.join(cfg.model for role, cfg in client.roles.items() if role != 'generator')}); "
        f"kept {len(kept)} of {len(source_entries)} entries. Awaiting operator review; the "
        "lead agent has not read these entries."
    )
    out_dir = Path(out_dir)
    path = out_dir / DRAFT_FILE
    write_sealed(path, {"header": header, "entries": kept})
    (out_dir / "review.jsonl").write_text(_review_lines(rows, strip_text=True), encoding="utf-8")
    return {**summarize_sealed(path), "rejects": rejects}
