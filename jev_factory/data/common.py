"""Helpers shared by the data stages (``augment``, ``draft`` and ``targeted``).

All three build prompts from the same operation table, ask a reviewer to judge
a stated handling in words, parse a JSON list out of a teacher reply, refuse
repeats through :func:`jev_factory.core.leakage.match` and hash a list of
entries. They used to carry a copy each; this is the one home.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from jev_factory.core.leakage import match
from jev_factory.data.teachers import TeacherError
from jev_factory.domain.model import DECLINE_PREFIX, Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/augment.py",
    "commit": "9debdc6",
    "adaptations": [
        "new module: the operation-table text (augment.py:233-247), the expected-handling words"
        " and strict reviewer preamble (draft_sources.py / targeted_augment.py) and the JSON-list"
        " parse, duplicate check and entry hash that nvsh repeats in augment, draft and"
        " targeted-augment are defined once here and imported by the three data modules",
    ],
    "licence": "Apache-2.0",
}


def arg_text(arg: Any) -> str:
    """One argument as prompt text."""
    if arg.kind == "choice":
        return f"{arg.name} (one of: {'/'.join(arg.choices)})"
    return f"{arg.name} (free text)"


def operation_table_text(domain: Domain) -> str:
    """The operation table as prompt text: name, description, effect, arguments."""
    lines = []
    for op in domain.operations:
        args = ", ".join(arg_text(a) for a in op.args) or "no arguments"
        effect = "read-only" if op.read_only else "changes state"
        lines.append(f"- {op.name}: {op.description} [{effect}; arguments: {args}]")
    return "\n".join(lines)


def reviewer_intro(domain: Domain) -> str:
    """The opening every strict dataset reviewer prompt shares: role, task, answer policy."""
    policy = f" {domain.answer_policy}" if domain.answer_policy else ""
    return (
        "You are a strict reviewer for a training dataset that teaches a small assistant when to "
        "run one of a fixed set of operations, when to hand a request off to a human, and when to "
        f"just answer a question in words.{policy}"
    )


def expect_words(expect: dict[str, Any], cls: str | None, domain: Domain) -> str:
    """The stated handling as a reviewer is asked to judge it, in words."""
    if expect.get("escalate"):
        base = "no listed operation can safely handle this request, so it is escalated to a human"
        definition = domain.reason_definitions().get((cls or "").removeprefix(DECLINE_PREFIX))
        return f"{base}, because {definition}" if definition else base
    if expect.get("explain"):
        answer = expect.get("answer", "")
        return f"this is a knowledge question needing no action, correctly answered as: {answer}"
    op = domain.get(str(expect.get("operation")))
    what = op.description.rstrip(".").lower() if op is not None else "the expected action"
    values = ", ".join(f"{k} {v}" for k, v in sorted((expect.get("args") or {}).items()))
    detail = f" -- {values}" if values else ""
    verb = (
        "run this read-only check" if (op is not None and op.read_only) else "propose this change"
    )
    return f"{verb}: {what}{detail}"


#: A code fence opening a line (```` ``` ```` or ```` ```json ````) and one closing a line.
_FENCE_OPEN = re.compile(r"^```(?:json)?", re.M)
_FENCE_CLOSE = re.compile(r"```$", re.M)


def parse_json_list(raw: str) -> list[Any]:
    """The JSON list in a reply (a code fence or prose around it is fine)."""
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", raw.strip())).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        raise ValueError("no JSON list in the reply")
    items = json.loads(text[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("the reply is not a JSON list")
    return items


def list_check(text: str) -> tuple[bool, str]:
    """``TeacherClient`` parse hook: a reply holding no JSON list is an error the client
    retries (and never caches), not a usable draft."""
    try:
        parse_json_list(text)
    except ValueError:
        raise TeacherError("reply is not a JSON list") from None
    return True, ""


def duplicates_against(text: str, others: Sequence[str]) -> str | None:
    """``"exact"``, ``"near-duplicate"`` or ``None``, from ``core.leakage.match``."""
    for other in others:
        kind = match(text, other)
        if kind is not None:
            return kind
    return None


def sha256_of_entries(entries: list[dict[str, Any]]) -> str:
    """The sha256 of a list of entries, serialised the one canonical way."""
    body = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def bump(counts: dict[str, int], key: str, by: int = 1) -> None:
    """``counts[key] += by``."""
    counts[key] = counts.get(key, 0) + by
