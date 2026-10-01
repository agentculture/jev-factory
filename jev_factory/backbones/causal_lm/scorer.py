"""The causal-LM scorer adapter: lettered prompt, one next-token read, grounded arguments.

A causal decoder is turned into a jev-like scorer by listing every offered
candidate of a :class:`~jev_factory.domain.model.Domain` under its own
one-letter label and reading the next-token log-probabilities once. This
module is that adapter, and nothing about letters, token variants or the
distribution is defined here: :mod:`jev_factory.backbones.causal_lm.readout`
is the one definition and every path (served top-k, in-process logits) goes
through it.

What lives here:

* **the prompt** (:func:`prompt_messages`, :func:`render_prompt`): the
  domain's ``instruction``, then ``"Actions:"`` and one line
  ``"<L>) <name>: <description>"`` per offered candidate, then the request as
  the user turn. Operation text comes from the domain, ``escalate:<reason>``
  text from the domain's reasons, and the two controls from
  :data:`CONTROL_DESCRIPTIONS` (every description can be overridden, for
  paraphrase probes);
* **letters, order and subsets** (:func:`labels_for`, :func:`permute`,
  :class:`Permutation`): by default a candidate keeps its positional letter
  in the full pool when others are left out; :func:`permute` draws a seeded,
  JSON round-trippable order + letter map + optional subset;
* **scoring** (:func:`score`): through an injectable top-k function
  ``(prompt, top) -> {token text: logprob}``. :func:`served_top_k` builds
  one over an OpenAI-style ``/completions`` endpoint with an injectable
  ``post``; :class:`InProcessTopK` builds one over a local model's
  full-vocabulary log-probabilities;
* **grounding** (:func:`ground_arguments`): the model never generates an
  argument. A ``choice`` argument is matched against its declared spellings
  in the request's words; a grounded ``str`` argument is whichever request
  word the domain's :meth:`~jev_factory.domain.model.Domain.ground` accepts
  (sentence punctuation stripped, each lookup run once). Exactly one value
  must ground; none or several is reported, never guessed;
* **the predictions record** (:class:`ScorerPrediction`): backbone-agnostic
  fields (entry id, offered labels and names, raw scores, probabilities,
  outcome, operation, arguments, grounded) that other backbones emit too.

Heavy imports (torch) happen only inside :func:`transformers_logprobs`.
"""

from __future__ import annotations

import json
import math
import random
import re
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.domain.model import (
    ESCALATE,
    ESCALATE_PREFIX,
    EXPLAIN,
    Domain,
    GroundDecline,
    Operation,
)

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/scorer.py",
    "commit": "9debdc6",
    "adaptations": [
        "nvsh imports :82-86 (nvsh.ops ground/table/_model, nvsh.tiers bench/lfm) rewired to"
        " jev_factory.domain.model.Domain; letters, variants and the distribution come from"
        " jev_factory.backbones.causal_lm.readout and are never redefined here",
        "prompt :137-140 and :407-483 (_INSTRUCTION, _description, _label_map,"
        " prompt_messages, render_prompt): the instruction is Domain.instruction plus"
        " 'Actions:'; operation and escalate:<reason> text come from the Domain;"
        " explain/escalate text is CONTROL_DESCRIPTIONS (nvsh lfm.tools_for wording"
        " made domain-neutral); descriptions stay overridable",
        "candidates/labels_for/candidate_pool/positional_labels :189-255 take the Domain;"
        " permute/Permutation/same_choice :258-339 kept, pool now required",
        "ground_arguments :551-629: ops_ground.ground over a memoised runner becomes"
        " Domain.ground over a world snapshot or memoised live lookups; punctuation"
        " stripping, exactly-one-value rule and ArgSpec.spellings choice matching kept;"
        " a str argument with no ground kind is reported, never taken from the text",
        "score :632-686 takes an injectable top-k callable (prompt, top) instead of the"
        " NextTokenScorer protocol and returns ScorerPrediction (the backbone-agnostic"
        " predictions record) instead of Scored",
        "served path: nvsh/tiers/toolchat.py score_next_token body and the"
        " _shape_l/_shape_c logprob parsers become served_top_k with an injectable post",
        "TransformersScorer :689-748 becomes InProcessTopK over any full-vocabulary"
        " log-probability function, scanning variants for the whole label alphabet once",
    ],
    "licence": "Apache-2.0",
}

#: The two controls' default prompt text (nvsh ``lfm.tools_for`` wording, domain-neutral).
CONTROL_DESCRIPTIONS: Mapping[str, str] = {
    EXPLAIN: "Answer the request in plain words, with no action.",
    ESCALATE: "Hand this request to the full agent, saying why.",
}

#: What follows the domain's instruction, before the candidate lines.
ACTIONS_HEADER = "\n\nActions:\n"

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*")

#: The injectable scoring seam: ``(prompt, top) -> {token text: natural-log probability}``.
TopK = Callable[[str, int], Mapping[str, float]]

#: ``post(url, json_body, timeout) -> parsed JSON reply`` for :func:`served_top_k`.
Post = Callable[[str, Mapping[str, Any], float], Any]


# ---------------------------------------------------------------------------
# Candidates and labels
# ---------------------------------------------------------------------------


def candidate_pool(domain: Domain, reasons: bool = False) -> tuple[str, ...]:
    """Every candidate one request may be offered, in positional-letter order."""
    return domain.candidates(with_reasons=reasons)


def positional_labels(offered: Sequence[str], full: Sequence[str]) -> dict[str, str]:
    """Candidate -> letter for *offered*, each lettered by its position in *full*."""
    if len(full) > len(ro.LABEL_ALPHABET):
        raise ValueError(f"{len(full)} candidates but only {len(ro.LABEL_ALPHABET)} labels")
    index = {name: position for position, name in enumerate(full)}
    missing = [name for name in offered if name not in index]
    if missing:
        raise ValueError(f"not in the candidate pool: {', '.join(missing)}")
    return {name: ro.LABEL_ALPHABET[index[name]] for name in offered}


def labels_for(domain: Domain, offered: Sequence[str], reasons: bool = False) -> dict[str, str]:
    """Candidate -> letter for *offered*; a letter does not move when others are left out."""
    return positional_labels(offered, candidate_pool(domain, reasons))


@dataclass(frozen=True)
class Permutation:
    """A seeded order + letter map (:func:`permute`), round-trippable through JSON.

    ``order`` is the offered candidates in listing order (so it carries the
    subset too); ``labels`` maps each to its letter.
    """

    order: tuple[str, ...]
    labels: dict[str, str]

    def to_json(self) -> dict:
        return {"order": list(self.order), "labels": dict(self.labels)}

    @classmethod
    def from_json(cls, data: Mapping) -> Permutation:
        return cls(order=tuple(data["order"]), labels=dict(data["labels"]))


def permute(
    seed,
    pool: Sequence[str],
    *,
    subset: int | None = None,
    keep: str | Sequence[str] | None = None,
) -> Permutation:
    """A seeded random order + letter map (optionally a random subset) of *pool*.

    The same *seed* always gives the same :class:`Permutation`. *subset*
    keeps that many candidates, always including *keep* (the gold answer).
    Letters are drawn from the readout's label alphabet without replacement.
    """
    names = list(pool)
    if len(names) > len(ro.LABEL_ALPHABET):
        raise ValueError(f"{len(names)} candidates but only {len(ro.LABEL_ALPHABET)} labels")
    rng = random.Random(seed)  # nosec B311 -- seeded shuffles, not cryptography
    order = list(names)
    rng.shuffle(order)
    if subset is not None:
        keep_names = (keep,) if isinstance(keep, str) else tuple(keep or ())
        for name in keep_names:
            if name not in order:
                raise ValueError(f"{name!r} to keep is not in the candidate pool")
        if subset < len(keep_names):
            raise ValueError(f"subset {subset} is smaller than the {len(keep_names)} to keep")
        if subset > len(order):
            raise ValueError(f"subset {subset} is larger than the {len(order)}-candidate pool")
        kept = [name for name in order if name in keep_names]
        rest = [name for name in order if name not in keep_names]
        order = sorted(kept + rest[: subset - len(kept)], key=order.index)
    letters = list(ro.LABEL_ALPHABET)
    rng.shuffle(letters)
    return Permutation(order=tuple(order), labels={n: letters[i] for i, n in enumerate(order)})


def _label_map(
    domain: Domain,
    offered: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    order: Sequence[str] | None,
    reasons: bool,
) -> dict[str, str]:
    """Candidate -> letter for one request, in listing order."""
    if order is not None:
        names = list(order)
    elif offered is not None:
        names = list(offered)
    elif labels is not None:
        names = list(labels)
    else:
        names = list(candidate_pool(domain, reasons))
    if labels is None:
        return labels_for(domain, names, reasons)
    missing = [name for name in names if name not in labels]
    if missing:
        raise ValueError(f"no label given for: {', '.join(missing)}")
    return {name: labels[name] for name in names}


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def description(domain: Domain, name: str, descriptions: Mapping[str, str] | None = None) -> str:
    """*name*'s prompt text: an override, else the operation's, reason's or control's text."""
    if descriptions is not None and name in descriptions:
        return descriptions[name]
    operation = domain.get(name)
    if operation is not None:
        return operation.description
    if name.startswith(ESCALATE_PREFIX):
        reason_text = domain.reason_descriptions().get(name)
        if reason_text is not None:
            return reason_text
    if name in CONTROL_DESCRIPTIONS:
        return domain.control_description(name) or CONTROL_DESCRIPTIONS[name]
    raise ValueError(f"{name!r} is not a candidate of domain {domain.name!r}")


def prompt_messages(
    domain: Domain,
    request_text: str,
    offered: Sequence[str] | None = None,
    *,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
    reasons: bool = False,
) -> list[dict]:
    """System message listing the offered candidates under their letters, then the request.

    With no *labels*/*order* every candidate of the pool is offered under its
    positional letter. Pass a stored :class:`Permutation`'s ``labels`` and
    ``order`` to render a randomised prompt.
    """
    resolved = _label_map(domain, offered, labels, order, reasons)
    lines = [
        f"{letter}) {name}: {description(domain, name, descriptions)}"
        for name, letter in resolved.items()
    ]
    return [
        {"role": "system", "content": domain.instruction + ACTIONS_HEADER + "\n".join(lines)},
        {"role": "user", "content": request_text},
    ]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """The prompt text up to the answer, thinking off when the template has that switch."""
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Grounding: arguments come from the domain, never from the model
# ---------------------------------------------------------------------------


def _memoised_lookups(domain: Domain) -> Domain:
    """*domain* with each live lookup run at most once (for one request)."""

    def once(lookup):
        cache: list = []

        def run():
            if not cache:
                try:
                    cache.append(("ok", lookup()))
                except Exception as exc:  # noqa: BLE001 -- replayed to Domain.ground
                    cache.append(("err", exc))
            status, value = cache[0]
            if status == "err":
                raise value
            return value

        return run

    kinds = tuple(
        replace(kind, lookup=once(kind.lookup)) if kind.lookup is not None else kind
        for kind in domain.ground_kinds
    )
    return replace(domain, ground_kinds=kinds)


def _choice_value(name: str, choices: Sequence[str], spellings, text: str) -> str:
    folded = f" {' '.join(_WORD_RE.findall(text)).casefold()} "
    found = [
        choice
        for choice in choices
        if any(f" {spelling.casefold()} " in folded for spelling in spellings(choice))
    ]
    if len(found) == 1:
        return found[0]
    if found:
        raise LookupError(f"{name} is ambiguous: {', '.join(found)}")
    raise LookupError(f"no {name} named in the request")


def _grounded_value(
    domain: Domain,
    operation: Operation,
    name: str,
    text: str,
    world: Mapping[str, Any] | None,
) -> str:
    """The one value the request's words ground to for argument *name*."""
    found: list[str] = []
    # Sentence punctuation is not a name: "restart vllm." must still ground vllm.
    words = (word.rstrip(".:") for word in _WORD_RE.findall(text))
    for word in dict.fromkeys(word for word in words if word):
        grounded = domain.ground(operation.name, {name: word}, world)
        if isinstance(grounded, GroundDecline):
            if grounded.code in ("lookup_failed", "unknown_operation"):
                raise LookupError(grounded.message)
            continue
        value = grounded.args[name]
        if value not in found:
            found.append(value)
    if len(found) == 1:
        return found[0]
    if found:
        raise LookupError(f"{name} is ambiguous: {', '.join(found)}")
    raise LookupError(f"no {name} named in the request grounds against the world")


def ground_arguments(
    domain: Domain,
    operation: Operation,
    request_text: str,
    world: Mapping[str, Any] | None = None,
) -> dict[str, str] | str:
    """Arguments for *operation* from the request's words, or one line saying why not.

    ``choice`` arguments match one declared choice (any of its spellings);
    grounded ``str`` arguments are whichever request word
    :meth:`Domain.ground` accepts against *world* (or the kinds' live
    lookups, each run at most once). A ``str`` argument with no ground kind
    cannot be grounded and is reported. The result is checked with
    :meth:`Domain.validate_args`.
    """
    if world is None:
        domain = _memoised_lookups(domain)
    args: dict[str, str] = {}
    try:
        for spec in operation.args:
            if spec.kind == "choice":
                args[spec.name] = _choice_value(
                    spec.name, spec.choices, spec.spellings, request_text
                )
            elif spec.ground is None:
                raise LookupError(
                    f"{spec.name} declares no ground kind; it is never taken from the model"
                )
            else:
                args[spec.name] = _grounded_value(domain, operation, spec.name, request_text, world)
    except LookupError as problem:
        return str(problem.args[0]) if problem.args else str(problem)
    invalid = domain.validate_args(operation.name, args)
    return invalid.message if invalid is not None else args


# ---------------------------------------------------------------------------
# The predictions record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorerPrediction:
    """One request's scores: the backbone-agnostic predictions record.

    ``offered`` is ``(label, name)`` in listing order. ``raw_scores`` is each
    offered candidate's unnormalised score (for a causal LM, the summed
    next-token probability of its label's token variants; ``None`` for a
    label the scorer did not return). ``probabilities`` is the distribution
    over the offered candidates, empty when no label had mass or the readout
    is incomplete (``incomplete``/``missing`` then say why; it is never
    renormalised). ``outcome`` is the raw, pre-gate outcome: ``"propose"``
    for an operation, ``"explain"``, ``"escalate"`` (``reason`` names an
    ``escalate:<r>`` choice), or ``None`` when nothing was chosen. For a
    proposal, ``arguments`` are the grounded values and ``grounded`` is
    ``True``; otherwise ``grounding`` says why not.
    """

    id: str | None
    offered: tuple[tuple[str, str], ...]
    raw_scores: dict[str, float | None]
    probabilities: dict[str, float]
    outcome: str | None
    operation: str | None
    arguments: dict[str, str] | None
    grounded: bool
    choice: str | None = None
    reason: str | None = None
    confidence: float = 0.0
    mass: float = 0.0
    grounding: str | None = None
    incomplete: str | None = None
    missing: tuple[str, ...] = field(default=())

    @property
    def candidates(self) -> dict[str, float] | None:
        """The distribution under the calibration labels, or ``None`` when there is none."""
        if not self.probabilities or self.incomplete is not None:
            return None
        return {Domain.calibration_label(name): p for name, p in self.probabilities.items()}

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe line; ``candidates`` adds the calibration-labelled distribution."""
        return {
            "id": self.id,
            "offered": [{"label": label, "name": name} for label, name in self.offered],
            "raw_scores": dict(self.raw_scores),
            "probabilities": dict(self.probabilities),
            "candidates": self.candidates,
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "grounded": self.grounded,
            "choice": self.choice,
            "reason": self.reason,
            "confidence": self.confidence,
            "mass": self.mass,
            "grounding": self.grounding,
            "incomplete": self.incomplete,
            "missing": list(self.missing),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScorerPrediction:
        return cls(
            id=data.get("id"),
            offered=tuple((o["label"], o["name"]) for o in data.get("offered", ())),
            raw_scores=dict(data.get("raw_scores") or {}),
            probabilities=dict(data.get("probabilities") or {}),
            outcome=data.get("outcome"),
            operation=data.get("operation"),
            arguments=None if data.get("arguments") is None else dict(data["arguments"]),
            grounded=bool(data.get("grounded")),
            choice=data.get("choice"),
            reason=data.get("reason"),
            confidence=float(data.get("confidence", 0.0)),
            mass=float(data.get("mass", 0.0)),
            grounding=data.get("grounding"),
            incomplete=data.get("incomplete"),
            missing=tuple(data.get("missing") or ()),
        )


def same_choice(a, b) -> bool:
    """True when *a* and *b* chose the same candidate, by name (never by letter)."""
    left = a.choice if isinstance(a, ScorerPrediction) else a
    right = b.choice if isinstance(b, ScorerPrediction) else b
    return left == right


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _outcome(domain: Domain, choice: str | None) -> tuple[str | None, str | None, str | None]:
    """``(outcome, operation, reason)`` for a choice."""
    if choice is None:
        return (None, None, None)
    if domain.get(choice) is not None:
        return ("propose", choice, None)
    if choice.startswith(ESCALATE_PREFIX):
        return (ESCALATE, None, choice)
    return (choice, None, None)


def score(
    domain: Domain,
    top_k: TopK,
    prompt: str,
    request_text: str,
    *,
    offered: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    reasons: bool = False,
    world: Mapping[str, Any] | None = None,
    entry_id: str | None = None,
) -> ScorerPrediction:
    """Score every offered candidate for one request. Never raises on a scorer failure.

    *prompt* is the rendered prompt (:func:`render_prompt` of
    :func:`prompt_messages` built with the same *offered*/*labels*/*order*)
    and *request_text* the request, which grounding reads against *world*
    (or the domain's live lookups). Ties go to the earlier listed candidate.
    """
    letters = _label_map(domain, offered, labels, order, reasons)
    listing = tuple((letter, name) for name, letter in letters.items())
    try:
        logprobs = top_k(prompt, ro.READOUT_TOP)
    except Exception:  # noqa: BLE001 -- a scorer that fails makes no choice
        logprobs = {}
    if not isinstance(logprobs, Mapping):
        logprobs = {}
    result = ro.readout(logprobs, letters)
    source = result.distribution or result.probabilities or {}
    missing = set(result.missing) if source else set(letters)
    raw_scores = {
        name: (None if name in missing else source[name] * result.mass) for name in letters
    }
    choice = result.choice
    if choice is None:
        return ScorerPrediction(
            id=entry_id,
            offered=listing,
            raw_scores=raw_scores,
            probabilities={},
            outcome=None,
            operation=None,
            arguments=None,
            grounded=False,
        )
    if result.complete:
        confidence, incomplete = result.distribution[choice], None
    else:
        confidence = source[choice] * result.mass  # the raw probability, never renormalised
        incomplete = (
            f"{len(result.missing)} of {len(letters)} candidate labels missing from the"
            " scorer's top log-probabilities"
        )
    outcome, operation_name, reason = _outcome(domain, choice)
    arguments, grounded, grounding = None, False, None
    if operation_name is not None:
        found = ground_arguments(domain, domain.get(operation_name), request_text, world)
        if isinstance(found, str):
            grounding = found
        else:
            arguments, grounded = found, True
    return ScorerPrediction(
        id=entry_id,
        offered=listing,
        raw_scores=raw_scores,
        probabilities=dict(result.distribution),
        outcome=outcome,
        operation=operation_name,
        arguments=arguments,
        grounded=grounded,
        choice=choice,
        reason=reason,
        confidence=confidence,
        mass=result.mass,
        grounding=grounding,
        incomplete=incomplete,
        missing=tuple(result.missing),
    )


# ---------------------------------------------------------------------------
# Top-k sources: a served endpoint, or a model in this process
# ---------------------------------------------------------------------------


class ServedError(RuntimeError):
    """A served top-k request failed or its reply carried no log-probabilities."""


def _default_post(url: str, body: Mapping[str, Any], timeout: float) -> Any:
    if not url.startswith(("http://", "https://")):
        raise ServedError(f"refusing a non-http endpoint: {url}")
    request = urllib.request.Request(  # nosec B310 -- scheme checked above
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read())


def _valid(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
        and value <= 0
    )


def parse_top_logprobs(reply: Any) -> dict[str, float]:
    """``{token: logprob}`` from a ``/completions`` reply (legacy or content shape)."""
    choices = reply.get("choices") if isinstance(reply, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    logprobs = first.get("logprobs") if isinstance(first, dict) else None
    if not isinstance(logprobs, dict):
        raise ServedError("the server reply carries no logprobs")
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        scores = {t: float(v) for t, v in top[0].items() if isinstance(t, str) and _valid(v)}
    else:
        content = logprobs.get("content")
        head = content[0] if isinstance(content, list) and content else None
        entries = head.get("top_logprobs") if isinstance(head, dict) else None
        scores = {
            e["token"]: float(e["logprob"])
            for e in (entries if isinstance(entries, list) else ())
            if isinstance(e, dict) and isinstance(e.get("token"), str) and _valid(e.get("logprob"))
        }
    if not scores:
        raise ServedError("the server reply carries no logprobs")
    return scores


def served_top_k(
    base_url: str, model: str, *, post: Post | None = None, timeout: float = 60.0
) -> TopK:
    """A :data:`TopK` over an OpenAI-style ``<base_url>/completions`` endpoint.

    One prefill and one token (``max_tokens=1``, ``temperature=0``), asking
    for *top* log-probabilities. *post* is injectable (tests need no server).
    """
    send = post or _default_post
    url = base_url.rstrip("/") + "/completions"

    def top_k(prompt: str, top: int) -> dict[str, float]:
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": top,
            "stream": False,
        }
        try:
            reply = send(url, body, timeout)
        except ServedError:
            raise
        except Exception as exc:  # noqa: BLE001 -- transport and decoding alike
            raise ServedError(f"request to {url} failed: {type(exc).__name__}: {exc}") from exc
        return parse_top_logprobs(reply)

    return top_k


class InProcessTopK:
    """A :data:`TopK` over a local model: every label variant's true next-token log-probability.

    *logprobs_fn* returns the full-vocabulary log-softmax at the next
    position for a prompt (:func:`transformers_logprobs` builds one; any
    indexable row works). The vocabulary is scanned once for every letter of
    the label alphabet (:func:`readout.label_variant_ids`), so any letter map
    a :class:`Permutation` draws is read completely. *top* is ignored.
    """

    def __init__(self, tokenizer, logprobs_fn: Callable[[str], Sequence[float]]) -> None:
        letters = {letter: letter for letter in ro.LABEL_ALPHABET}
        self._variants = ro.label_variant_ids(tokenizer, letters)
        ro.label_token_ids(tokenizer, letters)  # each bare letter is one distinct token
        self._tokenizer = tokenizer
        self._logprobs_fn = logprobs_fn

    def __call__(self, prompt: str, top: int = ro.READOUT_TOP) -> dict[str, float]:
        del top
        return ro.variant_logprobs(self._logprobs_fn(prompt), self._variants, self._tokenizer)


def transformers_logprobs(model, tokenizer) -> Callable[[str], Any]:
    """A full-vocabulary next-token log-probability function over a transformers model.

    One forward pass keeping only the last position's logits. torch is
    imported on first call.
    """

    def logprobs(prompt: str):
        import torch

        device = next(model.parameters()).device
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
        with torch.no_grad():
            logits = model(input_ids=ids.to(device), logits_to_keep=1).logits[0, -1]
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    return logprobs
