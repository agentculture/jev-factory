"""The request contract shared by every reference-provider adapter.

There is exactly one way to turn a :class:`~jev_factory.evals.cases.Case`
into a provider call: :func:`build_choice_request`, the jev-like scorer's own
lettered prompt (:func:`jev_factory.backbones.causal_lm.scorer.prompt_messages`)
over the case's offered candidates plus the two controls, so a reference model
reads exactly the text a candidate checkpoint was scored on.

Every adapter MUST build its provider payload by calling
:func:`canonical_content` on the :class:`~jev_factory.evals.providers.base.CallRequest`
this module returns, never by re-deriving system/user text itself: that is
what makes the content byte-identical across provider kinds by construction.

Answers are parsed back with :func:`parse_choice`, routed through
:func:`jev_factory.evals.providers.errors.classify_answer` so "invalid" is
classified in exactly one place. Where a served model returned per-label
log-probabilities, :func:`distribution_from_logprobs` gives the candidate
distribution through the readout's one definition, never an estimate when a
label's mass did not come back.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jev_factory.backbones.causal_lm import readout, scorer
from jev_factory.domain.model import CONTROLS, Domain

from .cases import Case
from .providers.base import CallRequest
from .providers.errors import Classification, Outcome, classify_answer

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/request.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 57-94: nvsh.agent/nvsh.tiers.lfm/nvsh.platform imports and the path-loaded"
        " scripts/lfm-finetune/scorer.py are replaced by jev_factory.backbones.causal_lm"
        " (scorer.prompt_messages/labels_for, readout.distribution/missing_labels) over the"
        " Domain",
        "Track A (build_tool_call_request, _tools_for_case, parse_tool_call, lines 100-247) is"
        " deferred per c14: only the choice interface is ported",
        "the request text is the case's own text: nvsh's AgentRequest/AgentContext rendering"
        " of a failed command (lfm.request_message) was nvsh-specific",
        "canonical_content returns (system, messages, labels); there are no tools and no"
        " history without Track A",
    ],
    "licence": "Apache-2.0",
}

#: The reasoning/output-length knobs every request carries in ``params``.
DEFAULT_REASONING = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 512


def offered_names(case: Case, domain: Domain) -> tuple[str, ...]:
    """Every candidate *case* offers, in pool order: its operations, then the two controls."""
    if case.candidates is None:
        return domain.candidates()
    return tuple(case.candidates) + CONTROLS


def build_choice_request(case: Case, domain: Domain) -> CallRequest:
    """The ``CallRequest`` for *case*: the model answers with one candidate's letter.

    Built exactly the way the causal-LM scorer builds its own prompt: the
    domain's instruction, the offered candidates under their positional
    letters (a letter does not move when others are left out), then the
    request text as the user turn. ``params["labels"]`` is the candidate ->
    letter map; ``offered_candidates`` is the letters.
    """
    if case.text is None:
        raise ValueError(f"{case.id}: a held-out case has no request text to build a request from")
    names = offered_names(case, domain)
    labels = scorer.labels_for(domain, names)
    messages = scorer.prompt_messages(domain, case.text, names, labels=labels)
    params = {
        "labels": labels,
        "reasoning": DEFAULT_REASONING,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }
    return CallRequest(
        case_id=case.id,
        split=case.split,
        case_text=messages[1]["content"],
        prompt=messages[0]["content"],
        offered_candidates=tuple(labels.values()),
        params=params,
        interface="choice",
    )


def parse_choice(
    answer_text: str | None, labels: Mapping[str, str]
) -> tuple[Classification, str | None]:
    """Parse one answer into ``(classification, chosen candidate name)``.

    *labels* is the candidate -> letter map :func:`build_choice_request` put
    in ``params["labels"]``. The first non-whitespace character is the label
    (the prompt says "answer with the action's letter only"); a character that
    is not one of the letters is malformed, never guessed into a pick.
    """
    offered = tuple(labels)
    if not isinstance(answer_text, str):
        return classify_answer(None, offered, malformed=True), None
    token = answer_text.strip()[:1]
    by_label = {label: name for name, label in labels.items()}
    name = by_label.get(token) if token else None
    if name is None:
        return classify_answer(None, offered, malformed=True), None
    classification = classify_answer(name, offered)
    if classification.outcome is not Outcome.OK:
        return classification, None
    return classification, name


def distribution_from_logprobs(
    top_logprobs_of_first_answer_token: Mapping[str, float],
    labels: Mapping[str, str],
    offered_order: tuple[str, ...],
) -> dict[str, float] | None:
    """The candidate distribution over *labels*, in *offered_order*, or ``None``.

    ``None`` when any offered label's mass did not come back at all (never
    renormalised over the labels that did) or when no label had any mass;
    otherwise every offered candidate's renormalised probability.
    """
    if readout.missing_labels(top_logprobs_of_first_answer_token, labels):
        return None
    distribution, mass = readout.distribution(top_logprobs_of_first_answer_token, labels)
    if mass <= 0 or not distribution:
        return None
    return {name: distribution[name] for name in offered_order if name in distribution}


def canonical_content(request: CallRequest) -> tuple[str, list[dict], dict[str, str] | None]:
    """``(system_text, messages, labels)``: the only content an adapter may send.

    A pure read of *request*'s own fields; ``messages`` is a fresh
    ``[{"role": "user", "content": request.case_text}]`` so an adapter cannot
    mutate the request through it.
    """
    labels: Any = request.params.get("labels")
    messages: list[dict] = [{"role": "user", "content": request.case_text}]
    return request.prompt, messages, (dict(labels) if isinstance(labels, Mapping) else None)
