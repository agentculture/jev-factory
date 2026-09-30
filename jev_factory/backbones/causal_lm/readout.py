"""The label readout: the one definition of letters, token variants and the distribution.

A causal-decoder scorer offers N candidates under one-letter labels and reads
one next-token distribution. This module is the **only** place that defines
how that read becomes a distribution over the offered candidates, for every
path (served top-k log-probabilities, in-process logits, training):

- the label alphabet is ``A-Z`` then ``a-z`` (:data:`LABEL_ALPHABET`, at most 52);
- a candidate's raw mass is the sum of the next-token probabilities of every
  vocabulary token whose text, stripped of whitespace, is its label
  (``"A"``, ``" A"`` and ``"\\tA"`` all count for ``A``), found by scanning the
  vocabulary (:func:`label_variant_ids`), never assumed;
- probabilities are those masses normalised over the labels actually offered
  (:func:`distribution`);
- a served model only returns its top-k log-probabilities. A label absent
  from them makes the readout **incomplete** (:func:`readout`): it is named,
  carries no distribution and is never renormalised over the labels that did
  come back. :class:`ReadoutTally` counts complete and incomplete readouts
  for a run.

Heavy imports (torch) happen inside the functions that need them.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Mapping, Sequence

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/scorer.py",
    "commit": "9debdc6",
    "adaptations": [
        "lifted LABEL_ALPHABET, READOUT_TOP, label_token_ids, label_variant_ids,"
        " label_logits_from_vocab, distribution, missing_labels",
        "added Readout/readout (complete flag, never renormalised) and ReadoutTally",
        "added torch-free in-process helpers (log_softmax, variant_logprobs)"
        " so the same distribution is reproducible from raw logits",
        "removed the nvsh ops/table, grounding and ToolChat coupling",
    ],
    "licence": "Apache-2.0",
}

#: Label alphabet, in candidate order. Each is one token in the Qwen3.5 and
#: LFM2.5 vocabularies (checked by :func:`label_token_ids`, never assumed).
LABEL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase

#: How many next-token log-probabilities a served request asks for. nvsh's
#: 2026-09-25 probe found the top 22 missing 10-15 labels on every prompt and
#: the top 5000 complete on 6/6; a Q4_K_M GGUF still missed labels on 5 of 64
#: prompts at top 5000 and on none at top 20000. Anything still missing is
#: marked incomplete.
READOUT_TOP = 20000


def label_token_ids(tokenizer, labels: Mapping[str, str]) -> dict[str, int]:
    """Candidate -> the single token id of its bare label; refuses multi-token or shared labels."""
    ids: dict[str, int] = {}
    for name, label in labels.items():
        encoded = list(tokenizer.encode(label, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} for {name!r} is not a single token: {encoded}")
        ids[name] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("two labels share a token id; the scores would not be distinct")
    return ids


def label_variant_ids(tokenizer, labels: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    """Candidate -> every token id whose text, stripped of whitespace, is its label.

    The whole vocabulary is scanned once. Ids are sorted. Refuses a label
    with no token at all. Distinct labels never share an id.
    """
    by_label = {label: name for name, label in labels.items()}
    found: dict[str, list[int]] = {name: [] for name in labels}
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        name = by_label.get(tokenizer.decode([token_id]).strip())
        if name is not None:
            found[name].append(token_id)
    empty = [name for name, ids in found.items() if not ids]
    if empty:
        raise ValueError(f"no token in the vocabulary is the label of: {', '.join(empty)}")
    return {name: tuple(ids) for name, ids in found.items()}


def label_logits_from_vocab(logits, variant_ids: Sequence[Sequence[int]]):
    """``(..., labels)`` label logits from ``(..., vocab)`` logits, one log-sum-exp per label.

    The training half of the definition: a softmax over the result equals
    :func:`distribution` of the same next-token log-probabilities.
    Differentiable. *variant_ids* is in candidate order.
    """
    import torch

    columns = [
        torch.logsumexp(
            logits.index_select(-1, torch.tensor(list(ids), device=logits.device)), dim=-1
        )
        for ids in variant_ids
    ]
    return torch.stack(columns, dim=-1)


def _valid_logprob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
        and value <= 0
    )


def distribution(
    logprobs: Mapping[str, float], labels: Mapping[str, str]
) -> tuple[dict[str, float], float]:
    """``(candidate -> probability, mass)`` over *labels*. Never raises.

    Each token whose text stripped of whitespace is a label adds its
    probability to that label; the masses are normalised over the offered
    candidates. Tokens that are no label, and junk values, are skipped.
    ``({}, 0.0)`` when no label has mass.
    """
    by_label = {label: name for name, label in labels.items()}
    masses = {name: 0.0 for name in labels}
    for token, logprob in logprobs.items():
        if not isinstance(token, str) or not _valid_logprob(logprob):
            continue
        name = by_label.get(token.strip())
        if name is not None:
            masses[name] += math.exp(logprob)
    mass = sum(masses.values())
    if mass <= 0:
        return ({}, 0.0)
    return ({name: value / mass for name, value in masses.items()}, mass)


def missing_labels(logprobs: Mapping[str, float], labels: Mapping[str, str]) -> tuple[str, ...]:
    """Candidates in *labels* whose label (in any token variant) *logprobs* does not carry.

    A label reported with a valid log-probability is present even when that
    probability is zero; junk values count as absent, as in :func:`distribution`.
    """
    seen = {
        token.strip()
        for token, logprob in logprobs.items()
        if isinstance(token, str) and _valid_logprob(logprob)
    }
    return tuple(name for name, label in labels.items() if label not in seen)


@dataclass(frozen=True)
class Readout:
    """One prompt's readout. ``distribution`` is empty unless the readout is complete.

    ``probabilities`` keeps the (over-labels normalised) values of an
    incomplete readout only for diagnosis; never feed them to calibration.
    """

    distribution: dict[str, float]
    mass: float
    missing: tuple[str, ...] = ()
    probabilities: dict[str, float] | None = None

    @property
    def complete(self) -> bool:
        return bool(self.distribution) and not self.missing

    @property
    def choice(self) -> str | None:
        source = self.distribution or self.probabilities
        if not source:
            return None
        return max(source, key=lambda name: source[name])  # first maximum wins


def readout(logprobs: Mapping[str, float], labels: Mapping[str, str]) -> Readout:
    """Read *logprobs* over *labels*; an offered label missing makes it incomplete.

    An incomplete readout is never renormalised over the labels that came
    back (that would turn a label holding 1% of the raw mass into a
    certainty): its ``distribution`` is empty and ``missing`` names them.
    """
    probabilities, mass = distribution(logprobs, labels)
    if not probabilities:
        return Readout({}, 0.0, missing=tuple(labels))
    missing = missing_labels(logprobs, labels)
    if missing:
        return Readout({}, mass, missing=missing, probabilities=probabilities)
    return Readout(probabilities, mass)


@dataclass
class ReadoutTally:
    """Complete/incomplete readout counts for a run (goal: N complete / 0 incomplete)."""

    complete: int = 0
    incomplete: int = 0

    def record(self, result: Readout) -> Readout:
        if result.complete:
            self.complete += 1
        else:
            self.incomplete += 1
        return result

    @property
    def total(self) -> int:
        return self.complete + self.incomplete

    def summary(self) -> str:
        return f"{self.complete} complete / {self.incomplete} incomplete"

    def as_dict(self) -> dict[str, int]:
        return {"complete": self.complete, "incomplete": self.incomplete, "total": self.total}


# -- the in-process half (torch-free: takes plain numbers) --


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Log-softmax of one next-token logit row over the whole vocabulary."""
    values = [float(value) for value in logits]
    peak = max(values)
    shift = peak + math.log(sum(math.exp(value - peak) for value in values))
    return [value - shift for value in values]


def variant_logprobs(
    logprobs: Sequence[float], variants: Mapping[str, Sequence[int]], tokenizer
) -> dict[str, float]:
    """Variant token text -> log-probability, as a server keys them.

    *logprobs* is the full-vocabulary log-softmax at the next position and
    *variants* is :func:`label_variant_ids`. Two ids that decode to the same
    text have their probabilities added.
    """
    result: dict[str, float] = {}
    for ids in variants.values():
        for token_id in ids:
            text = tokenizer.decode([token_id])
            value = float(logprobs[token_id])
            if text in result:
                high, low = max(result[text], value), min(result[text], value)
                value = high if low == -math.inf else high + math.log1p(math.exp(low - high))
            result[text] = value
    return result
