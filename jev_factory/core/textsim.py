"""Text normalisation and similarity shared by the leakage check and its callers.

The three helpers (:func:`normalize_text`, :func:`shingles`, :func:`jaccard`)
and their thresholds are the ones nvsh's ``leakage_check.py`` borrowed from
``jetson_skills.py``. They live here so nothing in jev-factory depends on a
skills scanner. Stdlib only.
"""

from __future__ import annotations

import re

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/jetson_skills.py",
    "commit": "9debdc6",
    "adaptations": [
        "only NEAR_DUP_SHINGLE_SIZE/NEAR_DUP_JACCARD_THRESHOLD/MIN_TOKENS_FOR_EXACT_MATCH"
        " (:445-447), normalize_text (:458), _shingles (:482) and _jaccard (:489) are"
        " taken; the skills scanner itself is not imported",
        "_shingles/_jaccard gain public aliases shingles/jaccard; behaviour unchanged",
        "WORD_JACCARD (0.8) moves here from leakage_check.py beside its sibling thresholds",
    ],
    "licence": "Apache-2.0",
}

#: Shingle length in tokens, and the shingle-Jaccard threshold for a near-duplicate.
NEAR_DUP_SHINGLE_SIZE = 5
NEAR_DUP_JACCARD_THRESHOLD = 0.8
#: Texts shorter than this many tokens only ever match exactly.
MIN_TOKENS_FOR_EXACT_MATCH = 4
#: Word-set Jaccard at or above this (texts of >= MIN_TOKENS_FOR_EXACT_MATCH words) is a
#: near-duplicate: stricter than a skills scan's 0.6, because requests are short.
WORD_JACCARD = 0.8

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case, punctuation to spaces, whitespace collapsed."""
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def shingles(text: str, size: int = NEAR_DUP_SHINGLE_SIZE) -> frozenset[tuple[str, ...]]:
    """The set of *size*-token runs of already-normalised *text*."""
    tokens = text.split()
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    """``|a & b| / |a | b|``; two empty sets are identical (1.0), one empty is 0.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


_shingles = shingles
_jaccard = jaccard
