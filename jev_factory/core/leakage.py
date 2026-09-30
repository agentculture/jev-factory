"""No val/test/held-out entry or near-duplicate in training (nvsh issue 46, t19).

``merge_variations --exclude`` drops a variation whose normalized text equals a
protected entry's. This also refuses a *near*-duplicate: a rewrite of a train
request can land a word or two away from a validation, test or held-out request.
It compares every training text with every protected text:

- **exact**: the normalized texts are equal;
- **near-duplicate**: 5-token-shingle Jaccard >= 0.8, or word-set Jaccard >= 0.8
  for texts of at least 4 words.

It prints counts and matching **ids only**, never a text, so it can be run
against the sealed held-out file before the final run. It **fails closed**:
any match exits 1 (unless ``--out-filtered`` writes the training file without
the matched entries), and anything it cannot check (an unreadable or malformed
file, an entry without checkable text) exits 2. It never passes a file it did
not check.

Usage::

    jev-leakage --train TRAIN.json --protected val.json test.json HELD_OUT.json \\
        [--out-filtered OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jev_factory.core.textsim import (
    MIN_TOKENS_FOR_EXACT_MATCH,
    NEAR_DUP_JACCARD_THRESHOLD,
    WORD_JACCARD,
    jaccard,
    normalize_text,
    shingles,
)

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/leakage_check.py",
    "commit": "9debdc6",
    "adaptations": [
        "the jetson_skills imports (:36-44) become jev_factory.core.textsim",
        "fails closed: an unreadable/malformed protected or train file, and a text that"
        " normalises to nothing, exit 2 naming the path or id only (nvsh raised a traceback"
        " or passed the empty text unchecked)",
        "`match` no longer skips silently on an empty normalised text: _texts refuses it first",
    ],
    "licence": "Apache-2.0",
}


def _load(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """``(document, entries)``: a split-format file ``{header, entries}``, or a
    JSONL file of records (document ``None``)."""
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix == ".jsonl":
            return None, [json.loads(line) for line in text.splitlines() if line.strip()]
        doc = json.loads(text)
        return doc, list(doc["entries"])
    except (ValueError, KeyError, TypeError) as exc:
        # Name the file and the error type only, never the parser's message.
        raise ValueError(
            f"{path}: not a readable split or JSONL file ({type(exc).__name__})"
        ) from None


def _texts(entries: list[dict[str, Any]], source: Path) -> list[str]:
    """Every entry's ``text``; an entry without a non-empty string text is refused
    (Codex review: it would otherwise pass unchecked, or match other missing
    texts as the string "None")."""
    texts = []
    for index, entry in enumerate(entries):
        text = entry.get("text")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError(f"{source}: entry {entry.get('id', index)!r} has no text to check")
        texts.append(text)
    return texts


def match(train_text: str, protected_text: str) -> str | None:
    """``"exact"``, ``"near-duplicate"`` or ``None``."""
    a, b = normalize_text(train_text), normalize_text(protected_text)
    if not a or not b:
        return None
    if a == b:
        return "exact"
    if jaccard(shingles(a), shingles(b)) >= NEAR_DUP_JACCARD_THRESHOLD:
        return "near-duplicate"
    words_a, words_b = a.split(), b.split()
    if (
        min(len(words_a), len(words_b)) >= MIN_TOKENS_FOR_EXACT_MATCH
        and jaccard(frozenset(words_a), frozenset(words_b)) >= WORD_JACCARD
    ):
        return "near-duplicate"
    return None


def find_matches(
    train: list[dict[str, Any]], protected: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """One match per training row that matches any protected entry; ``row`` is
    the training row's index, so filtering never depends on ids being unique."""
    found = []
    for row, entry in enumerate(train):
        for name, entries in protected.items():
            other = next((o for o in entries if match(entry["text"], o["text"])), None)
            if other is not None:
                found.append(
                    {
                        "row": row,
                        "train_id": str(entry.get("id")),
                        "protected": name,
                        "protected_id": str(other.get("id")),
                        "kind": match(entry["text"], other["text"]),
                    }
                )
                break
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-leakage", description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--protected", required=True, nargs="+", type=Path)
    parser.add_argument("--out-filtered", type=Path)
    args = parser.parse_args(argv)

    try:
        doc, train = _load(args.train)
        # Keyed by the full path: two files named test.json (issue 46: the new
        # split's and issue 39's) must both be checked (Codex review).
        protected = {str(path): _load(path)[1] for path in args.protected}
        _texts(train, args.train)
        for path, entries in protected.items():
            _texts(entries, Path(path))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        # Fail closed without echoing file content: the exception type only.
        print(f"error: could not read a checked file ({type(exc).__name__})", file=sys.stderr)
        return 2
    matches = find_matches(train, protected)
    report = {
        "train": len(train),
        "protected": {name: len(entries) for name, entries in protected.items()},
        "hits": len(matches),
        "by_kind": {k: sum(m["kind"] == k for m in matches) for k in ("exact", "near-duplicate")},
        "matches": matches,
    }
    print(json.dumps(report, indent=1))
    if args.out_filtered is None:
        return 1 if matches else 0
    drop = {m["row"] for m in matches}
    kept = [entry for row, entry in enumerate(train) if row not in drop]
    if doc is None:
        body = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in kept)
    else:
        body = json.dumps({**doc, "entries": kept}, indent=2, ensure_ascii=False) + "\n"
    args.out_filtered.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
