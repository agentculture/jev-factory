"""Eval slices of a split: the missing-candidate slice.

Reads an existing split (train/val/test) file, takes every entry whose
``expect`` block names a concrete operation, and emits a new slice in which
``candidates`` lists every operation of the domain **except** the gold one and
``expect`` is ``{"escalate": true}``: the model must hand off a request whose
right action was not offered.

Usage::

    python -m jev_factory.measure.slices --domain <module|file.json> \\
        --split out/val.json --out out/val-missing-candidate.json

Stdlib only; prints only the number of entries written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/eval_slices.py",
    "commit": "9debdc6",
    "adaptations": [
        "missing_candidate_slice (lines 32-66) unchanged",
        "main (lines 69-88): nvsh.ops.table.names() becomes the --domain's Domain.names();"
        " the sys.path hack (line 27) is removed",
    ],
    "licence": "Apache-2.0",
}


def missing_candidate_slice(split: dict, all_operations: Sequence[str]) -> dict:
    """Return a new slice with missing-candidate entries for every operation expect.

    For each entry whose ``expect`` dict has an ``"operation"`` key, emit a
    new entry with:

    - ``id`` = original ``id`` + ``"-nocand"``
    - ``kind`` = same kind
    - ``text`` = byte-identical to the original
    - ``source_id`` = original id
    - ``candidates`` = ``all_operations`` with the gold operation removed
    - ``expect`` = ``{"escalate": True}``

    Entries without an ``"operation"`` key in their ``expect`` dict are
    **not** included. The input *split* dict is never mutated.
    """
    header = f"Eval slice: missing candidates for {split['header'].strip()}."
    new_entries: list[dict] = []
    for entry in split.get("entries", []):
        expect = entry.get("expect", {})
        if "operation" not in expect:
            continue
        gold_op = expect["operation"]
        new_entries.append(
            {
                "id": entry["id"] + "-nocand",
                "kind": entry["kind"],
                "text": entry["text"],
                "source_id": entry["id"],
                "candidates": [op for op in all_operations if op != gold_op],
                "expect": {"escalate": True},
            }
        )
    return {"header": header, "entries": new_entries}


def main(argv: list[str] | None = None) -> int:
    from jev_factory.domain.validate import DomainError, load_domain

    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.measure.slices", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--domain", required=True, help="domain module or JSON domain file")
    parser.add_argument("--split", required=True, help="Path to a split JSON file")
    parser.add_argument("--out", required=True, help="Path to write the eval-slice JSON")
    args = parser.parse_args(argv)

    try:
        domain = load_domain(args.domain)
    except DomainError as exc:
        parser.exit(1, f"error: {exc}\n")
    with open(Path(args.split), encoding="utf-8") as handle:
        split = json.load(handle)

    slice_dict = missing_candidate_slice(split, domain.names())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(slice_dict, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"entries={len(slice_dict['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
