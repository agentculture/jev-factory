"""Contamination check of evaluation sources against upstream training data and eval suites.

Each evaluation source name gets one of three statuses against the checked-in manifest
``data/upstream_sources.json`` (strands-decider-2B-hobson-v21's 29 card datasets with the
role from its ``data/sources.md``, plus Clef's Decision Index suites; every entry cites the
URL it came from):

* ``overlapping``: v21 trained on it, or it is a Clef Decision Index suite;
* ``v21-evaluated``: v21 only held it out / evaluated on it. It does not inflate v21 but is
  not independent of it either (when a name hits both, ``overlapping`` wins);
* ``clean``: not found.

jev-bench configs are matched through aliases (``banking77``, ``clinc150``, ``massive``,
``civil_comments``, ``measuring_hate_speech`` ...) or by upstream HF id. ``Record.hf_id`` holds
``meta.hf_id`` for jev-bench rows: pass ``records=`` and each record's ``hf_id`` is
derived automatically (keyed by its ``source``); an explicit ``hf_ids={source: hf_id}`` wins and
the id is tried first; a ``jev-bench:<config>`` source name is also accepted.

Known limitations (v1): matching is by NAME only, after normalisation (lowercase,
alphanumerics only) plus the manifest's alias table. No row-level or near-duplicate
check is done, so a renamed or re-mirrored copy of an upstream dataset is reported
clean. Clef's own training data is unreleased, so its Decision Index entries only cite the
leaderboard and cannot be checked at row level; "clean" means "not found in the manifest",
not "proven clean". Of the 21 jev-bench configs without a known upstream id (the page shows
``meta.hf_id`` only for ``arc_challenge``), unaliased ones fall to ``clean`` by name.
Stdlib only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill.records import Record
from jev_factory.factory import safepath

STAGE = "distill-contamination"
MANIFEST_PATH = Path(__file__).parent / "data" / "upstream_sources.json"
CLEAN = "clean"
OVERLAPPING = "overlapping"
V21_EVALUATED = "v21-evaluated"
EVAL_ONLY_ROLE = "v21-eval-only"
JEV_BENCH_PREFIX = "jev-bench:"


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = data["sources"]
        for s in sources:
            if not (s["name"] and s["source_url"] and isinstance(s["aliases"], list)):
                raise ValueError("entry needs name, url and aliases")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path}: not a valid upstream sources manifest ({exc})",
        ) from None
    return data


def _index(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = {}
    for s in manifest["sources"]:
        for key in {normalise(s["name"]), *(normalise(a) for a in s["aliases"])}:
            idx.setdefault(key, []).append(s)
    return idx


def _status(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return CLEAN
    if all(h["role"] == EVAL_ONLY_ROLE for h in hits):
        return V21_EVALUATED
    return OVERLAPPING


def _config_name(name: str) -> str:
    """Strip ``jev-bench:`` or ``<org>/jev-bench:`` so the config name is matched."""
    head, sep, config = name.partition(JEV_BENCH_PREFIX)
    if sep and head.rstrip("/").count("/") == 0 and (head == "" or head.endswith("/")):
        return config
    return name


def check_sources(
    names: Iterable[str],
    manifest: dict[str, Any] | None = None,
    hf_ids: Mapping[str, str] | None = None,
    records: Iterable[Record] | None = None,
) -> dict[str, Any]:
    """Mark each source name clean, v21-evaluated or overlapping; returns the report dict.

    *hf_ids* optionally maps a source name to its upstream HF id (``meta.hf_id``); when
    present the id is matched in addition to the name. *records* supplies ``Record.hf_id``
    values for their ``source`` names; an explicit *hf_ids* entry overrides them.
    """
    manifest = manifest if manifest is not None else load_manifest()
    idx = _index(manifest)
    hf_ids = {**{r.source: r.hf_id for r in (records or ()) if r.hf_id}, **(hf_ids or {})}
    results: dict[str, Any] = {}
    for name in sorted(set(names)):
        keys = [_config_name(name)]
        if name in hf_ids:
            keys.insert(0, hf_ids[name])
        hits: list[dict[str, Any]] = []
        for key in keys:
            for h in idx.get(normalise(key), []):
                if h not in hits:
                    hits.append(h)
        results[name] = {
            "status": _status(hits),
            "matches": [
                {
                    "name": h["name"],
                    "role": h["role"],
                    "source_url": h["source_url"],
                    "card": h["card"],
                }
                for h in hits
            ],
        }

    def of(status: str) -> list[str]:
        return sorted(n for n, r in results.items() if r["status"] == status)

    return {
        "stage": STAGE,
        "method": "name/hf-id match plus alias table",
        "manifest_note": manifest.get("note", ""),
        "results": results,
        "overlapping": of(OVERLAPPING),
        "v21_evaluated": of(V21_EVALUATED),
        "clean": of(CLEAN),
    }


def write_report(report: dict[str, Any], run_dir: Path | str) -> Path:
    """Write the report into the run directory for the final report."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = safepath.contained(out / "contamination.json", out)  # no symlink out of the run
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
