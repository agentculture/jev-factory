"""Contamination check of evaluation sources against upstream training data and eval suites.

Each evaluation source name is marked ``overlapping`` if it matches an entry of the
checked-in manifest ``data/upstream_sources.json`` (strands-decider-2B-hobson-v21's
public training datasets and evaluation sets, and Clef's Decision Index suites; every
entry cites the URL it came from), else ``clean``.

Known limitations (v1): matching is by NAME only, after normalisation (lowercase,
alphanumerics only) plus the manifest's alias table. No row-level or near-duplicate
check is done, so a renamed or re-mirrored copy of an upstream dataset is reported
clean. The v21 card displays only 3 of its 29 metadata datasets, so the manifest is
incomplete for v21; "clean" means "not found in the manifest", not "proven clean".
Stdlib only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

STAGE = "distill-contamination"
MANIFEST_PATH = Path(__file__).parent / "data" / "upstream_sources.json"
CLEAN = "clean"
OVERLAPPING = "overlapping"


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources = data["sources"]
        for s in sources:
            if not (s["name"] and s["url"] and isinstance(s["aliases"], list)):
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


def check_sources(names: Iterable[str], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mark each source name clean or overlapping; returns the report dict."""
    manifest = manifest if manifest is not None else load_manifest()
    idx = _index(manifest)
    results: dict[str, Any] = {}
    for name in sorted(set(names)):
        hits = idx.get(normalise(name), [])
        results[name] = {
            "status": OVERLAPPING if hits else CLEAN,
            "matches": [
                {"name": h["name"], "role": h["role"], "url": h["url"], "card": h["card"]}
                for h in hits
            ],
        }
    return {
        "stage": STAGE,
        "method": "name-level match plus alias table",
        "manifest_note": manifest.get("note", ""),
        "results": results,
        "overlapping": sorted(n for n, r in results.items() if r["status"] == OVERLAPPING),
        "clean": sorted(n for n, r in results.items() if r["status"] == CLEAN),
    }


def write_report(report: dict[str, Any], run_dir: Path | str) -> Path:
    """Write the report into the run directory for the final report."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contamination.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
