"""Jetson-ops source registry: pinned repos, licences and attribution (issue #20, t9).

Loads ``sources.json`` and validates it, failing closed: an unknown licence, a commit that is
not a full 40-hex sha or a missing attribution refuses the whole registry. Stdlib only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

REGISTRY_PATH = Path(__file__).with_name("sources.json")
_CC_BY = "CC-BY-4.0"
ALLOWED_LICENCES = frozenset({"MIT", "Apache-2.0", _CC_BY})
PATH_CLASSES = ("code", "docs")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+$")


@dataclass(frozen=True)
class Source:
    id: str
    repo_url: str
    commit: str
    paths: tuple[str, ...]
    licences: dict[str, str]
    attribution: str

    def licence_for(self, path_class: str) -> str:
        """Licence of a path class (``code`` or ``docs``); unknown classes are refused."""
        if path_class not in self.licences:
            raise CliError(
                EXIT_USER_ERROR,
                f"source {self.id!r}: unknown path class {path_class!r}",
                f"use one of {', '.join(PATH_CLASSES)}",
            )
        return self.licences[path_class]

    def record_terms(self, path_class: str) -> dict[str, str]:
        """What a record drafted from this source must carry."""
        return {
            "source": self.id,
            "repo_url": self.repo_url,
            "commit": self.commit,
            "licence": self.licence_for(path_class),
            "attribution": self.attribution,
        }


def _bad(msg: str) -> CliError:
    return CliError(EXIT_ENV_ERROR, f"jetson-ops sources: {msg}", "fix sources.json")


def _text(entry: dict[str, Any], key: str, sid: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _bad(f"source {sid!r} has no {key}")
    return value


def _licences(entry: dict[str, Any], sid: str) -> dict[str, str]:
    raw = entry.get("licences")
    if not isinstance(raw, dict) or set(raw) != set(PATH_CLASSES):
        raise _bad(f"source {sid!r} must give a licence for each of {PATH_CLASSES}")
    for cls, lic in raw.items():
        if lic not in ALLOWED_LICENCES:
            raise _bad(f"source {sid!r}: unknown licence {lic!r} for {cls}")
    return dict(raw)


def _parse(entry: Any) -> Source:
    if not isinstance(entry, dict):
        raise _bad("source entry is not an object")
    sid = _text(entry, "id", "?")
    url = _text(entry, "repo_url", sid)
    if not _URL.match(url):
        raise _bad(f"source {sid!r}: bad repo_url {url!r}")
    commit = entry.get("commit")
    if not isinstance(commit, str) or not _SHA.match(commit):
        raise _bad(f"source {sid!r}: missing or non-full commit")
    paths = entry.get("paths")
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise _bad(f"source {sid!r} has no paths")
    return Source(
        sid, url, commit, tuple(paths), _licences(entry, sid), _text(entry, "attribution", sid)
    )


def _check_cc_by(source: Source) -> None:
    if _CC_BY in source.licences.values() and _CC_BY not in source.attribution:
        raise _bad(f"source {source.id!r}: CC-BY-4.0 attribution must name the licence")


def load_sources(path: Path | None = None) -> dict[str, Source]:
    """Load and validate the registry; any defect raises ``CliError``."""
    target = path or REGISTRY_PATH
    try:
        data = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _bad(f"cannot read {target}: {exc}") from exc
    entries = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise _bad("no sources listed")
    out: dict[str, Source] = {}
    for entry in entries:
        source = _parse(entry)
        if source.id in out:
            raise _bad(f"duplicate source {source.id!r}")
        _check_cc_by(source)
        out[source.id] = source
    return out


def record_terms(source_id: str, path_class: str, path: Path | None = None) -> dict[str, str]:
    """Licence and attribution a record drafted from ``source_id`` must carry."""
    sources = load_sources(path)
    if source_id not in sources:
        raise CliError(
            EXIT_USER_ERROR, f"unknown jetson-ops source {source_id!r}", "see sources.json"
        )
    return sources[source_id].record_terms(path_class)
