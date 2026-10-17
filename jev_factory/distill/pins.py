"""Student input pins: revisions and sha256 verification from a local cache dir.

No network. The cache holds one directory per repo (``cache_dirname``), each with a
``REVISION`` file recording the commit it was fetched at. Upstream recorded the base revision
only as "inferred", so the pin lives in the run config and is checked here, failing closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile

# Module-level aliases of the built-in profile (DEFAULT_2610), kept importable.
BASE_REPO = DEFAULT_2610.base_repo
DECIDER_REPO = DEFAULT_2610.decider_repo
#: The decider commit the digests below were read at (its provenance.json).
DECIDER_REVISION = DEFAULT_2610.decider_revision
REVISION_FILE = "REVISION"
#: 2610 stores the adapter under ``lora/`` (with ``lora/adapter_config.json``); no root file.
ADAPTER_FILE = DEFAULT_2610.adapter_file
HEAD_FILE = DEFAULT_2610.head_file
PROVENANCE_FILE = DEFAULT_2610.provenance_file
#: 2610's full digests, from its provenance.json.
DECIDER_ADAPTER_SHA256 = DEFAULT_2610.adapter_sha256
DECIDER_HEAD_SHA256 = DEFAULT_2610.head_sha256
#: Default base pin. Upstream marks it "inferred: Hub main at training time; hosts did not pin
#: it", so it is our pin; a run config may override it.
DEFAULT_BASE_REVISION = DEFAULT_2610.base_revision
_HEX = re.compile(r"^[0-9a-f]{8,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def cache_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


@dataclass(frozen=True)
class RepoPin:
    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if not _HEX.match(self.revision):
            raise CliError(
                EXIT_USER_ERROR,
                f"{self.repo_id}: revision {self.revision!r} is not a pinned commit hash",
                "pin a hex commit hash (>= 8 chars) in the run config, never a branch name",
            )


@dataclass(frozen=True)
class StudentPins:
    base: RepoPin
    decider: RepoPin
    adapter_sha256: str
    head_sha256: str

    def __post_init__(self) -> None:
        for name in ("adapter_sha256", "head_sha256"):
            if not _SHA256.match(getattr(self, name)):
                raise CliError(
                    EXIT_USER_ERROR,
                    f"{name} must be a full 64-hex sha256, got {getattr(self, name)!r}",
                    "pin the full digest from provenance.json; prefixes are not accepted",
                )


def default_pins(
    *,
    decider_revision: str,
    base_revision: str | None = None,
    profile: StudentProfile | None = None,
) -> StudentPins:
    """Decider pins from a profile: digests built in, base revision overridable, commit required.

    ``profile=None`` means :data:`DEFAULT_2610`, read through this module's alias constants
    (``BASE_REPO``, ``DECIDER_ADAPTER_SHA256`` ...) so a test that patches them still bites.
    """
    if profile is None:
        return StudentPins(
            base=RepoPin(BASE_REPO, base_revision or DEFAULT_BASE_REVISION),
            decider=RepoPin(DECIDER_REPO, decider_revision),
            adapter_sha256=DECIDER_ADAPTER_SHA256,
            head_sha256=DECIDER_HEAD_SHA256,
        )
    return StudentPins(
        base=RepoPin(profile.base_repo, base_revision or profile.base_revision),
        decider=RepoPin(profile.decider_repo, decider_revision),
        adapter_sha256=profile.adapter_sha256,
        head_sha256=profile.head_sha256,
    )


def _fail(kind: str, what: str, detail: str) -> CliError:
    return CliError(
        EXIT_ENV_ERROR,
        f"{kind}: {what}: {detail}",
        "re-fetch the artifact into the cache at the pinned revision; do not edit the pin",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_revision(cache: Path, pin: RepoPin) -> Path:
    root = cache / cache_dirname(pin.repo_id)
    rev_file = root / REVISION_FILE
    if not rev_file.is_file():
        raise _fail("missing", str(rev_file.name), f"{pin.repo_id} not in cache {cache}")
    have = rev_file.read_text().strip().lower()
    want = pin.revision.lower()
    if not (have.startswith(want) or want.startswith(have)) or len(have) < 8:
        raise _fail("revision-mismatch", pin.repo_id, f"cache has {have}, pinned {want}")
    return root


def _check_file(root: Path, name: str, expected: str) -> str:
    path = root / name
    if not path.is_file():
        raise _fail("missing", name, f"not found under {root}")
    got = sha256_file(path)
    if got != expected.lower():
        raise _fail("hash-mismatch", name, f"sha256 {got} does not match expected {expected}")
    return got


def _read_provenance(root: Path) -> dict[str, Any]:
    path = root / PROVENANCE_FILE
    if not path.is_file():
        raise _fail("missing", PROVENANCE_FILE, f"not found under {root}")
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise _fail("bad-provenance", PROVENANCE_FILE, str(exc)) from exc
    if not isinstance(data, dict):
        raise _fail("bad-provenance", PROVENANCE_FILE, "not a JSON object")
    return data


def verify_pins(cache: Path, pins: StudentPins) -> dict[str, Any]:
    """Verify revisions and adapter/head hashes; raise CliError on any mismatch."""
    cache = Path(cache)
    _check_revision(cache, pins.base)
    root = _check_revision(cache, pins.decider)
    prov = _read_provenance(root)
    out: dict[str, Any] = {"status": "ok"}
    for key, fname, run_pin in (
        ("adapter_sha256", ADAPTER_FILE, pins.adapter_sha256),
        ("head_sha256", HEAD_FILE, pins.head_sha256),
    ):
        want = prov.get(key)
        if not isinstance(want, str) or not _SHA256.match(want.lower()):
            raise _fail("bad-provenance", PROVENANCE_FILE, f"missing or malformed {key}")
        if want.lower() != run_pin.lower():
            raise _fail("provenance", PROVENANCE_FILE, f"{key} {want} disagrees with pin {run_pin}")
        out[key] = _check_file(root, fname, want)
    out["base_revision"] = pins.base.revision
    out["decider_revision"] = pins.decider.revision
    return out
