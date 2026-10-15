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

BASE_REPO = "Qwen/Qwen3.5-2B-Base"
DECIDER_REPO = "StrandsAgents/strands-decider-2B-hobson-v21"
REVISION_FILE = "REVISION"
#: v21 stores the adapter under ``lora/`` (with ``lora/adapter_config.json``); no root file.
ADAPTER_FILE = "lora/adapter_model.safetensors"
HEAD_FILE = "head.safetensors"
PROVENANCE_FILE = "provenance.json"
#: v21's full digests, from its provenance.json.
V21_ADAPTER_SHA256 = "59be987f5eb664a7f74f11ef69383a0de526fe30c684932102984eb9c11f4028"
V21_HEAD_SHA256 = "0fc78684d7504d334082d6ac8e6cb7825b5eb0e230601181cd273757f9f6ebef"
#: Default base pin. Upstream marks it "inferred: Hub main at training time; hosts did not pin
#: it", so it is our pin; a run config may override it.
DEFAULT_BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
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
    *, decider_revision: str, base_revision: str = DEFAULT_BASE_REVISION
) -> StudentPins:
    """v21 pins: full digests built in, base revision overridable, decider commit required."""
    return StudentPins(
        base=RepoPin(BASE_REPO, base_revision),
        decider=RepoPin(DECIDER_REPO, decider_revision),
        adapter_sha256=V21_ADAPTER_SHA256,
        head_sha256=V21_HEAD_SHA256,
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
