"""Upload one scanned bundle to a PRIVATE hub repository and fetch it back.

This is the only module that talks to the Hugging Face Hub. It refuses, before any
hub call:

- a repository id outside the run's hub prefix (``hub_prefix`` of the run config, else
  the Domain's; lower-case letters, digits and ``-`` after it);
- a bundle that fails :func:`jev_factory.release.bundle.check_bundle` (a symlink, a
  missing ``calibration.json`` / ``gate.json`` / ``scorer-train.json``, an absolute
  path in GGUF metadata) or that ``scan.verify`` does not pass (no scan, findings, or a
  file changed after the scan).

Without ``apply`` it stops there and returns the plan: nothing is sent, no token is read.
With ``apply`` it reads the token from the variable the run config's ``hf_token_env``
names (never printed), creates the repository private (``exist_ok``), sets it private
again (an existing repository might not be), uploads the folder, downloads that exact
commit into a temporary folder next to the bundle, and compares the sha256 of every file
both ways plus the repository's own file list: any changed, missing or extra file fails
loudly (a ``.gitattributes`` the Hub adds itself is the one expected extra). Last it reads
the repository's ``private`` flag back and fails unless it is true. No code path here ever
makes a repository public: going public is a human decision outside this module.

``huggingface_hub`` is imported lazily, only when ``apply`` is set and no client was
passed in.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jev_factory.domain.model import Domain
from jev_factory.factory.config import RunConfig
from jev_factory.factory.secrets import read_secret, scrub
from jev_factory.release import scan
from jev_factory.release.bundle import bundle_problems

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/hub_upload.py",
    "commit": "9debdc6",
    "adaptations": [
        "ALLOWED_PREFIX constant -> hub_prefix from the run config / Domain",
        "FINAL=1 environment gate -> dry-run by default, --apply (apply=True) commits",
        "importlib path-loading of scan_bundle.py -> jev_factory.release.scan",
        "token read through factory.secrets.read_secret (config names the env var)",
        "bundle required files / symlinks / GGUF paths checked by release.bundle.bundle_problems",
        "the argparse main() is dropped; the upload verb calls upload()",
    ],
    "licence": "Apache-2.0",
}

REPO_TYPES = ("model", "dataset")
EXPECTED_REMOTE_EXTRAS = frozenset({".gitattributes"})
_DOWNLOAD_CACHE = ".cache"
_SUFFIX_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class UploadError(ValueError):
    """A refusal or a failed check; the message never holds the token."""


def effective_prefix(config: RunConfig | None, domain: Domain | None = None) -> str:
    """The hub prefix uploads go under: the run config's, else the Domain's."""
    prefix = str((config.get("hub_prefix") if config is not None else "") or "")
    if not prefix and domain is not None:
        prefix = domain.hub_prefix
    if not prefix:
        raise UploadError("no hub_prefix: set it in the run config or on the Domain")
    return prefix if prefix.endswith(("-", "/")) else prefix + "-"


def check_repo(repo: str, prefix: str) -> None:
    """Refuse a repository id that is not ``<prefix><suffix>``."""
    suffix = repo[len(prefix) :] if repo.startswith(prefix) else ""
    if not _SUFFIX_RE.match(suffix):
        raise UploadError(
            f"refusing {repo!r}: uploads go only to {prefix}<suffix>"
            " (lower-case letters, digits and '-')"
        )


def digests(folder: Path, *, skip_cache: bool = False) -> dict[str, str]:
    """``relative POSIX path -> sha256`` for every regular file under *folder*."""
    found: dict[str, str] = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if skip_cache and rel.split("/", 1)[0] == _DOWNLOAD_CACHE:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        found[rel] = digest.hexdigest()
    return found


def check_inventory(local: Path, remote_files: list[str]) -> list[str]:
    """Every file the repository holds that the bundle does not, or the reverse."""
    mine = set(digests(local))
    theirs = set(remote_files)
    extra = theirs - mine - EXPECTED_REMOTE_EXTRAS
    problems = [f"{rel}: in the repository but not in the bundle" for rel in sorted(extra)]
    problems += [f"{rel}: missing from the repository" for rel in sorted(mine - theirs)]
    return problems


def compare(local: Path, fetched: Path) -> list[str]:
    """Every difference between the local bundle and the fetched copy, one line each."""
    mine = digests(local)
    theirs = digests(fetched, skip_cache=True)
    problems = []
    for rel in sorted(set(mine) | set(theirs)):
        if rel not in theirs:
            problems.append(f"{rel}: missing from the fetched copy")
        elif rel not in mine:
            if rel not in EXPECTED_REMOTE_EXTRAS:
                problems.append(f"{rel}: in the repository but not in the bundle")
        elif mine[rel] != theirs[rel]:
            problems.append(f"{rel}: sha256 differs")
    return problems


def _set_private(api: Any, repo: str, repo_type: str) -> None:
    """Make *repo* private: ``update_repo_settings`` (huggingface_hub 1.x), or the older
    ``update_repo_visibility``. Only ever ``private=True``."""
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo, private=True, repo_type=repo_type)
    else:
        api.update_repo_visibility(repo, private=True, repo_type=repo_type)


def check_bundle_for_upload(bundle: Path) -> None:
    """Every local refusal, before any hub call."""
    if not bundle.is_dir():
        raise UploadError(f"{bundle} is not a folder")
    problems = bundle_problems(bundle)
    if problems:
        raise UploadError(f"refusing {bundle}:\n  " + "\n  ".join(problems))
    reason = scan.verify(bundle)
    if reason is not None:
        raise UploadError(f"scan verify failed for {bundle}: {reason}")


def upload(
    *,
    bundle: Path,
    repo: str,
    repo_type: str,
    config: RunConfig,
    domain: Domain | None = None,
    apply: bool = False,
    hub: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Upload *bundle* to *repo* privately and check it; return a summary.

    Without *apply* the checks run and the plan is returned, and nothing is sent."""
    prefix = effective_prefix(config, domain)
    check_repo(repo, prefix)
    if repo_type not in REPO_TYPES:
        raise UploadError(f"unknown repo type {repo_type!r} (one of {', '.join(REPO_TYPES)})")
    check_bundle_for_upload(bundle)
    files = len(digests(bundle))
    if not apply:
        return {
            "applied": False,
            "repo": repo,
            "repo_type": repo_type,
            "files": files,
            "private": True,
            "note": "dry run: nothing was sent; pass --apply to upload",
        }

    secret = read_secret(config, "hf_token_env", environ=environ)
    token = secret.reveal()
    if hub is None:
        hub = importlib.import_module("huggingface_hub")
    try:
        return _apply(hub, token, bundle, repo, repo_type, files)
    except UploadError:
        raise
    except Exception as exc:  # noqa: BLE001 -- a hub failure must not echo the token
        raise UploadError(scrub(f"{type(exc).__name__}: {exc}", [secret])) from None


def _apply(hub: Any, token: str, bundle: Path, repo: str, repo_type: str, files: int):
    api = hub.HfApi(token=token)
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    _set_private(api, repo, repo_type)
    commit = api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo,
        repo_type=repo_type,
        commit_message=f"jev-factory: upload {bundle.name}",
    )
    revision = getattr(commit, "oid", None)

    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=bundle.parent) as fetched:
        hub.snapshot_download(
            repo, repo_type=repo_type, revision=revision, token=token, local_dir=fetched
        )
        problems = compare(bundle, Path(fetched))
        problems += check_inventory(
            bundle, api.list_repo_files(repo, repo_type=repo_type, revision=revision)
        )
    if problems:
        raise UploadError(
            f"the copy fetched back from {repo} differs from {bundle}:\n  " + "\n  ".join(problems)
        )

    private = getattr(api.repo_info(repo, repo_type=repo_type), "private", None)
    if private is not True:
        raise UploadError(f"{repo} is not private (the Hub reports private={private})")
    return {
        "applied": True,
        "repo": repo,
        "repo_type": repo_type,
        "revision": revision,
        "files": files,
        "private": private,
    }
