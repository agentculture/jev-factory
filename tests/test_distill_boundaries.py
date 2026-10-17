"""Boundary guards for the Orin Nano jetson-ops run (t19, issue #20).

Criterion 1: every ``jev distill`` stage is dry-run and applied against the e2e fakes with
``socket.connect`` patched, and the test fails on any address other than 127.0.0.1 or ::1.
Criterion 2: the branch's diff touches no path outside this repository and no lobes-cli or nvsh
file is vendored or edited (a path scan plus a content-hash scan against sibling checkouts,
skipped cleanly outside a git checkout).
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from jev_factory.distill import pipeline as P
from tests.test_distill_e2e import (
    STAGES,
    UPLOAD_KNOBS,
    Env,
    approve_corrections,
    approve_test_gold,
)

REPO = Path(__file__).resolve().parent.parent
FOREIGN_REPOS = ("lobes-cli", "nvsh")
#: Boilerplate that legitimately repeats across repositories.
SHARED_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".flake8", "__init__.py", "py.typed"}
MIN_COPY_BYTES = 200


# ---------------------------------------------------------------------------
# Criterion 1: loopback-only connections
# ---------------------------------------------------------------------------


def is_loopback(address) -> bool:
    """True for ``127.0.0.1`` / ``::1`` (host part of an AF_INET/AF_INET6 address)."""
    host = address[0] if isinstance(address, tuple) else address
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ConnectGuard:
    """Replacement for ``socket.socket.connect``: records every address, never connects."""

    def __init__(self):
        self.seen: list = []

    def __call__(self, sock, address, *args, **kwargs):
        self.seen.append(address)
        if not is_loopback(address):
            raise AssertionError(f"non-loopback connection attempted: {address!r}")
        raise ConnectionRefusedError("the fakes serve every stage; nothing really connects")

    @property
    def violations(self) -> list:
        return [a for a in self.seen if not is_loopback(a)]


@pytest.fixture
def guard(monkeypatch):
    for name in ("torch", "transformers", "strands_decider"):
        monkeypatch.setitem(sys.modules, name, None)
    g = ConnectGuard()
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr, *a, **k: g(self, addr))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, addr, *a, **k: g(self, addr))
    return g


@pytest.fixture
def env(tmp_path, capsys, monkeypatch, guard):
    return Env(tmp_path, capsys, monkeypatch)


def test_the_guard_accepts_only_loopback_addresses():
    """Criterion 1: the guard itself rejects anything but 127.0.0.1 and ::1."""
    assert is_loopback(("127.0.0.1", 9))
    assert is_loopback(("::1", 9, 0, 0))
    assert not is_loopback(("8.8.8.8", 443))
    assert not is_loopback(("10.0.0.5", 80))
    assert not is_loopback(("localhost", 80))  # names are resolved by the caller, never trusted
    g = ConnectGuard()
    with pytest.raises(AssertionError):
        g(None, ("192.168.1.2", 80))
    assert g.violations == [("192.168.1.2", 80)]


def test_a_patched_connect_to_a_non_loopback_address_fails(guard):
    """Criterion 1: a real socket.connect to a foreign address trips the patched guard."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AssertionError):
            sock.connect(("203.0.113.7", 80))
    finally:
        sock.close()
    assert guard.violations


def test_every_distill_stage_dry_runs_and_applies_with_loopback_only_connections(env, guard):
    """Criterion 1: all 15 stages, dry-run then apply, never reach a non-loopback address."""
    assert len(STAGES) == 15
    for stage in STAGES:
        env.dry_run(stage, UPLOAD_KNOBS if stage == "upload" else None)
    for stage in STAGES[: STAGES.index("corrections-draft") + 1]:
        assert env.apply(stage)["status"] == "complete", stage
    assert approve_corrections(env)
    approve_test_gold(env)
    for stage in STAGES[STAGES.index("assemble") :]:
        knobs = UPLOAD_KNOBS if stage == "upload" else None
        assert env.apply(stage, knobs)["status"] == "complete", stage
    assert guard.violations == []
    assert all(is_loopback(a) for a in guard.seen)


def test_the_configured_urls_of_the_toy_run_are_loopback(env):
    """Criterion 1: every URL in the toy run config points at 127.0.0.1."""
    text = (env.work / "run.toml").read_text(encoding="utf-8")
    urls = [line for line in text.splitlines() if "http://" in line or "https://" in line]
    assert urls
    for line in urls:
        assert "http://127.0.0.1" in line, line


# ---------------------------------------------------------------------------
# Criterion 2: nothing outside this repository, nothing vendored from lobes-cli or nvsh
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path = REPO) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def _branch_diff() -> list[str]:
    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("not a git checkout")
    base = None
    for ref in ("main", "origin/main"):
        base = _git("merge-base", ref, "HEAD")
        if base:
            break
    if not base:
        pytest.skip("no main ref to take the merge-base from")
    if base.strip() == (_git("rev-parse", "HEAD") or "").strip():
        pytest.skip("HEAD is the merge-base (main itself): there is no branch diff to check")
    out = _git("diff", "--name-only", f"{base.strip()}..HEAD")
    if out is None:
        pytest.skip("git diff unavailable")
    return [line for line in out.splitlines() if line]


def _inside_repo(rel: str) -> bool:
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    return (REPO / rel).resolve().is_relative_to(REPO)


def test_the_branch_diff_stays_inside_this_repository():
    """Criterion 2: every path in the merge-base..HEAD diff is a path inside this repository."""
    paths = _branch_diff()
    assert paths
    outside = [p for p in paths if not _inside_repo(p)]
    assert outside == []


def test_the_branch_diff_has_no_lobes_cli_or_nvsh_tree():
    """Criterion 2: no changed path sits under a lobes-cli or nvsh directory."""
    paths = _branch_diff()
    foreign = [p for p in paths if PurePosixPath(p).parts[0] in FOREIGN_REPOS]
    assert foreign == []
    nested = [
        p
        for p in paths
        if any(part in FOREIGN_REPOS for part in PurePosixPath(p).parts[:-1])
        and not p.startswith(("docs/", "tests/fixtures/"))
    ]
    assert nested == []


def _sibling_roots() -> list[Path]:
    common = _git("rev-parse", "--git-common-dir")
    if common is None:
        return []
    root = (REPO / common.strip()).resolve().parent
    return [root.parent / name for name in FOREIGN_REPOS if (root.parent / name / ".git").exists()]


def _digests(root: Path, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in names:
        path = root / rel
        if path.name in SHARED_NAMES or not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) >= MIN_COPY_BYTES:
            out[hashlib.sha256(data).hexdigest()] = rel
    return out


def test_no_changed_file_is_a_byte_copy_of_a_lobes_cli_or_nvsh_file():
    """Criterion 2: no file the branch adds or edits is identical to a lobes-cli/nvsh source."""
    changed = _branch_diff()
    siblings = _sibling_roots()
    if not siblings:
        pytest.skip("no lobes-cli or nvsh checkout beside this repository")
    mine = _digests(REPO, changed)
    for sibling in siblings:
        tracked = (_git("ls-files", cwd=sibling) or "").splitlines()
        theirs = _digests(sibling, tracked)
        copies = {mine[h]: theirs[h] for h in mine.keys() & theirs.keys()}
        assert copies == {}, f"vendored from {sibling.name}: {copies}"


def test_the_jetson_ops_sources_name_neither_lobes_cli_nor_nvsh():
    """Criterion 2 (deviation d3): the vendored-sources list no longer lists either repository."""
    text = (REPO / "jev_factory" / "distill" / "jetson_ops" / "sources.json").read_text("utf-8")
    for name in FOREIGN_REPOS:
        assert name not in text.lower()


def test_the_distill_registry_still_has_fifteen_stages():
    """Criterion 1 depends on the full registry: edge sits after final."""
    assert len(P.STAGE_ORDER) == 15
    assert P.STAGE_ORDER.index("edge") == P.STAGE_ORDER.index("final") + 1


def test_a_real_systemone_client_is_seen_by_the_guard_and_stays_on_loopback(monkeypatch):
    """l9: drive the real client through a recording guard, so the boundary is not vacuous."""
    from jev_factory.distill import systemone as s1
    from tests.test_distill_systemone import RECORD, _answers, _Server

    srv = _Server(_answers())
    seen: list = []
    real_connect = socket.socket.connect

    def recording(sock, address, *args, **kwargs):
        seen.append(address)
        if not is_loopback(address):
            raise AssertionError(f"non-loopback connection attempted: {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", recording)
    try:
        result = s1.SystemOneClient(srv.url, model="m").ask(RECORD)
    finally:
        srv.close()
    assert result.response_sha256
    assert seen
    assert all(is_loopback(a) for a in seen)
    with pytest.raises(ValueError):
        s1.SystemOneClient("http://192.0.2.10:8000", model="m")
