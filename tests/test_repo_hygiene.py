"""Repo hygiene that CI enforces, pinned as tests (o43).

CI runs the hygiene scripts itself; these tests make the same checks part of
the local suite and pin the CI wiring, so a change that would fail (or silently
stop running) a hygiene gate fails here first.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PORTABILITY = ROOT / ".claude" / "skills" / "cicd" / "scripts" / "portability-lint.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
#: Cited verbatim from elsewhere or written as planning records, so not ours to reword.
PORTABILITY_EXEMPT = (".claude/skills/", ".devague/", "docs/specs/", "docs/plans/")


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout  # nosec B607


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


@pytest.mark.behavioral("o43")
def test_harness_smoke_config_stage_passes():
    proc = _run([sys.executable, "scripts/harness-smoke.py", "--stage", "config"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.behavioral("o43")
def test_scan_secrets_passes():
    proc = _run([sys.executable, "scripts/scan-secrets.py"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.behavioral("o43")
@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_portability_lint_passes_on_our_own_files(tmp_path):
    """Lint every tracked file we author, in a scratch repo (the script lints a git tree)."""
    tracked = [
        f
        for f in _git("ls-files", "-z").split("\0")
        if f and not f.startswith(PORTABILITY_EXEMPT) and (ROOT / f).is_file()
    ]
    for f in tracked:
        dest = tmp_path / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    _git("init", "-q", cwd=tmp_path)
    _git("add", "-A", cwd=tmp_path)
    proc = subprocess.run(
        ["bash", str(PORTABILITY), "--all"], cwd=tmp_path, capture_output=True, text=True
    )  # nosec B603 B607
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.behavioral("o43")
def test_portability_lint_flags_a_leaked_path(tmp_path):
    leak = "/ho" + "me/someone/project/x"
    (tmp_path / "doc.md").write_text(f"see {leak}\n", encoding="utf-8")
    _git("init", "-q", cwd=tmp_path)
    _git("add", "-A", cwd=tmp_path)
    proc = subprocess.run(
        ["bash", str(PORTABILITY), "--all"], cwd=tmp_path, capture_output=True, text=True
    )  # nosec B603 B607
    assert proc.returncode == 1


@pytest.mark.behavioral("o43")
def test_ci_lint_job_runs_portability_lint_and_scan_secrets():
    text = WORKFLOW.read_text(encoding="utf-8")
    lint = text[text.index("\n  lint:") : text.index("\n  harness-smoke:")]
    assert "portability-lint.sh" in lint
    assert "scripts/scan-secrets.py" in lint
    assert "harness-smoke.py" in text


@pytest.mark.behavioral("o43")
def test_there_is_no_root_agents_md():
    assert not (ROOT / "AGENTS.md").exists()
    assert not any(p.name.lower() == "agents.md" for p in ROOT.iterdir())


@pytest.mark.behavioral("o43")
def test_coverage_floor_is_sixty_percent_on_jev_factory():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["coverage"]
    assert cfg["report"]["fail_under"] >= 60
    assert cfg["run"]["source"] == ["jev_factory"]
    assert "--cov=jev_factory" in WORKFLOW.read_text(encoding="utf-8")


@pytest.mark.behavioral("o43")
def test_no_module_level_pragma_no_cover_in_jev_factory():
    """A module-level pragma would drop a whole def, class or module from the coverage floor."""
    pattern = re.compile(r"^\S.*pragma:\s*no cover", re.IGNORECASE)
    offenders = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in sorted((ROOT / "jev_factory").rglob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.match(line)
    ]
    assert offenders == []


def test_the_pragma_scan_sees_a_module_level_pragma():
    pattern = re.compile(r"^\S.*pragma:\s*no cover", re.IGNORECASE)
    assert pattern.match("def f():  # pragma: no cover")
    assert not pattern.match("    x = 1  # pragma: no cover")


def test_pytest_ignores_the_deepeval_plugin():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "-p no:deepeval" in cfg["tool"]["pytest"]["ini_options"]["addopts"]
