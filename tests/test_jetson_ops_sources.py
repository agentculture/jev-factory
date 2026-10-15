"""Jetson-ops source registry (t9): pins, licences, attribution, fail-closed refusals."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill.jetson_ops import sources

EXPECTED = {
    "jetson-containers-nv",
    "jetson-ai-lab",
    "jetson-device-skills",
    "jetson-bsp-skills",
    "lobes-cli",
    "nvsh",
}


def _raw() -> dict:
    return json.loads(sources.REGISTRY_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _mutated(tmp_path: Path, edit) -> Path:
    data = copy.deepcopy(_raw())
    edit(data["sources"][0])
    return _write(tmp_path, data)


def test_registry_pins_every_source_with_full_commit():
    """AC1: each source is pinned by repo URL and full commit."""
    loaded = sources.load_sources()
    assert set(loaded) == EXPECTED
    for src in loaded.values():
        assert len(src.commit) == 40
        assert src.repo_url.startswith("https://github.com/")


def test_nvidia_skills_split_licences_and_cc_by_attribution():
    """AC1: skills docs are CC-BY-4.0, code Apache-2.0, with attribution text."""
    loaded = sources.load_sources()
    for sid in ("jetson-device-skills", "jetson-bsp-skills"):
        src = loaded[sid]
        assert src.licence_for("docs") == "CC-BY-4.0"
        assert src.licence_for("code") == "Apache-2.0"
        assert "CC-BY-4.0" in src.attribution
        assert "NVIDIA" in src.attribution


def test_other_licences_match_the_repos():
    """AC1: containers and ai-lab are MIT; lobes-cli and nvsh Apache-2.0."""
    loaded = sources.load_sources()
    assert loaded["jetson-containers-nv"].licence_for("code") == "MIT"
    assert loaded["jetson-ai-lab"].licence_for("docs") == "MIT"
    assert loaded["lobes-cli"].licence_for("docs") == "Apache-2.0"
    assert loaded["nvsh"].licence_for("code") == "Apache-2.0"


def test_local_clone_pins_match_when_clones_exist():
    """AC1: pinned commits of lobes-cli and nvsh exist in their local clones."""
    home = Path.home() / "git"
    loaded = sources.load_sources()
    for sid, name in (("lobes-cli", "lobes-cli"), ("nvsh", "nvsh")):
        clone = home / name
        if not clone.is_dir():
            pytest.skip("no local clone")
        res = subprocess.run(
            ["git", "-C", str(clone), "cat-file", "-t", loaded[sid].commit],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.stdout.strip() == "commit"


def test_record_terms_carry_licence_and_attribution():
    """AC2: per source, the licence and attribution a drafted record must carry."""
    terms = sources.record_terms("jetson-bsp-skills", "docs")
    assert terms["licence"] == "CC-BY-4.0"
    assert "CC-BY-4.0" in terms["attribution"]
    assert terms["commit"] == sources.load_sources()["jetson-bsp-skills"].commit


def test_unknown_licence_is_refused(tmp_path):
    """AC2: unknown licence fails closed."""
    path = _mutated(tmp_path, lambda s: s["licences"].update(code="GPL-3.0"))
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_missing_commit_is_refused(tmp_path):
    """AC2: missing commit fails closed."""
    path = _mutated(tmp_path, lambda s: s.pop("commit"))
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_short_commit_is_refused(tmp_path):
    """AC2: an abbreviated commit is not a pin."""
    path = _mutated(tmp_path, lambda s: s.update(commit="c8bad15f"))
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_missing_attribution_is_refused(tmp_path):
    """AC2: missing attribution fails closed."""
    path = _mutated(tmp_path, lambda s: s.pop("attribution"))
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_cc_by_attribution_must_name_licence(tmp_path):
    """AC2: CC-BY-4.0 content needs attribution that names the licence."""
    data = copy.deepcopy(_raw())
    nv = next(s for s in data["sources"] if s["id"] == "jetson-device-skills")
    nv["attribution"] = "From NVIDIA."
    path = _write(tmp_path, data)
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_missing_path_class_licence_is_refused(tmp_path):
    """AC2: both code and docs licences are required."""
    path = _mutated(tmp_path, lambda s: s["licences"].pop("docs"))
    with pytest.raises(CliError):
        sources.load_sources(path)


def test_unreadable_registry_is_refused(tmp_path):
    """AC2: a missing or malformed file fails closed."""
    missing = tmp_path / "nope.json"
    with pytest.raises(CliError):
        sources.load_sources(missing)


def test_unknown_source_and_path_class_are_refused():
    """AC2: record_terms refuses unknown sources and path classes."""
    with pytest.raises(CliError):
        sources.record_terms("nope", "docs")
    with pytest.raises(CliError):
        sources.record_terms("nvsh", "binary")
