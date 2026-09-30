"""Provenance convention: imported nvsh modules declare NVSH_PROVENANCE."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "jev_factory"
IMPORTED_PACKAGES = ("core", "data", "measure", "release", "backbones/causal_lm", "evals")
NAME = "NVSH_PROVENANCE"
COMMIT = "9debdc6"
LICENCE = "Apache-2.0"


def _provenance(path: Path):
    """Return the literal NVSH_PROVENANCE value, or None when absent."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == NAME for t in targets):
            try:
                return ast.literal_eval(value)
            except ValueError:
                return "not-a-literal"
    return None


def imported_modules(pkg: Path = PKG) -> list[Path]:
    found = set()
    for sub in IMPORTED_PACKAGES:
        base = pkg / sub
        if base.is_dir():
            found.update(
                p
                for p in base.rglob("*.py")
                if p.name != "__init__.py" and "__pycache__" not in p.parts
            )
    for p in pkg.rglob("*.py"):
        if "__pycache__" not in p.parts and NAME in p.read_text() and _provenance(p) is not None:
            found.add(p)
    return sorted(found)


def shape_errors(value) -> list[str]:
    if not isinstance(value, dict):
        return [f"{NAME} must be a literal dict"]
    errs = []
    if set(value) != {"upstream", "commit", "adaptations", "licence"}:
        errs.append(f"keys must be upstream/commit/adaptations/licence, got {sorted(value)}")
        return errs
    if not isinstance(value["upstream"], str) or not value["upstream"].strip():
        errs.append("upstream must be a non-empty string")
    if value["commit"] != COMMIT:
        errs.append(f"commit must be {COMMIT}")
    ad = value["adaptations"]
    if not (isinstance(ad, list) and ad and all(isinstance(a, str) and a for a in ad)):
        errs.append("adaptations must be a non-empty list of strings")
    if value["licence"] != LICENCE:
        errs.append(f"licence must be {LICENCE}")
    return errs


def nvsh_root() -> Path | None:
    for cand in (os.environ.get("NVSH_ROOT"), str(ROOT.parent / "nvsh")):
        if cand and (Path(cand) / "scripts").is_dir():
            return Path(cand)
    return None


def problems(pkg: Path = PKG, upstream_root: Path | None = None) -> list[str]:
    out = []
    for path in imported_modules(pkg):
        rel = path.relative_to(pkg)
        value = _provenance(path)
        if value is None:
            out.append(f"{rel}: missing {NAME}")
            continue
        errs = shape_errors(value)
        out.extend(f"{rel}: {e}" for e in errs)
        if not errs and upstream_root is not None:
            if not (upstream_root / value["upstream"]).is_file():
                out.append(f"{rel}: upstream file missing: {value['upstream']}")
    return out


def _good(upstream="scripts/lfm-finetune/x.py") -> str:
    return (
        f'{NAME} = {{"upstream": "{upstream}", "commit": "{COMMIT}", '
        f'"adaptations": ["a"], "licence": "{LICENCE}"}}\n'
    )


@pytest.mark.behavioral("o42")
def test_every_imported_module_has_valid_provenance():
    assert problems() == []


@pytest.mark.behavioral("o42")
def test_listed_upstream_files_exist():
    root = nvsh_root()
    if root is None:
        pytest.skip("no nvsh checkout (set NVSH_ROOT or place it at ../nvsh)")
    assert problems(upstream_root=root) == []


def test_fails_when_header_missing(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "__init__.py").write_text("")
    (tmp_path / "core" / "m.py").write_text("x = 1\n")
    assert problems(tmp_path) == ["core/m.py: missing NVSH_PROVENANCE"]


def test_fails_on_bad_shape(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "m.py").write_text(_good().replace(COMMIT, "abc1234"))
    assert any("commit must be" in e for e in problems(tmp_path))
    (tmp_path / "data" / "m.py").write_text(_good().replace('["a"]', "[]"))
    assert any("adaptations" in e for e in problems(tmp_path))
    (tmp_path / "data" / "m.py").write_text(f"{NAME} = dict(upstream='x')\n")
    assert any("literal dict" in e for e in problems(tmp_path))


def test_fails_when_upstream_missing_and_passes_when_present(tmp_path):
    pkg, up = tmp_path / "pkg", tmp_path / "nvsh"
    (pkg / "measure").mkdir(parents=True)
    (up / "scripts" / "lfm-finetune").mkdir(parents=True)
    (pkg / "measure" / "m.py").write_text(_good())
    assert any("upstream file missing" in e for e in problems(pkg, up))
    (up / "scripts" / "lfm-finetune" / "x.py").write_text("")
    assert problems(pkg, up) == []


def test_module_outside_imported_packages_is_validated_when_it_declares_it(tmp_path):
    (tmp_path / "factory").mkdir()
    (tmp_path / "factory" / "m.py").write_text(f"{NAME} = 3\n")
    assert any("literal dict" in e for e in problems(tmp_path))
    (tmp_path / "factory" / "plain.py").write_text("y = 1\n")
    assert not any("plain.py" in e for e in problems(tmp_path))
