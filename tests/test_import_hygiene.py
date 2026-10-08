"""Import-hygiene tests: the base install stays light and self-contained."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "jev_factory"
BLOCKED = (
    "torch",
    "transformers",
    "peft",
    "unsloth",
    "llmcompressor",
    "huggingface_hub",
    "datasets",
    "deepeval",
)
FORBIDDEN_ROOTS = {"nvsh", "jetson_skills"}

_IMPORT_ALL = """
import importlib, pkgutil, sys
for name in {blocked!r}:
    sys.modules[name] = None  # any import of it raises ImportError
import jev_factory
failed = []
count = 0
for info in pkgutil.walk_packages(jev_factory.__path__, "jev_factory."):
    try:
        importlib.import_module(info.name)
        count += 1
    except BaseException as exc:
        failed.append(f"{{info.name}}: {{exc!r}}")
if failed:
    print("\\n".join(failed))
    sys.exit(1)
print(count)
"""


def _modules() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.behavioral("o5")
def test_dependencies_stay_empty():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["dependencies"] == []


@pytest.mark.behavioral("o5")
def test_import_every_module_with_heavy_packages_blocked():
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL.format(blocked=BLOCKED)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert int(proc.stdout.strip().splitlines()[-1]) >= 1


@pytest.mark.behavioral("o5")
def test_heavy_blocking_actually_blocks():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "try:\n import torch\nexcept ImportError:\n sys.exit(0)\nsys.exit(1)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    roots.add(arg.value.split(".")[0])
    return roots


def _uses_spec_from_file_location(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "spec_from_file_location":
            return True
        if isinstance(node, ast.Name) and node.id == "spec_from_file_location":
            return True
        if isinstance(node, ast.alias) and node.name == "spec_from_file_location":
            return True
    return False


def _violations(source: str) -> tuple[set[str], bool]:
    tree = ast.parse(source)
    return _imported_roots(tree) & FORBIDDEN_ROOTS, _uses_spec_from_file_location(tree)


@pytest.mark.behavioral("o44")
@pytest.mark.behavioral("o5")
def test_no_module_imports_nvsh_or_jetson_skills():
    # Every module is checked, so a transitive path through jev_factory would also
    # surface as a direct import in some module.
    bad = {}
    for path in _modules():
        roots, _ = _violations(path.read_text())
        if roots:
            bad[str(path.relative_to(ROOT))] = sorted(roots)
    assert not bad, bad


@pytest.mark.behavioral("o45")
def test_no_module_uses_spec_from_file_location():
    bad = [str(p.relative_to(ROOT)) for p in _modules() if _violations(p.read_text())[1]]
    assert not bad, bad


@pytest.mark.behavioral("o44")
@pytest.mark.parametrize(
    "source",
    [
        "import nvsh",
        "from nvsh.ops import table",
        "import jetson_skills.x",
        "import importlib\nimportlib.import_module('nvsh.ops')",
        "__import__('nvsh')",
    ],
)
def test_detector_flags_forbidden_imports(source):
    assert _violations(source)[0]


@pytest.mark.behavioral("o45")
@pytest.mark.parametrize(
    "source",
    [
        "import importlib.util\nimportlib.util.spec_from_file_location('a', 'b')",
        "from importlib.util import spec_from_file_location",
    ],
)
def test_detector_flags_spec_from_file_location(source):
    assert _violations(source)[1]


def test_detector_passes_clean_source():
    assert _violations("import json\nfrom jev_factory import cli") == (set(), False)
