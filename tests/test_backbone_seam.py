"""The backbone seam: a registry of lazily imported adapters, and a distribution-only core.

The invariant contract is a bounded candidate set, a calibrated distribution, the gate, then
grounding. Modules that only need a distribution must not import a backbone adapter.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = ("jev_factory.backbones.causal_lm", "jev_factory.backbones.pointer")
DISTRIBUTION_ONLY = (
    "jev_factory.core.calibration",
    "jev_factory.core.gate",
    "jev_factory.core.sweep_gate",
    "jev_factory.distill.metrics",
    "jev_factory.decide",
)


def _find(name: str):
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


def _source_files(spec) -> list[Path]:
    if spec.submodule_search_locations:
        return sorted(Path(spec.submodule_search_locations[0]).rglob("*.py"))
    return [Path(spec.origin)]


def _imported(path: Path, module: str, is_pkg: bool) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = module.split(".") if is_pkg else module.split(".")[:-1]
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


def _closure(root: str) -> set[str]:
    """Every jev_factory module reachable from *root* by static imports."""
    seen: set[str] = set()
    todo = [root]
    found: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen or not name.startswith("jev_factory"):
            continue
        seen.add(name)
        spec = _find(name)
        if spec is None or spec.origin is None:
            continue
        for path in _source_files(spec):
            rel = path.relative_to(
                Path(spec.origin).parent
                if not spec.submodule_search_locations
                else Path(spec.submodule_search_locations[0])
            )
            mod = name if str(rel) in ("__init__.py", Path(spec.origin).name) else None
            if mod is None:
                mod = ".".join([name, *rel.with_suffix("").parts])
            imports = _imported(path, mod, path.name == "__init__.py")
            found |= imports
            todo.extend(imports)
    return found


@pytest.mark.parametrize("module", DISTRIBUTION_ONLY)
def test_distribution_only_modules_do_not_import_a_backbone(module):
    if _find(module) is None:
        pytest.skip(f"{module} not present in this tree yet")
    imported = _closure(module)
    bad = sorted(i for i in imported if any(i == f or i.startswith(f + ".") for f in FORBIDDEN))
    assert not bad, f"{module} reaches a backbone adapter: {bad}"


def test_distill_metrics_is_checked_once_present():
    if _find("jev_factory.distill.metrics") is None:
        pytest.skip("distill.metrics lands with another task")
    assert "jev_factory.distill.metrics" in DISTRIBUTION_ONLY


def test_registry_registers_both_backbones_without_importing_them():
    for f in FORBIDDEN:
        sys.modules.pop(f, None)
    from jev_factory.backbones import registry

    assert {"causal_lm", "pointer"} <= set(registry.names())
    assert not any(f in sys.modules for f in FORBIDDEN), "registry imported an adapter eagerly"


def test_registry_unknown_name_is_a_clear_error():
    from jev_factory.backbones import registry

    with pytest.raises(KeyError, match="nope"):
        registry.get("nope")


def test_registry_loads_lazily_via_importlib(monkeypatch):
    from jev_factory.backbones import registry

    calls = []
    real = importlib.import_module

    def spy(name, *a, **k):
        calls.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(registry.importlib, "import_module", spy)
    sentinel = type("Fake", (), {"name": "causal_lm"})
    monkeypatch.setitem(registry._LOADERS, "fake", ("json", "JSONDecoder"))
    assert registry.get("fake") is real("json").JSONDecoder
    assert calls == ["json"]
    del sentinel


def test_protocol_declares_the_adapter_surface():
    from jev_factory.backbones import base

    for member in (
        "render",
        "predict_distributions",
        "train_plan",
        "merge_verify",
        "export",
        "quantize_candidates",
    ):
        assert hasattr(base.Backbone, member), member
