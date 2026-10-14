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


def _leaks(imported) -> list[str]:
    return sorted(i for i in imported if any(i == f or i.startswith(f + ".") for f in FORBIDDEN))


@pytest.mark.parametrize("module", DISTRIBUTION_ONLY)
def test_distribution_only_modules_do_not_import_a_backbone(module):
    if _find(module) is None:
        pytest.skip(f"{module} not present in this tree yet")
    imported = _closure(module)
    bad = _leaks(imported)
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


def test_get_causal_lm_returns_a_backbone():
    from jev_factory.backbones import base, registry

    adapter = registry.get("causal_lm")
    assert isinstance(adapter, base.Backbone)
    assert adapter.name == "causal_lm"
    assert tuple(adapter.quantize_candidates({})) == ("gguf-q4_k_m",)
    with pytest.raises(NotImplementedError, match="merge_verify"):
        adapter.merge_verify("a", "b")
    with pytest.raises(NotImplementedError, match="export"):
        adapter.export("a", "b")


def test_causal_lm_adapter_delegates_to_scorer_and_train():
    from jev_factory.backbones import registry
    from jev_factory.backbones.causal_lm import scorer, train

    adapter = registry.get("causal_lm")
    calls = {}

    class Pred:
        probabilities = {"a": 0.75, "b": 0.25}

    def fake_score(domain, top_k, prompt, request_text, *, offered=None, **_):
        calls["score"] = (domain, top_k, prompt, request_text, offered)
        return Pred()

    def fake_messages(domain, request, offered=None, **_):
        calls["messages"] = (domain, request, offered)
        return [{"role": "user", "content": request}]

    def fake_plan(**kw):
        calls["plan"] = kw
        return {"stage": "train"}

    orig = (scorer.score, scorer.prompt_messages, train.plan_train)
    try:
        scorer.score, scorer.prompt_messages, train.plan_train = (
            fake_score,
            fake_messages,
            fake_plan,
        )
        assert adapter.render("dom", "hi", ["a", "b"])[0]["content"] == "hi"
        assert calls["messages"] == ("dom", "hi", ["a", "b"])
        row = {"domain": "dom", "prompt": "P", "request_text": "hi"}
        assert adapter.predict_distributions("topk", [row], [["a", "b"]]) == [
            {"a": 0.75, "b": 0.25}
        ]
        assert calls["score"] == ("dom", "topk", "P", "hi", ["a", "b"])
        assert adapter.train_plan({"run_dir": "r"}) == {"stage": "train"}
        assert calls["plan"] == {"run_dir": "r"}
    finally:
        scorer.score, scorer.prompt_messages, train.plan_train = orig


def test_get_pointer_names_the_missing_module(monkeypatch):
    from jev_factory.backbones import registry

    missing = "jev_factory.backbones.no_such_backbone_xyz"
    monkeypatch.setitem(registry._LOADERS, "pointer", (missing, "BACKBONE"))
    with pytest.raises(registry.BackboneUnavailable, match=missing) as info:
        registry.get("pointer")
    assert isinstance(info.value, LookupError)
    assert not isinstance(info.value, ImportError)


def test_registry_does_not_mask_a_missing_dependency_of_a_present_module(monkeypatch, tmp_path):
    from jev_factory.backbones import registry

    (tmp_path / "zz_present_mod.py").write_text("import zz_not_installed_dep\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry._LOADERS, "x", ("zz_present_mod", "BACKBONE"))
    with pytest.raises(ModuleNotFoundError, match="zz_not_installed_dep"):
        registry.get("x")


def test_negative_control_closure_flags_leaking_modules(tmp_path, monkeypatch):
    """Throwaway 'distribution-only' modules: direct and transitive leaks flagged, clean not."""
    pkg = tmp_path / "jev_factory_fake"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "direct.py").write_text("import jev_factory.backbones.causal_lm\n")
    (pkg / "direct_from.py").write_text("from jev_factory.backbones.causal_lm import scorer\n")
    (pkg / "mid.py").write_text("from . import direct\n")
    clean_pkg = tmp_path / "jev_factory_fake_clean"  # a package walk covers every file in it
    clean_pkg.mkdir()
    (clean_pkg / "__init__.py").write_text("")
    (clean_pkg / "helper.py").write_text("X = 1\n")
    (clean_pkg / "clean.py").write_text("import json\nfrom . import helper\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    assert _leaks(_closure("jev_factory_fake.direct"))
    assert _leaks(_closure("jev_factory_fake.direct_from"))
    transitive = _leaks(_closure("jev_factory_fake.mid"))
    assert "jev_factory.backbones.causal_lm" in transitive
    assert _leaks(_closure("jev_factory_fake_clean.clean")) == []
