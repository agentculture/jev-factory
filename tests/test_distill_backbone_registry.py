"""d1/d2 (d-t16): the distillery reaches its student only through the backbone registry.

* No module in ``jev_factory/distill`` imports ``jev_factory.backbones.pointer`` directly
  (absolute or relative); the stages call ``registry.get(profile.backbone)`` and the hooks of
  :class:`~jev_factory.backbones.base.Backbone`. The allow-list is empty.
* The shared layer (``jev_factory.wire``, ``jev_factory.runtime``) imports neither the
  distillery nor a backbone, and the old distill import paths re-export the very same objects.
* The Backbone protocol carries the layout, config-file, serve, train and quantize hooks;
  the pointer backbone implements them by delegating to its modules, and the causal-LM
  backbone refuses them (``NotImplementedError``) rather than guessing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "jev_factory"
POINTER = "jev_factory.backbones.pointer"
#: Distill modules allowed to import the pointer backbone directly. Keep it empty.
ALLOWED_POINTER_IMPORTERS: set[str] = set()

HOOKS = (
    "layout",
    "config_files",
    "serve_argv",
    "server",
    "plan_distill_train",
    "run_distill_train",
    "trainer",
    "quantizer",
)


def _module_name(path: Path, top: Path) -> str:
    rel = path.relative_to(top).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path, top: Path) -> set[str]:
    """Every module (and ``module.name``) *path* imports, relative imports resolved."""
    module = _module_name(path, top)
    is_pkg = path.name == "__init__.py"
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
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


def _hits(package: str, prefixes: tuple[str, ...], top: Path = ROOT.parent) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((top / package.replace(".", "/")).rglob("*.py")):
        bad = sorted(
            name
            for name in _imports(path, top)
            if any(name == p or name.startswith(p + ".") for p in prefixes)
        )
        if bad:
            found[str(path.relative_to(top))] = bad
    return found


def test_no_distill_module_imports_the_pointer_backbone_directly():
    hits = _hits("jev_factory.distill", (POINTER,))
    assert set(hits) <= ALLOWED_POINTER_IMPORTERS, hits


def test_the_ast_check_sees_a_relative_or_from_package_pointer_import(tmp_path):
    """The guard is not vacuous: both spellings of a pointer import are caught."""
    pkg = tmp_path / "jev_factory" / "distill"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("from ..backbones import pointer\n")
    (pkg / "b.py").write_text("from jev_factory.backbones.pointer import layout\n")
    (pkg / "c.py").write_text("from jev_factory.backbones import registry\n")
    hits = _hits("jev_factory.distill", (POINTER,), top=tmp_path)
    assert sorted(hits) == ["jev_factory/distill/a.py", "jev_factory/distill/b.py"]


def test_the_pointer_backbone_imports_no_distill_module():
    assert _hits("jev_factory.backbones.pointer", ("jev_factory.distill",)) == {}


@pytest.mark.parametrize("package", ["jev_factory.wire", "jev_factory.runtime"])
def test_the_shared_layer_imports_neither_the_distillery_nor_a_backbone(package):
    assert (ROOT.parent / package.replace(".", "/") / "__init__.py").is_file()
    assert _hits(package, ("jev_factory.distill", "jev_factory.backbones")) == {}


def test_old_distill_paths_re_export_the_shared_objects():
    from jev_factory.backbones.pointer import train_distill as td
    from jev_factory.distill import envs as denvs
    from jev_factory.distill import prereg as dp
    from jev_factory.distill import systemone as dsys
    from jev_factory.runtime import envs as renvs
    from jev_factory.runtime import ids, prereg_lock
    from jev_factory.wire import systemone as wsys

    for name in (
        "SystemOneClient",
        "CallResult",
        "QuestionResult",
        "build_body",
        "parse_response",
        "SYSTEMONE_PATH",
        "SUM_TOLERANCE",
        "MODEL_HEADER",
    ):
        assert getattr(dsys, name) is getattr(wsys, name), name
    assert denvs.EnvSpec is renvs.EnvSpec
    assert denvs.ROLES is renvs.ROLES
    assert td.id_digest is ids.id_digest
    assert dp.LOCK_REL == prereg_lock.LOCK_REL


def test_the_backbone_protocol_carries_the_distill_hooks():
    from jev_factory.backbones import base

    for member in HOOKS:
        assert hasattr(base.Backbone, member), member


def test_the_pointer_backbone_hooks_delegate_to_its_modules():
    from jev_factory.backbones import registry
    from jev_factory.backbones.pointer import layout, quantized, serve, train_distill

    pointer = registry.get("pointer")
    assert pointer.layout() is layout
    assert pointer.server() is serve
    assert pointer.trainer() is train_distill
    assert pointer.quantizer() is quantized
    names = pointer.config_files()
    assert names["served_config"] == layout.CONFIG
    assert names["manifest"] == layout.MANIFEST
    assert names["provenance"] == layout.PROVENANCE
    assert names["train_config"] == train_distill.CONFIG_FILE


def test_plan_and_run_distill_train_go_through_the_module_attributes(monkeypatch):
    """Delegation looks the trainer up at call time, so a patched module is what runs."""
    from jev_factory.backbones import registry
    from jev_factory.backbones.pointer import train_distill as td

    seen = {}
    monkeypatch.setattr(td, "plan_train", lambda **kw: seen.setdefault("plan", kw))
    monkeypatch.setattr(td, "run_train", lambda plan, **kw: seen.setdefault("run", (plan, kw)))
    pointer = registry.get("pointer")
    assert pointer.plan_distill_train(a=1) == {"a": 1}
    assert pointer.run_distill_train({"p": 1}, apply=False) == ({"p": 1}, {"apply": False})


def test_the_serve_argv_hook_is_the_pointer_serve_argv(monkeypatch):
    from jev_factory.backbones import registry
    from jev_factory.backbones.pointer import serve

    monkeypatch.setattr(serve, "serve_argv", lambda python, bundle, **kw: [python, bundle, kw])
    out = registry.get("pointer").serve_argv("py", "b", port=9, host="127.0.0.1")
    assert out == ["py", "b", {"port": 9, "host": "127.0.0.1"}]


@pytest.mark.parametrize("member", HOOKS)
def test_the_causal_lm_backbone_refuses_the_distill_hooks(member):
    from jev_factory.backbones import registry

    adapter = registry.get("causal_lm")
    with pytest.raises(NotImplementedError, match=member):
        if member == "serve_argv":
            adapter.serve_argv("py", "b", port=1)
        elif member == "run_distill_train":
            adapter.run_distill_train({})
        else:
            getattr(adapter, member)()


def test_the_distill_student_helper_resolves_the_profile_backbone():
    import dataclasses

    from jev_factory.backbones import registry
    from jev_factory.distill import student
    from jev_factory.distill.profile import DEFAULT_2610

    assert student.backbone() is registry.get(DEFAULT_2610.backbone)
    other = dataclasses.replace(DEFAULT_2610, backbone="causal_lm")
    assert student.backbone(other) is registry.get("causal_lm")
