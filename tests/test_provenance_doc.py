"""docs/nvsh-import-provenance.md is generated from the NVSH_PROVENANCE headers (o42)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.test_provenance import COMMIT, LICENCE, imported_modules, nvsh_root

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "gen-provenance-doc.py"
DOC = ROOT / "docs" / "nvsh-import-provenance.md"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_provenance_doc_under_test", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _generator()


@pytest.mark.behavioral("o42")
def test_the_committed_doc_is_what_the_headers_generate():
    assert DOC.read_text(encoding="utf-8") == gen.render(
        gen.collect()
    ), "docs/nvsh-import-provenance.md is stale: run scripts/gen-provenance-doc.py"
    assert gen.main(["--check"]) == 0


@pytest.mark.behavioral("o42")
def test_every_imported_module_is_listed_and_exists():
    text = DOC.read_text(encoding="utf-8")
    listed = {module for module, _ in gen.collect()}
    for path in imported_modules():
        rel = path.relative_to(ROOT).as_posix()
        assert rel in listed, f"{rel} is imported from nvsh but missing from the doc"
        assert f"| `{rel}` |" in text
    for module in listed:
        assert (ROOT / module).is_file(), f"{module} is listed but does not exist"


@pytest.mark.behavioral("o42")
def test_every_entry_names_its_nvsh_path_commit_adaptations_and_licence():
    entries = gen.collect()
    assert entries
    text = DOC.read_text(encoding="utf-8")
    assert COMMIT in text and LICENCE in text
    for module, value in entries:
        assert value["upstream"].strip(), module
        assert value["commit"] == COMMIT, module
        assert value["licence"] == LICENCE, module
        assert value["adaptations"], module
        assert f"| `{module}` | `{value['upstream']}` | `{COMMIT}` | {LICENCE} |" in text
        for item in value["adaptations"]:
            assert f"- {item}" in text, module


@pytest.mark.behavioral("o42")
def test_the_scorer_path_and_the_evals_modules_are_all_listed():
    listed = {module for module, _ in gen.collect()}
    for package in ("core", "data", "measure", "release", "backbones/causal_lm", "evals"):
        files = [
            p
            for p in (ROOT / "jev_factory" / package).rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "__init__.py"
        ]
        assert files, package
        for p in files:
            assert p.relative_to(ROOT).as_posix() in listed


@pytest.mark.behavioral("o42")
def test_every_listed_upstream_file_exists_in_nvsh():
    root = nvsh_root()
    if root is None:
        pytest.skip("no nvsh checkout (set NVSH_ROOT or place it at ../nvsh)")
    missing = [
        f"{module}: {value['upstream']}"
        for module, value in gen.collect()
        if not (root / value["upstream"]).is_file()
    ]
    assert missing == []


def test_a_stale_doc_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "DOC", tmp_path / "doc.md")
    (tmp_path / "doc.md").write_text("old\n", encoding="utf-8")
    assert gen.main(["--check"]) == 1
    assert gen.main([]) == 0
    assert gen.main(["--check"]) == 0


def test_render_lists_each_module_with_its_fields():
    value = {
        "upstream": "scripts/lfm-finetune/x.py",
        "commit": COMMIT,
        "adaptations": ["a <b> _c_ change"],
        "licence": LICENCE,
    }
    out = gen.render([("jev_factory/core/x.py", value)])
    assert "| `jev_factory/core/x.py` | `scripts/lfm-finetune/x.py` |" in out
    assert "- a <b> _c_ change" in out
