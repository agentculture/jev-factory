"""t32: docs/lessons-encoded.md maps each nvsh failure to the test that prevents it, and
``jev learn`` / ``jev explain`` / the prompt files describe the factory as it now exists."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from jev_factory.cli import main
from jev_factory.explain.catalog import ENTRIES

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "lessons-encoded.md"
REF = re.compile(r"`(tests/[\w/]+\.py)::(\w+)`")

LESSONS = {
    "#46 b1-rule": "mid-run rule",
    "#39 l5": "leakage",
    "#46 l4": "silent merge",
    "issue #4 stage 16": "hand-copied bundle files",
    "issue #4,": "stale snapshot",
    "issue #4 stage 3": "parser blacklist",
}


def _rows() -> dict[str, str]:
    text = DOC.read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    }


def _defined(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_the_doc_has_one_section_per_failure_with_its_record_and_tests() -> None:
    sections = _rows()
    for record, name in LESSONS.items():
        hits = [title + body for title, body in sections.items() if name in title.lower()]
        assert hits, f"no section for {name}"
        body = hits[0]
        assert record in body, f"{name}: nvsh record {record!r} not cited"
        assert len(REF.findall(body)) >= 2, f"{name}: needs at least two named tests"


def test_every_cited_test_exists() -> None:
    refs = REF.findall(DOC.read_text(encoding="utf-8"))
    assert len(refs) >= 20
    for path, name in refs:
        assert (ROOT / path).is_file(), path
        assert name in _defined(path), f"{path}::{name} does not exist"


def test_the_doc_leaks_no_path_host_or_nvsh_doc_pointer() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "/home/" not in text and "../nvsh" not in text
    assert "github.com/agentculture/nvsh/blob" not in text


# --- learn / explain / README / prompt files --------------------------------


def _learn_text(capsys) -> str:
    assert main(["learn"]) == 0
    return capsys.readouterr().out


def test_learn_names_the_three_readers_and_the_domain_author_path(capsys) -> None:
    text = _learn_text(capsys)
    for needle in (
        "operator",
        "mesh agent",
        "domain author",
        "jev_factory/domain/model.py",
        "tests/fixtures/toy_domain",
        "status",
        "decisions.jsonl",
        "docs/lessons-encoded.md",
    ):
        assert needle in text, needle
    assert "playbook" not in text and "../nvsh" not in text


def test_learn_json_carries_the_audiences(capsys) -> None:
    assert main(["learn", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {a["reader"] for a in payload["audiences"]} == {
        "operator",
        "mesh agent",
        "domain author",
    }
    assert payload["domain_module"]["contract"] == "jev_factory/domain/model.py"
    assert payload["domain_module"]["example"] == "tests/fixtures/toy_domain"


def test_explain_has_a_domain_module_entry_and_no_nvsh_doc_pointers(capsys) -> None:
    body = ENTRIES[("domain",)]
    for needle in ("jev_factory/domain/model.py", "tests/fixtures/toy_domain", "jev-factory init"):
        assert needle in body
    assert main(["explain", "domain"]) == 0
    for key, text in ENTRIES.items():
        assert "playbook" not in text and "../nvsh" not in text, key
    root = ENTRIES[()]
    assert "operator" in root and "mesh agent" in root and "domain author" in root


@pytest.mark.parametrize(
    "name", ["README.md", "CLAUDE.md", "AGENTS.override.md", "AGENTS.colleague.md", "QWEN.md"]
)
def test_the_factory_is_no_longer_described_as_a_scaffold(name) -> None:
    text = (ROOT / name).read_text(encoding="utf-8").lower()
    assert "scaffold only" not in text
    assert "none of the factory" not in text
    assert "nothing of the factory is built" not in text
    assert "not yet trained" in text.replace("\n", " ") or "not trained" in text


def test_readme_does_not_point_at_nvsh_docs() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "nvsh/blob" not in text
    assert "docs/lessons-encoded.md" in text
