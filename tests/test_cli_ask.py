"""``jev ask`` (t29): a model proposes one jev verb from a bundle; nothing is executed.

Every test uses a synthetic bundle directory and a fake top-k: no model, no server,
no network.
"""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.cli import _build_parser, main
from jev_factory.cli._commands import ask
from jev_factory.cli._errors import CliError
from jev_factory.domains.jev_cli.generate import cli_surface_sha256, generate_domain

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "jev_factory"
DOMAIN = generate_domain()
POOL = sc.candidate_pool(DOMAIN)
LETTERS = sc.labels_for(DOMAIN, POOL)
OPEN_GATE = {"escalate": None, "read_only": {}, "mutating": {}}


def make_bundle(
    tmp: Path, *, surface: str | None = None, calibration=None, gate=None, name="bundle"
) -> Path:
    folder = tmp / name
    folder.mkdir()
    info = {
        "kind": "gguf",
        "domain": DOMAIN.name,
        "surface_sha256": surface or cli_surface_sha256(),
    }
    files = {
        "bundle.json": info,
        "calibration.json": calibration or {"temperature": 1.0, "vector": {}},
        "gate.json": gate or OPEN_GATE,
        "scorer-train.json": [],
    }
    for name, data in files.items():
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    return folder


def fake_top_k(masses: dict[str, float]):
    """A top-k giving each named candidate *mass* (by probability), the rest a little."""
    rest = [n for n in POOL if n not in masses]
    logprobs = {LETTERS[n]: math.log(p) for n, p in masses.items()}
    logprobs.update({LETTERS[n]: math.log(0.001) for n in rest})

    def top_k(prompt: str, top: int) -> dict[str, float]:
        assert prompt == "PROMPT"
        return dict(logprobs)

    return top_k


def patch_scorer(monkeypatch, masses):
    monkeypatch.setattr(
        ask, "build_scorer", lambda args, bundle: (fake_top_k(masses), lambda messages: "PROMPT")
    )


def run_cli(capsys, *argv):
    capsys.readouterr()
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


READ_ONLY = "jev.whoami"
MUTATING = "jev.run.split"


@pytest.mark.behavioral("o34")
def test_ask_proposes_a_read_only_verb_with_its_probability(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {READ_ONLY: 0.9, "explain": 0.05})
    rc, out, err = run_cli(capsys, "ask", "who am i", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert rc == 0
    assert doc["outcome"] == "propose" and doc["operation"] == READ_ONLY
    assert doc["executed"] is False
    assert 0.8 < doc["probability"] <= 1.0
    assert doc["surface_mismatch"] is None


@pytest.mark.behavioral("o34")
def test_ask_prints_exactly_one_outcome_in_text_mode(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {"explain": 0.9})
    rc, out, _ = run_cli(capsys, "ask", "what is a jev model", "--bundle", str(bundle))
    assert rc == 0
    first = out.splitlines()[0].split()[0]
    assert first == "explain" and len(out.splitlines()[0].split()) == 2  # outcome, then p=...


@pytest.mark.behavioral("o34")
def test_ask_applies_calibration_and_gate_from_the_bundle(tmp_path, monkeypatch, capsys):
    # raw: explain narrowly ahead of the verb. A per-label vector flips the argmax and the
    # gate's floor then abstains on it -- both files are read from the bundle.
    calibration = {"temperature": 1.0, "vector": {"(explain)": 0.1}}
    gate = {"escalate": None, "read_only": {"floor": 0.99}, "mutating": {}}
    bundle = make_bundle(tmp_path, calibration=calibration, gate=gate)
    patch_scorer(monkeypatch, {"explain": 0.5, READ_ONLY: 0.45})
    _, out, _ = run_cli(capsys, "ask", "who am i", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert doc["outcome"] == "abstain_uncertain" and doc["reason"] == "floor"
    # without the vector the argmax stays explain
    plain = make_bundle(tmp_path, name="plain")
    _, out, _ = run_cli(capsys, "ask", "who am i", "--bundle", str(plain), "--json")
    assert json.loads(out)["outcome"] == "explain"


@pytest.mark.behavioral("o34")
def test_ask_escalates(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {"escalate": 0.95})
    _, out, _ = run_cli(capsys, "ask", "reformat my disk", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert doc["outcome"] == "escalate" and doc["operation"] is None


@pytest.mark.behavioral("o34")
def test_ask_grounds_arguments_deterministically(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {"jev.explain": 0.95})
    _, out, _ = run_cli(capsys, "ask", "tell me about whoami", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert doc["operation"] == "jev.explain"
    assert doc["arguments"] == {"path": "whoami"} and doc["grounded"] is True
    # a word the catalog does not know is never invented by the model
    _, out, _ = run_cli(capsys, "ask", "tell me about zzzz", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert doc["grounded"] is False and doc["arguments"] is None and doc["grounding"]


@pytest.mark.behavioral("o34")
def test_ask_without_a_distribution_abstains(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    monkeypatch.setattr(
        ask, "build_scorer", lambda args, bundle: (lambda p, t: {}, lambda m: "PROMPT")
    )
    _, out, _ = run_cli(capsys, "ask", "x", "--bundle", str(bundle), "--json")
    assert json.loads(out)["outcome"] == "abstain_uncertain"


def test_ask_refuses_an_incomplete_bundle(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    (bundle / "gate.json").unlink()
    patch_scorer(monkeypatch, {READ_ONLY: 0.9})
    rc, _, err = run_cli(capsys, "ask", "x", "--bundle", str(bundle))
    assert rc == 1 and "gate.json" in err
    rc, _, err = run_cli(capsys, "ask", "x", "--bundle", str(tmp_path / "nope"))
    assert rc == 1 and "hint:" in err


@pytest.mark.behavioral("o36")
def test_ask_reports_a_surface_hash_mismatch(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path, surface="0" * 64)
    patch_scorer(monkeypatch, {READ_ONLY: 0.9})
    rc, out, err = run_cli(capsys, "ask", "who am i", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert rc == 0 and "surface" in doc["surface_mismatch"] and "0" * 64 in doc["surface_mismatch"]
    assert "surface" in err  # also a diagnostic on stderr, never mixed into stdout
    rc, _, err = run_cli(capsys, "ask", "who am i", "--bundle", str(bundle), "--strict-surface")
    assert rc == 2 and "surface" in err


@pytest.mark.behavioral("o36")
def test_a_matching_bundle_reports_no_mismatch(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {READ_ONLY: 0.9})
    rc, out, err = run_cli(capsys, "ask", "who am i", "--bundle", str(bundle), "--json")
    assert rc == 0 and json.loads(out)["surface_mismatch"] is None and err == ""


# ---------------------------------------------------------------------------
# o35: nothing a model says is ever run
# ---------------------------------------------------------------------------


@pytest.mark.behavioral("o35")
def test_ask_with_a_mutating_proposal_only_prints(tmp_path, monkeypatch, capsys):
    bundle = make_bundle(tmp_path)
    patch_scorer(monkeypatch, {MUTATING: 0.97})

    def boom(*a, **k):
        raise AssertionError("ask must not start a process")

    for target in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, target, boom)
    for target in ("system", "execv", "execvp", "spawnv", "popen"):
        monkeypatch.setattr(os, target, boom, raising=False)
    import jev_factory.cli._commands.run as run_cmd

    monkeypatch.setattr(run_cmd, "cmd_run", boom, raising=False)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    rc, out, _ = run_cli(capsys, "ask", "split the data", "--bundle", str(bundle), "--json")
    doc = json.loads(out)
    assert rc == 0 and doc["outcome"] == "propose" and doc["operation"] == MUTATING
    assert doc["executed"] is False and "--apply" not in json.dumps(doc)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before  # wrote nothing


#: Modules that may name the ``--apply`` literal, each with why.
APPLY_ALLOWED = {
    "cli/_commands/run.py": "declares --apply; --detach re-runs the stage the OPERATOR named",
    "cli/_commands/init.py": "declares --apply",
    "backbones/causal_lm/quantize.py": "declares --apply on its own argparse",
    "domains/jev_cli/generate.py": "detects an --apply option on a parser",
}


def _code_literals(path: Path) -> list[str]:
    """String constants in *path* outside docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]


@pytest.mark.behavioral("o35")
def test_only_named_modules_construct_the_apply_flag():
    offenders = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel in APPLY_ALLOWED:
            continue
        if any(lit.strip() == "--apply" for lit in _code_literals(path)):
            offenders.append(rel)
    assert offenders == [], f"--apply constructed outside the allowlist: {offenders}"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            found.add(base)
            found.update(f"{base}.{a.name}" for a in n.names)
    return found


@pytest.mark.behavioral("o35")
@pytest.mark.parametrize("module", ["cli/_commands/ask.py", "cli/_commands/decide.py"])
def test_model_driven_verbs_cannot_reach_a_writer(module):
    """ask and decide import no process launcher and no verb that writes or runs a stage."""
    imports = _imports(PKG / module)
    forbidden = {
        "subprocess",
        "os.system",
        "jev_factory.cli._commands.run",
        "jev_factory.cli._commands.init",
    }
    forbidden |= {"jev_factory.factory.detach", "jev_factory.release.hub", "multiprocessing"}
    assert not (imports & forbidden), sorted(imports & forbidden)
    src = (PKG / module).read_text(encoding="utf-8")
    for call in ("subprocess", "os.system", "os.exec", "Popen", '--apply"'):
        assert call not in src.replace("never passes ``--apply``", ""), call


def test_the_verb_is_annotated_read_only_and_in_the_domain():
    op = DOMAIN.get("jev.ask")
    assert op is not None and op.read_only is True
    assert {a.name for a in op.args} == {"request", "bundle"}
    assert "ask" in _build_parser().format_help()


def test_build_scorer_needs_a_server_or_a_checkpoint(tmp_path):
    bundle = make_bundle(tmp_path)  # no weights, no --server
    args = ask.argparse.Namespace(server=None, model=None)
    with pytest.raises(CliError) as caught:
        ask.build_scorer(args, bundle)
    assert caught.value.code == 2 and "--server" in caught.value.remediation
