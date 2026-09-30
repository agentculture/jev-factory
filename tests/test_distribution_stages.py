"""Distribution-only stages run on a synthetic predictions record (o9) and read the
domain only through the Domain seam (o44).

The run happens in a fresh interpreter where the heavy ML packages and the
backbone adapter package are blocked outright, so any import of them (direct
or transitive) fails the test rather than going unnoticed.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "jev_factory" / "core"
#: The distribution-only modules this task ships.
MODULES = ("predictions", "metrics", "gate")
BLOCKED = (
    "torch",
    "transformers",
    "peft",
    "unsloth",
    "llmcompressor",
    "huggingface_hub",
    "datasets",
    "deepeval",
    "tokenizers",
    "vllm",
    "jev_factory.backbones",
    "nvsh",
    "jetson_skills",
)

_RUN = r"""
import json, sys
for name in {blocked!r}:
    sys.modules[name] = None  # any import of it raises ImportError
from jev_factory.core import gate, metrics, predictions
from tests.fixtures.toy_domain import DOMAIN

rows = [
    {{"id": "a", "expected": {{"operation": "lamp_status", "args": {{}}}}, "outcome": "propose",
     "operation": "lamp_status", "arguments": {{}},
     "offered": ["lamp_status", "set_scene", "(explain)", "(escalate)"],
     "raw_scores": {{"lamp_status": -0.2, "set_scene": -2.5, "(explain)": -3.0,
                     "(escalate)": -3.1}},
     "raw_probabilities": {{"lamp_status": 0.8, "set_scene": 0.1, "(explain)": 0.05,
                            "(escalate)": 0.05}},
     "candidates": {{"lamp_status": 0.7, "set_scene": 0.15, "(explain)": 0.1, "(escalate)": 0.05}},
     "grounded": True, "tokens": 0, "ttfd_ms": 3.0, "latency_ms": 4.0}},
    {{"id": "b", "expected": {{"escalate": True}}, "outcome": "propose",
     "operation": "set_scene", "arguments": {{"scene": "bright"}},
     "candidates": {{"set_scene": 0.55, "escalate:injection": 0.45}},
     "grounded": True, "tokens": 0, "ttfd_ms": 1.0, "latency_ms": 2.0}},
]
records = predictions.from_rows(rows)
thresholds = gate.Thresholds(mutating=gate.ThresholdSet(floor=0.6))
decisions = [gate.decide_prediction(p, thresholds, DOMAIN) for p in records]
result = metrics.compute(records, DOMAIN, bootstrap_resamples=20)
leaked = sorted(
    m for m, mod in sys.modules.items() if m.startswith("jev_factory.backbones") and mod
)
print(json.dumps({{
    "decisions": [[d.outcome, d.label, d.reason] for d in decisions],
    "right": result["right_proposals"]["n"],
    "wrong_mutating": result["wrong_mutating"]["total"],
    "leaked": leaked,
}}))
"""


@pytest.mark.behavioral("o9")
def test_metrics_and_gate_run_on_a_synthetic_record_with_no_model_or_backbone():
    proc = subprocess.run(
        [sys.executable, "-c", _RUN.format(blocked=BLOCKED)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["decisions"] == [
        ["propose", "lamp_status", None],
        ["abstain_uncertain", "set_scene", "floor"],
    ]
    assert out["right"] == 1
    assert out["wrong_mutating"] == 1
    assert out["leaked"] == []


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("jev_factory.core." + (node.module or ""))
            elif node.module in ("jev_factory", "jev_factory.core"):
                # `from jev_factory.core import metrics` imports the submodule.
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif node.module:
                names.add(node.module)
    return names


@pytest.mark.behavioral("o9")
@pytest.mark.parametrize("name", MODULES)
def test_no_distribution_module_imports_a_backbone(name):
    imported = _imports(CORE / f"{name}.py")
    assert not any(n.startswith("jev_factory.backbones") for n in imported), imported


@pytest.mark.behavioral("o44")
@pytest.mark.parametrize("name", MODULES)
def test_only_stdlib_and_the_domain_seam_are_imported(name):
    """The operation table and control labels come from jev_factory.domain, nothing else."""
    imported = _imports(CORE / f"{name}.py")
    project = {n for n in imported if n.split(".")[0] == "jev_factory"}
    allowed = {"jev_factory.domain.model", "jev_factory.domain.validate"} | {
        f"jev_factory.core.{m}" for m in MODULES
    }
    assert project <= allowed, project - allowed
    third_party = {
        n.split(".")[0]
        for n in imported
        if n.split(".")[0] not in sys.stdlib_module_names and n.split(".")[0] != "jev_factory"
    }
    assert third_party <= {"__future__"}, third_party


@pytest.mark.behavioral("o44")
@pytest.mark.parametrize("name", ("metrics", "gate"))
def test_labels_come_from_the_domain_module(name):
    from jev_factory.core import gate, metrics
    from jev_factory.domain import model

    module = {"metrics": metrics, "gate": gate}[name]
    assert module.ESCALATE_LABEL is model.ESCALATE_LABEL
    assert module.EXPLAIN_LABEL is model.EXPLAIN_LABEL
