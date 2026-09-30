"""o30: a Q4_K_M build is measured through llama-server with no nvsh package installed.

The whole served path runs for real in a child interpreter in which ``nvsh``
(and every heavy ML package) is blocked in ``sys.modules``: the measure CLI
starts the stub ``llama-server`` (tests/fixtures/stub_servers.py) on a free
localhost port through jev_factory.measure.serve, waits for it, preflights it
(alias and context from ``/props``), scores every entry over HTTP
``/v1/completions``, writes the predictions and results page, and stops the
server. Only the prompt renderer is injected (a chat template needs a real
tokenizer); the stub stands in for llama.cpp, so this proves the harness and
its contract, not a real model's quality.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm.readout import READOUT_TOP
from tests.fixtures.stub_servers import free_port, path_with, stub_bin

ROOT = Path(__file__).resolve().parent.parent

CHILD = """
import sys
for name in ("nvsh", "torch", "transformers", "peft", "unsloth", "huggingface_hub", "datasets"):
    sys.modules[name] = None  # importing any of these now raises ImportError
from jev_factory.measure import run as measure

def render(messages):
    return "\\n".join(message["content"] for message in messages)

seams = measure.Seams(
    build_scorer=lambda spec: measure.build_scorer(spec, renderer=lambda _spec: render),
)
code = measure.main(sys.argv[1:], seams=seams)
assert "nvsh" not in {name.split(".")[0] for name, mod in sys.modules.items() if mod}
sys.exit(code)
"""


def _split(tmp_path: Path) -> Path:
    entries = [
        {
            "id": "on1",
            "kind": "explicit",
            "text": "turn on the kitchen",
            "expect": {"operation": "lamp_on", "args": {"room": "kitchen"}},
        },
        {
            "id": "esc1",
            "kind": "explicit",
            "text": "what is the weather tomorrow",
            "expect": {"escalate": True},
        },
    ]
    path = tmp_path / "splits" / "test.json"
    path.parent.mkdir(parents=True)
    header = "Toy corpus. Split 'test' of toy.json (seed=39)."
    path.write_text(json.dumps({"header": header, "entries": entries}))
    return path


def test_nvsh_is_not_installed_in_this_environment():
    assert importlib.util.find_spec("nvsh") is None


@pytest.mark.behavioral("o30")
def test_a_q4_k_m_build_is_measured_through_llama_server_without_nvsh(tmp_path):
    stubs = stub_bin(tmp_path / "bin")
    gguf = tmp_path / "quant" / "toy-jev.Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF stub weights")
    requests = tmp_path / "requests.jsonl"
    run_dir = tmp_path / "run"
    port = free_port()
    env = {
        **os.environ,
        "PATH": path_with(stubs),  # stub nvidia-smi: no foreign GPU process
        "STUB_LLAMA_LOG": str(requests),
        "STUB_LLAMA_PICKS": json.dumps({"kitchen": "lamp_on"}),
        "JEV_MEASURE_POLL_SECONDS": "0.05",
        "JEV_MEASURE_WAIT_SECONDS": "30",
        "PYTHONPATH": str(ROOT),
    }
    argv = [
        "--domain",
        "tests.fixtures.toy_domain",
        "--run-dir",
        str(run_dir),
        "--split",
        str(_split(tmp_path)),
        "--final",
        "--scorer",
        "served",
        "--serve",
        str(gguf),
        "--port",
        str(port),
        "--llama-server",
        str(stubs / "llama-server"),
        "--tokenizer",
        str(tmp_path / "merged"),
        "--model",
        "toy-jev.Q4_K_M",
        "--revision",
        "build-1",
        "--label",
        "final-q4_k_m",
    ]
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, *argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Every scoring request went over HTTP and asked for the readout's READOUT_TOP.
    bodies = [json.loads(line) for line in requests.read_text().splitlines()]
    assert len(bodies) == 2
    assert {b["path"] for b in bodies} == {"/v1/completions"}
    assert {b["logprobs"] for b in bodies} == {READOUT_TOP}
    assert {(b["model"], b["max_tokens"], b["temperature"]) for b in bodies} == {
        ("toy-jev.Q4_K_M", 1, 0)
    }

    measure_dir = run_dir / "measure"
    [predictions] = (measure_dir / "final-q4_k_m").glob("*.predictions.jsonl")
    lines = [json.loads(line) for line in predictions.read_text().splitlines()]
    on1, esc1 = lines
    assert (on1["outcome"], on1["arguments"], on1["grounded"]) == (
        "propose",
        {"room": "kitchen"},
        True,
    )
    assert esc1["outcome"] == "escalate"

    page = next(measure_dir.glob("*-final-q4_k_m.md")).read_text()
    assert "backend=llama-server" in page
    assert "9999 (stub llama-server)" in page
    assert "- Final run: yes" in page
    assert "- Context: 2048" in page
    import hashlib

    assert hashlib.sha256(gguf.read_bytes()).hexdigest() in page

    record = json.loads(next((measure_dir / "serve").glob("*.record.json")).read_text())
    argv_served = record["argv"]
    assert argv_served[argv_served.index("--ctx-size") + 1] == "2048"
    assert argv_served[argv_served.index("--host") + 1] == "127.0.0.1"
    assert argv_served[argv_served.index("--alias") + 1] == "toy-jev.Q4_K_M"

    # The server this run started is stopped again, and the final run is on record.
    assert not (measure_dir / "serve" / f"jev-measure-{port}.json").exists()
    ledger = (measure_dir / "once-ledger.jsonl").read_text().splitlines()
    assert [json.loads(line)["status"] for line in ledger] == ["measured"]
