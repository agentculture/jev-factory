"""t12 integration: a produced pointer bundle served by the pinned strands-decider env.

Skipped unless ``JEV_STRANDS_PY`` names the python of the pinned student env (one with
``strands_decider`` installed) and ``JEV_STRANDS_SRC`` names a local checkpoint directory
(for example a verified snapshot of StrandsAgents/strands-decider-2B-hobson-v21). The test
exports a bundle from it, serves that bundle on 127.0.0.1 and asks a choice, a noul and a
score question over /v1/systemone. It needs the GPU (or a slow CPU) and the base model in the
local HF cache; it never runs in CI.
"""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess  # nosec B404 - the test's own pinned python
from pathlib import Path

import pytest

PY = os.environ.get("JEV_STRANDS_PY")
SRC = os.environ.get("JEV_STRANDS_SRC")

pytestmark = pytest.mark.skipif(
    not (PY and SRC), reason="set JEV_STRANDS_PY and JEV_STRANDS_SRC to run the live pointer test"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _student_env():
    """The pinned student env: JEV_STRANDS_LOCK, else the sha256 of its ``pip freeze``."""
    from jev_factory.distill.envs import EnvSpec

    lock = os.environ.get("JEV_STRANDS_LOCK")
    if not lock:
        frozen = subprocess.run(  # nosec B603
            [str(PY), "-m", "pip", "freeze"], check=True, capture_output=True
        ).stdout
        lock = hashlib.sha256(frozen).hexdigest()
    return EnvSpec(role="student", python=str(PY), lockfile_hash=lock)


def test_served_bundle_answers_choice_noul_and_score(tmp_path):
    from jev_factory.backbones import pointer, registry

    bundle = tmp_path / "bundle"
    registry.get("pointer").export(str(SRC), str(bundle), apply=True)
    port = _free_port()
    pid = pointer.launch(tmp_path / "jobs", "student", _student_env(), bundle, port=port)
    try:
        url = f"http://127.0.0.1:{port}"
        pointer.wait_ready(url, timeout=900.0, jobdir=tmp_path / "jobs", name="student")
        record = {
            "id": "live",
            "state": "The nightly build failed on the lint step; tests passed.",
            "questions": {
                "c": {
                    "type": "choice",
                    "instructions": "What should happen next?",
                    "criteria": {"fix_lint": "fix the lint error", "rerun": "rerun the tests"},
                },
                "n": {"type": "noul", "instructions": "Did the tests pass?", "criteria": None},
                "s": {
                    "type": "score",
                    "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "somewhat urgent", "very urgent"],
                },
            },
        }
        result = pointer.PointerClient(url, model="student").ask(record)
        by = {q.qid: q for q in result.questions}
        assert all(q.ok for q in result.questions), [q.error for q in result.questions]
        assert set(by["c"].distribution) == {"fix_lint", "rerun"}
        assert 0.0 <= by["n"].p_true <= 1.0
        assert set(by["s"].distribution) == {"not urgent", "somewhat urgent", "very urgent"}
    finally:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    assert Path(tmp_path / "jobs" / "student.serve.json").exists()
