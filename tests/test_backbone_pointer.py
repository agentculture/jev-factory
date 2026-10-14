"""t12: the pointer backbone (Strands Decider) behind the Backbone protocol.

Predictions come from a served ``strands-decider serve`` over /v1/systemone, never from
letter-token logprobs; bundles are the hobson-hf-export/1 layout. No network, no torch: the
server here is a stdlib fake on 127.0.0.1 that answers in the real server's wire shape.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from jev_factory.backbones import registry
from jev_factory.backbones.base import Backbone
from jev_factory.cli._errors import CliError
from tests.fixtures.toy_domain import DOMAIN

REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def _pointer():
    from jev_factory.backbones import pointer

    return pointer


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _source(tmp_path: Path, *, head_type: str = "pointer", with_provenance: bool = True) -> Path:
    """A trained-checkpoint directory in the shape the student trainer leaves behind."""
    src = tmp_path / "ckpt"
    adapter = _write(src / "lora" / "adapter_model.safetensors", b"adapter-bytes")
    _write(src / "lora" / "adapter_config.json", b'{"r": 16}')
    head = _write(src / "head.safetensors", b"head-bytes")
    config = {"base_model": "Qwen/Qwen3.5-2B-Base", "head_type": head_type, "max_length": 4096}
    _write(src / "strands_decider_config.json", json.dumps(config).encode())
    _write(src / "tokenizer.json", b"{}")
    _write(src / "tokenizer_config.json", b"{}")
    if with_provenance:
        prov = {
            "format": "hobson-hf-export/1",
            "base_model": "Qwen/Qwen3.5-2B-Base",
            "base_model_revision": REVISION,
            "adapter_sha256": adapter,
            "head_sha256": head,
        }
        _write(src / "provenance.json", json.dumps(prov).encode())
    return src


# --------------------------------------------------------------------- protocol / registry


def test_registry_resolves_pointer_to_a_backbone():
    adapter = registry.get("pointer")
    assert isinstance(adapter, Backbone)
    assert adapter.name == "pointer"


def test_pointer_package_does_not_import_strands_decider_or_torch():
    pkg = Path(_pointer().__file__).parent
    for path in pkg.rglob("*.py"):
        text = path.read_text()
        for banned in ("import strands_decider", "from strands_decider", "import torch"):
            assert banned not in text, f"{path.name} imports in-process: {banned}"
    assert "strands_decider" not in sys.modules


# --------------------------------------------------------------------- render


def test_render_builds_a_systemone_choice_record_over_the_offered_names():
    offered = ["lamp_status", "list_rooms", "explain", "escalate"]
    rec = registry.get("pointer").render(DOMAIN, "is the kitchen lamp on?", offered)
    assert rec["state"] == "is the kitchen lamp on?"
    (qid, q), *rest = rec["questions"].items()
    assert not rest
    assert q["type"] == "choice"
    assert list(q["criteria"]) == offered
    assert q["criteria"]["lamp_status"] == DOMAIN.get("lamp_status").description
    assert q["instructions"]
    # no lettering: a pointer head scores each named option, there is no letter readout
    assert not any(") " in str(v) for v in q["criteria"].values())


def test_render_refuses_an_unknown_candidate_and_a_single_option():
    pb = registry.get("pointer")
    with pytest.raises(CliError, match="not a candidate"):
        pb.render(DOMAIN, "x", ["lamp_status", "no_such_verb"])
    with pytest.raises(CliError, match="at least 2"):
        pb.render(DOMAIN, "x", ["lamp_status"])


# --------------------------------------------------------------------- fake server


class _Fake:
    """Answers /v1/systemone the way strands_decider.server does (score keyed by level index)."""

    def __init__(self, probabilities):
        outer = self
        self.requests = []
        self.probabilities = probabilities

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                answers = {}
                for qid, q in body["questions"].items():
                    if q["type"] == "noul":
                        answers[qid] = {"type": "noul", "noul": 0.7}
                    elif q["type"] == "score":
                        n = len(q["criteria"])
                        answers[qid] = {
                            "type": "score",
                            "score": 0.5,
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                            "probabilities": {str(i): 1.0 / n for i in range(n)},
                            "confidence": 0.9,
                        }
                    else:
                        probs = outer.probabilities or {
                            k: 1.0 / len(q["criteria"]) for k in q["criteria"]
                        }
                        answers[qid] = {
                            "type": "choice",
                            "choice": max(probs, key=probs.get),
                            "probabilities": probs,
                            "confidence": 0.99,
                        }
                raw = json.dumps({"model": "m", "answers": answers, "usage": {}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    made = []

    def make(probabilities=None):
        srv = _Fake(probabilities)
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.close()


# --------------------------------------------------------------------- predict


def test_predictions_come_from_the_served_systemone_probabilities(fake):
    srv = fake({"lamp_status": 0.7, "list_rooms": 0.2, "explain": 0.1})
    pb = registry.get("pointer")
    offered = ["lamp_status", "list_rooms", "explain"]
    rendered = [pb.render(DOMAIN, "is the lamp on?", offered)]
    client = _pointer().PointerClient(srv.url, model="student")
    dists = pb.predict_distributions(client, rendered, [offered])
    assert dists == [{"lamp_status": 0.7, "list_rooms": 0.2, "explain": 0.1}]
    path, body = srv.requests[0]
    assert path == "/v1/systemone"
    assert body["model"] == "student"


def test_predict_accepts_a_base_url_as_the_model_handle(fake):
    srv = fake()
    pb = registry.get("pointer")
    offered = ["lamp_status", "explain"]
    dists = pb.predict_distributions(srv.url, [pb.render(DOMAIN, "q", offered)], [offered])
    assert dists == [{"lamp_status": 0.5, "explain": 0.5}]


def test_predict_fails_closed_when_the_server_answer_is_not_a_distribution(fake):
    srv = fake({"lamp_status": 0.9, "explain": 0.9})  # sums to 1.8
    pb = registry.get("pointer")
    offered = ["lamp_status", "explain"]
    with pytest.raises(CliError, match="sum"):
        pb.predict_distributions(srv.url, [pb.render(DOMAIN, "q", offered)], [offered])


def test_predict_refuses_offered_that_disagrees_with_the_rendered_record(fake):
    srv = fake()
    pb = registry.get("pointer")
    rec = pb.render(DOMAIN, "q", ["lamp_status", "explain"])
    with pytest.raises(CliError, match="offered"):
        pb.predict_distributions(srv.url, [rec], [["lamp_status", "escalate"]])


def test_predict_refuses_a_non_localhost_server():
    pb = registry.get("pointer")
    rec = pb.render(DOMAIN, "q", ["lamp_status", "explain"])
    with pytest.raises((CliError, ValueError)):
        pb.predict_distributions("http://example.com:8000", [rec], [["lamp_status", "explain"]])


def test_client_rekeys_index_keyed_score_answers_to_the_criteria(fake):
    srv = fake()
    record = {
        "id": "r",
        "state": "s",
        "questions": {
            "c": {"type": "choice", "instructions": "pick", "criteria": {"a": "", "b": ""}},
            "n": {"type": "noul", "instructions": "is it?", "criteria": None},
            "s": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]},
        },
    }
    result = _pointer().PointerClient(srv.url, model="m").ask(record)
    by = {q.qid: q for q in result.questions}
    assert all(q.ok for q in result.questions), [q.error for q in result.questions]
    assert by["n"].p_true == 0.7
    assert set(by["s"].distribution) == {"low", "mid", "high"}
    assert result.response_sha256


# --------------------------------------------------------------------- layout


def test_read_layout_accepts_a_complete_hobson_export(tmp_path):
    info = _pointer().read_layout(_source(tmp_path))
    assert info["format"] == "hobson-hf-export/1"
    assert set(info["files"]) >= set(_pointer().REQUIRED_FILES) - {"provenance.json"}


@pytest.mark.parametrize(
    "missing",
    [
        "lora/adapter_model.safetensors",
        "lora/adapter_config.json",
        "head.safetensors",
        "strands_decider_config.json",
        "provenance.json",
    ],
)
def test_read_layout_refuses_a_layout_missing_any_file(tmp_path, missing):
    src = _source(tmp_path)
    (src / missing).unlink()
    with pytest.raises(CliError, match="missing"):
        _pointer().read_layout(src)


def test_read_layout_refuses_a_wrong_format_or_hash_or_head_type(tmp_path):
    src = _source(tmp_path)
    prov = json.loads((src / "provenance.json").read_text())
    (src / "provenance.json").write_text(json.dumps({**prov, "format": "other/2"}))
    with pytest.raises(CliError, match="format"):
        _pointer().read_layout(src)
    (src / "provenance.json").write_text(json.dumps(prov))
    (src / "head.safetensors").write_bytes(b"tampered")
    with pytest.raises(CliError, match="head_sha256"):
        _pointer().read_layout(src)
    slot = _source(tmp_path / "slot", head_type="slot")
    with pytest.raises(CliError, match="pointer"):
        _pointer().read_layout(slot)


def test_export_is_dry_run_by_default_and_writes_nothing(tmp_path):
    out = tmp_path / "bundle"
    plan = registry.get("pointer").export(str(_source(tmp_path)), str(out))
    assert plan["apply"] is False
    assert "provenance.json" in plan["would_write"]
    assert not out.exists()


def test_export_apply_writes_a_layout_that_reads_back(tmp_path):
    src = _source(tmp_path, with_provenance=False)
    out = tmp_path / "bundle"
    pb = registry.get("pointer")
    with pytest.raises(CliError, match="base_model_revision"):
        pb.export(str(src), str(out), apply=True)
    src2 = _source(tmp_path / "again")
    report = pb.export(str(src2), str(out), apply=True)
    assert report["apply"] is True
    info = _pointer().read_layout(out)
    assert info["provenance"]["base_model_revision"] == REVISION
    assert (out / "tokenizer.json").is_file()
    manifest = (out / "MANIFEST.sha256").read_text()
    assert "  head.safetensors\n" in manifest


def test_export_refuses_a_source_missing_a_required_file(tmp_path):
    src = _source(tmp_path)
    (src / "head.safetensors").unlink()
    with pytest.raises(CliError, match="missing"):
        registry.get("pointer").export(str(src), str(tmp_path / "out"))


def test_merge_verify_checks_the_layout_and_the_base(tmp_path):
    pb = registry.get("pointer")
    report = pb.merge_verify(str(_source(tmp_path)), "Qwen/Qwen3.5-2B-Base")
    assert report["merged"] is False
    with pytest.raises(CliError, match="base"):
        pb.merge_verify(str(_source(tmp_path / "b")), "Qwen/Qwen3.5-0.8B")


def test_train_plan_and_quantize_candidates_are_descriptive_only():
    pb = registry.get("pointer")
    plan = pb.train_plan({"train_file": "t.jsonl", "output_dir": "o"})
    assert plan["dry_run"] is True
    assert plan["argv"][1:4] == ["-m", "training.train", "--config"]
    assert list(pb.quantize_candidates({})) == ["bf16"]


# --------------------------------------------------------------------- serve argv


def test_serve_argv_is_the_real_strands_decider_cli_on_loopback(tmp_path):
    bundle = tmp_path / "bundle"
    registry.get("pointer").export(str(_source(tmp_path)), str(bundle), apply=True)
    argv = _pointer().serve_argv("/env/bin/python", bundle, port=8123, model_name="student")
    assert argv[:5] == ["/env/bin/python", "-m", "strands_decider.cli", "serve", str(bundle)]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8123"
    assert "--revision" not in argv and "--model" not in argv


def test_serve_argv_refuses_a_non_loopback_host_and_an_unverified_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    registry.get("pointer").export(str(_source(tmp_path)), str(bundle), apply=True)
    with pytest.raises(CliError, match="127.0.0.1"):
        _pointer().serve_argv("/py", bundle, port=1, host="0.0.0.0")  # nosec B104
    (bundle / "tokenizer.json").unlink()
    with pytest.raises(CliError, match="tokenizer"):
        _pointer().serve_argv("/py", bundle, port=1)
    with pytest.raises(CliError):
        _pointer().serve_argv("/py", "StrandsAgents/strands-decider-2B-hobson-v21", port=1)


def test_launch_needs_the_pinned_student_env_and_records_provenance(tmp_path, monkeypatch):
    from jev_factory.backbones.pointer import serve
    from jev_factory.distill.envs import EnvSpec

    bundle = tmp_path / "bundle"
    registry.get("pointer").export(str(_source(tmp_path)), str(bundle), apply=True)
    with pytest.raises(CliError, match="student"):
        serve.launch(tmp_path / "jobs", "s", "/env/bin/python", bundle, port=9)
    started = []
    monkeypatch.setattr(serve, "start_detached", lambda d, n, a: started.append(a) or 4242)
    env = EnvSpec(role="student", python="/env/bin/python", lockfile_hash="a" * 64)
    assert serve.launch(tmp_path / "jobs", "s", env, bundle, port=9) == 4242
    record = json.loads((tmp_path / "jobs" / "s.serve.json").read_text())
    assert record["argv"] == started[0]
    assert record["environment"]["lockfile_hash"] == "a" * 64
    assert "head.safetensors" in record["bundle"]
