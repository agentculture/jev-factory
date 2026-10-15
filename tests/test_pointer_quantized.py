"""tq (deviation d3): a torchao weight-only int8/int4 pointer student and its serving wrapper.

The pure-Python halves always run here: the build plan (dry-run writes nothing), the artifact
manifest validation, the loopback-only refusal, the subprocess wiring with a fake runner and
the HTTP wrapper with a fake evaluator. The torch/torchao half skips cleanly when either is
missing. The live test (build int8 from a real checkpoint, serve it, ask a choice, a noul and
a score question) is skipped unless ``JEV_STRANDS_PY`` and ``JEV_STRANDS_SRC`` are set.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess  # nosec B404 - the live test's own pinned python
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill.envs import EnvSpec

REV = "b" * 40
CONFIG = {"head_type": "pointer", "base_model": "Qwen/Qwen3.5-2B", "use_lora": True}


def _checkpoint(root: Path, *, config=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "lora").mkdir(exist_ok=True)
    (root / player.ADAPTER_CONFIG).write_text('{"r": 16}')
    (root / player.ADAPTER).write_bytes(b"adapter-weights")
    (root / player.HEAD).write_bytes(b"head-weights")
    (root / player.CONFIG).write_text(json.dumps(config or CONFIG))
    for name in player.TOKENIZER_FILES:
        (root / name).write_text("{}")
    return root


def _artifact(root: Path, *, bits=8, source=None, **overrides) -> Path:
    """What the in-env build writes, with fake weights (no torch)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / qz.TORSO_FILE).write_bytes(b"quantized-torso")
    (root / player.HEAD).write_bytes(b"fp32-head")
    (root / player.CONFIG).write_text(json.dumps({**CONFIG, "use_lora": False}))
    for name in player.TOKENIZER_FILES:
        (root / name).write_text("{}")
    fields = dict(
        bits=bits,
        torchao_api=qz.TORCHAO_APIS[bits][0],
        torchao_version="0.18.0",
        torch_version="2.12.1",
        source={"checkpoint": "/ckpt", "sha256": {player.HEAD: "0" * 64}},
        base_model=CONFIG["base_model"],
        base_model_revision=REV,
    )
    if source is not None:
        fields["source"] = source
    fields.update(overrides)
    qz.write_manifest(root, **fields)
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / qz.MANIFEST_FILE).read_text())


def _rewrite(root: Path, **changes) -> None:
    doc = _manifest(root)
    doc.update(changes)
    (root / qz.MANIFEST_FILE).write_text(json.dumps(doc))


def _plan(tmp_path, bits=8, **kw):
    ckpt = _checkpoint(tmp_path / "ckpt")
    args = dict(python="/envs/student/bin/python", decider_repo=tmp_path / "sd", base_revision=REV)
    args.update(kw)
    return ckpt, qz.build_weight_only(ckpt, tmp_path / "out", bits, **args)


# --------------------------------------------------------------------- the build plan


def test_dry_run_returns_the_plan_and_writes_nothing(tmp_path):
    ckpt, plan = _plan(tmp_path)
    assert plan["dry_run"] is True
    assert plan["format"] == "torch_int8_wo"
    assert plan["bits"] == 8
    assert plan["argv"] == [
        "/envs/student/bin/python",
        "-m",
        qz.MODULE,
        "build",
        str(ckpt),
        str(tmp_path / "out"),
        "--bits",
        "8",
        "--embedding-bits",
        "4",
        "--block-group-size",
        "256",
        "--embedding-group-size",
        "128",
        "--base-revision",
        REV,
        "--device",
        "cuda",
    ]
    assert plan["scheme"] == qz.QuantScheme(block_bits=8).record()
    assert plan["size_budget_bytes"] == 10**9
    assert plan["tensor_policy"] == qz.tensor_policy()
    path = plan["env"]["PYTHONPATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "sd"), str(tmp_path / "sd" / "src")]
    assert (Path(path[2]) / "jev_factory" / "backbones" / "pointer" / "quantized.py").is_file()
    assert plan["base_model_revision"] == REV
    assert plan["head_dtype"] == "float32"
    assert plan["lora_merged"] is True
    assert not (tmp_path / "out").exists()


def test_int4_plans_the_int4_api(tmp_path):
    _ckpt, plan = _plan(tmp_path, bits=4)
    assert plan["format"] == "torch_int4_wo"
    assert plan["argv"][7] == "4"
    assert plan["torchao_api"] == qz.TORCHAO_APIS[4]


def test_the_build_refuses_bits_other_than_8_or_4(tmp_path):
    with pytest.raises(CliError, match="bits"):
        _plan(tmp_path, bits=2)


def test_the_build_refuses_an_incomplete_checkpoint(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    (ckpt / player.HEAD).unlink()
    with pytest.raises(CliError, match="head.safetensors"):
        qz.build_weight_only(ckpt, tmp_path / "out", 8, python="py", base_revision=REV)


def test_the_build_refuses_an_unpinned_base_revision(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    with pytest.raises(CliError, match="base_model_revision"):
        qz.build_weight_only(ckpt, tmp_path / "out", 8, python="py")


def test_the_build_takes_the_revision_from_the_checkpoints_provenance(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    (ckpt / player.PROVENANCE).write_text(json.dumps({"base_model_revision": REV}))
    plan = qz.build_weight_only(ckpt, tmp_path / "out", 8, python="py")
    assert plan["base_model_revision"] == REV


def test_the_build_refuses_a_non_empty_output_directory(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("x")
    with pytest.raises(CliError, match="not empty"):
        _plan(tmp_path)


def test_without_a_decider_repo_only_jev_factory_goes_on_the_path(tmp_path):
    _ckpt, plan = _plan(tmp_path, decider_repo=None)
    assert len(plan["env"]["PYTHONPATH"].split(os.pathsep)) == 1


class FakeRunner:
    def __init__(self, rc=0, write=True, stderr=""):
        self.rc, self.write, self.stderr, self.calls = rc, write, stderr, []

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env)))
        if self.write:
            _artifact(Path(argv[5]), bits=int(argv[7]))
        return subprocess.CompletedProcess(argv, self.rc, stdout="", stderr=self.stderr)


def test_apply_runs_the_planned_command_in_the_student_env_and_verifies(tmp_path):
    runner = FakeRunner()
    _ckpt, out = _plan(tmp_path, apply=True, runner=runner, memory_max="100G")
    assert out["dry_run"] is False
    argv, env = runner.calls[0]
    assert argv == out["argv"]
    assert env["PYTHONPATH"] == out["env"]["PYTHONPATH"]
    assert out["artifact"]["manifest"]["format"] == "torch_int8_wo"


def test_apply_reports_a_failed_build_with_its_error(tmp_path):
    runner = FakeRunner(rc=1, write=False, stderr="RuntimeError: no int4 kernel for sm_121")
    with pytest.raises(CliError, match="no int4 kernel") as err:
        _plan(tmp_path, bits=4, apply=True, runner=runner, memory_max="100G")
    assert err.value.code == EXIT_ENV_ERROR


def test_apply_refuses_a_build_that_left_no_valid_artifact(tmp_path):
    runner = FakeRunner(write=False)
    with pytest.raises(CliError, match="jev_quant.json"):
        _plan(tmp_path, apply=True, runner=runner, memory_max="100G")


class GpuStage:
    """Stands in for factory.gpu.run_gpu_stage: records the call, writes the artifact."""

    def __init__(self, rc=0, stderr=""):
        self.rc, self.stderr, self.calls = rc, stderr, []

    def __call__(self, stage, run_dir, cmd, **kw):
        self.calls.append((stage, Path(run_dir), list(cmd), kw))
        if self.rc == 0:
            _artifact(Path(cmd[5]), bits=int(cmd[7]))
        return subprocess.CompletedProcess(cmd, self.rc, stdout="", stderr=self.stderr)


def _no_bare_subprocess(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a bare subprocess ran the quantize build")

    monkeypatch.setattr(qz.subprocess, "run", boom)


def test_apply_without_a_memory_cap_is_refused_and_runs_nothing(tmp_path, monkeypatch):
    stage = GpuStage()
    monkeypatch.setattr(qz.gpu, "run_gpu_stage", stage)
    _no_bare_subprocess(monkeypatch)
    runner = FakeRunner()
    for kw in ({}, {"runner": runner}):
        with pytest.raises(CliError, match="memory cap") as err:
            _plan(tmp_path, apply=True, **kw)
        assert err.value.code == EXIT_USER_ERROR
    assert stage.calls == []
    assert runner.calls == []


def test_dry_run_needs_no_memory_cap(tmp_path):
    assert _plan(tmp_path)[1]["dry_run"] is True


def test_apply_goes_through_run_gpu_stage_never_a_bare_subprocess(tmp_path, monkeypatch):
    stage = GpuStage()
    monkeypatch.setattr(qz.gpu, "run_gpu_stage", stage)
    _no_bare_subprocess(monkeypatch)
    _ckpt, out = _plan(
        tmp_path, apply=True, memory_max="100G", memory_floor="12G", allow_foreign=True
    )
    name, run_dir, cmd, kw = stage.calls[0]
    assert name == "quantize"
    assert cmd == out["argv"]
    assert kw["memory_max"] == "100G"
    assert kw["memory_floor"] == "12G"
    assert kw["allow_foreign"] is True
    assert kw["env"]["PYTHONPATH"] == out["env"]["PYTHONPATH"]
    assert run_dir.is_dir()
    assert run_dir != tmp_path / "out"
    assert out["artifact"]["manifest"]["format"] == "torch_int8_wo"
    assert out["capped"] == {"memory_max": "100G", "memory_floor": "12G"}


def test_a_watchdog_stop_is_reported_as_the_memory_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(qz.gpu, "run_gpu_stage", GpuStage(rc=3, stderr="MemAvailable low"))
    with pytest.raises(CliError, match="memory-floor watchdog") as err:
        _plan(tmp_path, apply=True, memory_max="100G")
    assert err.value.code == EXIT_ENV_ERROR


def test_launch_records_that_the_server_is_not_capped(tmp_path, monkeypatch):
    root = _artifact(tmp_path / "a")
    monkeypatch.setattr(qz, "start_detached", lambda j, n, argv: 1)
    env = EnvSpec(role="student", python="/envs/student/bin/python", lockfile_hash="c" * 64)
    qz.launch(tmp_path / "jobs", "q", env, root, port=8000)
    record = json.loads((tmp_path / "jobs" / "q.serve.json").read_text())
    assert record["capped"] is False
    assert "long-lived" in record["capped_note"]


# --------------------------------------------------------------------- the artifact


def test_a_complete_artifact_reads_back(tmp_path):
    info = qz.read_artifact(_artifact(tmp_path / "a", bits=4))
    m = info["manifest"]
    assert (m["format"], m["bits"], m["head_dtype"]) == ("torch_int4_wo", 4, "float32")
    assert m["artifact_format"] == qz.ARTIFACT_FORMAT
    assert set(m["files"]) == {qz.TORSO_FILE, player.HEAD, player.CONFIG, *player.TOKENIZER_FILES}
    assert m["torchao_version"] == "0.18.0"
    assert m["torch_version"] == "2.12.1"
    assert m["base_model_revision"] == REV
    assert m["lora_merged"] is True


@pytest.mark.parametrize(
    "name", [qz.MANIFEST_FILE, qz.TORSO_FILE, player.HEAD, player.CONFIG, "tokenizer.json"]
)
def test_an_artifact_missing_a_file_is_refused(tmp_path, name):
    root = _artifact(tmp_path / "a")
    (root / name).unlink()
    with pytest.raises(CliError, match=name):
        qz.read_artifact(root)


def test_an_edited_weight_file_fails_the_hash_check(tmp_path):
    root = _artifact(tmp_path / "a")
    (root / qz.TORSO_FILE).write_bytes(b"tampered")
    with pytest.raises(CliError, match="sha256"):
        qz.read_artifact(root)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"artifact_format": "other/1"}, "artifact_format"),
        ({"bits": 3}, "bits"),
        ({"format": "torch_int4_wo"}, "format"),
        ({"head_dtype": "bfloat16"}, "float32"),
        ({"base_model_revision": "main"}, "base_model_revision"),
        ({"torchao_version": ""}, "torchao_version"),
        ({"torch_version": None}, "torch_version"),
        ({"source": {}}, "source"),
        ({"torchao_api": ""}, "torchao_api"),
    ],
)
def test_a_manifest_that_does_not_describe_itself_is_refused(tmp_path, changes, match):
    root = _artifact(tmp_path / "a")
    _rewrite(root, **changes)
    with pytest.raises(CliError, match=match):
        qz.read_artifact(root)


def test_a_non_pointer_config_is_refused(tmp_path):
    root = _artifact(tmp_path / "a")
    (root / player.CONFIG).write_text(json.dumps({"head_type": "slot"}))
    qz.write_manifest(root, **{k: _manifest(root)[k] for k in qz.MANIFEST_INPUTS})
    with pytest.raises(CliError, match="pointer"):
        qz.read_artifact(root)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(CliError, match="not a local"):
        qz.read_artifact(tmp_path / "nope")


# --------------------------------------------------------------------- serving: loopback only


def test_serve_argv_is_loopback_only(tmp_path):
    root = _artifact(tmp_path / "a")
    with pytest.raises(CliError, match="127.0.0.1") as err:
        qz.serve_argv("py", root, port=8000, host="0.0.0.0")  # nosec B104 - refused
    assert err.value.code == EXIT_USER_ERROR


def test_serve_argv_runs_the_wrapper_module_with_jev_factory_on_the_path(tmp_path):
    root = _artifact(tmp_path / "a")
    argv = qz.serve_argv("py", root, port=8123, decider_repo=tmp_path / "sd", model_name="m")
    assert argv[0] == "env"
    assert argv[1].startswith("PYTHONPATH=")
    assert str(tmp_path / "sd" / "src") in argv[1]
    assert argv[2:] == [
        "py",
        "-m",
        qz.MODULE,
        "serve",
        str(root),
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--model-name",
        "m",
    ]


def test_serve_argv_refuses_a_hobson_bundle_or_a_broken_artifact(tmp_path):
    root = _artifact(tmp_path / "a")
    (root / qz.MANIFEST_FILE).unlink()
    with pytest.raises(CliError, match=qz.MANIFEST_FILE):
        qz.serve_argv("py", root, port=8000)


def test_the_serve_entry_refuses_a_non_loopback_host_before_loading_anything(tmp_path, capsys):
    root = _artifact(tmp_path / "a")
    rc = qz.main(["serve", str(root), "--host", "0.0.0.0", "--port", "8000"])  # nosec B104
    assert rc == EXIT_USER_ERROR
    assert "127.0.0.1" in capsys.readouterr().err


def test_make_server_refuses_a_non_loopback_host():
    with pytest.raises(CliError, match="127.0.0.1"):
        qz.make_server("0.0.0.0", 0, evaluate=dict, health=dict)  # nosec B104 - refused


def test_launch_needs_the_student_env_and_records_what_answers(tmp_path, monkeypatch):
    root = _artifact(tmp_path / "a")
    started = []
    monkeypatch.setattr(qz, "start_detached", lambda j, n, argv: started.append(argv) or 4242)
    with pytest.raises(CliError, match="student"):
        qz.launch(tmp_path / "jobs", "q", None, root, port=8000)
    env = EnvSpec(role="student", python="/envs/student/bin/python", lockfile_hash="c" * 64)
    pid = qz.launch(tmp_path / "jobs", "q", env, root, port=8000, decider_repo=tmp_path / "sd")
    assert pid == 4242
    assert started[0][2] == "/envs/student/bin/python"
    record = json.loads((tmp_path / "jobs" / "q.serve.json").read_text())
    assert record["argv"] == started[0]
    assert record["environment"]["lockfile_hash"] == "c" * 64
    assert record["artifact"]["format"] == "torch_int8_wo"
    assert record["artifact"]["files"][qz.TORSO_FILE] == _manifest(root)["files"][qz.TORSO_FILE]


# --------------------------------------------------------------------- the HTTP wrapper


#: The upstream response shape (strands_decider.schema.SystemOneResponse @ 8b6faa5).
ANSWER = {
    "model": "q",
    "answers": {
        "c": {
            "type": "choice",
            "choice": "fix",
            "probabilities": {"fix": 0.8, "wait": 0.2},
            "confidence": 0.6,
        },
        "n": {"type": "noul", "noul": 0.7},
        "s": {
            "type": "score",
            "score": 1.2,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
            "confidence": 0.5,
        },
    },
    "usage": {"input_tokens": 10, "output_tokens": 0},
}


@pytest.fixture
def server():
    seen = []

    def evaluate(body):
        seen.append(body)
        if body.get("state") == "too many options":
            raise qz.RequestError(422, "option count over num_slots")
        return dict(ANSWER)

    srv = qz.make_server(
        "127.0.0.1", 0, evaluate=evaluate, health=lambda: {"status": "ok", "model": "q"}
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    finally:
        srv.shutdown()
        srv.server_close()


def _post(url, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(  # nosec B310 - loopback test server
        url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_answers_ok(server):
    url, _ = server
    with urllib.request.urlopen(url + "/health", timeout=10) as resp:  # nosec B310
        assert json.loads(resp.read()) == {"status": "ok", "model": "q"}


def test_systemone_answers_with_the_upstream_shape_plus_latency(server):
    url, seen = server
    status, body = _post(url, {"id": "r", "state": "s", "questions": {}})
    assert status == 200
    assert seen == [{"id": "r", "state": "s", "questions": {}}]
    assert {k: v for k, v in body.items() if k != "latency_ms"} == ANSWER
    assert isinstance(body["latency_ms"], float)


def test_a_caller_error_is_a_422_with_detail_like_upstream(server):
    url, _ = server
    assert _post(url, {"state": "too many options"}) == (
        422,
        {"detail": "option count over num_slots"},
    )
    status, body = _post(url, b"{not json")
    assert status == 422
    assert "detail" in body


def test_an_unknown_path_is_a_404(server):
    url, _ = server
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(url + "/v1/other", timeout=10)  # nosec B310
    assert err.value.code == 404


def test_the_shared_systemone_client_reads_the_wrappers_answers(server):
    from jev_factory.backbones.pointer.serve import PointerClient

    url, _ = server
    record = {
        "id": "r",
        "state": "s",
        "questions": {
            "c": {"type": "choice", "criteria": {"fix": "fix it", "wait": "wait"}},
            "n": {"type": "noul", "criteria": None},
            "s": {"type": "score", "criteria": ["low", "mid", "high"]},
        },
    }
    result = PointerClient(url, model="q").ask(record)
    by = {q.qid: q for q in result.questions}
    assert all(q.ok for q in result.questions), [q.error for q in result.questions]
    assert by["c"].distribution == {"fix": 0.8, "wait": 0.2}
    assert by["n"].p_true == pytest.approx(0.7)
    assert set(by["s"].distribution) == {"low", "mid", "high"}


def test_importing_the_module_loads_no_heavy_package():
    code = (
        "import sys, jev_factory.backbones.pointer.quantized as q; "
        "heavy = [m for m in ('torch', 'torchao', 'strands_decider', 'transformers') "
        "if m in sys.modules]; print(heavy)"
    )
    out = subprocess.run(  # nosec B603 - this interpreter
        [os.sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


# --------------------------------------------------------------------- torch / torchao


def _toy_torso(torch):
    """A two-block decoder-shaped module with the names the policy keys on."""

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(256)
            self.linear_attn = torch.nn.Module()
            self.linear_attn.in_proj_qkv = torch.nn.Linear(256, 256, bias=False)
            self.linear_attn.in_proj_a = torch.nn.Linear(256, 8, bias=False)
            self.linear_attn.A_log = torch.nn.Parameter(torch.zeros(8))
            self.mlp = torch.nn.Module()
            self.mlp.gate_proj = torch.nn.Linear(256, 256, bias=False)

        def forward(self, x):
            h = self.input_layernorm(x)
            return x + self.mlp.gate_proj(self.linear_attn.in_proj_qkv(h))

    class Torso(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(1000, 256)
            self.layers = torch.nn.ModuleList([Block(), Block()])
            self.visual = torch.nn.Linear(4, 4)
            self.mtp = torch.nn.Linear(4, 4)

        def forward(self, ids):
            x = self.embed_tokens(ids)
            for block in self.layers:
                x = block(x)
            return x

    return Torso()


def test_packed_embedding_keeps_every_row_and_roundtrips(tmp_path):
    """c9: the full-vocabulary embedding is packed int4 / int8 and reloads bit-identically."""
    torch = pytest.importorskip("torch")
    for bits, qdtype in ((4, torch.uint8), (8, torch.int8)):
        torso = _toy_torso(torch).to(torch.bfloat16)
        ids = torch.randint(0, 1000, (2, 9))
        want = torso.embed_tokens(ids).float()
        dropped = qz.drop_unused(torso)
        assert [d["module"] for d in dropped] == ["visual", "mtp"]
        record = qz.pack_embedding(torso, qz.QuantScheme(embedding_bits=bits))
        assert (record["rows"], record["dim"], record["bits"]) == (1000, 256, bits)
        got = torso.embed_tokens(ids)
        assert got.dtype == torch.bfloat16
        tolerance = 0.08 if bits == 4 else 0.01
        assert (got.float() - want).abs().max() <= tolerance * want.abs().max()
        state = torso.state_dict()
        assert state["embed_tokens.qweight"].dtype == qdtype
        assert state["embed_tokens.qweight"].shape[0] == 1000  # every row kept
        assert not any(k.startswith(("visual", "mtp")) for k in state)
        torch.save(state, tmp_path / f"e{bits}.pt")
        fresh = _toy_torso(torch).to(torch.bfloat16)
        qz.drop_unused(fresh)
        qz.install_packed_embedding(fresh, record)
        fresh.load_state_dict(torch.load(tmp_path / f"e{bits}.pt", weights_only=True), assign=True)
        assert torch.equal(fresh.embed_tokens(ids), got)


def test_weight_only_quantization_follows_the_tensor_policy_and_roundtrips(tmp_path):
    """c9: block linears quantized, in_proj_a / A_log / norms kept, embedding packed."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    # int4 blocks need a CUDA kernel (torchao 0.18 on CPU: "Requires mslk >= 1.0.0"); int8
    # runs on CPU. An int8 embedding keeps the tolerance about the block linears.
    scheme = qz.QuantScheme(block_bits=8, embedding_bits=8, block_group_size=128)
    torso = _toy_torso(torch).to(torch.bfloat16)
    ids = torch.randint(0, 1000, (2, 9))
    want = torso(ids)
    record = qz.quantize_torso(torso, scheme)
    assert record["torchao_api"] in qz.TORCHAO_APIS[8]
    state = torso.state_dict()
    summary = qz.check_inventory(qz.state_inventory(state), scheme)
    assert summary["tensors"][qz.BLOCK_LINEAR] == 4
    assert type(state["layers.0.linear_attn.in_proj_a.weight"]) is torch.Tensor
    got = torso(ids)
    assert torch.allclose(want.float(), got.float(), atol=0.1, rtol=0.1)
    path = tmp_path / "t8.pt"
    torch.save(state, path)
    fresh = _toy_torso(torch).to(torch.bfloat16)
    qz.drop_unused(fresh)
    qz.install_packed_embedding(fresh, record["embedding"])
    qz.load_quantized_state(fresh, path, device="cpu")
    assert torch.allclose(fresh(ids).float(), got.float())


# --------------------------------------------------------------------- live (opt-in)

PY = os.environ.get("JEV_STRANDS_PY")
SRC = os.environ.get("JEV_STRANDS_SRC")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(
    not (PY and SRC), reason="set JEV_STRANDS_PY and JEV_STRANDS_SRC to run the live int8 test"
)
def test_live_int8_build_serves_and_answers_choice_noul_and_score(tmp_path):
    from jev_factory.backbones.pointer.serve import PointerClient, wait_ready

    lock = (
        os.environ.get("JEV_STRANDS_LOCK")
        or hashlib.sha256(
            subprocess.run(  # nosec B603
                [str(PY), "-m", "pip", "freeze"], check=True, capture_output=True
            ).stdout
        ).hexdigest()
    )
    repo = os.environ.get("JEV_STRANDS_REPO")
    art = tmp_path / "int8"
    qz.build_weight_only(Path(str(SRC)), art, 8, python=str(PY), decider_repo=repo, apply=True)
    assert qz.read_artifact(art)["manifest"]["bits"] == 8
    env = EnvSpec(role="student", python=str(PY), lockfile_hash=lock)
    port = _free_port()
    pid = qz.launch(tmp_path / "jobs", "q8", env, art, port=port, decider_repo=repo)
    try:
        url = f"http://127.0.0.1:{port}"
        wait_ready(url, timeout=900.0, jobdir=tmp_path / "jobs", name="q8")
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
        result = PointerClient(url, model="q8").ask(record)
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


# --------------------------------------------------------------------- path containment


def test_read_artifact_refuses_a_weight_file_symlinked_out_of_the_artifact(tmp_path):
    root = _artifact(tmp_path / "a")
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(b"quantized-torso")
    (root / qz.TORSO_FILE).unlink()
    (root / qz.TORSO_FILE).symlink_to(outside)
    with pytest.raises(CliError, match="escapes"):
        qz.read_artifact(root)


def test_launch_refuses_a_server_name_that_escapes_the_job_directory(tmp_path, monkeypatch):
    root = _artifact(tmp_path / "a")
    monkeypatch.setattr(qz, "start_detached", lambda j, n, argv: 1)
    env = EnvSpec(role="student", python="/envs/student/bin/python", lockfile_hash="c" * 64)
    with pytest.raises(CliError, match="escapes"):
        qz.launch(tmp_path / "jobs", "../../evil", env, root, port=8000)
    assert not (tmp_path / "evil.serve.json").exists()


def test_the_build_entry_refuses_a_checkpoint_member_symlink_before_loading_torch(
    tmp_path, capsys, monkeypatch
):
    ckpt = _checkpoint(tmp_path / "ckpt")
    (ckpt / "provenance.json").write_text(json.dumps({"base_model_revision": REV}))
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps(CONFIG))
    (ckpt / player.CONFIG).unlink()
    (ckpt / player.CONFIG).symlink_to(outside)
    monkeypatch.setattr(qz, "_load", lambda name: pytest.fail(f"loaded {name}"))
    argv = ["build", str(ckpt), str(tmp_path / "out"), "--bits", "8", "--base-revision", REV]
    assert qz.main(argv) == EXIT_USER_ERROR
    assert "escapes" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------- #20 t10: the scheme


def test_the_default_scheme_is_int4_blocks_and_an_int4_embedding():
    """c9: text-block linears 4-bit, embedding INT4 by default; INT8 is a recorded choice."""
    scheme = qz.QuantScheme()
    assert (scheme.block_bits, scheme.embedding_bits) == (4, 4)
    assert qz.as_scheme(8) == qz.QuantScheme(block_bits=8)
    assert qz.as_scheme(qz.QuantScheme(embedding_bits=8)).embedding_bits == 8


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"block_bits": 2}, "block_bits"),
        ({"embedding_bits": 16}, "embedding_bits"),
        ({"block_group_size": 0}, "block_group_size"),
        ({"embedding_group_size": True}, "embedding_group_size"),
    ],
)
def test_a_scheme_outside_int4_int8_or_a_bad_group_is_refused(fields, match):
    scheme = qz.QuantScheme(**fields)
    with pytest.raises(CliError, match=match):
        scheme.validated()


def test_the_build_argv_carries_the_whole_scheme(tmp_path):
    scheme = qz.QuantScheme(block_bits=8, embedding_bits=8, embedding_group_size=64)
    _ckpt, plan = _plan(tmp_path, bits=scheme)
    flags = dict(zip(plan["argv"][6::2], plan["argv"][7::2]))
    assert flags["--bits"] == "8"
    assert flags["--embedding-bits"] == "8"
    assert flags["--block-group-size"] == "256"
    assert flags["--embedding-group-size"] == "64"
    assert plan["scheme"] == scheme.record()
    assert plan["format"] == "torch_int8_wo"


def test_the_build_entry_parses_the_scheme_flags(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(qz, "_build_in_env", lambda *a: seen.append(a) or 0)
    argv = ["build", "c", "o", "--bits", "4", "--embedding-bits", "8", "--base-revision", REV]
    assert qz.main(argv) == 0
    assert seen[0][2] == qz.QuantScheme(block_bits=4, embedding_bits=8)


# --------------------------------------------------------------------- the tensor policy


@pytest.mark.parametrize(
    "name, cls",
    [
        ("model.layers.3.self_attn.q_proj", qz.BLOCK_LINEAR),
        ("model.layers.3.self_attn.o_proj", qz.BLOCK_LINEAR),
        ("model.layers.0.linear_attn.in_proj_qkv", qz.BLOCK_LINEAR),
        ("model.layers.0.linear_attn.in_proj_z", qz.BLOCK_LINEAR),
        ("model.layers.0.linear_attn.out_proj", qz.BLOCK_LINEAR),
        ("model.layers.0.mlp.down_proj", qz.BLOCK_LINEAR),
        ("model.layers.0.linear_attn.in_proj_a", qz.KEEP),
        ("model.layers.0.linear_attn.in_proj_b", qz.KEEP),
        ("model.layers.0.linear_attn.conv1d", qz.KEEP),
        ("model.layers.0.linear_attn.norm", qz.KEEP),
        ("model.layers.0.input_layernorm", qz.KEEP),
        ("model.norm", qz.KEEP),
        ("model.embed_tokens", qz.EMBEDDING),
        ("model.language_model.embed_tokens", qz.EMBEDDING),
        ("model.visual.blocks.0.attn.qkv", qz.DROP),
        ("visual.merger.mlp.0", qz.DROP),
        ("mtp.layers.0.mlp.gate_proj", qz.DROP),
        ("model.mtp_layers.0.fc", qz.DROP),
    ],
)
def test_the_module_policy_names_each_tensor_class(name, cls):
    """c9: block linears 4-bit; in_proj_a/b, conv1d, norms kept; vision and MTP dropped."""
    assert qz.module_class(name) == cls


def test_state_keys_take_their_modules_class():
    assert qz.key_class("model.layers.0.linear_attn.A_log") == qz.KEEP
    assert qz.key_class("model.layers.0.linear_attn.dt_bias") == qz.KEEP
    assert qz.key_class("model.layers.0.mlp.up_proj.weight") == qz.BLOCK_LINEAR
    assert qz.key_class("model.embed_tokens.qweight") == qz.EMBEDDING


def _row(key, tensor_type="Tensor", dtype="torch.bfloat16"):
    return {"key": key, "tensor_type": tensor_type, "dtype": dtype}


GOOD_INVENTORY = [
    _row("model.embed_tokens.qweight", dtype="torch.uint8"),
    _row("model.embed_tokens.scales"),
    _row("model.layers.0.self_attn.q_proj.weight", "AffineQuantizedTensor"),
    _row("model.layers.0.linear_attn.in_proj_qkv.weight", "Int4TilePackedTo4dTensor"),
    _row("model.layers.0.linear_attn.in_proj_a.weight"),
    _row("model.layers.0.linear_attn.in_proj_b.weight"),
    _row("model.layers.0.linear_attn.conv1d.weight"),
    _row("model.layers.0.linear_attn.A_log", dtype="torch.float32"),
    _row("model.layers.0.linear_attn.dt_bias"),
    _row("model.layers.0.input_layernorm.weight"),
    _row("model.norm.weight"),
]


def test_a_torso_that_follows_the_policy_passes_with_its_counts():
    summary = qz.check_inventory(GOOD_INVENTORY, qz.QuantScheme())
    assert summary["tensors"] == {
        qz.DROP: 0,
        qz.EMBEDDING: 2,
        qz.BLOCK_LINEAR: 2,
        qz.KEEP: 7,
    }
    assert summary["kept_unquantized"] == {
        "A_log": 1,
        "conv1d": 1,
        "dt_bias": 1,
        "in_proj_a": 1,
        "in_proj_b": 1,
        "norm": 2,
    }


@pytest.mark.parametrize(
    "change, match",
    [
        ({"add": _row("model.visual.patch_embed.proj.weight")}, "vision tower or the MTP"),
        ({"add": _row("mtp.fc.weight")}, "vision tower or the MTP"),
        (
            {"replace": 2, "row": _row("model.layers.0.self_attn.q_proj.weight")},
            "left unquantized",
        ),
        ({"replace": 0, "row": _row("model.embed_tokens.weight")}, "not packed"),
        (
            {"replace": 0, "row": _row("model.embed_tokens.qweight", dtype="torch.int8")},
            "not torch.uint8",
        ),
        (
            {"replace": 4, "row": _row("model.layers.0.linear_attn.in_proj_a.weight", "AQT")},
            "outside the text-block linears",
        ),
        (
            {"replace": 7, "row": _row("model.layers.0.linear_attn.A_log", dtype="torch.int8")},
            "unquantized floating",
        ),
        ({"replace": 10, "row": _row("model.norm.weight", "Int8Tensor")}, "outside the text"),
    ],
)
def test_a_torso_that_breaks_the_policy_is_refused(change, match):
    """c9: vision/MTP absent, block linears quantized, embedding packed, small params kept."""
    rows = list(GOOD_INVENTORY)
    if "add" in change:
        rows.append(change["add"])
    else:
        rows[change["replace"]] = change["row"]
    scheme = qz.QuantScheme()
    with pytest.raises(CliError, match=match):
        qz.check_inventory(rows, scheme)


def test_an_int8_embedding_expects_int8_storage():
    rows = list(GOOD_INVENTORY)
    rows[0] = _row("model.embed_tokens.qweight", dtype="torch.int8")
    summary = qz.check_inventory(rows, qz.QuantScheme(embedding_bits=8))
    assert summary["tensors"][qz.EMBEDDING] == 2


def test_a_torso_without_block_linears_or_embedding_is_refused():
    keep_only = [r for r in GOOD_INVENTORY if qz.key_class(r["key"]) == qz.KEEP]
    scheme = qz.QuantScheme()
    with pytest.raises(CliError, match="no text-block linear"):
        qz.check_inventory(keep_only, scheme)


def test_every_tokenizer_id_needs_an_embedding_row():
    """c9: full vocabulary; no id is unreachable."""
    assert qz.check_full_vocabulary(248320, 248077)["full_vocabulary"] is True
    assert qz.check_full_vocabulary(248320, 248320)["embedding_rows"] == 248320
    with pytest.raises(CliError, match="every id"):
        qz.check_full_vocabulary(50000, 248077)


# --------------------------------------------------------------------- manifest: hashes, bytes


def test_the_manifest_records_per_file_sha256_and_the_weight_byte_sum(tmp_path):
    """c9/h21: per-file sha256 (tokenizer too) and the byte sum of the weight files."""
    root = _artifact(tmp_path / "a", bits=4)
    m = qz.read_artifact(root)["manifest"]
    for name in (qz.TORSO_FILE, player.HEAD, player.CONFIG, *player.TOKENIZER_FILES):
        assert m["files"][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    sizes = {n: (root / n).stat().st_size for n in qz.WEIGHT_FILES}
    assert m["weight_files"] == sizes
    assert m["weight_bytes"] == sum(sizes.values())
    assert m["size_budget_bytes"] == 10**9
    assert m["within_size_budget"] is True
    assert m["scheme"] == qz.QuantScheme(block_bits=4).record()


def test_a_weight_byte_sum_that_does_not_match_the_files_is_refused(tmp_path):
    root = _artifact(tmp_path / "a")
    _rewrite(root, weight_bytes=1)
    with pytest.raises(CliError, match="weight_bytes"):
        qz.read_artifact(root)


def test_the_weight_record_checks_the_budget_exactly_at_ten_to_the_ninth(tmp_path):
    """c9/h21: 10^9 bytes is over budget; 10^9 - 1 is within (sparse files, no disk used)."""
    for total, within in ((10**9, False), (10**9 - 1, True)):
        root = tmp_path / str(total)
        root.mkdir()
        with (root / qz.TORSO_FILE).open("wb") as fh:
            fh.truncate(total - 7)
        (root / player.HEAD).write_bytes(b"x" * 7)
        record = qz.weight_record(root)
        assert record["weight_bytes"] == total
        assert record["within_size_budget"] is within


def test_a_scheme_that_disagrees_with_bits_is_refused(tmp_path):
    root = _artifact(tmp_path / "a", bits=4)
    _rewrite(root, scheme=qz.QuantScheme(block_bits=8).record())
    with pytest.raises(CliError, match="block_bits"):
        qz.read_artifact(root)


def test_the_source_tokenizer_ships_unmodified(tmp_path):
    """c9: the artifact's tokenizer files are the source checkpoint's, byte for byte."""
    ckpt = _checkpoint(tmp_path / "ckpt")
    source = qz.source_record(ckpt)
    for name in player.TOKENIZER_FILES:
        assert source["sha256"][name] == player.sha256_file(ckpt / name)
    root = _artifact(tmp_path / "a", source=source)
    tok = qz.read_artifact(root)["manifest"]["tokenizer"]
    assert tok["unmodified"] is True
    assert set(tok["source_sha256"]) == set(player.TOKENIZER_FILES)


def test_a_tokenizer_that_differs_from_its_source_is_refused(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    (ckpt / "tokenizer.json").write_text('{"model": "edited"}')
    source = qz.source_record(ckpt)  # the artifact keeps the original "{}"
    root = _artifact(tmp_path / "a", source=source)
    assert _manifest(root)["tokenizer"]["unmodified"] is False
    with pytest.raises(CliError, match="tokenizer"):
        qz.read_artifact(root)


def test_a_manifest_missing_a_tokenizer_hash_is_refused(tmp_path):
    root = _artifact(tmp_path / "a")
    files = dict(_manifest(root)["files"])
    files.pop("tokenizer.json")
    _rewrite(root, files=files)
    with pytest.raises(CliError, match="tokenizer.json"):
        qz.read_artifact(root)


# --------------------------------------------------------------------- /v1/bundle, /v1/stats


def _tree(root: Path, *, cgroup: str, limits: dict, tegra: str | None, model: bytes | None):
    """A fake /proc, /sys/fs/cgroup, /etc and device tree under *root*."""
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "status").write_text("Name:\tpython\nVmRSS:\t  812340 kB\nVmSwap:\t0 kB\n")
    (proc / "self" / "cgroup").write_text(cgroup)
    (proc / "meminfo").write_text(
        "MemTotal:        7799340 kB\nMemAvailable:    5123456 kB\nCached: 1 kB\n"
    )
    cg = root / "cgroup"
    for rel, value in limits.items():
        (cg / rel).mkdir(parents=True, exist_ok=True)
        name = "memory.max" if not rel.startswith("memory") else "memory.limit_in_bytes"
        (cg / rel / name).write_text(value + "\n")
    if tegra is not None:
        (root / "etc").mkdir()
        (root / "etc" / "nv_tegra_release").write_text(tegra)
    models = []
    if model is not None:
        (root / "dt").mkdir()
        (root / "dt" / "model").write_bytes(model)
        models.append(root / "dt" / "model")
    return qz.HostPaths(
        proc=proc,
        cgroup_root=cg,
        tegra_release=root / "etc" / "nv_tegra_release",
        device_models=(*models, root / "dmi" / "product_name"),
    )


SCOPE = "user.slice/user-1000.slice/user@1000.service/app.slice/run-u7.scope"
TEGRA = "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64\n"


def test_host_stats_reports_rss_memavailable_board_l4t_and_the_cgroup_cap(tmp_path):
    """c1/h1: RSS, MemAvailable, device model, nv_tegra_release and the cgroup memory.max."""
    paths = _tree(
        tmp_path,
        cgroup=f"0::/{SCOPE}\n",
        limits={SCOPE: "4294967296", "user.slice": "max"},
        tegra=TEGRA,
        model=b"NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\x00",
    )
    stats = qz.host_stats(paths)
    assert stats["rss_bytes"] == 812340 * 1024
    assert stats["mem_available_bytes"] == 5123456 * 1024
    assert stats["device_model"] == (
        "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"
    )
    assert stats["nv_tegra_release"] == TEGRA.strip()
    assert stats["cgroup"] == {
        "version": 2,
        "path": "/" + SCOPE,
        "memory_max": 4294967296,
        "effective_memory_max": 4294967296,
    }


def test_host_stats_without_a_tegra_release_or_a_cap_says_so(tmp_path):
    """c1/h1: nv_tegra_release only when present; an uncapped cgroup reports 'max'."""
    paths = _tree(tmp_path, cgroup=f"0::/{SCOPE}\n", limits={SCOPE: "max"}, tegra=None, model=None)
    stats = qz.host_stats(paths)
    assert stats["nv_tegra_release"] is None
    assert stats["device_model"] is None
    assert stats["cgroup"]["memory_max"] == "max"
    assert stats["cgroup"]["effective_memory_max"] is None


def test_an_ancestor_cap_is_the_effective_cap(tmp_path):
    paths = _tree(
        tmp_path,
        cgroup=f"0::/{SCOPE}\n",
        limits={SCOPE: "max", "user.slice/user-1000.slice": "8589934592"},
        tegra=None,
        model=None,
    )
    cgroup = qz.cgroup_memory(paths)
    assert cgroup["memory_max"] == "max"
    assert cgroup["effective_memory_max"] == 8589934592


def test_a_cgroup_v1_memory_limit_is_read_too(tmp_path):
    paths = _tree(
        tmp_path,
        cgroup="5:cpuset:/\n4:memory:/jev\n",
        limits={"memory/jev": "4294967296", "memory": str(1 << 63)},
        tegra=None,
        model=None,
    )
    cgroup = qz.cgroup_memory(paths)
    assert cgroup["version"] == 1
    assert cgroup["memory_max"] == 4294967296
    assert cgroup["effective_memory_max"] == 4294967296


def test_no_cgroup_file_reports_no_cap(tmp_path):
    paths = qz.HostPaths(proc=tmp_path / "none", cgroup_root=tmp_path / "cg")
    assert qz.cgroup_memory(paths)["memory_max"] is None


@pytest.fixture
def attesting(tmp_path):
    art = _artifact(tmp_path / "a", bits=4)
    paths = _tree(
        tmp_path / "host",
        cgroup=f"0::/{SCOPE}\n",
        limits={SCOPE: "4294967296"},
        tegra=TEGRA,
        model=b"NVIDIA Jetson AGX Orin Developer Kit\x00",
    )
    srv = qz.make_server(
        "127.0.0.1",
        0,
        evaluate=lambda body: dict(ANSWER),
        health=lambda: {"status": "ok"},
        bundle=lambda: qz.bundle_attestation(art),
        stats=lambda: qz.host_stats(paths),
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", art
    finally:
        srv.shutdown()
        srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310 - loopback test server
        return json.loads(resp.read())


def test_get_bundle_answers_the_manifests_per_file_sha256(attesting):
    """c1/h1: GET /v1/bundle carries the manifest's per-file sha256 and weight bytes."""
    url, art = attesting
    body = _get(url + "/v1/bundle")
    manifest = _manifest(art)
    assert body["files"] == manifest["files"]
    assert body["weight_bytes"] == manifest["weight_bytes"]
    assert body["within_size_budget"] is True
    assert body["manifest_sha256"] == player.sha256_file(art / qz.MANIFEST_FILE)
    assert body["format"] == "torch_int4_wo"
    assert body["scheme"] == manifest["scheme"]


def test_get_stats_answers_the_host_facts(attesting):
    """c1/h1: GET /v1/stats: RSS, MemAvailable, board, L4T release and the cgroup cap."""
    url, _art = attesting
    body = _get(url + "/v1/stats")
    assert body["rss_bytes"] == 812340 * 1024
    assert body["mem_available_bytes"] == 5123456 * 1024
    assert body["device_model"] == "NVIDIA Jetson AGX Orin Developer Kit"
    assert body["nv_tegra_release"].startswith("# R36 (release)")
    assert body["cgroup"]["memory_max"] == 4294967296


def test_a_server_without_attestation_callables_answers_404(server):
    url, _ = server
    for path in ("/v1/bundle", "/v1/stats"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url + path, timeout=10)  # nosec B310
        assert err.value.code == 404


def test_a_failing_stats_read_is_a_500_not_a_dropped_connection(tmp_path):
    def broken():
        raise OSError("no /proc")

    srv = qz.make_server("127.0.0.1", 0, evaluate=lambda b: {}, health=lambda: {}, stats=broken)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(  # nosec B310
                f"http://127.0.0.1:{srv.server_address[1]}/v1/stats", timeout=10
            )
        assert err.value.code == 500
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------- pruned vocabulary (c38/h9)

KEPT = [0, 3, 4, 7, 151643]


class FakeTokenizer:
    all_special_ids = [151643, 151645]
    added_tokens_decoder = {151644: "<|im_start|>", 151645: "<|im_end|>"}
    pad_token_id = 151643
    eos_token_id = 151645
    bos_token_id = None
    unk_token_id = None


def _prompts(path: Path, sides=("train", "fit", "selection")) -> Path:
    body = {"id": "r1", "state": "s", "questions": {"c": {"type": "choice", "criteria": {}}}}
    path.write_text("".join(json.dumps({"side": s, "body": body}) + "\n" for s in sides))
    return path


def test_the_kept_vocabulary_is_the_prompt_ids_plus_the_special_tokens():
    """h9: the embedding keeps only the ids the prompts use, plus the special tokens."""
    special = qz.special_token_ids(FakeTokenizer())
    assert special == [151643, 151644, 151645]
    kept = qz.kept_vocabulary([7, 3, 3, 0, 4], special, vocab_rows=248320)
    assert kept == [0, 3, 4, 7, 151643, 151644, 151645]


def test_the_kept_vocabulary_refuses_an_id_without_a_row_or_an_empty_prompt_set():
    with pytest.raises(CliError, match="outside the embedding"):
        qz.kept_vocabulary([5, 300], [], vocab_rows=256)
    with pytest.raises(CliError, match="empty"):
        qz.kept_vocabulary([], [1, 2], vocab_rows=256)


def test_a_pruned_manifest_records_the_kept_id_list_sha256_and_weight_bytes(tmp_path):
    """h9: the pruned artifact's manifest records the kept id list sha256 and weight bytes."""
    art = _artifact(tmp_path / "p", bits=4, kept_ids=KEPT)
    manifest = qz.read_artifact(art)["manifest"]
    record = manifest["pruned_vocabulary"]
    on_disk = player.sha256_file(art / qz.KEPT_IDS_FILE)
    assert record["kept_ids_sha256"] == on_disk
    assert (
        record["kept_ids_sha256"]
        == hashlib.sha256(json.dumps(KEPT).replace(" ", "").encode()).hexdigest()
    )
    assert record["kept_ids_count"] == len(KEPT)
    assert record["comparison_only"] is True
    assert manifest["files"][qz.KEPT_IDS_FILE] == on_disk
    assert manifest["weight_bytes"] == sum((art / n).stat().st_size for n in qz.WEIGHT_FILES)
    assert manifest["format"] == "torch_int4_wo_pruned_vocab"
    assert manifest["format"] not in qz.FORMAT_OF_BITS.values()  # never a shipped format name
    assert qz.read_kept_ids(art) == KEPT
    attested = qz.bundle_attestation(art)
    assert attested["pruned_vocabulary"] == record
    assert attested["weight_bytes"] == manifest["weight_bytes"]


def test_a_full_vocabulary_manifest_is_not_pruned(tmp_path):
    art = _artifact(tmp_path / "f", bits=4)
    manifest = qz.read_artifact(art)["manifest"]
    assert not qz.is_pruned(manifest)
    assert manifest["format"] == "torch_int4_wo"
    assert not (art / qz.KEPT_IDS_FILE).exists()
    assert "pruned_vocabulary" not in qz.bundle_attestation(art)


def test_an_edited_kept_id_list_is_refused(tmp_path):
    art = _artifact(tmp_path / "p", bits=8, kept_ids=KEPT)
    (art / qz.KEPT_IDS_FILE).write_text("[0,3,4,7,151643,9]")
    with pytest.raises(CliError, match="sha256"):
        qz.read_artifact(art)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"format": "torch_int8_wo"}, "does not match"),
        ({"pruned_vocabulary": {"kept_ids_file": "x.json"}}, "must name"),
    ],
)
def test_a_pruned_manifest_that_misdescribes_itself_is_refused(tmp_path, changes, match):
    art = _artifact(tmp_path / "p", bits=8, kept_ids=KEPT)
    _rewrite(art, **changes)
    with pytest.raises(CliError, match=match):
        qz.read_artifact(art)


def test_a_pruned_manifest_must_stay_comparison_only(tmp_path):
    art = _artifact(tmp_path / "p", bits=8, kept_ids=KEPT)
    record = {**_manifest(art)["pruned_vocabulary"], "comparison_only": False}
    _rewrite(art, pruned_vocabulary=record)
    with pytest.raises(CliError, match="comparison only"):
        qz.read_artifact(art)


@pytest.mark.parametrize("ids", [[3, 1], [1, 1], [-1, 2], [], [True]])
def test_a_kept_id_list_must_be_sorted_unique_non_negative_ids(tmp_path, ids):
    root = tmp_path / "p"
    with pytest.raises(CliError, match="kept_ids"):
        _artifact(root, bits=8, kept_ids=ids)


def test_the_pruned_build_plan_carries_the_prompt_set_and_writes_nothing(tmp_path):
    """h9: the pruned build is planned with the run's prompt set; dry-run writes nothing."""
    ckpt = _checkpoint(tmp_path / "ckpt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    plan = qz.build_pruned_vocabulary(
        ckpt, tmp_path / "out", 4, prompts, python="py", base_revision=REV
    )
    assert plan["dry_run"] is True
    assert plan["format"] == "torch_int4_wo_pruned_vocab"
    assert plan["comparison_only"] is True
    assert plan["argv"][-2:] == ["--prompts", str(prompts)]
    assert plan["prompt_set"]["counts"] == {"train": 1, "fit": 1, "selection": 1}
    assert plan["prompt_set"]["sha256"] == player.sha256_file(prompts)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["test", "held_out", "heldout", None])
def test_a_prompt_set_with_a_protected_or_unknown_side_is_refused(tmp_path, side):
    """h9: only train/fit/selection prompts shape the kept ids (test and held-out never do)."""
    ckpt = _checkpoint(tmp_path / "ckpt")
    prompts = _prompts(tmp_path / "prompts.jsonl", sides=("train", side))
    args = dict(python="py", base_revision=REV, prompts=prompts)
    with pytest.raises(CliError, match="may not shape"):
        qz.plan_weight_only(ckpt, tmp_path / "out", 8, **args)


def test_an_empty_or_missing_prompt_set_is_refused(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    args = dict(python="py", base_revision=REV, prompts=empty)
    with pytest.raises(CliError, match="holds no prompts"):
        qz.plan_weight_only(ckpt, tmp_path / "out", 8, **args)
    with pytest.raises(CliError, match="does not exist"):
        qz.read_prompt_set(tmp_path / "missing.jsonl")


def test_apply_builds_and_verifies_the_pruned_artifact(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    seen = []

    def runner(argv, env):
        seen.append(list(argv))
        _artifact(Path(argv[5]), bits=int(argv[7]), kept_ids=KEPT)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    info = qz.build_pruned_vocabulary(
        ckpt,
        tmp_path / "out",
        8,
        prompts,
        apply=True,
        python="py",
        base_revision=REV,
        runner=runner,
        memory_max="90G",
    )
    assert info["artifact"]["manifest"]["pruned_vocabulary"]["kept_ids_count"] == len(KEPT)
    assert "--prompts" in seen[0]


def test_a_pruned_build_that_left_a_full_vocabulary_artifact_is_refused(tmp_path):
    ckpt = _checkpoint(tmp_path / "ckpt")
    prompts = _prompts(tmp_path / "prompts.jsonl")

    def runner(argv, env):
        _artifact(Path(argv[5]), bits=int(argv[7]))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    args = dict(apply=True, python="py", base_revision=REV, runner=runner, memory_max="9G")
    with pytest.raises(CliError, match="full-vocabulary artifact"):
        qz.build_pruned_vocabulary(ckpt, tmp_path / "out", 8, prompts, **args)


def test_the_build_entry_passes_the_prompt_set(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(qz, "_build_in_env", lambda *a: seen.append(a) or 0)
    argv = ["build", "c", "o", "--bits", "4", "--base-revision", REV, "--prompts", "p.jsonl"]
    assert qz.main(argv) == 0
    assert seen[0][-1] == Path("p.jsonl")
    assert qz.main(["build", "c", "o", "--bits", "4", "--base-revision", REV]) == 0
    assert seen[1][-1] is None


def test_the_pruned_vocabulary_check_matches_rows_to_kept_ids():
    record = qz.check_pruned_vocabulary(KEPT, 248320, {"rows": len(KEPT), "vocab_rows": 248320})
    assert record["full_vocabulary"] is False
    assert record["kept_ids_sha256"] == qz.kept_ids_sha256(KEPT)
    with pytest.raises(CliError, match="rows"):
        qz.check_pruned_vocabulary(KEPT, 248320, {"rows": 4})
    with pytest.raises(CliError, match="not a tokenizer id"):
        qz.check_pruned_vocabulary(KEPT, 1000, {"rows": len(KEPT)})


@pytest.fixture
def oov_server():
    def evaluate(body):
        def call():
            if "unseen" in body.get("state", ""):
                raise qz.OutOfVocabulary([248000, 5, 248000])
            return dict(ANSWER)

        return qz.refuse_out_of_vocabulary(call)

    srv = qz.make_server("127.0.0.1", 0, evaluate=evaluate, health=lambda: {"status": "ok"})
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_request_outside_the_kept_ids_is_refused_out_of_vocabulary(oov_server):
    """h9: a token outside the kept ids is refused with out_of_vocabulary, never answered."""
    status, body = _post(oov_server, {"state": "an unseen word", "questions": {}})
    assert status == 422
    assert body["detail"]["error"] == "out_of_vocabulary"
    assert body["detail"]["token_ids"] == [5, 248000]
    assert "decision" not in json.dumps(body)  # no answer rides along with the refusal
    status, _ = _post(oov_server, {"state": "seen", "questions": {}})
    assert status == 200


def test_the_shared_client_sees_the_out_of_vocabulary_reason(oov_server):
    from jev_factory.backbones.pointer import serve as pserve

    record = {"id": "r", "state": "unseen", "questions": {"c": {"type": "choice", "criteria": {}}}}
    result = pserve.PointerClient(oov_server, model="p").ask(record)
    assert "out_of_vocabulary" in result.error


def test_pruned_embedding_answers_kept_rows_and_refuses_any_other_id(tmp_path):
    """h9: the pruned embedding returns each kept id's own row and raises OutOfVocabulary
    for any other id before reading a row (so it is never answered from a wrong row)."""
    torch = pytest.importorskip("torch")
    kept = [0, 3, 17, 500, 999]
    for bits in (4, 8):
        full = _toy_torso(torch).to(torch.bfloat16)
        pruned = _toy_torso(torch).to(torch.bfloat16)
        pruned.load_state_dict(full.state_dict())
        qz.pack_embedding(full, qz.QuantScheme(embedding_bits=bits))
        record = qz.pack_embedding(pruned, qz.QuantScheme(embedding_bits=bits), kept)
        assert (record["rows"], record["vocab_rows"], record["pruned"]) == (5, 1000, True)
        ids = torch.tensor([[999, 0, 17], [3, 500, 3]])
        assert torch.equal(pruned.embed_tokens(ids), full.embed_tokens(ids))
        for bad in ([[1]], [[0, 998]], [[1000]], [[-1]]):
            outside = torch.tensor(bad)
            with pytest.raises(qz.OutOfVocabulary):
                pruned.embed_tokens(outside)
        state = pruned.state_dict()
        assert state["embed_tokens.qweight"].shape[0] == 5
        assert "embed_tokens.id_map" not in state  # rebuilt from the kept list at load
        torch.save(state, tmp_path / f"p{bits}.pt")
        fresh = _toy_torso(torch).to(torch.bfloat16)
        qz.drop_unused(fresh)
        qz.install_packed_embedding(fresh, record, kept)
        fresh.load_state_dict(torch.load(tmp_path / f"p{bits}.pt", weights_only=True), assign=True)
        assert torch.equal(fresh.embed_tokens(ids), pruned.embed_tokens(ids))
        unkept = torch.tensor([[2]])
        with pytest.raises(qz.OutOfVocabulary):
            fresh.embed_tokens(unkept)
