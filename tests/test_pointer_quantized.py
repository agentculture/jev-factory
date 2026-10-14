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
    assert plan["format"] == "torch_int8_wo" and plan["bits"] == 8
    assert plan["argv"] == [
        "/envs/student/bin/python",
        "-m",
        qz.MODULE,
        "build",
        str(ckpt),
        str(tmp_path / "out"),
        "--bits",
        "8",
        "--base-revision",
        REV,
        "--device",
        "cuda",
    ]
    path = plan["env"]["PYTHONPATH"].split(os.pathsep)
    assert path[:2] == [str(tmp_path / "sd"), str(tmp_path / "sd" / "src")]
    assert (Path(path[2]) / "jev_factory" / "backbones" / "pointer" / "quantized.py").is_file()
    assert plan["base_model_revision"] == REV
    assert plan["head_dtype"] == "float32" and plan["lora_merged"] is True
    assert not (tmp_path / "out").exists()


def test_int4_plans_the_int4_api(tmp_path):
    _ckpt, plan = _plan(tmp_path, bits=4)
    assert plan["format"] == "torch_int4_wo" and plan["argv"][7] == "4"
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
    _ckpt, out = _plan(tmp_path, apply=True, runner=runner)
    assert out["dry_run"] is False
    argv, env = runner.calls[0]
    assert argv == out["argv"] and env["PYTHONPATH"] == out["env"]["PYTHONPATH"]
    assert out["artifact"]["manifest"]["format"] == "torch_int8_wo"


def test_apply_reports_a_failed_build_with_its_error(tmp_path):
    runner = FakeRunner(rc=1, write=False, stderr="RuntimeError: no int4 kernel for sm_121")
    with pytest.raises(CliError, match="no int4 kernel") as err:
        _plan(tmp_path, bits=4, apply=True, runner=runner)
    assert err.value.code == EXIT_ENV_ERROR


def test_apply_refuses_a_build_that_left_no_valid_artifact(tmp_path):
    with pytest.raises(CliError, match="jev_quant.json"):
        _plan(tmp_path, apply=True, runner=FakeRunner(write=False))


# --------------------------------------------------------------------- the artifact


def test_a_complete_artifact_reads_back(tmp_path):
    info = qz.read_artifact(_artifact(tmp_path / "a", bits=4))
    m = info["manifest"]
    assert (m["format"], m["bits"], m["head_dtype"]) == ("torch_int4_wo", 4, "float32")
    assert m["artifact_format"] == qz.ARTIFACT_FORMAT
    assert set(m["files"]) == {qz.TORSO_FILE, player.HEAD, player.CONFIG}
    assert m["torchao_version"] == "0.18.0" and m["torch_version"] == "2.12.1"
    assert m["base_model_revision"] == REV and m["lora_merged"] is True


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
    assert argv[0] == "env" and argv[1].startswith("PYTHONPATH=")
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
    assert pid == 4242 and started[0][2] == "/envs/student/bin/python"
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
    assert status == 200 and seen == [{"id": "r", "state": "s", "questions": {}}]
    assert {k: v for k, v in body.items() if k != "latency_ms"} == ANSWER
    assert isinstance(body["latency_ms"], float)


def test_a_caller_error_is_a_422_with_detail_like_upstream(server):
    url, _ = server
    assert _post(url, {"state": "too many options"}) == (
        422,
        {"detail": "option count over num_slots"},
    )
    status, body = _post(url, b"{not json")
    assert status == 422 and "detail" in body


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


def test_weight_only_quantization_replaces_linear_weights_and_roundtrips(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    for bits in (8,):  # int4 needs a CUDA kernel; int8 runs on CPU
        model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU())
        model = model.to(torch.bfloat16)
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        want = model(x)
        api = qz.quantize_torso(model, bits)
        assert api in qz.TORCHAO_APIS[bits]
        assert type(model[0].weight.detach()) is not torch.Tensor  # a torchao subclass
        got = model(x)
        assert torch.allclose(want.float(), got.float(), atol=0.1, rtol=0.1)
        path = tmp_path / f"t{bits}.pt"
        torch.save(model.state_dict(), path)
        fresh = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU())
        qz.load_quantized_state(fresh, path, device="cpu")
        assert torch.allclose(fresh(x).float(), got.float())


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
