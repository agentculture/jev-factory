"""Pinned environments and the localhost-only server launcher (distill t6)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import envs

SHA = "a" * 40
LOCK = "b" * 64
DIGEST = "sha256:" + "c" * 64


def _env(role, **kw):
    base = {"role": role, "python": "/opt/venv/bin/python", "lockfile_hash": LOCK}
    base.update(kw)
    return envs.EnvSpec(**base)


def _server(**kw):
    base = dict(
        role="student",
        kind="strands-decider",
        env=_env("student"),
        model="org/model",
        hf_revision=SHA,
        packages={"strands-decider": "1.2.3", "transformers": "4.50.0"},
        host="127.0.0.1",
        port=8123,
    )
    base.update(kw)
    return envs.ServerSpec(**base)


def test_three_environments_described_and_recorded():
    cfg = {
        "teacher": {"image_digest": DIGEST, "lockfile_hash": LOCK},
        "student": {"python": "/x/py", "lockfile_hash": LOCK},
        "trainer": {"python": "/y/py", "lockfile_hash": LOCK},
    }
    specs = envs.describe_environments(cfg)
    assert [s.role for s in specs] == ["teacher", "student", "trainer"]
    prov = envs.record_provenance({"run": "r1"}, specs)
    assert prov["run"] == "r1"
    rec = prov["environments"]
    assert rec["teacher"]["image_digest"] == DIGEST
    assert rec["student"]["python"] == "/x/py"
    assert rec["trainer"]["lockfile_hash"] == LOCK


def test_environment_needs_python_xor_digest_and_lock_hash():
    with pytest.raises(CliError):
        _env("teacher", python=None)
    with pytest.raises(CliError):
        _env("teacher", image_digest=DIGEST)  # both given
    with pytest.raises(CliError):
        _env("teacher", python=None, image_digest="sha256:short")
    with pytest.raises(CliError):
        _env("teacher", lockfile_hash="nothex")
    with pytest.raises(CliError):
        envs.describe_environments({"teacher": {}})


@pytest.mark.parametrize("rev", ["main", "", "v1.0", "abc123", "latest"])
def test_unpinned_hf_revision_refused(rev, tmp_path):
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", _server(hf_revision=rev))
    assert "revision" in exc.value.message


@pytest.mark.parametrize("ver", ["", "latest", ">=1.0", "~=1.0", "1.*", "*"])
def test_unpinned_package_version_refused(ver, tmp_path):
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", _server(packages={"transformers": ver}))
    assert "transformers" in exc.value.message


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1", "192.168.1.5", ""])
def test_non_loopback_bind_refused(host, tmp_path):
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", _server(host=host))
    assert "127.0.0.1" in exc.value.message


def test_trainer_is_not_a_server():
    with pytest.raises(CliError):
        _server(role="trainer")


def _bundle(root: Path) -> Path:
    """A minimal verified hobson-hf-export/1 bundle directory."""

    def put(name: str, data: bytes) -> str:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    adapter = put("lora/adapter_model.safetensors", b"adapter")
    put("lora/adapter_config.json", b"{}")
    head = put("head.safetensors", b"head")
    cfg = {"base_model": "Qwen/Qwen3.5-2B-Base", "head_type": "pointer", "max_length": 4096}
    put("strands_decider_config.json", json.dumps(cfg).encode())
    put("tokenizer.json", b"{}")
    put("tokenizer_config.json", b"{}")
    prov = {
        "format": "hobson-hf-export/1",
        "base_model": "Qwen/Qwen3.5-2B-Base",
        "base_model_revision": SHA,
        "adapter_sha256": adapter,
        "head_sha256": head,
    }
    put("provenance.json", json.dumps(prov).encode())
    return root


def test_strands_decider_argv_is_the_real_serve_cli_on_a_local_bundle(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    argv = envs.build_argv(_server(model=str(bundle)))
    assert argv[:5] == ["/opt/venv/bin/python", "-m", "strands_decider.cli", "serve", str(bundle)]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8123"
    assert "--model" not in argv and "--revision" not in argv and "strands_decider" not in argv[2:3]
    assert argv == envs.build_argv(_server(model=str(bundle)))


def test_strands_decider_refuses_a_hub_id_instead_of_a_local_bundle(tmp_path):
    with pytest.raises(CliError):
        envs.build_argv(_server(model="StrandsAgents/strands-decider-2B-hobson-v21"))


def test_valid_launch_uses_detach_with_loopback_argv(tmp_path, monkeypatch):
    seen = {}

    def fake(jobdir, name, argv):
        seen.update(jobdir=jobdir, name=name, argv=argv)
        return 4242

    monkeypatch.setattr(envs, "start_detached", fake)
    bundle = _bundle(tmp_path / "bundle")
    assert envs.launch_server(tmp_path, "student", _server(model=str(bundle))) == 4242
    argv = seen["argv"]
    assert argv[0] == "/opt/venv/bin/python"
    assert "127.0.0.1" in argv and "0.0.0.0" not in argv


def test_image_launch_binds_loopback_and_pins_digest(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(envs, "start_detached", lambda j, n, a: seen.setdefault("argv", a) and 1)
    spec = _server(
        role="teacher",
        kind="sglang-dev-clef",
        env=_env("teacher", python=None, image_digest=DIGEST),
    )
    envs.launch_server(tmp_path, "teacher", spec)
    argv = seen["argv"]
    assert f"127.0.0.1:{spec.port}:{spec.port}" in argv
    assert any(a.endswith("@" + DIGEST) for a in argv)


def test_import_with_heavy_modules_blocked_and_no_dependencies():
    code = (
        "import sys\n"
        "for m in ('torch','transformers','peft','strands_decider','sglang'):\n"
        "    sys.modules[m] = None\n"
        "import jev_factory, jev_factory.distill.envs\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    root = Path(__file__).resolve().parent.parent
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    assert meta["project"]["dependencies"] == []
