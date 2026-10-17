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
    spec = _server(hf_revision=rev)
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", spec)
    assert "revision" in exc.value.message


@pytest.mark.parametrize("ver", ["", "latest", ">=1.0", "~=1.0", "1.*", "*"])
def test_unpinned_package_version_refused(ver, tmp_path):
    spec = _server(packages={"transformers": ver})
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", spec)
    assert "transformers" in exc.value.message


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1", "192.168.1.5", ""])
def test_non_loopback_bind_refused(host, tmp_path):
    spec = _server(host=host)
    with pytest.raises(CliError) as exc:
        envs.launch_server(tmp_path, "s", spec)
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
    assert "--model" not in argv
    assert "--revision" not in argv
    assert "strands_decider" not in argv[2:3]
    assert argv == envs.build_argv(_server(model=str(bundle)))


def test_strands_decider_refuses_a_hub_id_instead_of_a_local_bundle(tmp_path):
    spec = _server(model="StrandsAgents/strands-decider-2B-hobson-v21")
    with pytest.raises(CliError):
        envs.build_argv(spec)


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
    assert "127.0.0.1" in argv
    assert "0.0.0.0" not in argv


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


# --------------------------------------------------------------------- exact-pin regex (S5852)

#: The pre-fix pattern, kept here only as the reference language for the equivalence check.
_OLD_EXACT = r"^[0-9]+(\.[0-9]+)*([.+-]?[A-Za-z0-9]+)*$"


@pytest.mark.parametrize(
    "ver", ["1", "1.2.3", "4.50.0", "2.12.1+cu130", "1.0rc1", "0.10.0.dev3", "1.2-post1", "1a"]
)
def test_exact_pin_accepts_exact_versions(ver, tmp_path):
    assert envs._EXACT.match(ver)
    envs.validate_launch(_server(packages={"transformers": ver}))


@pytest.mark.parametrize(
    "ver", ["", "latest", ">=1.0", "~=1.0", "1.*", "*", "1.", "1..2", "1.2+", ".1", "1 .2", "1+-a"]
)
def test_exact_pin_refuses_ranges_and_malformed_versions(ver):
    assert not envs._EXACT.match(ver)


def test_exact_pin_accepts_the_same_language_as_the_old_pattern():
    import itertools
    import re

    old = re.compile(_OLD_EXACT)
    alphabet = "0a.+-*"
    for n in range(0, 7):
        for chars in itertools.product(alphabet, repeat=n):
            text = "".join(chars)
            assert bool(envs._EXACT.match(text)) == bool(old.match(text)), text


def test_exact_pin_is_linear_on_an_adversarial_input():
    import time

    evil = "1" + "a" * 50_000 + "!"
    started = time.perf_counter()
    assert not envs._EXACT.match(evil)
    assert not envs._EXACT.match("1" + ".a" * 25_000 + "..")
    assert time.perf_counter() - started < 0.5


def test_variant_recipes_name_engine_and_quantization():
    assert envs.variant_recipe("bf16") == ("clef-transformers", None)
    kind, quant = envs.variant_recipe("W4A16")
    assert kind in envs.KINDS and quant
    assert envs.variant_recipe("NVFP4")[0] in envs.KINDS
    with pytest.raises(CliError, match="recipe"):
        envs.variant_recipe("INT3")


def test_unknown_engine_kind_refused_and_quantization_in_argv():
    with pytest.raises(CliError):
        envs.check_engine_kind("tensorrt")
    spec = _server(
        role="teacher", kind="vllm-clef", env=_env("teacher"), quantization="compressed-tensors"
    )
    argv = envs.build_argv(spec)
    assert "vllm.entrypoints.openai.api_server" in argv
    assert argv[-2:] == ["--quantization", "compressed-tensors"]
    plain = envs.build_argv(_server(role="teacher", kind="sglang-dev-clef", env=_env("teacher")))
    assert "--quantization" not in plain
