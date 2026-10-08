"""Run-config loader: precedence (o24) and secrets-by-name (o41)."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.factory import config as cfg
from jev_factory.factory.config import KEYS, load_config
from jev_factory.factory.secrets import Secret, read_secret, scrub

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "docs" / "run-config.example.toml"
REQUIRED = {
    "work": "w",
    "base": "org/base",
    "base_rev": "abc",
    "hub_prefix": "org/fam",
    "licence": "Apache-2.0",
}


def _write(tmp_path: Path, **extra: object) -> Path:
    body = {**REQUIRED, **extra}
    lines = [f"{k} = {json.dumps(v)}" for k, v in body.items()]
    p = tmp_path / "run.toml"
    p.write_text("\n".join(lines) + "\n")
    return p


# ---- precedence (o24) ----


@pytest.mark.behavioral("o24")
def test_precedence_order_cli_file_env_default(tmp_path):
    f = _write(tmp_path, seed=2)
    env = {"JEV_SEED": "3"}
    assert load_config(f, cli={"seed": 1}, environ=env)["seed"] == 1
    c = load_config(f, environ=env)
    assert c["seed"] == 2
    assert c.sources["seed"] == "file"
    c = load_config(_write(tmp_path), environ=env)
    assert c["seed"] == 3
    assert c.sources["seed"] == "env"
    c = load_config(_write(tmp_path), environ={})
    assert c["seed"] == 46
    assert c.sources["seed"] == "default"


@pytest.mark.behavioral("o24")
def test_measure_context_follows_the_same_rule(tmp_path):
    # nvsh's exception (environment outranked the file for MEASURE_CTX) is gone.
    f = _write(tmp_path, measure_ctx=4096)
    c = load_config(f, environ={"JEV_MEASURE_CTX": "2048"})
    assert c["measure_ctx"] == 4096
    assert c.sources["measure_ctx"] == "file"
    c = load_config(f, cli={"measure_ctx": 8192}, environ={"JEV_MEASURE_CTX": "2048"})
    assert c["measure_ctx"] == 8192
    c = load_config(_write(tmp_path), environ={"JEV_MEASURE_CTX": "4096"})
    assert c["measure_ctx"] == 4096
    assert c.sources["measure_ctx"] == "env"
    assert load_config(_write(tmp_path), environ={})["measure_ctx"] == 2048


@pytest.mark.behavioral("o24")
def test_precedence_applies_to_every_key(tmp_path):
    """Each declared key is overridable at every level, in order."""
    samples = {"str": "x", "path": "/p", "int": 7, "float": 0.5, "bool": True, "envname": "VAR_X"}
    for key in KEYS:
        value = samples[key.kind]
        other = {
            "str": "y",
            "path": "/q",
            "int": 8,
            "float": 0.25,
            "bool": False,
            "envname": "VAR_Y",
        }[key.kind]
        base = {k: v for k, v in REQUIRED.items() if k != key.name}
        f = tmp_path / f"{key.name}.toml"
        f.write_text("\n".join(f"{k} = {json.dumps(v)}" for k, v in base.items()) + "\n")
        env = {key.env_var: str(other)}
        if key.name in REQUIRED:
            f.write_text(f.read_text() + f"{key.name} = {json.dumps(REQUIRED[key.name])}\n")
        got = load_config(f, cli={key.name: value}, environ=env)
        assert got.sources[key.name] == "cli", key.name
        assert got[key.name] == (value if key.kind != "path" else value)


def test_env_used_when_file_silent_and_empty_env_ignored(tmp_path):
    f = _write(tmp_path)
    assert load_config(f, environ={"JEV_WORKERS": ""})["workers"] == 2
    assert load_config(f, environ={"JEV_WORKERS": "5"})["workers"] == 5


def test_cli_none_means_not_passed(tmp_path):
    f = _write(tmp_path, seed=2)
    assert load_config(f, cli={"seed": None})["seed"] == 2


def test_required_keys_listed_together():
    with pytest.raises(CliError) as e:
        load_config(None, environ={})
    for k in REQUIRED:
        assert k in e.value.message


def test_required_keys_can_come_from_env_alone():
    env = {f"JEV_{k.upper()}": v for k, v in REQUIRED.items()}
    c = load_config(None, environ=env)
    assert c["base"] == "org/base"
    assert c.sources["base"] == "env"


def test_unknown_keys_fail_loudly(tmp_path):
    bogus = _write(tmp_path, bogus=1)
    with pytest.raises(CliError, match="unknown run config key"):
        load_config(bogus, environ={})
    plain = _write(tmp_path)
    with pytest.raises(CliError, match="unknown run config key"):
        load_config(plain, cli={"bogus": 1}, environ={})


def test_type_errors_and_bool_coercion(tmp_path):
    path = _write(tmp_path)
    with pytest.raises(CliError, match="measure_port"):
        load_config(path, environ={"JEV_MEASURE_PORT": "abc"})
    assert load_config(_write(tmp_path), environ={"JEV_ENABLE_THINKING": "true"})["enable_thinking"]
    path = _write(tmp_path)
    with pytest.raises(CliError):
        load_config(path, environ={"JEV_ENABLE_THINKING": "maybe"})


def test_missing_and_bad_file(tmp_path):
    with pytest.raises(CliError) as e:
        load_config(tmp_path / "nope.toml", environ={})
    assert e.value.code == 2
    bad = tmp_path / "bad.toml"
    bad.write_text("x = = 1")
    with pytest.raises(CliError, match="not valid TOML"):
        load_config(bad, environ={})


def test_tool_paths_and_metadata_are_declared_keys():
    names = {k.name for k in KEYS}
    assert {
        "llama_cpp_convert",
        "llama_cpp_quantize",
        "llama_cpp_imatrix",
        "llama_server",
        "awq_python",
        "base",
        "base_rev",
        "hub_prefix",
        "licence",
        "issue_refs",
    } <= names
    required = {k.name for k in KEYS if k.default is cfg.REQUIRED}
    assert {"base", "base_rev", "hub_prefix", "licence"} <= required
    assert all(k.doc for k in KEYS)


def test_unset_tool_path_is_demanded_by_require(tmp_path):
    c = load_config(_write(tmp_path), environ={})
    assert c["llama_server"] is None
    with pytest.raises(CliError, match="llama_server") as e:
        c.require("llama_server")
    assert "JEV_LLAMA_SERVER" in e.value.remediation
    assert (
        load_config(_write(tmp_path, llama_server="/x/ls"), environ={}).require("llama_server")
        == "/x/ls"
    )


def test_every_key_is_documented_in_the_example():
    text = EXAMPLE.read_text()
    for key in KEYS:
        assert re.search(rf"^#?\s*{key.name}\s*=", text, re.M), key.name


def test_example_parses_and_loads():
    data = tomllib.loads(EXAMPLE.read_text())
    assert set(data) <= {k.name for k in KEYS}
    assert load_config(EXAMPLE, environ={})["base"] == "org/base-model"


# ---- secrets by name (o41) ----

CANARY = "CANARY-9f3a1c7e-not-a-real-token"


@pytest.fixture
def secret_config(tmp_path):
    f = _write(tmp_path, hf_token_env="MY_HUB_TOKEN", aug_key_env="MY_AUG_KEY")
    return load_config(f, environ={})


@pytest.mark.behavioral("o41")
def test_canary_never_appears_in_any_output(secret_config, capsys, caplog, tmp_path):
    env = {"MY_HUB_TOKEN": CANARY, "MY_AUG_KEY": CANARY + "-2"}
    caplog.set_level(logging.DEBUG)
    held = read_secret(secret_config, "hf_token_env", environ=env)
    assert held.reveal() == CANARY
    logging.getLogger("jev").info("using %s %r %s", held, held, f"{held}")
    print(held, repr(held), [held], {"k": held})
    manifest = json.dumps(secret_config.to_manifest())
    (tmp_path / "manifest.json").write_text(manifest)
    # the config loaded with the canary in the environment resolves names only
    loaded = load_config(tmp_path / "run.toml", environ=env)
    blob = json.dumps(loaded.to_manifest()) + repr(loaded)
    # a missing held errors without a value; a value pasted as a name is refused quietly
    with pytest.raises(CliError) as e:
        read_secret(secret_config, "aug_key_env", environ={})
    err = json.dumps(e.value.to_dict())
    bad = _write(tmp_path, hf_token_env=CANARY)
    with pytest.raises(CliError) as e2:
        load_config(bad, environ={})
    err += json.dumps(e2.value.to_dict())
    out = capsys.readouterr()
    everything = "\n".join([out.out, out.err, caplog.text, manifest, blob, err])
    assert CANARY not in everything
    assert "MY_HUB_TOKEN" in manifest  # the name is recorded, the value is not
    assert CANARY not in (tmp_path / "manifest.json").read_text()


@pytest.mark.behavioral("o41")
def test_secret_is_only_read_via_the_configured_name(secret_config):
    env = {"HF_TOKEN": "wrong-default-var", "MY_HUB_TOKEN": CANARY}
    assert read_secret(secret_config, "hf_token_env", environ=env).reveal() == CANARY
    with pytest.raises(CliError, match="MY_AUG_KEY"):
        read_secret(secret_config, "aug_key_env", environ=env)


def test_secret_wrapper_redacts_and_does_not_pickle():
    import pickle

    s = Secret(CANARY)
    assert CANARY not in repr(s)
    assert CANARY not in str(s)
    assert CANARY not in f"{s:>40}"
    with pytest.raises(TypeError):
        pickle.dumps(s)
    assert scrub(f"key={CANARY}", [s]) == "key=[REDACTED]"


@pytest.mark.behavioral("o41")
def test_run_config_example_has_no_host_and_no_secret():
    text = EXAMPLE.read_text()
    assert not re.search(r"https?://", text)
    assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text)
    assert "/home/" not in text
    assert "/Users/" not in text
    for pattern in (r"sk-[A-Za-z0-9]{20,}", r"gh[pousr]_[A-Za-z0-9]{36,}", r"hf_[A-Za-z0-9]{20,}"):
        assert not re.search(pattern, text)
    data = tomllib.loads(text)
    for key in KEYS:
        if key.kind == "envname" and key.name in data:
            assert re.fullmatch(r"[A-Z_][A-Z0-9_]*", data[key.name])
