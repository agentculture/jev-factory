"""Run-config loader: one precedence rule for every key.

Replaces nvsh's ``pipeline.sh`` env-file sourcing and its ``MEASURE_CTX``
special case (an exported value outranking the file's) with a single rule
that applies to **every** key, the measure context included::

    CLI flag  >  run file (TOML)  >  environment (JEV_<KEY>)  >  default

Every knob is a declared :class:`Key`: nothing is an undocumented export.
Tool paths default to ``None`` and are demanded by the stage that needs them
through :meth:`RunConfig.require`. Secrets never live in config: the secret
keys hold the *name* of an environment variable, read by
:mod:`jev_factory.factory.secrets`.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/pipeline.sh",
    "commit": "9debdc6",
    "adaptations": [
        "lines 174-191 env sourcing and the env.example knob list replaced by a typed TOML loader",
        "the MEASURE_CTX environment-outranks-file exception generalised to one precedence rule",
        "tool paths, licence, hub prefix and issue refs made declared keys",
        "secrets are referenced by env-var name only (see secrets.py)",
    ],
    "licence": "Apache-2.0",
}

ENV_PREFIX = "JEV_"
SOURCES = ("cli", "file", "env", "default")
"""Precedence order, highest first."""

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

REQUIRED = object()
"""Sentinel default: the key has no default and must be supplied."""


@dataclass(frozen=True)
class Key:
    """One documented run-config key."""

    name: str
    kind: str  # str | int | float | bool | path | envname
    default: Any
    doc: str

    @property
    def env_var(self) -> str:
        return ENV_PREFIX + self.name.upper()


def _k(name: str, kind: str, default: Any, doc: str) -> Key:
    return Key(name, kind, default, doc)


KEYS: tuple[Key, ...] = (
    # identity of the run
    _k("work", "path", REQUIRED, "work root for every intermediate file (outside any git tree)"),
    _k("base", "str", REQUIRED, "base model id (for example an org/name hub id)"),
    _k("base_rev", "str", REQUIRED, "base model revision (commit) the stock model is measured as"),
    _k("hub_prefix", "str", REQUIRED, "hub repo prefix the private bundles are uploaded under"),
    _k("licence", "str", REQUIRED, "licence of the shipped bundle (SPDX id)"),
    _k("issue_refs", "str", "", "comma-separated issue refs the model and data cards cite"),
    _k("seed", "int", 46, "seed for the grouped split and assembly"),
    # tool paths: optional until a stage needs them (RunConfig.require)
    _k("train_py", "path", None, "training environment's python (torch + unsloth + peft)"),
    _k("llama_cpp_convert", "path", None, "llama.cpp convert_hf_to_gguf.py"),
    _k("llama_cpp_quantize", "path", None, "llama.cpp llama-quantize binary"),
    _k("llama_cpp_imatrix", "path", None, "llama.cpp llama-imatrix binary"),
    _k("llama_server", "path", None, "llama.cpp llama-server binary (serves quantized builds)"),
    _k("awq_python", "path", None, "python of the AWQ quantization environment"),
    _k("hf_cache", "path", None, "Hugging Face cache directory the serving stage mounts"),
    # measurement
    _k("measure_image", "str", None, "serving image pinned by @sha256: digest (tags are refused)"),
    _k("measure_port", "int", 18060, "localhost port the measured model is served on"),
    _k("measure_ctx", "int", 2048, "context length the measured model is served at"),
    _k("measure_gpu_fraction", "float", 0.08, "GPU memory fraction for the measure server"),
    _k("measure_max_logprobs", "int", 20000, "serving --max-logprobs (readout top-k)"),
    _k("tool_call_parser", "str", None, "serving tool-call parser for the base, if any"),
    _k(
        "enable_thinking",
        "bool",
        False,
        "chat-template enable_thinking sent with measured requests",
    ),
    _k("ground_snapshot", "path", None, "fixed grounding snapshot every measure stage uses"),
    # resource caps
    _k("train_memory_max", "str", None, "hard OS memory cap for training/quantize (e.g. 24G)"),
    _k("train_memory_floor", "str", "8G", "available-memory floor the watchdog enforces"),
    _k("train_watchdog_seconds", "int", 5, "watchdog poll interval in seconds"),
    _k("train_gpu_memory_gb", "float", None, "per-process GPU memory budget in GB; unset = no cap"),
    # teachers and augmentation
    _k("aug_url", "str", None, "OpenAI-compatible chat-completions URL of the teacher gateway"),
    _k("workers", "int", 2, "concurrent teacher requests"),
    _k("teacher_generator_model", "str", "worker", "gateway model alias of the generator role"),
    _k("teacher_reviewer_a_model", "str", "senses", "gateway model alias of reviewer A (senses)"),
    _k("teacher_reviewer_b_model", "str", "cortex", "gateway model alias of reviewer B (cortex)"),
    _k(
        "teacher_models",
        "path",
        None,
        "JSON file {alias: {name, licence}} extending the teacher list",
    ),
    _k("teacher_generator_max_tokens", "int", 12000, "generator reply budget in tokens"),
    _k("teacher_reviewer_max_tokens", "int", 8192, "reviewer reply budget in tokens"),
    _k("teacher_timeout", "float", 300.0, "per-request teacher timeout in seconds"),
    _k("teacher_reasoning_effort", "str", None, "reasoning_effort sent to teachers, if any"),
    _k("teacher_cache", "path", None, "teacher response cache directory (default: under work)"),
    # secrets, by the NAME of the environment variable that holds them
    _k("aug_key_env", "envname", None, "name of the env var holding the teacher gateway key"),
    _k("hf_token_env", "envname", "HF_TOKEN", "name of the env var holding the hub token"),
)

_BY_NAME = {k.name: k for k in KEYS}


def _err(message: str, remediation: str = "", code: int = EXIT_USER_ERROR) -> CliError:
    return CliError(code=code, message=message, remediation=remediation)


def _coerce(key: Key, raw: Any, origin: str) -> Any:
    """Convert ``raw`` to the key's type. Error text never echoes the value."""
    kind = key.kind
    try:
        if kind in ("str", "path", "envname"):
            if not isinstance(raw, str):
                raise ValueError("expected a string")
            value: Any = raw
            if kind == "path":
                value = str(Path(raw).expanduser())
            if kind == "envname" and not _ENV_NAME.match(raw):
                # A pasted token would land here: say nothing about the value.
                raise ValueError("expected an environment-variable NAME, not a value")
            return value
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().lower() in _TRUE | _FALSE:
                return raw.strip().lower() in _TRUE
            raise ValueError("expected a boolean")
        if kind == "int":
            if isinstance(raw, bool):
                raise ValueError("expected an integer")
            return int(raw)
        if kind == "float":
            if isinstance(raw, bool):
                raise ValueError("expected a number")
            return float(raw)
    except (TypeError, ValueError) as exc:
        raise _err(
            f"run config key {key.name!r} from {origin}: {exc}",
            f"fix {key.name} ({kind}); see docs/run-config.example.toml",
        ) from None
    raise _err(f"run config key {key.name!r} has unknown kind {kind!r}")  # pragma: no cover


@dataclass(frozen=True)
class RunConfig:
    """Resolved config: ``values`` plus the ``sources`` each value came from."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]
    path: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str) -> Any:
        return self.values.get(name)

    def require(self, name: str) -> Any:
        """Return a value a stage cannot run without, or raise a CliError."""
        value = self.values.get(name)
        if value is None or value == "":
            key = _BY_NAME[name]
            raise _err(
                f"run config key {name!r} is required by this stage and is not set",
                f"set {name} in the run file, pass it as a flag, or export {key.env_var}",
                EXIT_ENV_ERROR,
            )
        return value

    def to_manifest(self) -> dict[str, Any]:
        """JSON-safe record of the resolved config for a run manifest.

        Holds secret *names* (config values), never secret values: this
        module never reads them.
        """
        return {
            "path": self.path,
            "values": {k: self.values[k] for k in sorted(self.values)},
            "sources": {k: self.sources[k] for k in sorted(self.sources)},
        }


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    cli: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RunConfig:
    """Resolve a run config by ``CLI flag > run file > environment > default``.

    ``cli`` holds only flags the operator actually passed (``None`` values
    are treated as not passed). ``environ`` defaults to ``os.environ``.
    Unknown keys (file or CLI) fail loudly; missing required keys are listed
    together.
    """
    env = os.environ if environ is None else environ
    cli_values = {k: v for k, v in (cli or {}).items() if v is not None}
    file_values: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        try:
            file_values = tomllib.loads(p.read_text(encoding="utf-8"))
        except OSError as exc:
            raise _err(
                f"cannot read run config file: {exc.strerror or 'unreadable'}",
                "pass --config <run.toml> (copy docs/run-config.example.toml)",
                EXIT_ENV_ERROR,
            ) from None
        except tomllib.TOMLDecodeError as exc:
            raise _err(f"run config is not valid TOML: {exc}", "fix the TOML syntax") from None
    for origin, mapping in (("the CLI", cli_values), ("the run file", file_values)):
        unknown = sorted(set(mapping) - set(_BY_NAME))
        if unknown:
            raise _err(
                f"unknown run config key(s) in {origin}: {', '.join(unknown)}",
                "valid keys are listed in docs/run-config.example.toml",
            )

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for key in KEYS:
        if key.name in cli_values:
            src, raw = "cli", cli_values[key.name]
        elif key.name in file_values:
            src, raw = "file", file_values[key.name]
        elif env.get(key.env_var, "") != "":
            src, raw = "env", env[key.env_var]
        elif key.default is REQUIRED:
            missing.append(key.name)
            continue
        else:
            values[key.name] = key.default
            sources[key.name] = "default"
            continue
        values[key.name] = _coerce(
            key, raw, {"cli": "the CLI", "file": "the run file"}.get(src, "the environment")
        )
        sources[key.name] = src
    if missing:
        raise _err(
            f"run config is missing required key(s): {', '.join(missing)}",
            "set them in the run file, pass flags, or export "
            + ", ".join(_BY_NAME[m].env_var for m in missing),
        )
    return RunConfig(values=values, sources=sources, path=str(path) if path is not None else None)
