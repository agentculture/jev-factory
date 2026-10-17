"""Pinned environments and a localhost-only server launcher for the distillery.

Three environments (teacher, student runtime, trainer) are each named by a python
path or an image digest plus a lockfile hash, and are recorded into the run's
provenance. A teacher or student server is started as a detached subprocess only
when every HF revision and package version is exactly pinned and the bind is
``127.0.0.1``. Nothing heavy is imported here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jev_factory.backbones import registry
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory.detach import start_detached
from jev_factory.measure.preflight import require_localhost

# EnvSpec and ROLES live in the shared layer (deviation d1) and keep this import path.
from jev_factory.runtime.envs import ROLES, EnvSpec

SERVER_ROLES = ("teacher", "student")
KINDS = ("strands-decider", "clef-transformers", "sglang-dev-clef", "vllm-clef")
#: Clef serving recipes: variant (upper-case) -> (engine kind, quantization or None). Data, so a
#: new variant is a table row; the quantization is passed to the engine as ``--quantization``.
VARIANT_RECIPES: dict[str, tuple[str, str | None]] = {
    "BF16": ("clef-transformers", None),
    "NVFP4": ("sglang-dev-clef", "modelopt_fp4"),
    "FP4": ("sglang-dev-clef", "modelopt_fp4"),
    "W4A16": ("vllm-clef", "compressed-tensors"),
}
LOOPBACK = "127.0.0.1"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: Server kinds a backbone serves, and the registered backbone that builds their command line.
KIND_BACKBONES = {"strands-decider": "pointer"}
#: An exact version pin: a digit first, then alphanumeric runs joined by single ``.``, ``+``
#: or ``-`` separators, ending alphanumeric (``1.2.3``, ``2.12.1+cu130``, ``1.0rc1``). The same
#: language as the earlier ``^[0-9]+(\.[0-9]+)*([.+-]?[A-Za-z0-9]+)*$``, but every separator in
#: the repeated group is mandatory, so each character has one parse and matching is linear
#: (that pattern backtracked exponentially on inputs like ``1aaaa...!``, Sonar S5852).
_EXACT = re.compile(r"^\d[A-Za-z0-9]*(?:[.+-][A-Za-z0-9]+)*$", re.ASCII)


def check_engine_kind(kind: str) -> str:
    """Return *kind* if it is a known engine kind, else refuse."""
    if kind not in KINDS:
        raise _refuse(f"unknown server kind {kind!r}", f"use one of {KINDS}")
    return kind


def variant_recipe(variant: str) -> tuple[str, str | None]:
    """The (engine kind, quantization) a Clef *variant* needs; unknown variants are refused."""
    recipe = VARIANT_RECIPES.get(str(variant).strip().upper())
    if recipe is None:
        raise _refuse(
            f"no serving recipe for Clef variant {variant!r}",
            f"known variants: {sorted(VARIANT_RECIPES)}; add a VARIANT_RECIPES row",
        )
    return recipe


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def describe_environments(config: Mapping[str, Mapping[str, Any]]) -> tuple[EnvSpec, ...]:
    """Build the teacher, student and trainer specs from a run-config mapping."""
    specs = []
    for role in ROLES:
        entry = config.get(role)
        if not isinstance(entry, Mapping):
            raise _refuse(f"run config has no {role} environment", f"add a {role} entry")
        specs.append(
            EnvSpec(
                role=role,
                lockfile_hash=str(entry.get("lockfile_hash", "")),
                python=entry.get("python"),
                image_digest=entry.get("image_digest"),
            )
        )
    return tuple(specs)


def record_provenance(provenance: Mapping[str, Any], specs: tuple[EnvSpec, ...]) -> dict[str, Any]:
    """Return a copy of *provenance* with the environments recorded under ``environments``."""
    out = dict(provenance)
    out["environments"] = {s.role: s.record() for s in specs}
    return out


@dataclass(frozen=True)
class ServerSpec:
    """A server to launch: which environment, which model at which revision, which pins."""

    role: str
    kind: str
    env: EnvSpec
    model: str
    hf_revision: str
    packages: Mapping[str, str] = field(default_factory=dict)
    host: str = LOOPBACK
    port: int = 8000
    quantization: str | None = None

    def __post_init__(self) -> None:
        if self.role not in SERVER_ROLES:
            raise _refuse(f"only {SERVER_ROLES} are served, not {self.role!r}")
        check_engine_kind(self.kind)


def validate_launch(spec: ServerSpec) -> None:
    """Raise ``CliError`` unless every pin is exact and the bind is 127.0.0.1."""
    if spec.host != LOOPBACK:
        raise _refuse(
            f"refusing to bind {spec.host!r}: servers bind {LOOPBACK} only",
            f"set the host to {LOOPBACK}",
        )
    try:
        require_localhost(f"http://{spec.host}:{spec.port}")
    except ValueError as exc:
        raise _refuse(f"refusing to launch: {exc}") from exc
    if not _COMMIT.match(spec.hf_revision or ""):
        raise _refuse(
            f"HF revision {spec.hf_revision!r} is not a pinned 40-hex commit",
            "pin the model to a commit hash, not a branch or tag",
        )
    for name, version in spec.packages.items():
        if not _EXACT.match(version or ""):
            raise _refuse(
                f"package {name} version {version!r} is not an exact pin",
                "use an exact version such as 1.2.3",
            )


def build_argv(spec: ServerSpec) -> list[str]:
    """The command line for a validated server spec."""
    validate_launch(spec)
    port = str(spec.port)
    if spec.env.image_digest:
        image = f"{spec.model}@{spec.env.image_digest}"
        return [
            "docker",
            "run",
            "--rm",
            "-p",
            f"{spec.host}:{port}:{port}",
            image,
            "--host",
            "0.0.0.0",  # nosec B104 - inside the container; published to 127.0.0.1 only
            "--port",
            port,
            "--revision",
            spec.hf_revision,
        ]
    python = str(spec.env.python)
    if spec.kind in KIND_BACKBONES:
        # The backbone's own serve hook. ``model`` must be a verified local bundle directory (a
        # Hub id would be fetched unpinned); the revision pin above refers to the bundle's
        # source commit and is not a CLI flag.
        backbone = registry.get(KIND_BACKBONES[spec.kind])
        return backbone.serve_argv(python, spec.model, port=spec.port, host=spec.host)
    module = {
        "clef-transformers": ["-m", "transformers_sidecar", "load_release_model"],
        "sglang-dev-clef": ["-m", "sglang.launch_server"],
        "vllm-clef": ["-m", "vllm.entrypoints.openai.api_server"],
    }[spec.kind]
    quant = ["--quantization", spec.quantization] if spec.quantization else []
    return [
        python,
        *module,
        "--model",
        spec.model,
        "--revision",
        spec.hf_revision,
        "--host",
        spec.host,
        "--port",
        port,
        *quant,
    ]


def launch_server(jobdir: Path, name: str, spec: ServerSpec) -> int:
    """Validate then start the server detached; return its pid."""
    return start_detached(Path(jobdir), name, build_argv(spec))
