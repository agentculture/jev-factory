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

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory.detach import start_detached
from jev_factory.measure.preflight import require_localhost

ROLES = ("teacher", "student", "trainer")
SERVER_ROLES = ("teacher", "student")
KINDS = ("strands-decider", "clef-transformers", "sglang-dev-clef")
LOOPBACK = "127.0.0.1"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCK = re.compile(r"^[0-9a-f]{64}$")
_EXACT = re.compile(r"^[0-9]+(\.[0-9]+)*([.+-]?[A-Za-z0-9]+)*$")


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


@dataclass(frozen=True)
class EnvSpec:
    """One pinned environment: a python path xor an image digest, plus a lockfile hash."""

    role: str
    lockfile_hash: str
    python: str | None = None
    image_digest: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise _refuse(f"unknown environment role {self.role!r}", f"use one of {ROLES}")
        if bool(self.python) == bool(self.image_digest):
            raise _refuse(
                f"{self.role} environment needs exactly one of python or image_digest",
                "give a python path or an image digest, not both and not neither",
            )
        if self.image_digest and not _DIGEST.match(self.image_digest):
            raise _refuse(f"{self.role} image_digest must be sha256:<64 hex>")
        if not _LOCK.match(self.lockfile_hash or ""):
            raise _refuse(f"{self.role} lockfile_hash must be 64 hex characters (sha256)")

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "lockfile_hash": self.lockfile_hash}
        if self.python:
            out["python"] = self.python
        if self.image_digest:
            out["image_digest"] = self.image_digest
        return out


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

    def __post_init__(self) -> None:
        if self.role not in SERVER_ROLES:
            raise _refuse(f"only {SERVER_ROLES} are served, not {self.role!r}")
        if self.kind not in KINDS:
            raise _refuse(f"unknown server kind {self.kind!r}", f"use one of {KINDS}")


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
    module = {
        "strands-decider": ["-m", "strands_decider", "serve"],
        "clef-transformers": ["-m", "transformers_sidecar", "load_release_model"],
        "sglang-dev-clef": ["-m", "sglang.launch_server"],
    }[spec.kind]
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
    ]


def launch_server(jobdir: Path, name: str, spec: ServerSpec) -> int:
    """Validate then start the server detached; return its pid."""
    return start_detached(Path(jobdir), name, build_argv(spec))
