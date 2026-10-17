"""Drive ``strands-decider serve`` as a localhost subprocess and ask it over /v1/systemone.

The real command line (strands-labs/strands-decider @ 8b6faa5, ``src/strands_decider/cli.py``
``serve_cmd``; console script ``strands-decider = strands_decider.cli:main``) is::

    strands-decider serve CHECKPOINT [--host 127.0.0.1] [--port 8000] [--device D]
                          [--model-name NAME] [--strict-window] ...

``CHECKPOINT`` is positional and there is no ``--revision``: a Hub id is fetched with
``snapshot_download(revision=None)``, i.e. the default branch. So this module serves only a
local, verified bundle directory; the base model's revision comes from that bundle's
provenance.json. The package has no ``__main__``, so the module form is
``python -m strands_decider.cli serve``. Nothing here imports strands_decider or torch.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jev_factory.backbones.pointer.layout import TOKENIZER_FILES, read_layout
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.factory.detach import job_status, start_detached
from jev_factory.measure.preflight import _NoRedirect, require_localhost
from jev_factory.runtime.envs import EnvSpec
from jev_factory.wire import systemone

LOOPBACK = "127.0.0.1"
SERVE_MODULE = "strands_decider.cli"


def serve_argv(
    python: str,
    bundle: str | Path,
    *,
    port: int,
    host: str = LOOPBACK,
    model_name: str | None = None,
    device: str | None = None,
) -> list[str]:
    """The ``strands-decider serve`` command for a verified local bundle on 127.0.0.1."""
    if host != LOOPBACK:
        raise CliError(
            EXIT_USER_ERROR,
            f"refusing to bind {host!r}: the student server binds {LOOPBACK} only",
            f"use host {LOOPBACK}",
        )
    root = Path(bundle)
    read_layout(root)  # refuses a Hub id, a missing file or a hash mismatch
    if missing := [name for name in TOKENIZER_FILES if not (root / name).is_file()]:
        raise CliError(
            EXIT_USER_ERROR,
            f"{root}: cannot serve without {', '.join(missing)} (tokenizer loads from the bundle)",
            "re-export the bundle from a checkpoint that carries its tokenizer",
        )
    argv = [python, "-m", SERVE_MODULE, "serve", str(root), "--host", host, "--port", str(port)]
    if model_name:
        argv += ["--model-name", model_name]
    if device:
        argv += ["--device", device]
    return argv


def launch(
    jobdir: str | Path,
    name: str,
    env: EnvSpec,
    bundle: str | Path,
    *,
    port: int,
    model_name: str | None = None,
    device: str | None = None,
) -> int:
    """Start the server detached in the pinned student *env*; return its pid.

    ``<name>.serve.json`` beside the pid file records the argv, the environment (python plus
    lockfile hash) and the bundle's file hashes, so a measurement can cite what answered it.
    """
    if not isinstance(env, EnvSpec) or env.role != "student" or not env.python:
        raise CliError(
            EXIT_USER_ERROR,
            "the pointer server runs in the pinned student environment (a python path)",
            "describe it with distill.envs.EnvSpec(role='student', python=..., lockfile_hash=...)",
        )
    argv = serve_argv(env.python, bundle, port=port, model_name=model_name or name, device=device)
    info = read_layout(bundle)
    jobdir = Path(jobdir)
    jobdir.mkdir(parents=True, exist_ok=True)
    record = {"argv": argv, "environment": env.record(), "bundle": info["files"]}
    (jobdir / f"{name}.serve.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return start_detached(jobdir, name, argv)


def wait_ready(
    base_url: str,
    *,
    timeout: float = 600.0,
    poll: float = 2.0,
    jobdir: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Poll ``GET /health`` until it answers ok; raise if the job dies or time runs out."""
    require_localhost(base_url)
    url = base_url.rstrip("/") + "/health"
    opener = urllib.request.build_opener(_NoRedirect)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with opener.open(url, timeout=poll) as response:  # nosec B310 - localhost only
                health = json.loads(response.read())
            if isinstance(health, dict) and health.get("status") == "ok":
                return health
        except (OSError, ValueError):
            pass
        if jobdir is not None and name is not None:
            state = job_status(Path(jobdir), name)["state"]
            if state in ("done", "died"):
                raise CliError(
                    EXIT_ENV_ERROR,
                    f"strands-decider server {name!r} exited before it was ready",
                    f"read {Path(jobdir) / (name + '.log')}",
                )
        if time.monotonic() >= deadline:
            raise CliError(EXIT_ENV_ERROR, f"{url} not ready after {timeout:.0f}s")
        time.sleep(poll)


class PointerClient(systemone.SystemOneClient):
    """strands-decider's /v1/systemone client: the shared one, with no overrides.

    The shared parser reads score ``probabilities`` keyed by level index directly.
    """
