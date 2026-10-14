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

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from jev_factory.backbones.pointer.layout import TOKENIZER_FILES, read_layout
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill import systemone
from jev_factory.distill.envs import EnvSpec
from jev_factory.factory.detach import job_status, start_detached
from jev_factory.measure.preflight import _NoRedirect, require_localhost

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


def rekey_scores(record: Mapping[str, Any], payload: Any) -> Any:
    """Map a score answer keyed by level index ("0".."n-1") onto the record's criteria.

    ``strands_decider.infer`` keys score ``probabilities`` (and ``legend``) by level index;
    :mod:`jev_factory.distill.systemone` checks keys against the offered criteria. The legend
    must name the same rubric in the same order, else the answer is left as is (and fails).
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get("answers"), Mapping):
        return payload
    answers = dict(payload["answers"])
    for qid, spec in record["questions"].items():
        answer = answers.get(qid)
        if spec.get("type") != "score" or not isinstance(answer, Mapping):
            continue
        criteria = [str(c) for c in spec.get("criteria") or []]
        probs = answer.get("probabilities")
        index = [str(i) for i in range(len(criteria))]
        if not isinstance(probs, Mapping) or sorted(map(str, probs)) != sorted(index):
            continue
        legend = answer.get("legend")
        if isinstance(legend, Mapping) and [str(legend.get(i)) for i in index] != criteria:
            continue
        rekeyed = {criteria[int(k)]: v for k, v in probs.items()}
        answers[qid] = {**answer, "probabilities": rekeyed}
    return {**payload, "answers": answers}


class PointerClient(systemone.SystemOneClient):
    """A :class:`~jev_factory.distill.systemone.SystemOneClient` for strands-decider's wire.

    Same body, same localhost-only no-redirect transport and the same probability-only parser;
    the one difference is :func:`rekey_scores` applied before parsing.
    """

    def ask(self, record: Mapping[str, Any]) -> systemone.CallResult:
        body = systemone.build_body(record, model=self.model)
        data = json.dumps(body, sort_keys=True).encode()
        url = self.base_url + systemone.SYSTEMONE_PATH
        config = {
            "url": url,
            "timeout": self.timeout,
            "body_sha256": hashlib.sha256(data).hexdigest(),
        }
        request = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=self.timeout) as response:  # nosec B310
                raw = response.read()
        except (OSError, ValueError) as exc:
            return self._failure(record, config, None, f"request to {url} failed: {exc}")
        digest = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            return self._failure(record, config, digest, f"response is not JSON: {exc}")
        questions = systemone.parse_response(record, rekey_scores(record, payload))
        return systemone.CallResult(str(record["id"]), self.model, config, digest, questions)
