"""A torchao weight-only (int8 / int4) pointer student and a thin server for it (deviation d3).

``strands-decider serve`` @ 8b6faa5 loads the torso from the config's base id at the
checkpoint dtype and wraps it in ``PeftModel`` unmerged, so it has no quantized load path.
This module adds one, in two halves:

* **Here, stdlib only.** :func:`build_weight_only` plans (dry-run, writes nothing) or runs the
  build as a subprocess in the student env; :func:`read_artifact` verifies a built artifact
  (every file present, the self-description complete, every weight hash matching, fail
  closed); :func:`serve_argv` / :func:`launch` start the wrapper on 127.0.0.1 only;
  :func:`make_server` is the HTTP layer (``GET /health``, ``POST /v1/systemone``).
* **In the student env.** ``python -m jev_factory.backbones.pointer.quantized build`` loads the
  winner with upstream ``StrandsDeciderModel.load`` (base at the pinned
  ``base_model_revision``, its LoRA, ``head.safetensors``), merges the LoRA into the torso
  (``merge_and_unload``), quantizes the torso's ``nn.Linear`` weights weight-only with
  ``torchao.quantization.quantize_`` (``Int8WeightOnlyConfig`` / ``Int4WeightOnlyConfig``,
  or the pre-0.10 ``int8_weight_only`` / ``int4_weight_only`` functions where those are what
  the env has; the one used is recorded), keeps the pointer head fp32 and writes a
  self-describing artifact directory (:data:`ARTIFACT_FORMAT`): the quantized torso state
  (:data:`TORSO_FILE`, ``torch.save`` of the state dict), ``head.safetensors`` (fp32),
  ``strands_decider_config.json`` (``use_lora: false``, the pinned ``base_revision``), the
  tokenizer files and :data:`MANIFEST_FILE` (format, bits, torchao API and version, torch
  version, the source checkpoint's sha256s, ``base_model_revision`` and the sha256 of every
  written weight file). ``python -m jev_factory.backbones.pointer.quantized serve`` loads the
  artifact (torso skeleton at the pinned revision, the quantized state assigned over it with
  ``torch.load(weights_only=True)`` after the hashes are checked) and answers with upstream's
  own engine and schema.

Upstream reuse (strands-labs/strands-decider @ 8b6faa553038a59e73fa7f16c4081f8ceb21d27e,
Copyright the strands-decider authors, Apache License 2.0; see :data:`UPSTREAM_PROVENANCE`):
``src/strands_decider/modeling.py`` (``StrandsDeciderModel.load``, ``_load_torso``,
``build_head``, ``StrandsDeciderConfig``, ``ensure_bos``, ``serving_attn_implementation``)
and ``src/strands_decider/infer.py`` (``SystemOneEngine``, ``EngineConfig``) are called
as-is; ``src/strands_decider/schema.py`` ``SystemOneRequest`` validates every request and
``SystemOneResponse.model_dump`` renders every answer. The HTTP layer re-expresses
``src/strands_decider/server.py`` (same paths, the same ``/health`` fields, one engine call
at a time, a ``ValueError`` is a 422 with ``detail``, ``latency_ms`` added to the payload) on
the stdlib ``http.server`` instead of FastAPI/uvicorn. Heavy packages are loaded with
:func:`importlib.import_module` inside the in-env functions only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - the pinned student python, argv built here
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.pointer import layout as player
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill.envs import EnvSpec
from jev_factory.factory import gpu
from jev_factory.factory.detach import start_detached

MODULE = "jev_factory.backbones.pointer.quantized"
ARTIFACT_FORMAT = "jev-pointer-torchao-wo/1"
MANIFEST_FILE = "jev_quant.json"
TORSO_FILE = "torso_quantized.pt"
LOOPBACK = "127.0.0.1"
HEAD_DTYPE = "float32"
#: bits -> the deployed-format name :mod:`jev_factory.distill.quant` registers.
FORMAT_OF_BITS: dict[int, str] = {8: "torch_int8_wo", 4: "torch_int4_wo"}
#: bits -> the torchao API names tried in order (config class, then the pre-0.10 function).
TORCHAO_APIS: dict[int, tuple[str, str]] = {
    8: ("Int8WeightOnlyConfig", "int8_weight_only"),
    4: ("Int4WeightOnlyConfig", "int4_weight_only"),
}
#: The weight files the manifest hashes.
HASHED_FILES = (TORSO_FILE, player.HEAD, player.CONFIG)
REQUIRED_FILES = (MANIFEST_FILE, *HASHED_FILES, *player.TOKENIZER_FILES)
#: Tokenizer files carried when the source checkpoint has them.
TOKENIZER_EXTRA = (
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
#: The source checkpoint files whose sha256 the manifest records.
SOURCE_FILES = (player.ADAPTER_CONFIG, player.ADAPTER, player.HEAD, player.CONFIG)
#: :func:`write_manifest`'s required inputs (all kept verbatim in the manifest).
MANIFEST_INPUTS = (
    "bits",
    "torchao_api",
    "torchao_version",
    "torch_version",
    "source",
    "base_model",
    "base_model_revision",
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")

UPSTREAM_PROVENANCE: dict[str, Any] = {
    "repo": "strands-labs/strands-decider",
    "commit": "8b6faa553038a59e73fa7f16c4081f8ceb21d27e",
    "licence": "Apache-2.0",
    "files": {
        "src/strands_decider/modeling.py": (
            "StrandsDeciderModel.load / _load_torso, build_head, StrandsDeciderConfig, "
            "ensure_bos, serving_attn_implementation called as-is"
        ),
        "src/strands_decider/infer.py": "SystemOneEngine and EngineConfig called as-is",
        "src/strands_decider/schema.py": (
            "SystemOneRequest validates requests; SystemOneResponse.model_dump renders answers"
        ),
        "src/strands_decider/server.py": (
            "/health and /v1/systemone paths, health fields, the engine lock, 422 on "
            "ValueError and latency_ms re-expressed on stdlib http.server"
        ),
    },
}


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(EXIT_USER_ERROR, message, remediation)


def _jev_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pythonpath(decider_repo: Path | str | None) -> str:
    parts = []
    if decider_repo:
        repo = Path(decider_repo)
        parts += [str(repo), str(repo / "src")]
    return os.pathsep.join([*parts, str(_jev_root())])


def _require_loopback(host: str) -> None:
    if host != LOOPBACK:
        raise _refuse(
            f"refusing to bind {host!r}: the quantized student server binds {LOOPBACK} only",
            f"use host {LOOPBACK}",
        )


def _json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _refuse(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _refuse(f"{path.name} is not a JSON object")
    return data


def _bits(bits: Any) -> int:
    if isinstance(bits, bool) or bits not in FORMAT_OF_BITS:
        raise _refuse(f"bits must be 8 or 4 (weight-only int8 / int4), got {bits!r}")
    return int(bits)


# --------------------------------------------------------------------- the artifact


def source_record(checkpoint: Path | str) -> dict[str, Any]:
    """The source checkpoint's path and the sha256 of its adapter, head and config."""
    root = Path(checkpoint)
    return {
        "checkpoint": str(root),
        "sha256": {n: player.sha256_file(root / n) for n in SOURCE_FILES if (root / n).is_file()},
    }


def write_manifest(
    root: Path | str,
    *,
    bits: int,
    torchao_api: str,
    torchao_version: str,
    torch_version: str,
    source: Mapping[str, Any],
    base_model: Any,
    base_model_revision: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write :data:`MANIFEST_FILE` over the weight files already in *root*; return it."""
    root = Path(root)
    bits = _bits(bits)
    manifest = {
        **dict(extra or {}),
        "artifact_format": ARTIFACT_FORMAT,
        "format": FORMAT_OF_BITS[bits],
        "bits": bits,
        "torchao_api": torchao_api,
        "torchao_version": torchao_version,
        "torch_version": torch_version,
        "head_dtype": HEAD_DTYPE,
        "lora_merged": True,
        "source": dict(source),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "files": {name: player.sha256_file(root / name) for name in HASHED_FILES},
        "upstream": {k: UPSTREAM_PROVENANCE[k] for k in ("repo", "commit", "licence")},
    }
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _nonempty_str(manifest: Mapping[str, Any], key: str) -> None:
    value = manifest.get(key)
    if not (isinstance(value, str) and value.strip()):
        raise _refuse(f"{MANIFEST_FILE}: {key} must name what built the artifact, got {value!r}")


def _check_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise _refuse(
            f"{MANIFEST_FILE}: artifact_format is {manifest.get('artifact_format')!r}, "
            f"not {ARTIFACT_FORMAT!r}"
        )
    bits = manifest.get("bits")
    if isinstance(bits, bool) or bits not in FORMAT_OF_BITS:
        raise _refuse(f"{MANIFEST_FILE}: bits must be 8 or 4, got {bits!r}")
    if manifest.get("format") != FORMAT_OF_BITS[bits]:
        raise _refuse(
            f"{MANIFEST_FILE}: format {manifest.get('format')!r} does not match "
            f"{bits} bits ({FORMAT_OF_BITS[bits]})"
        )
    if manifest.get("head_dtype") != HEAD_DTYPE or manifest.get("lora_merged") is not True:
        raise _refuse(
            f"{MANIFEST_FILE}: the pointer head must stay {HEAD_DTYPE} over a merged torso "
            f"(head_dtype {manifest.get('head_dtype')!r}, lora_merged "
            f"{manifest.get('lora_merged')!r})"
        )
    revision = manifest.get("base_model_revision")
    if not (isinstance(revision, str) and _REVISION.match(revision)):
        raise _refuse(f"{MANIFEST_FILE}: base_model_revision must be a 40-hex commit")
    for key in ("torchao_api", "torchao_version", "torch_version"):
        _nonempty_str(manifest, key)
    source = manifest.get("source")
    if not (
        isinstance(source, dict) and isinstance(source.get("sha256"), dict) and source["sha256"]
    ):
        raise _refuse(f"{MANIFEST_FILE}: source must carry the source checkpoint's sha256s")


def read_artifact(path: Path | str) -> dict[str, Any]:
    """Verify a :data:`ARTIFACT_FORMAT` directory; return its manifest and config. Fail closed."""
    root = Path(path)
    if not root.is_dir():
        raise _refuse(f"{root} is not a local artifact directory", "build the artifact first")
    if missing := [n for n in REQUIRED_FILES if not (root / n).is_file()]:
        raise _refuse(
            f"{root}: the quantized artifact is missing {', '.join(missing)}",
            f"an artifact needs every one of: {', '.join(REQUIRED_FILES)}",
        )
    manifest = _json_object(root / MANIFEST_FILE)
    _check_manifest(manifest)
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for name in HASHED_FILES:
        actual = player.sha256_file(root / name)
        if str(files.get(name, "")).lower() != actual:
            raise _refuse(
                f"{MANIFEST_FILE}: the sha256 recorded for {name} does not match the file "
                f"(sha256 {actual})",
                "rebuild the artifact; never edit a built artifact by hand",
            )
    config = _json_object(root / player.CONFIG)
    if config.get("head_type") != "pointer":
        raise _refuse(
            f"{player.CONFIG} has head_type {config.get('head_type')!r}; this serves pointer "
            "heads only"
        )
    return {"root": str(root), "manifest": manifest, "config": config}


# --------------------------------------------------------------------- the build (plan / run)


def plan_weight_only(
    checkpoint_dir: Path | str,
    out_dir: Path | str,
    bits: int,
    *,
    python: str,
    decider_repo: Path | str | None = None,
    base_revision: str | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """The in-env build command for *checkpoint_dir*; validates the inputs, runs nothing."""
    bits = _bits(bits)
    src, out = Path(checkpoint_dir), Path(out_dir)
    player.plan_export(src, out)  # refuses an incomplete or non-pointer checkpoint
    revision = player._base_revision(src, base_revision)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise _refuse(f"{out} exists and is not empty", "build into a fresh directory")
    if not (isinstance(python, str) and python.strip()):
        raise _refuse("the build needs the student env's python", "pass python=<student python>")
    argv = [
        python,
        "-m",
        MODULE,
        "build",
        str(src),
        str(out),
        "--bits",
        str(bits),
        "--base-revision",
        revision,
        "--device",
        device,
    ]
    return {
        "backbone": "pointer",
        "stage": "quantize",
        "format": FORMAT_OF_BITS[bits],
        "bits": bits,
        "dry_run": True,
        "argv": argv,
        "env": {"PYTHONPATH": _pythonpath(decider_repo)},
        "checkpoint": str(src),
        "out_dir": str(out),
        "base_model_revision": revision,
        "torchao_api": TORCHAO_APIS[bits],
        "head_dtype": HEAD_DTYPE,
        "lora_merged": True,
        "device": device,
        "upstream": {k: UPSTREAM_PROVENANCE[k] for k in ("repo", "commit", "licence")},
    }


#: argv, extra env -> a completed process (stdout/stderr text).
Runner = Callable[[Sequence[str], Mapping[str, str]], "subprocess.CompletedProcess[str]"]


def build_weight_only(
    checkpoint_dir: Path | str,
    out_dir: Path | str,
    bits: int,
    *,
    apply: bool = False,
    python: str = "python",
    decider_repo: Path | str | None = None,
    base_revision: str | None = None,
    device: str = "cuda",
    runner: Runner | None = None,
    timeout: float | None = None,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    allow_foreign: bool = False,
) -> dict[str, Any]:
    """Dry-run: the plan (argv, env), nothing written. *apply*: build and verify the artifact.

    The build runs through ``factory.gpu.run_gpu_stage("quantize", ...)``: the GPU residency
    guard, a hard *memory_max* cap and the *memory_floor* watchdog (GB10 GPU allocations are
    not charged to the cgroup). *apply* is refused without *memory_max*. *runner* replaces
    that call (a test seam); the cap is still required.
    """
    plan = plan_weight_only(
        checkpoint_dir,
        out_dir,
        bits,
        python=python,
        decider_repo=decider_repo,
        base_revision=base_revision,
        device=device,
    )
    if not apply:
        return plan
    if not memory_max:
        raise _refuse(
            "a GPU quantize build needs a memory cap",
            "pass memory_max (e.g. '100G') with apply; GB10 GPU memory skips the cgroup cap",
        )
    if runner is None:
        out_path = Path(plan["out_dir"])
        run_dir = out_path.parent / f"{out_path.name}.capped"
        run_dir.mkdir(parents=True, exist_ok=True)
        done = gpu.run_gpu_stage(
            "quantize",
            run_dir,
            list(plan["argv"]),
            allow_foreign=allow_foreign,
            memory_max=memory_max,
            memory_floor=memory_floor,
            env={**os.environ, **plan["env"]},
            timeout=timeout,
        )
    else:
        done = runner(plan["argv"], plan["env"])
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-5:]
        why = (
            f"stopped by the memory-floor watchdog (MemAvailable below {memory_floor})"
            if done.returncode == gpu.WATCHDOG_STATUS
            else f"exited {done.returncode}"
        )
        raise CliError(
            EXIT_ENV_ERROR,
            f"the torchao {plan['format']} build {why}: {' | '.join(tail)}",
            "read the build's stderr; the student env needs torch, torchao and strands_decider",
        )
    try:
        info = read_artifact(plan["out_dir"])
    except CliError as exc:
        raise CliError(
            EXIT_ENV_ERROR,
            f"the build exited 0 but left no valid {MANIFEST_FILE} artifact: {exc.message}",
        ) from exc
    return {
        **plan,
        "dry_run": False,
        "artifact": info,
        "capped": {"memory_max": memory_max, "memory_floor": memory_floor},
    }


# --------------------------------------------------------------------- serving (here)


#: Recorded in ``<name>.serve.json``: servers are long-lived and run without the build cap.
UNCAPPED_NOTE = (
    "long-lived server, not run under capped.sh (no hard memory cap, no floor watchdog); "
    "residual risk: it can use GPU/unified memory the cgroup does not charge"
)


def serve_argv(
    python: str,
    artifact: Path | str,
    *,
    port: int,
    host: str = LOOPBACK,
    decider_repo: Path | str | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> list[str]:
    """The wrapper's command for a verified artifact, on 127.0.0.1 only."""
    _require_loopback(host)
    root = Path(artifact)
    read_artifact(root)
    argv = [
        "env",
        f"PYTHONPATH={_pythonpath(decider_repo)}",
        python,
        "-m",
        MODULE,
        "serve",
        str(root),
        "--host",
        host,
        "--port",
        str(int(port)),
    ]
    if model_name:
        argv += ["--model-name", model_name]
    if device:
        argv += ["--device", device]
    return argv


def launch(
    jobdir: Path | str,
    name: str,
    env: EnvSpec,
    artifact: Path | str,
    *,
    port: int,
    decider_repo: Path | str | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> int:
    """Start the wrapper detached in the pinned student *env*; return its pid.

    ``<name>.serve.json`` records the argv, the environment and the artifact's format and
    weight hashes, so a measurement can cite what answered it.
    """
    if not isinstance(env, EnvSpec) or env.role != "student" or not env.python:
        raise _refuse(
            "the quantized pointer server runs in the pinned student environment (a python path)",
            "describe it with distill.envs.EnvSpec(role='student', python=..., lockfile_hash=...)",
        )
    argv = serve_argv(
        env.python,
        artifact,
        port=port,
        decider_repo=decider_repo,
        model_name=model_name or name,
        device=device,
    )
    manifest = read_artifact(artifact)["manifest"]
    jobdir = Path(jobdir)
    jobdir.mkdir(parents=True, exist_ok=True)
    record = {
        "argv": argv,
        "environment": env.record(),
        "artifact": {
            "format": manifest["format"],
            "files": manifest["files"],
            "manifest_sha256": player.sha256_file(Path(artifact) / MANIFEST_FILE),
        },
        "capped": False,
        "capped_note": UNCAPPED_NOTE,
    }
    (jobdir / f"{name}.serve.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return start_detached(jobdir, name, argv)


class RequestError(Exception):
    """A caller error: answered with *status* and ``{"detail": detail}`` (FastAPI's shape)."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def make_server(
    host: str,
    port: int,
    *,
    evaluate: Callable[[dict[str, Any]], Mapping[str, Any]],
    health: Callable[[], Mapping[str, Any]],
) -> HTTPServer:
    """A single-threaded HTTP server (one engine call at a time) on 127.0.0.1 only."""
    _require_loopback(host)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            if self.path == "/health":
                self._send(200, dict(health()))
            else:
                self._send(404, {"detail": "Not Found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            if self.path != "/v1/systemone":
                self._send(404, {"detail": "Not Found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"null")
            except ValueError as exc:
                self._send(422, {"detail": f"request body is not valid JSON: {exc}"})
                return
            if not isinstance(body, dict):
                self._send(422, {"detail": "request body must be a JSON object"})
                return
            started = time.perf_counter()
            try:
                payload = dict(evaluate(body))
            except RequestError as exc:
                self._send(exc.status, {"detail": exc.detail})
                return
            except Exception as exc:  # noqa: BLE001 - answered, never a dropped connection
                self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})
                return
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self._send(200, payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    return HTTPServer((host, int(port)), Handler)


# --------------------------------------------------------------------- the student env


def _load(name: str) -> Any:
    return importlib.import_module(name)


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _quant_config(bits: int) -> tuple[str, Any]:
    aoq = _load("torchao.quantization")
    for api in TORCHAO_APIS[bits]:
        factory = getattr(aoq, api, None)
        if factory is not None:
            return api, factory()
    raise CliError(
        EXIT_ENV_ERROR,
        f"torchao {_version('torchao')} has none of {list(TORCHAO_APIS[bits])}",
        "install a torchao with weight-only int8/int4 quantization in the student env",
    )


def quantize_torso(torso: Any, bits: int) -> str:
    """Quantize *torso*'s ``nn.Linear`` weights weight-only in place; return the API used."""
    api, config = _quant_config(_bits(bits))
    _load("torchao.quantization").quantize_(torso, config)
    return api


def load_quantized_state(module: Any, path: Path | str, *, device: str) -> None:
    """Assign a saved quantized state dict over *module* (``weights_only`` unpickling only)."""
    torch = _load("torch")
    _load("torchao")  # registers torchao's tensor subclasses as safe globals
    state = torch.load(str(path), map_location=device, weights_only=True)
    module.load_state_dict(state, assign=True)


def _load_pinned(modeling: Any, checkpoint: Path, revision: str) -> Any:
    """``StrandsDeciderModel.load`` with the base at *revision*, never the default branch."""
    pinned = _json_object(checkpoint / player.CONFIG).get("base_revision")
    if pinned and pinned != revision:
        raise _refuse(f"{player.CONFIG} pins base_revision {pinned}, not {revision}")
    original = modeling.base_revision
    modeling.base_revision = lambda path, config: revision
    try:
        model = modeling.StrandsDeciderModel.load(str(checkpoint))
    finally:
        modeling.base_revision = original
    if model.config.base_revision != revision:
        raise CliError(EXIT_ENV_ERROR, "the base model did not load at the pinned revision")
    return model


def _build_in_env(checkpoint: Path, out: Path, bits: int, revision: str, device: str) -> int:
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    safetensors = _load("safetensors.torch")
    plan_weight_only(checkpoint, out, bits, python=sys.executable, base_revision=revision)
    model = _load_pinned(modeling, checkpoint, revision)
    torso = model.torso
    if hasattr(torso, "merge_and_unload"):
        torso = torso.merge_and_unload()
    torso = torso.to(device)
    with torch.no_grad():
        api, config = _quant_config(bits)
        _load("torchao.quantization").quantize_(torso, config)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(torso.state_dict(), str(out / TORSO_FILE))
    head = {
        k: v.detach().to(torch.float32).cpu().contiguous()
        for k, v in model.head.state_dict().items()
    }
    safetensors.save_file(head, str(out / player.HEAD))
    config_doc = _json_object(checkpoint / player.CONFIG)
    config_doc.update({"base_revision": revision, "use_lora": False})
    (out / player.CONFIG).write_text(json.dumps(config_doc, indent=2) + "\n")
    for name in (*player.TOKENIZER_FILES, *TOKENIZER_EXTRA):
        if (checkpoint / name).is_file():
            shutil.copyfile(checkpoint / name, out / name)
    manifest = write_manifest(
        out,
        bits=bits,
        torchao_api=api,
        torchao_version=_version("torchao") or "",
        torch_version=str(torch.__version__),
        source=source_record(checkpoint),
        base_model=config_doc.get("base_model"),
        base_model_revision=revision,
        extra={
            "quantization": {
                "call": f"torchao.quantization.quantize_(torso, {api}(...))",
                "config": repr(config),
                "scope": "torso nn.Linear weights, weight-only, after merge_and_unload of "
                "the LoRA; the pointer head is not quantized (fp32)",
            },
            "device": device,
            "versions": {
                d: _version(d) for d in ("transformers", "peft", "safetensors", "strands-decider")
            },
        },
    )
    print(json.dumps({"artifact": str(out), "format": manifest["format"], "torchao_api": api}))
    return 0


def _load_engine(artifact: Path, device: str, model_name: str) -> tuple[Any, dict[str, Any]]:
    info = read_artifact(artifact)
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    infer = _load("strands_decider.infer")
    transformers = _load("transformers")
    safetensors = _load("safetensors.torch")
    revision = info["manifest"]["base_model_revision"]
    config = modeling.StrandsDeciderConfig.from_json(str(artifact / player.CONFIG))
    if config.base_revision != revision or config.use_lora:
        raise _refuse(f"{player.CONFIG} must pin {revision} with use_lora false")
    tok = transformers.AutoTokenizer.from_pretrained(str(artifact))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if config.force_bos:
        modeling.ensure_bos(tok)
    device_map = device if device not in ("cpu", "mps", "mlx") else None
    torso = modeling.StrandsDeciderModel._load_torso(config, device_map, None)
    attn = modeling.serving_attn_implementation(torso.config)
    if attn:
        torso.set_attn_implementation(attn)
    load_quantized_state(torso, artifact / TORSO_FILE, device=device)
    model = modeling.StrandsDeciderModel.__new__(modeling.StrandsDeciderModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model.torso = torso
    model.tokenizer = tok
    model.head = modeling.build_head(config, modeling.StrandsDeciderModel.hidden_size(torso))
    model.head.load_state_dict(safetensors.load_file(str(artifact / player.HEAD)))
    model.head.to(torch.float32)
    with torch.inference_mode():
        engine = infer.SystemOneEngine(
            model, infer.EngineConfig(device=device, model_name=model_name)
        )
    return engine, info


def _serve_in_env(artifact: Path, host: str, port: int, device: str, model_name: str) -> int:
    engine, info = _load_engine(artifact, device, model_name)
    torch = _load("torch")
    schema = _load("strands_decider.schema")
    pydantic = _load("pydantic")
    lock = threading.Lock()

    def evaluate(body: dict[str, Any]) -> Mapping[str, Any]:
        try:
            request = schema.SystemOneRequest.model_validate(body)
        except pydantic.ValidationError as exc:
            raise RequestError(422, json.loads(exc.json())) from exc
        try:
            with lock, torch.inference_mode():
                response = engine.evaluate(request)
        except ValueError as exc:
            raise RequestError(422, str(exc)) from exc
        return response.model_dump()

    def health() -> Mapping[str, Any]:
        cfg = engine.model.config
        return {
            "status": "ok",
            "model": engine.cfg.model_name,
            "checkpoint": str(artifact),
            "base_model": cfg.base_model,
            "num_slots": cfg.num_slots,
            "max_length": cfg.max_length,
            "temperature": cfg.temperature,
            "device": engine.cfg.device,
            "prefix_cache": engine.cfg.use_prefix_cache,
            "vision": False,
            "quant_format": info["manifest"]["format"],
            "torchao_api": info["manifest"]["torchao_api"],
        }

    server = make_server(host, port, evaluate=evaluate, health=health)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=MODULE, description="torchao weight-only pointer student.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="merge, quantize and save (student env)")
    b.add_argument("checkpoint")
    b.add_argument("out")
    b.add_argument("--bits", type=int, required=True, choices=sorted(FORMAT_OF_BITS))
    b.add_argument("--base-revision", required=True)
    b.add_argument("--device", default="cuda")
    s = sub.add_parser("serve", help="answer /v1/systemone on 127.0.0.1 (student env)")
    s.add_argument("artifact")
    s.add_argument("--host", default=LOOPBACK)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="cuda")
    s.add_argument("--model-name", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point in the student env: ``build`` or ``serve``."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            _require_loopback(args.host)
            artifact = Path(args.artifact)
            name = args.model_name or artifact.name
            return _serve_in_env(artifact, args.host, args.port, args.device, name)
        return _build_in_env(
            Path(args.checkpoint), Path(args.out), args.bits, args.base_revision, args.device
        )
    except CliError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.remediation:
            print(f"hint: {exc.remediation}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":  # runs in the student env
    sys.exit(main())
