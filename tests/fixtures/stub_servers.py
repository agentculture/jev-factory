"""Stub ``llama-server``, ``docker`` and ``nvidia-smi`` binaries for the measure tests.

No real server, GPU or model is ever started: each stub is a small Python
script written into a temporary directory with this interpreter as its
shebang, so it can be put on ``PATH`` or named as ``llama_server``.

The stub ``llama-server`` answers ``--version``, then serves on
``127.0.0.1:--port``:

* ``GET /v1/models`` -> its ``--alias``, ``owned_by: "llamacpp"``;
* ``GET /props`` -> ``default_generation_settings.n_ctx`` = ``--ctx-size``;
* ``POST /v1/completions`` -> one JSON line per request appended to
  ``$STUB_LLAMA_LOG`` (the body, minus the prompt), and a legacy
  ``top_logprobs`` reply holding every offered letter: the candidate picked by
  ``$STUB_LLAMA_PICKS`` (``{"substring of the request": candidate name}``,
  default ``escalate``) at probability 0.9, the rest sharing 0.1.

``$STUB_LLAMA_FAIL=1`` makes it exit at once; ``$STUB_LLAMA_DELAY`` delays
listening by that many seconds.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

LLAMA_SERVER = r"""
import json, math, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("version: 9999 (stub llama-server)")
    sys.exit(0)
if os.environ.get("STUB_LLAMA_FAIL") == "1":
    sys.exit(1)

def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default

PORT = int(opt("--port"))
ALIAS = opt("--alias", "model")
CTX = int(opt("--ctx-size", "0"))
LOG = os.environ.get("STUB_LLAMA_LOG")
PICKS = json.loads(os.environ.get("STUB_LLAMA_PICKS") or "{}")
LINE = re.compile(r"^([A-Za-z])\) (\S+): ", re.MULTILINE)

class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("/v1/models", "/models"):
            self._send({"object": "list", "data": [{"id": ALIAS, "owned_by": "llamacpp"}]})
        elif path == "/props":
            self._send({"default_generation_settings": {"n_ctx": CTX}})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = body.get("prompt", "")
        if LOG:
            logged = {key: value for key, value in body.items() if key != "prompt"}
            logged["path"] = self.path
            with open(LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(logged) + "\n")
        letters = {name: letter for letter, name in LINE.findall(prompt)}
        request = prompt.rsplit("\n", 1)[-1]
        pick = next((name for key, name in PICKS.items() if key in request), "escalate")
        rest = 0.1 / max(len(letters) - 1, 1)
        top = {
            letter: math.log(0.9 if name == pick else rest) for name, letter in letters.items()
        }
        choice = {"text": letters.get(pick, ""), "logprobs": {"top_logprobs": [top]}}
        self._send({"choices": [choice]})

    def log_message(self, *_args):
        pass

time.sleep(float(os.environ.get("STUB_LLAMA_DELAY") or 0))
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""

DOCKER = r"""
import json, os, sys
args = sys.argv[1:]
log = os.environ.get("STUB_DOCKER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(args) + "\n")
if args[:2] == ["ps", "-a"]:
    print("\n".join(os.environ.get("STUB_DOCKER_EXISTING", "").split()))
    sys.exit(int(os.environ.get("STUB_DOCKER_PS_CODE", "0")))
if args[:1] == ["run"]:
    print("0123456789ab")
    sys.exit(int(os.environ.get("STUB_DOCKER_RUN_CODE", "0")))
if args[:1] == ["inspect"]:
    print(os.environ.get("STUB_DOCKER_RUNNING", "true"))
    sys.exit(0)
if args[:1] == ["logs"]:
    print("stub vllm log line")
    sys.exit(0)
sys.exit(0)
"""

NVIDIA_SMI = r"""
import os, sys
if "--query-compute-apps=pid,process_name" in sys.argv:
    print(os.environ.get("STUB_GPU_APPS", ""))
    sys.exit(0)
print("NVIDIA-SMI stub")
"""


def write_stub(directory: Path, name: str, source: str) -> Path:
    """Write *source* as an executable script called *name* in *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{source.lstrip()}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def stub_bin(directory: Path) -> Path:
    """A directory holding stub ``llama-server``, ``docker`` and ``nvidia-smi``."""
    write_stub(directory, "llama-server", LLAMA_SERVER)
    write_stub(directory, "docker", DOCKER)
    write_stub(directory, "nvidia-smi", NVIDIA_SMI)
    return directory


def path_with(directory: Path) -> str:
    """``PATH`` with *directory* first."""
    return f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


def free_port() -> int:
    """A localhost port nothing listens on right now."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
