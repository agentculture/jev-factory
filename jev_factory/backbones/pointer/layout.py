"""The ``hobson-hf-export/1`` bundle layout that ``strands-decider serve`` loads.

Upstream (strands-labs/strands-decider @ 8b6faa5, ``src/strands_decider/hf_export.py`` and
``modeling.py``; published example StrandsAgents/strands-decider-2B-hobson-v21 @ 2b52a62):
the LoRA adapter lives under ``lora/`` (``adapter_config.json`` + ``adapter_model.safetensors``),
the pointer head is ``head.safetensors``, the model config is ``strands_decider_config.json`` and
``provenance.json`` carries ``format``, ``base_model``, ``base_model_revision`` and the full
sha256 of the adapter and head. ``modeling.load`` takes the base revision from provenance.json,
so a bundle without a pinned revision would load the base from the Hub's default branch.

Reading verifies every required file, the format string, the pointer head type and both
hashes, failing closed. Writing is dry-run unless asked. Stdlib only: no torch, so a trainer's
pickled ``slot_head.pt`` is not converted here (convert it in the student env first).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

FORMAT = "hobson-hf-export/1"
ADAPTER = "lora/adapter_model.safetensors"
ADAPTER_CONFIG = "lora/adapter_config.json"
HEAD = "head.safetensors"
CONFIG = "strands_decider_config.json"
PROVENANCE = "provenance.json"
MANIFEST = "MANIFEST.sha256"
REQUIRED_FILES = (ADAPTER_CONFIG, ADAPTER, HEAD, CONFIG, PROVENANCE)
#: ``modeling.load`` reads the tokenizer from the bundle itself; required to serve.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
#: Carried when present; never required.
OPTIONAL_FILES = ("chat_template.jinja", "train_config.json", "history.json")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise _refuse(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _refuse(f"{path.name} is not a JSON object")
    return data


def _missing(root: Path, names) -> list[str]:
    return [name for name in names if not (root / name).is_file()]


def _check_config(root: Path) -> dict[str, Any]:
    config = _json_object(root / CONFIG)
    if config.get("head_type") != "pointer":
        raise _refuse(
            f"{CONFIG} has head_type {config.get('head_type')!r}; the pointer backbone "
            "serves pointer heads only",
            "export a checkpoint trained with head_type: pointer",
        )
    return config


def read_layout(path: str | Path) -> dict[str, Any]:
    """Verify a ``hobson-hf-export/1`` bundle; return its format, config, provenance, hashes."""
    root = Path(path)
    if not root.is_dir():
        raise _refuse(f"{root} is not a local bundle directory", "export or fetch the bundle")
    if missing := _missing(root, REQUIRED_FILES):
        raise _refuse(
            f"{root}: hobson-hf-export/1 layout is missing {', '.join(missing)}",
            f"a bundle needs every one of: {', '.join(REQUIRED_FILES)}",
        )
    provenance = _json_object(root / PROVENANCE)
    if provenance.get("format") != FORMAT:
        raise _refuse(f"{PROVENANCE} format is {provenance.get('format')!r}, not {FORMAT!r}")
    config = _check_config(root)
    files = {name: sha256_file(root / name) for name in (ADAPTER_CONFIG, ADAPTER, HEAD, CONFIG)}
    for key, name in (("adapter_sha256", ADAPTER), ("head_sha256", HEAD)):
        if str(provenance.get(key, "")).lower() != files[name]:
            raise _refuse(
                f"{key} in {PROVENANCE} does not match {name} (sha256 {files[name]})",
                "re-export the bundle; never edit provenance by hand",
            )
    return {
        "root": str(root),
        "format": FORMAT,
        "config": config,
        "provenance": provenance,
        "files": files,
    }


def _base_revision(src: Path, override: str | None) -> str:
    revision = override
    if revision is None and (src / PROVENANCE).is_file():
        revision = _json_object(src / PROVENANCE).get("base_model_revision")
    if not isinstance(revision, str) or not _REVISION.match(revision):
        raise _refuse(
            f"no pinned base_model_revision for {src} (got {revision!r})",
            "pass the base model's 40-hex commit; strands-decider loads the base at it",
        )
    return revision


def plan_export(src: str | Path, out: str | Path) -> dict[str, Any]:
    """What an export of *src* into *out* would write; refuses an incomplete source."""
    src, out = Path(src), Path(out)
    needed = [name for name in REQUIRED_FILES if name != PROVENANCE] + list(TOKENIZER_FILES)
    if missing := _missing(src, needed):
        hint = (
            " (found slot_head.pt: convert it in the student env)"
            if (HEAD in missing and (src / "slot_head.pt").is_file())
            else ""
        )
        raise _refuse(
            f"{src}: source checkpoint is missing {', '.join(missing)}{hint}",
            "export from a complete pointer checkpoint",
        )
    _check_config(src)
    copies = needed + [name for name in OPTIONAL_FILES if (src / name).is_file()]
    return {"src": str(src), "out": str(out), "copy": copies, "write": [PROVENANCE, MANIFEST]}


def write_layout(
    src: str | Path,
    out: str | Path,
    *,
    base_revision: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy *src* into *out* as a ``hobson-hf-export/1`` bundle, then read it back."""
    plan = plan_export(src, out)
    src, out = Path(src), Path(out)
    revision = _base_revision(src, base_revision)
    if out.exists() and any(out.iterdir()):
        raise _refuse(f"{out} exists and is not empty", "export into a fresh directory")
    config = _json_object(src / CONFIG)
    for name in plan["copy"]:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / name, out / name)
    provenance = {
        **dict(extra or {}),
        "format": FORMAT,
        "base_model": config.get("base_model"),
        "base_model_revision": revision,
        "adapter_sha256": sha256_file(out / ADAPTER),
        "head_sha256": sha256_file(out / HEAD),
        "strands_decider_config_sha256": sha256_file(out / CONFIG),
    }
    (out / PROVENANCE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    (out / MANIFEST).write_text("".join(f"{sha256_file(out / n)}  {n}\n" for n in names))
    return read_layout(out)
