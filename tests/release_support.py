"""Shared builders for the release tests: a toy run config, checkpoint, GGUF and fake hub."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jev_factory.factory.config import RunConfig
from jev_factory.release import scan
from jev_factory.release.bundle import BundleFiles, ModelPayload
from jev_factory.release.dataset_bundle import DatasetSources

TOKEN = "hf_fake_token_value_for_tests_only"
HUB_PREFIX = "example-org/toy-lamps-jev-"
REPO = HUB_PREFIX + "scorer"
APACHE_HEAD = "Apache License\nVersion 2.0, January 2004\n"
SAFETENSORS = (2).to_bytes(8, "little") + b"{}"


def make_config(**overrides: Any) -> RunConfig:
    values = {
        "work": "w",
        "base": "org/base",
        "base_rev": "abc",
        "hub_prefix": HUB_PREFIX,
        "licence": "Apache-2.0",
        "issue_refs": "",
        "hf_token_env": "HF_TOKEN",
        **overrides,
    }
    return RunConfig(values=values, sources={})


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def gguf_bytes(meta: dict[str, Any], version: int = 3) -> bytes:
    """A GGUF header with *meta* (str / int / list-of-str values) and no tensors."""

    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", len(meta))
    for key, value in meta.items():
        out += string(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + string(value)
        elif isinstance(value, list):
            out += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(string(v) for v in value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    return out + b"\x00" * 16


def write_gguf(path: Path, meta: dict[str, Any] | None = None) -> Path:
    meta = {"general.architecture": "qwen3", "quantize.imatrix.file": "imatrix.dat"} | (meta or {})
    path.write_bytes(gguf_bytes(meta))
    return path


_BUNDLE_FILES = ("calibration", "gate", "scorer_train")
_MODEL_PAYLOAD = ("merged", "kind", "gguf", "awq_dir", "quantized_from")
_DATASET_SOURCES = ("splits", "train_augmented", "accepted", "rejected")


def _take(kwargs: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: kwargs.pop(name) for name in names if name in kwargs}


def model_bundle_args(**flat: Any) -> dict[str, Any]:
    """``build_model_bundle`` keyword arguments from the flat names the tests use: the
    three bundle files grouped as ``files``, the checkpoint and its kind as ``payload``."""
    kwargs = dict(flat)
    kwargs["files"] = BundleFiles(**_take(kwargs, _BUNDLE_FILES))
    kwargs["payload"] = ModelPayload(**_take(kwargs, _MODEL_PAYLOAD))
    return kwargs


def dataset_bundle_args(**flat: Any) -> dict[str, Any]:
    """``build_dataset_bundle`` keyword arguments from flat names (``files``, ``sources``)."""
    kwargs = dict(flat)
    kwargs["files"] = BundleFiles(**_take(kwargs, _BUNDLE_FILES))
    kwargs["sources"] = DatasetSources(**_take(kwargs, _DATASET_SOURCES))
    return kwargs


def dataset_build_args(**flat: Any) -> dict[str, Any]:
    """``dataset_bundle.build`` keyword arguments from flat names (the run files as
    ``sources``; ``scorer_train`` stays its own argument there)."""
    kwargs = dict(flat)
    kwargs["sources"] = DatasetSources(**_take(kwargs, _DATASET_SOURCES))
    return kwargs


class Checkpoint(SimpleNamespace):
    """merged dir, base snapshot and report files for one toy model bundle."""


def make_inputs(tmp: Path) -> Checkpoint:
    snapshot = tmp / "cache" / "models--org--base" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "LICENSE").write_text(APACHE_HEAD + "...\n", encoding="utf-8")
    (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    merged = tmp / "merged"
    merged.mkdir()
    (merged / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (merged / "tokenizer.json").write_text("{}", encoding="utf-8")
    (merged / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1}), "utf-8")
    (merged / "model.safetensors").write_bytes(SAFETENSORS)
    report = tmp / "final.md"
    report.write_text("# Final run\n\n| metric | value |\n|---|---|\n| ece | 0.02 |\n", "utf-8")
    return Checkpoint(
        snapshot=snapshot,
        merged=merged,
        report=report,
        calibration=write_json(tmp / "cal-src.json", {"temperature": 1.1}),
        gate=write_json(tmp / "gate-src.json", {"floor": 0.5}),
        scorer_train=write_json(tmp / "st-src.json", {"entries": [{"text": "turn on the lamp"}]}),
    )


class FakeHub:
    """Stands in for the huggingface_hub module; 'remote' repos live in a temp folder."""

    def __init__(self, remote: Path, *, private: bool = True, tamper=None) -> None:
        self.remote = remote
        self.calls: list[tuple] = []
        self.private = private
        self.tamper = tamper
        hub = self

        class HfApi:
            def __init__(self, token=None) -> None:
                hub.calls.append(("HfApi", token == TOKEN))

            def create_repo(self, repo_id, **kwargs):
                hub.calls.append(("create_repo", repo_id, kwargs))
                folder = hub.remote / repo_id
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ".gitattributes").write_text("*.gguf filter=lfs\n")

            def update_repo_visibility(self, repo_id, **kwargs):
                hub.calls.append(("update_repo_visibility", repo_id, kwargs))

            def upload_folder(self, *, folder_path, repo_id, **kwargs):
                hub.calls.append(("upload_folder", repo_id, kwargs))
                shutil.copytree(folder_path, hub.remote / repo_id, dirs_exist_ok=True)
                return SimpleNamespace(oid="c0ffee")

            def repo_info(self, repo_id, **kwargs):
                hub.calls.append(("repo_info", repo_id, kwargs))
                return SimpleNamespace(private=hub.private)

            def list_repo_files(self, repo_id, **kwargs):
                hub.calls.append(("list_repo_files", repo_id, kwargs))
                root = hub.remote / repo_id
                return sorted(
                    p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                )

        self.HfApi = HfApi

    def snapshot_download(self, repo_id, *, local_dir, **kwargs):
        self.calls.append(("snapshot_download", repo_id, kwargs))
        shutil.copytree(self.remote / repo_id, local_dir, dirs_exist_ok=True)
        cache = Path(local_dir) / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "download.lock").write_text("x")
        if self.tamper:
            self.tamper(Path(local_dir))
        return str(local_dir)


def scanned_bundle(tmp: Path, name: str = "scorer") -> Path:
    """A finished, scanned toy bundle (required files + a bundle.json) ready to upload."""
    bundle = tmp / "bundles" / name
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "sub").mkdir(parents=True)
    (bundle / "README.md").write_text("---\nlicense: apache-2.0\n---\n# card\n")
    (bundle / "model.safetensors").write_bytes(SAFETENSORS + b"\x00weights\x01")
    (bundle / "sub" / "tokenizer.json").write_text("{}")
    for required in ("calibration.json", "gate.json", "scorer-train.json"):
        (bundle / required).write_text("{}")
    (bundle / "bundle.json").write_text(json.dumps({"surface_sha256": "a" * 64}))
    scan.write_scan(bundle)
    return bundle
