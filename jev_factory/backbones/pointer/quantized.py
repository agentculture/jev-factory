"""The quantized pointer student (< 10^9 bytes of weights) and a thin server for it (d3, #20).

``strands-decider serve`` @ 8b6faa5 loads the torso from the config's base id at the
checkpoint dtype and wraps it in ``PeftModel`` unmerged, so it has no quantized load path.
This module adds one, in two halves:

* **Here, stdlib only.** :func:`build_weight_only` plans (dry-run, writes nothing) or runs the
  build as a subprocess in the student env; :func:`read_artifact` verifies a built artifact
  (every file present, the self-description complete, every hash and the weight byte sum
  matching, the tokenizer unchanged from its source, fail closed); :func:`serve_argv` /
  :func:`launch` start the wrapper on 127.0.0.1 only; :func:`make_server` is the HTTP layer
  (``GET /health``, ``GET /v1/bundle``, ``GET /v1/stats``, ``POST /v1/systemone``).
  :func:`module_class` / :func:`check_inventory` are the tensor policy both halves share;
  :func:`host_stats` reads what ``/v1/stats`` reports from ``/proc``, ``/sys`` and ``/etc``.
* **In the student env.** ``python -m jev_factory.backbones.pointer.quantized build`` loads the
  winner with upstream ``StrandsDeciderModel.load`` (base at the pinned
  ``base_model_revision``, its LoRA, ``head.safetensors``), merges the LoRA into the torso
  (``merge_and_unload``) and applies :class:`QuantScheme`:

  - the vision tower and the MTP head are removed (:data:`DROP_SEGMENTS`);
  - the full-vocabulary token embedding (every row kept) becomes a packed embedding:
    grouped symmetric int4 (two values per byte) or int8, one scale per
    ``embedding_group_size`` values in the model dtype; a lookup is a gather plus a
    multiply, so it needs no quantized kernel;
  - the text blocks' linears (:data:`QUANTIZED_LINEARS` under ``layers``) are quantized
    weight-only with ``torchao.quantization.quantize_`` (``Int4WeightOnlyConfig`` with
    ``block_group_size`` / ``Int8WeightOnlyConfig``, or the pre-0.10 functions; the one used
    is recorded);
  - ``in_proj_a`` / ``in_proj_b``, ``conv1d``, ``A_log``, ``dt_bias`` and every norm stay
    unquantized, and the pointer head stays fp32 in its own file.

  The torso's state is checked against that policy (:func:`check_inventory`) before it is
  written. The artifact directory (:data:`ARTIFACT_FORMAT`) holds the torso state
  (:data:`TORSO_FILE`, ``torch.save``), ``head.safetensors`` (fp32),
  ``strands_decider_config.json`` (``use_lora: false``, the pinned ``base_revision``), the
  source checkpoint's tokenizer files byte for byte, and :data:`MANIFEST_FILE`: the scheme,
  the torchao API and versions, the source checkpoint's sha256s, the sha256 of every file,
  the byte size of each weight file and their sum against :data:`SIZE_BUDGET_BYTES`.
  ``python -m jev_factory.backbones.pointer.quantized serve`` loads the artifact (torso
  skeleton at the pinned revision with the same modules dropped and the packed embedding
  installed, the quantized state assigned over it with ``torch.load(weights_only=True)``
  after the hashes are checked) and answers with upstream's own engine and schema.

**Pruned-vocabulary comparison (decision c38, h9).** Given a prompt set (the run's train,
fit and selection ``/v1/systemone`` bodies; :data:`PROMPT_SIDES` only, never test or held-out),
the build runs every prompt through upstream's engine over the merged bf16 torso, records
each token id the embedding looks up, adds the tokenizer's special and added tokens
(:func:`special_token_ids`) and packs only those rows (:func:`kept_vocabulary`). The kept id
list ships as :data:`KEPT_IDS_FILE`; the manifest records its sha256 and the weight bytes,
and its format is :data:`PRUNED_FORMAT_OF_BITS` (never a shipped format name). Served, an id
outside the kept list raises :class:`OutOfVocabulary` at the embedding gather itself, so a
request is refused with an ``out_of_vocabulary`` 422 and never answered from a wrong row.
It is a comparison only: :mod:`jev_factory.distill.quant` measures it beside the
full-vocabulary build and never selects it.

Which block kernel runs on the Jetson (int4 on ``sm_87``, else int8) is the t1 spike's
verdict; the scheme is a parameter recorded in the manifest, never assumed. A build whose
weight files total 10^9 bytes or more is still written and verified here;
:mod:`jev_factory.distill.quant` records it unavailable.

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
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - the pinned student python, argv built here
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jev_factory.backbones.pointer import layout as player
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.factory import gpu, safepath
from jev_factory.factory.detach import start_detached
from jev_factory.runtime.envs import EnvSpec

MODULE = "jev_factory.backbones.pointer.quantized"
ARTIFACT_FORMAT = "jev-pointer-torchao-wo/2"
MANIFEST_FILE = "jev_quant.json"
TORSO_FILE = "torso_quantized.pt"
LOOPBACK = "127.0.0.1"
HEAD_DTYPE = "float32"
TORCHAO_QUANTIZATION = "torchao.quantization"
#: bits -> the deployed-format name :mod:`jev_factory.distill.quant` registers.
FORMAT_OF_BITS: dict[int, str] = {8: "torch_int8_wo", 4: "torch_int4_wo"}
#: bits -> the torchao API names tried in order (config class, then the pre-0.10 function).
TORCHAO_APIS: dict[int, tuple[str, str]] = {
    8: ("Int8WeightOnlyConfig", "int8_weight_only"),
    4: ("Int4WeightOnlyConfig", "int4_weight_only"),
}
#: The int4 layout asked of torchao configs that take one (tinygemm: torch's own int4 mm).
INT4_PACKING_FORMAT = "tile_packed_to_4d"
#: bits -> the pruned-vocabulary comparison build's format name (c38: never shipped).
PRUNED_FORMAT_OF_BITS: dict[int, str] = {b: f"{f}_pruned_vocab" for b, f in FORMAT_OF_BITS.items()}
#: The pruned build's kept token ids (canonical JSON: sorted, no spaces).
KEPT_IDS_FILE = "kept_token_ids.json"
#: The sides whose prompts may shape the kept vocabulary (test and held-out never do).
PROMPT_SIDES = ("train", "fit", "selection")
#: The error a pruned build answers for a token outside its kept ids (and its HTTP status).
OOV_ERROR = "out_of_vocabulary"
OOV_STATUS = 422
#: The files the manifest always hashes (the tokenizer files present are hashed too).
HASHED_FILES = (TORSO_FILE, player.HEAD, player.CONFIG)
REQUIRED_FILES = (MANIFEST_FILE, *HASHED_FILES, *player.TOKENIZER_FILES)
#: The files whose bytes count as the artifact's weights (the size budget's numerator).
WEIGHT_FILES = (TORSO_FILE, player.HEAD)
#: '< 1 GB' means < 10^9 bytes of weight files on disk (spec decision; plan risk r3).
SIZE_BUDGET_BYTES = 10**9
#: Tokenizer files carried when the source checkpoint has them.
TOKENIZER_EXTRA = (
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
TOKENIZER_ALL = (*player.TOKENIZER_FILES, *TOKENIZER_EXTRA)
#: The source checkpoint files whose sha256 the manifest records (plus its tokenizer files).
SOURCE_FILES = (player.ADAPTER_CONFIG, player.ADAPTER, player.HEAD, player.CONFIG)

# The tensor policy (spec #20, quantize requirement). A module's class comes from its
# dotted name, so the build's quantize filter and the saved-state check share one rule.
DROP, EMBEDDING, BLOCK_LINEAR, KEEP = "drop", "embedding", "block_linear", "keep"
#: Path segments of the modules removed before anything is saved (vision tower, MTP head).
DROP_SEGMENTS = ("visual", "vision_tower", "vision_model", "mtp")
#: The token embedding's module name (kept full-vocabulary, packed int4 / int8).
EMBEDDING_MODULES = ("embed_tokens",)
#: The text-block linears quantized weight-only (attention, DeltaNet projections, MLP).
QUANTIZED_LINEARS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
#: The path segment the text blocks live under.
BLOCK_SEGMENT = "layers"
#: Names that must stay unquantized floating tensors ("norm" matches every norm).
KEPT_NAMES = ("in_proj_a", "in_proj_b", "conv1d", "A_log", "dt_bias", "norm")
#: The packed embedding's two buffers.
EMBED_QWEIGHT, EMBED_SCALES = "qweight", "scales"
#: bits -> the packed embedding's storage dtype.
EMBED_QDTYPE = {4: "torch.uint8", 8: "torch.int8"}
PLAIN_TENSOR_TYPES = frozenset({"Tensor", "Parameter"})
FLOAT_DTYPES = frozenset({"torch.float32", "torch.bfloat16", "torch.float16"})
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


def _bits(bits: Any, what: str = "bits") -> int:
    if isinstance(bits, bool) or bits not in FORMAT_OF_BITS:
        raise _refuse(f"{what} must be 8 or 4 (weight-only int8 / int4), got {bits!r}")
    return int(bits)


def _group(size: Any, what: str) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise _refuse(f"{what} must be a positive integer, got {size!r}")
    return size


@dataclass(frozen=True)
class QuantScheme:
    """What the build quantizes, and how. Every field is recorded in the manifest.

    The defaults are the smallest-error choice estimated to fit the 10^9-byte budget:
    int4 blocks with one scale and zero per 256 weights and an int4 embedding with one
    scale per 128 values. That is an estimate; the size check is on the built files.
    ``block_bits=8`` is the fallback when no int4 kernel runs on the target (t1's spike).
    """

    block_bits: int = 4
    embedding_bits: int = 4
    block_group_size: int = 256
    embedding_group_size: int = 128

    def validated(self) -> "QuantScheme":
        _bits(self.block_bits, "block_bits")
        _bits(self.embedding_bits, "embedding_bits")
        _group(self.block_group_size, "block_group_size")
        _group(self.embedding_group_size, "embedding_group_size")
        return self

    def record(self) -> dict[str, int]:
        return asdict(self)


def as_scheme(scheme: "QuantScheme | int") -> QuantScheme:
    """A validated :class:`QuantScheme`; a bare int is the block bits with default rest."""
    if isinstance(scheme, QuantScheme):
        return scheme.validated()
    return QuantScheme(block_bits=_bits(scheme)).validated()


def _scheme_from(doc: Any) -> QuantScheme:
    if not isinstance(doc, dict) or set(doc) != set(QuantScheme.__dataclass_fields__):
        raise _refuse(f"{MANIFEST_FILE}: scheme must name {sorted(QuantScheme().record())}")
    return QuantScheme(**doc).validated()


# --------------------------------------------------------------------- the tensor policy


def module_class(fqn: str) -> str:
    """:data:`DROP`, :data:`EMBEDDING`, :data:`BLOCK_LINEAR` or :data:`KEEP` for a module name."""
    parts = fqn.split(".")
    if any(part in DROP_SEGMENTS or part.startswith("mtp") for part in parts):
        return DROP
    if parts[-1] in EMBEDDING_MODULES:
        return EMBEDDING
    if BLOCK_SEGMENT in parts and parts[-1] in QUANTIZED_LINEARS:
        return BLOCK_LINEAR
    return KEEP


def key_class(key: str) -> str:
    """The class of a state-dict key (its owning module's class)."""
    return module_class(key.rpartition(".")[0])


def tensor_policy() -> dict[str, Any]:
    """The policy as recorded in the manifest."""
    return {
        "drop_segments": list(DROP_SEGMENTS),
        "embedding_modules": list(EMBEDDING_MODULES),
        "quantized_linears": list(QUANTIZED_LINEARS),
        "block_segment": BLOCK_SEGMENT,
        "kept_unquantized": list(KEPT_NAMES),
        "head": f"{HEAD_DTYPE}, its own file ({player.HEAD})",
    }


def _kept_name(key: str) -> str | None:
    return next((n for n in KEPT_NAMES if any(n in part for part in key.split("."))), None)


def _embedding_row_problem(key: str, dtype: str, scheme: QuantScheme) -> str | None:
    leaf = key.rpartition(".")[2]
    if leaf not in (EMBED_QWEIGHT, EMBED_SCALES):
        return f"{key}: the embedding is not packed ({EMBED_QWEIGHT} + {EMBED_SCALES})"
    want = EMBED_QDTYPE[scheme.embedding_bits] if leaf == EMBED_QWEIGHT else None
    return f"{key} is {dtype}, not {want}" if want and dtype != want else None


def _row_problem(row: Mapping[str, Any], cls: str, scheme: QuantScheme) -> str | None:
    key, ttype, dtype = row["key"], str(row["tensor_type"]), str(row["dtype"])
    plain = ttype in PLAIN_TENSOR_TYPES
    if cls == DROP:
        return f"{key} belongs to the vision tower or the MTP head"
    if cls == BLOCK_LINEAR:
        return (
            f"{key} is a text-block linear left unquantized"
            if (key.endswith(".weight") and plain)
            else None
        )
    if cls == EMBEDDING:
        return _embedding_row_problem(key, dtype, scheme)
    if not plain:
        return f"{key} is quantized ({ttype}) outside the text-block linears"
    if _kept_name(key) and dtype not in FLOAT_DTYPES:
        return f"{key} must stay an unquantized floating tensor (it is {dtype})"
    return None


def check_inventory(rows: Sequence[Mapping[str, Any]], scheme: QuantScheme) -> dict[str, Any]:
    """Check a torso's saved state against the policy; return its per-class counts.

    *rows* are ``{"key", "tensor_type", "dtype"}`` per state-dict entry (the in-env build
    passes ``type(v).__name__`` and ``str(v.dtype)``). Refuses a vision or MTP tensor, an
    unquantized block linear, an embedding that is not packed at *scheme*'s bits, a
    quantized tensor outside the blocks, or a torso with no block linear or no embedding.
    """
    scheme = scheme.validated()
    counts = {DROP: 0, EMBEDDING: 0, BLOCK_LINEAR: 0, KEEP: 0}
    kept: dict[str, int] = {}
    problems = []
    for row in rows:
        cls = key_class(row["key"])
        counts[cls] += 1
        if cls == KEEP and (name := _kept_name(row["key"])):
            kept[name] = kept.get(name, 0) + 1
        if problem := _row_problem(row, cls, scheme):
            problems.append(problem)
    if not counts[BLOCK_LINEAR]:
        problems.append("no text-block linear was found")
    if counts[EMBEDDING] != 2:
        problems.append(f"the packed embedding must be exactly {EMBED_QWEIGHT} + {EMBED_SCALES}")
    if problems:
        raise _refuse(
            f"the quantized torso breaks the tensor policy: {'; '.join(problems[:8])}",
            "fix the build; the policy is jev_factory.backbones.pointer.quantized.tensor_policy",
        )
    return {"tensors": counts, "kept_unquantized": dict(sorted(kept.items()))}


class OutOfVocabulary(Exception):
    """A token id the pruned embedding has no row for; raised before any row is read."""

    def __init__(self, token_ids: Sequence[int]) -> None:
        self.token_ids = sorted({int(i) for i in token_ids})
        super().__init__(
            f"{OOV_ERROR}: token id(s) {self.token_ids[:16]} are outside the pruned "
            "vocabulary's kept ids"
        )

    def detail(self) -> dict[str, Any]:
        return {"error": OOV_ERROR, "token_ids": self.token_ids, "message": str(self)}


def kept_ids_bytes(kept_ids: Sequence[int]) -> bytes:
    """The canonical encoding of a kept id list (what :data:`KEPT_IDS_FILE` holds)."""
    return json.dumps(list(kept_ids), separators=(",", ":")).encode("utf-8")


def kept_ids_sha256(kept_ids: Sequence[int]) -> str:
    return hashlib.sha256(kept_ids_bytes(kept_ids)).hexdigest()


def _id_list(ids: Any, what: str) -> list[int]:
    if not isinstance(ids, (list, tuple)) or not ids:
        raise _refuse(f"{what} must be a non-empty list of token ids")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise _refuse(f"{what} must hold non-negative integer token ids")
    if list(ids) != sorted(set(ids)):
        raise _refuse(f"{what} must be sorted with no repeats")
    return list(ids)


def kept_vocabulary(
    prompt_ids: Iterable[int], special_ids: Iterable[int], vocab_rows: int
) -> list[int]:
    """The sorted ids a pruned embedding keeps: every prompt id plus every special id.

    Refuses an id outside ``[0, vocab_rows)`` (it has no row to keep) and an empty prompt
    set (a pruned build over nothing is not a measurement).
    """
    used = {int(i) for i in prompt_ids}
    if not used:
        raise _refuse("no prompt token ids were recorded; the kept vocabulary would be empty")
    kept = sorted(used | {int(i) for i in special_ids})
    if bad := [i for i in kept if not 0 <= i < vocab_rows]:
        raise _refuse(f"token id(s) {bad[:8]} are outside the embedding's {vocab_rows} rows")
    return kept


def special_token_ids(tokenizer: Any) -> list[int]:
    """The tokenizer's special, added, pad, eos, bos and unk ids (kept by every pruned build)."""
    ids = set(getattr(tokenizer, "all_special_ids", None) or ())
    ids.update(getattr(tokenizer, "added_tokens_decoder", None) or {})
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            ids.add(value)
    return sorted(int(i) for i in ids)


def _prompt_row(text: str, where: str) -> dict[str, Any]:
    try:
        row = json.loads(text)
    except ValueError as exc:
        raise _refuse(f"{where} is not valid JSON: {exc}") from exc
    side = row.get("side") if isinstance(row, dict) else None
    if side not in PROMPT_SIDES:
        raise _refuse(
            f"{where}: side {side!r} may not shape the pruned vocabulary",
            f"a prompt set holds {', '.join(PROMPT_SIDES)} prompts only",
        )
    body = row.get("body")
    if not (isinstance(body, dict) and isinstance(body.get("questions"), dict)):
        raise _refuse(f"{where}: body must be a /v1/systemone request")
    return row


def read_prompt_set(path: Path | str) -> list[dict[str, Any]]:
    """The prompt set's rows (``{"side", "body"}``); a side outside :data:`PROMPT_SIDES` fails.

    The bodies are ``/v1/systemone`` requests; only train, fit and selection prompts may
    shape the pruned vocabulary, so a test or held-out row refuses the whole file.
    """
    path = Path(path)
    if not path.is_file():
        raise _refuse(f"the prompt set {path} does not exist")
    path = safepath.regular_file(path, path.parent)  # a regular file, never a symlink
    rows = []
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        rows.append(_prompt_row(text, f"{path.name} row {line}"))
    if not rows:
        raise _refuse(f"the prompt set {path} holds no prompts")
    return rows


def prompt_set_record(path: Path | str) -> dict[str, Any]:
    """The prompt set's path, sha256 and per-side counts (validated)."""
    rows = read_prompt_set(path)
    counts = {side: sum(1 for r in rows if r["side"] == side) for side in PROMPT_SIDES}
    return {"path": str(path), "sha256": player.sha256_file(Path(path)), "counts": counts}


def check_pruned_vocabulary(
    kept_ids: Sequence[int], tokenizer_size: int, embedding: Mapping[str, Any]
) -> dict[str, Any]:
    """The pruned embedding keeps exactly *kept_ids*, each a real tokenizer id."""
    kept = _id_list(list(kept_ids), "the kept ids")
    if kept[-1] >= tokenizer_size:
        raise _refuse(f"kept id {kept[-1]} is not a tokenizer id (size {tokenizer_size})")
    if embedding.get("rows") != len(kept):
        raise _refuse(
            f"the pruned embedding holds {embedding.get('rows')} rows for {len(kept)} kept ids"
        )
    return {
        "full_vocabulary": False,
        "kept_rows": len(kept),
        "tokenizer_size": tokenizer_size,
        "embedding_vocab_rows": embedding.get("vocab_rows"),
        "kept_ids_sha256": kept_ids_sha256(kept),
    }


def check_full_vocabulary(embedding_rows: int, tokenizer_size: int) -> dict[str, Any]:
    """Every tokenizer id has an embedding row (no id is unreachable)."""
    if not (isinstance(tokenizer_size, int) and 0 < tokenizer_size <= embedding_rows):
        raise _refuse(
            f"the tokenizer has {tokenizer_size} ids but the embedding keeps {embedding_rows} "
            "rows; a full-vocabulary build keeps a row for every id"
        )
    return {
        "embedding_rows": embedding_rows,
        "tokenizer_size": tokenizer_size,
        "full_vocabulary": True,
    }


# --------------------------------------------------------------------- the artifact


def source_record(checkpoint: Path | str) -> dict[str, Any]:
    """The source checkpoint's path and the sha256 of its adapter, head, config, tokenizer."""
    root = Path(checkpoint)
    return {
        "checkpoint": str(root),
        "sha256": {
            n: player.sha256_file(player.member(root, n))
            for n in (*SOURCE_FILES, *TOKENIZER_ALL)
            if (root / n).is_file()
        },
    }


def _tokenizer_record(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    names = [n for n in TOKENIZER_ALL if (root / n).is_file()]
    from_source = (source.get("sha256") or {}) if isinstance(source, Mapping) else {}
    files = {n: player.sha256_file(player.member(root, n)) for n in names}
    return {
        "files": names,
        "source_sha256": {n: from_source[n] for n in names if n in from_source},
        "unmodified": bool(names) and all(from_source.get(n) == files[n] for n in names),
    }


def weight_record(root: Path | str) -> dict[str, Any]:
    """The byte size of each weight file in *root*, their sum and the budget verdict."""
    root = Path(root)
    sizes = {n: player.member(root, n).stat().st_size for n in WEIGHT_FILES}
    total = sum(sizes.values())
    return {
        "weight_files": sizes,
        "weight_bytes": total,
        "size_budget_bytes": SIZE_BUDGET_BYTES,
        "within_size_budget": total < SIZE_BUDGET_BYTES,
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
    scheme: QuantScheme | None = None,
    extra: Mapping[str, Any] | None = None,
    kept_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write :data:`MANIFEST_FILE` over the files already in *root*; return it.

    *bits* is the block bits (the deployed-format name follows it); *scheme* defaults to
    :class:`QuantScheme` at those bits. Every file is hashed and the weight files' byte sum
    recorded, so the manifest is what ``GET /v1/bundle`` attests. *kept_ids* marks a
    pruned-vocabulary comparison build (c38): :data:`KEPT_IDS_FILE` is written and hashed,
    ``pruned_vocabulary`` records its sha256, and the format is
    :data:`PRUNED_FORMAT_OF_BITS`'s.
    """
    root = Path(root)
    bits = _bits(bits)
    scheme = as_scheme(scheme if scheme is not None else bits)
    if scheme.block_bits != bits:
        raise _refuse(f"bits {bits} disagrees with the scheme's block_bits {scheme.block_bits}")
    pruned = None
    if kept_ids is not None:
        kept = _id_list(list(kept_ids), "kept_ids")
        safepath.contained(root / KEPT_IDS_FILE, root).write_bytes(kept_ids_bytes(kept))
        pruned = {
            "kept_ids_file": KEPT_IDS_FILE,
            "kept_ids_sha256": kept_ids_sha256(kept),
            "kept_ids_count": len(kept),
            "comparison_only": True,
            "decision": "c38: a comparison build, reported only; never the shipped format",
        }
    tokenizer = _tokenizer_record(root, source)
    hashed = (*HASHED_FILES, *tokenizer["files"], *((KEPT_IDS_FILE,) if pruned else ()))
    manifest = {
        **dict(extra or {}),
        "artifact_format": ARTIFACT_FORMAT,
        "format": format_name(bits, pruned=pruned is not None),
        "bits": bits,
        "scheme": scheme.record(),
        "torchao_api": torchao_api,
        "torchao_version": torchao_version,
        "torch_version": torch_version,
        "head_dtype": HEAD_DTYPE,
        "lora_merged": True,
        "source": dict(source),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "files": {name: player.sha256_file(player.member(root, name)) for name in hashed},
        "tokenizer": tokenizer,
        **weight_record(root),
        "upstream": {k: UPSTREAM_PROVENANCE[k] for k in ("repo", "commit", "licence")},
    }
    if pruned:
        manifest["pruned_vocabulary"] = pruned
    target = safepath.contained(root / MANIFEST_FILE, root)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def format_name(bits: int, *, pruned: bool = False) -> str:
    """The format name of a build at *bits* (the pruned comparison has its own names)."""
    return (PRUNED_FORMAT_OF_BITS if pruned else FORMAT_OF_BITS)[_bits(bits)]


def is_pruned(manifest: Mapping[str, Any]) -> bool:
    """Whether a manifest describes a pruned-vocabulary comparison build."""
    return "pruned_vocabulary" in manifest


def _nonempty_str(manifest: Mapping[str, Any], key: str) -> None:
    value = manifest.get(key)
    if not (isinstance(value, str) and value.strip()):
        raise _refuse(f"{MANIFEST_FILE}: {key} must name what built the artifact, got {value!r}")


def _check_format(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise _refuse(
            f"{MANIFEST_FILE}: artifact_format is {manifest.get('artifact_format')!r}, "
            f"not {ARTIFACT_FORMAT!r}"
        )
    bits = manifest.get("bits")
    if isinstance(bits, bool) or bits not in FORMAT_OF_BITS:
        raise _refuse(f"{MANIFEST_FILE}: bits must be 8 or 4, got {bits!r}")
    want = format_name(bits, pruned=is_pruned(manifest))
    if manifest.get("format") != want:
        raise _refuse(
            f"{MANIFEST_FILE}: format {manifest.get('format')!r} does not match "
            f"{bits} bits ({want})"
        )
    if _scheme_from(manifest.get("scheme")).block_bits != bits:
        raise _refuse(f"{MANIFEST_FILE}: scheme.block_bits disagrees with bits {bits}")


def _check_manifest(manifest: Mapping[str, Any]) -> None:
    _check_format(manifest)
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


def _check_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    if missing := [n for n in (*HASHED_FILES, *player.TOKENIZER_FILES) if n not in files]:
        raise _refuse(f"{MANIFEST_FILE}: files must hash {', '.join(missing)}")
    for name, recorded in files.items():
        actual = player.sha256_file(player.member(root, name))
        if str(recorded).lower() != actual:
            raise _refuse(
                f"{MANIFEST_FILE}: the sha256 recorded for {name} does not match the file "
                f"(sha256 {actual})",
                "rebuild the artifact; never edit a built artifact by hand",
            )


def _check_weights(root: Path, manifest: Mapping[str, Any]) -> None:
    actual = weight_record(root)
    for key in ("weight_files", "weight_bytes"):
        if manifest.get(key) != actual[key]:
            raise _refuse(
                f"{MANIFEST_FILE}: {key} {manifest.get(key)!r} does not match the files "
                f"({actual[key]!r})",
                "rebuild the artifact; the weight byte sum is computed from its files",
            )


def _check_tokenizer(manifest: Mapping[str, Any]) -> None:
    """A tokenizer file whose bytes differ from the source checkpoint's is refused."""
    record = manifest.get("tokenizer")
    if not isinstance(record, dict):
        raise _refuse(f"{MANIFEST_FILE}: tokenizer must record the tokenizer files")
    files = manifest["files"]
    for name, digest in (record.get("source_sha256") or {}).items():
        if files.get(name) != digest:
            raise _refuse(
                f"{MANIFEST_FILE}: {name} differs from the source checkpoint's tokenizer",
                "the artifact ships the source tokenizer unmodified; rebuild it",
            )


def read_kept_ids(root: Path | str) -> list[int]:
    """The pruned build's kept ids from :data:`KEPT_IDS_FILE` (validated)."""
    try:
        ids = json.loads(player.member(Path(root), KEPT_IDS_FILE).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _refuse(f"{KEPT_IDS_FILE} is not valid JSON: {exc}") from exc
    return _id_list(ids, KEPT_IDS_FILE)


def _check_pruned(root: Path, manifest: Mapping[str, Any]) -> None:
    """A pruned build's kept id list is hashed, matches its recorded sha256 and count."""
    if not is_pruned(manifest):
        return
    record = manifest["pruned_vocabulary"]
    if not isinstance(record, dict) or record.get("kept_ids_file") != KEPT_IDS_FILE:
        raise _refuse(f"{MANIFEST_FILE}: pruned_vocabulary must name {KEPT_IDS_FILE}")
    if record.get("comparison_only") is not True:
        raise _refuse(f"{MANIFEST_FILE}: a pruned-vocabulary build is a comparison only (c38)")
    if manifest["files"].get(KEPT_IDS_FILE) != record.get("kept_ids_sha256"):
        raise _refuse(f"{MANIFEST_FILE}: the kept id list is not hashed as recorded")
    kept = read_kept_ids(root)
    if kept_ids_sha256(kept) != record["kept_ids_sha256"] or len(kept) != record.get(
        "kept_ids_count"
    ):
        raise _refuse(
            f"{KEPT_IDS_FILE} is not the canonical kept id list its manifest records",
            "rebuild the artifact; never edit a built artifact by hand",
        )


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
    paths = {name: player.member(root, name) for name in REQUIRED_FILES}
    manifest = _json_object(paths[MANIFEST_FILE])
    _check_manifest(manifest)
    _check_hashes(root, manifest)
    _check_weights(root, manifest)
    _check_tokenizer(manifest)
    _check_pruned(root, manifest)
    config = _json_object(paths[player.CONFIG])
    if config.get("head_type") != "pointer":
        raise _refuse(
            f"{player.CONFIG} has head_type {config.get('head_type')!r}; this serves pointer "
            "heads only"
        )
    return {"root": str(root), "manifest": manifest, "config": config}


def file_snapshot(path: Path | str) -> dict[str, tuple[int, int, int]]:
    """(size, ctime_ns, inode) of every file of the artifact: cheap, no hashing; a write,
    rename or replacement changes ctime, which (unlike a stored timestamp) cannot be set back."""
    root = Path(path)
    return {
        f.name: (st.st_size, st.st_ctime_ns, st.st_ino)
        for f in sorted(root.iterdir())
        if f.is_file() and (st := f.stat())
    }


def bundle_attestation(path: Path | str, info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What ``GET /v1/bundle`` answers: the verified manifest's per-file sha256 and sizes.

    *info* is a :func:`read_artifact` result for *path* already in hand (the server's own
    load), so the weights are not hashed a second time."""
    info = read_artifact(path) if info is None else info
    manifest = info["manifest"]
    keys = ("artifact_format", "format", "bits", "scheme", "files", "base_model_revision")
    return {
        **{k: manifest[k] for k in keys},
        **{k: manifest[k] for k in weight_record(path)},
        **({"pruned_vocabulary": manifest["pruned_vocabulary"]} if is_pruned(manifest) else {}),
        "manifest_sha256": player.sha256_file(player.member(Path(path), MANIFEST_FILE)),
    }


# --------------------------------------------------------------------- the build (plan / run)


def plan_weight_only(
    checkpoint_dir: Path | str,
    out_dir: Path | str,
    scheme: QuantScheme | int,
    *,
    python: str,
    decider_repo: Path | str | None = None,
    base_revision: str | None = None,
    device: str = "cuda",
    prompts: Path | str | None = None,
) -> dict[str, Any]:
    """The in-env build command for *checkpoint_dir*; validates the inputs, runs nothing.

    *scheme* is a :class:`QuantScheme` (a bare int is the block bits, the rest default).
    *prompts* (a prompt set, :func:`read_prompt_set`) plans the pruned-vocabulary
    comparison build instead of the full-vocabulary one.
    """
    scheme = as_scheme(scheme)
    bits = scheme.block_bits
    src, out = Path(checkpoint_dir), Path(out_dir)
    player.plan_export(src, out)  # refuses an incomplete or non-pointer checkpoint
    revision = player._base_revision(src, base_revision)
    prompt_set = prompt_set_record(prompts) if prompts is not None else None
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
        "--embedding-bits",
        str(scheme.embedding_bits),
        "--block-group-size",
        str(scheme.block_group_size),
        "--embedding-group-size",
        str(scheme.embedding_group_size),
        "--base-revision",
        revision,
        "--device",
        device,
    ]
    if prompt_set:
        argv += ["--prompts", prompt_set["path"]]
    pruned = {"comparison_only": True, "prompt_set": prompt_set} if prompt_set else {}
    return {
        "backbone": "pointer",
        "stage": "quantize",
        "format": format_name(bits, pruned=bool(prompt_set)),
        **pruned,
        "bits": bits,
        "scheme": scheme.record(),
        "tensor_policy": tensor_policy(),
        "size_budget_bytes": SIZE_BUDGET_BYTES,
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
    scheme: QuantScheme | int,
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
        scheme,
        python=python,
        decider_repo=decider_repo,
        base_revision=base_revision,
        device=device,
    )
    if not apply:
        return plan
    limits = {"timeout": timeout, "memory_max": memory_max, "memory_floor": memory_floor}
    return _run_build(plan, runner=runner, allow_foreign=allow_foreign, **limits)


def build_pruned_vocabulary(
    checkpoint_dir: Path | str,
    out_dir: Path | str,
    scheme: QuantScheme | int,
    prompts: Path | str,
    *,
    apply: bool = False,
    python: str = "python",
    decider_repo: Path | str | None = None,
    base_revision: str | None = None,
    device: str = "cuda",
    **limits: Any,
) -> dict[str, Any]:
    """The pruned-vocabulary comparison build (c38): as :func:`build_weight_only`, but the
    embedding keeps only the ids *prompts* (train / fit / selection) look up, plus the
    special tokens. *limits* are :func:`build_weight_only`'s ``runner``, ``timeout``,
    ``memory_max``, ``memory_floor`` and ``allow_foreign``. Dry-run by default."""
    plan = plan_weight_only(
        checkpoint_dir,
        out_dir,
        scheme,
        python=python,
        decider_repo=decider_repo,
        base_revision=base_revision,
        device=device,
        prompts=prompts,
    )
    if not apply:
        return plan
    info = _run_build(plan, **limits)
    if not is_pruned(info["artifact"]["manifest"]):
        raise CliError(
            EXIT_ENV_ERROR, "the pruned-vocabulary build left a full-vocabulary artifact"
        )
    return info


def _run_build(
    plan: Mapping[str, Any],
    *,
    runner: Runner | None = None,
    timeout: float | None = None,
    memory_max: str | None = None,
    memory_floor: str = "8G",
    allow_foreign: bool = False,
) -> dict[str, Any]:
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
    record_path = safepath.contained(jobdir / f"{name}.serve.json", jobdir)
    jobdir.mkdir(parents=True, exist_ok=True)
    record = {
        "argv": argv,
        "environment": env.record(),
        "artifact": {
            "format": manifest["format"],
            "files": manifest["files"],
            "weight_bytes": manifest["weight_bytes"],
            "manifest_sha256": player.sha256_file(player.member(Path(artifact), MANIFEST_FILE)),
        },
        "capped": False,
        "capped_note": UNCAPPED_NOTE,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return start_detached(jobdir, name, argv)


# --------------------------------------------------------------------- /v1/stats (host facts)


@dataclass(frozen=True)
class HostPaths:
    """Where :func:`host_stats` reads; tests point it at a fake tree."""

    proc: Path = Path("/proc")
    cgroup_root: Path = Path("/sys/fs/cgroup")
    tegra_release: Path = Path("/etc/nv_tegra_release")
    #: Board model sources, in order (a Jetson's device tree, then DMI).
    device_models: tuple[Path, ...] = (
        Path("/proc/device-tree/model"),
        Path("/sys/devices/virtual/dmi/id/product_name"),
    )


#: A cgroup v1 memory.limit_in_bytes at or above this is "no limit".
_V1_UNLIMITED = 1 << 62


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _kib_field(path: Path, field: str) -> int | None:
    """``<field>: <n> kB`` from a /proc status or meminfo file, in bytes."""
    for line in (_read_text(path) or "").splitlines():
        name, _, rest = line.partition(":")
        if name.strip() == field and rest.split() and rest.split()[0].isdigit():
            return int(rest.split()[0]) * 1024
    return None


def _limit(text: str | None, unlimited_at: int | None = None) -> int | str | None:
    value = (text or "").strip()
    if not value:
        return None
    if value == "max" or (value.isdigit() and unlimited_at and int(value) >= unlimited_at):
        return "max"
    return int(value) if value.isdigit() else None


def _cgroup_line(proc_cgroup: str | None) -> tuple[int, str] | None:
    """``(2, path)`` for the unified hierarchy, else ``(1, path)`` for the v1 memory one."""
    v1 = None
    for line in (proc_cgroup or "").splitlines():
        hid, _, rest = line.partition(":")
        controllers, _, rel = rest.partition(":")
        if hid == "0" and controllers == "":
            return 2, rel
        if "memory" in controllers.split(","):
            v1 = (1, rel)
    return v1


def cgroup_memory(paths: HostPaths = HostPaths(), pid: str = "self") -> dict[str, Any]:
    """The memory cap of the cgroup *pid* runs in, and the tightest cap on its ancestors.

    ``memory_max`` is that cgroup's own limit (bytes, ``"max"`` for none, ``None`` when it
    cannot be read); ``effective_memory_max`` is the smallest byte limit from it up to the
    hierarchy root (``None`` when no ancestor is capped).
    """
    found = _cgroup_line(_read_text(paths.proc / pid / "cgroup"))
    if found is None:
        return {"version": None, "path": None, "memory_max": None, "effective_memory_max": None}
    version, rel = found
    base = paths.cgroup_root if version == 2 else paths.cgroup_root / "memory"
    name = "memory.max" if version == 2 else "memory.limit_in_bytes"
    unlimited = None if version == 2 else _V1_UNLIMITED
    parts = [p for p in rel.split("/") if p and p not in (".", "..")]
    limits = [
        _limit(_read_text(base.joinpath(*parts[:depth]) / name), unlimited)
        for depth in range(len(parts), -1, -1)
    ]
    numeric = [v for v in limits if isinstance(v, int)]
    return {
        "version": version,
        "path": "/" + "/".join(parts),
        "memory_max": limits[0],
        "effective_memory_max": min(numeric) if numeric else None,
    }


def host_stats(paths: HostPaths = HostPaths(), pid: str = "self") -> dict[str, Any]:
    """What ``GET /v1/stats`` answers: process RSS, MemAvailable, board, L4T release, cap."""
    model = next(
        (t.strip("\x00 \n") for t in map(_read_text, paths.device_models) if t and t.strip()),
        None,
    )
    release = _read_text(paths.tegra_release)
    return {
        "rss_bytes": _kib_field(paths.proc / pid / "status", "VmRSS"),
        "mem_available_bytes": _kib_field(paths.proc / "meminfo", "MemAvailable"),
        "device_model": model or None,
        "nv_tegra_release": release.strip() if release and release.strip() else None,
        "cgroup": cgroup_memory(paths, pid),
    }


class RequestError(Exception):
    """A caller error: answered with *status* and ``{"detail": detail}`` (FastAPI's shape).

    *reason* replaces the HTTP reason phrase, so a client that keeps only the status line
    (``HTTP Error 422: out_of_vocabulary``) still sees which refusal it got.
    """

    def __init__(self, status: int, detail: Any, reason: str | None = None) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail
        self.reason = reason


def refuse_out_of_vocabulary(call: Callable[[], Any]) -> Any:
    """Run *call*; an :class:`OutOfVocabulary` becomes an ``out_of_vocabulary`` 422."""
    try:
        return call()
    except OutOfVocabulary as exc:
        raise RequestError(OOV_STATUS, exc.detail(), reason=OOV_ERROR) from exc


def make_server(
    host: str,
    port: int,
    *,
    evaluate: Callable[[dict[str, Any]], Mapping[str, Any]],
    health: Callable[[], Mapping[str, Any]],
    bundle: Callable[[], Mapping[str, Any]] | None = None,
    stats: Callable[[], Mapping[str, Any]] | None = None,
) -> HTTPServer:
    """A single-threaded HTTP server (one engine call at a time) on 127.0.0.1 only.

    ``GET /health``, ``GET /v1/bundle`` (*bundle*: the artifact's attested hashes) and
    ``GET /v1/stats`` (*stats*: the host and process facts); a route without a callable
    is a 404.
    """
    _require_loopback(host)
    routes = {"/health": health, "/v1/bundle": bundle, "/v1/stats": stats}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any, reason: str | None = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status, reason)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            answer = routes.get(self.path)
            if answer is None:
                self._send(404, {"detail": "Not Found"})
                return
            try:
                payload = dict(answer())
            except Exception as exc:  # noqa: BLE001 - answered rather than a dropped connection
                self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, payload)

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
                self._send(exc.status, {"detail": exc.detail}, exc.reason)
                return
            except Exception as exc:  # noqa: BLE001 - answered rather than a dropped connection
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


def _accepts(factory: Any, name: str) -> bool:
    """Whether the torchao config class (or function) *factory* takes the keyword *name*."""
    import inspect

    try:
        return name in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _quant_config(bits: int, group_size: int | None = None) -> tuple[str, Any]:
    """The torchao weight-only config for the block linears (int4 takes *group_size*)."""
    aoq = _load(TORCHAO_QUANTIZATION)
    kwargs: dict[str, Any] = {"group_size": group_size} if bits == 4 and group_size else {}
    for api in TORCHAO_APIS[bits]:
        factory = getattr(aoq, api, None)
        if factory is not None:
            if bits == 4 and _accepts(factory, "int4_packing_format"):
                # torchao >= 0.16 defaults int4 to the "plain" layout, whose CUDA kernel needs
                # the separate mslk package; the tinygemm layout runs on torch's own int4 mm
                # (sm_80+, so the GB10 and the Orin's sm_87 alike).
                kwargs["int4_packing_format"] = INT4_PACKING_FORMAT
                return f"{api}(int4_packing_format={INT4_PACKING_FORMAT})", factory(**kwargs)
            return api, factory(**kwargs)
    raise CliError(
        EXIT_ENV_ERROR,
        f"torchao {_version('torchao')} has none of {list(TORCHAO_APIS[bits])}",
        "install a torchao with weight-only int8/int4 quantization in the student env",
    )


#: Rows quantized per step when packing the embedding (bounds the fp32 working copy).
_EMBED_CHUNK_ROWS = 8192


def _kept_id_map(torch: Any, kept: Sequence[int], rows: int, vocab: int, device: Any) -> Any:
    """A pruned build's token id -> kept row map, ``-1`` for an id with no row."""
    if len(kept) != rows:
        raise ValueError(f"{len(kept)} kept ids for {rows} rows")
    id_map = torch.full((vocab,), -1, dtype=torch.long, device=device)
    id_map[torch.tensor(kept, dtype=torch.long, device=device)] = torch.arange(rows, device=device)
    return id_map


def _packed_embedding_class(torch: Any) -> Any:
    """The packed full-vocabulary embedding module (built lazily: torch is in-env only)."""

    class PackedEmbedding(torch.nn.Module):
        """Grouped symmetric int4 (two values per byte, offset 8) or int8 rows, one scale per
        *group_size* values; a lookup gathers the rows and dequantizes only those."""

        def __init__(
            self, rows, dim, bits, group_size, scale_dtype, device=None, kept=None, vocab=None
        ):
            super().__init__()
            if dim % group_size or (bits == 4 and dim % 2):
                raise ValueError(f"embedding dim {dim} is not a multiple of {group_size}")
            self.rows, self.dim, self.bits, self.group_size = rows, dim, bits, group_size
            # A pruned build (c38) maps a token id to its kept row, -1 for none; the map is
            # rebuilt from the kept id list at load, so it is not part of the saved state.
            self.vocab = rows if kept is None else int(vocab)
            id_map = None if kept is None else _kept_id_map(torch, kept, rows, self.vocab, device)
            self.register_buffer("id_map", id_map, persistent=False)
            qdtype, width = (torch.uint8, dim // 2) if bits == 4 else (torch.int8, dim)
            self.register_buffer(
                EMBED_QWEIGHT, torch.empty(rows, width, dtype=qdtype, device=device)
            )
            self.register_buffer(
                EMBED_SCALES,
                torch.empty(rows, dim // group_size, dtype=scale_dtype, device=device),
            )

        @property
        def num_embeddings(self) -> int:
            return self.vocab

        @property
        def embedding_dim(self) -> int:
            return self.dim

        def kept_rows(self, ids):
            """The kept row of each id; any id without one refuses before a row is read."""
            flat = ids.reshape(-1)
            outside = (flat < 0) | (flat >= self.vocab)
            mapped = self.id_map[flat.clamp(0, self.vocab - 1)]
            missing = outside | (mapped < 0)
            if bool(missing.any()):
                raise OutOfVocabulary(flat[missing].tolist())
            return mapped.reshape(ids.shape)

        def forward(self, ids):
            if self.id_map is not None:
                ids = self.kept_rows(ids)
            q = getattr(self, EMBED_QWEIGHT)[ids]
            if self.bits == 4:
                low = (q & 0x0F).to(torch.int16) - 8
                high = (q >> 4).to(torch.int16) - 8
                q = torch.stack((low, high), dim=-1).flatten(-2)
            scales = getattr(self, EMBED_SCALES)[ids]
            values = q.to(scales.dtype).unflatten(-1, (scales.shape[-1], self.group_size))
            return (values * scales.unsqueeze(-1)).flatten(-2)

        def pack_(self, weight) -> None:
            """Quantize a ``(rows, dim)`` float weight into the buffers, chunk by chunk."""
            qmax = 7 if self.bits == 4 else 127
            for start in range(0, self.rows, _EMBED_CHUNK_ROWS):
                block = weight[start : start + _EMBED_CHUNK_ROWS].detach().float()
                block = block.unflatten(-1, (self.dim // self.group_size, self.group_size))
                scale = (block.abs().amax(dim=-1) / qmax).clamp(min=1e-12)
                scale = scale.to(getattr(self, EMBED_SCALES).dtype)  # the stored scale is used
                q = torch.round(block / scale.float().unsqueeze(-1)).clamp(-qmax - 1, qmax)
                q = q.flatten(-2).to(torch.int16)
                if self.bits == 4:
                    q = (q + 8).to(torch.uint8)
                    q = q[..., 0::2] | (q[..., 1::2] << 4)
                else:
                    q = q.to(torch.int8)
                end = start + q.shape[0]
                getattr(self, EMBED_QWEIGHT)[start:end] = q
                getattr(self, EMBED_SCALES)[start:end] = scale

    return PackedEmbedding


def _set_module(root: Any, fqn: str, module: Any) -> None:
    parent_name, _, leaf = fqn.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, module)


def drop_unused(torso: Any) -> list[dict[str, Any]]:
    """Remove the vision tower and MTP head modules (:data:`DROP`); return what went."""
    dropped: list[dict[str, Any]] = []
    modules = list(torso.named_modules())  # a snapshot: the loop replaces modules
    for fqn, module in modules:
        if not fqn or module_class(fqn) != DROP:
            continue
        if any(fqn.startswith(d["module"] + ".") for d in dropped):
            continue
        params = sum(p.numel() for p in module.parameters())
        _set_module(torso, fqn, None)
        dropped.append({"module": fqn, "params": int(params)})
    return dropped


def _embedding(torso: Any) -> tuple[str, Any]:
    found = [(f, m) for f, m in torso.named_modules() if f and module_class(f) == EMBEDDING]
    if len(found) != 1:
        raise CliError(
            EXIT_ENV_ERROR,
            f"expected one token embedding named {EMBEDDING_MODULES}, found "
            f"{[f for f, _ in found]}",
        )
    return found[0]


def install_packed_embedding(
    torso: Any, record: Mapping[str, Any], kept_ids: Sequence[int] | None = None
) -> Any:
    """Replace the torso's embedding with an empty packed one shaped by *record* (serving).

    *kept_ids* (a pruned build) installs the id-to-row map that refuses any other id.
    """
    torch = _load("torch")
    fqn, current = _embedding(torso)
    weight = getattr(current, "weight", None)
    scale_dtype = getattr(torch, str(record["scale_dtype"]).replace("torch.", ""))
    packed = _packed_embedding_class(torch)(
        int(record["rows"]),
        int(record["dim"]),
        int(record["bits"]),
        int(record["group_size"]),
        scale_dtype,
        device=weight.device if weight is not None else None,
        kept=None if kept_ids is None else list(kept_ids),
        vocab=record.get("vocab_rows"),
    )
    _set_module(torso, fqn, packed)
    return packed


def pack_embedding(
    torso: Any, scheme: QuantScheme, kept_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    """Replace the torso's embedding with a packed copy of every row; return its record.

    *kept_ids* (the pruned comparison build, c38) packs only those rows, in that order.
    """
    torch = _load("torch")
    fqn, current = _embedding(torso)
    weight = current.weight
    vocab_rows, dim = (int(x) for x in weight.shape)
    kept = None if kept_ids is None else _id_list(list(kept_ids), "kept_ids")
    if kept is not None and kept[-1] >= vocab_rows:
        raise _refuse(f"kept id {kept[-1]} is outside the embedding's {vocab_rows} rows")
    rows = vocab_rows if kept is None else len(kept)
    packed = _packed_embedding_class(torch)(
        rows,
        dim,
        scheme.embedding_bits,
        scheme.embedding_group_size,
        weight.dtype,
        device=weight.device,
        kept=kept,
        vocab=vocab_rows,
    )
    with torch.no_grad():
        if kept is None:
            packed.pack_(weight)
        else:
            packed.pack_(weight[torch.tensor(kept, dtype=torch.long, device=weight.device)])
    _set_module(torso, fqn, packed)
    return {
        "module": fqn,
        "rows": rows,
        "vocab_rows": vocab_rows,
        "pruned": kept is not None,
        "dim": dim,
        "bits": scheme.embedding_bits,
        "group_size": scheme.embedding_group_size,
        "scale_dtype": str(weight.dtype),
        "packing": "symmetric per group; int4 two per byte (low nibble first, offset 8), "
        "or int8; value = q * scale",
    }


def quantize_torso(
    torso: Any, scheme: QuantScheme | int, kept_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    """Apply *scheme* to *torso* in place: drop, pack the embedding, quantize block linears.

    Returns the record the manifest keeps (``torchao_api``, ``dropped``, ``embedding``).
    *kept_ids* packs only those embedding rows (the pruned comparison build).
    """
    scheme = as_scheme(scheme)
    torch = _load("torch")
    dropped = drop_unused(torso)
    embedding = pack_embedding(torso, scheme, kept_ids)
    api, config = _quant_config(scheme.block_bits, scheme.block_group_size)

    def blocks(module: Any, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear) and module_class(fqn) == BLOCK_LINEAR

    with torch.no_grad():
        _load(TORCHAO_QUANTIZATION).quantize_(torso, config, filter_fn=blocks)
    return {"torchao_api": api, "config": repr(config), "dropped": dropped, "embedding": embedding}


def state_inventory(state: Mapping[str, Any]) -> list[dict[str, str]]:
    """``{"key", "tensor_type", "dtype"}`` per state-dict entry, for :func:`check_inventory`."""
    return [
        {"key": k, "tensor_type": type(v).__name__, "dtype": str(getattr(v, "dtype", ""))}
        for k, v in state.items()
    ]


def _torso_shell(config: Any, revision: str) -> Any:
    """The base's text torso with its parameters on the meta device (risk r9), or ``None``.

    Built from the base's config alone (its text config's causal-LM class), so the bf16
    weights are never loaded; the saved quantized state is then assigned over it and the
    server peaks near the int4 artifact's size, not the bf16 model's. Buffers (rotary
    tables) are built for real. ``None`` when the config maps to no causal-LM class."""
    transformers = _load("transformers")
    torch = _load("torch")
    accelerate = _load("accelerate")
    base_cfg = transformers.AutoConfig.from_pretrained(config.base_model, revision=revision)
    try:
        with accelerate.init_empty_weights(include_buffers=False):
            lm = transformers.AutoModelForCausalLM.from_config(
                base_cfg.get_text_config(), dtype=getattr(torch, config.torch_dtype)
            )
    except (ValueError, KeyError):
        return None
    torso = getattr(lm, "model", None)
    if torso is not None:
        torso.config.use_cache = True
    return torso


def _filled_torso(
    torso: Any, modeling: Any, manifest: Mapping[str, Any], artifact: Path, device: str
) -> Any:
    """*torso* made servable: attention, unused parts dropped, packed embedding, int4 state."""
    attn = modeling.serving_attn_implementation(torso.config)
    if attn:
        torso.set_attn_implementation(attn)
    drop_unused(torso)
    kept = read_kept_ids(artifact) if is_pruned(manifest) else None
    if kept is not None and manifest["embedding"].get("rows") != len(kept):
        raise _refuse(f"{MANIFEST_FILE}: the embedding rows disagree with {KEPT_IDS_FILE}")
    install_packed_embedding(torso, manifest["embedding"], kept)
    load_quantized_state(torso, player.member(artifact, TORSO_FILE), device=device)
    return torso


def _meta_parameters(torso: Any) -> list[str]:
    return [name for name, p in torso.named_parameters() if p.is_meta]


def load_quantized_state(module: Any, path: Path | str, *, device: str) -> None:
    """Assign a saved quantized state dict over *module* (``weights_only`` unpickling only)."""
    torch = _load("torch")
    _load("torchao")  # registers torchao's tensor subclasses as safe globals
    state = torch.load(str(path), map_location=device, weights_only=True)
    module.load_state_dict(state, assign=True)


def _load_pinned(modeling: Any, checkpoint: Path, revision: str) -> Any:
    """``StrandsDeciderModel.load`` with the base at *revision*, never the default branch."""
    pinned = _json_object(player.member(checkpoint, player.CONFIG)).get("base_revision")
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


def _write_side_files(model: Any, checkpoint: Path, out: Path, revision: str) -> dict:
    """The fp32 head, the config (merged, pinned) and the tokenizer files byte for byte."""
    torch = _load("torch")
    head = {
        k: v.detach().to(torch.float32).cpu().contiguous()
        for k, v in model.head.state_dict().items()
    }
    _load("safetensors.torch").save_file(head, str(safepath.contained(out / player.HEAD, out)))
    config_doc = _json_object(player.member(checkpoint, player.CONFIG))
    config_doc.update({"base_revision": revision, "use_lora": False})
    safepath.contained(out / player.CONFIG, out).write_text(json.dumps(config_doc, indent=2) + "\n")
    for name in TOKENIZER_ALL:
        if (checkpoint / name).is_file():
            shutil.copyfile(player.member(checkpoint, name), safepath.contained(out / name, out))
    return config_doc


def collect_prompt_ids(model: Any, rows: Sequence[Mapping[str, Any]], device: str) -> list[int]:
    """Every token id the embedding looks up while upstream's engine answers *rows*.

    A forward pre-hook on the token embedding records the ids each forward pass reads, so
    the kept set is what the engine actually feeds the model (its own prompt rendering),
    not a re-tokenization of the request text.
    """
    torch = _load("torch")
    infer = _load("strands_decider.infer")
    schema = _load("strands_decider.schema")
    _fqn, embedding = _embedding(model.torso)
    seen: set[int] = set()

    def record(_module: Any, args: tuple) -> None:
        seen.update(int(i) for i in args[0].reshape(-1).tolist())

    handle = embedding.register_forward_pre_hook(record)
    try:
        with torch.inference_mode():
            engine = infer.SystemOneEngine(
                model, infer.EngineConfig(device=device, model_name="kept-ids")
            )
            for row in rows:
                engine.evaluate(schema.SystemOneRequest.model_validate(row["body"]))
    finally:
        handle.remove()
    return sorted(seen)


def _kept_ids_in_env(model: Any, torso: Any, prompts: Path, device: str) -> list[int]:
    """The pruned build's kept ids: the prompt set's looked-up ids plus the special tokens."""
    model.torso = torso
    used = collect_prompt_ids(model, read_prompt_set(prompts), device)
    _fqn, embedding = _embedding(torso)
    return kept_vocabulary(used, special_token_ids(model.tokenizer), embedding.num_embeddings)


def _build_in_env(
    checkpoint: Path,
    out: Path,
    scheme: QuantScheme,
    revision: str,
    device: str,
    prompts: Path | None = None,
):
    # Validate (every checkpoint file a regular file inside the checkpoint, a fresh output
    # directory, the prompt set's sides) before anything heavy is imported.
    plan_weight_only(
        checkpoint, out, scheme, python=sys.executable, base_revision=revision, prompts=prompts
    )
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    model = _load_pinned(modeling, checkpoint, revision)
    torso = model.torso
    if hasattr(torso, "merge_and_unload"):
        torso = torso.merge_and_unload()
    torso = torso.to(device)
    kept = None if prompts is None else _kept_ids_in_env(model, torso, prompts, device)
    quantization = quantize_torso(torso, scheme, kept)
    state = torso.state_dict()
    inventory = check_inventory(state_inventory(state), scheme)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(safepath.contained(out / TORSO_FILE, out)))
    config_doc = _write_side_files(model, checkpoint, out, revision)
    tokenizer = _load("transformers").AutoTokenizer.from_pretrained(str(out))
    if kept is None:
        vocabulary = check_full_vocabulary(quantization["embedding"]["rows"], len(tokenizer))
    else:
        vocabulary = check_pruned_vocabulary(kept, len(tokenizer), quantization["embedding"])
        vocabulary["prompt_set"] = prompt_set_record(prompts)
    source = source_record(checkpoint)
    if not _tokenizer_record(out, source)["unmodified"]:
        raise CliError(EXIT_ENV_ERROR, "the copied tokenizer differs from the checkpoint's")
    manifest = write_manifest(
        out,
        bits=scheme.block_bits,
        torchao_api=quantization["torchao_api"],
        torchao_version=_version("torchao") or "",
        torch_version=str(torch.__version__),
        source=source,
        base_model=config_doc.get("base_model"),
        base_model_revision=revision,
        scheme=scheme,
        extra={
            "quantization": {
                "call": f"torchao.quantization.quantize_(torso, "
                f"{quantization['torchao_api']}(...), filter_fn=<text-block linears>)",
                "config": quantization["config"],
                "scope": "text-block linears weight-only after merge_and_unload of the LoRA; "
                + (
                    "the full-vocabulary embedding packed"
                    if kept is None
                    else "only the kept ids' embedding rows packed (c38 comparison)"
                )
                + "; the pointer head fp32",
            },
            "tensor_policy": tensor_policy(),
            "inventory": inventory,
            "dropped": quantization["dropped"],
            "embedding": quantization["embedding"],
            "vocabulary": vocabulary,
            "device": device,
            "versions": {
                d: _version(d) for d in ("transformers", "peft", "safetensors", "strands-decider")
            },
        },
        kept_ids=kept,
    )
    keys = ("format", "torchao_api", "weight_bytes", "within_size_budget")
    print(json.dumps({"artifact": str(out), **{k: manifest[k] for k in keys}}))
    return 0


def _load_engine(artifact: Path, device: str, model_name: str) -> tuple[Any, dict[str, Any]]:
    info = read_artifact(artifact)
    manifest = info["manifest"]
    if not isinstance(manifest.get("embedding"), dict):
        raise _refuse(f"{MANIFEST_FILE} records no packed embedding; rebuild the artifact")
    torch = _load("torch")
    modeling = _load("strands_decider.modeling")
    infer = _load("strands_decider.infer")
    transformers = _load("transformers")
    safetensors = _load("safetensors.torch")
    revision = manifest["base_model_revision"]
    config = modeling.StrandsDeciderConfig.from_json(str(player.member(artifact, player.CONFIG)))
    if config.base_revision != revision or config.use_lora:
        raise _refuse(f"{player.CONFIG} must pin {revision} with use_lora false")
    tok = transformers.AutoTokenizer.from_pretrained(str(artifact))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if config.force_bos:
        modeling.ensure_bos(tok)
    device_map = device if device not in ("cpu", "mps", "mlx") else None
    torso, shell = None, _torso_shell(config, revision)
    if shell is not None:
        try:
            torso = _filled_torso(shell, modeling, manifest, artifact, device)
        except RuntimeError:  # the saved state does not fit the shell's keys
            torso = None
        if torso is not None and _meta_parameters(torso):
            torso = None
    if torso is None:  # load the base, as before r9
        base = modeling.StrandsDeciderModel._load_torso(config, device_map, None)
        torso = _filled_torso(base, modeling, manifest, artifact, device)
    if meta := _meta_parameters(torso):
        raise _refuse(f"the saved torso state left {len(meta)} parameter(s) unset: {meta[:3]}")
    model = modeling.StrandsDeciderModel.__new__(modeling.StrandsDeciderModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model.torso = torso
    model.tokenizer = tok
    model.head = modeling.build_head(config, modeling.StrandsDeciderModel.hidden_size(torso))
    model.head.load_state_dict(safetensors.load_file(str(player.member(artifact, player.HEAD))))
    model.head.to(device=device, dtype=torch.float32)
    # no_grad, not inference_mode: the engine's model.to(device) swaps each parameter, and
    # torchao's int4 tensor subclasses (tinygemm layout) can only be swapped outside it.
    with torch.no_grad():
        engine = infer.SystemOneEngine(
            model, infer.EngineConfig(device=device, model_name=model_name)
        )
    return engine, info


def _accelerator(torch: Any) -> str | None:
    try:
        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - a stats field; never a reason to fail serving
        return None


def _serve_in_env(artifact: Path, host: str, port: int, device: str, model_name: str) -> int:
    engine, info = _load_engine(artifact, device, model_name)
    torch = _load("torch")
    schema = _load("strands_decider.schema")
    pydantic = _load("pydantic")
    lock = threading.Lock()
    attested = bundle_attestation(artifact, info)  # from the load's own verification
    snapshot = file_snapshot(artifact)

    def bundle() -> Mapping[str, Any]:
        # The attested files must still be the ones loaded: unchanged stat -> the load's
        # attestation; any change -> re-verify from disk, so a swapped file shows up.
        if file_snapshot(artifact) == snapshot:
            return attested
        return bundle_attestation(artifact)

    accelerator = _accelerator(torch)

    def evaluate(body: dict[str, Any]) -> Mapping[str, Any]:
        try:
            request = schema.SystemOneRequest.model_validate(body)
        except pydantic.ValidationError as exc:
            raise RequestError(422, json.loads(exc.json())) from exc

        def call() -> Any:
            with lock, torch.inference_mode():
                return engine.evaluate(request)

        try:
            response = refuse_out_of_vocabulary(call)
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
            "pruned_vocabulary": is_pruned(info["manifest"]),
            "torchao_api": info["manifest"]["torchao_api"],
        }

    server = make_server(
        host,
        port,
        evaluate=evaluate,
        health=health,
        bundle=bundle,
        stats=lambda: {**host_stats(), "accelerator": accelerator},
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=MODULE, description="quantized pointer student.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="merge, drop, pack, quantize and save (student env)")
    b.add_argument("checkpoint")
    b.add_argument("out")
    b.add_argument("--bits", type=int, required=True, choices=sorted(FORMAT_OF_BITS))
    defaults = QuantScheme()
    b.add_argument(
        "--embedding-bits",
        type=int,
        default=defaults.embedding_bits,
        choices=sorted(FORMAT_OF_BITS),
    )
    b.add_argument("--block-group-size", type=int, default=defaults.block_group_size)
    b.add_argument("--embedding-group-size", type=int, default=defaults.embedding_group_size)
    b.add_argument("--base-revision", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument(
        "--prompts",
        default=None,
        help="a train/fit/selection prompt set: build the pruned-vocabulary comparison (c38)",
    )
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
        scheme = QuantScheme(
            block_bits=args.bits,
            embedding_bits=args.embedding_bits,
            block_group_size=args.block_group_size,
            embedding_group_size=args.embedding_group_size,
        ).validated()
        prompts = Path(args.prompts) if args.prompts else None
        return _build_in_env(
            Path(args.checkpoint), Path(args.out), scheme, args.base_revision, args.device, prompts
        )
    except CliError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.remediation:
            print(f"hint: {exc.remediation}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":  # runs in the student env
    sys.exit(main())
