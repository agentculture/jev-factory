"""Quantize a merged scorer checkpoint, and decide whether the quant needs healing.

The quantization half of nvsh's ``quantize.py`` (issue 46, task t12) as a package
module, with its separate-venv AWQ recipe in
:mod:`jev_factory.backbones.causal_lm.awq_oneshot`. It builds a **train-side-only**
calibration set (calibration and healing data never come from val, test or the
held-out set), then drives llama.cpp's converter, ``llama-imatrix`` and
``llama-quantize`` to a text-only (no ``mmproj``) ``bf16`` GGUF, imatrix-quantized to
``Q4_K_M``; optionally it exports INT4 AWQ by running the recipe as a **subprocess of
the AWQ venv's python** (risk r14: stock ``llm-compressor`` needs a newer transformers
than the training venv pins), never in-process. Tool paths come from the run config
(``llama_cpp_convert``, ``llama_cpp_quantize``, ``llama_cpp_imatrix``,
``awq_python``): none is hard-coded and none is a dependency of this package.

**No absolute paths in the GGUF.** ``llama-quantize`` copies the imatrix file path
and the calibration dataset name into the GGUF's ``quantize.imatrix.*`` metadata
verbatim, which would leak the builder's directory layout into a shipped bundle
(:mod:`jev_factory.release.bundle` refuses such a file). So this module never
*writes* one: ``llama-imatrix`` and ``llama-quantize`` run with the work dir as their
working directory and work-dir-relative arguments, and the finished GGUF is read back
with the bundle's own reader, refusing (:func:`assert_no_absolute_paths`) if any
absolute path remains.

**The llama.cpp commit is recorded**: from ``LLAMA_CPP_DIR`` if exported, else the
checkout the converter script lives in; ``"unknown (...)"`` when git cannot say,
never invented.

**The heal trigger has one owner.** :func:`heal_needed` decides, and
:func:`heal_trigger` logs the decision *before* any heal is planned, and refuses a
second round. The margin and round count are the ones
:mod:`jev_factory.decide.rules` uses (:data:`HEAL_MARGIN_POINTS`,
:data:`HEAL_ROUNDS`); the train side's own one-round guard is
:func:`jev_factory.backbones.causal_lm.train.plan_heal` and is not duplicated here.

Quantize is a GPU stage: :func:`run_quantize` sends every subprocess through the
guarded runner (:func:`jev_factory.factory.gpu.run_gpu_stage`) and is a dry run
unless ``apply``. Every subprocess goes through an injected ``run`` seam
(``argv, timeout, cwd -> (returncode, output)``) so tests use stub binaries.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.backbones.causal_lm import awq_oneshot, gen_config
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.core.split import HELD_OUT_NAME
from jev_factory.decide.rules import HEAL_MARGIN_POINTS, HEAL_ROUNDS
from jev_factory.factory.config import RunConfig
from jev_factory.factory.gpu import WATCHDOG_STATUS, run_gpu_stage

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/quantize.py",
    "commit": "9debdc6",
    "adaptations": [
        "kept (lines 1-633): calibration set, calibration writers, mtp_without_weights,"
        " convert_gguf, compute_imatrix, quantize_q4_k_m, export_awq, vLLM support files,"
        " tool-version recording, QuantSummary and heal_needed",
        "_sibling/importlib loading of gen_config.py replaced by a package import; the AWQ script"
        " is now awq_oneshot.py next to this module, run by file path under the AWQ python",
        "tool paths read from the run config (llama_cpp_convert/quantize/imatrix, awq_python)"
        " instead of LLAMA_CPP_*/AWQ_PY env vars; the commit falls back to the converter's"
        " checkout when LLAMA_CPP_DIR is unset",
        "new: imatrix and quantize run with the work dir as cwd and relative arguments, and the"
        " GGUF is checked with the bundle's reader so quantize.imatrix.* never holds an"
        " absolute path",
        "new: heal_trigger logs the trigger before any heal and refuses a second round; the"
        " margin and round count are shared with decide.rules",
        "new: plan_quantize/run_quantize drive the stage through run_gpu_stage, dry-run by default"
        " (replaces main's straight-line script)",
        "tests ported from tests/test_lfm_finetune_quantize.py",
    ],
    "licence": "Apache-2.0",
}

DEFAULT_TIMEOUT = 3600.0
VERSION_TIMEOUT = 30.0

#: awq_oneshot.py, run as a subprocess of the AWQ python -- never imported in-process.
AWQ_ONESHOT_SCRIPT = Path(awq_oneshot.__file__).resolve()

#: Where llama.cpp's checked-out commit is read from, if the operator exported it.
LLAMA_CPP_DIR_VAR = "LLAMA_CPP_DIR"

#: llm-compressor's save_pretrained does not write these; vLLM needs them to serve the
#: AWQ export, so they are copied from the source model dir.
VLLM_SUPPORT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

#: A serving launcher that cannot pass this vLLM flag must record it, not drop it.
AWQ_SERVE_ARGS = ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']

#: Run inside the AWQ venv to report the two package versions llm-compressor needs.
_AWQ_VERSION_PROBE = (
    "import importlib.metadata as m; print(m.version('llmcompressor'), m.version('transformers'))"
)

#: ``split.py``'s note in a side's header: ``Split '<side>' of <corpus> (seed=N).``
_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")
_HELD_OUT_MARKER = "held-out split"

#: File names inside the work dir. The GPU steps use these *relative* names.
CALIBRATION_TXT = "calibration.txt"
CALIBRATION_JSONL = "calibration.jsonl"
GGUF_BF16 = "model-bf16.gguf"
IMATRIX_FILE = "imatrix.dat"
GGUF_Q4_K_M = "model-q4_k_m.gguf"
AWQ_DIR = "awq"
RUN_LOG = "quantize-run.json"
HEAL_LOG = "heal-trigger-log.jsonl"

RunFn = Callable[..., "tuple[int, str]"]


class QuantizeError(CliError):
    """A refusal or a failed tool invocation: one structured line, never a traceback."""

    def __init__(self, message: str, remediation: str = "", code: int = EXIT_USER_ERROR) -> None:
        super().__init__(code, message, remediation)


def default_run(  # pragma: no cover - exercised through stub binaries in tests
    argv: Sequence[str], timeout: float, cwd: Path | None = None
) -> tuple[int, str]:
    """Run *argv* (a fixed list, no shell) in *cwd* and return ``(exit code, output)``."""
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv list, no shell=True
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None if cwd is None else str(cwd),
        )
    except FileNotFoundError:
        return (127, f"{argv[0]}: not found")
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, f"{type(exc).__name__}: {exc}")
    return (completed.returncode, (completed.stdout or "") + (completed.stderr or ""))


#: ``bash -c`` wrapper so a GPU-stage command starts in a chosen directory
#: (``run_capped`` has no cwd of its own).
_CD_WRAPPER = 'cd "$1" && shift && exec "$@"'


def gpu_run(
    run_dir: Path,
    *,
    memory_max: str,
    memory_floor: str = "8G",
    watchdog_seconds: int = 5,
    allow_foreign: bool = False,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., Any] | None = None,
) -> RunFn:
    """A :data:`RunFn` that runs each command as the guarded ``quantize`` GPU stage.

    Residency guard, then the memory cap and floor watchdog
    (:func:`jev_factory.factory.gpu.run_gpu_stage`). *runner* replaces that function in tests.
    """
    stage_runner = runner or run_gpu_stage

    def run(argv: Sequence[str], timeout: float, cwd: Path | None = None) -> tuple[int, str]:
        cmd = list(argv)
        if cwd is not None:
            cmd = ["bash", "-c", _CD_WRAPPER, "jev-quantize", str(cwd), *cmd]
        done = stage_runner(
            "quantize",
            run_dir,
            cmd,
            allow_foreign=allow_foreign,
            memory_max=memory_max,
            memory_floor=memory_floor,
            watchdog_seconds=watchdog_seconds,
            env=None if env is None else dict(env),
            timeout=timeout,
        )
        output = (done.stdout or "") + (done.stderr or "")
        if done.returncode == WATCHDOG_STATUS:
            output = f"stopped by the memory-floor watchdog: {output}"
        return (done.returncode, output)

    return run


# ---------------------------------------------------------------------------
# Calibration set: train-side only
# ---------------------------------------------------------------------------


def _load_split(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"header": "", "entries": raw}


def _entries_of(raw: dict) -> list[dict]:
    return raw.get("entries", [])


def _source_ids(entries: Iterable[dict]) -> set[str]:
    return {entry.get("source_id", entry["id"]) for entry in entries}


def _verify_split_side(path: Path, raw: dict, expected: str) -> None:
    """Refuse *path* unless its ``split.py`` header names *expected*.

    Checking only for disjoint ``source_id``\\ s cannot catch a swapped
    ``--train``/``--val`` pair when both splits are otherwise ordinary, so each
    file's header must name the role it is used for. The held-out split is refused
    outright, by file name and by its header's own marker.
    """
    header = raw.get("header")
    header = header if isinstance(header, str) else ""
    if path.name == HELD_OUT_NAME or header.casefold().startswith(_HELD_OUT_MARKER):
        raise QuantizeError(
            f"{path}: the held-out split is never used for calibration or healing",
            "calibrate from the train split only",
        )
    match = _SPLIT_SIDE_RE.search(header)
    found = match.group(1) if match else None
    if found != expected:
        raise QuantizeError(
            f"{path}: its header names side {found!r}, expected {expected!r} -- "
            "calibration refuses a file whose header does not match its --train/--val/--test role"
        )


def build_calibration_set(
    train: Path, val: Path, test: Path, limit: int | None = None
) -> list[str]:
    """Calibration text for imatrix/AWQ, drawn only from *train*.

    Verifies each of *train*, *val* and *test* against its ``split.py`` header before
    trusting which side it is, refuses the held-out split, and refuses if any train
    entry's ``source_id`` also appears on *val* or *test* (a hand-edited split file
    could reintroduce one and silently leak validation/test data into calibration or a
    later heal).
    """
    train_raw, val_raw, test_raw = _load_split(train), _load_split(val), _load_split(test)
    _verify_split_side(train, train_raw, "train")
    _verify_split_side(val, val_raw, "val")
    _verify_split_side(test, test_raw, "test")

    train_entries = _entries_of(train_raw)
    other_ids = _source_ids(_entries_of(val_raw)) | _source_ids(_entries_of(test_raw))
    overlap = sorted(_source_ids(train_entries) & other_ids)
    if overlap:
        raise QuantizeError(
            f"calibration set would use non-train source_id(s) {overlap}; calibration and"
            " healing data must be train-side only"
        )
    texts = [entry["text"] for entry in train_entries if "text" in entry]
    if limit is not None:
        texts = texts[:limit]
    return texts


def write_calibration_file(texts: Sequence[str], out_path: Path) -> Path:
    """Write one calibration text per line, plain text, for llama.cpp's ``imatrix`` step.

    ``imatrix`` reads plain text as one undifferentiated mass, so an embedded newline is
    replaced with a space: one line per entry, without inventing a record. This is NOT
    the file AWQ calibrates from; :func:`write_calibration_jsonl` keeps record boundaries.
    """
    content = "\n".join(text.replace("\n", " ") for text in texts)
    out_path.write_text(content + ("\n" if texts else ""), encoding="utf-8")
    return out_path


def write_calibration_jsonl(texts: Sequence[str], out_path: Path) -> Path:
    """Write one JSON-encoded string per line, for AWQ's ``--calibration-file``.

    AWQ calibrates on discrete samples: a record with an embedded newline must still
    read back as exactly one sample. JSON-encoding keeps the newline as the escape
    ``\\n``, so the physical line count equals the record count.
    :func:`jev_factory.backbones.causal_lm.awq_oneshot.read_calibration_texts` reads it.
    """
    lines = [json.dumps(text) for text in texts]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Tool paths, from the run config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPaths:
    """Where each llama.cpp tool lives on this machine; never hard-coded."""

    convert: str
    quantize: str
    imatrix: str


def tool_paths_from_config(config: RunConfig) -> ToolPaths:
    """Every llama.cpp tool path from *config*; a stage that needs one demands it."""
    return ToolPaths(
        convert=config.require("llama_cpp_convert"),
        quantize=config.require("llama_cpp_quantize"),
        imatrix=config.require("llama_cpp_imatrix"),
    )


def awq_python_from_config(config: RunConfig) -> str:
    """The AWQ venv's python (run config ``awq_python``); refuses when unset (risk r14).

    The AWQ step never falls back to the training venv's python: stock llm-compressor
    needs a newer transformers than the training venv pins.
    """
    return config.require("awq_python")


# ---------------------------------------------------------------------------
# GGUF conversion + imatrix + Q4_K_M (text-only, no mmproj)
# ---------------------------------------------------------------------------


def _tensor_names(model_dir: Path) -> set[str]:
    """Every tensor name in *model_dir*'s safetensors files (headers only)."""
    names: set[str] = set()
    for shard in sorted(model_dir.glob("*.safetensors")):
        with open(shard, "rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(size))
        names.update(name for name in header if name != "__metadata__")
    return names


def mtp_without_weights(model_dir: Path) -> bool:
    """True when *model_dir*'s config declares an MTP head that has no ``mtp.*`` tensor.

    The text-only merge (``Qwen3_5ForCausalLM``) keeps the base config's
    ``mtp_num_hidden_layers`` but none of its ``mtp.*`` tensors, so llama.cpp's
    converter writes one block more than it has weights for and llama.cpp refuses the
    file. Such a model converts with ``--no-mtp``: the model that was measured, since the
    served bf16 checkpoint never had an MTP head either.
    """
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    holders = [c for c in (config.get("text_config"), config) if isinstance(c, dict)]
    declared = any(int(c.get("mtp_num_hidden_layers") or 0) > 0 for c in holders)
    return declared and not any("mtp" in name for name in _tensor_names(model_dir))


def convert_gguf(
    run: RunFn,
    tools: ToolPaths,
    model_dir: Path,
    out_file: Path,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    no_mtp: bool = False,
) -> str:
    """Convert a merged HF checkpoint to a bf16 GGUF. Refuses a vision-projector output.

    ``bf16``, not ``f16``: the base weights are bf16 and converting through f16 first
    risks a silent range-overflow rounding pass. The served artifact is text-only, so a
    converter that writes a separate ``mmproj-*`` file alongside *out_file* is a
    refusal, not silently accepted.
    """
    argv = [tools.convert, str(model_dir), "--outfile", str(out_file), "--outtype", "bf16"]
    if no_mtp:
        argv.append("--no-mtp")
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"GGUF conversion failed (exit {code}): {output}")
    mmproj = out_file.parent / f"mmproj-{out_file.name}"
    if mmproj.exists():
        raise QuantizeError(
            f"conversion wrote a vision projector {mmproj}; this export is text-only, no mmproj"
        )
    return output


def compute_imatrix(
    run: RunFn,
    tools: ToolPaths,
    gguf_file: Path | str,
    calibration_file: Path | str,
    out_file: Path | str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    cwd: Path | None = None,
) -> str:
    """Build an importance matrix from the train-side calibration file.

    Pass work-dir-relative paths with *cwd* set to the work dir so the dataset name the
    imatrix records (and ``llama-quantize`` later copies into the GGUF) is relative.
    """
    argv = [tools.imatrix, "-m", str(gguf_file), "-f", str(calibration_file), "-o", str(out_file)]
    code, output = run(argv, timeout) if cwd is None else run(argv, timeout, cwd=cwd)
    if code != 0:
        raise QuantizeError(f"imatrix computation failed (exit {code}): {output}")
    return output


def quantize_q4_k_m(
    run: RunFn,
    tools: ToolPaths,
    gguf_file: Path | str,
    imatrix_file: Path | str,
    out_file: Path | str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    cwd: Path | None = None,
) -> str:
    """Quantize a bf16 GGUF to Q4_K_M, imatrix-guided.

    With *cwd* set and work-dir-relative paths, ``quantize.imatrix.file`` in the output
    is relative; :func:`assert_no_absolute_paths` then verifies it.
    """
    argv = [tools.quantize, "--imatrix", str(imatrix_file), str(gguf_file), str(out_file)]
    argv.append("Q4_K_M")
    code, output = run(argv, timeout) if cwd is None else run(argv, timeout, cwd=cwd)
    if code != 0:
        raise QuantizeError(f"Q4_K_M quantization failed (exit {code}): {output}")
    return output


def assert_no_absolute_paths(gguf: Path) -> None:
    """Refuse a GGUF whose ``quantize.imatrix.*`` (or any string) metadata holds an absolute path.

    Read back with the bundle's own GGUF reader, so the two can never disagree about
    what counts as leaking the builder's directory layout.
    """
    from jev_factory.release.bundle import BundleError, gguf_absolute_paths

    try:
        leaked = gguf_absolute_paths(gguf)
    except BundleError as exc:
        raise QuantizeError(f"{gguf.name}: cannot verify its metadata ({exc})") from None
    if leaked:
        raise QuantizeError(
            f"{gguf.name}: GGUF metadata {', '.join(sorted(leaked))} holds an absolute path",
            "run llama-imatrix and llama-quantize with the work dir as cwd and relative arguments",
        )


def export_awq(
    run: RunFn,
    awq_py: str,
    model_dir: Path,
    calibration_file: Path,
    out_dir: Path,
    num_calibration_samples: int,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Run the proven AWQ recipe as a subprocess of *awq_py* (risk r14).

    Never an ``llm-compressor`` command line, and never imported in-process:
    ``awq_oneshot.py`` runs inside a separate venv (llm-compressor + a newer
    transformers) that the shared training venv does not carry.
    """
    argv = [
        awq_py,
        str(AWQ_ONESHOT_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--calibration-file",
        str(calibration_file),
        "--out-dir",
        str(out_dir),
        "--num-calibration-samples",
        str(num_calibration_samples),
    ]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"INT4 AWQ export failed (exit {code}): {output}")
    return output


def copy_vllm_support_files(model_dir: Path, out_dir: Path) -> list[str]:
    """Copy each of :data:`VLLM_SUPPORT_FILES` from *model_dir* to *out_dir* when present.

    ``save_pretrained(save_compressed=True)`` does not write these, but vLLM needs them.
    Symlinks are resolved (a Hugging Face snapshot dir is a tree of symlinks into a
    shared blob store) so *out_dir* stays self-contained. Returns the names copied.
    """
    copied = []
    for name in VLLM_SUPPORT_FILES:
        src = model_dir / name
        if src.is_file():  # is_file() follows symlinks; a dangling one is skipped
            shutil.copyfile(src.resolve(), out_dir / name)
            copied.append(name)
    return copied


def finish_awq_export(model_dir: Path, out_dir: Path) -> dict:
    """Post-processing after ``awq_oneshot.py`` saves *out_dir*.

    Copies the vLLM-serving files the AWQ save does not write, then writes a
    greedy-decoding ``generation_config.json``. Returns what the run record needs: the
    files copied, and the vLLM serve flag a launcher may not be able to pass on its own,
    recorded as a known limitation rather than silently dropped.
    """
    copied = copy_vllm_support_files(model_dir, out_dir)
    gen_config.write(out_dir)
    return {"copied_files": copied, "serve_args": list(AWQ_SERVE_ARGS)}


# ---------------------------------------------------------------------------
# Tool version recording: every result names its exact tool version
# ---------------------------------------------------------------------------


def tool_version(run: RunFn, path: str, timeout: float = VERSION_TIMEOUT) -> str:
    """The tool's own ``--version`` output, or ``"unknown (...)"`` if it fails."""
    code, output = run([path, "--version"], timeout)
    output = output.strip()
    if code != 0:
        return f"unknown ({output or code})"
    return output


def llama_cpp_commit(run: RunFn, directory: str | Path | None, timeout: float = VERSION_TIMEOUT):
    """llama.cpp's checked-out commit, from *directory*'s git HEAD.

    ``"unknown (...)"`` when there is no directory to ask or git fails: never invented.
    """
    if not directory:
        return f"unknown ({LLAMA_CPP_DIR_VAR} not set)"
    code, output = run(["git", "-C", str(directory), "rev-parse", "HEAD"], timeout)
    output = output.strip()
    if code != 0:
        return f"unknown ({output or code})"
    return output


def llama_cpp_dir(tools: ToolPaths, env: Mapping[str, str]) -> str:
    """The llama.cpp checkout: ``LLAMA_CPP_DIR`` if exported, else the converter's own dir."""
    return env.get(LLAMA_CPP_DIR_VAR) or str(Path(tools.convert).resolve().parent)


def awq_tool_versions(run: RunFn, awq_py: str, timeout: float = VERSION_TIMEOUT) -> dict[str, str]:
    """llm-compressor's and transformers' versions, read from the AWQ venv itself."""
    code, output = run([awq_py, "-c", _AWQ_VERSION_PROBE], timeout)
    output = output.strip()
    if code != 0:
        unknown = f"unknown ({output or code})"
        return {"llm-compressor": unknown, "transformers": unknown}
    parts = output.split()
    if len(parts) != 2:
        unknown = f"unknown (unexpected output: {output!r})"
        return {"llm-compressor": unknown, "transformers": unknown}
    compressor, transformers = parts
    return {"llm-compressor": compressor, "transformers": transformers}


def record_tool_versions(
    run: RunFn,
    tools: ToolPaths,
    env: Mapping[str, str],
    awq_py: str | None,
    timeout: float = VERSION_TIMEOUT,
) -> dict[str, str]:
    """Every tool's version, keyed by name, for the run log (AWQ only when it was run)."""
    versions = {
        "llama.cpp convert": tool_version(run, tools.convert, timeout),
        "llama.cpp imatrix": tool_version(run, tools.imatrix, timeout),
        "llama.cpp quantize": tool_version(run, tools.quantize, timeout),
        "llama.cpp commit": llama_cpp_commit(run, llama_cpp_dir(tools, env), timeout),
    }
    if awq_py:
        versions.update(awq_tool_versions(run, awq_py, timeout))
    return versions


# ---------------------------------------------------------------------------
# The heal trigger: one owner (decisions c42, c43)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSummary:
    """What :func:`heal_needed` compares between a bf16 build and a quantized one.

    ``wrong_mutating_ids`` is the *set* of entry ids with a wrong mutating proposal,
    not a count: two builds can have the same count while disagreeing on which entries
    they are wrong on, and only the set catches that.
    """

    right_pct: float
    wrong_mutating_ids: frozenset[str]


def loss_points(bf16: QuantSummary, quant: QuantSummary) -> float:
    """Right-proposal points *quant* lost against its own *bf16* (rounded against float dust)."""
    return round(bf16.right_pct - quant.right_pct, 9)


def new_wrong_mutating_ids(bf16: QuantSummary, quant: QuantSummary) -> list[str]:
    """Entry ids *quant* gets a wrong mutating proposal on that bf16 did not."""
    return sorted(set(quant.wrong_mutating_ids) - set(bf16.wrong_mutating_ids))


def heal_needed(bf16: QuantSummary, quant: QuantSummary) -> bool:
    """True when *quant* has drifted enough from its own *bf16* to need a healing round.

    Two independent triggers, either sufficient: right proposals drop by MORE than
    :data:`HEAL_MARGIN_POINTS` points, or *quant* has a wrong mutating proposal on an
    entry id *bf16* did not have one on. Quantization fixing one entry while breaking
    another leaves a count unchanged but is a new failure. A drop of exactly the margin,
    or an id set that only loses members, does not trigger a heal.
    """
    return loss_points(bf16, quant) > HEAL_MARGIN_POINTS or bool(
        new_wrong_mutating_ids(bf16, quant)
    )


@dataclass(frozen=True)
class HealTrigger:
    """A logged decision to heal one quantized build (round *heal_round* of one)."""

    candidate: str
    heal_round: int
    loss_points: float
    new_wrong_mutating_ids: list[str]
    reasons: list[str] = field(default_factory=list)


def _read_heal_log(log: Path) -> list[dict[str, Any]]:
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def _append_heal_log(log: Path, event: dict[str, Any]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def heal_trigger(
    bf16: QuantSummary,
    quant: QuantSummary,
    *,
    log: Path,
    candidate: str,
    heal_rounds: int = 0,
) -> HealTrigger | None:
    """Decide whether *candidate*'s quant needs healing; log a trigger before anything heals.

    Returns ``None`` when :func:`heal_needed` is false. Otherwise the trigger (the
    numbers and ids that fired it) is **appended to *log* first**, and only then is the
    :class:`HealTrigger` returned for the caller to plan a heal from. Heal is one round
    only: rounds already spent are the larger of *heal_rounds* (for example a train log
    that says the build is itself a heal) and the non-refused triggers already in *log*
    for this candidate. A trigger with no round left is logged as refused and raises.
    """
    if not heal_needed(bf16, quant):
        _append_heal_log(log, {"event": "heal_check", "candidate": candidate, "needed": False})
        return None
    lost, new_ids = loss_points(bf16, quant), new_wrong_mutating_ids(bf16, quant)
    reasons = []
    if lost > HEAL_MARGIN_POINTS:
        reasons.append(f"right proposals down {lost} pts (margin {HEAL_MARGIN_POINTS})")
    if new_ids:
        reasons.append(f"new wrong-mutating ids: {new_ids}")
    spent = max(
        heal_rounds,
        sum(
            1
            for event in _read_heal_log(log)
            if event.get("event") == "heal_trigger"
            and event.get("candidate") == candidate
            and not event.get("refused")
        ),
    )
    event = {
        "event": "heal_trigger",
        "candidate": candidate,
        "heal_round": spent + 1,
        "loss_points": lost,
        "new_wrong_mutating_ids": new_ids,
        "reasons": reasons,
        "refused": spent >= HEAL_ROUNDS,
    }
    _append_heal_log(log, event)
    if spent >= HEAL_ROUNDS:
        raise QuantizeError(
            f"{candidate}: heal is one round only ({HEAL_ROUNDS}) and it has been spent; "
            f"the quant still fails ({'; '.join(reasons)})",
            "drop this build rather than heal it again, or record a deviation",
        )
    return HealTrigger(candidate, spent + 1, lost, new_ids, reasons)


# ---------------------------------------------------------------------------
# The stage: plan (dry run) and run (apply)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizePlan:
    """What the quantize stage will run; :func:`run_quantize` writes nothing unless applied."""

    model_dir: Path
    work_dir: Path
    texts: tuple[str, ...]
    no_mtp: bool
    tools: ToolPaths
    awq_python: str | None
    commands: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "quantize",
            "model_dir": str(self.model_dir),
            "work_dir": str(self.work_dir),
            "calibration_entries": len(self.texts),
            "gguf_no_mtp": self.no_mtp,
            "awq": self.awq_python is not None,
            "commands": [list(c) for c in self.commands],
            "cwd_for_imatrix_and_quantize": str(self.work_dir),
        }


def plan_quantize(
    *,
    config: RunConfig,
    model_dir: Path,
    train: Path,
    val: Path,
    test: Path,
    work_dir: Path,
    calibration_limit: int | None = None,
    with_awq: bool = False,
) -> QuantizePlan:
    """Validate the inputs and list the commands: train-side calibration text, tool paths."""
    model_dir, work_dir = Path(model_dir), Path(work_dir)
    tools = tool_paths_from_config(config)
    awq_py = awq_python_from_config(config) if with_awq else None
    texts = build_calibration_set(Path(train), Path(val), Path(test), calibration_limit)
    try:
        no_mtp = mtp_without_weights(model_dir)
    except (OSError, ValueError) as exc:
        raise QuantizeError(f"{model_dir}: cannot read a merged checkpoint ({exc})") from None
    convert = [tools.convert, str(model_dir), "--outfile", str(work_dir / GGUF_BF16)]
    convert += ["--outtype", "bf16"] + (["--no-mtp"] if no_mtp else [])
    commands = [
        convert,
        [tools.imatrix, "-m", GGUF_BF16, "-f", CALIBRATION_TXT, "-o", IMATRIX_FILE],
        [tools.quantize, "--imatrix", IMATRIX_FILE, GGUF_BF16, GGUF_Q4_K_M, "Q4_K_M"],
    ]
    if awq_py:
        commands.append(
            [awq_py, str(AWQ_ONESHOT_SCRIPT), "--model-dir", str(model_dir)]
            + ["--calibration-file", str(work_dir / CALIBRATION_JSONL)]
            + ["--out-dir", str(work_dir / AWQ_DIR)]
            + ["--num-calibration-samples", str(len(texts))]
        )
    return QuantizePlan(
        model_dir, work_dir, tuple(texts), no_mtp, tools, awq_py, tuple(tuple(c) for c in commands)
    )


def run_quantize(
    plan: QuantizePlan,
    *,
    apply: bool = False,
    run: RunFn | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Dry run (the plan, nothing written) unless *apply*; then the whole quantize stage.

    *run* is the subprocess seam; callers applying a real stage pass :func:`gpu_run`.
    Order: greedy ``generation_config.json`` on the source dir (so the GGUF can pick the
    sampling defaults up), bf16 GGUF, imatrix, ``Q4_K_M`` (both from the work dir, relative
    paths), the no-absolute-path check, then the optional AWQ export. Writes
    ``quantize-run.json`` with the llama.cpp commit and every tool version.
    """
    if not apply:
        return {"applied": False, "plan": plan.to_dict()}
    if run is None:
        raise QuantizeError(
            "quantize is a GPU stage; applying it needs a guarded runner",
            "build one with gpu_run(...) from the run config's train_memory_max",
            EXIT_ENV_ERROR,
        )
    environ = dict(env or {})
    work = plan.work_dir
    work.mkdir(parents=True, exist_ok=True)
    write_calibration_file(plan.texts, work / CALIBRATION_TXT)
    write_calibration_jsonl(plan.texts, work / CALIBRATION_JSONL)
    gen_config.write(plan.model_dir)
    convert_gguf(run, plan.tools, plan.model_dir, work / GGUF_BF16, no_mtp=plan.no_mtp)
    compute_imatrix(run, plan.tools, GGUF_BF16, CALIBRATION_TXT, IMATRIX_FILE, cwd=work)
    quantize_q4_k_m(run, plan.tools, GGUF_BF16, IMATRIX_FILE, GGUF_Q4_K_M, cwd=work)
    assert_no_absolute_paths(work / GGUF_Q4_K_M)
    log: dict[str, Any] = {
        "calibration_entries": len(plan.texts),
        "gguf_q4_k_m": GGUF_Q4_K_M,
        "gguf_no_mtp": plan.no_mtp,
    }
    if plan.awq_python:
        export_awq(
            run, plan.awq_python, plan.model_dir, work / CALIBRATION_JSONL, work / AWQ_DIR,
            len(plan.texts),
        )  # fmt: skip
        result = finish_awq_export(plan.model_dir, work / AWQ_DIR)
        log.update(
            awq_dir=AWQ_DIR,
            awq_copied_files=result["copied_files"],
            awq_serve_args=result["serve_args"],
            awq_serve_args_note="a launcher that cannot pass these needs them set by hand",
        )
    log["tool_versions"] = record_tool_versions(run, plan.tools, environ, plan.awq_python)
    (work / RUN_LOG).write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"applied": True, "plan": plan.to_dict(), "log": log}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize a merged scorer checkpoint (dry run).")
    parser.add_argument("--config", type=Path, default=None, help="run config TOML")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--calibration-limit", type=int, default=None)
    parser.add_argument("--awq", action="store_true", help="also export INT4 AWQ")
    parser.add_argument("--apply", action="store_true", help="run it (default: dry run)")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - drives real tools
    from jev_factory.factory.config import load_config

    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        plan = plan_quantize(
            config=config,
            model_dir=args.model_dir,
            train=args.train,
            val=args.val,
            test=args.test,
            work_dir=args.work_dir,
            calibration_limit=args.calibration_limit,
            with_awq=args.awq,
        )
        run = None
        if args.apply:
            run = gpu_run(
                args.work_dir,
                memory_max=config.require("train_memory_max"),
                memory_floor=config["train_memory_floor"],
                watchdog_seconds=config["train_watchdog_seconds"],
            )
        import os

        result = run_quantize(plan, apply=args.apply, run=run, env=os.environ)
    except CliError as exc:
        print(f"quantize: {exc.message}", file=sys.stderr)
        return exc.code
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
