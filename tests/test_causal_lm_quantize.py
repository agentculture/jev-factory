"""jev_factory.backbones.causal_lm.quantize: quantize, AWQ hand-off and the heal trigger (t23).

Ported from nvsh's tests/test_lfm_finetune_quantize.py (the env-var tool-path tests become
run-config tests), plus the tests this task adds: the llama.cpp commit, relative imatrix paths
(no absolute path in the GGUF's ``quantize.imatrix.*`` metadata), the GPU stage, and the
one-round heal trigger (behavioral o17). llama.cpp and the AWQ venv are subprocesses: the
stage tests put stub binaries on disk instead of invoking them.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import quantize
from jev_factory.cli._errors import CliError
from jev_factory.decide import rules
from jev_factory.release import bundle
from tests.release_support import make_config

TOOLS = quantize.ToolPaths(
    convert="/opt/llama/convert.py", quantize="/opt/llama/quantize", imatrix="/opt/llama/imatrix"
)


def _write_split(path: Path, side: str, entries: list[dict], header: str | None = None) -> Path:
    """Write a split file with a ``split.py``-shaped header naming *side*."""
    if header is None:
        header = f"Split '{side}' of corpus.json (seed=1)."
    path.write_text(json.dumps({"header": header, "entries": entries}), encoding="utf-8")
    return path


def _entry(entry_id: str, source_id: str | None = None, text: str = "hi") -> dict:
    entry = {"id": entry_id, "text": text, "expect": {"operation": "thermal_stats"}}
    if source_id is not None:
        entry["source_id"] = source_id
    return entry


# ---------------------------------------------------------------------------
# Calibration set: train-side only (h27)
# ---------------------------------------------------------------------------


def test_calibration_set_uses_only_train_text(tmp_path) -> None:
    module = quantize
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", text="train text")])
    val = _write_split(tmp_path / "val.json", "val", [_entry("v1", text="val text")])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", text="test text")])
    texts = module.build_calibration_set(train, val, test)
    assert texts == ["train text"]


def test_calibration_set_refuses_a_val_source_id_in_train(tmp_path) -> None:
    module = quantize
    # A hand-edited train file that reintroduces a val source_id (h27's belt-and-braces check).
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", "val", [_entry("v1", source_id="shared")])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_a_test_source_id_in_train(tmp_path) -> None:
    module = quantize
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", source_id="shared")])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_respects_a_limit(tmp_path) -> None:
    module = quantize
    train = _write_split(
        tmp_path / "train.json", "train", [_entry(f"t{i}", text=f"text{i}") for i in range(5)]
    )
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    texts = module.build_calibration_set(train, val, test, limit=2)
    assert texts == ["text0", "text1"]


def test_calibration_set_refuses_swapped_train_and_val(tmp_path) -> None:
    """Codex finding #3: --train/--val swapped, but otherwise ordinary disjoint splits."""
    module = quantize
    # These are ordinary, mutually disjoint splits: the swap is purely in which
    # path is passed as --train and which as --val, so a disjointness check alone
    # cannot catch it. The header on each file names its real side.
    real_train = _write_split(tmp_path / "train.json", "train", [_entry("t1", text="train text")])
    real_val = _write_split(tmp_path / "val.json", "val", [_entry("v1", text="val text")])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", text="test text")])
    with pytest.raises(module.QuantizeError, match="val.*expected 'train'"):
        # --train is given val.json, --val is given train.json.
        module.build_calibration_set(real_val, real_train, test)


def test_calibration_set_refuses_a_file_whose_header_names_the_wrong_side(tmp_path) -> None:
    module = quantize
    train = _write_split(tmp_path / "train.json", "test", [_entry("t1")])
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="expected 'train'"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_the_held_out_split_by_file_name(tmp_path) -> None:
    module = quantize
    train = _write_split(
        tmp_path / "held-out.json", "train", [_entry("t1")]
    )  # header lies about its side; the file name alone must refuse it
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="held-out"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_the_held_out_split_by_header(tmp_path) -> None:
    module = quantize
    train = _write_split(
        tmp_path / "train.json", "train", [_entry("t1")], header="Held-out split of corpus.json."
    )
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="held-out"):
        module.build_calibration_set(train, val, test)


def test_write_calibration_file_writes_one_text_per_line(tmp_path) -> None:
    module = quantize
    out = module.write_calibration_file(["a", "b"], tmp_path / "calib.txt")
    assert out.read_text(encoding="utf-8") == "a\nb\n"


def test_write_calibration_file_handles_no_texts(tmp_path) -> None:
    module = quantize
    out = module.write_calibration_file([], tmp_path / "calib.txt")
    assert out.read_text(encoding="utf-8") == ""


def test_write_calibration_file_replaces_embedded_newlines_with_spaces(tmp_path) -> None:
    """Codex finding #7: this plain-text file is only for llama.cpp's imatrix step, which

    reads raw text mass, not discrete records -- collapsing an embedded newline to a
    space there is acceptable, unlike the AWQ JSONL file (below) where it is not.
    """
    module = quantize
    out = module.write_calibration_file(["a\nb", "c"], tmp_path / "calib.txt")
    assert out.read_text(encoding="utf-8") == "a b\nc\n"


# ---------------------------------------------------------------------------
# JSONL calibration file: for AWQ, one JSON string per line (Codex finding #7)
# ---------------------------------------------------------------------------


def test_write_calibration_jsonl_writes_one_json_string_per_line(tmp_path) -> None:
    module = quantize
    out = module.write_calibration_jsonl(["a", "b"], tmp_path / "calib.jsonl")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == ["a", "b"]


def test_write_calibration_jsonl_handles_no_texts(tmp_path) -> None:
    module = quantize
    out = module.write_calibration_jsonl([], tmp_path / "calib.jsonl")
    assert out.read_text(encoding="utf-8") == ""


def test_write_calibration_jsonl_preserves_a_record_with_an_embedded_newline(tmp_path) -> None:
    """Codex finding #7: an embedded newline must not split one record into two samples."""
    module = quantize
    texts = ["request\nerror details", "second request"]
    out = module.write_calibration_jsonl(texts, tmp_path / "calib.jsonl")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(texts)
    assert [json.loads(line) for line in lines] == texts


# ---------------------------------------------------------------------------
# GGUF conversion, imatrix, Q4_K_M, INT4 AWQ: subprocess seams, no real tools
# ---------------------------------------------------------------------------


def _recording_run(returncode: int = 0, output: str = "ok", side_effect=None):
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float, cwd=None) -> tuple[int, str]:
        calls.append(argv)
        if side_effect is not None:
            side_effect(argv)
        return (returncode, output)

    return run, calls


def test_convert_gguf_calls_the_env_configured_tool(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, calls = _recording_run()
    out_file = tmp_path / "model-bf16.gguf"
    module.convert_gguf(run, tools, tmp_path / "hf-model", out_file)
    assert calls[0][0] == tools.convert
    assert str(out_file) in calls[0]


def test_convert_gguf_requests_bf16_not_f16(tmp_path) -> None:
    """Risk r14 / t16 spike: bf16, never f16."""
    module = quantize
    tools = TOOLS
    run, calls = _recording_run()
    module.convert_gguf(run, tools, tmp_path / "hf-model", tmp_path / "model-bf16.gguf")
    assert "bf16" in calls[0]
    assert "f16" not in calls[0]


def test_convert_gguf_raises_on_nonzero_exit(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, _ = _recording_run(returncode=1, output="boom")
    with pytest.raises(module.QuantizeError, match="boom"):
        module.convert_gguf(run, tools, tmp_path / "hf-model", tmp_path / "out.gguf")


def test_convert_gguf_refuses_a_vision_projector_output(tmp_path) -> None:
    """text-only, no mmproj: a tool that writes one anyway is a refusal, not silent success."""
    module = quantize
    tools = TOOLS
    out_file = tmp_path / "model-f16.gguf"

    def plants_mmproj(argv: list[str]) -> None:
        (tmp_path / "mmproj-model-f16.gguf").write_bytes(b"x")

    run, _ = _recording_run(side_effect=plants_mmproj)
    with pytest.raises(module.QuantizeError, match="mmproj"):
        module.convert_gguf(run, tools, tmp_path / "hf-model", out_file)


def test_compute_imatrix_calls_the_env_configured_tool(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, calls = _recording_run()
    module.compute_imatrix(
        run, tools, tmp_path / "model-f16.gguf", tmp_path / "calib.txt", tmp_path / "im.dat"
    )
    assert calls[0][0] == tools.imatrix


def test_compute_imatrix_raises_on_nonzero_exit(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, _ = _recording_run(returncode=2, output="bad calib")
    with pytest.raises(module.QuantizeError, match="bad calib"):
        module.compute_imatrix(
            run, tools, tmp_path / "model-f16.gguf", tmp_path / "calib.txt", tmp_path / "im.dat"
        )


def test_quantize_q4_k_m_calls_the_env_configured_tool_with_that_quant_name(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, calls = _recording_run()
    module.quantize_q4_k_m(
        run, tools, tmp_path / "model-f16.gguf", tmp_path / "im.dat", tmp_path / "model-q4km.gguf"
    )
    assert calls[0][0] == tools.quantize
    assert "Q4_K_M" in calls[0]


def test_quantize_q4_k_m_raises_on_nonzero_exit(tmp_path) -> None:
    module = quantize
    tools = TOOLS
    run, _ = _recording_run(returncode=1, output="bad imatrix")
    with pytest.raises(module.QuantizeError, match="bad imatrix"):
        module.quantize_q4_k_m(
            run,
            tools,
            tmp_path / "model-f16.gguf",
            tmp_path / "im.dat",
            tmp_path / "model-q4km.gguf",
        )


def test_export_awq_invokes_awq_py_with_the_oneshot_script(tmp_path) -> None:
    """Risk r14: a subprocess of AWQ_PY running awq_oneshot.py, never an llm-compressor CLI."""
    module = quantize
    run, calls = _recording_run()
    model_dir, calib_file, out_dir = tmp_path / "hf-model", tmp_path / "calib.txt", tmp_path / "awq"
    module.export_awq(run, "/opt/awq/bin/python", model_dir, calib_file, out_dir, 128)
    assert calls[0][0] == "/opt/awq/bin/python"
    assert calls[0][1] == str(module.AWQ_ONESHOT_SCRIPT)
    assert calls[0][1].endswith("awq_oneshot.py")
    assert str(model_dir) in calls[0]
    assert str(calib_file) in calls[0]
    assert str(out_dir) in calls[0]
    assert "128" in calls[0]


def test_export_awq_raises_on_nonzero_exit(tmp_path) -> None:
    module = quantize
    run, _ = _recording_run(returncode=1, output="unsupported layer")
    with pytest.raises(module.QuantizeError, match="unsupported layer"):
        module.export_awq(
            run,
            "/opt/awq/bin/python",
            tmp_path / "hf-model",
            tmp_path / "calib.txt",
            tmp_path / "awq",
            128,
        )


# ---------------------------------------------------------------------------
# vLLM support files + generation_config.json after the AWQ save (risk r14)
# ---------------------------------------------------------------------------


def test_copy_vllm_support_files_copies_present_files_resolving_symlinks(tmp_path) -> None:
    module = quantize
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    real = tmp_path / "blob-tokenizer.json"
    real.write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").symlink_to(real)
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")

    copied = module.copy_vllm_support_files(model_dir, out_dir)

    assert set(copied) == {"tokenizer.json", "vocab.json"}
    assert not (out_dir / "tokenizer.json").is_symlink()
    assert (out_dir / "tokenizer.json").read_text(encoding="utf-8") == "{}"


def test_copy_vllm_support_files_skips_files_not_present(tmp_path) -> None:
    module = quantize
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    (model_dir / "merges.txt").write_text("a b", encoding="utf-8")

    copied = module.copy_vllm_support_files(model_dir, out_dir)

    assert copied == ["merges.txt"]
    assert not (out_dir / "tokenizer.json").exists()


def test_finish_awq_export_writes_generation_config_and_reports_serve_args(tmp_path) -> None:
    module = quantize
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    result = module.finish_awq_export(model_dir, out_dir)

    assert result["copied_files"] == ["tokenizer_config.json"]
    assert result["serve_args"] == module.AWQ_SERVE_ARGS
    gen_config_path = out_dir / "generation_config.json"
    assert gen_config_path.is_file()
    payload = json.loads(gen_config_path.read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["do_sample"] is False


# ---------------------------------------------------------------------------
# Tool version recording (c44)
# ---------------------------------------------------------------------------


def test_tool_version_reports_the_run_output() -> None:
    module = quantize
    run, _ = _recording_run(output="convert.py version 1.2.3")
    assert module.tool_version(run, "/opt/llama/convert.py") == "convert.py version 1.2.3"


def test_tool_version_reports_unknown_on_failure() -> None:
    module = quantize
    run, _ = _recording_run(returncode=127, output="not found")
    assert "unknown" in module.tool_version(run, "/opt/llama/missing")


def test_llama_cpp_commit_reads_git_head_when_dir_is_set() -> None:
    module = quantize
    run, calls = _recording_run(output="abc123\n")
    commit = module.llama_cpp_commit(run, "/opt/llama.cpp")
    assert commit == "abc123"
    assert calls[0] == ["git", "-C", "/opt/llama.cpp", "rev-parse", "HEAD"]


def test_llama_cpp_commit_is_unknown_when_dir_is_not_set() -> None:
    module = quantize
    run, calls = _recording_run()
    commit = module.llama_cpp_commit(run, None)
    assert "unknown" in commit
    assert calls == []  # never runs git without a directory to point it at


def test_llama_cpp_commit_is_unknown_on_a_failed_git_call() -> None:
    module = quantize
    run, _ = _recording_run(returncode=128, output="not a git repository")
    commit = module.llama_cpp_commit(run, "/not/a/repo")
    assert "unknown" in commit
    assert "not a git repository" in commit


def test_awq_tool_versions_parses_the_probe_output() -> None:
    module = quantize
    run, calls = _recording_run(output="0.14.0 5.17.0\n")
    versions = module.awq_tool_versions(run, "/opt/awq/bin/python")
    assert versions == {"llm-compressor": "0.14.0", "transformers": "5.17.0"}
    assert calls[0][0] == "/opt/awq/bin/python"
    assert calls[0][1] == "-c"


def test_awq_tool_versions_is_unknown_on_failure() -> None:
    module = quantize
    run, _ = _recording_run(returncode=1, output="ModuleNotFoundError")
    versions = module.awq_tool_versions(run, "/opt/awq/bin/python")
    assert "unknown" in versions["llm-compressor"]
    assert "unknown" in versions["transformers"]


def test_record_tool_versions_covers_llama_cpp_and_awq() -> None:
    module = quantize
    tools = TOOLS
    run, calls = _recording_run(output="v1 v2")
    versions = module.record_tool_versions(
        run, tools, {"LLAMA_CPP_DIR": "/opt/llama.cpp"}, "/opt/awq/bin/python"
    )
    # 3 llama.cpp --version calls + 1 git rev-parse + 1 AWQ venv probe.
    assert len(calls) == 5
    assert set(versions) == {
        "llama.cpp convert",
        "llama.cpp imatrix",
        "llama.cpp quantize",
        "llama.cpp commit",
        "llm-compressor",
        "transformers",
    }


# ---------------------------------------------------------------------------
# heal_needed (c42, c43)
# ---------------------------------------------------------------------------


def test_heal_needed_false_when_quant_matches_bf16() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_drop_more_than_3_points() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=86.9, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_at_exactly_the_3_point_margin() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=87.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_a_new_wrong_mutating_proposal_appears() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_when_wrong_mutating_ids_do_not_change() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_false_when_a_wrong_mutating_id_is_fixed_and_none_added() -> None:
    """Fewer wrong mutating proposals, no new ones: the count drops, no healing needed."""
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_improve_but_wrong_mutating_appears() -> None:
    module = quantize
    bf16 = module.QuantSummary(right_pct=80.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=95.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_true_when_the_wrong_mutating_set_changes_with_equal_counts() -> None:
    """Codex finding #5: bf16 gets A wrong, quant fixes A but breaks B -- counts tie at 1."""
    module = quantize
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True


# -- the GGUF conversion source (issue 46, t25, P67) --


def _safetensors(path: Path, names: list[str]) -> None:
    header = json.dumps(
        {name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for name in names}
    )
    raw = header.encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0\0")


def _model_dir(tmp_path: Path, *, mtp_layers: int, tensors: list[str], nested: bool) -> Path:
    model = tmp_path / "merged"
    model.mkdir()
    text = {"num_hidden_layers": 24, "mtp_num_hidden_layers": mtp_layers}
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        **({"text_config": text} if nested else text),
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    _safetensors(model / "model.safetensors", tensors)
    return model


@pytest.mark.parametrize("nested", [True, False])
def test_a_declared_mtp_head_without_weights_is_detected(nested: bool, tmp_path: Path) -> None:
    """P67: the text-only merge keeps mtp_num_hidden_layers=1 but no mtp.* tensor."""
    model = _model_dir(tmp_path, mtp_layers=1, tensors=["model.layers.0.w"], nested=nested)
    assert quantize.mtp_without_weights(model) is True


def test_an_mtp_head_with_weights_is_kept(tmp_path: Path) -> None:
    model = _model_dir(tmp_path, mtp_layers=1, tensors=["mtp.fc.weight"], nested=True)
    assert quantize.mtp_without_weights(model) is False


def test_a_model_without_an_mtp_head_is_left_alone(tmp_path: Path) -> None:
    model = _model_dir(tmp_path, mtp_layers=0, tensors=["model.layers.0.w"], nested=True)
    assert quantize.mtp_without_weights(model) is False


@pytest.mark.parametrize("no_mtp", [True, False])
def test_convert_gguf_passes_no_mtp_only_when_asked(no_mtp: bool, tmp_path: Path) -> None:
    mod = quantize
    calls = []

    def run(argv, timeout, cwd=None):
        calls.append(argv)
        return 0, ""

    tools = mod.ToolPaths(convert="convert", quantize="q", imatrix="i")
    mod.convert_gguf(run, tools, tmp_path, tmp_path / "out.gguf", no_mtp=no_mtp)
    assert ("--no-mtp" in calls[0]) is no_mtp


# ---------------------------------------------------------------------------
# The heal trigger (behavioral o17): > 3 pts or a new wrong-mutating id, logged
# first, one round only
# ---------------------------------------------------------------------------


def _summary(right: float, ids=()) -> quantize.QuantSummary:
    return quantize.QuantSummary(right_pct=right, wrong_mutating_ids=frozenset(ids))


def _events(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


@pytest.mark.behavioral("o17")
def test_heal_fires_only_past_three_points_or_on_a_new_wrong_mutating_id() -> None:
    bf16 = _summary(90.0, {"a"})
    assert quantize.heal_needed(bf16, _summary(87.0, {"a"})) is False  # exactly the margin
    assert quantize.heal_needed(bf16, _summary(86.9, {"a"})) is True  # past it
    assert quantize.heal_needed(bf16, _summary(93.0, set())) is False  # better, id fixed
    assert quantize.heal_needed(bf16, _summary(90.0, {"a", "b"})) is True  # a new id
    assert quantize.heal_needed(bf16, _summary(90.0, {"b"})) is True  # swapped ids, equal counts
    # float dust on the margin must not fire it
    assert quantize.heal_needed(_summary(93.3), _summary(90.3)) is False


@pytest.mark.behavioral("o17")
def test_the_trigger_is_logged_before_the_caller_can_heal(tmp_path: Path) -> None:
    log = tmp_path / "heal-log.jsonl"
    trigger = quantize.heal_trigger(_summary(90.0), _summary(80.0), log=log, candidate="r3b")
    # the trigger is on disk by the time the caller holds it: nothing has healed yet
    assert not (tmp_path / "heal").exists()
    (event,) = _events(log)
    assert event["event"] == "heal_trigger"
    assert event["candidate"] == "r3b"
    assert event["loss_points"] == 10.0
    assert event["refused"] is False
    assert (trigger.heal_round, trigger.loss_points) == (1, 10.0)


@pytest.mark.behavioral("o17")
def test_a_second_heal_round_is_refused_and_the_refusal_is_logged(tmp_path: Path) -> None:
    log = tmp_path / "heal-log.jsonl"
    quantize.heal_trigger(_summary(90.0), _summary(80.0), log=log, candidate="r3b")
    bf16, quant = _summary(90.0), _summary(85.0, {"x"})
    with pytest.raises(quantize.QuantizeError, match="one round only"):
        quantize.heal_trigger(bf16, quant, log=log, candidate="r3b")
    first, second = _events(log)
    assert second["refused"] is True
    assert second["heal_round"] == 2
    assert second["new_wrong_mutating_ids"] == ["x"]
    # refusals are not spent rounds: the count stays at one, and a third try refuses too
    bf16, quant = _summary(90.0), _summary(70.0)
    with pytest.raises(quantize.QuantizeError):
        quantize.heal_trigger(bf16, quant, log=log, candidate="r3b")
    # a different build has its own round
    assert quantize.heal_trigger(_summary(90.0), _summary(70.0), log=log, candidate="r4")


@pytest.mark.behavioral("o17")
def test_a_build_that_is_itself_a_heal_is_refused_even_with_an_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "heal-log.jsonl"
    bf16, quant = _summary(90.0), _summary(70.0)
    with pytest.raises(quantize.QuantizeError, match="one round only"):
        quantize.heal_trigger(bf16, quant, log=log, candidate="r3b", heal_rounds=1)
    assert _events(log)[0]["refused"] is True


@pytest.mark.behavioral("o17")
def test_no_trigger_means_no_heal_and_no_spent_round(tmp_path: Path) -> None:
    log = tmp_path / "heal-log.jsonl"
    assert quantize.heal_trigger(_summary(90.0), _summary(89.0), log=log, candidate="r3b") is None
    assert _events(log) == [{"candidate": "r3b", "event": "heal_check", "needed": False}]
    assert quantize.heal_trigger(_summary(90.0), _summary(80.0), log=log, candidate="r3b")


@pytest.mark.behavioral("o17")
def test_the_margin_and_rounds_are_the_decide_rule_s_one_definition() -> None:
    assert quantize.HEAL_MARGIN_POINTS is rules.HEAL_MARGIN_POINTS
    assert quantize.HEAL_ROUNDS is rules.HEAL_ROUNDS
    # and the two verdict paths agree on a grid, edges included
    for bf, qt in [(0.90, 0.87), (0.90, 0.869), (0.90, 0.95), (0.5, 0.5)]:
        for b_ids, q_ids in [((), ()), (("a",), ("a",)), (("a",), ("b",)), ((), ("b",))]:
            check = rules.QuantCheck(
                "c", bf, qt, frozenset(b_ids), frozenset(q_ids), 0, "f.json", "0" * 64
            )
            assert check.heal_needed() == quantize.heal_needed(
                _summary(bf * 100, b_ids), _summary(qt * 100, q_ids)
            ), (bf, qt, b_ids, q_ids)


# ---------------------------------------------------------------------------
# The stage with stub binaries: bf16 GGUF, imatrix, Q4_K_M, relative paths, commit
# ---------------------------------------------------------------------------

_STUB_HEAD = f"#!{sys.executable}\nimport json, os, struct, sys\n"
_GGUF_WRITER = """
def gguf(path, meta):
    def s(t):
        raw = t.encode()
        return struct.pack("<Q", len(raw)) + raw
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(meta))
    for k, v in meta.items():
        out += s(k) + struct.pack("<I", 8) + s(v)
    open(path, "wb").write(out + b"\\0" * 16)
"""
_CONVERT = _STUB_HEAD + _GGUF_WRITER + """
a = sys.argv[1:]
if "--version" in a:
    print("convert stub 1"); sys.exit(0)
open(os.environ["STUB_LOG"], "a").write(json.dumps(["convert", a, os.getcwd()]) + "\\n")
gguf(a[a.index("--outfile") + 1], {"general.architecture": "qwen3"})
"""
_IMATRIX = _STUB_HEAD + """
a = sys.argv[1:]
if "--version" in a:
    print("imatrix stub 1"); sys.exit(0)
open(os.environ["STUB_LOG"], "a").write(json.dumps(["imatrix", a, os.getcwd()]) + "\\n")
out = a[a.index("-o") + 1]
open(out, "w").write(a[a.index("-f") + 1])  # the dataset name, as llama-imatrix records it
"""
# llama-quantize copies the imatrix file path and the dataset name into the GGUF verbatim;
# LEAK=1 makes it resolve them first, as a tool that absolutises would.
_QUANTIZE = _STUB_HEAD + _GGUF_WRITER + """
a = sys.argv[1:]
if "--version" in a:
    print("quantize stub 1"); sys.exit(0)
open(os.environ["STUB_LOG"], "a").write(json.dumps(["quantize", a, os.getcwd()]) + "\\n")
imatrix = a[a.index("--imatrix") + 1]
dataset = open(imatrix).read()
if os.environ.get("LEAK"):
    imatrix, dataset = os.path.abspath(imatrix), os.path.abspath(dataset)
gguf(a[-2], {"general.architecture": "qwen3", "quantize.imatrix.file": imatrix,
             "quantize.imatrix.dataset": dataset})
"""
_GIT = _STUB_HEAD + 'print("deadbeefcafe")\n'


def _stub(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _model(tmp: Path) -> Path:
    model = tmp / "merged"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "mtp_num_hidden_layers": 1}),
        encoding="utf-8",
    )
    header = json.dumps(
        {"model.layers.0.w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    )
    (model / "model.safetensors").write_bytes(
        len(header).encode()
        if False
        else len(header.encode()).to_bytes(8, "little") + header.encode() + b"\0\0"
    )
    return model


def _splits(tmp: Path) -> dict[str, Path]:
    return {
        "train": _write_split(
            tmp / "train.json", "train", [_entry("t1", text="a\nb"), _entry("t2")]
        ),
        "val": _write_split(tmp / "val.json", "val", [_entry("v1", text="val only")]),
        "test": _write_split(tmp / "test.json", "test", [_entry("s1", text="test only")]),
    }


@pytest.fixture
def stage(tmp_path, monkeypatch):
    tools = tmp_path / "llama.cpp"
    tools.mkdir()
    (tools / "build").mkdir()
    paths = {
        "llama_cpp_convert": _stub(tools / "convert_hf_to_gguf.py", _CONVERT),
        "llama_cpp_imatrix": _stub(tools / "build" / "llama-imatrix", _IMATRIX),
        "llama_cpp_quantize": _stub(tools / "build" / "llama-quantize", _QUANTIZE),
    }
    # git on PATH is a stub too: the recorded commit must come from it, never invented
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _stub(bindir / "git", _GIT)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.delenv("LEAK", raising=False)
    return {
        "config": make_config(**paths),
        "model": _model(tmp_path),
        "work": tmp_path / "work",
        "splits": _splits(tmp_path),
        "log": tmp_path / "calls.jsonl",
        "tmp": tmp_path,
    }


def _plan(stage, **kw):
    s = stage["splits"]
    return quantize.plan_quantize(
        config=stage["config"],
        model_dir=stage["model"],
        train=s["train"],
        val=s["val"],
        test=s["test"],
        work_dir=stage["work"],
        **kw,
    )


def _calls(stage) -> list[list]:
    return [json.loads(line) for line in stage["log"].read_text().splitlines()]


def test_the_plan_is_a_dry_run_that_writes_nothing(stage) -> None:
    plan = _plan(stage)
    result = quantize.run_quantize(plan)
    assert result["applied"] is False
    assert not stage["work"].exists()
    assert not stage["log"].exists()
    commands = result["plan"]["commands"]
    assert commands[0][-1] == "--no-mtp"  # declared MTP head, no mtp.* tensor
    assert commands[2][-1] == "Q4_K_M"
    assert result["plan"]["gguf_no_mtp"] is True
    assert result["plan"]["calibration_entries"] == 2


def test_the_plan_needs_its_tool_paths_from_the_config(stage) -> None:
    config = make_config(llama_cpp_convert=stage["config"]["llama_cpp_convert"])
    s = stage["splits"]
    with pytest.raises(CliError, match="llama_cpp_quantize"):
        quantize.plan_quantize(
            config=config, model_dir=stage["model"], train=s["train"], val=s["val"],
            test=s["test"], work_dir=stage["work"],
        )  # fmt: skip
    with pytest.raises(CliError, match="awq_python"):
        _plan(stage, with_awq=True)


def test_an_applied_stage_needs_a_guarded_runner(stage) -> None:
    plan = _plan(stage)
    with pytest.raises(quantize.QuantizeError, match="GPU stage"):
        quantize.run_quantize(plan, apply=True)


def test_the_stage_builds_bf16_then_imatrix_then_q4_k_m_and_records_the_commit(stage) -> None:
    result = quantize.run_quantize(_plan(stage), apply=True, run=quantize.default_run, env={})
    work = stage["work"]
    assert [c[0] for c in _calls(stage)] == ["convert", "imatrix", "quantize"]
    convert = _calls(stage)[0][1]
    assert "bf16" in convert
    assert "--no-mtp" in convert
    # the greedy generation config landed on the source before conversion
    assert json.loads((stage["model"] / "generation_config.json").read_text())["do_sample"] is False
    # train-side text only, one line per entry (newline collapsed) and JSONL for AWQ
    assert (work / "calibration.txt").read_text() == "a b\nhi\n"
    assert [json.loads(x) for x in (work / "calibration.jsonl").read_text().splitlines()] == [
        "a\nb",
        "hi",
    ]
    log = json.loads((work / "quantize-run.json").read_text())
    assert log == result["log"]
    # the llama.cpp commit comes from git in the converter's checkout; versions recorded
    assert log["tool_versions"]["llama.cpp commit"] == "deadbeefcafe"
    assert log["tool_versions"]["llama.cpp quantize"] == "quantize stub 1"
    assert "llm-compressor" not in log["tool_versions"]  # AWQ was not asked for


def test_the_commit_reads_llama_cpp_dir_when_exported(stage) -> None:
    calls = []

    def run(argv, timeout, cwd=None):
        calls.append(argv)
        return (0, "abc\n")

    tools = quantize.tool_paths_from_config(stage["config"])
    assert quantize.llama_cpp_dir(tools, {"LLAMA_CPP_DIR": "/srv/llama"}) == "/srv/llama"
    default = quantize.llama_cpp_dir(tools, {})
    assert default == str(Path(tools.convert).parent)
    quantize.record_tool_versions(run, tools, {}, None)
    assert ["git", "-C", default, "rev-parse", "HEAD"] in calls


def test_the_gguf_carries_no_absolute_path_in_its_imatrix_metadata(stage) -> None:
    quantize.run_quantize(_plan(stage), apply=True, run=quantize.default_run, env={})
    gguf = stage["work"] / "model-q4_k_m.gguf"
    meta = bundle.gguf_metadata(gguf)  # the bundle's own stdlib GGUF reader
    assert meta["quantize.imatrix.file"] == "imatrix.dat"
    assert meta["quantize.imatrix.dataset"] == "calibration.txt"
    assert not any(v.startswith("/") for k, v in meta.items() if k.startswith("quantize.imatrix"))
    assert bundle.gguf_absolute_paths(gguf) == []
    # imatrix and quantize ran in the work dir with relative arguments
    for name, argv, cwd in _calls(stage)[1:]:
        assert cwd == str(stage["work"].resolve())
        assert not any(arg.startswith("/") for arg in argv if arg != "Q4_K_M"), argv


def test_a_tool_that_absolutises_the_paths_is_refused(stage, monkeypatch) -> None:
    monkeypatch.setenv("LEAK", "1")
    plan = _plan(stage)
    with pytest.raises(quantize.QuantizeError, match="absolute path"):
        quantize.run_quantize(plan, apply=True, run=quantize.default_run, env={})


def test_a_failing_tool_stops_the_stage_with_its_output(stage, tmp_path) -> None:
    def run(argv, timeout, cwd=None):
        return (1, "boom") if argv[0] == stage["config"]["llama_cpp_imatrix"] else (0, "")

    plan = _plan(stage)
    with pytest.raises(quantize.QuantizeError, match="boom"):
        quantize.run_quantize(plan, apply=True, run=run, env={})


def test_awq_runs_as_a_subprocess_of_the_awq_python_and_is_finished_after(stage) -> None:
    calls = []

    def run(argv, timeout, cwd=None):
        calls.append(list(argv))
        if "--outfile" in argv:
            Path(argv[argv.index("--outfile") + 1]).write_bytes(b"x")
        if "--imatrix" in argv:
            from tests.release_support import write_gguf

            write_gguf(stage["work"] / argv[-2])
        if argv[0] == "awq-python" and "-c" not in argv:
            Path(argv[argv.index("--out-dir") + 1]).mkdir(parents=True)
        if argv[0] == "awq-python" and "-c" in argv:
            return (0, "0.14.0 5.17.0")
        return (0, "ok")

    config = make_config(**{**stage["config"].values, "awq_python": "awq-python"})
    s = stage["splits"]
    plan = quantize.plan_quantize(
        config=config, model_dir=stage["model"], train=s["train"], val=s["val"], test=s["test"],
        work_dir=stage["work"], with_awq=True,
    )  # fmt: skip
    result = quantize.run_quantize(plan, apply=True, run=run, env={})
    awq = next(c for c in calls if c[0] == "awq-python" and "--model-dir" in c)
    assert awq[1] == str(quantize.AWQ_ONESHOT_SCRIPT)
    assert awq[1].endswith("awq_oneshot.py")
    assert result["log"]["tool_versions"]["llm-compressor"] == "0.14.0"
    assert (
        json.loads((stage["work"] / "awq" / "generation_config.json").read_text())["do_sample"]
        is False
    )
    assert result["log"]["awq_serve_args"] == quantize.AWQ_SERVE_ARGS


def test_gpu_run_sends_each_command_through_the_guarded_stage_runner(tmp_path) -> None:
    seen = []

    class Done:
        returncode, stdout, stderr = 0, "out", "err"

    def runner(stage_name, run_dir, cmd, **kwargs):
        seen.append((stage_name, run_dir, cmd, kwargs))
        return Done()

    run = quantize.gpu_run(tmp_path, memory_max="24G", runner=runner)
    assert run(["tool", "x"], 5.0) == (0, "outerr")
    assert run(["tool", "y"], 5.0, cwd=tmp_path) == (0, "outerr")
    plain, wrapped = seen
    assert plain[0] == "quantize"
    assert plain[2] == ["tool", "x"]
    assert plain[3]["memory_max"] == "24G"
    assert plain[3]["allow_foreign"] is False
    assert wrapped[2][:2] == ["bash", "-c"]
    assert wrapped[2][-3:] == [str(tmp_path), "tool", "y"][-3:]


def test_gpu_run_reports_a_watchdog_stop(tmp_path) -> None:
    class Done:
        returncode, stdout, stderr = 3, "", "low memory"

    run = quantize.gpu_run(tmp_path, memory_max="24G", runner=lambda *a, **k: Done())
    code, output = run(["tool"], 1.0)
    assert code == 3
    assert "watchdog" in output


def test_gpu_run_really_runs_a_command_in_the_work_dir(tmp_path, monkeypatch) -> None:
    """The cd wrapper end to end through run_gpu_stage's capped runner (no GPU to guard)."""
    from jev_factory.factory import gpu

    if gpu.run_capped(tmp_path, ["true"], memory_max="8G").returncode != 0:
        pytest.skip("run_capped needs systemd-run user scope here")
    smi = tmp_path / "smibin"
    smi.mkdir()
    _stub(smi / "nvidia-smi", "#!/bin/sh\nexit 0\n")  # an empty GPU, whatever this box has
    monkeypatch.setenv("PATH", f"{smi}:{os.environ['PATH']}")
    work = tmp_path / "w"
    work.mkdir()
    run = quantize.gpu_run(tmp_path, memory_max="8G")
    code, _ = run(["sh", "-c", "pwd > where.txt"], 30.0, cwd=work)
    assert code == 0
    assert (work / "where.txt").read_text().strip() == str(work)


def test_provenance_header() -> None:
    assert quantize.NVSH_PROVENANCE["upstream"] == "scripts/lfm-finetune/quantize.py"
    assert quantize.NVSH_PROVENANCE["commit"] == "9debdc6"
