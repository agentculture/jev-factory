"""docs/distill-bundle-format.md lists every file the bundle stage writes (t18, c11/h11).

The file lists come from the code: a fake bundle is built with the real ``build_bundles`` in
both deployed formats, and the constants of ``distill.bundle`` and ``pointer.quantized`` are
read, so a file the stage starts writing without a doc line fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.distill import bundle as B
from tests.test_distill_bundle import build

DOC = Path(__file__).resolve().parent.parent / "docs" / "distill-bundle-format.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _code_spans(text: str) -> set[str]:
    return set(re.findall(r"`([^`\n]+)`", text))


def written_files(tmp_path: Path, fmt: str) -> dict[str, set[str]]:
    """Every file ``build_bundles`` wrote, per upload folder, relative to that folder."""
    _, out, _ = build(tmp_path, fmt=fmt)
    return {
        folder: {
            p.relative_to(out / folder).as_posix() for p in (out / folder).rglob("*") if p.is_file()
        }
        for folder in (B.MODEL_DIR, B.DATASET_DIR)
    }


def undocumented(names: set[str], text: str) -> list[str]:
    spans = _code_spans(text)
    return sorted(n for n in names if n not in spans)


@pytest.mark.parametrize("fmt", ["bf16", "int8"])
def test_every_file_the_bundle_stage_writes_is_in_the_doc(tmp_path, fmt):
    """AC2: a file written by the bundle stage that the doc does not list fails."""
    written = written_files(tmp_path, fmt)
    for folder, names in written.items():
        assert names, folder
        assert undocumented(names, _doc()) == [], folder


def test_the_check_fails_on_a_file_the_doc_does_not_list(tmp_path):
    """AC2: the guard itself reports an extra file (not vacuous)."""
    names = written_files(tmp_path, "int8")[B.MODEL_DIR] | {"surprise.bin"}
    assert undocumented(names, _doc()) == ["surprise.bin"]


def test_every_required_and_tokenizer_file_constant_is_in_the_doc():
    """AC1: the files the readers require or carry, from the constants."""
    names = {
        *qz.REQUIRED_FILES,
        *qz.TOKENIZER_ALL,
        *player.REQUIRED_FILES,
        *player.OPTIONAL_FILES,
        player.MANIFEST,
        B.NOTICE_FILE,
        B.BUNDLE_FILE,
    }
    assert undocumented(names, _doc()) == []


def test_the_doc_names_the_quant_scheme_per_tensor_class():
    """AC1: the artifact format, scheme defaults and tensor policy match the code."""
    spans = _code_spans(_doc())
    text = _doc()
    scheme = qz.QuantScheme()
    assert qz.ARTIFACT_FORMAT in spans
    assert player.FORMAT in spans
    for field, value in scheme.record().items():
        assert f"`{field}` {value}" in text or f"`{field}`: {value}" in text, field
    for name in (*qz.QUANTIZED_LINEARS, *qz.KEPT_NAMES, *qz.DROP_SEGMENTS, *qz.EMBEDDING_MODULES):
        assert name in spans, name
    assert qz.HEAD_DTYPE in text
    assert qz.EMBED_QWEIGHT in spans
    assert qz.EMBED_SCALES in spans


def test_the_doc_names_the_tokenizer_and_the_serve_contract():
    """AC1: tokenizer, loader/serve entry point and the three endpoints."""
    spans = _code_spans(_doc())
    text = _doc()
    assert f"python -m {qz.MODULE} serve" in text
    for route in ("POST /v1/systemone", "GET /v1/bundle", "GET /v1/stats", "GET /health"):
        assert route in spans, route
    assert qz.LOOPBACK in text
    for key in ("rss_bytes", "mem_available_bytes", "device_model", "nv_tegra_release", "cgroup"):
        assert key in spans, key
    assert "unmodified" in text
    assert "full-vocabulary" in text


def test_the_pruned_build_is_described_as_a_comparison_never_bundled():
    """Boundary: a pruned-vocabulary build is a comparison artifact, not a bundle member."""
    text = _doc().lower()
    assert "pruned" in text
    assert "never bundled" in text
