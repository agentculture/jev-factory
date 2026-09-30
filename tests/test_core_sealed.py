"""Sealed held-out and test side loaders expose ids, counts and sha256 only."""

from __future__ import annotations

import hashlib
import json

import pytest

from jev_factory.core import split as split_module

SEALED_TEXT = "sealed wording that must never be disclosed"


def _sealed(tmp_path, name="held-out.json"):
    entries = [
        {"id": "s1", "text": SEALED_TEXT, "expect": {"operation": "lamp_status", "args": {}}},
        {"id": "s2", "text": SEALED_TEXT + " two", "expect": {"escalate": True}},
        {"id": "s3", "text": SEALED_TEXT + " three", "expect": {"explain": True}},
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    return path


@pytest.mark.behavioral("o10")
def test_sealed_loader_exposes_only_ids_counts_and_sha256(tmp_path) -> None:
    path = _sealed(tmp_path)
    side = split_module.load_sealed(path)
    assert side.ids == ("s1", "s2", "s3")
    assert side.count == 3
    assert side.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dict(side.by_kind) == {"operation": 1, "escalate": 1, "explain": 1}
    disclosed = repr(side) + str(side) + json.dumps(side.__dict__, default=str)
    assert SEALED_TEXT not in disclosed
    assert {f for f in side.__dataclass_fields__} == {"path", "sha256", "ids", "count", "by_kind"}


@pytest.mark.behavioral("o10")
def test_test_side_loader_is_the_same_text_free_view(tmp_path) -> None:
    side = split_module.load_sealed(_sealed(tmp_path, "test.json"))
    assert SEALED_TEXT not in repr(side) and side.count == 3


@pytest.mark.behavioral("o10")
def test_malformed_sealed_file_error_names_the_path_and_never_the_content(tmp_path) -> None:
    bad = tmp_path / "held-out.json"
    bad.write_text('{"entries": [{"id": "x", "text": "' + SEALED_TEXT + '"}]}', encoding="utf-8")
    with pytest.raises(ValueError) as err:
        split_module.load_sealed(bad)  # no 'expect' block: not a readable split
    assert SEALED_TEXT not in str(err.value)
    assert "held-out.json" in str(err.value)


@pytest.mark.behavioral("o10")
def test_the_held_out_split_is_refused_as_a_split_input(tmp_path) -> None:
    path = _sealed(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        split_module.build_splits(path)
    with pytest.raises(ValueError, match="held-out"):
        split_module.merge_corpora([path])
