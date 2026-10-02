"""The eval-slice builder: jev_factory/measure/slices.py (ported from nvsh's
tests/test_lfm_finetune_eval_slices.py; the CLI takes the toy domain)."""

from __future__ import annotations

import copy
import json

from jev_factory.measure import slices
from tests.fixtures.toy_domain import DOMAIN


def _entry(entry_id: str, expect: dict | None = None) -> dict:
    return {
        "id": entry_id,
        "kind": "explicit",
        "text": f"fixture text for {entry_id}",
        "expect": expect if expect is not None else {"escalate": True},
        "source": "fixture",
    }


def _op(entry_id: str) -> dict:
    return _entry(entry_id, {"operation": "lamp_status", "args": {}})


def _split() -> dict:
    return {"header": "Fixture split", "entries": [_op("op01"), _op("op02"), _entry("esc01")]}


def test_every_slice_entry_text_equals_its_source_entry_text():
    split = _split()
    sources = {e["id"]: e["text"] for e in split["entries"]}
    for entry in slices.missing_candidate_slice(split, DOMAIN.names())["entries"]:
        assert entry["text"] == sources[entry["source_id"]]


def test_every_slice_candidates_equals_all_ops_minus_gold():
    split = {"header": "x", "entries": [_op("op01")]}
    (entry,) = slices.missing_candidate_slice(split, ("op_a", "op_b", "lamp_status"))["entries"]
    assert entry["candidates"] == ["op_a", "op_b"]


def test_entries_without_operation_expectation_are_skipped():
    ids = {e["id"] for e in slices.missing_candidate_slice(_split(), DOMAIN.names())["entries"]}
    assert ids == {"op01-nocand", "op02-nocand"}


def test_every_slice_entry_expects_escalate():
    for entry in slices.missing_candidate_slice(_split(), DOMAIN.names())["entries"]:
        assert entry["expect"] == {"escalate": True}


def test_input_dict_is_unchanged_after_the_call():
    split = _split()
    original = copy.deepcopy(split)
    slices.missing_candidate_slice(split, DOMAIN.names())
    assert split == original


def test_main_writes_the_file_and_prints_entries_n(tmp_path, capsys):
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(_split()), encoding="utf-8")
    out_path = tmp_path / "out" / "slice.json"
    argv = ["--domain", "tests.fixtures.toy_domain", "--split", str(split_path)]
    assert slices.main(argv + ["--out", str(out_path)]) == 0
    assert "entries=2" in capsys.readouterr().out
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2
    for entry in payload["entries"]:
        assert entry["candidates"] == [n for n in DOMAIN.names() if n != "lamp_status"]
