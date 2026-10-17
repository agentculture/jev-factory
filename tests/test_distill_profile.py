"""Student profile: DEFAULT_2610 equals today's constants; loading fails closed (t2)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout, quantized
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark, contamination, dataset, pins, prereg, profile

P = profile.DEFAULT_2610


def _table() -> dict:
    d = dataclasses.asdict(P)
    d["lora_targets"] = list(P.lora_targets)
    d["temperature_by_kind"] = dict(P.temperature_by_kind)
    return d


def test_default_2610_matches_current_constants() -> None:
    assert P.base_repo == pins.BASE_REPO == "Qwen/Qwen3.5-2B-Base"
    assert P.base_revision == pins.DEFAULT_BASE_REVISION
    assert P.decider_repo == pins.DECIDER_REPO
    assert P.decider_revision == pins.DECIDER_REVISION
    assert P.adapter_sha256 == pins.DECIDER_ADAPTER_SHA256
    assert P.head_sha256 == pins.DECIDER_HEAD_SHA256
    assert (P.adapter_file, P.head_file, P.provenance_file) == (
        pins.ADAPTER_FILE,
        pins.HEAD_FILE,
        pins.PROVENANCE_FILE,
    )
    assert (P.adapter_file, P.adapter_config_file, P.head_file) == (
        layout.ADAPTER,
        layout.ADAPTER_CONFIG,
        layout.HEAD,
    )
    assert (P.decider_config_file, P.provenance_file) == (layout.CONFIG, layout.PROVENANCE)
    assert P.train_config_file == prereg.TRAIN_CONFIG
    assert P.decider_config_file == prereg.DECIDER_CONFIG
    assert P.adapter_config_file == prereg.ADAPTER_CONFIG
    assert P.upstream_model == prereg.UPSTREAM_MODEL
    assert P.upstream_revision == prereg.UPSTREAM_REVISION
    assert P.upstream_source == prereg.UPSTREAM_SOURCE
    assert P.lora_targets == tuple(prereg.STOCK_LORA_TARGETS)
    assert dict(P.temperature_by_kind) == prereg.STOCK_TEMPERATURE_BY_KIND
    knobs = {k["name"]: k["value"] for k in prereg.example_doc()["upstream_knobs"]}
    assert (P.lora_r, P.lora_alpha, P.lora_dropout) == (
        knobs["lora_r"],
        knobs["lora_alpha"],
        knobs["lora_dropout"],
    )
    assert P.max_tokens == knobs["max_length"] == dataset.MAX_TOKENS
    assert P.num_slots == dataset.DEFAULT_MAX_SLOTS
    assert P.weight_budget_bytes == quantized.SIZE_BUDGET_BYTES
    assert P.card == profile.CardBaseline(
        benchmark.CARD_N, benchmark.CARD_CORRECT, benchmark.CARD_TASKS_SHA256
    )
    assert Path(P.contamination_manifest) == contamination.MANIFEST_PATH
    assert P.backbone == "pointer"


def test_default_pins_take_a_profile() -> None:
    d = pins.default_pins(decider_revision=pins.DECIDER_REVISION)
    assert d == pins.default_pins(decider_revision=pins.DECIDER_REVISION, profile=P)
    other = dataclasses.replace(P, base_repo="x/other", adapter_sha256="a" * 64)
    o = pins.default_pins(decider_revision="c0ffee11", profile=other)
    assert o.base.repo_id == "x/other" and o.adapter_sha256 == "a" * 64


def test_round_trip_from_table_and_run_toml() -> None:
    assert profile.profile_from_table(_table()) == P
    assert profile.profile_from_run_toml({profile.RUN_TOML_KEY: _table()}) == P
    assert profile.profile_from_run_toml({}) is P


def test_load_from_toml_file(tmp_path: Path) -> None:
    t = _table()
    t.pop("card")
    lines = []
    for k, v in t.items():
        if isinstance(v, dict):
            continue
        lines.append(f"{k} = {_toml(v)}")
    lines.append("[temperature_by_kind]")
    lines += [f"{k} = {v!r}" for k, v in t["temperature_by_kind"].items()]
    f = tmp_path / "p.toml"
    f.write_text("\n".join(lines) + "\n")
    got = profile.load_profile_file(f)
    assert got == dataclasses.replace(P, card=None)


def _toml(v: object) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\") + '"'
    return repr(v)


def _refused(table: dict, field: str) -> None:
    with pytest.raises(CliError) as exc:
        profile.profile_from_table(table)
    assert exc.value.code == EXIT_USER_ERROR
    assert field in exc.value.message


def test_unknown_key_refused() -> None:
    _refused({**_table(), "bogus": 1}, "bogus")


def test_missing_required_refused() -> None:
    for name in ("base_repo", "adapter_sha256", "num_slots", "lora_targets"):
        t = _table()
        del t[name]
        _refused(t, name)


def test_malformed_sha256_refused() -> None:
    _refused({**_table(), "head_sha256": "abc123"}, "head_sha256")
    _refused({**_table(), "adapter_sha256": "A" * 64}, "adapter_sha256")
    _refused({**_table(), "card": {"n": 1, "correct": 1, "tasks_sha256": "zz"}}, "tasks_sha256")


def test_bad_type_refused() -> None:
    _refused({**_table(), "num_slots": "24"}, "num_slots")


def test_unreadable_file_refused(tmp_path: Path) -> None:
    with pytest.raises(CliError) as exc:
        profile.load_profile_file(tmp_path / "missing.toml")
    assert exc.value.code == EXIT_USER_ERROR
