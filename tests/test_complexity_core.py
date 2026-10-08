"""Characterization tests pinning behaviour before the d17 complexity refactor.

Each test here records what a function did at a0e4bda for a branch (or a
conditional-expression arm) the existing suite did not exercise, so the
behaviour-neutral refactor of gen_config, the causal-LM scorer, calibration,
merge_variations, predictions and split can be checked against it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import gen_config, scorer
from jev_factory.core import calibration, merge_variations
from jev_factory.core import predictions as pr
from jev_factory.core import split
from jev_factory.domain.model import ESCALATE_LABEL, EXPLAIN_LABEL

# ---------------------------------------------------------------------------
# gen_config._with_tokenizer_eos
# ---------------------------------------------------------------------------


@pytest.fixture
def _turn_end(monkeypatch):
    def set_id(value):
        monkeypatch.setattr(gen_config, "_tokenizer_eos_id", lambda _model_dir: value)

    return set_id


def test_eos_unchanged_when_tokenizer_eos_unreadable(_turn_end) -> None:
    _turn_end(None)
    ids = {"eos_token_id": 5}
    assert gen_config._with_tokenizer_eos(ids, Path(".")) is ids


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (None, 7),
        ([], 7),
        (5, [7, 5]),
        ([5, 6], [7, 5, 6]),
    ],
)
def test_eos_turn_end_is_put_first(_turn_end, current, expected) -> None:
    _turn_end(7)
    out = gen_config._with_tokenizer_eos({"eos_token_id": current, "bos_token_id": 1}, Path("."))
    assert out == {"eos_token_id": expected, "bos_token_id": 1}


def test_eos_key_absent_gets_the_turn_end_as_a_scalar(_turn_end) -> None:
    _turn_end(7)
    assert gen_config._with_tokenizer_eos({}, Path(".")) == {"eos_token_id": 7}


@pytest.mark.parametrize("current", [7, [7], [5, 7]])
def test_eos_already_listed_returns_ids_itself(_turn_end, current) -> None:
    _turn_end(7)
    ids = {"eos_token_id": current}
    assert gen_config._with_tokenizer_eos(ids, Path(".")) is ids


# ---------------------------------------------------------------------------
# calibration.split_markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, set()),
        ("", set()),
        ({}, set()),
        ([], set()),
        (0, set()),
        ("a test side", {"test"}),
        ("the HELD-OUT side", {"held-out"}),
        ({"note": "test"}, {"test"}),
        ({"note": "held_out"}, {"held-out"}),
        (["test", "held out"], {"test", "held-out"}),
        (1, set()),
    ],
)
def test_split_markers_header_shapes(header, expected) -> None:
    assert calibration.split_markers(Path("out/val.predictions.jsonl"), header) == expected


def test_split_markers_combine_name_and_header() -> None:
    markers = calibration.split_markers(Path("run-test.json"), {"side": "held-out"})
    assert markers == {"test", "held-out"}


# ---------------------------------------------------------------------------
# scorer.parse_top_logprobs
# ---------------------------------------------------------------------------


def _reply(logprobs: object) -> dict:
    return {"choices": [{"logprobs": logprobs}]}


@pytest.mark.parametrize(
    "reply",
    [
        None,
        [],
        "text",
        {},
        {"choices": None},
        {"choices": {}},
        {"choices": []},
        {"choices": ["x"]},
        {"choices": [{}]},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": [1]}]},
    ],
)
def test_parse_top_logprobs_refuses_a_reply_without_logprobs(reply) -> None:
    with pytest.raises(scorer.ServedError, match="^the server reply carries no logprobs$"):
        scorer.parse_top_logprobs(reply)


def test_parse_top_logprobs_legacy_shape_keeps_only_valid_entries() -> None:
    top = {
        "A": -0.5,
        "B": 0.5,
        "C": True,
        "D": float("nan"),
        7: -1.0,
        "E": "-1",
        "F": 0,
        "G": -3,
    }
    scores = scorer.parse_top_logprobs(_reply({"top_logprobs": [top, {"Z": -0.1}]}))
    assert scores == {"A": -0.5, "F": 0.0, "G": -3.0}
    assert isinstance(scores["G"], float)


def test_parse_top_logprobs_legacy_shape_all_invalid_raises() -> None:
    reply = _reply({"top_logprobs": [{"A": 1.0, "B": None}]})
    with pytest.raises(scorer.ServedError, match="^the server reply carries no logprobs$"):
        scorer.parse_top_logprobs(reply)


def test_parse_top_logprobs_content_shape_keeps_only_valid_entries() -> None:
    entries = [
        {"token": "A", "logprob": -0.2},
        "junk",
        {"token": 1, "logprob": -1.0},
        {"token": "B", "logprob": 0.1},
        {"token": "C"},
        {"logprob": -0.3},
        {"token": "D", "logprob": -2},
        {"token": "A", "logprob": -0.4},
    ]
    scores = scorer.parse_top_logprobs(
        _reply({"content": [{"top_logprobs": entries}, {"top_logprobs": []}]})
    )
    assert scores == {"A": -0.4, "D": -2.0}


@pytest.mark.parametrize(
    "top",
    [None, [], ["x"], [[{"A": -0.1}]], {"A": -0.1}],
)
def test_parse_top_logprobs_falls_back_to_content_when_legacy_is_unusable(top) -> None:
    logprobs = {"top_logprobs": top, "content": [{"top_logprobs": [{"token": "Q", "logprob": -1}]}]}
    assert scorer.parse_top_logprobs(_reply(logprobs)) == {"Q": -1.0}


def test_parse_top_logprobs_legacy_shape_wins_over_content() -> None:
    logprobs = {
        "top_logprobs": [{"L": -0.1}],
        "content": [{"top_logprobs": [{"token": "Q", "logprob": -1}]}],
    }
    assert scorer.parse_top_logprobs(_reply(logprobs)) == {"L": -0.1}


def test_parse_top_logprobs_legacy_empty_first_dict_does_not_fall_back() -> None:
    logprobs = {
        "top_logprobs": [{}],
        "content": [{"top_logprobs": [{"token": "Q", "logprob": -1}]}],
    }
    with pytest.raises(scorer.ServedError, match="carries no logprobs"):
        scorer.parse_top_logprobs(_reply(logprobs))


@pytest.mark.parametrize(
    "content",
    [
        None,
        {},
        [],
        ["x"],
        [{}],
        [{"top_logprobs": None}],
        [{"top_logprobs": {"token": "A", "logprob": -1}}],
        [{"top_logprobs": []}],
        [{"top_logprobs": [{"token": "A", "logprob": 1}]}],
    ],
)
def test_parse_top_logprobs_content_shape_without_scores_raises(content) -> None:
    with pytest.raises(scorer.ServedError, match="^the server reply carries no logprobs$"):
        scorer.parse_top_logprobs(_reply({"content": content}))


def test_parse_top_logprobs_reads_only_the_first_choice() -> None:
    reply = {
        "choices": [
            {"logprobs": {"top_logprobs": [{"A": -0.1}]}},
            {"logprobs": {"top_logprobs": [{"B": -0.1}]}},
        ]
    }
    assert scorer.parse_top_logprobs(reply) == {"A": -0.1}


# ---------------------------------------------------------------------------
# merge_variations.merge
# ---------------------------------------------------------------------------

_TRAIN = "Dev. Split 'train' of dev.json (seed=39)."


def _split_payload() -> dict:
    return {
        "header": _TRAIN,
        "entries": [
            {"id": "s1", "text": "Turn the lamp on.", "expect": {"operation": "on"}},
            {"id": "s2", "text": "Is it on?", "expect": {"operation": "status"}},
        ],
    }


def _var(vid: str, text: str, source: str = "s1", **over) -> dict:
    row = {
        "id": vid,
        "text": text,
        "source_id": source,
        "side": "train",
        "expect": {"operation": "on"},
        "models": ["m"],
        "seed_format": "x",
        "verdicts": [1],
        "extra": "kept",
    }
    row.update(over)
    return row


def test_merge_counts_every_outcome_and_strips_review_fields() -> None:
    variations = [
        _var("v1", "Switch the lamp on"),
        _var("v2", "switch  the LAMP on!"),
        _var("v3", "Leaky wording"),
        _var("v4", "Off split", source="nope"),
        _var("v5", "Lamp, please, on"),
    ]
    merged, counts = merge_variations.merge(
        _split_payload(),
        variations,
        exclude=frozenset({"leaky wording"}),
        filter_to_split=True,
    )
    assert counts == {"kept": 2, "duplicate": 1, "leaked": 1, "off_split": 1}
    assert [e["id"] for e in merged["entries"]] == ["s1", "s2", "v1", "v5"]
    assert merged["entries"][2] == {
        "id": "v1",
        "text": "Switch the lamp on",
        "source_id": "s1",
        "side": "train",
        "expect": {"operation": "on"},
        "extra": "kept",
    }
    assert merged["header"] == f"{_TRAIN} Plus 2 accepted variations (augment.py)."


def test_merge_a_leaked_repeat_of_the_source_counts_as_leaked() -> None:
    _, counts = merge_variations.merge(
        _split_payload(),
        [_var("v1", "turn the lamp on")],
        exclude=frozenset({"turn the lamp on"}),
    )
    assert counts == {"kept": 0, "duplicate": 0, "leaked": 1, "off_split": 0}


def test_merge_checks_the_id_before_the_side() -> None:
    variations = [_var("s1", "x", side="val")]
    with pytest.raises(ValueError, match="variation id is already used"):
        merge_variations.merge(_split_payload(), variations)


def test_merge_checks_the_side_before_the_source() -> None:
    variations = [_var("v1", "x", source="nope", side="val")]
    with pytest.raises(ValueError, match="^v1: side 'val' is not train$"):
        merge_variations.merge(_split_payload(), variations, filter_to_split=True)


def test_merge_an_off_split_variation_is_still_refused_without_the_filter() -> None:
    variations = [_var("v1", "x", source="nope")]
    with pytest.raises(ValueError, match="^v1: source 'nope' is not in the split$"):
        merge_variations.merge(_split_payload(), variations)


def test_merge_an_off_split_variation_still_claims_its_id() -> None:
    variations = [_var("v1", "x", source="nope"), _var("v1", "y")]
    with pytest.raises(ValueError, match="^'v1': variation id is already used"):
        merge_variations.merge(_split_payload(), variations, filter_to_split=True)


def test_merge_an_expect_mismatch_is_refused() -> None:
    variations = [_var("v1", "x", expect={"operation": "off"})]
    with pytest.raises(ValueError, match="^v1: expected answer differs from its source$"):
        merge_variations.merge(_split_payload(), variations)


# ---------------------------------------------------------------------------
# predictions.Prediction.from_dict
# ---------------------------------------------------------------------------


def _row(**over) -> dict:
    row = {
        "id": "e1",
        "expected": {"operation": "lamp_status", "args": {}},
        "outcome": "propose",
        "operation": "lamp_status",
        "arguments": {},
        "candidates": {"lamp_status": 0.7, EXPLAIN_LABEL: 0.2, ESCALATE_LABEL: 0.1},
        "tokens": 0,
        "ttfd_ms": 1.0,
        "latency_ms": 2.0,
    }
    row.update(over)
    return row


@pytest.mark.parametrize("row", [None, [], "x", 3])
def test_from_dict_refuses_a_non_object(row) -> None:
    with pytest.raises(pr.PredictionError, match="^a line must be a JSON object$"):
        pr.Prediction.from_dict(row)


def test_from_dict_names_every_missing_field_in_order() -> None:
    row = _row()
    del row["tokens"]
    del row["id"]
    with pytest.raises(pr.PredictionError, match="^missing field\\(s\\): id, tokens$"):
        pr.Prediction.from_dict(row)


@pytest.mark.parametrize("value", ["", None, 3])
def test_from_dict_refuses_a_bad_id(value) -> None:
    with pytest.raises(pr.PredictionError, match="^id must be a non-empty string$"):
        pr.Prediction.from_dict(_row(id=value))


def test_from_dict_checks_id_before_expected() -> None:
    with pytest.raises(pr.PredictionError, match="^id must be"):
        pr.Prediction.from_dict(_row(id="", expected=None))


def test_from_dict_refuses_an_unknown_outcome() -> None:
    with pytest.raises(pr.PredictionError, match="^outcome 'maybe' is not one of propose, "):
        pr.Prediction.from_dict(_row(outcome="maybe"))


@pytest.mark.parametrize("value", [True, -1, 1.0, "1", None])
def test_from_dict_refuses_bad_tokens(value) -> None:
    with pytest.raises(pr.PredictionError, match="^tokens must be a non-negative integer$"):
        pr.Prediction.from_dict(_row(tokens=value))


@pytest.mark.parametrize("name", ["ttfd_ms", "latency_ms"])
@pytest.mark.parametrize("value", [True, -0.5, "1", None])
def test_from_dict_refuses_bad_timings(name, value) -> None:
    with pytest.raises(pr.PredictionError, match=f"^{name} must be a non-negative number$"):
        pr.Prediction.from_dict(_row(**{name: value}))


def test_from_dict_checks_tokens_before_timings() -> None:
    with pytest.raises(pr.PredictionError, match="^tokens"):
        pr.Prediction.from_dict(_row(tokens=-1, ttfd_ms=-1, invalid_reason=3))


def test_from_dict_checks_ttfd_before_latency() -> None:
    with pytest.raises(pr.PredictionError, match="^ttfd_ms"):
        pr.Prediction.from_dict(_row(ttfd_ms=-1, latency_ms=-1))


@pytest.mark.parametrize("value", [3, ["x"], {}])
def test_from_dict_refuses_a_non_string_invalid_reason(value) -> None:
    with pytest.raises(pr.PredictionError, match="^invalid_reason must be a string$"):
        pr.Prediction.from_dict(_row(invalid_reason=value))


def test_from_dict_checks_invalid_reason_before_offered() -> None:
    with pytest.raises(pr.PredictionError, match="^invalid_reason"):
        pr.Prediction.from_dict(_row(invalid_reason=3, offered="bad"))


@pytest.mark.parametrize("value", [0, 1, "yes", []])
def test_from_dict_refuses_a_non_bool_grounded(value) -> None:
    with pytest.raises(pr.PredictionError, match="^grounded must be true, false or null$"):
        pr.Prediction.from_dict(_row(grounded=value))


def test_from_dict_checks_raw_probabilities_before_grounded() -> None:
    with pytest.raises(pr.PredictionError, match="^raw_probabilities"):
        pr.Prediction.from_dict(_row(raw_probabilities={"lamp_status": 2.0}, grounded=1))


def test_from_dict_builds_a_record_with_copies_and_float_timings() -> None:
    row = _row(
        outcome="explain",
        operation=None,
        arguments=None,
        candidates=None,
        tokens=3,
        ttfd_ms=4,
        latency_ms=5,
        invalid_reason="why",
        grounded=False,
        raw_scores=None,
        raw_probabilities=None,
        unknown="ignored",
    )
    p = pr.Prediction.from_dict(row)
    assert p == pr.Prediction(
        id="e1",
        expected={"operation": "lamp_status", "args": {}},
        outcome="explain",
        operation=None,
        arguments=None,
        candidates=None,
        tokens=3,
        ttfd_ms=4.0,
        latency_ms=5.0,
        invalid_reason="why",
        offered=None,
        raw_scores=None,
        raw_probabilities=None,
        grounded=False,
    )
    assert isinstance(p.ttfd_ms, float)
    assert isinstance(p.latency_ms, float)


def test_from_dict_copies_every_mapping() -> None:
    raw_scores = {"lamp_status": -0.3, EXPLAIN_LABEL: -1.6, ESCALATE_LABEL: -2.3}
    raw_probabilities = {"lamp_status": 0.6, EXPLAIN_LABEL: 0.25, ESCALATE_LABEL: 0.15}
    row = _row(
        arguments={"a": 1},
        offered=["lamp_status", EXPLAIN_LABEL, ESCALATE_LABEL],
        raw_scores=raw_scores,
        raw_probabilities=raw_probabilities,
        grounded=True,
    )
    p = pr.Prediction.from_dict(row)
    assert p.expected == row["expected"]
    assert p.expected is not row["expected"]
    assert p.arguments == {"a": 1}
    assert p.arguments is not row["arguments"]
    assert p.candidates is not row["candidates"]
    assert p.raw_scores == raw_scores
    assert p.raw_scores is not raw_scores
    assert p.raw_probabilities == raw_probabilities
    assert p.raw_probabilities is not raw_probabilities
    assert p.offered == ("lamp_status", EXPLAIN_LABEL, ESCALATE_LABEL)
    assert p.grounded is True


# ---------------------------------------------------------------------------
# split.stratified_split / stratified_split_sized
# ---------------------------------------------------------------------------


def _entry(entry_id: str, expect: dict, **extra) -> dict:
    return {"id": entry_id, "text": f"text {entry_id}", "expect": expect, **extra}


def _corpus_entries() -> list[dict]:
    entries = [_entry(f"op{i:02d}", {"operation": "lamp_status"}) for i in range(10)]
    entries += [
        _entry(f"esc{i:02d}", {"escalate": True}, **{"class": f"c{i % 2}"}) for i in range(6)
    ]
    entries += [_entry("var00", {"operation": "lamp_status"}, source_id="op03")]
    return entries


def test_expectation_kind_refuses_an_unknown_block() -> None:
    with pytest.raises(ValueError, match="expect block the factory doesn't recognize"):
        split.expectation_kind({"other": True})


@pytest.mark.parametrize(
    "fractions",
    [(0.5, 0.5, float("nan")), (1.5, -0.25, -0.25), (0.5, 0.5, -0.0001), (0.5, 0.5, math.inf)],
)
def test_stratified_split_refuses_a_fraction_out_of_range(fractions) -> None:
    with pytest.raises(ValueError, match="^each fraction must be between 0 and 1, got"):
        split.stratified_split(_corpus_entries(), fractions=fractions)


def test_stratified_split_refuses_fractions_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="^fractions must sum to 1.0, got \\(0.5, 0.2, 0.2\\)$"):
        split.stratified_split(_corpus_entries(), fractions=(0.5, 0.2, 0.2))


def test_stratified_split_refuses_duplicate_ids_sorted() -> None:
    entries = _corpus_entries() + [_entry("op05", {"explain": True}), _entry("op01", {})]
    with pytest.raises(ValueError, match="\\['op01', 'op05'\\]$"):
        split.stratified_split(entries)


def test_stratified_split_refuses_a_source_mixing_kinds() -> None:
    entries = _corpus_entries() + [_entry("var01", {"escalate": True}, source_id="op03")]
    with pytest.raises(
        ValueError, match="^source 'op03' mixes expectation kinds \\['escalate', 'operation'\\]$"
    ):
        split.stratified_split(entries)


def test_stratified_split_keeps_a_source_together_and_reports_missing_kinds() -> None:
    sides, missing = split.stratified_split(_corpus_entries(), seed=3)
    assert missing == ["explain"]
    by_side = {name: [e["id"] for e in sides[name]] for name in split.SPLIT_NAMES}
    for ids in by_side.values():
        assert ids == sorted(ids)
    home = [name for name, ids in by_side.items() if "op03" in ids]
    assert home == [name for name, ids in by_side.items() if "var00" in ids]
    assert all(e["source_id"] == "op03" for e in sides[home[0]] if e["id"] == "var00")
    assert sum(len(ids) for ids in by_side.values()) == len(_corpus_entries())


def test_stratified_split_sized_refuses_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="^cannot split an empty corpus$"):
        split.stratified_split_sized([], 1, 1, 1)


def test_stratified_split_sized_refuses_sizes_over_the_corpus() -> None:
    with pytest.raises(
        ValueError, match="^val_size \\+ test_size \\(18\\) exceeds the corpus size \\(17\\)$"
    ):
        split.stratified_split_sized(_corpus_entries(), 1, 9, 9)


def test_stratified_split_sized_refuses_duplicate_ids() -> None:
    entries = _corpus_entries() + [_entry("op02", {"operation": "x"})]
    with pytest.raises(ValueError, match="^duplicate entry ids would split one source across"):
        split.stratified_split_sized(entries, 1, 2, 2)


def test_stratified_split_sized_refuses_a_source_mixing_kinds() -> None:
    entries = _corpus_entries() + [_entry("var01", {"explain": True}, source_id="op03")]
    with pytest.raises(ValueError, match="^source 'op03' mixes expectation kinds"):
        split.stratified_split_sized(entries, 1, 2, 2)


def test_stratified_split_sized_pins_a_seeded_split() -> None:
    sides, missing = split.stratified_split_sized(_corpus_entries(), 5, 4, 4)
    assert missing == ["explain"]
    ids = {name: [e["id"] for e in sides[name]] for name in split.SPLIT_NAMES}
    again, _ = split.stratified_split_sized(_corpus_entries(), 5, 4, 4)
    assert ids == {name: [e["id"] for e in again[name]] for name in split.SPLIT_NAMES}
    assert sum(len(v) for v in ids.values()) == 17


# ---------------------------------------------------------------------------
# split._main_v2 (through split.main)
# ---------------------------------------------------------------------------


def _write_corpus(path: Path, entries: list[dict], **extra) -> Path:
    path.write_text(json.dumps({"header": "h", "entries": entries, **extra}), encoding="utf-8")
    return path


def _v2_args(corpus: list[Path], out_dir: Path, *more: str) -> list[str]:
    argv: list[str] = []
    for path in corpus:
        argv += ["--corpus", str(path)]
    return argv + [
        "--val-size",
        "4",
        "--test-size",
        "4",
        "--fold-seed",
        "3",
        "--out-dir",
        str(out_dir),
        *more,
    ]


def test_main_v2_a_value_error_becomes_a_usage_error(tmp_path, capsys) -> None:
    good = _write_corpus(tmp_path / "a.json", _corpus_entries())
    held = _write_corpus(tmp_path / "held-out.json", _corpus_entries())
    with pytest.raises(SystemExit) as exc:
        split.main(_v2_args([good, held], tmp_path / "out"))
    assert exc.value.code == 2
    assert "the held-out split is for judging a tuned model" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_v2_refuses_a_bad_version_as_a_usage_error(tmp_path, capsys) -> None:
    good = _write_corpus(tmp_path / "a.json", _corpus_entries())
    with pytest.raises(SystemExit):
        split.main(_v2_args([good], tmp_path / "out", "--version", "val"))
    assert "must not name a split side" in capsys.readouterr().err


def test_main_v2_reports_kinds_that_cannot_reach_every_side(tmp_path, capsys) -> None:
    entries = _corpus_entries() + [_entry("exp00", {"explain": True})]
    good = _write_corpus(tmp_path / "a.json", entries)
    with pytest.raises(SystemExit):
        split.main(_v2_args([good], tmp_path / "out"))
    assert "too few entries to reach every side: missing 'explain' on" in capsys.readouterr().err


def test_main_v2_full_run_pins_outputs(tmp_path, capsys) -> None:
    first = _write_corpus(tmp_path / "a.json", _corpus_entries()[:8])
    second = _write_corpus(tmp_path / "b.json", _corpus_entries()[6:], world={"platform": "toy"})
    third = _write_corpus(tmp_path / "c.json", [], world={"platform": "late"})
    extra = _write_corpus(tmp_path / "t.json", [_entry("t1", {"operation": "x"})])
    out_dir = tmp_path / "out"
    code = split.main(
        _v2_args([first, second, third], out_dir, "--train-only", str(extra), "--seed", "11")
    )
    assert code == 0
    assert capsys.readouterr().out == (
        "note: no 'explain' entries in the merged corpus; not present on any side\n"
        "train=8\nval=6\ntest=4\n"
        "merged 18 unique entries from 3 corpus file(s)\n"
    )
    sides = {n: json.loads((out_dir / f"{n}.json").read_text()) for n in split.SPLIT_NAMES}
    folds = json.loads((out_dir / "folds.json").read_text())
    for name, payload in sides.items():
        assert payload["world"] == {"platform": "toy"}
        assert payload["split"]["version"] == "v2"
        assert payload["split"]["seed"] == 11
        assert payload["split"]["sizes"] == {"train": 8, "val": 6, "test": 4}
        assert [s["path"] for s in payload["split"]["sources"]] == [
            str(first),
            str(second),
            str(third),
            str(extra),
        ]
        assert payload["split"]["sources"][-1]["train_only"] is True
        assert ("fit_ids" in payload["split"]) is (name == "val")
    assert sides["val"]["split"]["fold_seed"] == 3
    assert sides["val"]["split"]["fit_ids"] == folds["fit_ids"]
    assert sides["val"]["split"]["selection_ids"] == folds["selection_ids"]
    assert folds["seed"] == 3
    assert folds["source"] == str(out_dir / "val.json")
    assert sorted(folds["fit_ids"] + folds["selection_ids"]) == sorted(
        e["id"] for e in sides["val"]["entries"]
    )
    assert any(e["id"] == "t1" and e["train_only"] for e in sides["train"]["entries"])


def test_main_v2_without_any_world_writes_none(tmp_path) -> None:
    first = _write_corpus(tmp_path / "a.json", _corpus_entries(), world=None)
    out_dir = tmp_path / "out"
    assert split.main(_v2_args([first], out_dir, "--version", "v9")) == 0
    payload = json.loads((out_dir / "train.json").read_text())
    assert "world" not in payload
    assert payload["split"]["version"] == "v9"


def test_main_v2_reads_world_from_a_list_corpus_as_absent(tmp_path) -> None:
    listed = tmp_path / "a.json"
    listed.write_text(json.dumps(_corpus_entries()), encoding="utf-8")
    later = _write_corpus(tmp_path / "b.json", [], world={"platform": "second"})
    out_dir = tmp_path / "out"
    assert split.main(_v2_args([listed, later], out_dir)) == 0
    assert json.loads((out_dir / "val.json").read_text())["world"] == {"platform": "second"}
