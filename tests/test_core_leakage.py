"""core/leakage.py, ported from nvsh tests/test_lfm_finetune_leakage_check.py (issue 46, t19)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_factory.core import leakage as leakage_module
from jev_factory.core import textsim


def _module():
    return leakage_module


def _split(path: Path, texts: dict[str, str]) -> Path:
    entries = [{"id": k, "text": v, "expect": {"escalate": True}} for k, v in texts.items()]
    path.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    return path


def test_finds_exact_and_near_duplicates_by_id_only(tmp_path, capsys) -> None:
    module = _module()
    train = _split(
        tmp_path / "train.json",
        {
            "t1": "Show me the current GPU utilisation please",
            "t2": "show me the current gpu utilisation, please!",  # exact after normalizing
            "t3": "please show me the current GPU utilisation",  # same words, reordered
            "t4": "How hot is the board right now?",
        },
    )
    held = _split(tmp_path / "held.json", {"h1": "Show me the current GPU utilisation please"})
    rc = module.main(["--train", str(train), "--protected", str(held)])
    out = capsys.readouterr().out
    assert rc == 1
    report = json.loads(out)
    assert report["hits"] == 3
    assert {h["train_id"] for h in report["matches"]} == {"t1", "t2", "t3"}
    assert "GPU" not in out  # never prints a protected (or training) text


def test_drops_hits_into_a_filtered_file(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "Show the GPU stats now", "t4": "how hot is it"})
    held = _split(tmp_path / "held.json", {"h1": "Show the GPU stats now"})
    out_file = tmp_path / "filtered.json"
    rc = module.main(
        ["--train", str(train), "--protected", str(held), "--out-filtered", str(out_file)]
    )
    assert rc == 0
    kept = json.loads(out_file.read_text(encoding="utf-8"))
    assert [e["id"] for e in kept["entries"]] == ["t4"]
    assert kept["header"] == "h"


def test_a_clean_training_file_passes(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "Restart the vllm service for me"})
    held = _split(tmp_path / "held.json", {"h1": "What does unified memory mean on GB10?"})
    assert module.main(["--train", str(train), "--protected", str(held)]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 0


def test_short_texts_only_match_exactly(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "gpu stats", "t2": "stats gpu"})
    held = _split(tmp_path / "held.json", {"h1": "gpu stats"})
    module.main(["--train", str(train), "--protected", str(held)])
    report = json.loads(capsys.readouterr().out)
    assert [h["train_id"] for h in report["matches"]] == ["t1"]


def test_two_protected_files_with_the_same_name_are_both_checked(tmp_path, capsys) -> None:
    # Codex review: files keyed by basename let one test.json replace another.
    module = _module()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _split(tmp_path / "a" / "test.json", {"x1": "Restart the vllm service now please"})
    second = _split(tmp_path / "b" / "test.json", {"y1": "How hot is the Jetson board right now"})
    train = _split(
        tmp_path / "train.json",
        {
            "t1": "Restart the vllm service now please",
            "t2": "How hot is the Jetson board right now",
        },
    )
    assert module.main(["--train", str(train), "--protected", str(first), str(second)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["hits"] == 2
    assert len(report["protected"]) == 2


def test_filtering_drops_only_the_matching_rows(tmp_path, capsys) -> None:
    module = _module()
    train = tmp_path / "train.jsonl"
    rows = [
        {"id": "dup", "text": "Show the GPU stats now please"},
        {"id": "dup", "text": "What is unified memory on GB10"},
    ]
    train.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    held = _split(tmp_path / "held.json", {"h1": "Show the GPU stats now please"})
    out_file = tmp_path / "out.jsonl"
    module.main(["--train", str(train), "--protected", str(held), "--out-filtered", str(out_file)])
    kept = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in kept] == ["What is unified memory on GB10"]


def test_an_entry_without_a_text_is_refused(tmp_path, capsys) -> None:
    module = _module()
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"id": "m1", "messages": []}) + "\n", encoding="utf-8")
    held = _split(tmp_path / "held.json", {"h1": "anything at all here"})
    assert module.main(["--train", str(train), "--protected", str(held)]) == 2
    assert "m1" in capsys.readouterr().err


def test_a_text_that_normalises_to_nothing_is_refused_not_passed(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "?!..."})
    held = _split(tmp_path / "held.json", {"h1": "anything at all here"})
    assert module.main(["--train", str(train), "--protected", str(held)]) == 2
    assert "t1" in capsys.readouterr().err


@pytest.mark.behavioral("o16")
def test_every_protected_file_is_checked_by_every_rule_and_only_ids_are_printed(
    tmp_path, capsys
) -> None:
    """Exact, 5-token-shingle and word-set-Jaccard matches each fail closed, ids only."""
    module = _module()
    hidden = "kitchen lamp hidden wording"
    # Protected texts, one per protected file, one per rule.
    exact = _split(tmp_path / "exact.json", {"p-exact": "Turn the kitchen lamp on now"})
    # 8 shared tokens + 1 extra: shingle Jaccard = 4/5 = 0.8 (>= 0.8) on 5-token shingles.
    shingle_text = "set the kitchen lamp to a warm reading glow"
    shingle = _split(tmp_path / "shingle.json", {"p-shingle": shingle_text + " now"})
    # Reordered words: no shingle overlap, word-set Jaccard = 1.0.
    words = _split(tmp_path / "words.json", {"p-words": "bedroom dim the lights fully please"})
    train = _split(
        tmp_path / "train.json",
        {
            "t-exact": "turn the kitchen LAMP on, now!",
            "t-shingle": shingle_text,
            "t-words": "please dim the bedroom lights fully",
            "t-clean": hidden,
        },
    )
    rc = module.main(["--train", str(train), "--protected", str(exact), str(shingle), str(words)])
    out = capsys.readouterr().out
    assert rc == 1  # fails closed: any hit is non-zero without --out-filtered
    report = json.loads(out)
    kinds = {m["train_id"]: (m["protected_id"], m["kind"]) for m in report["matches"]}
    assert kinds == {
        "t-exact": ("p-exact", "exact"),
        "t-shingle": ("p-shingle", "near-duplicate"),
        "t-words": ("p-words", "near-duplicate"),
    }
    assert len(report["protected"]) == 3  # every protected file was loaded
    for text in ("kitchen", "lamp", "bedroom", "lights", hidden):
        assert text not in out  # ids and counts only, never a text


@pytest.mark.behavioral("o16")
def test_unreadable_or_malformed_protected_file_fails_closed_without_echoing_content(
    tmp_path, capsys
) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "a perfectly clean request here"})
    bad = tmp_path / "held.json"
    bad.write_text('{"entries": [ SEALED_TEXT-TEXT', encoding="utf-8")
    assert module.main(["--train", str(train), "--protected", str(bad)]) == 2
    captured = capsys.readouterr()
    assert "SEALED_TEXT-TEXT" not in captured.out + captured.err
    assert "held.json" in captured.err
    missing = tmp_path / "nope.json"
    assert module.main(["--train", str(train), "--protected", str(missing)]) == 2


def test_textsim_helpers_keep_nvsh_behaviour() -> None:
    assert textsim.normalize_text("  Hello,   WORLD!! ") == "hello world"
    assert textsim.shingles("a b c") == frozenset({("a", "b", "c")})
    assert textsim.shingles("") == frozenset()
    assert len(textsim.shingles("a b c d e f")) == 2
    assert textsim.jaccard(frozenset(), frozenset()) == 1.0
    assert textsim.jaccard(frozenset({1}), frozenset()) == 0.0
    assert textsim.jaccard(frozenset({1, 2}), frozenset({2, 3})) == pytest.approx(1 / 3)
    assert textsim.NEAR_DUP_SHINGLE_SIZE == 5
    assert textsim.NEAR_DUP_JACCARD_THRESHOLD == 0.8 == textsim.WORD_JACCARD
