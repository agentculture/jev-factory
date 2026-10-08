"""Sonar S8786: three regexes backtracked super-linearly on long input. Their replacements
must match the old patterns on ordinary input and stay fast on a pathological one."""

from __future__ import annotations

import re
import time

import pytest

from jev_factory.cli._commands import status
from jev_factory.cli._errors import CliError
from jev_factory.data.teachers import strip_fence
from jev_factory.review.core import split_number

OLD_FENCE = re.compile(r"(?:^```[A-Za-z]*\s*)|(?:\s*```$)")
OLD_NUMBERED = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")
LONG = 200_000


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "yes"}',
        '```json\n{"verdict": "yes"}\n```',
        '```\n{"verdict": "no", "reason": "x"}\n```',
        '  ```JSON {"verdict":"yes"}```  ',
        '{"verdict": "yes"}\n```',
        "```",
        "```json```",
        "",
        "plain text with ``` inside",
    ],
)
def test_strip_fence_matches_the_old_regex(text):
    assert strip_fence(text) == OLD_FENCE.sub("", text.strip()).strip()


@pytest.mark.parametrize("entry_id", ["jcs-op-042", "toy-7", "a1b22", "123", "no-digits", ""])
def test_split_number_matches_the_old_regex(entry_id):
    match = OLD_NUMBERED.match(entry_id)
    expected = (match["prefix"], match["num"]) if match else None
    assert split_number(entry_id) == expected


def _seconds(call) -> float:
    start = time.perf_counter()
    try:
        call()
    except CliError:
        pass  # a refusal is fine; only the time is measured here
    return time.perf_counter() - start


def test_the_replacements_are_fast_on_pathological_input():
    assert _seconds(lambda: strip_fence("{" + " " * LONG + "x")) < 0.5
    assert _seconds(lambda: split_number("1" * LONG + "x")) < 0.5
    bad = " " * LONG + "1" + " " * LONG + "q"
    assert _seconds(lambda: status.parse_every(bad)) < 0.5
    with pytest.raises(CliError, match="is not a duration"):
        status.parse_every("1 q")
    assert status.parse_every("  30m  ") == 1800


OLD_JSON_FENCE = re.compile(r"(?:^```(?:json)?)|(?:```$)", re.M)


def test_parse_json_list_fences_match_the_old_single_regex():
    """The two-pass fence strip (Sonar S5850) removes exactly what the old one-pass regex did."""
    import random

    from jev_factory.data import common

    rng = random.Random(7)
    pieces = ["```", "json", "`", "\n", " ", "[1]", "x", "``````"]
    samples = ["```json\n[1, 2]\n```", "[1]", "```\n[1]\n```", "```json```", "``````", "`````"]
    samples += ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 12))) for _ in range(3000)]
    for raw in samples:
        old = OLD_JSON_FENCE.sub("", raw.strip())
        new = common._FENCE_CLOSE.sub("", common._FENCE_OPEN.sub("", raw.strip()))
        assert new == old, repr(raw)


OLD_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def test_think_block_rewrite_matches_the_old_pattern():
    """Sonar S6019: the two-branch pattern removes exactly what the old one did."""
    import random

    from jev_factory.evals.providers import base

    rng = random.Random(11)
    pieces = ["<think>", "</think>", "<THINK>", "</Think>", "a", "\n", " ", "answer", "<", ">"]
    samples = ["<think>x</think>A", "<think>never closed", "A<think>b</think>C<think>d", ""]
    samples += ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 14))) for _ in range(4000)]
    for text in samples:
        assert base._THINK_BLOCK.sub("", text) == OLD_THINK.sub("", text), repr(text)


def test_split_exits_by_default_and_raises_when_asked(tmp_path, capsys):
    """split.main keeps argparse's exit for the CLI; in-process callers get SplitRefused."""
    from jev_factory.core import split

    argv = ["--seed", "not-a-number", "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as exited:
        split.main(argv)
    assert exited.value.code == 2
    assert "jev-split: error:" in capsys.readouterr().err
    with pytest.raises(split.SplitRefused) as refused:
        split.main(argv, exit_on_error=False)
    assert str(refused.value)
    assert capsys.readouterr().err == ""
