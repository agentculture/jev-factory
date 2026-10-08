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
