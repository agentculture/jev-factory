"""Behaviour pins for the SonarCloud cleanups of ``jev_factory/release`` (S5843, S1313, S107).

- The secret-assignment scanner was one regex too complex to read (S5843); it is now a
  key-name, separator and value pattern matched in turn. A differential test runs the
  old single regex (inlined here as the reference) and the split matcher over a large
  hand-picked and randomized corpus and asserts the same matches: positions, key name,
  quoting style and value.
- The CGNAT network is defined once, without an address literal (S1313).
- The release builders take small parameter groups (S107); the groups' defaults are the
  old parameters' defaults.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import random
import re
from pathlib import Path

import pytest

from jev_factory.release import bundle, dataset_bundle, scan

#: The single regex ``scan.py`` used before the split (the reference for the differential).
_REFERENCE_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
        \b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|token|password|passwd|pwd)
        ["']?
        \s*[:=]\s*
        (?:
            "(?P<dq>[^"\n]{20,})"
          | '(?P<sq>[^'\n]{20,})'
          | (?P<bare>[^\s"'`,;)\]}]{20,})
        )
    """,
)

_LONG = "abcdefghijklmnopqrstuvwxyz0123"

_HAND_PICKED = [
    "",
    "nothing to see here",
    f"api_key={_LONG}",
    f"API-KEY = '{_LONG}'",
    f'apikey: "{_LONG}"',
    f"secret={_LONG}",
    f"secret_key={_LONG}",
    f"secret-key: {_LONG}",
    f"SecretKey = {_LONG}",
    f"secretkey={_LONG}",
    f"access_key={_LONG}",
    f"accesskey = {_LONG}",
    f"token={_LONG}",
    f"Token: '{_LONG}'",
    f"password={_LONG}",
    f"passwd={_LONG}",
    f"pwd = {_LONG}",
    f'"token": "{_LONG}"',
    f"'password': '{_LONG}'",
    f"mytoken={_LONG}",
    f"my_token={_LONG}",
    f"my-token={_LONG}",
    f"api_key_token={_LONG}",
    f"secret_token={_LONG}",
    f"token = {'a' * 19}",
    f"token = {'a' * 20}",
    f"token = \"{'a' * 19}\"",
    f"token = \"{'a' * 20}\"",
    f"token = '{'a' * 20}",
    f'token = "{_LONG}',
    f"token = \"{'a' * 10}\n{'b' * 15}\"",
    f"token=\t{_LONG}",
    f"token=\n{_LONG}",
    f"token ={_LONG};password={_LONG}",
    f"token={_LONG},api_key='{_LONG}'",
    f"token={_LONG}) secret={_LONG}]",
    f"token={'x' * 25}`tail",
    f"tokentoken={_LONG}",
    f"token token={_LONG}",
    f"secret_key_extra={_LONG} secret={_LONG}",
    f"secret_={_LONG}",
    f"secret-={_LONG}",
    f"api_-key={_LONG}",
    f"token:: {_LONG}",
    f"token=:{_LONG}",
    f"token = = {_LONG}",
    f"token'\"={_LONG}",
    f'token"= {_LONG}',
    f"tokén={_LONG}",
    f"étoken={_LONG}",
    f"token = {_LONG}",
    f"PASSWORD={_LONG} pwd={_LONG} passwd='{_LONG}'",
    f"x=1; token: ${{{{ secrets.HF }}}} api_key = {_LONG}",
]

_FRAGMENTS = [
    "api",
    "key",
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "secret_key",
    "secret-",
    "SECRET",
    "access",
    "access_key",
    "token",
    "TOKEN",
    "password",
    "passwd",
    "pwd",
    "my",
    "_",
    "-",
    ".",
    " ",
    "  ",
    "\t",
    "\n",
    ":",
    "=",
    '"',
    "'",
    "`",
    ",",
    ";",
    ")",
    "]",
    "}",
    "(",
    "{",
    "$",
    "é",
    " ",
    "x" * 5,
    "y" * 12,
    "z" * 19,
    "a" * 20,
    "b" * 23,
    _LONG,
]


_KEYS = [
    "api_key",
    "API-Key",
    "apikey",
    "api__key",
    "secret",
    "Secret_Key",
    "secret-key",
    "secretkey",
    "secret_",
    "access_key",
    "accesskey",
    "access-",
    "token",
    "Token",
    "toke",
    "password",
    "passwd",
    "passwrd",
    "pwd",
    "PWD",
]
_VALUE_CHARS = "abcXYZ019_-./+$" + " \t\n\"'`,;)]}:=é"
_SEPARATORS = ["=", ":", " = ", "\t:\t", "==", ":=", "", " ", "=\n", "\u00a0=\u00a0"]


def _assignment_like(rng: random.Random) -> str:
    """One key, quote, separator and value, each sometimes slightly off."""
    key = rng.choice(_KEYS)
    key_quote = rng.choice(["", "", '"', "'", "`"])
    value = "".join(
        rng.choice(_VALUE_CHARS) if rng.random() < 0.08 else rng.choice("abcdef0123456789")
        for _ in range(rng.randint(15, 30))
    )
    open_quote, close_quote = rng.choice(
        [("", ""), ('"', '"'), ("'", "'"), ('"', ""), ("'", '"'), ("`", "`")]
    )
    prefix = rng.choice(["", "", " ", "my", "my_", "my-", "x.", "(", '"', "{"])
    separator = rng.choice(_SEPARATORS)
    return f"{prefix}{key_quote}{key}{key_quote}{separator}{open_quote}{value}{close_quote}"


def _random_corpus(count: int, seed: int) -> list[str]:
    """Half free-form fragment soup, half chains of near-miss and real assignments."""
    rng = random.Random(seed)  # nosec B311 -- a deterministic test corpus, not crypto
    corpus = []
    for index in range(count):
        if index % 2:
            parts = rng.choices(_FRAGMENTS, k=rng.randint(1, 14))
        else:
            parts = [
                _assignment_like(rng) + rng.choice(["", " ", ", ", "; ", "\n", "", "x"])
                for _ in range(rng.randint(1, 3))
            ]
        corpus.append("".join(parts))
    return corpus


def _reference(line: str) -> list[tuple[int, int, str, str, str]]:
    found = []
    for match in _REFERENCE_ASSIGNMENT_RE.finditer(line):
        style = next(name for name in ("dq", "sq", "bare") if match.group(name) is not None)
        found.append((match.start(), match.end(), match.group(1), style, match.group(style)))
    return found


def _split(line: str) -> list[tuple[int, int, str, str, str]]:
    return [
        (a.start, a.end, a.key, a.style, a.value) for a in scan._assignments(line)  # noqa: SLF001
    ]


_CORPUS = _HAND_PICKED + _random_corpus(20000, seed=5843)


def test_the_corpus_exercises_every_quoting_style_and_misses():
    styles = {m[3] for line in _CORPUS for m in _reference(line)}
    assert styles == {"dq", "sq", "bare"}
    assert sum(1 for line in _CORPUS if _reference(line)) > 1000
    assert sum(1 for line in _CORPUS if not _reference(line)) > 1000


def test_the_split_assignment_matcher_matches_the_single_regex_it_replaced():
    differing = [line for line in _CORPUS if _split(line) != _reference(line)]
    assert differing == []


@pytest.mark.parametrize("line", _HAND_PICKED)
def test_each_hand_picked_line_matches_the_single_regex(line):
    assert _split(line) == _reference(line)


def test_the_single_regex_is_gone_from_the_scanner():
    assert not hasattr(scan, "_ASSIGNMENT_RE")


# ---- S1313: one CGNAT constant ---------------------------------------------------------


def test_the_cgnat_constant_is_rfc_6598_shared_address_space():
    assert scan.CGNAT == ipaddress.ip_network("100.64.0.0/10")


def test_the_dataset_bundle_redacts_with_the_scanners_cgnat_network():
    assert dataset_bundle.CGNAT is scan.CGNAT
    assert not hasattr(scan, "_CGNAT")
    assert not hasattr(dataset_bundle, "_CGNAT")
    assert dataset_bundle.publishable_text("at 100.64.0.1") == "at 192.0.2.1"
    assert dataset_bundle.publishable_text("at 100.128.0.1") == "at 100.128.0.1"
    assert scan.private_hosts("at 100.127.255.254") == [(1, "100.127.255.254")]


# ---- S107: parameter groups --------------------------------------------------------------


def test_the_parameter_groups_are_frozen_with_the_old_defaults():
    facts = bundle.BuildFacts()
    assert dataclasses.astuple(facts) == ("bf16", None, None, None, (), None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.kind = "gguf"  # type: ignore[misc]
    payload = bundle.ModelPayload(merged=Path("m"))
    assert (payload.kind, payload.gguf, payload.awq_dir, payload.quantized_from) == (
        "bf16",
        None,
        None,
        None,
    )
    for group in (bundle.BundleFiles, dataset_bundle.DatasetSources):
        assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(group))


@pytest.mark.parametrize(
    "function",
    [
        bundle.model_card,
        bundle.build_model_bundle,
        bundle.build_dataset_bundle,
        dataset_bundle.build,
    ],
)
def test_each_release_builder_takes_at_most_thirteen_keyword_only_parameters(function):
    code = function.__code__
    assert code.co_argcount == 0
    assert code.co_kwonlyargcount <= 13
