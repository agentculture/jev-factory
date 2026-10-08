"""Release-gate cases (Domain-classified, held-out guarded), the request contract, the manifest.

Ported from nvsh evals/tool_jev/tests/test_cases.py, test_request.py (choice
interface) and test_manifest.py onto the toy domain.
"""

from __future__ import annotations

import json

import pytest

from jev_factory.evals import cases as cases_mod
from jev_factory.evals import manifest as manifest_mod
from jev_factory.evals import request as contract
from jev_factory.evals.providers.errors import Outcome
from tests.evals_support import CASES, HELDOUT, MANIFEST, write_json
from tests.fixtures.toy_domain import DOMAIN


@pytest.fixture
def splits(tmp_path):
    test = write_json(tmp_path / "test.json", {"header": "h", "entries": CASES})
    held = write_json(tmp_path / "held.json", {"header": "h", "entries": HELDOUT})
    return {"splits": {"test": str(test), "heldout": str(held)}}


def test_cases_take_read_only_from_the_domain_and_unknown_is_mutating(tmp_path, splits):
    loaded = {c.id: c for c in cases_mod.load_case_set(splits, "test", domain=DOMAIN)}
    assert loaded["c-1"].read_only is False  # lamp_on mutates
    assert loaded["c-2"].read_only is True  # lamp_status reads
    assert loaded["c-3"].read_only is None
    assert loaded["c-3"].expects_explain
    assert loaded["c-4"].expects_escalate
    assert loaded["c-5-nocand"].candidates == ("lamp_status", "list_rooms")
    assert "nocand" in loaded["c-5-nocand"].tags
    assert loaded["c-1"].expected_operation == "lamp_on"
    assert loaded["c-1"].expected_args == {"room": "kitchen"}
    unknown = [{"id": "u", "text": "t", "expect": {"operation": "self_destruct"}}]
    path = write_json(tmp_path / "u.json", {"entries": unknown})
    (case,) = cases_mod.load_case_set({"test": str(path)}, "test", domain=DOMAIN)
    assert case.read_only is False


def test_heldout_needs_the_flag_and_never_returns_text(splits):
    with pytest.raises(cases_mod.HeldOutAccessError):
        cases_mod.load_case_set(splits, "heldout", domain=DOMAIN)
    held = cases_mod.load_case_set(splits, "heldout", domain=DOMAIN, include_heldout=True)
    assert [c.text for c in held] == [None, None]
    with pytest.raises(ValueError):
        cases_mod.Case("x", "heldout", "leak", None, {"escalate": True}, None)
    with pytest.raises(ValueError):
        cases_mod.Case("x", "train", "t", None, {"escalate": True}, None)


def test_loader_refuses_bad_inputs(tmp_path, splits):
    with pytest.raises(ValueError):
        cases_mod.load_case_set(splits, "nope", domain=DOMAIN)
    with pytest.raises(KeyError):
        cases_mod.load_case_set(splits, "test-mc", domain=DOMAIN)
    bad = write_json(tmp_path / "bad.json", {"entries": {"not": "a list"}})
    with pytest.raises(ValueError):
        cases_mod.load_case_set({"test": str(bad)}, "test", domain=DOMAIN)
    foreign = [{"id": "f", "text": "t", "expect": {"escalate": True}, "candidates": ["warp"]}]
    path = write_json(tmp_path / "f.json", {"entries": foreign})
    with pytest.raises(ValueError, match="warp"):
        cases_mod.load_case_set({"test": str(path)}, "test", domain=DOMAIN)
    manifest_file = write_json(tmp_path / "m.json", splits)
    assert cases_mod.load_manifest(manifest_file) == splits["splits"]
    with pytest.raises(ValueError):
        cases_mod.load_manifest({"splits": ["x"]})


def test_training_overlap_is_refused(tmp_path):
    train = write_json(tmp_path / "train.json", {"entries": [{"id": "c-2"}, {"id": "t-9"}]})
    assert cases_mod.training_overlap(["c-1"], train) == ()
    with pytest.raises(cases_mod.TrainingOverlapError) as info:
        cases_mod.training_overlap(["c-2", "c-1"], train)
    assert info.value.overlapping_ids == ("c-2",)


def test_case_sets_from_a_run_manifest(tmp_path):
    data = __import__("tomllib").loads(
        MANIFEST.format(domain="tests.fixtures.toy_domain", extra_refs="", openrouter_cap=1.0)
    )
    manifest = manifest_mod.parse_manifest(data)
    paths = cases_mod.case_sets_from_manifest(manifest, tmp_path)
    assert paths == {
        "test": str(tmp_path / "splits/toy-test.json"),
        "heldout": str(tmp_path / "splits/toy-heldout.json"),
    }


# ---------------------------------------------------------------------------
# the request contract
# ---------------------------------------------------------------------------


def test_choice_request_is_the_scorer_prompt_with_stable_letters(splits):
    loaded = {c.id: c for c in cases_mod.load_case_set(splits, "test", domain=DOMAIN)}
    full = contract.build_choice_request(loaded["c-1"], DOMAIN)
    assert full.params["labels"] == {
        "lamp_status": "A",
        "list_rooms": "B",
        "room_status": "C",
        "lamp_on": "D",
        "set_scene": "E",
        "explain": "F",
        "escalate": "G",
    }
    assert full.prompt.startswith(DOMAIN.instruction)
    assert "D) lamp_on: Turn on the lamps in one named room." in full.prompt
    assert full.case_text == "turn on the lamps in the kitchen"
    narrowed = contract.build_choice_request(loaded["c-5-nocand"], DOMAIN)
    assert narrowed.params["labels"] == {
        "lamp_status": "A",
        "list_rooms": "B",
        "explain": "F",
        "escalate": "G",
    }
    assert narrowed.offered_candidates == ("A", "B", "F", "G")
    system, messages, labels = contract.canonical_content(narrowed)
    assert system == narrowed.prompt
    assert labels == narrowed.params["labels"]
    assert messages == [{"role": "user", "content": narrowed.case_text}]
    held = cases_mod.Case("h", "heldout", None, None, {"escalate": True}, None)
    with pytest.raises(ValueError):
        contract.build_choice_request(held, DOMAIN)


def test_parse_choice_reads_the_first_letter_only():
    labels = {"lamp_status": "A", "explain": "F"}
    classification, name = contract.parse_choice(" F) explain", labels)
    assert classification.outcome is Outcome.OK
    assert name == "explain"
    assert contract.parse_choice("Z", labels)[0].reason == "malformed"
    assert contract.parse_choice(None, labels)[0].reason == "malformed"
    assert contract.parse_choice("", labels)[1] is None


def test_distribution_from_logprobs_is_never_renormalised_over_a_subset():
    labels = {"lamp_status": "A", "explain": "F"}
    dist = contract.distribution_from_logprobs({"A": -0.1, " F": -2.0}, labels, ("lamp_status",))
    assert list(dist) == ["lamp_status"]
    assert 0.8 < dist["lamp_status"] < 1.0
    assert contract.distribution_from_logprobs({"A": -0.1}, labels, tuple(labels)) is None


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _parse(text: str):
    import tomllib

    return manifest_mod.parse_manifest(tomllib.loads(text))


BASE = 'domain = "tests.fixtures.toy_domain"\n'


def test_the_example_manifest_parses():
    manifest = _parse(
        MANIFEST.format(domain="tests.fixtures.toy_domain", extra_refs="", openrouter_cap=10.0)
    )
    assert manifest.domain == "tests.fixtures.toy_domain"
    assert manifest.world == "world.json"
    (cand,) = manifest.candidates
    assert cand.policies == ("raw", "mutating-strict-example")
    assert cand.predictions_for("toy-test") == cand.predictions_path
    assert [cs.name for cs in manifest.sendable_case_sets()] == ["toy-test"]
    assert manifest.budget_for("openrouter").usd_cap == 10.0
    assert manifest.budget_for("nvidia") is None
    assert manifest.roster_keys() == {("openrouter", "vendor/fake-sync")}
    assert manifest.stops.min_answers == 3


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "domain"),
        (BASE + '[[reference]]\nprovider = "openai"\nmodel = "m"\n', "unknown provider"),
        (BASE + '[[reference]]\nprovider = "local"\nmodel = "m"\nbatch = true\n', "batch"),
        (
            BASE + '[[reference]]\nprovider = "local"\nmodel = "m"\n'
            '[[reference]]\nprovider = "local"\nmodel = "m"\n',
            "duplicate",
        ),
        (BASE + '[[reference]]\nprovider = "local"\nmodel = "m"\nreasoning = "max"\n', "reason"),
        (
            BASE + '[[case_set]]\nname = "x"\ncount = 1\nsplit = "heldout"\npath = "p.json"\n',
            "include_heldout",
        ),
        (
            BASE + '[[case_set]]\nname = "x"\ncount = 1\nsplit = "test"\npath = "/abs.json"\n',
            "relative",
        ),
        (BASE + 'world = "~/w.json"\n', "relative"),
        (
            BASE + "[budget.local]\nusd_cap = 1\nconcurrency_cap = 1\nbatch_discount = 0.5\n",
            "batch",
        ),
        (BASE + "[budget.local]\nusd_cap = -1\nconcurrency_cap = 1\n", "usd_cap"),
        (BASE + "[budget.local]\nusd_cap = 1\nconcurrency_cap = 0\n", "concurrency_cap"),
        (BASE + "[stops]\nmax_truncated_share = 0\n", "max_truncated_share"),
        (BASE + '[[candidate]]\nname = "c"\npredictions_path = "p"\npolicies = []\n', "policies"),
    ],
)
def test_manifest_validation_fails_loudly(text, message):
    with pytest.raises(manifest_mod.ManifestError, match=message):
        _parse(text)


def test_manifest_path_comes_from_the_environment(tmp_path):
    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.resolve_manifest_path({})
    path = tmp_path / "m.toml"
    path.write_text(BASE)
    loaded = manifest_mod.load_manifest_from_env({manifest_mod.ENV_MANIFEST_PATH: str(path)})
    assert loaded.domain == "tests.fixtures.toy_domain"
    assert loaded.references == ()


def test_split_file_shape_matches_the_core_split(tmp_path):
    # The same {"header", "entries"} documents jev_factory.core.split writes load here.
    path = write_json(tmp_path / "s.json", {"header": "h", "entries": CASES[:1]})
    (case,) = cases_mod.load_case_set({"test": str(path)}, "test", domain=DOMAIN)
    assert json.loads(path.read_text())["entries"][0]["id"] == case.id


def test_a_validation_side_is_evaluated_under_its_own_tag(tmp_path):
    val = write_json(tmp_path / "val.json", {"header": "h", "entries": CASES})
    loaded = cases_mod.load_case_set({"val": str(val)}, "val", domain=DOMAIN)
    assert loaded
    assert {c.split for c in loaded} == {"val"}
    assert all(c.text for c in loaded)  # validation text is readable, unlike held-out
    with pytest.raises(ValueError):
        cases_mod.Case("x", "train", "t", None, {"escalate": True}, None)
