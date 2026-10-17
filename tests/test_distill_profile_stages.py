"""t11: the stages take their student-specific values from the profile."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import benchmark as bm
from jev_factory.distill import contamination, dataset, pins
from jev_factory.distill.profile import DEFAULT_2610
from tests import test_distill_dataset as td_
from tests import test_distill_train as tt


def test_contamination_loads_the_profiles_manifest(tmp_path):
    manifest = {
        "sources": [
            {
                "name": "only-this",
                "role": "training",
                "source_url": "https://example.test/x",
                "card": "https://example.test/c",
                "aliases": [],
            }
        ]
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest))
    prof = replace(DEFAULT_2610, contamination_manifest=str(p))
    assert contamination.load_manifest(profile=prof) == manifest
    assert contamination.load_manifest() == contamination.load_manifest(
        Path(contamination.MANIFEST_PATH)
    )
    # check_sources with no explicit manifest uses the profile's
    rep = contamination.check_sources(["only-this"], profile=prof)
    assert rep["results"]["only-this"]["status"] != "clean"


def test_dataset_limits_come_from_the_profile():
    r = td_._td(1)
    n = td_._words(dataset.render_prompt(r))
    tight = replace(DEFAULT_2610, max_tokens=n - 1)
    ds = dataset.build_dataset([r], token_counter=td_._words, seed=1, profile=tight)
    assert ds.manifest["max_tokens"] == n - 1
    assert ds.manifest["dropped_too_long"] == {"LocalLLaMA/typed-decisions": 1}
    one_slot = replace(DEFAULT_2610, num_slots=1)
    ds = dataset.build_dataset([r], token_counter=td_._words, seed=1, profile=one_slot)
    assert ds.manifest["max_slots"] == 1
    assert len(ds.all_records()) == 0
    # explicit arguments still win over the profile
    ds = dataset.build_dataset(
        [r], token_counter=td_._words, seed=1, profile=one_slot, max_slots=24
    )
    assert ds.manifest["max_slots"] == 24


def test_train_inits_from_the_profiles_decider_repo(tmp_path):
    prof = replace(DEFAULT_2610, decider_repo="Other/decider")
    cache = tmp_path / "cache"
    cache.mkdir()
    # the pins still name the default repo, so verification fails before init_from matters;
    # exercise the derived path through the dry-run plan with matching pins instead.
    plan_init = tt.T.init_from_path(cache, prof)
    assert plan_init == cache / pins.cache_dirname("Other/decider")
    assert tt.T.init_from_path(cache) == cache / pins.cache_dirname(DEFAULT_2610.decider_repo)


def test_no_card_reproduction_record_needs_a_deviation(tmp_path):
    rec = bm.not_available_record(model="m", served_model="m", base_revision="r")
    assert rec["reproduction"] == "not_available"
    assert rec["base_mismatch"] is None
    bm.write_reproduction(tmp_path, rec)
    with pytest.raises(CliError, match="not available"):
        bm.require_base_match(tmp_path)
    with pytest.raises(CliError, match="deviation"):
        bm.require_base_match(tmp_path, " ", required=True)
    bm.require_base_match(tmp_path, "d3")
