import json

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import contamination as c


def test_banking77_and_clinc150_overlap_unlisted_clean():
    rep = c.check_sources(["banking77", "CLINC150", "my_private_set"])
    assert rep["results"]["banking77"]["status"] == "overlapping"
    assert rep["results"]["CLINC150"]["status"] == "overlapping"
    assert rep["results"]["my_private_set"]["status"] == "clean"
    assert rep["overlapping"] == ["CLINC150", "banking77"]
    assert rep["clean"] == ["my_private_set"]


def test_alias_and_v21_training_match():
    rep = c.check_sources(["PolyAI/banking77", "nyu-mll/glue", "clinc_oos"])
    assert all(r["status"] == "overlapping" for r in rep["results"].values())


def test_every_manifest_entry_cites_a_url():
    m = c.load_manifest()
    assert m["sources"]
    assert all(
        s["url"].startswith("https://") and s["card"].startswith("https://") for s in m["sources"]
    )
    assert {s["role"] for s in m["sources"]} >= {"v21-training", "clef-decision-index-eval"}


def test_matches_carry_url():
    rep = c.check_sources(["banking77"])
    assert rep["results"]["banking77"]["matches"][0]["url"].startswith("https://")


def test_report_written_to_run(tmp_path):
    path = c.write_report(c.check_sources(["banking77", "zzz"]), tmp_path / "run")
    data = json.loads(path.read_text())
    assert data["results"]["zzz"]["status"] == "clean"


def test_bad_manifest(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{}")
    with pytest.raises(CliError):
        c.load_manifest(p)


def test_hate_speech_is_v21_evaluated_and_civil_comments_overlapping():
    rep = c.check_sources(["civil_comments", "measuring_hate_speech", "jev-bench:banking77"])
    r = rep["results"]
    assert r["civil_comments"]["status"] == "overlapping"
    assert r["jev-bench:banking77"]["status"] == "overlapping"
    assert r["measuring_hate_speech"]["status"] == "v21-evaluated"
    assert rep["v21_evaluated"] == ["measuring_hate_speech"]


def test_eval_only_source_gets_separate_status():
    rep = c.check_sources(["hotpotqa", "massive", "emotion"])
    assert {v["status"] for v in rep["results"].values()} == {"v21-evaluated"}
    assert rep["overlapping"] == [] and rep["clean"] == []


def test_hf_id_match_beats_name():
    rep = c.check_sources(["mystery"], hf_ids={"mystery": "google/civil_comments"})
    assert rep["results"]["mystery"]["status"] == "overlapping"


def test_all_29_v21_ids_present():
    names = {s["name"] for s in c.load_manifest()["sources"] if s["card"].endswith("sources.md")}
    assert len(names) >= 29
    assert {"legacy-datasets/banking77", "ucberkeley-dlab/measuring-hate-speech"} <= names
