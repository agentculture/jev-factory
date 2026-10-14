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
