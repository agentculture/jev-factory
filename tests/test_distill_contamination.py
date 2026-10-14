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
        s["source_url"].startswith("https://") and s["card"].startswith("https://")
        for s in m["sources"]
    )
    assert {s["role"] for s in m["sources"]} >= {"v21-training", "clef-decision-index-eval"}


def test_matches_carry_url():
    rep = c.check_sources(["banking77"])
    assert rep["results"]["banking77"]["matches"][0]["source_url"].startswith("https://")


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


def test_hf_id_is_derived_from_records_automatically():
    import json as _json
    from pathlib import Path

    from jev_factory.distill import records as R

    row = _json.loads(
        (
            Path(__file__).parent / "fixtures/typed_decisions/real_jev_bench_banking77_row.json"
        ).read_text()
    )
    rec = R.from_jev_bench_row(row)
    assert rec.hf_id == "mteb/banking77"
    name = rec.source  # "Praveenrajus/jev-bench:banking77"
    # the config name inside the source is matched even without records (lapse l13)
    assert c.check_sources([name])["results"][name]["status"] == "overlapping"
    rep = c.check_sources([name], records=[rec])
    assert rep["results"][name]["status"] == "overlapping"
    m = rep["results"][name]["matches"][0]
    assert m["source_url"].startswith("https://huggingface.co/")
    # an explicit hf_ids argument is tried first, the config-name match still applies
    rep = c.check_sources([name], records=[rec], hf_ids={name: "nobody/nothing"})
    assert rep["results"][name]["status"] == "overlapping"
    assert [m["name"] for m in rep["results"][name]["matches"]][0] != "mteb/banking77"


def _real_record(name):
    from pathlib import Path

    from jev_factory.distill.records import from_jev_bench_row

    path = Path(__file__).parent / "fixtures" / "typed_decisions" / name
    return from_jev_bench_row(json.loads(path.read_text(encoding="utf-8")))


def test_real_banking77_record_source_overlaps_with_and_without_records():
    rec = _real_record("real_jev_bench_banking77_row.json")
    assert rec.source == "Praveenrajus/jev-bench:banking77"
    bare = c.check_sources([rec.source])
    with_records = c.check_sources([rec.source], records=[rec])
    assert bare["results"][rec.source]["status"] == "overlapping"
    assert with_records["results"][rec.source]["status"] == "overlapping"


def test_real_go_emotions_record_source_matches_manifest_status():
    rec = _real_record("real_jev_bench_go_emotions_row.json")
    assert rec.source == "Praveenrajus/jev-bench:go_emotions"
    expected = c.check_sources(["go_emotions"])["results"]["go_emotions"]["status"]
    bare = c.check_sources([rec.source])["results"][rec.source]["status"]
    with_records = c.check_sources([rec.source], records=[rec])["results"][rec.source]["status"]
    assert bare == with_records == expected


@pytest.mark.parametrize(
    "name",
    ["jev-bench:banking77", "Praveenrajus/jev-bench:banking77", "other-org/jev-bench:banking77"],
)
def test_any_org_jev_bench_prefix_is_normalised(name):
    assert c.check_sources([name])["results"][name]["status"] == "overlapping"
