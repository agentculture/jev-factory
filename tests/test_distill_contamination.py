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


def test_alias_and_training_match():
    rep = c.check_sources(["PolyAI/banking77", "nyu-mll/glue", "clinc_oos"])
    assert all(r["status"] == "overlapping" for r in rep["results"].values())


def test_every_manifest_entry_cites_a_url():
    m = c.load_manifest()
    assert m["sources"]
    assert all(
        s["source_url"].startswith("https://") and s["card"].startswith("https://")
        for s in m["sources"]
    )
    assert {s["role"] for s in m["sources"]} >= {"training", "clef-decision-index-eval"}


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


def test_hate_speech_is_stock_evaluated_and_civil_comments_overlapping():
    rep = c.check_sources(["civil_comments", "measuring_hate_speech", "jev-bench:banking77"])
    r = rep["results"]
    assert r["civil_comments"]["status"] == "overlapping"
    assert r["jev-bench:banking77"]["status"] == "overlapping"
    assert r["measuring_hate_speech"]["status"] == "stock-evaluated"
    assert rep["stock_evaluated"] == ["measuring_hate_speech"]


def test_eval_only_source_gets_separate_status():
    rep = c.check_sources(["hotpotqa", "massive", "emotion"])
    assert {v["status"] for v in rep["results"].values()} == {"stock-evaluated"}
    assert rep["overlapping"] == []
    assert rep["clean"] == []


def test_hf_id_match_beats_name():
    rep = c.check_sources(["mystery"], hf_ids={"mystery": "google/civil_comments"})
    assert rep["results"]["mystery"]["status"] == "overlapping"


def test_manifest_lists_the_48_card_datasets_and_clef_suites():
    """t6 criterion 1: exactly the 2610 card's 48 datasets plus Clef Decision Index suites."""
    sources = c.load_manifest()["sources"]
    card = [s for s in sources if s["role"] != "clef-decision-index-eval"]
    clef = [s for s in sources if s["role"] == "clef-decision-index-eval"]
    assert len(card) == 48
    assert len({s["name"] for s in card}) == 48
    assert clef
    assert "216d1ccb46a9e9b036eb9187b248db98480f77d4" in card[0]["source_url"]
    assert {"legacy-datasets/banking77", "tasksource/tasksource-jev-typed-decisions"} <= {
        s["name"] for s in card
    }


def test_eval_only_label_is_model_neutral():
    """t6 criterion 1: no manifest role or status is named after v21."""
    roles = {s["role"] for s in c.load_manifest()["sources"]}
    assert roles == {"training", "stock-evaluated", "clef-decision-index-eval"}
    assert c.STOCK_EVALUATED == "stock-evaluated"


def test_tasksource_typed_decisions_hf_id_is_overlapping():
    """t6 criterion 2: a record with hf_id tasksource/tasksource-jev-typed-decisions overlaps."""
    rep = c.check_sources(
        ["some-source"], hf_ids={"some-source": "tasksource/tasksource-jev-typed-decisions"}
    )
    assert rep["results"]["some-source"]["status"] == "overlapping"


def test_localllama_typed_decisions_is_overlapping_fail_closed():
    """t6 criterion 2: LocalLLaMA/typed-decisions maps to the upstream set, so overlapping."""
    rep = c.check_sources(["LocalLLaMA/typed-decisions"])
    assert rep["results"]["LocalLLaMA/typed-decisions"]["status"] == "overlapping"
    assert rep["overlapping"] == ["LocalLLaMA/typed-decisions"]


def test_jetson_ops_source_is_clean():
    """t6 criterion 2: a Jetson-ops source is clean."""
    rep = c.check_sources(["jetson-ops", "jetson-ops/hand-written-v1"])
    assert rep["clean"] == ["jetson-ops", "jetson-ops/hand-written-v1"]


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


def test_report_refuses_a_contamination_file_symlinked_out_of_the_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    target = tmp_path / "elsewhere.json"
    target.write_text("keep", encoding="utf-8")
    (run / "contamination.json").symlink_to(target)
    report = c.check_sources(["banking77"])
    with pytest.raises(CliError, match="escapes"):
        c.write_report(report, run)
    assert target.read_text(encoding="utf-8") == "keep"
