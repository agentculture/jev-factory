"""nvsh release-test parity (lapse l15): coverage ported from nvsh 9debdc6.

Sources: ``tests/test_lfm_finetune_release_bundle.py``,
``tests/test_lfm_finetune_dataset_bundle.py`` and ``tests/test_lfm_finetune_scan_bundle.py``.
Only behaviour jev-factory kept is ported; the LFM licence mode, issue-numbered card prose,
the Track A tool-call parser and the argparse ``main()`` were dropped on import and have no
counterpart here.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.release import dataset_bundle as ds
from jev_factory.release import scan
from jev_factory.release.bundle import (
    BundleError,
    BundleFiles,
    build_dataset_bundle,
    build_model_bundle,
    drop_undeclared_mtp,
    results_section,
)
from tests.fixtures.toy_domain import DOMAIN
from tests.release_support import (
    dataset_build_args,
    make_config,
    make_inputs,
    model_bundle_args,
    write_json,
)

# Credential-shaped strings are assembled at run time so this file is never a finding.
_HF = "hf" + "_" + "abcdefghijklmnopqrstuvwxyz0123456789"
_HF_ESCAPED = "\\u0068\\u0066_" + _HF[3:]  # JSON-escaped: no raw "hf_" in the bytes

# ---------------------------------------------------------------------------
# Model bundle helpers
# ---------------------------------------------------------------------------


def _safetensors(path: Path, names: list[str]) -> Path:
    """A minimal safetensors file whose JSON header names *names* (no real data)."""
    header: dict = {"__metadata__": {"format": "pt"}}
    for index, name in enumerate(names):
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [2 * index, 2 * index + 2]}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0\0" * len(names))
    return path


def _build(tmp_path: Path, inp, **overrides):
    kwargs = dict(
        domain=DOMAIN,
        config=make_config(),
        merged=inp.merged,
        base_snapshot=inp.snapshot,
        repo="example-org/toy-lamps-jev-scorer",
        run="r1",
        results=inp.report,
        data_summary="40 toy rows",
        out=tmp_path / "out",
        calibration=inp.calibration,
        gate=inp.gate,
        scorer_train=inp.scorer_train,
    )
    kwargs.update(overrides)
    return build_model_bundle(**model_bundle_args(**kwargs))


def _no_licence(tmp_path, inp):
    (inp.snapshot / "LICENSE").unlink()
    return {}


def _blank_summary(tmp_path, inp):
    return {"data_summary": "  \n "}


def _non_empty_out(tmp_path, inp):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old").write_text("x")
    return {}


def _tableless_report(tmp_path, inp):
    report = tmp_path / "empty.md"
    report.write_text("# nothing\n\nno table here\n")
    return {"results": report}


def _no_reports(tmp_path, inp):
    return {"results": []}


def _gguf_without_file(tmp_path, inp):
    return {"kind": "gguf", "gguf": tmp_path / "missing.gguf"}


def _gguf_none(tmp_path, inp):
    return {"kind": "gguf", "gguf": None}


def _awq_template_mismatch(tmp_path, inp):
    awq = tmp_path / "q" / "awq"
    shutil.copytree(inp.merged, awq)
    (awq / "chat_template.jinja").write_text("other", encoding="utf-8")
    write_json(tmp_path / "q" / "quantize-run.json", {"awq_serve_args": []})
    return {"kind": "awq", "awq_dir": awq}


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (_no_licence, "ships no LICENSE"),
        (_blank_summary, "data_summary"),
        (_non_empty_out, "is not empty"),
        (_tableless_report, "no metric table"),
        (_no_reports, "at least one results report"),
        (_gguf_without_file, "kind gguf needs gguf"),
        (_gguf_none, "kind gguf needs gguf"),
        (_awq_template_mismatch, "AWQ export's chat template differs"),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
def test_model_bundle_refusals(tmp_path, setup, message):
    inp = make_inputs(tmp_path)
    overrides = setup(tmp_path, inp)
    with pytest.raises(BundleError, match=message):
        _build(tmp_path, inp, **overrides)
    out = tmp_path / "out"
    if setup is _non_empty_out:
        assert (out / "old").read_text() == "x"  # an occupied folder is never touched
    else:
        assert not out.exists()  # nothing half-built is left behind


# ---------------------------------------------------------------------------
# MTP head (nvsh t27)
# ---------------------------------------------------------------------------


def test_mtp_head_is_left_alone_when_the_weights_hold_mtp_tensors(tmp_path):
    inp = make_inputs(tmp_path)
    _safetensors(inp.merged / "model.safetensors", ["model.layers.0.w", "mtp.layers.0.w"])
    _build(tmp_path, inp)
    out = tmp_path / "out"
    assert json.loads((out / "config.json").read_text())["mtp_num_hidden_layers"] == 1
    assert "mtp_num_hidden_layers" not in (out / "README.md").read_text()


def test_merged_config_is_never_modified_only_the_bundle_copy(tmp_path):
    inp = make_inputs(tmp_path)
    config = inp.merged / "config.json"
    config.write_text(
        json.dumps({"architectures": ["X"], "mtp_num_hidden_layers": 2}, indent=2), "utf-8"
    )
    before = config.read_bytes()
    _build(tmp_path, inp)
    assert config.read_bytes() == before
    shipped = json.loads((tmp_path / "out" / "config.json").read_text())
    assert shipped == {"architectures": ["X"], "mtp_num_hidden_layers": 0}
    card = (tmp_path / "out" / "README.md").read_text()
    assert "from 2 to 0" in card
    assert "no `mtp.*` tensor" in card


def test_a_sharded_checkpoint_is_read_through_its_index(tmp_path):
    inp = make_inputs(tmp_path)
    (inp.merged / "model.safetensors").unlink()
    shard = "model-00001-of-00001.safetensors"
    _safetensors(inp.merged / shard, ["model.layers.0.w", "mtp.fc.weight"])
    write_json(
        inp.merged / "model.safetensors.index.json",
        {"weight_map": {"model.layers.0.w": shard, "mtp.fc.weight": shard}},
    )
    _build(tmp_path, inp)
    assert json.loads((tmp_path / "out" / "config.json").read_text())["mtp_num_hidden_layers"] == 1


def test_drop_undeclared_mtp_trusts_the_index_over_a_shard_glob(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    write_json(folder / "config.json", {"text_config": {"mtp_num_hidden_layers": 1}})
    # only the index names the tensors: a glob over *.safetensors would find nothing
    write_json(folder / "model.safetensors.index.json", {"weight_map": {"mtp.fc.weight": "s"}})
    assert drop_undeclared_mtp(folder) is None
    assert json.loads((folder / "config.json").read_text())["text_config"] == {
        "mtp_num_hidden_layers": 1
    }
    # without the mtp tensor the nested text_config declaration is zeroed too
    write_json(folder / "model.safetensors.index.json", {"weight_map": {"model.w": "s"}})
    assert drop_undeclared_mtp(folder) == 1
    assert json.loads((folder / "config.json").read_text())["text_config"] == {
        "mtp_num_hidden_layers": 0
    }


def test_drop_undeclared_mtp_writes_a_new_file_never_through_a_link(tmp_path):
    source = write_json(tmp_path / "merged-config.json", {"mtp_num_hidden_layers": 1})
    before = source.read_bytes()
    folder = tmp_path / "copy"
    folder.mkdir()
    os.link(source, folder / "config.json")  # a hard-linked copy shares the run's bytes
    assert drop_undeclared_mtp(folder) == 1
    assert source.read_bytes() == before
    assert json.loads((folder / "config.json").read_text()) == {"mtp_num_hidden_layers": 0}


# ---------------------------------------------------------------------------
# Results quoting
# ---------------------------------------------------------------------------

_REPORT = """# {title}

- Command: `measure --split test.json`

| Metric | bench |
|---|---|
| Bench row | 30 of 32 |

## Metrics

metrics over each model's predictions file.

| Metric | value |
|---|---|
| Right proposals | {right} of 32 |

- a note line

## Background
"""


def _report(path: Path, title: str, right: int) -> Path:
    path.write_text(_REPORT.format(title=title, right=right), encoding="utf-8")
    return path


def test_every_results_report_is_quoted(tmp_path):
    inp = make_inputs(tmp_path)
    first = _report(tmp_path / "final.md", "final run a3", 31)
    edge = _report(tmp_path / "edge-orin.md", "edge run a3 q4", 32)
    _build(tmp_path, inp, results=[first, edge], results_heading="## Metrics")
    card = (tmp_path / "out" / "README.md").read_text()
    assert "### final run a3" in card
    assert "### edge run a3 q4" in card
    assert "From `final.md`" in card
    assert "From `edge-orin.md`" in card
    assert "| Right proposals | 31 of 32 |" in card
    assert "| Right proposals | 32 of 32 |" in card
    # only the table under the heading is quoted: not the bench table, notes or command
    assert "Bench row" not in card
    assert "a note line" not in card
    assert "measure --split" not in card


def test_results_section_picks_the_heading_and_refuses_when_missing(tmp_path):
    report = _report(tmp_path / "r.md", "a run", 29)
    caption, first = results_section(report)
    assert caption == "a run"
    assert "Bench row" in first
    caption, table = results_section(report, "## Metrics")
    assert table.splitlines() == [
        "| Metric | value |",
        "|---|---|",
        "| Right proposals | 29 of 32 |",
    ]
    with pytest.raises(BundleError, match="no '## Missing' section"):
        results_section(report, "## Missing")
    # a heading whose section holds no table is refused, never quoting the next section's
    with pytest.raises(BundleError, match="no metric table"):
        results_section(report, "## Background")
    prose_first = tmp_path / "prose-first.md"
    prose_first.write_text(
        "# r\n\n## Notes\n\nno table here\n\n## Metrics\n\n| a | b |\n|---|---|\n", "utf-8"
    )
    with pytest.raises(BundleError, match="no metric table"):
        results_section(prose_first, "## Notes")
    assert results_section(prose_first, "## Metrics")[1] == "| a | b |\n|---|---|"
    inp = make_inputs(tmp_path / "b")
    with pytest.raises(BundleError, match="no '## Missing' section"):
        _build(tmp_path / "b", inp, results=report, results_heading="## Missing")
    assert not (tmp_path / "b" / "out").exists()


def test_a_report_without_an_h1_is_captioned_by_its_stem(tmp_path):
    report = tmp_path / "edge-run.md"
    report.write_text("| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    assert results_section(report)[0] == "edge-run"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_MODELS = {
    "GENERATOR": "worker",
    "CORRECTOR": "cortex",
    "REVIEWER_A": "senses",
    "REVIEWER_B": "cortex",
}
_ROLE_MODELS = {
    "worker": {"name": "Teacher Worker 35B", "licence": "Apache-2.0"},
    "cortex": {"name": "Teacher Cortex 27B", "licence": "Apache-2.0"},
    "senses": {"name": "Teacher Senses 26B", "licence": "Apache-2.0"},
}


def _entry(entry_id: str, expect: dict, *, source: str = "seed", **extra) -> dict:
    return {
        "id": entry_id,
        "text": f"text {entry_id}",
        "expect": expect,
        "kind": "explicit",
        "source": source,
        **extra,
    }


def _op(name: str = "lamp_status") -> dict:
    return {"operation": name, "args": {}}


def _role_models(tmp_path: Path, table: dict | None = None) -> dict:
    path = write_json(tmp_path / "role-models.json", _ROLE_MODELS if table is None else table)
    return ds.load_role_models(path)


def _dataset(
    tmp_path: Path,
    *,
    train: list[dict] | None = None,
    accepted: list[dict] | None = None,
    test_extra: tuple = (),
    role_models: dict | None = None,
    val_header: str | None = None,
    **overrides,
) -> dict:
    """``dataset_bundle.build`` kwargs for a small run with one variation."""
    splits = tmp_path / "splits"
    splits.mkdir(exist_ok=True)
    val = {"entries": [_entry("dev-v1", {"escalate": True})]}
    if val_header is not None:
        val["header"] = val_header
    write_json(splits / "val.json", val)
    write_json(
        splits / "test.json",
        {"entries": [_entry("dev-t1", {"explain": True, "answer": "a"}), *test_extra]},
    )
    if train is None:
        train = [
            _entry("dev-a", _op(), side="train"),
            _entry("sup-01", {"escalate": True}, side="train", source="supplement-2026-09-23"),
            _entry("dev-a~v1", _op(), side="train", source_id="dev-a"),
        ]
    write_json(tmp_path / "train-augmented.json", {"entries": train})
    if accepted is None:
        accepted = [{"id": "dev-a~v1", "models": _MODELS}]
    (tmp_path / "accepted.jsonl").write_text("".join(json.dumps(r) + "\n" for r in accepted))
    (tmp_path / "rejected.jsonl").write_text(json.dumps({"id": "dev-a~v2"}) + "\n")
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    kwargs = dict(
        domain=DOMAIN,
        splits=splits,
        train_augmented=tmp_path / "train-augmented.json",
        accepted=tmp_path / "accepted.jsonl",
        rejected=tmp_path / "rejected.jsonl",
        licence=tmp_path / "LICENSE",
        role_models=role_models if role_models is not None else _role_models(tmp_path),
        scorer_train=write_json(tmp_path / "scorer-src.json", {"header": "h", "entries": []}),
        out=tmp_path / "bundle",
    )
    kwargs.update(overrides)
    return dataset_build_args(**kwargs)


def _manifest(out: Path) -> dict[str, dict]:
    return {row["id"]: row for row in json.loads((out / "manifest.json").read_text())}


def _card(tmp_path: Path) -> str:
    return (tmp_path / "bundle" / "README.md").read_text()


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------


def test_dataset_manifest_records_origin_source_teachers_and_transformation(tmp_path):
    train = [
        _entry("dev-a", _op(), side="train"),
        _entry("sup-01", {"escalate": True}, side="train", source="supplement-2026-09-23"),
        _entry("t-01", {"escalate": True}, side="train", source="t15-diagnosis-explain"),
        _entry("dr-01", _op("list_rooms"), side="train", source="draft-sources"),
        _entry("dev-a~v1", _op(), side="train", source_id="dev-a"),
    ]
    counts = ds.build(**_dataset(tmp_path, train=train))
    out = tmp_path / "bundle"
    manifest = _manifest(out)
    assert {k: manifest[k]["origin"] for k in manifest} == {
        "dev-a": "corpus",
        "sup-01": "supplement",
        "t-01": "supplement",
        "dr-01": "draft",
        "dev-a~v1": "variation",
        "dev-v1": "corpus",
        "dev-t1": "corpus",
    }
    assert (counts["corpus"], counts["supplement"], counts["draft"], counts["variation"]) == (
        1,
        2,
        1,
        1,
    )
    variation = manifest["dev-a~v1"]
    assert variation["transformed"] is True
    assert variation["source"] == "seed"
    assert variation["source_id"] == "dev-a"
    assert variation["teachers"] == {
        "GENERATOR": "Teacher Worker 35B",
        "CORRECTOR": "Teacher Cortex 27B",
        "REVIEWER_A": "Teacher Senses 26B",
        "REVIEWER_B": "Teacher Cortex 27B",
    }
    for rid, row in manifest.items():
        assert {"split", "origin", "source", "source_id", "teachers", "transformed"} <= set(row)
        if rid != "dev-a~v1":
            assert row["transformed"] is False
            assert row["teachers"] == {}
            assert row["source_id"] == rid
    assert manifest["sup-01"]["source"] == "supplement-2026-09-23"
    assert manifest["dev-a"]["source_file"] == DOMAIN.seed_corpus.name
    assert manifest["dev-t1"]["split"] == "test"
    assert manifest["dev-v1"]["split"] == "validation"
    for split in ("train", "validation", "test"):
        for line in (out / "data" / f"{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            assert record["source_id"] == manifest[record["id"]]["source_id"]
            assert record["split"] == manifest[record["id"]]["split"]
    test_line = next(line for line in _card(tmp_path).splitlines() if line.startswith("| test |"))
    assert "never trained on" in test_line


# ---------------------------------------------------------------------------
# Dataset teacher refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "accepted",
    [[{"id": "dev-a~v1"}], [{"id": "dev-a~v1", "models": {}}], []],
    ids=["no-models", "empty-models", "no-accepted-row"],
)
def test_a_variation_with_no_models_is_refused(tmp_path, accepted):
    kwargs = _dataset(tmp_path, accepted=accepted)
    with pytest.raises(ValueError, match="dev-a~v1 has no accepted record naming its models"):
        ds.build(**kwargs)


def test_a_variation_missing_one_role_is_refused(tmp_path):
    models = {k: v for k, v in _MODELS.items() if k != "REVIEWER_B"}
    kwargs = _dataset(tmp_path, accepted=[{"id": "dev-a~v1", "models": models}])
    with pytest.raises(ValueError, match="names no REVIEWER_B teacher"):
        ds.build(**kwargs)


def test_an_unknown_teacher_alias_is_refused(tmp_path):
    table = {"worker": ("Teacher Worker 35B", "Apache-2.0")}
    kwargs = _dataset(tmp_path, role_models=table)
    with pytest.raises(ValueError, match="'cortex' is not in the run's teacher-models table"):
        ds.build(**kwargs)


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"worker": {"name": "Qwen"}}, "'worker' needs a non-empty 'name' and 'licence'"),
        ({"worker": {"name": "", "licence": "Apache-2.0"}}, "non-empty 'name'"),
        ({"worker": "Qwen"}, "needs a non-empty"),
        ({}, "expected a JSON object"),
        ([["worker", "Qwen"]], "expected a JSON object"),
    ],
)
def test_load_role_models_refuses_a_malformed_table(tmp_path, table, message):
    path = write_json(tmp_path / "bad.json", table)
    with pytest.raises(ValueError, match=message):
        ds.load_role_models(path)


def test_load_role_models_reads_name_and_licence(tmp_path):
    assert _role_models(tmp_path)["cortex"] == ("Teacher Cortex 27B", "Apache-2.0")


def test_a_variation_on_a_non_train_side_is_refused(tmp_path):
    extra = (_entry("dev-t1~v1", {"escalate": True}),)
    kwargs = _dataset(tmp_path, test_extra=extra)
    with pytest.raises(ValueError, match="never leave the train side"):
        ds.build(**kwargs)


def test_a_train_file_entry_marked_for_another_side_is_refused(tmp_path):
    train = [_entry("dev-a", _op(), side="validation")]
    kwargs = _dataset(tmp_path, train=train, accepted=[])
    with pytest.raises(ValueError, match="is not a train-side entry"):
        ds.build(**kwargs)


def test_a_record_without_source_is_refused(tmp_path):
    train = [_entry("dev-a", _op(), side="train")]
    del train[0]["source"]
    kwargs = _dataset(tmp_path, train=train, accepted=[])
    with pytest.raises(ValueError, match="needs a 'source' field"):
        ds.build(**kwargs)
    assert not (tmp_path / "bundle").exists()


def test_default_source_fills_a_missing_source(tmp_path):
    train = [_entry("dev-a", _op(), side="train"), _entry("dev-b", _op(), side="train")]
    del train[0]["source"]
    extra = ({"id": "dev-t2", "text": "x", "expect": {"escalate": True}},)
    ds.build(
        **_dataset(tmp_path, train=train, accepted=[], test_extra=extra),
        default_source="draft-sources",
    )
    manifest = _manifest(tmp_path / "bundle")
    assert manifest["dev-a"]["source"] == "draft-sources"
    assert manifest["dev-a"]["origin"] == "draft"
    assert manifest["dev-t2"]["source"] == "draft-sources"
    assert manifest["dev-b"]["source"] == "seed"  # an explicit source is kept


# ---------------------------------------------------------------------------
# Teacher aggregation and disclosure
# ---------------------------------------------------------------------------


def _two_variations(tmp_path: Path, second: dict, table: dict, **kw) -> dict:
    train = [
        _entry("dev-a", _op(), side="train"),
        _entry("dev-b", _op(), side="train"),
        _entry("dev-a~v1", _op(), side="train", source_id="dev-a"),
        _entry("dev-b~v1", _op(), side="train", source_id="dev-b"),
    ]
    accepted = [{"id": "dev-a~v1", "models": _MODELS}, {"id": "dev-b~v1", "models": second}]
    return _dataset(
        tmp_path, train=train, accepted=accepted, role_models=_role_models(tmp_path, table), **kw
    )


def test_every_distinct_teacher_per_role_is_listed(tmp_path):
    table = {**_ROLE_MODELS, "oracle": {"name": "Teacher Oracle 22B", "licence": "Apache-2.0"}}
    counts = ds.build(**_two_variations(tmp_path, {**_MODELS, "REVIEWER_A": "oracle"}, table))
    assert counts["variation"] == 2
    card = _card(tmp_path)
    assert "| Teacher Senses 26B | Apache-2.0 | accepted it (reviewer A) |" in card
    assert "| Teacher Oracle 22B | Apache-2.0 | accepted it (reviewer A) |" in card


def test_apache_only_checks_every_teacher_not_only_the_first(tmp_path):
    table = {**_ROLE_MODELS, "oracle": {"name": "Closed Oracle", "licence": "Proprietary"}}
    kwargs = _two_variations(tmp_path, {**_MODELS, "REVIEWER_A": "oracle"}, table)
    with pytest.raises(ValueError, match="dev-b~v1: teacher 'Closed Oracle'.*apache_only"):
        ds.build(**kwargs, apache_only=True)
    assert ds.build(**kwargs)["variation"] == 2  # allowed when not apache_only


def test_shared_corrector_reviewer_b_is_detected_by_resolved_name(tmp_path):
    table = {**_ROLE_MODELS, "cortex-2": {"name": "Teacher Cortex 27B", "licence": "Apache-2.0"}}
    models = {**_MODELS, "REVIEWER_B": "cortex-2"}  # two aliases, one model
    accepted = [{"id": "dev-a~v1", "models": models}]
    ds.build(**_dataset(tmp_path, accepted=accepted, role_models=_role_models(tmp_path, table)))
    card = _card(tmp_path)
    assert card.lower().count("reviewer b is also the corrector") == 1
    assert "(Teacher Cortex 27B)" in card


def test_shared_corrector_reviewer_b_is_disclosed_once(tmp_path):
    # both variations share cortex as corrector and reviewer B: one disclosure, one name
    ds.build(**_two_variations(tmp_path, dict(_MODELS), _ROLE_MODELS))
    card = _card(tmp_path)
    assert card.lower().count("reviewer b is also the corrector") == 1
    assert "(Teacher Cortex 27B):" in card


def test_no_disclosure_when_corrector_and_reviewer_b_differ(tmp_path):
    models = {**_MODELS, "REVIEWER_B": "senses"}
    ds.build(**_dataset(tmp_path, accepted=[{"id": "dev-a~v1", "models": models}]))
    assert "reviewer b is also the corrector" not in _card(tmp_path).lower()


def _reviewer_b_row(row_id: str = "dev-a~v1") -> dict:
    return {
        "id": row_id,
        "models": _MODELS,
        "decided_by": "reviewer_b",
        "verdicts": {"reviewer_a": {"accept": False}, "reviewer_b": {"accept": True}},
    }


def _rereview_row(row_id: str = "dev-a~v1") -> dict:
    return {
        "id": row_id,
        "models": _MODELS,
        "verdicts": {"reviewer_b": {"accept": True}},
        "prior_verdicts": {"reviewer_a": {"accept": True}},
    }


@pytest.mark.parametrize("row", [_reviewer_b_row(), _rereview_row()], ids=["fresh", "rereview"])
def test_a_run_decided_by_reviewer_b_says_reviewer_a_was_advisory(tmp_path, row):
    ds.build(**_dataset(tmp_path, accepted=[row]))
    card = _card(tmp_path)
    assert "kept only when both said yes" not in card
    assert "reviewer B's verdict alone decided" in card
    assert "| Teacher Senses 26B | Apache-2.0 | reviewer A, advisory: asked and recorded" in card
    assert "| Teacher Cortex 27B | Apache-2.0 | accepted it (reviewer B, deciding) |" in card


def test_a_run_decided_by_both_reviewers_keeps_the_both_wording(tmp_path):
    ds.build(**_dataset(tmp_path))
    card = _card(tmp_path)
    assert "kept only when both said yes" in card
    assert "| Teacher Senses 26B | Apache-2.0 | accepted it (reviewer A) |" in card
    assert "| Teacher Cortex 27B | Apache-2.0 | accepted it (reviewer B) |" in card


def test_decision_sentence_and_role_description_cover_every_mix():
    both, by_b, mixed = {"both": 3}, {"reviewer_b": 2}, {"both": 2, "reviewer_b": 1}
    assert ds.decision_sentence(both).endswith("kept only when both said yes.")
    assert "reviewer B's verdict alone decided" in ds.decision_sentence(by_b)
    assert "both" not in ds.decision_sentence(by_b).split(":")[1]
    sentence = ds.decision_sentence(mixed)
    assert "For 2 kept variations both said yes; for 1, reviewer B's verdict" in sentence
    assert ds.role_description("REVIEWER_A", both) == "accepted it (reviewer A)"
    assert ds.role_description("REVIEWER_A", by_b).startswith("reviewer A, advisory")
    assert ds.role_description("REVIEWER_A", mixed) == (
        "accepted it (reviewer A; deciding for 2 variations, advisory for 1)"
    )
    assert ds.role_description("REVIEWER_B", by_b) == "accepted it (reviewer B, deciding)"
    assert ds.role_description("REVIEWER_B", mixed) == "accepted it (reviewer B)"
    assert ds.role_description("GENERATOR", by_b) == "wrote the variation"


def test_teacher_summary_derives_roles_decisions_and_shared_names(tmp_path):
    train = [{"id": "dev-a", "text": "t"}, {"id": "dev-a~v1", "text": "u"}]
    summary = ds.teacher_summary(
        train, {"dev-a~v1": _reviewer_b_row()}, _role_models(tmp_path), apache_only=True
    )
    assert summary.role_teachers["GENERATOR"] == {"Teacher Worker 35B": "Apache-2.0"}
    assert summary.role_teachers["REVIEWER_B"] == {"Teacher Cortex 27B": "Apache-2.0"}
    assert summary.decisions == {"reviewer_b": 1}
    assert summary.shared_corrector_reviewer_names == ["Teacher Cortex 27B"]
    assert list(summary.per_variation) == ["dev-a~v1"]  # the non-variation is skipped
    rows = ds.teacher_rows(summary)
    assert [r[0] for r in rows] == [
        "Teacher Worker 35B",
        "Teacher Cortex 27B",
        "Teacher Senses 26B",
        "Teacher Cortex 27B",
    ]
    assert "advisory" in rows[2][2]
    table = {**_ROLE_MODELS, "senses": {"name": "Closed", "licence": "Proprietary"}}
    rows, role_models = {"dev-a~v1": _reviewer_b_row()}, _role_models(tmp_path, table)
    with pytest.raises(ValueError, match="not Apache-2.0; refused by apache_only"):
        ds.teacher_summary([{"id": "dev-a~v1"}], rows, role_models, apache_only=True)


def test_decision_rule_reads_decided_by_and_verdicts():
    assert ds.decision_rule({"decided_by": "reviewer_b"}) == "reviewer_b"
    assert ds.decision_rule({"verdicts": {"reviewer_b": {}}}) == "reviewer_b"
    assert ds.decision_rule({"verdicts": {"reviewer_a": {}, "reviewer_b": {}}}) == "both"
    assert ds.decision_rule({}) == "both"


# ---------------------------------------------------------------------------
# Dataset card counts and redaction
# ---------------------------------------------------------------------------


def test_the_card_names_the_splits_seed(tmp_path):
    ds.build(**_dataset(tmp_path, val_header="Corpus. Split 'val' of seed.json (seed=46)."))
    assert "(seed 46)" in _card(tmp_path)


def test_a_split_without_a_seed_says_seeded(tmp_path):
    ds.build(**_dataset(tmp_path))
    assert "(seeded)" in _card(tmp_path)


def test_the_card_lists_the_model_repos(tmp_path):
    repos = ["example-org/toy-jev-scorer", "example-org/toy-jev-scorer-gguf"]
    ds.build(**_dataset(tmp_path), model_repos=repos)
    card = _card(tmp_path)
    assert "`example-org/toy-jev-scorer`, `example-org/toy-jev-scorer-gguf`" in card
    assert "the models of this run" not in card


def test_several_rejected_files_are_summed(tmp_path):
    kwargs = _dataset(tmp_path)
    second = tmp_path / "rereview-rejected.jsonl"
    second.write_text(json.dumps({"id": "dev-a~v3"}) + "\n\n" + json.dumps({"id": "x"}) + "\n")
    sources = replace(kwargs["sources"], rejected=[kwargs["sources"].rejected, second])
    counts = ds.build(**{**kwargs, "sources": sources})
    assert counts["rejected"] == 3
    assert counts["accepted"] == 1
    assert "Of 4\n  reviewed rewrites, 1 were accepted (25%)" in _card(tmp_path)


@pytest.mark.parametrize(
    ("text", "published"),
    [
        ("tailnet 100.93.248.8 or 8.8.8.8", "tailnet 192.0.2.8 or 8.8.8.8"),
        ("edge 100.64.0.1 and 100.127.255.254", "edge 192.0.2.1 and 192.0.2.254"),
        ("outside 100.63.0.1 and 100.128.0.1", "outside 100.63.0.1 and 100.128.0.1"),
        ("lan 10.0.0.5 and 172.16.3.4", "lan 192.0.2.5 and 192.0.2.4"),
        ("local 127.0.0.1 and 0.0.0.0", "local 127.0.0.1 and 0.0.0.0"),
        ("version 5.5.0.1 and 999.1.1.1", "version 5.5.0.1 and 999.1.1.1"),
    ],
)
def test_publishable_text_redacts_private_and_cgnat_addresses(text, published):
    assert ds.publishable_text(text) == published


def _full_dataset(tmp_path: Path, scorer_doc: dict, test_text: str) -> Path:
    kwargs = _dataset(tmp_path)
    test = json.loads((kwargs["sources"].splits / "test.json").read_text())
    test["entries"].append({**_entry("dev-t2", {"escalate": True}), "text": test_text})
    write_json(kwargs["sources"].splits / "test.json", test)
    inp = make_inputs(tmp_path / "m")
    build_dataset_bundle(
        domain=DOMAIN,
        config=make_config(),
        sources=kwargs["sources"],
        licence_file=kwargs["licence"],
        role_models=kwargs["role_models"],
        out=tmp_path / "bundle",
        files=BundleFiles(
            calibration=inp.calibration,
            gate=inp.gate,
            scorer_train=write_json(tmp_path / "scorer-train-src.json", scorer_doc),
        ),
    )
    return tmp_path / "bundle"


def test_scorer_train_is_redacted_and_named_in_the_card(tmp_path):
    doc = {
        "header": "Split 'train' of x",
        "entries": [{"id": "a", "text": "ssh into 192.168.1.50 then 100.93.248.8", "expect": {}}],
    }
    out = _full_dataset(tmp_path, doc, "ssh into 10.1.2.3, not 127.0.0.1")
    shipped = json.loads((out / "scorer-train.json").read_text())
    assert shipped["entries"][0]["text"] == "ssh into 192.0.2.50 then 192.0.2.8"
    assert shipped["header"] == "Split 'train' of x"
    record = json.loads((out / "data" / "test.jsonl").read_text().splitlines()[-1])
    assert record["text"] == "ssh into 192.0.2.3, not 127.0.0.1"
    card = (out / "README.md").read_text()
    assert "`scorer-train.json` is the file the candidate scorer trained on" in card
    assert "1 record(s) named a private network address" in card
    assert "documentation address 192.0.2.x (RFC 5737)" in card
    assert scan.write_scan(out)["clean"] is True  # nothing private survives into the bundle


def test_no_redaction_note_when_nothing_was_redacted(tmp_path):
    counts = ds.build(**_dataset(tmp_path))
    assert counts["redacted_hosts"] == 0
    assert "private network address" not in _card(tmp_path)


# ---------------------------------------------------------------------------
# Scan edge cases (nvsh scan_bundle tests)
# ---------------------------------------------------------------------------


def _folder(tmp_path: Path, **files) -> Path:
    folder = tmp_path / "b"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    return folder


def _safetensors_bytes(header, data: bytes = b"\x00\x01") -> bytes:
    raw = json.dumps(header).encode("utf-8")
    return len(raw).to_bytes(8, "little") + raw + data


def test_folder_hash_ignores_scan_json(tmp_path):
    folder = _folder(tmp_path, **{"a.txt": "data\n"})
    before = scan.folder_hash(folder)
    payload = scan.write_scan(folder)
    assert scan.folder_hash(folder) == before == payload["hash"]
    (folder / "scan.json").write_text("{}")  # rewriting scan.json alone changes nothing
    assert scan.folder_hash(folder) == before
    (folder / "b.txt").write_text("more")
    assert scan.folder_hash(folder) != before


def test_scan_payload_keys_and_shape(tmp_path):
    folder = _folder(tmp_path, **{"env.txt": f"HF_TOKEN={_HF}\n", "m.safetensors": b""})
    (folder / "m.safetensors").write_bytes(_safetensors_bytes({"__metadata__": {}}))
    scan.write_scan(folder)
    payload = json.loads((folder / "scan.json").read_text())
    assert set(payload) == {"hash", "findings", "clean", "binaries"}
    assert isinstance(payload["hash"], str)
    assert len(payload["hash"]) == 64
    int(payload["hash"], 16)
    assert payload["clean"] is False
    assert payload["binaries"] == ["m.safetensors"]
    assert payload["findings"]
    for finding in payload["findings"]:
        assert set(finding) == {"path", "line", "kind", "detail"}
        assert isinstance(finding["line"], int)


def test_jsonl_finding_carries_the_physical_line_number(tmp_path):
    body = '{"ok": "nothing here"}\n\n{"field": "' + _HF_ESCAPED + '"}\n'
    folder = _folder(tmp_path, **{"records.jsonl": body})
    hits = [
        f for f in scan.scan_folder(folder) if f["kind"] == "redact" and f["detail"] == "hf_token"
    ]
    assert len(hits) == 1
    assert hits[0]["line"] == 3


def test_json_scan_recurses_through_nested_objects_and_arrays(tmp_path):
    body = '{"outer": [{"inner": ["x", {"deep": "' + _HF_ESCAPED + '"}]}]}'
    folder = _folder(tmp_path, **{"nested.json": body})
    findings = scan.scan_folder(folder)
    assert any(f["kind"] == "redact" and f["detail"] == "hf_token" for f in findings)
    assert all(f["path"] == "nested.json" for f in findings)


def test_a_token_in_safetensors_metadata_is_found(tmp_path):
    header = {"__metadata__": {"note": _HF}}
    folder = _folder(tmp_path, **{"model.safetensors": _safetensors_bytes(header)})
    findings = scan.scan_folder(folder)
    assert any(f["kind"] == "redact" and f["detail"] == "hf_token" for f in findings)
    assert all(f["path"] == "model.safetensors" for f in findings)


def test_a_bin_file_is_scanned_like_any_other_file(tmp_path):
    folder = _folder(
        tmp_path,
        **{"notes.bin": f"token {_HF}\n", "training_args.bin": b"PK\x03\x04\x80\x02\xff\xfe"},
    )
    findings = scan.scan_folder(folder)
    assert any(f["path"] == "notes.bin" and f["detail"] == "hf_token" for f in findings)
    assert {
        "path": "training_args.bin",
        "line": 0,
        "kind": "unscanned_binary",
        "detail": "non-UTF-8 file",
    } in findings
    assert scan.write_scan(folder)["binaries"] == []


def test_private_lan_cgnat_and_local_hosts_are_flagged(tmp_path):
    folder = _folder(
        tmp_path,
        **{
            "README.md": "Served at http://192.168.1.138:8000/v1 during the run.\n",
            "notes.txt": "gateway on 10.0.0.5 and 100.93.248.8, box node2.local\n",
        },
    )
    findings = scan.scan_folder(folder)
    hosts = {f["detail"] for f in findings if f["kind"] == "private_host"}
    assert hosts == {"192.168.1.138", "10.0.0.5", "100.93.248.8", "node2.local"}
    lines = {(f["path"], f["line"]) for f in findings if f["kind"] == "private_host"}
    assert lines == {("README.md", 1), ("notes.txt", 1)}
    assert scan.write_scan(folder)["clean"] is False


def test_public_urls_versions_and_documentation_nets_are_not_private_hosts(tmp_path):
    text = (
        "Try http://localhost:8000 or 127.0.0.1; see https://huggingface.co/Qwen and"
        " https://github.com/agentculture/nvsh; transformers 5.5.0, torch 2.12.1, 8.8.8.8,"
        " version 10.0.0.5.1 and docs nets 192.0.2.50, 198.51.100.7, 203.0.113.9.\n"
    )
    folder = _folder(tmp_path, **{"README.md": text})
    assert [f for f in scan.scan_folder(folder) if f["kind"] == "private_host"] == []


@pytest.mark.parametrize(
    "content",
    [
        b"\x00\x01\x02\x03",  # shorter than the 8-byte length prefix
        (2).to_bytes(8, "little") + b"[]",  # a header that is not an object
        (1 << 40).to_bytes(8, "little") + b"{}",  # an implausibly huge header length
        (100).to_bytes(8, "little") + b'{"a": 1}',  # a header cut off before its length
        (5).to_bytes(8, "little") + b"\xff\xfe\xfd\xfc\xfb",  # a header that is not UTF-8
    ],
    ids=["short", "array", "huge", "truncated", "non-utf8"],
)
def test_a_malformed_safetensors_header_is_unrecognised_not_a_crash(tmp_path, content):
    folder = _folder(tmp_path, **{"model.safetensors": content})
    findings = scan.scan_folder(folder)
    assert [(f["path"], f["kind"]) for f in findings] == [
        ("model.safetensors", "unrecognised_binary")
    ]
    assert scan.write_scan(folder)["clean"] is False


def test_weight_files_that_prove_their_format_are_listed_not_flagged(tmp_path):
    folder = _folder(
        tmp_path,
        **{
            "model.safetensors": _safetensors_bytes({"__metadata__": {"format": "pt"}}),
            "adapter.gguf": b"GGUF\x03\x00\x00\x00",
            "readme.txt": "harmless\n",
        },
    )
    assert scan.scan_folder(folder) == []
    payload = scan.write_scan(folder)
    assert payload["clean"] is True
    assert payload["binaries"] == ["adapter.gguf", "model.safetensors"]
