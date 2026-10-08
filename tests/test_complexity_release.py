"""Characterization tests for the release and review functions refactored under d17.

They pin the branches the rest of the suite did not reach (inputs -> outputs, exceptions
and messages), so the cognitive-complexity refactor of these functions stays
behaviour-neutral: ``bundle_problems``, ``build_model_bundle``, ``_resolve_roles``,
``_answer_counts``, ``dataset_bundle.build``, ``private_hosts``, ``_scan_json_strings``,
``verb_tree``, ``make_record``, ``_plan_one`` (through ``plan_changes``) and the review
site's request handler.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import shutil
import threading
from pathlib import Path

import pytest

from jev_factory.domain.model import Operation
from jev_factory.release import bundle as bundle_module
from jev_factory.release import dataset_bundle, scan
from jev_factory.release.bundle import BundleError, build_model_bundle, bundle_problems
from jev_factory.review import core, server
from tests.fixtures.toy_domain import DOMAIN
from tests.release_support import (
    dataset_build_args,
    make_config,
    make_inputs,
    model_bundle_args,
    write_gguf,
    write_json,
)

AT = "2026-10-02T12:00:00Z"


# -- bundle_problems -----------------------------------------------------------------


def _good_bundle(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for name in ("calibration.json", "gate.json", "scorer-train.json"):
        write_json(folder / name, {})
    write_json(folder / "bundle.json", {"surface_sha256": "ab" * 32})
    return folder


def test_bundle_problems_of_a_path_that_is_not_a_folder(tmp_path):
    missing = tmp_path / "nowhere"
    assert bundle_problems(missing) == [f"{missing} is not a folder"]
    a_file = tmp_path / "f.txt"
    a_file.write_text("x")
    assert bundle_problems(a_file) == [f"{a_file} is not a folder"]


def test_bundle_problems_of_a_clean_bundle_is_empty(tmp_path):
    assert bundle_problems(_good_bundle(tmp_path / "b")) == []


def test_bundle_problems_without_bundle_json_names_only_the_missing_file(tmp_path):
    folder = _good_bundle(tmp_path / "b")
    (folder / "bundle.json").unlink()
    (folder / "gate.json").unlink()
    assert bundle_problems(folder) == ["missing gate.json", "missing bundle.json"]


@pytest.mark.parametrize("text", ["not json {", "[1, 2]", "{}", '{"surface_sha256": ""}'])
def test_bundle_problems_of_an_unusable_bundle_json(tmp_path, text):
    folder = _good_bundle(tmp_path / "b")
    (folder / "bundle.json").write_text(text)
    assert bundle_problems(folder) == ["bundle.json records no surface_sha256"]


def test_bundle_problems_skips_a_symlinked_gguf_after_refusing_the_link(tmp_path):
    folder = _good_bundle(tmp_path / "b")
    real = write_gguf(tmp_path / "real.gguf", {"quantize.imatrix.file": "/abs/imatrix.dat"})
    (folder / "link.gguf").symlink_to(real)
    assert bundle_problems(folder) == ["symlink link.gguf: a bundle never holds a link"]


def test_bundle_problems_reports_gguf_paths_and_unreadable_ggufs_in_order(tmp_path):
    folder = _good_bundle(tmp_path / "b")
    (folder / "sub").mkdir()
    write_gguf(folder / "sub" / "a.gguf", {"quantize.imatrix.file": "/abs/imatrix.dat"})
    (folder / "z.gguf").write_bytes(b"NOPE")
    assert bundle_problems(folder) == [
        "sub/a.gguf: GGUF metadata quantize.imatrix.file holds an absolute path",
        "z.gguf: no GGUF magic",
    ]


# -- build_model_bundle --------------------------------------------------------------


def _model_kwargs(tmp_path: Path, inp, **overrides):
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
    return model_bundle_args(**kwargs)


def _teachers(role_teachers):
    summary = dataset_bundle.TeacherSummary()
    summary.role_teachers = role_teachers
    return summary


def test_an_unknown_bundle_kind_is_refused_first(tmp_path):
    inp = make_inputs(tmp_path)
    kwargs = _model_kwargs(tmp_path, inp, kind="fp8", calibration=None)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == "unknown bundle kind 'fp8' (one of bf16, gguf, awq)"


@pytest.mark.parametrize("awq_dir", [None, "missing"])
def test_kind_awq_needs_an_existing_export_folder(tmp_path, awq_dir):
    inp = make_inputs(tmp_path)
    folder = None if awq_dir is None else tmp_path / awq_dir
    kwargs = _model_kwargs(tmp_path, inp, kind="awq", awq_dir=folder, calibration=None)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == "kind awq needs awq_dir, an existing export folder"


@pytest.mark.parametrize("gguf", [None, "missing.gguf"])
def test_kind_gguf_needs_an_existing_file(tmp_path, gguf):
    inp = make_inputs(tmp_path)
    path = None if gguf is None else tmp_path / gguf
    kwargs = _model_kwargs(tmp_path, inp, kind="gguf", gguf=path, calibration=None)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == "kind gguf needs gguf, an existing .gguf file"


def test_no_results_report_and_an_empty_data_summary_are_refused(tmp_path):
    inp = make_inputs(tmp_path)
    no_reports = _model_kwargs(tmp_path, inp, results=[])
    with pytest.raises(BundleError, match="pass at least one results report"):
        build_model_bundle(**no_reports)
    no_summary = _model_kwargs(tmp_path, inp, data_summary="  \n")
    with pytest.raises(BundleError, match="data_summary must describe the training data"):
        build_model_bundle(**no_summary)
    assert not (tmp_path / "out").exists()


def test_an_apache_bundle_refuses_a_non_apache_teacher(tmp_path):
    inp = make_inputs(tmp_path)
    teachers = _teachers({"GENERATOR": {"Gen A": "Apache-2.0", "Gen M": "MIT"}})
    kwargs = _model_kwargs(tmp_path, inp, teachers=teachers)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == (
        "teacher 'Gen M' (MIT) is not Apache-2.0; an Apache bundle names only Apache-2.0"
        " teachers"
    )
    assert not (tmp_path / "out").exists()


def test_an_apache_bundle_with_apache_teachers_lists_them_on_the_card(tmp_path):
    inp = make_inputs(tmp_path)
    teachers = _teachers({"GENERATOR": {"Gen A": "Apache-2.0"}})
    build_model_bundle(**_model_kwargs(tmp_path, inp, teachers=teachers))
    card = (tmp_path / "out" / "README.md").read_text()
    assert "| Gen A | Apache-2.0 | wrote the variation |" in card


def test_a_non_apache_bundle_does_not_check_its_teachers_licences(tmp_path):
    inp = make_inputs(tmp_path)
    (inp.snapshot / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    teachers = _teachers({"GENERATOR": {"Gen M": "Proprietary"}})
    config = make_config(licence="MIT")
    build_model_bundle(**_model_kwargs(tmp_path, inp, teachers=teachers, config=config))
    card = (tmp_path / "out" / "README.md").read_text()
    assert "| Gen M | Proprietary | wrote the variation |" in card


def test_an_awq_export_with_another_chat_template_is_refused(tmp_path):
    inp = make_inputs(tmp_path)
    awq = tmp_path / "q" / "awq"
    shutil.copytree(inp.merged, awq)
    (awq / "chat_template.jinja").write_text("other", encoding="utf-8")
    kwargs = _model_kwargs(tmp_path, inp, kind="awq", awq_dir=awq)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == "the AWQ export's chat template differs from the base model's"


def test_a_base_snapshot_without_a_licence_file_is_refused(tmp_path):
    inp = make_inputs(tmp_path)
    (inp.snapshot / "LICENSE").unlink()
    kwargs = _model_kwargs(tmp_path, inp)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == f"{inp.snapshot} ships no LICENSE file"


def test_an_existing_empty_out_folder_is_replaced(tmp_path):
    inp = make_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    revision = build_model_bundle(**_model_kwargs(tmp_path, inp))
    assert len(revision) == 40
    assert (tmp_path / "out" / "README.md").is_file()


def test_an_existing_non_empty_out_folder_is_refused_and_kept(tmp_path):
    inp = make_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    kwargs = _model_kwargs(tmp_path, inp)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == f"{tmp_path / 'out'} is not empty"
    assert (tmp_path / "out" / "keep.txt").is_file()


def test_several_reports_under_a_heading_are_quoted_in_order(tmp_path):
    inp = make_inputs(tmp_path)
    second = tmp_path / "second.md"
    second.write_text("# Second\n\n## Metrics\n\n| m | v |\n|---|---|\n| f1 | 0.9 |\n")
    first = tmp_path / "first.md"
    first.write_text("# First\n\n| x | y |\n\n## Metrics\n\n| m | v |\n|---|---|\n| ece | 1 |\n")
    kwargs = _model_kwargs(tmp_path, inp, results=[first, second], results_heading="## Metrics")
    build_model_bundle(**kwargs)
    card = (tmp_path / "out" / "README.md").read_text()
    assert card.index("### First") < card.index("### Second")
    assert "| ece | 1 |" in card
    assert "| x | y |" not in card


@pytest.mark.parametrize(
    ("kind", "card", "missing"),
    [
        ("bf16", "nothing", ["license: apache-2.0", "base_model:", "derivative work"]),
        (
            "gguf",
            "license: apache-2.0 base_model: derivative work",
            ["llama-server", "--jinja"],
        ),
    ],
)
def test_a_card_missing_a_required_phrase_is_refused(tmp_path, monkeypatch, kind, card, missing):
    inp = make_inputs(tmp_path)
    monkeypatch.setattr(bundle_module, "model_card", lambda **_: card)
    extra = {"gguf": write_gguf(tmp_path / "m.gguf")} if kind == "gguf" else {}
    kwargs = _model_kwargs(tmp_path, inp, kind=kind, **extra)
    with pytest.raises(BundleError) as caught:
        build_model_bundle(**kwargs)
    assert str(caught.value) == f"model card is missing {missing}"
    assert not (tmp_path / "out").exists()


def test_the_bundle_files_and_notice_of_a_bf16_build(tmp_path):
    inp = make_inputs(tmp_path)
    revision = build_model_bundle(**_model_kwargs(tmp_path, inp))
    out = tmp_path / "out"
    assert revision == bundle_module.revision_of(inp.merged)
    assert sorted(p.name for p in out.iterdir()) == [
        "LICENSE",
        "NOTICE",
        "README.md",
        "bundle.json",
        "calibration.json",
        "chat_template.jinja",
        "config.json",
        "gate.json",
        "model.safetensors",
        "scorer-train.json",
        "tokenizer.json",
    ]
    assert (out / "scorer-train.json").read_bytes() == inp.scorer_train.read_bytes()
    assert (out / "calibration.json").read_bytes() == inp.calibration.read_bytes()
    notice = (out / "NOTICE").read_text()
    assert notice.startswith("example-org/toy-lamps-jev-scorer\n\n")
    assert "org/base (revision rev1)" in notice
    assert "mtp_num_hidden_layers was changed from 1 to 0" in notice


# -- dataset_bundle._resolve_roles / _answer_counts ------------------------------------

_ROLE_MODELS = {"g": ("Gen", "Apache-2.0"), "c": ("Cor", "MIT")}
_ALL_ROLES = {"GENERATOR": "g", "CORRECTOR": "c", "REVIEWER_A": "g", "REVIEWER_B": "c"}


def test_resolve_roles_from_an_nvsh_models_record():
    resolved = dataset_bundle._resolve_roles("e~v1", {"models": _ALL_ROLES}, _ROLE_MODELS)
    assert resolved == {
        "GENERATOR": ("Gen", "Apache-2.0"),
        "CORRECTOR": ("Cor", "MIT"),
        "REVIEWER_A": ("Gen", "Apache-2.0"),
        "REVIEWER_B": ("Cor", "MIT"),
    }


def test_resolve_roles_from_a_models_record_missing_a_role_or_alias():
    partial = {k: v for k, v in _ALL_ROLES.items() if k != "REVIEWER_A"}
    with pytest.raises(ValueError) as caught:
        dataset_bundle._resolve_roles("e~v1", {"models": partial}, _ROLE_MODELS)
    assert str(caught.value) == "variation e~v1's accepted record names no REVIEWER_A teacher"
    unknown = {**_ALL_ROLES, "CORRECTOR": "zz"}
    with pytest.raises(ValueError) as caught:
        dataset_bundle._resolve_roles("e~v1", {"models": unknown}, _ROLE_MODELS)
    assert str(caught.value) == "'zz' is not in the run's teacher-models table"


def _teacher(model="M", licence="Apache-2.0", key="model_name"):
    return {key: model, "licence": licence}


def test_resolve_roles_from_a_jev_teachers_record():
    teachers = {
        "generator": _teacher("G", key="model"),
        "reviewer_b": _teacher("B", "MIT"),
        "reviewer_a": _teacher("A"),
    }
    resolved = dataset_bundle._resolve_roles("e", {"teachers": teachers}, {})
    assert resolved == {
        "GENERATOR": ("G", "Apache-2.0"),
        "CORRECTOR": ("B", "MIT"),
        "REVIEWER_B": ("B", "MIT"),
        "REVIEWER_A": ("A", "Apache-2.0"),
    }
    assert list(resolved) == ["GENERATOR", "CORRECTOR", "REVIEWER_B", "REVIEWER_A"]


def test_resolve_roles_prefers_model_name_and_skips_an_absent_reviewer_a():
    teachers = {
        "generator": {"model_name": "Named", "model": "id", "licence": "Apache-2.0"},
        "reviewer_b": _teacher("B"),
    }
    resolved = dataset_bundle._resolve_roles("e", {"teachers": teachers}, {})
    assert resolved == {
        "GENERATOR": ("Named", "Apache-2.0"),
        "CORRECTOR": ("B", "Apache-2.0"),
        "REVIEWER_B": ("B", "Apache-2.0"),
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({}, "variation e has no accepted record naming its models"),
        ({"models": {}}, "variation e has no accepted record naming its models"),
        ({"teachers": {}}, "variation e has no accepted record naming its models"),
        ({"teachers": ["x"]}, "variation e has no accepted record naming its models"),
        (
            {"teachers": {"reviewer_b": _teacher()}},
            "variation e's accepted record names no generator teacher",
        ),
        (
            {"teachers": {"generator": _teacher()}},
            "variation e's accepted record names no reviewer_b teacher",
        ),
        (
            {"teachers": {"generator": "a string", "reviewer_b": _teacher()}},
            "variation e: teacher 'generator' is not a record",
        ),
        (
            {"teachers": {"generator": _teacher(), "reviewer_b": {"licence": "MIT"}}},
            "variation e: teacher 'reviewer_b' needs a model and a licence",
        ),
        (
            {"teachers": {"generator": {"model": "G"}, "reviewer_b": _teacher()}},
            "variation e: teacher 'generator' needs a model and a licence",
        ),
        (
            {"teachers": {"generator": _teacher(), "reviewer_b": _teacher(), "reviewer_a": 3}},
            "variation e: teacher 'reviewer_a' is not a record",
        ),
    ],
)
def test_resolve_roles_refuses_an_unusable_record(row, message):
    with pytest.raises(ValueError) as caught:
        dataset_bundle._resolve_roles("e", row, {})
    assert str(caught.value) == message


def test_answer_counts_classifies_explain_then_escalate_then_propose():
    records = [
        {"expect": {"explain": True, "escalate": True}},
        {"expect": {"escalate": True}},
        {"expect": {"operation": "x"}},
        {"expect": {"explain": False, "escalate": 0}},
        {"expect": {"explain": 1}},
    ]
    assert dataset_bundle._answer_counts(records) == {"explain": 2, "escalate": 1, "propose": 2}
    assert dataset_bundle._answer_counts([]) == {}


# -- dataset_bundle.build ---------------------------------------------------------------


def _entry(eid, text, **extra):
    return {
        "id": eid,
        "text": text,
        "expect": {"operation": "lamp_status", "args": {}},
        "source": "seed",
        **extra,
    }


def _dataset_kwargs(tmp: Path, train=None, **overrides):
    splits = tmp / "splits"
    splits.mkdir(exist_ok=True)
    write_json(splits / "val.json", {"header": "split (seed=7)", "entries": [_entry("v1", "va")]})
    write_json(splits / "test.json", {"entries": [_entry("t1", "te")]})
    write_json(tmp / "train.json", {"entries": train or [_entry("a1", "turn on the lamp")]})
    (tmp / "accepted.jsonl").write_text("")
    (tmp / "rejected.jsonl").write_text("\n")
    (tmp / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    write_json(tmp / "scorer.json", {"entries": [{"text": "at 10.0.0.9"}, {"text": 5}, {}]})
    kwargs = dict(
        domain=DOMAIN,
        splits=splits,
        train_augmented=tmp / "train.json",
        accepted=tmp / "accepted.jsonl",
        rejected=tmp / "rejected.jsonl",
        licence=tmp / "LICENSE",
        role_models={},
        scorer_train=tmp / "scorer.json",
        out=tmp / "ds",
    )
    kwargs.update(overrides)
    return dataset_build_args(**kwargs)


def test_build_refuses_an_empty_licence(tmp_path):
    kwargs = _dataset_kwargs(tmp_path)
    (tmp_path / "LICENSE").write_text("  \n")
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == f"{tmp_path / 'LICENSE'} is empty"


def test_build_refuses_duplicate_ids_across_the_splits(tmp_path):
    kwargs = _dataset_kwargs(tmp_path, train=[_entry("v1", "something else")])
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == "duplicate record ids across the splits"


def test_build_refuses_a_non_train_side_entry_and_a_variation_off_the_train_side(tmp_path):
    kwargs = _dataset_kwargs(tmp_path, train=[_entry("a1", "x", side="validation")])
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == f"a1 in {tmp_path / 'train.json'} is not a train-side entry"
    write_json(tmp_path / "splits" / "test.json", {"entries": [_entry("t1~v1", "te")]})
    train_augmented = write_json(tmp_path / "t2.json", [_entry("a1", "x")])
    kwargs["sources"] = dataclasses.replace(kwargs["sources"], train_augmented=train_augmented)
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == "t1~v1: variations never leave the train side"


def test_build_needs_a_source_unless_a_default_is_given(tmp_path):
    bare = {k: v for k, v in _entry("a1", "x").items() if k != "source"}
    kwargs = _dataset_kwargs(tmp_path, train=[bare])
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == "a1: every published record needs a 'source' field"
    counts = dataset_bundle.build(**kwargs, default_source="supplement-x")
    assert counts["supplement"] == 1
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest[0]["source"] == "supplement-x"
    assert manifest[0]["source_file"] == "train-supplement"


def test_build_into_an_existing_folder(tmp_path):
    kwargs = _dataset_kwargs(tmp_path)
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "keep.txt").write_text("x")
    with pytest.raises(ValueError) as caught:
        dataset_bundle.build(**kwargs)
    assert str(caught.value) == f"{tmp_path / 'ds'} is not empty"
    (tmp_path / "ds" / "keep.txt").unlink()
    counts = dataset_bundle.build(**kwargs)
    assert counts["train"] == 1


def test_build_writes_every_file_and_the_counts(tmp_path):
    train = [
        _entry("a1", "turn on the lamp at 10.0.0.5"),
        _entry("a2", "draft one", source="draft-x", expect={"explain": True}),
        _entry("a3", "targeted one", source="tgt-recipe", expect={"escalate": True}),
        _entry("a4", "t15 one", source="t15-old"),
        _entry("s1", "drafted by a teacher", source="supplement"),
    ]
    supplement = {
        "s1": {
            "teachers": {"generator": _teacher("G"), "reviewer_b": _teacher("B")},
            "decided_by": "reviewer_b",
        }
    }
    kwargs = _dataset_kwargs(
        tmp_path,
        train=train,
        rejected=[tmp_path / "rejected.jsonl", write_json(tmp_path / "r2.jsonl", {"id": "r"})],
        supplement_teachers=supplement,
        source_files={"draft": "drafts.json"},
        model_repos=["acme/m"],
        issue_refs="issue #9",
        apache_only=True,
    )
    counts = dataset_bundle.build(**kwargs)
    assert counts == {
        "train": 5,
        "validation": 1,
        "test": 1,
        "corpus": 1,
        "supplement": 3,
        "variation": 0,
        "draft": 1,
        "redacted_hosts": 1,
        "accepted": 0,
        "rejected": 1,
        "answers": {"propose": 3, "explain": 1, "escalate": 1},
    }
    out = tmp_path / "ds"
    assert sorted(p.name for p in out.iterdir()) == [
        "LICENSE",
        "README.md",
        "data",
        "manifest.json",
        "scorer-train.json",
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert [m["source_file"] for m in manifest] == [
        "seed.json",
        "drafts.json",
        "train-supplement",
        "train-supplement",
        "train-supplement",
        "seed.json",
        "seed.json",
    ]
    assert manifest[4]["teachers"] == {"GENERATOR": "G", "CORRECTOR": "B", "REVIEWER_B": "B"}
    assert [m["split"] for m in manifest] == ["train"] * 5 + ["validation", "test"]
    scorer = json.loads((out / "scorer-train.json").read_text())
    assert scorer == {"entries": [{"text": "at 192.0.2.9"}, {"text": 5}, {}]}
    train_rows = (out / "data" / "train.jsonl").read_text().splitlines()
    assert json.loads(train_rows[0])["text"] == "turn on the lamp at 192.0.2.5"
    card = (out / "README.md").read_text()
    assert "license: apache-2.0" in card
    assert "seed 7" in card
    assert "`acme/m`" in card
    assert "(issue #9)" in card


# -- scan.private_hosts / _scan_json_strings ---------------------------------------------


def test_private_hosts_keeps_only_private_addresses_and_private_names():
    text = (
        "a 999.1.1.1 b 127.0.0.1 c 0.0.0.0 d 192.0.2.4 e 8.8.8.8\n"
        "f 10.1.2.3 g 100.64.1.1 h nas.local i box.LAN 192.168.0.1\n"
        "\n"
        "x.internal 203.0.113.9 198.51.100.1 172.16.0.1 1.2.3.4.5 a.home"
    )
    assert scan.private_hosts(text) == [
        (2, "10.1.2.3"),
        (2, "100.64.1.1"),
        (2, "192.168.0.1"),
        (2, "nas.local"),
        (2, "box.LAN"),
        (4, "172.16.0.1"),
        (4, "x.internal"),
        (4, "a.home"),
    ]
    assert scan.private_hosts("") == []


def test_scan_json_strings_of_json_jsonl_and_other_suffixes():
    secretless = json.dumps({"a": ["10.0.0.1", {"b": "ok"}], "n": 3})
    assert scan._scan_json_strings("f.json", ".json", secretless) == [
        {"path": "f.json", "line": 1, "kind": "private_host", "detail": "10.0.0.1"}
    ]
    assert scan._scan_json_strings("f.json", ".json", "{not json") == []
    jsonl = '{"a": "x"}\n\n   \nnot json\n["10.0.0.2"]\n"192.168.1.1 and\\n10.0.0.3"\n'
    assert scan._scan_json_strings("f.jsonl", ".jsonl", jsonl) == [
        {"path": "f.jsonl", "line": 5, "kind": "private_host", "detail": "10.0.0.2"},
        {"path": "f.jsonl", "line": 6, "kind": "private_host", "detail": "192.168.1.1"},
        {"path": "f.jsonl", "line": 7, "kind": "private_host", "detail": "10.0.0.3"},
    ]
    assert scan._scan_json_strings("f.txt", ".txt", secretless) == []


# -- review core: verb_tree / make_record / plan_changes ------------------------------------


def test_verb_tree_when_an_operation_comes_after_its_dotted_child():
    ops = (
        Operation(name="grp.sub.leaf", description="Leaf op.", read_only=False),
        Operation(name="grp", description="Group op.", read_only=True),
        Operation(name="grp.other", description="Other op.", read_only=True),
    )
    domain = dataclasses.replace(DOMAIN, operations=ops, paraphrases=())
    nodes = core.verb_tree(domain)
    by_id = {n["id"]: n for n in nodes}
    assert [n["id"] for n in nodes][:5] == [
        "toy-lamps",
        "grp",
        "grp.sub",
        "grp.sub.leaf",
        "grp.other",
    ]
    assert by_id["grp"] == {
        "id": "grp",
        "label": "grp",
        "parent": "toy-lamps",
        "kind": "operation",
        "read_only": True,
        "description": "Group op.",
    }
    assert by_id["grp.sub"]["kind"] == "group"
    assert by_id["grp.sub"]["read_only"] is None
    assert by_id["grp.sub.leaf"]["parent"] == "grp.sub"
    assert by_id["grp.sub.leaf"]["read_only"] is False
    assert by_id["grp.other"]["parent"] == "grp"
    assert len(nodes) == len(by_id)


@pytest.fixture
def toy(tmp_path):
    seed = tmp_path / "seed.json"
    shutil.copyfile(DOMAIN.seed_corpus, seed)
    return dataclasses.replace(DOMAIN, seed_corpus=seed)


def _raw(domain) -> dict:
    return json.loads(Path(domain.seed_corpus).read_text(encoding="utf-8"))


def test_make_record_checks_note_and_entry_id_types(toy):
    raw = _raw(toy)
    cases = [
        ({"action": "approve", "entry_id": 5}, "a decision needs an entry_id"),
        ({"action": "approve", "entry_id": "  "}, "a decision needs an entry_id"),
        ({"action": "approve", "entry_id": "toy-01", "note": 3}, "note must be a string"),
        ({"action": None, "entry_id": "toy-01"}, "action must be one of"),
    ]
    for decision, message in cases:
        with pytest.raises(core.ReviewError, match=message):
            core.make_record(raw, toy, decision, at=AT)


def test_make_record_of_an_approval_and_a_withdrawn_proposal(toy):
    raw = _raw(toy)
    record = core.make_record(
        raw, toy, {"action": "approve", "entry_id": "toy-01", "note": " ok "}, at=AT
    )
    entry = next(e for e in raw["entries"] if e["id"] == "toy-01")
    assert record == {
        "at": AT,
        "action": "approve",
        "entry_id": "toy-01",
        "before": entry,
        "after": None,
        "note": "ok",
        "seed_sha256": core.seed_sha256(raw),
    }
    proposals = {"toy-90": {"id": "toy-90"}}
    withdrawn = core.make_record(
        raw, toy, {"action": "reject", "entry_id": "toy-90"}, proposals=proposals, at=AT
    )
    assert withdrawn["before"] is None
    assert withdrawn["after"] is None
    assert core.make_record(raw, toy, {"action": "reject", "entry_id": "toy-01"})["at"]


def _record(action, entry_id, before=None, after=None):
    return {"action": action, "entry_id": entry_id, "before": before, "after": after}


def test_plan_changes_covers_every_decision_outcome(toy):
    raw = _raw(toy)
    first, second, third, fourth, fifth = (dict(e) for e in raw["entries"][:5])
    changed = {**second, "text": "changed text"}
    new = {**first, "id": "toy-90", "text": "a new request"}
    records = [
        _record("approve", first["id"], before={**first, "text": "old"}),
        _record("approve", "toy-91"),
        _record("approve", third["id"]),
        _record("reject", "toy-92"),
        _record("reject", second["id"], before=changed),
        _record("edit", "toy-93", before=first, after=changed),
        _record("edit", fourth["id"], before={**fourth, "text": "old"}, after=changed),
        _record("propose", fifth["id"], after=new),
        _record("propose", "toy-90", after=new),
        _record("propose", "toy-94", after=None),
    ]
    plan = core.plan_changes(raw, records, toy)
    assert plan.conflicts == [
        f"{first['id']}: approved, but the seed entry changed since",
        "toy-91: approved, but it is no longer in the seed",
        f"{second['id']}: rejected, but the seed entry changed since",
        "toy-93: edited, but it is no longer in the seed",
        f"{fourth['id']}: edited, but the seed entry changed since",
        f"{fifth['id']}: proposed, but the id is already taken in the seed",
    ]
    assert plan.changes == [{"action": "add", "entry_id": "toy-90", "after": new}]
    assert [e["id"] for e in plan.seed["entries"]][-1] == "toy-90"


def test_plan_changes_applies_reject_edit_and_skips_what_is_already_applied(toy):
    raw = _raw(toy)
    first, second, third = (dict(e) for e in raw["entries"][:3])
    edited = {**third, "text": "a better request"}
    records = [
        _record("reject", first["id"], before=first),
        _record("reject", second["id"]),
        _record("edit", third["id"], before=third, after=edited),
        _record("edit", raw["entries"][3]["id"], before=None, after=raw["entries"][3]),
    ]
    plan = core.plan_changes(raw, records, toy)
    assert plan.conflicts == []
    assert plan.changes == [
        {"action": "remove", "entry_id": first["id"], "before": first},
        {"action": "remove", "entry_id": second["id"], "before": second},
        {"action": "edit", "entry_id": third["id"], "before": third, "after": edited},
    ]
    ids = [e["id"] for e in plan.seed["entries"]]
    assert first["id"] not in ids
    assert second["id"] not in ids
    assert plan.seed["entries"][0] == edited


# -- review server: the request handler ------------------------------------------------------


@pytest.fixture
def site(toy):
    app = server.ReviewApp(toy, Path(toy.seed_corpus), core.review_path(Path(toy.seed_corpus)))
    httpd = server.make_server(app, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield app, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _raw_call(port, method, path, *, body=b"", headers=None):
    conn = http.client.HTTPConnection(server.HOST, port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    res = conn.getresponse()
    payload = res.read()
    conn.close()
    return res.status, res.getheader("Content-Type", ""), payload, res


def test_the_handler_class_and_its_quiet_log():
    handler = server.make_handler(object())
    assert handler.server_version == "jev-review"
    assert handler.log_message(None, "%s", "x") is None


def test_an_unknown_get_path_is_404_and_index_html_is_the_page(site):
    app, port = site
    host = {"Host": f"{server.HOST}:{port}"}
    status, ctype, body, res = _raw_call(port, "GET", "/nowhere", headers=host)
    assert status == 404
    assert json.loads(body) == {"error": "not found"}
    assert ctype == "application/json; charset=utf-8"
    assert res.getheader("Content-Security-Policy") is None
    assert res.getheader("Cache-Control") == "no-store"
    status, ctype, body, res = _raw_call(port, "GET", "/index.html", headers=host)
    assert status == 200
    assert body == app.page
    status, _, _, _ = _raw_call(port, "GET", "/", headers={"Host": f"localhost:{port}"})
    assert status == 200


@pytest.mark.parametrize(
    ("body", "length", "error"),
    [
        (b"", "0", "a JSON body up to 1 MiB is needed"),
        (b"", None, "a JSON body up to 1 MiB is needed"),
        (b"{}", str(server.MAX_BODY + 1), "a JSON body up to 1 MiB is needed"),
        (b"{not json", None, "the body is not JSON"),
        (b"\xff\xfe", None, "the body is not JSON"),
        (b'"text"', None, "the body must be a JSON object"),
    ],
)
def test_a_bad_post_body_is_a_400(site, body, length, error):
    app, port = site
    headers = {"Host": f"{server.HOST}:{port}", server.AUTH_HEADER: app.token}
    if length is not None:
        headers["Content-Length"] = length
    elif body:
        headers["Content-Length"] = str(len(body))
    status, _, payload, _ = _raw_call(port, "POST", "/api/check", body=body, headers=headers)
    assert status == 400
    assert json.loads(payload) == {"error": error}
    assert not app.review_file.exists()


def test_a_post_with_a_wrong_host_is_403_and_an_unknown_path_is_404_first(site):
    app, port = site
    headers = {"Host": "evil.example:1", server.AUTH_HEADER: app.token}
    status, _, payload, _ = _raw_call(port, "POST", "/api/check", body=b"{}", headers=headers)
    assert status == 403
    assert json.loads(payload) == {"error": "unexpected Host header"}
    status, _, payload, _ = _raw_call(port, "POST", "/nope", body=b"{}", headers=headers)
    assert status == 404
