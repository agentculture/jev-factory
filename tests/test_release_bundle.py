"""release/bundle.py: model and dataset bundles, the required-file, symlink and GGUF checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from jev_factory.domains.jev_cli.generate import cli_surface_sha256, generate_domain
from jev_factory.release import bundle as bundle_module
from jev_factory.release import scan
from jev_factory.release.bundle import (
    BundleError,
    build_dataset_bundle,
    build_model_bundle,
    check_bundle,
    gguf_absolute_paths,
    gguf_metadata,
    surface_mismatch,
)
from jev_factory.release.hub import upload
from tests.fixtures.toy_domain import DOMAIN
from tests.release_support import (
    TOKEN,
    FakeHub,
    make_config,
    make_inputs,
    write_gguf,
    write_json,
)


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
    return build_model_bundle(**kwargs)


# ---- acceptance 1 / o18: required files, symlinks, GGUF metadata ------------------


@pytest.mark.behavioral("o18")
@pytest.mark.parametrize("missing", ["calibration", "gate", "scorer_train"])
def test_a_bundle_without_calibration_gate_or_scorer_train_is_refused(tmp_path, missing):
    inp = make_inputs(tmp_path)
    with pytest.raises(BundleError, match="must carry"):
        _build(tmp_path, inp, **{missing: None})
    with pytest.raises(BundleError, match="must carry"):
        _build(tmp_path, inp, **{missing: tmp_path / "does-not-exist.json"})
    assert not (tmp_path / "out").exists()


@pytest.mark.behavioral("o18")
def test_a_finished_bundle_missing_a_required_file_fails_the_check(tmp_path):
    inp = make_inputs(tmp_path)
    _build(tmp_path, inp)
    check_bundle(tmp_path / "out")  # the good bundle passes
    for name in ("calibration.json", "gate.json", "scorer-train.json"):
        broken = tmp_path / f"no-{name}"
        shutil.copytree(tmp_path / "out", broken)
        (broken / name).unlink()
        with pytest.raises(BundleError, match=f"missing {name}"):
            check_bundle(broken)


@pytest.mark.behavioral("o18")
def test_a_symlink_in_a_bundle_refuses_it_at_build_and_at_check(tmp_path):
    inp = make_inputs(tmp_path)
    outside = tmp_path / "sealed.json"
    outside.write_text("{}")
    (inp.merged / "leak.json").symlink_to(outside)
    with pytest.raises(BundleError, match="symlink leak.json"):
        _build(tmp_path, inp)
    assert not (tmp_path / "out").exists()  # the refused bundle is removed, not left behind

    inp2 = make_inputs(tmp_path / "ok")
    _build(tmp_path / "ok", inp2)
    out = tmp_path / "ok" / "out"
    (out / "later.json").symlink_to(outside)
    with pytest.raises(BundleError, match="symlink later.json"):
        check_bundle(out)


@pytest.mark.behavioral("o18")
@pytest.mark.parametrize(
    "value", ["/ho" + "me/someone/work/imatrix.dat", "C:\\work\\imatrix.dat", "~/imatrix.dat"]
)
def test_an_absolute_path_in_gguf_imatrix_metadata_refuses_the_bundle(tmp_path, value):
    inp = make_inputs(tmp_path)
    gguf = write_gguf(tmp_path / "model-q4_k_m.gguf", {"quantize.imatrix.file": value})
    assert gguf_absolute_paths(gguf) == ["quantize.imatrix.file"]
    with pytest.raises(BundleError, match="quantize.imatrix.file holds an absolute path"):
        _build(tmp_path, inp, kind="gguf", gguf=gguf)
    assert not (tmp_path / "out").exists()


@pytest.mark.behavioral("o18")
def test_a_symlinked_gguf_is_refused(tmp_path):
    inp = make_inputs(tmp_path)
    real = write_gguf(tmp_path / "real.gguf")
    link = tmp_path / "link.gguf"
    link.symlink_to(real)
    with pytest.raises(BundleError, match="symlink link.gguf"):
        _build(tmp_path, inp, kind="gguf", gguf=link)


@pytest.mark.behavioral("o18")
def test_a_relative_imatrix_path_and_a_gguf_bundle_pass(tmp_path):
    inp = make_inputs(tmp_path)
    gguf = write_gguf(tmp_path / "model-q4_k_m.gguf", {"tokenizer.chat_template": "/not/a/path"})
    _build(tmp_path, inp, kind="gguf", gguf=gguf, quantized_from="example-org/x")
    out = tmp_path / "out"
    assert (out / "model-q4_k_m.gguf").is_file()
    card = (out / "README.md").read_text()
    assert "llama-server" in card
    assert "--jinja" in card
    assert "quantized from" in card
    assert not (out / "model.safetensors").exists()  # a gguf bundle ships the gguf only


@pytest.mark.behavioral("o18")
def test_an_unreadable_gguf_refuses_the_bundle(tmp_path):
    inp = make_inputs(tmp_path)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF\x03\x00\x00\x00")  # header cut off
    with pytest.raises(BundleError, match="truncated GGUF header"):
        _build(tmp_path, inp, kind="gguf", gguf=bad)


def test_gguf_metadata_reads_strings_numbers_and_skips_arrays(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", {"n": 7, "tokenizer.ggml.tokens": ["a", "b"]})
    meta = gguf_metadata(path)
    assert meta["general.architecture"] == "qwen3"
    assert meta["n"] == 7
    assert meta["tokenizer.ggml.tokens"] is None


# ---- build output, config-derived card, surface hash ------------------------------


def test_the_model_bundle_carries_everything_and_the_card_comes_from_domain_and_config(tmp_path):
    inp = make_inputs(tmp_path)
    revision = _build(tmp_path, inp, config=make_config(licence="Apache-2.0"))
    out = tmp_path / "out"
    names = {p.name for p in out.iterdir()}
    assert {
        "README.md",
        "NOTICE",
        "LICENSE",
        "calibration.json",
        "gate.json",
        "scorer-train.json",
        "bundle.json",
        "model.safetensors",
    } <= names
    assert (out / "gate.json").read_bytes() == inp.gate.read_bytes()
    card = (out / "README.md").read_text()
    assert DOMAIN.card_text in card
    assert "license: apache-2.0" in card
    assert "derivative work" in card
    assert "ece" in card  # the results table is quoted
    assert json.loads((out / "config.json").read_text())["mtp_num_hidden_layers"] == 0
    assert len(revision) == 40
    info = json.loads((out / "bundle.json").read_text())
    assert info["hub_prefix"] == "example-org/toy-lamps-jev-"
    assert info["kind"] == "model"


def test_licence_and_hub_prefix_follow_the_run_config(tmp_path):
    inp = make_inputs(tmp_path)
    (inp.snapshot / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    _build(tmp_path, inp, config=make_config(licence="MIT", hub_prefix="acme/fam-"))
    info = json.loads((tmp_path / "out" / "bundle.json").read_text())
    assert info["licence"] == "MIT"
    assert info["hub_prefix"] == "acme/fam-"
    assert "license: mit" in (tmp_path / "out" / "README.md").read_text()
    # an Apache claim over a base licence file that is not Apache is refused
    inp2 = make_inputs(tmp_path / "b")
    (inp2.snapshot / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    with pytest.raises(BundleError, match="Apache"):
        _build(tmp_path / "b", inp2, out=tmp_path / "b" / "out")


def test_a_chat_template_that_differs_from_the_base_is_refused(tmp_path):
    inp = make_inputs(tmp_path)
    (inp.merged / "chat_template.jinja").write_text("different", encoding="utf-8")
    with pytest.raises(BundleError, match="chat template differs"):
        _build(tmp_path, inp)


def test_awq_needs_its_quantization_record(tmp_path):
    inp = make_inputs(tmp_path)
    awq = tmp_path / "q" / "awq"
    shutil.copytree(inp.merged, awq)
    with pytest.raises(BundleError, match="quantize-run.json"):
        _build(tmp_path, inp, kind="awq", awq_dir=awq)
    write_json(tmp_path / "q" / "quantize-run.json", {"awq_serve_args": ["--dtype", "half"]})
    _build(tmp_path, inp, kind="awq", awq_dir=awq)
    assert "--dtype half" in (tmp_path / "out" / "README.md").read_text()


@pytest.mark.behavioral("o36")
def test_a_bundle_records_the_jev_cli_surface_hash_and_ask_reports_a_mismatch(tmp_path):
    cli_domain = generate_domain()
    inp = make_inputs(tmp_path)
    _build(tmp_path, inp, domain=cli_domain)
    out = tmp_path / "out"
    info = json.loads((out / "bundle.json").read_text())
    assert info["surface_sha256"] == cli_surface_sha256() == cli_domain.surface_sha256()
    assert info["domain"] == cli_domain.name
    assert info["surface_sha256"] in (out / "README.md").read_text()
    assert surface_mismatch(out) is None  # same CLI: no report
    message = surface_mismatch(out, current_sha256="0" * 64)  # the CLI's surface moved
    assert message
    assert info["surface_sha256"] in message
    assert "0" * 64 in message
    # a bundle trained on another domain's surface differs from the running CLI's
    other = tmp_path / "o2"
    inp2 = make_inputs(other)
    _build(other, inp2, out=other / "out")
    assert "no longer match" in surface_mismatch(other / "out")


def test_a_bundle_with_no_surface_hash_is_refused(tmp_path):
    inp = make_inputs(tmp_path)
    _build(tmp_path, inp)
    write_json(tmp_path / "out" / "bundle.json", {"kind": "model"})
    with pytest.raises(BundleError, match="no surface_sha256"):
        check_bundle(tmp_path / "out")


def test_a_built_bundle_scans_clean_and_uploads_through_the_fake_hub(tmp_path):
    inp = make_inputs(tmp_path)
    _build(tmp_path, inp)
    out = tmp_path / "out"
    assert scan.write_scan(out)["clean"] is True
    hub = FakeHub(tmp_path / "remote")
    result = upload(
        bundle=out,
        repo="example-org/toy-lamps-jev-scorer",
        repo_type="model",
        config=make_config(),
        domain=DOMAIN,
        apply=True,
        hub=hub,
        environ={"HF_TOKEN": TOKEN},
    )
    assert result["private"] is True
    assert result["applied"] is True


# ---- dataset bundle --------------------------------------------------------------


def _dataset_inputs(tmp: Path):
    splits = tmp / "splits"
    splits.mkdir()

    def entry(eid, text, **extra):
        return {
            "id": eid,
            "text": text,
            "expect": {"operation": "lamp_status", "args": {}},
            "source": "seed",
            **extra,
        }

    write_json(
        splits / "val.json", {"header": "split (seed=46)", "entries": [entry("v1", "val one")]}
    )
    write_json(splits / "test.json", {"entries": [entry("t1", "test one")]})
    train = [
        entry("a1", "turn on the lamp at 192.168.1.50"),
        entry("a1~v1", "switch the lamp on", source_id="a1"),
    ]
    write_json(tmp / "train.json", {"entries": train})
    models = {r: "alias" for r in ("GENERATOR", "CORRECTOR", "REVIEWER_A", "REVIEWER_B")}
    (tmp / "accepted.jsonl").write_text(json.dumps({"id": "a1~v1", "models": models}) + "\n")
    (tmp / "rejected.jsonl").write_text(json.dumps({"id": "x"}) + "\n")
    (tmp / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    return splits, models


def _build_dataset(tmp: Path, **overrides):
    splits, _ = _dataset_inputs(tmp)
    inp = make_inputs(tmp / "m")
    kwargs = dict(
        domain=DOMAIN,
        config=make_config(issue_refs="issue #1"),
        splits=splits,
        train_augmented=tmp / "train.json",
        accepted=tmp / "accepted.jsonl",
        rejected=tmp / "rejected.jsonl",
        licence_file=tmp / "LICENSE",
        role_models={"alias": ("Teacher Model", "Apache-2.0")},
        out=tmp / "ds",
        calibration=inp.calibration,
        gate=inp.gate,
        scorer_train=inp.scorer_train,
    )
    kwargs.update(overrides)
    return build_dataset_bundle(**kwargs)


def test_the_dataset_bundle_carries_the_required_files_and_the_surface_hash(tmp_path):
    counts = _build_dataset(tmp_path)
    out = tmp_path / "ds"
    assert counts["train"] == 2
    assert counts["variation"] == 1
    assert counts["redacted_hosts"] == 1
    for name in ("calibration.json", "gate.json", "scorer-train.json", "bundle.json", "LICENSE"):
        assert (out / name).is_file()
    info = json.loads((out / "bundle.json").read_text())
    assert info["surface_sha256"] == DOMAIN.surface_sha256()
    assert info["kind"] == "dataset"
    train = (out / "data" / "train.jsonl").read_text()
    assert "192.168.1.50" not in train  # private address redacted
    assert "192.0.2.50" in train
    card = (out / "README.md").read_text()
    assert DOMAIN.card_text in card
    assert "Teacher Model" in card
    assert "issue #1" in card
    assert scan.write_scan(out)["clean"] is True
    check_bundle(out)


@pytest.mark.behavioral("o18")
@pytest.mark.parametrize("missing", ["calibration", "gate", "scorer_train"])
def test_a_dataset_bundle_without_a_required_file_is_refused(tmp_path, missing):
    with pytest.raises(BundleError, match="must carry"):
        _build_dataset(tmp_path, **{missing: None})
    assert not (tmp_path / "ds").exists()


def test_the_dataset_bundle_refuses_held_out_leaks_and_non_apache_teachers(tmp_path):
    with pytest.raises(ValueError, match="Apache-2.0"):
        _build_dataset(tmp_path, role_models={"alias": ("Closed Model", "Proprietary")})
    held = tmp_path / "held-out.json"
    held.write_text("[]")
    with pytest.raises(ValueError, match="held-out"):
        bundle_module._dataset.load_entries(held)  # noqa: SLF001


def test_the_dataset_bundle_refuses_a_train_row_repeating_a_held_apart_entry(tmp_path):
    splits, _ = _dataset_inputs(tmp_path)
    write_json(
        tmp_path / "train.json",
        {
            "entries": [
                {"id": "a1", "text": "Val one!", "expect": {"escalate": True}, "source": "s"}
            ]
        },
    )
    config = make_config()
    calibration = write_json(tmp_path / "c.json", {})
    gate = write_json(tmp_path / "g.json", {})
    scorer_train = write_json(tmp_path / "s.json", {"entries": []})
    with pytest.raises(ValueError, match="repeat a validation or test entry"):
        build_dataset_bundle(
            domain=DOMAIN,
            config=config,
            splits=splits,
            train_augmented=tmp_path / "train.json",
            accepted=tmp_path / "accepted.jsonl",
            rejected=tmp_path / "rejected.jsonl",
            licence_file=tmp_path / "LICENSE",
            role_models={},
            out=tmp_path / "ds",
            calibration=calibration,
            gate=gate,
            scorer_train=scorer_train,
        )


# ---- l15: what nvsh's release tests pinned that the rewrite dropped -----------------


def test_the_notice_states_the_quantization_and_the_mtp_change(tmp_path):
    inp = make_inputs(tmp_path)
    gguf = write_gguf(tmp_path / "model-q4_k_m.gguf")
    _build(tmp_path, inp, kind="gguf", gguf=gguf)
    notice = (tmp_path / "out" / "NOTICE").read_text()
    assert "Q4_K_M GGUF quantization" in notice
    assert "llama.cpp" in notice
    shutil.rmtree(tmp_path / "out")
    _build(tmp_path, inp)  # bf16: the declared MTP head is zeroed in the bundle copy
    notice = (tmp_path / "out" / "NOTICE").read_text()
    assert "mtp_num_hidden_layers was changed from" in notice
    assert "quantization" not in notice


def test_the_card_names_the_logprob_window_and_a_text_only_gguf(tmp_path):
    from jev_factory.backbones.causal_lm.readout import READOUT_TOP

    inp = make_inputs(tmp_path)
    _build(tmp_path, inp)
    assert f"--max-logprobs {READOUT_TOP}" in (tmp_path / "out" / "README.md").read_text()
    shutil.rmtree(tmp_path / "out")
    _build(tmp_path, inp, kind="gguf", gguf=write_gguf(tmp_path / "model-q4_k_m.gguf"))
    card = (tmp_path / "out" / "README.md").read_text()
    assert "text-only" in card
    assert "no `--mmproj`" in card
    assert "--temp 0 --top-k 1" in card


def test_a_dataset_bundle_refuses_a_licence_file_that_is_not_apache(tmp_path):
    mit = _write(tmp_path / "MIT", "MIT License\n")
    with pytest.raises(BundleError, match="Apache License 2.0"):
        _build_dataset(tmp_path, licence_file=mit)
    assert not (tmp_path / "ds").exists()


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _jev_record(decided_by="reviewer_b", **teachers):
    def role(name, licence="Apache-2.0"):
        return {"role": name, "model": f"{name}-alias", "model_name": name, "licence": licence}

    record = {
        "generator": role("Gen Model"),
        "reviewer_b": role("Judge Model"),
        **{k: role(*v) if isinstance(v, tuple) else v for k, v in teachers.items()},
    }
    return {
        "teachers": {k: v for k, v in record.items() if v is not None},
        "decided_by": decided_by,
    }


def test_teacher_summary_reads_the_records_jev_augment_writes():
    ds = bundle_module._dataset  # noqa: SLF001
    rows = {"a~v1": _jev_record(), "b~v1": _jev_record("both", reviewer_a=("Second Judge",))}
    train = [{"id": "a"}, {"id": "a~v1"}, {"id": "b~v1"}]
    summary = ds.teacher_summary(train, rows, {}, apache_only=True)
    assert summary.role_teachers["GENERATOR"] == {"Gen Model": "Apache-2.0"}
    assert summary.role_teachers["CORRECTOR"] == {"Judge Model": "Apache-2.0"}
    assert summary.role_teachers["REVIEWER_A"] == {"Second Judge": "Apache-2.0"}
    assert summary.shared_corrector_reviewer_names == ["Judge Model"]
    assert summary.decisions == {"reviewer_b": 1, "both": 1}
    closed = {"a~v1": _jev_record(generator=("Closed Gen", "Proprietary"))}
    with pytest.raises(ValueError, match="apache_only"):
        ds.teacher_summary([{"id": "a~v1"}], closed, {}, apache_only=True)
    no_generator = {"a~v1": _jev_record(generator=None)}
    with pytest.raises(ValueError, match="no generator teacher"):
        ds.teacher_summary([{"id": "a~v1"}], no_generator, {})


def test_targeted_rows_are_published_as_supplement_with_their_teachers(tmp_path):
    def entry(eid, text, source, **extra):
        expect = {"operation": "lamp_status", "args": {}}
        return {"id": eid, "text": text, "expect": expect, "source": source, **extra}

    train = [
        entry("a1", "turn on the lamp at 192.168.1.50", "seed"),
        entry("a1~v1", "switch the lamp on", "seed", source_id="a1"),
        entry("tgt-missing-argument-0001", "dim it", "tgt-missing-argument"),
    ]
    plus = write_json(tmp_path / "train-plus.json", {"entries": train})
    drafted = {"tgt-missing-argument-0001": _jev_record()}
    _build_dataset(tmp_path, train_augmented=plus, supplement_teachers=drafted)
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    row = next(r for r in manifest if r["id"] == "tgt-missing-argument-0001")
    assert row["origin"] == "supplement"
    assert row["transformed"] is False
    assert row["teachers"]["GENERATOR"] == "Gen Model"
