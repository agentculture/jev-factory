"""t10: a bf16 ship (deployed_format bf16): quantize and heal skipped; gate, bundle and edge
take the attested bf16 build, the train-stage calibration and its predictions.

The bf16 deployed contract: the deployed student is ``distill/select/summary.json``'s
``winner``; its checkpoint is ``distill/train/<winner>/bundle`` (``hobson-hf-export/1``), its
calibration ``distill/select/calibration.json``, its selection-fold predictions
``distill/train/<winner>/predictions.json``; the attestation is the bundle folder's per-file
sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.cli._errors import EXIT_ENV_ERROR, CliError
from jev_factory.distill import bundle as B
from jev_factory.distill import deployed as D
from jev_factory.distill import edge
from jev_factory.distill import gate as G
from jev_factory.distill import prereg as dp
from jev_factory.distill import quant as Q
from jev_factory.distill.runspec import TrainerEnv
from jev_factory.release import hub
from tests import test_bundle_format_doc as docs
from tests import test_distill_bundle as tb
from tests import test_distill_edge as te
from tests import test_distill_gate as tg

WINNER = "c003"
SELECT_CAL = {
    "temperature_by_kind": {"choice": 1.0, "score": 1.0, "noul": 1.0},
    "vector_by_kind": {},
    "upstream_temperature_by_kind_used": False,
    "candidate": WINNER,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hobson(tmp_path: Path, out: Path) -> Path:
    src = tmp_path / "ckpt-src"
    (src / "lora").mkdir(parents=True)
    (src / player.ADAPTER_CONFIG).write_text('{"r": 16}')
    (src / player.ADAPTER).write_bytes(tb.ST + b"adapter")
    (src / player.HEAD).write_bytes(tb.ST + b"head")
    (src / player.CONFIG).write_text(json.dumps(tb.UPSTREAM_CONFIG))
    for name in player.TOKENIZER_FILES:
        (src / name).write_text("{}")
    player.write_layout(src, out, base_revision=tb.REV, extra={"distill_candidate": WINNER})
    return out


def _preds(fold_records):
    return {
        fold: [
            {"record_id": r.id, "error": None, "questions": tg._answer(int(r.id[1:]))} for r in recs
        ]
        for fold, recs in fold_records.items()
    }


def make_bf16_run(tmp_path: Path, deployed_format: str = "bf16"):
    """A locked v3 run at the end of train/select, with the winner's bundle exported."""
    run = tmp_path / "run"
    prereg = tmp_path / "prereg.json"
    doc = dp.example_doc_v3(deployed_format=deployed_format)
    doc["gate"]["min_coverage"] = {"choice": 0.5, "noul": 0.5, "score": 0.5}
    prereg.write_text(json.dumps(doc), encoding="utf-8")
    dp.lock(prereg, run)
    cand = run / "distill" / "train" / WINNER
    ckpt = cand / "checkpoint"
    ckpt.mkdir(parents=True)
    bundle = _hobson(tmp_path, cand / "bundle")
    (cand / "predictions.json").write_text(json.dumps(_preds(tg.folds())), encoding="utf-8")
    sel = run / "distill" / "select"
    sel.mkdir(parents=True)
    summary = {
        "winner": WINNER,
        "candidates": [{"cid": WINNER, "knobs": {"epochs": 1}, "checkpoint": str(ckpt)}],
    }
    (sel / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (sel / "calibration.json").write_text(json.dumps(SELECT_CAL), encoding="utf-8")
    return run, prereg, bundle


# --------------------------------------------------------------------- the helper


def test_the_helper_resolves_the_bf16_contract_paths(tmp_path):
    run, prereg, bundle = make_bf16_run(tmp_path)
    reg = dp.require_locked(prereg, run)
    student = D.deployed_student(run, reg)
    assert student["format"] == "bf16"
    assert student["deployed_format"] == "bf16"
    assert Path(student["artifact"]) == run / "distill/train" / WINNER / "bundle"
    assert student["winner"] == WINNER
    assert student["calibration_file"] == "distill/select/calibration.json"
    assert student["calibration_sha256"] == _sha(run / "distill/select/calibration.json")
    assert student["calibration"]["temperature_by_kind"] == SELECT_CAL["temperature_by_kind"]
    assert Path(student["predictions_file"]) == run / "distill/train" / WINNER / "predictions.json"
    assert student["bundle"] == D.attestation(bundle)
    assert student["bundle"]["files"] == hub.digests(bundle)
    # one attestation shape for final, gate, bundle and edge: the MANIFEST.sha256 file's hash
    assert student["bundle"]["manifest_sha256"] == _sha(bundle / player.MANIFEST)
    assert student["bundle"]["weight_bytes"] > 0


def test_final_and_the_stages_share_one_bf16_resolver_and_attestation(tmp_path, monkeypatch):
    """Note 6: final.py's bf16 resolver and attestation are deployed.py's, not a second copy."""
    from jev_factory.distill import final as F

    run, _, bundle = make_bf16_run(tmp_path)
    assert F.bf16_attestation(bundle) == D.attestation(bundle)
    calls = []
    real = D.bf16_student
    monkeypatch.setattr(D, "bf16_student", lambda r: calls.append(Path(r)) or real(r))
    got = F.bf16_deployed_student(run)
    assert calls == [run]
    assert Path(got["artifact"]) == bundle
    assert got["winner"] == WINNER
    assert Path(got["selection_predictions"]) == run / "distill/train" / WINNER / "predictions.json"


def test_the_helper_refuses_a_bundle_that_does_not_verify(tmp_path):
    run, prereg, bundle = make_bf16_run(tmp_path)
    (bundle / player.HEAD).write_bytes(tb.ST + b"tampered")
    with pytest.raises(CliError):
        D.deployed_student(run, dp.require_locked(prereg, run))


def test_the_helper_refuses_a_calibration_of_another_candidate(tmp_path):
    run, prereg, _ = make_bf16_run(tmp_path)
    cal = run / "distill/select/calibration.json"
    cal.write_text(json.dumps({**SELECT_CAL, "candidate": "c999"}), encoding="utf-8")
    with pytest.raises(CliError, match="c999"):
        D.deployed_student(run, dp.require_locked(prereg, run))


def test_the_helper_refuses_without_predictions(tmp_path):
    run, prereg, _ = make_bf16_run(tmp_path)
    (run / "distill/train" / WINNER / "predictions.json").unlink()
    with pytest.raises(CliError, match="predictions"):
        D.deployed_student(run, dp.require_locked(prereg, run))


# --------------------------------------------------------------------- quantize and heal


@pytest.mark.parametrize("apply", [False, True])
def test_quantize_and_heal_are_skipped_under_bf16_and_build_nothing(tmp_path, apply):
    run, prereg, _ = make_bf16_run(tmp_path)
    out = Q.run_quant(
        prereg_path=prereg,
        run_dir=run,
        folds=tg.folds(),
        trainer_env=TrainerEnv(python="python3", decider_repo=tmp_path / "sd"),
        apply=apply,
        hooks=Q.QuantHooks(builders={}, measurer=None),
    )
    assert out["status"] == "skipped"
    assert out["reason"] == Q.BF16_SKIP_REASON == "deployed_format bf16"
    assert out["heal"] == "skipped"
    assert not (run / Q.QUANT_REL).exists()


def test_skip_reason_names_bf16_and_is_none_for_a_quant_ship(tmp_path):
    assert Q.skip_reason(dp.validate(dp.example_doc_v3(deployed_format="bf16"))) == (
        "deployed_format bf16"
    )
    assert Q.skip_reason(dp.validate(dp.example_doc_v3())) is None
    assert Q.skip_reason(dp.validate(dp.example_doc())) is None


# --------------------------------------------------------------------- gate


def test_gate_fits_on_the_train_predictions_and_select_calibration_under_bf16(tmp_path):
    run, prereg, bundle = make_bf16_run(tmp_path)
    measurer = tg.FakeMeasurer()
    out = G.run_gate(
        run,
        prereg_path=prereg,
        folds=tg.folds(),
        measurer=measurer,
        apply=True,
        resamples=tg.RESAMPLES,
    )
    assert measurer.calls == []  # the persisted predictions of the attested build are used
    assert out["deployed_format"] == "bf16"
    assert out["deployed_artifact"] == str(bundle)
    assert out["calibration_file"] == "distill/select/calibration.json"
    assert out["calibration_sha256"] == _sha(run / "distill/select/calibration.json")
    # the same answers as the quant-path test: the same fit
    assert out["by_kind"]["choice"]["fit"]["selective_accuracy"] == 1.0
    assert out["by_kind"]["choice"]["fit"]["coverage"] == 0.65
    assert json.loads((run / G.GATE_REL / G.GATE_FILE).read_text()) == out


def test_gate_under_bf16_needs_no_measurer(tmp_path):
    run, prereg, _ = make_bf16_run(tmp_path)
    out = G.run_gate(run, prereg_path=prereg, folds=tg.folds(), apply=True, resamples=50)
    assert out["fitted"] is True


def test_gate_dry_run_under_bf16_writes_nothing(tmp_path):
    run, prereg, _ = make_bf16_run(tmp_path)
    before = sorted(run.rglob("*"))
    out = G.run_gate(run, prereg_path=prereg, folds=tg.folds())
    assert out["dry_run"] is True and out["deployed_format"] == "bf16"
    assert sorted(run.rglob("*")) == before


# --------------------------------------------------------------------- bundle


def _bundle_inputs(run: Path, tmp_path: Path) -> Path:
    rows = [tb._row("t1", "q", "mit"), tb._row("t2", "q", "Apache-2.0", "human_gold")]
    a = run / "distill" / "assemble"
    a.mkdir(parents=True)
    tb._jsonl(a / "train_set.jsonl", rows)
    (a / "assemble_manifest.json").write_text(
        json.dumps(
            {
                "teacher_build_ids": ["build1"],
                "audited_builds": {"clef": "build1"},
                "output": {"sha256": _sha(a / "train_set.jsonl"), "rows": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    f = run / "distill" / "final"
    f.mkdir(parents=True)
    (f / "report.json").write_text(json.dumps({"verdict": "ok"}), encoding="utf-8")
    ds = tmp_path / "dataset"
    ds.mkdir()
    tb._jsonl(ds / "held_out.jsonl", [{"id": "h1", "source": "s", "state": tb.SEALED_TEXT}])
    tb._jsonl(ds / "test.jsonl", [{"id": "x1", "source": "s", "state": tb.SEALED_TEXT}])
    return ds


def build_bf16_bundle(tmp_path: Path, *, gate: bool = True):
    run, prereg, bundle = make_bf16_run(tmp_path)
    if gate:
        G.run_gate(run, prereg_path=prereg, folds=tg.folds(), apply=True, resamples=50)
    ds = _bundle_inputs(run, tmp_path)
    out = tmp_path / "out"
    summary = B.build_bundles(run, out, prereg_path=prereg, dataset_dir=ds)
    return run, out, summary, bundle


def test_bundle_ships_the_bf16_build_with_the_select_calibration(tmp_path):
    run, out, summary, _ = build_bf16_bundle(tmp_path)
    model = out / B.MODEL_DIR
    info = player.read_layout(model)  # strands-decider serve's layout, verified
    assert info["format"] == "hobson-hf-export/1"
    assert summary["deployed_format"] == "bf16"
    assert summary["gate_fitted"] is True
    assert (model / B.CALIBRATION_FILE).read_bytes() == (
        run / "distill/select/calibration.json"
    ).read_bytes()
    config = json.loads((model / player.CONFIG).read_text())
    assert config["temperature_by_kind"] == SELECT_CAL["temperature_by_kind"]
    assert config["temperature"] == 1.0
    prov = json.loads((model / B.PROVENANCE_FILE).read_text())
    assert prov["bundle_kind"] == "bf16"
    assert prov["deployed_format"] == "bf16"
    assert prov["deployed_student"]["winner"] == WINNER
    assert prov["deployed_student"]["checkpoint"] == f"distill/train/{WINNER}/bundle"
    assert prov["deployed_student"]["calibration"] == "distill/select/calibration.json"
    assert prov["calibration_sha256"] == _sha(run / "distill/select/calibration.json")
    assert prov["held_out_published"] is False
    ident = json.loads((model / B.BUNDLE_FILE).read_text())
    assert ident["deployed_format"] == "bf16"


def test_bundle_refuses_a_gate_fitted_on_another_calibration_under_bf16(tmp_path):
    run, prereg, _ = make_bf16_run(tmp_path)
    G.run_gate(run, prereg_path=prereg, folds=tg.folds(), apply=True, resamples=50)
    cal = run / "distill/select/calibration.json"
    cal.write_text(json.dumps({**SELECT_CAL, "temperature_by_kind": {"choice": 2.0}}))
    ds = _bundle_inputs(run, tmp_path)
    with pytest.raises(CliError, match="fitted on calibration"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg, dataset_dir=ds)


def test_every_file_of_a_bf16_kind_bundle_is_in_the_doc(tmp_path):
    _, out, _, _ = build_bf16_bundle(tmp_path)
    text = docs._doc()
    for folder in (B.MODEL_DIR, B.DATASET_DIR):
        names = {p.relative_to(out / folder).as_posix() for p in (out / folder).rglob("*")}
        names = {n for n in names if (out / folder / n).is_file()}
        assert docs.undocumented(names, text) == [], folder
    spans = docs._code_spans(text)
    for field in ("bundle_kind", "deployed_student", "distill/select/calibration.json"):
        assert field in spans, field


def test_a_quant_ship_with_no_qualifying_build_still_refuses_with_a_bf16_build_on_disk(tmp_path):
    """No bf16 fallback: a quant ship whose quantize failed refuses even though the winner's
    bf16 bundle, calibration and predictions are all on disk."""
    run, prereg, _ = make_bf16_run(tmp_path, deployed_format="quant")
    q = run / Q.QUANT_REL
    q.mkdir(parents=True)
    (q / Q.DECISION_FILE).write_text(json.dumps({"format": None}))
    (q / Q.MANIFEST_FILE).write_text(
        json.dumps(
            {
                "status": "failed",
                "outcome": Q.NO_QUALIFYING,
                "chosen_format": None,
                "shipped_format": None,
                "shippable": False,
                "deployed_artifact": None,
            }
        )
    )
    ds = _bundle_inputs(run, tmp_path)
    with pytest.raises(CliError, match="no shipped format"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg, dataset_dir=ds)
    with pytest.raises(CliError, match="no shipped format"):
        G.run_gate(run, prereg_path=prereg, folds=tg.folds())
    with pytest.raises(CliError, match="no shipped format"):
        D.deployed_student(run, dp.require_locked(prereg, run))
    assert not (tmp_path / "out" / B.MODEL_DIR).exists()


# --------------------------------------------------------------------- edge


@pytest.fixture
def server():
    made = []

    def make(**kw):
        made.append(te._Server(**kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


def _edge_world(tmp_path: Path, final_bundle: bool = True):
    run, _, bundle = make_bf16_run(tmp_path)
    ds = tmp_path / "ds"
    ds.mkdir()
    for stem in ("test", "held_out"):
        (ds / f"{stem}.jsonl").write_text(json.dumps({"id": f"{stem}-1", "source": "s"}) + "\n")
    # the selection fold: records the train predictions answered (r20, r21)
    te._jsonl(ds / "selection.jsonl", [tg._rec(20), tg._rec(21)])
    te._jsonl(tmp_path / "probe.jsonl", [te._rec("p1")])
    fdir = run / "distill/final"
    fdir.mkdir(parents=True)
    attest = D.attestation(bundle)
    final = {"deployed": {"bundle": attest}} if final_bundle else {"deployed": {}}
    (fdir / "final.json").write_text(json.dumps(final))
    return {
        "run": run,
        "ds": ds,
        "sel": ds / "selection.jsonl",
        "probe": tmp_path / "probe.jsonl",
    }, attest


def _edge(w, url, **kw):
    return edge.run_edge(
        run_dir=w["run"],
        dataset_dir=w["ds"],
        selection=w["sel"],
        probe=w["probe"],
        base_url=url,
        served_model="m",
        label="orin",
        apply=True,
        deployed_format="bf16",
        **kw,
    )


@pytest.mark.parametrize("final_bundle", [True, False])
def test_edge_measures_an_attested_bf16_server(tmp_path, server, final_bundle):
    w, attest = _edge_world(tmp_path, final_bundle)
    srv = server(bundle=attest)
    out = _edge(w, srv.url)
    assert out["deployed_format"] == "bf16"
    assert out["bundle"]["files"] == attest["files"]
    assert out["agreement"]["reference"] == f"distill/train/{WINNER}/predictions.json"
    assert out["agreement"]["n_compared"] == 2  # r20 and r21's choice; noul is not offered
    assert (w["run"] / "distill/edge/orin/edge_manifest.json").is_file()


def test_edge_refuses_a_bf16_server_with_no_bundle_attestation(tmp_path, server):
    """strands-decider serve has no /v1/bundle: edge refuses, it never skips the check."""
    w, _ = _edge_world(tmp_path)
    srv = server()

    def get(path):
        if path == edge.BUNDLE_PATH:
            raise CliError(EXIT_ENV_ERROR, "GET /v1/bundle failed: HTTP Error 404", "")
        return te.CAPPED

    with pytest.raises(CliError, match="cannot attest") as err:
        _edge(w, srv.url, get=get)
    assert "strands-decider serve" in err.value.message
    assert srv.posts == 0


def test_edge_refuses_a_bf16_server_serving_other_files(tmp_path, server):
    w, attest = _edge_world(tmp_path)
    files = {**attest["files"], player.HEAD: "0" * 64}
    srv = server(bundle={"files": files})
    with pytest.raises(CliError, match="differs"):
        _edge(w, srv.url)
    assert srv.posts == 0


def test_edge_refuses_when_the_bf16_bundle_on_disk_changed_since_the_final(tmp_path, server):
    w, attest = _edge_world(tmp_path)
    (w["run"] / "distill/train" / WINNER / "bundle" / "extra.txt").write_text("x")
    srv = server(bundle=attest)
    with pytest.raises(CliError):
        _edge(w, srv.url)
    assert srv.posts == 0
