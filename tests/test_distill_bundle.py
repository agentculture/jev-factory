"""Distill bundle (t21): model + dataset bundles and the private upload, with a fake hub."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.cli._errors import CliError
from jev_factory.distill import bundle as B
from jev_factory.distill import prereg as dp
from jev_factory.release import bundle as release_bundle
from jev_factory.release import hub as hub_module
from tests.release_support import REPO, TOKEN, FakeHub, make_config

REV = "b" * 40
ST = (2).to_bytes(8, "little") + b"{}"  # a valid empty safetensors header
CAL = {
    "temperature_by_kind": {"choice": 1.7, "score": 1.2, "noul": 1.0},
    "vector_by_kind": {},
    "upstream_temperature_by_kind_used": False,
    "format": "bf16",
}
UPSTREAM_CONFIG = {
    "head_type": "pointer",
    "base_model": "Qwen/Qwen3.5-2B",
    "temperature": 2.5,
    "temperature_by_kind": {"choice": 9.9},
}
SEALED_TEXT = "SEALED-HELD-OUT-STATE-TEXT"
DATASET_REPO = REPO + "-data"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _bf16(root: Path) -> Path:
    src = root / "src"
    (src / "lora").mkdir(parents=True)
    (src / player.ADAPTER_CONFIG).write_text('{"r": 16}')
    (src / player.ADAPTER).write_bytes(ST + b"adapter")
    (src / player.HEAD).write_bytes(ST + b"head")
    (src / player.CONFIG).write_text(json.dumps(UPSTREAM_CONFIG))
    for name in player.TOKENIZER_FILES:
        (src / name).write_text("{}")
    out = root / "bf16"
    player.write_layout(src, out, base_revision=REV)
    return out


def _quantized(root: Path) -> Path:
    art = root / "q"
    art.mkdir(parents=True)
    (art / qz.TORSO_FILE).write_bytes(b"torso")
    (art / player.HEAD).write_bytes(ST + b"head")
    (art / player.CONFIG).write_text(json.dumps(UPSTREAM_CONFIG))
    for name in player.TOKENIZER_FILES:
        (art / name).write_text("{}")
    qz.write_manifest(
        art,
        bits=8,
        torchao_api=qz.TORCHAO_APIS[8][0],
        torchao_version="0.18.0",
        torch_version="2.12.1",
        source={"checkpoint": "ckpt", "sha256": {player.HEAD: "0" * 64}},
        base_model="Qwen/Qwen3.5-2B",
        base_model_revision=REV,
    )
    return art


def _row(rid, qid, licence, target="teacher"):
    return {
        "record_id": rid,
        "question_id": qid,
        "kind": "choice",
        "state": f"state of {rid}",
        "instructions": "pick",
        "options": [["a", "A"], ["b", "B"]],
        "source": "Praveenrajus/jev-bench:scifact",
        "licence": licence,
        "gold": {"label": "a"},
        "target_source": target,
        "precedence_rule": "gold+teacher",
        "teacher_filter": "all",
        "teacher_build_id": "build1",
        "gold_provenance": "human",
    }


def make_run(tmp_path: Path, *, fmt="bf16", rows=None, sealed_ids=("h1",)):
    run = tmp_path / "run"
    prereg_path = tmp_path / "prereg.json"
    doc = dp.example_doc()
    prereg_path.write_text(json.dumps(doc), encoding="utf-8")
    dp.lock(prereg_path, run)
    artifact = _bf16(tmp_path) if fmt == "bf16" else _quantized(tmp_path)
    q = run / "distill" / "quant"
    q.mkdir(parents=True)
    cal_path = q / "calibration.json"
    cal_path.write_text(json.dumps(CAL), encoding="utf-8")
    (q / "format_decision.json").write_text(json.dumps({"format": fmt}), encoding="utf-8")
    (q / "quant_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "chosen_format": fmt,
                "deployed_artifact": str(artifact),
                "prereg_sha256": _sha(prereg_path),
                "outputs": {str(cal_path): _sha(cal_path)},
            }
        ),
        encoding="utf-8",
    )
    rows = rows or [
        _row("t1", "q", "mit"),
        _row("t2", "q", "Apache-2.0", "human_gold"),
        _row("t3", "q", "cc-by-nc-4.0"),
    ]
    a = run / "distill" / "assemble"
    a.mkdir(parents=True)
    _jsonl(a / "train_set.jsonl", rows)
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
    (f / "report.json").write_text(json.dumps({"verdict": "ok", "ece": 0.03}), encoding="utf-8")
    ds = tmp_path / "dataset"
    ds.mkdir()
    _jsonl(
        ds / "held_out.jsonl", [{"id": i, "source": "s", "state": SEALED_TEXT} for i in sealed_ids]
    )
    _jsonl(ds / "test.jsonl", [{"id": "x1", "source": "s", "state": SEALED_TEXT}])
    return run, prereg_path, ds


def build(tmp_path, **kw):
    run, prereg_path, ds = make_run(tmp_path, **kw)
    out = tmp_path / "out"
    summary = B.build_bundles(run, out, prereg_path=prereg_path, dataset_dir=ds)
    return run, out, summary


@pytest.mark.parametrize("fmt", ["bf16", "int8"])
def test_model_bundle_has_the_layout_and_the_fitted_temperatures(tmp_path, fmt):
    _, out, summary = build(tmp_path, fmt=fmt)
    model = out / "model"
    config = json.loads((model / player.CONFIG).read_text())
    assert config["temperature_by_kind"] == CAL["temperature_by_kind"]
    assert config["temperature"] == 1.0  # never Strands' upstream 2.5
    assert 9.9 not in config["temperature_by_kind"].values()
    if fmt == "bf16":
        assert player.read_layout(model)["format"] == "hobson-hf-export/1"
        assert summary["layout"] == {"layout": "hobson-hf-export/1"}
    else:
        assert qz.read_artifact(model)["manifest"]["bits"] == 8
    for name in (
        "calibration.json",
        "gate.json",
        "scorer-train.json",
        "report.json",
        "distill_provenance.json",
        "assemble_manifest.json",
        "bundle.json",
        "scan.json",
    ):
        assert (model / name).is_file(), name
    assert json.loads((model / "calibration.json").read_text()) == CAL
    assert release_bundle.bundle_problems(model) == []
    prov = json.loads((model / "distill_provenance.json").read_text())
    assert prov["deployed_format"] == fmt
    assert prov["held_out_published"] is False
    assert prov["teacher_build_ids"] == ["build1"]


def test_training_index_carries_every_rows_target_source_without_text(tmp_path):
    _, out, _ = build(tmp_path)
    index = json.loads((out / "model" / "scorer-train.json").read_text())
    assert [e["target_source"] for e in index["entries"]] == ["teacher", "human_gold", "teacher"]
    assert index["rows"] == 3
    assert "state of t1" not in (out / "model" / "scorer-train.json").read_text()


def _write_gate(run: Path, calibration_sha: str) -> dict:
    doc = {
        "stage": "distill-gate",
        "fitted": True,
        "thresholds_by_kind": {
            "choice": {"floor": 0.6, "margin": None, "max_entropy": None},
            "noul": {"floor": None, "margin": 0.2, "max_entropy": None},
            "score": {"floor": None, "margin": None, "max_entropy": 0.9},
        },
        "objective": {"objective": "max_selective_accuracy"},
        "calibration_sha256": calibration_sha,
        "prereg_sha256": "p" * 64,
    }
    g = run / "distill" / "gate"
    g.mkdir(parents=True)
    (g / "gate.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_without_a_gate_stage_the_bundle_records_none_fitted(tmp_path):
    # the example prereg now registers a gate objective and grid: that is not a fitted gate
    assert "gate" in dp.example_doc()
    _, out, summary = build(tmp_path)
    gate = json.loads((out / "model" / "gate.json").read_text())
    assert gate["fitted"] is False
    assert "reason" in gate
    assert summary["gate_fitted"] is False
    assert gate["thresholds_by_kind"] == {}


def test_the_bundle_ships_the_fitted_gate_from_the_gate_stage(tmp_path):
    run, prereg_path, ds = make_run(tmp_path)
    cal_sha = _sha(run / "distill" / "quant" / "calibration.json")
    written = _write_gate(run, cal_sha)
    summary = B.build_bundles(run, tmp_path / "out", prereg_path=prereg_path, dataset_dir=ds)
    for folder in ("model", "dataset"):
        gate = json.loads((tmp_path / "out" / folder / "gate.json").read_text())
        assert gate["fitted"] is True
        assert gate["thresholds_by_kind"] == written["thresholds_by_kind"]
        assert gate["calibration_sha256"] == cal_sha
        assert gate["source"] == "distill/gate/gate.json"
    assert summary["gate_fitted"] is True


def test_a_gate_fitted_on_another_calibration_is_refused(tmp_path):
    run, prereg_path, ds = make_run(tmp_path)
    _write_gate(run, "0" * 64)
    with pytest.raises(CliError, match="calibration"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg_path, dataset_dir=ds)
    assert not (tmp_path / "out" / "model").exists()


def test_a_quant_manifest_with_no_shipped_format_is_refused(tmp_path):
    """Risk r10: bundle refuses when quantize recorded a failed 'no qualifying format'."""
    run, prereg_path, ds = make_run(tmp_path, fmt="int8")
    manifest = run / "distill" / "quant" / "quant_manifest.json"
    doc = json.loads(manifest.read_text())
    doc.update(status="failed", chosen_format=None, shipped_format=None, shippable=False)
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CliError, match="no shipped format"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg_path, dataset_dir=ds)
    assert not (tmp_path / "out" / "model").exists()


def test_dataset_bundle_keeps_only_apache_or_mit_rows_with_their_licence(tmp_path):
    _, out, summary = build(tmp_path)
    rows = [json.loads(x) for x in (out / "dataset" / "train.jsonl").read_text().splitlines()]
    assert [r["record_id"] for r in rows] == ["t1", "t2"]
    assert {r["licence"] for r in rows} == {"mit", "Apache-2.0"}
    assert all(r["target_source"] for r in rows)
    meta = json.loads((out / "dataset" / "dataset.json").read_text())
    assert meta["rows_excluded_by_licence"] == {"cc-by-nc-4.0": 1}
    assert summary["dataset_rows"] == 2
    assert "state of t3" not in (out / "dataset" / "train.jsonl").read_text()


def test_sealed_text_never_appears_in_either_bundle(tmp_path):
    _, out, _ = build(tmp_path)
    for f in out.rglob("*"):
        if f.is_file():
            assert SEALED_TEXT.encode() not in f.read_bytes(), f
            assert f.name not in ("held_out.jsonl", "test.jsonl")


def test_a_training_row_on_a_sealed_side_refuses(tmp_path):
    run, prereg_path, ds = make_run(tmp_path, sealed_ids=("t1",))
    with pytest.raises(CliError, match="sealed"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg_path, dataset_dir=ds)


def test_a_symlink_in_the_student_or_inputs_is_refused(tmp_path):
    run, prereg_path, ds = make_run(tmp_path)
    artifact = Path(
        json.loads((run / "distill/quant/quant_manifest.json").read_text())["deployed_artifact"]
    )
    (artifact / "evil").symlink_to(tmp_path / "prereg.json")
    with pytest.raises(CliError, match="symlink"):
        B.build_bundles(run, tmp_path / "out", prereg_path=prereg_path)
    (artifact / "evil").unlink()
    report = run / "distill/final/report.json"
    real = tmp_path / "real-report.json"
    real.write_text("{}")
    report.unlink()
    report.symlink_to(real)
    with pytest.raises(CliError, match="symlink"):
        B.build_bundles(run, tmp_path / "out2", prereg_path=prereg_path)


def test_a_changed_training_set_or_calibration_refuses(tmp_path):
    run, prereg_path, _ = make_run(tmp_path)
    (run / "distill/assemble/train_set.jsonl").write_text(json.dumps(_row("zz", "q", "mit")) + "\n")
    with pytest.raises(CliError, match="frozen"):
        B.build_bundles(run, tmp_path / "o", prereg_path=prereg_path)
    (tmp_path / "b").mkdir()
    run2, prereg2, _ = make_run(tmp_path / "b")
    (run2 / "distill/quant/calibration.json").write_text(json.dumps({**CAL, "x": 1}))
    with pytest.raises(CliError, match="sha256"):
        B.build_bundles(run2, tmp_path / "o2", prereg_path=prereg2)


def test_upstream_temperatures_in_calibration_refuse(tmp_path):
    run, prereg_path, _ = make_run(tmp_path)
    cal = run / "distill/quant/calibration.json"
    cal.write_text(json.dumps({**CAL, "upstream_temperature_by_kind_used": True}))
    qm = run / "distill/quant/quant_manifest.json"
    doc = json.loads(qm.read_text())
    doc["outputs"] = {str(cal): _sha(cal)}
    qm.write_text(json.dumps(doc))
    with pytest.raises(CliError, match="upstream"):
        B.build_bundles(run, tmp_path / "o", prereg_path=prereg_path)


def test_no_apache_or_mit_rows_refuses_the_dataset(tmp_path):
    run, prereg_path, _ = make_run(tmp_path, rows=[_row("t1", "q", "cc-by-nc-4.0")])
    with pytest.raises(CliError, match="Apache-2.0 or MIT"):
        B.build_bundles(run, tmp_path / "o", prereg_path=prereg_path)


# --------------------------------------------------------------------- upload (fake hub)


def _scan_inputs(tmp_path: Path, probe_rows=(), names=("zz-side-model",)) -> dict:
    """Operator-held scan inputs, outside the bundle; returns the config overrides."""
    private = tmp_path / "private"
    private.mkdir(exist_ok=True)
    probe = private / "probe.jsonl"
    _jsonl(probe, list(probe_rows) or [{"id": "probe-1", "state": "PROBE-STATE-TEXT"}])
    side = private / "side.txt"
    side.write_text("# private\n" + "\n".join(names) + "\n", encoding="utf-8")
    return {B.PROBE_SET_KEY: str(probe), B.SIDE_MODEL_LIST_KEY: str(side)}


def _upload(bundles, hub, *, apply, environ=None, scan_cfg=None, tmp_path=None):
    cfg = scan_cfg if scan_cfg is not None else _scan_inputs(tmp_path or bundles["model"].parent)
    return B.upload_bundles(
        bundles,
        model_repo=REPO,
        dataset_repo=DATASET_REPO,
        config=make_config(**cfg),
        apply=apply,
        hub_client=hub,
        environ=environ if environ is not None else {"HF_TOKEN": TOKEN},
    )


def test_upload_without_apply_sends_nothing(tmp_path, monkeypatch):
    _, out, _ = build(tmp_path)
    hub = FakeHub(tmp_path / "remote")
    monkeypatch.setattr(hub_module.importlib, "import_module", lambda n: pytest.fail(n))
    result = _upload(
        {"model": out / "model", "dataset": out / "dataset"}, hub, apply=False, environ={}
    )
    assert hub.calls == []
    assert result["applied"] is False
    assert result["model"]["private"] is True
    assert not (tmp_path / "remote").exists()


def test_upload_apply_is_private_fetched_back_and_hash_verified(tmp_path):
    _, out, _ = build(tmp_path)
    hub = FakeHub(tmp_path / "remote")
    result = _upload({"model": out / "model", "dataset": out / "dataset"}, hub, apply=True)
    assert result["applied"]
    assert result["model"]["private"]
    assert result["dataset"]["private"]
    names = [c[0] for c in hub.calls]
    assert names.count("create_repo") == 2
    assert names.count("snapshot_download") == 2
    creates = [c for c in hub.calls if c[0] == "create_repo"]
    assert all(c[2]["private"] is True for c in creates)
    assert [c[2]["repo_type"] for c in creates] == ["model", "dataset"]
    assert not any("public" in n.lower() for n in names)


def test_upload_fails_when_the_fetched_copy_differs(tmp_path):
    _, out, _ = build(tmp_path)

    def flip(local: Path) -> None:
        (local / "gate.json").write_text("{}")

    hub = FakeHub(tmp_path / "remote", tamper=flip)
    with pytest.raises(hub_module.UploadError, match="differs"):
        _upload({"model": out / "model", "dataset": out / "dataset"}, hub, apply=True)


def test_no_public_visibility_call_exists_in_the_source():
    src = Path(B.__file__).read_text()
    assert "private=False" not in src
    assert "make_public" not in src


# --------------------------------------------------------------------- NOTICE (t16)

CCBY = "Dataset X by Alice, CC BY 4.0, https://example.org/x"


def _ccby_rows():
    row = {**_row("t2", "q", "cc-by-4.0", "human_gold"), "attribution": CCBY}
    return [_row("t1", "q", "mit"), row]


def _manifest(tmp_path: Path, attribution=CCBY) -> Path:
    path = tmp_path / "dataset" / "manifest.json"
    table = {"s": {"licences": ["cc-by-4.0"], "allowed": True}}
    if attribution:
        table["s"]["attribution"] = attribution
    path.write_text(json.dumps({"licences": table}), encoding="utf-8")
    return path


def test_notice_lists_every_manifest_attribution_in_both_bundles(tmp_path):
    """AC1: NOTICE carries every attribution string, in the model and the dataset bundle."""
    run, prereg_path, ds = make_run(tmp_path, rows=_ccby_rows())
    _manifest(tmp_path, [CCBY, "Other by Bob, CC BY 4.0"])
    out = tmp_path / "out"
    B.build_bundles(run, out, prereg_path=prereg_path, dataset_dir=ds)
    for side in ("model", "dataset"):
        text = (out / side / "NOTICE").read_text()
        assert CCBY in text
        assert "Other by Bob, CC BY 4.0" in text


def test_cc_by_record_without_attribution_is_left_out_of_the_dataset(tmp_path):
    """AC1: a cc-by-4.0 record with no attribution never reaches the dataset bundle."""
    rows = [_row("t1", "q", "mit"), _row("t2", "q", "cc-by-4.0")]
    run, prereg_path, ds = make_run(tmp_path, rows=rows)
    out = tmp_path / "out"
    B.build_bundles(run, out, prereg_path=prereg_path, dataset_dir=ds)
    text = (out / "dataset" / "train.jsonl").read_text()
    assert "t2" not in text


def test_cc_by_dataset_bundle_without_notice_is_refused_at_upload(tmp_path):
    """AC1: removing NOTICE from a built cc-by dataset bundle makes upload refuse."""
    run, prereg_path, ds = make_run(tmp_path, rows=_ccby_rows())
    _manifest(tmp_path)
    out = tmp_path / "out"
    B.build_bundles(run, out, prereg_path=prereg_path, dataset_dir=ds)
    (out / "dataset" / "NOTICE").unlink()
    with pytest.raises(CliError, match="NOTICE"):
        _upload(
            {"model": out / "model", "dataset": out / "dataset"},
            FakeHub(tmp_path / "r"),
            apply=False,
            tmp_path=tmp_path,
        )


# --------------------------------------------------------------------- pre-upload scans (t16)


def _refusal(tmp_path, out, **kw) -> str:
    bundles = {"model": out / "model", "dataset": out / "dataset"}
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(CliError) as err:
        _upload(bundles, hub, apply=False, tmp_path=tmp_path, **kw)
    assert hub.calls == []
    return err.value.message


def _append(path: Path, line: str) -> None:
    path.write_text(path.read_text() + line + "\n", encoding="utf-8")


def test_upload_refuses_a_side_model_name_and_prints_file_and_line_only(tmp_path):
    """AC2: a side-model name from the operator list refuses; only file:line is printed."""
    _, out, _ = build(tmp_path)
    _append(out / "model" / "distill_provenance.json", "note ZZ-Side-Model here")
    message = _refusal(tmp_path, out)
    assert "model/distill_provenance.json:" in message
    assert "zz-side-model" not in message.lower()


def test_upload_refuses_a_private_ip(tmp_path):
    """AC2: a hostname/private-IP hit refuses."""
    _, out, _ = build(tmp_path)
    _append(out / "dataset" / "dataset.json", "host 10.20.30.40")
    message = _refusal(tmp_path, out)
    assert "dataset/dataset.json:" in message
    assert "10.20.30.40" not in message


def test_upload_refuses_a_secret(tmp_path):
    """AC2: a secret-shaped string refuses, without echoing it."""
    _, out, _ = build(tmp_path)
    secret = "AKIA" + "ABCDEFGHIJKLMNOP"
    _append(out / "model" / "gate.json", secret)
    message = _refusal(tmp_path, out)
    assert "model/gate.json:" in message
    assert secret not in message


def test_upload_refuses_a_probe_set_record_by_id_or_state(tmp_path):
    """AC2: any record of the Clef probe set in the bundle refuses (id or state match)."""
    _, out, _ = build(tmp_path)
    _append(out / "dataset" / "train.jsonl", json.dumps({"record_id": "probe-1"}))
    message = _refusal(tmp_path, out)
    assert "dataset/train.jsonl:" in message
    assert "probe-1" not in message


def test_upload_refuses_a_probe_state_under_another_id(tmp_path):
    """AC2: a probe state text reused under a new record id still refuses."""
    _, out, _ = build(tmp_path)
    _append(
        out / "dataset" / "train.jsonl", json.dumps({"record_id": "n", "state": "PROBE-STATE-TEXT"})
    )
    message = _refusal(tmp_path, out)
    assert "probe-set record" in message
    assert "PROBE-STATE-TEXT" not in message


def test_upload_refuses_when_the_scan_inputs_are_not_configured(tmp_path):
    """AC2: fail closed, no probe set or side-model list path means no upload."""
    _, out, _ = build(tmp_path)
    bundles = {"model": out / "model", "dataset": out / "dataset"}
    with pytest.raises(CliError, match=B.PROBE_SET_KEY):
        _upload(bundles, FakeHub(tmp_path / "r"), apply=False, scan_cfg={})


def test_cc_by_rows_need_their_own_attribution_and_the_manifest_licence_table(tmp_path):
    """Scope addition: cc-by-4.0 rows are admitted only with a non-blank attribution; NOTICE
    is written from t7's licence_table whenever one is bundled."""
    rows = [
        _row("t1", "q", "mit"),
        {**_row("t2", "q", "cc-by-4.0"), "attribution": "Row by Carol, CC BY 4.0"},
        {**_row("t3", "q", "cc-by-4.0"), "attribution": "  "},
        _row("t4", "q", "cc-by-4.0"),
    ]
    run, prereg_path, ds = make_run(tmp_path, rows=rows)
    table = {"cc-by-4.0": {"count": 3, "attributions": [CCBY]}}
    (ds / "manifest.json").write_text(json.dumps({"licence_table": table}), encoding="utf-8")
    out = tmp_path / "out"
    B.build_bundles(run, out, prereg_path=prereg_path, dataset_dir=ds)
    kept = [
        json.loads(x)["record_id"]
        for x in (out / "dataset" / "train.jsonl").read_text().splitlines()
    ]
    assert kept == ["t1", "t2"]
    for side in ("model", "dataset"):
        text = (out / side / "NOTICE").read_text()
        assert CCBY in text
        assert "Row by Carol, CC BY 4.0" in text
