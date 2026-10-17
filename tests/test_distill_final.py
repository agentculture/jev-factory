"""Distill final measurement (t20, t14): Clef, stock, the bf16 student and the shipped quant.

Fake clients stand in for the models; the shipped quant is "pre-served" by a real loopback
``quantized.make_server`` answering ``/v1/bundle`` and ``/v1/stats`` (no GPU, no torch).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import quantized as qz
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import edge as dedge
from jev_factory.distill import final as F
from jev_factory.distill import prereg as dp
from jev_factory.distill import report as R
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.distill.runspec import Bootstrap
from jev_factory.distill.systemone import CallResult, QuestionResult
from jev_factory.measure import once
from jev_factory.review import gold

CLEAN = "Praveenrajus/jev-bench:scifact"
NOW = "2026-10-14T00:00:00+00:00"
STUDENT_CAL = {"temperature_by_kind": {"choice": 1.5}, "vector_by_kind": {}}
FORMAT = "torch_int8_wo"
CAP = 4 * 1024**3
STATS = {
    "rss_bytes": 700_000_000,
    "mem_available_bytes": 30_000_000_000,
    "device_model": "NVIDIA Jetson AGX Orin Developer Kit",
    "nv_tegra_release": "# R36 (release), REVISION: 4.3",
    "cgroup": {"version": 2, "path": "/x", "memory_max": CAP, "effective_memory_max": CAP},
}


def _rec(rid: str, split: str) -> Record:
    return Record(
        id=rid,
        state=f"state {rid}",
        questions={"c": Question("choice", [("a", "A"), ("b", "B")])},
        gold={"c": Gold(label="a")},
        split=split,
        source=CLEAN,
        licence="mit",
        gold_provenance="human",
    )


def _artifact(root: Path) -> Path:
    """A manifest-verified int8 artifact with fake weights (what the quant stage deploys)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / qz.TORSO_FILE).write_bytes(b"int8-torso")
    (root / player.HEAD).write_bytes(b"fp32-head")
    (root / player.CONFIG).write_text(
        json.dumps({"head_type": "pointer", "base_model": "Qwen/Qwen3.5-2B", "use_lora": False})
    )
    for name in player.TOKENIZER_FILES:
        (root / name).write_text("{}")
    qz.write_manifest(
        root,
        bits=8,
        torchao_api=qz.TORCHAO_APIS[8][0],
        torchao_version="0.18.0",
        torch_version="2.12.1",
        source={"checkpoint": "/ckpt", "sha256": {player.HEAD: "0" * 64}},
        base_model="Qwen/Qwen3.5-2B",
        base_model_revision="b" * 40,
    )
    return root


class FakeClient:
    """Answers ``a`` with *p_a*; logs every record id it was asked (and into *log*)."""

    def __init__(self, name: str, p_a: float, log=None):
        self.name, self.p_a, self.asked, self.log = name, p_a, [], log

    def ask(self, record):
        self.asked.append(record["id"])
        if self.log is not None:
            self.log.append(f"ask {self.name}")
        dist = {"a": self.p_a, "b": round(1.0 - self.p_a, 9)}
        return CallResult(
            str(record["id"]), self.name, {}, "sha", [QuestionResult("c", "choice", dist)]
        )


class Board:
    """The AGX Orin behind a tunnel: a loopback quantized server with /v1/bundle, /v1/stats."""

    def __init__(self, artifact: Path, log: list):
        self.artifact, self.log = artifact, log
        self.stats = dict(STATS)
        self.bundle_override = None
        self.bundle_after = None  # what /v1/bundle answers from the second call on
        self.bundle_calls = 0
        self.srv = qz.make_server(
            "127.0.0.1",
            0,
            evaluate=lambda body: {},
            health=lambda: {"status": "ok"},
            bundle=self._bundle,
            stats=self._stats,
        )
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(
            target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()

    def _bundle(self):
        self.log.append("GET /v1/bundle")
        self.bundle_calls += 1
        if self.bundle_after is not None and self.bundle_calls > 1:
            return self.bundle_after
        return self.bundle_override or qz.bundle_attestation(self.artifact)

    def _stats(self):
        self.log.append("GET /v1/stats")
        return self.stats

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _approve(path: Path, records, at="2026-10-14T00:00:00Z") -> None:
    for rec in records:
        entry, _ = gold.make_entry(rec, "approve", at=at)
        gold.append_entry(path, entry)


def _locked_run(tmp_path, doc):
    run_dir = tmp_path / "run"
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(doc), encoding="utf-8")
    dp.lock(prereg, run_dir)
    data = tmp_path / "data"
    data.mkdir()
    sides = {"test": [f"t{i}" for i in range(12)], "held_out": [f"h{i}" for i in range(6)]}
    for side, ids in sides.items():
        lines = [json.dumps(_rec(i, side).to_json(), sort_keys=True) for i in ids]
        (data / f"{side}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = gold.ledger_path(data / "test.jsonl")
    _approve(ledger, [_rec(i, "test") for i in sides["test"]])
    return run_dir, prereg, data, sides, ledger


@pytest.fixture
def env(tmp_path):
    yield from _quant_env(tmp_path, dp.example_doc())


def _quant_env(tmp_path, doc):
    run_dir, prereg, data, sides, ledger = _locked_run(tmp_path, doc)
    qdir = run_dir / "distill" / "quant"
    qdir.mkdir(parents=True)
    artifact = _artifact(qdir / "builds" / FORMAT)
    cal = qdir / "calibration.json"
    cal.write_text(json.dumps({**STUDENT_CAL, "format": FORMAT}), encoding="utf-8")
    (qdir / "format_decision.json").write_text(
        json.dumps({"decision": "deployed_format", "format": FORMAT}), encoding="utf-8"
    )
    (qdir / "quant_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "chosen_format": FORMAT,
                "deployed_artifact": str(artifact),
                "heal_count": 0,
                "quant_vs_bf16_delta_points": 0.0,
                "deployed_selection_ece": 0.05,
                "outputs": {str(cal): dp.sha256_file(cal)},
            }
        ),
        encoding="utf-8",
    )
    log: list = []
    board = Board(artifact, log)
    yield {
        "run_dir": run_dir,
        "prereg": prereg,
        "data": data,
        "sides": sides,
        "qdir": qdir,
        "artifact": artifact,
        "ledger": ledger,
        "board": board,
        "log": log,
    }
    board.close()


def _models(env, log, *, fail_start=None, p=None):
    """Clef, stock and the bf16 student are started and stopped; the quant is pre-served."""
    p = p or {"clef": 0.9, "stock": 0.6, "student_bf16": 0.8, "student": 0.8}
    out = {}
    for name in F.MODELS:

        def start(name=name):
            if name == fail_start:
                raise RuntimeError(f"{name} failed to start")
            log.append(f"start {name}")

        if name == R.STUDENT:
            target = bench.Target(name, env["board"].url, name)  # pre-served: no start, no stop
        else:
            target = bench.Target(
                name,
                "http://127.0.0.1:9",
                name,
                start=start,
                stop=lambda n=name: log.append(f"stop {n}"),
            )
        out[name] = F.FinalModel(target=target, client=FakeClient(name, p[name], log))
    return out


def _run(env, models, **kw):
    kw.setdefault("apply", True)
    return F.run_final(
        prereg_path=env["prereg"],
        run_dir=env["run_dir"],
        dataset_dir=env["data"],
        models=models,
        memory_probe=None,
        host="box",
        now=lambda: NOW,
        bootstrap=Bootstrap(resamples=50),
        **kw,
    )


def _ledger(env):
    return once.OnceLedger(env["run_dir"]).records()


def test_dry_run_by_default_scores_nothing_and_writes_no_ledger(env):
    log = []
    models = _models(env, log)
    plan = _run(env, models, apply=False)
    assert plan["dry_run"] is True
    assert plan["sides"]["test"]["count"] == 12
    assert plan["sides"]["held_out"]["count"] == 6
    assert log == []
    assert all(not m.client.asked for m in models.values())
    assert _ledger(env) == []


def test_one_ledgered_invocation_scores_all_four_on_test_and_held_out(env):
    log = []
    models = _models(env, log)
    out = _run(env, models)
    every = env["sides"]["test"] + env["sides"]["held_out"]
    for m in models.values():
        assert sorted(m.client.asked) == sorted(every)
    lifecycle = [e for e in log if e.split()[0] in ("start", "stop")]
    started = [n for n in F.MODELS if n != R.STUDENT]  # the quant is pre-served
    assert lifecycle == [f"{verb} {n}" for n in started for verb in ("start", "stop")]
    records = _ledger(env)
    assert sorted((r.side, r.slice, r.status) for r in records) == [
        (once.HELD_OUT, once.FULL, once.MEASURED),
        (once.TEST, once.FULL, once.MEASURED),
    ]
    sha = {r.side: r.split_sha256 for r in records}
    assert sha[once.TEST] == dataset.load_sealed(env["data"] / "test.jsonl").sha256
    final_dir = env["run_dir"] / F.FINAL_REL
    for name in F.MODELS:
        rows = json.loads((final_dir / "predictions" / f"{name}.json").read_text())
        assert len(rows) == 18
    written = json.loads((final_dir / F.REPORT_FILE).read_text())
    assert written["c17"]["builds"]
    assert written["c42"]["bars"]
    assert written["c43"]["bars"]
    assert out["report"]["result"] in (R.PASSED, R.MISSED)
    assert set(out["final"]["runtime"]) == set(F.MODELS)


@pytest.mark.parametrize("apply", [True, False])
def test_a_second_final_is_refused_without_a_deviation_id(env, apply):
    _run(env, _models(env, []))
    log = []
    models = _models(env, log)
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert exc.value.code == EXIT_USER_ERROR != 0
    assert "deviation" in exc.value.remediation
    assert log == []
    assert all(not m.client.asked for m in models.values())
    assert len(_ledger(env)) == 2


def test_a_final_for_only_some_of_the_models_is_refused(env):
    models = _models(env, [])
    del models["stock"], models["clef"]
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert exc.value.code != 0
    assert "missing ['clef', 'stock']" in exc.value.message
    assert _ledger(env) == []
    _run(env, _models(env, []))
    student_only = {"student": _models(env, [])["student"]}
    with pytest.raises(CliError):  # a later student-only (or any) final is refused too
        _run(env, student_only)


def test_a_second_final_with_a_deviation_id_runs_and_is_recorded(env):
    _run(env, _models(env, []))
    _run(env, _models(env, []), deviation_id="d21")
    records = _ledger(env)
    assert len(records) == 4
    assert [r.deviation for r in records[2:]] == ["d21", "d21"]


def test_a_bad_deviation_id_is_refused(env):
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models, deviation_id="not a deviation id!")


def test_a_run_whose_first_start_fails_writes_no_ledger_record(env):
    models = _models(env, [], fail_start="clef")
    with pytest.raises(RuntimeError):
        _run(env, models)
    assert _ledger(env) == []
    _run(env, _models(env, []))  # the real run is not blocked
    assert len(_ledger(env)) == 2


def test_a_run_that_fails_after_scoring_is_still_ledgered(env):
    models = _models(env, [], fail_start="student_bf16")
    with pytest.raises(RuntimeError):
        _run(env, models)
    statuses = {r.status for r in _ledger(env)}
    assert statuses == {once.FAILED_MID_RUN}
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models)


def test_the_calibration_each_row_used_is_recorded(env, tmp_path):
    clef_cal = tmp_path / "clef_cal.json"
    clef_cal.write_text(json.dumps({"temperature_by_kind": {"choice": 2.0}}), encoding="utf-8")
    out = _run(env, _models(env, []), clef_calibration=clef_cal)
    cals = out["final"]["calibrations"]
    assert cals["student"]["source"] == F.STUDENT_CALIBRATION_REL
    assert cals["student"]["sha256"] == dp.sha256_file(env["qdir"] / "calibration.json")
    assert cals["student"]["calibration"]["temperature_by_kind"] == {"choice": 1.5}
    assert cals["stock"]["source"] == F.CAL_SERVED
    assert cals["stock"]["calibration"] is None
    assert cals["clef"]["source"] == F.CAL_RAW
    assert cals["clef"]["calibration"] is None
    assert cals[R.CLEF_CALIBRATED]["sha256"] == dp.sha256_file(clef_cal)
    assert out["final"]["deployed"]["format"] == FORMAT
    assert cals["student_bf16"]["source"] == F.CAL_RAW
    stats = out["report"]["sides"]["test"]["pools"][R.HEADLINE]["stats"]
    assert "clef_calibrated.ece" in stats


def test_the_student_needs_the_quant_calibration_it_was_shipped_with(env):
    (env["qdir"] / "calibration.json").unlink()
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models)
    assert _ledger(env) == []


def _no_shipped_format(qdir):
    """What quant writes when no quantized build qualifies (risk r10): a failed result."""
    (qdir / "calibration.json").unlink()
    (qdir / "format_decision.json").write_text(json.dumps({"format": None}), encoding="utf-8")
    (qdir / "quant_manifest.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "outcome": "no qualifying format",
                "reason": "no qualifying format: every build is over the size budget",
                "chosen_format": None,
                "shipped_format": None,
                "shippable": False,
                "deployed_artifact": None,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("apply", [False, True])
def test_a_quant_manifest_with_no_shipped_format_is_refused(env, apply):
    """Risk r10: the final refuses when quantize recorded no shipped format."""
    _no_shipped_format(env["qdir"])
    models = _models(env, [])
    with pytest.raises(CliError, match="no shipped format") as exc:
        _run(env, models, apply=apply)
    assert "bf16 fallback" in exc.value.remediation
    assert _ledger(env) == []


def test_a_changed_quant_calibration_is_refused(env):
    (env["qdir"] / "calibration.json").write_text(json.dumps(STUDENT_CAL), encoding="utf-8")
    models = _models(env, [])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "sha256" in exc.value.message


def test_an_unlocked_prereg_is_refused(env):
    (env["run_dir"] / dp.LOCK_REL).unlink()
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models)


def test_runtime_is_recorded_for_all_three_on_the_same_host(env):
    out = _run(env, _models(env, []))
    runtime = out["final"]["runtime"]
    assert {runtime[n]["host"] for n in F.MODELS} == {"box"}
    assert all(runtime[n]["latency"]["n"] == 18 for n in F.MODELS)
    assert out["report"]["runtime"]["same_box"] is True


def test_final_reads_the_sealed_sides_only_by_their_sealed_sha(env, monkeypatch):
    real = dataset.load_sealed

    def tampered(path):
        side = real(path)
        return dataset.SealedSide(side.path, "0" * 64, side.ids, side.count, side.by_source)

    monkeypatch.setattr(F.dataset, "load_sealed", tampered)
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models)
    assert Path(env["run_dir"] / F.FINAL_REL / F.REPORT_FILE).exists() is False


# --------------------------------------------------------------------- the decision gate (t22)

GATE_THRESHOLDS = {
    "choice": {"floor": 0.95, "margin": None, "max_entropy": None},
    "noul": {"floor": None, "margin": None, "max_entropy": None},
    "score": {"floor": None, "margin": None, "max_entropy": None},
}


def _write_gate(env, calibration_sha256=None, fitted=True):
    path = env["run_dir"] / "distill" / "gate" / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = calibration_sha256 or dp.sha256_file(env["qdir"] / "calibration.json")
    doc = {"fitted": fitted, "thresholds_by_kind": GATE_THRESHOLDS, "calibration_sha256": sha}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_the_fitted_gate_is_passed_to_the_report(env):
    path = _write_gate(env)
    plan = _run(env, _models(env, []), apply=False)
    assert plan["gate"]["sha256"] == dp.sha256_file(path)
    out = _run(env, _models(env, []))
    dg = out["report"]["decision_gate"]
    assert dg["fitted"] is True  # reported only: "gated" stays False (never a bar)
    assert dg["thresholds_by_kind"] == GATE_THRESHOLDS
    assert set(dg["sides"]) == {"test", "held_out"}
    assert out["final"]["gate"]["sha256"] == dp.sha256_file(path)


def test_without_a_gate_the_report_says_none_was_fitted(env):
    out = _run(env, _models(env, []))
    assert out["report"]["decision_gate"]["fitted"] is False
    assert out["final"]["gate"] is None


@pytest.mark.parametrize("apply", [True, False])
def test_a_gate_fitted_on_another_calibration_is_refused(env, apply):
    _write_gate(env, calibration_sha256="d" * 64)
    models = _models(env, [])
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert "calibration" in exc.value.message
    assert _ledger(env) == []


def test_an_unfitted_gate_file_is_refused(env):
    _write_gate(env, fitted=False)
    models = _models(env, [])
    with pytest.raises(CliError):
        _run(env, models)
    assert _ledger(env) == []


# --------------------------------------------------------------------- t14: the attested quant


def _asks(log):
    return [e for e in log if e.startswith("ask ")]


def test_a_pre_served_student_is_attested_before_any_request_and_recorded(env):
    """AC1: final accepts a pre-served student at a 127.0.0.1 URL (an SSH tunnel to the AGX
    Orin) and checks GET /v1/bundle before any test request; the cap is recorded."""
    log = env["log"]
    models = _models(env, log)
    assert models[R.STUDENT].target.start is None
    out = _run(env, models)
    first_ask = log.index(next(e for e in log if e.startswith("ask ")))
    assert log.index("GET /v1/bundle") < first_ask
    assert log.index("GET /v1/stats") < first_ask
    assert log[-1] == "GET /v1/bundle"  # fetched again once the student was scored
    final = out["final"]
    attested = qz.bundle_attestation(env["artifact"])
    bundle = final["deployed"]["bundle"]
    assert bundle["files"] == attested["files"]
    assert bundle["manifest_sha256"] == attested["manifest_sha256"]
    assert bundle["weight_bytes"] == attested["weight_bytes"]
    server = final["student_server"]
    assert server["memory_cap"]["effective_memory_max"] == CAP
    assert server["memory_cap"]["memory_max"] == CAP
    assert server["board"]["device_model"] == STATS["device_model"]
    assert server["bundle"]["files"] == attested["files"]
    assert final["models"][R.STUDENT]["pre_served"] is True
    written = json.loads((env["run_dir"] / F.FINAL_REL / F.FINAL_FILE).read_text())
    assert written["student_server"]["memory_cap"]["effective_memory_max"] == CAP
    # the edge stage reads the final's bundle from exactly this key
    assert dedge.expected_bundle(env["run_dir"])["files"] == attested["files"]
    assert out["report"]["c17"]["builds"]["quant"]["bars"][-1]["value"] == attested["weight_bytes"]


@pytest.mark.parametrize(
    "served",
    [
        {"files": {qz.TORSO_FILE: "0" * 64}},
        "manifest",
    ],
)
def test_a_served_bundle_that_differs_is_refused_before_any_request(env, served):
    """AC1: a /v1/bundle mismatch (files or manifest sha256) is refused before any request."""
    board = env["board"]
    real = qz.bundle_attestation(env["artifact"])
    if served == "manifest":
        board.bundle_override = {**real, "manifest_sha256": "f" * 64}
    else:
        board.bundle_override = {**real, **served}
    models = _models(env, env["log"])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert exc.value.code == EXIT_USER_ERROR
    assert "/v1/bundle" in exc.value.message
    assert _asks(env["log"]) == []
    assert not any(e.startswith("start") for e in env["log"])
    assert _ledger(env) == []


def test_a_board_without_a_memory_cap_is_refused_before_any_request(env):
    env["board"].stats = {**STATS, "cgroup": {"memory_max": "max", "effective_memory_max": None}}
    models = _models(env, env["log"])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "memory cap" in exc.value.message
    assert _asks(env["log"]) == []
    assert _ledger(env) == []


def test_a_student_url_that_is_not_localhost_is_refused(env):
    models = _models(env, env["log"])
    remote = bench.Target(R.STUDENT, "http://10.0.0.5:8000", R.STUDENT)
    models[R.STUDENT] = F.FinalModel(target=remote, client=FakeClient("student", 0.8))
    with pytest.raises(CliError):
        _run(env, models)
    assert _asks(env["log"]) == []
    assert _ledger(env) == []


def test_a_bundle_swapped_while_measuring_fails_the_run_and_is_ledgered(env):
    env["board"].bundle_after = {"files": {qz.TORSO_FILE: "0" * 64}, "manifest_sha256": "f" * 64}
    models = _models(env, env["log"])
    with pytest.raises(CliError):
        _run(env, models)
    assert {r.status for r in _ledger(env)} == {once.FAILED_MID_RUN}
    assert not (env["run_dir"] / F.FINAL_REL / F.REPORT_FILE).exists()


@pytest.mark.parametrize("apply", [True, False])
def test_final_refuses_while_the_teacher_label_lock_exists(env, apply):
    """AC1: final refuses while the run's teacher-label lock exists."""
    dedge.acquire_teacher_lock(env["run_dir"])
    models = _models(env, env["log"])
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert "teacher-label lock" in exc.value.message
    assert env["log"] == []
    assert _ledger(env) == []
    dedge.release_teacher_lock(env["run_dir"])
    _run(env, _models(env, []))


# --------------------------------------------------------------------- t14: the gold ledger


def _rewrite_ledger(env, records):
    env["ledger"].unlink()
    _approve(env["ledger"], records)


@pytest.mark.parametrize("apply", [True, False])
def test_a_test_record_without_an_approve_entry_is_refused_printing_ids_only(env, apply):
    """AC2: final refuses any test record without an approve entry (approved_ids); ids only."""
    approved = [_rec(i, "test") for i in env["sides"]["test"] if i not in ("t3", "t7")]
    _rewrite_ledger(env, approved)
    models = _models(env, env["log"])
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert exc.value.code == EXIT_USER_ERROR
    assert exc.value.message.endswith(": t3, t7")
    assert "state t3" not in exc.value.message  # never record text
    assert env["log"] == []
    assert _ledger(env) == []


def test_a_rejected_or_edited_after_approval_record_is_refused(env):
    recs = [_rec(i, "test") for i in env["sides"]["test"]]
    entry, _ = gold.make_entry(recs[0], "reject", at="2026-10-14T01:00:00Z")
    gold.append_entry(env["ledger"], entry)  # the latest decision wins
    models = _models(env, [])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert exc.value.message.endswith(": t0")
    _rewrite_ledger(env, recs)
    edited = recs[1].to_json()
    edited["state"] = "an edited state"
    lines = [json.dumps(r.to_json(), sort_keys=True) for r in recs]
    lines[1] = json.dumps(edited, sort_keys=True)
    (env["data"] / "test.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    models = _models(env, [])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert exc.value.message.endswith(": t1")


@pytest.mark.parametrize("ledger", ["missing", "corrupt"])
def test_a_missing_or_corrupt_gold_ledger_fails_closed(env, ledger):
    if ledger == "missing":
        env["ledger"].unlink()
    else:
        env["ledger"].write_text("{not json\n", encoding="utf-8")
    models = _models(env, env["log"])
    with pytest.raises(CliError):
        _run(env, models)
    assert env["log"] == []
    assert _ledger(env) == []


def test_an_explicit_gold_ledger_path_is_used(env, tmp_path):
    moved = tmp_path / "review" / "drafted.gold-ledger.jsonl"
    moved.parent.mkdir()
    env["ledger"].rename(moved)
    plan = _run(env, _models(env, []), apply=False, inputs=F.FinalInputs(gold_ledger=moved))
    assert plan["gold_ledger"]["file"] == str(moved)
    assert plan["gold_ledger"]["approved_test"] == 12
    assert plan["gold_ledger"]["sha256"] == dp.sha256_file(moved)


def _model_drafted_sides(env) -> None:
    """Rewrite both sealed sides as model-drafted gold and remove the gold ledger."""
    for side, ids in env["sides"].items():
        lines = []
        for i in ids:
            data = _rec(i, side).to_json()
            data["gold_provenance"] = "model-generated"
            lines.append(json.dumps(data, sort_keys=True))
        (env["data"] / f"{side}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    env["ledger"].unlink()


@pytest.mark.parametrize("apply", [True, False])
def test_the_unverified_gold_final_is_refused_without_a_deviation_id(env, apply):
    """d10: measuring on gold no human approved needs a recorded deviation."""
    _model_drafted_sides(env)
    models = _models(env, env["log"])
    inputs = F.FinalInputs(unverified_gold=True)
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply, inputs=inputs)
    assert exc.value.code == EXIT_USER_ERROR
    assert "deviation" in exc.value.message
    assert env["log"] == []
    assert _ledger(env) == []


def test_the_unverified_gold_final_needs_no_ledger_and_marks_the_report(env):
    """d10: no gold ledger is read; the report carries an UNVERIFIED c17 block."""
    _model_drafted_sides(env)
    out = _run(
        env, _models(env, []), deviation_id="d10", inputs=F.FinalInputs(unverified_gold=True)
    )
    approval = out["final"]["gold_ledger"]
    assert approval["verification"] == "unverified"
    assert approval["file"] is None
    assert approval["unverified_test"] == 12
    assert approval["unverified_held_out"] == 6
    report = out["report"]
    assert report["headline_n"]["records"] == 0
    block = report["unverified_c17"]
    assert block["verified"] is False
    assert block["n_records"] == 12
    assert block["unverified_share"] == 1.0
    assert report["unverified_result"] == block["result"]
    assert report["verification"].startswith("UNVERIFIED")
    written = json.loads((env["run_dir"] / F.FINAL_REL / F.REPORT_FILE).read_text("utf-8"))
    assert written["unverified_c17"]["verified"] is False


def test_the_headline_counts_only_approved_human_gold_test_records_with_n(env):
    """AC2: the headline counts only approved human-gold test records, with n."""
    out = _run(env, _models(env, []))
    report = out["report"]
    assert report["headline_n"] == {"records": 12, "items": 12}
    assert report["c17"]["n_records"] == 12
    for build in ("bf16", "quant"):
        assert all(b["n"] == 12 for b in report["c17"]["builds"][build]["bars"][:2])
    # held-out records were never approved: they never count as human gold
    held = report["sides"]["held_out"]["pools"]
    assert held[R.HEADLINE]["records"] == 0
    assert held[R.UNAPPROVED]["records"] == 6
    assert report["prereg"]["sha256"] == dp.sha256_file(env["prereg"])


def test_the_probe_set_is_scored_reported_and_its_ids_hashed(env, tmp_path):
    """AC3: the Clef probe-set table is reported with no pass/fail; probe ids stay private."""
    probe = tmp_path / "probe.jsonl"
    rows = [_rec(f"probe-secret-{i}", "train").to_json() for i in range(4)]
    probe.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models = _models(env, [])
    out = _run(env, models, inputs=F.FinalInputs(probe=probe))
    assert len(models["student"].client.asked) == 18 + 4
    table = out["report"]["probe"]
    assert table["measured"] is True
    assert table["items"] == 4
    assert "result" not in table
    assert out["final"]["probe"] == {"count": 4, "sha256": dp.sha256_file(probe)}
    final_dir = env["run_dir"] / F.FINAL_REL
    sealed_rows = json.loads((final_dir / "predictions" / "student.json").read_text())
    assert len(sealed_rows) == 18
    probe_rows = json.loads((final_dir / "predictions" / "probe" / "student.json").read_text())
    assert len(probe_rows) == 4
    assert all(r["record_id"].startswith("sha256:") for r in probe_rows)
    for path in final_dir.rglob("*.json"):
        assert "probe-secret" not in path.read_text()


def test_a_probe_record_with_a_sealed_id_is_refused(env, tmp_path):
    probe = tmp_path / "probe.jsonl"
    probe.write_text(json.dumps(_rec("t0", "train").to_json()) + "\n", encoding="utf-8")
    models = _models(env, [])
    inputs = F.FinalInputs(probe=probe)
    with pytest.raises(CliError) as exc:
        _run(env, models, inputs=inputs)
    assert "t0" not in exc.value.message


def test_clefs_client_carries_the_key_and_the_model_header(monkeypatch):
    """benchmark and final ask Clef as teacher-label does; a bare client gets a 401."""
    from types import SimpleNamespace

    from jev_factory.distill import pipeline as dpipe

    cfg = {
        "distill_teacher_url": "http://127.0.0.1:18090",
        "distill_teacher_model": "decider:general",
        "distill_teacher_key_env": "JEV_TEST_TEACHER_KEY",
        "distill_teacher_model_header": "decider:general",
    }
    monkeypatch.setenv("JEV_TEST_TEACHER_KEY", "k")
    config = SimpleNamespace(get=cfg.get, require=cfg.__getitem__)
    client = dpipe._clef_client(SimpleNamespace(config=config))
    assert client.auth_env == "JEV_TEST_TEACHER_KEY"
    assert client.model_header == "decider:general"
    assert client._key == "k"


def test_runtime_counts_the_sealed_sides_and_the_board_reports_its_own_memory(env, tmp_path):
    """Review of PR #21: probe-set calls are counted apart from the sealed-side runtime, and
    the pre-served quant's memory comes from its own /v1/stats, not this box's GPU."""
    probe = tmp_path / "probe.jsonl"
    rows = [_rec(f"probe-{i}", "train").to_json() for i in range(4)]
    probe.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models = _models(env, env["log"])
    inputs = F.FinalInputs(probe=probe)
    out = _run(env, models, inputs=inputs)
    runtime = out["final"]["runtime"]
    for name in F.MODELS:
        assert runtime[name]["n_records"] == 18  # 12 test + 6 held-out
        assert runtime[name]["probe_records"] == 4
        assert runtime[name]["latency"]["n"] == 18
    assert runtime[R.STUDENT]["memory"]["probe"] == "v1/stats"
    assert runtime[R.STUDENT]["memory"]["rss_bytes_after"] == 700_000_000


# --------------------------------------------------------------------- roles and formats (t9)


def test_the_deployed_url_is_role_neutral_and_distill_agx_url_is_an_alias():
    """c9/h8: distill_deployed_url, with the old distill_agx_url still working."""
    url = "http://127.0.0.1:18071"
    assert F.DEPLOYED_URL_KEY == "distill_deployed_url"
    assert F.deployed_url({"distill_deployed_url": url}) == url
    assert F.deployed_url({"distill_agx_url": url}) == url
    assert F.deployed_url({"distill_deployed_url": url, "distill_agx_url": url}) == url
    assert F.deployed_url({"distill_deployed_url": None, "distill_agx_url": url}) == url
    with pytest.raises(CliError) as exc:  # two different servers: never guess
        F.deployed_url({"distill_deployed_url": url, "distill_agx_url": "http://127.0.0.1:1"})
    assert exc.value.code == EXIT_USER_ERROR
    with pytest.raises(CliError) as exc:
        F.deployed_url({})
    assert "distill_deployed_url" in exc.value.remediation
    assert F.deployed_url({}, required=False) is None


def test_final_json_shows_the_clef_variant_beside_the_teacher_row(env):
    out = _run(env, _models(env, []))
    clef = out["final"]["models"][R.CLEF]
    assert clef["teacher_variant"] == "W4A16"
    assert out["final"]["teacher"] == {"row": R.CLEF, "variant": "W4A16", "model": "Clef 27B"}
    assert out["report"]["teacher"]["variant"] == "W4A16"


@pytest.fixture
def env_v3_roles_absent(tmp_path):
    """A v3 quant run with the deployed server, edge board and alt teacher declared absent."""
    doc = dp.example_doc_v3(teacher_variant="BF16")
    for name in dp.OPTIONAL_ROLES:
        doc["roles"][name] = {"present": False}
    yield from _quant_env(tmp_path, doc)


def test_any_attested_loopback_server_is_accepted_without_a_deployed_server_role(
    env_v3_roles_absent,
):
    """c9: the deployed quant need not be on a board; /v1/bundle and a cap still gate it."""
    env = env_v3_roles_absent
    out = _run(env, _models(env, env["log"]))
    assert out["final"]["student_server"]["memory_cap"]["effective_memory_max"] == CAP
    assert out["final"]["models"][R.CLEF]["teacher_variant"] == "BF16"
    assert out["report"]["roles"]["deployed_server"]["present"] is False
    assert out["report"]["c17"]["builds"]["quant"]["bars"]
    # not on a board: this box's memory probe measures it, not the board's /v1/stats
    assert out["final"]["runtime"][R.STUDENT]["memory"].get("probe") != "v1/stats"


BF16_WINNER = "c003"
BF16_CAL = {"temperature_by_kind": {"choice": 1.2}, "vector_by_kind": {}}


def _bf16_build(run_dir: Path) -> Path:
    """The bf16 deployed contract: select/summary.json winner -> train/<winner>/bundle."""
    select = run_dir / "distill" / "select"
    select.mkdir(parents=True, exist_ok=True)
    (select / "summary.json").write_text(json.dumps({"winner": BF16_WINNER}), encoding="utf-8")
    (select / "calibration.json").write_text(json.dumps(BF16_CAL), encoding="utf-8")
    cand = run_dir / "distill" / "train" / BF16_WINNER
    bundle = cand / "bundle"
    (bundle / "lora").mkdir(parents=True)
    (bundle / player.HEAD).write_bytes(b"bf16-head")
    (bundle / player.ADAPTER).write_bytes(b"bf16-lora")
    (bundle / player.CONFIG).write_text("{}", encoding="utf-8")
    (bundle / player.MANIFEST).write_text("sha lines\n", encoding="utf-8")
    (cand / "predictions.json").write_text(json.dumps({"selection": []}), encoding="utf-8")
    return bundle


@pytest.fixture
def bf16_env(tmp_path):
    doc = dp.example_doc_v3(deployed_format="bf16")
    run_dir, prereg, data, sides, ledger = _locked_run(tmp_path, doc)
    bundle = _bf16_build(run_dir)
    return {
        "run_dir": run_dir,
        "prereg": prereg,
        "data": data,
        "sides": sides,
        "ledger": ledger,
        "bundle": bundle,
        "log": [],
    }


def _bf16_models(env, log, served_from=None):
    p = {"clef": 0.9, "stock": 0.6, "student_bf16": 0.8}
    out = {}
    for name in (R.CLEF, R.STOCK, R.STUDENT_BF16):
        target = bench.Target(
            name,
            "http://127.0.0.1:9",
            name,
            start=lambda n=name: log.append(f"start {n}"),
            stop=lambda n=name: log.append(f"stop {n}"),
        )
        extra = {}
        if name == R.STUDENT_BF16:
            extra["served_from"] = served_from or env["bundle"]
        out[name] = F.FinalModel(target=target, client=FakeClient(name, p[name], log), **extra)
    return out


def test_a_bf16_ship_measures_the_attested_bf16_build_once_as_the_deployed_student(bf16_env):
    """c7: final's model set follows the declared format; no quant manifest is needed."""
    env = bf16_env
    models = _bf16_models(env, env["log"])
    plan = _run(env, models, apply=False)
    assert plan["deployed"]["format"] == "bf16"
    assert plan["deployed"]["model"] == R.STUDENT_BF16
    assert set(plan["models"]) == {R.CLEF, R.STOCK, R.STUDENT_BF16}
    out = _run(env, models)
    every = env["sides"]["test"] + env["sides"]["held_out"]
    assert sorted(models[R.STUDENT_BF16].client.asked) == sorted(every)  # measured once
    final = out["final"]
    assert set(final["runtime"]) == {R.CLEF, R.STOCK, R.STUDENT_BF16}
    bundle = final["deployed"]["bundle"]
    assert bundle["files"][player.HEAD] == dp.sha256_file(env["bundle"] / player.HEAD)
    assert final["deployed"]["winner"] == BF16_WINNER
    server = final["student_server"]
    assert server["attestation"].startswith("file sha256")
    assert server["served_from"] == str(env["bundle"])
    cal = final["calibrations"][R.STUDENT]
    assert cal["source"] == "distill/select/calibration.json"
    assert cal["calibration"] == BF16_CAL
    assert final["calibrations"][R.STUDENT_BF16]["sha256"] == cal["sha256"]
    rep = out["report"]
    assert rep["deployed_format"] == "bf16"
    assert rep["c17"]["builds"]["bf16"]["result"] in (R.PASSED, R.FAILED)
    assert rep["c17"]["builds"]["quant"]["result"] == R.NOT_APPLICABLE
    assert rep["c43"]["result"] == R.NOT_APPLICABLE
    written = env["run_dir"] / F.FINAL_REL / "predictions"
    assert sorted(p.name for p in written.glob("*.json")) == [
        "clef.json",
        "stock.json",
        "student_bf16.json",
    ]


def test_a_bf16_ship_refuses_a_separate_student_model(bf16_env):
    env = bf16_env
    models = _bf16_models(env, env["log"])
    models[R.STUDENT] = models[R.STUDENT_BF16]
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "unexpected ['student']" in exc.value.message
    assert env["log"] == []


@pytest.mark.parametrize("apply", [True, False])
def test_a_bf16_ship_refuses_a_partial_model_set(bf16_env, apply):
    env = bf16_env
    models = _bf16_models(env, env["log"])
    del models[R.STOCK]
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert "missing ['stock']" in exc.value.message


def test_a_bf16_server_started_from_another_bundle_is_refused_before_any_request(
    bf16_env, tmp_path
):
    env = bf16_env
    other = tmp_path / "other"
    import shutil

    shutil.copytree(env["bundle"], other)
    (other / player.HEAD).write_bytes(b"swapped")
    models = _bf16_models(env, env["log"], served_from=other)
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "bundle" in exc.value.message
    assert not [e for e in env["log"] if e.startswith("ask")]
    assert _ledger(env) == []


def test_a_bf16_server_with_no_served_from_cannot_be_attested(bf16_env):
    env = bf16_env
    models = _bf16_models(env, env["log"])
    models[R.STUDENT_BF16] = F.FinalModel(
        target=models[R.STUDENT_BF16].target, client=models[R.STUDENT_BF16].client
    )
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "served_from" in exc.value.message
    assert not [e for e in env["log"] if e.startswith("ask")]


def test_a_bf16_ship_with_no_winner_is_refused(bf16_env):
    env = bf16_env
    (env["run_dir"] / "distill" / "select" / "summary.json").write_text(
        json.dumps({"winner": None}), encoding="utf-8"
    )
    with pytest.raises(CliError) as exc:
        _run(env, _bf16_models(env, env["log"]), apply=False)
    assert "winner" in exc.value.message


def test_a_bf16_ship_refuses_a_bf16_calibration_other_than_the_deployed_one(bf16_env, tmp_path):
    env = bf16_env
    other = tmp_path / "cal.json"
    other.write_text(json.dumps({"temperature_by_kind": {"choice": 9.0}}), encoding="utf-8")
    with pytest.raises(CliError) as exc:
        _run(
            env,
            _bf16_models(env, env["log"]),
            apply=False,
            inputs=F.FinalInputs(bf16_calibration=other),
        )
    assert "calibration" in exc.value.message
