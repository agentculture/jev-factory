"""Distill final measurement (t20): Clef, stock v21 and the deployed student, measured once.

Fake clients and targets stand in for the servers: no GPU, no network, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import benchmark as bench
from jev_factory.distill import dataset
from jev_factory.distill import final as F
from jev_factory.distill import prereg as dp
from jev_factory.distill import report as R
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.distill.systemone import CallResult, QuestionResult
from jev_factory.measure import once

CLEAN = "Praveenrajus/jev-bench:scifact"
NOW = "2026-10-14T00:00:00+00:00"
STUDENT_CAL = {"temperature_by_kind": {"choice": 1.5}, "vector_by_kind": {}}


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


class FakeClient:
    """Answers ``a`` with *p_a*; logs every record id it was asked."""

    def __init__(self, name: str, p_a: float):
        self.name, self.p_a, self.asked = name, p_a, []

    def ask(self, record):
        self.asked.append(record["id"])
        dist = {"a": self.p_a, "b": round(1.0 - self.p_a, 9)}
        return CallResult(
            str(record["id"]), self.name, {}, "sha", [QuestionResult("c", "choice", dist)]
        )


@pytest.fixture
def env(tmp_path):
    run_dir = tmp_path / "run"
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(dp.example_doc()), encoding="utf-8")
    dp.lock(prereg, run_dir)
    data = tmp_path / "data"
    data.mkdir()
    sides = {"test": [f"t{i}" for i in range(12)], "held_out": [f"h{i}" for i in range(6)]}
    for side, ids in sides.items():
        lines = [json.dumps(_rec(i, side).to_json(), sort_keys=True) for i in ids]
        (data / f"{side}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    qdir = run_dir / "distill" / "quant"
    qdir.mkdir(parents=True)
    cal = qdir / "calibration.json"
    cal.write_text(json.dumps({**STUDENT_CAL, "format": "bf16"}), encoding="utf-8")
    (qdir / "format_decision.json").write_text(
        json.dumps({"decision": "deployed_format", "format": "bf16"}), encoding="utf-8"
    )
    (qdir / "quant_manifest.json").write_text(
        json.dumps(
            {
                "status": "done",
                "chosen_format": "bf16",
                "deployed_artifact": "artifact",
                "heal_count": 0,
                "quant_vs_bf16_delta_points": 0.0,
                "deployed_selection_ece": 0.05,
                "outputs": {str(cal): dp.sha256_file(cal)},
            }
        ),
        encoding="utf-8",
    )
    return {"run_dir": run_dir, "prereg": prereg, "data": data, "sides": sides, "qdir": qdir}


def _models(log, *, fail_start=None, p=None):
    p = p or {"clef": 0.9, "stock_v21": 0.6, "student": 0.8}
    out = {}
    for name in F.MODELS:

        def start(name=name):
            if name == fail_start:
                raise RuntimeError(f"{name} failed to start")
            log.append(f"start {name}")

        target = bench.Target(
            name,
            "http://127.0.0.1:9",
            name,
            start=start,
            stop=lambda n=name: log.append(f"stop {n}"),
        )
        out[name] = F.FinalModel(target=target, client=FakeClient(name, p[name]))
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
        resamples=50,
        **kw,
    )


def _ledger(env):
    return once.OnceLedger(env["run_dir"]).records()


def test_dry_run_by_default_scores_nothing_and_writes_no_ledger(env):
    log = []
    models = _models(log)
    plan = _run(env, models, apply=False)
    assert plan["dry_run"] is True
    assert plan["sides"]["test"]["count"] == 12
    assert plan["sides"]["held_out"]["count"] == 6
    assert log == []
    assert all(not m.client.asked for m in models.values())
    assert _ledger(env) == []


def test_one_ledgered_invocation_scores_all_three_on_test_and_held_out(env):
    log = []
    models = _models(log)
    out = _run(env, models)
    every = env["sides"]["test"] + env["sides"]["held_out"]
    for m in models.values():
        assert sorted(m.client.asked) == sorted(every)
    assert log == [f"{verb} {n}" for n in F.MODELS for verb in ("start", "stop")]  # one at a time
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
    assert written["c42"]["bars"]
    assert written["c43"]["bars"]
    assert out["report"]["result"] in (R.PASSED, R.MISSED)
    assert set(out["final"]["runtime"]) == set(F.MODELS)


@pytest.mark.parametrize("apply", [True, False])
def test_a_second_final_is_refused_without_a_deviation_id(env, apply):
    _run(env, _models([]))
    log = []
    models = _models(log)
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert exc.value.code == EXIT_USER_ERROR != 0
    assert "deviation" in exc.value.remediation
    assert log == []
    assert all(not m.client.asked for m in models.values())
    assert len(_ledger(env)) == 2


def test_a_final_for_only_some_of_the_models_is_refused(env):
    models = _models([])
    del models["stock_v21"], models["clef"]
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert exc.value.code != 0
    assert "missing ['clef', 'stock_v21']" in exc.value.message
    assert _ledger(env) == []
    _run(env, _models([]))
    student_only = {"student": _models([])["student"]}
    with pytest.raises(CliError):  # a later student-only (or any) final is refused too
        _run(env, student_only)


def test_a_second_final_with_a_deviation_id_runs_and_is_recorded(env):
    _run(env, _models([]))
    _run(env, _models([]), deviation_id="d21")
    records = _ledger(env)
    assert len(records) == 4
    assert [r.deviation for r in records[2:]] == ["d21", "d21"]


def test_a_bad_deviation_id_is_refused(env):
    models = _models([])
    with pytest.raises(CliError):
        _run(env, models, deviation_id="not a deviation id!")


def test_a_run_whose_first_start_fails_writes_no_ledger_record(env):
    models = _models([], fail_start="clef")
    with pytest.raises(RuntimeError):
        _run(env, models)
    assert _ledger(env) == []
    _run(env, _models([]))  # the real run is not blocked
    assert len(_ledger(env)) == 2


def test_a_run_that_fails_after_scoring_is_still_ledgered(env):
    models = _models([], fail_start="student")
    with pytest.raises(RuntimeError):
        _run(env, models)
    statuses = {r.status for r in _ledger(env)}
    assert statuses == {once.FAILED_MID_RUN}
    models = _models([])
    with pytest.raises(CliError):
        _run(env, models)


def test_the_calibration_each_row_used_is_recorded(env, tmp_path):
    clef_cal = tmp_path / "clef_cal.json"
    clef_cal.write_text(json.dumps({"temperature_by_kind": {"choice": 2.0}}), encoding="utf-8")
    out = _run(env, _models([]), clef_calibration=clef_cal)
    cals = out["final"]["calibrations"]
    assert cals["student"]["source"] == F.STUDENT_CALIBRATION_REL
    assert cals["student"]["sha256"] == dp.sha256_file(env["qdir"] / "calibration.json")
    assert cals["student"]["calibration"]["temperature_by_kind"] == {"choice": 1.5}
    assert cals["stock_v21"]["source"] == F.CAL_SERVED
    assert cals["stock_v21"]["calibration"] is None
    assert cals["clef"]["source"] == F.CAL_RAW
    assert cals["clef"]["calibration"] is None
    assert cals[R.CLEF_CALIBRATED]["sha256"] == dp.sha256_file(clef_cal)
    assert out["final"]["deployed"]["format"] == "bf16"
    stats = out["report"]["sides"]["test"]["pools"][R.HEADLINE]["stats"]
    assert "clef_calibrated.ece" in stats


def test_the_student_needs_the_quant_calibration_it_was_shipped_with(env):
    (env["qdir"] / "calibration.json").unlink()
    models = _models([])
    with pytest.raises(CliError):
        _run(env, models)
    assert _ledger(env) == []


def test_a_changed_quant_calibration_is_refused(env):
    (env["qdir"] / "calibration.json").write_text(json.dumps(STUDENT_CAL), encoding="utf-8")
    models = _models([])
    with pytest.raises(CliError) as exc:
        _run(env, models)
    assert "sha256" in exc.value.message


def test_an_unlocked_prereg_is_refused(env):
    (env["run_dir"] / dp.LOCK_REL).unlink()
    models = _models([])
    with pytest.raises(CliError):
        _run(env, models)


def test_runtime_is_recorded_for_all_three_on_the_same_host(env):
    out = _run(env, _models([]))
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
    models = _models([])
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
    plan = _run(env, _models([]), apply=False)
    assert plan["gate"]["sha256"] == dp.sha256_file(path)
    out = _run(env, _models([]))
    dg = out["report"]["decision_gate"]
    assert dg["fitted"] is True  # reported only: "gated" stays False (never a bar)
    assert dg["thresholds_by_kind"] == GATE_THRESHOLDS
    assert set(dg["sides"]) == {"test", "held_out"}
    assert out["final"]["gate"]["sha256"] == dp.sha256_file(path)


def test_without_a_gate_the_report_says_none_was_fitted(env):
    out = _run(env, _models([]))
    assert out["report"]["decision_gate"]["fitted"] is False
    assert out["final"]["gate"] is None


@pytest.mark.parametrize("apply", [True, False])
def test_a_gate_fitted_on_another_calibration_is_refused(env, apply):
    _write_gate(env, calibration_sha256="d" * 64)
    models = _models([])
    with pytest.raises(CliError) as exc:
        _run(env, models, apply=apply)
    assert "calibration" in exc.value.message
    assert _ledger(env) == []


def test_an_unfitted_gate_file_is_refused(env):
    _write_gate(env, fitted=False)
    models = _models([])
    with pytest.raises(CliError):
        _run(env, models)
    assert _ledger(env) == []
