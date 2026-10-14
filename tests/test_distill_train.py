"""Distill train, select and calibrate stage (t18).

The knob grid runs through the pointer trainer under the GPU guard (faked here: no GPU, no
torch, no server), each candidate is scored on the fit and selection folds by an injected
scorer, calibration is fitted on the fit fold with ``core.calibration`` and the winner is
selected on the selection fold by the pre-registered rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.core import calibration as cal
from jev_factory.distill import pins as dpins
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as T
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.factory import gpu

REV = "a" * 40
V21_TEMPS = {"noul": 0.8177019773481682, "choice": 0.8177019773481682, "score": 1.192199469079091}

# --------------------------------------------------------------------- fixtures


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    base = root / dpins.cache_dirname(dpins.BASE_REPO)
    base.mkdir(parents=True)
    (base / dpins.REVISION_FILE).write_text(REV + "\n")
    dec = root / dpins.cache_dirname(dpins.DECIDER_REPO)
    (dec / "lora").mkdir(parents=True)
    (dec / dpins.REVISION_FILE).write_text(REV + "\n")
    (dec / dpins.ADAPTER_FILE).write_bytes(b"adapter")
    (dec / dpins.HEAD_FILE).write_bytes(b"head")
    (dec / dpins.PROVENANCE_FILE).write_text(
        json.dumps({"adapter_sha256": _sha(b"adapter"), "head_sha256": _sha(b"head")})
    )
    # v21's own calibration, which must never become the deployed one.
    (dec / "strands_decider_config.json").write_text(
        json.dumps({"head_type": "pointer", "temperature": 0.93, "temperature_by_kind": V21_TEMPS})
    )
    return root


def student_pins() -> dpins.StudentPins:
    return dpins.StudentPins(
        base=dpins.RepoPin(dpins.BASE_REPO, REV),
        decider=dpins.RepoPin(dpins.DECIDER_REPO, REV),
        adapter_sha256=_sha(b"adapter"),
        head_sha256=_sha(b"head"),
    )


def small_doc(**grid):
    doc = dp.example_doc()
    doc["knob_grid"].update(
        {
            "alpha": [0.5, 0.9],
            "temperature": {"choice": [2.0], "score": [1.0], "noul": [1.0]},
            "epochs": [1],
            "lr": [1e-4],
            "init_mode": ["continue"],
        }
    )
    doc["knob_grid"].update(grid)
    return doc


def locked(tmp_path, doc=None, lock=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(doc or small_doc()), encoding="utf-8")
    run_dir = tmp_path / "run"
    if lock:
        dp.lock(path, run_dir)
    return path, run_dir


def _rec(i: int, split: str = "train") -> Record:
    return Record(
        id=f"r{i}",
        state=f"state {i}",
        questions={
            "c": Question("choice", [("fix", "fix it"), ("wait", "wait")]),
            "s": Question("score", [("0", "low"), ("1", "mid"), ("2", "high")]),
            "n": Question("noul", [("false", ""), ("true", "")]),
        },
        gold={
            "c": Gold(label="fix" if i % 3 else "wait"),
            "s": Gold(label=str(i % 3)),
            "n": Gold(label="true", probability_true=1.0 if i % 2 else 0.0),
        },
        split=split,
        source="src",
        licence="mit",
        gold_provenance="human",
    )


def folds():
    return {"fit": [_rec(i) for i in range(0, 20)], "selection": [_rec(i) for i in range(20, 40)]}


def _answer(rec: Record, p_right: float, right: bool) -> dict:
    qs = {}
    for qid, q in rec.questions.items():
        names = q.names
        gold = rec.gold[qid]
        if q.type == "noul":
            truth = gold.probability_true >= 0.5
            p_true = p_right if truth == right else 1.0 - p_right
            qs[qid] = {"kind": "noul", "distribution": None, "p_true": p_true, "error": None}
            continue
        g = names.index(gold.label)
        target = g if right else (g + 1) % len(names)
        rest = (1.0 - p_right) / (len(names) - 1)
        dist = {n: (p_right if k == target else rest) for k, n in enumerate(names)}
        qs[qid] = {"kind": q.type, "distribution": dist, "p_true": None, "error": None}
    return qs


class FakeScorer:
    """Overconfident answers; alpha 0.9 is right on 90% of records, alpha 0.5 on 60%."""

    def __init__(self, p_right=0.97):
        self.calls = []
        self.p_right = p_right

    def __call__(self, candidate, checkpoint, fold_records):
        self.calls.append((candidate.cid, sorted(fold_records)))
        acc = 0.9 if candidate.knobs["alpha"] >= 0.9 else 0.6
        out = {}
        for fold, recs in fold_records.items():
            rows = []
            for k, rec in enumerate(recs):
                right = (k % 10) < round(acc * 10)
                rows.append(
                    {
                        "fold": fold,
                        "record_id": rec.id,
                        "error": None,
                        "questions": _answer(rec, self.p_right, right),
                    }
                )
            out[fold] = rows
        return out


def fake_gpu(calls, config=None):
    def run_gpu_stage(stage, run_dir, cmd, **kw):
        calls.append((stage, list(cmd), kw))
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        cfg = config or {"head_type": "pointer", "temperature": 1.0, "temperature_by_kind": {}}
        (out / "strands_decider_config.json").write_text(json.dumps(cfg))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    return run_gpu_stage


def run(tmp_path, cache, *, doc=None, apply=True, scorer=None, train=True, **kw):
    path, run_dir = locked(tmp_path, doc)
    train_file = run_dir / "distill" / "assemble" / "train_set.jsonl"
    if train:
        train_file.parent.mkdir(parents=True, exist_ok=True)
        train_file.write_text("{}\n")
    args = dict(
        prereg_path=path,
        run_dir=run_dir,
        cache_dir=cache,
        pins=student_pins(),
        folds=folds(),
        trainer_python="python3",
        decider_repo=tmp_path / "strands-decider",
        apply=apply,
        scorer=scorer if scorer is not None else FakeScorer(),
        memory_max="100G",
        resamples=50,
    )
    args.update(kw)
    return T.run_grid(**args), run_dir


# --------------------------------------------------------------------- refusals


def test_refuses_without_a_locked_prereg(tmp_path, cache):
    path, run_dir = locked(tmp_path, lock=False)
    with pytest.raises(CliError, match="locked"):
        T.run_grid(
            prereg_path=path,
            run_dir=run_dir,
            cache_dir=cache,
            pins=student_pins(),
            folds=folds(),
            trainer_python="python3",
            decider_repo=tmp_path,
        )


def test_refuses_without_verified_pins(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("trained"))
    with pytest.raises(CliError, match="pins"):
        run(tmp_path / "a", cache, pins=None)
    head = cache / dpins.cache_dirname(dpins.DECIDER_REPO) / dpins.HEAD_FILE
    head.write_bytes(b"tampered")
    with pytest.raises(CliError, match="hash-mismatch"):
        run(tmp_path / "b", cache)


def test_refuses_held_out_and_test_folds_before_training(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("trained"))
    with pytest.raises(CliError):
        run(tmp_path / "a", cache, folds={**folds(), "held_out": [_rec(99)]})
    with pytest.raises(CliError, match="test"):
        run(tmp_path / "b", cache, folds={"fit": [_rec(1, "test")], "selection": [_rec(2)]})
    sealed = tmp_path / "test.jsonl"
    sealed.write_text(json.dumps(_rec(3).to_json()) + "\n")
    with pytest.raises(CliError, match="sealed"):
        run(tmp_path / "c", cache, folds={"fit": sealed, "selection": [_rec(2)]})


def test_both_folds_are_required(tmp_path, cache):
    with pytest.raises(CliError, match="selection"):
        run(tmp_path, cache, folds={"fit": [_rec(1)]})


def test_apply_refuses_without_memory_cap_scorer_or_train_set(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("trained"))
    with pytest.raises(CliError, match="memory"):
        run(tmp_path / "a", cache, memory_max=None)
    with pytest.raises(CliError, match="scorer"):
        T.run_grid(
            prereg_path=locked(tmp_path / "b")[0],
            run_dir=tmp_path / "b" / "run",
            cache_dir=cache,
            pins=student_pins(),
            folds=folds(),
            trainer_python="python3",
            decider_repo=tmp_path,
            apply=True,
            memory_max="100G",
        )
    with pytest.raises(CliError, match="train_set"):
        run(tmp_path / "c", cache, train=False)


# --------------------------------------------------------------------- the grid


def test_grid_candidates_cover_the_registered_grid():
    reg = dp.validate(dp.example_doc())
    cands = T.grid_candidates(reg)
    grid = reg.knob_grid
    temps = math.prod(len(v) for v in grid["temperature"].values())
    expected = (
        len(grid["alpha"]) * temps * len(grid["epochs"]) * len(grid["lr"]) * len(grid["init_mode"])
    )
    assert len(cands) == expected
    assert len({c.cid for c in cands}) == expected
    assert len({json.dumps(c.knobs, sort_keys=True) for c in cands}) == expected
    for c in cands:
        dp.require_knobs(reg, c.knobs)
        assert set(c.knobs) == {"alpha", "temperature", "epochs", "lr", "init_mode"}


def test_dry_run_plans_every_candidate_and_runs_nothing(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("ran on dry-run"))
    scorer = FakeScorer()
    out, run_dir = run(tmp_path, cache, apply=False, scorer=scorer, train=False)
    assert out["dry_run"] is True
    assert [c["cid"] for c in out["candidates"]] == ["c000", "c001"]
    assert out["pins"]["status"] == "ok"
    init = cache / dpins.cache_dirname(dpins.DECIDER_REPO)
    for c in out["candidates"]:
        assert c["plan"]["init_from"] == str(init)
        assert c["plan"]["argv"][c["plan"]["argv"].index("--output-dir") + 1].startswith(
            str(run_dir / "distill" / "train" / c["cid"])
        )
    assert scorer.calls == []
    assert not (run_dir / "distill" / "select").exists()


def test_apply_trains_each_candidate_under_the_gpu_guard(tmp_path, cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu(calls))
    out, _ = run(tmp_path, cache)
    assert [c[0] for c in calls] == ["train", "train"]
    assert all(c[2]["memory_max"] == "100G" for c in calls)
    knobs = [json.loads(c[1][c[1].index("--knobs") + 1]) for c in calls]
    assert [k["alpha"] for k in knobs] == [0.5, 0.9]
    assert out["dry_run"] is False


def test_scorer_sees_only_the_fit_and_selection_folds(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))
    scorer = FakeScorer()
    run(tmp_path, cache, scorer=scorer)
    assert scorer.calls and all(f == ["fit", "selection"] for _, f in scorer.calls)


# --------------------------------------------------------------------- selection


def _score(cid, accuracy, ece, cw, safe=True):
    return {"cid": cid, "safe": safe, "accuracy": accuracy, "ece": ece, "confident_wrong": cw}


RULE = {
    "order": list(dp.SELECTION_ORDER),
    "tolerances": {"accuracy_floor_margin": 0.05, "ece": 0.01},
}


def test_rule_applies_safety_floor_calibration_confident_wrong_then_ties():
    scores = [
        _score("c000", 0.90, 0.08, 0.05),
        _score("c001", 0.87, 0.02, 0.06),
        _score("c002", 0.80, 0.00, 0.00),  # below the accuracy floor (0.90 - 0.05)
        _score("c003", 0.99, 0.00, 0.00, safe=False),
    ]
    winner, trail = T.apply_rule(RULE, scores)
    assert winner == "c001"
    steps = {s["step"]: s["survivors"] for s in trail}
    assert steps["safety"] == ["c000", "c001", "c002"]
    assert steps["accuracy_floor"] == ["c000", "c001"]
    assert steps["calibration"] == ["c001"]
    scores.append(_score("c004", 0.86, 0.025, 0.03))
    assert T.apply_rule(RULE, scores)[0] == "c004"  # same ECE band, fewer confident-wrong


def test_rule_ties_go_to_accuracy_then_grid_order():
    scores = [_score("c000", 0.86, 0.02, 0.0), _score("c001", 0.88, 0.02, 0.0)]
    assert T.apply_rule(RULE, scores)[0] == "c001"
    scores = [_score("c000", 0.88, 0.02, 0.0), _score("c001", 0.88, 0.02, 0.0)]
    assert T.apply_rule(RULE, scores)[0] == "c000"


def test_rule_with_no_safe_candidate_selects_none():
    assert T.apply_rule(RULE, [_score("c000", 0.9, 0.0, 0.0, safe=False)])[0] is None


def test_selection_is_on_the_selection_fold(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))

    class Flipped(FakeScorer):
        """alpha 0.5 is better on the fit fold, alpha 0.9 on the selection fold."""

        def __call__(self, candidate, checkpoint, fold_records):
            out = {}
            for fold, recs in fold_records.items():
                good = (candidate.knobs["alpha"] >= 0.9) == (fold == "selection")
                acc = 0.9 if good else 0.5
                out[fold] = [
                    {
                        "fold": fold,
                        "record_id": r.id,
                        "error": None,
                        "questions": _answer(r, 0.8, (k % 10) < round(acc * 10)),
                    }
                    for k, r in enumerate(recs)
                ]
            return out

    out, run_dir = run(tmp_path, cache, scorer=Flipped())
    summary = json.loads((run_dir / "distill" / "select" / "summary.json").read_text())
    assert summary["winner"] == "c001" == out["winner"]
    by = {c["cid"]: c for c in summary["candidates"]}
    assert by["c001"]["knobs"]["alpha"] == 0.9
    assert by["c001"]["selection_metrics"]["accuracy"]["all"]["value"] > (
        by["c000"]["selection_metrics"]["accuracy"]["all"]["value"]
    )


def test_summary_names_winner_metrics_with_cis_and_prereg_sha(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))
    out, run_dir = run(tmp_path, cache)
    summary = json.loads((run_dir / "distill" / "select" / "summary.json").read_text())
    assert summary["winner"] == "c001"
    assert summary["prereg_sha256"] == dp.sha256_file(tmp_path / "prereg.json")
    assert summary["fold"] == "selection"
    assert [s["step"] for s in summary["rule_trail"]] == list(dp.SELECTION_ORDER)
    for cand in summary["candidates"]:
        m = cand["selection_metrics"]
        for name in ("accuracy", "macro_f1", "confident_wrong", "log_loss", "brier", "ece"):
            block = m[name]["all"]
            assert {"value", "ci_low", "ci_high", "n"} <= set(block), name
        assert m["confident_wrong_threshold"] == 0.9
        assert cand["calibration"]["temperature_by_kind"]
        assert cand["safe"] is True


def test_unsafe_candidate_with_scoring_errors_is_never_selected(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))

    class Broken(FakeScorer):
        def __call__(self, candidate, checkpoint, fold_records):
            out = super().__call__(candidate, checkpoint, fold_records)
            if candidate.knobs["alpha"] >= 0.9:
                out["selection"][0]["questions"]["c"]["error"] = "timeout"
            return out

    out, _ = run(tmp_path, cache, scorer=Broken())
    assert out["winner"] == "c000"


def test_missing_confident_wrong_threshold_is_refused(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("trained"))
    doc = small_doc()
    del doc["selection_rule"]["confident_wrong_threshold"]
    with pytest.raises(CliError, match="confident_wrong_threshold"):
        run(tmp_path, cache, doc=doc)


# --------------------------------------------------------------------- calibration


def _noul_rows(p_true, golds):
    return [({"false": 1.0 - p_true, "true": p_true}, g) for g in golds]


def test_temperature_is_fitted_on_the_fit_fold_only():
    fit = {"choice": [({"a": 0.95, "b": 0.05}, "a" if i % 3 else "b") for i in range(30)]}
    sel_a = {"choice": [({"a": 0.9, "b": 0.1}, "a") for _ in range(10)]}
    sel_b = {"choice": [({"a": 0.6, "b": 0.4}, "b") for _ in range(10)]}
    one = T.fit_calibration(fit, sel_a)
    two = T.fit_calibration(fit, sel_b)
    want = cal.fit_temperature(fit["choice"])
    assert one["temperature_by_kind"]["choice"] == pytest.approx(want)
    assert two["temperature_by_kind"]["choice"] == pytest.approx(want)
    assert want > 1.0  # the overconfident fit fold is softened
    assert one["by_kind"]["choice"]["fit_examples"] == 30


def test_vector_is_kept_only_if_it_lowers_selection_fold_ece():
    # The model says P(true) = 0.3 but 70% of fit-fold answers are true: a label bias that
    # temperature cannot fix and the per-label vector can.
    fit = {"noul": _noul_rows(0.3, ["true"] * 7 + ["false"] * 3)}
    same = {"noul": _noul_rows(0.3, ["true"] * 7 + ["false"] * 3)}
    opposite = {"noul": _noul_rows(0.3, ["true"] * 3 + ["false"] * 7)}
    kept = T.fit_calibration(fit, same)
    assert kept["by_kind"]["noul"]["vector_kept"] is True
    assert kept["vector_by_kind"]["noul"]
    sel = kept["by_kind"]["noul"]["selection_ece"]
    assert sel["temperature+vector"] < sel["temperature"]
    dropped = T.fit_calibration(fit, opposite)
    assert dropped["by_kind"]["noul"]["vector_kept"] is False
    assert "noul" not in dropped["vector_by_kind"]
    sel = dropped["by_kind"]["noul"]["selection_ece"]
    assert not sel["temperature+vector"] < sel["temperature"]
    # The temperature itself never depends on the selection fold.
    assert kept["temperature_by_kind"]["noul"] == dropped["temperature_by_kind"]["noul"]


def test_calibration_json_is_fitted_not_upstream(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))
    scorer = FakeScorer()
    out, run_dir = run(tmp_path, cache, scorer=scorer)
    calib = json.loads((run_dir / "distill" / "select" / "calibration.json").read_text())
    assert calib["candidate"] == out["winner"] == "c001"
    assert calib["fitted_on"] == "fit"
    assert calib["upstream_temperature_by_kind_used"] is False
    assert set(calib["temperature_by_kind"]) == {"choice", "score", "noul"}
    for kind, t in calib["temperature_by_kind"].items():
        assert not math.isclose(t, V21_TEMPS[kind], rel_tol=1e-6), kind
    # Recompute the choice temperature from the winner's fit-fold predictions by hand.
    winner = next(c for c in out["candidates"] if c["cid"] == "c001")
    recs = {r.id: r for r in folds()["fit"]}
    fit_rows = []
    preds = scorer(winner_candidate(winner), None, {"fit": folds()["fit"]})["fit"]
    for row in preds:
        q = row["questions"]["c"]
        fit_rows.append((dict(q["distribution"]), recs[row["record_id"]].gold["c"].label))
    assert calib["temperature_by_kind"]["choice"] == pytest.approx(
        cal.fit_temperature(fit_rows), rel=1e-6
    )


def winner_candidate(entry):
    return T.Candidate(entry["cid"], entry["knobs"])


def test_a_checkpoint_carrying_a_served_temperature_is_refused(tmp_path, cache, monkeypatch):
    cfg = {"head_type": "pointer", "temperature": 1.0, "temperature_by_kind": V21_TEMPS}
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([], config=cfg))
    scorer = FakeScorer()
    with pytest.raises(CliError, match="temperature"):
        run(tmp_path, cache, scorer=scorer)
    assert scorer.calls == []


def test_calibrated_selection_metrics_use_the_fitted_temperature(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", fake_gpu([]))
    out, run_dir = run(tmp_path, cache, scorer=FakeScorer(p_right=0.99))
    summary = json.loads((run_dir / "distill" / "select" / "summary.json").read_text())
    for cand in summary["candidates"]:
        raw = cand["selection_metrics_raw"]["ece"]["all"]["value"]
        calibrated = cand["selection_metrics"]["ece"]["all"]["value"]
        assert calibrated < raw  # overconfident raw answers are softened by the fit-fold T


# --------------------------------------------------------------------- served scorer


def test_served_scorer_guards_the_gpu_exports_serves_and_always_stops(tmp_path, monkeypatch):
    from jev_factory.backbones.pointer import layout as player
    from jev_factory.backbones.pointer import serve as pserve
    from jev_factory.distill.systemone import CallResult, QuestionResult

    log = []
    monkeypatch.setattr(gpu, "assert_gpu_free", lambda stage, **k: log.append(("guard", stage)))
    monkeypatch.setattr(
        player, "write_layout", lambda src, out, **k: log.append(("export", k["base_revision"]))
    )
    monkeypatch.setattr(pserve, "launch", lambda *a, **k: log.append(("launch", a[1])) or 1)
    monkeypatch.setattr(pserve, "wait_ready", lambda *a, **k: log.append(("ready",)))
    monkeypatch.setattr(T, "_stop_job", lambda jobdir, name: log.append(("stop", name)))

    class Client:
        def __init__(self, base_url, *, model):
            assert base_url.startswith("http://127.0.0.1:")

        def ask(self, record):
            assert "gold" not in record
            qs = [
                QuestionResult(qid=q, kind=v["type"], distribution=None, p_true=0.5, error=None)
                for q, v in record["questions"].items()
            ]
            return CallResult(
                record_id=record["id"],
                model="m",
                request_config={},
                response_sha256="x",
                questions=qs,
            )

    monkeypatch.setattr(pserve, "PointerClient", Client)
    score = T.served_scorer(env=object(), jobdir=tmp_path / "jobs", port=8123, base_revision=REV)
    ckpt = tmp_path / "c000" / "checkpoint"
    out = score(T.Candidate("c000", {"alpha": 0.5}), ckpt, folds())
    assert set(out) == {"fit", "selection"}
    assert [r["record_id"] for r in out["selection"]] == [r.id for r in folds()["selection"]]
    assert log[0] == ("export", REV)
    assert log[1] == ("guard", "measure")
    assert log[2] == ("launch", "distill-c000")
    assert log[-1] == ("stop", "distill-c000")


def test_vector_is_not_fitted_over_an_open_label_set():
    fit = {"choice": [({f"x{i}": 0.8, f"y{i}": 0.2}, f"x{i}") for i in range(10)]}
    out = T.fit_calibration(fit, fit)
    assert out["by_kind"]["choice"]["vector_kept"] is False
    assert out["by_kind"]["choice"]["vector_skipped"].startswith("open label set")
    assert out["by_kind"]["choice"]["temperature"] == pytest.approx(
        cal.fit_temperature(fit["choice"])
    )
