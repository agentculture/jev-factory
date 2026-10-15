"""``jev distill <stage>`` and the distillery's own stage registry (t22).

Stdlib fakes stand in for the teacher, the drafter and the student tokenizer: no GPU, no
network, no torch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pytest

from jev_factory.cli import _build_parser, main
from jev_factory.distill import assemble as dassemble
from jev_factory.distill import audit as daudit
from jev_factory.distill import dataset as ddataset
from jev_factory.distill import gate as dgate
from jev_factory.distill import pipeline as P
from jev_factory.distill import prereg as dprereg
from jev_factory.distill.corrections import read_corrections
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.distill.systemone import CallResult, QuestionResult
from jev_factory.explain import known_paths
from jev_factory.factory import detach
from jev_factory.factory import pipeline as factory_pipeline
from jev_factory.factory.config import load_config
from jev_factory.factory.stages import DEFAULT_REGISTRY, read_manifest

STAGES = P.stage_names()
CHOICE = Question("choice", (("fix", "repair it"), ("wait", "do nothing")))
NOUL = Question("noul", (("false", ""), ("true", "")))
GOOD, WEAK = "Praveenrajus/jev-bench:good", "Praveenrajus/jev-bench:weak"
_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def _state(i: int) -> str:
    rng = random.Random(i)
    return " ".join(f"{rng.choice(_WORDS)}{rng.randrange(10**6)}" for _ in range(10))


def _rec(i: int, source: str, split: str = "train") -> Record:
    return Record(
        id=f"r{i:03d}",
        state=_state(i),
        questions={"a": CHOICE, "b": NOUL},
        gold={"a": Gold(label="fix"), "b": Gold(label="true", probability_true=1.0)},
        split=split,
        source=source,
        licence="MIT",
        gold_provenance="human",
    )


def _records() -> list[Record]:
    out = [_rec(i, GOOD) for i in range(40)] + [_rec(100 + i, WEAK) for i in range(40)]
    return out + [_rec(200 + i, GOOD, "test") for i in range(6)]


class FakeTeacher:
    """Answers ``fix``/true for the good source and ``wait``/false for the weak one."""

    def __init__(self, records):
        self.source = {r.id: r.source for r in records}
        self.asked: list[str] = []

    def ask(self, wire):
        self.asked.append(wire["id"])
        good = self.source[wire["id"]] == GOOD
        results = []
        for qid, spec in wire["questions"].items():
            if spec["type"] == "noul":
                results.append(QuestionResult(qid, "noul", p_true=0.9 if good else 0.1))
            else:
                dist = {"fix": 0.9, "wait": 0.1} if good else {"fix": 0.1, "wait": 0.9}
                results.append(QuestionResult(qid, "choice", distribution=dist))
        return CallResult(wire["id"], "clef", {"url": "fake"}, "sha", results)


def _write_config(work: Path, records: Path) -> Path:
    toml = work / "run.toml"
    toml.write_text(
        "\n".join(
            [
                f"work = {json.dumps(str(work))}",
                'base = "example-org/base"',
                f'base_rev = "{"0" * 40}"',
                'hub_prefix = "example-org/distill-"',
                'licence = "Apache-2.0"',
                f"distill_records = {json.dumps(str(records))}",
                'distill_teacher_url = "http://127.0.0.1:9"',
                'distill_teacher_model = "clef"',
                'distill_teacher_repo = "example-org/clef-27b"',
                'distill_teacher_revision = "abc123"',
                'distill_teacher_build_id = "clef-b1"',
                'distill_teacher_engine = "sglang"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return toml


@pytest.fixture
def work(tmp_path):
    """A fresh distill run: run.toml, a records file and the distill pre-registration."""
    work = tmp_path / "run"
    (work / "distill").mkdir(parents=True)
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps(r.to_json(), sort_keys=True) + "\n" for r in _records()), "utf-8"
    )
    _write_config(work, records)
    (work / dprereg.PREREG_REL).write_text(json.dumps(dprereg.example_doc()), encoding="utf-8")
    return work


def _ctx(work: Path, **services) -> P.DistillContext:
    cfg = load_config(work / "run.toml", cli={"work": str(work)}, environ={})
    return P.DistillContext(work=work, config=cfg, services=P.Services(**services))


def _services(**over):
    base = {
        "token_counter": lambda text: len(text.split()),
        "teacher_client": FakeTeacher(_records()),
        "drafter": lambda rec: {"a": {"label": "fix"}},
        "today": lambda: "2026-10-14",
    }
    return {**base, **over}


KNOBS = {"audit": {"floor": 0.6}}


def _apply(work: Path, *names: str, **services) -> dict:
    ctx = _ctx(work, **_services(**services))
    out = {}
    for name in names:
        out[name] = P.run(work, name, ctx, KNOBS.get(name), apply=True)
        assert out[name]["status"] == "complete", (name, out[name].get("error"))
    return out


THROUGH_ASSEMBLE = (
    "preregister",
    "dataset",
    "contamination",
    "teacher-label",
    "audit",
    "corrections-draft",
    "assemble",
)


def tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    rc = main(list(argv))
    out = capsys.readouterr()
    return rc, out.out, out.err


# ----------------------------------------------------------------- registry


def test_every_distill_stage_is_in_its_own_registry_in_dependency_order():
    assert STAGES == list(P.STAGE_ORDER)
    assert P.REGISTRY is not DEFAULT_REGISTRY
    assert P.REGISTRY is not factory_pipeline.REGISTRY
    # the jev run registry is untouched: no distill-only stage leaks into it
    assert not {"dataset", "contamination", "teacher-label", "gate", "final"} & set(
        factory_pipeline.stage_names()
    )
    for name in STAGES:
        stage = P.REGISTRY.get(name)
        assert stage.summary
        assert all(STAGES.index(d) < STAGES.index(name) for d in stage.deps)
        assert set(P.DEFAULT_KNOBS) == set(STAGES)


def test_the_layout_matches_the_modules_that_write_it():
    d = P.DISTILL_DIR
    assert dprereg.PREREG_REL == f"{d}/{P.PREREG_FILE}"
    assert dprereg.LOCK_REL == f"{d}/{P.PREREG_LOCK}"
    assert daudit.AUDIT_REL == f"{d}/{P.AUDIT_FILE}"
    assert dassemble.OUT_REL == f"{d}/{P.TRAIN_SET}"
    assert dassemble.MANIFEST_REL == f"{d}/{P.ASSEMBLE_MANIFEST}"
    assert f"{d}/{P.GATE_FILE}" == f"{dgate.GATE_REL}/{dgate.GATE_FILE}"


def test_jev_run_surface_is_unchanged():
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    run = sub.choices["run"]
    stages = next(a for a in run._actions if isinstance(a, argparse._SubParsersAction))
    assert list(stages.choices) == factory_pipeline.stage_names()
    assert "distill" in sub.choices


# ----------------------------------------------------------------- the verb


def test_distill_lists_its_stages_in_order(capsys):
    rc, out, _ = run_cli(capsys, "distill", "--json")
    assert rc == 0
    assert [s["stage"] for s in json.loads(out)["stages"]] == STAGES
    rc, out, _ = run_cli(capsys, "distill")
    assert rc == 0
    assert all(s in out for s in STAGES)


def test_distill_has_apply_detach_force_and_json():
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    flags = {o for a in sub.choices["distill"]._actions for o in a.option_strings}
    assert {"--apply", "--detach", "--force", "--json", "--knob", "--work"} <= flags


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_writes_nothing_without_apply(stage, work, capsys):
    """One case per stage, on a fresh run: a plan is printed, no file changes."""
    root = work.parent
    before = tree(root)
    rc, out, err = run_cli(capsys, "distill", stage, "--work", str(work), "--json")
    assert rc == 0, err
    plan = json.loads(out)
    assert plan["applied"] is False
    assert plan["stage"] == stage
    assert plan["would_run"] is True
    assert tree(root) == before
    assert not (work / "distill" / "manifests").exists()
    rc, out, err = run_cli(capsys, "distill", stage, "--work", str(work))
    assert rc == 0, err
    assert out.startswith(f"dry run: distill stage {stage}")
    assert tree(root) == before


@pytest.fixture
def assembled(work):
    _apply(work, *THROUGH_ASSEMBLE)
    return work


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_writes_nothing_without_apply_on_a_populated_run(stage, assembled, capsys):
    """One case per stage again, after the stages up to assemble have run."""
    root = assembled.parent
    before = tree(root)
    argv = ["distill", stage, "--work", str(assembled), "--json"]
    if stage == "upload":
        argv += ["--knob", "model_repo=example-org/distill-m", "--knob", "dataset_repo=x/y"]
    for key, value in KNOBS.get(stage, {}).items():
        argv += ["--knob", f"{key}={value}"]  # judged by the knobs it ran with
    rc, out, err = run_cli(capsys, *argv)
    assert rc == 0, err
    plan = json.loads(out)
    assert plan["applied"] is False
    assert plan["would_run"] is (stage not in THROUGH_ASSEMBLE)
    if stage in ("train", "quantize", "gate", "final", "upload"):
        assert "module_plan" in plan or "module_plan_blocked" in plan
    assert tree(root) == before


def test_apply_runs_through_assemble_with_per_stage_manifests(assembled):
    base = assembled / "distill"
    for name in THROUGH_ASSEMBLE:
        man = read_manifest(base, name)
        assert man["status"] == "complete"
        assert man["stage"] == name
        assert (base / "manifests" / f"{name}.json").is_file()
    assert (base / "prereg.lock.json").is_file()
    assert (base / "dataset" / "train.jsonl").is_file()
    labels = (base / "teacher" / "teacher_labels.jsonl").read_text().splitlines()
    assert labels
    table = json.loads((base / "teacher_audit.json").read_text())
    assert table["teacher_weak_sources"] == [WEAK]
    proposed = read_corrections(base / "corrections" / "corrections.jsonl")
    assert proposed
    assert {c.status for c in proposed} == {"proposed"}
    frozen = json.loads((base / "assemble" / "assemble_manifest.json").read_text())
    assert frozen["status"] == "frozen"
    assert frozen["counts"]["trainable_corrections"] == 0  # nothing is reviewed yet


def test_teacher_label_only_labels_the_train_side(work):
    teacher = FakeTeacher(_records())
    _apply(work, "preregister", "dataset", "teacher-label", teacher_client=teacher)
    train = {r.id for r in ddataset.read_records(work / "distill" / "dataset" / "train.jsonl")}
    assert set(teacher.asked) == train


def test_teacher_label_progress_is_shown_by_status(assembled, capsys):
    progress = detach.read_progress(assembled / "jobs" / "distill", "teacher-label")
    train = ddataset.read_records(assembled / "distill" / "dataset" / "train.jsonl")
    assert progress["done"] == progress["total"] == len(train)
    rc, out, _ = run_cli(capsys, "status", str(assembled), "--json")
    row = next(s for s in json.loads(out)["distill_stages"] if s["stage"] == "teacher-label")
    assert row["jobs"][0]["job"] == "distill/teacher-label"
    assert row["jobs"][0]["progress"]["done"] == len(train)


def test_a_resumed_teacher_label_asks_only_unanswered_items(assembled):
    teacher = FakeTeacher(_records())
    ctx = _ctx(assembled, **_services(teacher_client=teacher))
    man = P.run(assembled, "teacher-label", ctx, apply=True, force=True)
    assert man["status"] == "complete"
    assert teacher.asked == []  # every request is a cache hit
    labelled = json.loads((assembled / "distill" / P.TEACHER_MANIFEST).read_text())
    assert labelled["cache_hits"] == labelled["records"]
    assert labelled["requests_sent"] == 0


def test_a_fresh_stage_is_skipped_and_force_reruns_it(assembled, capsys):
    argv = ["distill", "contamination", "--work", str(assembled), "--apply", "--json"]
    rc, out, err = run_cli(capsys, *argv)
    assert rc == 0, err
    assert json.loads(out)["skipped"] is True
    first = read_manifest(assembled / "distill", "contamination")["finished"]
    rc, out, err = run_cli(capsys, *argv, "--force")
    assert rc == 0, err
    assert json.loads(out).get("skipped") is None
    assert read_manifest(assembled / "distill", "contamination")["finished"] > first


def test_a_changed_records_file_makes_the_dataset_stale(assembled, capsys):
    records = Path(load_config(assembled / "run.toml", environ={})["distill_records"])
    records.write_text(records.read_text() + "\n", encoding="utf-8")
    rc, out, _ = run_cli(capsys, "distill", "dataset", "--work", str(assembled), "--json")
    assert rc == 0
    assert json.loads(out)["stale"] == "knobs changed"


def test_a_failed_stage_exits_non_zero_and_records_it(work, capsys):
    rc, _, err = run_cli(capsys, "distill", "audit", "--work", str(work), "--apply")
    assert rc != 0
    assert "audit" in err
    assert read_manifest(work / "distill", "audit")["status"] == "failed"


def test_unknown_knobs_and_misused_flags_are_refused(work, capsys):
    rc, _, err = run_cli(capsys, "distill", "dataset", "--work", str(work), "--knob", "nope=1")
    assert rc == 1
    assert "unknown knob" in err
    rc, _, err = run_cli(capsys, "distill", "dataset", "--work", str(work), "--detach")
    assert rc == 1
    assert "--detach needs --apply" in err
    rc, _, err = run_cli(capsys, "distill", "--apply")
    assert rc == 1
    assert "name the distill stage" in err
    with pytest.raises(SystemExit) as stop:
        run_cli(capsys, "distill", "nope", "--json")
    assert stop.value.code == 1
    assert json.loads(capsys.readouterr().err)["code"] == 1
    assert not (work / "distill" / "manifests").exists()


def test_the_upload_stage_names_its_repositories_before_anything(work, capsys):
    rc, out, _ = run_cli(capsys, "distill", "upload", "--work", str(work), "--json")
    assert rc == 0
    assert "model_repo" in json.loads(out)["module_plan_blocked"]["message"]


# ----------------------------------------------------------------- status and detach


def test_status_shows_distill_stages(assembled, capsys):
    rc, out, err = run_cli(capsys, "status", str(assembled), "--json")
    assert rc == 0, err
    doc = json.loads(out)
    rows = {s["stage"]: s for s in doc["distill_stages"]}
    assert list(rows) == STAGES
    assert all(rows[n]["status"] == "complete" for n in THROUGH_ASSEMBLE)
    assert rows["train"]["status"] == "not-run"
    rc, out, _ = run_cli(capsys, "status", str(assembled))
    assert "distill stages:" in out
    assert "teacher-label" in out


def test_status_of_a_run_without_distill_lists_none(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    rc, out, _ = run_cli(capsys, "status", str(plain), "--json")
    assert rc == 0
    assert json.loads(out)["distill_stages"] == []
    rc, out, _ = run_cli(capsys, "status", str(plain))
    assert "distill stages:" not in out


def test_detached_distill_stage_survives_and_status_reports_its_job(work, capsys):
    argv = ["distill", "preregister", "--work", str(work), "--apply", "--detach", "--json"]
    rc, out, err = run_cli(capsys, *argv)
    assert rc == 0, err
    info = json.loads(out)
    assert info["detached"] is True
    assert info["pid"] > 0
    jobs = work / "jobs" / "distill"
    deadline = time.time() + 60
    while detach.job_status(jobs, "preregister")["state"] != "done":
        assert time.time() < deadline, "detached distill stage did not finish"
        time.sleep(0.2)
    assert detach.job_status(jobs, "preregister")["rc"] == 0, (jobs / "preregister.log").read_text()
    rc, out, _ = run_cli(capsys, "status", str(work), "--json")
    row = json.loads(out)["distill_stages"][0]
    assert row["stage"] == "preregister"
    assert row["status"] == "complete"
    assert row["jobs"][0]["job"] == "distill/preregister"
    assert row["jobs"][0]["state"] == "done"


def test_detach_refuses_a_distill_stage_that_is_already_running(work, capsys):
    import os

    jobs = work / "jobs" / "distill"
    jobs.mkdir(parents=True)
    (jobs / "dataset.pid").write_text(f"{os.getpid()}\n")
    argv = ["distill", "dataset", "--work", str(work), "--apply", "--detach"]
    rc, _, err = run_cli(capsys, *argv)
    assert rc == 1
    assert "already running" in err


# ----------------------------------------------------------------- docs surface


def test_explain_learn_and_overview_cover_distill(capsys):
    paths = known_paths()
    assert ("distill",) in paths
    for stage in STAGES:
        assert ("distill", stage) in paths
    rc, out, _ = run_cli(capsys, "explain", "distill")
    assert rc == 0
    assert all(f"`{s}`" in out for s in STAGES)
    rc, out, _ = run_cli(capsys, "learn", "--json")
    assert ["distill", "<stage>"] in [c["path"] for c in json.loads(out)["commands"]]
    rc, out, _ = run_cli(capsys, "learn")
    assert "distill <stage>" in out
    rc, out, _ = run_cli(capsys, "overview")
    assert "distill <stage>" in out


def test_benchmark_applies_on_the_fit_and_selection_folds_only(assembled):
    from jev_factory.distill import benchmark as bench

    clients = {"clef": FakeTeacher(_records()), "stock_v21": FakeTeacher(_records())}
    targets = {n: bench.Target(n, "http://127.0.0.1:9", n) for n in clients}
    _apply(
        assembled,
        "benchmark",
        benchmark_targets=targets,
        benchmark_clients=clients,
        memory_probe=lambda: None,
    )
    ds = assembled / "distill" / "dataset"
    folds = {
        r.id for side in ("fit", "selection") for r in ddataset.read_records(ds / f"{side}.jsonl")
    }
    for client in clients.values():
        assert set(client.asked) == folds
    result = json.loads((assembled / "distill" / P.BENCHMARK_FILE).read_text())
    assert set(result["models"]) == {"clef", "stock_v21"}
    assert read_manifest(assembled / "distill", "benchmark")["status"] == "complete"
