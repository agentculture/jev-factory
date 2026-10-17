"""``jev distill <stage>`` and the distillery's own stage registry (t22).

Stdlib fakes stand in for the teacher, the drafter and the student tokenizer: no GPU, no
network, no torch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.cli import _build_parser, main
from jev_factory.cli._errors import CliError
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
    if stage in ("train", "quantize", "gate", "final", "edge", "upload"):
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


def _jevbench(n: int = 231) -> list[Record]:
    return [replace(_rec(1000 + i, GOOD), id=f"jb{i:03d}") for i in range(n)]


def _probe() -> list[Record]:
    return [replace(_rec(2000 + i, GOOD), id=f"probe{i}") for i in range(3)]


def _add_config(work: Path, **keys: str) -> None:
    toml = work / "run.toml"
    extra = "".join(f"{k} = {json.dumps(v)}\n" for k, v in keys.items())
    toml.write_text(toml.read_text(encoding="utf-8") + extra, encoding="utf-8")


def _jsonl(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in records), "utf-8")
    return path


def test_benchmark_applies_on_the_fit_and_selection_folds_only(assembled):
    """Risk r12: benchmark also reproduces stock on JevBench w4096 and measures it on the probe
    set and the fit/selection folds; the sealed sides are never asked."""
    from jev_factory.distill import benchmark as bench

    tasks = _jsonl(assembled.parent / "jevbench.jsonl", _jevbench())
    probe = _jsonl(assembled.parent / "probe.jsonl", _probe())
    _add_config(assembled, distill_jevbench_tasks=str(tasks), distill_probe_set=str(probe))
    every = _records() + _jevbench() + _probe()
    clients = {"clef": FakeTeacher(every), "stock": FakeTeacher(every)}
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
    assert set(clients["clef"].asked) == folds
    jb = {r.id for r in _jevbench()}
    assert set(clients["stock"].asked) == folds | jb | {r.id for r in _probe()}
    sealed = {
        r.id for side in ("test", "held_out") for r in ddataset.read_records(ds / f"{side}.jsonl")
    }
    assert not sealed & set(clients["stock"].asked)
    result = json.loads((assembled / "distill" / P.BENCHMARK_FILE).read_text())
    assert set(result["models"]) == {"clef", "stock"}
    reproduction = json.loads((assembled / "distill" / P.STOCK_REPRODUCTION).read_text())
    assert reproduction["n"] == 2 * 231  # every gold-bearing question of the 231 tasks
    assert reproduction["tasks_sha256"] == dprereg.sha256_file(tasks)
    assert reproduction["base_mismatch"] is True  # 231/231 right: the CI excludes 0.7792
    measured = json.loads((assembled / "distill" / P.STOCK_MEASUREMENTS).read_text())
    assert set(measured["sets"]) == {"fit", "selection", "probe"}
    assert measured["probe_set"] == {"count": 3}
    assert "probe0" not in json.dumps(measured)  # the probe set is private
    manifest = read_manifest(assembled / "distill", "benchmark")
    assert manifest["status"] == "complete"
    assert set(manifest["outputs"]) >= {P.STOCK_REPRODUCTION, P.STOCK_MEASUREMENTS}


def test_benchmark_without_the_jevbench_tasks_refuses_before_serving(assembled):
    """Risk r12: no JevBench task file -> the stage fails before any model is asked."""
    from jev_factory.distill import benchmark as bench

    clients = {"clef": FakeTeacher(_records()), "stock": FakeTeacher(_records())}
    targets = {n: bench.Target(n, "http://127.0.0.1:9", n) for n in clients}
    ctx = _ctx(assembled, benchmark_targets=targets, benchmark_clients=clients)
    out = P.run(assembled, "benchmark", ctx, apply=True)
    assert out["status"] == "failed"
    assert "distill_jevbench_tasks" in out["error"]
    assert clients["stock"].asked == []
    assert clients["clef"].asked == []


def test_train_refuses_without_the_stock_reproduction_record(assembled, capsys):
    """Risk r12: an ABSENT stock reproduction makes train refuse (fail closed)."""
    out = P.run(assembled, "train", _ctx(assembled), apply=True)
    assert out["status"] == "failed"
    assert "no stock reproduction record" in out["error"]
    rc, plan, _ = run_cli(capsys, "distill", "train", "--work", str(assembled), "--json")
    assert rc == 0
    blocked = json.loads(plan)["module_plan_blocked"]
    assert "no stock reproduction record" in blocked["message"]


def test_a_deviation_id_lets_train_past_an_absent_reproduction(assembled):
    ctx = replace(_ctx(assembled), deviation_id="d9")
    out = P.run(assembled, "train", ctx, apply=True)
    assert "stock reproduction" not in (out.get("error") or "")


def test_train_keeps_refusing_a_recorded_base_mismatch(assembled):
    from jev_factory.distill import benchmark as bench

    bench.write_reproduction(
        assembled,
        {"base_mismatch": True, "n": 231, "n_correct": 150, "ci_low": 0.58, "ci_high": 0.70},
    )
    out = P.run(assembled, "train", _ctx(assembled), apply=True)
    assert out["status"] == "failed"
    assert "base-mismatch" in out["error"]


# ----------------------------------------------------------------- t17: run wiring


def test_edge_is_registered_after_final_and_there_are_fifteen_stages(capsys):
    """AC1: 'edge' is registered after final; jev distill --help lists 15 stages."""
    assert len(STAGES) == 15
    assert STAGES.index("edge") == STAGES.index("final") + 1
    assert P.REGISTRY.get("edge").deps == ("final",)
    assert set(P.DEFAULT_KNOBS["edge"]) == {"label", "served_model"}
    with pytest.raises(SystemExit):
        main(["distill", "--help"])
    assert "final, edge, bundle, upload." in " ".join(capsys.readouterr().out.split())
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    stage = next(a for a in sub.choices["distill"]._actions if a.dest == "stage")
    assert list(stage.choices) == STAGES
    assert stage.help == "The distill stage: " + ", ".join(STAGES) + "."


def test_every_new_run_config_key_is_declared_and_read_by_a_stage():
    """AC1: the six new keys exist and the stages that use them record them."""
    from jev_factory.factory.config import KEYS

    declared = {k.name: k for k in KEYS}
    new = {
        "distill_teacher_key_env": "teacher-label",
        "distill_teacher_model_header": "teacher-label",
        "distill_gold_ledger": "final",
        "distill_agx_url": "final",
        "distill_edge_url": "edge",
        "distill_edge_memory_cap": "edge",
        "distill_jevbench_tasks": "benchmark",
    }
    for key, stage in new.items():
        assert key in declared, key
        assert key in P.STAGE_CONFIG_KEYS[stage], (key, stage)
    assert declared["distill_teacher_key_env"].kind == "envname"  # a NAME, never the key


class LockCheckingTeacher(FakeTeacher):
    """Records whether the teacher-label lock was held while each request was asked."""

    def __init__(self, records, work, fail=False):
        super().__init__(records)
        self.work, self.fail, self.held = work, fail, []

    def ask(self, wire):
        from jev_factory.distill import edge as dedge

        self.held.append(dedge.teacher_lock_path(self.work).is_file())
        if self.fail:
            raise RuntimeError("the teacher went away")
        return super().ask(wire)


def test_teacher_label_holds_the_lock_while_labelling_and_releases_it(work):
    """Item 1 (t15): the lock is acquired at start and released when the stage ends."""
    from jev_factory.distill import edge as dedge

    teacher = LockCheckingTeacher(_records(), work)
    _apply(work, "preregister", "dataset", "teacher-label", teacher_client=teacher)
    assert teacher.held
    assert all(teacher.held)
    assert not dedge.teacher_lock_path(work).exists()


def test_teacher_label_releases_the_lock_when_it_fails(work):
    from jev_factory.distill import edge as dedge

    _apply(work, "preregister", "dataset")
    teacher = LockCheckingTeacher(_records(), work, fail=True)
    out = P.run(work, "teacher-label", _ctx(work, **_services(teacher_client=teacher)), apply=True)
    assert out["status"] == "failed"
    assert teacher.held == [True]
    assert not dedge.teacher_lock_path(work).exists()


def test_teacher_label_refuses_while_another_holds_the_lock(work):
    from jev_factory.distill import edge as dedge

    _apply(work, "preregister", "dataset")
    dedge.acquire_teacher_lock(work)
    teacher = FakeTeacher(_records())
    out = P.run(work, "teacher-label", _ctx(work, **_services(teacher_client=teacher)), apply=True)
    assert out["status"] == "failed"
    assert "already held" in out["error"]
    assert teacher.asked == []
    assert dedge.teacher_lock_path(work).exists()  # someone else's lock is left alone


def test_the_teacher_client_gets_the_key_env_name_and_the_model_header(work, monkeypatch):
    """Item 7 (t5): distill_teacher_key_env and distill_teacher_model_header reach the client."""
    from jev_factory.distill import systemone

    made = []

    class Recorder(FakeTeacher):
        def __init__(self, base_url, **kw):
            super().__init__(_records())
            made.append({"base_url": base_url, **kw})

    monkeypatch.setattr(systemone, "SystemOneClient", Recorder)
    _add_config(work, distill_teacher_key_env="CLEF_KEY", distill_teacher_model_header="clef-27b")
    _apply(work, "preregister", "dataset")
    ctx = _ctx(work, **{**_services(), "teacher_client": None})
    out = P.run(work, "teacher-label", ctx, apply=True)
    assert out["status"] == "complete", out.get("error")
    assert made[0]["auth_env"] == "CLEF_KEY"
    assert made[0]["model_header"] == "clef-27b"
    manifest = json.loads((work / "distill" / P.TEACHER_MANIFEST).read_text())
    assert manifest["auth_env"] == "CLEF_KEY"  # the NAME is recorded, never a value
    assert manifest["model_header"] == "clef-27b"


def test_final_builds_four_models_with_the_shipped_quant_on_the_agx(assembled):
    """Item 6 (t14): clef, stock, student_bf16 served here, student pre-served on the AGX."""
    from jev_factory.distill import final as dfinal

    _add_config(
        assembled,
        distill_stock_url="http://127.0.0.1:10",
        distill_stock_model="v21",
        distill_agx_url="http://127.0.0.1:18090",
        distill_student_python="python3",
        distill_student_lockfile="0" * 64,
    )
    artifact = assembled / "distill" / "quant" / "builds" / "bf16"
    artifact.mkdir(parents=True)
    measurements = assembled / "distill" / P.QUANT_MEASUREMENTS
    measurements.write_text(
        json.dumps({"formats": {"bf16": {"available": True, "artifact": str(artifact)}}})
    )
    models = P._final_models(_ctx(assembled))
    assert set(models) == set(dfinal.MODELS)
    assert set(dfinal.MODELS) == {"clef", "stock", "student_bf16", "student"}
    agx = models["student"]
    assert agx.target.base_url == "http://127.0.0.1:18090"
    assert agx.target.start is None  # pre-served: no start, no stop
    assert agx.target.stop is None
    assert agx.get is not None
    bf16 = models["student_bf16"]
    assert bf16.target.start is not None
    assert bf16.target.stop is not None
    assert bf16.target.base_url.startswith("http://127.0.0.1:")


def test_a_non_localhost_agx_url_is_refused(assembled):
    _add_config(assembled, distill_agx_url="http://10.0.0.7:18090")
    ctx = _ctx(assembled)
    with pytest.raises(CliError, match="distill_agx_url"):
        P._agx_final_model(ctx)


def test_the_final_inputs_come_from_the_run_config_and_earlier_stages(assembled, tmp_path):
    """Item 6: gold ledger, probe set, bf16 calibration (when train wrote one) and the
    contamination manifest the contamination stage matched against."""
    from jev_factory.distill import contamination

    ledger = tmp_path / "ledger.jsonl"
    probe = _jsonl(tmp_path / "probe.jsonl", _probe())
    _add_config(assembled, distill_gold_ledger=str(ledger), distill_probe_set=str(probe))
    inputs = P._final_inputs(_ctx(assembled))
    assert Path(inputs.gold_ledger) == ledger
    assert Path(inputs.probe) == probe
    assert inputs.bf16_calibration is None  # train has not written one
    assert inputs.contamination_manifest == contamination.load_manifest()
    cal = assembled / "distill" / P.SELECT_CALIBRATION
    cal.parent.mkdir(parents=True)
    cal.write_text("{}")
    assert P._final_inputs(_ctx(assembled)).bf16_calibration == cal


def test_the_unverified_gold_knob_reaches_the_final_and_must_be_a_boolean(assembled):
    """d10: ``--knob unverified_gold=true`` sets the proving mode; off by default."""
    ctx = _ctx(assembled)
    assert P.DEFAULT_KNOBS["final"]["unverified_gold"] is False
    assert P._final_inputs(ctx).unverified_gold is False
    assert P._final_inputs(ctx, {"unverified_gold": True}).unverified_gold is True
    with pytest.raises(CliError, match="unverified_gold must be true or false"):
        P._final_inputs(ctx, {"unverified_gold": "yes"})


def test_the_final_refuses_before_the_contamination_check_has_run(work):
    _apply(work, "preregister", "dataset")
    ctx = _ctx(work)
    with pytest.raises(CliError, match="contamination check has not run"):
        P._final_inputs(ctx)
    assert "contamination" in P.REGISTRY.get("final").deps


def test_quantize_asks_the_probe_set_of_the_pruned_comparison(assembled, tmp_path):
    """Item 4 (r14): the probe set reaches the quant stage; a sealed probe id is refused."""
    probe = _jsonl(tmp_path / "probe.jsonl", _probe())
    _add_config(assembled, distill_probe_set=str(probe))
    got = P._probe_records(_ctx(assembled))
    assert [r.id for r in got] == [r.id for r in _probe()]
    sealed = ddataset.read_records(assembled / "distill" / "dataset" / "test.jsonl")[0]
    _jsonl(probe, [sealed])
    ctx = _ctx(assembled)
    with pytest.raises(CliError, match="sealed"):
        P._probe_records(ctx)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("4G", 4 * 1024**3), ("512m", 512 * 1024**2), (4096, 4096)],
)
def test_the_edge_memory_cap_parses_like_systemd(value, expected):
    assert P.parse_memory_cap(value) == expected


@pytest.mark.parametrize("value", ["four", "4X", "-1", "0", "G"])
def test_a_bad_edge_memory_cap_is_refused(value):
    with pytest.raises(CliError, match="distill_edge_memory_cap"):
        P.parse_memory_cap(value)


def test_the_edge_stage_wires_url_cap_label_and_probe(assembled, tmp_path, monkeypatch):
    """Item 8: edge uses distill_edge_url and distill_edge_memory_cap; the label is a knob."""
    from jev_factory.distill import edge as dedge

    probe = _jsonl(tmp_path / "probe.jsonl", _probe())
    _add_config(
        assembled,
        distill_edge_url="http://127.0.0.1:18091",
        distill_edge_memory_cap="4G",
        distill_probe_set=str(probe),
    )
    seen = {}
    monkeypatch.setattr(dedge, "run_edge", lambda **kw: seen.update(kw) or {})
    reset = P._CURRENT.set(_ctx(assembled))
    try:
        P.stage_edge(assembled / "distill", {**P.DEFAULT_KNOBS["edge"], "label": "agx"})
    finally:
        P._CURRENT.reset(reset)
    assert seen["base_url"] == "http://127.0.0.1:18091"
    assert seen["expected_cap"] == 4 * 1024**3
    assert seen["label"] == "agx"
    assert seen["apply"] is True
    assert Path(seen["probe"]) == probe
    assert Path(seen["selection"]).name == "selection.jsonl"


def test_the_edge_stage_refuses_without_a_configured_cap(assembled, tmp_path):
    _add_config(assembled, distill_edge_url="http://127.0.0.1:18091")
    ctx, knobs = _ctx(assembled), P.DEFAULT_KNOBS["edge"]
    with pytest.raises(CliError, match="memory cap"):
        P._edge_kwargs(ctx, knobs)


def test_the_edge_dry_run_writes_nothing_and_plans_through_the_module(assembled, tmp_path, capsys):
    probe = _jsonl(tmp_path / "probe.jsonl", _probe())
    _add_config(
        assembled,
        distill_edge_url="http://127.0.0.1:18091",
        distill_edge_memory_cap="4G",
        distill_probe_set=str(probe),
    )
    before = tree(assembled.parent)
    rc, out, err = run_cli(
        capsys, "distill", "edge", "--work", str(assembled), "--knob", "label=agx", "--json"
    )
    assert rc == 0, err
    plan = json.loads(out)
    assert plan["module_plan"]["dry_run"] is True
    assert plan["module_plan"]["label"] == "agx"
    assert plan["module_plan"]["expected_cap"] == 4 * 1024**3
    assert tree(assembled.parent) == before


# ----------------------------------------------------------------- t12: pipeline wiring


def _v3(work: Path, *, absent=(), deployed_format: str = "quant") -> None:
    """Rewrite the run's pre-registration as v3 with *absent* optional roles."""
    doc = dprereg.example_doc_v3(deployed_format=deployed_format)
    for role in absent:
        doc["roles"][role] = {"present": False}
    (work / dprereg.PREREG_REL).write_text(json.dumps(doc), encoding="utf-8")


def _profile_toml(path: Path, **over) -> Path:
    """A student profile file: the built-in 2610 profile with *over* replaced."""
    import dataclasses

    from jev_factory.distill.profile import DEFAULT_2610

    table = dataclasses.asdict(dataclasses.replace(DEFAULT_2610, **over))
    table.pop("card")
    temps = table.pop("temperature_by_kind")
    lines = [
        f"{k} = {json.dumps(list(v) if isinstance(v, tuple) else v)}" for k, v in table.items()
    ]
    lines += ["[temperature_by_kind]", *(f"{k} = {v!r}" for k, v in temps.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_new_run_config_keys_are_declared_and_load(work):
    """Note 1: the four t8 keys are run-config keys; the executor defaults to local."""
    cfg = load_config(work / "run.toml", cli={"work": str(work)}, environ={})
    assert cfg["distill_executor"] == "local"
    for key in ("distill_student_profile", "distill_teacher_variant", "distill_deployed_url"):
        assert cfg[key] is None
    _add_config(
        work,
        distill_student_profile="/profiles/nano.toml",
        distill_teacher_variant="NVFP4",
        distill_executor="local",
        distill_deployed_url="http://127.0.0.1:18090",
    )
    cfg = load_config(work / "run.toml", cli={"work": str(work)}, environ={})
    assert cfg["distill_teacher_variant"] == "NVFP4"
    assert cfg["distill_deployed_url"] == "http://127.0.0.1:18090"


def test_an_executor_other_than_local_is_refused(work):
    """c11: only the local executor exists; any other name is a clear refusal (no cloud)."""
    _add_config(work, distill_executor="aws-batch")
    with pytest.raises(CliError, match="aws-batch") as exc:
        P._executor(_ctx(work))
    assert "local" in exc.value.message
    from jev_factory.distill.executor import LocalExecutor

    cfg = _ctx(work).config
    cfg = replace(cfg, values={**cfg.values, "distill_executor": "local"})
    assert isinstance(P._executor(P.DistillContext(work=work, config=cfg)), LocalExecutor)


def test_an_injected_executor_is_the_one_used(work):
    fake = object()
    ctx = _ctx(work, executor=fake)
    assert P._executor(ctx) is fake


def test_edge_writes_skipped_naming_the_role_when_the_edge_role_is_absent(work):
    """c17: an absent secondary role skips its stage (status skipped, the role named)."""
    _v3(work, absent=("edge",))
    _apply(work, "preregister")
    out = P.run(work, "edge", _ctx(work), apply=True)
    assert out["status"] == "skipped"
    assert "edge" in out["reason"] and "absent" in out["reason"]
    assert read_manifest(work / "distill", "edge")["status"] == "skipped"
    plan = P.plan(work, "edge", _ctx(work))
    assert "absent" in plan["skip_reason"]


def test_corrections_draft_writes_skipped_naming_the_alt_teacher_when_it_is_absent(work):
    """c17/h14: the alt drafter is needed (teacher-weak rows) but the role is absent: the stage
    is skipped, never a require() refusal, and assemble still runs without corrections."""
    _v3(work, absent=("alt_teacher",))
    out = _apply(work, *THROUGH_ASSEMBLE[:5])
    assert out["audit"]["status"] == "complete"
    ctx = _ctx(work, **_services(drafter=None))
    man = P.run(work, "corrections-draft", ctx, apply=True)
    assert man["status"] == "skipped", man.get("error")
    assert "alt_teacher" in man["reason"]
    assert not (work / "distill" / P.CORRECTIONS_FILE).exists()
    done = P.run(work, "assemble", ctx, KNOBS.get("assemble"), apply=True)
    assert done["status"] == "complete", done.get("error")
    assert P.CORRECTIONS_FILE not in done["inputs"]


def test_quantize_writes_skipped_under_a_bf16_ship(work):
    """c7: deployed_format bf16 skips quantize (and its heal) with quant's own reason."""
    from jev_factory.distill import quant as dquant

    _v3(work, deployed_format="bf16")
    _apply(work, "preregister")
    out = P.run(work, "quantize", _ctx(work), apply=True)
    assert out["status"] == "skipped"
    assert out["reason"] == dquant.BF16_SKIP_REASON


def test_a_quant_ship_with_every_role_present_skips_nothing(work):
    _apply(work, "preregister")
    reg = dprereg.require_locked(work / dprereg.PREREG_REL, work)
    for name in STAGES:
        assert P._skip_reason(name, _ctx(work), reg) is None, name


def test_the_profile_reaches_the_dataset_tokenizer_and_builder(work, tmp_path, monkeypatch):
    """Note 2: the run's profile names the tokenizer's repo and is passed to build_dataset."""
    from jev_factory.distill import dataset as dds
    from jev_factory.distill import pins as dpins

    prof = _profile_toml(tmp_path / "nano.toml", base_repo="example-org/nano-base")
    _add_config(
        work,
        distill_student_profile=str(prof),
        distill_pin_cache=str(tmp_path / "cache"),
        distill_student_python="python3",
    )
    seen = {}
    monkeypatch.setattr(
        dds,
        "subprocess_token_counter",
        lambda python, path: seen.setdefault("tokenizer", path) and (lambda t: len(t.split())),
    )
    real = dds.build_dataset
    monkeypatch.setattr(
        dds,
        "build_dataset",
        lambda *a, **kw: seen.setdefault("profile", kw["profile"]) and real(*a, **kw),
    )
    ctx = _ctx(work, **_services(token_counter=None))
    out = P.run(work, "dataset", ctx, apply=True)
    assert out["status"] == "complete", out.get("error")
    assert seen["tokenizer"] == tmp_path / "cache" / dpins.cache_dirname("example-org/nano-base")
    assert seen["profile"].base_repo == "example-org/nano-base"


def test_the_profile_reaches_the_pins_and_the_default_is_unchanged(work, tmp_path):
    from jev_factory.distill import pins as dpins

    _add_config(work, distill_decider_revision="c0ffee11")
    assert P._pins(_ctx(work)) == dpins.default_pins(decider_revision="c0ffee11")
    prof = _profile_toml(tmp_path / "nano.toml", decider_repo="example-org/nano-decider")
    _add_config(work, distill_student_profile=str(prof))
    pins = P._pins(_ctx(work))
    assert pins.decider.repo_id == "example-org/nano-decider"


def test_the_teacher_variant_reaches_the_teacher_config_and_assemble(work, monkeypatch):
    """Note 3: unset is today's behaviour; set, it is stamped and assemble checks it."""
    assert P._teacher_config(_ctx(work), {"joint": True}).variant is None
    _add_config(work, distill_teacher_variant="NVFP4")
    assert P._teacher_config(_ctx(work), {"joint": True}).variant == "NVFP4"
    seen = {}
    monkeypatch.setattr(P.dassemble, "assemble", lambda *a, **kw: seen.update(kw))
    reset = P._CURRENT.set(_ctx(work))
    try:
        _apply(work, "preregister")
        P.stage_assemble(work / "distill", dict(P.DEFAULT_KNOBS["assemble"]))
    finally:
        P._CURRENT.reset(reset)
    assert seen["run_variant"] == "NVFP4"
    assert seen["run_build_id"] == "clef-b1"


def test_assemble_gets_no_run_teacher_check_without_a_variant(work, monkeypatch):
    seen = {}
    monkeypatch.setattr(P.dassemble, "assemble", lambda *a, **kw: seen.update(kw))
    _apply(work, "preregister")
    reset = P._CURRENT.set(_ctx(work))
    try:
        P.stage_assemble(work / "distill", dict(P.DEFAULT_KNOBS["assemble"]))
    finally:
        P._CURRENT.reset(reset)
    assert seen["run_variant"] is None and seen["run_build_id"] is None


def _bf16_build(work: Path, tmp_path: Path) -> Path:
    from tests import test_distill_bf16_ship as tbs

    cand = work / "distill" / "train" / tbs.WINNER
    bundle = tbs._hobson(tmp_path, cand / "bundle")
    (cand / "predictions.json").write_text(json.dumps({"fit": [], "selection": []}))
    sel = work / "distill" / "select"
    sel.mkdir(parents=True, exist_ok=True)
    (sel / "summary.json").write_text(json.dumps({"winner": tbs.WINNER}))
    (sel / "calibration.json").write_text(json.dumps(tbs.SELECT_CAL))
    return bundle


def test_a_bf16_final_builds_three_models_served_from_the_deployed_bundle(work, tmp_path):
    """Note 4 / c7: under bf16 there is no separate student; the bf16 student is served from
    the deployed bundle and names it as served_from."""
    from jev_factory.distill import final as dfinal

    _v3(work, deployed_format="bf16")
    _apply(work, "preregister")
    bundle = _bf16_build(work, tmp_path)
    _add_config(
        work,
        distill_stock_url="http://127.0.0.1:10",
        distill_stock_model="v21",
        distill_student_python="python3",
        distill_student_lockfile="0" * 64,
    )
    models = P._final_models(_ctx(work))
    assert set(models) == set(dfinal.required_models("bf16"))
    assert "student" not in models
    assert Path(models["student_bf16"].served_from) == bundle


def test_the_deployed_url_key_names_the_pre_served_student(work, tmp_path):
    """Note 4: distill_deployed_url (or its alias distill_agx_url) is the student's URL."""
    _add_config(work, distill_deployed_url="http://127.0.0.1:18095")
    _apply(work, "preregister")
    model = P._agx_final_model(_ctx(work))
    assert model.target.base_url == "http://127.0.0.1:18095"
    assert model.target.start is None


def test_the_edge_stage_passes_the_declared_deployed_format(work, tmp_path, monkeypatch):
    from jev_factory.distill import edge as dedge

    _v3(work, deployed_format="bf16")
    _apply(work, "preregister", "dataset")
    probe = _jsonl(tmp_path / "probe.jsonl", _probe())
    _add_config(
        work,
        distill_edge_url="http://127.0.0.1:18091",
        distill_edge_memory_cap="4G",
        distill_probe_set=str(probe),
    )
    seen = {}
    monkeypatch.setattr(dedge, "run_edge", lambda **kw: seen.update(kw) or {})
    reset = P._CURRENT.set(_ctx(work))
    try:
        P.stage_edge(work / "distill", dict(P.DEFAULT_KNOBS["edge"]))
    finally:
        P._CURRENT.reset(reset)
    assert seen["deployed_format"] == "bf16"


def test_bf16_stages_read_the_select_calibration_not_the_quant_one(work):
    _v3(work, deployed_format="bf16")
    _apply(work, "preregister")
    reg = dprereg.require_locked(work / dprereg.PREREG_REL, work)
    for name in ("gate", "final"):
        inputs = P._stage_for(name, _ctx(work), reg).inputs
        assert P.QUANT_CALIBRATION not in inputs and P.SELECT_CALIBRATION in inputs
    assert P.QUANT_PREDICTIONS not in P._stage_for("edge", _ctx(work), reg).inputs
