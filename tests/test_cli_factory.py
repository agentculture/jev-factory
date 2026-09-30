"""The factory verbs: ``jev init``, ``jev run <stage>``, ``jev status``, ``jev decide`` (t28)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytest

from jev_factory.cli import _build_parser, main
from jev_factory.domains.jev_cli import annotations as ann
from jev_factory.explain import known_paths
from jev_factory.factory import detach, pipeline
from tests.fixtures.toy_domain import DOMAIN  # noqa: F401  (importable as the domain module)
from tests.test_factory_pipeline import Run

DOMAIN_REF = "tests.fixtures.toy_domain"
STAGES = pipeline.stage_names()
BASE = ["--base", "example-org/toy-base", "--base-rev", "0" * 40]
BASE += ["--hub-prefix", "example-org/toy-lamps-jev-", "--licence", "Apache-2.0", "--seed", "7"]


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


@pytest.fixture
def scaffold(tmp_path, capsys):
    work = tmp_path / "run"
    rc, _, err = run_cli(capsys, "init", DOMAIN_REF, "--work", str(work), *BASE, "--apply")
    assert rc == 0, err
    return work


def test_every_new_verb_is_registered_with_json_and_an_explain_entry():
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert {"init", "run", "status", "decide"} <= set(sub.choices)
    for verb in ("init", "status", "decide"):
        assert any("--json" in a.option_strings for a in sub.choices[verb]._actions)
    paths = known_paths()
    for path in [("init",), ("run",), ("status",), ("decide",), *[("run", s) for s in STAGES]]:
        assert path in paths, path
    notes = ann.ANNOTATIONS
    assert notes["jev.init"].read_only is False and notes["jev.decide"].read_only is False
    assert all(notes[f"jev.run.{s}"].read_only is False for s in STAGES)


@pytest.mark.behavioral("o2")
def test_run_help_lists_every_build_stage(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["run", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for stage in STAGES:
        assert stage in text
    rc, out, _ = run_cli(capsys, "run", "--json")
    assert rc == 0
    assert [s["stage"] for s in json.loads(out)["stages"]] == STAGES


def test_there_is_one_run_subcommand_per_registered_stage():
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    run = sub.choices["run"]
    inner = next(a for a in run._actions if isinstance(a, argparse._SubParsersAction))
    assert list(inner.choices) == STAGES


# --- dry run by default (acceptance 2, o33) ---------------------------------------


@pytest.mark.behavioral("o33")
def test_init_is_a_dry_run_until_apply(tmp_path, capsys):
    work = tmp_path / "run"
    rc, out, _ = run_cli(capsys, "init", DOMAIN_REF, "--work", str(work), *BASE, "--json")
    assert rc == 0
    doc = json.loads(out)
    assert doc["applied"] is False and len(doc["files"]) == 2
    assert not work.exists()
    rc, _, _ = run_cli(capsys, "init", DOMAIN_REF, "--work", str(work), *BASE, "--apply")
    assert rc == 0
    assert (work / "run.json").is_file() and (work / "run.toml").is_file()
    toml = (work / "run.toml").read_text()
    assert 'base = "example-org/toy-base"' in toml and "seed = 7" in toml


@pytest.mark.behavioral("o33")
@pytest.mark.parametrize("stage", STAGES)
def test_every_run_stage_leaves_the_work_dir_unchanged_without_apply(scaffold, capsys, stage):
    before = tree(scaffold)
    rc, out, err = run_cli(capsys, "run", stage, "--work", str(scaffold), "--json")
    assert rc == 0, err
    doc = json.loads(out)
    assert doc["stage"] == stage and doc["applied"] is False and doc["would_run"] is True
    assert tree(scaffold) == before
    rc, out, _ = run_cli(capsys, "run", stage, "--work", str(scaffold))
    assert rc == 0 and out.startswith(f"dry run: stage {stage}")
    assert tree(scaffold) == before


def test_apply_runs_a_stage_and_writes_its_manifest(scaffold, capsys):
    rc, out, err = run_cli(capsys, "run", "config", "--work", str(scaffold), "--apply", "--json")
    assert rc == 0, err
    assert json.loads(out)["status"] == "complete"
    assert (scaffold / "manifests" / "config.json").is_file()
    rc, out, _ = run_cli(capsys, "run", "config", "--work", str(scaffold), "--apply")
    assert rc == 0 and "skipped" in out


# --- CliError paths (o33) ---------------------------------------------------------


@pytest.mark.behavioral("o33")
def test_errors_are_structured_clierrors_in_text_and_json(scaffold, tmp_path, capsys):
    cases = [
        ["run", "config", "--work", str(scaffold), "--knob", "nope=1"],
        ["run", "config", "--work", str(scaffold), "--knob", "novalue"],
        ["run", "config", "--work", str(scaffold), "--detach"],
        ["run", "config", "--work", str(tmp_path / "missing")],
        ["run", "config", "--domain", "no.such.module", "--work", str(scaffold)],
        ["status", str(tmp_path / "missing")],
        ["decide", str(scaffold)],
        ["init", DOMAIN_REF, "--work", str(scaffold), "--apply"],
        ["init", "no.such.module", "--work", str(tmp_path / "x")],
    ]
    for argv in cases:
        rc, out, err = run_cli(capsys, *argv, "--json")
        assert rc in (1, 2), argv
        assert out == "" and json.loads(err)["message"], argv
        rc, out, err = run_cli(capsys, *argv)
        assert rc in (1, 2) and err.startswith("error: "), argv


def test_init_refuses_a_work_dir_inside_a_git_worktree(capsys):
    rc, _, err = run_cli(capsys, "init", DOMAIN_REF, "--work", str(Path.cwd() / "x"), "--apply")
    assert rc == 1 and "git worktree" in err


def test_a_failing_stage_is_an_error_with_the_recorded_cause(scaffold, capsys):
    rc, out, err = run_cli(capsys, "run", "preregister", "--work", str(scaffold), "--apply")
    assert rc != 0 and out == "" and "preregister failed" in err
    assert json.loads((scaffold / "manifests" / "preregister.json").read_text())["status"] == (
        "failed"
    )


# --- status (acceptance 2, o2) ----------------------------------------------------


@pytest.mark.behavioral("o2")
def test_status_reads_manifests_and_shows_items_done_of_total(scaffold, capsys):
    rc, out, _ = run_cli(capsys, "run", "config", "--work", str(scaffold), "--apply")
    assert rc == 0
    ledger = detach.ItemLedger(scaffold / "jobs", "augment", 10)
    for i in range(3):
        ledger.record(f"item-{i}")
    (scaffold / "jobs" / "augment.pid").write_text(f"{os.getpid()}\n")
    before = tree(scaffold)
    rc, out, _ = run_cli(capsys, "status", str(scaffold), "--json")
    assert rc == 0 and tree(scaffold) == before
    doc = json.loads(out)
    stages = {s["stage"]: s for s in doc["stages"]}
    assert list(stages) == STAGES
    assert stages["config"]["status"] == "complete" and stages["config"]["stale"] is None
    assert stages["split"]["status"] == "not-run"
    aug = stages["augment"]
    assert aug["status"] == "running"
    assert aug["jobs"][0]["progress"] == {"done": 3, "total": 10}
    rc, text, _ = run_cli(capsys, "status", str(scaffold))
    assert "augment" in text and "3/10 items" in text and "decision records: 0" in text


def test_status_marks_a_changed_output_stale(scaffold, capsys):
    run_cli(capsys, "run", "config", "--work", str(scaffold), "--apply")
    out_file = scaffold / pipeline.get_stage("config").outputs[0]
    out_file.write_text(out_file.read_text() + " ")
    rc, out, _ = run_cli(capsys, "status", str(scaffold), "--json")
    stage = json.loads(out)["stages"][0]
    assert rc == 0 and stage["stale"]


def test_detached_stage_survives_and_status_reports_its_job(scaffold, capsys):
    rc, out, err = run_cli(
        capsys, "run", "config", "--work", str(scaffold), "--apply", "--detach", "--json"
    )
    assert rc == 0, err
    info = json.loads(out)
    assert info["detached"] is True and info["pid"] > 0
    deadline = time.time() + 60
    while detach.job_status(scaffold / "jobs", "config")["state"] != "done":
        assert time.time() < deadline, "detached stage did not finish"
        time.sleep(0.2)
    assert detach.job_status(scaffold / "jobs", "config")["rc"] == 0
    rc, out, _ = run_cli(capsys, "status", str(scaffold), "--json")
    stage = json.loads(out)["stages"][0]
    assert stage["status"] == "complete" and stage["jobs"][0]["state"] == "done"


def test_detach_refuses_a_stage_that_is_already_running(scaffold, capsys):
    jobs = scaffold / "jobs"
    jobs.mkdir()
    (jobs / "config.pid").write_text(f"{os.getpid()}\n")
    rc, _, err = run_cli(capsys, "run", "config", "--work", str(scaffold), "--apply", "--detach")
    assert rc == 1 and "already running" in err


# --- decide (acceptance 3) --------------------------------------------------------


@pytest.fixture
def selected(tmp_path, capsys):
    run = Run(tmp_path).through(
        "config", "preregister", "seed", "draft-heldout", "split", "snapshot"
    )
    run.through("augment", "targeted", "assemble", "train", "select")
    rc, _, err = run_cli(
        capsys, "init", DOMAIN_REF, "--work", str(tmp_path / "cfg"), *BASE, "--apply"
    )
    assert rc == 0, err
    # point the scaffold's config at the populated run
    toml = (
        (tmp_path / "cfg" / "run.toml").read_text().replace(str(tmp_path / "cfg"), str(run.workdir))
    )
    (run.workdir / "run.toml").write_text(toml)
    (run.workdir / "run.json").write_text(json.dumps({"domain": DOMAIN_REF}))
    return run


def test_decide_writes_a_record_and_prints_the_verdict_with_cited_metrics(selected, capsys):
    work = selected.workdir
    rc, out, err = run_cli(capsys, "decide", str(work))
    assert rc == 0, err
    lines = [json.loads(x) for x in (work / "decisions.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["id"] in out and rec["verdict"] in out and "cited metrics:" in out
    for cited in rec["cited"]:
        assert cited["name"] in out
    rc, out, _ = run_cli(capsys, "decide", str(work), "--json")
    assert rc == 0
    second = json.loads(out)
    assert second["verdict"] == rec["verdict"] and second["id"] != rec["id"]
    assert second["cited"] and second["decider"] == "rule"
    assert len((work / "decisions.jsonl").read_text().splitlines()) == 2
    rc, out, _ = run_cli(capsys, "status", str(work), "--json")
    assert json.loads(out)["decisions"] == 2


def test_decide_with_a_hard_stop_records_stop_escalate(selected, capsys):
    rc, out, err = run_cli(
        capsys, "decide", str(selected.workdir), "--hard-stop", "public_publish", "--json"
    )
    assert rc == 0, err
    assert json.loads(out)["verdict"] == "stop_escalate"
