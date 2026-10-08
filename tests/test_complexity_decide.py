"""Characterization tests pinned before the d17 cognitive-complexity refactor.

They pin the current behaviour (outputs, error types and message text) of the
functions refactored for Sonar S3776/S3358 in the decide, domain, config,
detach, prereg, pipeline-select and status modules, covering the branches the
rest of the suite did not reach.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from jev_factory.cli._commands import status
from jev_factory.cli._errors import CliError
from jev_factory.decide import records, rules
from jev_factory.domain import validate as v
from jev_factory.domain.model import ArgSpec
from jev_factory.factory import config as config_mod
from jev_factory.factory import detach, pipeline, prereg
from tests.fixtures.toy_domain import DOMAIN

SHA = "a" * 64


# --- status: render_update / render ----------------------------------------------


def _upd(**kw):
    base = {"at": "T", "work": "w", "running": False, "jobs": [], "stages": {}, "changes": []}
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "running, final, head",
    [
        (True, False, "update"),
        (False, False, "nothing running"),
        (True, True, "final update"),
        (False, True, "final update"),
    ],
)
def test_render_update_headline(running, final, head):
    assert status.render_update(_upd(running=running), final=final) == f"[T] w: {head}"


def test_render_update_lists_jobs_failed_stages_and_changes():
    upd = _upd(
        running=True,
        jobs=[{"job": "j1", "state": "died", "progress": None}],
        stages={"train": "failed", "select": "complete", "quantize": "failed"},
        changes=["a", "b"],
    )
    assert status.render_update(upd) == (
        "[T] w: update\n"
        "  j1: died, no progress file\n"
        "  failed stages: train, quantize\n"
        "  since last update: a; b"
    )


def test_render_marks_every_stage_status_stale_error_and_jobs():
    job = {"job": "train/r1", "state": "done", "progress": {"done": 2, "total": 3}}
    other = {"job": "loose", "state": "absent", "progress": None}
    doc = {
        "work": "w",
        "stages": [
            {"stage": "config", "status": "not-run", "stale": None, "jobs": []},
            {"stage": "seed", "status": "complete", "stale": "knob x changed", "jobs": []},
            {"stage": "split", "status": "complete", "stale": None, "jobs": []},
            {"stage": "assemble", "status": "running", "stale": "ignored", "jobs": []},
            {"stage": "train", "status": "failed", "stale": None, "error": "boom", "jobs": [job]},
        ],
        "jobs": [job, other],
        "decisions": 2,
    }
    assert status.render(doc) == "\n".join(
        [
            "run: w",
            "",
            "  config           -",
            "  seed             ok       stale: knob x changed",
            "  split            ok",
            "  assemble         running",
            "  train            failed   error: boom [train/r1: done, 2/3 items]",
            "",
            "jobs:",
            "  loose: absent, no progress file",
            "",
            "decision records: 2",
        ]
    )


def test_render_without_loose_jobs_has_no_jobs_section():
    doc = {"work": "w", "stages": [], "jobs": [], "decisions": 0}
    assert status.render(doc) == "run: w\n\n\ndecision records: 0"


# --- records: _cited_errors / validate_record ------------------------------------


def _record(**kw):
    rec = {
        "schema_version": 1,
        "id": "D1",
        "created": "2026-01-01T00:00:00+00:00",
        "run": "run-1",
        "verdict": "ship_candidate",
        "params": {},
        "reasons": ["ok"],
        "cited": [{"name": "m", "value": 0.5, "source": "s.json", "sha256": SHA}],
        "rule_version": "jev-decide-rule/1",
        "decider": "rule",
        "prev_sha256": "0" * 64,
    }
    rec.update(kw)
    return rec


def test_cited_errors_lists_every_bad_citation_field():
    bad = {"name": " ", "value": float("nan"), "source": "", "sha256": "ABC"}
    assert records._cited_errors([bad, "x", {"name": "m"}]) == [
        "cited[0].name must be a non-empty string",
        "cited[0].value must be a finite number or null",
        "cited[0].source must be a non-empty string",
        "cited[0].sha256 must be 64 lowercase hex characters",
        "cited[1] must have exactly the keys ['name', 'sha256', 'source', 'value']",
        "cited[2] must have exactly the keys ['name', 'sha256', 'source', 'value']",
    ]


@pytest.mark.parametrize("cited", [[], None, {"a": 1}])
def test_cited_errors_needs_a_non_empty_list(cited):
    assert records._cited_errors(cited) == ["cited must be a non-empty list of metric citations"]


def test_cited_errors_accepts_a_null_value():
    assert records._cited_errors([{"name": "m", "value": None, "source": "s", "sha256": SHA}]) == []


def test_validate_record_rejects_a_non_object():
    assert records.validate_record(["x"]) == ["a record must be a JSON object"]


def test_validate_record_valid_record_and_human_override():
    assert records.validate_record(_record()) == []
    human = _record(
        decider="human",
        decider_ref="ori",
        overrides="D3",
        deviation_id="d9",
        details={},
        prereg_sha256=None,
    )
    assert records.validate_record(human) == []


def test_validate_record_reports_every_problem_in_order():
    rec = _record(
        schema_version=2,
        id="X1",
        created=" ",
        run="",
        verdict="nope",
        params=[],
        reasons=["ok", ""],
        cited=[],
        rule_version=3,
        prev_sha256="z",
        prereg_sha256="short",
        details="d",
        decider="robot",
        overrides="bad",
        deviation_id=" ",
        extra=1,
    )
    assert records.validate_record(rec) == [
        "unknown fields: extra",
        "schema_version must be 1",
        "id must look like D<n>",
        "created must be an ISO-8601 timestamp string",
        "run must be a non-empty string",
        f"verdict must be one of {list(records.VERDICTS)}",
        "params must be an object",
        "reasons must be a non-empty list of non-empty strings",
        "cited must be a non-empty list of metric citations",
        "rule_version must be a non-empty string",
        "prev_sha256 must be 64 lowercase hex characters",
        "prereg_sha256 must be 64 lowercase hex characters or null",
        "details must be an object",
        f"decider must be one of {list(records.DECIDERS)}",
        "overrides must be a record id (D<n>)",
        "only a human record may override another record",
        "an override needs a deviation_id (record it with /deviate)",
        "deviation_id must be a non-empty string",
    ]


def test_validate_record_reports_missing_fields_and_decider_ref():
    assert records.validate_record({"decider": "model"}) == [
        "missing fields: schema_version, id, created, run, verdict, params, reasons, cited, "
        "rule_version, prev_sha256",
        "decider_ref is required when decider is model (model bundle / who)",
    ]
    assert records.validate_record(_record(decider="human", decider_ref=" ")) == [
        "decider_ref is required when decider is human (model bundle / who)"
    ]


@pytest.mark.parametrize("reasons", [[], "r", None])
def test_validate_record_reasons_must_be_a_non_empty_list(reasons):
    assert records.validate_record(_record(reasons=reasons)) == [
        "reasons must be a non-empty list of non-empty strings"
    ]


def test_validate_record_override_by_a_human_without_deviation():
    rec = _record(decider="human", decider_ref="ori", overrides="D1")
    assert records.validate_record(rec) == [
        "an override needs a deviation_id (record it with /deviate)"
    ]


def test_validate_record_unhashable_verdict_and_decider():
    rec = _record(verdict=["x"], decider=["rule"])
    assert records.validate_record(rec) == [
        f"verdict must be one of {list(records.VERDICTS)}",
        f"decider must be one of {list(records.DECIDERS)}",
    ]


# --- rules: CandidateSummary.__post_init__ ----------------------------------------


def _cand(**kw):
    base = dict(
        name="r1",
        source="select/r1.json",
        sha256=SHA,
        wrong_mutating=0,
        wrong_mutating_mc=0,
        right_proposals=0.97,
        permutation_change=0.02,
        ece=0.02,
        mc_escalation=0.9,
    )
    base.update(kw)
    return rules.CandidateSummary(**base)


def test_candidate_summary_accepts_every_optional_field():
    c = _cand(
        mean_confidence=0.5,
        abstain_recall=1,
        train_loss=0.1,
        epochs=2,
        train_examples=10,
        batch_size=4,
        steps=5,
        gate_fold="fit",
        gate_predictions_sha256=SHA,
        predictions_sha256=SHA,
        failure_classes={"x": 0},
    )
    assert c.epochs == 2


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"name": " "}, "summary name must be a non-empty string"),
        ({"name": 3}, "summary name must be a non-empty string"),
        ({"source": ""}, "summary r1: source must be a non-empty path"),
        ({"sha256": "x"}, "summary r1: sha256 must be 64 lowercase hex characters"),
        ({"wrong_mutating": -1}, "summary r1: wrong_mutating must be a non-negative integer count"),
        ({"wrong_mutating_mc": True}, "summary r1: wrong_mutating_mc must be a non-negative"),
        ({"ece": 1.5}, "summary r1: ece must be a number in [0, 1]"),
        ({"mc_escalation": "x"}, "summary r1: mc_escalation must be a number in [0, 1]"),
        ({"abstain_recall": 2}, "summary r1: abstain_recall must be a number in [0, 1] or null"),
        ({"epochs": 0}, "summary r1: epochs must be a positive integer or null"),
        ({"steps": 1.5}, "summary r1: steps must be a positive integer or null"),
        ({"train_loss": "x"}, "summary r1: train_loss must be a number or null"),
        ({"failure_classes": []}, "summary r1: failure_classes must map class names"),
        ({"failure_classes": {" ": 1}}, "summary r1: failure_classes must map class names"),
        ({"failure_classes": {"a": -1}}, "summary r1: failure_classes must map class names"),
        ({"gate_fold": "test"}, "summary r1: gate_fold must be 'fit' or 'selection'"),
        (
            {"predictions_sha256": "nope"},
            "summary r1 predictions_sha256: sha256 must be 64 lowercase hex characters",
        ),
        (
            {"ece": 2.0, "wrong_mutating": -1},
            "summary r1: wrong_mutating must be a non-negative integer count",
        ),
    ],
)
def test_candidate_summary_refuses(kw, message):
    with pytest.raises(CliError) as exc:
        _cand(**kw)
    assert exc.value.message.startswith(message)


# --- domain validate: _arg_problems / _text_problems ------------------------------


def _names(found):
    return [(type(e).__name__, str(e)) for e in found]


def test_arg_problems_reports_every_argument_rule():
    args = [
        ArgSpec("Bad Name", "str"),
        ArgSpec("a", "str"),
        ArgSpec("a", "weird"),
        ArgSpec("c1", "choice"),
        ArgSpec("c2", "choice", choices=("x", " ")),
        ArgSpec("c3", "choice", choices=("x", "x")),
        ArgSpec("s1", "str", choices=("x",)),
        ArgSpec("g1", "choice", choices=("x",), ground="thing"),
        ArgSpec("g2", "str", ground="undeclared"),
        ArgSpec("g3", "str", ground="known"),
    ]
    found = v._arg_problems("op", args, {"known"})
    names = _names(found)
    assert names[0][0] == "InvalidName"
    assert names[1:] == [
        ("DuplicateArgument", "'op' declares argument 'a' twice"),
        (
            "UnknownArgKind",
            "'op' argument 'a' has unknown kind 'weird' (known: "
            + ", ".join(sorted(v.ARG_KINDS))
            + ")",
        ),
        ("BadChoices", "'op' argument 'c1' needs non-empty choices"),
        ("BadChoices", "'op' argument 'c2' needs non-empty choices"),
        ("BadChoices", "'op' argument 'c3' repeats a choice"),
        ("BadChoices", "'op' argument 's1' is not a choice but lists choices"),
        ("UnknownGroundKind", "'op' argument 'g1': only a str argument is grounded"),
        ("UnknownGroundKind", "'op' argument 'g2' grounds on undeclared kind 'undeclared'"),
    ]


def test_arg_problems_clean_arguments():
    args = [ArgSpec("a", "str"), ArgSpec("b", "choice", choices=("x", "y"))]
    assert v._arg_problems("op", args, set()) == []


def test_text_problems_reports_every_text_rule():
    op = DOMAIN.operations[0].name
    domain = dataclasses.replace(
        DOMAIN,
        name=" ",
        instruction="",
        answer_policy=" ",
        explain_topics=("ok", " "),
        phrasing_styles=("",),
        control_descriptions=(("explain", " "), ("nope", "text")),
        paraphrases=((op, ()), ("ghost", ("x", " "))),
    )
    assert _names(v._text_problems(domain)) == [
        ("EmptyText", "the domain needs a name"),
        ("EmptyText", "domain ' ' has an empty instruction"),
        ("EmptyText", "domain ' ' has an empty answer_policy"),
        ("EmptyText", "domain ' ' has an empty entry in explain_topics"),
        ("EmptyText", "domain ' ' has an empty entry in phrasing_styles"),
        ("EmptyText", "control_descriptions for 'explain' is empty"),
        ("UnknownOperationRef", "control_descriptions names 'nope', not a control"),
        ("EmptyText", f"paraphrases for {op!r} include an empty text"),
        ("UnknownOperationRef", "paraphrases name unknown operation 'ghost'"),
        ("EmptyText", "paraphrases for 'ghost' include an empty text"),
    ]


def test_text_problems_clean_toy_domain():
    assert v._text_problems(DOMAIN) == []


# --- config: _coerce ---------------------------------------------------------------


def _key(kind):
    return config_mod.Key("k", kind, None, "doc")


@pytest.mark.parametrize(
    "kind, raw, expected",
    [
        ("str", "x", "x"),
        ("envname", "MY_KEY", "MY_KEY"),
        ("bool", True, True),
        ("bool", " Yes ", True),
        ("bool", "off", False),
        ("int", "7", 7),
        ("int", 3.9, 3),
        ("float", "2.5", 2.5),
        ("float", 4, 4.0),
    ],
)
def test_coerce_converts(kind, raw, expected):
    out = config_mod._coerce(_key(kind), raw, "the CLI")
    assert out == expected
    assert type(out) is type(expected)


def test_coerce_expands_a_path(monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    assert config_mod._coerce(_key("path"), "~/x", "the CLI") == "/home/someone/x"


@pytest.mark.parametrize(
    "kind, raw, why",
    [
        ("str", 3, "expected a string"),
        ("path", None, "expected a string"),
        ("envname", "sk-secret value", "expected an environment-variable NAME, not a value"),
        ("bool", "maybe", "expected a boolean"),
        ("bool", 1, "expected a boolean"),
        ("int", True, "expected an integer"),
        ("float", False, "expected a number"),
        ("int", "x", "invalid literal for int() with base 10: 'x'"),
        ("int", None, "int() argument must be a string"),
        ("float", [], "float() argument must be a string or a real number, not 'list'"),
    ],
)
def test_coerce_refuses(kind, raw, why):
    with pytest.raises(CliError) as exc:
        config_mod._coerce(_key(kind), raw, "the run file")
    assert exc.value.message.startswith(f"run config key 'k' from the run file: {why}")
    assert exc.value.remediation == f"fix k ({kind}); see docs/run-config.example.toml"
    assert exc.value.code == 1


def test_coerce_unknown_kind():
    with pytest.raises(CliError) as exc:
        config_mod._coerce(_key("complex"), "1", "the CLI")
    assert exc.value.message == "run config key 'k' has unknown kind 'complex'"


# --- detach: job_status ------------------------------------------------------------


def test_job_status_garbled_done_marker_is_not_an_rc(tmp_path):
    (tmp_path / "j.done").write_text("not-a-number\n")
    assert detach.job_status(tmp_path, "j") == {
        "state": "absent",
        "pid": None,
        "rc": None,
        "progress": None,
    }


def test_job_status_done_marker_wins(tmp_path):
    (tmp_path / "j.done").write_text("3\n")
    (tmp_path / "j.pid").write_text("999999999\n")
    assert detach.job_status(tmp_path, "j") == {
        "state": "done",
        "pid": 999999999,
        "rc": 3,
        "progress": None,
    }


def test_job_status_dead_pid_is_died(tmp_path):
    (tmp_path / "j.pid").write_text("999999999\n")
    assert detach.job_status(tmp_path, "j")["state"] == "died"


def _write_progress(tmp_path, done, total, pid):
    doc = {"done": done, "total": total, "started": 1.0, "updated": 2.0, "pid": pid}
    (tmp_path / "j.progress.json").write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "done, total, pid, state",
    [
        (3, 3, 999999999, "complete"),
        (1, 3, 999999999, "stopped"),
        (1, 3, None, "absent"),
        (1, 3, 0, "absent"),
    ],
)
def test_job_status_from_progress_only(tmp_path, done, total, pid, state):
    _write_progress(tmp_path, done, total, pid)
    assert detach.job_status(tmp_path, "j")["state"] == state


def test_job_status_progress_of_a_live_process_is_running(tmp_path):
    import os

    _write_progress(tmp_path, 1, 3, os.getpid())
    assert detach.job_status(tmp_path, "j")["state"] == "running"


# --- prereg: validate ----------------------------------------------------------------


def _prereg_doc():
    return prereg.apply_defaults(
        {
            "schema_version": 1,
            "stock_baseline_record_id": " rec-1 ",
            "perms_per_entry": 8,
            "candidates": ["r1", "r2"],
            "bars": {
                "wrong_mutating": {"stock": 0.02, "minimum": 0.0},
                "ece": {"stock": 0.25, "minimum": 0.03},
                "permutation_change": {"stock": 0.3, "minimum": 0.05},
                "mc_escalation": {"stock": 0.1, "minimum": 0.8},
                "right_proposals": {"stock": 0.5},
            },
        }
    )


def test_prereg_validate_builds_the_registration():
    doc = _prereg_doc()
    got = prereg.validate(doc)
    assert got.stock_baseline_record_id == "rec-1"
    assert got.candidates == ("r1", "r2")
    assert got.stock_model == "Qwen/Qwen3.5-0.8B"
    assert got.raw is doc
    assert got.bars["mc_escalation"] == prereg.Bar("mc_escalation", 0.1, 0.8, 0.8, "higher")


def _broken(mutate):
    doc = _prereg_doc()
    mutate(doc)
    return doc


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(schema_version=2), "schema_version must be 1"),
        (lambda d: d.update(stock_baseline_record_id=" "), "stock_baseline_record_id is required"),
        (lambda d: d.update(bars=[]), "bars must be an object"),
        (lambda d: d["bars"].pop("ece"), "bars missing: ece"),
        (lambda d: d["bars"].update(zz={}, aa={}), "unknown bars: aa, zz"),
        (lambda d: d["bars"].update(ece=1), "bar ece must be an object"),
        (
            lambda d: d["bars"]["ece"].update(minimum=2),
            "bar ece.minimum must be a number in [0, 1]",
        ),
        (lambda d: d["bars"]["ece"].pop("bar"), "bar ece.bar must be a number in [0, 1]"),
        (
            lambda d: d["bars"]["ece"].update(bar=0.2),
            "bar ece.bar is 0.2 but the stricter of stock and minimum is 0.03",
        ),
        (lambda d: d.update(perms_per_entry=True), "perms_per_entry must be a positive integer"),
        (lambda d: d.update(perms_per_entry=0), "perms_per_entry must be a positive integer"),
        (lambda d: d.update(rule_order=["x"]), "rule_order must be exactly"),
        (lambda d: d.update(tolerances=[]), "tolerances must define exactly"),
        (lambda d: d["tolerances"].update(extra=0.1), "tolerances must define exactly"),
        (
            lambda d: d["tolerances"].update({next(iter(d["tolerances"])): 5}),
            "tolerances.",
        ),
        (lambda d: d.update(candidates=[]), "candidates must be a non-empty list of names"),
        (lambda d: d.update(candidates="r1"), "candidates must be a non-empty list of names"),
        (lambda d: d.update(candidates=["r1", " "]), "candidate names must be non-empty strings"),
        (lambda d: d.update(candidates=["r1", "r1"]), "candidate names must be unique"),
    ],
)
def test_prereg_validate_refuses(mutate, message):
    doc = _broken(mutate)
    with pytest.raises(CliError) as exc:
        prereg.validate(doc)
    assert exc.value.message.startswith(message)


def test_prereg_validate_refuses_a_non_object():
    with pytest.raises(CliError) as exc:
        prereg.validate([])
    assert exc.value.message == "pre-registration must be a JSON object"


# --- pipeline: stage_select (missing predictions, undefined ECE) --------------------


@pytest.fixture
def trained(tmp_path):
    from tests.test_factory_pipeline import CHAIN, DATA, Run

    return Run(tmp_path).through(*CHAIN, *DATA, "train")


def test_select_names_every_candidate_without_validation_predictions(trained):
    import shutil

    shutil.rmtree(trained.workdir / pipeline.CANDIDATES_DIR / "r1" / pipeline.VAL_MC_DIR)
    shutil.rmtree(trained.workdir / pipeline.CANDIDATES_DIR / "r2" / pipeline.VAL_DIR)
    error = trained.fail("select")
    assert "no validation predictions for pre-registered candidate(s): r1, r2" in error
    assert not (trained.workdir / pipeline.SELECT_DIR / "selection.json").exists()


def test_select_writes_the_summaries_it_can_before_refusing(trained):
    import shutil

    shutil.rmtree(trained.workdir / pipeline.CANDIDATES_DIR / "r2" / pipeline.VAL_DIR)
    error = trained.fail("select")
    assert "no validation predictions for pre-registered candidate(s): r2" in error
    assert (trained.workdir / pipeline.SELECT_DIR / "r1" / pipeline.SUMMARY_FILE).is_file()
    assert not (trained.workdir / pipeline.SELECT_DIR / "r2").exists()


def test_select_refuses_an_undefined_ece(trained, monkeypatch):
    real = pipeline.calibrate_and_gate

    def no_ece(*args, **kw):
        fitted = real(*args, **kw)
        fitted["numbers"]["ece"] = None
        return fitted

    monkeypatch.setattr(pipeline, "calibrate_and_gate", no_ece)
    error = trained.fail("select")
    assert "r1: no selection-fold line has a distribution; ECE is undefined" in error
