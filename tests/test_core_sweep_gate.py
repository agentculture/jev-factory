"""jev_factory.core.sweep_gate, ported from nvsh's tests/test_lfm_finetune_gate.py (sweep half).

The Domain is explicit (``redecide``/``evaluate``/``run_sweep`` take it), and the
toy lamp domain stands in for nvsh's operation table: ``lamp_status`` and
``list_rooms`` are read-only, ``set_scene`` is mutating. Operation names appear
only as fixture data; a grep test guards the sweep's source for that. t16 adds:
a newly chosen operation is marked not grounded, ``--final`` takes one value per
grid knob, and the gate is fit on the fit fold only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_factory.core import calibration, gate, metrics
from jev_factory.core import sweep_gate as sweep
from jev_factory.core.predictions import Prediction
from tests.fixtures.toy_domain import DOMAIN

DOMAIN_PATH = "tests.fixtures.toy_domain"
ROOT = Path(__file__).resolve().parent.parent

# Real toy operation names, used only as fixture data.
READ_ONLY_OP = "lamp_status"
READ_ONLY_OP_2 = "list_rooms"
MUTATING_OP = "set_scene"
MUTATING_ARGS = {"scene": "bright"}


def _op(name: str, **args) -> dict:
    return {"operation": name, "args": dict(args)}


def _line(
    id_: str,
    expected: dict,
    outcome: str,
    operation: str | None,
    arguments: dict | None,
    candidates: dict | None,
) -> dict:
    return {
        "id": id_,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": 0,
        "ttfd_ms": 1.0,
        "latency_ms": 1.0,
    }


# ---------------------------------------------------------------------------
# sweep_gate: grid parsing
# ---------------------------------------------------------------------------


def test_parse_grid_none():
    assert sweep.parse_grid("none") == [None]


def test_parse_grid_numbers():
    assert sweep.parse_grid("0.5, 0.7") == [0.5, 0.7]


def test_parse_grid_mixed():
    assert sweep.parse_grid("none,0.5") == [None, 0.5]


def test_parse_grid_rejects_garbage():
    with pytest.raises(sweep.SweepError):
        sweep.parse_grid("bogus")


def test_parse_grid_rejects_empty():
    with pytest.raises(sweep.SweepError):
        sweep.parse_grid("")


def test_build_threshold_grid_cartesian_size():
    grid = sweep.build_threshold_grid(
        escalate=[None, 0.5],
        ro_floor=[None, 0.6],
        ro_margin=[None],
        ro_max_entropy=[None],
    )
    # escalate(2) x ro_floor(2) x ro_margin(1) x ro_entropy(1), each squared again
    # by mut_* defaulting to the same ro_* grid (see build_threshold_grid's docstring).
    assert len(grid) == 8
    assert all(isinstance(t, gate.Thresholds) for t in grid)


def test_build_threshold_grid_mutating_defaults_to_read_only():
    grid = sweep.build_threshold_grid(
        escalate=[None], ro_floor=[0.6], ro_margin=[None], ro_max_entropy=[None]
    )
    (thresholds,) = grid
    assert thresholds.mutating.floor == 0.6


def test_build_threshold_grid_mutating_override():
    grid = sweep.build_threshold_grid(
        escalate=[None],
        ro_floor=[0.6],
        ro_margin=[None],
        ro_max_entropy=[None],
        mut_floor=[0.9],
    )
    (thresholds,) = grid
    assert thresholds.read_only.floor == 0.6
    assert thresholds.mutating.floor == 0.9


# ---------------------------------------------------------------------------
# sweep_gate: split safety
# ---------------------------------------------------------------------------


def test_refuse_unless_final_allows_a_plain_path(tmp_path):
    path = tmp_path / "dev-predictions.jsonl"
    sweep.refuse_unless_final(path, final=False)  # no raise


def test_refuse_unless_final_blocks_test_named_file(tmp_path):
    path = tmp_path / "test-predictions.jsonl"
    with pytest.raises(sweep.SweepError):
        sweep.refuse_unless_final(path, final=False)


def test_refuse_unless_final_blocks_final_directory(tmp_path):
    final_dir = tmp_path / "final" / "scorer-b1"
    final_dir.mkdir(parents=True)
    path = final_dir / "scorer-b1.predictions.jsonl"
    with pytest.raises(sweep.SweepError):
        sweep.refuse_unless_final(path, final=False)


def test_refuse_unless_final_allows_final_with_flag(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    path = final_dir / "final-scorer-b1.predictions.jsonl"
    sweep.refuse_unless_final(path, final=True)  # no raise


# ---------------------------------------------------------------------------
# sweep_gate: redecide()
# ---------------------------------------------------------------------------


def test_redecide_passes_through_a_null_candidates_line():
    prediction = Prediction.from_dict(
        _line("x1", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, None)
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result is prediction


def test_redecide_reproduces_a_propose_line_under_disabled_thresholds():
    candidates = {READ_ONLY_OP: 0.9, "(explain)": 0.05, "(escalate)": 0.05}
    prediction = Prediction.from_dict(
        _line("x2", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result.outcome == "propose"
    assert result.operation == READ_ONLY_OP
    assert result.arguments == {}


def test_redecide_can_turn_a_propose_line_into_abstain_uncertain():
    candidates = {READ_ONLY_OP: 0.6, "(explain)": 0.4}
    prediction = Prediction.from_dict(
        _line("x3", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, candidates)
    )
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.8))
    result = sweep.redecide(prediction, thresholds, DOMAIN)
    assert result.outcome == "abstain_uncertain"
    assert result.operation is None
    assert result.arguments is None


# ---------------------------------------------------------------------------
# sweep_gate: redecide() never invents {} arguments for an invalid original
# (issue 53, P2)
# ---------------------------------------------------------------------------


def test_redecide_keeps_an_invalid_line_invalid_under_disabled_thresholds():
    """A complete distribution whose winning operation could not be grounded stays invalid.

    Before the fix, disabling every threshold still turned this into a
    fabricated ``propose`` with ``{}`` arguments -- with a complete
    distribution and nothing overriding the argmax, ``gate.decide`` always
    proposes it, and ``redecide`` used to trust that blindly.
    """
    candidates = {READ_ONLY_OP: 0.93, READ_ONLY_OP_2: 0.05, "(explain)": 0.02}
    prediction = Prediction.from_dict(
        _line("invalid-1", _op(READ_ONLY_OP), "invalid", None, None, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == metrics.invalid_reason(prediction, DOMAIN)


def test_redecide_keeps_a_functionally_invalid_propose_invalid():
    """A "propose" line whose own arguments already fail the operation table stays invalid.

    ``metrics.invalid_reason`` calls this kind of line invalid even though
    its stored ``outcome`` field says ``"propose"``; redecide must not
    launder it into a clean propose just because the gate's argmax agrees
    with its operation.
    """
    candidates = {MUTATING_OP: 0.9, "(escalate)": 0.1}
    prediction = Prediction.from_dict(
        _line(
            "invalid-2", _op(MUTATING_OP, **MUTATING_ARGS), "propose", MUTATING_OP, {}, candidates
        )
    )
    assert metrics.invalid_reason(prediction, DOMAIN) is not None  # bad args: {} for set_scene
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == metrics.invalid_reason(prediction, DOMAIN)


def test_redecide_marks_not_grounded_by_sweep_for_a_different_operation():
    """The gate picking a different operation than the original never invents arguments.

    The stored line disagrees with its own candidates' argmax (a
    corrupted/tie-broken record) -- an edge case, but redecide must still
    refuse to attach the *original*'s arguments to the operation the gate
    actually picked, or to invent new ones.
    """
    candidates = {READ_ONLY_OP: 0.7, READ_ONLY_OP_2: 0.3}
    prediction = Prediction.from_dict(
        _line("mismatch", _op(READ_ONLY_OP_2), "propose", READ_ONLY_OP_2, {}, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == sweep.NOT_GROUNDED_BY_SWEEP


def test_sweep_reproduces_an_invalid_line_exactly_when_disabled():
    """The scorer-b1-shaped fixture plus an invalid line: 100% outcome+operation match."""
    lines = _scorer_b1_style_fixture() + [
        _line(
            "b1-06-invalid",
            _op(READ_ONLY_OP),
            "invalid",
            None,
            None,
            {READ_ONLY_OP: 0.93, READ_ONLY_OP_2: 0.05, "(explain)": 0.02},
        ),
    ]
    originals = [Prediction.from_dict(line) for line in lines]
    for original in originals:
        redecided = sweep.redecide(original, gate.Thresholds(), DOMAIN)
        assert redecided.outcome == original.outcome, original.id
        assert redecided.operation == original.operation, original.id


# ---------------------------------------------------------------------------
# sweep_gate: the exact-reproduction acceptance test
# ---------------------------------------------------------------------------


def _scorer_b1_style_fixture() -> list[dict]:
    """A handful of lines shaped like scorer-b1's stored exact predictions.

    Each line's ``outcome``/``operation`` already equals what its own
    ``candidates`` argmax would give -- exactly how a scorer.py-produced
    predictions file records a decision with no gate involved.
    """
    return [
        _line(
            "b1-01",
            _op(READ_ONLY_OP),
            "propose",
            READ_ONLY_OP,
            {},
            {READ_ONLY_OP: 0.97, READ_ONLY_OP_2: 0.02, "(explain)": 0.01},
        ),
        _line(
            "b1-02",
            _op(MUTATING_OP, **MUTATING_ARGS),
            "propose",
            MUTATING_OP,
            dict(MUTATING_ARGS),
            {MUTATING_OP: 0.88, "(escalate)": 0.12},
        ),
        _line(
            "b1-03",
            {"explain": True},
            "explain",
            None,
            None,
            {"(explain)": 0.7, READ_ONLY_OP: 0.3},
        ),
        _line(
            "b1-04",
            {"escalate": True},
            "escalate",
            None,
            None,
            {"(escalate)": 0.65, READ_ONLY_OP: 0.35},
        ),
        _line(
            "b1-05",
            _op(READ_ONLY_OP_2),
            "propose",
            READ_ONLY_OP_2,
            {},
            {READ_ONLY_OP_2: 0.51, READ_ONLY_OP: 0.49},
        ),
    ]


def test_sweep_reproduces_scorer_b1_argmax_exactly_when_disabled(tmp_path):
    lines = _scorer_b1_style_fixture()
    path = tmp_path / "scorer-b1-fixture.predictions.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    originals = metrics.read_predictions(path)
    matches, total = sweep.count_argmax_matches(originals, DOMAIN)
    assert total == len(lines)
    assert matches == total

    for original in originals:
        redecided = sweep.redecide(original, gate.Thresholds(), DOMAIN)
        assert redecided.outcome == original.outcome
        assert redecided.operation == original.operation


def test_evaluate_and_run_sweep_shape():
    originals = [Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    grid = [gate.Thresholds(), gate.Thresholds(read_only=gate.ThresholdSet(floor=0.99))]
    reports = sweep.run_sweep(originals, grid, DOMAIN)
    assert len(reports) == 2
    for report in reports:
        assert set(report) >= {
            "thresholds",
            "right_proposals",
            "wrong_mutating",
            "abstain_uncertain_count",
            "escalation",
            "false_positives",
            "missing_candidate_escalation_recall",
        }


def test_missing_candidate_escalation_recall_improves_with_a_threshold():
    # A -nocand line the scorer confidently (wrongly) proposed on.
    line = _line(
        "case-nocand",
        _op(READ_ONLY_OP),
        "propose",
        READ_ONLY_OP_2,
        {},
        {READ_ONLY_OP_2: 0.6, "(explain)": 0.4},
    )
    prediction = Prediction.from_dict(line)
    assert metrics.is_missing_candidate(prediction)

    disabled = sweep._missing_candidate_recall(
        [prediction], [sweep.redecide(prediction, gate.Thresholds(), DOMAIN)]
    )
    assert disabled == {"n": 0, "N": 1, "rate": 0.0}

    tight = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.7))
    gated = sweep._missing_candidate_recall(
        [prediction], [sweep.redecide(prediction, tight, DOMAIN)]
    )
    assert gated == {"n": 1, "N": 1, "rate": 1.0}


# ---------------------------------------------------------------------------
# sweep_gate: folds
# ---------------------------------------------------------------------------


def test_filter_by_fold_restricts_to_the_named_fold():
    originals = [Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    folds = {"fit_ids": ["b1-01", "b1-02"], "selection_ids": ["b1-03", "b1-04", "b1-05"]}
    fit = sweep.filter_by_fold(originals, folds, "fit")
    assert {p.id for p in fit} == {"b1-01", "b1-02"}
    selection = sweep.filter_by_fold(originals, folds, "selection")
    assert {p.id for p in selection} == {"b1-03", "b1-04", "b1-05"}


def test_filter_by_fold_none_folds_keeps_everything():
    originals = [Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    assert sweep.filter_by_fold(originals, None, None) == originals


def test_fold_ids_rejects_unknown_fold_name():
    with pytest.raises(sweep.SweepError):
        sweep.fold_ids({"fit_ids": [], "selection_ids": []}, "bogus")


# ---------------------------------------------------------------------------
# sweep_gate: CLI (main)
# ---------------------------------------------------------------------------


def _write_predictions(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    lines = _scorer_b1_style_fixture()
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_main_default_grid_is_a_single_disabled_combo(tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    rc = sweep.main(["--domain", DOMAIN_PATH, "--predictions", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 5
    assert len(out["reports"]) == 1
    assert out["reports"][0]["thresholds"] == gate_thresholds_json()


def gate_thresholds_json():
    return {
        "escalate": None,
        "read_only": {"floor": None, "margin": None, "max_entropy": None},
        "mutating": {"floor": None, "margin": None, "max_entropy": None},
    }


def test_main_refuses_a_test_named_file_without_final(tmp_path, capsys):
    path = _write_predictions(tmp_path, "test-predictions.jsonl")
    rc = sweep.main(["--domain", DOMAIN_PATH, "--predictions", str(path)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_main_allows_test_named_file_with_final(tmp_path, capsys):
    path = _write_predictions(tmp_path, "test-predictions.jsonl")
    rc = sweep.main(["--domain", DOMAIN_PATH, "--predictions", str(path), "--final"])
    assert rc == 0


def test_main_writes_markdown_when_asked(tmp_path):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    markdown_path = tmp_path / "report.md"
    rc = sweep.main(
        [
            "--domain",
            DOMAIN_PATH,
            "--predictions",
            str(path),
            "--out",
            str(tmp_path / "out.json"),
            "--markdown",
            str(markdown_path),
        ]
    )
    assert rc == 0
    text = markdown_path.read_text(encoding="utf-8")
    assert text.startswith("| escalate |")


def test_main_grid_flags_expand_the_report(tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    rc = sweep.main(["--domain", DOMAIN_PATH, "--predictions", str(path), "--floor", "none,0.99"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # --floor sets both ro_floor and (by default) mut_floor to the same 2-value
    # grid, so the two floor axes multiply: 2 x 2 = 4 combinations.
    assert len(out["reports"]) == 4


def test_main_fold_filters_the_report_count(tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps({"fit_ids": ["b1-01", "b1-02"], "selection_ids": ["b1-03", "b1-04", "b1-05"]}),
        encoding="utf-8",
    )
    rc = sweep.main(
        [
            "--domain",
            DOMAIN_PATH,
            "--predictions",
            str(path),
            "--folds",
            str(folds_path),
            "--fold",
            "fit",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 2
    assert out["fold"] == "fit"


# ---------------------------------------------------------------------------
# t16: newly chosen operations are marked not grounded
# ---------------------------------------------------------------------------


def test_a_newly_chosen_operation_is_marked_not_grounded():
    candidates = {READ_ONLY_OP: 0.7, READ_ONLY_OP_2: 0.3}
    prediction = Prediction.from_dict(
        dict(
            _line("m", _op(READ_ONLY_OP_2), "propose", READ_ONLY_OP_2, {}, candidates),
            grounded=True,
        )
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert result.grounded is False
    assert result.invalid_reason == sweep.NOT_GROUNDED_BY_SWEEP
    assert result.arguments is None


def test_a_kept_proposal_keeps_its_grounding_and_arguments():
    candidates = {MUTATING_OP: 0.9, "(escalate)": 0.1}
    prediction = Prediction.from_dict(
        dict(
            _line(
                "k",
                _op(MUTATING_OP, **MUTATING_ARGS),
                "propose",
                MUTATING_OP,
                MUTATING_ARGS,
                candidates,
            ),
            grounded=True,
        )
    )
    result = sweep.redecide(prediction, gate.Thresholds(), DOMAIN)
    assert (result.outcome, result.operation, result.arguments) == (
        "propose",
        MUTATING_OP,
        MUTATING_ARGS,
    )
    assert result.grounded is True


def test_redecide_changes_the_decision_only_not_the_distribution():
    candidates = {READ_ONLY_OP: 0.6, "(explain)": 0.4}
    prediction = Prediction.from_dict(
        dict(
            _line("d", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, candidates),
            raw_probabilities={READ_ONLY_OP: 0.8, "(explain)": 0.2},
            offered=["(explain)", READ_ONLY_OP],
        )
    )
    result = sweep.redecide(
        prediction, gate.Thresholds(read_only=gate.ThresholdSet(floor=0.9)), DOMAIN
    )
    assert result.outcome == "abstain_uncertain"
    assert result.candidates == prediction.candidates
    assert result.raw_probabilities == prediction.raw_probabilities
    assert result.offered == prediction.offered


# ---------------------------------------------------------------------------
# t16: --final takes one value per grid knob
# ---------------------------------------------------------------------------


def test_require_single_point_accepts_one_value_per_knob():
    sweep.require_single_point(escalate=[None], floor=[0.6], mutating_floor=None)


def test_require_single_point_refuses_a_multi_value_knob():
    with pytest.raises(sweep.SweepError, match="--floor"):
        sweep.require_single_point(escalate=[None], floor=[0.5, 0.6])


def test_main_final_refuses_a_grid(tmp_path, capsys):
    path = _write_predictions(tmp_path, "final-predictions.jsonl")
    rc = sweep.main(
        ["--domain", DOMAIN_PATH, "--predictions", str(path), "--final", "--floor", "none,0.99"]
    )
    assert rc == 1
    assert "one value per grid knob" in capsys.readouterr().err


def test_main_final_refuses_a_multi_value_mutating_knob(tmp_path, capsys):
    path = _write_predictions(tmp_path, "final-predictions.jsonl")
    rc = sweep.main(
        [
            "--domain",
            DOMAIN_PATH,
            "--predictions",
            str(path),
            "--final",
            "--mutating-margin",
            "0.1,0.2",
        ]
    )
    assert rc == 1
    assert "--mutating-margin" in capsys.readouterr().err


def test_main_final_accepts_one_value_per_knob(tmp_path, capsys):
    path = _write_predictions(tmp_path, "final-predictions.jsonl")
    rc = sweep.main(
        ["--domain", DOMAIN_PATH, "--predictions", str(path), "--final", "--floor", "0.9"]
    )
    assert rc == 0
    assert len(json.loads(capsys.readouterr().out)["reports"]) == 1


def test_main_needs_a_domain(tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    assert sweep.main(["--predictions", str(path)]) == 1
    assert "Domain" in capsys.readouterr().err


def test_main_accepts_a_domain_object(tmp_path):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    assert sweep.main(["--predictions", str(path)], DOMAIN) == 0


# ---------------------------------------------------------------------------
# t16: the gate is fit on the fit fold only
# ---------------------------------------------------------------------------


def _fold_lines() -> tuple[list[dict], dict]:
    lines = []
    for n in range(6):
        lines.append(
            _line(
                f"fit{n}",
                _op(READ_ONLY_OP),
                "propose",
                READ_ONLY_OP,
                {},
                {READ_ONLY_OP: 0.6, "(explain)": 0.4},
            )
        )
    for n in range(6):
        lines.append(
            _line(
                f"sel{n}",
                _op(READ_ONLY_OP),
                "propose",
                READ_ONLY_OP,
                {},
                {READ_ONLY_OP: 0.95, "(explain)": 0.05},
            )
        )
    folds = {
        "fit_ids": [f"fit{n}" for n in range(6)],
        "selection_ids": [f"sel{n}" for n in range(6)],
    }
    return lines, folds


GRID = [gate.Thresholds(), gate.Thresholds(read_only=gate.ThresholdSet(floor=0.8))]


def _most_right(reports):
    return max(range(len(reports)), key=lambda i: reports[i]["right_proposals"]["n"])


def test_fit_gate_scores_only_the_fit_fold():
    lines, folds = _fold_lines()
    records = [Prediction.from_dict(line) for line in lines]
    seen = []

    def choose(reports):
        seen.append(reports)
        return _most_right(reports)

    chosen, reports = sweep.fit_gate(records, folds, GRID, DOMAIN, choose)
    assert chosen == GRID[0]
    # 6 fit proposals, all right, in every report: selection rows never counted.
    assert all(r["right_proposals"]["N"] == 6 for r in reports)


def test_fit_gate_ignores_selection_fold_outcomes():
    lines, folds = _fold_lines()
    baseline = sweep.fit_gate(
        [Prediction.from_dict(line) for line in lines], folds, GRID, DOMAIN, _most_right
    )
    flipped = []
    for line in lines:
        if line["id"].startswith("sel"):
            line = dict(
                line,
                expected={"escalate": True},
                candidates={"(escalate)": 1.0},
                outcome="escalate",
                operation=None,
                arguments=None,
            )
        flipped.append(Prediction.from_dict(line))
    again = sweep.fit_gate(flipped, folds, GRID, DOMAIN, _most_right)
    assert again == baseline


def test_fit_gate_refuses_an_empty_fit_fold_and_a_bad_choice():
    lines, folds = _fold_lines()
    records = [Prediction.from_dict(line) for line in lines]
    with pytest.raises(sweep.SweepError):
        sweep.fit_gate(
            records, {"fit_ids": ["nope"], "selection_ids": []}, GRID, DOMAIN, _most_right
        )
    with pytest.raises(sweep.SweepError):
        sweep.fit_gate(records, folds, GRID, DOMAIN, lambda reports: 99)
    with pytest.raises(sweep.SweepError):
        sweep.fit_gate(records, folds, [], DOMAIN, _most_right)


# ---------------------------------------------------------------------------
# No operation names in the sweep/calibration source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [sweep, calibration])
def test_no_operation_names_in_source(module):
    text = Path(module.__file__).read_text(encoding="utf-8")
    for name in DOMAIN.names():
        assert name not in text, f"{module.__name__} names the operation {name!r}"


# ---------------------------------------------------------------------------
# o9: distribution-only stages run with no tokenizer, model or backbone adapter
# ---------------------------------------------------------------------------

_BLOCKED = (
    "torch",
    "transformers",
    "peft",
    "unsloth",
    "llmcompressor",
    "huggingface_hub",
    "datasets",
    "deepeval",
    "tokenizers",
    "vllm",
    "jev_factory.backbones",
    "nvsh",
    "jetson_skills",
)

_RUN = r"""
import json, sys
for name in {blocked!r}:
    sys.modules[name] = None  # any import of it raises ImportError
from jev_factory.core import calibration, gate, predictions, sweep_gate
from tests.fixtures.toy_domain import DOMAIN

rows = []
for n in range(10):
    right = n % 3 != 0
    top, other = ("lamp_status", "list_rooms") if right else ("list_rooms", "lamp_status")
    rows.append({{
        "id": f"r{{n}}", "expected": {{"operation": "lamp_status", "args": {{}}}},
        "outcome": "propose", "operation": top, "arguments": {{}},
        "candidates": {{top: 0.9, other: 0.1}}, "grounded": True,
        "tokens": 0, "ttfd_ms": 1.0, "latency_ms": 1.0,
    }})
records = predictions.from_rows(rows)
folds = {{"seed": 1, "fit_ids": [f"r{{n}}" for n in range(7)],
         "selection_ids": [f"r{{n}}" for n in range(7, 10)]}}
params = calibration.fit_params_from_predictions(records, folds)
chosen = calibration.select_calibration(records, params, folds)
calibrated = calibration.apply_to_predictions(records, chosen)
reports = sweep_gate.run_sweep(
    calibrated, [gate.Thresholds(read_only=gate.ThresholdSet(floor=0.5))], DOMAIN
)
leaked = sorted(
    m for m, mod in sys.modules.items() if m.startswith("jev_factory.backbones") and mod
)
print(json.dumps({{
    "temperature": params["temperature"], "vector_kept": chosen["vector_kept"],
    "raw_kept": calibrated[0].raw_probabilities is not None,
    "reports": len(reports), "leaked": leaked,
}}))
"""


@pytest.mark.behavioral("o9")
def test_calibrate_and_gate_sweep_run_with_no_model_tokenizer_or_backbone():
    proc = subprocess.run(
        [sys.executable, "-c", _RUN.format(blocked=_BLOCKED)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["temperature"] > 0
    assert out["raw_kept"] is True
    assert out["reports"] == 1
    assert out["leaked"] == []


@pytest.mark.behavioral("o9")
@pytest.mark.parametrize("module", [sweep, calibration])
def test_neither_stage_imports_the_backbone_adapter(module):
    import ast

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    assert not any(n.startswith("jev_factory.backbones") for n in imported), imported
