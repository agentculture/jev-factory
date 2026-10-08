"""The measure stage's scorer path: jev_factory/measure/run.py on the toy lamp domain.

Ported from nvsh's tests/test_lfm_finetune_measure.py (the Track B scorer,
split-guard, P71, tier-error, calibration and reasons-mode tests), plus the
once-only rule for the sealed sides. Everything that would touch the machine
goes through ``run.Seams``: a fake top-k scorer, a fake ``nvidia-smi``
runner and a no-op GPU guard, so no model is loaded and nothing is served
(the served tests talk to a localhost ``http.server`` thread).
"""

from __future__ import annotations

import hashlib
import http.server
import itertools
import json
import math
import re
import threading
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm.readout import LABEL_ALPHABET, READOUT_TOP
from jev_factory.core.calibration import apply_scaling
from jev_factory.core.predictions import Prediction
from jev_factory.measure import once
from jev_factory.measure import run as measure
from tests.fixtures.stub_servers import path_with, stub_bin
from tests.fixtures.toy_domain import DOMAIN, SEED_CORPUS

DOMAIN_ARG = "tests.fixtures.toy_domain"
WORLD = {"home": "toy-home", "rooms": ["kitchen", "bedroom", "study", "hallway"]}
STOCK = "toy-base"
TUNED = "toy-tuned"
LINE_RE = re.compile(r"^([A-Za-z])\) (\S+): ", re.MULTILINE)


def render(messages: list[dict]) -> str:
    """A trivial renderer: the messages' contents, one per line (no chat template)."""
    return "\n".join(message["content"] for message in messages)


def letters_in(prompt: str) -> dict[str, str]:
    """``name -> letter`` parsed back out of a rendered prompt."""
    return {name: letter for letter, name in LINE_RE.findall(prompt)}


def favour(prompt: str, winner: str, drop: str | None = None) -> dict[str, float]:
    """Every offered letter, *winner*'s at 0.9; *drop* names a letter left out."""
    letters = letters_in(prompt)
    rest = 0.1 / (len(letters) - 1)
    return {
        letter: math.log(0.9 if name == winner else rest)
        for name, letter in letters.items()
        if name != drop
    }


def by_request(picks: dict[str, str], default: str = "escalate"):
    """A pick function: the candidate for the first substring of the request that matches."""

    def pick(prompt: str) -> dict[str, float]:
        request = prompt.rsplit("\n", 1)[-1]
        winner = next((name for key, name in picks.items() if key in request), default)
        return favour(prompt, winner)

    return pick


class FakeTopK:
    def __init__(self, pick) -> None:
        self.pick = pick
        self.prompts: list[str] = []
        self.tops: list[int] = []

    def __call__(self, prompt: str, top: int) -> dict[str, float]:
        self.prompts.append(prompt)
        self.tops.append(top)
        return self.pick(prompt)


def fake_run(argv: list[str], timeout: float) -> tuple[int, str]:
    del timeout
    if argv == ["nvidia-smi"]:
        return (0, "NVIDIA-SMI stub   GB10\n")
    return (127, "not found")


class Harness:
    def __init__(self, pick=None, fail: str = "") -> None:
        self.fake = FakeTopK(pick or by_request({}))
        self.fail = fail
        self.built: list[measure.ScorerSpec] = []
        self.closed = 0
        counter = itertools.count()
        self.guarded: list[tuple] = []
        self.seams = measure.Seams(
            run=fake_run,
            today=lambda: "2026-09-30",
            clock=lambda: next(counter) * 0.25,
            build_scorer=self.build,
            preflight=lambda base_url, model, ctx=None: None,
            gpu_guard=lambda stage, allow_foreign=False: self.guarded.append(
                (stage, allow_foreign)
            ),
        )

    def build(self, spec: measure.ScorerSpec) -> measure.ScorerHandle:
        self.built.append(spec)
        if self.fail:
            raise ImportError(self.fail)

        def close() -> None:
            self.closed += 1

        return measure.ScorerHandle(
            top_k=self.fake, render=render, close=close, base_url=spec.base_url
        )


def _entry(entry_id: str, expect: dict, text: str, source_id: str | None = None) -> dict:
    return {
        "id": entry_id,
        "source_id": source_id or entry_id,
        "kind": "explicit",
        "text": text,
        "expect": expect,
        "source": "fixture",
    }


def _entries() -> list[dict]:
    return [
        _entry("on1", {"operation": "lamp_on", "args": {"room": "kitchen"}}, "turn on the kitchen"),
        _entry("st1", {"operation": "room_status", "args": {"room": "study"}}, "is it lit there"),
        _entry("esc1", {"escalate": True}, "what is the weather tomorrow"),
        _entry("exp1", {"explain": True}, "what does dimming mean"),
    ]


def _split(tmp_path: Path, name: str = "val.json", header: str | None = None, entries=None):
    header = header or "Toy corpus. Split 'val' of toy.json (seed=39)."
    path = tmp_path / "splits" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"header": header, "entries": entries if entries is not None else _entries()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _snapshot(tmp_path: Path, rooms=("kitchen", "bedroom", "study", "hallway")) -> Path:
    path = tmp_path / "snapshot.json"
    payload = {"home": "toy-home", "rooms": list(rooms), "source": "fixture", "created": "x"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _argv(tmp_path: Path, split: Path, *extra: str, models=(STOCK,), label="val-r1") -> list:
    argv = ["--domain", DOMAIN_ARG, "--run-dir", str(tmp_path / "run"), "--split", str(split)]
    argv += ["--label", label, "--scorer", "in-process"]
    for index, model in enumerate(models):
        argv += ["--model", model, "--revision", f"rev{index}"]
    return argv + list(extra)


def _page(tmp_path: Path, label: str = "val-r1") -> Path:
    return tmp_path / "run" / "measure" / f"2026-09-30-{label}.md"


def _lines(tmp_path: Path, label: str = "val-r1", index: int = 1, model: str = STOCK) -> list:
    path = tmp_path / "run" / "measure" / label / f"{label}-{index}-{model}.predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _row(text: str, metric: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(f"| {metric} |"):
            return [cell.strip() for cell in line.strip().strip("|").split("|")[1:]]
    raise AssertionError(f"no row {metric!r}")


PICKS = {"kitchen": "lamp_on", "lit there": "room_status", "dimming": "explain"}


# ---------------------------------------------------------------------------
# The predictions file
# ---------------------------------------------------------------------------


def test_scorer_run_writes_the_shared_predictions_file(tmp_path):
    harness = Harness(by_request(PICKS))
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 0
    on1, st1, esc1, exp1 = _lines(tmp_path)
    for row in (on1, st1, esc1, exp1):
        Prediction.from_dict(row)
        assert row["tokens"] == 0
    assert (on1["outcome"], on1["operation"], on1["arguments"]) == (
        "propose",
        "lamp_on",
        {"room": "kitchen"},
    )
    assert on1["grounded"] is True
    assert on1["candidates"]["lamp_on"] == pytest.approx(0.9)
    assert on1["raw_probabilities"] == on1["candidates"]
    assert on1["offered"] == [*DOMAIN.names(), "(explain)", "(escalate)"]
    assert set(on1["raw_scores"]) == set(on1["offered"])
    # "is it lit there" names no room: the proposal cannot be grounded.
    assert (st1["outcome"], st1["invalid_reason"], st1["grounded"]) == (
        "invalid",
        "not_grounded",
        False,
    )
    assert st1["operation"] is None
    assert st1["arguments"] is None
    assert esc1["outcome"] == "escalate"
    assert exp1["outcome"] == "explain"
    assert harness.fake.tops == [READOUT_TOP] * 4
    assert harness.guarded == [("measure", False)]
    text = _page(tmp_path).read_text()
    assert "## Metrics" in text
    assert _row(text, "Right proposals") == ["1 of 2"]
    metrics_file = next((tmp_path / "run" / "measure" / "val-r1").glob("*.metrics.json"))
    kept = json.loads(metrics_file.read_text())
    assert kept["readout"] == {"complete": 4, "incomplete": 0, "no_label_mass": 0, "call_error": 0}
    assert kept["top_asked"] == [READOUT_TOP]


def test_the_prompt_is_the_domain_prompt_and_the_request_is_the_entry_text(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    measure.main(_argv(tmp_path, split), seams=harness.seams)
    first = harness.fake.prompts[0]
    assert first.startswith(DOMAIN.instruction)
    assert list(letters_in(first)) == list(DOMAIN.candidates())
    assert first.rsplit("\n", 1)[-1] == "turn on the kitchen"


def test_two_models_run_back_to_back_with_identical_settings(tmp_path):
    harness = Harness(by_request(PICKS))
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split, models=(STOCK, TUNED)), seams=harness.seams) == 0
    assert [spec.model for spec in harness.built] == [STOCK, TUNED]
    assert harness.built[0].kind == harness.built[1].kind == "in-process"
    assert harness.closed == 2
    assert len(_lines(tmp_path, index=2, model=TUNED)) == 4


def test_the_tokenizer_option_reaches_the_scorer_spec(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    measure.main(_argv(tmp_path, split, "--tokenizer", "/models/merged"), seams=harness.seams)
    assert harness.built[0].tokenizer == "/models/merged"


def test_incomplete_readouts_are_counted_never_renormalised(tmp_path):
    def pick(prompt: str) -> dict[str, float]:
        if "kitchen" in prompt.rsplit("\n", 1)[-1]:
            return favour(prompt, "lamp_on")
        return favour(prompt, "escalate", drop="lamp_status")

    harness = Harness(pick)
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 0
    on1, *rest = _lines(tmp_path)
    assert on1["candidates"] is not None
    assert all(row["candidates"] is None for row in rest)
    assert all("raw_scores" not in row for row in rest)
    text = _page(tmp_path).read_text()
    assert _row(text, "Scorer readouts, complete / incomplete (never renormalised)") == ["1 / 3"]


def test_no_label_mass_is_invalid_and_not_a_tier_error(tmp_path):
    harness = Harness(lambda _prompt: {})
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 0
    rows = _lines(tmp_path)
    assert {row["invalid_reason"] for row in rows} == {"no_label_mass"}


def test_missing_candidate_slice_offers_every_operation_but_the_gold_one(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    argv = _argv(tmp_path, split, "--slice", "missing-candidate")
    assert measure.main(argv, seams=harness.seams) == 0
    assert len(harness.fake.prompts) == 2  # the two operation entries only
    offered = list(letters_in(harness.fake.prompts[0]))
    assert "lamp_on" not in offered
    assert offered == [n for n in DOMAIN.names() if n != "lamp_on"] + ["explain", "escalate"]
    rows = _lines(tmp_path)
    assert [row["id"] for row in rows] == ["on1-nocand", "st1-nocand"]
    assert all(row["expected"] == {"escalate": True} for row in rows)
    assert all(row["outcome"] == "escalate" for row in rows)
    assert "missing-candidate" in _page(tmp_path).read_text()


def test_reasons_mode_scores_the_reason_pool(tmp_path):
    entries = _entries()
    entries[2]["class"] = "decline:outside_table"

    def pick(prompt: str) -> dict[str, float]:
        request = prompt.rsplit("\n", 1)[-1]
        if "weather" in request:
            return favour(prompt, "escalate:outside_table")
        return by_request(PICKS)(prompt)

    harness = Harness(pick)
    split = _split(tmp_path, entries=entries)
    assert measure.main(_argv(tmp_path, split, "--reasons"), seams=harness.seams) == 0
    pool = DOMAIN.candidates(with_reasons=True)
    for prompt in harness.fake.prompts:
        assert list(letters_in(prompt)) == list(pool)
        for name, text in DOMAIN.reason_descriptions().items():
            assert f") {name}: {text}" in prompt
    esc1 = _lines(tmp_path)[2]
    assert esc1["outcome"] == "escalate"
    assert esc1["candidates"]["escalate:outside_table"] == pytest.approx(0.9)
    assert "(escalate)" not in esc1["candidates"]
    assert "reasons mode" in _page(tmp_path).read_text()


def test_reasons_request_keeps_every_reason_when_a_slice_drops_an_operation():
    offered = tuple(name for name in DOMAIN.names() if name != "lamp_on")
    messages, kwargs = measure.scorer_request(DOMAIN, "x", offered, reasons=True)
    assert "lamp_on" not in kwargs["order"]
    assert set(DOMAIN.escalate_labels()) <= set(kwargs["order"])
    full = DOMAIN.candidates(with_reasons=True)
    assert (
        kwargs["labels"]["escalate:injection"] == LABEL_ALPHABET[full.index("escalate:injection")]
    )
    assert "escalate:injection" in messages[0]["content"]


def test_the_default_request_is_the_adapter_prompt():
    messages, kwargs = measure.scorer_request(DOMAIN, "show the lamps", None, reasons=False)
    from jev_factory.backbones.causal_lm import scorer as sc

    assert messages == sc.prompt_messages(DOMAIN, "show the lamps", None)
    assert kwargs == {"offered": None}


# ---------------------------------------------------------------------------
# Grounding: a fixed snapshot from the domain's world schema
# ---------------------------------------------------------------------------


def test_a_ground_snapshot_grounds_every_proposal_and_its_hash_is_recorded(tmp_path):
    harness = Harness(by_request(PICKS))
    split = _split(tmp_path)
    snapshot = _snapshot(tmp_path, rooms=("bedroom",))  # no kitchen in this world
    argv = _argv(tmp_path, split, "--ground-snapshot", str(snapshot))
    assert measure.main(argv, seams=harness.seams) == 0
    on1 = _lines(tmp_path)[0]
    assert (on1["outcome"], on1["invalid_reason"]) == ("invalid", "not_grounded")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert digest in _page(tmp_path).read_text()


def test_a_malformed_snapshot_is_refused_before_any_model_starts(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rooms": "kitchen", "source": "x", "created": "y"}))
    argv = _argv(tmp_path, split, "--ground-snapshot", str(bad))
    assert measure.main(argv, seams=harness.seams) == 1
    assert harness.built == []


def test_live_and_snapshot_groundings_are_exclusive(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    argv = _argv(tmp_path, split, "--live", "--ground-snapshot", str(_snapshot(tmp_path)))
    assert measure.main(argv, seams=harness.seams) == 1


def test_the_split_world_is_used_when_present(tmp_path):
    harness = Harness(by_request(PICKS))
    path = tmp_path / "splits" / "val.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "header": "Toy corpus. Split 'val' of toy.json (seed=39).",
                "entries": _entries(),
                "world": {"home": "h", "rooms": ["attic"]},
            }
        )
    )
    assert measure.main(_argv(tmp_path, path), seams=harness.seams) == 0
    assert _lines(tmp_path)[0]["invalid_reason"] == "not_grounded"
    assert "fixture world from the split file" in _page(tmp_path).read_text()


# ---------------------------------------------------------------------------
# Split refusals
# ---------------------------------------------------------------------------


def test_refuses_held_out_without_acceptance(tmp_path, capsys):
    harness = Harness()
    split = _split(tmp_path, "held-out.json")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1
    assert "--acceptance" in capsys.readouterr().err
    assert harness.built == []
    assert not _page(tmp_path).exists()


def test_refuses_test_side_without_final(tmp_path, capsys):
    harness = Harness()
    split = _split(tmp_path, "test.json")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1
    assert "--final" in capsys.readouterr().err
    assert harness.built == []


@pytest.mark.parametrize("flag", ["--final", "--acceptance"])
def test_final_and_acceptance_refused_on_other_files(tmp_path, flag):
    harness = Harness()
    assert measure.main(_argv(tmp_path, _split(tmp_path), flag), seams=harness.seams) == 1
    assert harness.built == []


def test_a_renamed_test_side_still_needs_final(tmp_path):
    harness = Harness()
    split = _split(tmp_path, "eval.json", header="Toy corpus. Split 'test' of toy.json (seed=39).")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1


def test_a_renamed_held_out_still_needs_acceptance(tmp_path):
    harness = Harness()
    split = _split(tmp_path, "eval.json", header="Held-out split drafted for the toy domain.")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1


def test_an_undetermined_side_is_refused(tmp_path, capsys):
    harness = Harness()
    split = _split(tmp_path, "eval.json", header="some corpus")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1
    assert "cannot tell which side" in capsys.readouterr().err


def test_the_seed_corpus_itself_may_be_measured(tmp_path):
    harness = Harness()
    assert measure.main(_argv(tmp_path, SEED_CORPUS), seams=harness.seams) == 0


def test_revision_required_per_model(tmp_path, capsys):
    harness = Harness()
    argv = _argv(tmp_path, _split(tmp_path))
    argv += ["--model", TUNED]
    assert measure.main(argv, seams=harness.seams) == 1
    assert "--revision" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "ok"),
    [("final-a3.q4_k_m", True), ("val-350m", True), ("Final-A3", False), ("_x", False)],
)
def test_label_allows_a_quantized_build_name(tmp_path, label, ok):
    harness = Harness()
    argv = _argv(tmp_path, _split(tmp_path), label=label)
    assert (measure.main(argv, seams=harness.seams) == 0) is ok


def test_refuses_to_overwrite_a_results_page(tmp_path):
    harness = Harness()
    split = _split(tmp_path)
    page = _page(tmp_path)
    page.parent.mkdir(parents=True)
    page.write_text("keep me\n")
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 1
    assert page.read_text() == "keep me\n"
    assert measure.main(_argv(tmp_path, split, "--force"), seams=harness.seams) == 0


def test_a_run_dir_inside_a_git_worktree_is_refused(tmp_path):
    harness = Harness()
    repo_dir = Path(__file__).resolve().parent / "not-a-run-dir"
    argv = _argv(tmp_path, _split(tmp_path))
    argv[argv.index("--run-dir") + 1] = str(repo_dir)
    assert measure.main(argv, seams=harness.seams) == 1
    assert not repo_dir.exists()


# ---------------------------------------------------------------------------
# o10: the sealed sides are measured once
# ---------------------------------------------------------------------------


def _ledger(tmp_path: Path) -> list[dict]:
    path = tmp_path / "run" / "measure" / "once-ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.behavioral("o10")
@pytest.mark.parametrize(
    ("name", "flag", "side"),
    [("test.json", "--final", "test"), ("held-out.json", "--acceptance", "held-out")],
)
def test_a_second_sealed_measurement_without_a_deviation_exits_non_zero(
    tmp_path, capsys, name, flag, side
):
    split = _split(tmp_path, name)
    first = Harness(by_request(PICKS))
    assert measure.main(_argv(tmp_path, split, flag, label="final-1"), seams=first.seams) == 0
    [record] = _ledger(tmp_path)
    assert record["side"] == side
    assert record["status"] == "measured"
    assert record["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert "kitchen" not in json.dumps(record)  # ids, counts and hashes only, never text
    capsys.readouterr()

    second = Harness(by_request(PICKS))
    code = measure.main(_argv(tmp_path, split, flag, label="final-2"), seams=second.seams)
    assert code != 0
    err = capsys.readouterr().err
    assert "measured once" in err
    assert "--deviation" in err
    assert second.built == []  # refused before any model starts
    assert not _page(tmp_path, "final-2").exists()
    assert len(_ledger(tmp_path)) == 1


@pytest.mark.behavioral("o10")
def test_a_recorded_deviation_permits_one_more_sealed_measurement(tmp_path):
    split = _split(tmp_path, "test.json")
    assert measure.main(_argv(tmp_path, split, "--final", label="f1"), seams=Harness().seams) == 0
    argv = _argv(tmp_path, split, "--final", "--deviation", "d9", label="f2")
    assert measure.main(argv, seams=Harness().seams) == 0
    assert [r["deviation"] for r in _ledger(tmp_path)] == [None, "d9"]
    text = _page(tmp_path, "f2").read_text()
    assert "- Deviation: `d9`" in text
    assert "- Measurements of the test side in this run, including this one: 2" in text


@pytest.mark.behavioral("o10")
@pytest.mark.parametrize(
    ("name", "flag", "side"),
    [("test.json", "--final", "test"), ("held-out.json", "--acceptance", "held-out")],
)
def test_the_once_rule_is_per_side_and_slice(tmp_path, capsys, name, flag, side):
    """d9: the full side and its missing-candidate slice are each measured once."""
    split = _split(tmp_path, name)
    mc = ("--slice", "missing-candidate")
    assert measure.main(_argv(tmp_path, split, flag, label="f"), seams=Harness().seams) == 0
    # the slice is a different pair: no deviation needed the first time
    assert measure.main(_argv(tmp_path, split, flag, *mc, label="m1"), seams=Harness().seams) == 0
    assert [(r["side"], r["slice"]) for r in _ledger(tmp_path)] == [
        (side, "full"),
        (side, "missing-candidate"),
    ]
    capsys.readouterr()
    # the same pair again is refused without a deviation id
    second = Harness()
    code = measure.main(_argv(tmp_path, split, flag, *mc, label="m2"), seams=second.seams)
    assert code != 0
    assert second.built == []
    assert "measured once" in capsys.readouterr().err
    code = measure.main(_argv(tmp_path, split, flag, label="f2"), seams=Harness().seams)
    assert code != 0
    assert len(_ledger(tmp_path)) == 2
    # and a deviation id permits one more of that pair
    argv = _argv(tmp_path, split, flag, *mc, "--deviation", "d9", label="m3")
    assert measure.main(argv, seams=Harness().seams) == 0
    assert _ledger(tmp_path)[-1]["deviation"] == "d9"


@pytest.mark.behavioral("o10")
def test_a_ledger_written_with_side_only_keys_reads_as_the_full_slice(tmp_path, capsys):
    path = tmp_path / "run" / "measure" / "once-ledger.jsonl"
    path.parent.mkdir(parents=True)
    old = {
        "side": "test",
        "split_sha256": "0" * 64,
        "label": "old",
        "date": "2026-09-01",
        "status": "measured",
        "deviation": None,
    }
    path.write_text(json.dumps(old) + "\n")
    ledger = once.OnceLedger(tmp_path / "run")
    assert [r.slice for r in ledger.measured("test")] == ["full"]
    split = _split(tmp_path, "test.json")
    assert measure.main(_argv(tmp_path, split, "--final", label="n"), seams=Harness().seams) == 1
    assert "measured once" in capsys.readouterr().err
    argv = _argv(tmp_path, split, "--final", "--slice", "missing-candidate", label="n2")
    assert measure.main(argv, seams=Harness().seams) == 0


def test_a_deviation_must_look_like_an_id_and_apply_to_a_sealed_run(tmp_path, capsys):
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split, "--deviation", "a b"), seams=Harness().seams) == 1
    assert measure.main(_argv(tmp_path, split, "--deviation", "d1"), seams=Harness().seams) == 1
    assert "final or held-out" in capsys.readouterr().err


@pytest.mark.behavioral("o10")
def test_a_failed_start_up_never_leaves_a_page_or_record_that_blocks_the_real_run(tmp_path, capsys):
    """nvsh P71: a final run whose every start-up failed measured nothing."""
    split = _split(tmp_path, "test.json")
    argv = _argv(tmp_path, split, "--final", label="final")
    failing = Harness(fail="No module named 'transformers'")
    assert measure.main(argv, seams=failing.seams) == 2
    err = capsys.readouterr().err
    assert "scorer start-up failed" in err
    assert "transformers" in err
    page = _page(tmp_path, "final")
    assert measure.NOT_MEASURED_MARKER in page.read_text().splitlines()
    assert _ledger(tmp_path) == []

    working = Harness(by_request(PICKS))
    assert measure.main(argv, seams=working.seams) == 0  # no --force, no --deviation
    assert "replacing" in capsys.readouterr().err
    text = page.read_text()
    assert measure.NOT_MEASURED_MARKER not in text.splitlines()
    assert "- Measurements of the test side in this run, including this one: 1" in text
    assert [r["status"] for r in _ledger(tmp_path)] == ["measured"]


@pytest.mark.behavioral("o10")
def test_a_preflight_refusal_on_a_final_run_does_not_count_as_the_measurement(tmp_path):
    split = _split(tmp_path, "test.json")
    harness = Harness()

    def refuse(base_url, model, ctx=None):
        raise measure.PreflightError("not serving it")

    harness.seams.preflight = refuse
    argv = _argv(tmp_path, split, "--final", label="final")
    argv[argv.index("in-process")] = "served"
    argv += ["--base-url", "http://127.0.0.1:9/v1", "--max-logprobs", str(READOUT_TOP)]
    assert measure.main(argv, seams=harness.seams) == 2
    assert _ledger(tmp_path) == []
    assert not _page(tmp_path, "final").exists()
    assert harness.fake.prompts == []


def test_a_partly_measured_run_still_blocks_a_page_rerun(tmp_path):
    split = _split(tmp_path)
    harness = Harness()
    original = harness.build

    def build(spec):
        if spec.model == TUNED:
            raise ImportError("no weights")
        return original(spec)

    harness.seams.build_scorer = build
    argv = _argv(tmp_path, split, models=(STOCK, TUNED))
    assert measure.main(argv, seams=harness.seams) == 2
    assert measure.NOT_MEASURED_MARKER not in _page(tmp_path).read_text().splitlines()
    assert measure.main(argv, seams=Harness().seams) == 1


def test_two_sealed_measurements_never_overlap(tmp_path, capsys):
    split = _split(tmp_path, "test.json")
    ledger = once.OnceLedger(tmp_path / "run")
    with ledger:
        argv = _argv(tmp_path, split, "--final", label="final")
        assert measure.main(argv, seams=Harness().seams) == 1
    assert "holds" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Tier errors: a server that dies is never a plausible-looking run
# ---------------------------------------------------------------------------


def _dying(after: int):
    calls = itertools.count()

    def pick(prompt: str) -> dict[str, float]:
        if next(calls) >= after:
            raise ConnectionError("server dropped")
        return favour(prompt, "escalate")

    return pick


def test_a_tier_error_fails_the_run_and_writes_no_page(tmp_path):
    harness = Harness(_dying(after=1))
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 2
    assert not _page(tmp_path).exists()
    rows = _lines(tmp_path)  # kept for debugging
    assert [row.get("invalid_reason") for row in rows] == [
        None,
        "tier_error",
        "tier_error",
        "tier_error",
    ]


def test_allow_tier_errors_writes_the_page_with_the_count(tmp_path):
    harness = Harness(_dying(after=3))
    split = _split(tmp_path)
    assert (
        measure.main(_argv(tmp_path, split, "--allow-tier-errors", "1"), seams=harness.seams) == 0
    )
    text = _page(tmp_path).read_text()
    assert "1 tier-error prediction(s) permitted" in text
    assert "--allow-tier-errors 1" in text


def test_allow_tier_errors_rejects_a_negative_value(tmp_path, capsys):
    argv = _argv(tmp_path, _split(tmp_path), "--allow-tier-errors", "-1")
    assert measure.main(argv, seams=Harness().seams) == 1
    assert "--allow-tier-errors" in capsys.readouterr().err


@pytest.mark.behavioral("o10")
def test_a_final_run_whose_server_died_mid_run_is_still_the_one_measurement(tmp_path):
    split = _split(tmp_path, "test.json")
    argv = _argv(tmp_path, split, "--final", label="final")
    assert measure.main(argv, seams=Harness(_dying(after=2)).seams) == 2
    assert [r["status"] for r in _ledger(tmp_path)] == ["failed-mid-run"]
    again = _argv(tmp_path, split, "--final", label="final-b")
    assert measure.main(again, seams=Harness().seams) == 1


# ---------------------------------------------------------------------------
# Served scorers: attach flags, preflight, the GPU guard
# ---------------------------------------------------------------------------


def _served(argv: list[str]) -> list[str]:
    argv = list(argv)
    argv[argv.index("in-process")] = "served"
    return argv


def test_an_attached_scorer_needs_readout_top_max_logprobs(tmp_path, capsys):
    harness = Harness()
    split = _split(tmp_path)
    base = _served(_argv(tmp_path, split)) + ["--base-url", "http://127.0.0.1:9/v1"]
    for extra in ([], ["--max-logprobs", "20"], ["--max-logprobs", str(READOUT_TOP - 1)]):
        assert measure.main(base + extra, seams=harness.seams) == 1
        assert "max-logprobs" in capsys.readouterr().err
    assert harness.built == []
    ok = base + ["--max-logprobs", str(READOUT_TOP)]
    assert measure.main(ok, seams=harness.seams) == 0
    assert harness.guarded == []  # an attached server is the operator's job, not a new one


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--base-url", "http://127.0.0.1:9/v1", "--serve", "x.gguf", "--tokenizer", "t"],
        ["--serve", "x.gguf"],
    ],
)
def test_a_served_scorer_needs_exactly_one_endpoint_and_a_tokenizer(tmp_path, extra):
    harness = Harness()
    argv = _served(_argv(tmp_path, _split(tmp_path))) + extra
    assert measure.main(argv, seams=harness.seams) == 1
    assert harness.built == []


def test_in_process_refuses_served_flags(tmp_path):
    argv = _argv(tmp_path, _split(tmp_path), "--base-url", "http://127.0.0.1:9/v1")
    assert measure.main(argv, seams=Harness().seams) == 1


def test_the_gpu_guard_refuses_a_foreign_compute_process(tmp_path, monkeypatch, capsys):
    stubs = stub_bin(tmp_path / "bin")
    monkeypatch.setenv("PATH", path_with(stubs))
    monkeypatch.setenv("STUB_GPU_APPS", "4242, python3")
    harness = Harness()
    harness.seams.gpu_guard = measure.Seams().gpu_guard
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=harness.seams) == 2
    assert "4242" in capsys.readouterr().err
    assert harness.built == []
    argv = _argv(tmp_path, split, "--allow-foreign-gpu")
    assert measure.main(argv, seams=harness.seams) == 0


class _ScorerHandler(http.server.BaseHTTPRequestHandler):
    model_ids: list[str] = [STOCK]
    completions_status = 200
    calls = 0
    bodies: list[dict] = []

    def _send(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        data = [{"id": i, "max_model_len": 2048} for i in _ScorerHandler.model_ids]
        self._send({"data": data} if self.path.rstrip("/").endswith("/models") else {}, 200)

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        _ScorerHandler.calls += 1
        _ScorerHandler.bodies.append({k: v for k, v in body.items() if k != "prompt"})
        if _ScorerHandler.completions_status != 200:
            self._send({"error": "boom"}, _ScorerHandler.completions_status)
            return
        top = favour(body["prompt"], "escalate")
        self._send({"choices": [{"logprobs": {"top_logprobs": [top]}}]})

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def scorer_server():
    _ScorerHandler.model_ids = [STOCK]
    _ScorerHandler.completions_status = 200
    _ScorerHandler.calls = 0
    _ScorerHandler.bodies = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _ScorerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _real_served_seams() -> measure.Seams:
    harness = Harness()
    seams = harness.seams
    seams.build_scorer = lambda spec: measure.build_scorer(spec, renderer=lambda _spec: render)
    seams.preflight = measure.preflight_models
    return seams


def _attach_argv(tmp_path, split, url, *extra):
    argv = _served(_argv(tmp_path, split)) + ["--base-url", url]
    return argv + ["--max-logprobs", str(READOUT_TOP), *extra]


def test_an_attached_run_asks_readout_top_logprobs_over_http(tmp_path, scorer_server):
    split = _split(tmp_path)
    seams = _real_served_seams()
    assert measure.main(_attach_argv(tmp_path, split, scorer_server), seams=seams) == 0
    assert {body["logprobs"] for body in _ScorerHandler.bodies} == {READOUT_TOP}
    assert {body["max_tokens"] for body in _ScorerHandler.bodies} == {1}
    assert all(row["outcome"] == "escalate" for row in _lines(tmp_path))


def test_served_preflight_refuses_the_wrong_model_before_any_entry(
    tmp_path, scorer_server, monkeypatch
):
    monkeypatch.setattr(_ScorerHandler, "model_ids", ["another-model"])
    split = _split(tmp_path)
    assert (
        measure.main(_attach_argv(tmp_path, split, scorer_server), seams=_real_served_seams()) == 2
    )
    assert _ScorerHandler.calls == 0
    assert not _page(tmp_path).exists()


def test_served_preflight_checks_the_labelled_context(tmp_path, scorer_server):
    split = _split(tmp_path)
    argv = _attach_argv(tmp_path, split, scorer_server, "--ctx", "4096")
    assert measure.main(argv, seams=_real_served_seams()) == 2
    assert _ScorerHandler.calls == 0


def test_a_served_call_error_fails_the_run_and_writes_no_page(tmp_path, scorer_server, monkeypatch):
    monkeypatch.setattr(_ScorerHandler, "completions_status", 500)
    split = _split(tmp_path)
    assert (
        measure.main(_attach_argv(tmp_path, split, scorer_server), seams=_real_served_seams()) == 2
    )
    assert not _page(tmp_path).exists()
    assert {row["invalid_reason"] for row in _lines(tmp_path)} == {"tier_error"}


# ---------------------------------------------------------------------------
# --calibration
# ---------------------------------------------------------------------------


def _params(tmp_path: Path, name: str = "params.json", **fields) -> Path:
    payload = {
        "temperature": 2.0,
        "vector": {},
        "fit_examples": 10,
        "predictions_source": "work/val-1-stock.predictions.jsonl",
        **fields,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_calibration_rescales_candidates_and_keeps_the_raw_distribution(tmp_path):
    split = _split(tmp_path)
    assert (
        measure.main(_argv(tmp_path, split, label="raw"), seams=Harness(by_request(PICKS)).seams)
        == 0
    )
    raw = _lines(tmp_path, "raw")
    params = _params(tmp_path, vector={"(escalate)": 3.0})
    argv = _argv(tmp_path, split, "--calibration", str(params), label="cal")
    assert measure.main(argv, seams=Harness(by_request(PICKS)).seams) == 0
    calibrated = _lines(tmp_path, "cal")
    for before, after in zip(raw, calibrated):
        expected = apply_scaling(before["candidates"], 2.0, {"(escalate)": 3.0})
        assert after["candidates"] == pytest.approx(expected)
        assert after["raw_probabilities"] == pytest.approx(before["candidates"])
        assert after["outcome"] == before["outcome"]  # the decision is not rescaled
    text = _page(tmp_path, "cal").read_text()
    digest = hashlib.sha256(params.read_bytes()).hexdigest()
    assert f"sha256 `{digest}`" in text
    assert _row(text, "ECE / Brier before --calibration") != ["n/a"]


def test_no_calibration_is_recorded_as_none(tmp_path):
    assert measure.main(_argv(tmp_path, _split(tmp_path)), seams=Harness().seams) == 0
    text = _page(tmp_path).read_text()
    assert "- Calibration: none" in text
    assert _row(text, "ECE / Brier before --calibration") == ["n/a"]


@pytest.mark.parametrize(
    "source", ["work/test-1-stock.predictions.jsonl", "out/held-out-1.predictions.jsonl"]
)
def test_calibration_fitted_on_test_or_held_out_is_refused(tmp_path, capsys, source):
    harness = Harness()
    params = _params(tmp_path, predictions_source=source)
    argv = _argv(tmp_path, _split(tmp_path), "--calibration", str(params))
    assert measure.main(argv, seams=harness.seams) == 1
    assert "calibration" in capsys.readouterr().err
    assert harness.built == []


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": 0},
        {"temperature": "hot"},
        {"vector": {"lamp_on": -1.0}},
        {"vector": ["lamp_on"]},
        {"predictions_source": None},
    ],
)
def test_malformed_calibration_params_are_refused(tmp_path, fields):
    harness = Harness()
    params = _params(tmp_path, **fields)
    argv = _argv(tmp_path, _split(tmp_path), "--calibration", str(params))
    assert measure.main(argv, seams=harness.seams) == 1
    assert harness.built == []


def test_a_missing_calibration_file_is_refused(tmp_path):
    argv = _argv(tmp_path, _split(tmp_path), "--calibration", str(tmp_path / "nope.json"))
    assert measure.main(argv, seams=Harness().seams) == 1


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_carries_cis_per_slice_tables_and_provenance(tmp_path):
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=Harness(by_request(PICKS)).seams) == 0
    text = _page(tmp_path).read_text()
    assert _row(text, "Right proposals, 95% bootstrap CI")[0].startswith("[")
    for metric in (
        "Abstention recall, 95% bootstrap CI",
        "False-positive tool calls, 95% bootstrap CI",
        "ECE, 95% bootstrap CI",
        "Brier, 95% bootstrap CI",
    ):
        assert len(_row(text, metric)) == 1
    section = text.split("## Per-slice calibration", 1)[1]
    for name in ("read_only", "mutating", "escalate_or_explain"):
        assert f"#### {name}" in section
        assert f"| {name} |" in section
    assert DOMAIN.surface_sha256() in text
    assert hashlib.sha256(split.read_bytes()).hexdigest() in text
    assert "- Seed: 39 (from the split header)" in text
    assert "NVIDIA-SMI stub" in text
    assert "- Final run: no" in text


def test_reports_write_the_home_directory_as_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    split = _split(tmp_path)
    assert measure.main(_argv(tmp_path, split), seams=Harness().seams) == 0
    text = _page(tmp_path).read_text()
    assert str(tmp_path) + "/" not in text
    assert "$HOME/" in text


def test_a_measurement_reports_its_progress_for_jev_status(tmp_path):
    from jev_factory.factory import detach

    split = _split(tmp_path)
    jobs = tmp_path / "jobs"
    argv = _argv(tmp_path, split, "--progress-dir", str(jobs), label="val-r1.q4")
    assert measure.main(argv, seams=Harness().seams) == 0
    progress = detach.read_progress(jobs, "measure-val-r1_q4")
    n = len(json.loads(split.read_text())["entries"])
    assert (progress["done"], progress["total"]) == (n, n)
    assert progress["updated"] >= progress["started"]
    assert progress["start_done"] == 0
    assert detach.job_status(jobs, "measure-val-r1_q4")["state"] == "complete"
