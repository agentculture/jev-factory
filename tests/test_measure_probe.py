"""The permutation probe: jev_factory/measure/probe.py.

Ported from nvsh's tests/test_lfm_finetune_permutation_probe.py onto the toy
lamp domain. A fake top-k function stands in for the model; no tokenizer or
GPU is needed. Two fakes exercise the invariant the probe is built to catch:

* ``identity(target)`` always answers whichever offered candidate's *name*
  is *target*, wherever it landed, so answer changes stay at 0 for every kind;
* ``first_listed`` always answers the first listed candidate, so ``order``
  (and ``all``) must show changes while ``letters`` must not.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.measure import probe
from jev_factory.measure.corpus import CorpusEntry
from tests.fixtures.toy_domain import DOMAIN

POOL = DOMAIN.names()
_LINE_RE = re.compile(r"^([A-Za-z])\) (\S+):", re.MULTILINE)


def _render(messages: list[dict]) -> str:
    return "\n".join(message["content"] for message in messages)


def _lines(prompt: str) -> dict[str, str]:
    return {name: label for label, name in _LINE_RE.findall(prompt)}


def identity(target: str, prompts: list | None = None):
    def top_k(prompt: str, top: int) -> dict[str, float]:
        if prompts is not None:
            prompts.append(prompt)
        return {
            label: math.log(0.9 if name == target else 0.01)
            for name, label in _lines(prompt).items()
        }

    return top_k


def first_listed(prompts: list | None = None):
    def top_k(prompt: str, top: int) -> dict[str, float]:
        if prompts is not None:
            prompts.append(prompt)
        names = _lines(prompt)
        if not names:
            return {}
        first = _LINE_RE.search(prompt).group(1)
        return {label: math.log(0.9 if label == first else 0.01) for label in names.values()}

    return top_k


def narrow_alphabet(target: str):
    """Reads only letters A, B and C: any other offered letter goes missing."""

    def top_k(prompt: str, top: int) -> dict[str, float]:
        return {
            label: math.log(0.9 if name == target else 0.01)
            for name, label in _lines(prompt).items()
            if label.upper() in ("A", "B", "C")
        }

    return top_k


def _entry(entry_id: str, operation: str, text: str = "which lights are on") -> CorpusEntry:
    return CorpusEntry(
        id=entry_id, kind="explicit", text=text, expect={"operation": operation}, source="t"
    )


def _kinds(report: dict) -> dict:
    return {row["kind"]: row for row in report["kinds"]}


# -- gold / paraphrases --


def test_gold_name_operation_escalate_explain():
    assert probe.gold_name(DOMAIN, {"operation": "lamp_status"}) == "lamp_status"
    assert probe.gold_name(DOMAIN, {"escalate": True}) == "escalate"
    assert probe.gold_name(DOMAIN, {"explain": True}) == "explain"


def test_reasons_gold_rolls_an_unclassed_escalation_up():
    assert probe.gold_name(DOMAIN, {"escalate": True}, reasons=True, cls=None) == (
        DOMAIN.default_reason_label()
    )
    assert probe.gold_name(DOMAIN, {"escalate": True}, reasons=True, cls="decline:injection") == (
        "escalate:injection"
    )


def test_load_paraphrases_requires_two_alternatives(tmp_path):
    path = tmp_path / "paraphrases.json"
    path.write_text(json.dumps({"lamp_status": ["only one"]}))
    with pytest.raises(probe.ProbeError, match="fewer than 2"):
        probe.load_paraphrases(path)


def test_load_paraphrases_ok(tmp_path):
    path = tmp_path / "paraphrases.json"
    data = {"lamp_status": ["which lights are lit", "what is on right now"]}
    path.write_text(json.dumps(data))
    assert probe.load_paraphrases(path) == data


def test_the_domain_paraphrases_are_the_default():
    assert probe.domain_paraphrases(DOMAIN) == {
        "lamp_status": ["Report which lights are lit right now."],
        "lamp_on": ["Switch the lights on in a given room."],
    }


# -- the core probe --


def test_identity_scorer_never_changes_answer():
    report = probe.run_probe(
        DOMAIN,
        identity("lamp_status"),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=12,
        seed="probe-seed",
    )
    kinds = _kinds(report)
    for kind in ("order", "letters", "subset", "all"):
        assert kinds[kind]["changes"] == 0, kind
        assert kinds[kind]["rate"] == 0.0, kind
    assert "paraphrase" not in kinds


def test_identity_scorer_survives_paraphrase():
    paraphrases = {"lamp_status": ["which lamps glow", "what is lit"]}
    report = probe.run_probe(
        DOMAIN,
        identity("lamp_status"),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=8,
        seed="probe-seed",
        paraphrases=paraphrases,
    )
    kinds = _kinds(report)
    assert kinds["paraphrase"]["changes"] == 0
    assert kinds["paraphrase"]["trials"] == 8


def test_paraphrase_trials_render_the_alternative_description():
    prompts: list[str] = []
    paraphrases = {"lamp_status": ["which lamps glow", "what is lit"]}
    probe.run_probe(
        DOMAIN,
        identity("lamp_status", prompts),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=3,
        seed="p",
        paraphrases=paraphrases,
    )
    assert any(") lamp_status: which lamps glow" in p or "what is lit" in p for p in prompts)


def test_first_listed_scorer_changes_under_order_not_letters():
    report = probe.run_probe(
        DOMAIN,
        first_listed(),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=20,
        seed="probe-seed-2",
    )
    kinds = _kinds(report)
    assert kinds["order"]["changes"] > 0
    assert kinds["letters"]["changes"] == 0
    assert kinds["all"]["changes"] > 0


def test_subset_keeps_gold_and_reports_baseline_removed():
    report = probe.run_probe(
        DOMAIN,
        first_listed(),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=25,
        seed="probe-seed-3",
    )
    subset = _kinds(report)["subset"]
    removed = subset["baseline_choice_removed"]
    assert removed["trials"] == subset["trials"]
    assert 0 <= removed["removed"] <= removed["trials"]


def test_build_trial_subset_always_keeps_gold():
    baseline_labels = sc.labels_for(DOMAIN, POOL)
    for i in range(30):
        trial = probe.build_trial(
            "subset",
            probe.derive_seed("s", "e1", "subset", i),
            POOL,
            list(POOL),
            baseline_labels,
            "lamp_on",
            "lamp_status",
            None,
        )
        assert "lamp_on" in trial.order
        assert set(trial.labels) == set(trial.order)


def test_an_unknown_kind_is_refused():
    with pytest.raises(probe.ProbeError):
        probe.build_trial("sideways", "s", POOL, list(POOL), {}, "lamp_on", None, None)


# -- bootstrap over entries, not trials --


def test_kind_report_bootstraps_over_entries():
    changed = probe.EntryOutcome("a", "x", "x", trials={"order": [(True, False, None, False)] * 10})
    same = probe.EntryOutcome("b", "x", "x", trials={"order": [(False, False, None, False)] * 10})
    report = probe.kind_report([changed, same], "order", bootstrap_seed=0)
    assert (report["trials"], report["changes"], report["rate"]) == (20, 10, 0.5)
    assert report["ci_low"] in (0.0, 0.5)
    assert report["ci_high"] in (0.5, 1.0)
    assert "entries" in report["bootstrap_note"]


def test_kind_report_label_case_counts():
    outcome = probe.EntryOutcome(
        "a",
        "x",
        "x",
        trials={
            "letters": [
                (True, True, None, False),
                (False, False, None, False),
                (True, False, None, False),
            ]
        },
    )
    case = probe.kind_report([outcome], "letters", bootstrap_seed=0)["label_case"]
    assert (case["lowercase_trials"], case["lowercase_changes"]) == (1, 1)
    assert (case["uppercase_trials"], case["uppercase_changes"]) == (2, 1)


def test_kind_report_returns_none_when_no_trials():
    outcome = probe.EntryOutcome("a", "x", "x", trials={})
    assert probe.kind_report([outcome], "paraphrase", bootstrap_seed=0) is None


# -- incomplete trials: tallied separately, never scored as an answer --


def test_kind_report_counts_incomplete_separately_from_changes():
    outcome = probe.EntryOutcome(
        "a",
        "x",
        "x",
        trials={
            "letters": [
                (True, False, None, False),
                (False, False, None, False),
                (True, True, None, True),
                (False, True, None, True),
            ]
        },
    )
    report = probe.kind_report([outcome], "letters", bootstrap_seed=0)
    assert (report["trials"], report["changes"], report["rate"]) == (2, 1, 0.5)
    assert report["label_case"]["lowercase_trials"] == 0
    assert report["incomplete"] == {"trials": 2, "of": 4, "rate": 0.5}


def test_kind_report_reports_when_every_trial_is_incomplete():
    outcome = probe.EntryOutcome("a", "x", "x", trials={"letters": [(True, False, None, True)] * 5})
    report = probe.kind_report([outcome], "letters", bootstrap_seed=0)
    assert (report["trials"], report["changes"], report["rate"]) == (0, 0, None)
    assert report["incomplete"] == {"trials": 5, "of": 5, "rate": 1.0}


def test_narrow_alphabet_incomplete_trials_are_not_scored_as_changes():
    pool = ("lamp_status", "list_rooms", "room_status")
    report = probe.run_probe(
        DOMAIN,
        narrow_alphabet("lamp_status"),
        _render,
        [_entry("e1", "lamp_status")],
        pool=pool,
        per_entry=40,
        seed="narrow-alphabet",
    )
    letters = _kinds(report)["letters"]
    assert letters["incomplete"]["of"] == 40
    assert letters["incomplete"]["trials"] > 0
    assert letters["trials"] + letters["incomplete"]["trials"] == 40
    assert letters["changes"] == 0


# -- determinism, verbatim draws --


def test_same_seed_is_reproducible():
    entries = [_entry("e1", "lamp_status"), _entry("e2", "list_rooms")]
    a = probe.run_probe(DOMAIN, first_listed(), _render, entries, pool=POOL, per_entry=6, seed="f")
    b = probe.run_probe(DOMAIN, first_listed(), _render, entries, pool=POOL, per_entry=6, seed="f")
    assert a == b


def test_different_seed_can_change_the_draws():
    labels = sc.labels_for(DOMAIN, POOL)
    args = (POOL, list(POOL), labels, "lamp_status", "lamp_status", None)
    a = probe.build_trial("order", probe.derive_seed(1, "e1", "order", 0), *args)
    b = probe.build_trial("order", probe.derive_seed(2, "e1", "order", 0), *args)
    assert a != b


def test_scorer_sees_the_drawn_order_verbatim():
    prompts: list[str] = []
    probe.run_probe(
        DOMAIN,
        first_listed(prompts),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=5,
        seed="verbatim",
    )
    assert len(prompts) == 1 + 5 * 4
    assert len({_LINE_RE.search(p).group(2) for p in prompts}) > 1


def test_letters_trials_draw_from_the_whole_alphabet():
    prompts: list[str] = []
    probe.run_probe(
        DOMAIN,
        identity("lamp_status", prompts),
        _render,
        [_entry("e1", "lamp_status")],
        pool=POOL,
        per_entry=20,
        seed="alphabet",
    )
    letters = {letter for p in prompts for letter in _lines(p).values()}
    assert any(letter.islower() for letter in letters)


# -- reasons mode --


def test_reasons_probe_offers_the_reason_pool_and_its_gold():
    prompts: list[str] = []
    entry = CorpusEntry(
        id="e1",
        kind="explicit",
        text="reflash the controller",
        expect={"escalate": True},
        phrasing="decline:injection",
    )
    report = probe.run_probe(
        DOMAIN,
        identity("escalate:injection", prompts),
        _render,
        [entry],
        per_entry=6,
        seed="reasons",
        reasons=True,
    )
    kinds = _kinds(report)
    for kind in ("order", "letters", "subset", "all"):
        assert kinds[kind]["changes"] == 0, kind
        assert kinds[kind]["incomplete"]["trials"] == 0, kind
    pool = DOMAIN.candidates(with_reasons=True)
    assert list(_lines(prompts[0])) == list(pool)
    assert _lines(prompts[0]) == sc.positional_labels(pool, pool)
    for prompt in prompts:
        assert "escalate:injection" in _lines(prompt)
        for name, text in DOMAIN.reason_descriptions().items():
            if name in _lines(prompt):
                assert f") {name}: {text}" in prompt
    assert report["reasons"] is True


def test_probe_without_reasons_keeps_the_default_pool():
    prompts: list[str] = []
    report = probe.run_probe(
        DOMAIN,
        identity("lamp_status", prompts),
        _render,
        [_entry("e1", "lamp_status")],
        per_entry=1,
    )
    assert list(_lines(prompts[0])) == list(DOMAIN.candidates())
    assert report["reasons"] is False


def test_pooled_change_rate_and_markdown():
    report = probe.run_probe(
        DOMAIN, first_listed(), _render, [_entry("e1", "lamp_status")], pool=POOL, per_entry=4
    )
    rate = probe.pooled_change_rate(report)
    trials = sum(k["trials"] for k in report["kinds"])
    assert rate == sum(k["changes"] for k in report["kinds"]) / trials
    text = probe.render_markdown(report)
    assert "| order |" in text and "bootstrap over entries" in text


# -- CLI --


def _split_file(tmp_path: Path, name: str, header: str | None = None) -> Path:
    path = tmp_path / name
    payload: dict = {
        "entries": [
            {
                "id": "e1",
                "kind": "explicit",
                "text": "which lights are on",
                "expect": {"operation": "lamp_status", "args": {}},
            }
        ]
    }
    if header is not None:
        payload["header"] = header
    path.write_text(json.dumps(payload))
    return path


DOMAIN_ARG = ["--domain", "tests.fixtures.toy_domain"]


def test_cli_refuses_test_split_without_final(tmp_path, capsys):
    path = _split_file(tmp_path, "test.json")
    assert probe.main([*DOMAIN_ARG, "--split", str(path), "--model", "m"]) == 1
    assert "test" in capsys.readouterr().err


def test_cli_refuses_held_out_split_without_final(tmp_path, capsys):
    path = _split_file(tmp_path, "held-out.json")
    assert probe.main([*DOMAIN_ARG, "--split", str(path), "--model", "m"]) == 1
    assert "held-out" in capsys.readouterr().err


def test_cli_allows_val_split_but_needs_a_model(tmp_path, capsys):
    path = _split_file(tmp_path, "val.json")
    assert probe.main([*DOMAIN_ARG, "--split", str(path)]) == 1
    assert "--model" in capsys.readouterr().err


def test_cli_final_allows_test_split_to_reach_model_check(tmp_path, capsys):
    path = _split_file(tmp_path, "test.json")
    assert probe.main([*DOMAIN_ARG, "--split", str(path), "--final"]) == 1
    err = capsys.readouterr().err
    assert "looks like the test" not in err
    assert "--model" in err


def test_cli_runs_end_to_end_with_an_injected_scorer(tmp_path):
    from jev_factory.measure.run import ScorerHandle

    path = _split_file(tmp_path, "val.json")
    closed: list[bool] = []

    def build(_args):
        return ScorerHandle(
            top_k=identity("lamp_status"), render=_render, close=lambda: closed.append(True)
        )

    out, md = tmp_path / "probe.json", tmp_path / "probe.md"
    argv = [*DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "3"]
    argv += ["--out", str(out), "--markdown", str(md)]
    assert probe.main(argv, build_scorer=build) == 0
    report = json.loads(out.read_text())
    assert report["pooled_change_rate"] == 0.0
    assert {k["kind"] for k in report["kinds"]} == set(probe.KINDS)  # domain paraphrases
    assert md.read_text().startswith("# Permutation probe")
    assert closed == [True]


def test_the_probe_reports_its_progress_for_jev_status(tmp_path):
    from jev_factory.factory import detach
    from jev_factory.measure.run import ScorerHandle

    path = _split_file(tmp_path, "val.json")

    def build(_args):
        return ScorerHandle(top_k=identity("lamp_status"), render=_render, close=lambda: None)

    out = tmp_path / "final" / "probe.json"
    out.parent.mkdir()
    argv = [*DOMAIN_ARG, "--split", str(path), "--model", "m", "--per-entry", "2"]
    argv += ["--out", str(out), "--progress-dir", str(tmp_path / "jobs")]
    assert probe.main(argv, build_scorer=build) == 0
    progress = detach.read_progress(tmp_path / "jobs", "probe-final-probe")
    assert progress["done"] == progress["total"] > 0
