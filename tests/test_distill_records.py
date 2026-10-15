"""The typed-decision record contract and its source loaders (distillery t1)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_factory.distill import records as R
from tests.fixtures.toy_domain import DOMAIN

FIX = Path(__file__).parent / "fixtures" / "typed_decisions"
L = "Apache-2.0"


def _fx(name):
    return json.loads((FIX / name).read_text())


def _rec(**over):
    base = {
        "id": "r1",
        "state": "s",
        "questions": {"q": {"type": "choice", "criteria": [["a", ""], ["b", ""]]}},
        "gold": {"q": {"label": "a"}},
        "split": "train",
        "source": "x",
        "licence": "mit",
        "gold_provenance": "human",
    }
    base.update(over)
    return R.Record.from_json(base)


def test_valid_record_roundtrips():
    rec = _rec()
    assert R.Record.from_json(rec.to_json()) == rec
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.id = "other"  # frozen


@pytest.mark.parametrize(
    "over, err",
    [
        ({"questions": {"q": {"type": "essay", "criteria": ["a", "b"]}}}, R.UnknownQuestionType),
        ({"questions": {"q": {"type": "choice", "criteria": ["a"]}}}, R.TooFewOptions),
        ({"questions": {"q": {"type": "score", "criteria": ["a"]}}}, R.TooFewLevels),
        ({"gold": {"q": {"label": "zzz"}}}, R.GoldOutsideOptions),
        ({"gold": {"q": {"label": "a", "probabilities": {"zzz": 1.0}}}}, R.GoldOutsideOptions),
        ({"gold": {"q": {"label": "a", "probabilities": {"a": 1.5}}}}, R.BadProbability),
        ({"gold": {}}, R.MissingField),
        ({"gold_provenance": "vibes"}, R.UnknownGoldProvenance),
    ],
)
def test_named_errors(over, err):
    with pytest.raises(err) as ei:
        _rec(**over)
    assert ei.value.code in str(ei.value)


def test_noul_gold_is_a_probability():
    qs = {"q": {"type": "noul", "criteria": {"false": "", "true": ""}}}
    rec = _rec(questions=qs, gold={"q": {"probability_true": 0.25}})
    assert rec.gold["q"].probability_true == 0.25
    with pytest.raises(R.BadProbability):
        _rec(questions=qs, gold={"q": {"probability_true": 2}})
    with pytest.raises(R.MissingField):
        _rec(questions=qs, gold={"q": {"label": "true"}})


def test_typed_decisions_real_row_loads_and_roundtrips():
    row = _fx("real_typed_decisions_row.json")
    rec = R.from_typed_decisions_row(row)
    assert rec.id == row["id"]
    assert rec.gold_provenance == "model-generated"
    assert rec.licence == "Apache-2.0"
    assert rec.source == R.TYPED_DECISIONS_SOURCE
    assert rec.questions["action"].names == ("continue", "human_review", "observe", "stop")
    assert rec.questions["needs_review"].type == "noul"
    assert rec.gold["needs_review"].probability_true == 0.356667
    assert rec.gold["action"].probabilities["continue"] == 0.583333
    risk = rec.questions["risk"]
    assert risk.type == "score"
    assert risk.names == ("0", "1", "2", "3")
    assert rec.gold["risk"].label == "1"
    assert rec.gold["risk"].score == 0.79
    assert R.Record.from_json(json.loads(json.dumps(rec.to_json()))) == rec


@pytest.mark.parametrize(
    "name, gold, licence",
    [
        (
            "real_jev_bench_banking77_row.json",
            "getting_virtual_card",
            "cc-by-4.0 (PolyAI; mirror: mit)",
        ),
        ("real_jev_bench_go_emotions_row.json", "nervousness", "apache-2.0"),
    ],
)
def test_jev_bench_real_rows_load_and_roundtrip(name, gold, licence):
    row = _fx(name)
    rec = R.from_jev_bench_row(row)
    assert rec.id == row["id"]
    assert rec.gold_provenance == "human"
    assert rec.licence == licence  # verbatim from meta.license
    assert rec.source.startswith(R.JEV_BENCH_SOURCE)
    assert rec.gold["q"].label == gold
    assert len(rec.questions["q"].names) > 2
    assert R.Record.from_json(json.loads(json.dumps(rec.to_json()))) == rec


def test_jev_bench_null_descriptions_and_soft_label_kept():
    rec = R.from_jev_bench_row(_fx("real_jev_bench_go_emotions_row.json"))
    assert dict(rec.questions["q"].criteria)["anger"] == ""
    assert rec.gold["q"].probabilities["nervousness"] == pytest.approx(2 / 7)
    assert (
        R.from_jev_bench_row(_fx("real_jev_bench_banking77_row.json")).gold["q"].probabilities
        is None
    )


def test_jev_bench_hard_noul_and_score_labels():
    row = dict(_fx("real_jev_bench_banking77_row.json"), soft_label=None)
    noul = {"type": "noul", "criteria": {"false": "no", "true": "yes"}}
    rec = R.from_jev_bench_row(dict(row, question=json.dumps(noul), label="1"))
    assert rec.gold["q"].probability_true == 1.0
    score = {"type": "score", "criteria": ["lo", "mid", "hi"]}
    rec = R.from_jev_bench_row(dict(row, question=json.dumps(score), label="2"))
    assert rec.questions["q"].names == ("0", "1", "2")
    assert rec.gold["q"].label == "2"


def test_domain_renders_as_one_choice_question():
    rec = R.from_domain(
        DOMAIN,
        request="turn on the kitchen lamp",
        gold="lamp_on",
        gold_provenance="human",
        licence=L,
    )
    assert list(rec.questions) == [R.DOMAIN_QID]
    q = rec.questions[R.DOMAIN_QID]
    assert q.type == "choice"
    assert q.names == tuple(DOMAIN.names())
    assert rec.gold_provenance == "human"
    model = R.from_domain(
        DOMAIN, request="x", gold="lamp_on", gold_provenance="model-generated", licence=L
    )
    assert model.gold_provenance == "model-generated"
    with pytest.raises(R.GoldOutsideOptions):
        R.from_domain(DOMAIN, request="x", gold="nope", gold_provenance="human", licence=L)


def test_import_pulls_no_third_party():
    code = (
        "import sys; import jev_factory.distill.records as r;"
        "std = set(sys.stdlib_module_names);"
        "bad = sorted(m for m in sys.modules if '.' not in m and m not in std"
        " and m not in ('jev_factory','__main__') and not m.startswith('_'));"
        "print(bad); assert 'pyarrow' not in sys.modules"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_source_metadata_defaults_to_none_and_roundtrips():
    rec = _rec()
    assert rec.hf_id is None
    assert rec.group is None
    assert "hf_id" not in rec.to_json()
    assert "group" not in rec.to_json()
    rich = _rec(hf_id="mteb/banking77", group="agent_trace_observability")
    assert R.Record.from_json(rich.to_json()) == rich
    assert rich.to_json()["hf_id"] == "mteb/banking77"
    assert rich.to_json()["group"] == "agent_trace_observability"


def test_jev_bench_row_carries_meta_hf_id():
    rec = R.from_jev_bench_row(_fx("real_jev_bench_banking77_row.json"))
    assert rec.hf_id == "mteb/banking77"
    assert rec.group is None


def test_typed_decisions_row_carries_workflow_as_group():
    row = _fx("real_typed_decisions_row.json")
    rec = R.from_typed_decisions_row(row)
    assert rec.group == row["workflow"] == "agent_trace_observability"
    assert rec.hf_id is None
