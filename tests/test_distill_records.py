"""The typed-decision record contract and its source loaders (distillery t1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_factory.distill import records as R
from tests.fixtures.toy_domain import DOMAIN

FIX = Path(__file__).parent / "fixtures" / "typed_decisions"


def _rows(name):
    return json.loads((FIX / name).read_text())


def _rec(**over):
    base = {
        "id": "r1",
        "state": "s",
        "questions": {"q": {"type": "choice", "criteria": [["a", ""], ["b", ""]]}},
        "gold": {"q": "a"},
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
    with pytest.raises(Exception):
        rec.id = "other"  # frozen


@pytest.mark.parametrize(
    "over, err",
    [
        ({"questions": {"q": {"type": "essay", "criteria": ["a", "b"]}}}, R.UnknownQuestionType),
        ({"questions": {"q": {"type": "choice", "criteria": ["a"]}}}, R.TooFewOptions),
        (
            {"questions": {"q": {"type": "score", "criteria": ["a"]}}, "gold": {"q": "a"}},
            R.TooFewLevels,
        ),
        ({"gold": {"q": "zzz"}}, R.GoldOutsideOptions),
        ({"gold": {}}, R.MissingField),
        ({"gold_provenance": "vibes"}, R.UnknownGoldProvenance),
    ],
)
def test_named_errors(over, err):
    with pytest.raises(err) as ei:
        _rec(**over)
    assert ei.value.code in str(ei.value)


def test_noul_accepts_none_sentinel():
    rec = _rec(
        questions={"q": {"type": "noul", "criteria": ["a", "b"]}}, gold={"q": R.NONE_OF_THE_ABOVE}
    )
    assert rec.gold["q"] == "none"


def test_typed_decisions_row_loads_and_roundtrips():
    row = _rows("typed_decisions_rows.json")[0]
    rec = R.from_typed_decisions_row(row)
    assert rec.gold_provenance == "model-generated"
    assert rec.source == R.TYPED_DECISIONS_SOURCE
    assert rec.questions["q1"].names == ("env", "flaky", "code")
    assert rec.questions["q2"].type == "score"
    assert R.Record.from_json(json.loads(json.dumps(rec.to_json()))) == rec
    # id is content-derived, not positional
    assert R.from_typed_decisions_row(dict(row)).id == rec.id


def test_jev_bench_row_loads_and_roundtrips():
    row = _rows("jev_bench_rows.json")[0]
    rec = R.from_jev_bench_row(row)
    assert rec.id == "jb-0001"
    assert rec.gold_provenance == "human"
    assert rec.split == "test"
    assert rec.source.startswith(R.JEV_BENCH_SOURCE)
    assert rec.questions["q"].names == ("yes", "no")
    assert R.Record.from_json(json.loads(json.dumps(rec.to_json()))) == rec


def test_domain_renders_as_one_choice_question():
    rec = R.from_domain(
        DOMAIN, request="turn on the kitchen lamp", gold="lamp_on", gold_provenance="human"
    )
    assert list(rec.questions) == [R.DOMAIN_QID]
    q = rec.questions[R.DOMAIN_QID]
    assert q.type == "choice"
    assert q.names == tuple(DOMAIN.names())
    assert rec.gold_provenance == "human"
    model = R.from_domain(DOMAIN, request="x", gold="lamp_on", gold_provenance="model-generated")
    assert model.gold_provenance == "model-generated"
    with pytest.raises(R.GoldOutsideOptions):
        R.from_domain(DOMAIN, request="x", gold="nope", gold_provenance="human")


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
