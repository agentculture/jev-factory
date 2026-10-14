"""t17: assemble the distill training set (precedence, filters, one teacher build, freeze)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

from jev_factory.backbones.pointer.train_distill import load_items
from jev_factory.cli._errors import CliError
from jev_factory.core import leakage
from jev_factory.distill import assemble, prereg
from jev_factory.distill.audit import AUDIT_ALT_REL, AUDIT_REL
from jev_factory.distill.corrections import Correction, read_corrections, write_corrections
from jev_factory.distill.dataset import Dataset, SealedSide, write_dataset
from jev_factory.distill.records import Gold, Question, Record
from jev_factory.review import corrections as review

HP = {"ordinal_smoothing": 0.1, "temperature": {"choice": 1, "noul": 1, "score": 1}}
CHOICE = Question("choice", (("fix", "repair it"), ("wait", "do nothing"), ("ask", "ask")))
NOUL = Question("noul", (("false", ""), ("true", "")))
SCORE = Question("score", (("0", "bad"), ("1", "ok"), ("2", "good")))

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()


def _state(i: int) -> str:
    """A distinct, non-overlapping state per index (no accidental near-duplicates)."""
    rng = random.Random(i)
    return " ".join(f"{rng.choice(_WORDS)}{rng.randrange(10**6)}" for _ in range(12))


def _rec(rid, i, source="good", prov="human", split="train", gold="fix", state=None):
    return Record(
        id=rid,
        state=state if state is not None else _state(i),
        questions={"a": CHOICE, "b": NOUL, "c": SCORE},
        gold={
            "a": Gold(label=gold),
            "b": Gold(label="true", probability_true=1.0),
            "c": Gold(label="1", probabilities={"0": 0.1, "1": 0.7, "2": 0.2}),
        },
        split=split,
        source=source,
        licence="MIT",
        gold_provenance=prov,
    )


def _sealed(recs):
    return SealedSide("", "x", tuple(r.id for r in recs), len(recs), ())


def _dataset(tmp_path, train, fit=(), selection=(), held=(), test=()):
    fit = list(fit) or [_rec("f1", 101)]
    selection = list(selection) or [_rec("s1", 102)]
    held = list(held) or [_rec("h1", 103)]
    test = list(test) or [_rec("x1", 104, split="test")]
    ds = Dataset(
        train=tuple(train),
        fit=tuple(fit),
        selection=tuple(selection),
        held_out=_sealed(held),
        test=_sealed(test),
        manifest={"stage": "distill-dataset"},
        _held_out_records=tuple(held),
        _test_records=tuple(test),
    )
    out = tmp_path / "dataset"
    write_dataset(ds, out)
    return out


def _labels(rid, *, build="clef-b1", fix=0.8, p_true=0.9, score=None, error=None, qids=None):
    rows = []
    dists = {
        "a": ("choice", {"fix": fix, "wait": (1 - fix) / 2, "ask": (1 - fix) / 2}, None),
        "b": ("noul", None, p_true),
        "c": ("score", score or {"0": 0.2, "1": 0.6, "2": 0.2}, None),
    }
    for qid, (kind, dist, pt) in dists.items():
        if qids is not None and qid not in qids:
            continue
        rows.append(
            {
                "record_id": rid,
                "question_id": qid,
                "kind": kind,
                "distribution": None if error else dist,
                "p_true": None if error else pt,
                "teacher_error": error,
                "teacher_build_id": build,
            }
        )
    return rows


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _audit(run_dir, weak=(), build="clef-b1", rel=AUDIT_REL):
    sources = [
        {"source": s, "gold_provenance": "human", "teacher_weak": s in weak, "flaggable": True}
        for s in ("good", "weak")
    ]
    table = {
        "stage": "teacher-audit",
        "floor": 0.6,
        "teacher_build_id": build,
        "sources": sources,
        "teacher_weak_sources": sorted(weak),
    }
    path = run_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table), encoding="utf-8")


def _prereg(filters=("all", "agrees_with_gold")):
    doc = prereg.example_doc()
    doc["knob_grid"]["teacher_filter"] = list(filters)
    return prereg.validate(doc)


def _correction(rid, qid="a", label="ask", dist=None):
    return Correction(
        record_id=rid,
        question_id=qid,
        author="operator",
        date="2026-10-01",
        reason="weak",
        label=label,
        distribution=dist,
        text=f"state of {rid}",
    )


def _decide(cpath, corr, action, after=None):
    decision = {"action": action, "record_id": corr.record_id, "question_id": corr.question_id}
    if after is not None:
        decision["after"] = after
    rec = review.make_record(read_corrections(cpath), decision)
    review.append_record(review.review_path(cpath), rec)


class _Env:
    """One run: dataset dir, run dir, label files, corrections file."""

    def __init__(self, tmp_path, train, weak=(), **sides):
        self.tmp = tmp_path
        self.dataset = _dataset(tmp_path, train, **sides)
        self.run = tmp_path / "run"
        _audit(self.run, weak)
        self.clef = tmp_path / "clef" / "teacher_labels.jsonl"
        self.alt = None
        self.corrections = None

    def clef_rows(self, rows):
        _write_jsonl(self.clef, rows)

    def alt_rows(self, rows, audited="alt-b7", weak=()):
        self.alt = _write_jsonl(self.tmp / "alt" / "teacher_labels.jsonl", rows)
        if audited is not None:
            _audit(self.run, weak, build=audited, rel=AUDIT_ALT_REL)

    def go(self, teacher_filter="all", reg=None, deviation_id=None):
        return assemble.assemble(
            self.run,
            self.dataset,
            prereg=reg or _prereg(),
            teacher_filter=teacher_filter,
            clef_labels=self.clef,
            alt_labels=self.alt,
            corrections_path=self.corrections,
            deviation_id=deviation_id,
        )


def _rows(manifest):
    path = manifest["output"]["path"]
    text = open(path, encoding="utf-8").read()
    return {(r["record_id"], r["question_id"]): r for r in map(json.loads, text.splitlines())}


# ----------------------------------------------------------------- precedence


def test_clef_teacher_plus_gold_one_row_per_record_question(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("r2", 2)])
    env.clef_rows(_labels("r1") + _labels("r2"))
    m = env.go()
    rows = _rows(m)
    assert sorted(rows) == [(r, q) for r in ("r1", "r2") for q in ("a", "b", "c")]
    a = rows[("r1", "a")]
    assert a["target_source"] == "clef+gold"
    assert a["precedence_rule"] == "gold+teacher"
    assert a["gold"] == {"label": "fix"}
    assert a["teacher"] == pytest.approx({"fix": 0.8, "wait": 0.1, "ask": 0.1})
    assert a["teacher_build_id"] == "clef-b1" and a["correction_id"] is None
    assert a["teacher_filter"] == "kept"
    assert a["options"] == [["fix", "repair it"], ["wait", "do nothing"], ["ask", "ask"]]
    assert a["kind"] == "choice" and a["state"] == _state(1) and a["weight"] == 1.0
    b = rows[("r1", "b")]
    assert b["gold"] == {"p_true": 1.0}
    assert b["teacher"] == pytest.approx({"false": 0.1, "true": 0.9})
    c = rows[("r1", "c")]
    assert c["gold"]["probabilities"] == pytest.approx({"0": 0.1, "1": 0.7, "2": 0.2})


def test_alternative_teacher_takes_precedence_over_clef(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1", fix=0.8))
    env.alt_rows(_labels("r1", build="alt-b7", fix=0.6))
    rows = _rows(env.go())
    a = rows[("r1", "a")]
    assert a["target_source"] == "alt_teacher+gold"
    assert a["teacher_build_id"] == "alt-b7"
    assert a["teacher"]["fix"] == pytest.approx(0.6)


def test_approved_correction_replaces_both_targets(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    env.alt_rows(_labels("r1", build="alt-b7"))
    corr = _correction("r1", "a", label="ask")
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [corr])
    _decide(env.corrections, corr, "approve")
    rows = _rows(env.go())
    a = rows[("r1", "a")]
    assert a["target_source"] == "correction" and a["precedence_rule"] == "correction"
    assert a["gold"] == {"label": "ask"} and a["teacher"] is None
    assert a["correction_id"] == "r1/a" and a["correction_status"] == "approved"
    assert a["teacher_build_id"] is None
    assert rows[("r1", "b")]["target_source"] == "alt_teacher+gold"


def test_edited_correction_uses_the_reviewed_target(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    corr = _correction("r1", "b", label="false")
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [corr])
    _decide(env.corrections, corr, "edit", after={"distribution": {"false": 0.25, "true": 0.75}})
    rows = _rows(env.go())
    b = rows[("r1", "b")]
    assert b["target_source"] == "correction" and b["correction_status"] == "edited"
    assert b["gold"] == {"p_true": pytest.approx(0.75)} and b["teacher"] is None


def test_only_approved_or_edited_corrections_are_used(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("r2", 2), _rec("r3", 3)])
    env.clef_rows(_labels("r1") + _labels("r2") + _labels("r3"))
    proposed = _correction("r1")
    rejected = _correction("r2")
    claims = dataclasses.replace(_correction("r3"), status="approved")
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [proposed, rejected, claims])
    _decide(env.corrections, rejected, "reject")
    rows = _rows(env.go())
    for rid in ("r1", "r2", "r3"):
        assert rows[(rid, "a")]["target_source"] == "clef+gold"
        assert rows[(rid, "a")]["gold"] == {"label": "fix"}


def test_teacher_weak_source_takes_gold_only_or_correction(tmp_path):
    env = _Env(
        tmp_path,
        [_rec("w1", 1, source="weak"), _rec("w2", 2, source="weak"), _rec("g1", 3)],
        weak=("weak",),
    )
    env.clef_rows(_labels("w1") + _labels("w2") + _labels("g1"))
    corr = _correction("w2", "a", label="wait")
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [corr])
    _decide(env.corrections, corr, "approve")
    rows = _rows(env.go())
    w1 = rows[("w1", "a")]
    assert w1["target_source"] == "human_gold" and w1["precedence_rule"] == "teacher_weak"
    assert w1["teacher"] is None and w1["gold"] == {"label": "fix"}
    assert w1["teacher_filter"] == "not_applied:teacher_weak"
    assert rows[("w2", "a")]["target_source"] == "correction"
    assert rows[("w2", "b")]["target_source"] == "human_gold"
    assert rows[("g1", "a")]["target_source"] == "clef+gold"


def test_teacher_error_nulls_the_teacher_slot(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("r2", 2)])
    env.clef_rows(_labels("r1", error="timeout") + _labels("r2", qids=("a",)))
    rows = _rows(env.go())
    a = rows[("r1", "a")]
    assert a["teacher"] is None and a["gold"] == {"label": "fix"}
    assert a["target_source"] == "gold_only:teacher_error"
    assert a["teacher_filter"] == "no_teacher:teacher_error"
    assert rows[("r2", "b")]["target_source"] == "gold_only:no_teacher_label"


# ----------------------------------------------------------------- teacher filter


def test_agrees_with_gold_filter_drops_disagreeing_teacher(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    # a: teacher argmax "wait" vs gold "fix"; b: teacher p_true 0.2 vs gold true;
    # c: teacher argmax "1" == gold argmax "1".
    env.clef_rows(_labels("r1", fix=0.1, p_true=0.2))
    m = env.go(teacher_filter="agrees_with_gold")
    rows = _rows(m)
    for qid in ("a", "b"):
        assert rows[("r1", qid)]["teacher"] is None
        assert rows[("r1", qid)]["target_source"] == "gold_only:filtered"
        assert rows[("r1", qid)]["teacher_filter"] == "dropped:disagrees_with_gold"
    assert rows[("r1", "c")]["teacher"] is not None
    assert rows[("r1", "c")]["teacher_filter"] == "kept"
    assert m["knobs"]["teacher_filter"] == "agrees_with_gold"
    assert m["counts"]["teacher_filter"]["dropped:disagrees_with_gold"] == 2


def test_filter_all_keeps_disagreeing_teacher(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1", fix=0.1, p_true=0.2))
    rows = _rows(env.go(teacher_filter="all"))
    assert rows[("r1", "a")]["teacher"] is not None
    assert rows[("r1", "a")]["teacher_filter"] == "kept"


def test_unregistered_filter_knob_is_refused(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    with pytest.raises(CliError, match="teacher_filter"):
        env.go(teacher_filter="agrees_with_gold", reg=_prereg(filters=("all",)))
    with pytest.raises(CliError, match="teacher_filter"):
        env.go(teacher_filter="sometimes")


# ----------------------------------------------------------------- one teacher build


def test_mixed_teacher_build_ids_refused_without_deviation(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("r2", 2)])
    env.clef_rows(_labels("r1") + _labels("r2", build="clef-b2"))
    with pytest.raises(CliError, match="teacher build"):
        env.go()
    assert not (env.run / assemble.OUT_REL).exists()
    # a deviation allows a mixed set, but never an unaudited build (see fx4 tests)
    with pytest.raises(CliError, match="audit"):
        env.go(deviation_id="d9")


def test_alt_falling_back_to_clef_is_a_mixed_build(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("r2", 2)])
    env.clef_rows(_labels("r1") + _labels("r2"))
    env.alt_rows(_labels("r1", build="alt-b7"))  # r2 has no alternative label
    with pytest.raises(CliError, match="teacher build"):
        env.go()
    rows = _rows(env.go(deviation_id="d9"))
    assert rows[("r2", "a")]["target_source"] == "clef+gold"
    assert rows[("r1", "a")]["target_source"] == "alt_teacher+gold"


def test_unused_teacher_rows_do_not_count_as_mixed(tmp_path):
    env = _Env(tmp_path, [_rec("w1", 1, source="weak"), _rec("g1", 2)], weak=("weak",))
    env.clef_rows(_labels("w1", build="clef-old") + _labels("g1"))
    m = env.go()
    assert m["teacher_build_ids"] == ["clef-b1"] and m["deviation_id"] is None


# ----------------------------------------------------------------- train side only


def test_refuses_a_non_train_record(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1), _rec("bad", 2, split="test")])
    env.clef_rows(_labels("r1") + _labels("bad"))
    with pytest.raises(CliError, match="non-train"):
        env.go()


def test_refuses_teacher_rows_for_records_off_the_train_side(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1") + _labels("f1"))  # f1 is a fit record
    with pytest.raises(CliError, match="train side"):
        env.go()


def test_refuses_a_trainable_correction_off_the_train_side(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    corr = _correction("s1", "a")  # s1 is a selection record
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [corr])
    _decide(env.corrections, corr, "approve")
    with pytest.raises(CliError, match="train side"):
        env.go()


def test_refuses_without_the_teacher_audit(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    (env.run / AUDIT_REL).unlink()
    with pytest.raises(CliError, match="audit"):
        env.go()


# ----------------------------------------------------------------- leakage


@pytest.mark.parametrize("side", ["fit", "selection", "held", "test"])
def test_exact_duplicate_of_a_protected_side_fails_closed(tmp_path, side):
    secret = _state(1)
    twin = _rec("p1", 50, state=secret.upper(), split="test" if side == "test" else "train")
    env = _Env(tmp_path, [_rec("r1", 1)], **{side: [twin]})
    env.clef_rows(_labels("r1"))
    with pytest.raises(CliError) as err:
        env.go()
    assert "r1" in err.value.message and "p1" in err.value.message
    assert secret.split()[0] not in str(err.value) and secret.split()[0] not in err.value.message
    assert not (env.run / assemble.OUT_REL).exists()


def test_near_duplicate_of_held_out_fails_closed(tmp_path):
    words = _state(1).split()
    near = " ".join(words[:-1] + ["changed"])
    env = _Env(tmp_path, [_rec("r1", 1)], held=[_rec("h9", 60, state=near)])
    env.clef_rows(_labels("r1"))
    with pytest.raises(CliError, match="near-duplicate"):
        env.go()


def test_leakage_candidates_match_brute_force():
    rng = random.Random(3)
    base = [" ".join(rng.choice(_WORDS) for _ in range(rng.randrange(2, 14))) for _ in range(40)]
    train = [{"id": f"t{i}", "text": t} for i, t in enumerate(base)]
    prot_texts = []
    for t in base[:25]:
        w = t.split()
        if w and rng.random() < 0.5:
            w[rng.randrange(len(w))] = rng.choice(_WORDS)
        prot_texts.append(" ".join(w) or "zulu")
    protected = {"held_out": [{"id": f"p{i}", "text": t} for i, t in enumerate(prot_texts)]}
    brute = {(m["train_id"], m["protected_id"], m["kind"]) for m in _all_pairs(train, protected)}
    fast = {
        (m["train_id"], m["protected_id"], m["kind"])
        for m in assemble.leakage_matches(train, protected)
    }
    assert brute and fast == brute


def _all_pairs(train, protected):
    for t in train:
        for name, entries in protected.items():
            for p in entries:
                kind = leakage.match(t["text"], p["text"])
                if kind:
                    yield {"train_id": t["id"], "protected_id": p["id"], "kind": kind}


# ----------------------------------------------------------------- freeze


def test_freeze_writes_sha256_and_rows_load_in_the_trainer(tmp_path):
    env = _Env(
        tmp_path,
        [_rec("r1", 1), _rec("w1", 2, source="weak"), _rec("r2", 3)],
        weak=("weak",),
    )
    env.clef_rows(_labels("r1", fix=0.1) + _labels("w1") + _labels("r2"))
    corr = _correction("r2", "c", dist={"0": 0.5, "1": 0.5, "2": 0.0}, label=None)
    env.corrections = tmp_path / "corrections.jsonl"
    write_corrections(env.corrections, [corr])
    _decide(env.corrections, corr, "approve")
    m = env.go(teacher_filter="agrees_with_gold")
    path = env.run / assemble.OUT_REL
    assert m["output"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads((env.run / assemble.MANIFEST_REL).read_text(encoding="utf-8"))
    assert saved == m
    assert m["leakage"]["hits"] == 0
    assert set(m["leakage"]["protected"]) == {"fit", "selection", "held_out", "test"}
    assert m["inputs"]["dataset_train"]["sha256"]
    items = load_items(path, HP)
    assert len(items) == 9
    assert m["counts"]["rows"] == 9


def test_refreezing_a_different_set_needs_a_deviation(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1", fix=0.1))  # disagrees with gold on "a"
    first = env.go()
    assert env.go()["output"]["sha256"] == first["output"]["sha256"]  # identical: fine
    with pytest.raises(CliError, match="frozen"):
        env.go(teacher_filter="agrees_with_gold")
    assert env.go(teacher_filter="agrees_with_gold", deviation_id="d10")["deviation_id"] == "d10"


# ----------------------------------------------------------------- fx4: audit names the build


def test_clef_labels_from_an_unaudited_build_are_refused(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1", build="clef-b2"))  # audit names clef-b1
    with pytest.raises(CliError, match="audited") as exc:
        env.go()
    assert "clef-b2" in exc.value.message and "clef-b1" in exc.value.message
    assert not (env.run / assemble.OUT_REL).exists()
    with pytest.raises(CliError, match="audited"):
        env.go(deviation_id="d9")


def test_alternative_teacher_needs_its_own_audit(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    env.alt_rows(_labels("r1", build="alt-b7"), audited=None)
    with pytest.raises(CliError, match="audit"):
        env.go()


def test_alternative_teacher_build_must_match_its_audit(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    env.alt_rows(_labels("r1", build="alt-b8"), audited="alt-b7")
    with pytest.raises(CliError, match="audited"):
        env.go()


def test_audited_alternative_build_is_accepted_and_recorded(tmp_path):
    env = _Env(tmp_path, [_rec("r1", 1)])
    env.clef_rows(_labels("r1"))
    env.alt_rows(_labels("r1", build="alt-b7"))
    m = env.go()
    assert m["teacher_build_ids"] == ["alt-b7"]
    assert m["audited_builds"] == {"clef": "clef-b1", "alt": "alt-b7"}


def test_weak_flags_come_from_the_audit_of_the_teacher_used(tmp_path):
    env = _Env(tmp_path, [_rec("w1", 1, source="weak")], weak=())  # clef audit: not weak
    env.clef_rows(_labels("w1"))
    env.alt_rows(_labels("w1", build="alt-b7"), weak=("weak",))  # alt audit: weak
    m = env.go()
    assert m["teacher_weak_sources"] == ["weak"]
    assert _rows(m)[("w1", "a")]["target_source"] == "human_gold"
    # clef only: its own (clean) audit governs
    env2 = _Env(tmp_path / "x", [_rec("w1", 1, source="weak")], weak=())
    env2.clef_rows(_labels("w1"))
    m2 = env2.go()
    assert m2["teacher_weak_sources"] == []
    assert _rows(m2)[("w1", "a")]["target_source"] == "clef+gold"
