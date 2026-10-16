"""Run dataset build: licence allowlist, grouped folds, sealed sides, length filter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import dataset, records

FIX = Path(__file__).parent / "fixtures" / "typed_decisions"


def _row(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _td(i: int, split: str = "train", state: str | None = None) -> records.Record:
    row = _row("real_typed_decisions_row.json")
    row["id"] = f"td_{i:04d}"
    row["split"] = split
    if state is not None:
        row["state"] = json.dumps({"text": state})
    else:
        row["state"] = json.dumps({"n": i})
    return records.from_typed_decisions_row(row)


def _fit_slots(row: dict, slots: int = 24) -> dict:
    """Trim a real row's question to the student's 24 slots (the real go_emotions has 28)."""
    q = row["question"]
    q = json.loads(q) if isinstance(q, str) else dict(q)
    crit = q["criteria"]
    if isinstance(crit, dict):
        names = list(crit)[:slots]
        q["criteria"] = {k: crit[k] for k in names}
    else:
        names = crit[:slots]
        q["criteria"] = names
    row["question"] = json.dumps(q)
    soft = row.get("soft_label")
    if soft:
        soft = json.loads(soft) if isinstance(soft, str) else dict(soft)
        row["soft_label"] = json.dumps({k: v for k, v in soft.items() if k in names})
    return row


def _jb(name: str, i: int, split: str = "train", licence: str | None = None) -> records.Record:
    row = _row(name)
    if "go_emotions" in name:
        _fit_slots(row)
    row["id"] = f"{row['source']}/{split}/{i}"
    row["split"] = split
    row["state"] = json.dumps(f"text {i}")
    if licence is not None:
        meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else dict(row["meta"])
        meta["license"] = licence
        row["meta"] = json.dumps(meta)
    return records.from_jev_bench_row(row)


def _words(text: str) -> int:
    return len(text.split())


def _corpus() -> list[records.Record]:
    out = [_td(i) for i in range(40)]
    out += [_td(100 + i, "test") for i in range(5)]
    out += [_jb("real_jev_bench_go_emotions_row.json", i) for i in range(40)]
    out += [_jb("real_jev_bench_go_emotions_row.json", 100 + i, "test") for i in range(5)]
    return out


def _build(recs=None, **kw):
    return dataset.build_dataset(recs or _corpus(), token_counter=_words, seed=7, **kw)


# --- licence filter ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Apache-2.0", True),
        ("apache-2.0", True),
        (" MIT ", True),
        ("cc-by-4.0 (PolyAI; mirror: mit)", False),
        ("cc-by-4.0", True),
        (" CC-BY-4.0 ", True),
        ("cc-by-4.0 or mit", False),
        ("apache-2.0 OR gpl-3.0", False),
        ("mit/apache-2.0", False),
        ("", False),
        ("unknown", False),
    ],
)
def test_licence_allowlist_is_explicit_and_rejects_compound(text, ok):
    assert dataset.licence_allowed(text) is ok


def test_real_banking77_row_is_excluded_and_go_emotions_kept():
    recs = [
        _jb("real_jev_bench_banking77_row.json", 1),
        _jb("real_jev_bench_go_emotions_row.json", 1),
        _td(1),
    ]
    ds = _build(recs)
    kept = {r.source for r in ds.all_records()}
    assert not any("banking77" in s for s in kept)
    table = ds.manifest["licences"]
    bank = table["Praveenrajus/jev-bench:banking77"]
    assert bank["allowed"] is False
    assert bank["excluded"] == 1
    assert bank["licences"] == ["cc-by-4.0 (PolyAI; mirror: mit)"]
    assert table["Praveenrajus/jev-bench:go_emotions"]["allowed"] is True


# --- folds and sealed -------------------------------------------------------


def test_folds_are_disjoint_deterministic_and_from_train_only():
    a, b = _build(), _build()
    assert [r.id for r in a.fit] == [r.id for r in b.fit]
    assert [r.id for r in a.train] == [r.id for r in b.train]
    train = {r.id for r in a.train}
    fit, sel = {r.id for r in a.fit}, {r.id for r in a.selection}
    held = set(a.held_out.ids)
    test = set(a.test.ids)
    assert train
    assert fit
    assert sel
    assert held
    assert test
    four = [train, fit, sel, held]
    for i, x in enumerate(four):
        for y in four[i + 1 :]:
            assert not (x & y)
    assert not (test & (train | fit | sel | held))
    assert all(r.split == "train" for r in a.train + a.fit + a.selection)
    assert all(i.startswith(("td_01", "go_emotions/test")) for i in test)
    assert len(test) == 10


def test_each_sources_own_test_split_is_the_test_side():
    ds = _build()
    assert set(ds.test.ids) == {r.id for r in _corpus() if r.split == "test"}
    assert ds.manifest["counts"]["test"] == 10


def test_selection_groups_never_straddle_folds():
    recs = [_td(i, state="same") for i in range(10)] + [_td(50 + i) for i in range(30)]
    ds = _build(recs)
    same = {"text": "same"}
    sides = [
        any(json.loads(r.state) == same for r in ds.train),
        any(json.loads(r.state) == same for r in ds.fit),
        any(json.loads(r.state) == same for r in ds.selection),
        any(json.loads(r.state) == same for r in ds._held_out_records),
    ]
    assert sum(sides) == 1


def test_duplicate_ids_refused():
    recs = [_td(1), _td(1)]
    with pytest.raises(CliError):
        _build(recs)


def test_unknown_splits_are_not_trained_on_and_are_counted():
    recs = _corpus() + [replace(_td(900), split="validation")]
    ds = _build(recs)
    assert "td_0900" not in {r.id for r in ds.train + ds.fit + ds.selection}
    assert ds.manifest["unused_splits"] == {"validation": 1}


# --- length filter ----------------------------------------------------------


def test_over_length_records_dropped_not_truncated_with_per_source_counts():
    long_state = " ".join(["w"] * 5000)
    recs = _corpus() + [_td(700, state=long_state), _td(701, "test", state=long_state)]

    def counter(text: str) -> int:
        return len(text.split())

    ds = dataset.build_dataset(recs, token_counter=counter, seed=7, max_tokens=4096)
    ids = {r.id for r in ds.all_records()} | set(ds.held_out.ids) | set(ds.test.ids)
    assert "td_0700" not in ids
    assert "td_0701" not in ids
    assert ds.manifest["dropped_too_long"] == {"LocalLLaMA/typed-decisions": 2}
    assert ds.manifest["max_tokens"] == 4096
    for r in ds.all_records():
        assert counter(dataset.render_prompt(r)) <= 4096


def test_boundary_exactly_max_is_kept():
    r = _td(1)
    n = _words(dataset.render_prompt(r))
    ds = dataset.build_dataset([r], token_counter=_words, seed=1, max_tokens=n)
    assert ds.manifest["dropped_too_long"] == {}
    ds = dataset.build_dataset([r], token_counter=_words, seed=1, max_tokens=n - 1)
    assert ds.manifest["dropped_too_long"] == {"LocalLLaMA/typed-decisions": 1}


def test_render_prompt_includes_state_and_questions():
    r = _td(3)
    text = dataset.render_prompt(r)
    assert '"n": 3' in text
    for qid in r.questions:
        assert qid in text


def test_subprocess_counter_batches_through_student_python(tmp_path):
    fake = tmp_path / "py.sh"
    fake.write_text(
        "#!/bin/sh\nexec python3 -c "
        "'import json,sys; t=json.load(sys.stdin); "
        "print(json.dumps([len(x.split()) for x in t]))'\n"
    )
    fake.chmod(0o755)
    counter = dataset.subprocess_token_counter(str(fake), tmp_path / "tok")
    assert counter("a b c") == 3
    assert counter("x") == 1


def test_subprocess_counter_failure_is_a_cli_error(tmp_path):
    fake = tmp_path / "bad.sh"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    counter = dataset.subprocess_token_counter(str(fake), tmp_path / "tok")
    with pytest.raises(CliError):
        counter("a")


# --- sealed sides expose no text -------------------------------------------


def test_sealed_sides_expose_only_ids_counts_sha(tmp_path):
    ds = _build()
    out = dataset.write_dataset(ds, tmp_path)
    for name in ("held_out", "test"):
        side = dataset.load_sealed(out[name])
        assert side.count == len(side.ids) > 0
        assert len(side.sha256) == 64
        assert set(vars(side)) == {"path", "sha256", "ids", "count", "by_source"}
        assert side.sha256 == dataset.sha256_bytes(Path(out[name]).read_bytes())
        assert side.sha256 == getattr(ds, name).sha256
        assert side.ids == getattr(ds, name).ids


def test_sealed_loader_discards_state_text(tmp_path):
    ds = _build(
        [_td(i, state=f"SECRET{i}") for i in range(30)] + [_td(99, "test", state="SECRETT")]
    )
    out = dataset.write_dataset(ds, tmp_path)
    side = dataset.load_sealed(out["test"])
    assert "SECRET" not in repr(side)
    assert "SECRET" not in str(vars(side))
    assert not hasattr(side, "records")
    assert not hasattr(side, "entries")
    # the dataset object holds the sealed sides as SealedSide, not records
    assert type(ds.held_out).__name__ == "SealedSide"
    assert "SECRET" not in repr(ds.held_out) + repr(ds.test)


def test_write_dataset_manifest_and_files(tmp_path):
    ds = _build()
    out = dataset.write_dataset(ds, tmp_path)
    man = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert man["stage"] == "distill-dataset"
    assert set(man["outputs"]) == {"train", "fit", "selection", "held_out", "test"}
    for name, info in man["outputs"].items():
        assert info["sha256"] == dataset.sha256_bytes(Path(out[name]).read_bytes())
        assert info["count"] == man["counts"][name]
    assert man["knobs"]["seed"] == 7
    assert "licences" in man
    assert "dropped_too_long" in man
    # records round-trip
    back = dataset.read_records(out["fit"])
    assert [r.id for r in back] == [r.id for r in ds.fit]
    assert man["knobs"]["train_fraction"] == dataset.DEFAULT_TRAIN_FRACTION
    assert man["knobs"]["fit_fraction"] == dataset.DEFAULT_FIT_FRACTION
    assert man["knobs"]["held_out_fraction"] == dataset.DEFAULT_HELD_OUT_FRACTION


def test_test_rows_never_in_training_files(tmp_path):
    ds = _build()
    out = dataset.write_dataset(ds, tmp_path)
    seen = {r.id for name in ("train", "fit", "selection") for r in dataset.read_records(out[name])}
    assert not (seen & set(ds.test.ids))
    assert not (seen & set(ds.held_out.ids))


def test_group_field_is_the_fold_key_when_present_else_state():
    from dataclasses import replace

    def sides(ds, ids):
        where = {}
        for name, recs in (
            ("train", ds.train),
            ("fit", ds.fit),
            ("selection", ds.selection),
            ("held", ds._held_out_records),
        ):
            for r in recs:
                where[r.id] = name
        return {where[i] for i in ids if i in where}

    # 12 workflows of 10 records each, every record with its own distinct state
    def corpus(grouped):
        out = []
        for w in range(12):
            for k in range(10):
                rec = _td(w * 10 + k)
                out.append(replace(rec, group=f"wf-{w}" if grouped else None))
        return out

    wf_ids = [[f"td_{w * 10 + k:04d}" for k in range(10)] for w in range(12)]
    ds = _build(corpus(True))
    assert all(len(sides(ds, ids)) == 1 for ids in wf_ids)
    assert len({n for ids in wf_ids for n in sides(ds, ids)}) > 1  # not all on one side
    # control: without group every distinct state is its own group, so workflows spread
    plain = _build(corpus(False))
    assert any(len(sides(plain, ids)) > 1 for ids in wf_ids)


# --- the separate training side (deviation d2) -------------------------------


def test_sides_constant_and_default_fractions():
    assert dataset.SIDES == ("train", "fit", "selection", "held_out", "test")
    assert (dataset.DEFAULT_TRAIN_FRACTION, dataset.DEFAULT_FIT_FRACTION) == (0.70, 0.15)


def test_train_is_the_only_trainable_side():
    ds = _build()
    assert {r.id for r in ds.trainable_records()} == {r.id for r in ds.train}
    assert not ({r.id for r in ds.trainable_records()} & {r.id for r in ds.fit + ds.selection})
    ev = {r.id for r in ds.eval_records()}
    assert ev == {r.id for r in ds.fit + ds.selection}
    assert {r.id for r in ds.all_records()} == {r.id for r in ds.train} | ev


def test_fractions_are_args_recorded_and_roughly_honoured():
    recs = [replace(_td(i), group=None) for i in range(200)]
    ds = _build(recs, train_fraction=0.5, fit_fraction=0.25)
    assert ds.manifest["knobs"]["train_fraction"] == 0.5
    assert ds.manifest["knobs"]["fit_fraction"] == 0.25
    n = len(ds.train) + len(ds.fit) + len(ds.selection) + len(ds._held_out_records)
    assert n == 200
    assert abs(len(ds.train) / (n - len(ds._held_out_records)) - 0.5) < 0.05
    assert abs(len(ds.fit) - len(ds.selection)) <= 0.1 * n
    with pytest.raises(CliError):
        _build(train_fraction=0.9, fit_fraction=0.2)


def test_train_ids_from_dataset_and_from_directory(tmp_path):
    ds = _build()
    out = dataset.write_dataset(ds, tmp_path)
    want = {r.id for r in ds.train}
    assert set(dataset.train_ids(ds)) == want
    assert set(dataset.train_ids(tmp_path)) == want
    assert Path(out["train"]).name == "train.jsonl"
    man = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert man["counts"]["train"] == len(want)


# --- t7: cc-by-4.0, slots, length reasons -----------------------------------


def _ccby(i: int, attribution: str | None = "Ann Author, CC BY 4.0") -> records.Record:
    return replace(_td(i), id=f"cc_{i:04d}", licence="CC-BY-4.0", attribution=attribution)


def _many_options(i: int, n: int) -> records.Record:
    base = _td(i)
    q = records.Question(type="choice", criteria=tuple((f"o{k}", "") for k in range(n)))
    return replace(
        base, id=f"opt_{i}_{n}", questions={"q": q}, gold={"q": records.Gold(label="o0")}
    )


def test_allowlist_admits_apache_mit_ccby_only():
    """AC1: apache-2.0, mit and cc-by-4.0 pass; anything else is refused."""
    assert dataset.licence_allowed("CC-BY-4.0") is True
    assert dataset.licence_allowed("cc-by-sa-4.0") is False
    assert dataset.licence_allowed("gpl-3.0") is False


def test_ccby_with_attribution_admitted_and_listed_in_manifest():
    """AC1: licence table carries per-licence counts and attribution strings."""
    recs = [_ccby(1), _ccby(2, "Bob, CC BY 4.0"), _td(3), replace(_td(4), licence="mit")]
    ds = _build(recs)
    table = ds.manifest["licence_table"]
    assert table["cc-by-4.0"]["count"] == 2
    assert table["cc-by-4.0"]["attributions"] == ["Ann Author, CC BY 4.0", "Bob, CC BY 4.0"]
    assert table["apache-2.0"]["count"] == 1
    assert table["mit"]["count"] == 1
    assert table["mit"]["attributions"] == []


def test_ccby_without_attribution_is_refused_by_the_build():
    """AC1: a cc-by-4.0 record lacking attribution never reaches a side."""
    bad = _ccby(1)
    object.__setattr__(bad, "attribution", None)  # bypass Record.validate to test the stage
    ds = _build([bad, _td(2)])
    seen = {r.id for r in ds.all_records()} | set(ds.held_out.ids) | set(ds.test.ids)
    assert bad.id not in seen
    assert ds.manifest["refused"]["missing_attribution"] == 1
    assert ds.manifest["licence_table"].get("cc-by-4.0") is None


def test_other_licence_is_refused_and_counted():
    ds = _build([replace(_td(1), licence="gpl-3.0"), _td(2)])
    assert ds.manifest["refused"]["licence"] == 1


def test_max_slots_default_is_24_and_a_knob():
    """AC2: 24 options pass, 25 are refused as over_slots; the knob is recorded."""
    assert dataset.DEFAULT_MAX_SLOTS == 24
    ok, over = _many_options(1, 24), _many_options(2, 25)
    ds = _build([ok, over])
    seen = {r.id for r in ds.all_records()} | set(ds.held_out.ids) | set(ds.test.ids)
    assert ok.id in seen
    assert over.id not in seen
    assert ds.manifest["refused"]["over_slots"] == 1
    assert ds.manifest["max_slots"] == 24
    ds = _build([over], max_slots=25)
    assert ds.manifest["refused"]["over_slots"] == 0
    assert ds.manifest["knobs"]["max_slots"] == 25


def test_over_length_counted_by_reason_and_never_truncated():
    """AC2: over_length is counted in the manifest; kept records are whole."""
    long_state = " ".join(["w"] * 5000)
    ds = _build(_corpus() + [_td(700, state=long_state)])
    assert ds.manifest["refused"]["over_length"] == 1
    assert ds.manifest["refused"]["over_slots"] == 0
    assert ds.manifest["max_tokens"] == 4096
    for r in ds.all_records():
        assert _words(dataset.render_prompt(r)) <= 4096
