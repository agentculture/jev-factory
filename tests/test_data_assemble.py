"""Assemble + freeze, ported from nvsh's tests/test_lfm_finetune_dataset.py (Track B half).

The toy lamp domain stands in for nvsh's; a fake tokenizer stands in for a
model's chat template, so nothing here needs a GPU, network or model download.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.cli._errors import CliError
from jev_factory.data import assemble as asm
from tests.fixtures.toy_domain import DOMAIN, SEED_CORPUS

SEED = json.loads(SEED_CORPUS.read_text(encoding="utf-8"))
ENTRIES = SEED["entries"]


def write(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def side(name: str, entries: list[dict]) -> dict:
    return {"header": f"Split '{name}' of toy (seed=1).", "entries": entries}


def other(eid: str, text: str) -> dict:
    return {"id": eid, "kind": "explicit", "text": text, "expect": {"escalate": True}}


@pytest.fixture
def train(tmp_path) -> Path:
    doc = side("train", ENTRIES)
    doc["world"] = SEED["world"]
    return write(tmp_path / "train.json", doc)


@pytest.fixture
def val(tmp_path) -> Path:
    return write(tmp_path / "val.json", side("val", [other("v1", "zebra quantum harmonica")]))


def run(tmp_path, train, **kwargs):
    return asm.assemble(tmp_path / "run", domain=DOMAIN, split=train, **kwargs)


def rows_of(tmp_path) -> list[dict]:
    return json.loads((tmp_path / "run" / asm.SCORER_TRAIN_NAME).read_text())["entries"]


class FakeTokenizer:
    """A chat template that keeps the system and user text, or one that drops the system."""

    chat_template = ""

    def __init__(self, keep_system: bool = True) -> None:
        self.keep_system = keep_system

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        kept = [m for m in messages if self.keep_system or m["role"] != "system"]
        return "\n".join(f"<{m['role']}>{m['content']}" for m in kept) + "\n<assistant>"


# -- acceptance 1: rows with random order, letters, subsets and -nocand -----


def test_rows_carry_a_seeded_permutation_gold_and_seed(tmp_path, train):
    run(tmp_path, train, cfg=asm.AssembleConfig(missing_candidate_rate=0))
    rows = rows_of(tmp_path)
    assert [r["id"] for r in rows] == [e["id"] for e in ENTRIES]
    for row in rows:
        perm = sc.Permutation.from_json(row["permutation"])
        assert row["gold"] in perm.order
        assert set(perm.labels.values()) <= set(ro.LABEL_ALPHABET)
        assert len(set(perm.labels.values())) == len(perm.order)
        assert isinstance(row["perm_seed"], int)


def test_randomised_orders_letters_and_subsets_vary_across_rows(tmp_path, train):
    run(tmp_path, train, cfg=asm.AssembleConfig(missing_candidate_rate=0))
    perms = [sc.Permutation.from_json(r["permutation"]) for r in rows_of(tmp_path)]
    assert len({p.order for p in perms}) > 4
    assert len({tuple(sorted(p.labels.values())) for p in perms}) > 4
    sizes = {len(p.order) for p in perms}
    assert len(sizes) > 1 and max(sizes) == len(DOMAIN.candidates())
    assert min(sizes) >= asm.DEFAULT_MIN_SUBSET


def test_same_seed_replays_and_another_seed_differs(tmp_path, train):
    cfg = asm.AssembleConfig(perm_seed=7)
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    asm.assemble(a, domain=DOMAIN, split=train, cfg=cfg)
    asm.assemble(b, domain=DOMAIN, split=train, cfg=cfg)
    asm.assemble(c, domain=DOMAIN, split=train, cfg=asm.AssembleConfig(perm_seed=8))
    read = lambda d: (d / asm.SCORER_TRAIN_NAME).read_bytes()  # noqa: E731
    assert read(a) == read(b)
    assert read(a) != read(c)


def test_defaults_follow_issue_4(tmp_path, train):
    cfg = asm.AssembleConfig()
    assert cfg.randomize_labels is True
    assert cfg.missing_candidate_rate == 0.3
    assert cfg.perm_seed == 0


def test_not_randomised_keeps_the_fixed_order_and_positional_letters(tmp_path, train):
    cfg = asm.AssembleConfig(randomize_labels=False, missing_candidate_rate=0)
    run(tmp_path, train, cfg=cfg)
    pool = DOMAIN.candidates()
    for row in rows_of(tmp_path):
        assert tuple(row["permutation"]["order"]) == pool
        assert row["permutation"]["labels"] == sc.labels_for(DOMAIN, pool)


def test_nocand_rows_drop_the_operation_and_force_escalate(tmp_path, train):
    cfg = asm.AssembleConfig(missing_candidate_rate=1.0)
    run(tmp_path, train, cfg=cfg)
    rows = {r["id"]: r for r in rows_of(tmp_path)}
    operations = [e for e in ENTRIES if "operation" in e["expect"]]
    derived = [r for r in rows.values() if r["id"].endswith("-nocand")]
    assert len(derived) == len(operations) == 10
    for entry in operations:
        row = rows[entry["id"] + "-nocand"]
        assert entry["expect"]["operation"] not in row["permutation"]["order"]
        assert row["gold"] == "escalate" and row["expect"] == {"escalate": True}
        assert row["text"] == entry["text"] and row["source_id"] == entry["id"]
        assert row["gold"] in row["permutation"]["order"]


def test_nocand_rate_is_a_fraction_of_operation_rows(tmp_path, train):
    run(tmp_path, train, cfg=asm.AssembleConfig(missing_candidate_rate=0.3))
    derived = [r for r in rows_of(tmp_path) if r["id"].endswith("-nocand")]
    assert 0 < len(derived) < 10
    assert all(asm.select_missing_candidate(0, r["id"][: -len("-nocand")], 0.3) for r in derived)


def test_reasons_mode_golds_descriptions_and_nocand_class(tmp_path, train):
    cfg = asm.AssembleConfig(reasons=True, missing_candidate_rate=1.0, randomize_labels=False)
    run(tmp_path, train, cfg=cfg)
    rows = {r["id"]: r for r in rows_of(tmp_path)}
    assert rows["toy-14"]["gold"] == "escalate:missing_argument"
    assert rows["toy-13"]["descriptions"]["escalate:outside_table"]
    assert "descriptions" not in rows["toy-01"] or all(
        k.startswith("escalate:") for k in rows["toy-01"]["descriptions"]
    )
    nocand = rows["toy-01-nocand"]
    assert nocand["gold"] == DOMAIN.default_reason_label()
    assert nocand["class"] == "decline:" + DOMAIN.default_reason_label().split(":", 1)[1]


def test_merges_variations_and_supplement_and_drops_duplicates(tmp_path, train, val):
    variations = tmp_path / "accepted.jsonl"
    lines = [
        {
            "id": "toy-01-v1",
            "kind": "explicit",
            "text": "which lamps are lit right now",
            "expect": ENTRIES[0]["expect"],
            "source_id": "toy-01",
            "side": "train",
        },
        {  # repeats an existing train text: dropped as a duplicate
            "id": "toy-01-v2",
            "kind": "explicit",
            "text": "WHICH lights are on",
            "expect": ENTRIES[0]["expect"],
            "source_id": "toy-01",
            "side": "train",
        },
        {  # repeats a val text: dropped as leaked
            "id": "toy-01-v3",
            "kind": "explicit",
            "text": "zebra quantum harmonica",
            "expect": ENTRIES[0]["expect"],
            "source_id": "toy-01",
            "side": "train",
        },
    ]
    variations.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    supplement = write(
        tmp_path / "supp.json",
        {
            "header": "Split 'train' of toy supplement.",
            "entries": [other("sup-1", "open the pod bay door")],
        },
    )
    freeze = run(
        tmp_path,
        train,
        variations=[variations],
        supplement=supplement,
        exclude=[val],
        cfg=asm.AssembleConfig(missing_candidate_rate=0),
    )
    ids = [r["id"] for r in rows_of(tmp_path)]
    assert "toy-01-v1" in ids and "sup-1" in ids
    assert "toy-01-v2" not in ids and "toy-01-v3" not in ids
    assert freeze["counts"]["kept"] == 1
    assert freeze["counts"]["duplicate"] == 1 and freeze["counts"]["leaked"] == 1
    assert freeze["counts"]["supplement"] == 1


def test_a_supplement_repeating_an_excluded_side_is_refused(tmp_path, train, val):
    supplement = write(
        tmp_path / "supp.json",
        {
            "header": "Split 'train' of toy supplement.",
            "entries": [other("s", "Zebra quantum harmonica!")],
        },
    )
    with pytest.raises(CliError, match="merge refused"):
        run(tmp_path, train, supplement=supplement, exclude=[val])


def test_leakage_fails_closed_exact_and_near_duplicate(tmp_path, train):
    test_side = write(
        tmp_path / "test.json", side("test", [other("t1", ENTRIES[4]["text"].upper())])
    )
    with pytest.raises(CliError, match="leak"):
        run(tmp_path, train, protected=[test_side])
    assert not (tmp_path / "run" / asm.FREEZE_NAME).exists()


def test_unreadable_protected_side_fails_closed(tmp_path, train):
    bad = tmp_path / "test.json"
    bad.write_text("not json")
    with pytest.raises(CliError) as err:
        run(tmp_path, train, protected=[bad])
    assert err.value.code == 2


def test_clean_protected_sides_pass(tmp_path, train, val):
    run(tmp_path, train, protected=[val])


def test_held_out_and_non_train_sides_are_refused(tmp_path):
    for name, header in (
        ("held-out.json", "Split 'train' of toy."),
        ("renamed.json", "Held-out split of toy."),
        ("val.json", "Split 'val' of toy."),
        ("nohead.json", "no side named here"),
    ):
        path = write(tmp_path / name, {"header": header, "entries": ENTRIES})
        with pytest.raises(CliError):
            asm.assemble(tmp_path / "out", domain=DOMAIN, split=path)


def test_eval_side_markers_by_file_name():
    assert asm.eval_side_markers(Path("test.json")) == {"test"}
    assert asm.eval_side_markers(Path("held-out.json")) == {"held-out"}
    assert asm.eval_side_markers(Path("dev.json")) == set()


def test_check_train_source_refuses_an_eval_named_plain_corpus():
    cfg = asm.AssembleConfig(missing_candidate_rate=0.3)
    with pytest.raises(CliError, match="test side"):
        asm.check_train_source(Path("test.json"), {"header": ""}, is_split=False, cfg=cfg)


def test_bad_rate_is_refused(tmp_path, train):
    with pytest.raises(CliError):
        run(tmp_path, train, cfg=asm.AssembleConfig(missing_candidate_rate=1.5))


# -- rendering with an injectable tokenizer ---------------------------------


def test_render_check_passes_with_a_faithful_template(tmp_path, train):
    run(tmp_path, train, tokenizer=FakeTokenizer())


def test_render_check_fails_when_the_template_drops_the_offer(tmp_path, train):
    with pytest.raises(CliError, match="chat template"):
        run(tmp_path, train, tokenizer=FakeTokenizer(keep_system=False))


def test_render_row_uses_the_stored_permutation(tmp_path, train):
    run(tmp_path, train, cfg=asm.AssembleConfig(missing_candidate_rate=0))
    row = rows_of(tmp_path)[0]
    text = asm.render_row(DOMAIN, row, FakeTokenizer())
    for name, letter in row["permutation"]["labels"].items():
        assert f"{letter}) {name}:" in text
    assert row["text"] in text


# -- acceptance 2: freeze by sha256, training selects by it ------------------


@pytest.mark.behavioral("o28")
def test_freeze_records_the_sha256_of_every_data_file(tmp_path, train, val):
    freeze = run(tmp_path, train, exclude=[val], protected=[val])
    files = freeze["files"]
    assert {"split", "scorer_train", "exclude_0", "protected_0"} <= set(files)
    for record in files.values():
        assert record["sha256"] == asm.sha256_path(Path(record["path"]))
    on_disk = json.loads((tmp_path / "run" / asm.FREEZE_NAME).read_text())
    assert on_disk == freeze
    assert asm.verify_freeze(tmp_path / "run" / asm.FREEZE_NAME) == []


@pytest.mark.behavioral("o28")
def test_select_frozen_picks_by_sha256_never_by_mtime(tmp_path, train):
    run(tmp_path, train)
    freeze_path = tmp_path / "run" / asm.FREEZE_NAME
    frozen = tmp_path / "run" / asm.SCORER_TRAIN_NAME
    # A newer decoy with other content beside it does not win.
    decoy = tmp_path / "run" / "scorer-train-new.json"
    decoy.write_text(json.dumps({"header": "x", "entries": []}))
    now = time.time()
    os.utime(frozen, (now - 1000, now - 1000))
    os.utime(decoy, (now, now))
    assert asm.select_frozen(freeze_path).path == frozen
    # Moved elsewhere (and renamed), the frozen bytes are still found by hash.
    elsewhere = tmp_path / "store"
    elsewhere.mkdir()
    moved = elsewhere / "whatever.json"
    moved.write_bytes(frozen.read_bytes())
    frozen.unlink()
    os.utime(decoy, (now + 500, now + 500))
    choice = asm.select_frozen(freeze_path, search=[elsewhere])
    assert choice.path == moved and choice.deviation_id is None


@pytest.mark.behavioral("o28")
def test_a_change_after_freeze_needs_a_deviation_id(tmp_path, train):
    run(tmp_path, train)
    freeze_path = tmp_path / "run" / asm.FREEZE_NAME
    frozen = tmp_path / "run" / asm.SCORER_TRAIN_NAME
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(CliError, match="deviation"):
        asm.select_frozen(freeze_path)
    with pytest.raises(CliError, match="deviation"):
        asm.verify_freeze(freeze_path)
    choice = asm.select_frozen(freeze_path, deviation_id="d7")
    assert choice.path == frozen and choice.deviation_id == "d7"
    assert asm.verify_freeze(freeze_path, "d7") == ["scorer_train"]


def test_a_changed_input_after_freeze_is_caught(tmp_path, train):
    run(tmp_path, train)
    train.write_text(train.read_text() + "\n")
    with pytest.raises(CliError, match="split"):
        asm.verify_freeze(tmp_path / "run" / asm.FREEZE_NAME)


def test_reassembling_over_a_freeze_needs_a_deviation_id(tmp_path, train):
    first = run(tmp_path, train)
    with pytest.raises(CliError, match="frozen"):
        run(tmp_path, train)
    second = run(tmp_path, train, cfg=asm.AssembleConfig(perm_seed=3), deviation_id="d8")
    assert second["deviation_id"] == "d8"
    assert second["supersedes"]["scorer_train"] == first["files"]["scorer_train"]["sha256"]


def test_missing_data_file_cannot_be_frozen(tmp_path, train):
    with pytest.raises(CliError):
        run(tmp_path, train, exclude=[tmp_path / "nope.json"])


def test_select_frozen_unknown_role(tmp_path, train):
    run(tmp_path, train)
    with pytest.raises(CliError, match="no 'nope'"):
        asm.select_frozen(tmp_path / "run" / asm.FREEZE_NAME, role="nope")


def test_load_tokenizer_is_lazy_and_local_only(monkeypatch):
    import sys
    import types

    calls = {}

    class Auto:
        @staticmethod
        def from_pretrained(base, revision=None, local_files_only=False):
            calls.update(base=base, revision=revision, local=local_files_only)
            return "tok"

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = Auto
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert asm.load_tokenizer("some/base", "rev") == "tok"
    assert calls == {"base": "some/base", "revision": "rev", "local": True}
