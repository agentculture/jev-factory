"""jev_factory.backbones.causal_lm.train_scorer: the LoRA scorer trainer (t22).

Ported from nvsh's tests/test_lfm_finetune_train_scorer.py and
test_lfm_finetune_train_scorer_rows.py onto the toy lamp domain. The pure
helpers (reading splits, per-row letter maps, targets, columns, row maps,
the prereg/sha256 gates of ``main``) run everywhere; the loss and the loop
need torch and are skipped where it is not installed (the training env runs
them).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.backbones.causal_lm import train_scorer as ts
from jev_factory.data import assemble as asm
from jev_factory.factory import prereg
from tests.fixtures.toy_domain import DOMAIN, SEED_CORPUS

_WORLD = json.loads(SEED_CORPUS.read_text(encoding="utf-8"))["world"]


def _split(tmp_path: Path, side: str, entries: list[dict], name: str | None = None) -> Path:
    path = tmp_path / (name or f"{side}.json")
    payload = {
        "header": f"Toy corpus. Split '{side}' of seed.json (seed=46).",
        "entries": entries,
        "world": _WORLD,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(entry_id: str, text: str, expect: dict, **extra) -> dict:
    return {
        "id": entry_id,
        "kind": "explicit",
        "text": text,
        "expect": expect,
        "source_id": "s",
        **extra,
    }


_ENTRIES = [
    _entry("e1", "Which lights are on?", {"operation": "lamp_status", "args": {}}),
    _entry("e2", "Rewire the fuse box", {"escalate": True}),
    _entry("e3", "What is a scene?", {"explain": True, "answer": "A saved lighting set."}),
]


class _Tokenizer:
    """One token per character of the rendered prompt."""

    chat_template = "{{ messages }}"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 250 + 1 for char in text]


def _read(tmp_path, side, entries, **kwargs):
    return ts.read_split(DOMAIN, _split(tmp_path, side, entries, **kwargs), side)


def _perm(seed, **kwargs) -> dict:
    return sc.permute(seed, pool=sc.candidate_pool(DOMAIN), **kwargs).to_json()


# -- reading the split files --


def test_gold_candidate_maps_each_expectation_kind() -> None:
    assert ts.gold_candidate(DOMAIN, {"operation": "list_rooms", "args": {}}) == "list_rooms"
    assert ts.gold_candidate(DOMAIN, {"escalate": True}) == "escalate"
    assert ts.gold_candidate(DOMAIN, {"explain": True, "answer": "x"}) == "explain"
    with pytest.raises(ValueError, match="expect"):
        ts.gold_candidate(DOMAIN, {"surprise": 1})


def test_a_gold_operation_outside_the_domain_is_refused() -> None:
    with pytest.raises(ValueError, match="not a candidate"):
        ts.gold_candidate(DOMAIN, {"operation": "no_such_operation"})


def test_read_split_returns_request_text_and_gold_for_each_entry(tmp_path) -> None:
    examples = _read(tmp_path, "train", _ENTRIES)
    assert [example.gold for example in examples] == ["lamp_status", "escalate", "explain"]
    assert examples[0].request == "Which lights are on?"
    assert [example.entry_id for example in examples] == ["e1", "e2", "e3"]


def test_training_refuses_any_side_but_train(tmp_path) -> None:
    for side in ("val", "test"):
        path = _split(tmp_path, side, _ENTRIES)
        with pytest.raises(ValueError, match="side"):
            ts.read_split(DOMAIN, path, ts.TRAIN_SIDE)


def test_validation_reads_only_the_val_side(tmp_path) -> None:
    assert _read(tmp_path, "val", _ENTRIES)
    path = _split(tmp_path, "test", _ENTRIES)
    with pytest.raises(ValueError, match="side"):
        ts.read_split(DOMAIN, path, ts.VAL_SIDE)


def test_the_held_out_split_is_refused_by_name_and_by_header(tmp_path) -> None:
    named = _split(tmp_path, "train", _ENTRIES, name="held-out.json")
    with pytest.raises(ValueError, match="held-out"):
        ts.read_split(DOMAIN, named, ts.TRAIN_SIDE)
    path = tmp_path / "renamed.json"
    path.write_text(json.dumps({"header": "Held-out split: sealed.", "entries": _ENTRIES}))
    with pytest.raises(ValueError, match="held-out"):
        ts.read_split(DOMAIN, path, ts.TRAIN_SIDE)


def test_a_split_with_no_entries_or_a_malformed_entry_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="no entries"):
        _read(tmp_path, "train", [])
    with pytest.raises(ValueError, match="text"):
        _read(tmp_path, "train", [{"id": "x", "expect": {"escalate": True}}])


def test_the_assembled_scorer_train_file_is_read_as_the_train_side(tmp_path) -> None:
    seed = json.loads(SEED_CORPUS.read_text(encoding="utf-8"))
    split = _split(tmp_path, "train", seed["entries"])
    asm.assemble(tmp_path / "data", domain=DOMAIN, split=split)
    path = tmp_path / "data" / asm.SCORER_TRAIN_NAME
    examples = ts.read_split(DOMAIN, path, ts.TRAIN_SIDE)
    rows = json.loads(path.read_text())["entries"]
    assert [example.entry_id for example in examples] == [row["id"] for row in rows]
    assert all(example.permutation is not None for example in examples)
    assert any(example.entry_id.endswith("-nocand") for example in examples)


# -- per-row maps from the split file --


def test_read_split_keeps_each_rows_stored_permutation_seed_and_descriptions(tmp_path) -> None:
    perm = _perm(7, subset=5, keep="lamp_status")
    entries = [
        _entry(
            "p1",
            "Which lights are on?",
            {"operation": "lamp_status", "args": {}},
            permutation=perm,
            perm_seed=7,
            descriptions={"lamp_status": "Read the lamps."},
        ),
        _entry("f1", "What is a scene?", {"explain": True, "answer": "A set."}),
    ]
    permuted, fixed = _read(tmp_path, "train", entries)
    assert permuted.permutation == sc.Permutation.from_json(perm)
    assert permuted.perm_seed == 7
    assert permuted.descriptions == {"lamp_status": "Read the lamps."}
    assert permuted.gold == "lamp_status"
    assert fixed.permutation is None
    assert fixed.perm_seed is None
    assert fixed.descriptions is None


def test_a_stored_gold_must_agree_with_the_expect_block(tmp_path) -> None:
    reason = _entry("r1", "Rewire it", {"escalate": True}, gold="escalate:multi_step")
    [example] = _read(tmp_path, "train", [reason])
    assert example.gold == "escalate:multi_step"
    wrong = _entry("w1", "Lights?", {"operation": "lamp_status", "args": {}}, gold="list_rooms")
    with pytest.raises(ValueError, match="gold"):
        _read(tmp_path, "train", [wrong])


def test_a_gold_the_rows_permutation_does_not_offer_is_refused(tmp_path) -> None:
    perm = {"order": ["list_rooms", "explain"], "labels": {"list_rooms": "k", "explain": "B"}}
    entry = _entry("x", "Lights?", {"operation": "lamp_status", "args": {}}, permutation=perm)
    with pytest.raises(ValueError, match="not offered"):
        _read(tmp_path, "train", [entry])


def test_a_permutation_that_reuses_a_letter_is_refused(tmp_path) -> None:
    perm = {"order": ["lamp_status", "explain"], "labels": {"lamp_status": "Q", "explain": "Q"}}
    entry = _entry("x", "Lights?", {"operation": "lamp_status", "args": {}}, permutation=perm)
    with pytest.raises(ValueError, match="letter"):
        _read(tmp_path, "train", [entry])


def test_a_permutation_letter_outside_the_readout_alphabet_is_refused(tmp_path) -> None:
    perm = {"order": ["lamp_status", "explain"], "labels": {"lamp_status": "1", "explain": "B"}}
    entry = _entry("x", "Lights?", {"operation": "lamp_status", "args": {}}, permutation=perm)
    with pytest.raises(ValueError, match="letter"):
        _read(tmp_path, "train", [entry])


# -- encoding: per-row letters and targets --


def test_encode_puts_the_gold_index_on_the_rendered_prompt(tmp_path) -> None:
    examples = _read(tmp_path, "train", _ENTRIES)
    rows = ts.encode(DOMAIN, _Tokenizer(), examples, max_length=100_000)
    candidates = list(sc.candidate_pool(DOMAIN))
    assert [row["target"] for row in rows] == [
        candidates.index("lamp_status"),
        candidates.index("escalate"),
        candidates.index("explain"),
    ]
    assert all(row["input_ids"] for row in rows)


def test_encode_refuses_a_prompt_over_max_length_rather_than_cutting_it(tmp_path) -> None:
    examples = _read(tmp_path, "train", _ENTRIES)
    tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="over 5"):
        ts.encode(DOMAIN, tokenizer, examples, max_length=5)


def test_two_rows_with_different_maps_get_different_targets_for_the_same_gold(tmp_path) -> None:
    first, second = _perm(1), _perm(2)
    assert first["order"].index("lamp_status") != second["order"].index("lamp_status")
    entries = [
        _entry("a", "Lights?", {"operation": "lamp_status", "args": {}}, permutation=first),
        _entry("b", "Lights?", {"operation": "lamp_status", "args": {}}, permutation=second),
    ]
    rows = ts.encode(DOMAIN, _Tokenizer(), _read(tmp_path, "train", entries), max_length=100_000)
    for row, perm in zip(rows, (first, second)):
        assert row["target"] == perm["order"].index("lamp_status")
        assert row["letters"] == [perm["labels"][name] for name in perm["order"]]
        assert row["letters"][row["target"]] == perm["labels"]["lamp_status"]
    assert rows[0]["target"] != rows[1]["target"]
    assert rows[0]["input_ids"] != rows[1]["input_ids"]  # the prompts list them differently


def test_a_permuted_row_renders_the_prompt_from_its_own_map(tmp_path) -> None:
    perm = _perm(5, subset=3, keep="escalate")
    descriptions = {"escalate": "Hand it up."}
    entry = _entry(
        "a", "Rewire it", {"escalate": True}, permutation=perm, descriptions=descriptions
    )
    tokenizer = _Tokenizer()
    [row] = ts.encode(DOMAIN, tokenizer, _read(tmp_path, "train", [entry]), max_length=100_000)
    expected = sc.render_prompt(
        tokenizer,
        sc.prompt_messages(
            DOMAIN,
            "Rewire it",
            labels=perm["labels"],
            order=perm["order"],
            descriptions=descriptions,
        ),
    )
    assert row["input_ids"] == tokenizer.encode(expected)
    assert len(row["letters"]) == 3


def test_a_row_without_a_permutation_keeps_the_fixed_map_and_prompt(tmp_path) -> None:
    entry = _entry("a", "Lights?", {"operation": "lamp_status", "args": {}})
    tokenizer = _Tokenizer()
    [row] = ts.encode(DOMAIN, tokenizer, _read(tmp_path, "train", [entry]), max_length=100_000)
    names = list(sc.candidate_pool(DOMAIN))
    fixed = sc.labels_for(DOMAIN, names)
    assert row["target"] == names.index("lamp_status")
    assert row["letters"] == [fixed[name] for name in names]
    prompt = sc.render_prompt(tokenizer, sc.prompt_messages(DOMAIN, "Lights?"))
    assert row["input_ids"] == tokenizer.encode(prompt)


def test_attach_columns_maps_each_rows_letters_to_their_token_ids() -> None:
    rows = [
        {"input_ids": [1], "target": 0, "letters": ["B", "A"]},
        {"input_ids": [2], "target": 1, "letters": ["A", "C", "B"]},
    ]
    ids = {"A": (10, 11), "B": (20,), "C": (30, 31, 32)}
    attached = ts.attach_columns(rows, ids)
    assert attached[0]["columns"] == [(20,), (10, 11)]
    assert attached[1]["columns"] == [(10, 11), (30, 31, 32), (20,)]
    assert "columns" not in rows[0]  # the input rows are left alone


def test_the_loss_columns_are_exactly_each_rows_offered_letters(tmp_path) -> None:
    """CE is restricted to the offered letters: a row's columns are its own letters, no others."""
    perm = _perm(11, subset=4, keep="lamp_on")
    entry = _entry("a", "Kitchen lights on", {"operation": "lamp_on", "args": {}}, permutation=perm)
    [row] = ts.encode(DOMAIN, _Tokenizer(), _read(tmp_path, "train", [entry]), max_length=100_000)
    ids = {letter: (100 + i,) for i, letter in enumerate(ro.LABEL_ALPHABET)}
    [attached] = ts.attach_columns([row], ids)
    offered = [perm["labels"][name] for name in perm["order"]]
    assert attached["columns"] == [ids[letter] for letter in offered]
    assert len(attached["columns"]) == 4
    assert attached["columns"][attached["target"]] == ids[perm["labels"]["lamp_on"]]


def test_letter_ids_single_and_variants_readouts() -> None:
    class _Vocab(_Tokenizer):
        _vocab = {"A": 5, " A": 6, "\tA": 7, "B": 8, " B": 9, "x": 10}

        def get_vocab(self):
            return dict(self._vocab)

        def decode(self, ids):
            inverse = {v: k for k, v in self._vocab.items()}
            return "".join(inverse[i] for i in ids)

        def encode(self, text, add_special_tokens=False):
            return [self._vocab[text]]

    tokenizer = _Vocab()
    assert ts.letter_ids(tokenizer, ["A", "B"], "variants") == {"A": (5, 6, 7), "B": (8, 9)}
    assert ts.letter_ids(tokenizer, ["B", "A"], "single") == {"A": (5,), "B": (8,)}
    with pytest.raises(ValueError, match="readout"):
        ts.letter_ids(tokenizer, ["A"], "other")


def test_permutation_record_logs_seeds_counts_and_per_row_maps(tmp_path) -> None:
    perm = _perm(9)
    entries = [
        _entry(
            "a",
            "Lights?",
            {"operation": "lamp_status", "args": {}},
            permutation=perm,
            perm_seed=9,
            descriptions={"lamp_status": "Lamps."},
        ),
        _entry("b", "Rewire it", {"escalate": True}),
    ]
    train = _read(tmp_path, "train", entries)
    val = _read(tmp_path, "val", entries[:1])
    summary, maps = ts.permutation_record({"train": train, "val": val})
    assert summary["perm_seeds"] == [9]
    assert summary["permuted_rows"] == {"train": 1, "val": 1}
    assert summary["fixed_rows"] == {"train": 1, "val": 0}
    assert maps == [
        {
            "side": side,
            "entry_id": "a",
            "perm_seed": 9,
            "order": perm["order"],
            "labels": perm["labels"],
            "gold": "lamp_status",
            "target": perm["order"].index("lamp_status"),
            "descriptions": ["lamp_status"],
        }
        for side in ("train", "val")
    ]
    path = tmp_path / ts.ROW_MAPS_NAME
    written = ts.write_row_maps(path, maps)
    assert written["file"] == ts.ROW_MAPS_NAME
    assert written["rows"] == 2
    assert written["sha256"] == ts.file_sha256(path)
    assert json.loads(path.read_text()) == maps


def test_file_sha256_names_the_exact_split_bytes(tmp_path) -> None:
    path = _split(tmp_path, "train", _ENTRIES)
    first = ts.file_sha256(path)
    assert len(first) == 64
    path.write_text(path.read_text() + " ")
    assert ts.file_sha256(path) != first


# -- the train log --


def test_the_train_log_carries_what_replays_the_run(tmp_path) -> None:
    train_path = _split(tmp_path, "train", _ENTRIES)
    args = ts._parser().parse_args(
        ["--domain", "x", "--train", str(train_path), "--out", str(tmp_path / "run")]
        + ["--prereg", "p.json", "--lock-dir", "w"]
    )
    ts.resolve_hyperparameters(args)
    log = ts.train_log(
        args,
        domain=DOMAIN,
        labels={"lamp_status": "A"},
        label_token_ids={"lamp_status": 5},
        variants={"lamp_status": (5, 6)},
        readout_ids={"A": (5, 6)},
        perm_summary={"perm_seeds": [], "permuted_rows": {}, "fixed_rows": {}},
        row_maps={"file": ts.ROW_MAPS_NAME, "sha256": "0" * 64, "rows": 0},
        stats=ts.RunStats(train_rows=3, val_rows=0, seconds=1.23, max_gpu_memory_gb=None),
        val=None,
        history=[{"epoch": 1, "step": 1, "loss": 0.5}],
    )
    assert log["train_sha256"] == ts.file_sha256(train_path)
    assert log["precision"] == "bf16"
    assert log["heal"] is None
    assert log["domain"] == DOMAIN.name
    assert log["domain_surface_sha256"] == DOMAIN.surface_sha256()
    assert log["permutations"]["row_maps"]["file"] == ts.ROW_MAPS_NAME
    assert log["hyperparameters"]["epochs"] == 3
    assert json.loads(json.dumps(log)) == log


# -- the parser and the heal recipe --


def _args(*extra: str):
    base = ["--domain", "x", "--train", "t.json", "--out", "o", "--prereg", "p", "--lock-dir", "w"]
    return ts._parser().parse_args(base + list(extra))


def test_defaults_are_the_issue_46_recipe_and_readout_variants() -> None:
    args = ts.resolve_hyperparameters(_args())
    assert (args.epochs, args.lr, args.rank, args.alpha, args.batch) == (3, 2e-4, 16, 32, 4)
    assert args.label_readout == "variants"
    assert args.label_smoothing == 0.0
    assert args.brier_weight == 0.0
    assert args.heal is False


def test_heal_forces_one_epoch_at_lr_5e_5() -> None:
    args = ts.resolve_hyperparameters(_args("--heal"))
    assert (args.epochs, args.lr) == (1, 5e-5)
    assert ts.resolve_hyperparameters(_args("--heal", "--epochs", "1", "--lr", "5e-5")).lr == 5e-5
    three_epochs = _args("--heal", "--epochs", "3")
    with pytest.raises(ValueError, match="1 epoch"):
        ts.resolve_hyperparameters(three_epochs)
    recipe_lr = _args("--heal", "--lr", "2e-4")
    with pytest.raises(ValueError, match="5e-05"):
        ts.resolve_hyperparameters(recipe_lr)


@pytest.mark.parametrize(
    "flags", [["--label-smoothing", "1.0"], ["--label-smoothing", "-0.1"], ["--brier-weight", "-1"]]
)
def test_the_parser_refuses_calibration_terms_out_of_range(flags: list[str]) -> None:
    with pytest.raises(SystemExit):
        _args(*flags)
    ok = _args("--label-smoothing", "0.1", "--brier-weight", "0.5")
    assert (ok.label_smoothing, ok.brier_weight) == (0.1, 0.5)
    with pytest.raises(SystemExit):
        _args("--label-readout", "x")


# -- main's gates, before any heavy import --


def _prereg_doc() -> dict:
    return prereg.apply_defaults(_prereg_draft())


def _prereg_draft() -> dict:
    return {
        "schema_version": 1,
        "stock_baseline_record_id": "8b59d0afa150",
        "perms_per_entry": 8,
        "candidates": ["r1"],
        "bars": {
            "wrong_mutating": {"stock": 0.02, "minimum": 0.0},
            "ece": {"stock": 0.25, "minimum": 0.03},
            "permutation_change": {"stock": 0.3, "minimum": 0.05},
            "mc_escalation": {"stock": 0.1, "minimum": 0.8},
            "right_proposals": {"stock": 0.072},
        },
    }


def _main_argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--domain",
        "tests.fixtures.toy_domain",
        "--train",
        str(_split(tmp_path, "train", _ENTRIES)),
        "--out",
        str(tmp_path / "run"),
        "--base",
        "example-org/base-model",
        "--prereg",
        str(tmp_path / "prereg.json"),
        "--lock-dir",
        str(tmp_path / "work"),
        *extra,
    ]


def test_main_refuses_without_a_registered_preregistration(tmp_path, capsys) -> None:
    (tmp_path / "prereg.json").write_text(json.dumps(_prereg_doc()))
    assert ts.main(_main_argv(tmp_path)) == 1
    assert "pre-registration" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_main_refuses_data_whose_sha256_is_not_the_frozen_one(tmp_path, capsys) -> None:
    (tmp_path / "prereg.json").write_text(json.dumps(_prereg_doc()))
    prereg.register(tmp_path / "prereg.json", tmp_path / "work")
    assert ts.main(_main_argv(tmp_path, "--expect-sha256", "0" * 64)) == 1
    assert "sha256" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_main_passes_its_gates_and_prepares_the_examples(tmp_path) -> None:
    (tmp_path / "prereg.json").write_text(json.dumps(_prereg_doc()))
    prereg.register(tmp_path / "prereg.json", tmp_path / "work")
    argv = _main_argv(tmp_path)
    train_sha = ts.file_sha256(Path(argv[argv.index("--train") + 1]))
    args = ts.resolve_hyperparameters(
        ts._parser().parse_args(argv + ["--expect-sha256", train_sha])
    )
    domain, train, val = ts.prepare(args)
    assert domain.name == DOMAIN.name
    assert [example.entry_id for example in train] == ["e1", "e2", "e3"]
    assert val == []


# -- reasons-mode validation (PR #65 review) --


def test_reasons_mode_training_validates_on_the_reasons_pool(tmp_path) -> None:
    pool = sc.candidate_pool(DOMAIN, reasons=True)
    train_perm = sc.Permutation(order=tuple(pool), labels=sc.positional_labels(pool, pool))
    train = _read(
        tmp_path,
        "train",
        [
            _entry(
                "t1",
                "turn them on",
                {"escalate": True},
                **{
                    "class": "decline:missing_argument",
                    "permutation": train_perm.to_json(),
                    "gold": "escalate:missing_argument",
                },
            )
        ],
    )
    val = _read(
        tmp_path,
        "val",
        [
            _entry(
                "v1", "turn them on", {"escalate": True}, **{"class": "decline:missing_argument"}
            ),
            _entry("v2", "which lights are on", {"operation": "lamp_status", "args": {}}),
        ],
    )
    assert ts.reasons_mode(DOMAIN, train)
    matched = ts.match_validation(DOMAIN, train, val)
    assert matched[0].gold == "escalate:missing_argument"
    assert tuple(matched[0].permutation.order) == tuple(pool)
    assert matched[1].gold == "lamp_status"
    rows = ts.encode(DOMAIN, _Tokenizer(), matched, 100_000)
    assert len(rows[0]["letters"]) == len(pool)


def test_plain_training_leaves_validation_unchanged(tmp_path) -> None:
    op = {"operation": "lamp_status", "args": {}}
    train = _read(tmp_path, "train", [_entry("t1", "lights?", op)])
    val = _read(tmp_path, "val", [_entry("v1", "lights?", op)])
    assert not ts.reasons_mode(DOMAIN, train)
    assert ts.match_validation(DOMAIN, train, val) == val


# -- the loss and the loop (torch, when importable) --


def _toy(torch):
    """A tiny causal-LM stand-in: an embedding and a head, logits_to_keep honoured."""

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(64, 16)
            self.head = torch.nn.Linear(16, 64)
            self.kept: list = []

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            self.kept.append(logits_to_keep)
            hidden = self.embed(input_ids).cumsum(dim=1)
            if isinstance(logits_to_keep, int):
                keep = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
            else:
                keep = logits_to_keep
            return type("Out", (), {"logits": self.head(hidden[:, keep, :])})()

    return _Model


def _fixed_rows(n_labels: int = 4) -> list[dict]:
    return [
        {"input_ids": [1 + (i % 8), 9, 2 + (i % 5)] + [3] * (i % 3), "target": i % 8 % n_labels}
        for i in range(24)
    ]


def test_label_logits_are_restricted_to_the_candidate_tokens_at_each_last_position() -> None:
    torch = pytest.importorskip("torch")
    model_cls = _toy(torch)
    model = model_cls()
    rows = _fixed_rows()
    label_ids = [40, 41, 42, 43]
    input_ids, mask = ts.pad_right([row["input_ids"] for row in rows[:4]], pad_id=0)
    logits = ts.label_logits(model, input_ids, mask, torch.tensor(label_ids))
    assert tuple(logits.shape) == (4, len(label_ids))
    kept = model.kept[-1]
    assert not isinstance(kept, int)
    assert len(kept) <= 4
    for row_index, row in enumerate(rows[:4]):
        alone = torch.tensor([row["input_ids"]])
        full = model_cls.forward(model, alone, None, 0).logits[0, -1, label_ids]
        assert torch.allclose(logits[row_index], full, atol=1e-5)


def test_pad_right_masks_the_padding() -> None:
    pytest.importorskip("torch")
    input_ids, mask = ts.pad_right([[5, 6, 7], [8]], pad_id=0)
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_training_loss_decreases_and_is_reproducible_under_one_seed() -> None:
    torch = pytest.importorskip("torch")
    model_cls = _toy(torch)
    rows, label_ids = _fixed_rows(), [40, 41, 42, 43]

    def run(seed: int, epochs: int = 3) -> list[float]:
        ts.seed_everything(seed)
        history = ts.train_loop(
            model_cls(), rows, label_ids, epochs=epochs, lr=0.05, batch=4, seed=seed, pad_id=0
        )
        return [step["loss"] for step in history]

    long = run(46, epochs=30)
    assert long[-1] < 0.5 * long[0]
    first, second = run(46), run(46)
    assert first == second
    assert run(46) != run(47)


def test_evaluate_reports_accuracy_and_confidence() -> None:
    torch = pytest.importorskip("torch")
    model_cls = _toy(torch)
    rows, label_ids = _fixed_rows(), [40, 41, 42, 43]
    model = model_cls()
    ts.train_loop(model, rows, label_ids, epochs=30, lr=0.05, batch=4, seed=46, pad_id=0)
    report = ts.evaluate(model, rows, label_ids, batch=4, pad_id=0)
    assert report["n"] == len(rows)
    assert report["accuracy"] > 1 / len(label_ids)
    assert 0.0 < report["mean_confidence"] <= 1.0
    assert report["loss"] >= 0.0


class _OldReference:
    """The pre-per-row-map loss path, so ``single`` + fixed map is checked against it."""

    @staticmethod
    def train_loop(torch, model, rows, label_ids, *, epochs, lr, batch, seed, pad_id):
        labels = torch.tensor(label_ids)
        order_rng = random.Random(seed)  # nosec B311
        optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        model.train()
        history = []
        for _ in range(epochs):
            order = list(range(len(rows)))
            order_rng.shuffle(order)
            for start in range(0, len(order), batch):
                chunk = [rows[i] for i in order[start : start + batch]]
                input_ids, mask = ts.pad_right([row["input_ids"] for row in chunk], pad_id)
                targets = torch.tensor([row["target"] for row in chunk])
                last = mask.sum(dim=1) - 1
                keep, where = torch.unique(last, return_inverse=True)
                out = model(input_ids=input_ids, attention_mask=mask, logits_to_keep=keep).logits
                picked = out[torch.arange(input_ids.shape[0]), where]
                logits = picked.index_select(-1, labels).float()
                loss = torch.nn.functional.cross_entropy(logits, targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                history.append(loss.item())
        return history


def test_single_readout_on_the_fixed_map_matches_the_old_code_path_exactly() -> None:
    torch = pytest.importorskip("torch")
    model_cls = _toy(torch)
    label_ids = [40, 41, 42, 43]
    rows = _fixed_rows()
    letters = ["A", "B", "C", "D"]
    single = ts.attach_columns(
        [{**row, "letters": letters} for row in rows],
        {letter: (label_ids[i],) for i, letter in enumerate(letters)},
    )
    torch.manual_seed(3)
    old = _OldReference.train_loop(
        torch, model_cls(), rows, label_ids, epochs=3, lr=0.05, batch=4, seed=46, pad_id=0
    )
    torch.manual_seed(3)
    new = ts.train_loop(model_cls(), single, None, epochs=3, lr=0.05, batch=4, seed=46, pad_id=0)
    assert [step["loss"] for step in new] == old


def test_variant_readout_is_the_log_sum_exp_of_each_labels_variants() -> None:
    torch = pytest.importorskip("torch")
    vocab = torch.randn(2, 64)
    columns = [[(40, 41), (42,)], [(42,), (43, 44, 45), (40, 41)]]
    logits, mask = ts.column_logits(vocab, columns)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert torch.allclose(logits[0, 0], torch.logsumexp(vocab[0, [40, 41]], dim=0))
    assert torch.allclose(logits[1, 1], torch.logsumexp(vocab[1, [43, 44, 45]], dim=0))
    assert logits[0, 2] == float("-inf")
    expected = ro.label_logits_from_vocab(vocab[1:2].float(), columns[1])
    assert torch.allclose(logits[1], expected[0])


def test_a_mixed_batch_gives_each_row_the_loss_it_has_alone() -> None:
    torch = pytest.importorskip("torch")
    vocab = torch.randn(3, 64, requires_grad=True)
    columns = [[(40, 41), (42,)], [(43,), (44, 45), (46,), (47,)], [(48,), (49,), (50,)]]
    targets = torch.tensor([1, 2, 0])
    logits, mask = ts.column_logits(vocab, columns)
    loss = ts.label_loss(logits, mask, targets)
    alone = [
        torch.nn.functional.cross_entropy(
            ts.column_logits(vocab[i : i + 1], [columns[i]])[0], targets[i : i + 1]
        )
        for i in range(3)
    ]
    assert torch.allclose(loss, torch.stack(alone).mean())
    loss.backward()
    assert torch.isfinite(vocab.grad).all()


def test_the_default_loss_is_plain_cross_entropy_and_terms_are_bounded() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    targets = torch.tensor([0, 3, 1, 4])
    plain = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.equal(ts.label_loss(logits, mask, targets), plain)
    smoothed = ts.label_loss(logits, mask, targets, label_smoothing=0.1)
    assert torch.allclose(
        smoothed, torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
    )
    with pytest.raises(ValueError, match="smoothing"):
        ts.label_loss(logits, mask, targets, label_smoothing=1.0)
    with pytest.raises(ValueError, match="brier"):
        ts.label_loss(logits, mask, targets, brier_weight=-0.1)
