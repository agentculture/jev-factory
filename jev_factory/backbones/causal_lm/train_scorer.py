"""Train the causal-LM candidate scorer: a LoRA on letter-restricted cross-entropy.

The trainer reads the frozen scorer training set (``scorer-train.json``, the
train side :mod:`jev_factory.data.assemble` writes and freezes) and, for
validation, the ``val`` side ``split`` writes. Each entry becomes one prompt
from :mod:`.scorer` (:func:`~.scorer.prompt_messages` over the
:class:`~jev_factory.domain.model.Domain`; the request is the entry's own
``text``) and one target: the position of its gold candidate.

The loss reads the logits of the label tokens only, at the one position
after the prompt: the model is asked for ``logits_to_keep`` of just each
row's last real position, and a cross-entropy over the offered labels'
columns is taken there. The full vocabulary is never computed over the
whole sequence.

**Per-row letter maps.** An entry may carry its own ``"permutation"``
(:meth:`~.scorer.Permutation.to_json`: the offered candidates in listing
order and their letters), a ``"gold"`` candidate name (an operation,
``explain``, ``escalate`` or ``escalate:<reason>``), a ``"descriptions"``
override and its ``"perm_seed"``. Such a row's prompt is rendered from its
own map, its label columns are the letters in its own order and its target
is the gold's position there; a row without one keeps the fixed full
candidate map. Rows in a batch may offer different numbers of labels: the
label logits are padded per row, and a padded column is ``-inf`` so it gets
no probability and no loss. The cross-entropy is therefore restricted to
the letters each row actually offered.

**The label readout.** ``--label-readout variants`` (the default) is the
one definition in :mod:`.readout`: a label's logit is the log-sum-exp of
every token whose stripped text is the letter
(:func:`~.readout.label_logits_from_vocab`), so training optimises the
distribution the scorer reads. ``single`` reads the one id of the bare letter.

**Calibration terms.** ``--label-smoothing`` and ``--brier-weight`` are both
0, off, by default; with both off the loss is plain cross-entropy.

**Gates before any heavy import** (:func:`prepare`): the run's
pre-registration must be registered (:func:`jev_factory.factory.prereg.require_registered`),
the training file's sha256 must be the frozen one when ``--expect-sha256`` is
given, and the held-out split, test and any non-train side are refused.

**Heal** (``--heal``): 1 epoch at lr 5e-5 in bf16 from the run's own merged
checkpoint (``--base <run>/merged``), on the same frozen set; any other
epochs/lr is refused (:func:`resolve_hyperparameters`).

Runs are seeded and the batch order is drawn from the seed. ``train-log.json``
records each file's sha256 next to the hyperparameters, precision, heal
flag, readout and its token ids, the permutation seeds seen and how many
rows were permuted; every permuted row's map is in ``row-maps.json`` beside
it (name, sha256 and row count in the log). torch, transformers and peft are
imported lazily, inside the functions that need them.

    python -m jev_factory.backbones.causal_lm.train_scorer --domain <module-or-json> \\
        --train scorer-train.json --val val.json --out runs/scorer-r1 \\
        --base <base> --revision <commit> --prereg prereg.json --lock-dir <work>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer
from jev_factory.backbones.causal_lm.train import (
    HEAL_EPOCHS,
    HEAL_LR,
    HEAL_PRECISION,
    TRAIN_DEFAULTS,
    cap_gpu_memory,
    text_tokenizer,
)
from jev_factory.cli._errors import CliError
from jev_factory.core.split import HELD_OUT_NAME
from jev_factory.domain.model import ESCALATE, EXPLAIN, Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/train_scorer.py",
    "commit": "9debdc6",
    "adaptations": [
        "nvsh imports :73-74 (nvsh.tiers lfm, bench load_corpus/request_for/context_for)"
        " rewired to the Domain and the causal-LM scorer adapter: the candidate pool, labels,"
        " reasons and reason_for_class come from the Domain; the request is the entry's text",
        "_sibling path-loading :77-90 replaced by package imports (scorer, readout, train);"
        " label_variant_ids/label_token_ids/label_logits_from_vocab come from readout.py,"
        " the one letter definition; permutation letters must be in its LABEL_ALPHABET",
        "read_split/_example/encode/letter_ids/attach_columns/permutation_record/"
        "write_row_maps/label loss/train_loop/evaluate :107-569 ported with the Domain as an"
        " explicit argument; entries are checked for id/text/expect in place of load_corpus",
        "main :623-736 split into prepare (prereg gate, frozen sha256 check, split reading;"
        " no heavy import) and _train; the tokenizer goes through train.text_tokenizer"
        " (processor.tokenizer for Qwen3.5); train_log records precision, heal, deviation id"
        " and the domain surface sha256; row-maps.json is named relative to the run dir",
        "new --heal: 1 epoch, lr 5e-5, bf16 continuation (resolve_hyperparameters refuses"
        " other values); new --prereg/--lock-dir/--expect-sha256/--deviation/--domain",
        "tests ported from tests/test_lfm_finetune_train_scorer.py and"
        " test_lfm_finetune_train_scorer_rows.py onto the toy domain",
    ],
    "licence": "Apache-2.0",
}

DEFAULT_REVISION = None

TRAIN_SIDE = "train"
VAL_SIDE = "val"

TRAIN_LOG_NAME = "train-log.json"
ROW_MAPS_NAME = "row-maps.json"

#: ``split``'s note in a side's header: ``Split '<side>' of <corpus> (seed=N).``
_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")

#: The phrase the held-out split's header opens with.
_HELD_OUT_MARKER = "held-out split"


@dataclass(frozen=True)
class Example:
    """One entry of a split: the request as the model reads it, and its gold candidate.

    ``permutation`` is the row's own offered candidates and letters (``None``:
    the fixed full map), ``descriptions`` its prompt-description overrides and
    ``perm_seed`` the seed its permutation was drawn with, all as stored.
    """

    entry_id: str
    request: str
    gold: str
    permutation: scorer.Permutation | None = None
    descriptions: dict[str, str] | None = None
    perm_seed: int | None = None
    cls: str | None = None


def gold_candidate(domain: Domain, expect: dict) -> str:
    """The candidate an entry's ``expect`` block names (split's three kinds)."""
    if expect.get("escalate"):
        return ESCALATE
    if expect.get("explain"):
        return EXPLAIN
    if "operation" in expect:
        name = expect["operation"]
        if domain.get(name) is None:
            raise ValueError(f"{name!r} is not a candidate of domain {domain.name!r}")
        return name
    raise ValueError(f"an expect block the factory does not recognise: {expect!r}")


def _entries(raw: Any, path: Path) -> list[dict]:
    items = raw.get("entries") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: no entries")
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(f"{path}: entry {index} has no id")
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"{path}: entry {item['id']!r} has no request text")
        if not isinstance(item.get("expect"), dict):
            raise ValueError(f"{path}: entry {item['id']!r} has no expect block")
    return items


def read_split(domain: Domain, path: Path, side: str) -> list[Example]:
    """Every entry of a split file that must be *side*; refuses the held-out split.

    A file whose header names no side, or another side, is refused: the
    trainer reads the train side, validation reads the val side, and the
    test and held-out sides are never read here.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = raw.get("header") if isinstance(raw, dict) else None
    header = header if isinstance(header, str) else ""
    if path.name == HELD_OUT_NAME or header.casefold().startswith(_HELD_OUT_MARKER):
        raise ValueError(f"{path}: the held-out split is never read by the trainer")
    match = _SPLIT_SIDE_RE.search(header)
    found = match.group(1) if match else None
    if found != side:
        raise ValueError(f"{path}: its header names side {found!r}, expected {side!r}")
    return [_example(domain, item) for item in _entries(raw, path)]


def _example(domain: Domain, item: dict) -> Example:
    """One :class:`Example` from a raw entry and its per-row fields."""
    entry_id = str(item["id"])
    expected = gold_candidate(domain, item["expect"])
    gold = item.get("gold", expected)
    if not isinstance(gold, str) or gold.split(":", 1)[0] != expected:
        raise ValueError(f"entry {entry_id!r}: gold {gold!r} disagrees with its expect block")
    permutation = None
    if item.get("permutation") is not None:
        permutation = scorer.Permutation.from_json(item["permutation"])
        letters = [permutation.labels.get(name) for name in permutation.order]
        if (
            None in letters
            or len(set(letters)) != len(letters)
            or any(letter not in ro.LABEL_ALPHABET for letter in letters)
        ):
            raise ValueError(
                f"entry {entry_id!r}: its permutation must give each a distinct letter"
                " from the readout alphabet"
            )
        if gold not in permutation.order:
            raise ValueError(f"entry {entry_id!r}: gold {gold!r} is not offered by its permutation")
    descriptions = item.get("descriptions")
    return Example(
        entry_id=entry_id,
        request=item["text"],
        gold=gold,
        permutation=permutation,
        descriptions=dict(descriptions) if descriptions is not None else None,
        perm_seed=item.get("perm_seed"),
        cls=item.get("class") if isinstance(item.get("class"), str) else None,
    )


def reasons_mode(domain: Domain, examples: list[Example]) -> bool:
    """True when any row offers an ``escalate:<reason>`` candidate (reasons data)."""
    reasons = set(domain.escalate_labels())
    return any(
        example.permutation is not None
        and any(name in reasons for name in example.permutation.order)
        for example in examples
    )


def match_validation(domain: Domain, train: list[Example], val: list[Example]) -> list[Example]:
    """*val*, scored on the same candidate pool the train rows use.

    In reasons mode a validation row with no stored map gets the reasons
    pool's default map, its reason descriptions, and an escalate gold named by
    its class (:meth:`Domain.reason_for_class`), rather than the fixed map
    with a bare ``escalate`` the model never trained on. Otherwise *val* is
    returned unchanged.
    """
    if not reasons_mode(domain, train):
        return val
    pool = scorer.candidate_pool(domain, reasons=True)
    permutation = scorer.Permutation(order=tuple(pool), labels=scorer.positional_labels(pool, pool))
    matched = []
    for example in val:
        if example.permutation is not None:
            matched.append(example)
            continue
        gold = example.gold
        if gold == ESCALATE:
            gold = domain.reason_for_class(example.cls)
        matched.append(
            dataclasses.replace(
                example,
                gold=gold,
                permutation=permutation,
                descriptions={**domain.reason_descriptions(), **(example.descriptions or {})},
            )
        )
    return matched


def example_messages(domain: Domain, example: Example) -> list[dict]:
    """The prompt messages *example* is trained on: its own map, else the fixed one.

    In-process and served scoring render the same messages for the same
    candidate list and order (:func:`.scorer.prompt_messages`).
    """
    permutation = example.permutation
    return scorer.prompt_messages(
        domain,
        example.request,
        labels=permutation.labels if permutation is not None else None,
        order=permutation.order if permutation is not None else None,
        descriptions=example.descriptions,
    )


def encode(domain: Domain, tokenizer, examples: list[Example], max_length: int) -> list[dict]:
    """``{"input_ids", "target", "letters"}`` per example, from that example's own map.

    ``letters`` are the row's label letters in its listing order and
    ``target`` the gold's position in that order. An example with no
    permutation renders the prompt over every candidate, fixed map.
    """
    fixed_names = list(scorer.candidate_pool(domain))
    fixed_labels = scorer.labels_for(domain, fixed_names)
    rows: list[dict] = []
    for example in examples:
        permutation = example.permutation
        order = list(permutation.order) if permutation is not None else fixed_names
        labels = permutation.labels if permutation is not None else fixed_labels
        prompt = scorer.render_prompt(tokenizer, example_messages(domain, example))
        input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(input_ids) > max_length:
            raise ValueError(
                f"entry {example.entry_id!r} is {len(input_ids)} tokens, over {max_length};"
                " raise --max-length rather than cutting the request"
            )
        if example.gold not in order:
            raise ValueError(f"entry {example.entry_id!r}: gold {example.gold!r} is not offered")
        rows.append(
            {
                "input_ids": input_ids,
                "target": order.index(example.gold),
                "letters": [labels[name] for name in order],
            }
        )
    return rows


READOUTS = ("variants", "single")


def letter_ids(tokenizer, letters, readout: str) -> dict[str, tuple[int, ...]]:
    """Letter -> the token ids its label logit reads, under *readout*.

    ``variants``: every token whose stripped text is the letter
    (:func:`.readout.label_variant_ids`, the one readout definition).
    ``single``: the one id of the bare letter (:func:`.readout.label_token_ids`).
    """
    wanted = {letter: letter for letter in sorted(set(letters))}
    if readout == "variants":
        return dict(ro.label_variant_ids(tokenizer, wanted))
    if readout == "single":
        return {letter: (i,) for letter, i in ro.label_token_ids(tokenizer, wanted).items()}
    raise ValueError(f"unknown label readout {readout!r}; expected one of {READOUTS}")


def attach_columns(rows: list[dict], ids: dict[str, tuple[int, ...]]) -> list[dict]:
    """Copies of *rows*, each with ``columns``: its letters' token ids, in its order."""
    return [{**row, "columns": [tuple(ids[letter]) for letter in row["letters"]]} for row in rows]


def permutation_record(sides: dict[str, list[Example]]) -> tuple[dict, list[dict]]:
    """``(summary, maps)``: the permutation seeds and counts, and every permuted row's map."""
    seeds: set = set()
    permuted: dict[str, int] = {}
    fixed: dict[str, int] = {}
    maps: list[dict] = []
    for side, examples in sides.items():
        permuted[side] = fixed[side] = 0
        for example in examples:
            if example.permutation is None:
                fixed[side] += 1
                continue
            permuted[side] += 1
            if example.perm_seed is not None:
                seeds.add(example.perm_seed)
            order = list(example.permutation.order)
            maps.append(
                {
                    "side": side,
                    "entry_id": example.entry_id,
                    "perm_seed": example.perm_seed,
                    "order": order,
                    "labels": {name: example.permutation.labels[name] for name in order},
                    "gold": example.gold,
                    "target": order.index(example.gold),
                    "descriptions": sorted(example.descriptions or {}),
                }
            )
    summary = {
        "perm_seeds": sorted(seeds, key=lambda seed: (str(type(seed)), seed)),
        "permuted_rows": permuted,
        "fixed_rows": fixed,
    }
    return summary, maps


def write_row_maps(path: Path, maps: list[dict]) -> dict:
    """Write *maps* to *path*; ``{"file", "sha256", "rows"}`` for the train-log (file by name)."""
    path = Path(path)
    path.write_text(json.dumps(maps, indent=1) + "\n", encoding="utf-8")
    return {"file": path.name, "sha256": file_sha256(path), "rows": len(maps)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- the loss and the loop (torch) --


#: AdamW's weight decay: torch's default, named so a run's recipe states it.
ADAMW_WEIGHT_DECAY = 0.01


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def pad_right(sequences: list[list[int]], pad_id: int):
    """``(input_ids, attention_mask)`` tensors, padded on the right."""
    import torch

    width = max(len(sequence) for sequence in sequences)
    ids = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    mask = [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
    return torch.tensor(ids), torch.tensor(mask)


def label_logits(model, input_ids, attention_mask, label_ids):
    """``(batch, labels)`` logits of the label tokens at each row's last real position.

    Only the distinct last positions in the batch are kept (a tensor
    ``logits_to_keep``). Padding is on the right: every layer is causal, so
    pad tokens after a row's last real token cannot change what the model
    computes there.
    """
    rows = last_logits(model, input_ids, attention_mask)
    return rows.index_select(-1, label_ids.to(rows.device)).float()


def column_logits(vocab_rows, columns: list[list[tuple[int, ...]]]):
    """``(logits, mask)``: each row's label logits from its ``(vocab,)`` logits, padded.

    ``columns[i]`` is row *i*'s label columns, each a tuple of token ids. When
    every column is one id (the ``single`` readout) the logits are those ids'
    entries; otherwise each is the log-sum-exp of its ids
    (:func:`.readout.label_logits_from_vocab`). Columns past a row's own count
    are ``-inf`` and ``False`` in *mask*.
    """
    import torch

    device = vocab_rows.device
    width = max(len(row) for row in columns)
    mask = torch.tensor(
        [[j < len(row) for j in range(width)] for row in columns], dtype=torch.bool, device=device
    )
    if all(len(ids) == 1 for row in columns for ids in row):
        index = torch.tensor(
            [[ids[0] for ids in row] + [0] * (width - len(row)) for row in columns], device=device
        )
        logits = vocab_rows.gather(-1, index).float()
    else:
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(columns):
            groups.setdefault(tuple(tuple(ids) for ids in row), []).append(i)
        pieces: list = [None] * len(columns)
        for key, members in groups.items():
            part = ro.label_logits_from_vocab(vocab_rows[members].float(), key)
            part = torch.nn.functional.pad(part, (0, width - len(key)), value=-math.inf)
            for j, member in enumerate(members):
                pieces[member] = part[j]
        logits = torch.stack(pieces)
    return logits.masked_fill(~mask, -math.inf), mask


def label_loss(logits, mask, targets, *, label_smoothing: float = 0.0, brier_weight: float = 0.0):
    """Mean label-restricted loss: cross-entropy, plus the optional calibration terms."""
    import torch

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label smoothing must be in [0, 1), not {label_smoothing}")
    if brier_weight < 0.0:
        raise ValueError(f"brier weight must be at least 0, not {brier_weight}")
    if math.isclose(label_smoothing, 0.0, abs_tol=1e-12) and math.isclose(
        brier_weight, 0.0, abs_tol=1e-12
    ):
        return torch.nn.functional.cross_entropy(logits, targets)
    log_p = torch.log_softmax(logits, dim=-1)
    offered_log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
    per_row = -offered_log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if label_smoothing:
        uniform = -offered_log_p.sum(dim=-1) / mask.sum(dim=-1)
        per_row = (1.0 - label_smoothing) * per_row + label_smoothing * uniform
    if brier_weight:
        gold = torch.nn.functional.one_hot(targets, logits.shape[-1]).to(log_p.dtype)
        brier = ((log_p.exp() - gold) ** 2 * mask).sum(dim=-1)
        per_row = per_row + brier_weight * brier
    return per_row.mean()


def last_logits(model, input_ids, attention_mask):
    """``(batch, vocab)`` logits at each row's last real position (see :func:`label_logits`)."""
    import torch

    last = attention_mask.sum(dim=1) - 1
    keep, where = torch.unique(last, return_inverse=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep).logits
    return logits[torch.arange(input_ids.shape[0], device=logits.device), where]


def _batches(rows: list[dict], batch: int, order: list[int]):
    for start in range(0, len(order), batch):
        yield [rows[index] for index in order[start : start + batch]]


def _step_inputs(model, chunk: list[dict], pad_id: int):
    import torch

    device = next(model.parameters()).device
    input_ids, mask = pad_right([row["input_ids"] for row in chunk], pad_id)
    targets = torch.tensor([row["target"] for row in chunk], device=device)
    return input_ids.to(device), mask.to(device), targets


def _default_columns(label_ids) -> list[tuple[int, ...]] | None:
    """*label_ids* (one id, or a tuple of ids, per column) as columns; ``None`` stays ``None``."""
    if label_ids is None:
        return None
    return [(ids,) if isinstance(ids, int) else tuple(ids) for ids in label_ids]


def _chunk_logits(model, chunk: list[dict], default, pad_id: int):
    """``(logits, mask, targets)`` for one batch: each row's own columns, else *default*."""
    input_ids, attention_mask, targets = _step_inputs(model, chunk, pad_id)
    columns = [row.get("columns", default) for row in chunk]
    if any(row is None for row in columns):
        raise ValueError("a row has no label columns and no default label_ids was given")
    logits, mask = column_logits(last_logits(model, input_ids, attention_mask), columns)
    return logits, mask, targets


def train_loop(
    model,
    rows: list[dict],
    label_ids=None,
    *,
    epochs: int,
    lr: float,
    batch: int,
    seed: int,
    pad_id: int,
    label_smoothing: float = 0.0,
    brier_weight: float = 0.0,
) -> list[dict]:
    """AdamW over the label-restricted loss; returns ``[{epoch, step, loss}]``.

    Each row reads its own ``columns`` (:func:`attach_columns`); *label_ids*
    (one id or a tuple of ids per column) serves rows that carry none.
    """
    import torch

    default = _default_columns(label_ids)
    order_rng = random.Random(seed)  # nosec B311 - batch order, not security
    trainable = [p for p in model.parameters() if p.requires_grad]
    # weight_decay is torch's AdamW default, stated so the recipe records it.
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=ADAMW_WEIGHT_DECAY)
    model.train()
    history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        for chunk in _batches(rows, batch, order):
            logits, mask, targets = _chunk_logits(model, chunk, default, pad_id)
            loss = label_loss(
                logits,
                mask,
                targets,
                label_smoothing=label_smoothing,
                brier_weight=brier_weight,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1
            history.append({"epoch": epoch + 1, "step": step, "loss": loss.item()})
    return history


def evaluate(model, rows: list[dict], label_ids=None, *, batch: int, pad_id: int) -> dict:
    """Mean cross-entropy, accuracy and mean top probability over each row's offered labels."""
    import torch

    default = _default_columns(label_ids)
    model.eval()
    total_loss = 0.0
    right = 0
    confidence = 0.0
    with torch.no_grad():
        for chunk in _batches(rows, batch, list(range(len(rows)))):
            logits, _mask, targets = _chunk_logits(model, chunk, default, pad_id)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
            probabilities = torch.softmax(logits, dim=-1)
            top, picked = probabilities.max(dim=-1)
            right += int((picked == targets).sum())
            confidence += float(top.sum())
    model.train()
    n = len(rows)
    return {
        "n": n,
        "loss": total_loss / n,
        "accuracy": right / n,
        "mean_confidence": confidence / n,
    }


# -- the command line --


def _smoothing(text: str) -> float:
    value = float(text)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"label smoothing must be in [0, 1), not {value}")
    return value


def _non_negative(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"must be at least 0, not {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", required=True, help="domain module (dotted) or JSON file")
    parser.add_argument("--train", required=True, type=Path, help="the frozen scorer-train.json")
    parser.add_argument("--val", type=Path, help="split's val side (loss and accuracy)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default=None, help="base model id, or a merged dir to heal")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prereg", required=True, type=Path, help="the pre-registration file")
    parser.add_argument(
        "--lock-dir", required=True, type=Path, help="where prereg.lock.json was registered"
    )
    parser.add_argument("--expect-sha256", help="the frozen sha256 --train must have")
    parser.add_argument("--deviation", help="deviation id for changed prereg or frozen data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=TRAIN_DEFAULTS["rank"])
    parser.add_argument("--alpha", type=int, default=TRAIN_DEFAULTS["alpha"])
    parser.add_argument("--batch", type=int, default=TRAIN_DEFAULTS["batch"])
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-length", type=int, default=TRAIN_DEFAULTS["max_length"])
    parser.add_argument(
        "--heal",
        action="store_true",
        help=f"heal: {HEAL_EPOCHS} epoch at lr {HEAL_LR} ({HEAL_PRECISION}) from --base's merge",
    )
    parser.add_argument(
        "--label-readout",
        choices=READOUTS,
        default="variants",
        help="variants: log-sum-exp of every token that is the letter (readout.py's definition);"
        " single: the bare letter's one id",
    )
    calibration = parser.add_argument_group("calibration loss (both off by default)")
    calibration.add_argument(
        "--label-smoothing",
        type=_smoothing,
        default=0.0,
        help="label smoothing over each row's offered labels, in [0, 1)",
    )
    calibration.add_argument(
        "--brier-weight",
        type=_non_negative,
        default=0.0,
        help="weight of a multi-class Brier term over the offered labels, added to CE",
    )
    return parser


def resolve_hyperparameters(args: argparse.Namespace) -> argparse.Namespace:
    """Fill epochs/lr: the heal recipe under ``--heal`` (anything else refused), else defaults."""
    if args.heal:
        if args.epochs not in (None, HEAL_EPOCHS):
            raise ValueError(f"heal is {HEAL_EPOCHS} epoch, not {args.epochs}")
        if args.lr is not None and args.lr != HEAL_LR:
            raise ValueError(f"heal is lr {HEAL_LR}, not {args.lr}")
        args.epochs, args.lr = HEAL_EPOCHS, HEAL_LR
        return args
    args.epochs = TRAIN_DEFAULTS["epochs"] if args.epochs is None else args.epochs
    args.lr = TRAIN_DEFAULTS["lr"] if args.lr is None else args.lr
    return args


def prepare(args: argparse.Namespace) -> tuple[Domain, list[Example], list[Example]]:
    """The gates, then the examples; no heavy import. Raises CliError or ValueError."""
    from jev_factory.domain.validate import load_domain
    from jev_factory.factory.prereg import require_registered

    require_registered(args.prereg, args.lock_dir, args.deviation)
    if args.expect_sha256 and file_sha256(args.train) != args.expect_sha256:
        raise ValueError(
            f"{args.train}: its sha256 is not the frozen {args.expect_sha256[:12]}...;"
            " training selects its data by the frozen sha256"
        )
    domain = load_domain(args.domain)
    train = read_split(domain, args.train, TRAIN_SIDE)
    val = read_split(domain, args.val, VAL_SIDE) if args.val else []
    return domain, train, match_validation(domain, train, val)


@dataclass(frozen=True)
class RunStats:
    """How big and how long one training run was, for :func:`train_log`."""

    train_rows: int
    val_rows: int
    seconds: float
    max_gpu_memory_gb: float | None


def train_log(
    args: argparse.Namespace,
    *,
    domain: Domain,
    labels: dict,
    label_token_ids: dict,
    variants: dict,
    readout_ids: dict,
    perm_summary: dict,
    row_maps: dict,
    stats: RunStats,
    val: dict | None,
    history: list[dict],
) -> dict[str, Any]:
    """The ``train-log.json`` record: everything needed to replay the run."""
    objective = "cross-entropy over each row's offered label tokens at the decision position"
    if args.label_smoothing or args.brier_weight:
        objective += ", with calibration terms"
    return {
        "base": args.base,
        "revision": args.revision,
        "objective": objective,
        "precision": HEAL_PRECISION,
        "heal": (
            {"of": args.base, "epochs": HEAL_EPOCHS, "lr": HEAL_LR, "precision": HEAL_PRECISION}
            if args.heal
            else None
        ),
        "deviation_id": args.deviation,
        "domain": domain.name,
        "domain_surface_sha256": domain.surface_sha256(),
        "candidates": list(labels),
        "labels": dict(labels),
        "label_token_ids": dict(label_token_ids),
        "label_variant_ids": {name: list(found) for name, found in variants.items()},
        "label_readout": args.label_readout,
        "letter_ids": {letter: list(found) for letter, found in readout_ids.items()},
        "calibration_loss": {
            "label_smoothing": args.label_smoothing,
            "brier_weight": args.brier_weight,
        },
        "permutations": {**perm_summary, "row_maps": row_maps},
        "train_file": str(args.train),
        "train_sha256": file_sha256(args.train),
        "val_file": str(args.val) if args.val else None,
        "val_sha256": file_sha256(args.val) if args.val else None,
        "train_examples": stats.train_rows,
        "val_examples": stats.val_rows,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "rank": args.rank,
            "alpha": args.alpha,
            "batch": args.batch,
            "seed": args.seed,
            "max_length": args.max_length,
        },
        "seconds": round(stats.seconds, 1),
        "max_gpu_memory_gb": stats.max_gpu_memory_gb,
        "val": val,
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resolve_hyperparameters(args)
        if not args.base:
            raise ValueError("--base is required (a model id, or a merged dir under --heal)")
        domain, train_examples, val_examples = prepare(args)
    except CliError as exc:
        print(f"train_scorer: {exc.message} ({exc.remediation})", file=sys.stderr)
        return exc.code
    except (ValueError, OSError) as exc:
        print(f"train_scorer: {exc}", file=sys.stderr)
        return 1
    return _train(args, domain, train_examples, val_examples)


def _train(args, domain: Domain, train_examples, val_examples) -> int:  # needs a GPU stack
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        cap_gpu_memory(torch)
    except ValueError as exc:
        print(f"train_scorer: {exc}", file=sys.stderr)
        return 2
    seed_everything(args.seed)
    tokenizer = text_tokenizer(AutoTokenizer.from_pretrained(args.base, revision=args.revision))
    reasons = reasons_mode(domain, train_examples)
    labels = scorer.labels_for(domain, scorer.candidate_pool(domain, reasons), reasons)
    ids = ro.label_token_ids(tokenizer, labels)
    variants = ro.label_variant_ids(tokenizer, labels)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_rows = encode(domain, tokenizer, train_examples, args.max_length)
    val_rows = encode(domain, tokenizer, val_examples, args.max_length)
    readout_ids = letter_ids(
        tokenizer,
        [letter for row in train_rows + val_rows for letter in row["letters"]],
        args.label_readout,
    )
    train_rows = attach_columns(train_rows, readout_ids)
    val_rows = attach_columns(val_rows, readout_ids)
    perm_summary, maps = permutation_record({TRAIN_SIDE: train_examples, VAL_SIDE: val_examples})

    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, dtype=torch.bfloat16
    )
    if model.dtype != torch.bfloat16:
        print(f"train_scorer: the model loaded as {model.dtype}, not bf16", file=sys.stderr)
        return 2
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=args.alpha, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    started = time.time()
    history = train_loop(
        model,
        train_rows,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        seed=args.seed,
        pad_id=pad_id,
        label_smoothing=args.label_smoothing,
        brier_weight=args.brier_weight,
    )
    seconds = time.time() - started
    val = evaluate(model, val_rows, batch=args.batch, pad_id=pad_id) if val_rows else None

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    log = train_log(
        args,
        domain=domain,
        labels=labels,
        label_token_ids=ids,
        variants=variants,
        readout_ids=readout_ids,
        perm_summary=perm_summary,
        row_maps=write_row_maps(args.out / ROW_MAPS_NAME, maps),
        stats=RunStats(
            train_rows=len(train_rows),
            val_rows=len(val_rows),
            seconds=seconds,
            max_gpu_memory_gb=(
                round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None
            ),
        ),
        val=val,
        history=history,
    )
    (args.out / TRAIN_LOG_NAME).write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
