"""The label readout module: alphabet, variants, distribution, completeness, tally."""

from __future__ import annotations

import math

import pytest

from jev_factory.backbones.causal_lm import readout as ro

LABELS = {"read": "A", "write": "B", "explain": "C"}


class FakeTokenizer:
    """A tiny vocabulary: each label in bare, spaced and tabbed variants, plus filler."""

    def __init__(self, labels) -> None:
        self.texts: list[str] = ["<pad>", "the", " the"]
        for label in labels:
            self.texts += [label, " " + label, "\t" + label]
        self.texts += ["AB"]

    def get_vocab(self):
        return {text: index for index, text in enumerate(self.texts)}

    def decode(self, ids):
        return "".join(self.texts[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.texts.index(text)] if text in self.texts else [0, 1]

    def __len__(self):
        return len(self.texts)


def _logits(size: int) -> list[float]:
    return [((index * 7) % 11) / 3.0 - 1.5 for index in range(size)]


def test_alphabet_is_upper_then_lower_and_52_long():
    assert ro.LABEL_ALPHABET[:3] == "ABC"
    assert ro.LABEL_ALPHABET[25:28] == "Zab"
    assert len(ro.LABEL_ALPHABET) == len(set(ro.LABEL_ALPHABET)) == 52


def test_readout_top_is_at_least_20000():
    assert ro.READOUT_TOP >= 20000


def test_provenance_names_upstream():
    assert ro.NVSH_PROVENANCE["commit"] == "9debdc6"


def test_label_token_ids_refuses_multi_token_and_shared_ids():
    tok = FakeTokenizer(LABELS.values())
    with pytest.raises(ValueError, match="single token"):
        ro.label_token_ids(tok, {"x": "nope"})

    class Same:
        def encode(self, text, add_special_tokens=False):
            return [1]

    same = Same()
    with pytest.raises(ValueError, match="share"):
        ro.label_token_ids(same, {"a": "A", "b": "B"})


def test_variant_ids_are_bare_spaced_and_tabbed():
    tok = FakeTokenizer(LABELS.values())
    variants = ro.label_variant_ids(tok, LABELS)
    for name, label in LABELS.items():
        assert sorted(tok.texts[i] for i in variants[name]) == sorted(
            [label, " " + label, "\t" + label]
        )


def test_variant_ids_refuse_a_label_with_no_token():
    tok = FakeTokenizer(["A"])
    with pytest.raises(ValueError, match="no token"):
        ro.label_variant_ids(tok, {"a": "A", "z": "Z"})


def test_distribution_sums_variants_and_normalises():
    lp = {"A": math.log(0.1), " A": math.log(0.1), "\tA": math.log(0.1), "B": math.log(0.1)}
    dist, mass = ro.distribution(lp, {"a": "A", "b": "B"})
    assert dist["a"] == pytest.approx(0.75)
    assert mass == pytest.approx(0.4)


def test_distribution_skips_junk_and_empty_mass():
    assert ro.distribution({"A": float("nan"), "B": 0.5, 3: -1.0}, LABELS) == ({}, 0.0)


@pytest.mark.behavioral("o14")
def test_missing_label_is_incomplete_and_never_renormalised():
    served = {"A": math.log(0.01), " B": math.log(0.5)}  # C absent from the top-k
    result = ro.readout(served, LABELS)
    assert not result.complete
    assert result.distribution == {}
    assert result.missing == ("explain",)
    assert ro.missing_labels(served, LABELS) == ("explain",)
    assert result.mass == pytest.approx(0.51)


@pytest.mark.behavioral("o14")
def test_complete_readout_is_normalised():
    served = {"A": math.log(0.2), " B": math.log(0.2), "\tC": math.log(0.6)}
    result = ro.readout(served, LABELS)
    assert result.complete and result.missing == ()
    assert sum(result.distribution.values()) == pytest.approx(1.0)
    assert result.choice == "explain"


@pytest.mark.behavioral("o14")
def test_zero_probability_label_counts_as_present():
    served = {"A": -1.0, "B": -1.0, "C": -math.inf}
    assert ro.readout(served, LABELS).complete


@pytest.mark.behavioral("o14")
def test_tally_reports_complete_and_incomplete_counts():
    tally = ro.ReadoutTally()
    full = {"A": -1.0, "B": -1.5, "C": -2.0}
    tally.record(ro.readout(full, LABELS))
    tally.record(ro.readout(full, LABELS))
    assert tally.summary() == "2 complete / 0 incomplete"
    tally.record(ro.readout({"A": -1.0}, LABELS))
    tally.record(ro.readout({}, LABELS))
    assert tally.summary() == "2 complete / 2 incomplete"
    assert tally.as_dict() == {"complete": 2, "incomplete": 2, "total": 4}


@pytest.mark.behavioral("o27")
def test_in_process_and_served_paths_share_one_distribution():
    tok = FakeTokenizer(LABELS.values())
    variants = ro.label_variant_ids(tok, LABELS)
    logits = _logits(len(tok))

    in_process = ro.readout(ro.variant_logprobs(ro.log_softmax(logits), variants, tok), LABELS)

    # served: the same next-token distribution as a top-k map keyed by token text
    served_map: dict[str, float] = {}
    for index, value in enumerate(ro.log_softmax(logits)):
        text = tok.texts[index]
        served_map[text] = (
            value
            if text not in served_map
            else math.log(math.exp(served_map[text]) + math.exp(value))
        )
    served = ro.readout(served_map, LABELS)

    assert in_process.complete and served.complete
    for name in LABELS:
        assert in_process.distribution[name] == pytest.approx(served.distribution[name], abs=1e-6)
    assert in_process.mass == pytest.approx(served.mass, abs=1e-6)


def test_variant_logprobs_adds_ids_that_decode_to_one_text():
    class Dup:
        def decode(self, ids):
            return "A"

    out = ro.variant_logprobs([math.log(0.1), math.log(0.2)], {"a": (0, 1)}, Dup())
    assert math.exp(out["A"]) == pytest.approx(0.3)


def test_training_readout_matches_distribution():
    torch = pytest.importorskip("torch")
    tok = FakeTokenizer(LABELS.values())
    variants = ro.label_variant_ids(tok, LABELS)
    logits = torch.tensor([_logits(len(tok))] * 2)
    label_logits = ro.label_logits_from_vocab(logits, [variants[n] for n in LABELS])
    trained = torch.softmax(label_logits, dim=-1)[0].tolist()
    dist, _ = ro.distribution(
        ro.variant_logprobs(ro.log_softmax(logits[0].tolist()), variants, tok), LABELS
    )
    for position, name in enumerate(LABELS):
        assert trained[position] == pytest.approx(dist[name], abs=1e-6)
