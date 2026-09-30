"""The causal-LM scorer adapter: prompt, permutation, scoring, grounding, the predictions record.

Ported from nvsh's tests/test_lfm_finetune_scorer.py onto the toy lamp
domain. A fake top-k function stands in for the model everywhere; the
in-process path runs on plain-number logits (no torch), and the served path
on an injected ``post`` (no server).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from jev_factory.backbones.causal_lm import readout as ro
from jev_factory.backbones.causal_lm import scorer as sc
from jev_factory.domain.model import ArgSpec, Operation
from tests.fixtures.toy_domain import DOMAIN

WORLD = {"home": "toy-home", "rooms": ["kitchen", "bedroom", "study", "hallway"]}
POOL = DOMAIN.candidates()


class FakeTopK:
    """Answers with fixed log-probabilities; records each prompt and top asked."""

    def __init__(self, logprobs: dict[str, float]) -> None:
        self.logprobs = logprobs
        self.prompts: list[str] = []
        self.tops: list[int] = []

    def __call__(self, prompt: str, top: int) -> dict[str, float]:
        self.prompts.append(prompt)
        self.tops.append(top)
        return dict(self.logprobs)


def favouring(candidate: str, labels: dict[str, str] | None = None) -> dict[str, float]:
    """Log-probabilities where *candidate*'s label wins and every other label has some mass."""
    labels = labels or sc.labels_for(DOMAIN, POOL)
    return {
        label: (-0.5 if name == candidate else -5.0 - index * 0.1)
        for index, (name, label) in enumerate(labels.items())
    }


def score(top_k, text: str = "x", **kwargs) -> sc.ScorerPrediction:
    kwargs.setdefault("world", WORLD)
    return sc.score(DOMAIN, top_k, "prompt text", text, **kwargs)


# -- candidates and labels --


def test_default_labels_are_positional_over_the_domain_pool() -> None:
    labels = sc.labels_for(DOMAIN, POOL)
    assert labels == {name: ro.LABEL_ALPHABET[i] for i, name in enumerate(POOL)}
    assert POOL[-2:] == ("explain", "escalate")


def test_a_label_stays_with_its_candidate_when_others_are_not_offered() -> None:
    full = sc.labels_for(DOMAIN, POOL)
    offered = POOL[1:]
    assert sc.labels_for(DOMAIN, offered) == {name: full[name] for name in offered}


def test_an_unknown_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the candidate pool"):
        sc.labels_for(DOMAIN, ("no_such_operation",))


def test_reasons_mode_replaces_bare_escalate_with_the_domain_reasons() -> None:
    pool = sc.candidate_pool(DOMAIN, reasons=True)
    assert "escalate" not in pool
    assert pool[-4:] == DOMAIN.escalate_labels()
    labels = sc.labels_for(DOMAIN, pool, reasons=True)
    assert labels["escalate:injection"] == ro.LABEL_ALPHABET[len(pool) - 1]


# -- the prompt --


def test_default_prompt_text_is_pinned() -> None:
    messages = sc.prompt_messages(DOMAIN, "Turn on the kitchen", ("lamp_on", "explain", "escalate"))
    assert messages == [
        {
            "role": "system",
            "content": (
                DOMAIN.instruction + "\n\nActions:\n"
                "D) lamp_on: Turn on the lamps in one named room.\n"
                "F) explain: " + sc.CONTROL_DESCRIPTIONS["explain"] + "\n"
                "G) escalate: " + sc.CONTROL_DESCRIPTIONS["escalate"]
            ),
        },
        {"role": "user", "content": "Turn on the kitchen"},
    ]


def test_the_prompt_lists_every_offered_candidate_with_its_label() -> None:
    offered = POOL[1:]
    system = sc.prompt_messages(DOMAIN, "hi", offered)[0]["content"]
    labels = sc.labels_for(DOMAIN, offered)
    assert system.startswith(DOMAIN.instruction)
    for name in offered:
        assert f"{labels[name]}) {name}: " in system
    assert f") {POOL[0]}:" not in system


def test_reason_candidates_take_their_text_from_the_domain() -> None:
    pool = sc.candidate_pool(DOMAIN, reasons=True)
    system = sc.prompt_messages(DOMAIN, "x", pool, reasons=True)[0]["content"]
    for label, text in DOMAIN.reason_descriptions().items():
        assert f") {label}: {text}" in system


def test_explicit_labels_and_order_render_in_the_given_order() -> None:
    order = ("escalate", "explain")
    labels = {"escalate": "Z", "explain": "Y"}
    system = sc.prompt_messages(DOMAIN, "x", labels=labels, order=order)[0]["content"]
    assert system.index("Z) escalate:") < system.index("Y) explain:")


def test_descriptions_can_be_overridden_for_paraphrase_probes() -> None:
    system = sc.prompt_messages(
        DOMAIN, "x", ("lamp_status",), descriptions={"lamp_status": "Paraphrased."}
    )[0]["content"]
    assert "A) lamp_status: Paraphrased." in system


def test_a_label_missing_from_an_explicit_map_is_refused() -> None:
    with pytest.raises(ValueError, match="no label given"):
        sc.prompt_messages(DOMAIN, "x", labels={"explain": "A"}, order=("explain", "escalate"))


class PlainTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "\n".join(m["content"] for m in messages) + "\nANSWER:"


def test_render_prompt_turns_thinking_off_only_when_the_template_has_the_switch() -> None:
    plain = PlainTokenizer()
    sc.render_prompt(plain, [{"role": "user", "content": "x"}])
    assert "enable_thinking" not in plain.kwargs
    assert plain.kwargs["add_generation_prompt"] is True
    thinking = PlainTokenizer()
    thinking.chat_template = "{% if enable_thinking is defined %}{% endif %}"
    sc.render_prompt(thinking, [{"role": "user", "content": "x"}])
    assert thinking.kwargs["enable_thinking"] is False


# -- randomised letters, order and subsets --


def test_permute_is_deterministic_and_covers_the_pool() -> None:
    first, second = sc.permute("seed-1", POOL), sc.permute("seed-1", POOL)
    assert first == second
    assert set(first.order) == set(POOL)
    assert len(set(first.labels.values())) == len(POOL)
    assert set(first.labels.values()) <= set(ro.LABEL_ALPHABET)


def test_permute_differs_across_seeds() -> None:
    assert sc.permute("seed-1", POOL) != sc.permute("seed-2", POOL)


def test_permute_subset_keeps_the_gold_candidate() -> None:
    perm = sc.permute("seed-3", POOL, subset=3, keep="lamp_on")
    assert len(perm.order) == 3
    assert "lamp_on" in perm.order
    assert set(perm.labels) == set(perm.order)
    assert perm == sc.permute("seed-3", POOL, subset=3, keep="lamp_on")


def test_permute_refuses_a_subset_smaller_than_what_must_be_kept() -> None:
    with pytest.raises(ValueError, match="smaller"):
        sc.permute("seed-5", POOL, subset=0, keep="lamp_on")


def test_permutation_round_trips_through_json_and_scores_the_same_prompt() -> None:
    perm = sc.permute("seed-7", POOL, subset=4, keep="lamp_status")
    stored = sc.Permutation.from_json(json.loads(json.dumps(perm.to_json())))
    assert stored == perm
    rendered = sc.prompt_messages(DOMAIN, "which lights", labels=stored.labels, order=stored.order)
    assert rendered == sc.prompt_messages(
        DOMAIN, "which lights", labels=perm.labels, order=perm.order
    )
    fake = FakeTopK(favouring("lamp_status", dict(stored.labels)))
    result = score(fake, labels=stored.labels, order=stored.order)
    assert result.choice == "lamp_status"
    assert set(result.probabilities) == set(stored.order)


# -- scoring through readout.py --


def test_score_returns_a_normalised_distribution_and_its_argmax() -> None:
    fake = FakeTopK(favouring("explain"))
    result = score(fake, "What is a lighting scene?")
    assert math.isclose(sum(result.probabilities.values()), 1.0, rel_tol=1e-9)
    assert result.choice == "explain"
    assert result.outcome == "explain"
    assert result.operation is None
    assert result.confidence == max(result.probabilities.values())
    assert fake.prompts == ["prompt text"]
    assert fake.tops == [ro.READOUT_TOP]


def test_score_over_a_subset_scores_only_that_subset() -> None:
    offered = POOL[2:]
    result = score(FakeTopK(favouring(POOL[0])), offered=offered)
    assert set(result.probabilities) == set(offered)
    assert [name for _, name in result.offered] == list(offered)


def test_no_label_mass_makes_no_choice() -> None:
    result = score(FakeTopK({"the": -0.1}))
    assert result.choice is None
    assert result.outcome is None
    assert result.probabilities == {}
    assert result.confidence == 0.0
    assert result.candidates is None


def test_a_failing_top_k_makes_no_choice_and_does_not_raise() -> None:
    def broken(prompt, top):
        raise OSError("no server")

    assert score(broken).choice is None


def test_argmax_ties_break_by_listing_order() -> None:
    labels = sc.labels_for(DOMAIN, POOL)
    result = score(FakeTopK({label: -1.0 for label in labels.values()}))
    assert result.choice == POOL[0]


def test_a_result_missing_labels_is_incomplete_not_renormalised() -> None:
    labels = sc.labels_for(DOMAIN, POOL)
    fake = FakeTopK({labels["lamp_status"]: math.log(0.01), "the": math.log(0.9)})
    result = score(fake)
    assert result.probabilities == {}
    assert result.candidates is None
    assert result.incomplete and "missing" in result.incomplete
    assert set(result.missing) == set(POOL) - {"lamp_status"}
    assert result.choice == "lamp_status"
    assert result.confidence == pytest.approx(0.01)
    assert result.mass == pytest.approx(0.01)
    assert result.raw_scores["lamp_status"] == pytest.approx(0.01)
    assert result.raw_scores["explain"] is None


def test_token_variants_of_one_label_are_summed_into_raw_scores() -> None:
    labels = sc.labels_for(DOMAIN, ("explain", "escalate"))
    fake = FakeTopK(
        {
            labels["explain"]: math.log(0.2),
            " " + labels["explain"]: math.log(0.2),
            labels["escalate"]: math.log(0.2),
        }
    )
    result = score(fake, offered=("explain", "escalate"))
    assert result.probabilities["explain"] == pytest.approx(2 / 3)
    assert result.raw_scores == pytest.approx({"explain": 0.4, "escalate": 0.2})
    assert result.mass == pytest.approx(0.6)


def test_candidates_use_the_calibration_labels_for_the_controls() -> None:
    result = score(FakeTopK(favouring("escalate")))
    assert set(result.candidates) == set(DOMAIN.names()) | {"(explain)", "(escalate)"}
    assert result.candidates["(escalate)"] == result.probabilities["escalate"]
    assert result.outcome == "escalate"


def test_a_reason_choice_is_an_escalate_outcome_with_its_reason() -> None:
    pool = sc.candidate_pool(DOMAIN, reasons=True)
    labels = sc.labels_for(DOMAIN, pool, reasons=True)
    result = score(FakeTopK(favouring("escalate:injection", labels)), reasons=True)
    assert result.choice == "escalate:injection"
    assert result.outcome == "escalate"
    assert result.reason == "escalate:injection"
    assert result.operation is None


def test_same_choice_compares_by_name_not_letter() -> None:
    plain = score(FakeTopK(favouring("explain")))
    order = tuple(reversed(POOL))
    labels = {name: letter for name, letter in zip(order, reversed(ro.LABEL_ALPHABET))}
    permuted = score(FakeTopK(favouring("explain", labels)), labels=labels, order=order)
    assert sc.same_choice(plain, permuted)
    assert not sc.same_choice(plain, "escalate")


# -- the predictions record --


def test_the_predictions_record_carries_every_seam_field_and_round_trips() -> None:
    fake = FakeTopK(favouring("lamp_on"))
    result = sc.score(
        DOMAIN, fake, "p", "Turn on the kitchen lights", world=WORLD, entry_id="toy-07"
    )
    data = json.loads(json.dumps(result.to_dict()))
    for key in (
        "id",
        "offered",
        "raw_scores",
        "probabilities",
        "outcome",
        "operation",
        "arguments",
        "grounded",
    ):
        assert key in data
    assert data["id"] == "toy-07"
    assert data["offered"][3] == {"label": "D", "name": "lamp_on"}
    assert data["outcome"] == "propose"
    assert data["operation"] == "lamp_on"
    assert data["arguments"] == {"room": "kitchen"}
    assert data["grounded"] is True
    assert data["candidates"]["(explain)"] == data["probabilities"]["explain"]
    assert sc.ScorerPrediction.from_dict(data) == result


# -- arguments are grounded deterministically, never generated --


def test_a_room_argument_is_grounded_from_the_request_text() -> None:
    result = score(FakeTopK(favouring("lamp_on")), "Turn on the KITCHEN lights")
    assert result.choice == "lamp_on"
    assert result.arguments == {"room": "kitchen"}  # the world's own spelling
    assert result.grounded is True
    assert result.grounding is None


def test_the_model_output_never_supplies_an_argument() -> None:
    labels = sc.labels_for(DOMAIN, POOL)
    logprobs = favouring("lamp_on", labels)
    logprobs.update({"study": -0.01, " bedroom": -0.02})  # the model "saying" a room
    result = score(FakeTopK(logprobs), "Turn on the kitchen lights")
    assert result.arguments == {"room": "kitchen"}


def test_sentence_punctuation_after_a_name_does_not_hide_it() -> None:
    op = DOMAIN.get("lamp_on")
    assert sc.ground_arguments(DOMAIN, op, "turn on the study.", WORLD) == {"room": "study"}
    assert sc.ground_arguments(DOMAIN, op, "lights: hallway:", WORLD) == {"room": "hallway"}


def test_a_choice_argument_matches_any_declared_spelling() -> None:
    op = DOMAIN.get("set_scene")
    assert sc.ground_arguments(DOMAIN, op, "night light mode", WORLD) == {"scene": "night_light"}
    assert sc.ground_arguments(DOMAIN, op, "go night-light", WORLD) == {"scene": "night_light"}
    assert sc.ground_arguments(DOMAIN, op, "Set the READING scene", WORLD) == {"scene": "reading"}


def test_two_choices_named_are_ambiguous_not_a_pick() -> None:
    result = sc.ground_arguments(DOMAIN, DOMAIN.get("set_scene"), "bright or reading", WORLD)
    assert isinstance(result, str) and "ambiguous" in result


def test_an_argument_nothing_grounds_is_reported_not_guessed() -> None:
    result = score(FakeTopK(favouring("lamp_on")), "turn them on")
    assert result.choice == "lamp_on"
    assert result.outcome == "propose"
    assert result.arguments is None
    assert result.grounded is False
    assert "room" in result.grounding


def test_two_grounded_values_are_ambiguous_not_a_pick() -> None:
    result = sc.ground_arguments(DOMAIN, DOMAIN.get("lamp_on"), "kitchen or bedroom", WORLD)
    assert isinstance(result, str) and "ambiguous" in result


def test_an_operation_with_no_arguments_needs_no_lookup() -> None:
    def lookup():
        raise AssertionError("no lookup for an operation without arguments")

    kind = replace(DOMAIN.ground_kinds[0], lookup=lookup)
    domain = replace(DOMAIN, ground_kinds=(kind,))
    assert sc.ground_arguments(domain, domain.get("lamp_status"), "anything at all") == {}


def test_a_live_lookup_runs_once_per_request() -> None:
    calls: list[int] = []

    def lookup():
        calls.append(1)
        return ["kitchen", "study"]

    domain = replace(DOMAIN, ground_kinds=(replace(DOMAIN.ground_kinds[0], lookup=lookup),))
    result = sc.ground_arguments(domain, domain.get("lamp_on"), "a b c d e f study")
    assert result == {"room": "study"}
    assert len(calls) == 1


def test_a_failed_lookup_is_reported() -> None:
    def lookup():
        raise OSError("no controller")

    domain = replace(DOMAIN, ground_kinds=(replace(DOMAIN.ground_kinds[0], lookup=lookup),))
    result = sc.ground_arguments(domain, domain.get("lamp_on"), "kitchen")
    assert isinstance(result, str) and "could not list rooms" in result


def test_a_free_text_argument_with_no_ground_kind_is_never_generated() -> None:
    op = Operation("note", "Leave a note.", False, (ArgSpec("text", "str"),))
    domain = replace(DOMAIN, operations=DOMAIN.operations + (op,))
    result = sc.ground_arguments(domain, op, "remember milk", WORLD)
    assert isinstance(result, str) and "text" in result


# -- served top-k endpoint (injected post, no server) --


def test_the_served_request_asks_for_readout_top_and_one_token() -> None:
    sent: list[tuple[str, dict, float]] = []
    labels = sc.labels_for(DOMAIN, POOL)

    def post(url, body, timeout):
        sent.append((url, dict(body), timeout))
        return {"choices": [{"logprobs": {"top_logprobs": [favouring("explain", labels)]}}]}

    top_k = sc.served_top_k("http://127.0.0.1:8000/v1", "scorer", post=post)
    result = score(top_k)
    assert result.choice == "explain"
    url, body, _ = sent[0]
    assert url == "http://127.0.0.1:8000/v1/completions"
    assert body["max_tokens"] == 1
    assert body["logprobs"] == ro.READOUT_TOP
    assert body["temperature"] == 0
    assert body["prompt"] == "prompt text"
    assert body["model"] == "scorer"


def test_the_served_parser_reads_the_content_shape_too() -> None:
    def post(url, body, timeout):
        entries = [{"token": "A", "logprob": -0.1}, {"token": " B", "logprob": -2.0}]
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}]}

    assert sc.served_top_k("http://127.0.0.1:1/v1", "m", post=post)("p", 5) == {
        "A": -0.1,
        " B": -2.0,
    }


def test_a_served_reply_without_logprobs_raises_and_scoring_absorbs_it() -> None:
    top_k = sc.served_top_k("http://127.0.0.1:1/v1", "m", post=lambda u, b, t: {"choices": []})
    with pytest.raises(sc.ServedError):
        top_k("p", 5)
    assert score(top_k).choice is None


# -- in-process scoring (plain-number logits, no torch) --


class VocabTokenizer:
    """Every letter as itself, spaced and tabbed, plus filler; a label encodes to its bare token."""

    def __init__(self) -> None:
        self.texts = ["<pad>", "the", "\n"]
        for letter in ro.LABEL_ALPHABET:
            self.texts += [letter, " " + letter, "\t" + letter]
        self.texts += ["AB", " the"]

    def get_vocab(self):
        return {f"tok{i}": i for i in range(len(self.texts))}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.texts[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.texts.index(text)] if text in self.texts else [1, 2]

    def __len__(self):
        return len(self.texts)


def fixture_logits(size: int) -> list[float]:
    return [((i * 7919) % 97) / 13.0 - 3.0 for i in range(size)]


def test_in_process_and_served_paths_give_the_same_distribution() -> None:
    tokenizer = VocabTokenizer()
    logits = fixture_logits(len(tokenizer))
    prompts: list[str] = []

    def logprobs_fn(prompt):
        prompts.append(prompt)
        return ro.log_softmax(logits)

    in_process = sc.InProcessTopK(tokenizer, logprobs_fn)
    local = score(in_process)
    assert prompts == ["prompt text"]
    assert local.incomplete is None and local.missing == ()

    served_logprobs: dict[str, float] = {}
    for index, value in enumerate(ro.log_softmax(logits)):
        text = tokenizer.texts[index]
        prior = served_logprobs.get(text)
        served_logprobs[text] = (
            value if prior is None else math.log(math.exp(prior) + math.exp(value))
        )
    served = score(FakeTopK(served_logprobs))
    for name in POOL:
        assert local.probabilities[name] == pytest.approx(served.probabilities[name], abs=1e-9)
    assert local.mass == pytest.approx(served.mass, abs=1e-9)


def test_in_process_scoring_holds_under_a_permuted_letter_map() -> None:
    tokenizer = VocabTokenizer()
    in_process = sc.InProcessTopK(
        tokenizer, lambda p: ro.log_softmax(fixture_logits(len(tokenizer)))
    )
    perm = sc.permute("seed-9", POOL, subset=4)
    result = score(in_process, labels=perm.labels, order=perm.order)
    assert result.incomplete is None
    assert set(result.probabilities) == set(perm.order)


def test_in_process_refuses_a_tokenizer_missing_a_label_token() -> None:
    class Short(VocabTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.texts = [t if t.strip() != "z" else "zz" for t in self.texts]

    with pytest.raises(ValueError, match="no token"):
        sc.InProcessTopK(Short(), lambda p: [])
