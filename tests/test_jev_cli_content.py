"""The jev-CLI domain's content: answer policy, reasons, prose and the seed corpus (t33).

The seed corpus is agent-drafted and train-only, pending operator review (c68).
These tests pin its *shape*: every verb is covered, all eight decline classes
are covered, every operation entry validates against and grounds in the CLI's
own world, and the prose the generators read is present.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from jev_factory.domain.model import DECLINE_PREFIX, Grounded
from jev_factory.domain.validate import problems
from jev_factory.domains.jev_cli import content
from jev_factory.domains.jev_cli.generate import generate_domain, world
from jev_factory.measure.corpus import load_raw
from jev_factory.review.core import OPERATOR_SOURCE_PREFIX

DECLINE_CLASSES = (
    "outside_table",
    "repair",
    "diagnosis",
    "missing_argument",
    "not_a_request",
    "multi_step",
    "injection",
    "over_time",
)
PHRASING_CLASSES = {"imperative", "question", "terse", "jargon", "symptom"}
ID_RE = re.compile(r"^jcs-(op|ex|dc)-\d{3}$")


@pytest.fixture(scope="module")
def domain():
    return generate_domain()


@pytest.fixture(scope="module")
def corpus(domain):
    return domain.load_seed_corpus()


def _by(corpus, kind):
    return [e for e in corpus.entries if _kind(e) == kind]


def _kind(entry) -> str:
    expect = entry["expect"]
    if expect.get("escalate") is True:
        return "escalate"
    if expect.get("explain") is True:
        return "explain"
    return "operation"


# -- prose ---------------------------------------------------------------------


def test_domain_validates_and_carries_the_content(domain) -> None:
    assert problems(domain) == ()
    assert domain.answer_policy == content.ANSWER_POLICY
    assert domain.instruction == content.INSTRUCTION
    assert domain.persona == content.PERSONA
    assert domain.explain_topics == content.EXPLAIN_TOPICS
    assert domain.phrasing_styles == content.PHRASING_STYLES
    assert domain.seed_corpus == content.SEED_CORPUS
    assert domain.hub_prefix == ""  # parked operator question: hub prefix


def test_answer_policy_is_one_sentence_naming_propose_explain_and_hand_off() -> None:
    policy = content.ANSWER_POLICY
    assert policy.endswith(".")
    assert len(re.findall(r"[.!?](\s|$)", policy)) == 1, "one sentence only"
    for word in ("propose", "in words", "hand off"):
        assert word in policy


def test_instruction_asks_for_the_letter_only() -> None:
    assert content.INSTRUCTION.endswith("Answer with the action's letter only.")


def test_persona_topics_and_styles_are_substantial() -> None:
    assert "shell" in content.PERSONA
    assert len(content.EXPLAIN_TOPICS) >= 8
    assert len(set(content.EXPLAIN_TOPICS)) == len(content.EXPLAIN_TOPICS)
    assert len(content.PHRASING_STYLES) >= 5


def test_the_eight_decline_reasons_each_have_prompt_and_generator_text(domain) -> None:
    assert tuple(r.name for r in domain.reasons) == DECLINE_CLASSES
    assert domain.default_reason == "outside_table"
    for reason in domain.reasons:
        assert reason.description.startswith("Hand this off:")
        assert len(reason.definition) >= 40
    missing = domain.reason("missing_argument").definition
    assert "jev run" in missing and "jev explain" in missing


def test_paraphrases_cover_every_operation_with_at_least_two(domain) -> None:
    for name in domain.names():
        texts = domain.paraphrases_for(name)
        assert len(texts) >= 2, name
        assert domain.get(name).description not in texts
        assert len(set(texts)) == len(texts)


def test_candidate_count_with_reasons_fits_the_letters(domain) -> None:
    assert len(domain.candidates(with_reasons=True)) <= 52


# -- seed corpus -----------------------------------------------------------------


def test_seed_header_marks_it_train_only_and_pending_review(corpus) -> None:
    assert "Split 'train' of" in corpus.header
    assert "train-only" in corpus.header
    assert "operator review" in corpus.header
    assert "agent-drafted" in corpus.header


def test_seed_loads_against_the_domain_with_no_problems(domain) -> None:
    raw = json.loads(domain.seed_corpus.read_text(encoding="utf-8"))
    loaded = load_raw(raw, domain)
    assert loaded.problems == ()
    assert len(loaded.entries) == len(raw["entries"]) >= 250


def test_seed_ids_are_unique_and_prefixed(corpus) -> None:
    ids = [e["id"] for e in corpus.entries]
    assert len(ids) == len(set(ids))
    assert all(ID_RE.match(i) for i in ids), [i for i in ids if not ID_RE.match(i)]


def test_seed_texts_are_unique(corpus) -> None:
    norm = [" ".join(re.sub(r"[^\w\s]", " ", e["text"].lower()).split()) for e in corpus.entries]
    dupes = [t for t, n in Counter(norm).items() if n > 1]
    assert dupes == []


def test_seed_entries_carry_kind_class_and_source(corpus) -> None:
    for entry in corpus.entries:
        assert entry["kind"] == "explicit"
        # Agent drafts, or entries the operator proposed in `jev review`.
        assert entry["source"].startswith(("agent-draft", OPERATOR_SOURCE_PREFIX)), entry["id"]
        kind = _kind(entry)
        if kind == "operation":
            assert entry["class"] in PHRASING_CLASSES, entry["id"]
        elif kind == "explain":
            assert entry["class"].startswith("explain:"), entry["id"]
            assert entry["class"].removeprefix("explain:") in PHRASING_CLASSES
            assert len(entry["expect"]["answer"]) >= 20, entry["id"]
        else:
            assert entry["class"].removeprefix(DECLINE_PREFIX) in DECLINE_CLASSES, entry["id"]


def test_seed_covers_every_operation_several_times_in_several_phrasings(domain, corpus) -> None:
    ops = _by(corpus, "operation")
    counts = Counter(e["expect"]["operation"] for e in ops)
    assert set(counts) == set(domain.names())
    assert min(counts.values()) >= 4, counts.most_common()[-5:]
    for name in domain.names():
        classes = {e["class"] for e in ops if e["expect"]["operation"] == name}
        assert len(classes) >= 3, (name, classes)
    assert set().union(*({e["class"]} for e in ops)) == PHRASING_CLASSES


def test_seed_covers_every_decline_class(corpus) -> None:
    counts = Counter(e["class"].removeprefix(DECLINE_PREFIX) for e in _by(corpus, "escalate"))
    assert set(counts) == set(DECLINE_CLASSES)
    assert min(counts.values()) >= 6, counts


def test_seed_has_explain_entries_in_several_styles(corpus) -> None:
    explains = _by(corpus, "explain")
    assert len(explains) >= 30
    assert {e["class"] for e in explains} >= {"explain:question", "explain:terse"}


def test_mutating_verbs_have_dry_run_and_apply_phrasings(domain, corpus) -> None:
    ops = _by(corpus, "operation")
    mutating = [e for e in ops if domain.is_mutating(e["expect"]["operation"])]
    texts = " ".join(e["text"].lower() for e in mutating)
    assert "--apply" in texts and "dry" in texts
    for name in domain.names():
        if domain.is_mutating(name):
            assert any(e["expect"]["operation"] == name for e in mutating), name


def test_every_operation_entry_grounds_in_the_cli_world(domain, corpus) -> None:
    live = world()
    for entry in _by(corpus, "operation"):
        name, args = entry["expect"]["operation"], entry["expect"]["args"]
        assert domain.validate_args(name, args) is None, entry["id"]
        for snapshot in (live, corpus.world):
            grounded = domain.ground(name, args, world=snapshot)
            assert isinstance(grounded, Grounded), (entry["id"], grounded)
        if name == "jev.explain":
            assert args["path"] in live["explain_paths"], entry["id"]


def test_seed_world_is_a_subset_of_the_live_cli_world(domain, corpus) -> None:
    assert domain.check_world(corpus.world) == ()
    live = world()
    for key in ("explain_paths", "stages"):
        assert set(corpus.world[key]) <= set(live[key]), key
