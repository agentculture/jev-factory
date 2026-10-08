"""Drafting: eval pool (teachers) and the sealed held-out (non-teacher, in process).

A fake gateway caller replaces every teacher call and a fake drafter replaces
the held-out model: no network, no GPU, no model download, no nvsh checkout.
Ported from nvsh's test_lfm_finetune_draft_sources.py and
test_lfm_finetune_draft_heldout.py onto the toy domain.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.data import draft as D
from jev_factory.data.teachers import RoleConfig, TeacherClient
from jev_factory.domain.model import Domain
from tests.fixtures.toy_domain import DOMAIN

YES = json.dumps({"verdict": "yes", "reason": "ok"})
NO = json.dumps({"verdict": "no", "reason": "wrong"})
SECRET_TEXT = "zq-unique-sealed-phrase"


def _role(name: str) -> RoleConfig:
    return RoleConfig(
        role=name,
        url="http://example.invalid/v1/chat",
        model=f"{name}-model",
        model_name=name,
        licence="Apache-2.0",
        max_tokens=64,
        timeout=1.0,
    )


ROLES = {name: _role(name) for name in ("generator", "reviewer_a", "reviewer_b")}


class FakeCaller:
    """Routes a call to a scripted generator or reviewer by role and prompt text."""

    def __init__(self, reject_b_containing: str | None = None, batch_items: int = 1):
        self.calls: list[tuple[str, str]] = []
        self.reject_b_containing = reject_b_containing
        self.batch_items = batch_items

    def __call__(self, role: RoleConfig, system: str, user: str) -> str:
        self.calls.append((role.role, user))
        if role.role == "generator":
            return self._generate(user)
        if role.role == "reviewer_b" and self.reject_b_containing in (None, ""):
            return YES
        if role.role == "reviewer_b" and self.reject_b_containing in user:
            return NO
        return YES

    def _generate(self, user: str) -> str:
        batch = user.split("Batch ")[-1][:6] if "Batch " in user else "1"
        for op in DOMAIN.operations:
            if f"operation {op.name}." in user:
                args = {
                    a.name: (a.choices[0] if a.kind == "choice" else "kitchen") for a in op.args
                }
                items = [
                    {"text": f"please run {op.name} #{i} b{batch}", "args": args}
                    for i in range(self.batch_items)
                ]
                return json.dumps(items)
        for name, definition in DOMAIN.reason_definitions().items():
            if definition in user:
                items = [
                    {"text": f"escalate {name} #{i} b{batch}"} for i in range(self.batch_items)
                ]
                return json.dumps(items)
        if "answered in words" in user:
            items = [
                {"text": f"What is scene {i} b{batch}?", "answer": "A preset."}
                for i in range(self.batch_items)
            ]
            return json.dumps(items)
        return "[]"


def _client(tmp_path: Path, caller=None, name: str = "cache") -> TeacherClient:
    return TeacherClient(
        ROLES, tmp_path / name, caller=caller or FakeCaller(), sleep=lambda s: None
    )


def _draft(tmp_path: Path, client=None, **kw):
    params = dict(seed=53, per_op=1, per_reason=1, explain=1, dev_texts=[])
    params.update(kw)
    return D.run_draft(DOMAIN, tmp_path / "out", client=client or _client(tmp_path), **params)


def _doc(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "draft.json").read_text())


# ---------------------------------------------------------------------------
# prompts are read from the Domain
# ---------------------------------------------------------------------------


def test_generator_prompt_uses_the_domain_persona_and_styles() -> None:
    system = D.generator_system(DOMAIN)
    assert DOMAIN.persona in system
    for style in DOMAIN.phrasing_styles:
        assert style in system
    assert "Jetson" not in system


def test_table_text_lists_every_operation_with_its_effect() -> None:
    text = D.table_text(DOMAIN)
    for op in DOMAIN.operations:
        assert f"- {op.name}: {op.description}" in text
    assert text.count(D.READ_ONLY_LABEL) == sum(1 for o in DOMAIN.operations if o.read_only)
    assert text.count(D.MUTATING_LABEL) == sum(1 for o in DOMAIN.operations if not o.read_only)
    assert "one of: bright/reading/night_light" in text


def test_explain_prompt_uses_the_domain_topics() -> None:
    _, user = D.explain_prompt(DOMAIN, 3)
    for topic in DOMAIN.explain_topics:
        assert topic in user


def test_decline_prompt_embeds_the_reason_definition() -> None:
    _, user = D.decline_prompt(DOMAIN, "missing_argument", 3)
    assert DOMAIN.reason_definitions()["missing_argument"] in user


def test_prompts_show_the_table_and_nothing_from_the_seed_corpus() -> None:
    definitions = " ".join(DOMAIN.reason_definitions().values())
    corpus_texts = [t for t in D.load_dev_texts(DOMAIN) if t not in definitions]
    assert corpus_texts
    prompts = [D.op_prompt(DOMAIN, "lamp_on", 2), D.explain_prompt(DOMAIN, 2)]
    prompts += [D.decline_prompt(DOMAIN, r.name, 2) for r in DOMAIN.reasons]
    for system, user in prompts:
        assert "Operations:" in user
        assert '"expect"' not in user
        for text in corpus_texts:
            assert text not in system + user


def test_reviewer_system_carries_the_answer_policy() -> None:
    assert DOMAIN.answer_policy in D.reviewer_system(DOMAIN)


def test_escalate_reviewer_prompt_names_the_reason_only_when_classed() -> None:
    _, with_cls = D.reviewer_prompt(DOMAIN, "t", {"escalate": True}, "decline:multi_step")
    assert DOMAIN.reason_definitions()["multi_step"] in with_cls
    _, without = D.reviewer_prompt(DOMAIN, "t", {"escalate": True}, None)
    assert "no listed operation can safely handle" in without
    for definition in DOMAIN.reason_definitions().values():
        assert definition not in without


def test_explain_reviewer_prompt_carries_the_answer() -> None:
    _, user = D.reviewer_prompt(DOMAIN, "what is X", {"explain": True, "answer": "X is Y."})
    assert "X is Y." in user


def test_reviewer_prompt_uses_read_only_flag_for_the_verb() -> None:
    _, ro = D.reviewer_prompt(DOMAIN, "t", {"operation": "lamp_status", "args": {}})
    _, mut = D.reviewer_prompt(DOMAIN, "t", {"operation": "lamp_on", "args": {"room": "study"}})
    assert "run this read-only check" in ro
    assert "propose this change" in mut


# ---------------------------------------------------------------------------
# small batches
# ---------------------------------------------------------------------------


def test_batch_sizes_never_exceed_the_cap() -> None:
    assert D.batch_sizes(60, 8) == [8] * 7 + [4]
    assert D.batch_sizes(3, 8) == [3]
    assert D.batch_sizes(0) == []
    assert sum(D.batch_sizes(60)) == 60


def test_a_sixty_item_explain_request_is_asked_in_small_batches(tmp_path: Path) -> None:
    caller = FakeCaller()
    _draft(tmp_path, _client(tmp_path, caller), per_op=0, per_reason=0, explain=60)
    asked = [u for role, u in caller.calls if role == "generator"]
    assert len(asked) == 8
    assert "Write 60" not in "".join(asked)
    assert all("Write 8 " in u or "Write 4 " in u for u in asked)
    assert len({u for u in asked}) == len(asked)  # distinct prompts: no cache collision


def test_a_large_per_op_request_is_split_per_operation(tmp_path: Path) -> None:
    caller = FakeCaller()
    _draft(tmp_path, _client(tmp_path, caller), per_op=20, per_reason=0, explain=0)
    asked = [u for role, u in caller.calls if role == "generator"]
    assert len(asked) == len(DOMAIN.operations) * 3


# ---------------------------------------------------------------------------
# generator-side validation and retries
# ---------------------------------------------------------------------------


def test_op_candidates_validate_args_and_drop_invalid(tmp_path: Path) -> None:
    def bad(role, system, user):
        return json.dumps(
            [
                {"text": "show the kitchen", "args": {"room": "kitchen"}},
                {"text": "show some room", "args": {}},
            ]
        )

    rejects: dict[str, int] = {}
    out = D._op_candidates(DOMAIN, _client(tmp_path, bad), 1, rejects, lambda k, fn: fn())
    room_status = [c for c in out if c.expect["operation"] == "room_status"]
    assert len(room_status) == 1
    assert rejects["invalid_args"] >= 1


def test_a_reply_that_is_not_a_json_list_is_retried_not_cached(tmp_path: Path) -> None:
    replies = iter(["sorry, no", "still no", '[{"text": "lights?"}]'])

    def caller(role, system, user):
        return next(replies)

    client = _client(tmp_path, caller)
    rejects: dict[str, int] = {}
    cands = D._decline_candidates(
        DOMAIN, client, 1, rejects, lambda k, fn: fn(), ("outside_table",)
    )
    assert [c.text for c in cands] == ["lights?"]
    assert rejects == {}


def test_a_generator_that_never_parses_is_counted_as_an_error(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda r, s, u: "not json at all")
    rejects: dict[str, int] = {}
    cands = D._explain_candidates(DOMAIN, client, 1, rejects, lambda k, fn: fn())
    assert cands == []
    assert rejects == {"generator_error": 1}


def test_bare_string_items_become_text_only_items() -> None:
    assert D.as_item("lights on") == {"text": "lights on"}
    assert D.as_item({"text": "a", "args": {}}) == {"text": "a", "args": {}}
    assert D.as_item(7) == {}


def test_a_json_reply_in_a_code_fence_is_parsed() -> None:
    reply = 'Here:\n```json\n[{"text": "a", "args": {}}]\n```'
    assert D.parse_json_list(reply) == [{"text": "a", "args": {}}]


# ---------------------------------------------------------------------------
# reviewers: both must accept; an error is not a reject
# ---------------------------------------------------------------------------


def test_judge_requires_both_reviewers_to_accept(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeCaller(reject_b_containing="please run lamp_status"))
    out = D.judge(
        DOMAIN, "please run lamp_status", {"operation": "lamp_status", "args": {}}, None, client
    )
    assert out["accepted"] is False
    assert out["errored"] is False
    assert out["votes"]["reviewer_a"]["accept"] is True
    assert out["votes"]["reviewer_b"]["accept"] is False


def test_judge_accepts_when_both_say_yes(tmp_path: Path) -> None:
    out = D.judge(DOMAIN, "x", {"operation": "lamp_status", "args": {}}, None, _client(tmp_path))
    assert out["accepted"] is True


def test_an_unusable_reviewer_reply_is_an_error_not_a_reject(tmp_path: Path) -> None:
    def caller(role, system, user):
        return "" if role.role == "reviewer_a" else YES

    out = D.judge(DOMAIN, "x", {"escalate": True}, "decline:multi_step", _client(tmp_path, caller))
    assert out["accepted"] is False
    assert out["errored"] is True
    assert out["votes"]["reviewer_a"]["accept"] is None


def test_errored_candidates_are_counted_apart_from_rejects(tmp_path: Path) -> None:
    class Caller(FakeCaller):
        def __call__(self, role, system, user):
            if role.role == "reviewer_a":
                return "{not json"
            return super().__call__(role, system, user)

    result = _draft(tmp_path, _client(tmp_path, Caller()), per_reason=0, explain=0)
    assert result["kept"] == 0
    assert result["rejects"].get("reviewer_a_error", 0) >= 1
    assert "reviewer_a" not in result["rejects"]
    ids = [
        json.loads(line)["id"]
        for line in (tmp_path / "out" / "review.jsonl").read_text().splitlines()
    ]
    assert ids
    assert all("-error-" in i for i in ids)


def test_reviewer_words_in_the_reason_do_not_flip_the_verdict(tmp_path: Path) -> None:
    reply = json.dumps({"verdict": "yes", "reason": "no-argument op, ambiguous but fine"})
    out = D.judge(
        DOMAIN,
        "x",
        {"explain": True, "answer": "a"},
        None,
        _client(tmp_path, lambda r, s, u: reply),
    )
    assert out["accepted"] is True


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def test_dedupe_drops_exact_match_against_the_seed_corpus() -> None:
    rejects: dict[str, int] = {}
    kept = D.dedupe_candidates(
        [D.Candidate("explain", "Which lights are on?", {})], ["Which lights are on?"], rejects
    )
    assert kept == []
    assert rejects == {"dedupe_exact": 1}


def test_dedupe_drops_near_duplicates_and_within_draft_repeats() -> None:
    base = "please switch on the lamp in the kitchen right now"
    near = "please switch on the lamp in the kitchen right away"
    rejects: dict[str, int] = {}
    kept = D.dedupe_candidates([D.Candidate("a", near, {})], [base], rejects)
    assert kept == []
    assert rejects == {"dedupe_near": 1}
    rejects = {}
    twice = [D.Candidate("a", "Show me the lamps", {}), D.Candidate("a", "Show me the lamps", {})]
    assert len(D.dedupe_candidates(twice, [], rejects)) == 1
    assert rejects == {"dedupe_exact": 1}


def test_dedupe_keeps_genuinely_different_text() -> None:
    rejects: dict[str, int] = {}
    kept = D.dedupe_candidates(
        [D.Candidate("e", "What does dimming mean?", {})], ["Which lights are on?"], rejects
    )
    assert len(kept) == 1
    assert rejects == {}


def test_the_default_dedupe_source_is_the_domain_seed_corpus(tmp_path: Path) -> None:
    seed_texts = D.load_dev_texts(DOMAIN)

    def caller(role, system, user):
        if role.role == "generator" and "answered in words" in user:
            return json.dumps([{"text": seed_texts[0], "answer": "x"}])
        return YES if role.role != "generator" else "[]"

    result = D.run_draft(DOMAIN, tmp_path / "o", 1, 0, 0, 1, _client(tmp_path, caller))
    assert result["kept"] == 0
    assert result["rejects"]["dedupe_exact"] == 1


# ---------------------------------------------------------------------------
# ids, header, outputs
# ---------------------------------------------------------------------------


def test_ids_are_stable_across_identical_runs(tmp_path: Path) -> None:
    a = D.run_draft(DOMAIN, tmp_path / "1", 53, 1, 1, 1, _client(tmp_path, name="c1"), dev_texts=[])
    b = D.run_draft(DOMAIN, tmp_path / "2", 53, 1, 1, 1, _client(tmp_path, name="c2"), dev_texts=[])

    def ids(d: str) -> list[str]:
        doc = json.loads((tmp_path / d / "draft.json").read_text())
        return [e["id"] for e in doc["entries"]]

    assert ids("1") == ids("2")
    assert ids("1")
    assert a["sha256"] == b["sha256"]


def test_id_shape_carries_the_seed_and_source_id(tmp_path: Path) -> None:
    _draft(tmp_path, per_reason=0, explain=0)
    entries = _doc(tmp_path)["entries"]
    assert entries
    for entry in entries:
        assert entry["id"].startswith("s53-eval-op-")
        assert entry["source_id"] == entry["id"]
        assert entry["kind"] == "explicit"
        assert entry["source"] == "draft-eval"


def test_ids_differ_between_seeds(tmp_path: Path) -> None:
    _draft(tmp_path, seed=1, per_reason=0, explain=0)
    first = {e["id"] for e in _doc(tmp_path)["entries"]}
    (tmp_path / "out" / "draft.json").unlink()
    _draft(tmp_path, client=_client(tmp_path, name="c2"), seed=2, per_reason=0, explain=0)
    assert first.isdisjoint({e["id"] for e in _doc(tmp_path)["entries"]})


def test_decline_entries_carry_the_class(tmp_path: Path) -> None:
    _draft(tmp_path, per_op=0, explain=0)
    entries = _doc(tmp_path)["entries"]
    assert entries
    names = {r.name for r in DOMAIN.reasons}
    for entry in entries:
        assert entry["expect"] == {"escalate": True}
        assert entry["class"].removeprefix("decline:") in names


def test_header_records_pool_seed_models_counts_hash_and_domain(tmp_path: Path) -> None:
    _draft(tmp_path, seed=7)
    header = _doc(tmp_path)["header"]
    assert header["pool"] == "eval"
    assert header["seed"] == 7
    assert header["models"] == {n: c.model for n, c in ROLES.items()}
    assert header["domain"] == DOMAIN.name
    assert header["surface_sha256"] == DOMAIN.surface_sha256()
    assert "kept" in header["counts"]
    assert len(header["sha256"]) == 64
    assert header["sampling"]["per_call_seed"] is True


def test_only_reasons_limits_the_decline_draft(tmp_path: Path) -> None:
    _draft(tmp_path, per_op=0, explain=0, only_reasons=["multi_step", "injection"])
    doc = _doc(tmp_path)
    assert {e["class"] for e in doc["entries"]} == {"decline:injection", "decline:multi_step"}
    assert doc["header"]["reasons"] == ["multi_step", "injection"]


def test_only_reasons_refuses_an_unknown_reason() -> None:
    with pytest.raises(ValueError, match="unknown decline reason"):
        D.check_reasons(DOMAIN, ["multi_step", "nonsense"])
    assert D.check_reasons(DOMAIN, None) == tuple(r.name for r in DOMAIN.reasons)
    assert D.split_reasons("a, b,") == ["a", "b"]
    assert D.split_reasons(None) is None


def test_run_draft_return_value_carries_no_entry_text(tmp_path: Path) -> None:
    dumped = json.dumps(_draft(tmp_path))
    for banned in ("please run", "escalate ", "What is scene"):
        assert banned not in dumped


def test_the_review_log_keeps_text_for_the_eval_pool(tmp_path: Path) -> None:
    _draft(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "out" / "review.jsonl").read_text().splitlines()]
    assert rows
    assert all("text" in r for r in rows)


def test_the_teacher_pool_no_longer_drafts_the_heldout() -> None:
    assert D.POOL == "eval"


# ---------------------------------------------------------------------------
# item-level resume
# ---------------------------------------------------------------------------


def test_a_resumed_draft_sends_nothing_it_already_has(tmp_path: Path) -> None:
    jobdir = tmp_path / "jobs"
    first = FakeCaller()
    _draft(tmp_path, _client(tmp_path, first, "c1"), jobdir=jobdir)
    assert first.calls
    second = FakeCaller()
    (tmp_path / "out" / "draft.json").unlink()
    _draft(tmp_path, _client(tmp_path, second, "c2"), jobdir=jobdir)
    assert second.calls == []
    progress = json.loads((jobdir / "draft-generate.progress.json").read_text())
    assert progress["done"] == progress["total"] > 0


# ---------------------------------------------------------------------------
# per-call seeding
# ---------------------------------------------------------------------------


def test_call_seed_is_deterministic_and_keyed() -> None:
    base = D.call_seed(53, "generator", "p", 0)
    assert base == D.call_seed(53, "generator", "p", 0)
    assert base != D.call_seed(53, "reviewer_a", "p", 0)
    assert base != D.call_seed(53, "generator", "q", 0)
    assert base != D.call_seed(53, "generator", "p", 1)
    assert base != D.call_seed(99, "generator", "p", 0)


def test_a_seeded_caller_replays_the_same_seeds_and_varies_repeats() -> None:
    def run() -> list[int]:
        seen: list[int] = []
        caller = D.make_seeded_caller(53, raw_caller=lambda r, s, u, n: seen.append(n) or "x")
        caller(ROLES["reviewer_a"], "s", "u")
        caller(ROLES["reviewer_b"], "s", "u")
        caller(ROLES["reviewer_a"], "s", "u")  # a repeat
        return seen

    a = run()
    assert a == run()
    assert len(set(a)) == 3


def test_post_seeded_sends_a_top_level_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": " hi "}}]}).encode()

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return Resp()

    monkeypatch.setattr(D.urllib.request, "urlopen", fake_urlopen)
    assert D.post_seeded(ROLES["generator"], "sys", "usr", 12345) == "hi"
    assert captured["body"]["seed"] == 12345
    assert captured["body"]["model"] == "generator-model"
    assert captured["body"]["messages"][1]["content"] == "usr"


# ---------------------------------------------------------------------------
# sealed held-out: a non-teacher model, in process
# ---------------------------------------------------------------------------


class FakeDrafter:
    """A scripted in-process drafter; the requests it sees are recorded."""

    def __init__(self, per_batch: int = 2):
        self.prompts: list[tuple[str, str]] = []
        self.per_batch = per_batch
        self.n = 0

    def __call__(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        self.n += 1
        items = []
        for i in range(self.per_batch):
            text = f"{SECRET_TEXT} {self.n}-{i}"
            for op in DOMAIN.operations:
                if f"operation {op.name}." in user:
                    args = {
                        a.name: (a.choices[0] if a.kind == "choice" else "study") for a in op.args
                    }
                    items.append({"text": text, "args": args})
                    break
            else:
                items.append({"text": text, "answer": "an answer"})
        return json.dumps(items)


def _heldout(tmp_path: Path, drafter=None, **kw):
    params = dict(dev_texts=[], teachers={})
    params.update(kw)
    return D.draft_heldout(DOMAIN, tmp_path / "ho", 53, drafter or FakeDrafter(), **params)


def test_a_drafted_heldout_carries_the_header_prefix_and_is_read_only(tmp_path: Path) -> None:
    result = _heldout(tmp_path)
    path = Path(result["path"])
    doc = json.loads(path.read_text())
    assert doc["header"].startswith("Held-out split")
    assert doc["entries"]
    assert all(e["id"].startswith("ho53-") for e in doc["entries"])
    assert len({e["id"] for e in doc["entries"]}) == len(doc["entries"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_the_heldout_id_prefix_is_unique_per_seed(tmp_path: Path) -> None:
    assert D.held_out_id_prefix(46) != D.held_out_id_prefix(53)
    a = D.draft_heldout(DOMAIN, tmp_path / "a", 46, FakeDrafter(), dev_texts=[], teachers={})
    b = D.draft_heldout(DOMAIN, tmp_path / "b", 53, FakeDrafter(), dev_texts=[], teachers={})

    def ids(p: dict) -> set[str]:
        return {e["id"] for e in json.loads(Path(p["path"]).read_text())["entries"]}

    assert ids(a).isdisjoint(ids(b))


def test_only_counts_and_sha256_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _heldout(tmp_path)
    blob = json.dumps(result)
    assert SECRET_TEXT not in blob
    assert "an answer" not in blob
    assert set(result) == {"path", "entries", "by_kind", "sha256", "snapshot", "rejects"}
    assert len(result["sha256"]) == 64
    assert result["entries"] > 0
    assert result["sha256"] == D.sha256_file(Path(result["path"]))
    assert sum(result["by_kind"].values()) == result["entries"]
    captured = capsys.readouterr()
    assert SECRET_TEXT not in captured.out + captured.err
    side = D.load_sealed(Path(result["path"]))
    assert SECRET_TEXT not in repr(side)
    assert side.sha256 == result["sha256"]


def test_a_sealed_draft_is_never_overwritten(tmp_path: Path) -> None:
    result = _heldout(tmp_path)
    before = Path(result["path"]).read_bytes()
    drafter = FakeDrafter(per_batch=1)
    with pytest.raises(CliError) as err:
        _heldout(tmp_path, drafter)
    assert SECRET_TEXT not in str(err.value)
    assert "refusing to overwrite" in str(err.value)
    assert Path(result["path"]).read_bytes() == before


def test_the_heldout_drafter_sees_only_the_table(tmp_path: Path) -> None:
    drafter = FakeDrafter()
    _heldout(tmp_path, drafter, dev_texts=None)
    assert drafter.prompts
    definitions = " ".join(DOMAIN.reason_definitions().values())
    corpus = [t for t in D.load_dev_texts(DOMAIN) if t not in definitions]
    assert corpus
    for system, user in drafter.prompts:
        assert "Operations:" in user
        assert '"expect"' not in user
        assert all(text not in system + user for text in corpus)
    assert DOMAIN.persona in drafter.prompts[0][0]


def test_heldout_asks_are_small_batches_covering_ops_escalate_and_explain(tmp_path: Path) -> None:
    drafter = FakeDrafter()
    _heldout(tmp_path, drafter, per_op=3, escalate=16, explain=60)
    users = [u for _, u in drafter.prompts]
    assert len(users) == len(DOMAIN.operations) + 2 + 8
    assert not any("Write 16 " in u or "Write 60 " in u for u in users)
    kinds = [k for k, _, _ in D._held_out_asks(DOMAIN, 3, 16, 60, D.MAX_BATCH)]
    assert [k for k in kinds if k.startswith("escalate")] == ["escalate:1", "escalate:2"]


def test_heldout_drops_seed_corpus_repeats_and_bad_args(tmp_path: Path) -> None:
    seed_text = D.load_dev_texts(DOMAIN)[0]

    def drafter(system: str, user: str) -> str:
        if "operation lamp_on." in user:
            return json.dumps(
                [
                    {"text": seed_text, "args": {"room": "study"}},
                    {"text": "turn the study lamps on", "args": {}},
                    {"text": "turn the study lamps on please", "args": {"room": "study"}},
                ]
            )
        return "[]"

    result = D.draft_heldout(
        DOMAIN, tmp_path / "ho", 1, drafter, teachers={}, per_op=1, escalate=0, explain=0
    )
    assert result["entries"] == 1
    assert result["rejects"]["duplicate"] == 1
    assert result["rejects"]["invalid_op_args"] == 1


def test_an_unparsable_heldout_reply_is_counted_without_quoting_it(tmp_path: Path) -> None:
    result = _heldout(tmp_path, lambda s, u: f"{SECRET_TEXT} not json")
    assert result["entries"] == 0
    assert result["rejects"]["parse"] > 0
    assert SECRET_TEXT not in json.dumps(result)


def test_the_heldout_drafter_may_not_be_a_teacher(tmp_path: Path) -> None:
    for model in ("Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B", "senses", "org/gemma-4-26b-a4b"):
        drafter = FakeDrafter()
        with pytest.raises(CliError, match="is one of the teachers"):
            D.draft_heldout(DOMAIN, tmp_path / "x", 1, drafter, model=model, dev_texts=[])
    D.check_non_teacher(D.HELD_OUT_MODEL)  # the default drafter is not a teacher


def test_loading_the_real_drafter_is_lazy() -> None:
    import sys

    assert "transformers" not in sys.modules
    assert callable(D.load_qwen_drafter)


# ---------------------------------------------------------------------------
# review of an existing draft
# ---------------------------------------------------------------------------


def _write_draft(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"header": "drafted by X", "entries": entries}), encoding="utf-8")
    return path


ENTRY_A = {
    "id": "ho46-001",
    "kind": "explicit",
    "text": "please run lamp_status now",
    "expect": {"operation": "lamp_status", "args": {}},
    "source": "t",
}
ENTRY_B = {
    "id": "ho46-002",
    "kind": "explicit",
    "text": "rewire the whole house please",
    "expect": {"escalate": True},
    "source": "t",
}


def test_review_keeps_only_entries_both_reviewers_accept(tmp_path: Path) -> None:
    src = _write_draft(tmp_path / "draft.json", [ENTRY_A, ENTRY_B])
    client = _client(tmp_path, FakeCaller(reject_b_containing="rewire the whole house"))
    result = D.run_review(DOMAIN, src, tmp_path / "out", client, dev_texts=[])
    assert result["entries"] == 1
    out = tmp_path / "out" / "draft.json"
    doc = json.loads(out.read_text())
    assert [e["id"] for e in doc["entries"]] == ["ho46-001"]
    assert doc["header"].startswith("Held-out split")
    assert stat.S_IMODE(out.stat().st_mode) == 0o444
    assert "rewire" not in json.dumps(result)


def test_review_dedupes_against_the_seed_corpus(tmp_path: Path) -> None:
    text = D.load_dev_texts(DOMAIN)[0]
    src = _write_draft(tmp_path / "draft.json", [{**ENTRY_A, "text": text}])
    result = D.run_review(DOMAIN, src, tmp_path / "out", _client(tmp_path))
    assert result["entries"] == 0
    assert result["rejects"]["dedupe_exact"] == 1


def test_review_log_never_carries_text(tmp_path: Path) -> None:
    src = _write_draft(tmp_path / "draft.json", [ENTRY_A])
    D.run_review(DOMAIN, src, tmp_path / "out", _client(tmp_path), dev_texts=[])
    lines = (tmp_path / "out" / "review.jsonl").read_text().splitlines()
    assert lines
    assert all("text" not in json.loads(x) for x in lines)


def test_review_refuses_the_sealed_held_out_corpus(tmp_path: Path) -> None:
    sealed = tmp_path / "held-out.json"
    sealed.write_text(json.dumps({"header": "x", "entries": []}), encoding="utf-8")
    client = _client(tmp_path)
    with pytest.raises(D.HeldOutRefused):
        D.run_review(DOMAIN, sealed, tmp_path / "out", client)


def test_review_of_a_missing_file_names_no_text(tmp_path: Path) -> None:
    client = _client(tmp_path)
    with pytest.raises(CliError, match="not a readable draft file"):
        D.run_review(DOMAIN, tmp_path / "nope.json", tmp_path / "out", client, dev_texts=[])


def test_the_module_is_domain_agnostic_about_the_domain_name(tmp_path: Path) -> None:
    other = Domain.from_dict({**DOMAIN.to_dict(), "name": "other", "persona": "a tester"})
    assert "a tester" in D.generator_system(other)
    assert "other operation table" in D.held_out_header("m", "s", 1, other)
