"""The gate's durable ledger (resume after a stop or reset) and the billing journal.

Ported from nvsh evals/tool_jev/tests/test_ledger.py (the batch submit/requeue
cases are deferred with the batch APIs) plus runstate's billing behaviour.
"""

from __future__ import annotations

import json

import pytest

from jev_factory.evals import runstate
from jev_factory.evals.ledger import (
    DONE,
    INVALID,
    PENDING,
    CachedResponse,
    CallSpec,
    Ledger,
    LedgerCorrupt,
    LedgerLocked,
    LedgerStateError,
    canonical_json,
    ledger_key,
    prompt_hash,
)


def spec(case_id="c-1", **params):
    return CallSpec(
        provider="p",
        model="m",
        subject_role="subject",
        case_id=case_id,
        target="choice",
        prompt_hash=prompt_hash(f"prompt {case_id}"),
        params=params,
    )


def answer(text="D"):
    return CachedResponse(raw=text.encode(), model_id="m", response_id="r", usage={"x": 1})


def test_keys_are_canonical_and_parameter_sensitive():
    first, second = ledger_key(spec()), ledger_key(spec())
    assert first == second
    assert ledger_key(spec(max_output_tokens=1)) != ledger_key(spec(max_output_tokens=2))
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    nan_spec = spec(bad=float("nan"))
    with pytest.raises(ValueError):
        ledger_key(nan_spec)


def test_register_record_and_reopen(tmp_path):
    with Ledger(tmp_path) as ledger:
        key, other = ledger.register_many([spec("c-1"), spec("c-2")])
        assert ledger.register(spec("c-1")) == key
        assert ledger.keys(PENDING) == sorted([key, other])
        ledger.record_done(key, answer())
        ledger.mark_invalid(other, "malformed", answer("?"))
        again = answer()
        with pytest.raises(LedgerStateError):
            ledger.record_done(key, again)
        with pytest.raises(LedgerStateError):
            ledger.mark_invalid(key, "")
    with Ledger(tmp_path) as again:
        assert again.entry(key).state == DONE
        assert again.entry(other).state == INVALID
        assert again.entry(other).reason == "malformed"
        assert again.cached(key).raw == b"D"


def test_a_cached_answer_without_its_state_change_is_recovered_on_open(tmp_path):
    with Ledger(tmp_path) as ledger:
        key = ledger.register(spec())
        # Simulate a crash between the cache write and the state change.
        ledger._write_cache(key, answer(), DONE, None)
    with Ledger(tmp_path) as again:
        assert again.entry(key).state == DONE
    events = (tmp_path / "events.jsonl").read_text()
    assert "recovered_from_cache" in events


def test_one_live_ledger_per_run_dir(tmp_path):
    with Ledger(tmp_path):
        with pytest.raises(LedgerLocked):
            Ledger(tmp_path)


def test_corruption_is_never_silently_reset(tmp_path):
    with Ledger(tmp_path) as ledger:
        key = ledger.register(spec())
        ledger.record_done(key, answer())
    cache = tmp_path / "cache" / f"{key}.json"
    doc = json.loads(cache.read_text())
    doc["raw_sha256"] = "0" * 64
    cache.write_text(json.dumps(doc))
    with Ledger(tmp_path) as again:
        with pytest.raises(LedgerCorrupt):
            again.cached(key)
    (tmp_path / "ledger.json").write_text("{")
    with pytest.raises(LedgerCorrupt):
        Ledger(tmp_path)
    (tmp_path / "ledger.json").write_text(json.dumps({"schema": 99, "entries": {}}))
    with pytest.raises(LedgerCorrupt):
        Ledger(tmp_path)


def test_leftover_temps_are_removed_and_results_are_deterministic(tmp_path):
    (tmp_path / "ledger.json.tmp").write_text("partial")
    with Ledger(tmp_path) as ledger:
        assert not (tmp_path / "ledger.json.tmp").exists()
        key = ledger.register(spec())
        ledger.record_done(key, answer())
        first = ledger.results_json()
        with pytest.raises(LedgerStateError):
            ledger.return_to_pending([key])  # a done answer is final
        other = ledger.register(spec("c-2"))
        ledger.mark_invalid(other, "malformed", answer("?"))
        ledger.return_to_pending([other], reason="regrade")
        assert ledger.entry(other).state == PENDING
        assert ledger.cached(other) is None
        assert json.loads(first)["calls"][key]["state"] == DONE
    with Ledger(tmp_path) as again:
        again.mark_invalid(other, "malformed", answer("?"))
        first_again = again.results_json()
    with Ledger(tmp_path) as third:
        assert third.results_json() == first_again


def test_billing_counts_each_attempt_once_and_moves_a_torn_tail_aside(tmp_path):
    billing = runstate.Billing(tmp_path)
    entry = {"kind": "answer", "key": "k", "attempt_id": "k:r", "provider": "p", "cost_usd": 1.5}
    billing.append(entry)
    billing.append(entry)
    assert runstate.sums(billing.entries(), "provider") == {"p": 1.5}
    assert billing.billed_keys() == {"k"}
    with open(billing.path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "answer", "cost')
    assert len(billing.entries(tolerant=True)) == 1
    aside = billing.repair(now=5.0)
    assert aside is not None
    assert aside.name == "billing.torn.5"
    assert len(billing.entries()) == 1


def test_a_torn_line_in_the_middle_stops_and_asks(tmp_path):
    billing = runstate.Billing(tmp_path)
    billing.path.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
    with pytest.raises(runstate.BillingTorn):
        billing.repair(now=1.0)
    with pytest.raises(runstate.BillingTorn):
        billing.entries()


def test_state_writes_are_durable_json(tmp_path):
    runstate.save_state(tmp_path, {"b": 1, "a": 2})
    assert runstate.load_state(tmp_path) == {"a": 2, "b": 1}
    assert runstate.read_json(tmp_path / "missing.json", {"x": 1}) == {"x": 1}
    assert not list(tmp_path.glob("*.tmp"))
