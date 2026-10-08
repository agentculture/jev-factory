"""Runner for the release gate: run / continue / status / smoke.

One run directory holds one gate run: the call ledger and response cache
(``ledger.py``), the run record (``run.json``), the billing record
(``billing.jsonl``), and, once every call is answered, the traces, the
per-(subject, policy) metrics, ``result.json`` and ``report.md``. The run
directory is PRIVATE and must sit outside every git worktree (refused
otherwise, judged by ``git rev-parse``); the runner never writes into a
repository.

Run-dir layout::

    run.json                 run record: run id/date, scope (full or the smoke
                             selection), manifest sha256 history, hosts that
                             received case text, stops, capability entries,
                             reservations, uncertain charges
    billing.jsonl            one line per charge, priced when it was answered
    ledger.json cache/ events.jsonl .lock    the call ledger; .lock is the run lock
    traces/<subject>.jsonl   raw record + each policy's final decision
    metrics/<subject>__<policy>.json         metrics_bridge.compute() output
    deepeval/<subject>__<policy>/            deepeval's per-case results (when it runs)
    manifest.json result.json report.md      what report.py reads and writes
    smoke.json               smoke only: tokens, cost, projection, OK/CAPPED

Subjects: candidates and baselines replay their saved predictions (no GPU);
references answer through one choice call per case. Every subject is scored
model-only (policy ``raw``); a candidate or baseline is also scored under
each harness policy it names (model+harness rows).

Money and stops
---------------
- **Every charge is billed once, at answer time** (``billing.jsonl``); spend
  is never recomputed from the current manifest.
- **Every call reserves its worst-case cost before it is sent** (prompt size
  plus its whole output budget). A provider sends nothing that would take
  spent + reserved + estimate past its ``usd_cap``; a reservation is released
  when its answer (or failure) is recorded.
- A call is marked in flight (the reservation) before it is sent. A crash
  leaves the marker: the next pass resends the call and bills its estimate as
  an *uncertain charge* (continue loses at most the calls in flight).
- **money stop** (402 / insufficient credit, budget cap) stops one provider;
  ``continue`` retries it. **transient stops** (429, timeouts, network)
  pause one provider (a timeout: one model) for :data:`PAUSE_SECONDS`; a
  **rejected request** stops one model while the rejected parameters are
  unchanged and becomes a capability entry.
- **truncation stop**: once ``[stops]`` thresholds are met, a model whose
  replies are cut at the output budget stops for an operator decision; a cut
  reply is never an answer, under any policy.

Every host that received case text is recorded in ``run.json`` before the
call leaves. Nothing here reads a key: adapters read their key from the
environment variable the manifest names, at call time.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from jev_factory.backbones.causal_lm.scorer import ground_arguments
from jev_factory.cli._errors import CliError
from jev_factory.domain.model import Domain

from . import request as contract
from . import runstate
from .cases import Case
from .ledger import (
    DONE,
    INVALID,
    PENDING,
    CachedResponse,
    CallSpec,
    Entry,
    Ledger,
    LedgerCorrupt,
    LedgerLocked,
)
from .manifest import Manifest, load_manifest
from .providers.base import CallRequest, CallResult
from .providers.errors import Classification, Outcome, stop_message
from .runplan import (  # noqa: F401  (re-exported: jev_factory.evals.run is the public API)
    CHOICE_TARGET,
    DEEPEVAL_DIR,
    ENV_BASE_URL,
    ENV_PRIVATE,
    ENV_PRIVATE_ROOT,
    EXIT_ASK,
    EXIT_ENV,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_STOPPED,
    EXIT_USER,
    MAX_UNCERTAIN_ATTEMPTS,
    MONEY_REASONS,
    NONE_REASONING,
    PAGE_FILE,
    RESULT_FILE,
    STATUS_COMPLETE,
    STATUS_STOPPED,
    TRUNCATION_DECISION,
    UNCERTAIN_REASONS,
    EnvError,
    Model,
    Plan,
    ProviderFactory,
    RunError,
    StopAndAsk,
    Subject,
    WorkItem,
    _classification_of,
    _sha256_file,
    build_plan,
    choice_call,
    classify_exception,
    default_factory,
    policy_path,
    private_root,
    provider_host,
    provider_reasoning,
    refused_before_send,
    request_fingerprint,
    resolve_private,
    shared_limiter,
    smoke_scope,
    subject_name,
    tokens_in,
    tokens_out,
)
from .runstatus import render_status, status  # noqa: F401  (re-exported)
from .trace import RawRecord, Trace, inside_git_worktree, write_traces

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/run.py",
    "commit": "9debdc6",
    "adaptations": [
        "kept: the pass engine's claim/reserve (worst-case reservation against each"
        " provider's usd_cap), answer-time billing, uncertain charges for calls in flight at"
        " a crash, money/transient/rejected stops, the truncation stop, host recording,"
        " per-provider concurrency pools, smoke scope and cost projection, run/continue",
        "deferred per c14 and not imported: Track A replay (track_a_loop, LfmTier), the judge"
        " panel (judge_plan/judge_work/judge_results), batch APIs (resolve_orphans,"
        " poll_batches, submit_batches, ghosts, batch failures and backoff), the drive loop,"
        " Discord alerts, the docker driver, permutation entries",
        "line 88: nvsh.ops.table -> the Domain; choice_record grounds a proposal with"
        " jev_factory.backbones.causal_lm.scorer.ground_arguments against the world snapshot",
        "a reference's distribution is keyed by calibration labels ('(explain)',"
        " '(escalate)') like a saved predictions line, so the gate and the metrics read it",
        "the worktree guard (_check_run_dir) uses trace.inside_git_worktree: git rev-parse"
        " via factory.workroot (nvsh#71 item 2)",
        "finalize runs the DeepEval layer unless deepeval=False (--no-deepeval), which is"
        " recorded in result.json; without deepeval installed a full run stops before any"
        " call with an environment error naming `uv run --group evals`",
    ],
    "licence": "Apache-2.0",
}

RUN_FILE = runstate.RUN_FILE
SMOKE_FILE = runstate.SMOKE_FILE
BILLING_FILE = runstate.BILLING_FILE

#: ``invalid_reason`` of a reply cut at the output budget.
TRUNCATED = "truncated"

#: A transient pause lasts this long, not the rest of the pass.
PAUSE_SECONDS = 60.0

#: A round of sync calls stops claiming new calls after this many seconds;
#: calls in flight finish, the rest stay pending, and the pass loops.
ROUND_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


def _new_state(digest: str, run_id: str | None, date: str | None, scope: dict) -> dict:
    return {
        "schema": 2,
        "run_id": run_id or digest[:12],
        "date": date or datetime.date.today().isoformat(),
        "manifest_sha256": digest,
        "manifest_history": [digest],
        "scope": scope,
        "hosts": {},
        "stops": {},
        "capabilities": {},
        "reserved": {},
        "uncertain_charges": [],
    }


def _open_ledger(run_dir: Path) -> Ledger:
    try:
        return Ledger(run_dir)
    except LedgerLocked as exc:
        raise RunError(f"{run_dir} is in use by another runner: {exc}") from exc
    except LedgerCorrupt as exc:
        raise RunError(f"{run_dir}: the ledger cannot be trusted: {exc}") from exc


def _check_run_dir(run_dir: Path) -> None:
    try:
        inside = inside_git_worktree(run_dir)
    except CliError as exc:
        raise EnvError(f"cannot check the run dir: {exc.message}") from exc
    if inside:
        raise RunError(f"{run_dir} is inside a git worktree; the run dir must be private")


def _check_deepeval(use_deepeval: bool, scope: Mapping[str, Any]) -> None:
    """Refuse a full run that would need deepeval when it is not installed (before any call)."""
    from .deepeval_layer import deepeval_available

    if use_deepeval and scope.get("mode") != "smoke" and not deepeval_available():
        raise EnvError(
            "deepeval is not installed: run the gate under `uv run --group evals`, or pass "
            "--no-deepeval to score with the exact corpus metrics only (recorded in result.json)"
        )


# ---------------------------------------------------------------------------
# The pass engine
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    status: str
    messages: list[str] = field(default_factory=list)
    money_stopped: set[str] = field(default_factory=set)
    exit_code: int = EXIT_OK


class Runner:
    """One pass over a run dir, under the run lock: resume, send, finalize when done."""

    def __init__(
        self,
        run_dir: Path,
        plan: Plan,
        ledger: Ledger,
        state: dict,
        *,
        retry_rejected: bool = False,
        out: Callable[[str], None] = print,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        use_deepeval: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.plan = plan
        self.ledger = ledger
        self.state = state
        self.out = out
        self.env = os.environ if env is None else env
        self.clock = clock
        self.use_deepeval = use_deepeval
        self.messages: list[str] = []
        self._lock = threading.RLock()
        self.billing = runstate.Billing(self.run_dir)
        for name in ("reserved", "capabilities", "hosts", "uncertain_attempts"):
            self.state.setdefault(name, {})
        self.state.setdefault("uncertain_charges", [])
        self._spec_model: dict[tuple[str, str], Model] = {}
        for model in plan.models.values():
            self._spec_model[(model.provider.name, model.ref.model)] = model
            self._spec_model[(model.ref.provider, model.ref.model)] = model
        try:
            self.billing.repair(self.clock(), log=self.ledger._log)
            entries = self.billing.entries()
        except runstate.BillingTorn as exc:
            raise StopAndAsk(str(exc)) from exc
        self.billed: dict[str, float] = runstate.sums(entries, "provider")
        self._billed_keys = {e["key"] for e in entries if e.get("kind") == runstate.BILL_ANSWER}
        self._billed_attempts = {
            (e.get("kind"), e["attempt_id"]) for e in entries if e.get("attempt_id") is not None
        }
        self.answers: dict[str, int] = {}
        self.truncated: dict[str, int] = {}
        self.invalid: dict[str, int] = {}
        self.model_stops: dict[str, dict] = {}
        self.provider_stops: dict[str, dict] = {}
        self.paused: dict[str, str] = {}
        #: kind or label -> clock time a transient pause ends.
        self.paused_until: dict[str, float] = {}
        self._restore_stops(retry_rejected)
        self._reconcile_reservations()
        self._scan_ledger()
        for kind in sorted({m.kind for m in plan.models.values()}):
            self._check_budget(kind)
        for label in sorted(plan.models):
            self._check_truncation(plan.models[label])

    # -- bookkeeping -------------------------------------------------------

    def say(self, message: str) -> None:
        with self._lock:
            if message in self.messages:
                return
            self.messages.append(message)
        self.out(message)

    def _model_of_spec(self, spec: Mapping[str, Any]) -> Model | None:
        return self._spec_model.get((spec.get("provider"), spec.get("model")))

    def _current(self, model: Model, spec: Mapping[str, Any]) -> bool:
        """Whether *spec* was made with the model's current settings (stale ones never stop it)."""
        return spec.get("target") == CHOICE_TARGET and (spec.get("params") or {}) == model.knobs

    def _count(self, model: Model, key: str, spec: Mapping[str, Any], state: str) -> None:
        """Add one answered call to the model's answer / truncated / invalid counts."""
        if not self._current(model, spec):
            return
        label = model.label
        self.answers[label] = self.answers.get(label, 0) + 1
        cut = self.ledger.entry(key).reason == TRUNCATED
        cached = self.ledger.cached(key)
        if cached is not None and not cut:
            cut = _cut(model, cached.raw)
        if cut:
            self.truncated[label] = self.truncated.get(label, 0) + 1
        if state == INVALID:
            self.invalid[label] = self.invalid.get(label, 0) + 1

    def _scan_ledger(self) -> None:
        for entry in self.ledger.entries():
            if entry.state not in (DONE, INVALID):
                continue
            model = self._model_of_spec(entry.spec)
            if model is not None:
                self._count(model, entry.key, entry.spec, entry.state)

    # -- money: billing and reservations -----------------------------------

    def reserved_sum(self, kind: str) -> float:
        return sum(r["usd"] for r in self.state["reserved"].values() if r["provider"] == kind)

    def spent(self, kind: str) -> float:
        return self.billed.get(kind, 0.0)

    def _bill(self, entry: dict) -> bool:
        """Append one charge unless this attempt is already billed; True when appended."""
        attempt = (entry["kind"], entry["attempt_id"])
        if attempt in self._billed_attempts:
            return False
        self.billing.append(entry)
        self._billed_attempts.add(attempt)
        self.billed[entry["provider"]] = self.billed.get(entry["provider"], 0.0) + entry["cost_usd"]
        if entry["kind"] == runstate.BILL_ANSWER:
            self._billed_keys.add(entry["key"])
        return True

    def _bill_answer(self, model: Model, key: str, result: CallResult) -> None:
        self._bill(
            {
                "kind": runstate.BILL_ANSWER,
                "key": key,
                "attempt_id": f"{key}:{result.response_id}",
                "provider": model.kind,
                "label": model.label,
                "usd_per_mtok_in": model.ref.usd_per_mtok_in,
                "usd_per_mtok_out": model.ref.usd_per_mtok_out,
                "usage": dict(result.usage),
                "cost_usd": model.cost(result.usage),
            }
        )

    def _release(self, *keys: str) -> None:
        with self._lock:
            changed = False
            for key in keys:
                changed |= self.state["reserved"].pop(key, None) is not None
            if changed:
                self._persist()

    def _reconcile_reservations(self) -> None:
        """Settle reservations a crash left behind: a call in flight is an uncertain charge."""
        reserved = self.state["reserved"]
        for key in sorted(reserved):
            try:
                entry = self.ledger.entry(key)
            except Exception:  # noqa: BLE001 -- a key the ledger never knew: nothing to hold
                reserved.pop(key)
                continue
            if entry.state in (DONE, INVALID) or key in self._billed_keys:
                reserved.pop(key)
                continue
            self._uncertain(key, "in flight at a crash")
        self._persist()

    def _uncertain(self, key: str, why: str) -> None:
        """One uncertain attempt: bill its reservation once (by its id), count it, resend.

        The charge, the attempt count and the reservation's removal reach
        ``run.json`` in one save; the billing line is idempotent by attempt id,
        so a crash between the two never charges twice.
        """
        with self._lock:
            held = self.state["reserved"].pop(key)
            attempt_id = held.get("id") or f"legacy:{key}:{held.get('since')}"
            charge = {
                "kind": runstate.BILL_UNCERTAIN,
                "key": key,
                "attempt_id": attempt_id,
                "provider": held["provider"],
                "label": held["label"],
                "cost_usd": held["usd"],
                "since": held.get("since"),
                "why": why,
            }
            self._bill(charge)
            charges = self.state["uncertain_charges"]
            if all(c.get("attempt_id") != attempt_id for c in charges):
                charges.append(charge)
                self.ledger._log(
                    "uncertain_resend", [key], provider=held["provider"], usd=held["usd"]
                )
            attempts = self.state["uncertain_attempts"]
            attempts[key] = attempts.get(key, 0) + 1
            message = (
                f"{held['label']}: call {key[:12]} was {why}; it may have been billed, so its "
                f"estimated ${held['usd']:.4f} counts as an uncertain charge and it is resent"
            )
            model = self.plan.models.get(held["label"])
            # The stop guards money: a free model (local, a free tier) is resent.
            paid = bool(
                model is not None
                and (model.ref.usd_per_mtok_in > 0 or model.ref.usd_per_mtok_out > 0)
            )
            if attempts[key] >= MAX_UNCERTAIN_ATTEMPTS and paid:
                message = (
                    f"{held['label']}: call {key[:12]} had {attempts[key]} uncertain attempts "
                    f"(sent, maybe billed, no answer); {self._model_remaining(model)} call(s) "
                    "left pending; needs operator decision: check the provider's usage, then "
                    "continue with --retry-rejected"
                )
                self.model_stops[held["label"]] = {
                    "kind": "uncertain_attempts",
                    "reason": "uncertain_attempts",
                    "message": message,
                    "key": key,
                }
            self._persist()
        self.say(message)

    def claim(self, model: Model, key: str, request: CallRequest) -> bool:
        """Claim *key* for sending: fresh, unblocked, within budget; reserve its worst case.

        Called under the lock just before the call leaves. The reservation is
        persisted before the send, so it doubles as the call's in-flight marker.
        """
        with self._lock:
            if self.blocked(model) or key in self.state["reserved"] or not self._fresh(key):
                return False
            budget = self.plan.manifest.budget_for(model.kind)
            estimate = model.estimate(request)
            if budget is not None:
                room = budget.usd_cap - self.spent(model.kind) - self.reserved_sum(model.kind)
                if estimate > room and estimate > 0:
                    self._refuse_budget(model.kind, estimate, room)
                    return False
            self.state["reserved"][key] = {
                "id": uuid.uuid4().hex,
                "provider": model.kind,
                "label": model.label,
                "usd": round(estimate, 8),
                "since": self.clock(),
                "model": model.ref.model,
            }
            self._persist()
            return True

    def _refuse_budget(self, kind: str, estimate: float, room: float) -> None:
        message = (
            f"{kind}: stopping cleanly (budget_cap_reached: the next call's estimated "
            f"${estimate:.4f} exceeds the ${max(room, 0.0):.4f} left after "
            f"${self.spent(kind):.4f} spent and ${self.reserved_sum(kind):.4f} reserved); "
            f"{self._provider_remaining(kind)} call(s) left pending; raise [budget.{kind}] "
            "usd_cap, or continue once the reserved calls settle"
        )
        if self.reserved_sum(kind) > 0:
            self._pause(kind, message)
        else:
            self.provider_stops[kind] = {
                "kind": "budget_cap",
                "reason": "budget_cap_reached",
                "message": message,
            }
        self.say(message)

    # -- stops ---------------------------------------------------------------

    def _restore_stops(self, retry_rejected: bool) -> None:
        """Carry model stops over from the last pass.

        A rejected model stays stopped while the rejected request parameters
        are still sent, unless the operator retries it; an uncertain-attempts
        stop stays until ``--retry-rejected``. Provider money stops are retried
        by every new pass; budget-cap and truncation stops are recomputed.
        """
        stops = self.state.get("stops", {})
        for label, stop in stops.get("models", {}).items():
            model = self.plan.models.get(label)
            if model is None or retry_rejected:
                continue
            if stop.get("kind") == "uncertain_attempts":
                self.model_stops[label] = stop
            elif stop.get("kind") == "rejected":
                sent = (stop.get("request") or {}).get("params")
                if sent == request_fingerprint(model.knobs):
                    self.model_stops[label] = stop
        if retry_rejected:
            self.state["uncertain_attempts"] = {}

    def _persist(self) -> None:
        with self._lock:
            self.state["stops"] = {
                "models": dict(sorted(self.model_stops.items())),
                "providers": dict(sorted(self.provider_stops.items())),
            }
            self.state["models"] = {
                label: {
                    "provider": model.kind,
                    "host": model.host,
                    "spec_keys": sorted(
                        [list(k) for k, m in self._spec_model.items() if m is model]
                    ),
                    "answers": self.answers.get(label, 0),
                    "truncated": self.truncated.get(label, 0),
                    "invalid": self.invalid.get(label, 0),
                }
                for label, model in sorted(self.plan.models.items())
            }
            self.state["budgets"] = {
                b.provider: {"usd_cap": b.usd_cap} for b in self.plan.manifest.budgets
            }
            runstate.save_state(self.run_dir, self.state)

    def _note_host(self, model: Model) -> None:
        """Record *model*'s host as having received case text, before anything is sent."""
        with self._lock:
            names = self.state["hosts"].setdefault(model.host, [])
            if model.label not in names:
                names.append(model.label)
                names.sort()
                self._persist()

    def _remaining(self, keys_of: Callable[[Mapping[str, Any]], bool]) -> int:
        return sum(
            1 for entry in self.ledger.entries() if entry.state == PENDING and keys_of(entry.spec)
        )

    def _model_remaining(self, model: Model | None) -> int:
        return self._remaining(lambda spec: self._model_of_spec(spec) is model)

    def _provider_remaining(self, kind: str) -> int:
        return self._remaining(
            lambda spec: (m := self._model_of_spec(spec)) is not None and m.kind == kind
        )

    def _pause_for(self, model: Model, classification: Classification, message: str) -> None:
        """A timeout pauses only the model that timed out; anything else the whole provider."""
        scope = model.label if classification.reason == "timeout" else model.kind
        self._pause(scope, message)

    def _pause(self, kind: str, message: str) -> None:
        self.paused[kind] = message
        self.paused_until[kind] = self.clock() + PAUSE_SECONDS

    def is_paused(self, kind: str) -> bool:
        """Whether *kind* is in a transient pause that has not run out yet."""
        with self._lock:
            if kind not in self.paused:
                return False
            if self.clock() < self.paused_until.get(kind, float("inf")):
                return True
            self.paused.pop(kind, None)
            self.paused_until.pop(kind, None)
            return False

    def blocked(self, model: Model) -> bool:
        if self.is_paused(model.kind) or self.is_paused(model.label):
            return True
        with self._lock:
            return model.label in self.model_stops or model.kind in self.provider_stops

    def apply_stop(
        self, model: Model, classification: Classification, params: Mapping[str, Any]
    ) -> None:
        """Stop or pause for *classification*.

        A rejection is fingerprinted by the parameters that were rejected, so
        it stops the model only while those are still sent.
        """
        with self._lock:
            if classification.rejected or not classification.retryable:
                sent = {"params": request_fingerprint(params)}
                message = stop_message(model.label, classification, self._model_remaining(model))
                self.model_stops[model.label] = {
                    "kind": "rejected",
                    "reason": classification.reason,
                    "message": message,
                    "request": sent,
                }
                capabilities = self.state["capabilities"].setdefault(model.label, [])
                entry = {"request_rejected": classification.reason, "request": sent}
                if entry not in capabilities:
                    capabilities.append(entry)
            elif classification.reason in MONEY_REASONS:
                message = stop_message(
                    model.kind, classification, self._provider_remaining(model.kind)
                )
                self.provider_stops[model.kind] = {
                    "kind": "money",
                    "reason": classification.reason,
                    "message": message,
                    "since": self.clock(),
                }
            else:
                message = stop_message(
                    model.kind, classification, self._provider_remaining(model.kind)
                )
                self._pause_for(model, classification, message)
            self._persist()
        self.say(message)

    def _check_budget(self, kind: str) -> None:
        budget = self.plan.manifest.budget_for(kind)
        if budget is None:
            return
        spent = self.spent(kind)
        if spent >= budget.usd_cap and not (budget.usd_cap == 0 and spent == 0):
            message = (
                f"{kind}: stopping cleanly (budget_cap_reached: ${spent:.4f} of "
                f"${budget.usd_cap:.2f}); {self._provider_remaining(kind)} call(s) left "
                f"pending; raise [budget.{kind}] usd_cap, then continue"
            )
            with self._lock:
                self.provider_stops[kind] = {
                    "kind": "budget_cap",
                    "reason": "budget_cap_reached",
                    "message": message,
                }
            self.say(message)

    def _check_truncation(self, model: Model) -> None:
        rules = self.plan.manifest.stops
        answers = self.answers.get(model.label, 0)
        cut = self.truncated.get(model.label, 0)
        if answers < rules.min_answers or cut / answers < rules.max_truncated_share:
            return
        if self.model_stops.get(model.label, {}).get("kind") == "truncation":
            return
        message = (
            f"{model.label}: truncation stop ({cut} of {answers} answers cut at "
            f"max_output_tokens={model.knobs['max_output_tokens']}, "
            f"{self.invalid.get(model.label, 0)} invalid); "
            f"{self._model_remaining(model)} call(s) left pending; {TRUNCATION_DECISION}"
        )
        with self._lock:
            self.model_stops[model.label] = {
                "kind": "truncation",
                "reason": "truncated",
                "message": message,
                "truncated": cut,
                "answers": answers,
                "invalid": self.invalid.get(model.label, 0),
            }
        self.say(message)

    # -- recording ---------------------------------------------------------

    def record(self, model: Model, key: str, result: CallResult) -> None:
        """One answer: billed, then into the ledger, then its reservation released."""
        with self._lock:
            entry = self.ledger.entry(key)
            if entry.state != PENDING:
                self._release(key)
                return
            if result.outcome is Outcome.PENDING:
                self._release(key)
                self.apply_stop(model, _classification_of(result), entry.spec.get("params") or {})
                return
            if key not in self._billed_keys:
                # Billed before the cache: a replayed answer is never billed twice.
                self._bill_answer(model, key, result)
            cached = CachedResponse(
                raw=result.raw,
                model_id=result.returned_model or model.ref.model,
                response_id=result.response_id,
                usage=dict(result.usage),
            )
            if result.outcome is Outcome.OK:
                self.ledger.record_done(key, cached)
            else:
                self.ledger.mark_invalid(key, result.reason or "invalid", cached)
            self.state["reserved"].pop(key, None)
            self._count(model, key, entry.spec, self.ledger.entry(key).state)
            self._persist()
            self._check_budget(model.kind)
            self._check_truncation(model)

    # -- sync --------------------------------------------------------------

    def run_sync(self, work: list[tuple[Model, WorkItem]]) -> bool:
        """Send *work*, each provider in its own pool under its concurrency cap.

        A worker holds its provider's semaphore through the claim, the send,
        the billing and the ledger write, so with a cap of N at most N calls
        are in flight when a stop lands, and none starts after it.
        """
        if not work:
            return False
        caps = self._concurrency_caps(work)
        semaphores = {kind: threading.Semaphore(cap) for kind, cap in caps.items()}
        deadline = time.monotonic() + ROUND_SECONDS

        def task(model: Model, item: WorkItem) -> bool:
            with semaphores[model.kind]:
                if time.monotonic() > deadline or self.blocked(model):
                    return False  # stays pending: the next round (or pass) sends it
                return self._send_one(model, item)

        pools = {
            kind: concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, cap), thread_name_prefix=f"sync-{kind}"
            )
            for kind, cap in caps.items()
        }
        futures = [pools[model.kind].submit(task, model, item) for model, item in work]
        try:
            progress, failure = _drain(futures)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            raise failure
        return progress

    def _concurrency_caps(self, work: list[tuple[Model, WorkItem]]) -> dict[str, int]:
        """Each provider's concurrency cap, in first-seen order (1 without a budget)."""
        caps: dict[str, int] = {}
        for model, _item in work:
            if model.kind not in caps:
                budget = self.plan.manifest.budget_for(model.kind)
                caps[model.kind] = budget.concurrency_cap if budget else 1
        return caps

    def _send_one(self, model: Model, item: WorkItem) -> bool:
        """Claim, send and record one call; ``False`` when it stays pending."""
        if not self.claim(model, item.key, item.request):
            return False
        self._note_host(model)
        try:
            result = model.provider.submit_sync(item.request)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 -- classified or re-raised
            classification = classify_exception(exc, model.kind)
            if classification is None:
                raise
            self._send_failed(model, item, exc, classification)
            return False
        self.record(model, item.key, result)
        return True

    def _send_failed(
        self, model: Model, item: WorkItem, exc: Exception, classification: Classification
    ) -> None:
        """A failed send: uncertain if it may have reached the provider, else released."""
        if classification.reason in UNCERTAIN_REASONS and not refused_before_send(exc):
            self._uncertain(item.key, f"sent but not answered ({classification.reason})")
        else:
            self._release(item.key)
        self.apply_stop(model, classification, item.request.params)

    # -- work --------------------------------------------------------------

    def _fresh(self, key: str) -> bool:
        return self.ledger.entry(key).state == PENDING

    def _choice_calls(self, model: Model, subject: Subject) -> list[tuple[str, CallRequest]]:
        calls = [
            choice_call(model, case, self.plan.domain)
            for case in self.plan.cases[subject.case_set.name]
        ]
        specs: list[CallSpec] = [spec for spec, _request in calls]
        keys = self.ledger.register_many(specs)
        return [(key, request) for key, (_spec, request) in zip(keys, calls)]

    def subject_work(self) -> tuple[dict[str, list[WorkItem]], bool]:
        """Every reference call that can go out now, once per key; and whether all are final."""
        work: dict[str, list[WorkItem]] = {}
        seen: set[str] = set()
        final = True
        for subject in self.plan.references:
            model = self.plan.models[subject.model]
            for key, request in self._choice_calls(model, subject):
                if self.ledger.entry(key).state not in (DONE, INVALID):
                    final = False
                if key in seen or not self._fresh(key):
                    continue
                seen.add(key)
                work.setdefault(model.label, []).append(WorkItem(key, request))
        return work, final

    # -- records per subject -------------------------------------------------

    def choice_record(self, model: Model, case: Case, key: str, request: CallRequest) -> RawRecord:
        """A reference's answer as a raw record, in the saved predictions' vocabulary."""
        cached = self.ledger.cached(key)
        entry = self.ledger.entry(key)
        base = {"provider": model.provider.name, "model": model.ref.model, "interface": "choice"}

        def invalid(reason: str, result: CallResult | None = None) -> RawRecord:
            return RawRecord.from_provider_answer(
                returned_model=result.returned_model if result else None,
                outcome="invalid",
                candidates=_calibration_labels(result.candidates) if result else None,
                invalid_reason=reason,
                **base,
            )

        if cached is None:
            return invalid(entry.reason or "invalid")
        result = model.provider.result_from_raw(request, cached.raw)
        if _cut(model, cached.raw):
            # A reply cut at the output budget is never an answer, even when a
            # valid letter came before the cut.
            return invalid(TRUNCATED, result)
        classification, name = contract.parse_choice(result.answer, request.params["labels"])
        if result.outcome is not Outcome.OK or name is None:
            reason = result.reason if result.outcome is not Outcome.OK else classification.reason
            return invalid(reason, result)
        answered = {
            "returned_model": result.returned_model,
            "candidates": _calibration_labels(result.candidates),
            **base,
        }
        if name in ("explain", "escalate"):
            return RawRecord.from_provider_answer(outcome=name, **answered)
        operation = self.plan.domain.get(name)
        grounded = (
            ground_arguments(self.plan.domain, operation, case.text or "", self.plan.world)
            if operation is not None
            else "unknown operation"
        )
        if isinstance(grounded, str):
            return RawRecord.from_provider_answer(
                outcome="invalid", invalid_reason="not_grounded", **answered
            )
        return RawRecord.from_provider_answer(
            outcome="propose", operation=name, arguments=dict(grounded), **answered
        )

    def subject_traces(self) -> dict[str, list[Trace]]:
        """Every subject's traces, in case order; call only when final."""
        out: dict[str, list[Trace]] = {}
        for subject in self.plan.subjects:
            if subject.kind != "reference":
                out[subject.name] = list(self.plan.saved[subject.name].traces)
                continue
            model = self.plan.models[subject.model]
            cases = self.plan.cases[subject.case_set.name]
            traces = []
            for (key, request), case in zip(self._choice_calls(model, subject), cases):
                record = self.choice_record(model, case, key, request)
                traces.append(
                    Trace(
                        case_id=case.id,
                        split=case.split,
                        raw=record,
                        ground_truth=dict(case.expect),
                        subject=subject.name,
                    )
                )
            out[subject.name] = traces
        return out

    # -- the pass ------------------------------------------------------------

    def dispatch(self, work: Mapping[str, list[WorkItem]]) -> bool:
        sync: list[tuple[Model, WorkItem]] = []
        for label in sorted(work):
            model = self.plan.models[label]
            if not self.blocked(model):
                sync.extend((model, item) for item in work[label])
        return self.run_sync(interleave(sync))

    def run_pass(self) -> StepOutcome:
        final = False
        while True:
            work, final = self.subject_work()
            if final or not work or not self.dispatch(work):
                break
        self._persist()
        if final:
            if self.plan.smoke:
                self.write_smoke()
            else:
                self.finalize()
            return StepOutcome(STATUS_COMPLETE, list(self.messages), exit_code=EXIT_OK)
        money = {k for k, v in self.provider_stops.items() if v.get("kind") == "money"}
        if not self.messages:
            self.say("no call could be sent; see status")
        return StepOutcome(STATUS_STOPPED, list(self.messages), money, EXIT_STOPPED)

    # -- outputs -------------------------------------------------------------

    def finalize(self) -> None:
        """Traces, per-(subject, policy) metrics, result.json and the page."""
        from . import policies, report

        run_dir = self.run_dir
        loaded_policies = {
            name: policies.load_policy(policy_path(name, self.env))
            for subject in self.plan.subjects
            for name in subject.policies
        }
        subjects_doc = []
        for subject, traces in self.subject_traces().items():
            spec = next(s for s in self.plan.subjects if s.name == subject)
            final_traces = [
                self._with_policies(trace, spec.policies, loaded_policies) for trace in traces
            ]
            write_traces(report.traces_path(run_dir, subject), final_traces)
            for policy in spec.policies:
                figures = self._policy_figures(
                    subject, policy, final_traces, loaded_policies[policy]
                )
                runstate.write_json_durable(report.metrics_path(run_dir, subject, policy), figures)
            subjects_doc.append(self._subject_doc(subject, spec))
        runstate.write_json_durable(
            run_dir / report.MANIFEST_FILENAME,
            {
                "run_id": self.state["run_id"],
                "date": self.state["date"],
                "deepeval": self.use_deepeval,
                "subjects": subjects_doc,
            },
        )
        report.generate(
            run_dir, result_path=run_dir / RESULT_FILE, markdown_path=run_dir / PAGE_FILE
        )
        self.state["status"] = STATUS_COMPLETE
        self._persist()
        self.say(f"complete: {run_dir / RESULT_FILE} and {run_dir / PAGE_FILE}")

    def _with_policies(
        self, trace: Trace, names: tuple[str, ...], loaded: Mapping[str, Any]
    ) -> Trace:
        """*trace* with each named policy's decision applied, in order."""
        from . import policies

        truncated = trace.raw.outcome == "invalid" and trace.raw.invalid_reason == TRUNCATED
        for policy in names:
            if truncated:
                # A cut reply stays invalid under every policy.
                trace = trace.with_policy(policy, "invalid", TRUNCATED)
                continue
            decision, reason, _n, _v = policies.apply(
                loaded[policy],
                trace.raw.to_dict(),
                trace.raw.offered_order(),
                self.plan.domain,
            )
            trace = trace.with_policy(policy, decision, reason or "")
        return trace

    def _policy_figures(
        self, subject: str, policy: str, traces: list[Trace], policy_json: Any
    ) -> Any:
        """One (subject, policy)'s corpus metrics, through the DeepEval layer when it runs."""
        from . import deepeval_layer

        domain = self.plan.domain
        if not self.use_deepeval:
            return deepeval_layer.corpus_metrics(traces, policy_json, domain)
        folder = self.run_dir / DEEPEVAL_DIR / f"{subject}__{policy}"
        folder.mkdir(parents=True, exist_ok=True)
        # deepeval prints its own summary; keep stdout for the runner's lines.
        log = folder / "deepeval.log"
        with open(log, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            outcome = deepeval_layer.evaluate_traces(
                traces, policy_json, domain, results_folder=folder
            )
        return outcome.corpus_metrics

    def _subject_doc(self, subject: str, spec: Subject) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "name": subject,
            "kind": spec.kind,
            "policies": list(spec.policies),
        }
        if spec.kind != "reference":
            doc["artifact"] = self.plan.saved[subject].artifact
        return doc

    def write_smoke(self) -> None:
        """Per-model tokens, cost, truncation and a projected full-run cost (smoke run)."""
        smoke_ids = {case.id for cases in self.plan.cases.values() for case in cases}
        smoke_count = len(smoke_ids)
        scale = self.plan.full_case_count / smoke_count if smoke_count else 0.0
        rows = self._smoke_rows(smoke_ids)
        for label, row in rows.items():
            self._finish_smoke_row(label, row, scale)
        smoke = {
            "cases": smoke_count,
            "full_run_cases": self.plan.full_case_count,
            "models": dict(sorted(rows.items())),
            "providers": dict(sorted(_provider_totals(rows).items())),
        }
        runstate.write_json_durable(self.run_dir / SMOKE_FILE, smoke)
        self.state["smoke"] = smoke
        self.state["status"] = STATUS_COMPLETE
        self._persist()
        for label, row in sorted(rows.items()):
            self.out(_smoke_line(label, row))

    def _smoke_rows(self, smoke_ids: set[str]) -> dict[str, dict[str, Any]]:
        """Per-model call, token, cost and truncation counts over the final smoke calls."""
        billed = {
            e["key"]: e for e in self.billing.entries() if e.get("kind") == runstate.BILL_ANSWER
        }
        rows: dict[str, dict[str, Any]] = {}
        for entry in self.ledger.entries():
            model = self._smoke_model(entry, smoke_ids)
            if model is None:
                continue
            row = rows.setdefault(model.label, _new_smoke_row(model.kind))
            self._add_smoke_call(row, model, entry, billed)
        return rows

    def _smoke_model(self, entry: Entry, smoke_ids: set[str]) -> Model | None:
        """*entry*'s model when it is a final smoke call made with current settings."""
        model = self._model_of_spec(entry.spec)
        if model is None or entry.spec.get("case_id") not in smoke_ids:
            return None
        if entry.state not in (DONE, INVALID) or not self._current(model, entry.spec):
            return None
        return model

    def _add_smoke_call(
        self, row: dict[str, Any], model: Model, entry: Entry, billed: Mapping[str, Any]
    ) -> None:
        cached = self.ledger.cached(entry.key)
        row["calls"] += 1
        row["invalid"] += entry.state == INVALID
        usage = cached.usage if cached else {}
        row["input_tokens"] += tokens_in(usage)
        row["output_tokens"] += tokens_out(usage)
        row["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
        bill = billed.get(entry.key)
        row["cost_usd"] += bill["cost_usd"] if bill else model.cost(usage)
        cut = entry.reason == TRUNCATED
        if cached is not None and not cut:
            cut = _cut(model, cached.raw)
        row["truncated"] += bool(cut)

    def _finish_smoke_row(self, label: str, row: dict[str, Any], scale: float) -> None:
        model = self.plan.models[label]
        row["cost_usd"] = round(row["cost_usd"], 6)
        row["projected_full_run_usd"] = round(row["cost_usd"] * scale, 4)
        row["max_output_tokens"] = model.knobs["max_output_tokens"]
        row["reasoning"] = model.ref.reasoning
        row["flag"] = "CAPPED" if row["truncated"] else "OK"
        stop = self.model_stops.get(label)
        if stop:
            row["stop"] = stop["kind"]


def _new_smoke_row(provider: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "calls": 0,
        "invalid": 0,
        "truncated": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }


def _provider_totals(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """The smoke rows' cost and projected full-run cost summed per provider."""
    providers: dict[str, dict[str, float]] = {}
    for row in rows.values():
        total = providers.setdefault(
            row["provider"], {"cost_usd": 0.0, "projected_full_run_usd": 0.0}
        )
        total["cost_usd"] = round(total["cost_usd"] + row["cost_usd"], 6)
        total["projected_full_run_usd"] = round(
            total["projected_full_run_usd"] + row["projected_full_run_usd"], 4
        )
    return providers


def _smoke_line(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"smoke {label}: {row['flag']} calls={row['calls']} truncated={row['truncated']} "
        f"invalid={row['invalid']} tokens in/out/reasoning={row['input_tokens']}/"
        f"{row['output_tokens']}/{row['reasoning_tokens']} cost=${row['cost_usd']:.4f} "
        f"projected full run=${row['projected_full_run_usd']:.2f}"
    )


def _calibration_labels(candidates: Mapping[str, float] | None) -> dict[str, float] | None:
    """A reference's distribution keyed as a predictions line keys it ('(explain)', ...)."""
    if candidates is None:
        return None
    return {Domain.calibration_label(name): value for name, value in candidates.items()}


def interleave(work: list[tuple[Model, WorkItem]]) -> list[tuple[Model, WorkItem]]:
    """*work* reordered one call per model in turn (first-seen model order kept).

    Listed model by model, one model's hundreds of calls would hold every
    slot of its provider's pool for a whole round.
    """
    queues: dict[str, list[tuple[Model, WorkItem]]] = {}
    for pair in work:
        queues.setdefault(pair[0].label, []).append(pair)
    out: list[tuple[Model, WorkItem]] = []
    for turn in range(max((len(q) for q in queues.values()), default=0)):
        out.extend(q[turn] for q in queues.values() if turn < len(q))
    return out


def _drain(futures: list[concurrent.futures.Future]) -> tuple[bool, BaseException | None]:
    """Wait for every future: whether any made progress, and the first failure.

    The first failure cancels every future that has not started; the rest
    still run to the end, so the caller re-raises only after the drain.
    """
    progress = False
    failure: BaseException | None = None
    for future in concurrent.futures.as_completed(futures):
        if future.cancelled():
            continue
        try:
            progress |= future.result()
        except BaseException as exc:  # noqa: BLE001 -- re-raised after the drain
            if failure is None:
                failure = exc
                _cancel(futures)
    return progress, failure


def _cancel(futures: list[concurrent.futures.Future]) -> None:
    for other in futures:
        other.cancel()


def _cut(model: Model, raw: bytes) -> bool:
    try:
        return model.provider.reply_text(raw).truncated
    except Exception:  # noqa: BLE001 -- an unreadable reply is not a cut one
        return False


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> Manifest:
    try:
        return load_manifest(manifest_path)
    except (OSError, ValueError) as exc:  # ManifestError is a ValueError
        raise RunError(f"manifest {manifest_path}: {exc}") from exc


def step(
    run_dir: Path,
    manifest_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    factory: ProviderFactory | None = None,
    retry_rejected: bool = False,
    out: Callable[[str], None] = print,
    clock: Callable[[], float] = time.time,
    use_deepeval: bool = True,
    _start: Mapping[str, Any] | None = None,
) -> StepOutcome:
    """One pass over *run_dir* (``continue``), holding the run lock from the first read
    of ``run.json`` to its last write. A smoke run stays a smoke run however it is resumed.
    """
    env = os.environ if env is None else env
    run_dir = Path(run_dir)
    _check_run_dir(run_dir)
    if _start is None and not (run_dir / RUN_FILE).exists():
        raise RunError(f"{run_dir} holds no run; start one with `run` (or `smoke`)")
    digest = _sha256_file(manifest_path)
    manifest = _load_manifest(manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _open_ledger(run_dir) as ledger:
        state = runstate.load_state(run_dir)
        if _start is not None:
            state = _begin(state, manifest, env, digest, _start)
        elif not state:
            raise RunError(f"{run_dir} holds no run; start one with `run` (or `smoke`)")
        scope = state.get("scope") or {"mode": "full"}
        _check_deepeval(use_deepeval, scope)
        if digest != state.get("manifest_sha256"):
            state["manifest_sha256"] = digest
            state.setdefault("manifest_history", []).append(digest)
            out(f"manifest changed since the last pass (now {digest[:12]})")
        plan = build_plan(manifest, env, factory or default_factory, scope=scope)
        if _start is not None and scope.get("mode") == "full":
            state["started_full"] = True
        state["mode"] = scope.get("mode", "full")
        runstate.save_state(run_dir, state)
        runner = Runner(
            run_dir,
            plan,
            ledger,
            state,
            retry_rejected=retry_rejected,
            out=out,
            env=env,
            clock=clock,
            use_deepeval=use_deepeval,
        )
        try:
            outcome = runner.run_pass()
        finally:
            runner._persist()
        state["status"] = outcome.status
        state["messages"] = outcome.messages
        runstate.save_state(run_dir, state)
    return outcome


def _begin(
    state: dict, manifest: Manifest, env: Mapping[str, str], digest: str, start: Mapping
) -> dict:
    """``run`` / ``smoke`` on *state* (under the lock): create it, or check the scope."""
    smoke_cases = start.get("smoke_cases")
    if not state:
        scope = smoke_scope(manifest, env, smoke_cases) if smoke_cases else {"mode": "full"}
        return _new_state(digest, start.get("run_id"), start.get("date"), scope)
    scope = state.get("scope") or {"mode": "full"}
    if smoke_cases:
        if scope.get("mode") != "smoke":
            raise RunError("this run dir holds a full run; a smoke run needs its own run dir")
        if len(scope.get("case_ids", [])) != smoke_cases:
            raise RunError(
                f"this run dir holds a {len(scope.get('case_ids', []))}-case smoke run; "
                "use the same --cases, or a new run dir"
            )
        return state
    if scope.get("mode") == "smoke":
        if not start.get("expand"):
            raise RunError(
                "this run dir holds a smoke run; expand it to the full run with "
                "`run --expand`, or start the full run in a new run dir"
            )
        state["scope"] = {"mode": "full"}
        state["expanded_from"] = scope
        return state
    if state.get("started_full"):
        raise RunError("this run dir already holds a run; use `continue`")
    return state


def start(
    run_dir: Path,
    manifest_path: Path,
    *,
    run_id: str | None = None,
    date: str | None = None,
    smoke_cases: int | None = None,
    expand: bool = False,
    **kwargs: Any,
) -> StepOutcome:
    """``run`` (and ``smoke``): initialize or check *run_dir*, then take a pass.

    Nothing is written until the plan has been built, so a configuration
    error can be fixed and ``run`` repeated.
    """
    start_args = {"run_id": run_id, "date": date, "smoke_cases": smoke_cases, "expand": expand}
    return step(Path(run_dir), Path(manifest_path), _start=start_args, **kwargs)
