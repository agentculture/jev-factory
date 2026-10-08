"""Plan-building for the release-gate runner: errors, private paths, providers,
subjects, and the call each reference subject makes.

``run.py`` re-exports every public name here, so ``jev_factory.evals.run``
stays the one import the CLI and tests use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.domain.model import Domain
from jev_factory.domain.validate import DomainError, load_domain

from . import request as contract
from .cases import Case, TrainingOverlapError, load_case_set, training_overlap
from .ledger import CallSpec, canonical_json, prompt_hash
from .manifest import CaseSet, Manifest, Reference, RunEntry
from .providers import openai_compat
from .providers.base import (
    CallRequest,
    CallResult,
    HeldoutSplitRefused,
    MissingProviderKey,
    Provider,
    ProviderCapabilities,
)
from .providers.errors import Classification, Outcome, classify_transport
from .trace import PredictionError, Trace

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/runplan.py",
    "commit": "9debdc6",
    "adaptations": [
        "line 21: nvsh.tiers.bench (Track A platform) and the track_a_loop import are gone;"
        " the Domain is loaded from the manifest (jev_factory.domain.validate.load_domain) and"
        " the optional world snapshot replaces the Track A ground snapshot",
        "references answer through the choice interface only: one subject per (reference,"
        " case set); Track A subjects, judges and batch routing are deferred (c14)",
        "default_factory builds the OpenAI-compatible adapter only (OpenAI/Anthropic adapters"
        " not ported); classify_exception keeps the adapter's own classification first",
        "Model.rate/estimate drop the batch discount; the worst-case reservation (prompt size"
        " plus the whole output budget) is unchanged",
        "policies may be a builtin name or a .json path under the private root",
        "env vars NVSH_EVALS_* -> JEV_EVALS_*",
        "d17 (2026-10-08): restructured for SonarCloud code quality (cognitive "
        "complexity split into private helpers, plus lint-level cleanups); behaviour "
        "unchanged, pinned by tests/test_complexity_*.py and differential checks against "
        "the imported version; fixed: a saved prediction row with a non-string id is "
        "skipped, not a TypeError",
    ],
    "licence": "Apache-2.0",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_USER = 1
EXIT_ENV = 2
EXIT_ASK = 3
EXIT_STOPPED = 4
EXIT_INTERRUPTED = 130

STATUS_COMPLETE = "complete"
STATUS_STOPPED = "stopped"

#: The private data root case-set paths, the world snapshot and relative
#: prediction paths are joined onto.
ENV_PRIVATE_ROOT = "JEV_EVALS_PRIVATE_ROOT"
#: What a manifest's ``${JEV_EVALS_PRIVATE}`` placeholder expands to
#: (falls back to ``JEV_EVALS_PRIVATE_ROOT``).
ENV_PRIVATE = "JEV_EVALS_PRIVATE"
PLACEHOLDER = "${JEV_EVALS_PRIVATE}"
#: Optional per-kind base URL override for an OpenAI-compatible host, e.g.
#: ``JEV_EVALS_BASE_URL_LOCAL``; validated by the adapter.
ENV_BASE_URL = "JEV_EVALS_BASE_URL_{kind}"

RESULT_FILE = "result.json"
PAGE_FILE = "report.md"
DEEPEVAL_DIR = "deepeval"

SUBJECT_ROLE = "subject"
CHOICE_TARGET = "choice"

#: Reasons that make a stop a *money* stop (one provider).
MONEY_REASONS = frozenset({"insufficient_credit", "budget_cap_reached"})

#: Uncertain sync attempts (sent, maybe billed, no answer) per key before the
#: model stops for an operator decision.
MAX_UNCERTAIN_ATTEMPTS = 2
#: Classification reasons that mean a sync request may have been accepted.
UNCERTAIN_REASONS = frozenset({"timeout", "network_loss", "machine_reset"})

#: The truncation stop's operator question.
TRUNCATION_DECISION = (
    "needs operator decision: raise max_output_tokens, lower reasoning, or replace the model"
)

#: How each provider is asked for ``reasoning = "none"``; ``None`` omits the parameter.
NONE_REASONING: dict[str, str | None] = {
    # OpenRouter's unified reasoning.effort accepts "none" to turn reasoning off.
    "openrouter": "none",
    # build.nvidia.com and local servers: sent only when the manifest lists the
    # "reasoning" capability; "none" omits it.
    "nvidia": None,
    "local": None,
}


class RunError(Exception):
    """A configuration or input problem the operator must fix (exit 1)."""


class EnvError(RunError):
    """The environment lacks something the run needs (exit 2)."""


class StopAndAsk(Exception):
    """The run cannot proceed safely without the operator (exit 3)."""


def provider_reasoning(provider: str, level: str) -> str | None:
    """The reasoning value sent to *provider* for manifest level *level* (``None`` = omit)."""
    if level != "none":
        return level
    return NONE_REASONING.get(provider)


# ---------------------------------------------------------------------------
# Private paths
# ---------------------------------------------------------------------------


def private_root(env: Mapping[str, str]) -> Path:
    raw = env.get(ENV_PRIVATE_ROOT) or env.get(ENV_PRIVATE)
    if not raw:
        raise EnvError(
            f"{ENV_PRIVATE_ROOT} is not set: case sets, the world snapshot and saved "
            "predictions live in the operator's private data root, never in the repository"
        )
    return Path(raw)


def resolve_private(value: str, env: Mapping[str, str]) -> Path:
    """A manifest path: ``${JEV_EVALS_PRIVATE}`` expanded, relative ones under the root."""
    if PLACEHOLDER in value:
        base = env.get(ENV_PRIVATE) or env.get(ENV_PRIVATE_ROOT)
        if not base:
            raise EnvError(f"{value!r} needs {ENV_PRIVATE} (or {ENV_PRIVATE_ROOT}) to be set")
        value = value.replace(PLACEHOLDER, base)
    if "${" in value:
        raise RunError(f"{value!r} carries a placeholder this runner does not expand")
    path = Path(value)
    return path if path.is_absolute() else private_root(env) / path


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise RunError(f"cannot read {path}: {exc}") from exc


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


# ---------------------------------------------------------------------------
# Domain, world and policies
# ---------------------------------------------------------------------------


def load_run_domain(manifest: Manifest, env: Mapping[str, str]) -> Domain:
    """The manifest's Domain: a dotted module, or a ``.json`` path under the private root."""
    source: Any = manifest.domain
    if source.endswith(".json"):
        source = resolve_private(source, env)
    try:
        return load_domain(source)
    except DomainError as exc:
        raise RunError(f"domain {manifest.domain!r}: {exc}") from exc


def load_world(manifest: Manifest, domain: Domain, env: Mapping[str, str]) -> dict | None:
    """The manifest's world snapshot (checked against the Domain's world schema), or ``None``."""
    if manifest.world is None:
        return None
    path = resolve_private(manifest.world, env)
    try:
        world = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"world snapshot {path}: {exc}") from exc
    problems = domain.check_world(world)
    if problems:
        raise RunError(f"world snapshot {path}: " + "; ".join(problems))
    return world


def policy_path(name: str, env: Mapping[str, str]) -> Path:
    """A builtin policy's file, or a ``.json`` policy path under the private root."""
    from . import policies

    if name.endswith(".json"):
        return resolve_private(name, env)
    return policies.builtin_policy_path(name)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

#: ``factory(reference, budget, env) -> Provider``; tests inject FakeProviders.
ProviderFactory = Callable[..., Provider]


def default_factory(ref: Reference, budget: Any, env: Mapping[str, str]) -> Provider:
    """The real adapter for *ref* (its key is read from the environment at call time)."""
    capabilities = ProviderCapabilities(
        logprobs="logprobs" in ref.capabilities,
        batch=False,
        reasoning="reasoning" in ref.capabilities,
    )
    limiter = None
    if budget is not None and budget.requests_per_minute:
        limiter = shared_limiter(ref.provider, budget.requests_per_minute)
    kwargs: dict[str, Any] = {}
    if ref.api_key_env:
        kwargs["api_key_env"] = ref.api_key_env
    if budget is not None:
        kwargs["timeout_seconds"] = budget.timeout_seconds
    return openai_compat.OpenAICompatProvider(
        ref.provider,
        ref.model,
        base_url=env.get(ENV_BASE_URL.format(kind=ref.provider.upper())) or None,
        capabilities=capabilities,
        rate_limiter=limiter,
        **kwargs,
    )


#: One synchronized limiter per (provider account, rate), shared by every model
#: of that provider and kept for the life of the process.
_LIMITERS: dict[tuple[str, float], openai_compat.RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_limiter(provider: str, requests_per_minute: float) -> openai_compat.RateLimiter:
    with _LIMITERS_LOCK:
        key = (provider, float(requests_per_minute))
        if key not in _LIMITERS:
            _LIMITERS[key] = openai_compat.RateLimiter(requests_per_minute)
        return _LIMITERS[key]


def provider_host(provider: Provider) -> str:
    """The network host *provider* sends case text to."""
    host = getattr(provider, "host", None)
    if isinstance(host, str) and host:
        return host
    return provider.name


def classify_exception(exc: BaseException, provider: str) -> Classification | None:
    """The stop an adapter exception stands for.

    An adapter's own classification wins. An HTTP error carrying its body is
    read the way the adapter reads a sync error. A transport failure is
    network loss. Anything else pauses that provider as
    ``unexpected_error:<type>`` so one provider's surprise never stops the
    others, except the held-out guard and test assertions, which are
    re-raised (``None``).
    """
    if isinstance(exc, (HeldoutSplitRefused, AssertionError)):
        return None
    found = getattr(exc, "classification", None)
    if isinstance(found, Classification):
        return found
    if isinstance(exc, MissingProviderKey):
        return Classification(Outcome.PENDING, "missing_key", stop=True, retryable=True)
    status = getattr(exc, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        body = getattr(exc, "body", None)
        error_type = openai_compat._error_type_from_body(body) if isinstance(body, bytes) else None
        return classify_transport("openai_compat", status_code=status, error_type=error_type)
    if isinstance(exc, (OSError, openai_compat.TransportError)):
        return classify_transport("openai_compat", error_type="network_error")
    return Classification(
        Outcome.PENDING, f"unexpected_error:{type(exc).__name__}", stop=True, retryable=True
    )


def _classification_of(result: CallResult) -> Classification:
    """A ``pending`` result as the stop it stands for."""
    reason = result.reason or "unrecognized_infra_condition"
    rejected = reason.startswith("request_rejected:")
    return Classification(Outcome.PENDING, reason, stop=True, retryable=not rejected)


@dataclass
class Model:
    """One reference model: its adapter and where it sends."""

    ref: Reference
    provider: Provider
    host: str

    @property
    def label(self) -> str:
        return f"{self.ref.provider}/{self.ref.model}"

    @property
    def kind(self) -> str:
        return self.ref.provider

    @property
    def knobs(self) -> dict[str, Any]:
        """Reasoning and output budget: sent with, and keyed into, every call."""
        out: dict[str, Any] = {
            "max_output_tokens": self.ref.max_output_tokens or contract.DEFAULT_MAX_OUTPUT_TOKENS
        }
        effort = provider_reasoning(self.ref.provider, self.ref.reasoning)
        if effort:
            out["reasoning"] = effort
        return out

    def rate(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.ref.usd_per_mtok_in + tokens_out * self.ref.usd_per_mtok_out) / 1e6

    def cost(self, usage: Mapping[str, int]) -> float:
        """What an answer cost, from its reported usage at the current prices."""
        return self.rate(tokens_in(usage), tokens_out(usage))

    def estimate(self, request: CallRequest) -> float:
        """A call's worst-case cost before it is sent: prompt size plus its whole output budget.

        Conservative on purpose (about three characters per token, every
        output token used): the reservation must not undershoot the bill.
        """
        system, messages, labels = contract.canonical_content(request)
        size = len(json.dumps([system, messages, labels], ensure_ascii=False))
        budget = int(request.params.get("max_output_tokens") or contract.DEFAULT_MAX_OUTPUT_TOKENS)
        return self.rate(size // 3 + 1, budget)


def tokens_in(usage: Mapping[str, int]) -> int:
    return int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))


def tokens_out(usage: Mapping[str, int]) -> int:
    return int(usage.get("output_tokens", usage.get("completion_tokens", 0)))


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """One report subject: a checkpoint or a reference model, on one case set."""

    name: str
    kind: str  # candidate | baseline | reference
    case_set: CaseSet
    policies: tuple[str, ...]
    entry: RunEntry | None = None
    model: str | None = None  # Model.label for a reference


@dataclass
class Saved:
    """A candidate/baseline's saved predictions for one case set."""

    traces: list[Trace]
    artifact: dict[str, Any]


@dataclass
class Plan:
    manifest: Manifest
    domain: Domain
    world: Mapping[str, Any] | None
    cases: dict[str, tuple[Case, ...]]
    subjects: list[Subject]
    saved: dict[str, Saved]
    models: dict[str, Model]
    smoke: bool = False
    full_case_count: int = 0
    scope: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {"mode": "full"})

    @property
    def references(self) -> list[Subject]:
        return [s for s in self.subjects if s.kind == "reference"]


def subject_name(base: str, case_set: str) -> str:
    return f"{_safe(base)}.{_safe(case_set)}"


def _load_cases(cs: CaseSet, domain: Domain, env: Mapping[str, str]) -> tuple[Case, ...]:
    path = private_root(env) / cs.path
    try:
        cases = load_case_set(
            {cs.split: str(path)}, cs.split, domain=domain, include_heldout=cs.include_heldout
        )
    except (OSError, ValueError, KeyError) as exc:
        raise RunError(f"case set {cs.name!r}: cannot load {path}: {exc}") from exc
    if len(cases) != cs.count:
        raise RunError(
            f"case set {cs.name!r}: the manifest says {cs.count} cases, the file has {len(cases)}"
        )
    if len({c.id for c in cases}) != len(cases):
        raise RunError(f"case set {cs.name!r}: case ids are not unique")
    return cases


def _load_saved(
    entry: RunEntry, cs: CaseSet, cases: Sequence[Case], name: str, env: Mapping[str, str]
) -> Saved | None:
    """The saved predictions of *entry* on *cs*, in case order; ``None`` when it has none."""
    path = resolve_private(entry.predictions_for(cs.name), env)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RunError(f"{entry.name}: cannot read saved predictions {path}: {exc}") from exc
    wanted = {c.id for c in cases}
    rows = _saved_rows(entry, path, data, wanted)
    if not rows:
        return None
    missing = [c.id for c in cases if c.id not in rows]
    if missing:
        raise RunError(
            f"{entry.name}: saved predictions for case set {cs.name!r} miss {len(missing)} of "
            f"{len(cases)} cases; a partial replay would bias every figure"
        )
    if entry.train_split:
        _check_training_overlap(entry, wanted, env)
    traces = [_saved_trace(entry, rows[case.id], case, cs.split, name) for case in cases]
    return Saved(traces=traces, artifact=_saved_artifact(entry, data))


def _saved_rows(entry: RunEntry, path: Path, data: bytes, wanted: set[str]) -> dict[str, dict]:
    """The prediction lines of *data* whose case id is in *wanted*, each at most once."""
    rows: dict[str, dict] = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RunError(f"{entry.name}: {path} line {number} is not JSON") from exc
        case_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or case_id not in wanted:
            continue
        if case_id in rows:
            raise RunError(f"{entry.name}: {path} has case {case_id!r} twice")
        rows[case_id] = row
    return rows


def _check_training_overlap(entry: RunEntry, wanted: set[str], env: Mapping[str, str]) -> None:
    try:
        training_overlap(wanted, resolve_private(entry.train_split, env))
    except TrainingOverlapError as exc:
        raise RunError(f"{entry.name}: {exc}") from exc
    except OSError as exc:
        raise RunError(f"{entry.name}: cannot read its train split: {exc}") from exc


def _saved_trace(entry: RunEntry, row: dict, case: Case, split: str, name: str) -> Trace:
    try:
        return Trace.from_prediction_line(row, split=split, subject=name)
    except PredictionError as exc:
        raise RunError(f"{entry.name}: case {case.id!r}: {exc}") from exc


def _saved_artifact(entry: RunEntry, data: bytes) -> dict[str, str]:
    artifact = {"predictions_sha256": hashlib.sha256(data).hexdigest()}
    if entry.repo_id:
        artifact["repo_id"] = entry.repo_id
    if entry.revision:
        artifact["revision"] = entry.revision
    return artifact


def build_plan(
    manifest: Manifest,
    env: Mapping[str, str],
    factory: ProviderFactory,
    *,
    scope: Mapping[str, Any] | None = None,
) -> Plan:
    """Everything one pass needs, from the manifest and the private data root.

    *scope* is the run's persisted scope: ``{"mode": "full"}`` or a smoke
    selection ``{"mode": "smoke", "case_set": name, "case_ids": [...]}``. A
    smoke run never sends a case outside its selection, however it is resumed.
    """
    scope = dict(scope or {"mode": "full"})
    domain = load_run_domain(manifest, env)
    world = load_world(manifest, domain, env)
    cases: dict[str, tuple[Case, ...]] = {}
    for cs in manifest.case_sets:
        cases[cs.name] = _load_cases(cs, domain, env)
    sendable = list(manifest.sendable_case_sets())
    full_case_count = sum(len(cases[cs.name]) for cs in sendable)
    smoke = scope.get("mode") == "smoke"
    subjects: list[Subject] = []
    saved: dict[str, Saved] = {}
    if smoke:
        cases, sendable = _smoke_selection(scope, cases, sendable)
    else:
        subjects, saved = _saved_subjects(manifest, cases, env)
    models: dict[str, Model] = {}
    for ref in manifest.references:
        model = _reference_model(ref, manifest, factory, env)
        models[model.label] = model
        subjects.extend(_reference_subjects(model, sendable))
    names = [s.name for s in subjects]
    if len(set(names)) != len(names):
        raise RunError("two subjects map to the same file name; rename a checkpoint")
    return Plan(
        manifest=manifest,
        domain=domain,
        world=world,
        cases=cases,
        subjects=subjects,
        saved=saved,
        models=models,
        smoke=smoke,
        full_case_count=full_case_count,
        scope=scope,
    )


def _smoke_selection(
    scope: Mapping[str, Any], cases: dict[str, tuple[Case, ...]], sendable: list[CaseSet]
) -> tuple[dict[str, tuple[Case, ...]], list[CaseSet]]:
    """The smoke run's cases and its one sendable case set, from its persisted selection."""
    chosen = [cs for cs in sendable if cs.name == scope.get("case_set")]
    if not chosen:
        raise RunError(f"the smoke case set {scope.get('case_set')!r} is not in the manifest")
    wanted = list(scope.get("case_ids", []))
    by_id = {case.id: case for case in cases[chosen[0].name]}
    missing = [case_id for case_id in wanted if case_id not in by_id]
    if missing:
        raise RunError(f"smoke cases {missing} are no longer in {chosen[0].name!r}")
    return {chosen[0].name: tuple(by_id[case_id] for case_id in wanted)}, chosen


def _saved_subjects(
    manifest: Manifest, cases: Mapping[str, tuple[Case, ...]], env: Mapping[str, str]
) -> tuple[list[Subject], dict[str, Saved]]:
    """Every candidate and baseline subject with saved predictions, and those predictions."""
    subjects: list[Subject] = []
    saved: dict[str, Saved] = {}
    for kind, entries in (("candidate", manifest.candidates), ("baseline", manifest.baselines)):
        for entry in entries:
            _check_policies(entry, env)
            for cs in manifest.case_sets:
                name = subject_name(entry.name, cs.name)
                loaded = _load_saved(entry, cs, cases[cs.name], name, env)
                if loaded is None:
                    continue
                subjects.append(Subject(name, kind, cs, tuple(entry.policies), entry=entry))
                saved[name] = loaded
    return subjects, saved


def _check_policies(entry: RunEntry, env: Mapping[str, str]) -> None:
    for policy in entry.policies:
        if not policy_path(policy, env).is_file():
            raise RunError(f"{entry.name}: unknown policy {policy!r}")


def _reference_model(
    ref: Reference, manifest: Manifest, factory: ProviderFactory, env: Mapping[str, str]
) -> Model:
    budget = manifest.budget_for(ref.provider)
    if budget is None:
        raise RunError(f"{ref.provider}/{ref.model}: no [budget.{ref.provider}] table")
    provider = factory(ref, budget, env)
    if provider.capabilities.batch:
        raise RunError(f"{ref.provider}/{ref.model}: batch adapters are not supported")
    return Model(ref=ref, provider=provider, host=provider_host(provider))


def _reference_subjects(model: Model, sendable: Sequence[CaseSet]) -> list[Subject]:
    ref = model.ref
    return [
        Subject(
            subject_name(f"{ref.provider}.{ref.model}", cs.name),
            "reference",
            cs,
            ("raw",),
            model=model.label,
        )
        for cs in sendable
    ]


def smoke_scope(manifest: Manifest, env: Mapping[str, str], count: int) -> dict[str, Any]:
    """The smoke selection persisted at init: the first *count* cases of the first sendable set."""
    sendable = list(manifest.sendable_case_sets())
    if not sendable:
        raise RunError("smoke needs a sendable (non-held-out) case set")
    first = sendable[0]
    domain = load_run_domain(manifest, env)
    ids = [case.id for case in _load_cases(first, domain, env)[:count]]
    return {"mode": "smoke", "case_set": first.name, "case_ids": ids}


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def refused_before_send(exc: BaseException) -> bool:
    """True when the request certainly never reached the provider (refused, unknown host)."""
    import socket
    import urllib.error

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionRefusedError, socket.gaierror)):
            return True
        reason = getattr(current, "reason", None)
        if isinstance(current, urllib.error.URLError) and isinstance(
            reason, (ConnectionRefusedError, socket.gaierror)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def request_fingerprint(params: Mapping[str, Any]) -> str:
    """The request parameters a rejection is about."""
    kept = {k: params[k] for k in ("max_output_tokens", "reasoning") if k in params}
    return canonical_json(kept)


def choice_call(model: Model, case: Case, domain: Domain) -> tuple[CallSpec, CallRequest]:
    """One choice call for *case*, with the model's own knobs."""
    base = contract.build_choice_request(case, domain)
    request = dataclasses.replace(base, params={"labels": base.params["labels"], **model.knobs})
    content = canonical_json(
        {"system": request.prompt, "user": request.case_text, "labels": request.params["labels"]}
    )
    spec = CallSpec(
        provider=model.provider.name,
        model=model.ref.model,
        subject_role=SUBJECT_ROLE,
        case_id=case.id,
        target=CHOICE_TARGET,
        prompt_hash=prompt_hash(content),
        params=dict(model.knobs),
    )
    return spec, request


@dataclass
class WorkItem:
    key: str
    request: CallRequest
