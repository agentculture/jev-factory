"""Run manifest: the domain, candidates, baselines, references, case sets, budgets.

The manifest is the single, data-driven description of one release-gate run:
which domain the cases belong to, which checkpoints (candidates and baselines)
to score from their saved predictions, which reference models to ask the same
cases, which case sets to run, the per-provider budget caps and the
truncation-stop thresholds. Every list is an array of tables, so a new
checkpoint is one ``[[candidate]]`` table: a data change, never a code change.

The operator's real manifest lives outside the repository and is never read
from a hard-coded path: its location comes from ``JEV_EVALS_MANIFEST``
(:data:`ENV_MANIFEST_PATH`). Only environment variable *names* are stored
here (``api_key_env``), never secret values, and provider base URLs belong
in the adapters, not in the manifest.

Top-level keys::

    domain = "jev_factory.domains.jev_cli"   # dotted module with DOMAIN, or a
                                             # domain JSON path (private-root relative)
    world = "world.json"                     # optional world snapshot for grounding
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .cases import SPLIT_TAGS

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/manifest.py",
    "commit": "9debdc6",
    "adaptations": [
        "new top-level 'domain' (required) and 'world' (optional snapshot) keys: the Domain"
        " replaces nvsh.ops.table and the world snapshot replaces [track_a] snapshot",
        "[[judge]], [judging] and [track_a] are removed (judge panel and Track A deferred,"
        " c14); RunEntry.track and permutation_probes go with them",
        "batch routing is refused: a reference with batch = true and the budget's"
        " batch_discount are errors now (batch APIs deferred, c14)",
        "ALLOWED_PROVIDERS narrows to the ported adapters (openrouter, nvidia, local)",
        "NVSH_EVALS_MANIFEST -> JEV_EVALS_MANIFEST",
        "kept: [[candidate]]/[[baseline]] with per-case-set predictions and policies,"
        " [[reference]] roster uniqueness, [[case_set]] held-out consistency, [budget.*]"
        " usd_cap/concurrency_cap/requests_per_minute/timeout_seconds, [stops]",
    ],
    "licence": "Apache-2.0",
}

#: The environment variable that points at the operator's private manifest.
ENV_MANIFEST_PATH = "JEV_EVALS_MANIFEST"

#: Split tags whose case sets must never be sent to a provider.
_HELDOUT_SPLITS = frozenset({"heldout", "heldout-mc"})

#: Providers this schema knows (the ported adapters). Anything else fails loudly.
ALLOWED_PROVIDERS = frozenset({"openrouter", "nvidia", "local"})

#: Reasoning levels a ``[[reference]]`` may ask for; ``"none"`` is mapped per provider.
REASONING_LEVELS = ("none", "low", "medium", "high")


class ManifestError(ValueError):
    """A manifest file failed validation."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEntry:
    """One candidate or baseline checkpoint: a name plus its saved predictions.

    ``train_split`` is the checkpoint's own training-side split file, so a
    runner can refuse to score it on any case id it trained on.
    ``predictions`` maps a case-set name to that set's own predictions file
    (overriding ``predictions_path``). ``policies`` are the harness policies
    scored for it: ``"raw"`` is model-only, any other is model+harness (a
    builtin policy name or a ``.json`` path under the private root).
    """

    name: str
    predictions_path: str
    train_split: str | None = None
    predictions: Mapping[str, str] = field(default_factory=dict)
    policies: tuple[str, ...] = ("raw",)
    repo_id: str | None = None
    revision: str | None = None

    def predictions_for(self, case_set: str) -> str:
        """The saved predictions file for *case_set* (the override, else the default)."""
        return self.predictions.get(case_set, self.predictions_path)


@dataclass(frozen=True)
class Reference:
    """One reference model in the roster; ``(provider, model)`` is its unique key."""

    provider: str
    model: str
    reasoning: str = "medium"
    capabilities: tuple[str, ...] = ()
    api_key_env: str | None = None
    quantization: str | None = None
    #: Output budget per call; ``None`` = the request contract's default.
    max_output_tokens: int | None = None
    #: Price per million input / output tokens, for spend tracking and reservations.
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.model)


@dataclass(frozen=True)
class CaseSet:
    """One named slice of cases to run.

    ``path`` is RELATIVE to the operator's private data root, never absolute
    or home-shaped. ``include_heldout`` must be ``True`` exactly for the
    held-out splits: such a set is scored from saved predictions only and is
    never sent to a provider.
    """

    name: str
    count: int
    split: str
    path: str
    include_heldout: bool = False


@dataclass(frozen=True)
class Budget:
    """Per-provider spend and concurrency cap."""

    provider: str
    usd_cap: float
    concurrency_cap: int
    #: Client-side request spacing for sync calls (e.g. a free tier's RPM).
    requests_per_minute: float | None = None
    #: Seconds a sync call may take before it counts as a timeout.
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class StopRules:
    """The truncation stop: once a model has ``min_answers`` answers and at least
    ``max_truncated_share`` of them were cut at the output budget, its calls stop
    until the operator decides."""

    min_answers: int = 5
    max_truncated_share: float = 0.3


@dataclass(frozen=True)
class Manifest:
    domain: str
    candidates: tuple[RunEntry, ...]
    baselines: tuple[RunEntry, ...]
    references: tuple[Reference, ...]
    case_sets: tuple[CaseSet, ...]
    budgets: tuple[Budget, ...]
    world: str | None = None
    stops: StopRules = StopRules()

    def budget_for(self, provider: str) -> Budget | None:
        """The ``[budget.<provider>]`` table, or ``None`` when absent."""
        for budget in self.budgets:
            if budget.provider == provider:
                return budget
        return None

    def sendable_case_sets(self) -> tuple[CaseSet, ...]:
        """Case sets that may be sent to providers (excludes held-out sets)."""
        return tuple(cs for cs in self.case_sets if not cs.include_heldout)

    def roster_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(ref.key for ref in self.references)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def _require_str(table: Mapping, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: missing or empty required string field {key!r}")
    return value


def _require_provider(table: Mapping, where: str) -> str:
    provider = _require_str(table, "provider", where)
    if provider not in ALLOWED_PROVIDERS:
        raise ManifestError(
            f"{where}: unknown provider {provider!r} "
            f"(must be one of {sorted(ALLOWED_PROVIDERS)})"
        )
    return provider


def _optional_str(table: Mapping, key: str, where: str) -> str | None:
    value = table.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ManifestError(f"{where}: {key} must be a non-empty string")
    return value


def _str_table(table: Mapping, key: str, where: str) -> dict[str, str]:
    value = table.get(key, {})
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in value.items()
    ):
        raise ManifestError(f"{where}: {key} must be a table of case-set name -> path strings")
    return dict(value)


def _number(table: Mapping, key: str, where: str, default: float) -> float:
    value = table.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{where}: {key} must be a non-negative number")
    return float(value)


def _positive_int(table: Mapping, key: str, where: str, default: int | None) -> int | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestError(f"{where}: {key} must be a positive integer")
    return value


def _parse_run_entry(table: Mapping, where: str) -> RunEntry:
    name = _require_str(table, "name", where)
    predictions_path = _require_str(table, "predictions_path", where)
    train_split = table.get("train_split")
    where = f"{where} (name={name!r})"
    if train_split is not None and not isinstance(train_split, str):
        raise ManifestError(f"{where}: train_split must be a string")
    policies = table.get("policies", ["raw"])
    if (
        not isinstance(policies, list)
        or not policies
        or not all(isinstance(p, str) and p for p in policies)
    ):
        raise ManifestError(f"{where}: policies must be a non-empty list of policy names")
    return RunEntry(
        name=name,
        predictions_path=predictions_path,
        train_split=train_split,
        predictions=_str_table(table, "predictions", where),
        policies=tuple(policies),
        repo_id=_optional_str(table, "repo_id", where),
        revision=_optional_str(table, "revision", where),
    )


def _parse_reference(table: Mapping, where: str) -> Reference:
    provider = _require_provider(table, where)
    model = _require_str(table, "model", where)
    reasoning = table.get("reasoning", "medium")
    if reasoning not in REASONING_LEVELS:
        raise ManifestError(f"{where}: reasoning must be one of {list(REASONING_LEVELS)}")
    if table.get("batch", False) is not False:
        raise ManifestError(
            f"{where}: batch routing is not supported by this gate (batch APIs are deferred); "
            "remove 'batch' or set it to false"
        )
    raw_capabilities = table.get("capabilities", ())
    if not isinstance(raw_capabilities, (list, tuple)) or not all(
        isinstance(c, str) for c in raw_capabilities
    ):
        raise ManifestError(f"{where}: capabilities must be a list of strings")
    api_key_env = table.get("api_key_env")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ManifestError(f"{where}: api_key_env must be a string (an env var NAME, not a value)")
    quantization = table.get("quantization")
    if quantization is not None and not isinstance(quantization, str):
        raise ManifestError(f"{where}: quantization must be a string")
    return Reference(
        provider=provider,
        model=model,
        reasoning=reasoning,
        capabilities=tuple(raw_capabilities),
        api_key_env=api_key_env,
        quantization=quantization,
        max_output_tokens=_positive_int(table, "max_output_tokens", where, None),
        usd_per_mtok_in=_number(table, "usd_per_mtok_in", where, 0.0),
        usd_per_mtok_out=_number(table, "usd_per_mtok_out", where, 0.0),
    )


def _private_relative(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    if value.startswith("/") or "~" in value:
        raise ManifestError(
            f"{where} must be relative to the private data root, never absolute or "
            f"home-shaped, got {value!r}"
        )
    return value


def _parse_case_set(table: Mapping, where: str) -> CaseSet:
    name = _require_str(table, "name", where)
    where = f"{where} (name={name!r})"
    count = table.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ManifestError(f"{where}: count must be a non-negative integer")
    split = _require_str(table, "split", where)
    if split not in SPLIT_TAGS:
        raise ManifestError(f"{where}: unknown split {split!r} (must be one of {SPLIT_TAGS})")
    path = _private_relative(_require_str(table, "path", where), f"{where}: path")
    include_heldout = table.get("include_heldout", False)
    if not isinstance(include_heldout, bool):
        raise ManifestError(f"{where}: include_heldout must be a boolean")
    expected_heldout = split in _HELDOUT_SPLITS
    if include_heldout != expected_heldout:
        raise ManifestError(
            f"{where}: include_heldout must be {expected_heldout} for split {split!r}"
        )
    return CaseSet(name=name, count=count, split=split, path=path, include_heldout=include_heldout)


def _is_real(value: object) -> bool:
    """An int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_budget(provider: str, table: Mapping, where: str) -> Budget:
    if provider not in ALLOWED_PROVIDERS:
        raise ManifestError(
            f"{where}: unknown provider {provider!r} "
            f"(must be one of {sorted(ALLOWED_PROVIDERS)})"
        )
    usd_cap = table.get("usd_cap")
    if not _is_real(usd_cap) or usd_cap < 0:
        raise ManifestError(f"{where}: usd_cap must be a non-negative number")
    concurrency_cap = table.get("concurrency_cap")
    if (
        not isinstance(concurrency_cap, int)
        or isinstance(concurrency_cap, bool)
        or concurrency_cap < 1
    ):
        raise ManifestError(f"{where}: concurrency_cap must be a positive integer")
    if "batch_discount" in table:
        raise ManifestError(f"{where}: batch_discount is not supported (batch APIs are deferred)")
    rpm = table.get("requests_per_minute")
    if rpm is not None and (not _is_real(rpm) or rpm <= 0):
        raise ManifestError(f"{where}: requests_per_minute must be a positive number")
    return Budget(
        provider=provider,
        usd_cap=float(usd_cap),
        concurrency_cap=concurrency_cap,
        requests_per_minute=None if rpm is None else float(rpm),
        timeout_seconds=_number(table, "timeout_seconds", where, 60.0),
    )


def _parse_budgets(raw: Mapping, where: str) -> tuple[Budget, ...]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where} must be a table")
    return tuple(
        _parse_budget(provider, table, f"{where}.{provider}") for provider, table in raw.items()
    )


def _parse_stops(raw: Mapping) -> StopRules:
    if not isinstance(raw, Mapping):
        raise ManifestError("stops must be a table")
    min_answers = _positive_int(raw, "min_answers", "stops", StopRules.min_answers)
    share = raw.get("max_truncated_share", StopRules.max_truncated_share)
    if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 < float(share) <= 1:
        raise ManifestError("stops.max_truncated_share must be a number in (0, 1]")
    return StopRules(min_answers=int(min_answers), max_truncated_share=float(share))


def parse_manifest(data: Mapping) -> Manifest:
    """Validate and build a :class:`Manifest` from a parsed TOML mapping.

    Raises :class:`ManifestError` for a missing domain, an unknown provider
    anywhere, a duplicate ``(provider, model)`` reference, a batch request,
    a held-out flag that disagrees with its split, or a private path that is
    absolute or home-shaped.
    """
    domain = _require_str(data, "domain", "manifest")
    candidates = tuple(
        _parse_run_entry(t, f"candidate[{i}]") for i, t in enumerate(data.get("candidate", []))
    )
    baselines = tuple(
        _parse_run_entry(t, f"baseline[{i}]") for i, t in enumerate(data.get("baseline", []))
    )
    references: list[Reference] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, table in enumerate(data.get("reference", [])):
        where = f"reference[{i}]"
        ref = _parse_reference(table, where)
        if ref.key in seen_pairs:
            raise ManifestError(
                f"{where}: duplicate (provider, model) pair {ref.key!r} in reference roster"
            )
        seen_pairs.add(ref.key)
        references.append(ref)
    case_sets = tuple(
        _parse_case_set(t, f"case_set[{i}]") for i, t in enumerate(data.get("case_set", []))
    )
    names = [cs.name for cs in case_sets]
    if len(set(names)) != len(names):
        raise ManifestError("case_set names must be unique")
    return Manifest(
        domain=domain,
        candidates=candidates,
        baselines=baselines,
        references=tuple(references),
        case_sets=case_sets,
        budgets=_parse_budgets(data.get("budget", {}), "budget"),
        world=_private_relative(data.get("world"), "world"),
        stops=_parse_stops(data.get("stops", {})),
    )


def load_manifest(path: str | os.PathLike) -> Manifest:
    """Load and validate a manifest TOML file at ``path``."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return parse_manifest(data)


def resolve_manifest_path(env: Mapping[str, str] | None = None) -> Path:
    """The operator's private manifest path from ``JEV_EVALS_MANIFEST`` (never a default)."""
    env = os.environ if env is None else env
    raw = env.get(ENV_MANIFEST_PATH)
    if not raw:
        raise ManifestError(
            f"{ENV_MANIFEST_PATH} is not set. The private run manifest lives outside this "
            f"repo; point {ENV_MANIFEST_PATH} at the operator's manifest file."
        )
    return Path(raw)


def load_manifest_from_env(env: Mapping[str, str] | None = None) -> Manifest:
    """Load the operator's private manifest via ``JEV_EVALS_MANIFEST``."""
    return load_manifest(resolve_manifest_path(env))
