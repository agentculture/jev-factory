"""Student profile: every pin and limit that names one distillation student, in one place.

``DEFAULT_2610`` is the built-in profile (``strands-decider-2B-qwen3.5-v1-2610``). A run
selects another with the run-config key ``distill_student_profile``, a path to a standalone
TOML file (:func:`load_profile_file`; run.toml itself refuses tables). Loading fails closed with
exit 1 and the field named: an unknown key, a missing required field, a malformed sha256 or a
wrong type.

Stdlib only; no heavy imports.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

#: The ``run.toml`` table that holds a student profile.
RUN_TOML_KEY = "student"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX = re.compile(r"^[0-9a-f]{8,64}$")


@dataclass(frozen=True)
class CardBaseline:
    """The upstream card's published reproduction target (informational)."""

    n: int
    correct: int
    tasks_sha256: str


@dataclass(frozen=True)
class StudentProfile:
    # base model
    base_repo: str
    base_revision: str
    backbone: str
    # decider (the released student the pins verify)
    decider_repo: str
    decider_revision: str
    adapter_sha256: str
    head_sha256: str
    # limits
    num_slots: int
    max_tokens: int
    weight_budget_bytes: int
    # prereg seeds
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    temperature_by_kind: Mapping[str, float]
    contamination_manifest: str
    # optional
    upstream_model: str = ""
    upstream_revision: str = ""
    upstream_source: str = ""
    train_config_file: str = "train_config.json"
    decider_config_file: str = "strands_decider_config.json"
    adapter_config_file: str = "lora/adapter_config.json"
    adapter_file: str = "lora/adapter_model.safetensors"
    head_file: str = "head.safetensors"
    provenance_file: str = "provenance.json"
    card: CardBaseline | None = None

    def __post_init__(self) -> None:
        for name in ("adapter_sha256", "head_sha256"):
            if not _SHA256.match(str(getattr(self, name))):
                raise _bad(name, "must be a full 64-hex lowercase sha256")
        for name in ("base_revision", "decider_revision"):
            if not _HEX.match(str(getattr(self, name))):
                raise _bad(name, "must be a pinned hex commit hash (>= 8 chars)")
        if self.card is not None and not _SHA256.match(self.card.tasks_sha256):
            raise _bad("card.tasks_sha256", "must be a full 64-hex lowercase sha256")


def _bad(field: str, why: str) -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        f"student profile: field {field!r} {why}",
        f"fix {field!r} in the [{RUN_TOML_KEY}] table or profile file",
    )


_REQUIRED = {
    f.name for f in fields(StudentProfile) if f.default is MISSING and f.default_factory is MISSING
}
_KNOWN = {f.name for f in fields(StudentProfile)}
_INT = {"num_slots", "max_tokens", "weight_budget_bytes", "lora_r", "lora_alpha"}
_STR = _KNOWN - _INT - {"lora_dropout", "lora_targets", "temperature_by_kind", "card"}


def profile_from_table(table: Mapping[str, Any]) -> StudentProfile:
    """Build a profile from a TOML table, refusing (exit 1, field named) anything malformed."""
    if not isinstance(table, Mapping):
        raise _bad(RUN_TOML_KEY, "must be a table")
    for key in table:
        if key not in _KNOWN:
            raise _bad(str(key), "is not a known key")
    for name in sorted(_REQUIRED):
        if name not in table:
            raise _bad(name, "is required but missing")
    kw: dict[str, Any] = dict(table)
    for name in _INT:
        if isinstance(kw[name], bool) or not isinstance(kw[name], int):
            raise _bad(name, "must be an integer")
    for name in _STR & kw.keys():
        if not isinstance(kw[name], str):
            raise _bad(name, "must be a string")
    if isinstance(kw["lora_dropout"], bool) or not isinstance(kw["lora_dropout"], (int, float)):
        raise _bad("lora_dropout", "must be a number")
    targets = kw["lora_targets"]
    if not isinstance(targets, (list, tuple)) or not all(isinstance(t, str) for t in targets):
        raise _bad("lora_targets", "must be a list of strings")
    kw["lora_targets"] = tuple(targets)
    temps = kw["temperature_by_kind"]
    if not isinstance(temps, Mapping) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in temps.values()
    ):
        raise _bad("temperature_by_kind", "must be a table of numbers")
    kw["temperature_by_kind"] = dict(temps)
    if kw.get("card") is not None:
        card = kw["card"]
        if not isinstance(card, Mapping):
            raise _bad("card", "must be a table with n, correct, tasks_sha256")
        for key in card:
            if key not in ("n", "correct", "tasks_sha256"):
                raise _bad(f"card.{key}", "is not a known key")
        for key in ("n", "correct", "tasks_sha256"):
            if key not in card:
                raise _bad(f"card.{key}", "is required but missing")
        for key in ("n", "correct"):
            if isinstance(card[key], bool) or not isinstance(card[key], int):
                raise _bad(f"card.{key}", "must be an integer")
        if not isinstance(card["tasks_sha256"], str):
            raise _bad("card.tasks_sha256", "must be a string")
        kw["card"] = CardBaseline(card["n"], card["correct"], card["tasks_sha256"])
    return StudentProfile(**kw)


def load_profile_file(path: Path | str) -> StudentProfile:
    """Load a profile from a TOML file: the top-level table, or its ``[student]`` table.

    A relative ``contamination_manifest`` is resolved against the profile file's directory.
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CliError(
            EXIT_USER_ERROR,
            f"student profile: cannot read {p}: {exc}",
            "point at a valid TOML file",
        ) from exc
    prof = profile_from_table(data[RUN_TOML_KEY] if RUN_TOML_KEY in data else data)
    manifest = Path(prof.contamination_manifest)
    if not manifest.is_absolute():
        prof = replace(prof, contamination_manifest=str(p.parent / manifest))
    return prof


def profile_from_run_toml(data: Mapping[str, Any]) -> StudentProfile:
    """The profile in a parsed ``run.toml``'s ``[student]`` table, else :data:`DEFAULT_2610`."""
    if RUN_TOML_KEY not in data:
        return DEFAULT_2610
    return profile_from_table(data[RUN_TOML_KEY])


DEFAULT_2610 = StudentProfile(
    base_repo="Qwen/Qwen3.5-2B-Base",
    # Upstream marks the base revision "inferred: Hub main at training time"; this is our pin.
    base_revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    backbone="pointer",
    decider_repo="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
    decider_revision="216d1ccb46a9e9b036eb9187b248db98480f77d4",
    adapter_sha256="ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7",
    head_sha256="ace56742545b505e6fae34ada85c8edbeedb23ba8974d8680aebe0380d6083a1",
    num_slots=24,
    max_tokens=4096,
    weight_budget_bytes=10**9,
    lora_r=48,
    lora_alpha=96,
    lora_dropout=0.05,
    lora_targets=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    ),
    temperature_by_kind={
        "noul": 1.0704394404346953,
        "choice": 0.8629556148274228,
        "score": 1.834409163182587,
    },
    contamination_manifest=str(Path(__file__).parent / "data" / "upstream_sources.json"),
    upstream_model="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
    upstream_revision="216d1ccb46a9e9b036eb9187b248db98480f77d4",
    upstream_source=(
        "https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
        "/raw/216d1ccb46a9e9b036eb9187b248db98480f77d4"
    ),
    card=CardBaseline(
        n=231,
        correct=180,
        tasks_sha256="771ea8ac252af155606eeb09cad920b32b326c5c2b1e9c9af868be00fe8e2fad",
    ),
)
