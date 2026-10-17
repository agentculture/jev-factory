"""Distill pre-registration: success bars, selection rule, knob grid, upstream knobs.

Mirrors :mod:`jev_factory.factory.prereg` (validate, hash into a lock, refuse a
changed file without a deviation id) but in its own files, ``distill/prereg.json``
and ``distill/prereg.lock.json``, so the jev-tool pre-registration is untouched.
Every value taken from the upstream Strands Decider training config must cite
the file and key it came from; every knob a run uses must be in the grid.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile
from jev_factory.distill.student import backbone as student_backbone
from jev_factory.factory import safepath
from jev_factory.runtime import prereg_lock
from jev_factory.runtime.prereg_lock import sha256_file  # noqa: F401  (re-exported)

#: The deployed build's weight budget: the student backbone's own (the c17 bar's 10^9 bytes),
#: read through the registry (deviation d2).
SIZE_BUDGET_BYTES = student_backbone(DEFAULT_2610).quantizer().SIZE_BUDGET_BYTES

#: 3 = roles and a deployed format: ``roles`` declares the teacher (Clef, with its locked
#: build ``variant``), the stock target and the student (all mandatory) and the alt teacher,
#: the deployed-quant server and the edge board (each present or absent); ``deployed_format``
#: is ``quant`` or ``bf16``; the c17 teacher bar is variant-neutral
#: (``top1_under_teacher_max_points``). v3 carries no ``reference_rows`` (``roles`` replace them).
#:
#: 2 = the Jetson-ops (#20) shape: ``bars.c17`` replaces #18's gap-closure / latency bars as the
#: run's success bars and the upstream seed is strands-decider-2B-qwen3.5-v1-2610. A locked v2
#: file still loads, unchanged and never rewritten: its v2-shaped view is returned as-is and
#: the v3 accessors read it through :func:`v2_to_v3`.
#:
#: A v1 (#18-shape) file is refused by name, never reinterpreted.
SCHEMA_VERSION = 3
V2_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (V2_SCHEMA_VERSION, SCHEMA_VERSION)
PREREG_REL = "distill/prereg.json"
LOCK_REL = prereg_lock.LOCK_REL

#: Knobs a run may set that are searched over (each needs a registered grid).
KNOB_NAMES: tuple[str, ...] = (
    "alpha",
    "temperature",
    "teacher_filter",
    "teacher_weak_floor",
    "data_mix",
    "epochs",
    "lr",
)
TEACHER_FILTERS = ("all", "agrees_with_gold")
REFERENCE_ROWS = ("clef", "stock")

#: v3 roles. The first three are mandatory (declaring one absent is refused); the others are
#: each declared present or absent before the lock.
MANDATORY_ROLES: tuple[str, ...] = ("teacher", "stock", "student")
OPTIONAL_ROLES: tuple[str, ...] = ("alt_teacher", "deployed_server", "edge")
ROLE_NAMES: tuple[str, ...] = MANDATORY_ROLES + OPTIONAL_ROLES
#: Fields a present role must carry (the alt teacher's model comes from run config,
#: ``distill_alt_teacher_model``, so it needs none here).
ROLE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "teacher": ("model", "variant"),
    "stock": ("model",),
}
#: What a v3 run ships: the quantized student (``quant``) or the bf16 student (``bf16``).
DEPLOYED_FORMATS: tuple[str, ...] = ("quant", "bf16")
#: The Clef build every v2 file was registered against (decision c43: the teacher bar is
#: relative to the Clef variant the run serves, locked here and reported beside every number).
V2_TEACHER_VARIANT = "W4A16"
DEFAULT_TEACHER_MODEL = "Clef 27B"

#: bar name -> (kind, bound); kind "rate" in [0, 1], "pos" > 0, "flag" must be bool.
TEST_BARS: dict[str, str] = {
    "accuracy_diff_ci_lower_min": "rate",
    "gap_closure_min": "rate",
    "ece_max": "rate",
    "ece_no_worse_than_stock": "flag",
    "confident_wrong_no_worse_than_stock": "flag",
    "latency_ratio_max": "pos",
}
QUANT_BARS: dict[str, str] = {
    "max_accuracy_drop_points": "pos",
    "ece_max": "rate",
    "heal_rounds_max": "int",
}
#: The c17 success bars (issue #20), pre-registered per build: ``bf16`` (the student) and
#: ``quant`` (the shipped quantized student), on the human-gold Jetson-ops test side.
C17_BUILDS: tuple[str, ...] = ("bf16", "quant")
#: top-1 must beat stock by at least this many points ...
C17_STOCK_MARGIN_POINTS = 10.0
#: ... and may trail the teacher (the registered Clef variant) by at most this many points.
C17_TEACHER_ALLOWANCE_POINTS = 5.0
C17_CLEF_W4A16_ALLOWANCE_POINTS = C17_TEACHER_ALLOWANCE_POINTS  # the v2 name
C17_AGREEMENT_MIN = 0.97
#: The teacher bar's name in v3 (variant-neutral) and in v2 (W4A16 in the name).
TEACHER_BAR = "top1_under_teacher_max_points"
V2_TEACHER_BAR = "top1_under_clef_w4a16_max_points"
#: The explicit v2 -> v3 c17 bar mapping (:func:`v2_to_v3`); every other bar keeps its name.
V2_BAR_MAP: dict[str, str] = {V2_TEACHER_BAR: TEACHER_BAR}
#: bar name -> (kind, weakest allowed value); ``min`` bars may be stricter (larger), ``max``
#: bars stricter (smaller), but a registration weaker than this floor is refused.
#: (v2 names; :data:`C17_TOP1_BARS_V3` holds the v3 ones.)
C17_TOP1_BARS: dict[str, tuple[str, float]] = {
    "top1_over_stock_min_points": ("min", C17_STOCK_MARGIN_POINTS),
    V2_TEACHER_BAR: ("max", C17_TEACHER_ALLOWANCE_POINTS),
}
C17_TOP1_BARS_V3: dict[str, tuple[str, float]] = {
    "top1_over_stock_min_points": ("min", C17_STOCK_MARGIN_POINTS),
    TEACHER_BAR: ("max", C17_TEACHER_ALLOWANCE_POINTS),
}
#: The c17 bars relative to the stock and teacher rows: both roles are mandatory, so these
#: can never be marked not applicable (h7).
C17_RELATIVE_BARS: frozenset[str] = frozenset({*C17_TOP1_BARS, *C17_TOP1_BARS_V3})
C17_QUANT_ONLY_BARS: dict[str, tuple[str, float]] = {
    "quant_vs_bf16_agreement_min": ("min", C17_AGREEMENT_MIN),
    "weight_bytes_max": ("max", float(SIZE_BUDGET_BYTES)),
}
#: Latency is reported, never gated: each build lists the statistics it must report.
C17_LATENCY_REPORTED: tuple[str, ...] = ("median", "p95")
V1_REFUSAL = (
    "this is a #18-shape (schema_version 1) distill pre-registration; the Jetson-ops run "
    "(#20) needs schema_version 2 with bars.c17"
)


def example_c17_bars(weight_bytes_max: int = SIZE_BUDGET_BYTES) -> dict[str, Any]:
    """The c17 bars as pre-registered for the Jetson-ops run (a v2 draft until the lock)."""
    out: dict[str, Any] = {}
    for build in C17_BUILDS:
        bars: dict[str, Any] = {
            "top1_over_stock_min_points": C17_STOCK_MARGIN_POINTS,
            V2_TEACHER_BAR: C17_TEACHER_ALLOWANCE_POINTS,
        }
        if build == "quant":
            bars["quant_vs_bf16_agreement_min"] = C17_AGREEMENT_MIN
            bars["weight_bytes_max"] = weight_bytes_max
        bars["latency_reported"] = list(C17_LATENCY_REPORTED)
        out[build] = bars
    return out


#: Question kinds the decision gate is fitted for (deviation d4), one threshold set each.
GATE_KINDS: tuple[str, ...] = ("choice", "noul", "score")
#: The gate thresholds swept per kind (floor, margin, max_entropy; ``None`` = disabled).
GATE_THRESHOLDS: tuple[str, ...] = ("floor", "margin", "max_entropy")
#: Registered gate objectives: maximise selective accuracy subject to a coverage minimum.
GATE_OBJECTIVES: tuple[str, ...] = ("max_selective_accuracy",)
GATE_PLACEHOLDER = (
    "operator placeholder (deviation d4): confirm the objective, coverage minimums and grid "
    "before the lock"
)
SELECTION_ORDER: tuple[str, ...] = (
    "safety",
    "accuracy_floor",
    "calibration",
    "confident_wrong",
    "ties",
)


class UnregisteredKnobError(CliError):
    """A run knob (or its value) is not in the pre-registered grid."""

    def __init__(self, message: str) -> None:
        super().__init__(
            EXIT_USER_ERROR,
            message,
            "add the knob to the distill pre-registration before the run, or record a deviation",
        )


class UpstreamSourceMissingError(CliError):
    """An upstream knob value does not cite its source file and key."""

    def __init__(self, message: str) -> None:
        super().__init__(
            EXIT_USER_ERROR,
            message,
            "every value taken from strands-decider config needs source_file and source_key",
        )


@dataclass(frozen=True)
class UpstreamKnob:
    name: str
    value: Any
    source_file: str
    source_key: str


@dataclass(frozen=True)
class DistillPrereg:
    bars: dict[str, dict[str, Any]]
    selection_rule: dict[str, Any]
    knob_grid: dict[str, Any]
    reference_rows: dict[str, dict[str, Any]]
    upstream_knobs: tuple[UpstreamKnob, ...]
    gate: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    #: The file's own schema version. ``bars`` is the file's own shape (v2 names for a v2
    #: file); read the normalised v3 view through :func:`roles`, :func:`deployed_format`,
    #: :func:`teacher_variant`, :func:`c17_bars` and :func:`teacher_bar`.
    schema_version: int = V2_SCHEMA_VERSION
    v3_roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    v3_deployed_format: str = "quant"
    v3_c17: dict[str, dict[str, Any]] = field(default_factory=dict)


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation or "fix the distill pre-registration file (distill/prereg.json)",
    )


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


#: Where every ``upstream_knobs`` value in :func:`example_doc` was read (once, by hand):
#: ``<UPSTREAM_SOURCE>/<source_file>``, key ``source_key``. Placeholders for the operator to
#: confirm before the lock; nothing fetches these at run time.
UPSTREAM_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
UPSTREAM_REVISION = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
UPSTREAM_SOURCE = f"https://huggingface.co/{UPSTREAM_MODEL}/raw/{UPSTREAM_REVISION}"
TRAIN_CONFIG = "train_config.json"
DECIDER_CONFIG = "strands_decider_config.json"
ADAPTER_CONFIG = "lora/adapter_config.json"
#: Where the heal recipe in :func:`example_doc` comes from: nvsh's d18, a short bf16
#: continuation (not quantization-aware training) from the healed model's own checkpoint.
HEAL_SOURCE = (
    "nvsh#46 decision d18 (docs/deliveries/2026-09-23-qwen3-5-0-8b-tool-jev-fine-tune-"
    "issue-46.md): "
    "heal = 1 epoch, lr 5e-5, continuing from the model's own checkpoint on the frozen "
    "train set (a plain continued fine-tune, not QAT)"
)
#: The heal's soft-target mix (c10): weight on the bf16 distilled student's per-option
#: distribution and on Clef's. A draft for the operator to confirm before the lock.
HEAL_MIX = {"bf16_student": 0.5, "clef": 0.5}
HEAL_MIX_SOURCE = (
    "spec 2026-10-15-orin-nano-jetson-ops-decider c10: heal trains the quantized student "
    "against the bf16 distilled student's and Clef's soft labels; the equal split is an "
    "operator placeholder until the lock"
)
#: The stock model's ``lora_targets``: exactly the 12 modules of 2610's
#: ``lora/adapter_config.json`` ``target_modules`` (its train_config.json and
#: strands_decider_config.json list the same set).
STOCK_LORA_TARGETS = (
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
)


#: ``temperature_by_kind`` in 2610's strands_decider_config.json (the stock calibration).
STOCK_TEMPERATURE_BY_KIND = {
    "noul": 1.0704394404346953,
    "choice": 0.8629556148274228,
    "score": 1.834409163182587,
}


def _stock_model(profile: StudentProfile) -> str:
    repo = profile.upstream_model or profile.decider_repo
    rev = profile.upstream_revision or profile.decider_revision
    return f"{repo}@{rev}"


def example_doc(profile: StudentProfile = DEFAULT_2610) -> dict[str, Any]:
    """A valid v2 pre-registration draft, seeded from the student ``profile``.

    With :data:`~jev_factory.distill.profile.DEFAULT_2610` (strands-decider-2B-qwen3.5-v1-2610
    @ 216d1ccb) the draft is exactly the pre-t7 one. The profile seeds the LoRA knobs, the
    stock calibration (``temperature_by_kind``), the source file names, the stock reference
    model and the quant's ``weight_bytes_max``. The draft stays schema v2 so every current
    reader of the v2 bar names keeps working; :func:`example_doc_v3` is the v3 draft.

    Carries every hyperparameter the pointer distill trainer resolves (``init_mode`` in the
    grid; the fixed upstream values, each citing :data:`UPSTREAM_SOURCE` file and key) and
    the selection rule's ``confident_wrong_threshold``, and the one ``heal`` round's knobs
    (:data:`HEAL_SOURCE`) that :mod:`jev_factory.distill.quant` reads. All values are drafts
    for the operator to confirm before the lock.
    """
    p = profile
    train_cfg, decider_cfg, adapter_cfg = (
        p.train_config_file,
        p.decider_config_file,
        p.adapter_config_file,
    )
    upstream = [
        ("lr", 1e-4, TRAIN_CONFIG, "lr"),
        ("head_lr", 1e-3, TRAIN_CONFIG, "head_lr"),
        ("epochs", 1, TRAIN_CONFIG, "epochs"),
        ("kl_frozen_weight", 0.3, TRAIN_CONFIG, "kl_frozen_weight"),
        ("teacher_weight", 1.0, TRAIN_CONFIG, "teacher_weight"),
        ("ordinal_smoothing", 0.1, TRAIN_CONFIG, "ordinal_smoothing"),
        ("micro_batch_size", 8, TRAIN_CONFIG, "micro_batch_size"),
        ("grad_accum", 4, TRAIN_CONFIG, "grad_accum"),
        ("weight_decay", 0.01, TRAIN_CONFIG, "weight_decay"),
        ("warmup_ratio", 0.03, TRAIN_CONFIG, "warmup_ratio"),
        ("max_grad_norm", 1.0, TRAIN_CONFIG, "max_grad_norm"),
        ("max_length", 4096, TRAIN_CONFIG, "max_length"),
        ("seed", 0, TRAIN_CONFIG, "seed"),
        ("shuffle_options", True, TRAIN_CONFIG, "shuffle_options"),
        ("reverse_score_prob", 0.5, TRAIN_CONFIG, "reverse_score_prob"),
        # 2610 trains with r48 / alpha 96: the adapter file and strands_decider_config.json say
        # so (its train_config.json still lists the r16 / alpha 32 of the earlier recipe).
        ("lora_r", p.lora_r, adapter_cfg, "r"),
        ("lora_alpha", p.lora_alpha, adapter_cfg, "lora_alpha"),
        ("lora_dropout", p.lora_dropout, adapter_cfg, "lora_dropout"),
        ("lora_targets", list(p.lora_targets), adapter_cfg, "target_modules"),
        (
            "temperature_by_kind",
            dict(p.temperature_by_kind),
            decider_cfg,
            "temperature_by_kind",
        ),
    ]
    # the train_config.json values above cite the profile's own file name
    upstream = [(n, v, train_cfg if f == TRAIN_CONFIG else f, k) for n, v, f, k in upstream]
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "bars": {
            "c17": example_c17_bars(p.weight_budget_bytes),
            "test": {
                "accuracy_diff_ci_lower_min": 0.0,
                "gap_closure_min": 0.5,
                "ece_max": 0.10,
                "ece_no_worse_than_stock": True,
                "confident_wrong_no_worse_than_stock": True,
                "latency_ratio_max": 0.2,
            },
            "quant": {
                "max_accuracy_drop_points": 3.0,
                "ece_max": 0.10,
                "heal_rounds_max": 1,
            },
        },
        "selection_rule": {
            "order": list(SELECTION_ORDER),
            "tolerances": {"accuracy_floor_margin": 0.05, "ece": 0.01},
            # A wrong answer at or above this top probability is "confident-wrong".
            "confident_wrong_threshold": 0.9,
        },
        "knob_grid": {
            "alpha": [0.5, 0.7, 0.9],
            "temperature": {"choice": [1.0, 2.0], "score": [1.0, 2.0], "noul": [1.0]},
            "teacher_filter": list(TEACHER_FILTERS),
            "teacher_weak_floor": [0.6, 0.7],
            "data_mix": ["teacher_only", "teacher_plus_corrections"],
            "epochs": [1, 2],
            "lr": [5e-5, 1e-4],
            "init_mode": ["continue", "merge_fresh"],
        },
        # The one heal round (distill quant): one more pass from the winner's own checkpoint
        # on the train records, with the winner's other knobs. Each value must be in
        # knob_grid (the trainer refuses an unregistered value). ``mix`` weighs the heal's
        # soft targets: the bf16 distilled student's and Clef's per-option distributions
        # (c10); the 0.5 / 0.5 split is an operator placeholder until the lock.
        "heal": {
            "epochs": 1,
            "lr": 5e-5,
            "init_mode": "continue",
            "mix": dict(HEAL_MIX),
            "mix_source": HEAL_MIX_SOURCE,
            "source": HEAL_SOURCE,
        },
        # The decision gate (deviation d4): per question kind, sweep floor / margin /
        # max_entropy on the deployed build's calibrated fit-fold predictions and keep the
        # point with the best selective accuracy whose coverage meets the kind's minimum.
        # Every value here is an operator placeholder until the lock.
        "gate": {
            "objective": GATE_OBJECTIVES[0],
            "min_coverage": {"choice": 0.8, "noul": 0.8, "score": 0.8},
            "grid": {
                "floor": [None, 0.5, 0.6, 0.7, 0.8],
                "margin": [None, 0.1, 0.2],
                "max_entropy": [None, 0.8, 0.9],
            },
            "status": GATE_PLACEHOLDER,
        },
        "reference_rows": {
            "clef": {"model": DEFAULT_TEACHER_MODEL, "role": "teacher"},
            "stock": {
                "model": _stock_model(p),
                "role": "stock baseline",
            },
        },
        "upstream_knobs": [
            {"name": n, "value": v, "source_file": f, "source_key": k} for n, v, f, k in upstream
        ],
    }


def example_doc_v3(
    profile: StudentProfile = DEFAULT_2610,
    *,
    teacher_variant: str = V2_TEACHER_VARIANT,
    deployed_format: str = "quant",
    teacher_model: str = DEFAULT_TEACHER_MODEL,
) -> dict[str, Any]:
    """A valid v3 pre-registration draft, seeded from ``profile``.

    Every field of :func:`example_doc` keeps its value (the teacher bar under its v3 name;
    ``reference_rows`` become ``roles``) and v3 adds ``roles`` and ``deployed_format``. With
    ``deployed_format="bf16"`` the quant-only bars (``bars.c17.quant``, ``bars.quant``) and
    the quant ``heal`` round are left out. Every optional role is drafted present.
    """
    doc = v2_to_v3(example_doc(profile))
    doc["roles"]["teacher"].update(model=teacher_model, variant=teacher_variant)
    doc["deployed_format"] = deployed_format
    if deployed_format == "bf16":
        doc["bars"]["c17"].pop("quant", None)
        doc["bars"].pop("quant", None)
        doc.pop("heal", None)
    return doc


def v2_to_v3(doc: dict[str, Any]) -> dict[str, Any]:
    """The explicit v2 -> v3 mapping, in memory only (a locked file is never rewritten).

    ``top1_under_clef_w4a16_max_points`` becomes the teacher bar
    (:data:`TEACHER_BAR`) with the teacher's variant :data:`V2_TEACHER_VARIANT`;
    ``reference_rows.clef`` / ``.stock`` become the ``teacher`` / ``stock`` roles. A v2 file
    declared no roles or format, so the mapping takes today's pipeline as the default: the
    student and every optional role (alt teacher, deployed-quant server, edge) present, and
    ``deployed_format`` ``quant``. The input is not mutated.
    """
    if not isinstance(doc, dict) or doc.get("schema_version") != V2_SCHEMA_VERSION:
        raise _err("v2_to_v3 maps a schema_version 2 distill pre-registration only")
    out = copy.deepcopy(doc)
    out["schema_version"] = SCHEMA_VERSION
    refs = out.pop("reference_rows", None)
    refs = refs if isinstance(refs, dict) else {}
    clef = refs.get("clef") if isinstance(refs.get("clef"), dict) else {}
    stock = refs.get("stock") if isinstance(refs.get("stock"), dict) else {}
    roles: dict[str, dict[str, Any]] = {
        "teacher": {
            "present": True,
            "model": clef.get("model"),
            "variant": V2_TEACHER_VARIANT,
        },
        "stock": {"present": True, "model": stock.get("model")},
        "student": {"present": True},
    }
    roles.update({name: {"present": True} for name in OPTIONAL_ROLES})
    out["roles"] = roles
    out["deployed_format"] = "quant"
    c17 = out.get("bars", {}).get("c17") if isinstance(out.get("bars"), dict) else None
    if isinstance(c17, dict):
        for build, entry in c17.items():
            if isinstance(entry, dict):
                c17[build] = {V2_BAR_MAP.get(k, k): v for k, v in entry.items()}
    return out


def _check_bar(section: str, name: str, kind: str, value: Any) -> None:
    ok = {
        "rate": _is_num(value) and 0.0 <= value <= 1.0,
        "pos": _is_num(value) and value > 0,
        "int": _is_int(value) and value >= 0,
        "flag": isinstance(value, bool),
    }[kind]
    if not ok:
        raise _err(f"bars.{section}.{name} is invalid ({kind} expected)")


def _check_c17_bar(build: str, name: str, spec: tuple[str, float], value: Any) -> None:
    sense, limit = spec
    where = f"bars.c17.{build}.{name}"
    if name in C17_RELATIVE_BARS and (value is None or isinstance(value, str)):
        raise _err(
            f"{where} cannot be marked not applicable: it is relative to the stock and teacher "
            "rows, and those roles are mandatory (present in every run)",
            "register a number; c17 bars may be tightened, never loosened or dropped",
        )
    if name == "weight_bytes_max":
        if not _is_int(value):
            raise _err(f"{where} must be an integer number of bytes")
    elif not _is_num(value):
        raise _err(f"{where} must be a number")
    weaker = value < limit if sense == "min" else value > limit
    if weaker:
        raise _err(
            f"{where} is {value}, weaker than the c17 bar ({'>=' if sense == 'min' else '<='} "
            f"{limit:g})",
            "c17 bars may be tightened, never loosened; a change is a recorded deviation",
        )


def _c17_build(
    build: str, entry: Any, top1: dict[str, tuple[str, float]] = C17_TOP1_BARS
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise _err(f"bars.c17.{build} must be an object")
    spec = dict(top1)
    if build == "quant":
        spec.update(C17_QUANT_ONLY_BARS)
    missing = [n for n in (*spec, "latency_reported") if n not in entry]
    if missing:
        raise _err(f"bars.c17.{build} missing: {', '.join(missing)}")
    extra = sorted(set(entry) - set(spec) - {"latency_reported"})
    if extra:
        raise _err(f"bars.c17.{build} has unknown bars: {', '.join(extra)}")
    for name, limit in spec.items():
        _check_c17_bar(build, name, limit, entry[name])
    if entry["latency_reported"] != list(C17_LATENCY_REPORTED):
        raise _err(
            f"bars.c17.{build}.latency_reported must be exactly {list(C17_LATENCY_REPORTED)} "
            "(latency is reported with no bar)"
        )
    return dict(entry)


def _c17(
    doc: Any,
    top1: dict[str, tuple[str, float]] = C17_TOP1_BARS,
    required: tuple[str, ...] = C17_BUILDS,
) -> dict[str, Any]:
    """Validate ``bars.c17``: every ``required`` build, plus any other known build given."""
    if not isinstance(doc, dict):
        raise _err("bars.c17 must be an object")
    extra = sorted(set(doc) - set(C17_BUILDS))
    if extra:
        raise _err(f"bars.c17 has unknown builds: {', '.join(extra)}")
    builds = [b for b in C17_BUILDS if b in required or b in doc]
    return {build: _c17_build(build, doc.get(build), top1) for build in builds}


def _bars(doc: Any, deployed_format: str | None = None) -> dict[str, dict[str, Any]]:
    """Validate ``bars``. ``deployed_format`` ``None`` is the v2 shape (v2 bar names, the
    quant bars always required); a v3 format uses the v3 names and requires the quant-only
    bars (``bars.c17.quant``, ``bars.quant``) only for ``quant``, checking them if given."""
    if not isinstance(doc, dict):
        raise _err("bars must be an object")
    if deployed_format is None:
        top1, needs_quant = C17_TOP1_BARS, True
    else:
        top1, needs_quant = C17_TOP1_BARS_V3, deployed_format == "quant"
    required = C17_BUILDS if needs_quant else ("bf16",)
    out: dict[str, dict[str, Any]] = {"c17": _c17(doc.get("c17"), top1, required)}
    if deployed_format is not None:
        unknown = sorted(set(doc) - {"c17", "test", "quant"})
        if unknown:
            raise _err(f"bars has unknown sections: {', '.join(unknown)}")
    for section, spec in (("test", TEST_BARS), ("quant", QUANT_BARS)):
        if section == "quant" and not needs_quant and section not in doc:
            continue
        entry = doc.get(section)
        if not isinstance(entry, dict):
            raise _err(f"bars.{section} must be an object")
        missing = [n for n in spec if n not in entry]
        if missing:
            raise _err(f"bars.{section} missing: {', '.join(missing)}")
        extra = sorted(set(entry) - set(spec))
        if extra:
            raise _err(f"bars.{section} has unknown bars: {', '.join(extra)}")
        for name, kind in spec.items():
            _check_bar(section, name, kind, entry[name])
        out[section] = dict(entry)
    return out


def _selection_rule(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("order") != list(SELECTION_ORDER):
        raise _err(
            f"selection_rule.order must be exactly {list(SELECTION_ORDER)}",
            "the selection order is fixed; change it only via a deviation",
        )
    tol = doc.get("tolerances")
    if not isinstance(tol, dict) or not all(_is_num(v) and v >= 0 for v in tol.values()):
        raise _err("selection_rule.tolerances must map names to non-negative numbers")
    return dict(doc)


def _nonempty_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise _err(f"knob_grid.{name} must be a non-empty list")
    return value


def _check_numeric_grid(doc: dict[str, Any]) -> None:
    for name in ("alpha", "teacher_weak_floor"):
        if not all(_is_num(v) and 0.0 <= v <= 1.0 for v in _nonempty_list(name, doc[name])):
            raise _err(f"knob_grid.{name} values must be numbers in [0, 1]")
    for name in ("lr",):
        if not all(_is_num(v) and v > 0 for v in _nonempty_list(name, doc[name])):
            raise _err(f"knob_grid.{name} values must be positive numbers")
    if not all(_is_int(v) and v >= 1 for v in _nonempty_list("epochs", doc["epochs"])):
        raise _err("knob_grid.epochs values must be positive integers")


def _check_temperature_grid(temp: Any) -> None:
    if not isinstance(temp, dict) or not temp:
        raise _err("knob_grid.temperature must map each question kind to a list")
    for kind, vals in temp.items():
        if not all(_is_num(v) and v > 0 for v in _nonempty_list(f"temperature.{kind}", vals)):
            raise _err(f"knob_grid.temperature.{kind} values must be positive numbers")


def _grid(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise _err("knob_grid must be an object")
    missing = [n for n in KNOB_NAMES if n not in doc]
    if missing:
        raise _err(f"knob_grid missing: {', '.join(missing)}")
    _check_numeric_grid(doc)
    if not all(_nonempty_str(v) for v in _nonempty_list("data_mix", doc["data_mix"])):
        raise _err("knob_grid.data_mix values must be non-empty strings")
    tf = _nonempty_list("teacher_filter", doc["teacher_filter"])
    if not all(v in TEACHER_FILTERS for v in tf):
        raise _err(f"knob_grid.teacher_filter values must be in {list(TEACHER_FILTERS)}")
    _check_temperature_grid(doc["temperature"])
    return dict(doc)


def _gate(doc: Any) -> dict[str, Any] | None:
    """The optional decision-gate block; validated in full when present."""
    if doc is None:
        return None
    if not isinstance(doc, dict):
        raise _err("gate must be an object")
    if doc.get("objective") not in GATE_OBJECTIVES:
        raise _err(f"gate.objective must be one of {list(GATE_OBJECTIVES)}")
    cov = doc.get("min_coverage")
    if not isinstance(cov, dict) or sorted(cov) != sorted(GATE_KINDS):
        raise _err(f"gate.min_coverage must map exactly {list(GATE_KINDS)} to a rate")
    if not all(_is_num(v) and 0.0 < v <= 1.0 for v in cov.values()):
        raise _err("gate.min_coverage values must be numbers in (0, 1]")
    _check_gate_grid(doc.get("grid"))
    return dict(doc)


def _check_gate_grid(grid: Any) -> None:
    if not isinstance(grid, dict) or sorted(grid) != sorted(GATE_THRESHOLDS):
        raise _err(f"gate.grid must carry exactly {list(GATE_THRESHOLDS)}")
    for name in GATE_THRESHOLDS:
        values = grid[name]
        if not isinstance(values, list) or not values:
            raise _err(f"gate.grid.{name} must be a non-empty list")
        if not all(v is None or (_is_num(v) and 0.0 <= v <= 1.0) for v in values):
            raise _err(f"gate.grid.{name} values must be null or numbers in [0, 1]")


def _references(doc: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(doc, dict):
        raise _err("reference_rows must be an object")
    missing = [n for n in REFERENCE_ROWS if n not in doc]
    if missing:
        raise _err(f"reference_rows missing: {', '.join(missing)}")
    for name in REFERENCE_ROWS:
        if not isinstance(doc[name], dict) or not _nonempty_str(doc[name].get("model")):
            raise _err(f"reference_rows.{name}.model must be a non-empty string")
    return {n: dict(doc[n]) for n in REFERENCE_ROWS}


def _upstream(doc: Any) -> tuple[UpstreamKnob, ...]:
    if not isinstance(doc, list):
        raise _err("upstream_knobs must be a list")
    out: list[UpstreamKnob] = []
    for i, entry in enumerate(doc):
        if not isinstance(entry, dict) or not _nonempty_str(entry.get("name")):
            raise _err(f"upstream_knobs[{i}] needs a name")
        name = entry["name"]
        if "value" not in entry:
            raise _err(f"upstream knob {name} has no value")
        if not (_nonempty_str(entry.get("source_file")) and _nonempty_str(entry.get("source_key"))):
            raise UpstreamSourceMissingError(
                f"upstream knob {name} does not cite its source_file and source_key"
            )
        out.append(UpstreamKnob(name, entry["value"], entry["source_file"], entry["source_key"]))
    if len({k.name for k in out}) != len(out):
        raise _err("upstream_knobs names must be unique")
    return tuple(out)


def _role(name: str, entry: Any) -> dict[str, Any]:
    where = f"roles.{name}"
    if not isinstance(entry, dict) or not isinstance(entry.get("present"), bool):
        raise _err(f"{where} must be an object with a boolean 'present'")
    if name in MANDATORY_ROLES and not entry["present"]:
        raise _err(
            f"{where} is mandatory and cannot be declared absent "
            "(the teacher, stock and student roles are present in every run)",
            "declare it present; only alt_teacher, deployed_server and edge may be absent",
        )
    if entry["present"]:
        for key in ROLE_REQUIRED_FIELDS.get(name, ()):
            if not _nonempty_str(entry.get(key)):
                raise _err(f"{where}.{key} must be a non-empty string")
    return dict(entry)


def _roles(doc: Any) -> dict[str, dict[str, Any]]:
    """Validate v3 ``roles``: every role in :data:`ROLE_NAMES` declared, no other."""
    if not isinstance(doc, dict):
        raise _err(f"roles must be an object declaring {list(ROLE_NAMES)}")
    missing = [n for n in ROLE_NAMES if n not in doc]
    if missing:
        raise _err(
            f"roles missing: {', '.join(missing)}",
            "declare every role, each present or absent, before the lock",
        )
    unknown = sorted(set(doc) - set(ROLE_NAMES))
    if unknown:
        raise _err(f"roles has unknown roles: {', '.join(unknown)}")
    return {name: _role(name, doc[name]) for name in ROLE_NAMES}


def _deployed_format(value: Any) -> str:
    if value not in DEPLOYED_FORMATS:
        raise _err(f"deployed_format must be one of {list(DEPLOYED_FORMATS)}")
    return str(value)


def _references_from_roles(roles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The v2-shaped ``reference_rows`` view of v3 roles (``clef`` = the teacher role)."""
    teacher, stock = roles["teacher"], roles["stock"]
    return {
        "clef": {"model": teacher["model"], "role": "teacher", "variant": teacher["variant"]},
        "stock": {"model": stock["model"], "role": "stock baseline"},
    }


def _common(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_rule": _selection_rule(doc.get("selection_rule")),
        "knob_grid": _grid(doc.get("knob_grid")),
        "upstream_knobs": _upstream(doc.get("upstream_knobs")),
        "gate": _gate(doc.get("gate")),
    }


def _validate_v2(doc: dict[str, Any]) -> DistillPrereg:
    bars = _bars(doc.get("bars"))
    refs = _references(doc.get("reference_rows"))
    common = _common(doc)
    # The normalised v3 view, through the explicit mapping (never written back).
    mapped = v2_to_v3(doc)
    return DistillPrereg(
        bars=bars,
        reference_rows=refs,
        raw=doc,
        schema_version=V2_SCHEMA_VERSION,
        v3_roles=_roles(mapped["roles"]),
        v3_deployed_format=mapped["deployed_format"],
        v3_c17=_c17(mapped["bars"]["c17"], C17_TOP1_BARS_V3),
        **common,
    )


def _validate_v3(doc: dict[str, Any]) -> DistillPrereg:
    if "reference_rows" in doc:
        raise _err(
            "a schema_version 3 pre-registration declares roles, not reference_rows",
            "move the teacher and stock models into roles.teacher / roles.stock",
        )
    roles = _roles(doc.get("roles"))
    fmt = _deployed_format(doc.get("deployed_format"))
    bars = _bars(doc.get("bars"), fmt)
    return DistillPrereg(
        bars=bars,
        reference_rows=_references_from_roles(roles),
        raw=doc,
        schema_version=SCHEMA_VERSION,
        v3_roles=roles,
        v3_deployed_format=fmt,
        v3_c17=copy.deepcopy(bars["c17"]),
        **_common(doc),
    )


def validate(doc: Any) -> DistillPrereg:
    """Validate a parsed distill pre-registration; raise CliError on the first problem.

    Accepts schema v3 and v2 (a v2 file keeps its v2-shaped ``bars`` and ``reference_rows``;
    the v3 accessors read it through :func:`v2_to_v3`). v1 is refused by name.
    """
    if not isinstance(doc, dict):
        raise _err("distill pre-registration must be a JSON object")
    version = doc.get("schema_version")
    if version == 1:
        raise _err(V1_REFUSAL, "re-seed the file from `example_doc_v3()` (schema_version 3)")
    if version == V2_SCHEMA_VERSION:
        return _validate_v2(doc)
    if version == SCHEMA_VERSION:
        return _validate_v3(doc)
    raise _err(f"schema_version must be one of {list(SUPPORTED_SCHEMA_VERSIONS)}")


# --- the normalised (v3) view: what readers call, for a v2 or a v3 file alike ---------------


def roles(reg: DistillPrereg) -> dict[str, dict[str, Any]]:
    """Every role in :data:`ROLE_NAMES` with its ``present`` flag (a copy)."""
    return copy.deepcopy(reg.v3_roles)


def role_present(reg: DistillPrereg, name: str) -> bool:
    if name not in ROLE_NAMES:
        raise _err(f"unknown role {name!r}; roles are {list(ROLE_NAMES)}")
    return bool(reg.v3_roles[name]["present"])


def deployed_format(reg: DistillPrereg) -> str:
    """``quant`` or ``bf16`` (a v2 file is ``quant``)."""
    return reg.v3_deployed_format


def teacher_variant(reg: DistillPrereg) -> str:
    """The pre-registered Clef build the teacher bar is relative to (a v2 file: W4A16)."""
    return str(reg.v3_roles["teacher"]["variant"])


def c17_bars(reg: DistillPrereg) -> dict[str, dict[str, Any]]:
    """The c17 bars per registered build under the v3 names (a copy)."""
    return copy.deepcopy(reg.v3_c17)


def teacher_bar(reg: DistillPrereg, build: str) -> float:
    """The teacher bar (max points under the registered Clef variant) for ``build``."""
    if build not in reg.v3_c17:
        raise _err(f"no c17 bars are registered for build {build!r}")
    return float(reg.v3_c17[build][TEACHER_BAR])


def load(path: Path) -> DistillPrereg:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(2, f"cannot read {path}: {exc}", "check the path") from exc
    except json.JSONDecodeError as exc:
        raise _err(f"{path} is not valid JSON: {exc}") from exc
    return validate(doc)


def lock(prereg_path: Path, run_dir: Path) -> dict[str, Any]:
    """Validate the file and hash it into ``<run_dir>/distill/prereg.lock.json``."""
    prereg_path, run_dir = Path(prereg_path), Path(run_dir)
    lock_path = safepath.contained(run_dir / LOCK_REL, run_dir)  # no symlink out of the run
    if lock_path.exists():
        raise _err(
            f"run already has a distill pre-registration lock at {lock_path}",
            "a changed registration is a recorded deviation, never a re-lock",
        )
    reg = load(prereg_path)
    record = {
        "schema_version": reg.schema_version,  # the file's own version (a v2 lock stays 2)
        "file": prereg_path.name,
        "sha256": sha256_file(prereg_path),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def require_locked(
    prereg_path: Path, run_dir: Path, deviation_id: str | None = None
) -> DistillPrereg:
    """Return the validated pre-registration, or refuse.

    Refuses without a lock, with an invalid file, or when the sha256 no longer
    matches the lock and no ``deviation_id`` is given.
    """
    recorded = prereg_lock.read_lock(Path(run_dir))
    reg = load(prereg_path)
    prereg_lock.check_unchanged(Path(prereg_path), recorded, deviation_id)
    return reg


def _in_grid(value: Any, grid: list[Any]) -> bool:
    return any(
        _is_num(value) and _is_num(g) and math.isclose(value, g, rel_tol=1e-9) or value == g
        for g in grid
    )


def _check_temperature(value: Any, grid: dict[str, list[Any]]) -> None:
    if not isinstance(value, dict):
        raise UnregisteredKnobError("knob temperature must map question kind to a value")
    for kind, v in value.items():
        if kind not in grid or not _in_grid(v, grid[kind]):
            raise UnregisteredKnobError(
                f"knob temperature.{kind}={v!r} is not in the pre-registered grid"
            )


def require_knobs(prereg: DistillPrereg, run_knobs: dict[str, Any]) -> None:
    """Refuse any run knob that the pre-registration does not carry (by name and value)."""
    fixed = {k.name: k.value for k in prereg.upstream_knobs}
    for name, value in run_knobs.items():
        if name == "temperature":
            _check_temperature(value, prereg.knob_grid["temperature"])
        elif name in prereg.knob_grid:
            if not _in_grid(value, prereg.knob_grid[name]):
                raise UnregisteredKnobError(
                    f"knob {name}={value!r} is not in the pre-registered grid"
                )
        elif name in fixed:
            if value != fixed[name]:
                raise UnregisteredKnobError(
                    f"knob {name}={value!r} differs from the pre-registered upstream "
                    f"value {fixed[name]!r}"
                )
        else:
            raise UnregisteredKnobError(f"knob {name} is not in the distill pre-registration")
