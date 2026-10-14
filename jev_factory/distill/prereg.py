"""Distill pre-registration: success bars, selection rule, knob grid, upstream knobs.

Mirrors :mod:`jev_factory.factory.prereg` (validate, hash into a lock, refuse a
changed file without a deviation id) but in its own files, ``distill/prereg.json``
and ``distill/prereg.lock.json``, so the jev-tool pre-registration is untouched.
Every value taken from the upstream Strands Decider training config must cite
the file and key it came from; every knob a run uses must be in the grid.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError

SCHEMA_VERSION = 1
PREREG_REL = "distill/prereg.json"
LOCK_REL = "distill/prereg.lock.json"

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
REFERENCE_ROWS = ("clef", "stock_v21")

#: bar name -> (kind, bound); kind "rate" in [0, 1], "pos" > 0, "flag" must be bool.
TEST_BARS: dict[str, str] = {
    "accuracy_diff_ci_lower_min": "rate",
    "gap_closure_min": "rate",
    "ece_max": "rate",
    "ece_no_worse_than_v21": "flag",
    "confident_wrong_no_worse_than_v21": "flag",
    "latency_ratio_max": "pos",
}
QUANT_BARS: dict[str, str] = {
    "max_accuracy_drop_points": "pos",
    "ece_max": "rate",
    "heal_rounds_max": "int",
}
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
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


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
UPSTREAM_SOURCE = "https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21/raw/main"
TRAIN_CONFIG = "train_config.json"
DECIDER_CONFIG = "strands_decider_config.json"
#: Where the heal recipe in :func:`example_doc` comes from: nvsh's d18, a short bf16
#: continuation (not quantization-aware training) from the healed model's own checkpoint.
HEAL_SOURCE = (
    "nvsh#46 decision d18 (docs/deliveries/2026-09-23-qwen3-5-0-8b-tool-jev-fine-tune-"
    "issue-46.md): "
    "heal = 1 epoch, lr 5e-5, continuing from the model's own checkpoint on the frozen "
    "train set (a plain continued fine-tune, not QAT)"
)
#: v21's ``lora_targets`` (train_config.json and strands_decider_config.json agree).
V21_LORA_TARGETS = (
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


def example_doc() -> dict[str, Any]:
    """A valid pre-registration draft, seeded from strands-decider-2B-hobson-v21.

    Carries every hyperparameter the pointer distill trainer resolves (``init_mode`` in the
    grid; the fixed upstream values, each citing :data:`UPSTREAM_SOURCE` file and key) and
    the selection rule's ``confident_wrong_threshold``, and the one ``heal`` round's knobs
    (:data:`HEAL_SOURCE`) that :mod:`jev_factory.distill.quant` reads. All values are drafts
    for the operator to confirm before the lock.
    """
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
        ("seed", 5, TRAIN_CONFIG, "seed"),
        ("shuffle_options", True, TRAIN_CONFIG, "shuffle_options"),
        ("reverse_score_prob", 0.5, TRAIN_CONFIG, "reverse_score_prob"),
        ("lora_r", 16, TRAIN_CONFIG, "lora_r"),
        ("lora_alpha", 32, TRAIN_CONFIG, "lora_alpha"),
        # train_config.json has no lora_dropout; the model config carries v21's value.
        ("lora_dropout", 0.05, DECIDER_CONFIG, "lora_dropout"),
        ("lora_targets", list(V21_LORA_TARGETS), TRAIN_CONFIG, "lora_targets"),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "bars": {
            "test": {
                "accuracy_diff_ci_lower_min": 0.0,
                "gap_closure_min": 0.5,
                "ece_max": 0.10,
                "ece_no_worse_than_v21": True,
                "confident_wrong_no_worse_than_v21": True,
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
        # on the same frozen train set, with the winner's other knobs. Each value must be in
        # knob_grid (the trainer refuses an unregistered value).
        "heal": {
            "epochs": 1,
            "lr": 5e-5,
            "init_mode": "continue",
            "source": HEAL_SOURCE,
        },
        "reference_rows": {
            "clef": {"model": "Clef 27B", "role": "teacher"},
            "stock_v21": {
                "model": "StrandsAgents/strands-decider-2B-hobson-v21",
                "role": "stock baseline",
            },
        },
        "upstream_knobs": [
            {"name": n, "value": v, "source_file": f, "source_key": k} for n, v, f, k in upstream
        ],
    }


def _check_bar(section: str, name: str, kind: str, value: Any) -> None:
    ok = {
        "rate": _is_num(value) and 0.0 <= value <= 1.0,
        "pos": _is_num(value) and value > 0,
        "int": _is_int(value) and value >= 0,
        "flag": isinstance(value, bool),
    }[kind]
    if not ok:
        raise _err(f"bars.{section}.{name} is invalid ({kind} expected)")


def _bars(doc: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(doc, dict):
        raise _err("bars must be an object")
    out: dict[str, dict[str, Any]] = {}
    for section, spec in (("test", TEST_BARS), ("quant", QUANT_BARS)):
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


def _grid(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise _err("knob_grid must be an object")
    missing = [n for n in KNOB_NAMES if n not in doc]
    if missing:
        raise _err(f"knob_grid missing: {', '.join(missing)}")
    for name in ("alpha", "teacher_weak_floor"):
        if not all(_is_num(v) and 0.0 <= v <= 1.0 for v in _nonempty_list(name, doc[name])):
            raise _err(f"knob_grid.{name} values must be numbers in [0, 1]")
    for name in ("lr",):
        if not all(_is_num(v) and v > 0 for v in _nonempty_list(name, doc[name])):
            raise _err(f"knob_grid.{name} values must be positive numbers")
    if not all(_is_int(v) and v >= 1 for v in _nonempty_list("epochs", doc["epochs"])):
        raise _err("knob_grid.epochs values must be positive integers")
    if not all(_nonempty_str(v) for v in _nonempty_list("data_mix", doc["data_mix"])):
        raise _err("knob_grid.data_mix values must be non-empty strings")
    tf = _nonempty_list("teacher_filter", doc["teacher_filter"])
    if not all(v in TEACHER_FILTERS for v in tf):
        raise _err(f"knob_grid.teacher_filter values must be in {list(TEACHER_FILTERS)}")
    temp = doc["temperature"]
    if not isinstance(temp, dict) or not temp:
        raise _err("knob_grid.temperature must map each question kind to a list")
    for kind, vals in temp.items():
        if not all(_is_num(v) and v > 0 for v in _nonempty_list(f"temperature.{kind}", vals)):
            raise _err(f"knob_grid.temperature.{kind} values must be positive numbers")
    return dict(doc)


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


def validate(doc: Any) -> DistillPrereg:
    """Validate a parsed distill pre-registration; raise CliError on the first problem."""
    if not isinstance(doc, dict):
        raise _err("distill pre-registration must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise _err(f"schema_version must be {SCHEMA_VERSION}")
    return DistillPrereg(
        bars=_bars(doc.get("bars")),
        selection_rule=_selection_rule(doc.get("selection_rule")),
        knob_grid=_grid(doc.get("knob_grid")),
        reference_rows=_references(doc.get("reference_rows")),
        upstream_knobs=_upstream(doc.get("upstream_knobs")),
        raw=doc,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    lock_path = run_dir / LOCK_REL
    if lock_path.exists():
        raise _err(
            f"run already has a distill pre-registration lock at {lock_path}",
            "a changed registration is a recorded deviation, never a re-lock",
        )
    load(prereg_path)
    record = {
        "schema_version": SCHEMA_VERSION,
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
    lock_path = Path(run_dir) / LOCK_REL
    if not lock_path.exists():
        raise _err(
            "no distill pre-registration is locked for this run",
            "lock the distill pre-registration before any training command",
        )
    recorded = json.loads(lock_path.read_text(encoding="utf-8"))
    reg = load(prereg_path)
    if sha256_file(prereg_path) != recorded.get("sha256") and not (deviation_id or "").strip():
        raise _err(
            "the distill pre-registration file changed since it was locked",
            "record a deviation and pass its id (--deviation <id>)",
        )
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
