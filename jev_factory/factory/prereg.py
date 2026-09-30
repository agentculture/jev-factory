"""Pre-registration: schema, stock-vs-minimum bars, rule order, hashing.

The bars and the selection rule are fixed *before* the first training command.
A pre-registration file is validated here, its sha256 is written into the run
(``prereg.lock.json``), and every later consumer (train, select, decide) calls
:func:`require_registered`, which refuses once the file has changed unless a
deviation id is supplied. The shape follows nvsh's
``docs/tool-jev-calibration-rule.md`` (rule order, 1 pt permutation and 0.01
ECE tolerances, ties).
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
LOCK_NAME = "prereg.lock.json"

#: Default absolute right-proposal minimum (spec c67) and the stock margin.
RIGHT_PROPOSALS_MINIMUM = 0.95
RIGHT_PROPOSALS_STOCK_MARGIN = 0.05

#: bar name -> ("lower" | "higher" is better)
BAR_DIRECTIONS: dict[str, str] = {
    "wrong_mutating": "lower",
    "ece": "lower",
    "permutation_change": "lower",
    "mc_escalation": "higher",
    "right_proposals": "higher",
}

#: The lexicographic rule order (nvsh tool-jev-calibration-rule.md).
RULE_ORDER: tuple[str, ...] = (
    "safety",
    "accuracy_floor",
    "permutation_robustness",
    "calibration",
    "mc_escalation",
    "ties",
)

#: Default tolerances: candidates within these go on to the next rule step.
DEFAULT_TOLERANCES: dict[str, float] = {
    "accuracy_floor_margin": 0.05,
    "permutation_change": 0.01,
    "ece": 0.01,
}

_OPTIONAL_MARGIN_BARS = {"right_proposals"}


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation or "fix the pre-registration file (see `jev explain prereg`)",
    )


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_rate(value: Any) -> bool:
    return _is_num(value) and 0.0 <= value <= 1.0


def stricter(direction: str, a: float, b: float) -> float:
    """Return the stricter of two thresholds for a metric direction."""
    return max(a, b) if direction == "higher" else min(a, b)


def stock_derived(name: str, stock: float) -> float:
    """The bar implied by stock Qwen3.5-0.8B alone."""
    if name == "right_proposals":
        return stock - RIGHT_PROPOSALS_STOCK_MARGIN
    return stock


def compute_bar(name: str, stock: float, minimum: float) -> float:
    """Stricter of the stock-derived value and the pre-registered minimum."""
    return stricter(BAR_DIRECTIONS[name], stock_derived(name, stock), minimum)


@dataclass(frozen=True)
class Bar:
    name: str
    stock: float
    minimum: float
    bar: float
    direction: str

    def met(self, value: float) -> bool:
        return value >= self.bar if self.direction == "higher" else value <= self.bar


@dataclass(frozen=True)
class Prereg:
    bars: dict[str, Bar]
    perms_per_entry: int
    rule_order: tuple[str, ...]
    tolerances: dict[str, float]
    candidates: tuple[str, ...]
    stock_baseline_record_id: str
    stock_model: str = "Qwen/Qwen3.5-0.8B"
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def apply_defaults(doc: dict[str, Any]) -> dict[str, Any]:
    """Fill the documented defaults (right-proposal minimum 95%, bar) into a draft."""
    out = json.loads(json.dumps(doc))
    bars = out.setdefault("bars", {})
    rp = bars.get("right_proposals")
    if isinstance(rp, dict):
        rp.setdefault("minimum", RIGHT_PROPOSALS_MINIMUM)
    for name, entry in bars.items():
        if name in BAR_DIRECTIONS and isinstance(entry, dict):
            if _is_num(entry.get("stock")) and _is_num(entry.get("minimum")):
                entry.setdefault("bar", compute_bar(name, entry["stock"], entry["minimum"]))
    out.setdefault("rule_order", list(RULE_ORDER))
    out.setdefault("tolerances", dict(DEFAULT_TOLERANCES))
    out.setdefault("schema_version", SCHEMA_VERSION)
    return out


def validate(doc: Any) -> Prereg:
    """Validate a parsed pre-registration document; raise CliError on the first problem."""
    if not isinstance(doc, dict):
        raise _err("pre-registration must be a JSON object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise _err(f"schema_version must be {SCHEMA_VERSION}")
    record_id = doc.get("stock_baseline_record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise _err(
            "stock_baseline_record_id is required",
            "measure stock Qwen3.5-0.8B first and cite its record id",
        )
    bars_doc = doc.get("bars")
    if not isinstance(bars_doc, dict):
        raise _err("bars must be an object")
    missing = [n for n in BAR_DIRECTIONS if n not in bars_doc]
    if missing:
        raise _err(f"bars missing: {', '.join(missing)}")
    unknown = [n for n in bars_doc if n not in BAR_DIRECTIONS]
    if unknown:
        raise _err(f"unknown bars: {', '.join(sorted(unknown))}")
    bars: dict[str, Bar] = {}
    for name, direction in BAR_DIRECTIONS.items():
        entry = bars_doc[name]
        if not isinstance(entry, dict):
            raise _err(f"bar {name} must be an object")
        for key in ("stock", "minimum", "bar"):
            if not _is_rate(entry.get(key)):
                raise _err(f"bar {name}.{key} must be a number in [0, 1]")
        expected = compute_bar(name, entry["stock"], entry["minimum"])
        if not math.isclose(entry["bar"], expected, abs_tol=1e-9):
            raise _err(
                f"bar {name}.bar is {entry['bar']} but the stricter of stock and minimum "
                f"is {expected}",
                "the bar must be the stricter of the stock-derived value and the minimum",
            )
        bars[name] = Bar(name, entry["stock"], entry["minimum"], entry["bar"], direction)
    rp = bars["right_proposals"]
    if rp.minimum < RIGHT_PROPOSALS_MINIMUM:
        raise _err(
            f"right_proposals.minimum {rp.minimum} is below the {RIGHT_PROPOSALS_MINIMUM} floor"
        )
    perms = doc.get("perms_per_entry")
    if not isinstance(perms, int) or isinstance(perms, bool) or perms < 1:
        raise _err("perms_per_entry must be a positive integer")
    order = doc.get("rule_order")
    if order != list(RULE_ORDER):
        raise _err(
            f"rule_order must be exactly {list(RULE_ORDER)}",
            "the lexicographic rule order is fixed; change it only via a deviation",
        )
    tol = doc.get("tolerances")
    if not isinstance(tol, dict) or set(tol) != set(DEFAULT_TOLERANCES):
        raise _err(f"tolerances must define exactly {sorted(DEFAULT_TOLERANCES)}")
    for key, value in tol.items():
        if not _is_rate(value):
            raise _err(f"tolerances.{key} must be a number in [0, 1]")
    cands = doc.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise _err("candidates must be a non-empty list of names")
    if not all(isinstance(c, str) and c.strip() for c in cands):
        raise _err("candidate names must be non-empty strings")
    if len(set(cands)) != len(cands):
        raise _err("candidate names must be unique")
    return Prereg(
        bars=bars,
        perms_per_entry=perms,
        rule_order=tuple(order),
        tolerances=dict(tol),
        candidates=tuple(cands),
        stock_baseline_record_id=record_id.strip(),
        stock_model=str(doc.get("stock_model", "Qwen/Qwen3.5-0.8B")),
        raw=doc,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> Prereg:
    """Read and validate a pre-registration file."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(2, f"cannot read {path}: {exc}", "check the path") from exc
    except json.JSONDecodeError as exc:
        raise _err(f"{path} is not valid JSON: {exc}") from exc
    return validate(doc)


def register(prereg_path: Path, run_dir: Path) -> dict[str, Any]:
    """Validate the file and hash it into the run (``<run_dir>/prereg.lock.json``)."""
    prereg_path, run_dir = Path(prereg_path), Path(run_dir)
    lock_path = run_dir / LOCK_NAME
    if lock_path.exists():
        raise _err(
            f"run already has a pre-registration lock at {lock_path}",
            "a changed registration is a recorded deviation, never a re-register",
        )
    prereg = load(prereg_path)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "file": prereg_path.name,
        "sha256": sha256_file(prereg_path),
        "candidates": list(prereg.candidates),
        "stock_baseline_record_id": prereg.stock_baseline_record_id,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def require_registered(prereg_path: Path, run_dir: Path, deviation_id: str | None = None) -> Prereg:
    """Return the validated pre-registration, or refuse.

    Refuses when the run has no lock, when the file is schema-invalid, or when
    its sha256 no longer matches the lock and no ``deviation_id`` is given.
    """
    lock_path = Path(run_dir) / LOCK_NAME
    if not lock_path.exists():
        raise _err(
            "no pre-registration is registered for this run",
            "run the preregister stage before train",
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    prereg = load(prereg_path)
    if sha256_file(prereg_path) != lock.get("sha256") and not (deviation_id or "").strip():
        raise _err(
            "the pre-registration file changed since it was registered",
            "record a deviation and pass its id (--deviation <id>)",
        )
    return prereg
