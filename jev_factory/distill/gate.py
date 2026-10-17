"""Distill decision gate (deviation d4): answer or abstain, per typed-question kind.

The jev-like contract is a calibrated distribution followed by a deterministic gate. Typed
questions (``choice`` / ``noul`` / ``score``) have no read-only/mutating split, so the gate
here decides, per question kind, whether to **answer** (the top option) or **abstain**, from
three thresholds (:class:`KindThresholds`):

* ``floor``: abstain when the top probability is below it;
* ``margin``: abstain when the top probability minus the second is below it;
* ``max_entropy``: abstain when the normalised entropy is above it.

Checked in that order; a threshold left ``None`` never fires, so an all-``None`` set answers
every argmax. The comparisons and the entropy are those of
:func:`jev_factory.core.gate.decide` (``p < floor``, ``margin < margin``, ``entropy >
max_entropy``; :func:`jev_factory.core.gate.normalized_entropy` is reused as is), without
its escalate/explain controls or Domain, which typed questions do not have.

**Fitting** (:func:`fit_gate`) follows :mod:`jev_factory.core.sweep_gate`'s approach: every
combination of the pre-registered grid values (``itertools.product``, as
``build_threshold_grid`` does, ``None`` = disabled) is evaluated on the **fit fold only**
(``sweep_gate.fit_gate``'s rule, nvsh D47), and a pre-registered rule picks one point. It
does not route through ``core.metrics.compute``/``sweep_gate.evaluate``, which count
tool-scorer outcomes (right proposals, wrong mutating); here each kind's point is the one
with the best **selective accuracy** (accuracy over answered questions) whose **coverage**
(share answered) meets the kind's pre-registered minimum (``gate`` in
:mod:`jev_factory.distill.prereg`), ties to higher coverage, then grid order. The chosen
thresholds' coverage and selective accuracy are then reported on the **selection** fold,
each with a seeded bootstrap CI (:func:`jev_factory.core.metrics.bootstrap_ci`). Held-out
and test are never read: :func:`run_gate` resolves folds through
:func:`jev_factory.distill.benchmark.collect_folds`, which refuses them.

**The stage** (:func:`run_gate`) gates the deployed build: its fit and selection predictions
are not persisted by :mod:`jev_factory.distill.quant` (``measurements.json`` keeps only the
metrics), so an injected *measurer* (``quant.Measurer``'s shape, e.g.
:func:`jev_factory.distill.quant.served_measurer`) asks the deployed artifact again, and the
shipped ``distill/quant/calibration.json`` is applied before fitting. ``distill/gate/gate.json``
records the thresholds, the objective, the selection coverage and selective accuracy with
CIs, and the sha256 of the pre-registration and of the calibration it was fitted on. Under a
bf16 ship (``deployed_format bf16``) the deployed build is the train stage's winner
(:mod:`jev_factory.distill.deployed`): its fit and selection predictions are persisted in
``distill/train/<winner>/predictions.json`` and ``distill/select/calibration.json`` is applied,
so no measurer is asked. Dry-run by default: it writes nothing and asks nothing. Stdlib only.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core import metrics as core_metrics
from jev_factory.core.gate import normalized_entropy
from jev_factory.distill import metrics as dm
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as dtrain

ANSWER = "answer"
ABSTAIN = "abstain"
KINDS = dp.GATE_KINDS
THRESHOLD_NAMES = dp.GATE_THRESHOLDS
GATE_REL = "distill/gate"
GATE_FILE = "gate.json"
STAGE = "distill-gate"
_EPS = 1e-12

RULE = (
    "per kind, on the calibrated fit fold only: keep grid points whose coverage >= the "
    "pre-registered min_coverage, then the highest selective accuracy, then the higher "
    "coverage, then the earlier grid point"
)
REUSE = {
    "jev_factory.core.gate": "normalized_entropy; decide()'s threshold comparisons and order",
    "jev_factory.core.sweep_gate": "grid = itertools.product of every knob's values "
    "(build_threshold_grid); fit on the fit fold only (fit_gate)",
    "jev_factory.core.metrics": "bootstrap_ci (seeded percentile CIs)",
}


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(EXIT_USER_ERROR, message, remediation or "fix the gate inputs and rerun")


# --------------------------------------------------------------------- decide


def _check_threshold(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not (ok and math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be None or a number in [0, 1], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class KindThresholds:
    """One question kind's gate; ``None`` disables a threshold."""

    floor: float | None = None
    margin: float | None = None
    max_entropy: float | None = None

    def __post_init__(self) -> None:
        for name in THRESHOLD_NAMES:
            object.__setattr__(self, name, _check_threshold(name, getattr(self, name)))

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> "KindThresholds":
        doc = doc or {}
        return cls(**{name: doc.get(name) for name in THRESHOLD_NAMES})


@dataclass(frozen=True)
class GateDecision:
    """``answer`` or ``abstain`` for one distribution, with what it was decided on."""

    outcome: str
    index: int
    label: str | None
    top: float
    margin: float
    entropy: float
    reason: str | None = None


def decide(
    distribution: Sequence[float] | Mapping[str, float], thresholds: KindThresholds
) -> GateDecision:
    """Gate one calibrated distribution. Deterministic; ties go to the earlier option.

    *distribution* is a probability sequence (option order) or a label -> probability
    mapping (insertion order); it must have at least two entries summing to 1.
    """
    if isinstance(distribution, Mapping):
        labels: list[str] | None = [str(k) for k in distribution]
        values = list(distribution.values())
    else:
        labels, values = None, list(distribution)
    probs = dm._check_distribution("distribution", values)
    index = dm.argmax(probs)
    top = probs[index]
    second = max(p for i, p in enumerate(probs) if i != index)
    margin = top - second
    entropy = normalized_entropy({str(i): p for i, p in enumerate(probs)}, len(probs))
    reason = None
    if thresholds.floor is not None and top < thresholds.floor:
        reason = "floor"
    elif thresholds.margin is not None and margin < thresholds.margin:
        reason = "margin"
    elif thresholds.max_entropy is not None and entropy > thresholds.max_entropy:
        reason = "entropy"
    return GateDecision(
        ABSTAIN if reason else ANSWER,
        index,
        labels[index] if labels else None,
        top,
        margin,
        entropy,
        reason,
    )


# --------------------------------------------------------------------- statistics

#: One gated question: ``(answered, correct)``; an abstention or no answer is ``(False, False)``.
Unit = tuple[bool, bool]


def unit(item: dm.PredItem, thresholds: KindThresholds) -> Unit:
    answered = decide(item.probs, thresholds).outcome == ANSWER
    return answered, answered and dm.argmax(item.probs) == item.gold_index


def coverage(units: Sequence[Unit]) -> float | None:
    return (sum(1 for a, _ in units if a) / len(units)) if units else None


def selective_accuracy(units: Sequence[Unit]) -> float | None:
    answered = [ok for a, ok in units if a]
    return (sum(1 for ok in answered if ok) / len(answered)) if answered else None


def summarize(
    units: Sequence[Unit],
    *,
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED,
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Coverage and selective accuracy of *units*, each a seeded ``bootstrap_ci`` dict."""
    units = list(units)
    return {
        "coverage": core_metrics.bootstrap_ci(units, coverage, seed=seed, resamples=resamples),
        "selective_accuracy": core_metrics.bootstrap_ci(
            units, selective_accuracy, seed=seed, resamples=resamples
        ),
        "n": len(units),
        "answered": sum(1 for a, _ in units if a),
        "abstained": sum(1 for a, _ in units if not a),
    }


# --------------------------------------------------------------------- fitting


def threshold_grid(grid: Mapping[str, Sequence[float | None]]) -> list[KindThresholds]:
    """Every combination of *grid*'s values, in ``floor``-major product order."""
    if not isinstance(grid, Mapping) or sorted(grid) != sorted(THRESHOLD_NAMES):
        raise _refuse(f"the gate grid must carry exactly {list(THRESHOLD_NAMES)}")
    lists = [list(grid[name]) for name in THRESHOLD_NAMES]
    if not all(lists):
        raise _refuse("every gate grid knob needs at least one value")
    try:
        return [KindThresholds(*combo) for combo in itertools.product(*lists)]
    except ValueError as exc:
        raise _refuse(f"invalid gate grid value: {exc}") from exc


def _min_coverage(objective: Mapping[str, Any]) -> dict[str, float]:
    if objective.get("objective") not in dp.GATE_OBJECTIVES:
        raise _refuse(f"gate objective must be one of {list(dp.GATE_OBJECTIVES)}")
    cov = objective.get("min_coverage")
    if not isinstance(cov, Mapping) or any(k not in cov for k in KINDS):
        raise _refuse(f"gate objective needs a min_coverage for each of {list(KINDS)}")
    return {k: float(cov[k]) for k in KINDS}


def _choose(kind: str, items: Sequence[dm.PredItem], points, minimum: float):
    scored = []
    for index, th in enumerate(points):
        units = [unit(i, th) for i in items]
        scored.append((index, th, coverage(units), selective_accuracy(units)))
    feasible = [s for s in scored if s[3] is not None and s[2] + _EPS >= minimum]
    if not feasible:
        best = max(s[2] for s in scored)
        raise _refuse(
            f"no gate grid point for {kind} reaches the pre-registered coverage minimum "
            f"{minimum} on the fit fold (best coverage {best:.4f})",
            "widen the registered grid or lower the minimum, through a recorded deviation",
        )
    index, th, cov, acc = max(feasible, key=lambda s: (s[3], s[2], -s[0]))
    return th, {
        "coverage": cov,
        "selective_accuracy": acc,
        "n": len(items),
        "grid_points": len(points),
        "feasible_points": len(feasible),
        "grid_index": index,
    }


def fit_gate(
    fit_rows: Sequence[dm.PredItem],
    selection_rows: Sequence[dm.PredItem],
    *,
    objective: Mapping[str, Any],
    grid: Mapping[str, Sequence[float | None]],
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED,
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Per-kind thresholds chosen on CALIBRATED *fit_rows*; reported on *selection_rows*.

    Only gold-bearing rows count. *selection_rows* never influence the choice. A kind with no
    gold-bearing fit row is left ungated (every threshold ``None``) and says why.
    """
    minimum = _min_coverage(objective)
    points = threshold_grid(grid)
    by_kind: dict[str, Any] = {}
    for kind in KINDS:
        fit = [r for r in fit_rows if r.kind == kind and r.gold_index is not None]
        sel = [r for r in selection_rows if r.kind == kind and r.gold_index is not None]
        entry: dict[str, Any] = {"min_coverage": minimum[kind]}
        if fit:
            th, fit_stats = _choose(kind, fit, points, minimum[kind])
            entry.update(fitted=True, reason=None, fit=fit_stats)
        else:
            th = KindThresholds()
            entry.update(
                fitted=False,
                reason=f"no gold-bearing {kind} question on the fit fold; the kind is ungated",
                fit={"n": 0, "coverage": None, "selective_accuracy": None},
            )
        entry["thresholds"] = asdict(th)
        entry["selection"] = summarize([unit(r, th) for r in sel], seed=seed, resamples=resamples)
        by_kind[kind] = entry
    return {
        "by_kind": by_kind,
        "objective": dict(objective),
        "grid": {k: list(v) for k, v in grid.items()},
        "rule": RULE,
        "seed": seed,
        "resamples": resamples,
    }


def thresholds_by_kind(result: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    """``{kind: {floor, margin, max_entropy}}`` of a :func:`fit_gate` result or ``gate.json``."""
    if "thresholds_by_kind" in result:
        return {k: dict(v) for k, v in result["thresholds_by_kind"].items()}
    return {k: dict(v["thresholds"]) for k, v in result["by_kind"].items()}


# --------------------------------------------------------------------- the stage

#: deployed format, artifact, ``{fold: records}`` -> ``{"predictions": {fold: rows}, ...}``.
Measurer = Callable[[str, Path, Mapping[str, Sequence[Any]]], Mapping[str, Any]]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _artifact(run_dir: Path, recorded: Any) -> Path:
    if not recorded:
        raise _refuse("the quant manifest names no deployed artifact", "run distill quant first")
    path = Path(str(recorded))
    return path if path.is_absolute() else run_dir / path


def _measured_predictions(
    run_dir: Path,
    student: Mapping[str, Any],
    measurer: Measurer | None,
    fold_records: Mapping[str, Sequence[Any]],
) -> Mapping[str, Any]:
    if measurer is None:
        raise _refuse(
            "apply needs a measurer to ask the deployed build on the fit and selection folds",
            "pass measurer=quant.served_measurer(...) (or a test double)",
        )
    artifact = _artifact(run_dir, student["artifact"])
    measured = measurer(
        student["format"], artifact, {f: list(fold_records[f]) for f in dtrain.FOLDS}
    )
    return measured.get("predictions") or {}


def run_gate(
    run_dir: Path | str,
    *,
    prereg_path: Path | str,
    folds: Mapping[str, Any],
    apply: bool = False,
    measurer: Measurer | None = None,
    sealed: Sequence[Any] = (),
    deviation_id: str | None = None,
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED,
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Plan (dry-run) or fit the deployed build's per-kind gate into ``distill/gate/gate.json``.

    Under ``deployed_format bf16`` the deployed build is the train stage's winner
    (:mod:`jev_factory.distill.deployed`): its persisted fit and selection predictions are
    used and *measurer* is not called."""
    from jev_factory.distill import deployed as ddeployed  # lazy: final -> report -> gate

    run_dir, prereg_path = Path(run_dir), Path(prereg_path)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    if reg.gate is None:
        raise _refuse(
            "the locked distill pre-registration registers no gate objective or grid",
            "register a gate block before the lock, or record a deviation",
        )
    student = ddeployed.deployed_student(run_dir, reg)
    bf16 = student.get("deployed_format") == ddeployed.BF16
    fold_records = dtrain._folds(folds, sealed)
    out_path = run_dir / GATE_REL / GATE_FILE
    base = {
        "stage": STAGE,
        "prereg_sha256": dp.sha256_file(prereg_path),
        "calibration_sha256": student["calibration_sha256"],
        "calibration_file": student["calibration_file"],
        "deployed_format": student["format"],
        "deployed_artifact": str(student["artifact"]),
        "objective": {k: reg.gate[k] for k in ("objective", "min_coverage")},
        "folds": {f: len(fold_records[f]) for f in dtrain.FOLDS},
        "output": str(out_path),
    }
    if not apply:
        return {**base, "dry_run": True, "grid_points": len(threshold_grid(reg.gate["grid"]))}
    if bf16:
        predictions = ddeployed.read_predictions(student)
    else:
        predictions = _measured_predictions(run_dir, student, measurer, fold_records)
    rows: dict[str, list[dm.PredItem]] = {}
    for fold in dtrain.FOLDS:
        items, problems = dtrain.question_items(fold_records[fold], predictions.get(fold) or [])
        if problems:
            raise _refuse(
                f"the deployed build left {problems} {fold}-fold question(s) unanswered",
                "the gate is fitted on complete folds; fix the serve path first",
            )
        calibrated = [dtrain.calibrate_item(i, student["calibration"]) for i in items]
        rows[fold] = [dtrain._pred_item(i) for i in calibrated]
    fitted = fit_gate(
        rows["fit"],
        rows["selection"],
        objective=reg.gate,
        grid=reg.gate["grid"],
        seed=seed,
        resamples=resamples,
    )
    by_kind = fitted["by_kind"]
    doc = {
        **base,
        "dry_run": False,
        "fitted": True,
        "thresholds_by_kind": thresholds_by_kind(fitted),
        "coverage": {k: v["selection"]["coverage"] for k, v in by_kind.items()},
        "selective_accuracy": {k: v["selection"]["selective_accuracy"] for k, v in by_kind.items()},
        "by_kind": by_kind,
        "grid": fitted["grid"],
        "rule": RULE,
        "fitted_on": "fit",
        "reported_on": "selection",
        "seed": seed,
        "resamples": resamples,
        "reuse": REUSE,
    }
    _write_json(out_path, doc)
    return doc
