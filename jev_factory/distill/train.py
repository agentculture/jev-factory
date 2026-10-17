"""Distill train, select and calibrate: the knob grid, one winner, its calibration.

One stage, three steps, all reading the **locked** distill pre-registration
(:func:`jev_factory.distill.prereg.require_locked`) and the **verified** student pins
(:func:`jev_factory.distill.pins.verify_pins`); either missing refuses before anything runs.

1. **Train.** Every combination of the registered trainer knobs (:func:`grid_candidates`:
   ``alpha``, per-kind ``temperature``, ``epochs``, ``lr``, ``init_mode``) is planned with
   :func:`jev_factory.backbones.pointer.train_distill.plan_train` on assemble's
   ``train_set.jsonl``, from the verified stock checkpoint in the pin cache. Dry-run by default;
   with ``apply`` each plan runs through ``train_distill.run_train`` and so through
   :func:`jev_factory.factory.gpu.run_gpu_stage` (residency guard, memory cap, floor
   watchdog), one candidate at a time.
2. **Score and calibrate.** Each trained checkpoint is scored on the fit and selection folds
   by an injectable *scorer* (:func:`served_scorer` serves it with ``strands-decider serve``
   and asks it over /v1/systemone). Only the fit and selection folds are accepted
   (:func:`jev_factory.distill.benchmark.collect_folds` refuses held-out and test). A
   checkpoint whose ``strands_decider_config.json`` still carries a served temperature is
   refused: its answers would arrive pre-scaled. Calibration is fitted per question kind
   with :mod:`jev_factory.core.calibration`: the temperature on the **fit** fold only
   (``fit_temperature``), then a per-label vector on the temperature-scaled fit rows
   (``fit_vector``) that is kept only if it lowers **selection**-fold ECE over the
   temperature alone (``select_calibration`` semantics; a tie drops it). Strands' upstream
   ``temperature_by_kind`` is never read: the deployed calibration is this fit.
3. **Select.** Calibrated selection-fold metrics (:func:`jev_factory.distill.metrics.assess`
   plus ECE, each with a seeded bootstrap CI) feed the pre-registered lexicographic rule
   (:func:`apply_rule`). ``<run_dir>/distill/select/summary.json`` names the winner, every
   candidate's metrics and the pre-registration's sha256; ``calibration.json`` beside it is
   the winner's calibration (``temperature_by_kind`` is what the bundle writes into
   ``strands_decider_config.json``).

The rule's steps, as applied here on the selection fold (recorded in the summary as
``rule_semantics``): ``safety`` keeps candidates that answered every fit and selection
question without an error; ``accuracy_floor`` keeps accuracy within
``tolerances.accuracy_floor_margin`` of the best survivor; ``calibration`` keeps calibrated
ECE within ``tolerances.ece`` of the best; ``confident_wrong`` keeps the lowest calibrated
confident-wrong rate (at ``selection_rule.confident_wrong_threshold``); ``ties`` takes the
higher accuracy, then the earlier grid candidate. Stdlib only.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jev_factory.backbones.pointer import layout as player
from jev_factory.backbones.pointer import serve as pserve
from jev_factory.backbones.pointer import train_distill as td
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.core import calibration as cal
from jev_factory.core import metrics as core_metrics
from jev_factory.distill import benchmark as bench
from jev_factory.distill import metrics as dm
from jev_factory.distill import pins as dpins
from jev_factory.distill import prereg as dp
from jev_factory.distill.assemble import OUT_REL as TRAIN_SET_REL
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile
from jev_factory.distill.records import NOUL_OPTIONS, Record
from jev_factory.distill.runspec import Bootstrap, GpuLimits, TrainerEnv
from jev_factory.factory import gpu

TRAIN_REL = "distill/train"
SELECT_REL = "distill/select"
SUMMARY_FILE = "summary.json"
CALIBRATION_FILE = "calibration.json"
PREDICTIONS_FILE = "predictions.json"
CHECKPOINT_DIR = "checkpoint"
SERVED_CONFIG = player.CONFIG
FOLDS = ("fit", "selection")
KINDS = ("choice", "noul", "score")
#: Trainer knobs a candidate sets, in enumeration order (outermost first).
GRID_KNOBS = td.SEARCHED
VARIANT_VECTOR = cal.VARIANT_TEMPERATURE_VECTOR
#: Calibration labels are namespaced so no option name can roll up as an escalate label in
#: ``core.metrics.canonical_label`` (the core fitters are shared with the tool scorer).
_LABEL_PREFIX = "opt:"
_EPS = 1e-12

RULE_SEMANTICS = {
    "safety": "answered every fit and selection question without an error",
    "accuracy_floor": "selection accuracy >= best survivor - tolerances.accuracy_floor_margin",
    "calibration": "calibrated selection ECE <= best survivor + tolerances.ece",
    "confident_wrong": "lowest calibrated selection confident-wrong rate",
    "ties": "higher selection accuracy, then the earlier grid candidate",
}

#: candidate, its checkpoint directory, ``{fold: records}`` -> ``{fold: prediction rows}``.
Scorer = Callable[["Candidate", Path, Mapping[str, Sequence[Record]]], Mapping[str, Any]]


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation or "fix the distill run inputs; a change after the lock is a deviation",
    )


@dataclass(frozen=True)
class Candidate:
    """One grid point: an id (grid order) and the trainer knobs it sets."""

    cid: str
    knobs: dict[str, Any]


# --------------------------------------------------------------------- the grid


def _grid_values(name: str, values: Any) -> list[Any]:
    if name == "temperature":
        kinds = sorted(values)
        return [dict(zip(kinds, combo)) for combo in itertools.product(*(values[k] for k in kinds))]
    return list(values)


def grid_candidates(reg: dp.DistillPrereg) -> list[Candidate]:
    """Every combination of the registered trainer knobs, in a fixed order.

    A trainer knob absent from ``knob_grid`` is not searched (the trainer reads it from
    ``upstream_knobs`` or refuses). Data-shaping knobs (``teacher_filter``,
    ``teacher_weak_floor``, ``data_mix``) are fixed by assemble's training set, not here.
    """
    names = [n for n in GRID_KNOBS if n in reg.knob_grid]
    axes = [_grid_values(n, reg.knob_grid[n]) for n in names]
    out = []
    for i, combo in enumerate(itertools.product(*axes)):
        knobs = dict(zip(names, combo))
        dp.require_knobs(reg, knobs)
        out.append(Candidate(f"c{i:03d}", knobs))
    return out


# --------------------------------------------------------------------- predictions


@dataclass(frozen=True)
class QItem:
    """One answered question: its kind, the offered labels, the distribution and gold."""

    kind: str
    labels: tuple[str, ...]
    probs: dict[str, float]
    gold: str | None


def _gold_label(rec: Record, qid: str) -> str | None:
    q, gold = rec.questions[qid], rec.gold[qid]
    if q.type == "noul":
        if gold.probability_true is not None:
            return NOUL_OPTIONS[1] if gold.probability_true >= 0.5 else NOUL_OPTIONS[0]
        return gold.label
    return gold.label if gold.label in q.names else None


def _answer_probs(rec: Record, qid: str, ans: Mapping[str, Any]) -> dict[str, float] | None:
    q = rec.questions[qid]
    names = q.names
    if q.type == "noul":
        p = ans.get("p_true")
        if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
            return None
        return {NOUL_OPTIONS[0]: 1.0 - float(p), NOUL_OPTIONS[1]: float(p)}
    dist = ans.get("distribution")
    if not isinstance(dist, Mapping) or len(dist) != len(names):
        return None
    # A score answer is keyed by each level's description (the wire sends descriptions only).
    # Map by key, never by position: a stored prediction file has its keys sorted.
    descriptions = tuple(desc for _, desc in q.criteria)
    if set(dist) == set(names):
        values = [dist[n] for n in names]
    elif len(set(descriptions)) == len(names) and set(dist) == set(descriptions):
        values = [dist[d] for d in descriptions]
    else:
        return None
    if not all(isinstance(v, (int, float)) and v >= 0 for v in values):
        return None
    total = math.fsum(values)
    if total <= 0:
        return None
    return {n: float(v) / total for n, v in zip(names, values)}


def question_items(
    records: Sequence[Record], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[QItem], int]:
    """``(answered items, problems)`` for *rows* against *records*.

    A problem is a record with no answer row, a row-level error, a question the model did not
    answer, answered with an error, or answered with an unusable distribution.
    """
    by_id = {str(r.get("record_id")): r for r in rows}
    items: list[QItem] = []
    problems = 0
    for rec in records:
        row = by_id.get(rec.id)
        if row is None or row.get("error"):
            problems += 1
            continue
        answers = row.get("questions") or {}
        for qid, q in rec.questions.items():
            ans = answers.get(qid)
            probs = None if ans is None or ans.get("error") else _answer_probs(rec, qid, ans)
            if probs is None:
                problems += 1
                continue
            items.append(QItem(q.type, q.names, probs, _gold_label(rec, qid)))
    return items, problems


def calibration_rows(items: Sequence[QItem]) -> dict[str, list[tuple[dict[str, float], str]]]:
    """Per kind, ``(label -> probability, gold label)`` for every gold-bearing item."""
    out: dict[str, list[tuple[dict[str, float], str]]] = {k: [] for k in KINDS}
    for item in items:
        if item.gold is not None and item.gold in item.probs:
            out[item.kind].append((dict(item.probs), item.gold))
    return out


# --------------------------------------------------------------------- calibration


def _wrap(rows: Sequence[tuple[Mapping[str, float], str]]) -> list[tuple[dict[str, float], str]]:
    return [
        ({_LABEL_PREFIX + k: float(v) for k, v in cands.items()}, _LABEL_PREFIX + gold)
        for cands, gold in rows
    ]


def _unwrap(vector: Mapping[str, float]) -> dict[str, float]:
    return {k[len(_LABEL_PREFIX) :]: v for k, v in vector.items()}


def _pairs(rows, temperature: float, vector: Mapping[str, float] | None):
    pairs = []
    for cands, gold in rows:
        scaled = cal.apply_scaling(cands, temperature, vector)
        label, conf = core_metrics.top_candidate(scaled)
        pairs.append((conf, label == gold))
    return pairs


def _fit_kind(fit_rows, selection_rows) -> dict[str, Any]:
    fit = [r for r in _wrap(fit_rows) if r[1] in r[0]]
    sel = [r for r in _wrap(selection_rows) if r[1] in r[0]]
    temperature = cal.fit_temperature(fit)  # 1.0 (a no-op) when the fit fold has no rows
    label_sets = {frozenset(cands) for cands, _ in fit}
    closed = len(label_sets) == 1
    scaled_fit = [(cal.temperature_scale(c, temperature), g) for c, g in fit]
    vector = cal.fit_vector(scaled_fit, sorted(next(iter(label_sets)))) if closed else {}
    ece = {
        "raw": core_metrics.ece(_pairs(sel, 1.0, None)),
        "temperature": core_metrics.ece(_pairs(sel, temperature, None)),
        VARIANT_VECTOR: core_metrics.ece(_pairs(sel, temperature, vector)),
    }
    with_vector, temp_only = ece[VARIANT_VECTOR], ece["temperature"]
    keep = bool(vector) and with_vector is not None and temp_only is not None
    keep = keep and with_vector < temp_only
    if closed:
        vector_skipped = None
    elif not fit:
        vector_skipped = "no fit rows"
    else:
        vector_skipped = "open label set: rows offer different labels"
    return {
        "temperature": temperature,
        "vector": _unwrap(vector) if keep else {},
        "vector_fitted": _unwrap(vector),
        "vector_kept": keep,
        "vector_skipped": vector_skipped,
        "fit_examples": len(fit),
        "selection_examples": len(sel),
        "selection_ece": ece,
    }


def fit_calibration(
    fit_rows: Mapping[str, Sequence[tuple[Mapping[str, float], str]]],
    selection_rows: Mapping[str, Sequence[tuple[Mapping[str, float], str]]],
) -> dict[str, Any]:
    """Per-kind calibration: temperature from *fit_rows* only, vector kept on selection ECE.

    The temperature and the vector are fitted on the fit fold (``core.calibration``); the
    selection fold only decides whether the vector is kept (strictly lower ECE than the
    temperature alone). A per-label scale means something only when every row offers the
    same labels, so the vector is fitted only for a kind whose fit rows share one label set
    (always ``noul``; ``score``/``choice`` when their levels/options agree); otherwise the
    kind is calibrated by temperature alone and ``vector_skipped`` says why.
    """
    by_kind = {
        kind: _fit_kind(fit_rows.get(kind, ()), selection_rows.get(kind, ())) for kind in KINDS
    }
    return {
        "temperature_by_kind": {k: v["temperature"] for k, v in by_kind.items()},
        "vector_by_kind": {k: v["vector"] for k, v in by_kind.items() if v["vector_kept"]},
        "by_kind": by_kind,
        "fitted_on": "fit",
        "vector_decided_on": "selection",
        "method": "jev_factory.core.calibration fit_temperature / fit_vector",
        "upstream_temperature_by_kind_used": False,
    }


def calibrate_item(item: QItem, calibration: Mapping[str, Any]) -> QItem:
    t = float(calibration["temperature_by_kind"].get(item.kind, 1.0))
    vector = calibration.get("vector_by_kind", {}).get(item.kind) or None
    return QItem(item.kind, item.labels, cal.apply_scaling(item.probs, t, vector), item.gold)


# --------------------------------------------------------------------- metrics


def _pred_item(item: QItem) -> dm.PredItem:
    probs = tuple(item.probs[label] for label in item.labels)
    total = math.fsum(probs)
    probs = tuple(p / total for p in probs)
    gold = item.labels.index(item.gold) if item.gold in item.labels else None
    return dm.PredItem(item.kind, probs, gold)


def selection_metrics(
    items: Sequence[QItem], *, threshold: float, seed: int, resamples: int
) -> dict[str, Any]:
    """``distill.metrics.assess`` plus ECE (top-probability), each with a bootstrap CI."""
    preds = [_pred_item(i) for i in items]
    out = dm.assess(preds, confident_wrong_threshold=threshold, seed=seed, resamples=resamples)
    graded = [p for p in preds if p.gold_index is not None]
    out["ece"] = {}
    for group in (*dm.KINDS, "all"):
        pairs = [
            (max(p.probs), dm.argmax(p.probs) == p.gold_index)
            for p in graded
            if group == "all" or p.kind == group
        ]
        out["ece"][group] = core_metrics.bootstrap_ci(
            pairs, core_metrics.ece, seed=seed, resamples=resamples
        )
    return out


# --------------------------------------------------------------------- the rule


def apply_rule(
    rule: Mapping[str, Any], scores: Sequence[Mapping[str, Any]]
) -> tuple[str | None, list[dict[str, Any]]]:
    """The pre-registered lexicographic rule over selection-fold scores.

    Each score: ``cid``, ``safe``, ``accuracy``, ``ece``, ``confident_wrong``. *scores* is in
    grid order, which is the last tie-break. Returns the winner's cid (or ``None``) and the
    trail of survivors after each step.
    """
    tol = rule.get("tolerances", {})
    order = {s["cid"]: i for i, s in enumerate(scores)}
    alive = list(scores)
    trail = []
    for step in rule["order"]:
        if step == "safety":
            alive = [s for s in alive if s["safe"]]
        elif step == "accuracy_floor" and alive:
            floor = max(s["accuracy"] for s in alive) - float(tol["accuracy_floor_margin"])
            alive = [s for s in alive if s["accuracy"] >= floor - _EPS]
        elif step == "calibration" and alive:
            limit = min(s["ece"] for s in alive) + float(tol["ece"])
            alive = [s for s in alive if s["ece"] <= limit + _EPS]
        elif step == "confident_wrong" and alive:
            best = min(s["confident_wrong"] for s in alive)
            alive = [s for s in alive if s["confident_wrong"] <= best + _EPS]
        elif step == "ties" and alive:
            alive = sorted(alive, key=lambda s: (-s["accuracy"], order[s["cid"]]))[:1]
        trail.append({"step": step, "survivors": [s["cid"] for s in alive]})
    return (alive[0]["cid"] if alive else None), trail


def _rule_score(cid: str, safe: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    def value(name: str) -> float:
        v = metrics[name]["all"]["value"]
        return float(v) if v is not None else math.nan

    acc, ece, cw = value("accuracy"), value("ece"), value("confident_wrong")
    usable = not any(math.isnan(v) for v in (acc, ece, cw))
    return {
        "cid": cid,
        "safe": bool(safe and usable),
        "accuracy": acc if usable else 0.0,
        "ece": ece if usable else math.inf,
        "confident_wrong": cw if usable else math.inf,
    }


# --------------------------------------------------------------------- checks


def _threshold(reg: dp.DistillPrereg) -> float:
    value = reg.selection_rule.get("confident_wrong_threshold")
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not (ok and 0.0 < value <= 1.0):
        raise _refuse(
            "selection_rule.confident_wrong_threshold is not pre-registered (a number in (0, 1])",
            "register it before the lock; selection never defaults it",
        )
    return float(value)


def _verify_pins(cache_dir: Path | str | None, pins: dpins.StudentPins | None) -> dict[str, Any]:
    if cache_dir is None or not isinstance(pins, dpins.StudentPins):
        raise _refuse(
            "distill training needs verified student pins (a pin cache and StudentPins)",
            "pass cache_dir and pins; verify_pins checks revisions and the stock hashes",
        )
    return dpins.verify_pins(Path(cache_dir), pins)


def _folds(folds: Mapping[str, Any], sealed: Sequence[Any]) -> dict[str, list[Record]]:
    resolved = bench.collect_folds(folds, sealed=sealed)
    missing = [f for f in FOLDS if not resolved.get(f)]
    if missing:
        raise _refuse(
            f"the {' and '.join(missing)} fold is missing or empty",
            "calibration fits on the fit fold and selection runs on the selection fold",
        )
    return {f: resolved[f] for f in FOLDS}


def check_raw_readout(checkpoint: Path | str) -> None:
    """Refuse a checkpoint whose served config would pre-scale its answers.

    The trainer saves ``temperature: 1.0`` and an empty ``temperature_by_kind``; anything
    else (such as the stock model's own upstream calibration) would be applied by the server
    before the fit, so the fitted calibration would compound it.
    """
    path = Path(checkpoint) / SERVED_CONFIG
    if not path.is_file():
        raise _refuse(f"{path} is missing; the trained checkpoint is incomplete")
    config = json.loads(path.read_text(encoding="utf-8"))
    temp = config.get("temperature", 1.0)
    by_kind = config.get("temperature_by_kind") or {}
    neutral = isinstance(temp, (int, float)) and math.isclose(float(temp), 1.0, abs_tol=1e-12)
    if not neutral or by_kind:
        raise _refuse(
            f"{path} carries a served temperature ({temp!r}, temperature_by_kind {by_kind!r})",
            "train with the distill trainer, which resets them; calibration is fitted here",
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------- the stage


def _score_candidate(
    cand: Candidate,
    checkpoint: Path,
    fold_records: Mapping[str, list[Record]],
    scorer: Scorer,
    *,
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    check_raw_readout(checkpoint)
    scored = scorer(cand, checkpoint, {f: list(fold_records[f]) for f in FOLDS})
    items: dict[str, list[QItem]] = {}
    problems = 0
    for fold in FOLDS:
        items[fold], bad = question_items(fold_records[fold], scored.get(fold) or [])
        problems += bad
    _write_json(checkpoint.parent / PREDICTIONS_FILE, {f: scored.get(f) or [] for f in FOLDS})
    calibration = fit_calibration(
        calibration_rows(items["fit"]), calibration_rows(items["selection"])
    )
    calibrated = [calibrate_item(i, calibration) for i in items["selection"]]
    kw = {"threshold": threshold, "seed": seed, "resamples": resamples}
    metrics = selection_metrics(calibrated, **kw)
    return {
        "cid": cand.cid,
        "knobs": cand.knobs,
        "checkpoint": str(checkpoint),
        "problems": problems,
        "safe": problems == 0,
        "calibration": calibration,
        "selection_metrics": metrics,
        "selection_metrics_raw": selection_metrics(items["selection"], **kw),
    }


def init_from_path(cache_dir: Path | str, profile: StudentProfile = DEFAULT_2610) -> Path:
    """Where the trainer starts from: the profile's decider repo inside the pin cache."""
    return Path(cache_dir) / dpins.cache_dirname(profile.decider_repo)


def run_grid(
    *,
    prereg_path: Path | str,
    run_dir: Path | str,
    cache_dir: Path | str | None,
    pins: dpins.StudentPins | None,
    folds: Mapping[str, Any],
    trainer_env: TrainerEnv,
    apply: bool = False,
    scorer: Scorer | None = None,
    sealed: Sequence[Any] = (),
    train_file: Path | str | None = None,
    deviation_id: str | None = None,
    limits: GpuLimits = GpuLimits(),
    bootstrap: Bootstrap = Bootstrap(),
    profile: StudentProfile = DEFAULT_2610,
) -> dict[str, Any]:
    """Plan (dry-run) or run the knob grid, then calibrate and select on apply."""
    trainer_python, decider_repo = trainer_env.python, trainer_env.decider_repo
    memory_max, memory_floor = limits.memory_max, limits.memory_floor
    allow_foreign, timeout = limits.allow_foreign, limits.timeout
    seed, resamples = bootstrap.seed, bootstrap.resamples
    prereg_path, run_dir = Path(prereg_path), Path(run_dir)
    reg = dp.require_locked(prereg_path, run_dir, deviation_id)
    bench.require_base_match(run_dir, deviation_id)
    threshold = _threshold(reg)
    pin_report = _verify_pins(cache_dir, pins)
    fold_records = _folds(folds, sealed)
    init_from = init_from_path(cache_dir, profile)  # type: ignore[arg-type]
    train_file = Path(train_file) if train_file else run_dir / TRAIN_SET_REL
    candidates = grid_candidates(reg)
    plans = {
        c.cid: td.plan_train(
            prereg_path=prereg_path,
            run_dir=run_dir,
            run_knobs=c.knobs,
            train_file=train_file,
            init_from=init_from,
            output_dir=run_dir / TRAIN_REL / c.cid / CHECKPOINT_DIR,
            trainer_python=trainer_python,
            decider_repo=decider_repo,
            deviation_id=deviation_id,
        )
        for c in candidates
    }
    base = {
        "stage": "distill-train",
        "prereg_sha256": dp.sha256_file(prereg_path),
        "pins": pin_report,
        "folds": {f: len(fold_records[f]) for f in FOLDS},
    }
    if not apply:
        return {
            **base,
            "dry_run": True,
            "candidates": [
                {"cid": c.cid, "knobs": c.knobs, "plan": plans[c.cid]} for c in candidates
            ],
        }
    if not memory_max:
        raise _refuse("a GPU training run needs a memory cap", "pass memory_max with apply")
    if scorer is None:
        raise _refuse(
            "apply needs a scorer for the fit and selection folds",
            "pass scorer=served_scorer(...) (or a test double)",
        )
    if not train_file.is_file():
        raise _refuse(
            f"the assembled training set {train_file} (train_set.jsonl) does not exist",
            "run the distill assemble stage first",
        )
    results = []
    for cand in candidates:
        td.run_train(
            plans[cand.cid],
            apply=True,
            allow_foreign=allow_foreign,
            memory_max=memory_max,
            memory_floor=memory_floor,
            timeout=timeout,
        )
        checkpoint = Path(plans[cand.cid]["output_dir"])
        result = _score_candidate(
            cand,
            checkpoint,
            fold_records,
            scorer,
            threshold=threshold,
            seed=seed,
            resamples=resamples,
        )
        _write_json(checkpoint.parent / CALIBRATION_FILE, result["calibration"])
        results.append(result)
    scores = [_rule_score(r["cid"], r["safe"], r["selection_metrics"]) for r in results]
    winner, trail = apply_rule(reg.selection_rule, scores)
    summary = {
        **base,
        "dry_run": False,
        "fold": "selection",
        "fold_ids": {f: [r.id for r in fold_records[f]] for f in FOLDS},
        "selection_rule": reg.selection_rule,
        "rule_semantics": RULE_SEMANTICS,
        "rule_trail": trail,
        "winner": winner,
        "candidates": results,
    }
    out_dir = run_dir / SELECT_REL
    _write_json(out_dir / SUMMARY_FILE, summary)
    if winner is not None:
        chosen = next(r for r in results if r["cid"] == winner)
        _write_json(
            out_dir / CALIBRATION_FILE,
            {
                **chosen["calibration"],
                "candidate": winner,
                "checkpoint": chosen["checkpoint"],
                "prereg_sha256": base["prereg_sha256"],
            },
        )
    return summary


# --------------------------------------------------------------------- served scorer


def _stop_job(jobdir: Path, name: str, grace: float = 30.0) -> None:
    pid_file = jobdir / f"{name}.pid"
    if not pid_file.is_file():
        return
    pid = int(pid_file.read_text().strip())
    try:
        os.killpg(pid, signal.SIGTERM)  # start_detached runs the server in its own session
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if pserve.job_status(jobdir, name)["state"] != "running":
            return
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def served_scorer(
    *,
    env: Any,
    jobdir: Path | str,
    port: int,
    base_revision: str,
    device: str | None = None,
    ready_timeout: float = 600.0,
    allow_foreign: bool = False,
) -> Scorer:
    """A scorer that serves each checkpoint with ``strands-decider serve`` on 127.0.0.1.

    The checkpoint is exported to a verified ``hobson-hf-export/1`` bundle beside it, the GPU
    residency guard runs before the server starts, every fit/selection record is asked over
    /v1/systemone (:func:`jev_factory.distill.benchmark.benchmark_model`) and the server is
    always stopped before the next candidate.
    """
    jobdir = Path(jobdir)
    base_url = f"http://{pserve.LOOPBACK}:{int(port)}"

    def score(
        cand: Candidate, checkpoint: Path, fold_records: Mapping[str, Sequence[Record]]
    ) -> dict[str, list[dict[str, Any]]]:
        bundle = Path(checkpoint).parent / "bundle"
        if not bundle.exists():
            player.write_layout(
                checkpoint,
                bundle,
                base_revision=base_revision,
                extra={"distill_candidate": cand.cid, "knobs": cand.knobs},
            )
        name = f"distill-{cand.cid}"
        gpu.assert_gpu_free("measure", allow_foreign=allow_foreign)
        target = bench.Target(
            name=name,
            base_url=base_url,
            model=name,
            start=lambda: pserve.launch(jobdir, name, env, bundle, port=port, device=device),
            ready=lambda: pserve.wait_ready(
                base_url, timeout=ready_timeout, jobdir=jobdir, name=name
            ),
            stop=lambda: _stop_job(jobdir, name),
        )
        out = bench.benchmark_model(
            target,
            fold_records,
            memory_probe=None,
            client=pserve.PointerClient(base_url, model=name),
        )
        return {f: [p for p in out["predictions"] if p["fold"] == f] for f in fold_records}

    return score
