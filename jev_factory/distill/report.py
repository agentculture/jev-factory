"""The distill final report: a paired gate over Clef, stock v21 and the deployed student.

Built from the single final measurement (:mod:`jev_factory.distill.final`), never from a new
one. Every item of a pool is one gold-bearing question ``(record id, qid)``; the three models
are **resampled jointly**: each bootstrap draw picks one list of item indices and every model
(and every difference between models) is computed on those same indices, so the CIs are on
paired per-item differences.

Pools, per sealed side (``test`` and ``held_out``):

* ``headline``: records whose gold is **human** (``Record.gold_provenance == "human"``) and
  whose source :func:`jev_factory.distill.contamination.check_sources` marks ``clean``. The
  c42 bars are computed here, on the test side only (held-out is reported, not gated).
* ``overlapping`` and ``v21_evaluated``: human-gold records from sources v21 trained on (or a
  Clef Decision Index suite), or that v21 only evaluated on. Reported separately, not gated.
* ``model_generated``: records whose gold a model produced (LocalLLaMA/typed-decisions). Their
  accuracy is **agreement** with that gold, reported as a secondary table, never gated.

Every bar is read from the locked distill pre-registration (``bars.test`` for c42,
``bars.quant`` for c43) and carries its value, n and CI. c43 (quant drop vs its own bf16,
heal count, recalibrated ECE) is read from ``distill/quant/quant_manifest.json``, not
re-measured. A bar that cannot be computed (no items, no v21-to-Clef gap) is a miss with the
reason; a miss is recorded as a miss and the report is still written. Student accuracy above
v21 with ECE above v21 is flagged as a regression. Latency (median/p95) and peak memory are
reported for all three models with the host they ran on, so "same box" is checkable.

A failed or unanswered question counts as wrong for accuracy and confident-wrong and is left
out of ECE (it has no confidence); the per-model ``unanswered`` count says how many.
Stdlib only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jev_factory.core import metrics as core_metrics
from jev_factory.distill import contamination
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as dtrain
from jev_factory.distill.records import JEV_BENCH_SOURCE, Record

CLEF = "clef"
V21 = "stock_v21"
STUDENT = "student"
MODELS = (CLEF, V21, STUDENT)
#: The extra row when Clef has a calibration of its own (reported, never gated).
CLEF_CALIBRATED = "clef_calibrated"

GATED_SIDE = "test"
SIDES = ("test", "held_out")
HEADLINE = "headline"
OVERLAPPING = "overlapping"
V21_EVALUATED = "v21_evaluated"
MODEL_GENERATED = "model_generated"
POOLS = (HEADLINE, OVERLAPPING, V21_EVALUATED, MODEL_GENERATED)

PASSED = "pass"  # a bar result, not a secret
MISSED = "miss"

Unit = tuple[bool, "float | None"]


# --------------------------------------------------------------------- sources


def contamination_key(source: str) -> str:
    """The name the contamination check matches for *source*.

    jev-bench records carry ``Praveenrajus/jev-bench:<config>``; the check aliases configs
    only under the ``jev-bench:<config>`` form, so the record source is mapped onto it.
    """
    prefix = JEV_BENCH_SOURCE + ":"
    if source.startswith(prefix):
        return contamination.JEV_BENCH_PREFIX + source[len(prefix) :]
    return source


def source_table(
    records: Sequence[Record], manifest: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Per source: contamination status, gold provenance, pool and the matches behind it."""
    keys = {r.source: contamination_key(r.source) for r in records}
    hf_ids = {keys[r.source]: r.hf_id for r in records if r.hf_id}
    checked = contamination.check_sources(
        set(keys.values()), manifest=dict(manifest) if manifest is not None else None, hf_ids=hf_ids
    )["results"]
    table: dict[str, dict[str, Any]] = {}
    for rec in records:
        row = table.setdefault(
            rec.source,
            {
                "checked_as": keys[rec.source],
                "status": checked[keys[rec.source]]["status"],
                "matches": checked[keys[rec.source]]["matches"],
                "gold_provenance": set(),
                "records": 0,
            },
        )
        row["gold_provenance"].add(rec.gold_provenance)
        row["records"] += 1
    for row in table.values():
        provenance = sorted(row["gold_provenance"])
        row["gold_provenance"] = provenance
        row["headline"] = provenance == ["human"] and row["status"] == contamination.CLEAN
    return table


def pool_of(rec: Record, sources: Mapping[str, Mapping[str, Any]]) -> str:
    if rec.gold_provenance != "human":
        return MODEL_GENERATED
    status = sources[rec.source]["status"]
    if status == contamination.CLEAN:
        return HEADLINE
    return OVERLAPPING if status == contamination.OVERLAPPING else V21_EVALUATED


# --------------------------------------------------------------------- per-item outcomes


def item_keys(records: Sequence[Record]) -> list[tuple[str, str]]:
    """Every gold-bearing question ``(record id, qid)`` of *records*, in record order."""
    keys = []
    for rec in records:
        for qid in rec.questions:
            if dtrain._gold_label(rec, qid) is not None:
                keys.append((rec.id, qid))
    return keys


def outcomes(
    records: Sequence[Record],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> dict[tuple[str, str], Unit]:
    """``(record id, qid) -> (correct, top probability or None)`` for one model.

    *calibration* (``temperature_by_kind`` / ``vector_by_kind``) is applied when given; an
    unanswered question is ``(False, None)``.
    """
    by_id = {str(r.get("record_id")): r for r in rows}
    out: dict[tuple[str, str], Unit] = {}
    for rec in records:
        row = by_id.get(rec.id) or {}
        answers = row.get("questions") or {}
        for qid, q in rec.questions.items():
            gold = dtrain._gold_label(rec, qid)
            if gold is None:
                continue
            ans = answers.get(qid)
            probs = None
            if ans is not None and not ans.get("error") and not row.get("error"):
                probs = dtrain._answer_probs(rec, qid, ans)
            if probs is None:
                out[(rec.id, qid)] = (False, None)
                continue
            item = dtrain.QItem(q.type, q.names, probs, gold)
            if calibration:
                item = dtrain.calibrate_item(item, calibration)
            label, conf = core_metrics.top_candidate(
                {name: item.probs[name] for name in item.labels}
            )
            out[(rec.id, qid)] = (label == gold, conf)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def model_stats(units: Sequence[Unit], threshold: float) -> dict[str, float | None]:
    """Accuracy, ECE (answered items only) and confident-wrong rate of one model's items."""
    answered = [(conf, ok) for ok, conf in units if conf is not None]
    return {
        "accuracy": _mean([1.0 if ok else 0.0 for ok, _ in units]),
        "ece": core_metrics.ece(answered),
        "confident_wrong": _mean(
            [
                1.0 if conf is not None and conf >= threshold and not ok else 0.0
                for ok, conf in units
            ]
        ),
    }


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def gap_closure(student: float | None, v21: float | None, clef: float | None) -> float | None:
    """Share of the v21-to-Clef accuracy gap the student closes; ``None`` without a gap."""
    if student is None or v21 is None or clef is None or clef - v21 <= 0:
        return None
    return (student - v21) / (clef - v21)


def joint_stats(
    per_model: Mapping[str, Sequence[Unit]], threshold: float
) -> Callable[[Sequence[int]], dict[str, float | None]]:
    """The statistic of one draw: every model and every difference on the SAME indices."""

    def stat(indices: Sequence[int]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        stats = {}
        for name, units in per_model.items():
            stats[name] = model_stats([units[i] for i in indices], threshold)
            for metric, value in stats[name].items():
                out[f"{name}.{metric}"] = value
        if {CLEF, V21, STUDENT} <= set(stats):
            s, v, c = stats[STUDENT], stats[V21], stats[CLEF]
            out["accuracy_diff_student_v21"] = _diff(s["accuracy"], v["accuracy"])
            out["accuracy_diff_clef_v21"] = _diff(c["accuracy"], v["accuracy"])
            out["gap_closure"] = gap_closure(s["accuracy"], v["accuracy"], c["accuracy"])
            out["ece_diff_student_v21"] = _diff(s["ece"], v["ece"])
            out["confident_wrong_diff_student_v21"] = _diff(
                s["confident_wrong"], v["confident_wrong"]
            )
        return out

    return stat


def paired_bootstrap(
    n: int,
    stat: Callable[[Sequence[int]], Mapping[str, float | None]],
    *,
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED,
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, dict[str, Any]]:
    """Percentile 95% CI of every statistic of *stat*, one index draw shared by all of them.

    Mirrors :func:`jev_factory.core.metrics.bootstrap_ci` (seeded, same percentile ranks, a
    ``None`` draw skipped), but *stat* sees the indices, so models are resampled jointly.
    """
    if n == 0:
        return {}
    point = stat(range(n))
    draws: dict[str, list[float]] = {name: [] for name in point}
    if n > 1:
        rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
        for _ in range(resamples):
            sample = stat([rng.randrange(n) for _ in range(n)])
            for name, value in sample.items():
                if value is not None:
                    draws.setdefault(name, []).append(value)
    out: dict[str, dict[str, Any]] = {}
    for name, value in point.items():
        values = sorted(draws.get(name, []))
        if n == 1:
            lo = hi = value
        elif values:
            lo = values[int(round(0.025 * (len(values) - 1)))]
            hi = values[int(round(0.975 * (len(values) - 1)))]
        else:
            lo = hi = None
        out[name] = {"n": n, "value": value, "ci_low": lo, "ci_high": hi}
    return out


# --------------------------------------------------------------------- bars


def _bar(
    name: str,
    threshold: Any,
    stat: Mapping[str, Any] | None,
    passed: bool | None,
    *,
    use: str = "value",
    reason: str | None = None,
) -> dict[str, Any]:
    stat = stat or {}
    value = stat.get(use)
    if passed is None or value is None:
        passed = False
        reason = reason or "not computable from the final measurement"
    return {
        "bar": name,
        "threshold": threshold,
        "compared": use,
        "value": value,
        "point": stat.get("value"),
        "ci_low": stat.get("ci_low"),
        "ci_high": stat.get("ci_high"),
        "n": stat.get("n", 0),
        "result": PASSED if passed else MISSED,
        "reason": reason,
    }


def _le(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a <= b


def latency_ratio(runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Student median latency over Clef median latency, from the same final run."""
    s = ((runtime.get(STUDENT) or {}).get("latency") or {}).get("median_s")
    c = ((runtime.get(CLEF) or {}).get("latency") or {}).get("median_s")
    n = ((runtime.get(STUDENT) or {}).get("latency") or {}).get("n", 0)
    value = s / c if s is not None and c else None
    return {"n": n, "value": value, "ci_low": None, "ci_high": None}


def c42_bars(
    stats: Mapping[str, Mapping[str, Any]],
    bars: Mapping[str, Any],
    runtime: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every ``bars.test`` bar, computed from one pool's paired-bootstrap statistics."""
    diff = stats.get("accuracy_diff_student_v21")
    gap = stats.get("gap_closure")
    s_ece, v_ece = stats.get(f"{STUDENT}.ece"), stats.get(f"{V21}.ece")
    s_cw = stats.get(f"{STUDENT}.confident_wrong")
    v_cw = stats.get(f"{V21}.confident_wrong")
    lat = latency_ratio(runtime)

    def value(stat: Mapping[str, Any] | None) -> float | None:
        return (stat or {}).get("value")

    lower = (diff or {}).get("ci_low")
    out = [
        _bar(
            "accuracy_diff_ci_lower_min",
            bars["accuracy_diff_ci_lower_min"],
            diff,
            None if lower is None else lower > bars["accuracy_diff_ci_lower_min"],
            use="ci_low",
        ),
        _bar(
            "gap_closure_min",
            bars["gap_closure_min"],
            gap,
            None if value(gap) is None else value(gap) >= bars["gap_closure_min"],
            reason=(
                "no v21-to-Clef accuracy gap to close"
                if value(gap) is None and value(stats.get("accuracy_diff_clef_v21")) is not None
                else None
            ),
        ),
        _bar("ece_max", bars["ece_max"], s_ece, _le(value(s_ece), bars["ece_max"])),
    ]
    for name, mine, theirs in (
        ("ece_no_worse_than_v21", s_ece, v_ece),
        ("confident_wrong_no_worse_than_v21", s_cw, v_cw),
    ):
        if bars[name]:
            out.append(
                _bar(
                    name,
                    {"v21": value(theirs)},
                    mine,
                    _le(value(mine), value(theirs)),
                )
            )
        else:
            row = _bar(name, False, mine, True)
            row["reason"] = "not gated: the pre-registration sets this bar to false"
            out.append(row)
    out.append(
        _bar(
            "latency_ratio_max",
            bars["latency_ratio_max"],
            lat,
            _le(lat["value"], bars["latency_ratio_max"]),
        )
    )
    return out


def c43_bars(quant_manifest: Mapping[str, Any], bars: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``bars.quant`` from the quant stage's manifest (read, not re-measured)."""
    delta = quant_manifest.get("quant_vs_bf16_delta_points")
    drop = None if delta is None else -float(delta)
    heals = quant_manifest.get("heal_count")
    ece = quant_manifest.get("deployed_selection_ece")
    rows = []
    for name, value, key in (
        ("max_accuracy_drop_points", drop, "quant_vs_bf16_delta_points"),
        ("ece_max", ece, "deployed_selection_ece"),
        ("heal_rounds_max", heals, "heal_count"),
    ):
        row = _bar(name, bars[name], {"n": None, "value": value}, _le(value, bars[name]))
        row["source"] = f"distill/quant/quant_manifest.json:{key}"
        rows.append(row)
    return rows


def _verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missed = [r["bar"] for r in rows if r["result"] != PASSED]
    return {"result": MISSED if missed else PASSED, "missed": missed, "bars": list(rows)}


def regression(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Student accuracy above v21 with ECE above v21 is a regression, not a win."""

    def v(key: str) -> float | None:
        return (stats.get(key) or {}).get("value")

    acc_s, acc_v = v(f"{STUDENT}.accuracy"), v(f"{V21}.accuracy")
    ece_s, ece_v = v(f"{STUDENT}.ece"), v(f"{V21}.ece")
    flagged = None not in (acc_s, acc_v, ece_s, ece_v) and acc_s > acc_v and ece_s > ece_v
    return {
        "regression": bool(flagged),
        "rule": "student accuracy > v21 accuracy and student ECE > v21 ECE",
        "student": {"accuracy": acc_s, "ece": ece_s},
        "stock_v21": {"accuracy": acc_v, "ece": ece_v},
    }


# --------------------------------------------------------------------- the report


def _pool_block(
    records: Sequence[Record],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    keys = item_keys(records)
    views = {}
    for name in MODELS:
        views[name] = outcomes(records, predictions.get(name, ()), calibrations.get(name))
    if calibrations.get(CLEF_CALIBRATED):
        views[CLEF_CALIBRATED] = outcomes(
            records, predictions.get(CLEF, ()), calibrations[CLEF_CALIBRATED]
        )
    per_model = {name: [view[k] for k in keys] for name, view in views.items()}
    stats = paired_bootstrap(
        len(keys), joint_stats(per_model, threshold), seed=seed, resamples=resamples
    )
    return {
        "items": len(keys),
        "records": len(records),
        "sources": sorted({r.source for r in records}),
        "unanswered": {n: sum(1 for _ok, c in u if c is None) for n, u in per_model.items()},
        "stats": stats,
        "regression": regression(stats),
    }


def runtime_table(runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Latency and peak memory per model, plus whether all three ran on one host."""
    rows = {}
    for name in MODELS:
        meta = runtime.get(name) or {}
        lat, mem = meta.get("latency") or {}, meta.get("memory") or {}
        rows[name] = {
            "host": meta.get("host"),
            "latency_median_s": lat.get("median_s"),
            "latency_p95_s": lat.get("p95_s"),
            "latency_n": lat.get("n"),
            "peak_memory_mib": mem.get("peak_mib"),
            "memory_probe": mem.get("probe"),
            "memory_measured": bool(mem.get("measured")),
        }
    hosts = {r["host"] for r in rows.values()}
    return {"models": rows, "same_box": len(hosts) == 1 and None not in hosts}


def build_report(
    *,
    records_by_side: Mapping[str, Sequence[Record]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    calibration_sources: Mapping[str, str],
    runtime: Mapping[str, Mapping[str, Any]],
    prereg: dp.DistillPrereg,
    quant_manifest: Mapping[str, Any],
    contamination_manifest: Mapping[str, Any] | None = None,
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED,
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """The final report (see the module docstring); pure, writes nothing."""
    threshold = float(prereg.selection_rule["confident_wrong_threshold"])
    every = [r for side in SIDES for r in records_by_side.get(side, ())]
    sources = source_table(every, contamination_manifest)
    sides: dict[str, Any] = {}
    for side in SIDES:
        recs = list(records_by_side.get(side, ()))
        pools = {}
        for pool in POOLS:
            members = [r for r in recs if pool_of(r, sources) == pool]
            pools[pool] = _pool_block(
                members, predictions, calibrations, threshold, seed, resamples
            )
        bars = _verdict(c42_bars(pools[HEADLINE]["stats"], prereg.bars["test"], runtime))
        bars["gated"] = side == GATED_SIDE
        sides[side] = {"pools": pools, "c42": bars}
    c42 = dict(sides[GATED_SIDE]["c42"])
    c42["side"] = GATED_SIDE
    c42["pool"] = HEADLINE
    reg = sides[GATED_SIDE]["pools"][HEADLINE]["regression"]
    if reg["regression"]:
        c42 = {**c42, "result": MISSED, "missed": [*c42["missed"], "regression"]}
    c43 = _verdict(c43_bars(quant_manifest, prereg.bars["quant"]))
    return {
        "stage": "distill-report",
        "headline_rule": "human-gold records from sources the contamination check marks clean",
        "confident_wrong_threshold": threshold,
        "seed": seed,
        "resamples": resamples,
        "paired": "one index draw per resample, shared by all models and differences",
        "unanswered_rule": "counted wrong for accuracy and confident-wrong, left out of ECE",
        "calibration_used": dict(calibration_sources),
        "sources": sources,
        "sides": sides,
        "c42": c42,
        "c43": c43,
        "regression": reg["regression"],
        "runtime": runtime_table(runtime),
        "result": PASSED if c42["result"] == PASSED and c43["result"] == PASSED else MISSED,
    }
