"""The distill final report: the c17 bars over Clef, the stock model and both student builds.

Built from the single final measurement (:mod:`jev_factory.distill.final`), never from a new
one. Every item of a pool is one gold-bearing question ``(record id, qid)``; the models are
**resampled jointly**: each bootstrap draw picks one list of item indices and every model
(and every difference or agreement between models) is computed on those same indices, so the
CIs are on paired per-item differences.

Models: Clef (W4A16, as served), the stock model, the distilled **bf16 student**
(``student_bf16``) and the **shipped quant** (``student``, the deployed artifact).

Pools, per sealed side (``test`` and ``held_out``):

* ``headline``: records whose gold is **human** (``Record.gold_provenance == "human"``), whose
  id the operator **approved** in the gold ledger (:func:`jev_factory.review.gold.approved_ids`)
  and whose source :func:`jev_factory.distill.contamination.check_sources` marks ``clean``.
  The c17 bars are computed here, on the test side only, with n.
* ``overlapping`` and ``stock_evaluated``: approved human-gold records from sources the stock
  model trained on, or that it only evaluated on. Reported separately, not gated.
* ``unapproved_human``: records labelled human gold without an approve entry. They never
  count as human gold (the final refuses such a test record outright); reported only.
* ``model_generated``: records whose gold a model produced. Their accuracy is **agreement**
  with that gold, reported as a secondary table, never gated.

c17 (the run's success bars, ``bars.c17`` of the locked, hashed distill pre-registration, cited
by sha256), for **each** build (``bf16`` and ``quant``): top-1 over stock (points, >= the
bar) and top-1 under Clef W4A16 (points, <= the bar); for the quant also its agreement with the
bf16 student (same draws) and its attested weight bytes (strictly below the bar: '< 1 GB' means
< 10^9 bytes). Every bar carries its value, n and paired-bootstrap CI and is compared on its
point value. A bar that cannot be computed is a miss with the reason; a miss is recorded as a
miss and the report is still written. A build with higher accuracy and higher ECE than stock is
flagged as a regression and fails the run.

Reported, never pass/fail: latency median/p95 per build (and per model with its host, so "same
box" is checkable) and peak memory; the Clef probe-set table (general ability, loss of it
measured, nothing gated); the #18-shape ``c42`` (``bars.test``) and ``c43`` (``bars.quant``)
tables, kept for comparison with the earlier run (c43 is read from
``distill/quant/quant_manifest.json``); and, given the fitted decision gate (deviation d4,
``distill/gate/gate.json``), the shipped student's per-kind coverage and selective accuracy
under ``decision_gate``.

A failed or unanswered question counts as wrong for accuracy and confident-wrong, as a
disagreement for quant-vs-bf16 agreement, and is left out of ECE (it has no confidence); the
per-model ``unanswered`` count says how many. Stdlib only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jev_factory.core import metrics as core_metrics
from jev_factory.distill import contamination
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dp
from jev_factory.distill import train as dtrain
from jev_factory.distill.records import JEV_BENCH_SOURCE, Record

CLEF = "clef"
STOCK = "stock"
#: The shipped quantized student (the deployed artifact the final attested).
STUDENT = "student"
#: The distilled bf16 student the quant was built from.
STUDENT_BF16 = "student_bf16"
MODELS = (CLEF, STOCK, STUDENT_BF16, STUDENT)
#: c17 build (``bars.c17`` key) -> the model row it is measured on.
C17_MODELS = {"bf16": STUDENT_BF16, "quant": STUDENT}
AGREEMENT = "quant_vs_bf16_agreement"
#: The extra row when Clef has a calibration of its own (reported, never gated).
CLEF_CALIBRATED = "clef_calibrated"

GATED_SIDE = "test"
SIDES = ("test", "held_out")
HEADLINE = "headline"
OVERLAPPING = "overlapping"
STOCK_EVALUATED = "stock_evaluated"
UNAPPROVED = "unapproved_human"
MODEL_GENERATED = "model_generated"
POOLS = (HEADLINE, OVERLAPPING, STOCK_EVALUATED, UNAPPROVED, MODEL_GENERATED)

PASSED = "pass"  # a bar result, not a secret
MISSED = "miss"
#: A row that is reported with no pass/fail (latency).
REPORTED = "reported"
#: Point comparisons tolerate float noise (``100 * (0.85 - 0.75)`` is not exactly 10).
_EPS = 1e-9

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


def pool_of(
    rec: Record, sources: Mapping[str, Mapping[str, Any]], approved: Collection[str]
) -> str:
    """The pool *rec* is reported in; human gold counts only with an approve entry."""
    if rec.gold_provenance != "human":
        return MODEL_GENERATED
    if rec.id not in approved:
        return UNAPPROVED
    status = sources[rec.source]["status"]
    if status == contamination.CLEAN:
        return HEADLINE
    return OVERLAPPING if status == contamination.OVERLAPPING else STOCK_EVALUATED


# --------------------------------------------------------------------- per-item outcomes


def item_keys(records: Sequence[Record]) -> list[tuple[str, str]]:
    """Every gold-bearing question ``(record id, qid)`` of *records*, in record order."""
    keys = []
    for rec in records:
        for qid in rec.questions:
            if dtrain._gold_label(rec, qid) is not None:
                keys.append((rec.id, qid))
    return keys


def predicted(
    records: Sequence[Record],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> dict[tuple[str, str], tuple[Any, float | None]]:
    """``(record id, qid) -> (top label, its probability)``, ``(None, None)`` when unanswered.

    Only gold-bearing questions are listed. *calibration* (``temperature_by_kind`` /
    ``vector_by_kind``) is applied when given.
    """
    by_id = {str(r.get("record_id")): r for r in rows}
    out: dict[tuple[str, str], tuple[Any, float | None]] = {}
    for rec in records:
        row = by_id.get(rec.id) or {}
        answers = row.get("questions") or {}
        for qid in rec.questions:
            gold = dtrain._gold_label(rec, qid)
            if gold is None:
                continue
            out[(rec.id, qid)] = _top(rec, qid, gold, row, answers.get(qid), calibration)
    return out


def outcomes(
    records: Sequence[Record],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> dict[tuple[str, str], Unit]:
    """``(record id, qid) -> (correct, top probability or None)`` for one model.

    *calibration* is applied when given; an unanswered question is ``(False, None)``.
    """
    gold = {(rec.id, qid): dtrain._gold_label(rec, qid) for rec in records for qid in rec.questions}
    return {
        key: (label is not None and label == gold[key], conf)
        for key, (label, conf) in predicted(records, rows, calibration).items()
    }


def agreement_series(
    keys: Sequence[tuple[str, str]],
    a: Mapping[tuple[str, str], tuple[Any, float | None]],
    b: Mapping[tuple[str, str], tuple[Any, float | None]],
) -> list[float]:
    """Per item, 1.0 when both models answered with the same top label, else 0.0."""
    out = []
    for key in keys:
        la, lb = a.get(key, (None, None))[0], b.get(key, (None, None))[0]
        out.append(1.0 if la is not None and la == lb else 0.0)
    return out


def _answered_probs(
    rec: Record, qid: str, row: Mapping[str, Any], ans: Mapping[str, Any] | None
) -> dict[str, float] | None:
    """The answer's probabilities, or ``None`` when the question or its row errored or is absent."""
    if ans is not None and not ans.get("error") and not row.get("error"):
        return dtrain._answer_probs(rec, qid, ans)
    return None


def _calibrated_item(
    rec: Record, qid: str, probs: Any, gold: Any, calibration: Mapping[str, Any] | None
) -> Any:
    """The question's item, calibrated when *calibration* is given."""
    q = rec.questions[qid]
    item = dtrain.QItem(q.type, q.names, probs, gold)
    if calibration:
        item = dtrain.calibrate_item(item, calibration)
    return item


def _top(
    rec: Record,
    qid: str,
    gold: Any,
    row: Mapping[str, Any],
    ans: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
) -> tuple[Any, float | None]:
    """One gold-bearing question's ``(top label, top probability)`` or ``(None, None)``."""
    probs = _answered_probs(rec, qid, row, ans)
    if probs is None:
        return (None, None)
    item = _calibrated_item(rec, qid, probs, gold, calibration)
    return core_metrics.top_candidate({name: item.probs[name] for name in item.labels})


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


def gap_closure(student: float | None, stock: float | None, clef: float | None) -> float | None:
    """Share of the stock-to-Clef accuracy gap the student closes; ``None`` without a gap."""
    if student is None or stock is None or clef is None or clef - stock <= 0:
        return None
    return (student - stock) / (clef - stock)


def _points(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def c17_key(build: str, name: str) -> str:
    """The joint-statistics key of one c17 quantity of *build* (``bf16`` or ``quant``)."""
    return f"c17.{build}.{name}"


def _c17_stats(stats: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    """Top-1 over stock and under Clef, in points, for every build measured."""
    out: dict[str, float | None] = {}
    for build, model in C17_MODELS.items():
        if model not in stats:
            continue
        acc = stats[model]["accuracy"]
        if STOCK in stats:
            out[c17_key(build, "top1_over_stock_points")] = _points(
                _diff(acc, stats[STOCK]["accuracy"])
            )
        if CLEF in stats:
            out[c17_key(build, "top1_under_clef_points")] = _points(
                _diff(stats[CLEF]["accuracy"], acc)
            )
    return out


def joint_stats(
    per_model: Mapping[str, Sequence[Unit]],
    threshold: float,
    series: Mapping[str, Sequence[float]] | None = None,
) -> Callable[[Sequence[int]], dict[str, float | None]]:
    """The statistic of one draw: every model, difference and *series* on the SAME indices.

    *series* are per-item values (e.g. quant-vs-bf16 agreement) whose mean is a statistic.
    """
    series = series or {}

    def stat(indices: Sequence[int]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        stats = {}
        for name, units in per_model.items():
            stats[name] = model_stats([units[i] for i in indices], threshold)
            for metric, value in stats[name].items():
                out[f"{name}.{metric}"] = value
        for name, values in series.items():
            out[name] = _mean([values[i] for i in indices])
        if {CLEF, STOCK, STUDENT} <= set(stats):
            s, v, c = stats[STUDENT], stats[STOCK], stats[CLEF]
            out["accuracy_diff_student_stock"] = _diff(s["accuracy"], v["accuracy"])
            out["accuracy_diff_clef_stock"] = _diff(c["accuracy"], v["accuracy"])
            out["gap_closure"] = gap_closure(s["accuracy"], v["accuracy"], c["accuracy"])
            out["ece_diff_student_stock"] = _diff(s["ece"], v["ece"])
            out["confident_wrong_diff_student_stock"] = _diff(
                s["confident_wrong"], v["confident_wrong"]
            )
        out.update(_c17_stats(stats))
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
        _resample(n, stat, draws, seed=seed, resamples=resamples)
    out: dict[str, dict[str, Any]] = {}
    for name, value in point.items():
        lo, hi = _percentile_ci(n, value, sorted(draws.get(name, [])))
        out[name] = {"n": n, "value": value, "ci_low": lo, "ci_high": hi}
    return out


def _resample(
    n: int,
    stat: Callable[[Sequence[int]], Mapping[str, float | None]],
    draws: dict[str, list[float]],
    *,
    seed: int,
    resamples: int,
) -> None:
    """Append every non-``None`` statistic of *resamples* seeded index draws to *draws*."""
    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
    for _ in range(resamples):
        sample = stat([rng.randrange(n) for _ in range(n)])
        for name, value in sample.items():
            if value is not None:
                draws.setdefault(name, []).append(value)


def _percentile_ci(
    n: int, value: float | None, values: Sequence[float]
) -> tuple[float | None, float | None]:
    """The 2.5th/97.5th percentile of sorted *values* (the point value itself when n == 1)."""
    if n == 1:
        return value, value
    if values:
        return (
            values[int(round(0.025 * (len(values) - 1)))],
            values[int(round(0.975 * (len(values) - 1)))],
        )
    return None, None


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


LATENCY_NOT_GATED = "latency is reported, never gated (bars.c17 latency_reported)"


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
    diff = stats.get("accuracy_diff_student_stock")
    gap = stats.get("gap_closure")
    s_ece, v_ece = stats.get(f"{STUDENT}.ece"), stats.get(f"{STOCK}.ece")
    s_cw = stats.get(f"{STUDENT}.confident_wrong")
    v_cw = stats.get(f"{STOCK}.confident_wrong")
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
                "no stock-to-Clef accuracy gap to close"
                if value(gap) is None and value(stats.get("accuracy_diff_clef_stock")) is not None
                else None
            ),
        ),
        _bar("ece_max", bars["ece_max"], s_ece, _le(value(s_ece), bars["ece_max"])),
    ]
    for name, mine, theirs in (
        ("ece_no_worse_than_stock", s_ece, v_ece),
        ("confident_wrong_no_worse_than_stock", s_cw, v_cw),
    ):
        if bars[name]:
            out.append(
                _bar(
                    name,
                    {"stock": value(theirs)},
                    mine,
                    _le(value(mine), value(theirs)),
                )
            )
        else:
            row = _bar(name, False, mine, True)
            row["reason"] = "not gated: the pre-registration sets this bar to false"
            out.append(row)
    row = _bar("latency_ratio_max", bars["latency_ratio_max"], lat, True)
    row["result"] = REPORTED
    row["reason"] = LATENCY_NOT_GATED
    out.append(row)
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
    missed = [r["bar"] for r in rows if r["result"] == MISSED]
    return {"result": MISSED if missed else PASSED, "missed": missed, "bars": list(rows)}


def regression(stats: Mapping[str, Mapping[str, Any]], model: str = STUDENT) -> dict[str, Any]:
    """*model* accuracy above stock with ECE above stock is a regression, not a win."""

    def v(key: str) -> float | None:
        return (stats.get(key) or {}).get("value")

    acc_s, acc_v = v(f"{model}.accuracy"), v(f"{STOCK}.accuracy")
    ece_s, ece_v = v(f"{model}.ece"), v(f"{STOCK}.ece")
    flagged = None not in (acc_s, acc_v, ece_s, ece_v) and acc_s > acc_v and ece_s > ece_v
    return {
        "regression": bool(flagged),
        "model": model,
        "rule": f"{model} accuracy > stock accuracy and {model} ECE > stock ECE",
        model: {"accuracy": acc_s, "ece": ece_s},
        "stock": {"accuracy": acc_v, "ece": ece_v},
    }


# --------------------------------------------------------------------- c17


def _ge(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a >= b - _EPS


def _le_eps(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a <= b + _EPS


def _weight_bar(threshold: Any, bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    """The attested weight bytes of the shipped bundle, strictly below the bar (no CI)."""
    value = (bundle or {}).get("weight_bytes")
    ok = None if not isinstance(value, int) or isinstance(value, bool) else value < threshold
    row = _bar(
        "weight_bytes_max",
        threshold,
        {"n": None, "value": value},
        ok,
        reason=None if ok is not None else "the final attested no weight_bytes for the bundle",
    )
    row["compared"] = "value < threshold"
    row["source"] = "GET /v1/bundle weight_bytes (the attested deployed bundle)"
    return row


def build_latency(runtime: Mapping[str, Mapping[str, Any]], model: str) -> dict[str, Any]:
    """Median and p95 latency of *model* in the final, reported with no pass/fail."""
    meta = runtime.get(model) or {}
    lat = meta.get("latency") or {}
    return {
        "model": model,
        "host": meta.get("host"),
        "n": lat.get("n"),
        "median_s": lat.get("median_s"),
        "p95_s": lat.get("p95_s"),
        "gated": False,
        "reason": LATENCY_NOT_GATED,
    }


def c17_build_bars(
    build: str,
    stats: Mapping[str, Mapping[str, Any]],
    bars: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Every ``bars.c17.<build>`` bar from one pool's paired-bootstrap statistics."""

    def stat(name: str) -> Mapping[str, Any] | None:
        return stats.get(c17_key(build, name))

    def value(entry: Mapping[str, Any] | None) -> float | None:
        return (entry or {}).get("value")

    over, under = stat("top1_over_stock_points"), stat("top1_under_clef_points")
    rows = [
        _bar(
            "top1_over_stock_min_points",
            bars["top1_over_stock_min_points"],
            over,
            _ge(value(over), bars["top1_over_stock_min_points"]),
        ),
        _bar(
            "top1_under_clef_w4a16_max_points",
            bars["top1_under_clef_w4a16_max_points"],
            under,
            _le_eps(value(under), bars["top1_under_clef_w4a16_max_points"]),
        ),
    ]
    if "quant_vs_bf16_agreement_min" in bars:
        agree = stats.get(AGREEMENT)
        rows.append(
            _bar(
                "quant_vs_bf16_agreement_min",
                bars["quant_vs_bf16_agreement_min"],
                agree,
                _ge(value(agree), bars["quant_vs_bf16_agreement_min"]),
            )
        )
    if "weight_bytes_max" in bars:
        rows.append(_weight_bar(bars["weight_bytes_max"], bundle))
    return rows


def c17_report(
    pool: Mapping[str, Any],
    bars: Mapping[str, Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any] | None,
    runtime: Mapping[str, Mapping[str, Any]],
    prereg_sha256: str,
) -> dict[str, Any]:
    """c17 for both builds on the test headline pool, citing the hashed pre-registration."""
    stats = pool["stats"]
    builds: dict[str, Any] = {}
    missed: list[str] = []
    for build, model in C17_MODELS.items():
        block = _verdict(c17_build_bars(build, stats, bars[build], bundle))
        reg = regression(stats, model)
        if reg["regression"]:
            block = {**block, "result": MISSED, "missed": [*block["missed"], "regression"]}
        block.update(model=model, regression=reg, latency=build_latency(runtime, model))
        builds[build] = block
        missed += [f"{build}.{name}" for name in block["missed"]]
    return {
        "side": GATED_SIDE,
        "pool": HEADLINE,
        "n_records": pool["records"],
        "n_items": pool["items"],
        "prereg_sha256": prereg_sha256,
        "cites": f"bars.c17 of the locked distill pre-registration (sha256 {prereg_sha256})",
        "compared": "point value; n and the paired-bootstrap CI are reported beside it",
        "clef_reference": "Clef W4A16 as served in the final (raw)",
        "builds": builds,
        "result": MISSED if missed else PASSED,
        "missed": missed,
    }


# --------------------------------------------------------------------- the decision gate


GATE_NOT_GATED = (
    "reported only: the pre-registration registers no coverage bar, so the decision gate "
    "is not a c42/c43 bar"
)


def gate_units(
    records: Sequence[Record],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
    thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[tuple[dgate.Unit, bool]]]:
    """Per kind, ``((answered, correct), unanswered)`` for every gold-bearing question.

    The student's final distribution is calibrated (as :func:`outcomes` does) and gated
    with its kind's thresholds. An unanswered question cannot be answered by the gate, so
    it counts as an abstention.
    """
    by_id = {str(r.get("record_id")): r for r in rows}
    out: dict[str, list[tuple[dgate.Unit, bool]]] = {k: [] for k in dgate.KINDS}
    for rec in records:
        row = by_id.get(rec.id) or {}
        answers = row.get("questions") or {}
        for qid, q in rec.questions.items():
            gold = dtrain._gold_label(rec, qid)
            if gold is None or q.type not in out:
                continue
            probs = _answered_probs(rec, qid, row, answers.get(qid))
            if probs is None:
                out[q.type].append(((False, False), True))
                continue
            item = _calibrated_item(rec, qid, probs, gold, calibration)
            th = dgate.KindThresholds.from_json(thresholds.get(q.type))
            out[q.type].append((dgate.unit(dtrain._pred_item(item), th), False))
    return out


def decision_gate(
    records_by_side: Mapping[str, Sequence[Record]],
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any] | None,
    gate: Mapping[str, Any] | None,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Per side and kind: coverage and selective accuracy of the gated student (reported)."""
    if not gate or gate.get("fitted") is not True:
        return {
            "fitted": False,
            "gated": False,
            "reason": "no fitted decision gate (the distill gate stage has not run)",
            "sides": {},
        }
    thresholds = dgate.thresholds_by_kind(gate)
    sides: dict[str, Any] = {}
    for side in SIDES:
        per_kind = gate_units(records_by_side.get(side, ()), rows, calibration, thresholds)
        block = {}
        for group in (*dgate.KINDS, "all"):
            pairs = [p for kind, ps in per_kind.items() if group in ("all", kind) for p in ps]
            stats = dgate.summarize([u for u, _ in pairs], seed=seed, resamples=resamples)
            stats["unanswered"] = sum(1 for _, missing in pairs if missing)
            block[group] = stats
        sides[side] = block
    return {
        "fitted": True,
        "gated": False,
        "reason": GATE_NOT_GATED,
        "thresholds_by_kind": thresholds,
        "calibration_sha256": gate.get("calibration_sha256"),
        "unanswered_rule": "an unanswered question counts as an abstention",
        "sides": sides,
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
    tops = {}
    for name in MODELS:
        tops[name] = predicted(records, predictions.get(name, ()), calibrations.get(name))
    views = {
        name: outcomes(records, predictions.get(name, ()), calibrations.get(name))
        for name in MODELS
    }
    if calibrations.get(CLEF_CALIBRATED):
        views[CLEF_CALIBRATED] = outcomes(
            records, predictions.get(CLEF, ()), calibrations[CLEF_CALIBRATED]
        )
    per_model = {name: [view[k] for k in keys] for name, view in views.items()}
    series = {AGREEMENT: agreement_series(keys, tops[STUDENT], tops[STUDENT_BF16])}
    stats = paired_bootstrap(
        len(keys), joint_stats(per_model, threshold, series), seed=seed, resamples=resamples
    )
    return {
        "items": len(keys),
        "records": len(records),
        "sources": sorted({r.source for r in records}),
        "unanswered": {n: sum(1 for _ok, c in u if c is None) for n, u in per_model.items()},
        "stats": stats,
        "regression": regression(stats),
    }


PROBE_NOT_GATED = "the Clef probe set measures general ability; reported, gates nothing"


def probe_table(
    records: Sequence[Record],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Every model on the (private) Clef probe set: no pass/fail, no ids, no sources."""
    if not records:
        return {"measured": False, "gated": False, "reason": "no probe set was given"}
    block = _pool_block(records, predictions, calibrations, threshold, seed, resamples)
    block.pop("sources")
    block.pop("regression")
    return {"measured": True, "gated": False, "reason": PROBE_NOT_GATED, **block}


def runtime_table(runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Latency and peak memory per model, plus whether all of them ran on one host."""
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
    return {
        "models": rows,
        "same_box": len(hosts) == 1 and None not in hosts,
        "gated": False,
        "reason": LATENCY_NOT_GATED,
    }


@dataclass(frozen=True)
class ReportContext:
    """What the report cites beyond the predictions: the hashed prereg, bundle, approvals."""

    prereg: dp.DistillPrereg
    prereg_sha256: str
    #: Record ids with an approve entry in the gold ledger (``approved_ids``).
    approved_ids: Collection[str]
    quant_manifest: Mapping[str, Any]
    #: The attested deployed bundle (``files``, ``manifest_sha256``, ``weight_bytes``).
    bundle: Mapping[str, Any] | None = None
    contamination_manifest: Mapping[str, Any] | None = None
    gate: Mapping[str, Any] | None = None
    probe_records: Sequence[Record] = ()
    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES
    #: Proving mode (deviation d10): also compute the c17 bars on model-drafted gold,
    #: marked UNVERIFIED (``unverified_c17``); ``c17`` and ``result`` are unchanged.
    unverified_gold: bool = False


UNVERIFIED = "unverified"
UNVERIFIED_NOTE = (
    "UNVERIFIED: model-drafted gold, never approved by a human; proves the process only, "
    "never a ship or quality claim"
)


def unverified_c17(
    records_by_side: Mapping[str, Sequence[Record]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    runtime: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    headline_records: int,
    ctx: ReportContext,
) -> dict[str, Any]:
    """The c17 bars on the test side's model-drafted gold from clean sources (deviation d10).

    Marked ``verified: False`` with the share of the test pool (unverified plus headline
    records) whose gold no human approved. Never feeds ``result``.
    """
    threshold = float(ctx.prereg.selection_rule["confident_wrong_threshold"])
    members = [
        r
        for r in records_by_side.get(GATED_SIDE, ())
        if r.gold_provenance != "human" and sources[r.source]["status"] == contamination.CLEAN
    ]
    pool = _pool_block(members, predictions, calibrations, threshold, ctx.seed, ctx.resamples)
    block = c17_report(
        pool,
        ctx.prereg.bars["c17"],
        bundle=ctx.bundle,
        runtime=runtime,
        prereg_sha256=ctx.prereg_sha256,
    )
    total = len(members) + headline_records
    block.update(
        pool=UNVERIFIED,
        verified=False,
        unverified_share=1.0 if not headline_records else len(members) / total,
        sources=sorted({r.source for r in members}),
        note=UNVERIFIED_NOTE,
    )
    return block


def _sides(
    records_by_side: Mapping[str, Sequence[Record]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    runtime: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    ctx: ReportContext,
) -> dict[str, Any]:
    threshold = float(ctx.prereg.selection_rule["confident_wrong_threshold"])
    approved = frozenset(ctx.approved_ids)
    sides: dict[str, Any] = {}
    for side in SIDES:
        recs = list(records_by_side.get(side, ()))
        pools = {}
        for pool in POOLS:
            members = [r for r in recs if pool_of(r, sources, approved) == pool]
            pools[pool] = _pool_block(
                members, predictions, calibrations, threshold, ctx.seed, ctx.resamples
            )
        bars = _verdict(c42_bars(pools[HEADLINE]["stats"], ctx.prereg.bars["test"], runtime))
        bars["gated"] = False
        sides[side] = {"pools": pools, "c42": bars}
    return sides


C42_NOT_GATED = (
    "#18-shape bars (bars.test, bars.quant), reported for comparison; bars.c17 are this "
    "run's success bars"
)


def build_report(
    *,
    records_by_side: Mapping[str, Sequence[Record]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Mapping[str, Any] | None],
    calibration_sources: Mapping[str, str],
    runtime: Mapping[str, Mapping[str, Any]],
    context: ReportContext,
) -> dict[str, Any]:
    """The final report (see the module docstring); pure, writes nothing.

    ``context.gate`` is the fitted ``distill/gate/gate.json``; when given, the shipped
    student's final predictions are gated with it and per-kind coverage and selective accuracy
    are reported under ``decision_gate`` (reported only, never a bar).
    """
    ctx = context
    threshold = float(ctx.prereg.selection_rule["confident_wrong_threshold"])
    every = [r for side in SIDES for r in records_by_side.get(side, ())]
    sources = source_table(every, ctx.contamination_manifest)
    sides = _sides(records_by_side, predictions, calibrations, runtime, sources, ctx)
    headline = sides[GATED_SIDE]["pools"][HEADLINE]
    c17 = c17_report(
        headline,
        ctx.prereg.bars["c17"],
        bundle=ctx.bundle,
        runtime=runtime,
        prereg_sha256=ctx.prereg_sha256,
    )
    c42 = dict(sides[GATED_SIDE]["c42"])
    c42.update(side=GATED_SIDE, pool=HEADLINE, reason=C42_NOT_GATED)
    reg = headline["regression"]
    if reg["regression"]:
        c42 = {**c42, "result": MISSED, "missed": [*c42["missed"], "regression"]}
    c43 = _verdict(c43_bars(ctx.quant_manifest, ctx.prereg.bars["quant"]))
    c43.update(gated=False, reason=C42_NOT_GATED)
    regressed = any(b["regression"]["regression"] for b in c17["builds"].values())
    report = {
        "stage": "distill-report",
        "prereg": {"file": dp.PREREG_REL, "lock": dp.LOCK_REL, "sha256": ctx.prereg_sha256},
        "headline_rule": (
            "operator-approved human-gold test records from sources the contamination check "
            "marks clean"
        ),
        "headline_n": {"records": headline["records"], "items": headline["items"]},
        "confident_wrong_threshold": threshold,
        "seed": ctx.seed,
        "resamples": ctx.resamples,
        "paired": "one index draw per resample, shared by all models, differences and agreement",
        "unanswered_rule": (
            "counted wrong for accuracy and confident-wrong, a disagreement for agreement, "
            "left out of ECE"
        ),
        "calibration_used": dict(calibration_sources),
        "sources": sources,
        "sides": sides,
        "c17": c17,
        "c42": c42,
        "c43": c43,
        "regression": regressed,
        "runtime": runtime_table(runtime),
        "probe": probe_table(
            ctx.probe_records, predictions, calibrations, threshold, ctx.seed, ctx.resamples
        ),
        "decision_gate": decision_gate(
            records_by_side,
            predictions.get(STUDENT, ()),
            calibrations.get(STUDENT),
            ctx.gate,
            seed=ctx.seed,
            resamples=ctx.resamples,
        ),
        "result": c17["result"],
    }
    if ctx.unverified_gold:
        block = unverified_c17(
            records_by_side, predictions, calibrations, runtime, sources, headline["records"], ctx
        )
        report["unverified_c17"] = block
        report["unverified_result"] = block["result"]
        report["verification"] = (
            f"UNVERIFIED: {100 * block['unverified_share']:.1f}% of the test pool's gold is "
            "model-drafted and unapproved; unverified_c17 proves the process only"
        )
    return report
