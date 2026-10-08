"""Post-hoc calibration: temperature scaling, then a per-label vector kept only on merit.

Distribution-only: it reads a predictions record (:mod:`jev_factory.core.predictions`)
and never a tokenizer, a model or a backbone adapter. Stdlib only.

Steps, each a CLI subcommand (``python -m jev_factory.core.calibration``):

``folds``
    Split a validation split file's entry ids into two disjoint folds -- a
    fit fold and a selection fold -- with a recorded seed
    (:func:`jev_factory.core.split.make_folds`, fit fraction 0.7).

``fit``
    Fit a temperature ``T`` and a per-candidate-label vector on the **fit
    fold's rows only** by minimising negative log-likelihood (NLL) of the
    expected label under the ``candidates`` distribution. The selection fold
    is never read here (nvsh D47). Refuses a predictions file that looks like
    the test or held-out split -- see :func:`refuse_if_test_or_held_out`.

``evaluate``
    ECE/Brier on one fold for three variants: raw, temperature, and
    temperature + vector. :func:`select_calibration` turns that into the
    decision: the vector is **kept only if it lowers selection-fold ECE**
    over temperature alone; otherwise it is dropped (all-ones).

``apply``
    Rescale a predictions file's ``candidates`` with fitted params. Apply
    changes probabilities only: the pre-calibration distribution is kept in
    ``raw_probabilities`` and every other field (outcome, operation,
    arguments, grounded, ...) is untouched.

Temperature scaling raises every candidate probability to the power ``1/T``
and renormalises. Vector scaling multiplies each candidate label's
probability by its own scale and renormalises. Both operate only on the
labels a line actually offers, never by operation name -- the vector is keyed
by whatever label :func:`jev_factory.core.metrics.expected_label` produces.

Lines whose ``candidates`` is ``null`` are skipped and counted, never treated
as a zero-confidence row. Likewise a fit-fold line whose gold label has no
matching candidate at all (not even under ``metrics.canonical_label``'s
escalate-family rollup) is skipped and counted (``skipped_gold_absent``).

Fitting minimises NLL in log-space (see ``_row_nll``): a candidate
probability is floored once, on input, and the gold's probability mass is the
sum of every candidate label that rolls up to the gold's canonical label. The
renormalised *output* probability is never clipped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from jev_factory.core import metrics
from jev_factory.core.predictions import Prediction, read_predictions, write_predictions
from jev_factory.core.split import DEFAULT_FIT_FRACTION, make_folds

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/calibration_fit.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 50-66: sys.path hack and the _sibling importlib path-loader removed; metrics and"
        " predictions are normal jev_factory.core imports",
        "lines 77-78, 162-207: DEFAULT_FIT_FRACTION and make_folds are reused from"
        " jev_factory.core.split (t13), not redefined",
        "lines 430-455: load_fit_rows takes Prediction records (fit_rows) with a file wrapper",
        "lines 456-501: fit_params split into fit_params_from_predictions (fit-fold ids only,"
        " never reads selection rows) and the file wrapper",
        "lines 508-520: apply returns Prediction records, changes probabilities only and keeps"
        " the pre-calibration distribution in raw_probabilities",
        "added select_calibration: the vector is kept only if it lowers selection-fold ECE",
        "evaluate reads jev_factory.core.predictions records and compute_calibration",
    ],
    "licence": "Apache-2.0",
}


#: A probability floor so log() and division never see exactly zero.
EPS = 1e-9
#: Search bounds for a per-coordinate multiplier, in log space (~[0.05, 20]).
_LOG_BOUND = math.log(20.0)
#: The folds file keys a fold name selects.
FOLD_KEYS = {"fit": "fit_ids", "selection": "selection_ids"}


class CalibrationError(ValueError):
    """A refused input: the wrong split, a malformed folds/params file, and so on."""


# ---------------------------------------------------------------------------
# Refusing test / held-out input (mirrors measure.py's check_split_allowed)
# ---------------------------------------------------------------------------


def _stem_words(path: Path) -> set[str]:
    return {word for word in re.split(r"[^a-z0-9]+", path.stem.lower()) if word}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def split_markers(path: Path, header: object = None) -> set[str]:
    """Which of ``{"test", "held-out"}`` *path* (and *header*, if given) name.

    Mirrors ``measure.py``'s ``check_split_allowed``: the side is read from
    both the file name and, when available, a JSON header, so renaming a
    file alone does not change what it is judged to be. A predictions
    JSONL file (``metrics.py``'s schema) carries no such header, so a fit
    input built from one is judged on its name alone -- the most robust
    signal that format offers. This is a documented limitation: naming a
    held-out predictions file without "test" or "held-out" in it (or
    renaming split.py's own test.json) would defeat the check.
    """
    words = _stem_words(path)
    compact_stem = _compact(path.stem)
    text = (
        header
        if isinstance(header, str)
        else (json.dumps(header, sort_keys=True) if header else "")
    )
    lowered_text = text.lower()
    markers: set[str] = set()
    if "test" in words or re.search(r"\btest\b", lowered_text):
        markers.add("test")
    if "heldout" in compact_stem or "heldout" in _compact(text):
        markers.add("held-out")
    return markers


def refuse_if_test_or_held_out(path: Path, header: object = None) -> None:
    """Raise :class:`CalibrationError` when *path* looks like test or held-out."""
    markers = split_markers(path, header)
    if markers:
        raise CalibrationError(
            f"{path} looks like the {' and '.join(sorted(markers))} split; "
            "calibration is fit on a validation fold only, never test or held-out data"
        )


# ---------------------------------------------------------------------------
# Folds: seeded, disjoint, sorted
# ---------------------------------------------------------------------------


def read_split_ids(path: Path) -> list[str]:
    """The entry ids of a split file (``{"header": ..., "entries": [...]}``)."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise CalibrationError(f"{path} is not a split file ({{header, entries}})")
    ids = []
    for entry in raw["entries"]:
        if not isinstance(entry, dict) or "id" not in entry:
            raise CalibrationError(f"{path}: an entry is missing its id")
        ids.append(str(entry["id"]))
    return ids


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def temperature_scale(candidates: Mapping[str, float], temperature: float) -> dict[str, float]:
    """``p_i ** (1/T)``, renormalised over *candidates*' own labels."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    powered = {label: max(float(p), EPS) ** (1.0 / temperature) for label, p in candidates.items()}
    total = math.fsum(powered.values())
    if total <= 0:
        return {label: 1.0 / len(powered) for label in powered}
    return {label: value / total for label, value in powered.items()}


def vector_scale(candidates: Mapping[str, float], vector: Mapping[str, float]) -> dict[str, float]:
    """Each label's probability times its own scale (default 1.0), renormalised."""
    scaled = {label: max(float(p), EPS) * vector.get(label, 1.0) for label, p in candidates.items()}
    total = math.fsum(scaled.values())
    if total <= 0:
        return {label: 1.0 / len(scaled) for label in scaled}
    return {label: value / total for label, value in scaled.items()}


def apply_scaling(
    candidates: Mapping[str, float],
    temperature: float = 1.0,
    vector: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Temperature scaling, then vector scaling (if given), over one line's candidates."""
    scaled = (
        temperature_scale(candidates, temperature)
        if not math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=1e-12)
        else dict(candidates)
    )
    if vector:
        scaled = vector_scale(scaled, vector)
    return scaled


# ---------------------------------------------------------------------------
# Fitting: golden-section search for T, coordinate descent for the vector
# ---------------------------------------------------------------------------


def _logsumexp(values: Iterable[float]) -> float:
    """``log(sum(exp(v) for v in values))``, computed without overflow."""
    values = list(values)
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def _gold_matches(label: str, gold_canonical: str) -> bool:
    """Whether *label* rolls up (via :func:`metrics.canonical_label`) to *gold_canonical*."""
    return metrics.canonical_label(label) == gold_canonical


def _row_log_scores(
    candidates: Mapping[str, float],
    temperature: float,
    vector: Mapping[str, float] | None,
) -> dict[str, float]:
    """Unnormalised log-score per label: ``log(max(p, EPS)) / T [+ log(vector)]``.

    The probability floor is applied exactly once, to the *input* ``p`` --
    never to a transformed/renormalised output -- so a candidate that is
    already vanishingly small does not get an artificial, T-independent
    floor imposed on its *rescaled* probability later. See
    :func:`_row_nll`, which turns these into an exact NLL via
    :func:`_logsumexp` instead of clipping a renormalised probability.
    """
    scores = {label: math.log(max(float(p), EPS)) / temperature for label, p in candidates.items()}
    if vector:
        for label in scores:
            scores[label] += math.log(max(float(vector.get(label, 1.0)), EPS))
    return scores


def _row_nll(
    candidates: Mapping[str, float],
    gold: str,
    temperature: float,
    vector: Mapping[str, float] | None = None,
) -> float | None:
    """The exact NLL of *gold* under a temperature/vector-scaled *candidates*.

    *gold*'s probability mass is the sum of every candidate label that rolls
    up to the same canonical label (``metrics.canonical_label``) -- an
    ``escalate:<reason>`` candidate counts towards a bare ``(escalate)``
    gold, matching how :func:`metrics.rollup_escalate_candidates` scores
    everywhere else. The whole computation is done in log-space via
    :func:`_logsumexp`, so the renormalising constant (shared by every
    label) never needs computing on its own and the gold probability is
    never clipped after the fact -- only the raw input probabilities are
    floored, once, in :func:`_row_log_scores`.

    Returns ``None`` when *gold*'s canonical label has no matching
    candidate at all (the line's model output never considered it):
    such a row carries no gradient for *T*/vector and the caller should
    skip and count it rather than charging a fixed, meaningless penalty.
    """
    gold_canonical = metrics.canonical_label(gold)
    scores = _row_log_scores(candidates, temperature, vector)
    matched = [value for label, value in scores.items() if _gold_matches(label, gold_canonical)]
    if not matched:
        return None
    return _logsumexp(scores.values()) - _logsumexp(matched)


def _nll(
    rows: Sequence[tuple[Mapping[str, float], str]],
    temperature: float,
    vector: Mapping[str, float] | None = None,
) -> float:
    """Total negative log-likelihood of each row's gold label.

    Rows whose gold has no matching candidate (see :func:`_row_nll`) are
    skipped; callers that need that count should filter with
    :func:`_rows_with_gold_present` up front, since presence does not
    depend on *temperature* or *vector*.
    """
    total = 0.0
    for candidates, gold in rows:
        row_nll = _row_nll(candidates, gold, temperature, vector)
        if row_nll is not None:
            total += row_nll
    return total


def _rows_with_gold_present(
    rows: Sequence[tuple[Mapping[str, float], str]],
) -> tuple[list[tuple[Mapping[str, float], str]], int]:
    """``(kept_rows, skipped_count)``: drop rows whose gold has no matching candidate.

    Presence is a structural property of a row's ``candidates`` keys and its
    gold label (via :func:`metrics.canonical_label`), independent of any
    fitted temperature or vector, so it is computed once up front rather
    than inside the fitting objective's hot loop.
    """
    kept = []
    skipped = 0
    for candidates, gold in rows:
        gold_canonical = metrics.canonical_label(gold)
        if any(_gold_matches(label, gold_canonical) for label in candidates):
            kept.append((candidates, gold))
        else:
            skipped += 1
    return kept, skipped


def _golden_section_min(
    f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-5
) -> float:
    """The x in [lo, hi] minimising f, by golden-section search (deterministic)."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    a, b = lo, hi
    span = b - a
    if span <= tol:
        return (a + b) / 2.0
    n = max(1, int(math.ceil(math.log(tol / span) / math.log(invphi))))
    c = a + invphi2 * span
    d = a + invphi * span
    yc, yd = f(c), f(d)
    for _ in range(n):
        if yc < yd:
            b, d, yd = d, c, yc
            span = invphi * span
            c = a + invphi2 * span
            yc = f(c)
        else:
            a, c, yc = c, d, yd
            span = invphi * span
            d = a + invphi * span
            yd = f(d)
    return (a + b) / 2.0


def fit_temperature(rows: Sequence[tuple[Mapping[str, float], str]]) -> float:
    """The temperature minimising NLL on *rows*, by 1-D search over log(T)."""
    if not rows:
        return 1.0

    def objective(log_t: float) -> float:
        return _nll(rows, math.exp(log_t))

    return math.exp(_golden_section_min(objective, -_LOG_BOUND, _LOG_BOUND))


def fit_vector(
    rows: Sequence[tuple[Mapping[str, float], str]],
    labels: Sequence[str],
    rounds: int = 25,
    tol: float = 1e-6,
) -> dict[str, float]:
    """A per-label scale minimising NLL on *rows*, by coordinate descent.

    Each round visits every label in sorted order and re-optimises its
    scale alone (golden-section search over its log), holding every other
    label's scale fixed, until no coordinate moves by more than *tol* or
    *rounds* is reached. Deterministic: no randomness, a fixed visit order.
    """
    vector = dict.fromkeys(labels, 1.0)
    if not rows or not labels:
        return vector
    ordered = sorted(labels)
    for _ in range(rounds):
        moved = False
        for label in ordered:

            def objective(log_s: float, label=label) -> float:
                trial = dict(vector)
                trial[label] = math.exp(log_s)
                return _nll(rows, 1.0, trial)

            best = math.exp(_golden_section_min(objective, -_LOG_BOUND, _LOG_BOUND))
            if abs(best - vector[label]) > tol:
                moved = True
            vector[label] = best
        if not moved:
            break
    return vector


def fit_rows(
    predictions: Iterable[Prediction], fit_ids: set[str]
) -> tuple[list[tuple[dict, str]], int, int]:
    """``(rows, skipped_null, considered)`` for the fit fold's predictions.

    *rows* pairs each non-null-candidates prediction's ``candidates`` with its
    gold calibration label (``metrics.expected_label``). Only predictions whose
    id is in *fit_ids* are looked at: a selection-fold row is skipped before
    its ``candidates`` or ``expected`` are touched. *considered* counts every
    prediction in *fit_ids*; *skipped_null* is how many of those had
    ``candidates: null``.
    """
    rows: list[tuple[dict, str]] = []
    skipped_null = 0
    considered = 0
    for prediction in predictions:
        if prediction.id not in fit_ids:
            continue
        considered += 1
        if prediction.candidates is None:
            skipped_null += 1
            continue
        rows.append((prediction.candidates, metrics.expected_label(prediction.expected)))
    return rows, skipped_null, considered


def load_fit_rows(
    predictions_path: Path, fit_ids: set[str]
) -> tuple[list[tuple[dict, str]], int, int]:
    """:func:`fit_rows` over a predictions file."""
    return fit_rows(read_predictions(predictions_path), fit_ids)


def fit_params_from_predictions(
    predictions: Iterable[Prediction], folds: Mapping[str, object], source: str | None = None
) -> dict:
    """Fit temperature and vector on the **fit fold only**; the params dict.

    The selection fold is never read: only ids in ``folds["fit_ids"]`` are
    looked at (nvsh D47). Whether the vector is worth keeping is a separate
    step on the selection fold (:func:`select_calibration`).
    """
    fit_ids = {str(entry_id) for entry_id in folds.get("fit_ids", [])}
    if not fit_ids:
        raise CalibrationError("the folds file has no fit_ids")
    rows, skipped_null, considered = fit_rows(predictions, fit_ids)
    if not rows:
        raise CalibrationError("no fit-fold prediction has a non-null candidates distribution")
    # A row whose gold label has no matching candidate at all carries no gradient for
    # temperature/vector fitting -- skip and count it (see _rows_with_gold_present).
    rows, skipped_gold_absent = _rows_with_gold_present(rows)
    if not rows:
        raise CalibrationError("no fit-fold prediction offers its gold label as a candidate")
    labels = sorted({label for candidates, _ in rows for label in candidates})
    temperature = fit_temperature(rows)
    # The vector is fit on top of the already-temperature-scaled rows, so it matches
    # apply_scaling's chain (temperature, then vector) exactly.
    temperature_scaled_rows = [
        (temperature_scale(candidates, temperature), gold) for candidates, gold in rows
    ]
    vector = fit_vector(temperature_scaled_rows, labels)
    return {
        "seed": folds.get("seed"),
        "temperature": temperature,
        "vector": vector,
        "fit_examples": len(rows),
        "skipped_null": skipped_null,
        "skipped_gold_absent": skipped_gold_absent,
        "fit_fold_size": considered,
        "predictions_source": source,
    }


def fit_params(predictions_path: Path, folds: Mapping[str, object]) -> dict:
    """:func:`fit_params_from_predictions` over a file; refuses a test/held-out file."""
    refuse_if_test_or_held_out(predictions_path)
    return fit_params_from_predictions(
        read_predictions(predictions_path), folds, source=str(predictions_path)
    )


# ---------------------------------------------------------------------------
# Apply: rescale a predictions record
# ---------------------------------------------------------------------------


def apply_to_predictions(
    predictions: Iterable[Prediction], params: Mapping[str, object]
) -> list[Prediction]:
    """*predictions* with ``candidates`` rescaled by *params*; nothing else changes.

    Apply changes probabilities only: the outcome, operation, arguments,
    grounded flag and every other field are kept as recorded (re-deciding is
    the gate's job). The pre-calibration distribution is recorded in
    ``raw_probabilities`` (kept if a raw one is already there, so re-applying
    never overwrites the true raw distribution). A line with no distribution
    passes through unchanged.
    """
    temperature = float(params.get("temperature", 1.0))
    vector = params.get("vector") or {}
    rescaled = []
    for prediction in predictions:
        if prediction.candidates is None:
            rescaled.append(prediction)
            continue
        raw = prediction.raw_probabilities
        rescaled.append(
            replace(
                prediction,
                candidates=apply_scaling(prediction.candidates, temperature, vector),
                raw_probabilities=dict(prediction.candidates) if raw is None else raw,
            )
        )
    return rescaled


# ---------------------------------------------------------------------------
# Evaluate: raw vs temperature vs temperature + vector on one fold
# ---------------------------------------------------------------------------

VARIANT_TEMPERATURE_VECTOR = "temperature+vector"
#: The three variants :func:`evaluate` compares, in this order.
EVALUATED_VARIANTS = ("raw", "temperature", VARIANT_TEMPERATURE_VECTOR)


def evaluate_predictions(
    predictions: Iterable[Prediction],
    params: Mapping[str, object],
    folds: Mapping[str, object],
    fold: str,
) -> dict:
    """ECE/Brier (``metrics.compute_calibration``, bootstrap CIs) on *fold*'s ids only.

    Three variants of the same lines: the model's own distributions, the fitted
    temperature alone, and temperature then vector -- so the selection fold
    decides whether vector scaling earns its extra parameters. Evaluating on the
    fit fold is allowed but says nothing about generalisation.
    """
    key = FOLD_KEYS.get(fold)
    if key is None:
        raise CalibrationError(f"--fold must be fit or selection, not {fold!r}")
    ids = set(folds.get(key) or [])
    if not ids:
        raise CalibrationError(f"the folds file has no {key}")
    chosen = [p for p in predictions if p.id in ids]
    if not chosen:
        raise CalibrationError(f"no predictions line has an id in the {fold} fold")
    temperature = float(params.get("temperature", 1.0))
    vector = params.get("vector") or {}
    scalings = {
        "raw": (1.0, {}),
        "temperature": (temperature, {}),
        VARIANT_TEMPERATURE_VECTOR: (temperature, vector),
    }
    report: dict = {"fold": fold, "n": len(chosen), "temperature": temperature, "variants": {}}
    for name in EVALUATED_VARIANTS:
        t, v = scalings[name]
        scaled = [
            (
                p
                if p.candidates is None
                else replace(p, candidates=apply_scaling(p.candidates, t, v))
            )
            for p in chosen
        ]
        block = metrics.compute_calibration(scaled)
        block.pop("bins", None)
        report["variants"][name] = block
    return report


def evaluate(
    predictions_path: Path, params: Mapping[str, object], folds: Mapping[str, object], fold: str
) -> dict:
    """:func:`evaluate_predictions` over a predictions file."""
    return evaluate_predictions(read_predictions(predictions_path), params, folds, fold)


def select_calibration(
    predictions: Iterable[Prediction], params: Mapping[str, object], folds: Mapping[str, object]
) -> dict:
    """Keep the fitted vector only if it lowers **selection-fold** ECE over temperature alone.

    Returns a copy of *params* with ``vector_kept`` (bool), and ``vector``
    replaced by ``{}`` when it was dropped, plus the selection-fold
    ``selection_ece`` of each of the three variants so the decision is cited.
    A tie (or an ECE that cannot be computed) drops the vector: it has to earn
    its extra parameters.
    """
    report = evaluate_predictions(predictions, params, folds, "selection")
    ece_by_variant = {name: report["variants"][name]["ece"] for name in EVALUATED_VARIANTS}
    with_vector = ece_by_variant[VARIANT_TEMPERATURE_VECTOR]
    temperature_only = ece_by_variant["temperature"]
    keep = (
        with_vector is not None and temperature_only is not None and with_vector < temperature_only
    )
    chosen = dict(params)
    if not keep:
        chosen["vector"] = {}
    chosen["vector_kept"] = keep
    chosen["selection_ece"] = ece_by_variant
    chosen["selection_n"] = report["n"]
    return chosen


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_folds(args: argparse.Namespace) -> int:
    split_path = Path(args.split)
    with open(split_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    header = raw.get("header") if isinstance(raw, dict) else None
    try:
        refuse_if_test_or_held_out(split_path, header)
        ids = read_split_ids(split_path)
        fit_ids, selection_ids = make_folds(ids, args.seed, args.fit_fraction)
    except ValueError as exc:  # CalibrationError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(
        Path(args.out),
        {
            "seed": args.seed,
            "source": str(split_path),
            "fit_ids": fit_ids,
            "selection_ids": selection_ids,
        },
    )
    print(f"fit={len(fit_ids)} selection={len(selection_ids)}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    predictions_path = Path(args.predictions)
    try:
        with open(args.folds, encoding="utf-8") as handle:
            folds = json.load(handle)
        params = fit_params(predictions_path, folds)
        if args.select:
            params = select_calibration(read_predictions(predictions_path), params, folds)
    except (OSError, ValueError) as exc:  # CalibrationError, MetricsError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(Path(args.out), params)
    print(f"temperature={params['temperature']:.4f} examples={params['fit_examples']}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    try:
        with open(args.params, encoding="utf-8") as handle:
            params = json.load(handle)
        rows = apply_to_predictions(read_predictions(args.predictions), params)
    except (OSError, ValueError) as exc:  # CalibrationError, MetricsError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_predictions(args.out, rows)
    print(f"rescaled {len(rows)} line(s)")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        with open(args.params, encoding="utf-8") as handle:
            params = json.load(handle)
        with open(args.folds, encoding="utf-8") as handle:
            folds = json.load(handle)
        report = evaluate(Path(args.predictions), params, folds, args.fold)
    except (OSError, ValueError) as exc:  # CalibrationError, MetricsError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        write_json(Path(args.out), report)
    for name in EVALUATED_VARIANTS:
        block = report["variants"][name]
        print(f"{name}: ece={block['ece']:.4f} brier={block['brier']:.4f} n={block['n']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.core.calibration", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    folds_parser = sub.add_parser("folds", help="split a validation file's ids into fit/selection")
    folds_parser.add_argument("--split", required=True)
    folds_parser.add_argument("--seed", type=int, required=True)
    folds_parser.add_argument("--out", required=True)
    folds_parser.add_argument("--fit-fraction", type=float, default=DEFAULT_FIT_FRACTION)
    folds_parser.set_defaults(func=_cmd_folds)

    fit_parser = sub.add_parser("fit", help="fit temperature and vector scaling on the fit fold")
    fit_parser.add_argument("--predictions", required=True)
    fit_parser.add_argument("--folds", required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.add_argument(
        "--select",
        action="store_true",
        help="keep the vector only if it lowers selection-fold ECE (reads that fold to decide)",
    )
    fit_parser.set_defaults(func=_cmd_fit)

    apply_parser = sub.add_parser("apply", help="rescale a predictions file with fitted params")
    apply_parser.add_argument("--predictions", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--out", required=True)
    apply_parser.set_defaults(func=_cmd_apply)

    eval_parser = sub.add_parser(
        "evaluate", help="ECE/Brier on one fold: raw vs temperature vs temperature+vector"
    )
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--params", required=True)
    eval_parser.add_argument("--folds", required=True)
    eval_parser.add_argument("--fold", default="selection", choices=("fit", "selection"))
    eval_parser.add_argument("--out", default=None)
    eval_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
