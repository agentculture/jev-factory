"""Offline threshold sweep for the uncertainty gate (issue 53): no GPU, no scorer call.

Distribution-only: no tokenizer, model or backbone adapter. Stdlib only.

Re-decides every line of an existing predictions file (a
:mod:`jev_factory.core.predictions` record) under a grid of
:class:`~jev_factory.core.gate.Thresholds`, using each line's already recorded
``candidates`` distribution -- no model call, so the sweep runs offline on a
laptop from a predictions file a GPU run produced earlier.
For each threshold set it reports ``right_proposals``, ``wrong_mutating``,
escalation recall/precision (including the escalation recall specifically
on missing-candidate lines -- ids ending ``-nocand`` or whose gold label
was never offered), the ``abstain_uncertain`` count and the escalation
false-positive count, by feeding the re-decided lines back through
``metrics.compute`` rather than re-implementing any of that
counting here.

With every grid value left at its default (``none``), the sweep runs
exactly one threshold set -- :class:`gate.Thresholds` with everything
``None`` -- which reproduces each line's own stored argmax decision (see
``gate.decide``'s docstring and this module's ``--final`` sanity mode
below).

**Split safety.** Like :mod:`jev_factory.core.calibration`, this script never sweeps a
predictions file that looks like the test or held-out split, or a "final"
evaluation artifact, unless ``--final`` is passed -- see
:func:`refuse_unless_final`. ``--final`` is the one measurement on that side, so
it takes exactly **one value per grid knob** (:func:`require_single_point`):
nothing is searched on final data.

**The gate is fit on the fit fold only.** :func:`fit_gate` filters to the fit
fold's ids before it re-decides or scores anything, so no selection-fold
outcome can influence the chosen thresholds (nvsh D47).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, cast

from jev_factory.core import calibration, gate, metrics
from jev_factory.core.predictions import Prediction, read_predictions
from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/sweep_gate.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 38-55: sys.path hack and the _sibling importlib path-loader removed; gate,"
        " metrics and calibration are normal jev_factory.core imports",
        "redecide/evaluate/run_sweep/count_argmax_matches take the Domain explicitly"
        " (gate.decide and metrics.compute need it); main loads it from --domain",
        "redecide marks a newly chosen operation grounded=False (not grounded by the sweep) and"
        " keeps grounded=None for lines it did not propose",
        "redecide keeps a propose line's raw_probabilities/offered/raw_scores (t12 record)",
        "--final requires exactly one value per grid knob (require_single_point)",
        "added fit_gate: the gate is fit on the fit fold only, before any outcome is read",
    ],
    "licence": "Apache-2.0",
}

#: A grid value spelled this way (case-insensitive) means "threshold disabled".
NONE_SPELLING = "none"
#: The default grid for every numeric flag: a single, disabled value.
DEFAULT_GRID = (NONE_SPELLING,)
#: A path whose name or any directory component contains this word is
#: treated as a final/held-out evaluation artifact requiring ``--final``,
#: in addition to calibration's own test/held-out name markers.
FINAL_MARKER = "final"


class SweepError(ValueError):
    """A refused input file, or a malformed folds/grid argument."""


# ---------------------------------------------------------------------------
# Split safety
# ---------------------------------------------------------------------------


def split_markers(path: Path) -> set[str]:
    """Which of ``{"test", "held-out", "final"}`` *path* names.

    Reuses ``calibration.split_markers`` for the "test"/"held-out"
    name check (a predictions JSONL file carries no split header, so this
    is judged on the file name alone, same documented limitation as
    ``calibration``'s own use of it) and additionally flags any path
    with a "final" word in its stem or in one of its directory names --
    scorer-b1's exact-run predictions file (``final-scorer-b1-exact-...``,
    under a ``final/`` directory) is exactly this shape.
    """
    markers = set(calibration.split_markers(path))
    words = set(path.stem.lower().replace("-", " ").replace("_", " ").split())
    words |= {part.lower() for part in path.parts}
    if FINAL_MARKER in words:
        markers.add(FINAL_MARKER)
    return markers


def refuse_unless_final(path: Path, final: bool) -> None:
    """Raise :class:`SweepError` when *path* looks final/test/held-out and *final* is false."""
    if final:
        return
    markers = split_markers(path)
    if markers:
        raise SweepError(
            f"{path} looks like {' and '.join(sorted(markers))} data; pass --final to sweep it"
        )


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def load_folds(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fold_ids(folds: Mapping, fold: str) -> set[str]:
    key = {"fit": "fit_ids", "selection": "selection_ids"}.get(fold)
    if key is None:
        raise SweepError(f"fold must be 'fit' or 'selection', got {fold!r}")
    ids = folds.get(key)
    if not isinstance(ids, list):
        raise SweepError(f"the folds file has no {key!r}")
    return {str(entry_id) for entry_id in ids}


def filter_by_fold(
    predictions: Sequence[Prediction], folds: Mapping | None, fold: str | None
) -> list[Prediction]:
    if folds is None:
        return list(predictions)
    ids = fold_ids(folds, fold or "fit")
    return [p for p in predictions if p.id in ids]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def parse_grid(text: str) -> list[float | None]:
    """A comma-separated list of floats and/or ``"none"`` -> the parsed grid values."""
    values: list[float | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == NONE_SPELLING:
            values.append(None)
        else:
            try:
                values.append(float(item))
            except ValueError as exc:
                raise SweepError(f"not a number or 'none': {item!r}") from exc
    if not values:
        raise SweepError("a grid must name at least one value")
    return values


def _grid_or_default(values: Sequence[float | None] | None, fallback: Sequence[float | None]):
    return list(values) if values is not None else list(fallback)


def build_threshold_grid(
    *,
    escalate: Sequence[float | None],
    ro_floor: Sequence[float | None],
    ro_margin: Sequence[float | None],
    ro_max_entropy: Sequence[float | None],
    mut_floor: Sequence[float | None] | None = None,
    mut_margin: Sequence[float | None] | None = None,
    mut_max_entropy: Sequence[float | None] | None = None,
) -> list[gate.Thresholds]:
    """Every combination of the given grids as a list of :class:`gate.Thresholds`.

    ``mut_*`` defaults to the corresponding ``ro_*`` grid when omitted, so a
    caller who wants one shared floor/margin/entropy grid across both
    operation classes only has to pass the ``ro_*`` values.
    """
    mut_floor = _grid_or_default(mut_floor, ro_floor)
    mut_margin = _grid_or_default(mut_margin, ro_margin)
    mut_max_entropy = _grid_or_default(mut_max_entropy, ro_max_entropy)
    grid = []
    for esc, rf, rm, rx, mf, mm, mx in itertools.product(
        escalate, ro_floor, ro_margin, ro_max_entropy, mut_floor, mut_margin, mut_max_entropy
    ):
        grid.append(
            gate.Thresholds(
                escalate=esc,
                read_only=gate.ThresholdSet(floor=rf, margin=rm, max_entropy=rx),
                mutating=gate.ThresholdSet(floor=mf, margin=mm, max_entropy=mx),
            )
        )
    return grid


# ---------------------------------------------------------------------------
# Re-deciding
# ---------------------------------------------------------------------------


#: redecide()'s invalid_reason when the gate would propose an operation the original line
#: never actually chose (schema forces "invalid"'s own operation field to null, so this is
#: the only place that mismatch can be recorded).
NOT_GROUNDED_BY_SWEEP = "not grounded by the sweep"


def _original_choice(prediction: Prediction, domain: Domain) -> str | None:
    """The operation/control label *prediction*'s own decision was about, win or lose.

    A ``"propose"`` line names it directly. Any other outcome -- including
    ``"invalid"``, whose ``operation`` field the schema forces to ``null``
    even when grounding failed on a real winning operation -- never recorded
    that label directly, so this reproduces it the same way :func:`redecide`
    would under every threshold disabled (:func:`gate.decide`'s top1 is
    computed before any threshold check, so it never depends on thresholds).
    """
    if prediction.outcome == "propose":
        return prediction.operation
    baseline = gate.decide_prediction(prediction, gate.Thresholds(), domain)
    if baseline is None:
        return None
    return baseline.label if baseline.outcome == "propose" else None


def redecide(prediction: Prediction, thresholds: gate.Thresholds, domain: Domain) -> Prediction:
    """*prediction* with its outcome/operation re-decided under *thresholds*.

    Lines with no recorded distribution (``candidates`` is ``null``) cannot be
    gated -- the gate has nothing to score -- so they pass through unchanged.
    Every field but ``outcome``/``operation``/``arguments``/``invalid_reason``/
    ``grounded`` is kept as recorded (the distribution, its offered order and
    raw probabilities included).

    The gate never re-grounds arguments, it only decides whether to trust an
    already-grounded argmax, so ``arguments`` are kept only when the re-decided
    outcome is ``propose`` for the exact operation an *original, valid*
    ``propose`` line already grounded. Two other ``propose``-shaped cases never
    fabricate ``{}`` arguments instead: when the gate's pick is the same
    operation the original line's own argmax already named but that line was
    invalid, the result stays ``invalid`` with the original's ``invalid_reason``
    -- the sweep did not fix what made it invalid. When the gate's pick names a
    *different* operation than the one the original line actually decided,
    nothing here grounds it, so the result is ``invalid`` with reason
    :data:`NOT_GROUNDED_BY_SWEEP` and ``grounded=False``: a newly chosen
    operation is marked not grounded, never given arguments.
    """
    decision = gate.decide_prediction(prediction, thresholds, domain)
    if decision is None:
        return prediction
    invalid_reason = None
    grounded = None
    if decision.outcome != "propose":
        outcome, operation, arguments = decision.outcome, None, None
    elif prediction.outcome == "propose" and prediction.operation == decision.label:
        original_reason = metrics.invalid_reason(prediction, domain)
        if original_reason is None:
            outcome, operation = "propose", decision.label
            arguments = dict(prediction.arguments) if prediction.arguments else {}
            grounded = prediction.grounded
        else:
            # An original "propose" whose own arguments already failed the Domain's
            # validation: still invalid, for the same reason as before.
            outcome, operation, arguments, invalid_reason = "invalid", None, None, original_reason
    elif _original_choice(prediction, domain) == decision.label:
        # Same operation the original line's argmax named, but that line never grounded
        # it (typically an "invalid" line: schema forces its own operation to null).
        outcome, operation, arguments = "invalid", None, None
        invalid_reason = metrics.invalid_reason(prediction, domain) or NOT_GROUNDED_BY_SWEEP
    else:
        # A different operation than whatever the original line actually decided --
        # nothing here has grounded arguments for it.
        outcome, operation, arguments, invalid_reason = "invalid", None, None, NOT_GROUNDED_BY_SWEEP
        grounded = False
    return cast(
        Prediction,
        replace(
            prediction,
            outcome=outcome,
            operation=operation,
            arguments=arguments,
            invalid_reason=invalid_reason,
            grounded=grounded,
        ),
    )


# ---------------------------------------------------------------------------
# Evaluating one threshold set
# ---------------------------------------------------------------------------


def _missing_candidate_recall(
    originals: Sequence[Prediction], redecided: Sequence[Prediction]
) -> dict:
    """Among lines whose gold label was missing from the offered candidates.

    the fraction the re-decided outcome escalated (semantically or by
    abstention) -- the gate's whole point on that slice.
    """
    hits = 0
    total = 0
    for original, new in zip(originals, redecided):
        if not metrics.is_missing_candidate(original):
            continue
        total += 1
        if new.outcome in ("escalate", "abstain_uncertain"):
            hits += 1
    rate = (hits / total) if total else None
    return {"n": hits, "N": total, "rate": rate}


def evaluate(originals: Sequence[Prediction], thresholds: gate.Thresholds, domain: Domain) -> dict:
    """One threshold set's report row, over *originals* (a list of predictions)."""
    redecided = [redecide(p, thresholds, domain) for p in originals]
    # Point values only: the sweep reads no confidence interval (PR #65 review).
    computed = metrics.compute(redecided, domain, bootstrap_resamples=0)
    return {
        "thresholds": thresholds.to_json(),
        "right_proposals": computed["right_proposals"],
        "wrong_mutating": computed["wrong_mutating"],
        "abstain_uncertain_count": computed["outcome_counts"]["abstain_uncertain"],
        "escalation": {
            "tp": computed["escalation"]["tp"],
            "fn": computed["escalation"]["fn"],
            "fp": computed["escalation"]["fp"],
            "recall": computed["escalation"]["recall"],
            "precision": computed["escalation"]["precision"],
        },
        "false_positives": computed["escalation"]["fp"],
        "missing_candidate_escalation_recall": _missing_candidate_recall(originals, redecided),
    }


def run_sweep(
    originals: Sequence[Prediction], grid: Sequence[gate.Thresholds], domain: Domain
) -> list[dict]:
    return [evaluate(originals, thresholds, domain) for thresholds in grid]


def fit_gate(
    predictions: Sequence[Prediction],
    folds: Mapping,
    grid: Sequence[gate.Thresholds],
    domain: Domain,
    choose: Callable[[list[dict]], int],
) -> tuple[gate.Thresholds, list[dict]]:
    """Choose a gate from *grid* using the **fit fold only**.

    Predictions are filtered to ``folds["fit_ids"]`` before anything is
    re-decided or scored, so no selection-fold outcome is ever read (nvsh
    D47). *choose* receives the fit-fold report rows (:func:`evaluate`'s
    shape, in grid order) and returns the index of the thresholds to keep --
    the pre-registered rule lives with the caller. Returns the chosen
    thresholds and the reports it chose from.
    """
    if not grid:
        raise SweepError("a gate fit needs at least one threshold set")
    fit = filter_by_fold(predictions, folds, "fit")
    if not fit:
        raise SweepError("no predictions line has an id in the fit fold")
    reports = run_sweep(fit, grid, domain)
    index = choose(reports)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(grid):
        raise SweepError(f"choose returned {index!r}, not an index into the grid")
    return grid[index], reports


# ---------------------------------------------------------------------------
# --final takes one value per knob
# ---------------------------------------------------------------------------


def require_single_point(**grids: Sequence[float | None] | None) -> None:
    """Raise :class:`SweepError` unless every given grid has exactly one value.

    ``--final`` is one measurement of one pre-selected threshold set on the
    final side; a multi-value grid there would be a search on final data. A
    grid passed as ``None`` (a ``mutating_*`` knob left to default to its
    read-only twin) is not a knob of its own and is skipped.
    """
    many = {name: list(values) for name, values in grids.items() if values and len(values) != 1}
    if many:
        names = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(many))
        raise SweepError(f"--final takes exactly one value per grid knob; got several for {names}")


# ---------------------------------------------------------------------------
# Exact-reproduction check (the --final sanity mode leans on this too)
# ---------------------------------------------------------------------------


def count_argmax_matches(originals: Sequence[Prediction], domain: Domain) -> tuple[int, int]:
    """``(matches, total)`` re-deciding every line under all-disabled thresholds.

    A line with no recorded distribution is skipped (there is nothing to
    re-decide); it counts toward neither *matches* nor *total*.
    """
    matches = 0
    total = 0
    for original in originals:
        if original.candidates is None:
            continue
        total += 1
        new = redecide(original, gate.Thresholds(), domain)
        if new.outcome == original.outcome and new.operation == original.operation:
            matches += 1
    return matches, total


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def report_markdown(reports: Iterable[dict]) -> str:
    lines = [
        "| escalate | ro floor/margin/entropy | mut floor/margin/entropy | right | "
        "wrong mutating | abstain_uncertain | esc recall | esc fp | missing-cand recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in reports:
        t = row["thresholds"]
        ro = t["read_only"]
        mut = t["mutating"]
        right = row["right_proposals"]
        missing = row["missing_candidate_escalation_recall"]
        recall = row["escalation"]["recall"]
        lines.append(
            "| {esc} | {rf}/{rm}/{rx} | {mf}/{mm}/{mx} | {right_n}/{right_N} | {wrong} | "
            "{abstain} | {recall} | {fp} | {mn}/{mN} |".format(
                esc=t["escalate"],
                rf=ro["floor"],
                rm=ro["margin"],
                rx=ro["max_entropy"],
                mf=mut["floor"],
                mm=mut["margin"],
                mx=mut["max_entropy"],
                right_n=right["n"],
                right_N=right["N"],
                wrong=row["wrong_mutating"]["total"],
                abstain=row["abstain_uncertain_count"],
                recall="-" if recall is None else f"{recall:.3f}",
                fp=row["false_positives"],
                mn=missing["n"],
                mN=missing["N"],
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _grid_arg(parser: argparse.ArgumentParser, *names: str) -> None:
    for name in names:
        parser.add_argument(f"--{name}", default=NONE_SPELLING)


def main(argv: list[str] | None = None, domain: Domain | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_factory.core.sweep_gate", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--domain", help="domain module (dotted name) or JSON domain file (unless passed in)"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--fold", choices=("fit", "selection"), default="fit")
    parser.add_argument(
        "--final",
        action="store_true",
        help="allow a test/held-out/final predictions file; one value per grid knob",
    )
    _grid_arg(parser, "escalate", "floor", "margin", "max-entropy")
    parser.add_argument("--mutating-floor")
    parser.add_argument("--mutating-margin")
    parser.add_argument("--mutating-max-entropy")
    parser.add_argument("--out", help="write the JSON report here (default: stdout)")
    parser.add_argument("--markdown", help="also write a markdown table here")
    args = parser.parse_args(argv)

    predictions_path = Path(args.predictions)
    folds = None
    try:
        if domain is None:
            if not args.domain:
                raise SweepError("a Domain is required: pass --domain")
            from jev_factory.domain.validate import load_domain

            domain = load_domain(args.domain)
        refuse_unless_final(predictions_path, args.final)
        grids = {
            "escalate": parse_grid(args.escalate),
            "floor": parse_grid(args.floor),
            "margin": parse_grid(args.margin),
            "max_entropy": parse_grid(args.max_entropy),
            "mutating_floor": parse_grid(args.mutating_floor) if args.mutating_floor else None,
            "mutating_margin": parse_grid(args.mutating_margin) if args.mutating_margin else None,
            "mutating_max_entropy": (
                parse_grid(args.mutating_max_entropy) if args.mutating_max_entropy else None
            ),
        }
        if args.final:
            require_single_point(**grids)
        originals = read_predictions(predictions_path)
        folds = load_folds(Path(args.folds)) if args.folds else None
        selected = filter_by_fold(originals, folds, args.fold)
        grid = build_threshold_grid(
            escalate=grids["escalate"],
            ro_floor=grids["floor"],
            ro_margin=grids["margin"],
            ro_max_entropy=grids["max_entropy"],
            mut_floor=grids["mutating_floor"],
            mut_margin=grids["mutating_margin"],
            mut_max_entropy=grids["mutating_max_entropy"],
        )
        reports = run_sweep(selected, grid, domain)
    except (OSError, ValueError) as exc:  # SweepError, MetricsError are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "predictions": str(predictions_path),
        "fold": args.fold if folds is not None else None,
        "n": len(selected),
        "reports": reports,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown:
        Path(args.markdown).write_text(report_markdown(reports) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
