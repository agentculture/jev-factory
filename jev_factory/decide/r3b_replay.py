"""Failure-time diagnostic: replay a frozen scorer run through the extracted select stages.

This is the only code that reads a frozen predictions tree from an earlier
project. It exists for one situation: a jev-tool evaluation **failed**, and the
question is whether the fault is in the pipeline (calibration, gate sweep,
metrics) or in the jev-CLI data or model. Replaying frozen #53 r3b validation
predictions, whose numbers are known, through the same stages separates the
two. The procedure is in ``docs/r3b-diagnostic.md``.

Guards (obligation o13):

* **Gated on a failed evaluation.** :func:`replay` refuses unless it is given
  the id of a decision record (``jev_factory.decide.records``) that marks a
  failed jev-tool evaluation: ``details.evaluation`` is
  ``{"domain": "jev-tool", "outcome": "failed"}`` and the verdict is not
  ``ship_candidate``. The check runs before any config or data is opened.
* **No defaults.** The frozen tree's paths, the domain definition and the
  reference numbers all come from an operator-supplied config file. Nothing
  here, and no stage or default config, names a location; with no config the
  replay does not run.
* **Validation side only.** Calibration fits on the fit fold and is judged on
  the selection fold; a predictions file that looks like test or held-out data
  is refused (``calibration.refuse_if_test_or_held_out``).

The replay reads predictions, never sealed or test text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.core import calibration, gate, sweep_gate
from jev_factory.core.predictions import read_predictions
from jev_factory.decide import records
from jev_factory.domain.validate import DomainError, load_domain

#: The model family whose failed evaluation opens this diagnostic.
JEV_TOOL = "jev-tool"
CONFIG_KEYS = ("domain", "predictions", "folds")
GRID_KEYS = (
    "escalate",
    "ro_floor",
    "ro_margin",
    "ro_max_entropy",
    "mut_floor",
    "mut_margin",
    "mut_max_entropy",
)
#: The gate grid used when the config gives none: disabled, plus a small floor/margin sweep.
DEFAULT_GRID = {
    "escalate": [None, 0.5],
    "ro_floor": [None, 0.5, 0.7],
    "ro_margin": [None, 0.1],
    "ro_max_entropy": [None],
}


def _refuse(message: str, remediation: str) -> CliError:
    return CliError(EXIT_USER_ERROR, message, remediation)


def require_failed_evaluation(decisions: Path, record_id: str) -> dict[str, Any]:
    """The decision record *record_id*, if it marks a failed jev-tool evaluation.

    Raises :class:`CliError` (a user error) when no id is given, the record is
    not in the store, or it does not mark a failed jev-tool evaluation.
    """
    hint = "record the failed evaluation with `jev decide` first; see docs/r3b-diagnostic.md"
    if not isinstance(record_id, str) or not record_id.strip():
        raise _refuse("the r3b replay needs a failed jev-tool evaluation record id", hint)
    found = next((r for r in records.load(Path(decisions)) if r["id"] == record_id.strip()), None)
    if found is None:
        raise _refuse(f"no decision record {record_id!r} in {decisions}", hint)
    evaluation = (found.get("details") or {}).get("evaluation")
    if not (
        isinstance(evaluation, dict)
        and evaluation.get("domain") == JEV_TOOL
        and evaluation.get("outcome") == "failed"
    ):
        raise _refuse(
            f"record {record_id} does not mark a failed {JEV_TOOL} evaluation "
            "(details.evaluation must be {domain: jev-tool, outcome: failed})",
            hint,
        )
    if found["verdict"] == "ship_candidate":
        raise _refuse(f"record {record_id} is a ship_candidate verdict, not a failure", hint)
    return found


def load_config(source: Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read and check the operator-supplied config: a JSON file path or a mapping.

    Keys: ``domain`` (a domain JSON file), ``predictions`` (frozen validation
    predictions, JSONL), ``folds`` (folds JSON with ``fit_ids``/``selection_ids``);
    optional ``grid`` (gate-sweep value lists) and ``reference`` (frozen numbers
    to compare against: ``right_proposals``, ``wrong_mutating``, ``ece``).
    Relative paths resolve against the config file's directory.
    """
    base: Path | None = None
    if isinstance(source, Mapping):
        cfg = dict(source)
    else:
        path = Path(source)
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CliError(
                EXIT_ENV_ERROR, f"cannot read replay config {path}: {exc}", "check the path"
            ) from exc
        base = path.resolve().parent
    if not isinstance(cfg, dict):
        raise _refuse("the replay config must be a JSON object", "see docs/r3b-diagnostic.md")
    missing = [k for k in CONFIG_KEYS if not cfg.get(k)]
    if missing:
        raise _refuse(
            f"replay config is missing: {', '.join(missing)}",
            "the frozen tree's paths are operator-supplied; there are no defaults",
        )
    for key in CONFIG_KEYS:
        p = Path(str(cfg[key]))
        cfg[key] = str(p if p.is_absolute() or base is None else base / p)
    return cfg


def _grid(config: Mapping[str, Any]) -> list[gate.Thresholds]:
    given = config.get("grid") or DEFAULT_GRID
    unknown = sorted(set(given) - set(GRID_KEYS))
    if unknown:
        raise _refuse(f"unknown grid keys: {', '.join(unknown)}", f"allowed: {list(GRID_KEYS)}")
    try:
        return sweep_gate.build_threshold_grid(
            **{k: list(given.get(k) or [None]) for k in GRID_KEYS[:4]},
            **{k: given[k] for k in GRID_KEYS[4:] if given.get(k)},
        )
    except TypeError as exc:
        raise _refuse(f"bad grid: {exc}", "each grid key is a list of numbers or null") from exc


def _choose(reports: Sequence[dict]) -> int:
    """Diagnostic pick: fewest wrong mutating, then most right proposals, earliest wins."""

    def key(item: tuple[int, dict]) -> tuple:
        i, r = item
        wrong = r["wrong_mutating"]
        right = r["right_proposals"]
        return (
            wrong["total"] if isinstance(wrong, dict) else wrong,
            -(right["n"] if isinstance(right, dict) else right),
            i,
        )

    return min(enumerate(reports), key=key)[0]


def _scalar(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else value


def _compare(reference: Mapping[str, Any], got: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage = {"right_proposals": "gate", "wrong_mutating": "gate", "ece": "calibration"}
    out = []
    for name, want in reference.items():
        if name not in stage:
            raise _refuse(f"unknown reference metric {name!r}", f"allowed: {sorted(stage)}")
        have = got[name]
        if want is None or have is None:
            same = want == have
        elif name == "ece":
            same = abs(float(want) - float(have)) <= 0.0005
        else:
            same = int(want) == int(have)
        out.append(
            {
                "metric": name,
                "stage": stage[name],
                "reference": want,
                "replayed": have,
                "same": same,
            }
        )
    return out


def replay(decisions: Path, record_id: str, config: Path | Mapping[str, Any]) -> dict[str, Any]:
    """Replay the frozen validation predictions through calibration, the gate sweep and metrics.

    Refuses first (no config or data is opened) unless *record_id* names a failed
    jev-tool evaluation in *decisions*. Returns a JSON-ready report.
    """
    failed = require_failed_evaluation(decisions, record_id)
    cfg = load_config(config)
    try:
        domain = load_domain(Path(cfg["domain"]))
    except DomainError as exc:
        raise _refuse(f"replay domain: {exc}", "supply a valid domain JSON file") from exc
    predictions_path = Path(cfg["predictions"])
    try:
        calibration.refuse_if_test_or_held_out(predictions_path)
        predictions = read_predictions(predictions_path)
        folds = sweep_gate.load_folds(Path(cfg["folds"]))
    except (calibration.CalibrationError, ValueError, OSError) as exc:
        raise _refuse(
            f"cannot replay: {exc}", "validation-side predictions and folds only"
        ) from exc
    try:
        params = calibration.fit_params_from_predictions(predictions, folds, source="replay")
        chosen = calibration.select_calibration(predictions, params, folds)
        calibrated = calibration.apply_to_predictions(predictions, chosen)
        thresholds, fit_rows = sweep_gate.fit_gate(calibrated, folds, _grid(cfg), domain, _choose)
        selection = sweep_gate.filter_by_fold(calibrated, folds, "selection")
        gated = sweep_gate.evaluate(selection, thresholds, domain)
        cal = calibration.evaluate_predictions(predictions, chosen, folds, "selection")
    except (calibration.CalibrationError, sweep_gate.SweepError) as exc:
        raise _refuse(f"replay stage failed: {exc}", "check the folds and predictions") from exc
    variant = "temperature+vector" if chosen["vector_kept"] else "temperature"
    got = {
        "right_proposals": _scalar(gated["right_proposals"], "n"),
        "wrong_mutating": _scalar(gated["wrong_mutating"], "total"),
        "ece": cal["variants"][variant]["ece"],
    }
    comparison = _compare(cfg.get("reference") or {}, got)
    off = sorted({c["stage"] for c in comparison if not c["same"]})
    if not comparison:
        verdict = "no reference numbers in the config: compare the stage outputs by hand"
    elif off:
        verdict = f"replay diverges from the frozen numbers in: {', '.join(off)} (pipeline suspect)"
    else:
        verdict = "replay reproduces the frozen numbers: suspect the jev-CLI data or model"
    return {
        "evaluation_record": {
            "id": failed["id"],
            "run": failed["run"],
            "verdict": failed["verdict"],
        },
        "calibration": {
            "temperature": chosen["temperature"],
            "vector_kept": chosen["vector_kept"],
            "selection_ece": chosen["selection_ece"],
            "selection_n": chosen["selection_n"],
        },
        "gate": {
            "thresholds": thresholds.to_json(),
            "grid_size": len(fit_rows),
            "fit_fold_choice": "fewest wrong mutating, then most right proposals",
        },
        "selection": got,
        "comparison": comparison,
        "localisation": verdict,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3b-replay", description="Replay frozen r3b predictions after a failed jev-tool eval."
    )
    parser.add_argument("--decisions", required=True, type=Path, help="decision record store")
    parser.add_argument(
        "--record", required=True, help="id of the failed jev-tool evaluation record"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="operator-supplied replay config"
    )
    args = parser.parse_args(argv)
    try:
        report = replay(args.decisions, args.record, args.config)
    except CliError as exc:
        print(f"error: {exc.message}\nhint: {exc.remediation}", file=sys.stderr)
        return exc.code
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
