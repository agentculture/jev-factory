# r3b failure diagnostic

This is a procedure for one situation only: a **jev-tool evaluation failed**.
It is not a regression gate and not a parity test. The jev-tool model passing
every evaluation is what shows the extracted stages are correct (issue #1's
parity test is dropped for this scope). Until an evaluation fails, nothing
here runs and no stage reads any r3b or #53 artifact.

## What it answers

When a bar is missed, the fault is either in the pipeline (calibration, the
gate sweep, the metrics) or in the jev-CLI data or model. nvsh's frozen #53
r3b validation predictions have known numbers. Replaying them through the
same extracted select stages tells the two apart:

- The replay reproduces the frozen numbers: the stages are sound, so look at
  the jev-CLI data, the recipe or the model.
- The replay diverges: the stage named in the output (`calibration` or `gate`)
  is suspect. Fix it, add a test, and re-run the failed evaluation.

## The gate: a failed jev-tool evaluation record id

The entry point is `jev_factory/decide/r3b_replay.py`. It **refuses** unless
it is given the id of a decision record (`jev_factory/decide/records.py`, the
append-only store `jev decide` writes) that marks the failure:

- `details.evaluation` is `{"domain": "jev-tool", "outcome": "failed"}`, and
- the verdict is not `ship_candidate`.

The check runs before the config or any data file is opened. An unknown id,
a record without the marker, a passed evaluation or another model's
evaluation are all refused with a user error (exit 1). The spec also asks for
an open deviation record; record the failure and the decision to replay with
`/deviate` first, and cite its id in the decision record's `details`.

## Operator-supplied config, never committed

The frozen tree lives outside this repository. Its paths, the domain
definition and the reference numbers are given in a JSON config file the
operator writes and keeps out of version control. There are no defaults, and
no shipped config or stage names the frozen artifacts.

```json
{
  "domain": "domain.json",
  "predictions": "val-predictions.jsonl",
  "folds": "folds.json",
  "grid": {"ro_floor": [null, 0.5, 0.7]},
  "reference": {"right_proposals": 79, "wrong_mutating": 0, "ece": 0.019}
}
```

- `domain` is a domain JSON file (`Domain.to_dict()` shape) describing the
  frozen run's operation table, so the gate's read-only and mutating split
  matches the frozen data. It is built by the operator, not imported.
- `predictions` is one validation predictions JSONL. A file whose name marks
  it test or held-out is refused. Sealed and test text is never read.
- `folds` holds `fit_ids` and `selection_ids`.
- `grid` (optional) are the gate-sweep value lists; `reference` (optional) are
  the frozen selection-fold numbers to compare against.

Relative paths resolve against the config file's directory.

## Run it

```bash
uv run python -m jev_factory.decide.r3b_replay \
  --decisions <decisions.jsonl> --record D<n> --config <replay.json>
```

The report is JSON on stdout: the fitted calibration (fit fold only), the gate
chosen on the fit fold, the selection-fold numbers, the comparison with the
reference and a one-line localisation. Calibration and the gate are fit on the
fit fold only; the selection fold only judges them.

## After the replay

Record what it found as a decision record (or a `/deviate` record if it
changes the plan). Do not retrain r3b and do not widen the replay: it exists
only to localise this failure.
