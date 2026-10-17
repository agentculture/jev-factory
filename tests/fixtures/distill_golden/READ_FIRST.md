# Golden baseline: read first

This directory holds main's (e3345d1) `jev distill` end-to-end output, produced by the fakes in
`tests/test_distill_e2e.py`. `tests/test_distill_golden.py` regenerates it and asserts 0
differing bytes. Later tasks must keep it identical under the default config.

## What is pinned

- `prereg.json` is the pre-registration the e2e run writes (`small_prereg()`), and
  `manifests/<stage>.json` is each of the 15 stage manifests.
- Normalised keys, all listed in `tests/distill_golden_support.py`: `started` and `finished`
  (timestamps), `jev_factory_version`, the pytest tmp dir inside any string (`<TMP>`), the
  `knobs._external.stock_reproduction` hash, and the sha256 of the 16 input/output paths in
  `VARYING_HASH_PATHS` (they embed tmp paths, measured latency or bootstrap draws).
- Every other hash is compared exactly, so the dataset, teacher, contamination, audit and
  preregister outputs are pinned by content. The varying files' own content is not pinned.

## What the stages assume (all under `jev_factory/distill/`)

The model set is four fixed names, and the quant is one deployed artifact.

- `report.py:62-70`: `MODELS = (clef, stock, student_bf16, student)`. `C17_MODELS` maps the two
  builds, `bf16` to `student_bf16` and `quant` to `student`.
- `report.py:293-308` (`_c17_stats`): loops `C17_MODELS` and emits, per build, top-1 over stock
  and under Clef. It skips a model absent from `stats`, so a new model name yields no c17 row.
- `report.py:449-508` (`c42_bars`): reads only `STUDENT` and `STOCK` keys
  (`accuracy_diff_student_stock`, `gap_closure`, `<model>.ece`, `<model>.confident_wrong`) plus
  `STUDENT` and `CLEF` latency. The headline bars are tied to the model named `student`.
- `report.py:511-526` (`c43_bars`): takes one flat quant manifest and reads exactly
  `quant_vs_bf16_delta_points`, `deployed_selection_ece` and `heal_count`. Its `source` string
  is the fixed path `distill/quant/quant_manifest.json`. `build_report` calls it at
  `report.py:965` with `ctx.quant_manifest`, and `c42_bars` at `report.py:920`.
- `quant.py:2005-2093` (`_finish`): writes the single manifest. It names one `chosen_format` and
  `shipped_format`, one `deployed_artifact`, one `deployed_checkpoint`, `shippable: True`, a
  fixed `heal_count` of 0 or 1, and compares against `measured[BF16]`. Its `outputs` are the
  measurements, decision, calibration, predictions and heal log. Calibration and predictions
  describe only the shipped format (`fitted_on_predictions_of`).
- `final.py:153-190` (`deployed_student`): requires `status == "done"`, a shipped format equal
  to `chosen_format`, `format_decision.json` agreeing, and the calibration sha256 recorded in
  the manifest `outputs`. Refuses when `shippable` is False. No bf16 fallback.
- `bundle.py:423-512` (`build_bundles`): ships one student. It calls `deployed_student`
  (`bundle.py:438`), then reads the single `quant/calibration.json` (`bundle.py:444`) and
  `final/report.json` (`bundle.py:442`), and `_gate` (`bundle.py:167-203`) refuses a gate whose
  `calibration_sha256` differs from that calibration. `_stage_student` (`bundle.py:240-267`)
  handles two layouts only: a torchao artifact (has `jev_quant.json`) or a bf16
  `hobson-hf-export/1` layout. The model bundle file set is fixed (`bundle.py:466-479`).
- `edge.py:240-258` (`expected_bundle`): takes `final.json` `deployed.bundle`, else the quant
  manifest `deployed_artifact`, so it assumes one deployed artifact.
- `edge.py:324-347` (`reference_rows`): reads `quant/deployed_predictions.json` and refuses unless
  its `bundle` matches the expected manifest sha256 and per-file sha256s.
- `edge.py:465-497` (`_finish`): the edge manifest records one bundle, one cap and the
  agreement against `REFERENCE_PREDICTIONS` (`edge.py:61`) on the selection fold only.

## Consequence for later tasks

Adding a model or a quant manifest shape must stay additive. Under the default config the
four model names, the flat quant manifest keys above and the file set in `bundle.py:466-479`
must produce the bytes pinned here.
