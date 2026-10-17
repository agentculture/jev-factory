# heal drop vs original bf16

> A healed quant's accuracy drop is gated against the original selected bf16 student, with the healed-bf16 drop and the heal drift recorded beside it as context
> instruction: grep `_healed_problem` and the manifest; both read the original measured\[BF16\]

## Audience

- The operator running jev distill quantize --apply, and the distill report's c43 bars.quant reader
  - instruction: read a `heal_result`: field names say original vs `heal_bf16`

## Before → After

- Before: After a heal, quantize gates the healed quant's accuracy drop against the bf16 rebuilt from the healed checkpoint (quant.py `_heal_round` -> `_measure_all`), so a heal that lowers the bf16 itself hides that loss; agreement already uses the original student
  - instruction: run the new test on main
- After: Every accuracy check in distill quantize, before and after a heal, compares against the original selected bf16 student; the healed bf16's own accuracy and the heal drift are recorded as context and gate nothing
  - instruction: assert on both the passing heal test and the new failing one

## Why it matters

- A heal could ship a build several points worse than the student the bars and final were pre-registered against while every quantize check passes; one reference across selection, agreement, heal and final removes that loophole
  - instruction: quant.py:855 on main

## Requirements

- After a heal round, the healed quant's `drop_points` (the value `_healed_problem` gates on, quant.py:1528) is 100\*(original bf16 accuracy - healed quant accuracy), where original is measured\[BF16\] from the pre-heal `_measure_all` of the selected winner, not the bf16 rebuilt from the healed checkpoint
  - instruction: `test_a_heal_that_lowers_the_bf16_is_judged_against_the_original`
  - honesty: `_healed_problem`'s drop test reads the drop against the original bf16; a healed quant within margin of the healed bf16 but past it vs the original fails
- The `heal_result` event in `heal_log.json`l and the healed measurement carry two context fields beside `drop_points`: `drop_points_vs_heal_bf16` (healed bf16 accuracy - healed quant accuracy) and `heal_bf16_drift_points` (original bf16 accuracy - healed bf16 accuracy); neither is read by any gate
  - instruction: assert the sum in the new test; grep shows the context keys only written, never read in a condition
  - honesty: drift + `drop_vs_heal_bf16` == `drop_points` to float rounding, and no gate reads the two context fields
- The original and healed accuracies compared are assessed identically: same `fold_records`, same selection fold, same d12 gating restriction (domain sources only when gating is set), same assess() threshold, so the drop subtracts like from like
  - instruction: a gating variant of the new test: guard rows differ, domain drop still against the original
  - honesty: The original and healed accuracies come from assess() with the same `fold_records`, gating and kw; with gating=clean both are domain-restricted
- Messages that name the reference say which bf16: `heal_reasons` (:1382) keeps 'its own bf16' for the pre-heal check (there it is the original); `_healed_problem` (:1530) says 'against the original bf16 student'; the module docstring step 4 and pipeline.`_bf16_artifact` docstring (:866) drop the follow-up note
  - instruction: grep -n "own bf16" `jev_factory`/distill/quant.py shows only pre-heal sites
  - honesty: No post-heal message says 'its own bf16'; pre-heal messages keep it
- After a heal, `quant_manifest.json`'s `quant_vs_bf16_delta_points` and `bf16_selection_accuracy` describe the ORIGINAL bf16 student (today `final_bf16` = (healed or measured)\[BF16\], quant.py:1974, i.e. the healed bf16); the healed bf16's accuracy is recorded under its own key (`healed_bf16_selection_accuracy`) as context
  - instruction: assert in the new test against measurements.json
  - honesty: After a heal, manifest `bf16_selection_accuracy` equals measurements.json formats.bf16.accuracy and `quant_vs_bf16_delta_points` equals -(healed drop vs original); `healed_bf16_selection_accuracy` holds the healed one
- The distill report's c43 bars.quant row `max_accuracy_drop_points` (report.py:513-519, reads `quant_vs_bf16_delta_points`) therefore reports the drop against the original bf16 student after a heal, with no change to report.py's code
  - instruction: git diff shows no report.py change; a report test or the e2e test reads the row
  - honesty: report.py's c43 code is unchanged and its `max_accuracy_drop_points` row value equals the manifest's drop vs the original
- A test in tests/`test_distill_quant.py`: FakeMeasurer original bf16 0.90, healed bf16 0.85, original quant below margin -> heal; healed quant 0.85 (within 1 of healed bf16, 5 below original) is shippable:false with a second-heal-trigger problem naming the original bf16, and `heal_result` records `drop_points`=5, `drop_points_vs_heal_bf16` ~0, `heal_bf16_drift_points`=5
  - instruction: run it on main first (red), then on the branch (green)
  - honesty: The new test fails on main and passes on the branch
- The change is filed as deviation d17 on plan orin-nano-jetson-ops-decider (task t23, affects t12, classification risky) and approved by the operator before the code lands; it is unconditional, not a new knob, so quantize needs no --deviation-id for it
  - instruction: devague deviate --list shows d17 approved
  - honesty: d17 is approved by the operator (origin user, devague deviate --confirm) before the code change is committed
- Docs follow the code: the open 'Heal rule' follow-up in docs/deliveries/2026-10-15-orin-nano-jetson-ops-decider.md (:173) is marked decided with the deviation id, and docs/distill.md's 'Heal, once' section (:261) gains one sentence saying the healed drop is against the original bf16 student, the healed-bf16 drop and drift are context only
  - instruction: markdownlint-cli2 docs/distill.md docs/deliveries/2026-10-15-orin-nano-jetson-ops-decider.md
  - honesty: markdownlint-cli2 passes on the edited docs and no other doc line changes

## Honesty conditions

- One reference: the original selected bf16 student's selection-fold measurement is the only baseline any distill quantize gate subtracts from
- Un-healed runs produce byte-identical measurements, `format_decision` and manifest values to before the change
- `agreement_vs_bf16` after a heal is still computed against predictions\[BF16\] of the pre-heal round
- prereg.py and its schema version are unchanged; an existing locked pre-registration still loads
- git diff shows no change under `jev_factory`/decide/
- The operator can tell from `heal_log.json`l alone which bf16 each drop is against
- On main, the new test's scenario ships the healed quant (verified red before the fix)
- The loophole is real in code, not hypothetical: `_measure_all` subtracts from out\[BF16\] of the healed call
- Both heal outcomes (shippable and refused) record the three numbers
- The full suite and CI lint pass on the branch

## Success signals

- A healed quant 5 points below the original bf16 but within 1 point of the healed bf16 is refused (shippable:false, `stop_escalate`) where today it ships; all existing tests in tests/`test_distill_quant.py` pass unchanged (0 regressions)
  - instruction: uv run pytest -n auto; black/isort/flake8/bandit

## Scope / boundaries

- The un-healed path is unchanged: pre-heal selection (`FORMAT_RULE`, :1122), `heal_reasons` and an un-healed build's `drop_points` already compare against the original bf16, and keep doing so byte-for-byte
  - instruction: existing un-healed tests pass unchanged
- The agreement check is unchanged: it already uses the original student's predictions (reference=`bf16_predictions`), and final's 0.97 quant-vs-bf16 bar keeps the same reference
  - instruction: existing `test_a_healed_build_below_the_agreement_bar_is_not_shippable` passes unchanged
- No pre-registration schema change: `QUANT_BARS` (prereg.py:54) keeps `max_accuracy_drop_points` as a bare number, `quant_bars` (quant.py:344) still pins it to `HEAL_MARGIN_POINTS`; only its reference (documented in quant.py) changes
  - instruction: git diff `jev_factory`/distill/prereg.py is empty
- jev decide's rule (decide/rules.py QuantCheck/`_quant_verdict`, `HEAL_MARGIN_POINTS`=3.0, `HEAL_ROUNDS`=1) is unchanged; it reads only the jev run workdir, never distill `heal_log`
  - instruction: git diff --stat main -- `jev_factory`/decide

## Non-goals

- Historical records are not rewritten: older specs, plans and deliveries that say 'within 3.0 points of its own bf16' (docs/specs/2026-10-14-distillery-\*.md:36,:88; split plan :224,:304; 2026-09-30 plans :209,:295) and the #20 frame stay as written
- No re-measurement of the proving run: the shipped int4 is un-healed (`heal_count` 0, docs/evidence/2026-10-16-jetson-ops-proving-run.md:17,52-58), so its drop already is against the original bf16 and d14's evidence stands

## Scope exploration

- `s1` — `jev_factory/distill/quant.py _measure_all (:810) / _drop_points (:704)`: `_measure_all` computes every format's `drop_points` against out\[BF16\] of the SAME call; `_heal_round` (:1585) calls it on `healed_checkpoint`, so a healed quant's drop is against the healed bf16, while agreement uses reference=original predictions
  - seeds: `c2`
- `s2` — `jev_factory/distill/quant.py _heal_round heal_result (:1597)`: `heal_result` records `drop_points` and `agreement_vs_bf16` only; nothing records how far the heal moved the bf16 torso itself
  - seeds: `c3`
- `s3` — `jev_factory/distill/quant.py _measure_format gating branch (:790)`: with gating, accuracy is assessed on domain sources only; the pre-heal measured\[BF16\] and the healed quant both pass through the same gating/kw, so they are comparable only if the same gating object reaches both calls (hr.gating does)
  - seeds: `c4`
- `s4` — `quant.py docstring step 4 (:50) + heal_reasons/_healed_problem messages`: pre-heal message 'against its own bf16' is correct (own = original pre-heal); post-heal message at :1530 uses the same words but means the healed bf16
  - seeds: `c5`
- `s5` — `quant.py format choice (:1122) and caller _deploy (:1731)`: pre-heal drop is from `_measure_all` on the winner checkpoint; predictions\[BF16\] passed to `_heal_round` as reference is the original — no change needed there
  - seeds: `c6`
- `s6` — `quant.py _heal_round reference= (:1591)`: agreement already against the original; final's `_bf16_artifact` uses the pre-heal bf16 export
  - seeds: `c7`
- `s7` — `jev_factory/distill/quant.py manifest (:1973-2031)`: `final_bf16` = (healed or measured)\[BF16\] and delta = final\['`delta_points`'\]: after a heal both the manifest's bf16 accuracy and its quant-vs-bf16 delta are relative to the healed bf16
  - seeds: `c8`
- `s8` — `jev_factory/distill/report.py c43_bars (:511)`: the `max_accuracy_drop_points` row reads `quant_manifest` `quant_vs_bf16_delta_points`; fixing the manifest field fixes the report row
  - seeds: `c9`
- `s9` — `jev_factory/distill/prereg.py QUANT_BARS (:54, example :281)`: the bar is a bare 'pos' value with no stated reference; the reference lives only in quant.py prose (:1136, :1382, :1530)
  - seeds: `c10`
- `s10` — `tests/test_distill_quant.py FakeMeasurer (:92) + only() helper (:491)`: only() hard-codes healed bf16 = original 0.9, so no existing test can see the reference; FakeMeasurer(healed={'bf16': ...}) can set it, accuracy in 5-point steps; no existing assertion breaks (manifest heal.`drop_points`==15 at :549 unchanged since healed bf16 == original there)
  - seeds: `c11`
- `s11` — `.devague/deliveries/orin-nano-jetson-ops-decider.json deviations + quant._checked_knobs (:1661)`: d1-d16 all approved on t23; knobs (`heal_mix`, gating, `heal_repeat`) need --deviation-id, but a semantic fix like this is not a knob; repo practice files a deviation for rule changes
  - seeds: `c12`
- `s12` — `docs/deliveries/2026-10-15-orin-nano-jetson-ops-decider.md:173, docs/distill.md:261, pipeline.py _bf16_artifact (:866)`: the delivery lists the heal-rule question as open for the operator; distill.md defines agreement but never the healed drop; the `_bf16_artifact` docstring points to that follow-up
  - seeds: `c13`
- `s13` — `docs/specs|plans|deliveries 2026-10-14 distillery and 2026-09-30 extract`: 'its own bf16' appears in historical spec/plan/delivery text; those are append-only records of what was decided then
  - seeds: `c14`
- `s14` — `docs/evidence/2026-10-16-jetson-ops-proving-run.md`: shipped artifact is un-healed `torch_int4_wo`; three heal rounds were snapshots (quant.round1-3-snapshot), not shipped
  - seeds: `c15`
- `s15` — `jev_factory/decide/rules.py (:63, :239-280, :505-520)`: QuantCheck takes bf16/quant `right_pct` from its caller (factory/pipeline.`_quant_check`); no distill input
  - seeds: `c16`
- `s16` — `jev_factory/factory/pipeline.py stage_heal (:1642) + backbones/causal_lm/quantize.py loss_points (:603)`: separate code with the same own-bf16 shape: `bf16_val`/`bf16_mc` are measured from the healed run's merged model; also `new_wrong_mutating_ids` compares against the healed bf16; feeds jev decide via `_quant_check` (:2496)
  - seeds: `q1` (question, resolved)

## Decisions

- jev run's heal (factory/pipeline.py `stage_heal`, `causal_lm`/quantize.`loss_points` + `new_wrong_mutating_ids`) is fixed separately, not in this change (operator decision on q1)

## Hard questions

- Does the same fix apply to jev run's heal (factory/pipeline.py `stage_heal` :1642-1710 measures `bf16_val` from the healed merged model; `causal_lm`/quantize.`loss_points` and `new_wrong_mutating_ids` compare against it), or is that a separate change? (resolved: Operator: separate follow-up; this change covers distill quantize only)

## Open parks

- [follow_up] jev run's heal compares the healed quant against the healed bf16 (factory/pipeline.py:1689-1710); file an issue and fix in its own change
