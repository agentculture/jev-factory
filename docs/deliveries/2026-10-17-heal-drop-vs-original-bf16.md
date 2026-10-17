# Delivery Summary — heal drop vs original bf16

plan: `heal-drop-vs-original-bf16` · run: `complete` · date: `2026-10-17`
baseline: `devague summary skeleton`

## Intent

> A healed quant's accuracy drop is gated against the original selected bf16
> student, with the healed-bf16 drop and the heal drift recorded beside it as
> context

This closes the heal-rule question left open by the PR #21 code review (in the
follow-up list of the #20 delivery). After the change, every accuracy check in
distill quantize, before and after a heal, compares against the original
selected bf16 student. The healed bf16's own accuracy and the heal drift are
recorded as context and gate nothing. It was built in one session on branch
`fix/heal-drop-vs-original-bf16`, with no workforce fan-out (two tasks, run
serially by the main agent).

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Rebase the healed quant's drop on the original bf16 student in
  quant.py (`_against_original`, `heal_result` fields, `_healed_problem`
  message, manifest bf16 fields) with the new tests
- `t2` — Record d17 on the #20 plan, update docs/distill.md and the #20
  delivery follow-up, file the jev run heal follow-up issue, bump the version

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `_against_original` in `jev_factory/distill/quant.py`; `heal_result` gains `drop_points_vs_heal_bf16`, `heal_bf16_drift_points` and the three accuracies; the second heal trigger names the original bf16 student; the manifest's `bf16_selection_accuracy` is the original bf16's and it gains `healed_bf16_selection_accuracy`; three new tests in `tests/test_distill_quant.py`, marked `behavioral` for o1–o10 (commits `bef8463`, `b4bd88b`) |
| `t2` | delivered | d17 recorded and operator-approved on plan `orin-nano-jetson-ops-decider`; `docs/distill.md` "Heal, once" and the #20 delivery's follow-up updated; `pipeline._bf16_artifact` docstring updated; issue #22 filed; version 0.12.1 with a CHANGELOG entry (commit `bef8463`) |

## Mid-work Decisions

- `d17` (plan `orin-nano-jetson-ops-decider`, approved) — the healed quant's
  `drop_points` is measured against the original selected bf16 student, not
  the bf16 rebuilt from the healed checkpoint; the drop vs the healed bf16 and
  the heal drift are context only — PR #21 code review: the healed-bf16
  reference hides a heal that lowers the bf16 itself.
- q1 (operator): `jev run`'s heal (`factory/pipeline.py` `stage_heal`) has the
  same flaw and is fixed separately, as issue #22.
- q2 (operator, raised by `/challenge`): gate on the original only, even when a
  heal *raises* the bf16. That direction loosens the gate (a healed quant
  within 3 points of the original ships even if it is further below its healed
  bf16), and that is intended.
- Found while writing the test (lapse `l1`, approved): the 0.97 argmax-agreement
  bar against the original already bounds the accuracy gap, pooled over fit and
  selection. The loophole is narrower than first claimed: it is real only when
  the loss falls mostly on the selection fold. The red test needed a per-fold
  accuracy override in `FakeMeasurer` (`fold_acc`) to show it.
- Code review of PR #23 (`/code-review medium`, 7 findings): 6 fixed in one
  commit. `_measure_all` takes a `baseline=` (the original bf16) instead of
  having its result patched afterwards, and its refusal no longer says "its own
  bf16". A healed bf16's cited `accuracy_delta_points_vs_bf16` is its drift. The
  un-healed manifest no longer gains a null key. The manifest's heal record
  labels the healed build's drops under `healed_result`. The o3 gating test now
  uses a healed bf16 of 1.0, so it fails on the old code. 1 pushed back:
  `heal_result` repeating the original bf16 accuracy is what operator-confirmed
  c23 requires (the log entry alone re-derives every drop).
- Evidence is plan-scoped: one record (`e26`) first landed on the #20 plan's
  unrelated `o5` by default, and the operator rejected it. Evidence for this
  run was re-filed on this plan, pinned to `bef8463`.

## Drift From Plan

no drift. Both tasks were delivered as confirmed (see Actual Delivery). The
`FakeMeasurer` `fold_acc` extension and the `behavioral` markers are test-side
additions within `t1`'s "with the new tests". `d17` is the deviation this plan
implements, not drift from it.

## Evidence

- tests: `tests/test_distill_quant.py::test_a_heal_that_lowers_the_bf16_is_judged_against_the_original` — pass (red before the change: the healed quant shipped)
- tests: `tests/test_distill_quant.py::test_a_heal_that_raises_the_bf16_is_still_judged_against_the_original` — pass (red before the change: refused at 10.0 points vs its own bf16)
- tests: `tests/test_distill_quant.py::test_a_gated_heal_drop_is_against_the_original_on_the_domain_sources` — pass
- tests: `uv run pytest -n auto` — 3358 passed, 28 skipped
- lint: black, isort, flake8, bandit, `scripts/scan-secrets.py`,
  `markdownlint-cli2` on every edited doc — clean
- unchanged: `git diff --stat main -- jev_factory/distill/prereg.py jev_factory/decide jev_factory/distill/report.py` — empty
- devague: obligations o1–o10 each with approved evidence e1–e10 (6 at
  sensitivity, 4 at execution), plus e11 (o3, sensitivity, after review); deltas b1–b4 operator-approved
- commits: `7e834cf..b4bd88b` (plus this summary)
- issues: #22 (follow-up)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `c2` — after a heal, `_healed_problem` gates on the drop against the original bf16 student | high | test `test_a_heal_that_lowers_the_bf16_is_judged_against_the_original` (e1, sensitivity) |
| `c3`/`c23` — `heal_result` records the vs-healed drop, the drift and the three accuracies; no gate reads them | high | same test (e2, e9, sensitivity); e8 for the shippable case |
| `c4` — with `gating=clean`, both sides of the drop are domain-only | high | test `test_a_gated_heal_drop_is_against_the_original_on_the_domain_sources` (e3 execution; after review, red on the old code, e11 sensitivity) |
| `c6` — the un-healed path is unchanged | high | no pre-existing assertion edited; full suite passes (e4) |
| `c7` — agreement still uses the original predictions | high | lowers test asserts 39/40 (e5) |
| `c8` — manifest bf16 fields describe the original after a heal | high | test `test_a_heal_that_raises_the_bf16_is_still_judged_against_the_original` (e6, sensitivity) |
| `c9` — the report's c43 drop row reads the drop vs the original, with `report.py` unchanged | high | raises test calls the real `c43_bars` (e7); `report.py` diff empty |
| `c24` — the `measurements.json` healed block matches `heal_result` | high | lowers test (e10, sensitivity) |
| `c10`/`c16` — no prereg, `decide/` or report code change | high | `git diff --stat` above, empty |
| `c19` — why it matters: a heal could ship a build several points worse than the student | medium | capped by lapse `l1`: the agreement bar already bounds the gap; real only when the loss concentrates on selection (shown at 5 points with 39/40 agreement) |
| a real heal run behaves as the fakes do | unverified | park `v2`: tested on `FakeMeasurer` only; no heal has run on hardware since the change |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `assumption-for-measurement` | challenge pass missed that the 0.97 argmax-agreement bar vs the original already bounds a healed quant's accuracy gap (d points of accuracy imply >= d% disagreement), pooled over fit+selection; c19/c21 overstate the loophole, which is real only when the loss concentrates on the selection fold |

## Remaining Work / Follow-up

- Issue #22 — the same fix for `jev run`'s heal (`factory/pipeline.py`
  `stage_heal`, `causal_lm/quantize.loss_points` and `new_wrong_mutating_ids`);
  needs its own deviation.
- Park `v2` — the change is untested on a real heal; the next #20 heal run is
  its first real exercise.
- The #20 proving run is unaffected: its shipped int4 is un-healed, so its
  drop already is against the original bf16.
