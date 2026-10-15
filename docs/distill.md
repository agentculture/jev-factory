# The distillery: Clef 27B to Strands Decider 2B

The distillery is `jev distill <stage>`: a staged, dry-run-by-default path that
turns a large teacher (Cloudflare **Clef 27B**) into a small calibrated student
(**Strands Decider 2B**, a pointer-scorer on `Qwen/Qwen3.5-2B-Base`). The
operator's brief is issue #18; the confirmed spec and plan are
`docs/specs/2026-10-14-distillery-clef-27b-to-strands-decider-2b.md` and
`docs/plans/2026-10-14-distillery-clef-27b-to-strands-decider-2b.md`.

> **Strands Decider is not the Layer-3 decider.** "Strands Decider" is the
> name of an external model. jev-factory's Layer-3 *decider* (the jev-like
> model whose candidates are `jev decide` verdicts, see `CLAUDE.md`) is a
> different thing. The distillery does not build, train or touch it, and the
> Layer-3 circularity guards are not affected. The distillery is also not an
> acceptance gate for the jev-tool model; that stays the first and only
> acceptance target of the extraction plan.

**State of the work.** The distillery code is built and tested on fixtures.
No distill run has been executed on the box yet (the operator run, t26, has
not happened), so no distilled student, bundle or final report exists. Do not
describe any of them as shipped.

## Before-state evidence

The spec's "Before" claim is that jev-factory could only train letter-token
causal-LM scorers with CE on gold, with no soft labels, no `/v1/systemone`
scorer and no typed-question data contract. That is checkable at the base
commit `8aec9c4` (the commit before the distillery work started). The commands
were run from the repo root on 2026-10-14:

```console
$ git grep -n -i -E "kl_div|soft.?label|systemone" 8aec9c4 -- jev_factory
$ echo "exit=$?"
exit=1
$ git ls-tree 8aec9c4 jev_factory/backbones/
100644 blob 44451bf5c109e65ddd5443801e4fb78a496a6879  jev_factory/backbones/__init__.py
040000 tree 9b225c67fdcfa3d81dd4e6be5ff3e87ca78c8438  jev_factory/backbones/causal_lm
```

`git grep` exits 1 with no output when nothing matches, so there was no
soft-label, KL or SystemOne code anywhere in `jev_factory/` at `8aec9c4`.
`jev_factory/backbones/` held only the package `__init__.py` and the
`causal_lm` backbone. The same commands against `HEAD` now find the
distillery code (`jev_factory/distill/`, `jev_factory/backbones/pointer/`).

## The two consumers

1. **The operator** runs `jev distill <stage>` on the DGX Spark to turn the
   teacher into a student. They approve every `--apply`, review drafted
   corrections, and make the call on going public (always a human decision).
2. **lobes' `decider` role** and the mesh callers behind it (routing, triage,
   guardrails) consume the distilled student over `/v1/systemone`. A bf16
   student ships in the `hobson-hf-export/1` layout that `strands-decider serve`
   loads unmodified; a torchao int8/int4 student (deviation d3) ships as a
   `jev_quant.json` artifact served by jev-factory's thin wrapper
   (`python -m jev_factory.backbones.pointer.quantized serve`), which answers
   the same wire.

## The lobes split

Serving and training live in different repos.

- **Serving** is in lobes-cli, in its `orin-decision-models` work: the
  `decider` role runs Clef (W4A16 build) on the Orin and the 2B (bf16) on any
  box. lobes passes `/v1/systemone` through unchanged.
- **Training** is here. jev-factory owns the data contract, teacher labelling,
  audit, corrections, the distill trainer, selection, calibration, the gate,
  quantization, the single final measurement and the private bundle.

jev-factory never edits lobes-cli. What crosses the boundary is the bundle
(model folder plus dataset folder) and the `/v1/systemone` wire format.

## The record contract

The wire format is `POST /v1/systemone` with a body
`{model, state, questions: {qid: {type, instructions?, criteria}}}`. Each
question answers with per-option `probabilities` (`choice`, `score`) or `noul`
(P(true)). The code is in `jev_factory/distill/systemone.py`.

- **Only probabilities are read.** The vendor's scalar `confidence` is defined
  differently per model and is never read, and neither are the reported argmax
  or expected score. An answer is a distribution only if it covers exactly the
  offered options and sums to 1 within `SUM_TOLERANCE`; anything else is an
  error result, never a label.
- **A record** (`jev_factory/distill/records.py`) is
  `{id, state, questions, gold per question, split, source, licence}` plus a
  `gold_provenance` of `human` or `model-generated`. Question types are
  `choice` (2 or more options), `noul` (true/false; gold is P(true)) and
  `score` (2 or more ordered levels named by index). Records and teacher
  labels are keyed by `(record id, question id)`, never by row position.
- **Loaders:** `from_typed_decisions_row` (LocalLLaMA/typed-decisions; gold is
  model-generated), `from_jev_bench_row` (Praveenrajus/jev-bench; labels are
  human) and `from_domain` (a tool domain as one `choice` question whose
  criteria are its verbs).
- **Metrics** (`jev_factory/distill/metrics.py`) are reported per question
  kind and overall, each with a seeded bootstrap CI.

## Data sides (deviation d2)

`jev_factory/distill/dataset.py` builds the sides. Only licences that are
exactly `apache-2.0` or `mit` are kept. Records over the student's 4096-token
window are dropped and counted per source, never truncated. Each source's own
`test` split is the test side. From its `train` split the build carves a sealed
**held-out** set and then grouped **train**, **fit** and **selection** sides.
Only the train side may become a training target; fit and selection are
evaluation folds (calibration and the gate are fitted on fit, the selection
rule reads selection). The sealed held-out and test sides are touched once, in
the `final` stage.

## The stages

`jev distill` with no stage lists them; the registry is
`jev_factory/distill/pipeline.py` (`STAGE_ORDER`). The stage is a positional
argument, so the jev-CLI domain gains one `jev.distill` operation (deviation
d5), not fourteen; the jev-tool model's candidate set stays inside its
52-letter bound. Every stage is dry-run by default and `--apply` commits;
`--apply --detach` runs it in its own session. Each writes a per-stage manifest
under `<work>/distill/manifests/`, which `jev status <work>` reads beside the
`jev run` manifests.

| # | Stage | What it does | Module |
|---|---|---|---|
| 1 | `preregister` | Locks bars, selection rule, knob grid and upstream knobs (each upstream value cites its file and key). A changed file is refused without a deviation id. | `prereg.py` |
| 2 | `dataset` | Licence allowlist, length filter, grouped train / fit / selection / held-out / test sides. | `dataset.py` |
| 3 | `contamination` | Marks each evaluation source `clean`, `v21-evaluated` or `overlapping` against v21's training data and Clef's eval suites. Name-based only; "clean" means "not found". | `contamination.py` |
| 4 | `teacher-label` | Asks Clef for per-option soft labels on the **train side only**, jointly by default, with a response cache so a killed run re-sends only unanswered items. | `teacher.py` |
| 5 | `audit` | Scores the teacher against human gold per source (n, accuracy, log-loss, ECE) and flags `teacher_weak` below the pre-registered floor. | `audit.py` |
| 6 | `corrections-draft` | Drafts proposed corrections (author, date, reason `weak` or `outdated`), leakage-checked. | `corrections.py` |
| 7 | `assemble` | One target per `(record, question)` with one winning source; freezes the training set. | `assemble.py` |
| 8 | `benchmark` | Zero-shot Clef and stock v21 on fit and selection folds only. | `benchmark.py` |
| 9 | `train` | Distill grid, calibrate per kind, select by the pre-registered rule. | `train.py` |
| 10 | `quantize` | Spikes deployed formats, heals at most once, recalibrates on the deployed build. | `quant.py` |
| 11 | `gate` | Fits the per-kind answer-or-abstain gate on the fit fold (deviation d4). | `gate.py` |
| 12 | `final` | Scores Clef, stock v21 and the student once on test and held-out. | `final.py` |
| 13 | `bundle` | Writes the model bundle and the dataset bundle. | `bundle.py` |
| 14 | `upload` | Private upload, fetch-back, sha256 verification. | `bundle.py`, `jev_factory/release/hub.py` |

Details that matter when operating them:

- **Teacher audit and corrections.** `assemble` refuses until a valid audit
  table naming the teacher build exists. One run uses one teacher build, and
  labels from another build are a new labelling run or a recorded deviation.
  Sources whose gold is model-generated (typed-decisions) are reported but
  never flag Clef, because a model's gold cannot indict the teacher.
- **Precedence per `(record, question)`:** operator correction, then human
  gold, then an alternative teacher, then Clef. The winner and the rule are
  written on the row. A `teacher_filter` knob (`all`, or only rows where the
  teacher agrees with gold) is pre-registered.
- **Operator review.** Drafted corrections enter as `proposed`. They train
  only after `jev review --corrections FILE` records an approval or edit
  (dry-run, `--serve`, `--apply`); the review file is the record. The pipeline
  never approves anything itself.
- **Trainer.** loss = `alpha` x KL(teacher, student) + (1 - `alpha`) x CE(gold),
  with a per-kind temperature, LoRA first, run as a subprocess under the GPU
  guard (`jev_factory/backbones/pointer/train_distill.py`). With alpha = 0 it
  equals the gold-CE path, and a row with no teacher target has zero KL.
- **Pins.** The Qwen3.5-2B-Base revision, the v21 commit and v21's
  `adapter_sha256` / `head_sha256` are verified before training
  (`pins.py`). The teacher, the student runtime and the trainer each run in
  their own pinned environment (`envs.py`), served on `127.0.0.1` only.
- **Quantized student path (deviation d3).** `strands-decider serve` has no
  int8 or int4 path for the 2B, so `quantize` builds torchao weight-only
  `torch_int8_wo` and `torch_int4_wo` artifacts (torso Linear weights
  quantized, pointer head kept fp32; `backbones/pointer/quantized.py`) next to
  the bf16 winner. A format whose build fails is recorded unavailable with its
  error, never dropped. Each is measured and calibrated on its own predictions.
- **Decision gate (deviation d4).** Typed questions have no read-only /
  mutating split. The gate decides per question kind whether to answer or
  abstain from `floor`, `margin` and `max_entropy` thresholds, fitted on the
  fit fold only (`gate.py`, reusing `jev_factory.core.gate` comparisons). Its
  coverage and selective accuracy are reported on selection and, at the end,
  on the final predictions. It is reported, not a pass/fail bar.
- **Calibration.** jev-factory fits it (temperature on the fit fold, then an
  optional per-label vector kept only if it lowers selection ECE). Strands'
  published `temperature_by_kind` is never the deployed calibration.

## Success signals and the headline rule

Both are pre-registered in `distill/prereg.json` before the first training
command, and computed from the single final measurement by
`jev_factory/distill/report.py`.

- **c42, on the final test side:** the student's accuracy beats stock v21 with
  the paired-bootstrap 95% CI of the difference above 0 and closes at least
  half of the v21-to-Clef accuracy gap; calibrated ECE is at most 0.10 and no
  worse than v21's; confident-wrong is no worse than v21's; median latency is
  at most one fifth of Clef's on the same box.
- **c43, on the quant:** within 3.0 accuracy points of its own bf16 after at
  most one heal, with a recalibrated ECE at most 0.10.
- **Headline rule:** the headline pool is **human-gold** records from
  **clean** sources only. Overlapping and v21-evaluated sources are reported
  separately. Agreement with model-generated gold (typed-decisions) is a
  secondary table and is never gated. The three models are resampled jointly,
  and a higher accuracy with a higher ECE than v21 is reported as a regression.
- A miss is recorded as a miss; the report is still written, and a run that
  misses either signal ships a failure report, not the announcement.

## The bundle

`bundle` writes two folders. `model/` is the deployed student exactly as
`quant/format_decision.json` chose it, with jev-factory's fitted
`temperature_by_kind`, plus `calibration.json`, `gate.json`, `scorer-train.json`
(an index of the training set with each row's target source, licence and
teacher build, no text), `report.json`, `assemble_manifest.json` and
`distill_provenance.json`. `dataset/` holds `train.jsonl`, only rows whose
licence is Apache-2.0 or MIT, each keeping its `licence`, `source` and
`target_source`. Neither folder holds sealed held-out or test text. Upload is
private-first, fetched back and hash-verified; there is no public-visibility
call.

## Invariants that carry over

These match the factory's and are enforced in code, not exposed as knobs: the
teacher labels the train side only; test and held-out are measured once and a
second final needs a deviation id; leakage fails closed; every write is
dry-run unless `--apply`; the base install keeps `dependencies = []` and heavy
packages stay behind the train extra or an external venv; servers bind
localhost only; remote model code runs only at a pinned revision.
