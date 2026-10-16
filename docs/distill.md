# The distillery: Clef 27B to Strands Decider 2B

The distillery is `jev distill <stage>`: a staged, dry-run-by-default path that
turns a large teacher (Cloudflare **Clef 27B**) into a small calibrated student
(**Strands Decider 2B**, a pointer-scorer on `Qwen/Qwen3.5-2B-Base`). The
operator's brief is issue #18; the confirmed spec and plan are
`docs/specs/2026-10-14-distillery-clef-27b-to-strands-decider-2b.md` and
`docs/plans/2026-10-14-distillery-clef-27b-to-strands-decider-2b.md`.

Issue #20 retargets the same distillery to a Jetson-ops specialist small enough
for an Orin Nano (spec and plan:
`docs/specs/2026-10-15-orin-nano-jetson-ops-decider.md`,
`docs/plans/2026-10-15-orin-nano-jetson-ops-decider.md`; runtime evidence:
`docs/evidence/2026-10-jetson-runtime-spike.md`; bundle contract:
`docs/distill-bundle-format.md`). The sections below describe the code as it
stands after that retarget. Where the #18 wording differed, this page follows
the code.

> **Strands Decider is not the Layer-3 decider.** "Strands Decider" is the
> name of an external model. jev-factory's Layer-3 *decider* (the jev-like
> model whose candidates are `jev decide` verdicts, see `CLAUDE.md`) is a
> different thing. The distillery does not build, train or touch it, and the
> Layer-3 circularity guards are not affected. The distillery is also not an
> acceptance gate for the jev-tool model; that stays the first and only
> acceptance target of the extraction plan.

**State of the work.** The distillery code is built and tested on fixtures.
No distill run has been executed on the box yet (the #18 operator run, t26,
and the #20 operator run, t23, have not happened). **No #20 model has been
trained or measured**: no distilled student, quantized build, bundle, edge
measurement or final report exists, and nothing here may be described as
shipped or as a measured result.

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
   loads unmodified; a quantized student (deviation d3) ships as a
   `jev_quant.json` artifact served by jev-factory's thin wrapper
   (`python -m jev_factory.backbones.pointer.quantized serve`), which answers
   the same wire plus `/v1/bundle` and `/v1/stats`. For #20 the quantized
   artifact is the one meant for an Orin Nano; hosting it as a lobes decider
   domain is lobes' separate plan, which consumes
   `docs/distill-bundle-format.md`.

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
- **Validation.** A record whose question kind is not `choice`, `noul` or
  `score`, or that is not a valid `/v1/systemone` body, is refused with its id.
  A `cc-by-4.0` record must carry an `attribution` string (older records
  without the field still load).
- **Loaders:** `from_typed_decisions_row` (LocalLLaMA/typed-decisions; gold is
  model-generated), `from_jev_bench_row` (Praveenrajus/jev-bench; labels are
  human) and `from_domain` (a tool domain as one `choice` question whose
  criteria are its verbs).
- **Metrics** (`jev_factory/distill/metrics.py`) are reported per question
  kind and overall, each with a seeded bootstrap CI.

## Data sides (deviation d2)

`jev_factory/distill/dataset.py` builds the sides. Only licences that are
exactly `apache-2.0`, `mit` or `cc-by-4.0` are kept; a `cc-by-4.0` record
without an attribution string is refused, and the manifest's licence table
carries per-licence counts and the attribution strings. A question with more
than 24 options (the student's `num_slots`) is refused as `over_slots`, and a
record over the student's 4096-token window as `over_length`. Both are counted
by reason (the length drops also per source) and nothing is truncated. Each source's own
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
d5), not fifteen; the jev-tool model's candidate set stays inside its
52-letter bound. Every stage is dry-run by default and `--apply` commits;
`--apply --detach` runs it in its own session. Each writes a per-stage manifest
under `<work>/distill/manifests/`, which `jev status <work>` reads beside the
`jev run` manifests.

| # | Stage | What it does | Module |
|---|---|---|---|
| 1 | `preregister` | Locks bars, selection rule, knob grid and upstream knobs (each upstream value cites its file and key). A changed file is refused without a deviation id. | `prereg.py` |
| 2 | `dataset` | Licence allowlist, length filter, grouped train / fit / selection / held-out / test sides. | `dataset.py` |
| 3 | `contamination` | Marks each evaluation source `clean`, `stock-evaluated` or `overlapping` against the 2610 card's 48 datasets, its non-card training sources (deviation d1) and Clef's eval suites. Name-based only; "clean" means "not found". | `contamination.py` |
| 4 | `teacher-label` | Asks Clef for per-option soft labels on the **train side only**, jointly by default, with a response cache so a killed run re-sends only unanswered items. | `teacher.py` |
| 5 | `audit` | Scores the teacher against human gold per source (n, accuracy, log-loss, ECE) and flags `teacher_weak` below the pre-registered floor. | `audit.py` |
| 6 | `corrections-draft` | Drafts proposed corrections (author, date, reason `weak` or `outdated`), leakage-checked. | `corrections.py` |
| 7 | `assemble` | One target per `(record, question)` with one winning source; freezes the training set. | `assemble.py` |
| 8 | `benchmark` | Zero-shot Clef and the stock model on fit and selection folds only; reproduces the stock model's published JevBench w4096 result. | `benchmark.py` |
| 9 | `train` | Distill grid, calibrate per kind, select by the pre-registered rule. | `train.py` |
| 10 | `quantize` | Builds the quantized candidates, applies the size and agreement bars, heals at most once, recalibrates on the deployed build. | `quant.py` |
| 11 | `gate` | Fits the per-kind answer-or-abstain gate on the fit fold (deviation d4). | `gate.py` |
| 12 | `final` | Scores Clef, the stock model, the bf16 student and the shipped quant once on test and held-out. | `final.py` |
| 13 | `edge` | Measures the shipped quant on a board (AGX Orin or Nano) through a tunnel, on non-test data only. | `edge.py` |
| 14 | `bundle` | Writes the model bundle and the dataset bundle, with a NOTICE. | `bundle.py` |
| 15 | `upload` | Pre-upload scans, private upload, fetch-back, sha256 verification. | `bundle.py`, `jev_factory/release/hub.py` |

Details that matter when operating them:

- **Teacher over a tunnel.** The teacher (Clef W4A16 behind the Orin
  gateway) is never contacted directly: the operator opens an SSH tunnel and
  the run config points at `127.0.0.1`. Non-localhost URLs are refused.

  ```console
  ssh -N -L 127.0.0.1:<port>:127.0.0.1:8000 <orin>
  ```

  The run config sets `distill_teacher_url` to `http://127.0.0.1:<port>`,
  `distill_teacher_key_env` to the **name** of the environment variable that
  holds the bearer key, and `distill_teacher_model_header` to
  `decider:general` (sent as `X-Lobes-Model`). The key's value is read from
  the environment at request time and is never written to a manifest, log or
  error; a missing variable exits 2 naming the variable. The teacher-label
  manifest records the variable's name and the header value, not the key.
  Labelling is sequential, one request at a time, and must not starve Clef's
  production lane.
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
- **Human gold for the test side (#20).** An agent-drafted test record counts
  as human gold only when the operator approved that exact record.
  `jev review --gold FILE` lists the drafted test records with their proposed
  gold and appends `approve`, `reject` or `edit` decisions (`--decide
  ID=approve|reject`, `--edit ID:QID=LABEL`, or the `--serve` page) to
  `<records stem>.gold-ledger.jsonl`. The ledger is append-only and binds each
  id to the record's sha256, so an edited record gets a new sha256 and its old
  approval no longer matches; a malformed ledger fails closed. Like every
  write verb it is dry-run unless `--apply`. `final` reads the ledger (run
  config key `distill_gold_ledger`) and refuses any test record without an
  approval for its current sha256, printing ids only; the headline counts only
  approved human-gold test records.
- **Draft review (t25).** `jev review --drafts FILE [--evidence FILE]` goes
  through drafted records of every split one at a time: `keep`, `drop`, `edit`
  (a label or per-option probabilities), `add_option` (for example `escalate`),
  `restore` or `note` (deviation d7), either on the `--serve` page or with `--decide` / `--edit` and
  `--apply`, into `<records stem>.draft-ledger.jsonl`. `--reviewed-out FILE`
  writes the reviewed drafts (dropped removed, edits applied). An edited test
  record still needs its gold approval through `--gold`.
- **Jetson-ops sources (deviation d3).** The Jetson-ops records are drafted
  by an agent from pinned public sources: NVIDIA public repositories and
  `jev_factory/distill/jetson_ops/sources.json` (each source pinned by repo
  URL and full commit, with its licence per path class and the CC-BY-4.0
  attribution text where it applies), plus agent-drafted general Jetson
  operations. They are **not** the operator's own stack. Test candidates are
  drafted from text disjoint from the train pool, hostnames and addresses are
  redacted at draft time, and no Clef probe-set text appears in them.
- **Trainer.** loss = `alpha` x KL(teacher, student) + (1 - `alpha`) x CE(gold),
  with a per-kind temperature, LoRA first, run as a subprocess under the GPU
  guard (`jev_factory/backbones/pointer/train_distill.py`). With alpha = 0 it
  equals the gold-CE path, and a row with no teacher target has zero KL.
- **Pins and the 2610 retarget.** The student is
  `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` at revision
  `216d1ccb` (`DECIDER_REPO` and the revision pin in `pins.py`; the
  adapter and head sha256 come from its `provenance.json`). The
  Qwen3.5-2B-Base revision, that commit and the two digests are verified
  before training. The pre-registration is schema version 2 and seeds LoRA
  rank 48, alpha 96, dropout 0.05, the 12 target modules of 2610's
  `adapter_config.json` and its per-kind temperatures; a #18-shape
  pre-registration is refused with a message, not reinterpreted. The old
  reference role `stock_v21` is now the model-neutral `stock` (config keys
  `distill_stock_url` / `distill_stock_model`, the `*_no_worse_than_stock`
  bars). The teacher, the student runtime and the trainer each run in their
  own pinned environment (`envs.py`), served on `127.0.0.1` only.
- **Benchmark gates.** `benchmark` serves the stock model on the pinned base
  revision and scores JevBench w4096. A result whose confidence interval
  excludes the card's published 0.7792 (180/231) marks the run
  `base-mismatch`, and `train` then refuses without a deviation id. The stock
  model is also measured here, on the Clef probe set and on the fit and
  selection folds, never copied from elsewhere.
- **Quantized student path (deviation d3, #20).** `strands-decider serve` has
  no int8 or int4 path for the 2B, so `quantize` builds its own artifact
  (`backbones/pointer/quantized.py`): the LoRA is merged, the vision tower and
  the MTP head are dropped, the text-block linears are int4 weight-only, the
  embedding is packed over the **full vocabulary** (the unmodified 2610
  tokenizer ships with it), and the pointer head stays fp32. `in_proj_a`/`b`,
  `conv1d`, `A_log`, `dt_bias` and the norms stay unquantized. The manifest
  records each file's sha256 and the byte sum of the weight files. A build at
  or over 10^9 bytes is recorded **unavailable** with its byte count and is
  never served, measured or chosen; a failed build is recorded unavailable
  with its error, never dropped. A second artifact with a pruned vocabulary
  can be built for comparison (the ids that train, fit and selection prompts
  use); it refuses out-of-vocabulary tokens, is reported beside the full
  build and is **never shipped**. The served wrapper adds `GET /v1/bundle`
  (per-file sha256 attestation) and `GET /v1/stats` (process RSS,
  `MemAvailable`, device model, the L4T release when present, and the cgroup
  `memory.max`). Whether int4 runs on `sm_87` is a question for the board:
  the spike evidence is in `docs/evidence/2026-10-jetson-runtime-spike.md`.
- **Heal, once.** When the chosen quant misses the bars, one heal round
  trains the quantized student against targets that mix the bf16 distilled
  student's and Clef's per-option distributions (the mix is pre-registered).
  Agreement with the bf16 student is computed on the fit and selection folds
  only and must reach **0.97**; a test or held-out id in the heal input exits
  non-zero, and a second heal of the same format is refused without a
  deviation id. That round may train on another mix
  (`--knob heal_mix='{"bf16_student": 1.0, "clef": 0.0}'`), which also needs
  the deviation id; the heal log records both mixes. A healed build under
  0.97 is not shippable.
- **No bf16 fallback.** When no quantized build is under the byte budget and
  survives safety and calibration, `quantize` records a failed `no qualifying
  format` result and exits non-zero. `final` and `bundle` refuse that
  manifest; nothing ships.
- **Final with the AGX-served quant.** `final` scores four models once: Clef
  W4A16, the stock model, the bf16 student and the shipped quant. The quant is
  pre-served on the AGX Orin and reached through a tunnel at
  `distill_agx_url` (`http://127.0.0.1:<port>`). Before any test request,
  `final` checks that `GET /v1/bundle` equals the deployed artifact's
  attestation (per-file sha256 and manifest sha256) and that `GET /v1/stats`
  reports a cgroup memory cap, and refuses otherwise. It also refuses while
  the teacher-label lock exists. The bundle is fetched again after scoring and
  a change fails the run. The cap `/v1/stats` reported is recorded in
  `final.json`.
- **Edge stage.** `edge` measures the shipped quant on a board (AGX Orin or
  Orin Nano, labelled by the `label` knob) through a tunnel at
  `distill_edge_url`, on the **selection fold and the probe set only**; test
  and held-out ids exit non-zero before any request. It records median and p95
  latency, resident memory and `MemAvailable` from `/v1/stats`, and
  prediction agreement with the quant stage's own selection-fold predictions
  of the same bundle. It refuses when `/v1/bundle` differs from the deployed
  bundle, when `/v1/stats` reports no cgroup cap (or one other than
  `distill_edge_memory_cap`), or while the teacher-label lock exists. The
  manifest records the board model, the L4T release and the cap. The probe
  set is private: only its sha256 and count are written.
- **Decision gate (deviation d4).** Typed questions have no read-only /
  mutating split. The gate decides per question kind whether to answer or
  abstain from `floor`, `margin` and `max_entropy` thresholds, fitted on the
  fit fold only (`gate.py`, reusing `jev_factory.core.gate` comparisons). Its
  coverage and selective accuracy are reported on selection and, at the end,
  on the final predictions. It is reported, not a pass/fail bar.
- **Calibration.** jev-factory fits it (temperature on the fit fold, then an
  optional per-label vector kept only if it lowers selection ECE). Strands'
  published `temperature_by_kind` is never the deployed calibration.

## Serving on a board: the tunnel and the cgroup cap

jev never ssh-executes on a board. The operator starts the server on the board
under a cgroup memory cap, opens a tunnel, and `final` or `edge` talks to
`127.0.0.1`. The wrapper binds loopback only and takes `ARTIFACT` (the
quantized folder), `--host` (default `127.0.0.1`), `--port` (default 8000),
`--device` (default `cuda`) and `--model-name`. On the board, in the student
environment:

```console
systemd-run --user -p MemoryMax=<cap> \
    python -m jev_factory.backbones.pointer.quantized serve ARTIFACT \
    --host 127.0.0.1 --port N
```

From the DGX Spark, forward the board's port (a Nano and an AGX Orin each get
their own local port), then set `distill_agx_url` or `distill_edge_url` to the
forwarded `http://127.0.0.1:<local port>`:

```console
ssh -N -L 127.0.0.1:<local port>:127.0.0.1:N <board>
```

`<cap>` is a systemd size such as `4G`; set `distill_edge_memory_cap` to the
same value so `edge` refuses a server running under a different cap. Run
nothing on the AGX Orin without the operator's go, and never while teacher
labelling runs. Which kernels actually run on `sm_87` is settled by the board
evidence in `docs/evidence/2026-10-jetson-runtime-spike.md`, not by this page.

## Success signals and the headline rule

Both are pre-registered in `distill/prereg.json` before the first training
command, and computed from the single final measurement by
`jev_factory/distill/report.py`.

- **c17 (#20), on the final test side, for the bf16 student and for the
  shipped quant each:** top-1 at least the stock model plus 10 points, and
  within 5 points of Clef W4A16 (at most 5 points below it). For the quant
  also: agreement with the bf16 student at least 0.97, and weight bytes under
  10^9. Median and p95 latency are reported with no bar. Each bar is
  computed with paired bootstrap CIs and cites the hashed pre-registration; a
  pre-registration missing any of them is refused. The Clef probe-set table
  carries no pass/fail.
- **#18's c42 and c43 bars** (gap closure, latency at most one fifth of
  Clef's, quant within 3.0 points of its bf16) were replaced for this run by
  c17; the schema is versioned so a #18-shape pre-registration is refused.
- **Headline rule:** the headline pool is **human-gold** records from
  **clean** sources only, and for the Jetson-ops test side only operator-
  approved records (see the gold ledger above). Overlapping and
  stock-evaluated sources are reported separately. Agreement with
  model-generated gold (typed-decisions) is a secondary table and is never
  gated. The models are resampled jointly, and a higher accuracy with a
  higher ECE than the stock model is reported as a regression.
- A miss is recorded as a miss; the report is still written, and a run that
  misses either signal ships a failure report, not the announcement.

## The bundle

`bundle` refuses a quant manifest that shipped nothing. It writes two folders.
`model/` is the deployed student exactly as `quant/format_decision.json`
chose it (see `docs/distill-bundle-format.md` for every file and the loader
contract), with jev-factory's fitted
`temperature_by_kind`, plus `calibration.json`, `gate.json`, `scorer-train.json`
(an index of the training set with each row's target source, licence and
teacher build, no text), `report.json`, `assemble_manifest.json` and
`distill_provenance.json`. `dataset/` holds `train.jsonl`, only rows whose
licence is Apache-2.0, MIT or CC-BY-4.0 with an attribution string, each
keeping its `licence`, `source` and `target_source`. Both folders carry a
`NOTICE` with every attribution string from the dataset manifest, and a
CC-BY-4.0 record without it is refused. Neither folder holds sealed held-out
or test text. Before anything is sent, `upload` runs the pre-upload scans
(`scripts/scan-secrets.py`, a hostname and private-IP scan, a scan for the
names on an operator-held list named by `distill_side_model_list`, and a check
that no Clef probe-set record is present) and prints file and line only.
Upload is private-first, fetched back and hash-verified; there is no
public-visibility call.

## Invariants that carry over

These match the factory's and are enforced in code, not exposed as knobs: the
teacher labels the train side only; test and held-out are measured once and a
second final needs a deviation id; leakage fails closed; every write is
dry-run unless `--apply`; the base install keeps `dependencies = []` and heavy
packages stay behind the train extra or an external venv; servers bind
localhost only; remote model code runs only at a pinned revision.
