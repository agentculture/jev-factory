# Build Plan — distillery: Clef 27B to Strands Decider 2B

slug: `distillery-clef-27b-to-strands-decider-2b` · status: `exported` · from frame: `distillery-clef-27b-to-strands-decider-2b`

> jev-factory ships a distillery: Cloudflare Clef 27B labels a train split with per-option soft probabilities, Strands Decider 2B is benchmarked zero-shot, distilled (KL to teacher + CE to gold, LoRA), re-evaluated and assessed against both teacher and its un-distilled self, then quantized and healed once — every step a dry-run-by-default stage with recorded provenance and a pre-registered gate (issue #18)

## Tasks

### t1 — Typed-decision record contract and source loaders

- instruction: New package `jev_factory`/distill/ (add `__init__.py`). Keep the record a frozen dataclass plus `to_json`/`from_json`; never key anything by row position. Fixtures go in tests/fixtures/`typed_decisions`/ (a few hand-written rows), never downloaded in tests. Mark gold provenance: typed-decisions gold is model-generated (a ~4B teacher's 3-sample mean); jev-bench labels are human (source datasets).
- covers: c27, h14
- acceptance:
  - `jev_factory`/distill/records.py defines a record {id, state, questions{qid:{type choice|noul|score, instructions?, criteria}}, gold per question, split, source, licence} and validates it with named errors (unknown type, choice with <2 options, score with <2 levels, gold outside the offered options)
  - A LocalLLaMA/typed-decisions row (state/questions/gold JSON columns) and a Praveenrajus/jev-bench row (id, source, primitive, split, state, question, label, `soft_label`) both load into the same record schema; tests/`test_distill_records.py` round-trips one fixture row of each
  - A tool Domain (tests/fixtures/`toy_domain`) renders as one choice question whose criteria are its operations; gold is marked human or model-generated per source
  - Importing the module pulls in no third-party package; parquet input, if supported, imports pyarrow lazily inside the loader

### t2 — Distill assessment metrics with bootstrap CIs

- instruction: Reuse core.metrics (`bootstrap_ci`, ece, `brier_one`) rather than re-implementing; do not route through core.metrics.compute, which is tool-scorer shaped (`expect_kind`). Pure Python, no numpy requirement unless already a dependency. Peak memory is a measurement field filled by the benchmark/final stages, not computed here.
- covers: c7, h5
- acceptance:
  - `jev_factory`/distill/metrics.py computes accuracy per kind, macro-F1, JSD to gold, JSD to teacher, confident-wrong rate (pre-registered threshold), log-loss and Brier over typed-question predictions, each with a seeded bootstrap CI via core.metrics.`bootstrap_ci`
  - tests/`test_distill_metrics.py` checks every metric against a hand-computed fixture and asserts the CI is identical across two runs with the same seed
  - Score questions are scored on the level index and on the full level distribution; noul on P(true)

### t3 — Distill pre-registration schema and lock

- instruction: Mirror factory/prereg.py's lock/`require_registered` pattern but keep a separate file (distill/prereg.json, distill/prereg.lock.json) so the jev-tool prereg is untouched. Seed the `upstream_knobs` example from strands-decider-2B-hobson-v21 `train_config.json` (lr 1e-4, `head_lr` 1e-3, 1 epoch, `kl_frozen_weight` 0.3, `teacher_weight` 1.0, `ordinal_smoothing` 0.1).
- covers: c10, h8, c15, h13
- acceptance:
  - `jev_factory`/distill/prereg.py defines a schema-validated distill prereg: success bars (c42/c43 thresholds), selection rule, knob grid (alpha, per-kind distillation temperature, `teacher_filter` all|`agrees_with_gold`, teacher-weak floor, data mix, epochs, lr), reference rows (Clef, stock v21) and an `upstream_knobs` list where every value taken from strands-decider config cites its source file and key
  - lock()/`require_locked`() hash the file like factory.prereg; a changed file is refused without a deviation id; tests/`test_distill_prereg.py` proves both
  - A knob present in the run but missing from the prereg, or an upstream value without a cited source, is refused with a named error

### t4 — SystemOne client and probability-only response parser

- instruction: Use a stdlib http.server fake in tests (see how evals/providers tests fake providers). Record model id, request config and a sha256 of the raw response body on every result for provenance.
- covers: c28, h15
- acceptance:
  - `jev_factory`/distill/systemone.py builds a POST /v1/systemone body from a record and parses answers into per-question distributions over exactly the offered options, using only 'probabilities' (choice, score) and 'noul' P(true)
  - tests/`test_distill_systemone.py`: a fake server returning a deliberately wrong 'confidence' yields a byte-identical parsed result; a missing option, extra option or sum outside 1 +/- 0.01 is an error result, never a distribution
  - Non-localhost base URLs are refused via measure/preflight.`require_localhost`; the client uses stdlib urllib only

### t5 — Backbone protocol and registry, with an import-graph test

- instruction: Do not rewire the existing `causal_lm` call sites in pipeline.py/measure/release here; that is out of scope and would collide with other tasks. Only add the protocol, the registry and the import-graph test (extend the existing layering test style).
- covers: h3
- acceptance:
  - `jev_factory`/backbones/base.py defines the per-backbone protocol (render, predict distributions, train plan, merge/verify, export, quantize candidates) and backbones/registry.py registers `causal_lm` and pointer by name, importing each lazily
  - tests/`test_backbone_seam.py` asserts the distribution-only modules (core.calibration, core.gate, core.`sweep_gate`, distill.metrics, decide) import neither `jev_factory`.backbones.`causal_lm` nor `jev_factory`.backbones.pointer
  - Existing `causal_lm` callers keep working unchanged (full suite green)

### t6 — Pinned environments and localhost-only server launcher

- instruction: Servers start as subprocesses (strands-decider serve; Clef via the transformers `load_release_model` sidecar or SGLang dev-clef image) with explicit pins from the run config; reuse factory/detach.py for detached processes and measure/preflight.`require_localhost`. Add run-config keys via factory/config.py KEYS only if needed, prefixed `distill_`.
- covers: c46, h31, c47, h32
- acceptance:
  - `jev_factory`/distill/envs.py describes three environments (teacher, student runtime, trainer) by python path or image digest plus a lockfile hash, and records them into the run provenance
  - Launching a teacher or student server with an unpinned HF revision or package version, or a bind other than 127.0.0.1, is refused; tests/`test_distill_envs.py` proves each refusal
  - Importing `jev_factory` with torch, transformers, peft, `strands_decider` and sglang blocked via sys.modules succeeds, and pyproject dependencies stays \[\]

### t7 — Student input pins and hash verification

- instruction: Work from a local cache dir; no network in tests. Upstream recorded its base revision only as 'inferred' (b1485b2f...), so the pin is ours and must be explicit in the run config.
- covers: c48, h33
- acceptance:
  - `jev_factory`/distill/pins.py verifies Qwen/Qwen3.5-2B-Base at the pinned revision, the strands-decider-2B-hobson-v21 repo at a pinned commit, and adapter/head sha256 against v21's provenance.json (adapter 59be987f..., head 0fc78684...)
  - tests/`test_distill_pins.py`: flipping one byte of a cached head.safetensors fixture makes verification fail closed with a hash-mismatch error naming the file

### t8 — Run dataset build: licence filter, folds, sealed held-out, length filter

- instruction: The real token count must come from the student tokenizer in the pinned student env (subprocess); keep the counter injectable so tests stay CPU-only. Never use typed-decisions test rows or anything derived from predictions on them for training (its card forbids it).
- depends on: t1
- covers: c49, h34
- acceptance:
  - `jev_factory`/distill/dataset.py builds the run dataset from typed-decisions and the Apache-2.0/MIT jev-bench configs, carves grouped fit/selection folds and a sealed held-out set from train splits (reusing core.split), and keeps each source's own test split as the test side
  - Records whose rendered state + questions exceed 4096 student tokens are dropped with a per-source count in the stage manifest; no record is truncated; tests/`test_distill_dataset.py` uses an injected token counter
  - Sealed held-out and test loaders expose only ids, counts and sha256, never text

### t9 — Contamination check against v21 training data and Clef eval suites

- instruction: Populate the manifest from the strands-decider-2B-hobson-v21 card's training data list and Clef's card/Decision Index suite list; cite each source URL in the JSON. Name-level matching plus an alias table is enough for v1; record that limitation in the module docstring.
- depends on: t8
- covers: c45
- acceptance:
  - `jev_factory`/distill/contamination.py marks every evaluation source clean or overlapping using a checked-in manifest `jev_factory`/distill/data/`upstream_sources.json` (v21's public training datasets, Clef's Decision Index suites) with the URL each entry came from
  - tests/`test_distill_contamination.py`: banking77 and clinc150 come out overlapping; an unlisted source comes out clean; the result is written to the run for the final report

### t10 — Teacher-label stage: soft labels on train only, resumable, with provenance

- instruction: Cache by sha256 of the request body like data/teachers.py `cache_key`, in its own cache dir. One teacher build id per labelling run; write it on every row. Default to sending a record's questions jointly (Clef's native mode) and record the mode.
- depends on: t1, t4, t6
- covers: c3, h1, h36
- acceptance:
  - `jev_factory`/distill/teacher.py sends every train-split record to the teacher over /v1/systemone and stores, keyed by (record id, question id), the probability vector, teacher repo id, revision, build/quant id, engine, request config and response sha256
  - A non-train record is refused before any request; an incomplete or malformed answer is stored as a teacher error, never as a label (tests/`test_distill_teacher.py` with a fake server)
  - Killed halfway and re-run, it re-sends only unanswered items and the manifest counts cache hits; it records whether questions were sent jointly or one at a time

### t11 — Correction records: schema, provenance and leakage-checked intake

- instruction: Corrections live in their own JSONL under the run; status changes come only from the review flow (next task). Keep the drafter injectable; the default drafter is an alternative teacher over /v1/systemone or the existing data/teachers TeacherClient.
- depends on: t1
- covers: c34, h20
- acceptance:
  - `jev_factory`/distill/corrections.py defines a correction {record id, question id, target distribution or label, author/source, date, reason weak|outdated, status proposed|approved|edited|rejected} and refuses one missing author/source, date or reason
  - A correction whose text exactly or near-duplicates a fit/selection, test or held-out item is refused using core.leakage; tests/`test_distill_corrections.py` covers both refusals
  - A drafting entry point turns teacher-weak rows into proposed corrections from an agent/model drafter, and never marks them approved

### t12 — Pointer backbone adapter for Strands Decider (render, predict, bundle layout)

- instruction: Do not import `strands_decider` in-process from `jev_factory`; drive it as a server/subprocess in the student env. Register as 'pointer' in backbones/registry.py (only that one line touches the registry).
- depends on: t5, t4, t6
- covers: c5, c44, h29, c38, h22
- acceptance:
  - `jev_factory`/backbones/pointer/ implements the backbone protocol: predictions come from a pinned 'strands-decider serve' over /v1/systemone (via distill.systemone), never from letter-token logprobs
  - It reads and writes the hobson-hf-export/1 layout (adapter, head.safetensors, `strands_decider_config.json`, provenance.json) and refuses a layout missing any file
  - An integration test (skipped unless `JEV_STRANDS_PY` is set) loads a produced bundle in the pinned strands-decider env and answers a choice, a noul and a score question on 127.0.0.1

### t13 — Distill trainer: KL-to-teacher + CE-to-gold, continuing from v21

- instruction: Read strands-labs/strands-decider training/train.py and training/data/collate.py first and cite what you reuse (Apache-2.0 attribution). Whether to continue v21's own LoRA or merge then add a fresh LoRA is a pre-registered knob (frame park v6); implement both behind it. Ordinal score questions: keep `ordinal_smoothing` as a knob (park v2).
- depends on: t5, t3, t12
- covers: c6, h4
- acceptance:
  - `jev_factory`/backbones/pointer/`train_distill.py` (run in the trainer env) trains from v21 with loss = alpha \* KL(teacher || student at per-kind temperature) + (1 - alpha) \* CE(gold), LoRA, with every hyperparameter read from the locked distill prereg
  - A pure loss function has tests (skipped without torch): alpha = 0 equals gold CE within 1e-6; a row with no teacher target has zero KL; KL is 0 when student equals teacher
  - `plan_train`() returns the subprocess command without running it, and execution goes through factory.gpu.`run_gpu_stage`

### t14 — Operator review of corrections through jev review

- instruction: Extend `jev_factory`/review/ (core.py, server.py, page.html) without changing the seed-review behaviour; existing review tests stay green.
- depends on: t11
- covers: c35, h21, c31
- acceptance:
  - jev review gains a corrections mode: dry-run plan, --serve on 127.0.0.1, --apply writes the operator's approve/edit/reject decisions to the corrections review file, which is the record
  - tests/`test_review_corrections.py`: a proposed correction is never exported as trainable; approved and edited ones are, with the reviewer decision cited

### t15 — Teacher audit: per-source trust table against human gold

- depends on: t10, t2, t3
- covers: c32, h18
- acceptance:
  - `jev_factory`/distill/audit.py scores the teacher's train-side soft labels against human gold per source (n, accuracy, log-loss, ECE) and flags a source teacher-weak below the pre-registered floor
  - Sources whose gold is model-generated (typed-decisions) are reported but never used to flag the teacher; tests/`test_distill_audit.py` proves both
  - The audit table is written before assemble can run (assemble refuses without it)

### t16 — Zero-shot benchmark of Clef and stock v21 on the fit/selection folds

- instruction: Start and stop each server in sequence (one GPU job at a time, plan risk on GPU sequencing); record nvidia-smi/peak memory per model on the same box.
- depends on: t4, t8, t2, t12
- covers: c4, h2
- acceptance:
  - `jev_factory`/distill/benchmark.py scores Clef and stock v21 on fit/selection folds into predictions records with latency (median, p95) and peak memory per model
  - A test or held-out file passed to it is refused; tests/`test_distill_benchmark.py` uses two fake servers

### t17 — Assemble the distill training set: precedence, filters, one teacher build

- depends on: t8, t10, t15, t14, t3
- covers: c33, h19, c29, h16, h17, c50, h35, c11, h9
- acceptance:
  - `jev_factory`/distill/assemble.py picks exactly one target per (record, question) by precedence operator correction > human gold > alternative teacher > Clef, writing the source, rule, correction or teacher-build id and `teacher_filter` outcome on the row
  - Only approved/edited corrections are used; teacher-weak sources take corrected or gold-only targets; agree-with-gold filtering follows the prereg knob
  - It refuses mixed teacher build ids without a deviation id, refuses any non-train row, and runs the exact + near-duplicate leakage check against fit/selection, test and held-out before freezing with sha256 (tests/`test_distill_assemble.py`)

### t18 — Distill train, select and calibrate stage

- depends on: t3, t7, t17, t13, t12, t2
- covers: c13, h11
- acceptance:
  - `jev_factory`/distill/train.py refuses without a locked prereg or verified pins, runs the knob grid through the trainer under the GPU guard, and selects on the selection fold by the pre-registered rule
  - Calibration (per-kind temperature, optional per-label vector kept only if it lowers selection-fold ECE) is fitted on the fit fold with core.calibration; Strands' upstream `temperature_by_kind` is never used as the deployed calibration (tests/`test_distill_train.py`)

### t19 — Quantize, heal once and recalibrate the student

- depends on: t18
- covers: c9, h7, c43, h27
- acceptance:
  - `jev_factory`/distill/quant.py spikes each deployed-format candidate (W4A16 torso + fp32 pointer head, ONNX int8/int4, GGUF torso + side-loaded head) and records the chosen format as a decision citing the measurements
  - Heal runs only when the quant loses > 3.0 points against its own bf16, at most once; a second trigger exits with stop/escalate; recalibration fits on the quant's own fit-fold predictions; heal count and delta are in the stage manifest (tests/`test_distill_quant.py`)

### t20 — Final measurement once, paired gate and comparison report

- depends on: t19, t9, t16, t2, t3
- covers: c12, h10, c8, h6, c42, h26, h30, c40, h24
- acceptance:
  - `jev_factory`/distill/final.py scores Clef, stock v21 and the deployed student on test and held-out in one ledgered invocation (measure/once.OnceLedger); a second measurement for any of them exits non-zero without a deviation id
  - `jev_factory`/distill/report.py resamples the three models jointly per bootstrap draw, computes every c42/c43 bar from the prereg with n and CI on human-gold clean sources, reports typed-decisions agreement and overlapping sources separately, and records a miss as a miss
  - Higher accuracy with higher ECE than v21 is reported as a regression; latency and peak memory are reported for all three on the same box (tests/`test_distill_final.py`, tests/`test_distill_report.py`)

### t21 — Distill bundle and private upload

- instruction: The sealed held-out set is never bundled. The dataset bundle carries only Apache-2.0/MIT sources with each row's licence.
- depends on: t20, t12
- covers: c14
- acceptance:
  - `jev_factory`/distill/bundle.py writes the hobson-hf-export/1 layout with jev-factory's fitted `temperature_by_kind` in `strands_decider_config.json`, plus calibration.json, gate.json, the training set with per-row target sources, the comparison report and provenance; symlinks are refused
  - Upload reuses release/hub: nothing is sent without --apply; with --apply repos are private, fetched back and hash-verified, and no public-visibility call exists (tests/`test_distill_bundle.py` with a fake hub)

### t22 — jev distill verb and distill stage registry

- instruction: Stage functions already live in the per-stage modules; this task only wires them. If factory/pipeline.py's global REGISTRY blocks a second registry, parameterise it minimally and keep the jev run surface identical.
- depends on: t8, t9, t10, t15, t14, t17, t16, t18, t19, t20, t21
- covers: h12, c51
- acceptance:
  - `jev_factory`/distill/pipeline.py registers every distill stage in its own factory.stages.Registry with per-stage manifests under the work dir; `jev_factory`/cli/`_commands`/distill.py exposes `jev distill <stage>` with --apply, --detach, --force and --json, dry-run by default
  - `jev status <work>` shows distill stages and detached jobs; every side-effecting stage writes nothing without --apply (tests/`test_cli_distill.py`, one case per stage)
  - explain/catalog.py has entries for distill and each stage; overview/learn list the verb; 'uv run teken cli doctor . --strict' passes

### t23 — jev-CLI domain: annotate the distill verb and stages, seed entries for review

- instruction: This changes the jev-tool model's candidate set and surface sha (plan risk on ordering with t34 of the extraction plan); land it before the jev-tool seed is frozen or record a deviation there.
- depends on: t22
- acceptance:
  - domains/`jev_cli` annotations cover 'distill' and its stages (`read_only`=False where --apply exists), with >= 2 paraphrases each and >= 4 seed entries per operation in >= 3 phrasing classes marked pending operator review
  - The domain generator still matches argparse exactly and stays within 52 candidates; tests/`test_jev_cli_domain.py` and `test_jev_cli_content.py` pass

### t24 — Docs and harness prompt files for the distillery

- instruction: Keep the 'Strands Decider is not the Layer-3 decider' note prominent. No root AGENTS.md.
- depends on: t22
- covers: c39, h23
- acceptance:
  - docs/distill.md documents the stages, the record contract, the lobes split (serving in lobes-cli orin-decision-models, training here) and both consumers; it records the before-state evidence (at base commit 8aec9c4 no soft-label/KL/systemone code and only backbones/`causal_lm`) with the command used
  - CLAUDE.md, QWEN.md, AGENTS.override.md and AGENTS.colleague.md gain the distillery facts in their own framing; harness-smoke and markdownlint pass

### t25 — End-to-end toy run of every distill stage

- depends on: t22
- covers: c41, h25
- acceptance:
  - tests/`test_distill_e2e.py` dry-runs every 'jev distill' stage on a tiny typed-question fixture, then runs them with --apply against fake teacher/student servers and a fake trainer, producing a bundle with every listed file
  - The e2e test runs CPU-only in CI without torch, transformers or `strands_decider` installed

### t26 — Operator run: distill Clef 27B into Strands Decider 2B on the Spark

- instruction: One GPU job at a time; stop the teacher server before training. Coordinate with lobes (orin-decision-models) on the bundle hand-off via the communicate skill. Any rerun of a final measurement is a /deviate record.
- depends on: t23, t24, t25
- covers: c1, h28
- acceptance:
  - With the operator: the distill prereg is locked before the first training command; Clef (4-bit) and v21 are served pinned on 127.0.0.1; every stage runs with --apply; corrections are drafted and operator-reviewed; progress is reported every 30 minutes from 'jev status --watch'
  - The final report states pass or miss for every c42/c43 bar from the single final measurement; on a pass the private bundle is uploaded and verified, on a miss a failed-evaluation record is written and the announcement is not made

## Risks

- [unknown_nonblocking] GPU sequencing: a running teacher or student server is a foreign GPU process to `assert_gpu_free`, so benchmark, labelling and training must start and stop servers in sequence (frame park v8). (task t16)
- [follow_up] Ordering with the jev-tool build: the distill verb changes the jev-CLI domain's candidate set and surface sha; if it lands after the extraction plan's t34 freezes the seed, the jev-tool model trains on a stale candidate set (frame park v9). (task t23)
- [unknown_nonblocking] Statistical power: about 400 typed-decisions test cases plus the human-gold sources may not resolve the >= 50% gap-closure bar with a tight paired CI (frame park v10). (task t20)
- [unknown_nonblocking] Strands' training code (training/train.py, `kl_frozen_weight` semantics) was not read during the challenge pass; the distill trainer must read and cite it before implementing. (task t13)
- [unknown_nonblocking] Per-option KL may be insufficient for ordinal score questions; ordinal smoothing is kept as a knob (frame park v2). (task t13)
- [unknown_nonblocking] GPU memory on the GB10 for a 4-bit Clef 27B next to the ~34 GB already resident is unmeasured; labelling may need the resident model stopped (frame park v3). (task t10)
- [unknown_nonblocking] Clef's joint scoring of multi-question records may produce different soft labels than asking each question alone; the labelling mode is recorded (frame park v5). (task t10)
- [unknown_nonblocking] Whether to continue v21's own LoRA + head or merge v21 and train a fresh LoRA is decided by a pre-registered knob (frame park v6). (task t13)
- [unknown_nonblocking] Detecting outdated teacher knowledge is not solved by the audit when gold is itself old; time-sensitive sources may need dated provenance (frame park v7). (task t11)
- [unknown_nonblocking] The deployed format for the pointer student may not exist on every target runtime (Orin/edge); the quantize spike may end with bf16 as the only verified format (frame v1 decision). (task t19)
- [unknown_nonblocking] Clef serving on GB10 is an assumption (c20): the transformers sidecar or SGLang dev-clef image must be proven on the Spark before the operator run. (task t26)
- [follow_up] Reusing jev decide's verdict set for distill runs is not planned; a distill-specific decide is a follow-up (frame park v4).
