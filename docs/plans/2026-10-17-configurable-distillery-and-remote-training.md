# Build Plan — configurable distillery and remote training

slug: `configurable-distillery-and-remote-training` · status: `exported` · from frame: `configurable-distillery-and-remote-training`

> jev distill runs with the models an operator actually has: a configurable student profile, Clef as the teacher in whichever build variant the operator serves (BF16, NVFP4, FP4, W4A16 and so on), optional secondary roles (alt teacher, the deployment board, edge) and a deployed format declared in the pre-registration (a quant or bf16). The training step runs through an executor seam that is agnostic to where the job runs, so a later session can add SageMaker (us-east-1) without touching the stages.

## Tasks

### t1 — Golden baseline and read-first pass: record main's e2e distill manifests and pre-registration draft as fixtures, plus a byte-identity test

- instruction: Read bundle.py, edge.py, quant.py's `_finish` and report.py's `c42_bars`, c43 and `_c17_stats` in full first (scope entry s29 says they were not read in full). Generate the golden files by running the e2e fakes on a clean checkout of e3345d1 into a tmp dir; normalise only fields that vary per run (created timestamps, version strings, absolute tmp paths). Touch only the tests/ tree.
- covers: c30, h21, c33, h24
- acceptance:
  - tests/fixtures/`distill_golden`/ holds the pre-registration draft and every stage manifest that the `test_distill_e2e` fakes write at e3345d1, with version and timestamp fields normalised
  - tests/`test_distill_golden.py` regenerates them on the working tree and asserts 0 differing bytes; it passes on main unchanged
  - A short notes file (tests/fixtures/`distill_golden`/`READ_FIRST.md`) records what bundle.py, edge.py, quant.py `_finish` and report.py's c42/c43 helpers assume about the model set and quant manifest, with file:line

### t2 — Student profile: new distill/profile.py with a validated StudentProfile and a `DEFAULT_2610` profile; pins.py derives its defaults from it

- instruction: Stdlib only (tomllib, dataclasses); no heavy imports. Keep the module-level constants in pins.py as aliases of `DEFAULT_2610` fields so no other module changes in this task. Files: `jev_factory`/distill/profile.py (new), `jev_factory`/distill/pins.py, tests/`test_distill_profile.py` (new).
- covers: c2, h2, c3, h3
- acceptance:
  - StudentProfile holds the base repo and default revision, decider repo, revision and adapter/head sha256, layout filenames, `num_slots`, `max_tokens`, prereg seeds (LoRA r/alpha/dropout/targets, `temperature_by_kind`, upstream strings), card baseline (n, correct, tasks sha256, or None), contamination manifest path, weight-byte budget and backbone name
  - Loading a profile from a run.toml table or a TOML file refuses, with exit 1 and the field named, an unknown key, a missing required field or a malformed sha256
  - A test compares `DEFAULT_2610` field by field with today's constants (pins.py, prereg.py `STOCK_`\*/`UPSTREAM_`\*, benchmark.py `CARD_`\*, dataset.py limits, quantized.`SIZE_BUDGET_BYTES`, the `upstream_sources.json` path)
  - pins.`default_pins` takes a profile, defaulting to `DEFAULT_2610`; the existing constants stay importable; tests/`test_distill_pins.py` passes unchanged

### t3 — Declared skips: factory/stages.py gains a skipped status that satisfies dependents and shows in jev status with its reason

- instruction: Change factory/stages.py (`_own_staleness` and the run path) and the status renderer only. Add a `write_skipped`(workdir, name, reason, knobs) helper the distill pipeline will call later. Tests in tests/`test_factory_stages_skipped.py` (new).
- covers: c39, h30
- acceptance:
  - A manifest with status skipped and a reason counts as satisfied for dependents; one whose recorded declaration (knobs) changed is stale
  - jev status prints skipped with the reason for such a stage
  - All jev run (factory) and distill tests pass unchanged; no jev run stage ever writes skipped

### t4 — Clef variant: TeacherConfig carries the variant and engine; envs gains per-variant serving recipes; teacher-label and assemble refuse a mismatched variant

- instruction: Files: `jev_factory`/distill/teacher.py, envs.py, assemble.py, audit.py and their tests. Do not rename files or role keys on disk (migration claims c37/c38); the display name and variant come from config. A row with no recorded variant (an existing run) is read as the run's configured variant only when the build id matches.
- covers: c5, h5, c27, h19
- acceptance:
  - TeacherConfig has a variant field (free text such as BF16, NVFP4, FP4, W4A16), stamped on every label row and written to the teacher-label and audit manifests
  - envs.py can name the engine and quantization a variant needs; unknown engine kinds are refused
  - assemble refuses labels whose recorded variant or build id differs from the run's, unless a deviation id is given
  - On-disk names and the 'clef' audit role key are unchanged, so existing run folders still read

### t5 — Executor seam: new distill/executor.py (JobSpec, Executor protocol, LocalExecutor); `train_distill`.`run_train` submits through it

- instruction: Files: `jev_factory`/distill/executor.py (new), the `run_train`/plan section of `jev_factory`/backbones/pointer/`train_distill.py`, tests/`test_distill_executor.py` (new). Reuse release/hub.py digests/compare and the existing scan helpers rather than re-implementing them. The seam owns the protected-id and scan checks so future executors inherit them. Keep dry run printing the job spec and touching nothing.
- covers: c10, h9, c12, h11, c13, h12, c14, h13, c40, h31, c36, h27
- acceptance:
  - JobSpec lists staged inputs and expected outputs with sha256, the environment (python path or image digest plus lockfile hash, non-secret variables only, paths relative to the staged folder) and the limits
  - Before calling any executor, the seam refuses with exit 1 when a staged record id is in the held-out, test, fit or selection sides, or when the staged folder fails scan-secrets or the hostname/private-IP scan
  - After the job, every expected output is verified by sha256; a missing or changed output fails non-zero, and the job manifest is copied into the stage manifest
  - LocalExecutor reproduces main's argv, env, GPU guard and capped.sh call exactly (asserted against a recorded plan); a fake executor in tests shows job state through the pid-or-job-id, done and log files
  - No AWS import anywhere: grep -rniE 'boto3|sagemaker' `jev_factory` returns nothing

### t6 — Pointer backbone stops importing distill: backbones/pointer/`__init__.py` takes its default `init_from` from the profile argument

- instruction: Files: `jev_factory`/backbones/pointer/`__init__.py` and tests. Pass the decider repo/`init_from` in rather than importing `DECIDER_REPO`; profile.py may import from backbones, never the reverse.
- depends on: t2
- covers: c4, h4
- acceptance:
  - `jev_factory`/backbones/pointer imports nothing from `jev_factory`.distill (a test walks the AST of every module in the package)
  - PointerBackbone behaviour with the default profile is unchanged; tests/`test_backbone_pointer.py` passes

### t7 — Pre-registration schema v3: role declarations, deployed format, a variant-neutral teacher bar, seeds from the profile, and a read path for locked v2 files

- instruction: File: `jev_factory`/distill/prereg.py and tests/`test_distill_prereg.py`. Keep v1 refused by name. The quant-only bars are required only when `deployed_format` is quant.
- depends on: t2
- covers: c37, h28, c8, h7
- acceptance:
  - v3 declares roles (teacher, stock and student mandatory; `alt_teacher`, `deployed_server` and edge each present or absent) and `deployed_format` quant or bf16; declaring teacher, stock or student absent is refused
  - The teacher bar is variant-neutral (for example `top1_under_teacher_max_points`), with the pre-registered Clef variant recorded; the floors and tighten-never-loosen rules hold, and relative c17 bars cannot be marked not applicable while Clef and stock are present
  - A locked v2 file (a copy of the jetson-ops-20 pre-registration shape) loads through an explicit mapping, w4a16 bar to teacher bar with variant W4A16; its sha256 is unchanged and it is never rewritten
  - `example_doc` seeds from the active profile; with `DEFAULT_2610` the v2-shaped fields match today's

### t8 — Knob-recording stability: `effective_knobs` leaves default-valued new keys out of recorded knobs, so upgrading leaves completed stages fresh

- instruction: File: `jev_factory`/distill/pipeline.py (`effective_knobs` and `STAGE_CONFIG_KEYS` only) and tests/`test_distill_knobs_compat.py` (new). Register the new keys (profile, teacher variant, role presence, deployed format, executor, deployed-server URL) with a 'record only when set' rule.
- depends on: t1
- covers: c38, h29
- acceptance:
  - On a copy of a completed run's manifests (fixture built from the golden set), `stale_stages` returns the same list before and after the change, with default config
  - Setting a new key to a non-default value does make the stages that read it stale

### t9 — Report and final follow the declared roles: a `not_applicable` result, every bar enumerated, the variant beside each number, and final accepts a bf16 deployed student and any attested loopback server

- instruction: Files: `jev_factory`/distill/report.py, final.py and their tests. Reuse the existing REPORTED path (report.py:86-87). Read the `READ_FIRST` notes from t1 before editing.
- depends on: t7
- covers: c7, h6, c9, h8
- acceptance:
  - Every bar in report.json has exactly one result from pass, fail, `not_applicable` or reported; `not_applicable` names the absent role or format; an uncomputable bar for a present role is fail
  - final's model set comes from the declared roles; with a bf16 format, the deployed student is the attested bf16 build
  - The deployed-student URL key is role-neutral, and `distill_agx_url` still works as an alias
  - Report and final manifests show the Clef variant beside the teacher numbers

### t10 — bf16 ship: quantize and heal record skipped under a bf16 format; gate, bundle and edge accept the attested bf16 artifact; the bundle contract documents a bf16 bundle kind

- instruction: Files: `jev_factory`/distill/quant.py (format decision and `_finish` only, not the heal trainer hook), gate.py, bundle.py, edge.py, docs/distill-bundle-format.md and tests. final.py belongs to t9.
- depends on: t7, t3
- covers: c26, h18, c41, h32
- acceptance:
  - With `deployed_format` bf16, quantize writes skipped (via t3's `write_skipped` helper) with the reason, and gate, bundle and edge run without refusing, using the bf16 build and the train-stage calibration and predictions
  - With `deployed_format` quant, a quantize stage with no qualifying build still fails and final and bundle refuse: no bf16 fallback (an existing test proves it unchanged)
  - The bf16 bundle loads with strands-decider serve's hobson-hf-export/1 layout and is identifiable as a bf16 kind in `distill_provenance.json`; docs/distill-bundle-format.md gains only that section and additive fields

### t11 — Profile-driven stages: benchmark's card baseline, the contamination manifest, dataset limits and train's `init_from` come from the profile, and the stages reach the student via registry.get

- instruction: Files: `jev_factory`/distill/benchmark.py, contamination.py, dataset.py, train.py and their tests. The pipeline passes the profile in through t12; here take it as a parameter defaulting to `DEFAULT_2610`.
- depends on: t2, t6
- covers: c2, h2, c4, h4
- acceptance:
  - benchmark reads the card baseline from the profile; a profile with no card baseline records the reproduction check as not available, and train then needs a deviation id, as with a base mismatch today
  - contamination uses the profile's manifest path, dataset gets `max_slots` and `max_tokens` from the profile, and train.py:525 uses the profile's decider repo
  - No distill module reads `BASE_REPO`, `DECIDER_REPO`, `CARD_`\* or `STOCK_LORA_TARGETS` directly (an AST test)

### t12 — Pipeline wiring: the run config selects the profile, Clef variant, role presence, deployed format and executor; absent-role stages are skipped; heal goes through the executor; local-only stages ignore the executor

- instruction: Files: `jev_factory`/distill/pipeline.py, the heal-trainer hook in quant.py (QuantHooks.trainer wiring only), tests/`test_cli_distill.py` and the fakes in tests/`test_distill_e2e.py`. Keep the jev distill verb and knob surface; add run.toml keys only.
- depends on: t3, t4, t5, t6, t7, t8, t9, t10, t11
- covers: c11, h10, c17, h14, c7, h6
- acceptance:
  - Stages that need an absent secondary role (edge; corrections-draft's alt drafter) write skipped with the role named instead of require()-refusing
  - quantize's heal round goes through the configured executor: a fake injected for train also receives the heal job
  - benchmark, gate, final, edge and the quantize builds never read the executor setting (a test asserts it)
  - Every existing distill test passes, and the golden byte-identity test from t1 passes with default config

### t13 — End-to-end scenarios: a second profile with an NVFP4 Clef, absent roles with a bf16 ship, a fake non-local executor, and the boundary checks

- instruction: File: tests/`test_distill_e2e_configurable.py` (new) plus fixtures under tests/fixtures/. Use one run.toml per scenario.
- depends on: t12
- covers: c1, h1, c22, h17, c32, h23, c34, h25, c35, h26, c18, h15
- acceptance:
  - A second-profile fixture (different repo names and digests) with Clef variant NVFP4 runs every stage from preregister to bundle in the fakes; no bar key or report field contains w4a16 or a 2610 constant
  - A run with alt teacher, deployed server and edge absent and a bf16 format completes preregister through bundle, with every bar pass, fail, `not_applicable` (with reason) or reported
  - A fake non-local executor receives a staged bundle with 0 protected ids and fails on a one-byte output change
  - git diff main -- tests/`test_distill_boundaries.py` is empty, the jev-CLI domain export shows the same verb set as main, and dependencies = \[\] holds

### t14 — Docs and prompt files: distill.md, run-config.example.toml, training-env.md, the jev explain distill entry and the four prompt files; draft the lobes-cli issue

- instruction: Files: docs/distill.md, docs/run-config.example.toml, docs/training-env.md, `jev_factory`/explain/catalog.py, CLAUDE.md, QWEN.md, AGENTS.override.md, AGENTS.colleague.md. Keep the harness-probe wording (jev-factory in CLAUDE.md and QWEN.md; no lobes-cli attribution in AGENTS.override.md; no root AGENTS.md).
- depends on: t12
- covers: c21, h16, c29, h20, c31, h22, c42, h33
- acceptance:
  - docs/distill.md states the minimum roles and the optional ones, the profile, the Clef variant, the deployed format and the executor seam; its State of the work matches the proving-run evidence
  - No W4A16-only, AGX-only or hobson-v21 wording remains in docs/distill.md, catalog.py or the four prompt files; .pi/SYSTEM.md is untouched
  - teken cli doctor --strict, markdownlint and scripts/harness-smoke.py --stage config pass
  - The text of a lobes-cli issue about the bf16 bundle kind is drafted in the PR description, to be filed only after operator approval

### t15 — Release: minor version bump and changelog, validate-delivery and summarize-delivery, then the PR via cicd

- instruction: Follow the memory rule: delivery records before the PR. No AWS calls, no uploads, no public visibility.
- depends on: t13, t14
- acceptance:
  - pyproject.toml and CHANGELOG.md bumped by /version-bump minor; the full suite, the lint set from CLAUDE.md and scan-secrets pass
  - /validate-delivery has filed evidence for every honesty condition, /summarize-delivery has run, and the PR is opened through the cicd skill

## Risks

- [unknown_nonblocking] If the golden byte-identity fixtures (t1) normalise too much, they can hide a real drift; if they normalise too little, they are flaky. Review the normalised field list before wave 1 merges. (task t1)
- [unknown_nonblocking] pipeline.py (1366 lines) is touched by t8 and t12, and quant.py by t10 and t12; keep those edits to the named sections, and expect SonarCloud complexity findings (S3776) on the wiring. (task t12)
