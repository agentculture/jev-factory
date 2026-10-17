# configurable distillery and remote training

> jev distill runs with the models an operator actually has: a configurable student profile, Clef as the teacher in whichever build variant the operator serves (BF16, NVFP4, FP4, W4A16 and so on), optional secondary roles (alt teacher, the deployment board, edge) and a deployed format declared in the pre-registration (a quant or bf16). The training step runs through an executor seam that is agnostic to where the job runs, so a later session can add SageMaker (us-east-1) without touching the stages.

## Audience

- Operators running jev distill on their own hardware with a subset of the models: Clef served in whatever variant they have (BF16, NVFP4, FP4, W4A16), possibly no AGX Orin, edge board or alt teacher, and their own strands-decider student. Also the later session that adds a SageMaker executor, and lobes' decider role, which consumes the bundle (format unchanged for quant ships).
  - instruction: Check that docs/distill.md opens with who can run the distillery and with which minimum roles: Clef (any variant), the stock target and the student.

## Before → After

- Before: The distillery assumes one fixed stack: 2610 pins and digests as module constants (pins.py:19-33), Clef W4A16 named in bar keys (prereg.py:63-89), exactly four final models including an AGX-served quant (final.py:381-389), every role's config key require()d (pipeline.py:163-226), quant-only shipping (quant.py:1939-1986), and training launched only as a local GPU subprocess (`train_distill.py` `run_train` calls gpu.`run_gpu_stage`).
  - instruction: Verify against main at e3345d1 with: git grep -n '`DECIDER_REPO`\|`top1_under_clef_w4a16`\|`_check_models`\|`run_gpu_stage`' e3345d1 -- `jev_factory`
- After: An operator writes a run.toml that names a student profile, the Clef variant and its serving recipe, which secondary roles are present, the deployed format (quant or bf16) in the pre-registration, and the executor (local by default). Every stage then runs, and every bar is computed or reported not applicable with a reason. A later session can add a SageMaker executor by implementing the seam alone, with no stage edits.
  - instruction: Demonstrate with the e2e fakes: one run.toml per combination in the success signals, all stages from preregister to bundle applied.

## Why it matters

- The #20 proving run worked only because the operator had the whole stack (Clef W4A16 on an Orin, an AGX, the 2610 cache). Anyone with a different Clef build, no board, or a different strands-decider revision cannot run it without code edits, and remote training has no seam to attach to.

## Requirements

- A declarative student profile replaces the hard-wired 2610 constants: base repo and default revision, decider repo, revision and adapter/head sha256, layout filenames, `num_slots` and `max_tokens`, prereg seeds (LoRA r/alpha/dropout/targets, `temperature_by_kind`, upstream source strings), the published benchmark baseline or none, the contamination manifest path, the weight-byte budget and the backbone registry name. Today these live in distill/pins.py:19-33, prereg.py:182-231, benchmark.py:476-488, dataset.py:48-50, data/`upstream_sources.json`, quantized.py:130, train.py:525 and pipeline.py:369-371,427-431,706-711.
  - honesty: No distill module reads `BASE_REPO`, `DECIDER_REPO`, `CARD_`\*, `STOCK_LORA_TARGETS`, the adapter/head digests or the default contamination manifest except through the active student profile.
- The 2610 profile ships as the default and reproduces current behaviour byte-for-byte: the same pins, prereg draft, card gate and contamination manifest, so the existing distill tests (`test_distill_pins.py`:119-177, `test_distill_prereg.py`) stay green with the default.
  - honesty: The default profile's values equal today's constants exactly (repos, revisions, digests, LoRA seeds, card 180/231 and tasks sha256, the 2610 manifest path, the 1e9 budget), checked by a test that compares them field by field.
- The distillery reaches the student through the existing backbone seam (backbones/registry.py:66-77, base.py Backbone protocol) instead of importing backbones.pointer modules directly (train.py:53-55, bundle.py:48, final.py:63, prereg.py:19, envs.py:166); backbones/pointer/`__init__.py`:33,138 stops importing `DECIDER_REPO` from distill.pins.
  - honesty: `jev_factory`/backbones/pointer imports nothing from `jev_factory`.distill, and the distill stages obtain the pointer backbone via registry.get(profile.backbone).
- The teacher stays Clef, but its build variant is configuration, not code: the variant (BF16, NVFP4, FP4, W4A16 and so on), its serving engine and its build id come from the run config and are recorded in provenance. Bar and field names that hard-code W4A16 (prereg.py:63-89 `top1_under_clef_w4a16_max_points`; report.py:305,617; the 'Clef W4A16' wording in final, report, docs and prompt files) become variant-neutral, with the variant reported beside each number. The transport (systemone.py, teacher.TeacherConfig) is already generic and does not change.
  - honesty: Changing only the run config's Clef variant (for example W4A16 to NVFP4) changes no code path, and the variant string appears in the teacher-label, audit, final and report manifests.
- The teacher (Clef, any variant), the stock fine-tuning target and the student are mandatory. The secondary roles (alt teacher, the deployed-quant server and the edge board) can be declared absent in the pre-registration before training. Every bar relative to an absent role or format is reported as not applicable, with the reason, and never as a silent miss. Today report.`_bar` turns an uncomputable stat into passed=False (report.py:405-427), and final.`_check_models` demands exactly four models (final.py:381-389).
  - honesty: A pre-registration that declares the teacher or stock absent is refused; one that declares the alt teacher, deployed-quant server or edge board absent is accepted, and stages needing an absent role are skipped with a recorded reason instead of require()-refusing.
- The deployed-student server for final is any loopback-attested server, not specifically an AGX Orin: the code already only requires a localhost URL with matching /v1/bundle and a cgroup cap from /v1/stats (final.py:351-373, pipeline.py:905-933), so the AGX-specific naming (`distill_agx_url`) becomes role-neutral, with the old key kept as an alias.
  - honesty: final accepts any loopback URL whose /v1/bundle matches and whose /v1/stats reports a cap; a run config using the old `distill_agx_url` key still works.
- The training subprocess runs through an executor seam at the single place where `run_train` calls gpu.`run_gpu_stage` (`train_distill.py` ~370-405). The plan dict it already builds (~300-345) becomes an executor-neutral job spec: a staged-inputs folder plus an expected-outputs contract, with no local absolute paths leaking into the job. Local execution, today's behaviour including the GPU guard and capped.sh, is the only executor this work ships, and it is selected by default in the run config.
  - honesty: With the default local executor, train and heal launch exactly as on main (same argv, env, GPU guard and capped.sh), and the executor interface takes only the job spec, with no stage-specific arguments.
- Heal goes through the same executor seam as train, because heal is `train_distill`.`run_train` with --heal (quant.py:58,118,1332,2096-2103) and the QuantHooks.trainer seam (quant.py:288) already accepts an injected trainer.
  - honesty: quantize's heal round goes through the same executor as train; a fake executor injected for train also receives the heal job.
- Only train-side rows leave the box: the train file, or for heal the heal set plus protected-id sha256s, together with prereg.json and its lock, the init checkpoint and the pinned code. Held-out, test, fit and selection records are never uploaded. This is enforced in code by checking every shipped row id against the protected sides before any upload. The trainer subprocess reads only train rows: fit and selection predictions and calibration are scored locally afterwards (distill/train.py:479), and heal already refuses protected ids (`train_distill.py`:18-24,599-606).
  - honesty: Before any executor runs, every record id in the staged inputs is checked against the protected sides (held-out, test, fit, selection), and one hit refuses the job with exit 1 before anything is staged out.
- Executor-neutral round trip: the job spec lists every staged input and every expected output with its sha256; outputs are verified against the manifest when they come back, whatever executor ran the job; a dry run prints the job spec and touches nothing; and the trainer environment is named by a python path or an image digest plus a lockfile hash (envs.EnvSpec, envs.py:42-68,90), so a container executor needs no stage change.
  - honesty: Every staged input and returned output is listed with its sha256 in the job manifest; a mismatch on return fails the stage, and the job manifest is recorded in the stage manifest.
- A remote job is visible in jev status through the same job files the detach path writes (factory/detach.py:22-100: pid, done with the real return code, log). A job-id file stands in for the pid, and done is written from the job's terminal state.
  - honesty: A job run by a non-local executor shows in jev status as running, done or died through the same pid-or-job-id, done and log files as a detached local job.
- docs/distill.md (including the stale State of the work section at :27-32), docs/run-config.example.toml, docs/training-env.md (with a short section on the executor seam), the jev explain distill entry (catalog.py:443-470, which still says hobson-v21) and the four prompt files (CLAUDE.md, QWEN.md, AGENTS.override.md, AGENTS.colleague.md; .pi/SYSTEM.md has no distill text) describe the student profile, the Clef variant, the optional roles, the deployed format and the executor seam.
  - honesty: The four prompt files and docs/distill.md mention the student profile, the Clef variant, the optional roles, the deployed format and the executor seam, and none still claims the W4A16-only, AGX-only or hobson-v21 wording.
- Tests: a second student profile fixture and a non-W4A16 Clef variant go through the e2e fakes (`test_distill_e2e.py`); a fake non-local executor receives exactly the staged train-side bundle and returns outputs that pass hash verification; a bf16-ship pre-registration runs end to end; and `test_distill_boundaries.py` stays unchanged, since this work opens no new network connection.
  - honesty: The listed tests exist and pass in CI, and `test_distill_boundaries.py` is byte-identical to main.
- A deployed-format choice is pre-registered: quant or bf16. With quant, today's no-bf16-fallback rule holds unchanged (a failed quantize stops the run; quant.py:1939-1986). With bf16, quantize and heal are skipped and recorded as such, the bf16 student is gated, attested and bundled in the hobson-hf-export/1 layout that strands-decider serve loads, and the quant-only bars (0.97 agreement, weight bytes, bars.quant; prereg.py:52-73) are reported not applicable. final (:154-168), gate (:319), bundle (:208) and edge (:250) accept the bf16 artifact.
  - honesty: A bf16-format pre-registration skips quantize and heal with a recorded reason, ships a bundle that strands-decider serve loads, and reports the quant-only bars not applicable. A quant-format run whose quantize fails still stops with no bf16 fallback.
- Clef variants have serving recipes: envs.KINDS (envs.py:21-23,170-171; today clef-transformers and sglang-dev-clef) gains a way to name the engine and quantization a variant needs, and the teacher-label and audit stages record the variant. The existing invariant holds: one run uses one teacher build, and labels from another build mean a new labelling run or a deviation.
  - honesty: Each Clef variant names its engine and quantization in the run config; teacher-label refuses labels whose recorded variant differs from the run's without a deviation id.

## Honesty conditions

- Every stage from preregister to bundle runs, in the e2e fakes, for each run.toml combination in c33-c36, with no code edit between combinations.
- With the stock and Clef rows present, the c17 relative bars cannot be loosened or marked not applicable, whatever roles or format are declared; a test tries each and is refused.
- quantize builds, benchmark, gate, final and edge do not consult the executor setting.
- The jev-CLI domain export (`test_jev_cli_domain_export.py`) shows the same verb set as main.
- docs/distill.md states the minimum roles (Clef in any variant, the stock target, the student) and which roles are optional.
- Each before-state citation resolves at e3345d1.
- The proving-run evidence doc (docs/evidence/2026-10-16-jetson-ops-proving-run.md) shows the run used Clef W4A16 on the Orin, the AGX and the 2610 cache.
- A SageMaker-shaped fake executor added in a test file touches no file under `jev_factory`/distill.
- The byte-identity check is automated, not eyeballed.
- The second profile is a test fixture with different repo names and digests, so the run cannot pass by reading 2610 constants.
- The report test enumerates every bar, so a newly added bar cannot silently fall outside the check.
- The protected-id check runs in the seam itself, not in each executor, so a future SageMaker executor inherits it.

## Success signals

- Default profile unchanged: with the default 2610 profile, all 1086 currently passing distill, pointer and review tests still pass with no assertion changed, and the e2e fixture's pre-registration draft and stage manifests are byte-identical to those written on main e3345d1 (0 differing bytes, apart from version and timestamp fields).
  - instruction: Run uv run pytest -n auto tests/`test_distill_`\*.py tests/`test_cli_distill.py` tests/`test_pointer_`\*.py tests/`test_backbone_pointer`\*.py tests/`test_review`\*.py on the branch and on e3345d1; diff the e2e manifests written by both.
- Variant and profile coverage: an e2e fake run with a second student profile and a non-W4A16 Clef variant (for example NVFP4) completes all 15 stages, its manifests and report name that variant, and 0 bar keys or report fields contain a hard-coded 'w4a16' or a 2610 constant.
  - instruction: New e2e test plus: grep -rn 'w4a16' `jev_factory`/distill/prereg.py `jev_factory`/distill/report.py returns only variant-registry data, not bar keys.
- Optional roles and bf16 ship: a run with the AGX, edge and alt-teacher roles absent and a bf16 deployed format completes preregister through bundle, the quantize manifest records status skipped, and 0 bars are a silent fail: each is passed, failed or not applicable with a named reason (at least the 3 quant-only bars are not applicable).
  - instruction: New e2e test asserting every bar in report.json has a result in {pass, fail, `not_applicable`, reported}, and that `not_applicable` carries a reason naming the absent role or format.
- Executor seam: a fake non-local executor receives a staged bundle holding 0 held-out, test, fit or selection record ids; an output with one changed byte fails hash verification with a non-zero exit; and grep -rniE 'boto3|sagemaker' `jev_factory` returns 0 lines, with dependencies = \[\] and `test_distill_boundaries.py` unchanged.
  - instruction: New executor tests; git diff main -- tests/`test_distill_boundaries.py` is empty.

## Scope / boundaries

- Declaring a secondary role absent, or choosing a bf16 ship, never loosens a bar that applies. '+10 over stock' and 'within 5 of the teacher' always apply because stock and Clef are mandatory; the c17 'tighten, never loosen' rule (prereg.py:349-367) holds. Changing role presence or the deployed format after the lock is a recorded deviation, never an edit.
- quantize builds, benchmark, gate, final and edge stay local: they serve models on localhost and read the 7.6G quant tree. Only the training subprocess (train, and heal) can go remote.
- Configuration happens through run.toml keys (docs/run-config.example.toml:59-89 already holds the `distill_`\* keys) and flags or knobs on the existing jev distill verb. No new verb is added, so the jev-tool model's candidate set is unchanged (domains/`jev_cli` annotations.py:104-108; a new verb would need an annotation, content and an export-test change).

## Non-goals

- No new student model family: GLiNER2.5-Decide, T5Gemma, LFM d1-omni and CLEF-style backends (issues #6, #13, #14, #16, #17) are out of scope. The student profile covers strands-decider pointer students (another 2610 revision or size, or a Qwen3.5-0.8B base) on the existing hobson-hf-export/1 layout and quantizer, and other families stay possible through the backbone seam later.
- The Layer-3 decider, the jev-tool acceptance path (jev run), lobes-cli and nvsh are not changed.
- This work adds no SageMaker executor, no AWS SDK dependency or extra, no ECR image, no non-loopback connection and no live cloud job. Those come in a later session that builds on the executor seam.

## Scope exploration

- `s1` — `jev_factory/distill/pins.py, prereg.py, dataset.py, train.py, pipeline.py (student pins)`: `BASE_REPO`, `DECIDER_REPO` and the adapter/head digests are module constants (pins.py:19-33), and only the revisions are overridable (pipeline.py:371,709). The prereg draft seeds 2610's LoRA, 12 Qwen3.5 targets and temperatures (prereg.py:182-231). `max_slots` never reaches `build_dataset` from the pipeline (dataset.py:50,324; pipeline.py:432-440). train.py:525 derives `init_from` from `DECIDER_REPO`.
  - seeds: `c2`, `c3`
- `s2` — `jev_factory/backbones/registry.py, base.py, pointer/__init__.py`: PointerBackbone already implements the Backbone protocol and is registered (registry.py:66-69), but no distill module calls registry.get; they import backbones.pointer.\* directly. pointer/`__init__.py`:33,138 imports `DECIDER_REPO` from distill.pins, so the backbone depends on distill.
  - seeds: `c4`, `c19`
- `s3` — `jev_factory/distill/systemone.py, teacher.py, audit.py, assemble.py, corrections.py (teacher)`: The transport is generic (POST /v1/systemone, key read from the env var named in config, optional X-Lobes-Model header; systemone.py:36,192-230). 'clef' is a functional role key in audit ROLES, the assemble precedence and `_require_audits`, the pipeline's audit default, `_weak_rows` and `_clef_client`, and the prereg `reference_rows`. An alt-teacher path exists only for assemble and corrections; alt-only fails because `stage_assemble` always passes `clef_labels` (pipeline.py:623).
  - seeds: `c5`, `c6` (rejected)
- `s4` — `jev_factory/backbones/pointer/distill_loss.py, train_distill.py (gold-only)`: alpha 0 is exactly gold CE and rows without a teacher get KL 0 (`distill_loss.py`:26,137-144); the trainer handles teacher=None rows (`train_distill.py`:566-573). Only the prereg alpha grid and the stage graph block a teacherless run.
  - seeds: `c6` (rejected)
- `s5` — `jev_factory/distill/report.py, final.py, prereg.py (reference roles and bars)`: Every role's config key is require()d, so an unset key refuses (pipeline.py:163-226). An uncomputable bar becomes passed=False (report.py:405-427). final.`_check_models` wants exactly clef, stock, `student_bf16` and student (final.py:381-389). prereg refuses a c17 bar that is missing or loosened (prereg.py:349-392) and requires `reference_rows` clef and stock (:506-508). A REPORTED result already exists for latency (report.py:86-87,437) and could carry 'not applicable'.
  - seeds: `c7`, `c8`, `q1` (question, resolved)
- `s6` — `jev_factory/distill/final.py, pipeline.py _agx_final_model, measure/preflight.py`: final needs only a pre-served loopback URL whose /v1/bundle matches the deployed artifact and whose /v1/stats reports a cgroup cap (final.py:351-373; pipeline.py:905-933; `require_localhost` in preflight.py:47-55). 'AGX' appears only in names, docstrings and the `distill_agx_url` key.
  - seeds: `c9`
- `s7` — `jev_factory/distill/quant.py, backbones/pointer/quantized.py (quantize and heal)`: The format registry (quant.py:156-186, injectable QuantHooks at :288) has one real builder (`torch_wo`, :488). The tensor policy is Qwen3.5/GatedDeltaNet-specific (quantized.py:148-167). When no quant qualifies, `_fail` writes `shipped_format` null (quant.py:1939-1986), and final, gate, bundle and edge refuse. Heal calls `train_distill`.`run_train` via `_default_trainer` (quant.py:2096-2103).
  - seeds: `c11`, `c19`, `q2` (question, resolved)
- `s8` — `jev_factory/backbones/pointer/train_distill.py run_train, factory/gpu.py, factory/detach.py (remote seam)`: `plan_train` builds the argv, env and paths without running anything; `run_train` with apply calls gpu.`run_gpu_stage`, which holds the nvidia-smi foreign-process guard and the capped.sh floor. Detached jobs are pid, done (real rc) and log files, which `job_status` reads as running, done, died or absent.
  - seeds: `c10`, `c14`
- `s9` — `jev_factory/distill/train.py, train_distill.py (data sides)`: predictions.json and calibration over the fit and selection folds are scored in distill/train.py (:71-72,:479) after training, not in the trainer subprocess. Heal refuses any protected (test, held-out, fit, selection) id before loading a model (`train_distill.py`:18-24,599-606). On the proving run, train/c000 outputs were checkpoint 224M, bundle 224M and predictions 452K, and dataset/train.jsonl was 1.3M.
  - seeds: `c12`
- `s10` — `jev_factory/distill/envs.py, release/hub.py (provenance and round-trip)`: EnvSpec already takes an `image_digest` (sha256:...) xor a python path, plus a lockfile hash, recorded in provenance (envs.py:42-68,90). hub.py's digests and compare functions (:87,:112) and its upload, fetch-back and verify flow (:149-226) are the pattern for an S3 round-trip.
  - seeds: `c13`
- `s11` — `pyproject.toml, tests/test_distill_envs.py, test_distill_boundaries.py`: dependencies = \[\] is asserted with heavy modules blocked (`test_distill_envs.py`:176-187). Every stage dry-runs and applies with socket.connect patched to fail on non-loopback (`test_distill_boundaries.py`:51-125). No boto3, sagemaker or s3:// code exists anywhere in the repo.
  - seeds: `c15` (rejected), `c16` (rejected), `c22`
- `s12` — `quant/benchmark/final/edge serving (localhost model servers)`: benchmark, final, edge and the quant measurer all serve or reach models through `require_localhost` (systemone.py:203, final.py:351, pipeline.py:924-933), and the quant tree is 7.6G on the proving run, so these stages stay local.
  - seeds: `c17`
- `s13` — `jev_factory/cli/_commands/distill.py, explain/catalog.py, domains/jev_cli`: The stage is a positional and knobs go through `parse_knobs`, which refuses unknown knobs (distill.py:90,133-148). --detach re-spawns locally (:100). jev.distill is one operation (annotations.py:104-108), and its stage list grows automatically from pipeline.`stage_names`() (generate.py:156-228). catalog.py:446 still says 'fine-tuned from hobson-v21'.
  - seeds: `c18`, `c21`
- `s14` — `docs/distill.md, run-config.example.toml, CLAUDE.md, QWEN.md, AGENTS.override.md, AGENTS.colleague.md, .pi/SYSTEM.md`: distill.md:27-32 says no run has happened, which is stale after the #20 proving run, and :395-407 states the localhost invariant. run-config.example.toml:59-89 already holds the teacher, stock, alt, AGX and edge keys, with no remote keys. CLAUDE.md:67-155, QWEN.md:74-114, AGENTS.colleague.md:66-81 and AGENTS.override.md:77-102 hard-code Clef and 2610; .pi/SYSTEM.md has no distill text.
  - seeds: `c21`, `c16` (rejected)
- `s15` — `GitHub issues and earlier devague frames (distillery #18, Orin Nano #20)`: Issue #6 (a pluggable decision-model contract) is the closest brief; #7, #13, #14, #16 and #17 are other model families. No issue mentions AWS or SageMaker. The earlier frames record no decision on configurability or cloud training; spec 2026-10-14 line 100 pins serving on 127.0.0.1.
  - seeds: `c19`, `c20`, `c16` (rejected)
- `s16` — `jev_factory/distill/benchmark.py, contamination.py, data/upstream_sources.json`: The card baseline (`CARD_N` 231, `CARD_CORRECT` 180, the tasks sha256) is module constants, and `reproduce_stock` refuses any count other than 231 (benchmark.py:476-488,585-607). The contamination manifest is 2610's 48 datasets at 216d1ccb, but `load_manifest` and `check_sources` already take a path (contamination.py:66,105); only the default and the pipeline call sites are hard-wired. Who drafts these for another student is parked as v3.
  - seeds: `c2`, `c7`
- `s17` — `Sibling repos and local AWS setup (open-bedrock-server, aws-agentcore-chat-bot, reachy_nova, unsloth-cli, ~/.aws)`: open-bedrock-server uses boto3 with the ambient chain (`AWS_PROFILE`, roles, assume-role; docs/guides/aws-authentication.md). No sibling uses SageMaker, and unsloth-cli has no remote training. ~/.aws/config names the profiles default and culture-nodes; aws CLI v2 is at /snap/bin/aws, and there is no SageMaker SDK. Rights were not tested (parked as v2).
  - seeds: `c15` (rejected), `q3` (question, resolved)
- `s18` — `docs/training-env.md, pyproject train extra (image architecture)`: The trainer env is pinned for GB10 aarch64 with cu130 torch, so SageMaker x86 GPU instances need their own image and lockfile hash. Spot resume is unverified (parked as v1).
  - seeds: `q4` (question, resolved)

## Decisions

- Clef (the teacher) and the stock fine-tuning target are mandatory roles in every run; Clef's build variant is configurable (BF16, NVFP4, FP4, W4A16 and so on).
- A bf16-only ship is allowed when it is decided.
- This work makes the training code agnostic to where the job runs; SageMaker support and live testing are a later session. Target region: us-east-1.

## Hard questions

- Can the stock role be absent? Stock is more than an evaluation row: train refuses without the stock JevBench reproduction (`_require_reproduction`, pipeline.py:744-748; benchmark.py:624-640), and it anchors '+10 over stock', gap closure and the ECE/confident-wrong regression rule. With no stock, what proves the student beat its starting point? (resolved: No. The teacher (Clef) and the fine-tuning target (the stock student) must always be present. Clef may be any build variant: BF16, NVFP4, FP4 (as NVIDIA did for GLM-5.3-FLASH), W4A16, and so on.)
- Is a bf16-only ship (quantize skipped or failed) in scope? It overrides risk r10's 'no bf16 fallback' (quant.py:1939-1986 `_fail`; final.py:154-168, gate.py:319, bundle.py:208, edge.py:250 all refuse) and the mandatory bars.quant / c17.quant bars (prereg.py:52-73). (resolved: Yes, a bf16-only ship can be done if it is decided.)
- AWS settings: which region, GPU instance type (and its training-job quota), spot or on-demand, S3 bucket and prefix, SageMaker execution role, which local profile (default or culture-nodes), and what MaxRuntime or cost cap per job? (resolved: Region us-east-1 ('us1'). Open to proposals for the other settings (instance, spot, bucket, role, profile, cost cap).)
- Image and how far 'preparation' goes: SageMaker GPU instances are x86, while the trainer env is pinned for GB10 aarch64 with cu130 (docs/training-env.md, the train extra), so a new x86 image and lockfile are needed. Does this work stop at a dry-run-verified executor with a Dockerfile, or does it include building and pushing to ECR and one live --apply job? (resolved: Preparation means the code is agnostic to where the job runs. Actual SageMaker support and live testing happen in another session.)

## Open parks

- [unknown_nonblocking] Whether `train_distill.py` can checkpoint to S3 and resume mid-run (needed for spot instances) is unverified.
- [unknown_nonblocking] Whether the default or culture-nodes AWS profile has SageMaker, ECR and S3 rights is unknown; it was not tested, to avoid making AWS calls during scope.
- [unknown_nonblocking] Who drafts the contamination manifest and the published-benchmark baseline for a non-2610 student profile, and whether a profile without a card baseline turns the base-mismatch gate (benchmark.py:578-607) off or makes it a recorded deviation.
- [unknown_nonblocking] Proposed defaults for the later SageMaker session, to discuss then: us-east-1; ml.g6e.xlarge (1x L40S 48 GB) or ml.g5.2xlarge (1x A10G 24 GB) for 2B LoRA training; on-demand first, spot once resume is proven (v1); a private, encrypted S3 bucket with public access blocked; the profile or role named in run config; MaxRuntimeInSeconds as the cost cap; an x86 image built from a new lockfile; and a loopback carve-out only for the AWS control plane and S3.
