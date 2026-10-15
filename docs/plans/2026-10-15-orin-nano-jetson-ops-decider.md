# Build Plan — Orin Nano Jetson-ops decider

slug: `orin-nano-jetson-ops-decider` · status: `exported` · from frame: `orin-nano-jetson-ops-decider`

> A Strands Decider 2B distilled from Clef 27B knows how to operate a Jetson Orin, and a quantized, healed build under 1 GB runs fast and accurately on a Jetson Orin Nano as a lobes decider domain

## Tasks

### t1 — Jetson runtime spike: prove the 2610 decider's torch path and a 4-bit kernel on JetPack 6.2 (`sm_87`)

- instruction: Operator-attended; outward actions on boards. Nano first (ssh nano; it was unreachable 2026-10-15, operator says it is coming online). On the AGX Orin run nothing without the operator's go, under a cgroup cap (systemd-run --user -p MemoryMax=4G), never while teacher labelling runs, and never touch Clef's lane. Work in a scratch venv on the board; no repo edits except the evidence doc. Never name side/experiment models in the doc. Resolves plan risk on the Jetson runtime (frame park v1).
- acceptance:
  - docs/evidence/2026-10-jetson-runtime-spike.md records, for the Nano (and the AGX Orin if run there): /etc/`nv_tegra_release`, CUDA, torch/transformers/torchao versions, MemAvailable, whether Qwen3.5-2B's Gated DeltaNet torch fallback runs one forward pass, and whether an int4 and an int8 weight-only linear runs on `sm_87`, each with the exact command and output
  - the doc ends with a recorded verdict per format (int4 torchao, int8 torchao, other) — works / fails with error text — and nothing in `jev_factory`/ changes

### t2 — Retarget student pins from hobson-v21 to strands-decider-2B-qwen3.5-v1-2610 @216d1ccb

- instruction: Own only `jev_factory`/distill/pins.py and tests/`test_distill_pins.py`. Base pin stays `DEFAULT_BASE_REVISION` b1485b2 (2610's provenance also says 'inferred'). Keep `ADAPTER_FILE` lora/`adapter_model`.safetensors and `HEAD_FILE` head.safetensors (2610 uses the same layout). Read <https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/raw/216d1ccb46a9e9b036eb9187b248db98480f77d4/provenance.json> to confirm the digests.
- covers: c6, h6
- acceptance:
  - `jev_factory`/distill/pins.py: `DECIDER_REPO` is StrandsAgents/strands-decider-2B-qwen3.5-v1-2610, the decider revision pin is 216d1ccb46a9e9b036eb9187b248db98480f77d4, adapter sha256 ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7 and head sha256 ace56742545b505e6fae34ada85c8edbeedb23ba8974d8680aebe0380d6083a1 (from its provenance.json); constants are named for 2610 or generically, no `V21_` names remain
  - tests/`test_distill_pins.py`: `verify_pins` passes on a fake cache with the 2610 digests and fails closed on a v21 digest; no 'v21' string remains in pins.py

### t3 — Rename the reference role `stock_v21` to a model-neutral stock role across the distillery

- instruction: Pure rename, no behaviour change. Do NOT touch pins.py, contamination.py or data/`upstream_sources.json` (other tasks own them). Keep 'v21-evaluated' semantics out of this task (contamination task renames that label). Update docs/distill.md mentions of config keys only. Run uv run pytest -n 4, black, isort, flake8, bandit.
- covers: c6, h6
- acceptance:
  - report.py, final.py, prereg.py, pipeline.py, benchmark.py, train.py, backbones/pointer/`train_distill.py` and layout.py use a 'stock' reference role (`REFERENCE_ROWS` ('clef','stock'), bars \*`_no_worse_than_stock`, config keys `distill_stock_url` / `distill_stock_model`); no identifier or config key contains v21 (grep -i v21 in those files finds only historical prose, if any)
  - the full suite passes with only renames in tests (no assertion weakened: the multiset of atomic assert conditions per test file is unchanged modulo the renamed identifiers)

### t4 — Retarget the pre-registration to 2610 and the c17 bars

- instruction: Own `jev_factory`/distill/prereg.py and tests/`test_distill_prereg.py`. Keep operator-placeholder semantics for knobs still undecided (frame park v2). The existing gap-closure / latency <= 1/5 Clef bars from #18 are replaced for this run by the c17 bars; keep the schema versioned so a #18-shape prereg is refused with a clear message rather than silently reinterpreted.
- depends on: t3
- covers: c6, h6, c17, h14
- acceptance:
  - prereg.`example_doc` seeds from 2610: upstream source URL at 216d1ccb, `lora_r` 48, `lora_alpha` 96, `lora_dropout` 0.05, `lora_targets` exactly the 12 modules in 2610's lora/`adapter_config.json`, `temperature_by_kind` from its `strands_decider_config.json`
  - the bars block pre-registers, for BOTH the bf16 student and the shipped quant: top-1 >= stock + 10 points and >= Clef W4A16 - 5 points on the human-gold Jetson-ops test side; quant-vs-bf16 agreement >= 0.97; weight bytes < 10^9; latency median and p95 reported with no bar; a prereg missing any of these is refused
  - tests/`test_distill_prereg.py` covers each new bar's validation (missing, wrong type, wrong value) and the 12-target list

### t5 — SystemOneClient: bearer key by env-var name and the X-Lobes-Model header, still localhost-only

- instruction: Own `jev_factory`/distill/systemone.py, `jev_factory`/distill/teacher.py and their tests. Do not edit pipeline.py (the wiring task adds config keys). The teacher is reached through an SSH tunnel (ssh -N -L 127.0.0.1:<port>:127.0.0.1:8000 <orin>), documented later by the docs task.
- covers: c12, h12
- acceptance:
  - systemone.py: the client takes `auth_env` (an env-var NAME) and `model_header`; requests carry Authorization: Bearer <value of that env var> and X-Lobes-Model: <`model_header`>; a missing env var fails with exit 2 naming the variable, never its value
  - a non-127.0.0.1/localhost URL is still refused; tests with a fake server assert both headers, and grep over a written teacher manifest, log and every raised CliError text finds no key value
  - teacher.py passes the two settings through and records the env-var NAME and header value (not the key) in the teacher-label manifest

### t6 — Contamination check against 2610's 48 card datasets

- instruction: Own `jev_factory`/distill/contamination.py, its data file and its test. Read the card YAML datasets list at <https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/raw/216d1ccb46a9e9b036eb9187b248db98480f77d4/README.md> and training/sources.md. Decide (and document in the module docstring) how LocalLLaMA/typed-decisions maps to the upstream tasksource typed-decisions set; when unsure, mark overlapping (fail closed).
- covers: c28, h16
- acceptance:
  - `jev_factory`/distill/data/`upstream_sources.json` lists exactly the datasets in the 2610 card at 216d1ccb (48 entries) plus Clef Decision Index suites; the label for eval-only upstream data is model-neutral (e.g. stock-evaluated)
  - tests/`test_distill_contamination.py`: a record whose `hf_id` is tasksource/tasksource-jev-typed-decisions, or a LocalLLaMA/typed-decisions record, is reported overlapping; a Jetson-ops source is clean

### t7 — Dataset stage: cc-by-4.0 with attribution, 24-option and 4096-token limits, typed-kind validation

- instruction: Own `jev_factory`/distill/dataset.py, `jev_factory`/distill/records.py and tests/`test_distill_dataset.py`, tests/`test_distill_records.py`. Add an optional attribution field to Record (required when licence is cc-by-4.0), serialised in `to_json`/`from_json`; pre-existing records without it still load. The 24 comes from 2610's `strands_decider_config.json` `num_slots`; take it as a knob with default 24, not a magic number.
- covers: c7, h7, c31, h19, c2, h2
- acceptance:
  - dataset.py admits apache-2.0, mit and cc-by-4.0; a cc-by-4.0 record without an attribution string, or any other licence, is refused; the manifest's licence table carries per-licence counts and the attribution strings
  - a question with more than 24 options or a record over `max_tokens` (4096) is refused and counted by reason (`over_slots`, `over_length`) in the manifest; no record is truncated
  - records.py: a record whose question kind is not choice/noul/score, or that is not a valid /v1/systemone body, is refused with its id; tests cover each refusal

### t8 — Human-gold approval ledger: jev review --gold FILE

- instruction: Own `jev_factory`/review/gold.py (new), `jev_factory`/cli/`_commands`/review.py, the review catalog entry and tests/`test_review_gold.py`. Mirror the corrections review (`jev_factory`/review/corrections.py) for UI and ledger style; reuse the local review site if it fits. Do not edit final.py (the final task consumes `approved_ids`).
- covers: c30, h18, c8, h8
- acceptance:
  - jev review --gold FILE lists drafted test records with proposed gold and records approve / edit / reject per record into an append-only ledger (record sha256, decision, ISO time); an edited record gets a new sha256 and its old approval no longer matches
  - a library function `approved_ids`(ledger, records) returns only ids whose current sha256 has an approve entry; tests cover approve, reject, edit-after-approve and a corrupted ledger (fails closed)
  - dry-run by default; --apply writes; jev explain review documents --gold; teken cli doctor --strict passes

### t9 — Jetson-ops source registry: pinned repos, licences and attribution

- instruction: New package `jev_factory`/distill/`jetson_ops`/ (stdlib only) plus tests/`test_jetson_ops_sources.py`. Read the licences from the repos themselves (~/git/jetson-containers-nv LICENSE is MIT, ~/git/jetson-ai-lab LICENSE is MIT, lobes-cli and nvsh are Apache-2.0; the skills repos: docs CC-BY-4.0, code Apache-2.0 per nvsh tests/fixtures/`jetson_skills`/README.md). Pin commits from the local clones (git rev-parse HEAD) and record them; clone the two NVIDIA-AI-IOT repos at nvsh's pinned commits.
- covers: c7, h7
- acceptance:
  - `jev_factory`/distill/`jetson_ops`/sources.json pins each source by repo URL and full commit (jetson-containers-nv, jetson-ai-lab, NVIDIA-AI-IOT/jetson-device-skills, NVIDIA-AI-IOT/jetson-bsp-skills, lobes-cli docs/evidence, nvsh device-CLI verbs) with its licence per path class (code vs docs) and the CC-BY-4.0 attribution text for the NVIDIA skills docs
  - `jev_factory`/distill/`jetson_ops`/sources.py loads and validates it (unknown licence, missing commit or missing attribution fail closed) and exposes, per source, the licence and attribution a drafted record must carry; tests cover each refusal

### t10 — Quantized student build: merge LoRA, drop vision/MTP, 4-bit blocks, quantized full-vocab embedding, size check, attestation endpoints

- instruction: Own `jev_factory`/backbones/pointer/quantized.py, the size check in `jev_factory`/distill/quant.py, and tests/`test_pointer_quantized.py`, tests/`test_distill_quant.py`. Use t1's spike verdict for the kernel choice; if int4 is unavailable on `sm_87`, build int8 blocks and record it. Keep build in the student env subprocess as today (dependencies = \[\] holds).
- covers: c9, h21, c1, h1
- acceptance:
  - backbones/pointer/quantized.py builds a full-vocabulary artifact: LoRA merged, vision tower and MTP head absent, text-block linears 4-bit, embedding INT4 or INT8, pointer head fp32, `in_proj_a`/b, conv1d, `A_log`, `dt_bias` and norms unquantized; the unmodified 2610 tokenizer ships with it
  - the artifact manifest records per-file sha256 and the byte sum of its weight files; quant.py records a build at or over 10^9 bytes as unavailable, never as a candidate
  - the serve wrapper answers GET /v1/bundle with the manifest's per-file sha256 and GET /v1/stats with process RSS, MemAvailable, device model, /etc/`nv_tegra_release` (when present) and the cgroup memory.max it runs under; tests use fakes (no torch needed)

### t11 — Pruned-vocabulary comparison build (reported, never shipped)

- instruction: Own `jev_factory`/backbones/pointer/quantized.py and `jev_factory`/distill/quant.py for this change and their tests; runs after t10. Implements decision c38 (comparison only) and h9.
- depends on: t10
- covers: h9
- acceptance:
  - quantized.py builds a second artifact whose embedding keeps only the ids the run's train/fit/selection prompts use (plus special tokens); its manifest records the kept id list sha256 and weight bytes
  - serving it, a request containing a token outside the kept ids is refused with an `out_of_vocabulary` error, never answered from a wrong row; quant.py reports it beside the full-vocabulary build (weight bytes, selection-fold accuracy, OOV refusal count) and never selects it as the shipped format

### t12 — Heal against the bf16 student's and Clef's soft labels, gated on >= 0.97 agreement

- instruction: Own the heal path in `jev_factory`/distill/quant.py and `jev_factory`/backbones/pointer/`train_distill.py` with their tests. Today's heal (nvsh d18) is a bf16 continuation then re-quantize; keep that as the fallback if quant-aware training is unavailable on the student env, and record which ran. Coordinate file ownership: t10/t11 also touch quant.py — this task runs in a later wave.
- depends on: t3, t11
- covers: c10, h10
- acceptance:
  - the heal round trains the quantized student (its LoRA/norms) on train records with targets mixing the bf16 distilled student's and Clef's per-option distributions; the pre-registered heal block names the mix
  - agreement with the bf16 student is computed on fit and selection only; a test or held-out id in the heal input exits non-zero; a second heal of the same format is refused without a deviation id; a healed build below 0.97 agreement is not shippable

### t13 — Benchmark: re-measure stock 2610 and reproduce its published JevBench w4096 result

- instruction: Own `jev_factory`/distill/benchmark.py and tests/`test_distill_benchmark.py` (the train-side refusal check lives in train.py: touch only the guard). Use the card's eval/jevbench-w4096 manifest and summary to define the comparison.
- depends on: t3
- covers: c29, h17, c3, h3
- acceptance:
  - benchmark.py has a stock-reproduction check: stock 2610 served on the pinned base revision scores JevBench w4096 and the result (n, accuracy, CI) is recorded; a result whose CI excludes the card's 0.7792 (180/231) marks the run base-mismatch, and train refuses without a deviation id
  - stock 2610 is also measured on the Clef probe set and on the Jetson-ops fit/selection folds, never copied from outside; tests use a fake server

### t14 — Final: measure the AGX-served quant via tunnel with attestation; human-gold only; bars for bf16 and quant

- instruction: Own `jev_factory`/distill/final.py, `jev_factory`/distill/report.py and their tests. The once ledger is unchanged: test and held-out are still measured once. Record the AGX memory cap reported by /v1/stats in the final manifest.
- depends on: t3, t4, t8, t10
- covers: c5, h5, c8, h8, c17, h14, c30, h18
- acceptance:
  - final.py accepts a pre-served student at a 127.0.0.1 URL (an SSH tunnel to the AGX Orin), checks GET /v1/bundle equals the bundle it is about to report, and refuses on mismatch before any test request; it refuses while the run's teacher-label lock exists
  - final refuses any test record without an approve entry in the gold ledger (`approved_ids`), printing ids only; the headline counts only approved human-gold test records with n
  - report.py computes each c17 bar for both the bf16 student and the shipped quant with paired bootstrap CIs and cites the hashed prereg; the Clef probe-set table and latency median/p95 carry no pass/fail

### t15 — Edge stage: non-test measurement on the AGX Orin and the Nano through a tunnel

- instruction: New module `jev_factory`/distill/edge.py and tests/`test_distill_edge.py` only; registering it as a stage is the wiring task's job. The board serves with the documented command under systemd-run --user -p MemoryMax=<cap>; jev never ssh-executes on the board.
- depends on: t10
- covers: c4, h4, c1, h1, c14, h22
- acceptance:
  - `jev_factory`/distill/edge.py measures a served bundle at a 127.0.0.1 URL on the selection fold and the probe set only: latency median and p95, resident memory and MemAvailable from /v1/stats, and prediction agreement with the final's predictions; a test or held-out id in its input exits non-zero
  - it refuses when /v1/bundle differs from the final's bundle sha256s, when /v1/stats reports no cgroup memory cap, or while the teacher-label lock exists; the manifest records board model, L4T release and the cap

### t16 — Bundle: CC-BY attribution NOTICE and pre-upload scans

- instruction: Own `jev_factory`/distill/bundle.py and tests/`test_distill_bundle.py`. The probe set and side-model list are private: read their paths from config, never copy their content into logs or the repo.
- covers: c32, h20, c5, h5
- acceptance:
  - bundle.py writes NOTICE with every attribution string from the dataset manifest into both the model and dataset bundles; a cc-by-4.0 record in the dataset bundle without NOTICE is refused
  - upload refuses when scripts/scan-secrets.py, a hostname/private-IP scan or a side-model-name scan (names from an operator-held list path, never committed) hits, or when any record of the Clef probe set is present; it prints file and line only

### t17 — Wire the run: config keys, edge stage registration, gold ledger, contamination and stock-reproduction gates, jev-CLI domain

- instruction: Own `jev_factory`/distill/pipeline.py, `jev_factory`/cli/`_commands`/distill.py, the explain catalog, `jev_factory`/domains/`jev_cli`/\* and their tests. This is the only task that edits pipeline.py.
- depends on: t5, t6, t7, t8, t11, t12, t13, t14, t15, t16
- covers: c14, h22
- acceptance:
  - pipeline.py adds `distill_teacher_key_env`, `distill_teacher_model_header`, `distill_gold_ledger`, `distill_agx_url`, `distill_edge_url` and `distill_edge_memory_cap` keys and wires them; 'edge' is registered after final; jev distill --help lists 15 stages, each dry-run by default
  - the jev-CLI domain's jev.distill stage grounding includes edge; `cli_surface_sha256` is updated and new seed entries are marked pending operator review; teken cli doctor --strict and harness-smoke pass

### t18 — Bundle-format doc for lobes' Nano backend

- instruction: Docs plus one test; no lobes-cli edits (c11 boundary).
- depends on: t16, t10
- covers: c11, h11
- acceptance:
  - docs/distill-bundle-format.md names every file the bundle stage writes, the quant scheme per tensor class, the tokenizer, and the loader/serve entry point with its /v1/systemone, /v1/bundle and /v1/stats contract
  - tests/`test_bundle_format_doc.py` fails when the bundle stage writes a file the doc does not list; markdownlint passes

### t19 — Boundary guards: localhost-only connections, no lobes/nvsh edits

- instruction: Own tests/`test_distill_boundaries.py` only.
- depends on: t17
- covers: c18, h15, c11, h11
- acceptance:
  - tests/`test_distill_boundaries.py` runs every distill stage's dry-run and apply path against fakes with socket.connect patched and fails on any address other than 127.0.0.1/::1
  - scripts or CI check: the PR diff touches no path outside this repository and no lobes-cli or nvsh file is vendored or edited (recorded in the delivery's evidence)

### t20 — Docs and prompt files for the Jetson-ops run

- instruction: No root AGENTS.md. lobes-cli attribution only in .pi/SYSTEM.md. Never name side/experiment models.
- depends on: t17
- acceptance:
  - docs/distill.md describes the 2610 retarget, the tunnel (command, env-var name only), the gold review, the edge stage and the board serve command under a cgroup cap; CLAUDE.md, QWEN.md, AGENTS.override.md and AGENTS.colleague.md updated in their own framing; markdownlint and harness-smoke pass

### t21 — Version bump and CHANGELOG

- instruction: Use /version-bump minor.
- depends on: t20
- acceptance:
  - pyproject.toml version bumped (minor), CHANGELOG entry in Keep a Changelog form; version-check passes

### t22 — Draft the Jetson-ops records: train pool and ~200-300 test candidates with proposed gold

- instruction: Agent-drafted data in the run directory (not committed). Redact hostnames, IPs and side-model names at draft time. The operator approves each test record via jev review --gold before it counts (c20/c30).
- depends on: t7, t9, t8
- covers: c7, h7
- acceptance:
  - a run-dir JSONL of records drafted from the pinned sources, each with source, licence, attribution (when cc-by-4.0), `gold_provenance` and split; test candidates are drafted from text disjoint from the train pool and carry proposed gold for jev review --gold
  - the general-guard share (typed-decisions) is a recorded knob; the dataset stage accepts the file with zero refusals other than counted `over_slots`/`over_length`; no Clef probe-set text appears

### t23 — Operator run: lock the prereg, label, train, quantize, heal, final, edge, bundle, upload private

- instruction: Blocked by design until the operator runs it. Ask before stopping any GPU process. Progress updates every 30 min from jev status --watch.
- depends on: t17, t21, t22, t1
- covers: c1, h1, c3, h3, c4, h4, c5, h5, c17, h14
- acceptance:
  - the operator decides the open pre-lock knobs (frame park v2) and the AGX/Nano memory caps before preregister --apply; every stage manifest exists; final reports every c17 bar with n and CI for bf16 and quant; the edge manifest exists for the AGX Orin and, once reachable, the Nano
  - the bundle and dataset bundle are uploaded private and fetched back with matching sha256; nothing is made public

## Risks

- [unknown_nonblocking] Jetson runtime unknown (frame park v1): torch + transformers 5.x for Qwen3.5 Gated DeltaNet and a 4-bit linear kernel on `sm_87` under JetPack 6.2 (CUDA 12.6) may not exist; t1 settles it before t10 picks a kernel; if neither int4 nor int8 runs, the Nano target needs a recorded deviation (task t1)
- [unknown_nonblocking] pre-lock knob decisions from #18 still open (frame park v2: grid and selection semantics, split fractions, audit thresholds, KD temperature, `init_mode`, `kl_frozen_weight` 0.3 used by 2610, `confident_wrong` 0.9, gate coverage 0.8, scalar temperature): the operator decides them before preregister --apply (task t23)
- [unknown_nonblocking] size budget is tight: INT4 blocks ~0.77 GB plus an INT4 embedding ~0.27-0.29 GB estimates 1.0-1.06 GB against < 10^9 bytes; an INT8 embedding does not fit, so the build may need INT4 embedding with grouped scales or lower-bit blocks — a miss is recorded unavailable, and changing the route is a deviation (task t10)
- [unknown_nonblocking] the Nano may stay unreachable; the AGX Orin edge numbers then stand alone and the Nano rows are reported as not measured (task t23)
- [unknown_nonblocking] teacher labelling throughput: one sequential request at a time over a tunnel to the Orin gateway; train-pool size is bounded by labelling time and must not starve Clef's production lane (task t22)
- [unknown_nonblocking] torchao int4 failed on CPU in #18 ('Requires mslk >= 1.0.0') and was never run on a GPU (lapse l14); the build may have to run on the AGX Orin or the Nano itself (task t10)
- [out_of_scope] lobes' Nano card, shape and quantized backend are a separate lobes plan consuming docs/distill-bundle-format.md (task t18)
- [out_of_scope] nvsh's /v1/systemone provider for evals/`tool_jev` and the Tool-Jev comparison
