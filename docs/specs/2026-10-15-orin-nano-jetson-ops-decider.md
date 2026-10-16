# Orin Nano Jetson-ops decider

> A Strands Decider 2B distilled from Clef 27B knows how to operate a Jetson Orin, and a quantized, healed build under 1 GB runs fast and accurately on a Jetson Orin Nano as a lobes decider domain

## Audience

- the operator and agents that run Jetson boxes (nvsh-style shells, lobes peers) asking typed decisions — choice / noul / score over /v1/systemone — about operating a Jetson Orin

## Before → After

- Before: the stock strands-decider-2B-qwen3.5-v1-2610 scores top-1 0.756 on the Clef probe set (Clef W4A16 0.9354; incident 0.655), knows nothing Jetson-specific, runs bf16 only (~3.8 GB of loaded weights: 1.88B params incl. a 508.6M-param 248,320x2048 embedding), and there is no Orin Nano card in lobes
- After: a distilled Jetson-ops 2B, LoRA merged, quantized and healed to under 1 GB on disk, ships as a private bundle (weights, calibration.json, gate.json and a documented loader contract) that runs on a Jetson Orin Nano through jev-factory's own /v1/systemone serve wrapper and agrees with its bf16 student and with Clef on the held-out human-gold Jetson-ops test set; hosting it as a lobes decider domain is lobes' separate plan

## Why it matters

- an Orin Nano (8 GB unified, shared with the OS and the workload) cannot host Clef 27B or a bf16 2B beside real work; a <1 GB specialist puts a calibrated Jetson-ops decider on the smallest board, local and offline

## Requirements

- distillation reuses jev-factory's distillery (jev distill: preregister .. upload), retargeted from hobson-v21 to strands-decider-2B-qwen3.5-v1-2610 @216d1ccb: pins.py `DECIDER_REPO` + adapter/head sha256, prereg.py `lora_r` 48 / `lora_alpha` 96 / `lora_targets` incl. `in_proj_qkv`,`in_proj_z`,`in_proj_a`,`in_proj_b`,`out_proj`, the `stock_v21` reference row
  - honesty: after retargeting no v21 pin, digest or reference row remains in pins.py or prereg.py, and the 2610 adapter and head sha256 match the snapshot at 216d1ccb
- a new Jetson-ops domain dataset of typed-decision records ({id,state,questions,gold,split,source,licence,`gold_provenance`}, licence apache-2.0, mit or cc-by-4.0 with attribution) built from the operator's measured Orin evidence in lobes-cli docs/evidence, jetson-containers-nv (`l4t_version.py`, setup/swap/ram scripts, package compatibility), jetson-ai-lab tutorials, NVIDIA-AI-IOT jetson-device-skills and jetson-bsp-skills at pinned commits and nvsh device-CLI verbs; Clef labels the train side
  - honesty: every record carries source, licence and `gold_provenance`; a source outside the licence allowlist is refused at the dataset stage; test records come from text disjoint from train and the contamination check fails closed
- the headline eval is a held-out Jetson-ops test set with HUMAN gold (jev-factory counts only human-gold records toward its bars), scored for accuracy, agreement with Clef, ECE and confident-wrong rate, plus the existing Clef probe set to measure loss of general ability
  - honesty: the headline table counts only operator-approved human-gold records on the test side, measured once in final; the Clef probe-set score is reported and gates nothing
- quantize after distilling: merge the LoRA, drop the unused vision tower (331.4M) and MTP head (60.8M), quantize the 24 text blocks to 4-bit, and quantize the full-vocabulary 248,320x2048 embedding (508.6M params, 1.0 GB in bf16) to INT4 or INT8 so the weight files total < 10^9 bytes; keep the fp32 pointer head (12.6 MB) and small DeltaNet params (`in_proj_a`/b, conv1d, `A_log`, `dt_bias`, norms) unquantized
  - honesty: on-disk weight size is computed from the bundle's files and is < 1 GB; a pruned vocabulary is recorded with its kept id list, and a prompt token outside it fails closed
  - honesty: the weight files' byte sum is < 10^9 and the tokenizer is the unmodified 2610 tokenizer (no id is unreachable)
- heal the quantized model: a short distillation pass that trains the quantized student (or its LoRA/norms) against the bf16 distilled student's and Clef's soft labels, and the healed build must reach agreement >= 0.97 with the bf16 student on the Jetson-ops fit/selection folds before it can ship
  - honesty: heal runs at most once; the >= 0.97 agreement is measured on the fit and selection folds only, never on test or held-out
- teacher-label reaches the Orin gateway through an SSH tunnel to 127.0.0.1, so SystemOneClient's localhost-only guard (`require_localhost`) stays, and the client gains the gateway bearer key (env-var name only) and the X-Lobes-Model: decider:general header, which today it does not send
  - honesty: no non-localhost URL is ever configured; the bearer key is passed by env-var name and is never printed or written to a manifest
- the contamination check is retargeted to 2610: data/`upstream_sources.json` is regenerated from the 2610 model card at 216d1ccb (48 datasets, incl. tasksource/tasksource-jev-typed-decisions and tasksource/procedural-typed-decisions), so typed-decisions general-guard records are marked overlapping, not clean
  - honesty: the regenerated manifest lists exactly the 2610 card's datasets at 216d1ccb, and a typed-decisions record is reported overlapping
- before training, stock 2610 is served on the inferred base revision b1485b2 (provenance.json: 'inferred: Hub main at training time; hosts did not pin it') and must reproduce its own published JevBench w4096 result (eval/summary.json, 0.7792 = 180/231) within its CI; a miss stops the run as a base-mismatch
  - honesty: the reproduction uses the same served path and w4096 window as the card, and its result is recorded in the benchmark manifest with n and CI
- a record counts as human gold only with an operator approval record (record sha256, decision, timestamp) written by a jev review gold-approval path; agent-drafted test records are stored as proposed and the final stage refuses a test record without a matching approval (records.py `gold_provenance` is a free string today)
  - honesty: a test record whose sha256 has no approval entry is refused by final with its id printed; editing an approved record voids its approval
- every Jetson-ops question offers at most 24 options (`strands_decider_config.json` `num_slots` = 24) and fits `max_length` 4096; the dataset stage refuses, never truncates, a record that exceeds either
  - honesty: the dataset manifest counts refusals by reason (`over_slots`, `over_length`) and no record is truncated
- the dataset bundle never contains the Clef probe set (private) and passes scripts/scan-secrets.py plus a hostname/IP and side-model-name scan before upload; evidence-derived records are redacted at draft time
  - honesty: the upload stage refuses when any scan finds a hit, printing file and line only

## Honesty conditions

- the bundle that runs on the Nano is the same sha256-pinned artifact measured in final; nothing is rebuilt on the board
- every Jetson-ops record is a /v1/systemone body (choice, noul or score) that an nvsh shell or lobes peer could send; no tool-calling or free-text records
- stock 2610's numbers are re-measured by jev-factory's benchmark stage on the probe set and the Jetson-ops sides, not copied from the lobes scope
- the quantized student's resident memory on the board is measured and reported next to the board's free memory under workload (c19)
- final measures the served bundle on the AGX Orin, and on the Nano once reachable, through jev-factory's serve wrapper, not the bf16 checkpoint
- no jev-factory task edits lobes-cli; the bundle-format doc is the hand-off
- the only model jev-factory starts on the AGX Orin is the <1 GB quant for measurement, under the recorded memory cap and never during teacher labelling; Clef's lane and every other box's served model are untouched
- every threshold is written into the pre-registration before the first train command and is measured on the shipped quantized artifact as well as the bf16 student
- no jev-factory task edits nvsh

## Success signals

- on the held-out human-gold Jetson-ops test set both the distilled bf16 student and the shipped healed quant beat the stock 2610 by >= 10 points top-1 and are within 5 points of Clef W4A16; the healed quant is < 1 GB of weights on disk and agrees >= 0.97 with the bf16 student; median and p95 latency on the AGX Orin and the Nano are measured and reported

## Scope / boundaries

- lobes' orin-nano machine card, shape and quantized decider backend are a separate lobes plan, not jev-factory tasks; jev-factory fixes and documents the bundle format (file layout, quantization scheme, loader entry point) that plan consumes
- Clef 27B stays served only on the AGX Orin (W4A16); this work changes no other box's served model or shape, and the existing decider:strands-general on spark2 stays as the stock baseline
- nvsh's /v1/systemone provider for evals/`tool_jev` is nvsh's work, not jev-factory's; the comparison with nvsh's Tool-Jev 0.8B `Q4_K_M` scorer is reported once that provider exists, never gated

## Non-goals

- not a general-purpose decider on the Nano: the student is a Jetson-ops specialist; general decisions keep routing to Clef via the mesh
- no lobes-side calibration layer (applying calibration.json/gate.json on the decider lane) in this plan; the bundle's own per-kind temperatures are the calibration

## Assumptions

- the Nano runs the decider with plain torch + transformers 5.x (the same single-forward-pass code path as the spark2 sidecar, whose torch fallback for the Gated DeltaNet needs no fla/causal-conv1d kernels) plus a 4-bit linear kernel that exists for `sm_87` on JetPack 6; GGUF/llama.cpp, TensorRT-LLM and vLLM would each need hidden-state extraction and the pointer head outside the runtime

## Scope exploration

- `s1` — `jev-factory jev_factory/distill (pins.py:20-33, prereg.py:136-254, pipeline.py STAGE_ORDER)`: 14-stage distillery exists, built and tested on fakes only, never run on a real checkpoint, GPU or Clef (docs/distill.md; delivery run partial, t26 blocked); every v21 pin listed for retargeting
  - seeds: `c6`
- `s2` — `jev-factory records.py:118-415, dataset stage; ~/git/jetson-containers-nv, ~/git/jetson-ai-lab, lobes-cli docs/evidence/*orin*`: record schema and loaders exist (`from_domain`, `distill_records` JSONL); no Jetson-ops dataset exists anywhere; the repos hold rich Orin ops text but almost nothing on the Orin Nano (only an 8 GB RAM tutorial and a `minimum_jetson` field)
  - seeds: `c7`
- `s3` — `jev-factory metrics.py, benchmark.py, final.py; docs/distill.md headline rule`: bars computed on human-gold test records only; typed-decisions gold is model-generated (lapse l1); risk r13: a small human-gold set may be too small for the paired CI
  - seeds: `c8`
- `s4` — `spark2 HF cache: Qwen3.5-2B-Base safetensors header + strands checkpoint 216d1ccb (strands_decider_config.json, modeling.py:228-482)`: loaded torso 1.882B params: blocks 1,373.3M + embedding 508.6M; `lm_head` never loaded (single forward pass, pointer head on hidden states); estimates INT4 blocks ~0.77 GB, INT4 embedding ~0.27-0.29 GB -> ~1.0-1.06 GB, or bf16 embedding pruned to 32-50k vocab -> ~0.9-0.97 GB: <1 GB is plausible but tight
  - seeds: `c9`
- `s5` — `lobes-cli scripts/clef_quant/quantize.py, scripts/clef_eval/compare.py; jev-factory quant.py:95-118, backbones/pointer/quantized.py`: our Clef quant script (llmcompressor 0.14.0, W4A16/NVFP4, keep-lists, joint head copied) is reusable with new keep-lists; nothing in either repo heals after quantizing; jev-factory's torchao int8/int4 builders were never run on a GPU (lapse l14) and all four of its formats are recorded unavailable; compare.py's PROMOTABLE bar is agreement >= 0.97
  - seeds: `c10`
- `s6` — `lobes-cli origin/main lobes/machines/orin.py, _strategy.py:51-110, __init__.py:29, profiles/loader.py:50, profiles/shapes.py:248-440, shape_render.py:235-376, decider_sidecar/server.py:235-262`: no Orin Nano card; name-only detection resolves a Nano to orin (64 GB budgets: util 0.45, window 32768 — wrong for 8 GB); the strands backend loads bf16 LoRA on the hub base with no quantization hook; a new `DECIDER_BACKENDS` entry must be added in server.py and shapes.py (pinned together by a test) plus a `DECIDER_DOCKERFILES` entry
  - seeds: `c11`
- `s7` — `jev-factory jev_factory/distill/systemone.py:75-198, teacher.py:37-130, measure/preflight.py:47`: client accepts only http:// to localhost with no redirects and sends only Content-Type: no auth header, no X-Lobes-Model; labelling is sequential, one request per record, cached by body sha256; records over 4096 student tokens are dropped, never truncated
  - seeds: `c12`
- `s8` — `strands_decider package in the spark2 decider image (modeling.py, infer.py:92-135, mps_kernels.py/cpu_kernels.py fallbacks); lobes-cli Dockerfile.decider`: inference is one forward pass, no generation, with prefix-cache forking over the transformers-5 layered cache incl. DeltaNet states; the image has no fla/causal-conv1d and runs on torch fallbacks; pins transformers 5.17.0, peft 0.21.2, torch 2.11+cu130 (not JetPack 6's CUDA 12.6)
  - seeds: `c13`
- `s9` — `lobes-cli docs/deliveries/2026-10-14-orin-decision-models.md (merged ed4aea6)`: orin-decider serves Clef W4A16 at the 32 GiB cap with ~3.4 GiB unreclaimable headroom: teacher-labelling load runs there and must not change that lane
  - seeds: `c14`
- `s10` — `nvsh evals/tool_jev (request.py, providers/base.py, openai_compat.py, manifest.example.toml); ~/.cache/huggingface/hub/models--jetson-ai-lab--qwen3.5-0.8b-nvsh-tool-jev*`: the gate scores replayed predictions or live /chat/completions first-token logprobs; no /v1/systemone provider; nvsh already ships a Jetson Tool-Jev scorer (Qwen3.5-0.8B `Q4_K_M`, ~355 ms/decision on an AGX Orin) — a prior art the Nano decider must be compared with
  - seeds: `c18`
- `s11` — `ssh host nano (~/.ssh/config Host nano, user nano)`: the Orin Nano is configured but unreachable on 2026-10-15 (ssh: connect to port 22 timed out); its JetPack/L4T version, power modes and free memory are unknown until it is online
- `s12` — `nvsh scripts/lfm-finetune (skills stage, augment-skills), tests/fixtures/jetson_skills/README.md, docs/lfm-finetune.md:280-287`: nvsh already builds training and eval data from NVIDIA-AI-IOT/jetson-device-skills and jetson-bsp-skills at pinned commits (38 tools, 104 test evals; docs CC-BY-4.0, code Apache-2.0) with a leakage check and resumable augmentation; reusable as a source for the Jetson-ops domain
  - seeds: `c23`, `c7`
- `s13` — `nvsh docs/qwen-tool-jev-finetune.md:85-108, 283-308, 519-520 (t25, d18)`: precedent for quantize-and-heal on a Qwen3.5 student: `heal_needed`() fired after a 3.125-point loss and the heal was a short bf16 continuation, not QAT (d18); shipped a3-heal.`q4_k_m` GGUF at ~1.75 GiB resident on GPU, 539 ms median on an AGX Orin
  - seeds: `c10`
- `s14` — `challenge pass / adjacent-systems lens: HF StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 @216d1ccb (README datasets, provenance.json, lora/adapter_config.json, strands_decider_config.json)`: card lists 48 training datasets incl. tasksource-jev-typed-decisions (contamination manifest is v21's 29); base revision inferred, not pinned; LoRA r48/alpha96 over 12 targets (incl. q/k/v/o/gate/up/down, beyond c6's list); `num_slots` 24, `max_length` 4096; `temperature_by_kind` present; `kl_frozen_weight` 0.3
  - seeds: `c28`, `c29`, `c31`
- `s15` — `challenge pass / data-flow lens: jev_factory/distill/records.py (GOLD_PROVENANCES), dataset.py (ALLOWED_LICENCES, _carve), pipeline.py stage_dataset`: `gold_provenance` 'human' is a free string with no approval binding; licence allowlist is a frozenset of apache-2.0/mit (cc-by-4.0 needs a code change per c24); held-out is carved per source from train-split records, test comes from split=test records
  - seeds: `c30`
- `s16` — `challenge pass / security lens: systemone.py require_localhost + c12/c27 tunnel, HANDOFF rules (private Clef probe set, side-model names off-record)`: tunnel keeps the guard; bearer key by env-var name covered by h12; new exposure is the dataset bundle carrying evidence text and the private probe set — seeded c32
  - seeds: `c32`
- `s17` — `challenge pass / reversibility+recovery lens: teacher.py (resumable, cached by body sha256), quant.py (heal once), bundle/upload (private-first)`: labelling resumes after a dropped tunnel; heal once; upload private-first with hash fetch-back; clean pass, residual risk only in the Orin gateway being down for long stretches
- `s18` — `challenge pass / unexamined surfaces`: NOT examined: lobes-cli docs/evidence content (redaction needs unknown), the Nano's actual JetPack/L4T and free memory (box unreachable), Jetson torch/transformers wheel availability (parked v1), the jetson-\*-skills repos at their pinned commits
- `s19` — `challenge pass / lifecycle lens: distill/final.py once ledger vs c22 (Nano unreachable)`: test is measured once; a later Nano run on test would be a second measurement needing a deviation; routed as pending decision 'Nano vs once-only test'
- `s20` — `challenge pass / operations lens: h5 vs h13 on the AGX Orin; lobes delivery s9 (Clef 32 GiB cap, ~3.4 GiB headroom)`: measuring the served bundle on the AGX Orin starts a model there, which h13 forbids; routed as pending decision on running the quant beside Clef

## Decisions

- the < 1 GB budget bounds the quantized weights on disk; resident memory on the board is measured and reported, not bounded
- the agent drafts about 200-300 held-out Jetson-ops test records with proposed gold from measured evidence and sources; the operator approves or corrects each one before it counts as human gold
- the student is a Jetson-ops specialist with a guard: mostly Jetson-ops data plus some general typed-decision data; the general Clef probe-set score is reported, not gated
- the target board is a Jetson Orin Nano Super 8GB (JetPack 6.2 / L4T 36.4 expected, to be read off the board); until it is reachable again, the build is tested on the AGX Orin 64GB
- the domain covers Jetson Containers usage, general Jetson commands (jtop, swap configuration, finding logs) and NVIDIA's Jetson skills (NVIDIA-AI-IOT/jetson-device-skills and jetson-bsp-skills at pinned commits), the same skills nvsh's lfm-finetune pipeline used
- the Jetson skills' CC-BY-4.0 documentation enters the dataset: the licence allowlist gains cc-by-4.0 for this run, and the dataset and model bundles carry NVIDIA's attribution NOTICE (a recorded deviation from apache-2.0/mit)
- the accuracy bars (>= +10 pts over stock 2610, within 5 pts of Clef W4A16, on the human-gold Jetson-ops test set) apply to both the bf16 student and the shipped healed quant
- Nano latency is measured and reported (median and p95 on the AGX Orin, then the Nano), not pre-registered as a bar; a latency bar is set by a recorded decision once a first Nano measurement exists
- teacher labelling reaches the Orin's lobes gateway through an SSH tunnel to 127.0.0.1, keeping the localhost-only guard; the client adds the bearer key by env-var name and X-Lobes-Model: decider:general
- the Nano (once reachable) measures latency, memory and prediction agreement on the selection fold and the probe set only, never the test side; the headline comes from the single AGX Orin final on the same sha256 bundle
- the <1 GB quant may run on the AGX Orin for measurement only, under a memory cap recorded in the manifest, never while teacher labelling runs, and without reconfiguring Clef's lane
- '< 1 GB' means < 10^9 bytes of weight files on disk
- the embedding stays full-vocabulary and is quantized (INT4 or INT8); the vocabulary is not pruned
- a pruned-vocabulary build (bf16 or quantized embedding over the ids the Jetson-ops and general prompts use, 32-50k) is also built as a comparison: it is measured beside the full-vocabulary quant on size, accuracy and out-of-vocabulary refusals, reported only; the full-vocabulary quant stays the shipped build

## Open parks

- [unknown_nonblocking] whether a Jetson torch + transformers 5.x stack runs Qwen3.5's Gated DeltaNet torch fallback and a 4-bit linear kernel on `sm_87` under JetPack 6.2 (CUDA 12.6), given strands-decider pins torch 2.11+cu130
- [unknown_nonblocking] the pre-lock knob decisions left open in #18's delivery (grid and selection semantics, split fractions, audit thresholds, KD temperature, `init_mode`, `confident_wrong` 0.9, gate coverage 0.8, scalar temperature) apply to this run too; 2610 itself trained with `kl_frozen_weight` 0.3, which the distillery does not apply
