# Delivery Summary — Orin Nano Jetson-ops decider

plan: `orin-nano-jetson-ops-decider` · run: `partial` · date: `2026-10-15`
baseline: `devague summary skeleton`

## Intent

> A Strands Decider 2B distilled from Clef 27B knows how to operate a Jetson Orin, and a quantized, healed build under 1 GB runs fast and accurately on a Jetson Orin Nano as a lobes decider domain

After: a distilled Jetson-ops 2B, LoRA merged, quantized and healed to under 1 GB on disk, ships as a private bundle (weights, calibration.json, gate.json and a documented loader contract) that runs on a Jetson Orin Nano through jev-factory's own /v1/systemone serve wrapper and agrees with its bf16 student and with Clef on the held-out human-gold Jetson-ops test set; hosting it as a lobes decider domain is lobes' separate plan

The run built every planned task. It then ran the operator run (t23) once, as an overnight **proving run on unverified, model-drafted gold** (d10). That run went through every stage up to `bundle`; `upload` and the Nano measurement were not done.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Jetson runtime spike: prove the 2610 decider's torch path and a 4-bit kernel on JetPack 6.2 (`sm_87`)
- `t2` — Retarget student pins from hobson-v21 to strands-decider-2B-qwen3.5-v1-2610 @216d1ccb
- `t3` — Rename the reference role `stock_v21` to a model-neutral stock role across the distillery
- `t4` — Retarget the pre-registration to 2610 and the c17 bars
- `t5` — SystemOneClient: bearer key by env-var name and the X-Lobes-Model header, still localhost-only
- `t6` — Contamination check against 2610's 48 card datasets
- `t7` — Dataset stage: cc-by-4.0 with attribution, 24-option and 4096-token limits, typed-kind validation
- `t8` — Human-gold approval ledger: jev review --gold FILE
- `t9` — Jetson-ops source registry: pinned repos, licences and attribution
- `t10` — Quantized student build: merge LoRA, drop vision/MTP, 4-bit blocks, quantized full-vocab embedding, size check, attestation endpoints
- `t11` — Pruned-vocabulary comparison build (reported, never shipped)
- `t12` — Heal against the bf16 student's and Clef's soft labels, gated on >= 0.97 agreement
- `t13` — Benchmark: re-measure stock 2610 and reproduce its published JevBench w4096 result
- `t14` — Final: measure the AGX-served quant via tunnel with attestation; human-gold only; bars for bf16 and quant
- `t15` — Edge stage: non-test measurement on the AGX Orin and the Nano through a tunnel
- `t16` — Bundle: CC-BY attribution NOTICE and pre-upload scans
- `t17` — Wire the run: config keys, edge stage registration, gold ledger, contamination and stock-reproduction gates, jev-CLI domain
- `t18` — Bundle-format doc for lobes' Nano backend
- `t19` — Boundary guards: localhost-only connections, no lobes/nvsh edits
- `t20` — Docs and prompt files for the Jetson-ops run
- `t21` — Version bump and CHANGELOG
- `t22` — Draft the Jetson-ops records: train pool and ~200-300 test candidates with proposed gold
- `t23` — Operator run: lock the prereg, label, train, quantize, heal, final, edge, bundle, upload private
- `t24` — jev review --serve: a gold page over the t8 ledger
- `t25` — jev review --drafts: a local draft-review page with evidence

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | runtime spike on the Orin Nano, docs/evidence/2026-10-jetson-runtime-spike.md |
| `t2` | delivered | `distill/pins.py` names 2610 @216d1ccb with full adapter/head digests |
| `t3` | delivered | reference role `stock` (config `distill_stock_url` / `distill_stock_model`) |
| `t4` | delivered | prereg schema 2 with the c17 bars; LoRA r48/alpha96, 12 targets from 2610 |
| `t5` | delivered | bearer key by env-var name and `X-Lobes-Model`; localhost-only. The benchmark and final stages built Clef's client without them (HTTP 401 live), fixed in `5679a19` |
| `t6` | delivered | contamination against the 2610 card plus non-card training sources (d1) |
| `t7` | delivered | cc-by-4.0 with attribution, 24-option and 4096-token refusals, typed-kind validation |
| `t8` | delivered | `jev review --gold` append-only approval ledger |
| `t9` | delivered | `jev_factory/distill/jetson_ops/` source registry |
| `t10` | delivered | int4 build of 985,675,969 weight bytes, measured; `/v1/bundle` and `/v1/stats`; serving now fills a meta torso (r9, `a17dc91`, `14be91f`) |
| `t11` | delivered | pruned-vocabulary comparison build, reported and never a candidate |
| `t12` | delivered | heal code works and is deviation-gated; in the run no healed build cleared both bars, so the shipped int4 is un-healed (d11, d13, d14) |
| `t13` | delivered | stock 2610 JevBench w4096 reproduced live: 165/213 = 0.775, CI 0.714–0.826 covers 0.7792 |
| `t14` | delivered | final ran twice (d15): four models measured live; Clef was 401 in run 1 and answered 538/538 in run 2 |
| `t15` | partial | edge ran on the AGX Orin (d16); the Orin Nano run is staged but paused by the operator |
| `t16` | delivered | NOTICE and pre-upload scans; the scan now proves `torch.save` archives (`0fd1c88`) |
| `t17` | delivered | run wired: config keys, edge stage, gold ledger, contamination and reproduction gates |
| `t18` | delivered | docs/distill-bundle-format.md, checked by a test |
| `t19` | delivered | localhost-only boundary tests (lapse l9 notes they pass vacuously for real clients) |
| `t20` | delivered | docs and the four prompt files describe the proving run (`856fe8d`) |
| `t21` | delivered | version 0.12.0 with its CHANGELOG entry |
| `t22` | delivered | 1371 drafted records (1133 train / 238 test), choice-only (d8, d9), sources per d3 |
| `t23` | partial | proving run through `bundle` on unverified gold (d10–d16); no verified gold, no upload, no Nano edge |
| `t24` | delivered | gold page over the t8 ledger (d2) |
| `t25` | delivered | draft-review page with evidence, redesign, tailnet host, soft gold and restore (d4–d7) |

## Mid-work Decisions

- `d1` — contamination manifest also lists 2610's non-card training sources — fail closed
- `d2` — add a task: a gold page over t8's ledger — ~200-300 test records are approved one by one
- `d3` — dataset sources are NVIDIA public repos plus agent-drafted general ops, not the operator's own stack — the model must not be opinionated about that stack
- `d4` — add a task: a draft-review mode on the review site — the operator reviews drafted records before and during the fan-out
- `d5` — the draft-review page redesigned (light and dark themes, dataset overview) — operator request
- `d6` — `--allow-host` for a tailnet proxy, still bound to 127.0.0.1 with a token — the operator wants the site on the tailnet
- `d7` — soft gold, `add_option` and restore in draft review — none of the pilot's drafted options was right, gold must be probabilities, and an edit had no way back
- `d8` — choice-only questions with "I'm not sure." — operator rule
- `d9` — `not_sure` added by the operator with one click, not drafted onto every question — amends d8
- `d10` — proving run of t23 on unverified model-drafted gold for every side, with a one-point grid — the operator wanted a trained, healed and tested 2B by morning to prove the process
- `d11` — second heal round with a bf16-only mix — round 1 left agreement at 0.775
- `d12` — quant rules gated on the clean (Jetson-ops) sources; the guard is a floor, not a target — the spec says the guard is never gated, and pooled it was 68% of the agreement questions
- `d13` — third, balanced heal round (heal set rows repeated) — scaling heal strength could not clear both bars
- `d14` — ship the un-healed int4 with the guard report-only — it passes every Jetson-ops bar; three heals each lowered Jetson-ops agreement
- `d15` — second final run after fixing Clef's auth — run 1 measured no Clef
- `d16` — edge on the AGX Orin with the fit fold as the probe set — the private probe set is not on this machine and the Nano image could not load torchao 0.16
- Not covered by a record: three blocking bugs were found and fixed only during the run. The trainer saved the head as a pickle (`6048f3d`); `--allow-foreign-gpu` did not reach three GPU guards (`378b368`, `64fb92c`); score answers were mapped to levels by position (`052831a`). The last one would have mis-scored every score question that `gate` and `final` read from disk.
- Not covered by a record: the student venv carried an older strands-decider than the pinned source; the pinned one was installed and the lockfile regenerated (backups kept).
- Not covered by a record: at the operator's request the AGX Orin server was stopped to free memory. A gateway recreate by this session's key-alignment script had dropped the gemma2 embedder route; it was restored with the gateway's full compose file set.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t6` (`d1`) | operator: fail closed — a source 2610 trained on but missing from the card would otherwise check clean | acceptable |
| `t8` (`d2`) | operator: ~200-300 test records are approved one by one; t8 shipped CLI-only because server.py was outside its files | acceptable |
| `t22` (`d3`) | operator: the model must not be opinionated about the operator's personal tech stack; it should know the public Jetson Orin tooling and defaults | acceptable |
| `t24` (`d4`) | operator wants a site to work with when reviewing drafted records before and during the fan-out | acceptable |
| `t25` (`d5`) | operator asked for light mode and a frontend-design + dataviz experience that reflects the data | acceptable |
| `t25` (`d6`) | operator wants the review site on the tailnet | acceptable |
| `t25` (`d7`) | operator review of the pilot: none of the drafted options was right (escalate / do not run was), gold must be probabilities not one answer, and a chosen edit had no clear way back to the drafted gold | acceptable |
| `t22` (`d8`) | operator rule: always offer an I'm-not-sure option; the systemone noul type is fixed to true/false and score levels are ordered, so only a choice can carry it | acceptable |
| `t22` (`d9`) | operator: with the add-option buttons there is no need to put `not_sure` on every question; they add it as needed | acceptable |
| `t23` (`d10`) | operator 2026-10-16: continue with the unverified gold, marked with its unverified share, on all datasets just to prove the process, and have a trained, healed and tested 2B model by morning | risky |
| `t23` (`d11`) | operator 2026-10-16: chose a 2nd heal with a bf16-only mix after quantize stopped (`stop_escalate`); the bar measures agreement with the bf16 student and round 1 half-targeted Clef. If it still misses 0.97, stop and report | risky |
| `t23` (`d12`) | operator 2026-10-16: the guard is a non-regression check, not a target (spec: typed-decisions agreement is never gated); pooled, it was 68% of the fit+selection agreement questions and 81% of selection, with weak gold (Clef 0.634) and overlapping contamination. Operator asked to run the measure-only per-source re-measure first | risky |
| `t23` (`d13`) | operator 2026-10-16 chose a balanced third heal after the per-source re-measure: int4 pre-heal passes Jetson-ops (agreement .976, accuracy >= bf16) but its guard is .469 (floor .575); round 2 recovered guard to .557 but Jetson-ops agreement fell to .961. Scaling heal strength cannot clear both bars, so the heal set weighs the domain rows up while still training the guard | risky |
| `t23` (`d14`) | operator 2026-10-16 chose option 1: the spec says typed-decisions agreement is never gated; the Nano model is a Jetson-ops specialist, and the most accurate domain build is the un-healed int4 | risky |
| `t23` (`d15`) | operator 2026-10-16 chose to rerun final rather than a Clef-only supplement | risky |
| `t23` (`d16`) | operator 2026-10-16 chose the fit fold as the probe set | risky |
| `t15` | the Orin Nano edge run is staged (artifact, torchao 0.16 image, config-only HF cache) but paused by the operator; only the AGX Orin was measured | needs-follow-up |
| `t23` | no human or model-verified gold, no `upload`, and test touched twice (d15); the run proves the process, not the model | needs-follow-up |
| `t5` | benchmark and final built Clef's client without the key and header, so the live run got HTTP 401 until `5679a19` | acceptable |

## Evidence

- tests: the 66 behavioral tests mapped to obligations o1–o18, listed per obligation in evidence records `e1`–`e18` (e.g. `tests/test_distill_edge.py::test_bundle_mismatch_refused`, `tests/test_distill_report.py::test_every_c17_bar_is_computed_for_bf16_and_quant_with_paired_ci_and_n`) — 66 pass at `856fe8d`
- tests: full suite `uv run pytest -n auto` — 3347 passed, 28 skipped
- lint: black, isort, flake8, bandit, markdownlint (CI globs), `scripts/scan-secrets.py`, `teken cli doctor --strict`, `harness-smoke.py --stage config` — all clean
- live run observations `e19`–`e25`: edge attestation and agreement (`e19`); Clef 401 in final run 1 (`e20`, **fail**) and 538/538 in run 2 (`e21`); contamination (`e22`); stock reproduction (`e23`); weight bytes (`e24`); heal rounds (`e25`). Behavioral deltas `b1`–`b10` trace d7–d16.
- run state: `docs/evidence/2026-10-16-jetson-ops-proving-run.md`
- commits: `29ec434..a48831a` on `spec/jetson-ops-nano-issue-20`
- issues: #20 (brief); agentculture/jetson-cli#11 (consumer instructions)

All evidence records are `llm`-origin and **proposed**: none is operator-confirmed yet.

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The distillery runs end to end on the real 2610 student, Clef teacher and Jetson-ops data: every stage through `bundle` completed | high | stage manifests under the run work dir; `docs/evidence/2026-10-16-jetson-ops-proving-run.md` |
| On unverified model-drafted Jetson-ops test gold, the bf16 student scores 0.968 and the int4 0.952, against stock 0.718 and Clef 0.952 | medium (gold unverified, d10; test touched twice, d15) | final report `distill/final/report.json`; `e21` |
| The shipped int4 is under 10^9 weight bytes (985,675,969) and attested by `/v1/bundle` | high | `e7`, `e24`; `distill/quant/measurements.json` |
| int4 agrees with bf16 ≥ 0.97 on test | **not met** — 0.960 (CI 0.935–0.984) | final report `unverified_c17.builds.quant` |
| c17 as written (human-gold test set, healed quant, Nano latency) | unverified | no human gold, quant un-healed (d14), Nano not measured |
| The heal step is deviation-gated, agreement-gated and protected-id-safe | medium | `e8`, `e25` |
| Clef is asked with the bearer key and model header in every stage | medium | `e10`, `e20` (fail, run 1), `e21` (pass, run 2); fix `5679a19` |
| The model serves on the AGX Orin under a cgroup cap and gives the same answers there as on the Spark (877/883) | high | `e19`; edge manifest |
| Serving without loading the bf16 base (r9) keeps the answers identical | medium | 883/883 selection on the Spark; not yet measured on a board |
| The bundle scans clean, including the `torch.save` pickle | high | `0fd1c88`; `tests/test_release_scan.py`; bundle `scan.json` |
| The model runs on an Orin Nano | unverified | paused; not claimed |

Lapse ledger evidence:

All nine lapses were approved by the operator on 2026-10-16, together with evidence `e1`–`e25` and deltas `b1`–`b10`. Each caps the claims it touches:

| Lapse | What | Claims it caps |
|---|---|---|
| `l1` | jetson-containers pinned by a fork-branch commit under the NVIDIA repo URL, not verified there | source provenance of the jetson-containers records |
| `l2` | int4 block path never run on a device; size was arithmetic | none now: the run measured 985,675,969 B and served it on the AGX Orin |
| `l3` | stock reproduction assumes the card grades argmax == gold | the stock-reproduction claim (`e23`) stays medium |
| `l4` | the card's CI taken as a 95% Wilson interval | same as `l3` |
| `l5` | pruned-vocabulary OOV count measured on its own prompts (≈0 by construction) | the pruned comparison (reported only, never shipped) |
| `l6` | pruned build's in-env paths never ran | same as `l5` |
| `l7` | final scores bf16 raw unless given a calibration, and measures agreement on the test pool | the test-side numbers and the 0.960 agreement miss |
| `l8` | heal QAT is a hand-written fake quant; trains LoRA and head, not norms | the heal claims (`e8`, `e25`): ran on a real checkpoint, recipe not validated |
| `l9` | the localhost boundary test passes vacuously for real clients | the boundary claim (`e11`) stays medium; the live run's 401 (`e20`) shows real clients matter |

## Remaining Work / Follow-up

- `t23` — verified gold, a fresh `final` and then `upload` (private). Next: data verification (model consensus for `[stated]` gold, humans on a public review site behind Cloudflare Access) and `jev distill publish-check`, the publish gate agreed on 2026-10-16. Owner: next PR.
- `t15` — Orin Nano edge run: start the staged image under a cap and run `edge` with `label=orin-nano`, following the resume doc. Owner: operator schedules it.
- int4 vs bf16 test agreement 0.960 < 0.97 — revisit once gold is verified; three heals did not help on the domain.
- The guard drop (int4 0.545 vs stock 0.596 on test) must go on any model card; scope the model to Jetson ops.
- `l1`–`l9` and `o18`'s live coverage await operator adjudication; evidence `e1`–`e25` and deltas `b1`–`b10` await confirmation.
- A second independent verifier model (Nemotron) is to be served by the operator.
- The JetPack 7.2 L4T topic text in `general-ops-topics.json` should say R39.
- Heal rule (code review of PR #21): after a heal, quantize measures the healed build's accuracy drop against the heal round's own bf16, while agreement (and final) use the selected bf16 student. Whether the drop should also use the selected student is a rule change for the operator to decide.
- PR #21 code review: nine of ten findings fixed (stale teacher-label lock, board memory from `/v1/stats`, sealed-only runtime, `/v1/bundle` re-verification, single hashing at serve start, CLI edits see ledger-added options, `--allow-host` error path, relative deployed artifact in edge, one agreement readout); the tenth is the heal-rule question above.
