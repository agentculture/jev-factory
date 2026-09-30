# extract jev process; jev-cli first model

> jev-factory ships nvsh's Tool-Jev fine-tune process (issue #4, under umbrella #1) as a domain-generic staged pipeline, and the first new model it builds is a jev scorer that operates the jev CLI itself
> instruction: Every artifact manifest in the jev-tool run's work dir names a jev-factory stage and version, and none points at an nvsh path

## Audience

- The operator, who approves every gate and runs builds; the jev-factory mesh agent, which runs the stages and records decisions; and future domain authors, nvsh first, who build their own jev-like models by writing a domain module instead of forking scripts
  - instruction: Check the exported spec and jev learn/explain name these three readers and what each does

## Before → After

- Before: Building a jev-like model means running nvsh's scripts/lfm-finetune by hand: 22 shell stages with no stage state, scripts wired to nvsh's operation table and tier runtime, domain facts duplicated across a dozen files, the selection rule in prose and decisions in a prose ledger, so a new domain means forking nvsh
  - instruction: Cite scope entries s12, s15, s16 and s13 for each part of this statement
- After: jev-factory owns the full jev-like build pipeline as `jev run <stage>` stages over a declarative domain module, with a state manifest per stage, machine-readable pre-registration and decision records, and every honesty check in code; its first product is a jev-tool scorer on Qwen3.5-0.8B that picks the right jev CLI verb, calibrated and gated, shipped privately as a GGUF bundle
  - instruction: Verify with 'jev run --help' listing every stage, 'jev status' reading the stage manifests, and the jev-tool bundle carrying calibration.json, gate.json and scorer-train.json

## Why it matters

- The operator will build many models across domains with this factory, and nvsh showed that each build costs days of GPU and reviewer time and fails silently when the process lives in a person's head (overfit rules chosen mid-run, leakage lapses, unverified merges, hand-copied bundle files); encoding the process once makes each new domain a module, not a fork
  - instruction: Trace each named failure to its nvsh record: #46 b1-rule, #39 l5 leakage, #46 l4 merge, and bundle gaps in issue #4 stage 16

## Requirements

- Stages that only need a candidate distribution (calibrate, gate sweep, metrics, selection rule, decide) take a backbone-agnostic predictions record; the letter-token readout, prompt rendering, trainer and quantizer live behind a per-backbone adapter, so #3's CLM heads and nvsh#67's GLiNER encoder can plug in later without touching those stages
  - instruction: A test feeds those stages a synthetic predictions file with no tokenizer or model present; an import-graph test asserts they never import the adapter module
  - honesty: calibrate, gate sweep, metrics, select and decide read only a backbone-agnostic predictions record (candidates, raw scores, probabilities) and never import the causal-LM adapter
- The checks listed in issue #4 extraction step 5 are code, not docs: readout completeness N/0, merge changed weights with 0 missing adapter keys, leakage fails closed against every protected file, heal trigger fires at most once, bundle carries calibration.json + gate.json + scorer-train.json with no symlinks and no absolute paths in GGUF imatrix metadata, private upload with hash-verified fetch-back, no final rerun without a deviation record
  - instruction: One test per check: readout N/0, merge weight changed with 0 missing keys, leakage fail-closed, heal once, bundle contents with no symlinks or absolute imatrix paths, private fetch-back, no final rerun without a deviation
  - honesty: Each check in issue #4 extraction step 5 has a named test that fails when the check is violated
- Teacher reviewers return a structured JSON verdict instead of free text parsed by augment.`parse_verdict`'s blacklist, which misread 'no-argument', 'ambiguous' and a mid-sentence 'but' (issue #4 stage 3)
  - instruction: Tests feed 'no-argument', 'ambiguous' and mid-sentence 'but' replies plus malformed JSON, and check the parsed verdict or error
  - honesty: Reviewer verdicts are parsed from JSON against a schema; a malformed or empty reply is retried twice and then recorded as an error, never as a reject
- Between-run decisions are machine-readable, append-only records (D-style: evidence with cited metrics, choice, who decided, record id), not only prose ledgers, because they are the Layer 3 decider's corpus (issue #4 'Process discipline'; issue #1 Layer 2)
  - instruction: Test: re-running decide appends a record and leaves earlier ones byte-identical; the schema test passes on every record
  - honesty: jev decide writes append-only JSON decision records (id, run, verdict, cited metric values with source-file sha256, rule version, decider rule|model|human) that a schema validates; an override is a new record, never an edit
- The release gate port is the last stage and keeps nvsh evals/' generic core (ledger/resume, manifest-as-data, provider adapters, per-provider USD caps with worst-case reservation, truncation stop, replay of saved predictions, model-only vs model+harness rows) while replacing its nvsh couplings: nvsh.ops.table (cases.py:49, request.py:64-67) becomes the domain module's `read_only` table with unknown=mutating, nvsh.redact (providers/base.py:50, judge.py:115) a jev-factory redactor, and path-loaded scripts/lfm-finetune metrics/gate/`calibration_fit` (trace.py:50, policies.py:82) the cited copies
  - instruction: Run the gate smoke with the fake provider in CI; grep evals code for nvsh imports (expect 0)
  - honesty: The imported release gate runs for the jev-tool candidate with raw policy and one provider, and its generic core has no nvsh import
- The evals worktree guard uses 'git rev-parse --is-inside-work-tree' rather than detecting a .git directory, since nvsh#71 item 2 shows an empty /tmp/.git trips every evals test in a sandbox
  - instruction: A test creates tmp/.git empty and runs the guard
  - honesty: The worktree guard uses git rev-parse --is-inside-work-tree, and the evals tests pass in a temp dir that contains an empty .git
- The jev-CLI domain module is generated from the argparse tree (`_build_parser`) plus a per-verb `read_only`/args annotation, not hand-copied: today three hand-kept lists (overview.py:70-112 `_VERBS`, learn.py:56-62 commands, explain/catalog.py:126-135) can drift and none carries `read_only` or arg schemas
  - instruction: A test compares the generated list with `_build_parser`() subcommands
  - honesty: The jev-CLI domain's operation list is generated from the argparse tree, and a test fails when a verb exists in argparse but not in the domain module, or the reverse
- Runtime package keeps dependencies = \[\] (pyproject.toml:15); training/quantize/serve/eval deps go in a jev-factory\[train\] extra or an external venv (plus a separate AWQ venv and an evals group), and verbs import them lazily
  - instruction: A test imports all modules with those packages blocked via sys.modules; uv pip install jev-factory pulls no ML packages
  - honesty: pyproject dependencies stays \[\] and every heavy import (torch, transformers, peft, unsloth, llmcompressor, `huggingface_hub`, deepeval) sits inside a function or behind an extra
- Imported nvsh code has a deliberate home: code inside `jev_factory`/ ships in the wheel, triggers publish.yml, and is subject to black/isort/flake8/bandit and the 60% coverage gate (pyproject coverage source=`jev_factory`); GPU-only paths that cannot be unit-tested import their heavy deps lazily and keep a tested pure-Python core
  - instruction: CI coverage job; grep for module-level pragma no cover
  - honesty: Coverage stays >= 60% on `jev_factory` with the imported code included, with no blanket 'pragma: no cover' on whole modules
- The absorbed nvsh code carries a one-time import provenance record (per file: upstream ../nvsh path, nvsh commit 9debdc6, adaptations made), laid out in docs/skill-sources.md's table style; after import jev-factory owns the code, and there is no ongoing re-sync from nvsh
  - instruction: A test parses the provenance doc and checks each listed file exists and no imported module is missing from it
  - honesty: Every imported file appears in a provenance table with its nvsh path, commit 9debdc6 and a list of adaptations
- Every new verb (init, run, status, decide) registers under cli/`_commands` with register(subparsers), raises CliError, supports --json, is dry-run by default with --apply for writes, and gets an explain/catalog.py entry so teken cli doctor --strict and `test_every_catalog_path_resolves` (tests/`test_cli.py`:111) pass
  - instruction: teken cli doctor --strict passes; a test runs each write verb without --apply and checks that no file in the work dir changed
  - honesty: Every new verb has an explain catalog entry, --json output and a CliError path, and write verbs change nothing without --apply
- The stale 'already domain-generic' list (issue #1, CLAUDE.md, nvsh playbook l.148-151) is not trusted when importing: gate.py:72-73 and metrics.py:127-128 import nvsh.ops.table and nvsh.tiers.bench, `train_scorer.py`:73-74 imports nvsh.tiers.lfm/bench, `scan_bundle.py`:34 imports nvsh.redact, `calibration_fit.py`:78 and `sweep_gate.py`:54-56 reach nvsh by path-loading metrics/gate, and `leakage_check.py`:38 imports Track-A-only `jetson_skills`; every import is rewired to the domain-module seam (operation table, explain/escalate labels, redactor, request formatting)
  - instruction: An import-graph test over `jev_factory` finds no nvsh module
  - honesty: No imported module imports nvsh, directly or transitively
- The letter/token readout (`LABEL_ALPHABET` scorer.py:121, `label_token_ids` :347, `label_variant_ids` :365, `READOUT_TOP` :128) is lifted out of nvsh-coupled scorer.py into the causal-LM backbone adapter as the single label definition that training, in-process scoring and served scoring all use
  - instruction: Test: the same prompt scored in-process and from served logprobs gives the same distribution within 1e-6 on a tiny model fixture; grep shows one `LABEL_ALPHABET` definition
  - honesty: One module defines the label alphabet and the token-variant summing, and training, in-process scoring and served scoring all call it
- Imported scripts become package modules with normal imports, replacing nvsh's `_sibling` importlib path-loading (e.g. gate -> metrics, `sweep_gate` -> gate + `calibration_fit`, `train_scorer` -> scorer + train)
  - instruction: grep -r `spec_from_file_location` `jev_factory` (expect 0)
  - honesty: No module uses importlib `spec_from_file_location` to load a sibling
- The pre-registered bars and lexicographic selection rule are a machine-readable, hashed artifact written before the first training command; nvsh's docs/tool-jev-calibration-rule.md is prose only (71 lines, committed b1c6cb6) and jev decide must evaluate it mechanically
  - instruction: Test: train without a pre-registration file exits non-zero; editing it after registration makes decide refuse until a deviation id is given
  - honesty: Pre-registration is a schema-validated file, hashed and recorded before the train stage may run; the train stage refuses without it, and any later change requires a deviation record
- jev run stages write a real per-stage state manifest (inputs with sha256, outputs, knobs, status) that makes them resumable; nvsh's pipeline.sh has no stage-state file — resumability is a docstring claim, sequencing is 'die if the prior artifact is missing', and `stage_cache.py` only stages a merged checkpoint into an HF cache (`stage_cache.py`:67-82)
  - instruction: Tests on the stage-manifest engine with toy stages
  - honesty: Re-running a completed stage with unchanged inputs is a no-op, and changing an input's sha256 marks that stage and all downstream stages stale
- jev-factory's run-config loader has one explicit, documented precedence rule, rather than copying pipeline.sh's surprise: the sourced env file overrides exported shell values (pipeline.sh:180), except `MEASURE_CTX`, which is saved and restored (:178-182); tool paths (`LLAMA_CPP_`\*, `AWQ_PY`, `LLAMA_SERVER`) are required config, not undocumented exports
  - instruction: Test: the same key set in the file and the environment resolves per the documented order
  - honesty: The config loader documents one precedence order and a test pins it, including the measure context value
- Knobs that pipeline.sh hard-codes to nvsh become domain or run config: `BUNDLE_REPO_PREFIX` (:500) and `hub_upload`.`ALLOWED_PREFIX` (`hub_upload.py`:51), `require_qwen_base` (:508-512), split always reading dev.json (split.py:664, no --corpus passed), bundle-dataset --issue 46 and nvsh LICENSE; train-scorer's mtime-based choice of scorer-train.json (:648-649) becomes an explicit choice by frozen sha256
  - instruction: grep -riE 'jetson-ai-lab|nvsh' `jev_factory` outside domains/nvsh (expect 0); train-scorer selects data by frozen sha256
  - honesty: No nvsh name, org prefix or Qwen-only check is hard-coded outside the nvsh domain module; the hub prefix, base model, licence and issue refs come from domain or run config
- The domain module is one declarative, loudly validated definition covering: operations (name, description, `read_only`, args of kind str|choice, order), groundable arg kinds with their world lookup and canonicalisation, the world/snapshot schema, escalate reasons as a single source (prompt description plus generator definition, deriving `escalate:<r>` and `decline:<r>`), one answer-policy sentence used by every generator and reviewer prompt, persona/explain topics/phrasing styles, probe paraphrases, the seed corpus {header, world, entries}, and hub prefix plus card text; today these are duplicated across nvsh/ops/table.py:15-133, ground.py:148-160, scorer.py:100-140, data/reasons.json, `draft_sources.py`:87-118 and :266-293, augment.py:507-601, measure.py:1788, `hub_upload.py`:51 and `release_bundle.py`:306-673
  - instruction: One validator test per rule
  - honesty: The domain module loader rejects an invalid module with a named error: duplicate operation, empty description, a reason missing its prompt description or generator definition, over 52 candidates including controls, or an unknown arg kind
- Served measurement does not depend on nvsh's tier runtime: measure.py:171-183 imports nvsh.tiers (bench, lfm, toolchat, router, base, runtime, `runtime_docker`), nvsh.config, nvsh.redact and nvsh.platform, so jev-factory needs its own decision contract (propose/explain/escalate/abstain), corpus loader, snapshot grounding runner and serve adapter behind the score seam (scorer.`score_next_token`)
  - instruction: The measure stage integration run on the Spark, from a venv without nvsh
  - honesty: Served measurement runs a jev-tool `Q4_K_M` through llama-server with no nvsh package installed
- Sequencing (operator decision): the factory verbs (jev init, `jev run <stage>`, jev status, jev decide) with dry-run by default and --apply ship before the first model is trained, and the first model is a jev scorer whose candidates are the grown jev CLI's verbs, including a real mutating half
  - instruction: Git history: the verb PRs merge before the draft stage's first --apply run
  - honesty: init, run, status and decide are implemented (in jev --help, with explain entries and tests) before any jev-tool training data is drafted
- jev-CLI training data is drafted and reviewed by nvsh's lobes teachers (generator Qwen3.6-35B-A3B; reviewers Gemma-4-26B-A4B and Qwen3.8-27B) through the OpenAI-compatible gateway, Apache-2.0 only, with reviewers returning structured JSON verdicts (operator decision)
  - instruction: The dataset bundle card lists the teachers and licences; a validator over the accepted file checks the role and model fields
  - honesty: Every drafted and reviewed jev-tool entry's provenance names the teacher model and its role, and all of them are Apache-2.0
- Correctness of the extracted factory is proven by the jev-tool model itself passing every evaluation the pipeline defines: selection-fold rule, fit-fold calibration and gate, permutation probe, missing-candidate slice, one final measurement on test and the sealed held-out set on the deployed `Q4_K_M`, and the release gate (operator decision)
  - instruction: The jev status run view shows each evaluation stage completed, with a record id
  - honesty: Every evaluation listed ran on the jev-tool model and its results are recorded, including misses, before the model is called done
- jev-factory owns the whole fine-tune pipeline (absorb, not cite): nvsh's scripts/lfm-finetune/ and evals/ move into jev-factory as its canonical implementation, and nvsh becomes a consumer (operator decision)
  - instruction: Provenance table coverage check (see c20); the nvsh#62 comment is posted
  - honesty: The imported nvsh code is jev-factory's canonical copy: all of scripts/lfm-finetune on the scorer path and evals/ are imported, and a handoff tells nvsh how to consume jev-factory
- The pipeline is built as a general factory for many models across other domains, not only the jev-tool model: nothing jev-CLI-specific lives outside that domain's module (operator decision)
  - instruction: grep for jev-CLI verb names outside the jev-tool domain module and its tests (expect 0)
  - honesty: Nothing jev-CLI-specific lives outside its domain module
- Docs that contradict the operator's decisions are corrected in the first PR: CLAUDE.md (and, in their own framing, AGENTS.override.md, AGENTS.colleague.md and QWEN.md) still say 'the first acceptance test is parity', 'cite (copy) rather than import' and list 6 nvsh-coupled scripts as 'already domain-generic'; issue #1's text says the same
  - instruction: grep the four files for 'parity' and 'already domain-generic' and review the matches
  - honesty: After the first PR, CLAUDE.md, AGENTS.override.md, AGENTS.colleague.md and QWEN.md say absorb (not cite), drop parity as the first acceptance test, and list no nvsh-coupled script as generic
- nvsh is told it is becoming a consumer (on nvsh#62, via the communicate skill), with the absorbed scope listed (scripts/lfm-finetune, evals/) and the seam nvsh must then implement as a jev-factory domain module
  - instruction: gh issue view 62 -R agentculture/nvsh --comments
  - honesty: nvsh#62 carries a comment from jev-factory that states the absorb decision and the domain-module interface nvsh will implement
- The reference for the jev-tool model's bars is stock Qwen3.5-0.8B plus pre-registered minimum requirements: for each metric the bar is the stricter of the stock-derived value and the minimum (operator decision). The minimums are nvsh's: 0 wrong mutating on test and held-out, ECE <= 0.10 on the deployed build, pooled permutation change <= 2.3%, missing-candidate escalation >= 80%, and right proposals >= stock - 5 pts
  - instruction: Schema test; the stock baseline measurement's record id is cited in the file
  - honesty: The pre-registration file records, for each bar, the stock Qwen3.5-0.8B value, the minimum, and the stricter one as the bar, before training
- Every step of building the jev-tool model is imported (operator decision): environment and run config, pre-registration, seed, teachers and reviewer pilot, sealed held-out set, eval pool with split and folds, grounding snapshot and baseline, augment and targeted recipes, assemble and freeze, scorer training with merge verification, selection (logprob calibration: temperature plus optional per-label vector, gate fit, permutation probe), quantize, one-round heal, recalibration on the deployed quant, one final measurement, edge check, bundle, private upload, and the release gate
  - instruction: Compare jev run --help with the step list; jev status for the jev-tool run
  - honesty: Each listed step exists as a jev run stage or a documented sub-step and ran for the jev-tool model
- Each jev-tool bundle records the sha256 of the generated jev-CLI domain module (the CLI surface) it was trained on, and using a bundle whose surface hash differs from the running jev's argparse surface is flagged, because the CLI keeps growing (c36) and the domain is generated from argparse (c17) (challenge / adjacent-systems)
  - honesty: A test builds a bundle, adds a verb to a fixture parser, and checks that the surface-hash mismatch is reported
- Long stages (teacher drafting and review take a day or more; scorer training about 26 min) run detached from the invoking shell with a PID and done marker, resume at item granularity from a response cache, and report progress through jev status; issue #4 'Shell and automation traps' records that tool-backgrounded jobs die at the tool timeout, and s15 found only augment.py resumes at item level (challenge / failure-modes)
  - honesty: A stage killed mid-run and restarted re-sends 0 cached teacher requests and resumes from the next item; jev status shows items done out of total
- GPU stages (train, heal, quantize, served measure) check for other GPU compute processes before starting, and refuse unless the operator overrides; training runs under the memory-floor watchdog, because GB10 does not charge GPU allocations to the cgroup cap. A probe on 2026-09-30 found a resident vLLM EngineCore holding about 32 GB alongside another process (challenge / operations: nvidia-smi --query-compute-apps)
  - honesty: A test with a stubbed nvidia-smi that lists a foreign GPU process makes train, quantize and measure refuse without the override flag
- A run's work dir holds a lock, so a second concurrent jev run against the same run (for example the mesh agent and the operator at once) is refused; nvsh#58 records a stop/start race in `serve_for_measure.sh` on one port (challenge / concurrency)
  - honesty: A second jev run on a locked work dir exits non-zero with the lock holder's PID, and a stale lock from a dead PID is reported, not silently taken
- Run work roots (drafts, splits, the sealed held-out set, predictions, bundles) live outside any git worktree, and stages refuse a work root inside one, as nvsh's evals writers already do (s6), so sealed or private data can never be committed (challenge / security + data-loss)
  - honesty: A test points a stage at a work root inside a git worktree (detected with rev-parse) and expects a refusal
- Secrets reach stages only by environment variable name (the hub token and the teacher gateway key), are never printed or logged, and a test pins that the run-config examples name no host and no secret, as nvsh's `test_neither_env_example_names_a_host_or_a_secret` does (challenge / security)
  - honesty: A secret's value never appears in stdout, stderr, logs or manifests; a test injects a canary token and greps every output
- Imported files keep their Apache-2.0 attribution, and the provenance record (c20) states their Apache-2.0 origin in nvsh; both repos are Apache-2.0, per the LICENSE files and pyproject license fields (challenge / migration: licence probe)
  - honesty: Every imported file keeps its licence header if it had one, and the provenance doc names nvsh's Apache-2.0 LICENSE

## Honesty conditions

- At the end, the jev-tool model was built by jev run stages in jev-factory, with no step run from nvsh's scripts
- No jev-factory code path prints, logs or returns the text of a test or sealed held-out entry, and measuring either twice without a deviation record is refused
- The upload verb creates and forces private, fetches back and compares every sha256 and the remote file list, refuses symlinks, and never changes visibility to public
- No code path sends training or eval data to codex, agy or kiro; the teacher config accepts only models listed with an Apache-2.0 licence
- scan-secrets.py and portability-lint.sh pass on the repo after import
- No root AGENTS.md exists and the harness smoke config stage passes after every PR
- No stage reads nvsh's #53 or r3b artifacts unless a jev-tool evaluation has failed and a deviation record is open
- Each named audience can do its job from jev learn, jev explain and the spec without reading nvsh's docs
- Each part of the before-state statement is backed by a recorded scope entry (s12, s13, s15, s16)
- The after-state is observable: jev run lists every stage, jev status reads the stage manifests, and the jev-tool bundle exists privately with the three required files
- Each failure named in the why-it-matters claim has a matching nvsh record, and the factory has a check or stage that prevents it
- The bars are read from the hashed pre-registration file, not re-derived after measurement, and the final measurement ran exactly once
- The gate report is produced by jev-factory code, and its rows include the jev-tool candidate
- The grep and the no-ML-import test run in CI on every PR
- The toy domain shares every stage implementation with the jev-tool domain; only its module differs
- No jev-factory PR, script or stage writes to the nvsh repo; the nvsh-side change is requested on nvsh#62 only
- No code path calls a write verb with --apply because of a model's output; a test confirms that decide and propose paths only print a proposal

## Success signals

- The jev-tool model's deployed `Q4_K_M` build, measured once, meets every pre-registered bar on both test and the sealed held-out set: 0 wrong mutating, ECE <= 0.10, pooled permutation change <= 2.3% (>= 10 permutations per entry), missing-candidate escalation >= 80%, and right proposals >= the stricter of (stock Qwen3.5-0.8B - 5 pts) and the pre-registered minimum
  - instruction: Read the final-measurement record and compare each metric (with n and bootstrap CI) against the hashed pre-registration file; any miss is reported as a miss
- The jev-tool bundle's release-gate run records 0 wrong mutating in both the model-only and model+harness rows, with the gate run from jev-factory's own evals code and no nvsh checkout present
  - instruction: Run the gate from jev-factory with the ../nvsh checkout moved away; check the report's rows
- jev-factory imports nothing from nvsh: 0 matches for 'import nvsh' or 'from nvsh' under `jev_factory`/, and the base install still has dependencies = \[\] with 0 heavy ML imports at module import time
  - instruction: grep -rE '^(from|import) nvsh' `jev_factory` (expect 0); a test imports every `jev_factory` module in a venv without torch, transformers or unsloth installed
- A second tiny domain in tests (at least 3 operations, at least 1 mutating) runs the CPU-only stages end to end — validate, split, assemble, calibrate, gate sweep, select, decide — through the same code as the jev-tool domain, proving the factory is not jev-specific
  - instruction: uv run pytest tests/ -k `toy_domain` runs green in CI without a GPU

## Scope / boundaries

- Sealed held-out and test sets are touched once, in the final run only; the lead never reads their text, only counts and sha256 (issue #4 stage 4 and 'Process discipline'); migrating nvsh's sealed set copies it only as a hash-verified file
  - instruction: Tests: sealed/test loaders expose ids, counts and hashes only; a second final measure without a deviation id exits non-zero
- Publishing is private-first: `hub_upload` creates private, fetches back and verifies every sha256 and the file list, and refuses symlinks; going public is a per-repo operator decision and a hard stop (issue #4 stage 16)
  - instruction: Tests with a fake hub client cover create-private, fetch-back mismatch, symlink refusal and no public call; --apply is required
- Code reviewers (codex, agy, kiro) never generate or correct training data; teachers are Apache-2.0 models so the data can be published (issue #4 'Reviewers for code')
  - instruction: The teacher-models config validator rejects a non-Apache licence entry; review the imported draft/augment code for any other model endpoint
- Imported code and committed docs carry no `/home/<user>` paths, hostnames, IPs or serving-model locations, and JSON configs carry no non-localhost URLs: scan-secrets.py check 2 (:97-110) and cicd portability-lint.sh flag them; use $WORK/$DRAFTS/$SEALED placeholders as nvsh's guides do
  - instruction: Run both in CI and pre-PR
- No root AGENTS.md is added and the four harness prompt files (CLAUDE.md, AGENTS.override.md + .pi/SYSTEM.md, AGENTS.colleague.md, QWEN.md) are each updated in their own framing when project facts change (tests/`test_harness_smoke.py`:200, docs/harness-invocations.yaml:35,83)
  - instruction: harness-smoke --stage config in CI
- No new r3b work: the nvsh r3b/#53 frozen data is not replayed, reproduced or retrained unless the jev-tool model fails an evaluation; only then is it replayed through the extracted stages, to find whether the fault is in the pipeline or in the jev-CLI data or model, and used to fix it (operator decision)
  - instruction: The r3b diagnostic path exists only as a documented procedure, gated on a failed-evaluation record id
- jev-factory never edits nvsh: nvsh retires its own copy (scripts/lfm-finetune, evals/, 37 `test_lfm_finetune_`\*.py files, and the CI lint of scripts and evals at nvsh .github/workflows/tests.yml:61-89) on its own schedule; until then two copies exist and jev-factory's is canonical (challenge / adjacent-systems)
- A jev-tool model's proposal is never executed with --apply on its own: when the model picks a mutating verb (train, upload, publish, retrain), a human still issues --apply, per issue #1 Layer 3 'the model proposes, and --apply plus the hard stops still apply' (challenge / security)

## Non-goals

- The CLM-style Track C experiment (issue #3) is out of scope: #3's own Timing section says it starts only after Tool-Jev is migrated out of nvsh, so this work is the migration #3 waits on (issue #4 title: 'unblocks #3'), not the experiment
- Track A (generative tool calling) is not a jev-factory product: issue #4 'Known weak spots' keeps it a contrast arm (12% missing-candidate escalation, 4 wrong mutating in gate run 8b59d0afa150), and CLAUDE.md says Track A is explicitly not jev-like
- The Layer 3 self-hosted decider model is not built in this scope: issue #1 says it comes after the rule-based jev decide loop has produced real records to learn from
- For the first jev-CLI model, the gate port defers Track A replay (`track_a_loop.py` via nvsh LfmTier), the judge panel and explain rubric, the Discord alert and docker drivers, batch APIs, and the execution-based layer (nvsh#70); a raw policy plus one provider (or the fake provider in tests) is enough to run it
- nvsh scorer-r3b parity is not an acceptance gate in this scope (operator decision, overriding issue #1's 'first acceptance test is parity' and CLAUDE.md's matching line); the jev-CLI model is the first and only acceptance target
- Code used only by Track A is not imported: `track_a_calibration.py`, `jetson_skills.py`, `skills_dataset.py`, `measure_skills.py`, the release gate's Track A replay (`track_a_loop.py` via LfmTier), and the pipeline's skills stages. Shared code on the scorer path is imported: train.py for merging and heal, and the text-similarity helpers `leakage_check.py`:38 takes from `jetson_skills`, which move into a jev-factory module

## Assumptions

- The operator's 'issue #3' means issue #4 ('Extraction guide: replicate nvsh's full Tool-Jev process'): #3 is the CLM experiment and #4 is the stage-by-stage extraction brief written by nvsh's run lead
- The jev-CLI domain's candidate table is small and almost entirely read-only until the factory verbs exist: today jev exposes whoami, learn, `explain <path>`, overview, doctor, cli, cli overview (cli/`__init__.py`:64-95), all read-only, so the mutating gate and the '0 wrong mutating' bar are vacuous until init, `run <stage>` and decide with --apply land
- nvsh's tests for scripts/lfm-finetune (37 `test_lfm_finetune_`\*.py files, about 24k lines, including tests/`test_lfm_finetune_pipeline.py` at 3,074 lines with a stubbed `_Pipeline` fixture) are the pinned behaviour to port with the scripts, re-fixtured onto a tiny test domain rather than nvsh's table
- The jev-CLI domain's groundable arguments come from the CLI's own catalog, with no machine lookup: `explain <path>` grounds against explain/catalog.py keys, and `run <stage>` against the stage list. Its world snapshot is the CLI introspection itself, so grounding is deterministic and offline
- jev run is a Python stage engine that replaces nvsh's pipeline.sh; pipeline.sh itself is not imported, and its stage contract and guards are re-expressed as stages and tests (implied by c29 and c30 but never stated) (challenge / unstated-assumptions)

## Scope exploration

- `s1` — `issue #3 (CLM experiment)`: \#3 is a downstream experiment gated on the migration ('Do this after Tool-Jev has been migrated out of nvsh'); its needs — same safety/calibration harness, frozen-backbone heads, candidate-identity-by-description, unseen-action eval — constrain the extraction's seams but its tracks are not built here
  - seeds: `c2`, `c3`
- `s2` — `issue #4 (extraction guide)`: \#4 is the stage-by-stage extraction brief (stages 0-17, parity targets, operational knowledge, suggested extraction order 1-7) and names itself the input for the migration #3 waits on; it is the working brief for this idea
  - seeds: `c4`, `c7`, `c8`, `c9`
- `s3` — `issue #4 stages 4, 14, 16 and 'Process discipline'`: sealed/test sets touched once and never read by the lead; private-first upload with hash-verified fetch-back; going public, sealed reruns, rule changes and paid-model replacement are operator gates
  - seeds: `c5`, `c6`
- `s4` — `issue #4 'Reviewers for code' and 'Known weak spots'`: codex/agy/kiro review code only, never data; Track A stays a contrast arm
  - seeds: `c12`, `c10`
- `s5` — `issue #1 (build brief) 'Suggested first milestone' and Layer 3`: first milestone is domain-module contract + nvsh module, cited generic scripts, jev run select over frozen r3b predictions, rule-based jev decide reproducing the issue-53 verdict; the self-hosted decider comes after rule-based records exist
  - seeds: `c11`
- `s6` — `nvsh evals/ + docs/deepeval-gate.md + pyproject evals group`: about 27k lines incl. ~12k tests; only dep is deepeval==4.2.6 (pyproject.toml:75-77); generic core is ledger/runstate/manifest/providers/cases/trace/report; nvsh couplings at cases.py:49, request.py:64-67, run.py:88, judge.py:115, providers/base.py:50, plus path-loading metrics.py/gate.py/`calibration_fit.py`/scorer.py/measure.py; input contract = metrics.Prediction JSONL per subject + test/test-mc splits + policy json; first run cost $35.78 across 5 provider keys
  - seeds: `c13`, `c14`
- `s7` — `nvsh#71 (gate follow-ups)`: item 2: worktree guard trips on an empty .git dir in /tmp; item 1: Track A predictions lack inspections/explanations; item 4: request contract ambiguity for read-only requests (propose vs explain) — relevant to a CLI-verb domain whose many verbs are read-only
  - seeds: `c15`
- `s8` — `jev_factory/cli (uv run jev --help, cli overview --json, learn --json)`: 7 callable entries (whoami, learn, explain, overview, doctor, cli, cli overview), all read-only, zero mutating; introspection is free-text 'verb — description' strings with no `read_only` or arg schema; three hand-kept verb lists can drift; parser description and catalog still say 'clonable template'
  - seeds: `c16`, `c17`, `c23`
- `s9` — `pyproject.toml + .github/workflows/tests.yml, publish.yml`: dependencies=\[\]; no optional extras yet; lint (black/isort/flake8/bandit) and coverage (`fail_under` 60) cover only `jev_factory` and tests; publish.yml triggers on `jev_factory`/\*\* so package code ships to PyPI; version-check forces a bump per PR
  - seeds: `c18`, `c19`
- `s10` — `docs/skill-sources.md`: provenance table (upstream relative path, origin, notes, last-synced with version) plus re-sync diff procedure and local-divergence sections — a direct template for a cited-nvsh-scripts ledger
  - seeds: `c20`
- `s11` — `scripts/scan-secrets.py, cicd/portability-lint.sh, scripts/harness-smoke.py`: scan-secrets flags token-like literals everywhere and non-localhost URLs in JSON only; portability-lint flags `/home/<user>` and `~/.dotfile` paths; harness-smoke config stage pins no root AGENTS.md and the four prompt files
  - seeds: `c21`, `c22`
- `s12` — `nvsh scripts/lfm-finetune/*.py (coupling survey, verified gate.py:72-73, metrics.py:127-128, train_scorer.py:73-74, scan_bundle.py:34, leakage_check.py:38)`: 6 of the 13 'generic' scripts are nvsh-coupled directly or through `_sibling` path-loading; no script imports numpy; the readout alphabet and token variants live only in nvsh-coupled scorer.py; issue #62's coupling table has drifted
  - seeds: `c24`, `c25`, `c26`
- `s13` — `nvsh docs/tool-jev-calibration-rule.md + qwen-tool-jev-calibration.md Decision path + scorer-finetune-playbook.md`: the rule is 71 lines of prose (committed b1c6cb6) with no machine-readable form; D1-D49 are prose bullets with inconsistent Evidence/Choice/decider/record fields; frozen hashes appear only as 16-hex prefixes (D37, D44, D45, D47); the playbook's 'already generic' list (l.148-151) repeats the stale claim
  - seeds: `c27`, `c9`, `c24`
- `s14` — `operator training work area: frozen #53 work tree (listed and hashed only; no sealed or test text read)`: splits, folds, frozen data (r1, r2, r3b/d7), per-candidate val and val-mc predictions, calib, rule/, probe/, final/ and the r3b `Q4_K_M` GGUF are all present; 18 hash prefixes match the docs; D47's 'params' matches the temp-only params file, and two calib files are uncited
  - seeds: `c28` (rejected)
- `s15` — `nvsh scripts/lfm-finetune/pipeline.sh + pipeline*.env.example + requirements-train.txt`: 22 stages; no stage-state file; env-file-over-export precedence with a `MEASURE_CTX` exception (:178-182); hard-coded qwen/jetson-ai-lab prefix (:500, :508); split never passes --corpus; train-scorer picks its data by mtime (:648); training venv pins torch 2.12.1+cu130/unsloth 2026.9.9; the AWQ venv has no requirements file; the causal-LM assumption sits in the train, train-scorer, assemble render, measure serve, quantize and bundle stages
  - seeds: `c29`, `c30`, `c31`, `c3`
- `s16` — `nvsh/ops (table.py, _model.py, ground.py), scorer.py, data/*.json, draft_sources/draft_heldout/augment/targeted_augment prompts, dev.json (structure and counts only)`: 16 operations (3 mutating); 2 groundable arg kinds; escalate reasons are stated in 6 places with no consistency assertion; no single answer-policy sentence exists (policy is restated in 5+ prompts); seed 431 entries (212 op / 116 explain / 103 escalate, 8 decline classes); persona strings are Jetson/DGX-specific
  - seeds: `c32`, `c35`
- `s17` — `nvsh measure.py:171-183 + nvsh/tiers (bench, lfm)`: served measurement is bound to the nvsh tier runtime and LfmTier; the scorer path goes through the scorer.`score_next_token` seam, which a non-nvsh measure harness can implement
  - seeds: `c33`
- `s18` — `nvsh tests/test_lfm_finetune_*.py`: 37 files, about 24k lines; the pipeline contract is pinned via a stubbed `_Pipeline` fixture (upload/FINAL guards, heal/quantize refusals, env-example hygiene, capped.sh watchdog, measure final/heldout allowlists, `MEASURE_MAX_LOGPROBS` == `READOUT_TOP`)
  - seeds: `c34`
- `s19` — `operator direction (2026-09-30)`: jev-factory absorbs the pipeline for many models; correctness is proven by the jev-tool model passing every evaluation; r3b is replayed only on failure
  - seeds: `c39`, `c40`, `c41`, `c42`
- `s20` — `operator direction (2026-09-30, second)`: the bar reference is stock Qwen3.5-0.8B combined with minimums, stricter per metric; import scope is every step of the jev-tool build path including healing, logprob calibration and evaluations
  - seeds: `c45`, `c46`, `c47`
- `s21` — `challenge pass / depth`: rigorous: migration (absorbing the pipeline), hardware (GB10 GPU jobs), hard-to-reverse ops (hub upload, once-only sealed measurement), data-loss surface (sealed held-out set)
- `s22` — `challenge pass / adjacent-systems lens: nvsh CI, nvsh/nvsh package, bundle consumers`: nvsh/nvsh never imports scripts/lfm-finetune, which only nvsh's tests and CI lint touch (tests.yml:61-89); the bundle's surface depends on the growing jev CLI
  - seeds: `c56`, `c57`
- `s23` — `challenge pass / unstated-assumptions lens: frame claims c29, c30, c38, c45, c50`: the pipeline.sh replacement was implied, never stated (c58); there is no absolute accuracy minimum (q6 on c45); seed authorship (q7 on c38) and the runtime consumer (q5 on c50) are unstated. CORRECTION: this entry's seed list wrongly names q2, q3 and q4, which were cited before the real ids existed; the correct questions are q5, q6 and q7
  - seeds: `c58`, `q2` (question, resolved), `q3` (question, resolved), `q4` (question, resolved)
- `s24` — `challenge pass / failure-modes + operations lens: issue #4 shell traps, s15, nvidia-smi probe`: detached long jobs, item-level resume, and a GPU residency check are missing from the spec; the probe found about 34 GB of resident GPU processes
  - seeds: `c59`, `c60`
- `s25` — `challenge pass / concurrency lens: nvsh#58, run work dir`: concurrent runs on one work dir are unguarded
  - seeds: `c61`
- `s26` — `challenge pass / security lens: issue #1 Layer 3, nvsh evals writers, env examples`: model proposals need a human --apply; work roots must sit outside a worktree; secrets pass by variable name only
  - seeds: `c62`, `c63`, `c64`
- `s27` — `challenge pass / migration lens: LICENSE and pyproject of both repos`: both Apache-2.0, so no licence conflict; attribution is kept in provenance
  - seeds: `c65`
- `s28` — `challenge pass / observability + rollback lens: decision records, stage manifests, private upload`: clean pass: stage manifests (c29), decision records citing metric hashes (c9) and private-only upload (c6) cover detection and containment; no rollback path exists for a bad private bundle beyond not promoting it, which is acceptable while public is a human hard stop
- `s29` — `challenge pass / cheap probes`: run read-only: licence, teacher gateway (HTTP 401: up, needs a key), GPU residency, llama-server not on PATH (the toolchain lives in the operator's training area); not probed: reviewer throughput and the Qwen3.5-4B held-out drafter

## Decisions

- jev gains a 'jev ask' verb that loads the jev-tool bundle and, for a natural-language request, proposes one jev verb: it applies the bundle's calibration.json and gate.json and grounds arguments deterministically, and it only prints the proposal, never running --apply itself (operator decision on q5; see c62)
- The pre-registered absolute minimum for right proposals is 95% (r3b's test side was 79/83 = 95.2%); the bar is the stricter of this and stock Qwen3.5-0.8B minus 5 pts (operator decision on q6; refines c45)
- The agent drafts the one-sentence answer policy and an initial jev-tool seed corpus from the domain module; the operator reviews and approves the policy and a sample before any teacher run, and the seed stays train-only (operator decision on q7)

## Hard questions

- Absorbing the whole pipeline for many future models includes nvsh's Track A (generative tool calling: train.py targets, `track_a_calibration.py`, the gate's Track A replay). Is Track A imported as a supported model family, or left behind as nvsh-only history? (resolved: Only steps that build the jev-tool (scorer) model are imported (c46); Track A-only code stays in nvsh (c47))
- With parity dropped, what proves the extracted calibrate/gate-sweep/select/decide stages still behave like nvsh's? Candidates: the ported nvsh unit tests (c34) alone, or a cheap no-GPU replay of the local frozen #53 predictions kept as a regression fixture rather than an acceptance gate (resolved: Proven by the jev-tool model passing all evaluations (c39); the frozen r3b replay is only a failure-time diagnostic (c42), not a regression gate)
- Who writes the jev-tool seed corpus? nvsh's seed was 431 hand-written entries (dev.json), and issue #4 stage 2 says the one-sentence answer policy must be written first. Is it operator-written, agent-drafted for operator review, or teacher-drafted from the domain module? (challenge / unstated-assumptions) (resolved: Agent drafts the policy and seed; the operator reviews them before any teacher run (c68))
- The accuracy bar 'right proposals >= reference - 5 pts' needs a reference model. nvsh used scorer-b1. The jev-CLI domain has no prior model, so is the reference stock Qwen3.5-0.8B, the first trained candidate, or a pre-registered absolute floor? (resolved: Stock Qwen3.5-0.8B plus pre-registered minimums, taking the stricter of the two per metric (c45))
- There is no absolute accuracy minimum: the only accuracy bar is 'right proposals >= stock - 5 pts', and stock Qwen3.5-0.8B scored 6/83 in nvsh's gate run 8b59d0afa150, so the stock-derived bar could be near 0%. What absolute right-proposal minimum should be pre-registered? (challenge / missing-counter-evidence) (resolved: Absolute right-proposal minimum 95% (c67))
- Who uses the jev-tool model at runtime? The frame lists only build steps (c46), with no inference path. Is the deliverable only the private bundle, or does jev itself load it (for example a 'jev ask' verb that proposes a verb, with the gate and grounding applied)? (challenge / overlooked-actors) (resolved: Add a jev ask verb (c66))
- x (resolved: Void: created by mistake during the challenge pass (lapse l1))

## Open parks

- [unknown_nonblocking] Hub prefix: bundles under a jev-factory prefix or per-domain prefixes? `hub_upload`.`ALLOWED_PREFIX` is hard-coded to jetson-ai-lab/qwen3.5-0.8b-nvsh- today (issue #1 parked question)
- [unknown_nonblocking] Decider autonomy: how much unattended autonomy the Layer 3 decider eventually gets (issue #1 parked question; out of this scope)

## Resolved vagueness

- [unknown_nonblocking] Absorb vs cite: does jev-factory absorb nvsh's scripts/lfm-finetune (nvsh becomes a consumer) or keep citing it with nvsh as owner? (issue #1 parked question; coordinate on nvsh#62) — resolved: Absorb: jev-factory takes the whole pipeline and nvsh becomes a consumer
