# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What jev-factory is

`jev-factory` is an AgentCulture mesh agent (`culture.yaml`: suffix
`jev-factory`, `backend: claude`, so this file is its mesh resident prompt).
It turns the process [`nvsh`](https://github.com/agentculture/nvsh) used to
build its **Tool-Jev** model, a calibrated candidate scorer on
`Qwen/Qwen3.5-0.8B`, into a reusable factory for **jev-like** models in any
domain. It has three layers, and each depends on the one before it:

1. **Factory.** nvsh's staged fine-tune pipeline, made domain-generic through
   a declarative *domain module*.
2. **Decision surface.** Every between-runs decision ("more epochs?",
   "targeted augment class X?", "heal the quant?", "ship?") becomes an
   explicit, rule-driven, recorded step with named knobs: `jev decide <run>`.
3. **Self-hosting.** The factory builds a jev-like *decider* model whose
   candidates are those verdicts. It then calls that model (`jev decide
   --model <bundle>`) to drive new jev builds, with the rule as baseline and
   fallback.

The brief is **[issue #1](https://github.com/agentculture/jev-factory/issues/1)**
(`gh issue view 1`). It is a brief, not a spec, and where it disagrees with
the confirmed spec, the spec wins. The operator's decisions are in
`docs/specs/2026-09-30-extract-jev-process-jev-cli-first-model.md` (read
it before designing anything). The intended flow is `/scope` → `/think` →
`/spec-to-plan` before code.

**Operator decisions that override older text** (spec claims c37-c47, c66):

- jev-factory **absorbs** nvsh's scorer-path pipeline (`scripts/lfm-finetune/`
  and `evals/`) as its canonical implementation, rather than citing it.
  nvsh becomes a consumer, on its own schedule. jev-factory never edits nvsh.
- The **first and only acceptance target is the jev-tool model**: a scorer
  whose candidates are the `jev` CLI's own verbs. Correctness of the
  extracted factory is proven by that model passing every evaluation the
  pipeline defines. nvsh `scorer-r3b` parity is **not** an acceptance gate.
- nvsh's r3b / issue-53 frozen data is replayed only if a jev-tool
  evaluation fails and a deviation record is open, and then only as a
  diagnostic.
- The factory verbs (`jev init`, `jev run`, `jev status`, `jev decide`) ship
  before the first model is trained.

## Current state: the factory exists; the jev-tool model is not yet trained

The factory code is built and tested: the domain-module contract
(`jev_factory/domain/model.py`, example `tests/fixtures/toy_domain`), the 22
`jev run <stage>` stages with per-stage manifests, `jev status`, `jev decide`
(pre-registered rule, append-only records), `jev ask`, the jev-CLI domain
(`jev_factory/domains/jev_cli`) and the release gate. **No jev-tool model has
been trained yet**: its bundle, calibration and gate do not exist, so never
claim one as shipped. Check a verb with `uv run jev --help` and
`uv run jev learn --json`. `docs/lessons-encoded.md` maps each nvsh failure to
the test that prevents it. The spec and plan are under `docs/specs` and
`docs/plans`. Some CLI strings may still describe a "clonable template";
rewrite them when next touched.

Names: the **command is `jev`**
(`[project.scripts] jev = "jev_factory.cli:main"`), the import package is
`jev_factory`, and the PyPI distribution is `jev-factory` (0.9.0 is
published; the bare `jev` name on PyPI belongs to someone else). Argparse's
`prog` string says `jev-factory`, but no `jev-factory` binary is installed.

## What "jev-like" means (from nvsh; don't redefine it silently)

This is the OpenJev-style **scorer**, nvsh's *Track B*. nvsh's Track A
(generative tool calling) is explicitly **not** jev-like.

- **Input:** one request plus N candidate actions, rendered as lettered lines
  `"<L>) <name>: <description>"`, using letters A–Z then a–z (at most 52).
  Two controls are always offered: `explain` (answer in words) and
  `escalate` (hand off). `escalate` may be split into named reasons
  (`escalate:<reason>`).
- **Inference:** one forward pass and one generated token. Each letter's
  probability is summed over its token variants, then normalised over the
  letters actually offered.
- **Calibration:** a temperature, plus an optional per-label vector that is
  kept only if it lowers ECE on the selection fold.
- **Gate:** separate floor, margin and max-entropy thresholds for
  **read-only** and **mutating** actions. The outcome is
  propose / explain / escalate / `abstain_uncertain`.
- **Arguments are grounded deterministically**, outside the model. The model
  never generates argument values.
- **Robustness bar:** the choice survives reordering, re-lettering,
  subsetting and paraphrasing of the candidates (`permutation_probe.py`;
  OpenJev's bar is 2.3% pooled answer change).

### Model families: keep the backbone pluggable

Qwen3.5-0.8B (and `LiquidAI/LFM2.5-350M`) are the only proven bases. The
"one token over lettered candidates" readout is how a **causal decoder** is
turned into a scorer. It is not the definition of jev-like.
[nvsh#67](https://github.com/agentculture/nvsh/issues/67) plans
**GLiNER2.5-Decide** (~340M, an encoder that scores explicit candidates
natively) as a head-to-head alternative. The factory is expected to support
such model families later, so:

- **The invariant contract is:** a bounded candidate set, then a calibrated
  distribution over it, then the deterministic read-only/mutating gate, then
  grounding outside the model. The letter/token readout, prompt rendering,
  trainer and quantizer are **per-backbone adapters**. Don't hard-code
  "causal LM + letter token" into stages that only need a distribution.
- A new family must go through the **same** splits, fit/selection folds,
  pre-registered rule and bars as Qwen. Raw scores, raw probabilities,
  fitted calibration, calibrated probabilities and the final gate decision
  are each recorded. Softmax confidence is never treated as calibrated.
  Serialization may differ per family but must not give one model
  information the other lacks, and any family-specific preprocessing is
  recorded in the run's artifacts.
- The bars for the jev-tool model are pre-registered: for each metric, the
  stricter of the stock Qwen3.5-0.8B-derived value and a minimum. The
  minimums are nvsh's (0 wrong mutating on test and held-out, ECE <= 0.10 on
  the deployed build, pooled permutation change <= 2.3%, missing-candidate
  escalation >= 80%) plus a **95% right-proposal minimum**, with right
  proposals also >= stock - 5 pts.
- nvsh's `scorer-r3b` Q4_K_M (nvsh#63) is **reference context only**, not a
  baseline to reproduce. If you read its numbers, compare like with like,
  because nvsh reports two sets for it:
  - **Selection fold** (nvsh D47): 79/84 right proposals, 0 wrong mutating,
    abstain recall 94.0%, 1/120 false-positive tool calls, ECE 0.019 after
    temperature.
  - **Final test side, measured once** (D48): 79/83 right, ECE 0.016.
    On the sealed held-out set: 49/60 right, ECE 0.044, and 76.7%
    missing-candidate escalation, which missed the 80% bar.
- An accuracy gain with worse mutation safety or calibration counts as a
  regression.

## Invariants: enforced in code, never knobs

- The sealed held-out set and the test set are touched **once**, in the final
  run only.
- The gate is fit on the **fit fold only** and never looks at the selection
  fold (nvsh D47).
- Protected-side leakage (exact and near-duplicate) **fails closed**.
- The bars and the selection rule are **pre-registered** before the first
  training command. Any later change, rerun or out-of-plan run is a recorded
  deviation (`/deviate`), never an edit.
- Publishing is **private-first**. Going public is always a human decision
  and a hard stop.
- Every write verb (train, quantize, upload, publish, retrain) is
  **dry-run by default**, and `--apply` commits.
- Decision records are append-only and cite the metrics they used. A human
  override is recorded as a deviation, never folded in.
- Self-hosting guards against circularity: a decider version never picks its
  own successor without a human gate, and it is never trained on verdicts it
  made itself unless a human or the rule confirmed them. The learned decider
  may drive the loop only after it beats the rule-based `jev decide` on the
  sealed real-decision set with **0 wrong mutating verdicts**. Below its gate
  it abstains and falls back to the rule or the human.

## Source material in `../nvsh`

nvsh is the origin. jev-factory **absorbs** its scorer-path pipeline: the
code is imported once, and after import jev-factory owns it (there is no
ongoing re-sync). Read nvsh; **don't fork its docs**. Coordinate with it on
[nvsh#62](https://github.com/agentculture/nvsh/issues/62), the domain-module
seam: nvsh will implement that interface as a jev-factory domain module
when it retires its own copy. The key files (paths are relative to
`../nvsh`):

- `docs/scorer-finetune-playbook.md`: the end-to-end recipe (steps 0–12).
  Its **Step 0 port checklist** lists exactly what is nvsh-specific, and its
  health-check table and "Lessons that transfer" are the operational
  knowledge the factory should encode.
- `docs/qwen-tool-jev-finetune.md` (issue 46), which covers the two tracks,
  the sweep and the overfit/underfit signature, and
  `docs/qwen-tool-jev-calibration.md` (issue 53).
- `docs/tool-jev-calibration-rule.md`: the pre-registered lexicographic
  selection rule, which is the model for `jev decide`'s "ship candidate" path.
- `docs/benchmarks/2026-09-2{4,5}-*`, `docs/deliveries/*jev*` and
  `docs/specs/*jev*` hold the results and the D1–D49 decision log, which is
  the seed corpus for the Layer 3 decider.
- `docs/specs/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md` and
  `evals/` cover the release gate, which is still in progress in nvsh.
- `scripts/lfm-finetune/`: `pipeline.sh` (resumable stages under `--env
  <file>`; `pipeline-qwen.env.example` shows every knob) and every script.

**Import scope.** Absorbed: every step on the path that builds a jev-tool
(scorer) model, from run config through the release gate, plus the `evals/`
generic core. Not absorbed: code used only by Track A (generative tool
calling): `track_a_calibration.py`, `jetson_skills.py`, `skills_dataset.py`,
`measure_skills.py`, the release gate's Track A replay and the pipeline's
skills stages. `pipeline.sh` itself is replaced by the `jev run` Python stage
engine, not imported. Each imported file goes in a one-time provenance table
(nvsh path, commit `9debdc6`, adaptations), and keeps its Apache-2.0
attribution.

**Do not trust nvsh's old "already domain-generic" list** (issue #1, the
playbook and earlier versions of this file). Verify every script's imports
as you absorb it. These are **nvsh-coupled**, not generic, and must be
rewired to the domain-module seam on the way in: `gate.py` and `metrics.py` (import `nvsh.ops.table` and
`nvsh.tiers.bench`), `train_scorer.py` (imports `nvsh.tiers.lfm`/`bench`),
`scan_bundle.py` (imports `nvsh.redact`), `calibration_fit.py` and
`sweep_gate.py` (path-load `metrics`/`gate`), and `leakage_check.py` (takes
text-similarity helpers from Track-A-only `jetson_skills`). After import, no
module imports nvsh and none path-loads a sibling.

**nvsh-specific, so this becomes the domain module:** the operation table
(`nvsh/ops/table.py`: `Operation(name, description, read_only, args)` plus the
`get`/`names`/`validate` API), grounding (`nvsh/ops/ground.py`, the world
snapshot in `measure.py`/`nvsh/tiers/bench.py`, `scorer.ground_arguments`),
the prompt strings in scorer/augment/draft/targeted-augment, escalation
reasons (three places that must agree: `scorer.REASON_CANDIDATES`,
`data/reasons.json`, `draft_sources.REASON_DEFINITIONS`), probe paraphrases
(`data/paraphrases.json`), the seed corpus (`dev.json`), the hub prefix and
card text, and the test fixtures. Served measurement also runs through
nvsh's tier runtime (`measure.py` imports `nvsh.tiers`), so jev-factory needs
its own decision contract, corpus loader, grounding runner and serve adapter
behind the score seam.

## Planned shape (from issue #1 and the spec; not built)

- **Domain module contract:** one declarative, loudly validated definition
  of actions with their `read_only` flags, grounding, prompts, seed corpus
  and escalate reason classes. **The first acceptance target is the
  jev-tool model** (candidates are the `jev` CLI's own verbs, with the domain
  module generated from the argparse tree plus per-verb `read_only`/args
  annotations). nvsh parity is not a gate. A second tiny toy domain in tests
  proves the stages are not jev-specific.
- **Stages** (nvsh's stages, absorbed and re-expressed as `jev run` stages),
  each resumable and each writing a per-stage state manifest (inputs with
  sha256, outputs, knobs, status):
  1. Pre-register the bars and the rule.
  2. Seed, then set up the teachers.
  3. Seal the held-out set.
  4. Draft a fresh eval pool and make the grouped split into fit and
     selection folds.
  5. Measure a baseline.
  6. Augment, including targeted recipes.
  7. Assemble (randomised letters, order, subsets and `-nocand` rows), then
     freeze with sha256.
  8. Train.
  9. Select: calibrate, sweep the gate, run the permutation probe.
  10. Quantize, and heal the quant if needed.
  11. Recalibrate on the deployed quant's own predictions.
  12. Run **one** final measurement.
  13. Bundle, which must carry `calibration.json` and `gate.json`
      (nvsh copies them by hand) and the training set actually used
      (`scorer-train.json`).
  14. Upload privately, then fetch back and verify the hash.
  15. Run the release gate.
- **CLI verbs:** `jev init <domain>`, `jev run <stage>`, `jev status`,
  `jev decide <run> [--model <bundle>]`, and later `jev ask`, which loads a
  jev-tool bundle, proposes one `jev` verb for a natural-language request
  (calibration and gate applied, arguments grounded) and only prints the
  proposal, never running `--apply` itself.
- **Minimum verdict set for `jev decide`:** ship candidate, more/fewer
  epochs, targeted augment for class X, fix the grader first, recalibrate,
  heal the quant (one round only), refit the gate, and stop/escalate to the
  human. Issue #1 has the table of triggers and knobs for each verdict.
- **Build order (spec c36):** the domain-module contract and the stage
  engine first; then the factory verbs (`init`, `run`, `status`, `decide`,
  dry-run by default) before any jev-tool training data is drafted; then the
  jev-tool run itself, through every evaluation, with the pre-registration
  file written before the first training command. The jev-tool bars are the
  stricter of stock Qwen3.5-0.8B and the minimums (95% right proposals
  minimum). nvsh#62 gets a comment stating the absorb decision and the
  domain-module interface nvsh will implement.
- **Parked operator questions (don't decide them):** whether bundles use a
  `jev-factory` hub prefix or per-domain prefixes; and how much unattended
  autonomy the decider eventually gets. (Absorb versus cite is decided:
  absorb.)

## Dependencies and environment

- The runtime package keeps **`dependencies = []`** (the CLI is cited from
  teken). Heavy ML dependencies (unsloth, torch cu130, vLLM) never go into
  the base install. Put them behind an extra (`jev-factory[train]`) or an
  external training venv, as nvsh does. Anything that needs them must
  import them lazily from the verbs that use them.
- Hardware follows nvsh: DGX Spark GB10 (128 GB **unified** memory) for
  training and measuring, and AGX Orin for the edge check. Use **one GPU for
  one job at a time**. On GB10, GPU allocations are not charged to the
  cgroup cap, so a memory-floor watchdog (`capped.sh`) protects the box.
  Serve with vLLM (images pinned by digest) or llama.cpp `llama-server`. The
  deployed format is GGUF `Q4_K_M`. Measure the artifact you ship, not the
  bf16 checkpoint.
- Related siblings: `unsloth-cli` (`sloth`, training), `dgx-spark-cli` and
  `jetson-ai-lab-cli` (hardware), `jetson-arena` (edge comparison),
  `evidence-cli` (grading the decision trail).

## Commands

```bash
uv sync                                         # dev env (Python >= 3.12)
uv run pytest -n auto                           # full suite (CI adds --cov=jev_factory; fail_under = 60)
uv run pytest tests/test_cli.py::test_version_flag -v  # a single test
uv run jev --help                               # the CLI; every verb takes --json

# lint, exactly as CI runs it
uv run black --check jev_factory tests
uv run isort --check-only jev_factory tests
uv run flake8 jev_factory tests
uv run bandit -c pyproject.toml -r jev_factory
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken"
python3 scripts/scan-secrets.py                 # committed secrets / non-localhost endpoints
uv run teken cli doctor . --strict              # agent-first rubric gate
uv run python scripts/harness-smoke.py --stage config   # all four harness configs
```

Black and isort use a line length of 100 (the `black` profile), matching
`.flake8`.

## CLI architecture (the part that exists)

The CLI is cited from teken's `python-cli` reference, so extend it in the
same style rather than restructuring it:

- `jev_factory/cli/__init__.py` builds the parser. Each verb or noun group is
  a module under `cli/_commands/` that exposes `register(subparsers)` and
  sets `func`. New noun groups (`run`, `decide`, `status`, `init`) register
  there the same way.
- Handlers return `None` or an exit code, and **raise `CliError`**
  (`cli/_errors.py`: code, message, remediation) on failure. `_dispatch`
  wraps any other exception, so no traceback leaks. Argparse errors go
  through the same structured path, and `--json` is pre-scanned from argv
  so parse errors render as JSON too.
- Output contract: results go to stdout, diagnostics to stderr, never mixed
  (`cli/_output.py`). Exit codes are `0` success, `1` user error,
  `2` environment error, `3+` reserved.
- `explain/catalog.py` holds the markdown for `jev explain <path>`. Every new
  verb needs a catalog entry, and `teken cli doctor --strict` checks the
  rubric.
- `doctor` checks `prompt-file-present` and `backend-consistency` against
  `culture.yaml`, using the backend → prompt-file map in
  `cli/_commands/doctor.py`.

## Four harnesses, four prompt files, no `AGENTS.md`

Each root prompt file is read by exactly one harness, and all four are live
over this clone at once:

| Harness | File(s) |
|---|---|
| Claude Code | `CLAUDE.md` (this file, and the mesh resident) |
| Pi / `associate` | `AGENTS.override.md` (context) + `.pi/SYSTEM.md` (system prompt, read-only non-coding role) |
| colleague | `AGENTS.colleague.md` |
| Qwen Code | `QWEN.md` |

- **Never add a root `AGENTS.md`.** It would shadow the Pi and colleague
  cascade, and `harness-smoke.py` and the tests fail on it.
- When project facts change here, update the other three files too, each in
  its own framing. Don't let them drift.
- The live probes in `docs/harness-invocations.yaml` depend on wording:
  `CLAUDE.md` and `QWEN.md` must describe the project as **jev-factory**,
  and the `lobes-cli` attribution must appear **only** in `.pi/SYSTEM.md`,
  never in `AGENTS.override.md`.
- `.qwen/skills`, `.pi/skills` and `.colleague/skills` are relative symlinks
  onto `.claude/skills`, so there is one skill tree for four loaders.
- The two selections are separate: the interactive harness is whichever
  binary you run, and the mesh resident is `culture.yaml`'s `backend`.
  Forcing a harness is invocation-level only and never mutates tracked
  files (`docs/harness-selection.md`, `docs/automation-contract.md`).

## Skills and workflow

- `.claude/skills/` is the guildmaster kit, vendored **verbatim**
  (cite-don't-import). Provenance and the re-sync procedure are in
  `docs/skill-sources.md`. Don't edit the vendored scripts; fixes go
  upstream. Every `SKILL.md` needs `type: command`, or the culture skill
  loader skips it.
- The devague lane (`scope`, `think`, `challenge`, `spec-to-plan`,
  `assign-to-workforce`, `deviate`, `validate-delivery`,
  `summarize-delivery`) is how features are specced and built here. The
  `deviate` record is the same mechanism the factory's decision records
  and human overrides mirror.
- PRs go through the `cicd` skill. `communicate` files issues on sibling
  repos, such as nvsh for #62. `ask-colleague review` gives a second opinion
  on non-trivial diffs.
- **Every PR bumps the version**, even for docs, config or CI: use
  `/version-bump` (it updates `pyproject.toml` and `CHANGELOG.md`, in Keep a
  Changelog format). CI's `version-check` job blocks merge otherwise.
  Pushing to `main` publishes to PyPI through Trusted Publishing
  (`.github/workflows/publish.yml`), and PRs do a TestPyPI dry-run.
