# QWEN.md

This file provides guidance to Qwen Code when working with code in this
repository. Qwen Code's context loader reads exactly `QWEN.md` and `AGENTS.md`
in a directory; this repo deliberately ships only `QWEN.md` — there is no
`AGENTS.md` here (each harness gets its own file; see "Prompt files by
harness" below), so this file is the sole source of project guidance for a
Qwen Code session.

## What this project is

`jev-factory` is an AgentCulture mesh agent. It turns the process
[`nvsh`](https://github.com/agentculture/nvsh) used to build its **Tool-Jev**
model (a calibrated scorer fine-tuned from `Qwen/Qwen3.5-0.8B`) into a
reusable factory for **jev-like** models. It has three layers:

1. **Factory.** nvsh's staged pipeline, absorbed into this repo (not cited;
   nvsh becomes a consumer) and made domain-generic through a
   declarative *domain module*: actions with `read_only` flags, grounding,
   prompts, seed corpus and escalation reasons.
2. **Decision surface.** `jev decide <run>` applies a pre-registered rule to
   a run's metrics and writes an append-only decision record.
3. **Self-hosting.** The factory builds a jev-like decider whose candidates
   are those verdicts, and uses it to drive new builds, with the rule as
   baseline and fallback.

The brief is issue #1 (`gh issue view 1`). The operator's decisions, which
override it where they differ, are in
`docs/specs/2026-09-30-extract-jev-process-jev-cli-first-model.md`.
The first and only acceptance target is a **jev-tool** model (a scorer whose
candidates are the `jev` CLI's own verbs); nvsh `scorer-r3b` parity is not a
gate. The factory code now exists (`jev init`, the `jev run` stages,
`jev status`, `jev decide`, `jev ask`, `jev review`, domain modules), but the jev-tool model
is not yet trained, so do not describe a bundle as shipped.

A **jev-like** model is a candidate scorer, not a generative tool caller:

- It reads one request plus up to 52 lettered candidates
  (`A) <name>: <description>`), always including `explain` and `escalate`.
- It produces a distribution over the offered candidates, calibrated with a
  temperature and an optional per-label vector.
- A gate with separate read-only and mutating thresholds turns that
  distribution into propose / explain / escalate / abstain.
- Arguments are grounded deterministically, outside the model.
- The choice must survive candidate reordering, re-lettering, subsetting and
  paraphrasing.

Other model families are planned, such as GLiNER2.5-Decide (nvsh issue #67),
so the backbone and its readout are pluggable adapters behind that contract.

The source material is in `../nvsh`: `docs/scorer-finetune-playbook.md` (its
Step 0 port checklist lists what is domain-specific),
`docs/tool-jev-calibration-rule.md`, `scripts/lfm-finetune/` and `evals/`.
jev-factory absorbs the scorer-path code once and then owns it. Not every
script there is domain-generic: `gate.py`, `metrics.py`,
`calibration_fit.py`, `sweep_gate.py`, `train_scorer.py`, `scan_bundle.py`
and `leakage_check.py` are nvsh-coupled and get rewired on import. nvsh
issue #62 is the domain-module seam.

Some rules are enforced in code, never exposed as knobs:

- Sealed held-out and test sets are touched once.
- The gate is fit on the fit fold only.
- Leakage fails closed.
- Every write verb is dry-run unless you pass `--apply`.
- Publishing is private-first.

It is a sibling to [`guildmaster`](https://github.com/agentculture/guildmaster)
(the **skills supplier**), [`steward`](https://github.com/agentculture/steward)
(**alignment**, through `steward doctor`), and
[`teken`](https://github.com/agentculture/teken) (the **afi-cli** scaffolder
this CLI is cited from) within the Organic Development framework.

## Prompt files by harness

This repo's root carries one prompt file per agent harness, each read by
exactly one of them — there is no shared base file for them to inherit from:

- **Claude Code** → [`CLAUDE.md`](CLAUDE.md) (the fullest write-up; read it
  first if you are new to the repo).
- **Pi / associate** → [`AGENTS.override.md`](AGENTS.override.md) for context,
  plus [`.pi/SYSTEM.md`](.pi/SYSTEM.md) for its system prompt.
- **colleague** → [`AGENTS.colleague.md`](AGENTS.colleague.md).
- **Qwen Code** → this file.

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: jev-factory
  backend: claude
```

`backend: claude` fixes the *mesh resident* prompt file to `CLAUDE.md` — the
mesh runtime reads that file, not this one. A Qwen Code session working in a
clone of this repo is a separate, local tool session; it reads `QWEN.md`
regardless of what `culture.yaml` declares, and running Qwen Code here neither
requires nor changes that declaration. The declaration and the resident prompt
together satisfy the two invariants `steward doctor` verifies:
**prompt-file-present** and **backend-consistency** (`claude` ↔ `CLAUDE.md`).

## The CLI

The CLI is cited (cite-don't-import) from teken's `python-cli` reference
(`teken cli cite`), so the runtime package has **no third-party dependencies** (keep it that way;
ML/training deps go behind an extra or an external venv);
`teken` (a.k.a. `afi-cli`) is a dev dependency only. The command is `jev` (package `jev_factory`, PyPI dist `jev-factory`).
Agent-first verbs:

- `jev whoami` — identity from `culture.yaml`.
- `jev learn` — structured self-teaching prompt.
- `jev explain <path>` — markdown docs for any noun/verb.
- `jev overview` — descriptive snapshot of the agent.
- `jev doctor` — check the agent-identity invariants.
- `jev cli overview` — describe the CLI surface itself.

Conventions: every command supports `--json`; results go to stdout, errors and
diagnostics to stderr (never mixed); exit codes are `0` success, `1` user
error, `2` environment error, `3+` reserved. The agent-first rubric is
enforced in CI by `teken cli doctor . --strict`.

## Skills

`.claude/skills/` vendors the **canonical guildmaster skill kit**
(cite-don't-import). Provenance and the re-sync procedure live in
`docs/skill-sources.md`. Do not reformat or edit vendored scripts — re-sync
from guildmaster instead.

## Conventions

- **Every PR bumps the version** — even docs/config/CI. Use the
  `version-bump` skill; the `version-check` CI job blocks merge otherwise.
- **Tests**: `uv run pytest -n auto`. **Lint**: black, isort, flake8 (line
  length 100), bandit, markdownlint.
- **Deploy**: pushing to `main` publishes to PyPI via Trusted Publishing
  (`.github/workflows/publish.yml`); PRs do a TestPyPI dry-run.

## Layout

```text
jev_factory/   agent-first CLI (cited from teken's python-cli reference)
  cli/                    parser, error/output contract, _commands/ (verbs)
  explain/                markdown catalog for `explain`
tests/                    pytest smoke + introspection tests
.claude/skills/           vendored guildmaster skill kit (cite-don't-import)
docs/skill-sources.md     skill provenance ledger
culture.yaml              mesh identity (suffix + backend)
.github/workflows/        tests + deploy (PyPI Trusted Publishing)
```

This file describes the repository **as it exists on disk today**. When you
edit, keep claims grounded in checked-in reality; if a section drifts ahead of
reality, mark it `(planned)` or move it under a `## Roadmap` heading. For the
full set of conventions (the jev-like definition, the nvsh source map, CI
commands, CLI architecture, skills workflow), see [`CLAUDE.md`](CLAUDE.md) — they apply
to work in this repo regardless of which harness is doing it.
