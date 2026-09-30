# AGENTS.override.md

This file is the **context layer** for the Pi harness (the `pi` CLI, and the
`associate` non-coding harness modelled on it) when it runs inside this repo.
Pi's CONTEXT loader concatenates `AGENTS.md` or `CLAUDE.md` from its user-level
config directory (see Pi's own docs), each parent directory, and the working
directory — but an `AGENTS.override.md`
present in a directory replaces that directory's `AGENTS.md`/`CLAUDE.md` entry
outright rather than adding to it. That is why this repo ships this file
instead of an `AGENTS.md`: Pi must **not** inherit `CLAUDE.md` (the Claude Code
guidance file) — the two harnesses read the same repository very differently,
and `CLAUDE.md` assumes a coding session with full repo-write authority that
Pi's non-coding lane does not have.

The identity and behavioral bounds for that lane — who Pi is here, what it may
and may not do — live one layer up, in Pi's **system prompt** file,
[`.pi/SYSTEM.md`](.pi/SYSTEM.md). That file replaces Pi's default
coding-assistant system prompt entirely. This file is project *context* only:
what the repo is and how it is laid out, not who is reading it.

## What this project is

`jev-factory` is an AgentCulture mesh agent that turns the process
[`nvsh`](https://github.com/agentculture/nvsh) used to build its **Tool-Jev**
model into a reusable factory for **jev-like** models. It has three layers:

1. **Factory.** nvsh's staged fine-tune pipeline, absorbed into this repo
   (not cited; nvsh becomes a consumer) and made domain-generic through a
   declarative domain module.
2. **Decision surface.** `jev decide` applies a pre-registered rule to each
   run's metrics and records the verdict.
3. **Self-hosting.** A jev-like decider model is built to make those
   decisions.

The brief is GitHub issue #1 in `agentculture/jev-factory`. The operator's
decisions, which override older text in that brief, are in
`docs/specs/2026-09-30-extract-jev-process-jev-cli-first-model.md`:
the first and only acceptance target is a **jev-tool** model (a scorer whose
candidates are the `jev` CLI's own verbs), and nvsh `scorer-r3b` parity is
not a gate.

**Current state.** The factory code exists: `jev init`, the 22 `jev run`
stages, `jev status`, `jev decide`, `jev ask` and domain modules are built
and tested. The jev-tool model itself is not yet trained, so report no bundle
as shipped. `docs/lessons-encoded.md` lists the failures the checks prevent.

A **jev-like** model is a calibrated candidate scorer (nvsh's "Track B"). It
reads one request plus up to 52 lettered candidate actions. Two controls are
always offered: `explain` and `escalate`. It outputs a distribution over
those candidates, which is calibrated with a temperature and an optional
per-label vector. A gate with separate read-only and mutating thresholds
turns that into propose / explain / escalate / abstain. Arguments are
grounded outside the model. nvsh's reference build, `scorer-r3b` on
Qwen3.5-0.8B, is context only; it is replayed only if a jev-tool evaluation
fails. Other model families are planned, such as GLiNER2.5-Decide
(nvsh issue #67) are planned.

When a question is about the fine-tune process itself, the source material
lives in the sibling checkout `../nvsh`. jev-factory absorbs its scorer-path
code once and then owns it, so read it as the origin, not as a dependency.
Not every script there is domain-generic: `gate.py`, `metrics.py`,
`calibration_fit.py`, `sweep_gate.py`, `train_scorer.py`, `scan_bundle.py`
and `leakage_check.py` are nvsh-coupled and get rewired on import.

- `docs/scorer-finetune-playbook.md`, especially its Step 0 port checklist;
- `docs/tool-jev-calibration-rule.md`;
- `scripts/lfm-finetune/` and `evals/`.

nvsh issue #62 is the domain-module seam between the two repos.

jev-factory is a sibling to
[`guildmaster`](https://github.com/agentculture/guildmaster) (the skills
supplier), [`steward`](https://github.com/agentculture/steward) (alignment),
and [`teken`](https://github.com/agentculture/teken) (the CLI scaffolder this
package is cited from).

## Four harnesses, four files, no shared base

This repo's root carries one prompt file per harness, each read by exactly
one of them — there is deliberately no shared `AGENTS.md` base for them to
cascade from:

- **Claude Code** reads [`CLAUDE.md`](CLAUDE.md).
- **Pi / associate** reads this file (`AGENTS.override.md`) for context, plus
  [`.pi/SYSTEM.md`](.pi/SYSTEM.md) for its system prompt.
- **colleague** reads [`AGENTS.colleague.md`](AGENTS.colleague.md) (the start
  of colleague's own cascade — see that file).
- **Qwen Code** reads [`QWEN.md`](QWEN.md).

If you are reading this as a human, `CLAUDE.md` is the fullest write-up of the
repo's conventions and is the one to read first; the other three exist to keep
each non-Claude harness from silently inheriting Claude-specific instructions
it cannot act on the same way.

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: jev-factory
  backend: claude
```

This repo's *mesh* resident runs on `backend: claude`, so `CLAUDE.md` is
the live resident prompt. A Pi session working in a clone of this repo is a
**local tool session**, not the mesh resident — it reads this file and
`.pi/SYSTEM.md` regardless of what `culture.yaml` declares, and running `pi`
here neither requires nor changes that declaration.

(A clone that wants `associate` as its *mesh* resident declares
`backend: colleague` with `model: associate` — see `docs/skill-sources.md`.
That is a per-clone choice; this repo does not ship it.)

## Layout (what you can read/find/summarize here)

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

## Conventions worth knowing before you answer a question about this repo

- The vendored skills under `.claude/skills/` are cited **verbatim** from
  guildmaster — never propose editing their scripts; the fix belongs upstream
  (`docs/skill-sources.md` has the re-sync procedure).
- Names: the command is `jev`, the import package is `jev_factory`, and the
  PyPI distribution is `jev-factory`.
- Some rules are enforced in code rather than exposed as knobs:
  - sealed held-out and test sets are touched once;
  - the gate is fit on the fit fold only;
  - leakage fails closed;
  - every write verb is dry-run unless you pass `--apply`;
  - publishing is private-first, and going public is a human decision.

  Don't summarize any of these as tunable.
- Every PR bumps the version (`version-bump` skill); CI's `version-check` job
  blocks merge otherwise.
- This file describes the repo **as it exists on disk today**. If you are
  asked to update it, keep claims grounded in checked-in reality.
