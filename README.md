# jev-factory

**jev-factory** turns the process [nvsh](https://github.com/agentculture/nvsh)
used to build its *Tool-Jev* model into a reusable factory for **jev-like
models**. It adds explicit knobs and a recorded decision step after every run.
It then fine-tunes its own jev-like model to make those decisions
(self-hosting).

> **Status: scaffold.** Nothing of the factory is built yet. What exists today
> is the agent scaffold described under [What's here now](#whats-here-now).
> The build brief is
> [issue #1](https://github.com/agentculture/jev-factory/issues/1).

## What "jev-like" means

A jev-like model is a small, **calibrated candidate scorer** (OpenJev-style,
nvsh's "Track B"), not a generative tool caller. It works like this:

- It reads one request plus up to 52 lettered candidate actions
  (`A) <name>: <description>`). Two controls are always offered: `explain`
  and `escalate`.
- It produces a probability distribution over those candidates, then
  calibrates it (a temperature, plus an optional per-label vector).
- A gate with separate thresholds for **read-only** and **mutating** actions
  turns the calibrated distribution into propose / explain / escalate /
  abstain.
- Action arguments are grounded **deterministically**, outside the model.
- The choice must survive reordering, re-lettering, subsetting and
  paraphrasing of the candidates (the permutation-probe robustness bar).

nvsh's reference build is `scorer-r3b` on `Qwen/Qwen3.5-0.8B`, served as
GGUF `Q4_K_M`. It scored 0 wrong mutating proposals and ECE 0.016 after
calibration on the test side, at about 355 ms per decision on an AGX Orin.
The recipe is in nvsh's
[`docs/scorer-finetune-playbook.md`](https://github.com/agentculture/nvsh/blob/main/docs/scorer-finetune-playbook.md).
Other model families are planned too, such as the GLiNER2.5-Decide encoder
([nvsh#67](https://github.com/agentculture/nvsh/issues/67)). The factory
treats the backbone as a pluggable adapter behind the same
candidates → calibrated distribution → gate contract.

## The plan (three layers)

1. **Factory.** nvsh's staged pipeline, made domain-generic through a
   declarative *domain module*: actions with `read_only` flags, grounding,
   prompts, seed corpus and escalation reasons. The stages run from
   pre-registering the bars and seeding the corpus through train, select,
   quantize, recalibrate, one final run, bundle and a private upload. Each
   stage is resumable and writes an artifact manifest. The first acceptance
   test is **parity**: reproduce nvsh's scorer-r3b selection verdict from
   its frozen data. This builds on nvsh's domain-module seam,
   [nvsh#62](https://github.com/agentculture/nvsh/issues/62).
2. **Decision surface.** `jev decide <run>` applies a **pre-registered rule**
   to a run's metrics. It emits a verdict (ship, more or fewer epochs,
   targeted augment, fix the grader, recalibrate, heal the quant, refit the
   gate, or escalate to a human) as an append-only decision record. A human
   override is recorded as a deviation.
3. **Self-hosting.** The factory builds a jev-like *decider* whose
   candidates are those verdicts, using real nvsh decisions as the sealed
   set. The decider drives the loop (`jev decide --model <bundle>`) only
   after it beats the rule with 0 wrong mutating verdicts. Below its gate it
   falls back to the rule or the human.

These honesty rules are enforced in code, not exposed as knobs:

- Sealed held-out and test sets are touched once.
- The gate is fit on the fit fold only.
- Leakage fails closed.
- Every write verb is dry-run unless you pass `--apply`.
- Publishing is private-first. Going public is always a human decision.

Planned CLI: `jev init <domain>`, `jev run <stage>`, `jev status` and
`jev decide`. Training dependencies will stay out of the base install,
behind an extra or an external venv.

## What's here now

- **An agent-first CLI**, `jev`, cited from
  [teken](https://github.com/agentculture/teken) (`afi-cli`). The runtime
  package has no third-party dependencies.
- **A mesh identity**: `culture.yaml` (`suffix: jev-factory`,
  `backend: claude`) and the matching resident prompt `CLAUDE.md`.
- **Four harness prompt files**, each read by exactly one harness (see
  below).
- **The guildmaster skill kit** under `.claude/skills/`, vendored
  cite-don't-import. See [`docs/skill-sources.md`](docs/skill-sources.md).
- **A build and deploy baseline**: pytest, lint, the agent-first rubric gate,
  a per-harness smoke check, and PyPI Trusted Publishing in GitHub Actions.

## Prompt files by harness

Four harnesses, four root files, no shared base — each file is read by
exactly one harness:

| Harness | File(s) |
|---------|---------|
| Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Pi / associate | [`AGENTS.override.md`](AGENTS.override.md) + [`.pi/SYSTEM.md`](.pi/SYSTEM.md) |
| colleague | [`AGENTS.colleague.md`](AGENTS.colleague.md) |
| Qwen Code | [`QWEN.md`](QWEN.md) |

**Claude Code** — `CLAUDE.md` is the fullest write-up of the repo's
conventions; read it first.

**Pi / associate** — `AGENTS.override.md` replaces this directory's
`AGENTS.md`/`CLAUDE.md` in Pi's context layer, so Pi does not inherit
`CLAUDE.md`. `.pi/SYSTEM.md` replaces Pi's default system prompt with the
non-coding `associate` identity (read/find/summarize only).

**colleague** — colleague's prompt cascade is `AGENTS.md` →
`AGENTS.colleague.md` → `AGENTS.colleague.<model>.md`. This repo ships only
the middle layer: there is no `AGENTS.md` (a shared base across harnesses was
considered and rejected) and no per-model override file.

**Qwen Code** — Qwen Code reads `QWEN.md` and `AGENTS.md`; since there is no
`AGENTS.md`, `QWEN.md` is its sole source of guidance.

There is intentionally **no `AGENTS.md`** at the root — each harness gets an
unrelated file rather than cascading from a shared base.

## Two selections, not one

It is tempting to read "switch harness" as one decision. It is actually two,
and this repo's layout exists partly to keep them separate:

1. **The interactive harness** — which binary you run (`claude`, `pi`,
   `colleague`, `qwen`). `cd` into the clone and run any of them; all four
   are live simultaneously, and none of them requires editing a file or
   flipping a switch. A harness can be force-selected for one invocation
   (e.g. a CI smoke check) without ever touching `culture.yaml` — see
   [`docs/automation-contract.md`](docs/automation-contract.md).
2. **The mesh resident** — the single `backend` `culture.yaml` declares,
   which is what the Culture daemon starts and what `steward doctor`
   checks. `guild harness use <name>` changes only this.

`culture.yaml`'s `backend` affects (2) only. It never affects which harness
you can invoke interactively in (1). See
[`docs/harness-selection.md`](docs/harness-selection.md) for the full
writeup, including who reads this config and why existing siblings are not
retrofitted by this arc.

## Quickstart

```bash
uv sync
uv run pytest -n auto                 # run the test suite
uv run jev whoami                     # identity from culture.yaml
uv run jev learn                      # self-teaching prompt (add --json)
uv run teken cli doctor . --strict    # the agent-first rubric gate CI runs
```

The command is `jev`, the import package is `jev_factory`, and the PyPI
distribution is `jev-factory`.

## CLI

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path. |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `cli overview` | Describe the CLI surface itself. |

Every command supports `--json`. Results go to stdout, and errors and
diagnostics go to stderr (never mixed). Exit codes: `0` success, `1` user
error, `2` environment error, `3+` reserved.

## Contributing

See [`CLAUDE.md`](CLAUDE.md) for the full conventions: the jev-like
invariants, the nvsh source material, version-bump-every-PR, and the `cicd`
PR lane. Every PR bumps the version (`/version-bump`), and CI blocks merge
otherwise.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
