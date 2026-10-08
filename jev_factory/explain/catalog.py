"""Markdown catalog for ``jev-factory explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("jev-factory",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# jev-factory

A factory for **jev-like** models: calibrated candidate scorers that read one
request plus a bounded set of lettered candidate actions, answer with one
token, and pass through a deterministic read-only/mutating gate. It turns the
process behind the first Tool-Jev scorer into a staged, resumable,
dry-run-by-default pipeline, records every between-runs decision, and is itself
an AgentCulture mesh agent (`culture.yaml` + `CLAUDE.md`).

## Who it is for

- **The operator** approves every gate: scaffold a run with `init`, dry-run
  each stage, pass `--apply`, read `status` and the decision records, and
  decide anything public.
- **The mesh agent** (this repo's resident) runs the stages in order, reads
  `status --json`, and records each between-runs verdict with `decide`.
- **A domain author** writes one domain module to build a jev-like model for a
  new domain: `jev-factory explain domain`.

The factory code exists and is tested; the first product, a jev-tool model
whose candidates are this CLI's own verbs, is not yet trained. The failures the
checks prevent, and the test behind each, are in `docs/lessons-encoded.md`.

## Verbs

- `jev-factory init <domain>` — scaffold a run directory and run config.
- `jev-factory run <stage>` — run one build stage (one subcommand per stage).
- `jev-factory status <run>` — stage manifests, staleness, job progress; `--watch` every 30m.
- `jev-factory decide <run>` — apply the pre-registered rule; append a record.
- `jev-factory ask <request>` — propose one jev verb from a bundle; executes nothing.
- `jev-factory whoami` — identity probe from `culture.yaml`.
- `jev-factory learn` — structured self-teaching prompt.
- `jev-factory explain <path>` — markdown docs for any noun/verb.
- `jev-factory overview` — descriptive snapshot of the agent.
- `jev-factory doctor` — check the agent-identity invariants.
- `jev-factory cli overview` — describe the CLI surface.

## Dry-run by default

Every write verb (`init`, each `run <stage>`) changes nothing without
`--apply`. The sealed held-out set is touched once, bars are pre-registered
before training, and publishing is private-first.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `jev-factory explain run`
- `jev-factory explain status`
- `jev-factory explain decide`
- `jev-factory explain domain`
"""

_WHOAMI = """\
# jev-factory whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    jev-factory whoami
    jev-factory whoami --json
"""

_LEARN = """\
# jev-factory learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer.

## Usage

    jev-factory learn
    jev-factory learn --json
"""

_EXPLAIN = """\
# jev-factory explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    jev-factory explain jev-factory
    jev-factory explain whoami
    jev-factory explain --json <path>
"""

_OVERVIEW = """\
# jev-factory overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the sibling-pattern artifacts the template carries. Accepts an
ignored `target` so a stray path never hard-fails.

## Usage

    jev-factory overview
    jev-factory overview --json
"""

_DOCTOR = """\
# jev-factory doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`claude` → `CLAUDE.md`), plus a
skills-present check. Exits 1 when unhealthy.

prompt-file-present requires the *resident* prompt the declared backend
actually reads. Other harness prompt files recognized under the same backend
name (`AGENTS.override.md`, `.pi/SYSTEM.md`, `QWEN.md`) belong to
interactively available harnesses the mesh daemon never loads; they are
reported by the informational harness-prompts check and never substituted.

## Usage

    jev-factory doctor
    jev-factory doctor --json
"""

_INIT = """\
# jev-factory init <domain>

Scaffolds a run: validates the domain (a dotted module exposing `DOMAIN`, or a
domain JSON file), then writes `run.json` (domain reference) and `run.toml`
(every documented run-config key; required ones are commented until you set
them) into `--work`. The work directory must be outside any git worktree, and
an existing run is never overwritten.

Dry-run by default: without `--apply` it lists the files and writes nothing.

## Usage

    jev-factory init <domain> --work <dir>
    jev-factory init <domain> --work <dir> --base org/name --base-rev <sha> --apply
    jev-factory init <domain> --work <dir> --json
"""

_RUN = """\
# jev-factory run <stage>

Runs one build stage of the pipeline. There is one subcommand per registered
stage, so `jev-factory run --help` lists them all, in order. Each stage reads
declared inputs, writes declared outputs and a manifest under `manifests/`; a
fresh stage is a no-op and a stale one says why.

Dry-run by default: without `--apply` a stage prints what it would read and
write and whether it is stale, and changes no file. `--apply` runs it under the
run lock. `--apply --detach` starts it in its own session (pid file, done
marker, log under `jobs/`) so a long stage survives the shell; follow it with
`jev-factory status <run>`.

Run context: `--work`, `--config` and `--domain` (defaults come from the run
scaffolded by `init`, or `$JEV_DOMAIN`); `--knob KEY=VALUE` sets a stage knob
(unknown knobs are refused).

## Stages

STAGES

## Usage

    jev-factory run --help
    jev-factory run split --work <dir>
    jev-factory run split --work <dir> --apply
    jev-factory run draft-eval --work <dir> --apply --detach
"""

_STATUS = """\
# jev-factory status <run>

Read-only view of a run's work directory: each stage's manifest status
(`ok`, `failed`, `running`, or `-` for not run), whether a finished stage is
stale and why, every job under `jobs/` (detached jobs, item ledgers,
measurements and probes) with its state, items done/total, rate and ETA, and
the number of decision records. It needs no domain and writes nothing.

## Watching long-running work

`--watch` is the standard progress cadence: it prints a timestamped update
now and every `--every` (default `30m`; also `1h`, `90s`, ...) with each job's
progress, rate and ETA and what changed since the last update, and stops with
a final update once nothing is running. With `--json`, each update is one
JSON line.

## Usage

    jev-factory status <run>
    jev-factory status <run> --json
    jev-factory status <run> --watch
    jev-factory status <run> --watch --every 10m --json
"""

_REVIEW = """\
# jev-factory review <domain>

Operator review of a domain's seed corpus. Every decision (approve, reject,
edit, or propose a new entry) is appended to the review file,
`<seed stem>.review.jsonl` beside the seed unless `--review-file` names
another. The file is append-only, and an entry's latest record is its
decision. The seed itself changes only with `--apply`.

- Default (dry run): lists what the recorded decisions would change in the
  seed (removals, edits, additions), with any conflicts (the seed entry changed
  since the decision) and problems (the result fails the domain's checks).
  Nothing is written.
- `--serve`: starts the local review site on 127.0.0.1 (`--port`, default
  18765). It shows the domain's verb tree in React Flow, each verb's seed
  entries attached to it, with filters for status and source. The site needs
  the per-run token it was served with and only appends to the review file.
- `--apply`: writes the reviewed seed. It refuses if there is any conflict or
  problem, and writes nothing in that case.

Approving changes nothing in the seed; the review file is the record.
Rejecting removes the entry. An edit replaces the entry. A proposal adds a
new entry, whose source defaults to `operator-review-<date>`.

## Usage

    jev-factory review <domain>
    jev-factory review <domain> --serve
    jev-factory review <domain> --serve --port 0 --json
    jev-factory review <domain> --apply
"""

_DECIDE = """\
# jev-factory decide <run>

Applies the pre-registered, hashed selection rule to the run's evidence (every
candidate summary `select` wrote, the teacher pilot's yields, the quantize/heal
check) and appends one decision record to `decisions.jsonl` in the run. The
verdict is printed with every metric it cited and the sha256 of the file each
came from. Records are append-only; a human override is a new record, never an
edit. It refuses a run whose pre-registration is missing or was changed.

It writes that record and nothing else, and never runs a stage.

## Usage

    jev-factory decide <run>
    jev-factory decide <run> --reference <summary.json> --hard-stop public_publish
    jev-factory decide <run> --json
"""

_ASK = """\
# jev-factory ask <request> --bundle <dir>

Proposes one jev verb for a request, using a jev-tool model bundle. It scores the
request over the running CLI's verbs as lettered candidates (one forward pass, one
token), applies the bundle's `calibration.json`, gates the result with its
`gate.json` (separate read-only and mutating thresholds) and prints exactly one of
`propose`, `explain`, `escalate` or `abstain_uncertain` with the probability.
A proposal's arguments are grounded from the CLI catalog, never taken from the model.

It executes nothing: no process is started, nothing is written, and no `--apply`
command is ever built from model output. A mutating proposal is text for you to
review and run yourself, dry-run first.

If the bundle's recorded CLI surface hash differs from the running CLI's, the
mismatch is reported (stderr, and `surface_mismatch` in JSON); `--strict-surface`
makes it an error.

Inference: `--server <llama-server base URL>` (with `--model` if the server names
it differently), or in this process for a bf16 bundle (needs the `train` extra).

## Usage

    jev-factory ask "show the run status" --bundle <dir> --server http://127.0.0.1:8080/v1
    jev-factory ask "explain doctor" --bundle <dir> --json
"""

_DOMAIN = """\
# jev-factory domain module

A domain module is everything that makes one jev-like model about one domain,
as one immutable `Domain` value. The contract is `jev_factory/domain/model.py`;
`jev_factory/domain/validate.py` checks it loudly, one named error per problem.
A complete small example (a smart-home lamp controller) is
`tests/fixtures/toy_domain`, and the jev CLI's own domain is generated from its
argparse tree.

## What to write

- **Operations**: name, description, `read_only`, and typed args (`str` or
  `choice`). Mutating operations are the ones that change something.
- **Groundable argument kinds**: how an argument value is matched against a
  world snapshot (or a live lookup), so the model never generates values.
- **The world snapshot schema**: the fields grounding reads.
- **Escalate reasons**: one list; `escalate:<r>` and `decline:<r>` derive from it.
- **Prose**: the one answer-policy sentence, persona, explain topics, phrasing
  styles, probe paraphrases and the bundle card text.
- **A seed corpus**: a JSON file of labelled requests.

## Use it

    jev-factory init my_pkg.my_domain --work <dir>          # dry-run: validates and lists files
    jev-factory init my_pkg.my_domain --work <dir> --apply
    jev-factory run --help                                   # the stages, in order
    jev-factory run config --work <dir> --apply
    jev-factory status <dir>                                 # manifests, staleness, jobs
    jev-factory decide <dir>                                 # appends to decisions.jsonl

`init` accepts a dotted module exposing `DOMAIN` or a domain JSON file.
"""

_CLI = """\
# jev-factory cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    jev-factory cli overview
    jev-factory cli overview --json
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("jev-factory",): _ROOT,
    ("jev",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("init",): _INIT,
    ("run",): _RUN,
    ("status",): _STATUS,
    ("decide",): _DECIDE,
    ("ask",): _ASK,
    ("review",): _REVIEW,
    ("domain",): _DOMAIN,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
}


def _stage_entries() -> dict[tuple[str, ...], str]:
    from jev_factory.factory.pipeline import SUBSTEPS, get_stage, stage_names

    out: dict[tuple[str, ...], str] = {}
    lines = []
    for name in stage_names():
        st = get_stage(name)
        lines.append(f"- `{name}` — {st.summary}")
        out[("run", name)] = (
            f"# jev-factory run {name}\n\n{st.summary}\n\n"
            f"Sub-steps: {', '.join(SUBSTEPS[name])}.\n"
            f"Upstream stages: {', '.join(st.deps) or 'none'}.\n"
            f"Reads: {', '.join(st.inputs) or 'nothing'}.\n"
            f"Writes: {', '.join(st.outputs) or 'nothing'}.\n\n"
            "Dry-run by default; `--apply` runs it, `--apply --detach` runs it detached.\n\n"
            f"## Usage\n\n    jev-factory run {name} --work <dir>\n"
            f"    jev-factory run {name} --work <dir> --apply\n"
        )
    out[("run",)] = _RUN.replace("STAGES", "\n".join(lines))
    return out


ENTRIES.update(_stage_entries())
