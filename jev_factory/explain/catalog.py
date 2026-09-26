"""Markdown catalog for ``jev-factory explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple
and ``("jev-factory",)`` both resolve to the root entry.

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# jev-factory

A clonable template for AgentCulture mesh agents. It carries an agent-first CLI
(cited from the teken `python-cli` reference), a mesh identity (`culture.yaml` +
`CLAUDE.md`), the canonical guildmaster skill kit under `.claude/skills/`, and a
buildable/deployable package baseline. Clone it, rename the package, edit
`culture.yaml`, and you have a new agent.

## Verbs

- `jev-factory whoami` — identity probe from `culture.yaml`.
- `jev-factory learn` — structured self-teaching prompt.
- `jev-factory explain <path>` — markdown docs for any noun/verb.
- `jev-factory overview` — descriptive snapshot of the agent.
- `jev-factory doctor` — check the agent-identity invariants.
- `jev-factory cli overview` — describe the CLI surface.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `jev-factory explain whoami`
- `jev-factory explain doctor`
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
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
}
