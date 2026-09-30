"""Generate the jev-CLI :class:`~jev_factory.domain.model.Domain` from argparse.

The domain is *derived*, not hand-copied: operations come from the subcommand
tree of :func:`jev_factory.cli._build_parser` (name, help text), and the
per-verb ``read_only`` flag and argument schema come from the annotation
registry (:mod:`jev_factory.domains.jev_cli.annotations`). A verb present on
one side only raises :class:`CliDomainError`. The prose (answer policy,
reasons, persona, topics, paraphrases) and the train-only seed corpus come
from :mod:`jev_factory.domains.jev_cli.content`.

Grounding is offline and deterministic: ``explain <path>`` grounds against
the explain catalog's own keys, and ``run <stage>`` against the stage
registry (:func:`jev_factory.factory.stages.list_stages`), so the domain
follows the stage list as stages register. The world snapshot is
:func:`world`, the CLI introspecting itself; no machine lookup is involved.

:func:`cli_surface_sha256` is the CLI surface hash: the running CLI's
:meth:`Domain.surface_sha256` (what a bundle records and ``jev ask`` compares).
Stdlib only: no GPU and no ML dependency.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from jev_factory.domain.model import Domain, GroundKind, Operation, WorldField
from jev_factory.domain.validate import validate
from jev_factory.domains.jev_cli import content
from jev_factory.domains.jev_cli.annotations import (
    ANNOTATIONS,
    EXPLAIN_PATH,
    STAGE,
    UNIVERSAL_DESTS,
    Annotation,
    verb_name,
)
from jev_factory.factory.stages import Registry, list_stages

DOMAIN_NAME = "jev-cli"


class CliDomainError(ValueError):
    """The argparse tree and the annotation registry disagree. ``code`` names how."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:  # noqa: SLF001 - argparse has no public accessor
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action
    return None


def _walk(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser, str]]:
    """Every verb as ``(path, parser, help)``, depth first, in registration order.

    A node is a verb when it is a leaf or carries its own handler (``func``).
    """
    sub = _subparsers(parser)
    if sub is None:
        return []
    helps = {a.dest: (a.help or "") for a in sub._choices_actions}  # noqa: SLF001
    found: list[tuple[tuple[str, ...], argparse.ArgumentParser, str]] = []
    seen: set[int] = set()
    for name, child in sub.choices.items():
        if id(child) in seen:  # an alias of a verb already taken
            continue
        seen.add(id(child))
        path = prefix + (name,)
        nested = _walk(child, path)
        if not nested or child.get_default("func") is not None:
            found.append((path, child, helps.get(name) or child.description or name))
        found.extend(nested)
    return found


def _arguments(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        a.dest: a
        for a in parser._actions  # noqa: SLF001
        if a.dest not in UNIVERSAL_DESTS
        and not isinstance(a, argparse._SubParsersAction)  # noqa: SLF001
    }


def _is_write_verb(parser: argparse.ArgumentParser) -> bool:
    return any("--apply" in a.option_strings for a in parser._actions)  # noqa: SLF001


def _required(action: argparse.Action) -> bool:
    if action.option_strings:
        return bool(action.required)
    return action.nargs in (None, "+")


def _problems(
    verbs: list[tuple[tuple[str, ...], argparse.ArgumentParser, str]],
    annotations: Mapping[str, Annotation],
) -> list[CliDomainError]:
    found: list[CliDomainError] = []
    names = {verb_name(path) for path, _, _ in verbs}
    for name in sorted(names - set(annotations)):
        found.append(
            CliDomainError("unannotated_verb", f"{name!r} is in argparse but not annotated")
        )
    for name in sorted(set(annotations) - names):
        found.append(CliDomainError("unknown_verb", f"{name!r} is annotated but not in argparse"))
    for path, parser, _ in verbs:
        name = verb_name(path)
        note = annotations.get(name)
        if note is None:
            continue
        if _is_write_verb(parser) and note.read_only:
            found.append(
                CliDomainError("write_verb_read_only", f"{name!r} has --apply so it is mutating")
            )
        real = _arguments(parser)
        declared = {a.name for a in note.args}
        for arg in sorted(declared - set(real)):
            found.append(CliDomainError("unknown_argument", f"{name!r} has no argument {arg!r}"))
        for arg, action in sorted(real.items()):
            if arg not in declared and _required(action):
                found.append(
                    CliDomainError("unannotated_argument", f"{name!r} argument {arg!r} is required")
                )
    return found


def _paths() -> list[str]:
    from jev_factory.explain import known_paths

    return [" ".join(p) for p in known_paths() if p]


def _stage_names(registry: Registry | None) -> list[str]:
    return [s["stage"] for s in list_stages(registry)]


def world(registry: Registry | None = None) -> dict[str, Any]:
    """The CLI's own world snapshot: explain paths and registered stages."""
    return {"explain_paths": _paths(), "stages": _stage_names(registry)}


def generate_domain(
    parser: argparse.ArgumentParser | None = None,
    annotations: Mapping[str, Annotation] | None = None,
    registry: Registry | None = None,
) -> Domain:
    """Build and validate the jev-CLI domain; raise :class:`CliDomainError` on drift."""
    if parser is None:
        from jev_factory.cli import _build_parser

        parser = _build_parser()
    annotations = ANNOTATIONS if annotations is None else annotations
    verbs = _walk(parser)
    found = _problems(verbs, annotations)
    if found:
        raise CliDomainError(found[0].code, "; ".join(e.message for e in found))
    operations = tuple(
        Operation(
            name=verb_name(path),
            description=help_text.strip(),
            read_only=annotations[verb_name(path)].read_only and not _is_write_verb(child),
            args=annotations[verb_name(path)].args,
        )
        for path, child, help_text in verbs
    )
    names = {op.name for op in operations}
    return validate(
        Domain(
            name=DOMAIN_NAME,
            description=content.DESCRIPTION,
            operations=operations,
            ground_kinds=(
                GroundKind(
                    name=EXPLAIN_PATH,
                    noun="explain path",
                    plural="explain paths",
                    world_field="explain_paths",
                    lookup=_paths,
                ),
                GroundKind(
                    name=STAGE,
                    noun="stage",
                    plural="stages",
                    world_field="stages",
                    lookup=lambda: _stage_names(registry),
                ),
            ),
            world_schema=(
                WorldField("explain_paths", "list", "Command paths the explain catalog knows."),
                WorldField("stages", "list", "Factory stages registered in the stage registry."),
            ),
            reasons=content.REASONS,
            default_reason=content.DEFAULT_REASON,
            instruction=content.INSTRUCTION,
            answer_policy=content.ANSWER_POLICY,
            persona=content.PERSONA,
            explain_topics=content.EXPLAIN_TOPICS,
            phrasing_styles=content.PHRASING_STYLES,
            paraphrases=tuple(
                (name, texts)
                for name, texts in content.PARAPHRASES.items()
                if name in names  # a caller's own parser may lack some verbs
            ),
            seed_corpus=content.SEED_CORPUS,
            card_text=content.CARD_TEXT,
        )
    )


def cli_surface_sha256(parser: argparse.ArgumentParser | None = None) -> str:
    """The CLI surface hash: sha256 of the generated domain's candidate surface."""
    return generate_domain(parser).surface_sha256()
