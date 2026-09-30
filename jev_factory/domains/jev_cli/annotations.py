"""The annotation registry: what argparse cannot say about each jev verb.

argparse knows a verb's name, help text and flags. It does not know whether
the verb is read-only or which of its arguments are groundable. This registry
adds exactly that, keyed by the verb's operation name, and nothing else:
names, descriptions and the verb list itself always come from the parser
(:mod:`jev_factory.domains.jev_cli.generate`), so the two cannot drift. A
verb on only one side fails the generator and its test.

Operation names: the validator requires ``^[A-Za-z0-9][A-Za-z0-9_.-]*$``, so a
multi-word argparse path is joined with ``.`` and prefixed ``jev.``
(``cli overview`` is ``jev.cli.overview``). The prefix is needed, not cosmetic:
the verb ``explain`` would otherwise collide with the ``explain`` control
candidate. :func:`verb_name` and :func:`verb_path` are the only places that
mapping lives.

Rules the generator enforces on top of this registry:

* a verb whose parser has ``--apply`` is a write verb and must be annotated
  ``read_only=False`` (every write verb is mutating, dry-run by default);
* an annotated argument must be a real argparse argument of that verb, and a
  required argparse positional must be annotated.

Add an entry here in the same change that registers a verb.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jev_factory.domain.model import ArgSpec

#: Ground kind names the jev-CLI domain declares (see generate.py).
EXPLAIN_PATH = "explain_path"
STAGE = "stage"

#: Every operation name starts with this.
PREFIX = "jev."

#: argparse dests that are never domain arguments.
UNIVERSAL_DESTS: frozenset[str] = frozenset({"help", "json", "apply", "version", "command"})


def verb_name(path: tuple[str, ...]) -> str:
    """The operation name of an argparse path: ``("cli", "overview")`` -> ``jev.cli.overview``."""
    return PREFIX + ".".join(path)


def verb_path(name: str) -> tuple[str, ...]:
    """The argparse path of an operation name: ``jev.cli.overview`` -> ``("cli", "overview")``."""
    return tuple(name.removeprefix(PREFIX).split("."))


@dataclass(frozen=True)
class Annotation:
    """Read-only flag and argument schema for one verb."""

    read_only: bool
    args: tuple[ArgSpec, ...] = ()


ANNOTATIONS: Mapping[str, Annotation] = {
    verb_name(("whoami",)): Annotation(read_only=True),
    verb_name(("learn",)): Annotation(read_only=True),
    verb_name(("explain",)): Annotation(
        read_only=True,
        args=(ArgSpec(name="path", kind="str", ground=EXPLAIN_PATH),),
    ),
    verb_name(("overview",)): Annotation(read_only=True),
    verb_name(("doctor",)): Annotation(read_only=True),
    verb_name(("cli",)): Annotation(read_only=True),
    verb_name(
        (
            "cli",
            "overview",
        )
    ): Annotation(read_only=True),
}
