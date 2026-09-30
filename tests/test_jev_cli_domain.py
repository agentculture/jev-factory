"""The jev-CLI domain generated from argparse (t14; obligations o26, o36)."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from jev_factory.cli import _build_parser
from jev_factory.domain.model import ArgSpec, Domain, GroundDecline, Grounded
from jev_factory.domains.jev_cli import annotations as ann
from jev_factory.domains.jev_cli.generate import (
    CliDomainError,
    cli_surface_sha256,
    generate_domain,
    world,
)
from jev_factory.explain.catalog import ENTRIES
from jev_factory.factory.stages import Registry, Stage


def _parser_verbs(parser: argparse.ArgumentParser) -> set[str]:
    out: set[str] = set()

    def walk(p: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, child in action.choices.items():
                    out.add(ann.verb_name(prefix + (name,)))
                    walk(child, prefix + (name,))

    walk(parser, ())
    return out


def _fake_parser(apply_flag: bool = False) -> argparse.ArgumentParser:
    """A parser shaped like the planned verbs: ``go <stage>`` and a noun group."""
    parser = _build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    run = sub.add_parser("go", help="Go to one factory stage.")
    run.add_argument("stage")
    run.add_argument("--json", action="store_true")
    if apply_flag:
        run.add_argument("--apply", action="store_true")
    run.set_defaults(func=lambda a: 0)
    return parser


RUN = ann.Annotation(read_only=True, args=(ArgSpec("stage", "str", ground=ann.STAGE),))


@pytest.mark.behavioral("o26")
def test_operations_match_argparse_subcommands_exactly() -> None:
    domain = generate_domain()
    assert set(domain.names()) == _parser_verbs(_build_parser())
    assert set(domain.names()) == set(ann.ANNOTATIONS)
    assert "jev.cli.overview" in domain.names()


@pytest.mark.behavioral("o26")
def test_verb_only_in_argparse_fails() -> None:
    with pytest.raises(CliDomainError) as exc:
        generate_domain(parser=_fake_parser())
    assert exc.value.code == "unannotated_verb"
    assert "'jev.go'" in str(exc.value)


@pytest.mark.behavioral("o26")
def test_verb_only_in_annotations_fails() -> None:
    notes = {**ann.ANNOTATIONS, "jev.ghost": ann.Annotation(read_only=True)}
    with pytest.raises(CliDomainError) as exc:
        generate_domain(annotations=notes)
    assert exc.value.code == "unknown_verb"
    assert "jev.ghost" in str(exc.value)


def test_write_verb_is_mutating_and_cannot_be_annotated_read_only() -> None:
    notes = {**ann.ANNOTATIONS, "jev.go": ann.Annotation(read_only=False, args=RUN.args)}
    domain = generate_domain(parser=_fake_parser(apply_flag=True), annotations=notes)
    assert domain.is_mutating("jev.go")
    notes["jev.go"] = RUN
    with pytest.raises(CliDomainError) as exc:
        generate_domain(parser=_fake_parser(apply_flag=True), annotations=notes)
    assert exc.value.code == "write_verb_read_only"


def test_annotated_argument_must_exist_and_required_positional_must_be_annotated() -> None:
    bad = {**ann.ANNOTATIONS, "jev.go": ann.Annotation(True, (ArgSpec("nope", "str"),))}
    with pytest.raises(CliDomainError, match="unknown_argument"):
        generate_domain(parser=_fake_parser(), annotations=bad)
    bare = {**ann.ANNOTATIONS, "jev.go": ann.Annotation(True)}
    with pytest.raises(CliDomainError, match="unannotated_argument"):
        generate_domain(parser=_fake_parser(), annotations=bare)


def test_verbs_are_read_only_except_the_write_verbs_and_all_have_descriptions() -> None:
    domain = generate_domain()
    stages = [n for n in domain.names() if n.startswith("jev.run.")]
    writes = {"jev.init", "jev.decide", *stages}
    assert stages and all(domain.is_mutating(n) for n in writes)
    assert all(domain.read_only(n) for n in domain.names() if n not in writes)
    assert all(domain.get(n).description.strip() for n in domain.names())


def test_dotted_name_mapping_round_trips() -> None:
    assert ann.verb_name(("cli", "overview")) == "jev.cli.overview"
    assert ann.verb_path("jev.cli.overview") == ("cli", "overview")


def test_explain_path_grounds_offline_against_catalog() -> None:
    domain = generate_domain()
    ok = domain.ground("jev.explain", {"path": "CLI Overview"})
    assert ok == Grounded(args={"path": "cli overview"})
    bad = domain.ground("jev.explain", {"path": "no such thing"})
    assert isinstance(bad, GroundDecline)
    snap = world()
    assert set(snap["explain_paths"]) == {" ".join(k) for k in ENTRIES if k}
    assert domain.ground("jev.explain", {"path": "doctor"}, world=snap) == Grounded(
        {"path": "doctor"}
    )


def test_run_stage_grounds_against_the_stage_registry() -> None:
    reg = Registry()
    notes = {**ann.ANNOTATIONS, "jev.go": RUN}
    domain = generate_domain(parser=_fake_parser(), annotations=notes, registry=reg)
    assert isinstance(domain.ground("jev.go", {"stage": "seal"}), GroundDecline)
    reg.register(Stage(name="seal", func=lambda w, k: None, summary="seal"))
    assert domain.ground("jev.go", {"stage": "Seal"}) == Grounded({"stage": "seal"})
    assert world(reg)["stages"] == ["seal"]


@pytest.mark.behavioral("o36")
def test_surface_hash_is_domain_hash_and_tracks_the_surface() -> None:
    domain = generate_domain()
    assert cli_surface_sha256() == domain.surface_sha256()
    assert len(cli_surface_sha256()) == 64
    assert cli_surface_sha256() == generate_domain().surface_sha256()
    notes = {**ann.ANNOTATIONS, "jev.go": RUN}
    other = generate_domain(parser=_fake_parser(), annotations=notes)
    assert other.surface_sha256() != domain.surface_sha256()
    flipped = {
        **ann.ANNOTATIONS,
        "jev.whoami": dataclasses.replace(ann.ANNOTATIONS["jev.whoami"], read_only=False),
    }
    assert generate_domain(annotations=flipped).surface_sha256() != domain.surface_sha256()


def test_domain_round_trips_through_dict() -> None:
    domain = generate_domain()
    assert Domain.from_dict(domain.to_dict()).surface_sha256() == domain.surface_sha256()
