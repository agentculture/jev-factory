"""jev_factory.factory.safepath: containment of CLI / run-config / manifest paths (Sonar S2083)."""

from __future__ import annotations

import os

import pytest

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.factory import safepath


def _root(tmp_path):
    root = tmp_path / "run"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    return root


def test_a_path_inside_the_root_resolves_and_is_returned(tmp_path):
    root = _root(tmp_path)
    got = safepath.contained(root / "sub" / "a.json", root)
    assert got == (root / "sub" / "a.json").resolve()
    assert safepath.regular_file(root / "sub" / "a.json", root) == got
    # a path that does not exist yet is fine for a write target
    assert safepath.contained(root / "sub" / "new.json", root).name == "new.json"


@pytest.mark.parametrize("rel", ["../outside.txt", "sub/../../outside.txt"])
def test_dot_dot_traversal_out_of_the_root_is_refused(tmp_path, rel):
    root = _root(tmp_path)
    with pytest.raises(CliError) as exc:
        safepath.contained(root / rel, root)
    assert exc.value.code == EXIT_USER_ERROR
    assert "escapes" in exc.value.message


def test_an_absolute_path_outside_the_root_is_refused(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(CliError, match="escapes"):
        safepath.contained(tmp_path / "outside.txt", root)
    with pytest.raises(CliError, match="escapes"):
        safepath.regular_file(str((tmp_path / "outside.txt").resolve()), root)


def test_a_symlink_escaping_the_root_is_refused(tmp_path):
    root = _root(tmp_path)
    (root / "link.txt").symlink_to(tmp_path / "outside.txt")
    (root / "linkdir").symlink_to(tmp_path)
    with pytest.raises(CliError, match="escapes"):
        safepath.contained(root / "link.txt", root)
    with pytest.raises(CliError, match="escapes"):
        safepath.contained(root / "linkdir" / "outside.txt", root)


def test_regular_file_refuses_a_symlink_even_inside_the_root(tmp_path):
    root = _root(tmp_path)
    (root / "inner-link.json").symlink_to(root / "sub" / "a.json")
    assert safepath.contained(root / "inner-link.json", root)  # contained, but a link
    with pytest.raises(CliError, match="symlink"):
        safepath.regular_file(root / "inner-link.json", root)


def test_regular_file_refuses_a_missing_file_and_a_directory(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(CliError, match="not a regular file"):
        safepath.regular_file(root / "sub" / "missing.json", root)
    with pytest.raises(CliError, match="not a regular file"):
        safepath.regular_file(root / "sub", root)


def test_a_nul_byte_is_refused(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(CliError, match="NUL"):
        safepath.contained(f"{root}/sub/a.json\x00.txt", root)
    with pytest.raises(CliError, match="NUL"):
        safepath.contained(root / "sub" / "a.json", f"{root}\x00")


def test_a_relative_path_resolves_against_the_working_directory(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.chdir(root)
    assert safepath.contained(os.path.join("sub", "a.json"), root) == root.resolve() / "sub" / (
        "a.json"
    )
    with pytest.raises(CliError, match="escapes"):
        safepath.contained(os.path.join("..", "outside.txt"), root)
