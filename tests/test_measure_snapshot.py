"""The fixed grounding snapshot: jev_factory/measure/snapshot.py.

Ported from nvsh's snapshot tests (tests/test_lfm_finetune_measure.py) onto
the toy lamp domain, whose world schema is ``rooms`` (a grounded list) and
``home`` (a string).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.domain.model import GroundKind
from jev_factory.measure import snapshot as snap
from tests.fixtures.toy_domain import DOMAIN

BASE = {"home": "toy-home", "rooms": ["kitchen", "study"], "comment": "not in the schema"}


def _split(tmp_path: Path, name: str, rooms: list[str]) -> Path:
    entries = [
        {
            "id": f"{name}-{i}",
            "kind": "explicit",
            "text": "x",
            "expect": {"operation": "lamp_on", "args": {"room": room}},
        }
        for i, room in enumerate(rooms)
    ]
    entries.append({"id": "esc", "kind": "explicit", "text": "y", "expect": {"escalate": True}})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"header": "h", "entries": entries}))
    return path


def _live_domain(values=("livingroom", "kitchen"), fail=False):
    def lookup():
        if fail:
            raise OSError("no controller")
        return list(values)

    kind = replace(DOMAIN.ground_kinds[0], lookup=lookup)
    return replace(DOMAIN, ground_kinds=(kind,))


def test_builder_merges_live_and_split_values_and_counts_only(tmp_path):
    splits = [_split(tmp_path, "val", ["cellar"]), _split(tmp_path, "test", ["kitchen", "attic"])]
    snapshot, counts = snap.build_snapshot(
        _live_domain(), splits, source="fixture", created="2026-09-30", base_world=BASE
    )
    assert snapshot == {
        "home": "toy-home",
        "rooms": ["attic", "cellar", "kitchen", "livingroom"],
        "source": "fixture",
        "created": "2026-09-30",
    }
    assert counts == {"rooms": 4, "rooms_live": 2, "rooms_splits_only": 2}
    assert DOMAIN.check_world(snapshot) == ()


def test_a_kind_without_a_lookup_takes_the_base_world(tmp_path):
    snapshot, counts = snap.build_snapshot(
        DOMAIN, [_split(tmp_path, "val", ["cellar"])], source="s", created="c", base_world=BASE
    )
    assert snapshot["rooms"] == ["cellar", "kitchen", "study"]
    assert counts["rooms_live"] == 2


def test_a_failed_lookup_is_an_environment_error(tmp_path):
    with pytest.raises(snap.SnapshotError) as excinfo:
        snap.build_snapshot(_live_domain(fail=True), [], source="s", created="c", base_world=BASE)
    assert excinfo.value.env


def test_a_missing_schema_field_is_refused(tmp_path):
    with pytest.raises(snap.SnapshotError, match="home"):
        snap.build_snapshot(DOMAIN, [], source="s", created="c", base_world={"rooms": []})


def test_the_world_spelling_uses_the_kind_suffix():
    kind = GroundKind(
        name="svc", noun="service", plural="services", world_field="s", suffixes=(".service",)
    )
    assert snap.world_spelling(kind, "nginx", []) == "nginx.service"
    assert snap.world_spelling(kind, "NGINX", ["nginx.service"]) == "nginx.service"
    assert snap.world_spelling(kind, "x.service", []) == "x.service"


def test_load_returns_the_snapshot_and_its_hash(tmp_path):
    path = tmp_path / "snap.json"
    payload = {"home": "h", "rooms": ["kitchen"], "source": "s", "created": "c"}
    path.write_text(json.dumps(payload))
    loaded, digest = snap.load_snapshot(path, DOMAIN)
    assert loaded == payload
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        {"home": "h", "rooms": "kitchen", "source": "s", "created": "c"},
        {"home": "h", "rooms": ["kitchen"], "created": "c"},
        {"rooms": ["kitchen"], "source": "s", "created": "c"},
        ["not", "an", "object"],
    ],
)
def test_a_malformed_snapshot_is_refused(tmp_path, payload):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(snap.SnapshotError):
        snap.load_snapshot(path, DOMAIN)


def test_cli_writes_the_snapshot_and_prints_counts_only(tmp_path, capsys):
    split = _split(tmp_path, "val", ["secretroom"])
    out = tmp_path / "snap.json"
    argv = ["--domain", "tests.fixtures.toy_domain", "--out", str(out), "--from-split", str(split)]
    assert snap.main(argv, today=lambda: "2026-09-30") == 0
    written = json.loads(out.read_text())
    assert "secretroom" in written["rooms"]
    assert written["created"] == "2026-09-30"
    printed = capsys.readouterr()
    for value in ("secretroom", "kitchen", "hallway"):
        assert value not in printed.out + printed.err
    assert "5 rooms (4 from this machine, 1 only from the split files)" in printed.out
    assert snap.main(argv, today=lambda: "x") == 1  # exists: --force needed
    assert snap.main([*argv, "--force"], today=lambda: "x") == 0
