"""Student input pins: revision and sha256 verification, fail closed (distill t7)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jev_factory.cli._errors import CliError
from jev_factory.distill import pins

ADAPTER = b"adapter-bytes"
HEAD = b"head-bytes-0123456789"
BASE_REV = "b1485b2f" + "0" * 32
DECIDER_REV = "c0ffee11" + "1" * 32


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_pins() -> pins.StudentPins:
    return pins.StudentPins(
        base=pins.RepoPin(pins.BASE_REPO, BASE_REV),
        decider=pins.RepoPin(pins.DECIDER_REPO, DECIDER_REV),
        adapter_sha256=_sha(ADAPTER),
        head_sha256=_sha(HEAD),
    )


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    base = tmp_path / pins.cache_dirname(pins.BASE_REPO)
    base.mkdir()
    (base / pins.REVISION_FILE).write_text(BASE_REV + "\n")
    dec = tmp_path / pins.cache_dirname(pins.DECIDER_REPO)
    dec.mkdir()
    (dec / pins.REVISION_FILE).write_text(DECIDER_REV + "\n")
    (dec / "lora").mkdir()
    (dec / "lora" / "adapter_model.safetensors").write_bytes(ADAPTER)
    (dec / "head.safetensors").write_bytes(HEAD)
    (dec / "provenance.json").write_text(
        json.dumps(
            {
                "adapter_sha256": _sha(ADAPTER),
                "head_sha256": _sha(HEAD),
            }
        )
    )
    return tmp_path


def test_verifies_clean_cache(cache: Path) -> None:
    result = pins.verify_pins(cache, _run_pins())
    assert result["status"] == "ok"
    assert result["head_sha256"] == _sha(HEAD)
    assert result["adapter_sha256"] == _sha(ADAPTER)


def test_flipped_head_byte_fails_closed_naming_file(cache: Path) -> None:
    p = cache / pins.cache_dirname(pins.DECIDER_REPO) / "head.safetensors"
    data = bytearray(p.read_bytes())
    data[0] ^= 0x01
    p.write_bytes(bytes(data))
    run_pins = _run_pins()
    with pytest.raises(CliError) as exc:
        pins.verify_pins(cache, run_pins)
    assert "hash-mismatch" in exc.value.message
    assert "head.safetensors" in exc.value.message


def test_flipped_adapter_byte_fails(cache: Path) -> None:
    p = cache / pins.cache_dirname(pins.DECIDER_REPO) / "lora" / "adapter_model.safetensors"
    p.write_bytes(ADAPTER + b"x")
    run_pins = _run_pins()
    with pytest.raises(CliError, match="adapter_model.safetensors"):
        pins.verify_pins(cache, run_pins)


def test_wrong_base_revision_fails(cache: Path) -> None:
    f = cache / pins.cache_dirname(pins.BASE_REPO) / pins.REVISION_FILE
    f.write_text("deadbeef" + "2" * 32)
    run_pins = _run_pins()
    with pytest.raises(CliError, match="revision-mismatch"):
        pins.verify_pins(cache, run_pins)


def test_wrong_decider_revision_fails(cache: Path) -> None:
    f = cache / pins.cache_dirname(pins.DECIDER_REPO) / pins.REVISION_FILE
    f.write_text("deadbeef" + "2" * 32)
    run_pins = _run_pins()
    with pytest.raises(CliError, match="revision-mismatch"):
        pins.verify_pins(cache, run_pins)


def test_provenance_disagreeing_with_run_pin_fails(cache: Path) -> None:
    prov = cache / pins.cache_dirname(pins.DECIDER_REPO) / "provenance.json"
    prov.write_text(json.dumps({"adapter_sha256": "f" * 64, "head_sha256": _sha(HEAD)}))
    run_pins = _run_pins()
    with pytest.raises(CliError, match="provenance"):
        pins.verify_pins(cache, run_pins)


def test_missing_file_and_missing_provenance_fail_closed(cache: Path) -> None:
    dec = cache / pins.cache_dirname(pins.DECIDER_REPO)
    (dec / "provenance.json").unlink()
    run_pins = _run_pins()
    with pytest.raises(CliError, match="provenance.json"):
        pins.verify_pins(cache, run_pins)


def test_unpinned_revision_is_rejected() -> None:
    with pytest.raises(CliError, match="pin"):
        pins.RepoPin(pins.BASE_REPO, "main")


def test_default_pins_carry_full_2610_digests_and_base_revision() -> None:
    """Criterion 1/2: repo, revision pin and digests are the 2610 values from provenance.json."""
    d = pins.default_pins(decider_revision=pins.DECIDER_REVISION)
    assert pins.DECIDER_REPO == "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
    assert d.decider.revision == "216d1ccb46a9e9b036eb9187b248db98480f77d4"
    assert d.adapter_sha256 == "ae1d0cd1ac3edec6df0a519a3ad18f1c382099032d493f8f98c96a80cfebb4c7"
    assert d.head_sha256 == "ace56742545b505e6fae34ada85c8edbeedb23ba8974d8680aebe0380d6083a1"
    assert d.base.revision == "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
    assert pins.default_pins(
        decider_revision=DECIDER_REV, base_revision=BASE_REV
    ).base.revision == (BASE_REV)


def test_no_v21_left_in_pins_module() -> None:
    """Criterion 2: no 'v21' string and no V21_ names remain in pins.py."""
    text = Path(pins.__file__).read_text().lower()
    assert "v21" not in text
    assert not [n for n in dir(pins) if n.startswith("V21_")]


def _cache_with_2610_digests(tmp_path: Path) -> Path:
    base = tmp_path / pins.cache_dirname(pins.BASE_REPO)
    base.mkdir()
    (base / pins.REVISION_FILE).write_text(pins.DEFAULT_BASE_REVISION)
    dec = tmp_path / pins.cache_dirname(pins.DECIDER_REPO)
    (dec / "lora").mkdir(parents=True)
    (dec / pins.REVISION_FILE).write_text(pins.DECIDER_REVISION)
    (dec / pins.ADAPTER_FILE).write_bytes(ADAPTER)
    (dec / pins.HEAD_FILE).write_bytes(HEAD)
    return dec


def test_verify_pins_with_2610_digests_passes_and_v21_digest_fails_closed(tmp_path: Path) -> None:
    """Criterion 2: a cache whose provenance carries the 2610 digests verifies; a v21 one fails."""
    dec = _cache_with_2610_digests(tmp_path)
    good = pins.StudentPins(
        base=pins.RepoPin(pins.BASE_REPO, pins.DEFAULT_BASE_REVISION),
        decider=pins.RepoPin(pins.DECIDER_REPO, pins.DECIDER_REVISION),
        adapter_sha256=_sha(ADAPTER),
        head_sha256=_sha(HEAD),
    )
    prov = {"adapter_sha256": _sha(ADAPTER), "head_sha256": _sha(HEAD)}
    (dec / pins.PROVENANCE_FILE).write_text(json.dumps(prov))
    assert pins.verify_pins(tmp_path, good)["status"] == "ok"
    # Built-in 2610 pins reject files that do not hash to the 2610 digests.
    builtin = pins.default_pins(decider_revision=pins.DECIDER_REVISION)
    with pytest.raises(CliError, match="provenance"):
        pins.verify_pins(tmp_path, builtin)
    # A provenance carrying the old v21 digests disagrees with the 2610 pins.
    v21 = {
        "adapter_sha256": "59be987f5eb664a7f74f11ef69383a0de526fe30c684932102984eb9c11f4028",
        "head_sha256": "0fc78684d7504d334082d6ac8e6cb7825b5eb0e230601181cd273757f9f6ebef",
    }
    (dec / pins.PROVENANCE_FILE).write_text(json.dumps(v21))
    with pytest.raises(CliError, match="provenance"):
        pins.verify_pins(tmp_path, builtin)


def test_decider_revision_is_required() -> None:
    with pytest.raises(TypeError):
        pins.default_pins()  # type: ignore[call-arg]


def test_prefix_only_adapter_or_head_pin_is_rejected() -> None:
    for kw in ({"adapter_sha256": "59be987f"}, {"head_sha256": "0fc78684"}):
        args = {
            "base": pins.RepoPin(pins.BASE_REPO, BASE_REV),
            "decider": pins.RepoPin(pins.DECIDER_REPO, DECIDER_REV),
            "adapter_sha256": _sha(ADAPTER),
            "head_sha256": _sha(HEAD),
            **kw,
        }
        with pytest.raises(CliError, match="64-hex"):
            pins.StudentPins(**args)


def test_adapter_lives_under_lora_not_at_the_repo_root(cache: Path) -> None:
    assert pins.ADAPTER_FILE == "lora/adapter_model.safetensors"
    dec = cache / pins.cache_dirname(pins.DECIDER_REPO)
    (dec / "adapter_model.safetensors").write_bytes(b"decoy")  # a root file is never read
    assert pins.verify_pins(cache, _run_pins())["status"] == "ok"
    (dec / "lora" / "adapter_model.safetensors").unlink()
    run_pins = _run_pins()
    with pytest.raises(CliError, match="lora/adapter_model.safetensors"):
        pins.verify_pins(cache, run_pins)
