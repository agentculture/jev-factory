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


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    base = tmp_path / pins.cache_dirname(pins.BASE_REPO)
    base.mkdir()
    (base / pins.REVISION_FILE).write_text(BASE_REV + "\n")
    dec = tmp_path / pins.cache_dirname(pins.DECIDER_REPO)
    dec.mkdir()
    (dec / pins.REVISION_FILE).write_text(DECIDER_REV + "\n")
    (dec / "adapter_model.safetensors").write_bytes(ADAPTER)
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
    with pytest.raises(CliError) as exc:
        pins.verify_pins(cache, _run_pins())
    assert "hash-mismatch" in exc.value.message
    assert "head.safetensors" in exc.value.message


def test_flipped_adapter_byte_fails(cache: Path) -> None:
    p = cache / pins.cache_dirname(pins.DECIDER_REPO) / "adapter_model.safetensors"
    p.write_bytes(ADAPTER + b"x")
    with pytest.raises(CliError, match="adapter_model.safetensors"):
        pins.verify_pins(cache, _run_pins())


def test_wrong_base_revision_fails(cache: Path) -> None:
    f = cache / pins.cache_dirname(pins.BASE_REPO) / pins.REVISION_FILE
    f.write_text("deadbeef" + "2" * 32)
    with pytest.raises(CliError, match="revision-mismatch"):
        pins.verify_pins(cache, _run_pins())


def test_wrong_decider_revision_fails(cache: Path) -> None:
    f = cache / pins.cache_dirname(pins.DECIDER_REPO) / pins.REVISION_FILE
    f.write_text("deadbeef" + "2" * 32)
    with pytest.raises(CliError, match="revision-mismatch"):
        pins.verify_pins(cache, _run_pins())


def test_provenance_disagreeing_with_run_pin_fails(cache: Path) -> None:
    prov = cache / pins.cache_dirname(pins.DECIDER_REPO) / "provenance.json"
    prov.write_text(json.dumps({"adapter_sha256": "f" * 64, "head_sha256": _sha(HEAD)}))
    with pytest.raises(CliError, match="provenance"):
        pins.verify_pins(cache, _run_pins())


def test_missing_file_and_missing_provenance_fail_closed(cache: Path) -> None:
    dec = cache / pins.cache_dirname(pins.DECIDER_REPO)
    (dec / "provenance.json").unlink()
    with pytest.raises(CliError, match="provenance.json"):
        pins.verify_pins(cache, _run_pins())


def test_unpinned_revision_is_rejected() -> None:
    with pytest.raises(CliError, match="pin"):
        pins.RepoPin(pins.BASE_REPO, "main")


def test_default_pins_carry_full_v21_digests_and_base_revision() -> None:
    d = pins.default_pins(decider_revision=DECIDER_REV)
    assert d.adapter_sha256 == "59be987f5eb664a7f74f11ef69383a0de526fe30c684932102984eb9c11f4028"
    assert d.head_sha256 == "0fc78684d7504d334082d6ac8e6cb7825b5eb0e230601181cd273757f9f6ebef"
    assert d.base.revision == "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
    assert pins.default_pins(
        decider_revision=DECIDER_REV, base_revision=BASE_REV
    ).base.revision == (BASE_REV)


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
