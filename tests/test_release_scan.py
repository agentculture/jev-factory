"""release/scan.py: the pre-upload bundle scan and its small redactor."""

from __future__ import annotations

import json

from jev_factory.release import scan

_SAFETENSORS = (2).to_bytes(8, "little") + b"{}"
# Secret-shaped strings are built at run time so this file is not itself a finding.
_SK = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1"
_AKIA = "AKIA" + "IOSFODNN7EXAMPLQ"


def _folder(tmp_path, **files):
    folder = tmp_path / "b"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    return folder


def test_clean_folder_scans_clean_and_verifies(tmp_path):
    folder = _folder(tmp_path, **{"README.md": "# hello\n", "m.safetensors": _SAFETENSORS})
    payload = scan.write_scan(folder)
    assert payload["clean"] is True
    assert payload["binaries"] == ["m.safetensors"]
    assert scan.verify(folder) is None


def test_credentials_hosts_and_redactions_are_findings(tmp_path):
    body = f"key {_SK}\naws {_AKIA}\nhost 192.168.1.50 and box.internal\n"
    folder = _folder(tmp_path, **{"notes.md": body})
    kinds = {f["kind"] for f in scan.scan_folder(folder)}
    assert {"credential", "private_host", "redact"} <= kinds
    assert scan.write_scan(folder)["clean"] is False
    assert scan.verify(folder) == "scan has findings"


def test_documentation_and_loopback_addresses_are_allowed(tmp_path):
    folder = _folder(tmp_path, **{"a.md": "see 192.0.2.7 and 127.0.0.1\n"})
    assert scan.scan_folder(folder) == []


def test_escaped_secret_inside_json_is_found(tmp_path):
    escaped = json.dumps({"note": _SK}).replace("sk-", "\\u0073k-")
    folder = _folder(tmp_path, **{"d.json": escaped})
    assert any(f["kind"] == "credential" for f in scan.scan_folder(folder))


def test_non_localhost_endpoint_in_json_is_found(tmp_path):
    folder = _folder(tmp_path, **{"c.json": json.dumps({"base_url": "https://example.org/v1"})})
    assert any(f["kind"] == "endpoint" for f in scan.scan_folder(folder))


def test_unknown_binary_and_fake_weights_are_findings(tmp_path):
    folder = _folder(
        tmp_path, **{"x.bin": b"\xff\xfe\x00", "bad.gguf": b"nope", "e.safetensors": b"x"}
    )
    kinds = sorted(f["kind"] for f in scan.scan_folder(folder))
    assert kinds == ["unrecognised_binary", "unrecognised_binary", "unscanned_binary"]


def test_a_changed_file_fails_verify(tmp_path):
    folder = _folder(tmp_path, **{"a.md": "x"})
    scan.write_scan(folder)
    (folder / "a.md").write_text("y")
    assert scan.verify(folder) == "folder changed after scan"
    assert scan.verify(tmp_path) == "no scan.json"


def test_redact_is_idempotent_and_typed():
    raw = f"Authorization: Bearer abcdefgh12345 and {_SK}".encode()
    once = scan.redact(raw)
    assert b"abcdefgh12345" not in once
    assert _SK.encode() not in once
    assert b"<REDACTED:" in once
    assert scan.redact(once) == once


def test_symlinks_are_listed(tmp_path):
    folder = _folder(tmp_path, **{"a.md": "x"})
    (folder / "link").symlink_to(tmp_path / "a-target")
    assert scan.check_no_symlinks(folder) == ["link"]
