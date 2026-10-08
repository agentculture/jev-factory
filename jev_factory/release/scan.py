"""Pre-upload bundle scan: credentials, endpoints, private hosts, redactable text.

Before a bundle is pushed to the hub this module scans the folder for leaked
credentials, non-localhost endpoints in JSON, private network addresses and
anything the small redactor below would redact, and writes a ``scan.json``
that :func:`verify` (and the upload step) check is clean and still matches the
folder. ``.json`` / ``.jsonl`` files are also parsed and every decoded string
value scanned, so an escape sequence hiding a credential from the raw scan does
not slip through. Weight binaries (``*.safetensors``, ``*.gguf``) are listed
under ``binaries`` instead of decoded as text, but only once their contents
prove the format; any other non-UTF-8 file is an ``unscanned_binary`` finding.

The credential and endpoint patterns are this repository's own, the ones
``scripts/scan-secrets.py`` applies to tracked files (factored here so the
package never imports a script by path).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/scan_bundle.py",
    "commit": "9debdc6",
    "adaptations": [
        "nvsh.redact.redact_report -> the small redactor in this module",
        "scan-secrets.py loaded by path -> its credential/endpoint patterns factored in here",
        "scan_folder/write_scan no longer take a scan_secrets module argument",
        "check_no_symlinks moved here from hub_upload.py (bundle + hub both refuse links)",
        "the argparse main() is dropped; scan/verify are called by bundle and hub code",
        "d17 (2026-10-08): restructured for SonarCloud code quality (cognitive "
        "complexity split into private helpers, plus lint-level cleanups); behaviour "
        "unchanged, pinned by tests/test_complexity_*.py and differential checks against "
        "the imported version",
    ],
    "licence": "Apache-2.0",
}

SCAN_FILE = "scan.json"

# ---------------------------------------------------------------------------
# Credentials and endpoints (same shapes as scripts/scan-secrets.py)
# ---------------------------------------------------------------------------

_KNOWN_TOKEN_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"gh[pousr]_[A-Za-z0-9]{36,}", "GitHub token"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style secret key"),
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----", "private key block"),
]

_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
        \b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|token|password|passwd|pwd)
        ["']?
        \s*[:=]\s*
        (?:
            "(?P<dq>[^"\n]{20,})"
          | '(?P<sq>[^'\n]{20,})'
          | (?P<bare>[^\s"'`,;)\]}]{20,})
        )
    """,
)

_PLACEHOLDER_RE = re.compile(
    r"^(?:\$\{?\{?|secrets\.|github\.|<.*>?$|\.\.\.$)",
    re.IGNORECASE,
)
_PLACEHOLDER_WORDS = frozenset(
    {"changeme", "your", "example", "redacted", "dummy", "fake", "placeholder", "sample", "todo"}
)
_MASK_RE = re.compile(r"(?i)(x{8,}|0{8,})")
_WORDY_RE = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")

_ENDPOINT_KEY_RE = re.compile(r"(?i)^(base[_-]?url|endpoint|url|host)$")
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}  # nosec -- an allow-list, no bind


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    detail: str


def _is_placeholder(value: str) -> bool:
    if _PLACEHOLDER_RE.match(value):
        return True
    if not _WORDY_RE.match(value):
        return False
    tokens = set(re.split(r"[-_.]+", value))
    return bool(tokens & _PLACEHOLDER_WORDS) or bool(_MASK_RE.search(value))


def scan_credentials(path: str, text: str) -> list[Finding]:
    """Credential-shaped strings in *text* (known token formats, secret-ish assignments)."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, label in _KNOWN_TOKEN_PATTERNS:
            if re.search(pattern, line):
                findings.append(Finding(path, lineno, "credential", label))
        for match in _ASSIGNMENT_RE.finditer(line):
            value = match.group("dq") or match.group("sq") or match.group("bare") or ""
            if _is_placeholder(value):
                continue
            findings.append(
                Finding(
                    path,
                    lineno,
                    "credential",
                    f"{match.group(1)}-shaped assignment with a literal value",
                )
            )
    return findings


def _endpoint_host(value: str) -> str | None:
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    return parts.hostname or None


def _walk_json(obj, lineage: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.extend(_walk_json(value, key))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_json(item, lineage))
    elif isinstance(obj, str):
        out.append((lineage, obj))
    return out


def scan_endpoints(path: str, text: str) -> list[Finding]:
    """Non-localhost ``http(s)`` endpoints under url-shaped keys of a JSON document."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    findings: list[Finding] = []
    for key, value in _walk_json(data):
        if not _ENDPOINT_KEY_RE.match(key):
            continue
        host = _endpoint_host(value)
        if host is None or host in _ALLOWED_HOSTS:
            continue
        findings.append(Finding(path, 1, "endpoint", f"{key}={value!r} is not localhost"))
    return findings


# ---------------------------------------------------------------------------
# The small redactor (replaces nvsh.redact)
# ---------------------------------------------------------------------------

_REDACT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
            r".*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "json_secret_field",
        re.compile(r'(?i)"(apiKey|api_key|token|secret|password)"\s*:\s*"([^"]{8,})"'),
    ),
    ("authorization_header", re.compile(r"(?i)\b(Authorization:\s*(?:Bearer|Basic))\s+(\S{8,})")),
    ("cli_flag_secret", re.compile(r"(--api-key|--token)(=|\s+)(\S{8,})")),
    ("url_credentials", re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^\s:/@]+):([^\s@]+)@")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{8,}\b")),
    ("slack_token", re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}\b")),
]


def redact_report(data: bytes) -> tuple[bytes, list[str]]:
    """Redact *data*; return the redacted bytes and the rule names that fired.

    Never raises: invalid UTF-8 round-trips through ``surrogateescape``."""
    text = data.decode("utf-8", errors="surrogateescape")
    fired: list[str] = []
    for name, pattern in _REDACT_RULES:
        text, count = pattern.subn(f"<REDACTED:{name}>", text)
        if count:
            fired.append(name)
    return text.encode("utf-8", errors="surrogateescape"), fired


def redact(data: bytes) -> bytes:
    """*data* with every secret-shaped span replaced by a typed marker."""
    return redact_report(data)[0]


# ---------------------------------------------------------------------------
# Folder hashing, symlinks, private hosts
# ---------------------------------------------------------------------------


def check_no_symlinks(bundle: Path) -> list[str]:
    """Relative paths of every symlink in *bundle* (scanning and uploading follow links,
    so one pointing outside the folder would ship what it points at)."""
    return sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_symlink())


def folder_hash(folder: Path) -> str:
    """A stable SHA-256 over every regular file under *folder*, excluding ``scan.json``."""
    digest = hashlib.sha256()
    for path in sorted(
        (f for f in folder.rglob("*") if f.is_file()),
        key=lambda p: p.relative_to(folder).as_posix(),
    ):
        rel = path.relative_to(folder).as_posix()
        if rel == SCAN_FILE:
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\x00")
    return digest.hexdigest()


_IPV4_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
_PRIVATE_NAME_RE = re.compile(
    r"(?<![\w.-])([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:local|lan|internal|home))\b", re.I
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_DOCUMENTATION = tuple(
    ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)

_EXPECTED_BINARY_EXTS = {".safetensors", ".gguf"}
_GGUF_MAGIC = b"GGUF"
_SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024


def private_hosts(text: str) -> list[tuple[int, str]]:
    """``(line, host)`` for each private address or private-suffix host name in *text*."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found.extend(
            (number, match.group(1))
            for match in _IPV4_RE.finditer(line)
            if _is_private_address(match.group(1))
        )
        found.extend((number, match.group(1)) for match in _PRIVATE_NAME_RE.finditer(line))
    return found


def _is_private_address(text: str) -> bool:
    """Whether dotted-quad *text* is a private or CGNAT address (not loopback, unspecified
    or an RFC 5737 documentation address); an invalid address is not one."""
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback or address.is_unspecified:
        return False
    if any(address in net for net in _DOCUMENTATION):
        return False
    return address.is_private or address in _CGNAT


# ---------------------------------------------------------------------------
# Weight files
# ---------------------------------------------------------------------------


def _safetensors_header(path: Path) -> dict | None:
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            (size,) = struct.unpack("<Q", prefix)
            if size > min(_SAFETENSORS_MAX_HEADER, path.stat().st_size - 8):
                return None
            header = json.loads(handle.read(size).decode("utf-8"))
    except (OSError, ValueError):  # UnicodeDecodeError is a ValueError
        return None
    return header if isinstance(header, dict) else None


def _has_gguf_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_GGUF_MAGIC)) == _GGUF_MAGIC
    except OSError:
        return False


def _scan_weight_file(rel: str, path: Path) -> list[dict]:
    if path.suffix.lower() == ".gguf":
        if _has_gguf_magic(path):
            return []
        detail = "named .gguf but has no GGUF magic"
        return [{"path": rel, "line": 0, "kind": "unrecognised_binary", "detail": detail}]
    header = _safetensors_header(path)
    if header is None:
        detail = "named .safetensors but has no valid safetensors header"
        return [{"path": rel, "line": 0, "kind": "unrecognised_binary", "detail": detail}]
    findings: list[dict] = []
    for fragment in _iter_json_strings(header):
        findings.extend(_scan_fragment(rel, 0, fragment))
    return findings


def _iter_json_strings(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_json_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_json_strings(item)


def _as_dict(finding: Finding, line: int) -> dict:
    return {"path": finding.path, "line": line, "kind": finding.kind, "detail": finding.detail}


def _scan_fragment(rel: str, base_line: int, fragment: str) -> list[dict]:
    """Credential / private-host / redact checks on one decoded JSON string."""
    findings = [_as_dict(f, base_line + f.line - 1) for f in scan_credentials(rel, fragment)]
    for offset, host in private_hosts(fragment):
        findings.append(
            {"path": rel, "line": base_line + offset - 1, "kind": "private_host", "detail": host}
        )
    _, fired_rules = redact_report(fragment.encode("utf-8"))
    for rule_name in fired_rules:
        findings.append({"path": rel, "line": base_line, "kind": "redact", "detail": rule_name})
    return findings


def _scan_json_strings(rel: str, suffix: str, text: str) -> list[dict]:
    if suffix == ".json":
        return _scan_json_document(rel, text)
    if suffix == ".jsonl":
        return _scan_json_lines(rel, text)
    return []


def _scan_json_document(rel: str, text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except ValueError:
        return []
    findings: list[dict] = []
    for fragment in _iter_json_strings(data):
        findings.extend(_scan_fragment(rel, 1, fragment))
    return findings


def _scan_json_lines(rel: str, text: str) -> list[dict]:
    findings: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        for fragment in _iter_json_strings(record):
            findings.extend(_scan_fragment(rel, lineno, fragment))
    return findings


def list_binaries(folder: Path) -> list[str]:
    """Relative POSIX paths of the expected weight-file binaries under *folder*."""
    return sorted(
        f.relative_to(folder).as_posix()
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in _EXPECTED_BINARY_EXTS
    )


def scan_folder(folder: Path) -> list[dict]:
    """One finding dict (``path``, ``line``, ``kind``, ``detail``) per issue found."""
    findings: list[dict] = []
    for path in sorted(
        (f for f in folder.rglob("*") if f.is_file()),
        key=lambda p: p.relative_to(folder).as_posix(),
    ):
        rel = path.relative_to(folder).as_posix()
        if rel == SCAN_FILE:
            continue
        if path.suffix.lower() in _EXPECTED_BINARY_EXTS:
            findings.extend(_scan_weight_file(rel, path))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(
                {"path": rel, "line": 0, "kind": "unscanned_binary", "detail": "non-UTF-8 file"}
            )
            continue
        except OSError:
            continue
        for finding in (*scan_credentials(rel, text), *scan_endpoints(rel, text)):
            findings.append(_as_dict(finding, finding.line))
        for line, host in private_hosts(text):
            findings.append({"path": rel, "line": line, "kind": "private_host", "detail": host})
        _, fired_rules = redact_report(path.read_bytes())
        for rule_name in fired_rules:
            findings.append({"path": rel, "line": 0, "kind": "redact", "detail": rule_name})
        findings.extend(_scan_json_strings(rel, path.suffix.lower(), text))
    return findings


def write_scan(folder: Path) -> dict:
    """Compute findings, hash and binaries, write ``scan.json``, return the payload."""
    findings = scan_folder(folder)
    payload = {
        "hash": folder_hash(folder),
        "findings": findings,
        "clean": not findings,
        "binaries": list_binaries(folder),
    }
    (folder / SCAN_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify(folder: Path) -> str | None:
    """``None`` when the folder is clean and unmodified since its scan, else a reason."""
    scan_path = folder / SCAN_FILE
    if not scan_path.is_file():
        return "no scan.json"
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    if not payload.get("clean"):
        return "scan has findings"
    if payload.get("hash") != folder_hash(folder):
        return "folder changed after scan"
    return None
