"""Append-only, schema-validated decision records for ``jev decide``.

Every between-runs decision (ship a candidate, more epochs, targeted augment,
heal the quant, ...) is written as one JSON object per line to a record store
(``decisions.jsonl``). The store is **append-only**:

* a new record is appended with ``O_APPEND`` under an exclusive ``flock``; the
  bytes of every earlier record are never rewritten;
* each record carries ``prev_sha256``, the sha256 of the whole store as it was
  before the record was appended, so :func:`load` detects any later edit of an
  earlier line (the chain breaks);
* a human override is a **new** record (``decider: human``) that names the
  record it overrides and the deviation id that authorises it. It is never an
  edit of the record it overrides.

A record cites the metric values it used, each with the sha256 of the file the
value came from, plus the rule version (and the pre-registration sha256) that
produced it. These records are the corpus the Layer 3 decider learns from, so
they are validated on write and on read. The shape mirrors nvsh's D-style
decision path (evidence, choice, who decided, record id), made machine
readable.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

SCHEMA_VERSION = 1
DEFAULT_NAME = "decisions.jsonl"

#: The minimum verdict set (issue #1 Layer 2). ``rules.VERDICTS`` is the same set.
VERDICTS: tuple[str, ...] = (
    "ship_candidate",
    "more_epochs",
    "fewer_epochs",
    "targeted_augment",
    "fix_grader",
    "recalibrate",
    "heal_quant",
    "refit_gate",
    "stop_escalate",
)
DECIDERS: tuple[str, ...] = ("rule", "model", "human")

_REQUIRED = (
    "schema_version",
    "id",
    "created",
    "run",
    "verdict",
    "params",
    "reasons",
    "cited",
    "rule_version",
    "decider",
    "prev_sha256",
)
_OPTIONAL = ("prereg_sha256", "decider_ref", "overrides", "deviation_id", "details")
_CITED_KEYS = {"name", "value", "source", "sha256"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^D[1-9]\d*$", re.ASCII)


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(
        EXIT_USER_ERROR,
        message,
        remediation or "decision records are append-only; see `jev explain decide`",
    )


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cite(name: str, value: float, source: Path | str, sha256: str | None = None) -> dict[str, Any]:
    """A cited metric: its value and the sha256 of the file it was read from."""
    digest = sha256 if sha256 is not None else sha256_file(Path(source))
    return {"name": name, "value": value, "source": str(source), "sha256": digest}


def _cited_errors(cited: Any) -> list[str]:
    if not isinstance(cited, list) or not cited:
        return ["cited must be a non-empty list of metric citations"]
    errs: list[str] = []
    for i, c in enumerate(cited):
        where = f"cited[{i}]"
        if not isinstance(c, dict) or set(c) != _CITED_KEYS:
            errs.append(f"{where} must have exactly the keys {sorted(_CITED_KEYS)}")
            continue
        if not _nonempty_str(c["name"]):
            errs.append(f"{where}.name must be a non-empty string")
        if not (_is_num(c["value"]) or c["value"] is None):
            errs.append(f"{where}.value must be a finite number or null")
        if not _nonempty_str(c["source"]):
            errs.append(f"{where}.source must be a non-empty string")
        if not (isinstance(c["sha256"], str) and _HEX64.match(c["sha256"])):
            errs.append(f"{where}.sha256 must be 64 lowercase hex characters")
    return errs


def validate_record(rec: Any) -> list[str]:
    """Return every schema problem with *rec* (empty when it is valid)."""
    if not isinstance(rec, dict):
        return ["a record must be a JSON object"]
    errs: list[str] = []
    missing = [k for k in _REQUIRED if k not in rec]
    if missing:
        errs.append(f"missing fields: {', '.join(missing)}")
    unknown = sorted(set(rec) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        errs.append(f"unknown fields: {', '.join(unknown)}")
    if "schema_version" in rec and rec["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    if "id" in rec and not (isinstance(rec["id"], str) and _ID.match(rec["id"])):
        errs.append("id must look like D<n>")
    if "created" in rec and not _nonempty_str(rec["created"]):
        errs.append("created must be an ISO-8601 timestamp string")
    if "run" in rec and not _nonempty_str(rec["run"]):
        errs.append("run must be a non-empty string")
    if "verdict" in rec and rec["verdict"] not in VERDICTS:
        errs.append(f"verdict must be one of {list(VERDICTS)}")
    if "params" in rec and not isinstance(rec["params"], dict):
        errs.append("params must be an object")
    if "reasons" in rec:
        reasons = rec["reasons"]
        if not (isinstance(reasons, list) and reasons and all(_nonempty_str(r) for r in reasons)):
            errs.append("reasons must be a non-empty list of non-empty strings")
    if "cited" in rec:
        errs.extend(_cited_errors(rec["cited"]))
    if "rule_version" in rec and not _nonempty_str(rec["rule_version"]):
        errs.append("rule_version must be a non-empty string")
    if "prev_sha256" in rec and not (
        isinstance(rec["prev_sha256"], str) and _HEX64.match(rec["prev_sha256"])
    ):
        errs.append("prev_sha256 must be 64 lowercase hex characters")
    psha = rec.get("prereg_sha256")
    if psha is not None and not (isinstance(psha, str) and _HEX64.match(psha)):
        errs.append("prereg_sha256 must be 64 lowercase hex characters or null")
    if "details" in rec and not isinstance(rec["details"], dict):
        errs.append("details must be an object")
    decider = rec.get("decider")
    if "decider" in rec and decider not in DECIDERS:
        errs.append(f"decider must be one of {list(DECIDERS)}")
    if decider in ("model", "human") and not _nonempty_str(rec.get("decider_ref")):
        errs.append(f"decider_ref is required when decider is {decider} (model bundle / who)")
    if "overrides" in rec:
        if not (isinstance(rec["overrides"], str) and _ID.match(rec["overrides"])):
            errs.append("overrides must be a record id (D<n>)")
        if decider != "human":
            errs.append("only a human record may override another record")
        if not _nonempty_str(rec.get("deviation_id")):
            errs.append("an override needs a deviation_id (record it with /deviate)")
    if "deviation_id" in rec and not _nonempty_str(rec["deviation_id"]):
        errs.append("deviation_id must be a non-empty string")
    return errs


def _parse(data: bytes, path: Path) -> list[dict[str, Any]]:
    """Parse and verify a store's bytes: schema, ids and the sha256 chain."""
    out: list[dict[str, Any]] = []
    if data and not data.endswith(b"\n"):
        raise _err(f"{path} does not end with a newline; a record was cut short")
    offset = 0
    for n, line in enumerate(data.splitlines(keepends=True), start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _err(f"{path}:{n} is not valid JSON: {exc}") from exc
        errs = validate_record(rec)
        if errs:
            raise _err(f"{path}:{n} is not a valid decision record: {'; '.join(errs)}")
        if rec["id"] != f"D{n}":
            raise _err(f"{path}:{n} has id {rec['id']}, expected D{n}")
        if rec["prev_sha256"] != hashlib.sha256(data[:offset]).hexdigest():
            raise _err(
                f"{path}:{n} breaks the sha256 chain: an earlier record was edited",
                "restore the store from version control; an override is a new record",
            )
        if "overrides" in rec and rec["overrides"] not in {r["id"] for r in out}:
            raise _err(f"{path}:{n} overrides {rec['overrides']}, which precedes no record")
        out.append(rec)
        offset += len(line)
    return out


def load(path: Path) -> list[dict[str, Any]]:
    """Read and verify every record in the store (an absent store is empty)."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CliError(EXIT_ENV_ERROR, f"cannot read {path}: {exc}", "check the path") from exc
    return _parse(data, path)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _append_record(path: Path, fields: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    draft = {"schema_version": SCHEMA_VERSION, "id": "D1", "created": _now()}
    draft.update({k: v for k, v in fields.items() if v is not None})
    draft["prev_sha256"] = "0" * 64  # placeholders; the real id and chain come under the lock
    errs = validate_record(json.loads(json.dumps(draft)))
    if errs:  # refuse before the store is even created
        raise _err(f"refusing an invalid decision record: {'; '.join(errs)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        data = os.pread(fd, size, 0) if size else b""
        existing = _parse(data, path)
        if fields.get("overrides") is not None and fields["overrides"] not in {
            r["id"] for r in existing
        }:
            raise _err(f"no record {fields['overrides']} to override in {path}")
        rec = {
            "schema_version": SCHEMA_VERSION,
            "id": f"D{len(existing) + 1}",
            "created": _now(),
            **{k: v for k, v in fields.items() if v is not None},
            "prev_sha256": hashlib.sha256(data).hexdigest(),
        }
        rec = json.loads(json.dumps(rec))  # plain JSON types only
        errs = validate_record(rec)
        if errs:
            raise _err(f"refusing an invalid decision record: {'; '.join(errs)}")
        line = (json.dumps(rec, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        os.write(fd, line)
        os.fsync(fd)
        return rec
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append(
    path: Path,
    *,
    run: str,
    verdict: str,
    reasons: Iterable[str],
    cited: Iterable[Mapping[str, Any]],
    rule_version: str,
    decider: str = "rule",
    params: Mapping[str, Any] | None = None,
    prereg_sha256: str | None = None,
    decider_ref: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and append one decision record; return it (with id and chain hash)."""
    return _append_record(
        path,
        {
            "run": run,
            "verdict": verdict,
            "params": dict(params or {}),
            "reasons": list(reasons),
            "cited": [dict(c) for c in cited],
            "rule_version": rule_version,
            "prereg_sha256": prereg_sha256,
            "decider": decider,
            "decider_ref": decider_ref,
            "details": dict(details) if details is not None else None,
        },
    )


def override(
    path: Path,
    *,
    overrides: str,
    verdict: str,
    reasons: Iterable[str],
    deviation_id: str,
    decided_by: str = "operator",
    params: Mapping[str, Any] | None = None,
    cited: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record a human override of an earlier record, as a new record.

    The overridden record stays byte-identical. The override inherits its run
    and rule version and, unless new citations are given, its cited metrics.
    """
    if not _nonempty_str(deviation_id):
        raise _err(
            "an override needs a deviation id",
            "record the override with /deviate first, then pass its id",
        )
    target = next((r for r in load(path) if r["id"] == overrides), None)
    if target is None:
        raise _err(f"no record {overrides} to override in {path}")
    return _append_record(
        path,
        {
            "run": target["run"],
            "verdict": verdict,
            "params": dict(params or {}),
            "reasons": list(reasons),
            "cited": [dict(c) for c in cited] if cited is not None else target["cited"],
            "rule_version": target["rule_version"],
            "prereg_sha256": target.get("prereg_sha256"),
            "decider": "human",
            "decider_ref": decided_by,
            "overrides": overrides,
            "deviation_id": deviation_id.strip(),
        },
    )
