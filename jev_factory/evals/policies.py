"""Harness decision policies: versioned JSON configs, applied offline.

A policy is a small JSON document that says how to turn one already recorded
prediction (its ``candidates`` distribution) into a final decision without
calling a model again:

- an optional ``calibration`` block (``{"temperature": ..., "vector": ...}``)
  rescaled onto the recorded ``candidates`` with
  :func:`jev_factory.core.calibration.apply_scaling`;
- an optional ``gate`` block (:meth:`jev_factory.core.gate.Thresholds.from_json`'s
  shape) that gates the (possibly rescaled) distribution's argmax with
  :func:`jev_factory.core.gate.decide`.

A policy with neither (``policies/raw.json``) is the bare argmax: the
*model-only* row. Any other policy is a *model+harness* row. Read-only vs
mutating dispatch is exactly the gate's: the Domain's ``read_only`` flag for
the argmax operation, never its name, and an operation the Domain does not
know is gated as mutating (the stricter set).

A policy cannot filter the operations offered to the scorer: that would
change what the model saw and cannot be replayed from one saved
distribution. A record with no distribution returns the explicit
``"not_gateable"`` decision rather than guessing an argmax never computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from jev_factory.core import calibration as calibration_fit
from jev_factory.core import gate
from jev_factory.domain.model import Domain

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/policies.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 54-85: the _load_sibling_script path-loader for gate.py and calibration_fit.py"
        " is replaced by jev_factory.core.gate / jev_factory.core.calibration imports",
        "apply takes the Domain (core gate.decide reads read-only from it, unknown ="
        " mutating) instead of nvsh.ops.table",
        "builtin policies: raw.json and mutating-strict-example.json ship; nvsh's"
        " scorer-r3b-shipped.json (one nvsh model's frozen calibration) does not; a run may"
        " name a policy JSON file instead (the bundle's own calibration + gate)",
    ],
    "licence": "Apache-2.0",
}

_POLICIES_DIR = Path(__file__).resolve().parent / "policies"


class PolicyError(ValueError):
    """A malformed policy payload (missing ``name``/``version``, bad JSON shape)."""


#: The decision string returned when a record carries no distribution to gate.
NOT_GATEABLE = "not_gateable"


def load_policy(path: Path | str) -> dict:
    """Read and validate one policy JSON file (does not call a model)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_policy(data)


def validate_policy(policy: Mapping) -> dict:
    """Check *policy* carries ``name`` and ``version``; return it as a plain dict."""
    if not isinstance(policy, Mapping):
        raise PolicyError(f"a policy must be a JSON object, got {policy!r}")
    if not policy.get("name"):
        raise PolicyError("policy must carry a non-empty 'name'")
    if policy.get("version") in (None, ""):
        raise PolicyError("policy must carry a non-empty 'version'")
    return dict(policy)


def builtin_policy_path(name: str) -> Path:
    """Path of one of this package's own ``policies/<name>.json`` files."""
    return _POLICIES_DIR / f"{name}.json"


def scaled_and_decided(
    policy: Mapping, candidates: Mapping[str, float], offered: Sequence[str], domain: Domain
) -> tuple[dict[str, float], gate.Decision]:
    """*policy*'s calibrated distribution and the gate's decision on it."""
    validated = validate_policy(policy)
    scaled = dict(candidates)
    calibration = validated.get("calibration")
    if calibration:
        temperature = float(calibration.get("temperature", 1.0))
        vector = calibration.get("vector") or None
        scaled = calibration_fit.apply_scaling(scaled, temperature=temperature, vector=vector)
    gate_json = validated.get("gate")
    thresholds = gate.Thresholds.from_json(gate_json) if gate_json else gate.Thresholds()
    return scaled, gate.decide(scaled, offered, thresholds, domain)


def apply(
    policy: Mapping,
    raw_record_like: Mapping,
    offered: Sequence[str],
    domain: Domain,
) -> tuple[str, str | None, str, object]:
    """Apply *policy* to one already-recorded prediction. Never calls a model.

    Returns ``(decision, reason, policy_name, policy_version)``: *decision* is
    ``"not_gateable"`` (reason ``"no_distribution"``) when there is no usable
    distribution, otherwise one of ``gate.OUTCOMES``.
    """
    validated = validate_policy(policy)
    name = validated["name"]
    version = validated["version"]
    candidates = raw_record_like.get("candidates") if isinstance(raw_record_like, Mapping) else None
    if not candidates:
        return NOT_GATEABLE, "no_distribution", name, version
    _scaled, decision = scaled_and_decided(validated, candidates, offered, domain)
    return decision.outcome, decision.reason, name, version
