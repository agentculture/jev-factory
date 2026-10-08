"""jev_factory.evals: the release gate, ported from nvsh's ``evals/tool_jev``.

A release gate run scores release candidates and baselines (their saved
predictions, replayed; no GPU) and reference models (asked the same cases
through one choice call each) with the exact metrics of
:mod:`jev_factory.core.metrics`, once model-only (policy ``raw``) and once
per model+harness policy (calibration plus the read-only/mutating gate of
:mod:`jev_factory.core.gate`). ``python -m jev_factory.evals`` is the runner.

Importing this package is the single choke point every ``jev_factory.evals``
module goes through before ``deepeval`` is imported anywhere, because deepeval
starts telemetry and reads its config at *import* time. Two environment
variables are forced to ``1`` here, and any other value already set is
refused rather than deferred to:

- ``DEEPEVAL_TELEMETRY_OPT_OUT=1``: no telemetry pings;
- ``DEEPEVAL_DISABLE_DOTENV=1``: deepeval never loads a stray ``.env``.

The package also refuses to run when ``CONFIDENT_API_KEY`` is set: that
variable opts deepeval into uploading results to a hosted service, which has
no place in an offline-first gate that runs with no secrets.

deepeval itself lives only in the ``evals`` dependency group (never in the
base install) and is imported lazily, inside the functions that need it.
"""

from __future__ import annotations

import os

NVSH_PROVENANCE = {
    "upstream": "evals/tool_jev/__init__.py",
    "commit": "9debdc6",
    "adaptations": [
        "lines 31-47: the env guard is kept verbatim (telemetry off, no .env, refuse"
        " CONFIDENT_API_KEY); only the package name in the messages changes",
        "evals/__init__.py's note (deepeval only in the evals dependency group) folded into"
        " this docstring; deepeval is imported lazily by the modules that need it",
    ],
    "licence": "Apache-2.0",
}

for _name in ("DEEPEVAL_TELEMETRY_OPT_OUT", "DEEPEVAL_DISABLE_DOTENV"):
    if os.environ.get(_name, "1") != "1":
        raise RuntimeError(
            f"jev_factory.evals refuses to run with {_name}={os.environ[_name]!r}: "
            f"this gate requires {_name}=1 (no telemetry, no .env loading). "
            f"Unset it or set it to 1."
        )
    os.environ[_name] = "1"

if os.environ.get("CONFIDENT_API_KEY"):
    raise RuntimeError(
        "jev_factory.evals refuses to run with CONFIDENT_API_KEY set: this gate is "
        "offline-first and must never upload results to Confident AI. Unset "
        "CONFIDENT_API_KEY before running it."
    )
