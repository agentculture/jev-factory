"""Regenerate the golden ``jev distill`` manifests from the e2e fakes (t1).

``generate(env)`` drives the same 15-stage run as ``tests/test_distill_e2e.py`` and returns
``{relative_name: normalised_text}`` for the pre-registration file and every stage manifest.
Later tasks must keep this output byte-identical under the default config.
"""

from __future__ import annotations

import json

from jev_factory.distill import pipeline as P
from jev_factory.distill import prereg as dprereg
from jev_factory.factory.stages import MANIFEST_DIR
from tests.test_distill_e2e import (
    STAGES,
    UPLOAD_KNOBS,
    approve_corrections,
    approve_test_gold,
)

# The ONLY normalisation, all listed here.
#: Per-run wall-clock values in a stage manifest.
TIMESTAMP_KEYS = frozenset({"started", "finished"})
#: The package version stamped by the stage runner.
VERSION_KEYS = frozenset({"jev_factory_version"})
#: Output/input paths whose sha256 differs run to run in the e2e fakes (measured by running the
#: e2e four times under different pytest tmp dirs): they embed the absolute tmp path, measured
#: latency, bootstrap draws or a torch pickle. Their hashes are replaced by VARIES; every other
#: input/output hash in a manifest is compared byte for byte.
VARYING_HASH_PATHS = frozenset(
    {
        "assemble/assemble_manifest.json",
        "benchmark/benchmark.json",
        "benchmark/stock_measurements.json",
        "benchmark/stock_reproduction.json",
        "bundle/dataset",
        "bundle/model",
        "corrections/draft.json",
        "edge",
        "final/final.json",
        "final/report.json",
        "gate/gate.json",
        "quant/calibration.json",
        "quant/deployed_predictions.json",
        "quant/format_decision.json",
        "quant/quant_manifest.json",
        "select/summary.json",
    }
)
#: A hash recorded under ``knobs._external`` of the sha256 of benchmark/stock_reproduction.json.
VARYING_VALUE_KEYS = frozenset({"stock_reproduction"})
HASH_SECTIONS = ("inputs", "outputs")
TMP_PATH = "<TMP>"  # the pytest tmp dir and the run dir, inside any string value
TIMESTAMP = "<TIMESTAMP>"
VERSION = "<VERSION>"
VARIES = "<VARIES>"


def run_all_stages(env) -> None:
    """The e2e driver: every stage applied, with the operator approvals in between."""
    for stage in STAGES[: STAGES.index("corrections-draft") + 1]:
        assert env.apply(stage)["status"] == "complete", stage
    approve_corrections(env)
    approve_test_gold(env)
    for stage in ("assemble", "benchmark", "train", "quantize", "gate", "final", "edge", "bundle"):
        assert env.apply(stage)["status"] == "complete", stage
    assert env.apply("upload", UPLOAD_KNOBS)["status"] == "complete"


def _normalise(value, roots: list[str]):
    if isinstance(value, dict):
        return {k: _normalise_item(k, v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v, roots) for v in value]
    if isinstance(value, str):
        for root in roots:
            value = value.replace(root, TMP_PATH)
    return value


def _normalise_item(key: str, item, roots: list[str]):
    if key in TIMESTAMP_KEYS and isinstance(item, (int, float)):
        return TIMESTAMP
    if key in VERSION_KEYS and isinstance(item, str):
        return VERSION
    if key in VARYING_VALUE_KEYS:
        return VARIES
    if key in HASH_SECTIONS and isinstance(item, dict):
        return {
            path: VARIES if path in VARYING_HASH_PATHS else _normalise(sha, roots)
            for path, sha in item.items()
        }
    return _normalise(item, roots)


def normalise_text(text: str, roots: list[str]) -> str:
    return json.dumps(_normalise(json.loads(text), roots), indent=2, sort_keys=True) + "\n"


def generate(env) -> dict[str, str]:
    """Run every stage on *env* and return the normalised golden files by relative name."""
    run_all_stages(env)
    roots = sorted({str(env.tmp), str(env.work)}, key=len, reverse=True)
    files = {"prereg.json": env.base / "prereg.json"}
    for stage in STAGES:
        files[f"{MANIFEST_DIR}/{stage}.json"] = env.base / MANIFEST_DIR / f"{stage}.json"
    assert dprereg.PREREG_REL == "distill/prereg.json" and P.STAGE_ORDER
    return {
        name: normalise_text(path.read_text(encoding="utf-8"), roots)
        for name, path in files.items()
    }
