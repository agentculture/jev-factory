"""d4 (d-t16): ``jev status`` judges distill staleness with the run's own stage inputs.

The pipeline runs each stage with the inputs the run actually has (``pipeline._stage_for``):
under ``deployed_format`` bf16 the deployed calibration is ``select/calibration.json`` and there
are no quant predictions; after a skipped corrections-draft, assemble reads no corrections file.
Status must use the same inputs, or a finished bf16 run shows gate, final and edge stale and a
run with a skipped corrections-draft shows assemble stale.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from jev_factory.cli._commands import status as cstatus
from jev_factory.distill import pipeline as P
from jev_factory.distill import prereg as dp
from jev_factory.factory.stages import Registry, run_stage, write_skipped


def _touch(base: Path, rel: str) -> None:
    path = base / rel
    if Path(rel).suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel, encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)
        (path / "x.json").write_text(rel, encoding="utf-8")


def _run(tmp_path: Path, doc: dict, skips: dict[str, str]) -> Path:
    """A run whose every stage completed (or was skipped) with the pipeline's own inputs."""
    work = tmp_path / "run"
    base = P.stage_dir(work)
    base.mkdir(parents=True)
    (work / dp.PREREG_REL).write_text(json.dumps(doc), encoding="utf-8")
    dp.lock(work / dp.PREREG_REL, work)
    # Files no stage declares as an output (the select calibration train writes beside its
    # summary, the corrections file under a non-skipped draft) exist as the real run has them.
    _touch(base, P.SELECT_CALIBRATION)
    _touch(base, P.DATASET_MANIFEST)
    if "corrections-draft" not in skips:
        _touch(base, P.CORRECTIONS_FILE)
    ctx = P.DistillContext(work=work, config=None)  # type: ignore[arg-type]
    reg = dp.require_locked(work / dp.PREREG_REL, work)
    for name in P.REGISTRY.names():
        if name in skips:
            write_skipped(base, name, skips[name], {})
            continue
        stage = P._stage_for(name, ctx, reg)  # noqa: SLF001 - the pipeline's own inputs

        def func(workdir, knobs, stage=stage):
            for rel in stage.outputs:
                if not (workdir / rel).exists():  # the lock and the dataset files stay
                    _touch(workdir, rel)

        one = Registry()
        for other in P.REGISTRY.names():
            st = P._stage_for(other, ctx, reg)  # noqa: SLF001
            one.register(replace(st, func=func) if other == name else st)
        man = run_stage(base, name, {}, registry=one)
        assert man["status"] == "complete", (name, man.get("error"))
    return work


def _stale(work: Path) -> dict[str, str | None]:
    return {s["stage"]: s["stale"] for s in cstatus.collect(work)["distill_stages"]}


def test_a_finished_bf16_run_shows_gate_final_and_edge_fresh(tmp_path):
    from jev_factory.distill import quant as dquant

    doc = dp.example_doc_v3(deployed_format="bf16")
    work = _run(tmp_path, doc, {"quantize": dquant.BF16_SKIP_REASON})
    stale = _stale(work)
    assert {k: v for k, v in stale.items() if v} == {}
    assert stale["gate"] is None and stale["final"] is None and stale["edge"] is None


def test_a_skipped_corrections_draft_leaves_assemble_fresh(tmp_path):
    doc = dp.example_doc_v3()
    doc["roles"]["alt_teacher"] = {"present": False}
    work = _run(tmp_path, doc, {"corrections-draft": "alt_teacher role absent: test"})
    stale = _stale(work)
    assert stale["assemble"] is None
    assert {k: v for k, v in stale.items() if v} == {}


def test_a_quant_run_is_judged_with_the_registered_inputs(tmp_path):
    """Unchanged for a quant ship: the effective registry is the registered one."""
    work = _run(tmp_path, dp.example_doc_v3(), {})
    assert {k: v for k, v in _stale(work).items() if v} == {}
    reg = P.effective_registry(work)
    for name in P.REGISTRY.names():
        assert reg.get(name).inputs == P.REGISTRY.get(name).inputs


def test_the_effective_registry_swaps_the_bf16_inputs(tmp_path):
    from jev_factory.distill import quant as dquant

    doc = dp.example_doc_v3(deployed_format="bf16")
    work = _run(tmp_path, doc, {"quantize": dquant.BF16_SKIP_REASON})
    reg = P.effective_registry(work)
    assert P.SELECT_CALIBRATION in reg.get("gate").inputs
    assert P.QUANT_PREDICTIONS not in reg.get("edge").inputs


def test_an_unlocked_run_falls_back_to_the_registered_inputs(tmp_path):
    work = tmp_path / "run"
    P.stage_dir(work).mkdir(parents=True)
    reg = P.effective_registry(work)
    assert [reg.get(n).inputs for n in reg.names()] == [
        P.REGISTRY.get(n).inputs for n in P.REGISTRY.names()
    ]
