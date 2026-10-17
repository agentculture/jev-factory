"""Knob-recording stability (t8): new run-config keys are recorded only when set.

A completed run's manifests (built by the e2e fakes, as the golden set is) must stay fresh under
the default config, and setting a new key to a non-default value must make exactly the stages
that read it stale.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from jev_factory.distill import pipeline as P
from jev_factory.factory.config import load_config
from jev_factory.factory.stages import read_manifest, stale_stages
from tests.distill_golden_support import run_all_stages
from tests.test_distill_e2e import env  # noqa: F401  (the e2e fixture)

#: new key -> the stages that read it.
NEW_KEYS = {
    "distill_student_profile": (
        "dataset",
        "contamination",
        "benchmark",
        "train",
        "quantize",
        "final",
        "bundle",
    ),
    "distill_teacher_variant": (
        "teacher-label",
        "audit",
        "assemble",
        "benchmark",
        "final",
        "bundle",
    ),
    "distill_executor": ("train", "quantize"),
    "distill_deployed_url": ("final",),
}
NON_DEFAULT = {
    "distill_student_profile": "/profiles/nano.toml",
    "distill_teacher_variant": "w4a16",
    "distill_executor": "remote",
    "distill_deployed_url": "http://127.0.0.1:9",
}


def _ctx(env_, **overrides):  # noqa: F811
    cfg = load_config(str(env_.work / "run.toml"), cli={"work": str(env_.work)})
    cfg = replace(cfg, values={**cfg.values, **overrides})
    return P.DistillContext(work=env_.work, config=cfg)


def _current(env_, ctx):  # noqa: F811
    """The knobs a re-run would record: each stage's recorded explicit knobs, re-derived."""
    out = {}
    for name in P.REGISTRY.names():
        recorded = (read_manifest(P.stage_dir(env_.work), name) or {}).get("knobs", {})
        given = {k: v for k, v in recorded.items() if k in P.DEFAULT_KNOBS[name]}
        out[name] = P.effective_knobs(name, given, ctx)
    return out


@pytest.fixture
def completed(env):  # noqa: F811
    run_all_stages(env)
    return env


def _recorded(env_):  # noqa: F811
    return {
        n: (read_manifest(P.stage_dir(env_.work), n) or {}).get("knobs", {})
        for n in P.REGISTRY.names()
    }


def _knobs_changed(env_, ctx):  # noqa: F811
    """The stages whose recorded knobs differ from what a re-run would record now."""
    current = _current(env_, ctx)
    recorded = _recorded(env_)
    return {n for n in current if current[n] != recorded[n]}


def test_default_config_records_the_same_knobs_as_the_completed_run(completed):
    assert _knobs_changed(completed, _ctx(completed)) == set()


def test_stale_stages_is_the_same_as_with_the_recorded_knobs(completed):
    # Later stages write into earlier stages' output dirs, so a finished e2e run already lists
    # some stages as stale; what must not move is that list when the knobs are the recorded ones.
    wd = P.stage_dir(completed.work)
    baseline = stale_stages(wd, _recorded(completed), P.REGISTRY)
    assert stale_stages(wd, _current(completed, _ctx(completed)), P.REGISTRY) == baseline


def test_default_valued_new_keys_are_not_recorded(completed):
    ctx = _ctx(completed, distill_executor="local")
    for name, knobs in _current(completed, ctx).items():
        assert not set(NEW_KEYS) & set(knobs["_config"]), name


@pytest.mark.parametrize("key", sorted(NEW_KEYS))
def test_a_non_default_new_key_makes_exactly_its_stages_stale(completed, key):
    ctx = _ctx(completed, **{key: NON_DEFAULT[key]})
    assert _knobs_changed(completed, ctx) == set(NEW_KEYS[key])
    for name in NEW_KEYS[key]:
        assert P.effective_knobs(name, None, ctx)["_config"][key] == NON_DEFAULT[key]


def test_agx_url_recording_is_unchanged_when_only_it_is_set(completed):
    final = read_manifest(P.stage_dir(completed.work), "final")["knobs"]["_config"]
    assert "distill_agx_url" in final
    assert "distill_deployed_url" not in final
