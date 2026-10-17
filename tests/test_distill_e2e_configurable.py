"""End-to-end scenarios for the configurable distillery (t13), CPU-only, on the e2e fakes.

Each scenario builds the t25 e2e ``Env`` and rewrites its ``run.toml`` (one per scenario):

* a second student profile (other repos, revisions, digests) with an NVFP4 Clef, every stage
  from ``preregister`` to ``bundle``, with no w4a16 and no 2610 constant anywhere in the report;
* the alt teacher, the deployed server and the edge board absent, with a bf16 deployed format;
* a fake non-local executor that receives a staged bundle with no protected id and whose
  one-byte output change fails verification;
* the boundary checks: ``tests/test_distill_boundaries.py`` untouched, the jev-CLI domain verb
  set equal to main's, and ``dependencies = []``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import layout as player
from jev_factory.distill import executor as dex
from jev_factory.distill import pins as dpins
from jev_factory.distill import pipeline as P
from jev_factory.distill import prereg as dprereg
from jev_factory.distill import profile as dprofile
from jev_factory.distill import quant as dquant
from jev_factory.factory.stages import read_manifest
from tests.test_distill_e2e import (
    FIX,
    STAGES,
    UPLOAD_KNOBS,
    Env,
    RecordingExecutor,
    approve_corrections,
    approve_test_gold,
    small_prereg,
)

REPO = Path(__file__).resolve().parent.parent
PROFILE_TEMPLATE = FIX / "distill_e2e" / "second_profile.toml"
MANIFEST = FIX / "distill_e2e" / "second_upstream_sources.json"


def render_profile(dest: Path) -> Path:
    """Copy the second-profile fixture, filling in the (absolute) contamination manifest."""
    text = PROFILE_TEMPLATE.read_text(encoding="utf-8")
    dest.write_text(text.replace("@CONTAMINATION_MANIFEST@", str(MANIFEST)), encoding="utf-8")
    return dest


with tempfile.TemporaryDirectory() as _tmp:
    ALT = dprofile.load_profile_file(render_profile(Path(_tmp) / "profile.toml"))
DECIDER_REV = ALT.decider_revision
_ALT_ADAPTER, _ALT_HEAD = b"alt-adapter", b"alt-head"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def small_prereg_v3(**kwargs) -> dict:
    """The e2e's shrunken knob grid (``small_prereg``), on a v3 draft seeded from ALT."""
    doc = dprereg.example_doc_v3(ALT, **kwargs)
    shrunk = small_prereg()
    doc["knob_grid"] = {**doc["knob_grid"], **shrunk["knob_grid"]}
    if "heal" in doc:
        doc["heal"]["lr"] = 1e-4
    return doc


class BundlingScorer:
    """The e2e scorer, plus what the served scorer also leaves: the winner's exported bundle.

    Under a bf16 ship the train stage's ``train/<cid>/bundle`` is the deployed build, so the
    fake must write it as ``served_scorer`` does (``layout.write_layout``)."""

    def __init__(self, inner, base_revision: str):
        self.inner, self.base_revision = inner, base_revision

    def __call__(self, candidate, checkpoint, fold_records):
        bundle = Path(checkpoint).parent / "bundle"
        if not bundle.exists():
            player.write_layout(
                checkpoint,
                bundle,
                base_revision=self.base_revision,
                extra={"distill_candidate": candidate.cid, "knobs": candidate.knobs},
            )
        return self.inner(candidate, checkpoint, fold_records)


class ConfigurableEnv(Env):
    """The e2e Env, with a pin cache for the second profile and a run.toml of its own."""

    def __init__(self, tmp_path, capsys, monkeypatch, *, absent: bool = False):
        self.absent = absent
        super().__init__(tmp_path, capsys, monkeypatch)
        self.profile = ALT
        profile_file = render_profile(tmp_path / "second_profile.toml")
        text = (self.work / "run.toml").read_text(encoding="utf-8")
        text = text.replace(f'distill_decider_revision = "{"a" * 40}"', "")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        lines += [
            f"distill_student_profile = {json.dumps(str(profile_file))}",
            f'distill_decider_revision = "{DECIDER_REV}"',
            'distill_teacher_variant = "NVFP4"',
        ]
        if absent:
            drop = ("distill_alt_teacher_", "distill_agx_url", "distill_edge_")
            lines = [ln for ln in lines if not ln.startswith(drop)]
        (self.work / "run.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        doc = (
            small_prereg_v3(teacher_variant="NVFP4", deployed_format="bf16", teacher_model="clef")
            if absent
            else small_prereg_v3(teacher_variant="NVFP4", teacher_model="clef")
        )
        if absent:
            for role in ("alt_teacher", "deployed_server", "edge"):
                doc["roles"][role]["present"] = False
        doc["roles"]["stock"]["model"] = "v21"
        self.prereg_doc = doc
        (self.work / dprereg.PREREG_REL).write_text(json.dumps(doc), encoding="utf-8")
        self.services.builders = {"bf16": dquant.bf16_builder(ALT.base_revision)} | dict(
            self.services.builders
        )
        if absent:
            # no deployed student server: the bf16 student is the deployed build
            self.services.final_models = {
                k: v for k, v in self.services.final_models.items() if k != "student"
            }
            self.services.scorer = BundlingScorer(self.scorer, ALT.base_revision)
            self.services.edge_client = None
            self.services.edge_get = None
            self.services.drafter = None

    def apply(self, stage, knobs=None):
        if stage == "final" and self.absent:
            # the bf16 student is served from the train stage's bundle, known only after train
            from jev_factory.distill import deployed

            bundle = Path(deployed.bf16_student(self.work)["artifact"])
            models = dict(self.services.final_models)
            models["student_bf16"] = replace(models["student_bf16"], served_from=bundle)
            self.services.final_models = models
        return super().apply(stage, knobs)

    def _pin_cache(self, root: Path, monkeypatch) -> Path:
        base = root / dpins.cache_dirname(ALT.base_repo)
        base.mkdir(parents=True)
        (base / dpins.REVISION_FILE).write_text(ALT.base_revision + "\n")
        dec = root / dpins.cache_dirname(ALT.decider_repo)
        (dec / "lora").mkdir(parents=True)
        (dec / dpins.REVISION_FILE).write_text(DECIDER_REV + "\n")
        (dec / ALT.adapter_file).write_bytes(_ALT_ADAPTER)
        (dec / ALT.head_file).write_bytes(_ALT_HEAD)
        (dec / ALT.provenance_file).write_text(
            json.dumps({"adapter_sha256": _sha(_ALT_ADAPTER), "head_sha256": _sha(_ALT_HEAD)})
        )
        assert (_sha(_ALT_ADAPTER), _sha(_ALT_HEAD)) == (ALT.adapter_sha256, ALT.head_sha256)
        return root


@pytest.fixture
def offline(tmp_path, capsys, monkeypatch):
    """The e2e's offline guard (no heavy imports, no sockets), without its Env."""
    import socket

    for name in ("torch", "transformers", "strands_decider"):
        monkeypatch.setitem(sys.modules, name, None)

    def refuse(*args, **kwargs):
        raise AssertionError("the e2e run opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return tmp_path, capsys, monkeypatch


@pytest.fixture
def second(offline):
    return ConfigurableEnv(*offline)


@pytest.fixture
def absent(offline):
    return ConfigurableEnv(*offline, absent=True)


def run_through(env: Env, last: str) -> None:
    """Every stage up to and including *last*, with the operator approvals in between."""
    for stage in STAGES[: STAGES.index("corrections-draft") + 1]:
        if STAGES.index(stage) > STAGES.index(last):
            return
        assert env.apply(stage)["status"] in ("complete", "skipped"), stage
    if STAGES.index(last) <= STAGES.index("corrections-draft"):
        return
    if (env.base / P.CORRECTIONS_FILE).exists():
        approve_corrections(env)
    approve_test_gold(env)
    for stage in STAGES[STAGES.index("assemble") : STAGES.index(last) + 1]:
        knobs = UPLOAD_KNOBS if stage == "upload" else None
        assert env.apply(stage, knobs)["status"] in ("complete", "skipped"), stage


def walk(value, path=""):
    """Every (path, key-or-None, value) of a JSON tree: keys and string values both."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield f"{path}/{k}", k, None
            yield from walk(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, None, value


def walk_dicts(value, path=""):
    if isinstance(value, dict):
        yield path, None, value
        for k, v in value.items():
            yield from walk_dicts(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk_dicts(v, f"{path}[{i}]")


# ------------------------------------------------------------------ second profile, NVFP4


def test_a_second_profile_with_an_nvfp4_clef_runs_every_stage_to_the_bundle(second):
    env = second
    assert env.prereg_doc["roles"]["teacher"]["variant"] == "NVFP4"
    run_through(env, "bundle")
    for stage in STAGES[: STAGES.index("bundle") + 1]:
        assert read_manifest(env.base, stage)["status"] == "complete", stage
    # the pins that ran are the second profile's: its repos, its revisions
    assert env.read("manifests/train.json")["knobs"]["_config"]["distill_student_profile"]
    assert env.read("manifests/assemble.json")["knobs"]["_config"]["distill_teacher_variant"] == (
        "NVFP4"
    )
    report = env.read("final/report.json")

    keys, strings = [], []
    for _path, key, value in walk(report):
        if key is not None:
            keys.append(key)
        if isinstance(value, str):
            strings.append(value)
    assert keys
    assert not [k for k in keys if "w4a16" in k.lower()]
    assert not [s for s in strings if "w4a16" in s.lower()]
    assert "NVFP4" in strings  # the variant is shown, as served
    # no value is a 2610 constant (repo, revisions, adapter / head digests)
    d = dprofile.DEFAULT_2610
    forbidden = {
        d.decider_repo,
        d.decider_revision,
        d.base_revision,
        d.adapter_sha256,
        d.head_sha256,
    }
    assert not [s for s in strings if s in forbidden or d.decider_repo in s]
    text = (env.base / "final" / "report.json").read_text(encoding="utf-8")
    assert not (forbidden & set(re.findall(r"[0-9a-f]{40,64}", text)))
    # the bars carry no w4a16 key, in the pre-registration or the report
    prereg_keys = [k for _p, k, _v in walk(env.read("prereg.json")) if k]
    assert not [k for k in prereg_keys if "w4a16" in k.lower()]
    # every bar row resolves to a legal result
    rows = [v for _p, _k, v in walk(report) if isinstance(v, str)]
    assert rows


# ------------------------------------------------------------------ absent roles, bf16 ship


def test_a_run_with_absent_roles_and_a_bf16_format_completes_through_the_bundle(absent):
    env = absent
    run_through(env, "bundle")
    status = {s: read_manifest(env.base, s)["status"] for s in STAGES[: STAGES.index("bundle") + 1]}
    assert status["quantize"] == "skipped"
    assert status["edge"] == "skipped"
    assert {s: v for s, v in status.items() if v not in ("complete", "skipped")} == {}
    report = env.read("final/report.json")
    rows = [v for _p, _k, v in walk_dicts(report) if "bar" in v]
    assert rows, "the report holds no bar rows"
    legal = {"pass", "fail", "not_applicable", "reported"}
    assert {r["result"] for r in rows} <= legal
    assert {r["result"] for r in rows if r["bar"].startswith("top1_")} >= {"pass", "not_applicable"}
    na = [r for r in rows if r["result"] == "not_applicable"]
    assert na, "bf16 with no edge leaves bars not applicable"
    for row in na:
        assert row.get("absent")
        assert row.get("reason")
    # the quant-only bars are the ones not applicable; nothing was quantized or measured on edge
    assert {r["bar"] for r in na} >= {"quant_vs_bf16_agreement_min", "weight_bytes_max"}
    assert not (env.base / "quant" / "quant_manifest.json").exists()
    assert env.read("manifests/quantize.json")["status"] == "skipped"
    assert env.read("manifests/edge.json")["status"] == "skipped"
    assert env.read("manifests/bundle.json")["status"] == "complete"
    assert env.prereg_doc["deployed_format"] == "bf16"
    assert {n for n, r in env.prereg_doc["roles"].items() if not r["present"]} == {
        "alt_teacher",
        "deployed_server",
        "edge",
    }


# ------------------------------------------------------------------ a fake non-local executor


class StagingExecutor(RecordingExecutor):
    """A non-local executor: it receives the staged bundle, inspects it, then runs it locally.

    ``tamper`` flips one byte of the first expected output after the job ran but before the
    result is reported, as a transfer that corrupts a file would."""

    name = "fake-remote"

    def __init__(self, tamper: bool = False):
        super().__init__()
        self.tamper = tamper
        self.received: list[dict] = []

    def submit(self, spec):
        protected, sources = dex.protected_digests(spec)
        ids = dex.staged_record_ids(spec)
        self.received.append(
            {
                "inputs": [i.path for i in spec.inputs],
                "ids": ids,
                "protected": protected,
                "sources": sources,
                "overlap": sorted(i for i in ids if dex.id_digest(i) in protected),
            }
        )
        return super().submit(spec)

    def fetch(self, handle, spec):
        result = super().fetch(handle, spec)  # digests of the outputs as the job wrote them
        if self.tamper:
            target = next(
                spec.local_path(rel) for rel in spec.outputs if spec.local_path(rel).is_file()
            )
            data = bytearray(target.read_bytes())
            data[0] ^= 0x01
            target.write_bytes(bytes(data))
        return result


def test_a_fake_non_local_executor_gets_a_bundle_with_no_protected_ids(second):
    env = second
    ex = StagingExecutor()
    env.services.executor = ex
    run_through(env, "train")
    assert ex.received, "train never reached the executor"
    for got in ex.received:
        assert got["ids"], "the staged bundle holds no record ids to check"
        assert got["protected"], "the seam knew no protected id"
        assert {"distill/dataset/fit.jsonl", "distill/dataset/selection.jsonl"} <= set(
            got["sources"]
        )
        assert got["overlap"] == []
    train = env.read("manifests/train.json")
    assert [j["executor"] for j in train["jobs"]] == ["fake-remote"] * len(ex.received)
    assert all(j["verified"] and j["protected_check"]["protected_ids"] > 0 for j in train["jobs"])


def test_a_fake_non_local_executor_fails_on_a_one_byte_output_change(second):
    env = second
    env.services.executor = StagingExecutor(tamper=True)
    run_through(env, "benchmark")
    rc, _out, err = env.cli("train", "--apply", "--json")
    assert rc != 0
    assert "failed verification" in err
    assert "sha256 differs" in err
    assert env.read("manifests/train.json")["status"] == "failed"
    assert not (env.base / "select" / "summary.json").exists()


# ------------------------------------------------------------------ the boundary checks


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def git_main():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if _git("rev-parse", "--verify", "--quiet", "main").returncode != 0:
        pytest.skip("no main ref to compare against")


def test_the_boundary_test_file_is_unchanged_against_main(git_main):
    diff = _git("diff", "main", "--", "tests/test_distill_boundaries.py")
    assert diff.returncode == 0
    assert diff.stdout == ""


VERBS = (
    "import json; from jev_factory.domains.jev_cli.generate import generate_domain;"
    "print(json.dumps(sorted(generate_domain().names())))"
)


def test_the_jev_cli_domain_export_has_the_verb_set_main_has(git_main, tmp_path):
    here = subprocess.run(
        [sys.executable, "-c", VERBS],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert here.returncode == 0, here.stderr
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "main", "jev_factory"], capture_output=True, check=False
    )
    assert archive.returncode == 0, archive.stderr
    tarball = tmp_path / "main.tar"
    tarball.write_bytes(archive.stdout)
    root = tmp_path / "main"
    root.mkdir()
    with tarfile.open(tarball) as tf:
        tf.extractall(root, filter="data")
    there = subprocess.run(
        [sys.executable, "-c", VERBS],
        capture_output=True,
        text=True,
        cwd=root,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert there.returncode == 0, there.stderr
    assert json.loads(here.stdout) == json.loads(there.stdout)
    assert json.loads(here.stdout)  # a non-empty verb set


def test_the_runtime_dependencies_stay_empty():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == []
