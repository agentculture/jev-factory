"""The distill executor seam: job spec, protected-id and scan checks, output verification.

The local executor must launch the trainer exactly as main did (same argv, env, GPU guard
and capped.sh call); a fake non-local executor shows the job-file contract and the sha256
return step. No GPU, no network, no subprocess beyond the patched ones.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from jev_factory.backbones.pointer import train_distill as td
from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import executor as ex
from jev_factory.distill import prereg as dp
from jev_factory.factory import gpu
from tests.test_pointer_train_distill import RUN_KNOBS, prereg_doc

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------- helpers


def _rows(path: Path, ids) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"record_id": i, "question_id": "q", "kind": "choice"}) for i in ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _side(path: Path, ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": i, "source": "s", "state": "x"}) for i in ids]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def pipeline_plan(tmp_path: Path, train_ids=("t1", "t2"), sides=True, **kw):
    """A plan in the pipeline's layout: prereg, rows and output under the run directory."""
    run = tmp_path / "run"
    (run / "distill").mkdir(parents=True)
    prereg = run / "distill" / "prereg.json"
    prereg.write_text(json.dumps(prereg_doc()), encoding="utf-8")
    dp.lock(prereg, run)
    train_file = _rows(run / "distill" / "assemble" / "train_set.jsonl", train_ids)
    if sides:
        data = run / "distill" / "dataset"
        _side(data / "train.jsonl", train_ids)
        _side(data / "fit.jsonl", ["f1"])
        _side(data / "selection.jsonl", ["s1"])
        _side(data / "held_out.jsonl", ["h1"])
        _side(data / "test.jsonl", ["x1"])
    init = tmp_path / "cache" / "stock"
    init.mkdir(parents=True)
    (init / "strands_decider_config.json").write_text("{}\n", encoding="utf-8")
    (init / "head.safetensors").write_bytes(b"\x00weights")
    return td.plan_train(
        prereg_path=prereg,
        run_dir=run,
        run_knobs=dict(RUN_KNOBS),
        train_file=train_file,
        init_from=init,
        output_dir=run / "distill" / "train" / "c1" / "checkpoint",
        trainer_python="/opt/trainer/bin/python",
        decider_repo=tmp_path / "strands-decider",
        prereg=dp,
        **kw,
    )


class _Done:
    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _write_checkpoint(plan, files=td.CHECKPOINT_OUTPUTS):
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        (out / name).write_text(f"{name}\n", encoding="utf-8")


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


# --------------------------------------------------------------------- job spec shape


def test_job_spec_lists_inputs_outputs_environment_and_limits(tmp_path):
    plan = pipeline_plan(tmp_path)
    spec = ex.job_spec_from_plan(
        plan,
        expected_outputs=td.CHECKPOINT_OUTPUTS,
        limits=ex.JobLimits(memory_max="100G", memory_floor="8G", timeout=60.0),
    )
    doc = spec.to_json()
    inputs = {i["path"]: i["sha256"] for i in doc["inputs"]}
    assert "distill/prereg.json" in inputs
    assert "distill/prereg.lock.json" in inputs
    assert "distill/assemble/train_set.jsonl" in inputs
    assert inputs["distill/prereg.json"] == dp.sha256_file(
        Path(plan["run_dir"]) / "distill/prereg.json"
    )
    assert all(len(v) == 64 for v in inputs.values())
    assert doc["outputs"] == [
        f"distill/train/c1/checkpoint/{name}" for name in td.CHECKPOINT_OUTPUTS
    ]
    env = doc["environment"]
    assert env["python"] == "/opt/trainer/bin/python"
    assert env["variables"] == {}
    assert doc["limits"] == {
        "memory_max": "100G",
        "memory_floor": "8G",
        "timeout": 60.0,
        "allow_foreign": False,
    }
    assert doc["stage"] == "train"


def test_clean_job_spec_holds_no_absolute_paths_outside_the_environment(tmp_path):
    plan = pipeline_plan(tmp_path)
    doc = ex.job_spec_from_plan(plan, expected_outputs=td.CHECKPOINT_OUTPUTS).to_json()
    env = doc.pop("environment")
    leaked = [s for s in _walk_strings(doc) if str(tmp_path) in s or s.startswith("/")]
    assert leaked == []
    assert all(not str(v).startswith("/") for v in env["variables"].values())
    argv = doc["argv"]
    assert argv[argv.index("--run-dir") + 1] == "."
    assert argv[argv.index("--train-file") + 1] == "distill/assemble/train_set.jsonl"
    assert argv[argv.index("--init-from") + 1] == "external/init_from"
    # the stock checkpoint (outside the run) is an input under external/, hashed per file
    paths = [i["path"] for i in doc["inputs"]]
    assert "external/init_from/head.safetensors" in paths


def test_job_spec_accepts_an_image_digest_environment(tmp_path):
    plan = pipeline_plan(tmp_path)
    environment = ex.JobEnvironment(
        image_digest="sha256:" + "a" * 64, lockfile_hash="b" * 64, python=None
    )
    doc = ex.job_spec_from_plan(plan, environment=environment).to_json()
    assert doc["environment"]["image_digest"] == "sha256:" + "a" * 64
    assert doc["environment"]["lockfile_hash"] == "b" * 64
    assert "python" not in doc["environment"]


def test_dry_run_prints_the_job_spec_and_touches_nothing(tmp_path, monkeypatch):
    plan = pipeline_plan(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("ran"))
    out = td.run_train(plan, jobs=ex)
    assert out["dry_run"] is True
    assert out["job_spec"]["stage"] == "train"
    assert out["argv"] == plan["argv"]  # the plan is still printed as before
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


# --------------------------------------------------------------------- local executor == main


RECORDED_PLAN = {
    "backbone": "pointer",
    "stage": "train",
    "dry_run": True,
    "argv": [
        "/opt/trainer/bin/python",
        "-m",
        "jev_factory.backbones.pointer.train_distill",
        "--prereg",
        "/work/run/distill/prereg.json",
        "--run-dir",
        "/work/run",
        "--knobs",
        '{"alpha": 0.7}',
        "--train-file",
        "/work/run/distill/assemble/train_set.jsonl",
        "--init-from",
        "/cache/stock",
        "--output-dir",
        "/work/run/distill/train/c1/checkpoint",
    ],
    "env": {"PYTHONPATH": "/src/strands-decider:/src/strands-decider/src:/src/jev"},
    "run_dir": "/work/run",
    "output_dir": "/work/run/distill/train/c1/checkpoint",
    "init_from": "/cache/stock",
    "train_file": "/work/run/distill/assemble/train_set.jsonl",
    "heal_spec": None,
}


def _main_run_train(plan, *, allow_foreign=False, memory_max=None, memory_floor="8G", timeout=None):
    """main's run_train apply body (e3345d1), verbatim apart from the return value."""
    env = {**os.environ, **plan["env"]}
    return gpu.run_gpu_stage(
        td.STAGE,
        plan["run_dir"],
        list(plan["argv"]),
        allow_foreign=allow_foreign,
        memory_max=memory_max,
        memory_floor=memory_floor,
        env=env,
        timeout=timeout,
    )


def _capture(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append({"argv": list(argv), **{k: v for k, v in kw.items()}})
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    guard = []
    real_guard = gpu.assert_gpu_free

    def spy_guard(stage, **kw):
        guard.append((stage, kw))
        return real_guard(stage, **kw)

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    monkeypatch.setattr(gpu, "assert_gpu_free", spy_guard)
    monkeypatch.setattr(gpu, "query_compute_apps", lambda env=None: [])
    return calls, guard


def test_local_executor_reproduces_mains_capped_call_exactly(monkeypatch):
    calls, guard = _capture(monkeypatch)
    _main_run_train(RECORDED_PLAN, memory_max="100G", timeout=5.0)
    main_calls, main_guard = list(calls), list(guard)
    calls.clear()
    guard.clear()
    out = td.run_train(RECORDED_PLAN, apply=True, memory_max="100G", timeout=5.0, jobs=ex)
    assert out["returncode"] == 0
    assert calls == main_calls
    assert guard == main_guard == [("train", {"allow_foreign": False})]
    (call,) = calls
    assert call["argv"] == [
        "bash",
        "-c",
        'source "$0"; run_capped "$@"',
        str(gpu.CAPPED_SH),
        "/work/run",
        *RECORDED_PLAN["argv"],
    ]
    assert call["env"]["PYTHONPATH"] == RECORDED_PLAN["env"]["PYTHONPATH"]
    assert call["env"]["TRAIN_MEMORY_MAX"] == "100G"
    assert call["env"]["TRAIN_MEMORY_FLOOR"] == "8G"
    assert call["timeout"] == 5.0


def test_local_executor_passes_the_foreign_gpu_override_through(monkeypatch):
    calls, guard = _capture(monkeypatch)
    td.run_train(RECORDED_PLAN, apply=True, memory_max="100G", allow_foreign=True, jobs=ex)
    assert guard == [("train", {"allow_foreign": True})]
    assert len(calls) == 1


def test_run_train_defaults_to_the_local_executor(tmp_path, monkeypatch):
    seen = []
    original = ex.LocalExecutor.submit

    def spy(self, spec):
        seen.append(spec)
        return original(self, spec)

    monkeypatch.setattr(ex.LocalExecutor, "submit", spy)
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: _Done())
    plan = pipeline_plan(tmp_path)
    td.run_train(plan, apply=True, memory_max="100G", jobs=ex)
    assert len(seen) == 1
    assert seen[0].stage == "train"


def test_run_train_accepts_an_injected_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("ran locally"))
    rec = _Recorder()
    out = td.run_train(
        pipeline_plan(tmp_path), apply=True, memory_max="100G", executor=rec, jobs=ex
    )
    assert len(rec.submitted) == 1
    assert out["job"]["executor"] == "recorder"


def test_run_train_verifies_declared_outputs_and_returns_the_job_manifest(tmp_path, monkeypatch):
    plan = pipeline_plan(tmp_path)
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: _write_checkpoint(plan) or _Done())
    out = td.run_train(
        plan, apply=True, memory_max="100G", expected_outputs=td.CHECKPOINT_OUTPUTS, jobs=ex
    )
    job = out["job"]
    assert job["executor"] == "local"
    assert job["verified"] is True
    assert sorted(job["outputs"]) == [
        f"distill/train/c1/checkpoint/{n}" for n in sorted(td.CHECKPOINT_OUTPUTS)
    ]
    assert job["spec"]["stage"] == "train"
    assert job["protected_check"]["staged_ids"] == 2
    stage_manifest = ex.record_job({"stage": "distill-train"}, job)
    assert stage_manifest["jobs"] == [job]
    assert ex.record_job(stage_manifest, job)["jobs"] == [job, job]


def test_a_missing_declared_output_fails_non_zero(tmp_path, monkeypatch):
    plan = pipeline_plan(tmp_path)
    monkeypatch.setattr(
        gpu,
        "run_gpu_stage",
        lambda *a, **k: _write_checkpoint(plan, td.CHECKPOINT_OUTPUTS[:1]) or _Done(),
    )
    with pytest.raises(CliError) as err:
        td.run_train(
            plan, apply=True, memory_max="100G", expected_outputs=td.CHECKPOINT_OUTPUTS, jobs=ex
        )
    assert err.value.code != 0
    assert "missing" in err.value.message


# --------------------------------------------------------------------- seam checks


class _Recorder:
    name = "recorder"

    def __init__(self):
        self.submitted = []

    def submit(self, spec):
        self.submitted.append(spec)
        return ex.JobHandle(executor=self.name, job_id="j1")

    def fetch(self, handle, spec):
        return ex.JobResult(returncode=0, outputs={})


@pytest.mark.parametrize("side_id", ["f1", "s1", "h1", "x1"])
def test_seam_refuses_a_protected_record_id_before_any_executor(tmp_path, side_id):
    plan = pipeline_plan(tmp_path, train_ids=("t1", side_id))
    spec = ex.job_spec_from_plan(plan)
    rec = _Recorder()
    with pytest.raises(CliError) as err:
        ex.run_job(spec, rec)
    assert err.value.code == EXIT_USER_ERROR
    assert "protected" in err.value.message
    assert side_id not in err.value.message  # ids are counted, never printed
    assert rec.submitted == []


def test_seam_refuses_a_protected_id_named_by_the_heal_spec(tmp_path):
    run = tmp_path / "run"
    heal = run / "distill" / "quant" / "heal_spec.json"
    plan = pipeline_plan(tmp_path, train_ids=("t1", "secret-test-id"), sides=False, heal_spec=heal)
    heal.parent.mkdir(parents=True, exist_ok=True)
    heal.write_text(
        json.dumps({"protected_id_sha256": [td.id_digest("secret-test-id")]}), encoding="utf-8"
    )
    spec = ex.job_spec_from_plan(plan)
    rec = _Recorder()
    with pytest.raises(CliError, match="protected"):
        ex.run_job(spec, rec)


def test_run_train_refuses_a_protected_id_before_the_gpu(tmp_path, monkeypatch):
    plan = pipeline_plan(tmp_path, train_ids=("t1", "h1"))
    monkeypatch.setattr(gpu, "run_gpu_stage", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(CliError) as err:
        td.run_train(plan, apply=True, memory_max="100G", jobs=ex)
    assert err.value.code == EXIT_USER_ERROR


@pytest.mark.parametrize(
    "planted",
    [
        "token: ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
        "see http://10.20.30.40:8000 for the box",
        "ssh orin.local",
    ],
)
def test_seam_refuses_a_secret_or_private_host_in_the_staged_inputs(tmp_path, planted):
    plan = pipeline_plan(tmp_path)
    train_file = Path(plan["train_file"])
    train_file.write_text(
        train_file.read_text(encoding="utf-8")
        + json.dumps({"record_id": "t3", "state": planted})
        + "\n",
        encoding="utf-8",
    )
    rec = _Recorder()
    spec = ex.job_spec_from_plan(plan)
    with pytest.raises(CliError) as err:
        ex.run_job(spec, rec)
    assert err.value.code == EXIT_USER_ERROR
    assert "ghp_" not in err.value.message
    assert "10.20.30.40" not in err.value.message
    assert rec.submitted == []


@pytest.mark.parametrize(
    "variables",
    [
        {"WANDB_API_KEY": "abc"},
        {"NOTE": "sk-" + "a" * 30},
        {"ENDPOINT": "http://192.168.1.20:9000"},
    ],
)
def test_seam_refuses_a_secret_or_private_host_in_the_environment(tmp_path, variables):
    plan = pipeline_plan(tmp_path)
    environment = ex.JobEnvironment(python="/opt/trainer/bin/python", variables=variables)
    rec = _Recorder()
    spec = ex.job_spec_from_plan(plan, environment=environment)
    with pytest.raises(CliError) as err:
        ex.run_job(spec, rec)
    assert err.value.code == EXIT_USER_ERROR
    assert rec.submitted == []


def test_seam_runs_scan_secrets_on_the_staged_inputs(tmp_path, monkeypatch):
    seen = []

    def fake_flags(folder, files):
        seen.extend(str(Path(folder) / rel) for rel, _ in files)
        return True

    monkeypatch.setattr(ex, "_script_flags", fake_flags)
    plan = pipeline_plan(tmp_path)
    rec = _Recorder()
    spec = ex.job_spec_from_plan(plan)
    with pytest.raises(CliError, match="scan-secrets"):
        ex.run_job(spec, rec)
    assert str(Path(plan["train_file"])) in seen
    assert rec.submitted == []


# --------------------------------------------------------------------- fake non-local executor


class FakeRemote:
    """A SageMaker-shaped fake: stages a copy, runs elsewhere, returns outputs by sha256."""

    name = "fake-remote"

    def __init__(self, jobdir: Path, remote: Path, tamper: bool = False):
        self.jobdir, self.remote, self.tamper = jobdir, remote, tamper
        self.received: list[str] = []

    def submit(self, spec):
        staged = self.remote / "staged"
        ex.stage_inputs(spec, staged)
        self.received = sorted(
            p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()
        )
        ex.write_job_id(self.jobdir, spec.name, "job-42")
        ex.append_log(self.jobdir, spec.name, "submitted job-42")
        return ex.JobHandle(executor=self.name, job_id="job-42")

    def complete(self, spec, rc=0):
        out = self.remote / "out"
        for rel in spec.outputs:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(f"trained {rel}\n", encoding="utf-8")
        ex.append_log(self.jobdir, spec.name, "finished")
        ex.mark_done(self.jobdir, spec.name, rc)

    def fetch(self, handle, spec):
        out = self.remote / "out"
        reported = {}
        for rel in spec.outputs:
            src = out / rel
            reported[rel] = ex.sha256_path(src)
            dst = spec.local_path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if self.tamper:
                dst.write_bytes(dst.read_bytes()[:-1] + b"X")
        rc = ex.job_status(self.jobdir, spec.name)["rc"]
        return ex.JobResult(returncode=rc, outputs=reported, log=["finished"])


def _remote_spec(tmp_path):
    plan = pipeline_plan(tmp_path)
    return ex.job_spec_from_plan(plan, expected_outputs=td.CHECKPOINT_OUTPUTS, name="c1")


def test_fake_remote_receives_only_the_staged_train_side_bundle(tmp_path):
    spec = _remote_spec(tmp_path)
    remote = FakeRemote(tmp_path / "jobs", tmp_path / "remote")
    original = remote.submit
    remote.submit = lambda s: (original(s), remote.complete(s))[0]
    job = ex.run_job(spec, remote)
    assert remote.received == sorted(i.path for i in spec.inputs)
    assert not any("dataset/" in rel for rel in remote.received)
    staged = (tmp_path / "remote" / "staged" / "distill/assemble/train_set.jsonl").read_text()
    for protected in ("f1", "s1", "h1", "x1"):
        assert f'"{protected}"' not in staged
    assert job["verified"] is True
    assert job["job_id"] == "job-42"
    assert job["returncode"] == 0


def test_fake_remote_job_state_through_job_id_done_and_log_files(tmp_path):
    spec = _remote_spec(tmp_path)
    jobs = tmp_path / "jobs"
    remote = FakeRemote(jobs, tmp_path / "remote")
    assert ex.job_status(jobs, spec.name)["state"] == "absent"
    remote.submit(spec)
    status = ex.job_status(jobs, spec.name)
    assert status["state"] == "running"
    assert status["job_id"] == "job-42"
    assert status["rc"] is None
    remote.complete(spec, rc=0)
    status = ex.job_status(jobs, spec.name)
    assert status["state"] == "done"
    assert status["rc"] == 0
    assert (jobs / f"{spec.name}.log").read_text().splitlines() == ["submitted job-42", "finished"]


def test_job_status_still_reads_a_local_detached_pid_job(tmp_path):
    (tmp_path / "local.pid").write_text("999999999\n")
    assert ex.job_status(tmp_path, "local")["state"] == "died"


def test_a_changed_output_byte_fails_hash_verification(tmp_path):
    spec = _remote_spec(tmp_path)
    remote = FakeRemote(tmp_path / "jobs", tmp_path / "remote", tamper=True)
    original = remote.submit
    remote.submit = lambda s: (original(s), remote.complete(s))[0]
    with pytest.raises(CliError) as err:
        ex.run_job(spec, remote)
    assert err.value.code != 0
    assert "sha256" in err.value.message


def test_a_failed_remote_job_reports_its_returncode_without_verifying(tmp_path):
    spec = _remote_spec(tmp_path)
    remote = FakeRemote(tmp_path / "jobs", tmp_path / "remote")
    original = remote.submit
    remote.submit = lambda s: (original(s), remote.complete(s, rc=3))[0]
    job = ex.run_job(spec, remote)
    assert job["returncode"] == 3
    assert job["verified"] is False


# --------------------------------------------------------------------- boundaries


def test_no_aws_sdk_anywhere_in_the_package():
    hits = []
    for path in (REPO / "jev_factory").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if "boto3" in text or "sagemaker" in text:
            hits.append(str(path))
    assert hits == []
