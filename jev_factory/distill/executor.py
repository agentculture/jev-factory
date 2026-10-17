"""The executor seam: where a distill training job runs, behind one small contract.

``train_distill.run_train`` (train and heal alike) hands its plan to :func:`run_job` as a
:class:`JobSpec`; an :class:`Executor` runs it and the seam verifies what comes back.

* **JobSpec.** Executor-neutral: the staged inputs (each with its sha256), the expected
  outputs, the environment (a python path or an image digest plus a lockfile hash, and
  non-secret variables only), the limits (memory cap and floor, timeout, the foreign-GPU
  override) and the argv. Every path in :meth:`JobSpec.to_json` is relative to the staged
  folder (the run directory); a path outside it (the pinned stock checkpoint, or a test's
  out-of-run file) is staged under ``external/<role>``. The local binding (main's exact
  argv and env, and where each relative path lives on this box) rides along in
  :attr:`JobSpec.binding`, which is never serialised: a non-local executor ships
  :meth:`JobSpec.to_json` plus the folder :func:`stage_inputs` builds, nothing else.
* **Checks in the seam** (:func:`preflight`, run by :func:`run_job` before any executor):
  a staged record id in the held-out, test, fit or selection side refuses the job with exit
  1, and so does a secret, endpoint or hostname/private-IP hit in the staged inputs or the
  environment (the upload stage's scans: :mod:`jev_factory.release.scan` and
  ``scripts/scan-secrets.py``). Executors inherit both and cannot skip them.
* **Executor protocol.** ``submit(spec) -> JobHandle`` (the job id) and
  ``fetch(handle, spec) -> JobResult`` (the return code, and each output's sha256 as the job
  side computed it, after placing the outputs at :meth:`JobSpec.local_path`). A non-local
  executor reports job state through the same files as a detached local job:
  ``<name>.job`` (the job id, in place of ``<name>.pid``), ``<name>.done`` (the real exit
  code) and ``<name>.log``; :func:`job_status` reads both kinds.
* **Return step.** After a zero exit every expected output is recomputed here and compared
  with what the executor reported; a missing or changed output fails non-zero. The job
  manifest :func:`run_job` returns is what a stage copies into its manifest
  (:func:`record_job`).

:class:`LocalExecutor`, the default, runs main's exact call: ``gpu.run_gpu_stage`` (the
residency guard, then capped.sh's memory cap and floor watchdog) with the plan's argv and
``{**os.environ, **plan_env}``. Nothing here imports a cloud SDK.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.distill import dataset as ddataset
from jev_factory.distill import prereg as dp
from jev_factory.factory import detach, gpu, stages
from jev_factory.release import hub, scan

#: Where inputs from outside the staged folder go inside it.
EXTERNAL = "external"
#: The sides no staged record may come from (spec c11; the dataset stage writes them).
PROTECTED_SIDES = ("held_out", "test", "fit", "selection")
#: The dataset stage's side files, relative to the run directory.
DATASET_REL = "distill/dataset"

#: Trainer flags that name a path, and the role each is staged under when outside the run.
_PATH_FLAGS = {
    "--prereg": "prereg",
    "--run-dir": "run_dir",
    "--train-file": "train_file",
    "--init-from": "init_from",
    "--output-dir": "output_dir",
    "--heal": "heal",
}
#: Flags whose path is a job input (the rest are the staged root and the output).
_INPUT_FLAGS = ("--prereg", "--train-file", "--init-from", "--heal")
#: Roles staged as a whole directory under ``external/<role>``.
_DIR_ROLES = ("init_from", "output_dir")
#: Inputs not scanned for secrets: the pinned checkpoint the trainer starts from (the public
#: stock model, verified against its pin by the stage, or the bf16 student the heal starts
#: from); its weights and tokenizer files are hashed but are not operator text.
_UNSCANNED_ROLES = ("init_from",)
_SECRET_NAME_RE = re.compile(
    r"(?i)(token|secret|passw|pwd|api[_-]?key|access[_-]?key|private[_-]?key|credential)"
)

sha256_path = stages.sha256_path


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


# --------------------------------------------------------------------- job spec


@dataclass(frozen=True)
class StagedFile:
    """One staged input: its path relative to the staged folder and its sha256 (None when the
    file does not exist; the local trainer refuses it itself, a non-local stage refuses it)."""

    path: str
    sha256: str | None
    role: str = ""


@dataclass(frozen=True)
class JobEnvironment:
    """The trainer environment: a python path xor an image digest, plus a lockfile hash, and
    the non-secret variables the job needs (never a key, never a local path)."""

    python: str | None = None
    image_digest: str | None = None
    lockfile_hash: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.python:
            out["python"] = self.python
        if self.image_digest:
            out["image_digest"] = self.image_digest
        out["lockfile_hash"] = self.lockfile_hash
        out["variables"] = dict(sorted(self.variables.items()))
        return out


@dataclass(frozen=True)
class JobLimits:
    """The memory cap and floor, the timeout and the foreign-GPU override."""

    memory_max: str | None = None
    memory_floor: str = "8G"
    timeout: float | None = None
    allow_foreign: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_max": self.memory_max,
            "memory_floor": self.memory_floor,
            "timeout": self.timeout,
            "allow_foreign": self.allow_foreign,
        }


@dataclass(frozen=True)
class LocalBinding:
    """This box's view of a job: main's exact argv, env overrides and run dir, and where each
    ``external/<role>`` root lives. Never serialised, never shipped."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    run_dir: str
    staged: Path
    external: Mapping[str, Path]


@dataclass(frozen=True)
class JobSpec:
    """An executor-neutral job; :meth:`to_json` is what a dry run prints and a job ships."""

    stage: str
    name: str
    argv: tuple[str, ...]
    inputs: tuple[StagedFile, ...]
    outputs: tuple[str, ...]
    environment: JobEnvironment
    limits: JobLimits
    binding: LocalBinding = field(repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "argv": list(self.argv),
            "inputs": [{"path": i.path, "sha256": i.sha256} for i in self.inputs],
            "outputs": list(self.outputs),
            "environment": self.environment.to_json(),
            "limits": self.limits.to_json(),
        }

    def local_path(self, rel: str) -> Path:
        """Where staged path *rel* lives on this box (inputs are read and outputs land here)."""
        if rel.split("/", 1)[0] == EXTERNAL:
            for root, path in self.binding.external.items():
                if rel == root:
                    return path
                if rel.startswith(root + "/"):
                    return path / rel[len(root) + 1 :]
        return self.binding.staged / rel


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    try:
        return argv[list(argv).index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _portable(value: str, role: str, staged: Path) -> tuple[str, Path, bool]:
    """``(relative token, local path, external)`` for a path the trainer is given."""
    local = Path(os.path.abspath(value))
    try:
        rel = local.relative_to(staged).as_posix()
        return (rel or "."), local, False
    except ValueError:
        pass
    if role in _DIR_ROLES:
        return f"{EXTERNAL}/{role}", local, True
    return f"{EXTERNAL}/{role}/{local.name}", local, True


def _input_files(rel: str, local: Path, role: str) -> list[StagedFile]:
    if local.is_dir():
        return [
            StagedFile(f"{rel}/{sub}", digest, role) for sub, digest in hub.digests(local).items()
        ]
    return [StagedFile(rel, sha256_path(local), role)]


def job_spec_from_plan(
    plan: Mapping[str, Any],
    *,
    expected_outputs: Sequence[str] = (),
    environment: JobEnvironment | None = None,
    limits: JobLimits = JobLimits(),
    name: str | None = None,
) -> JobSpec:
    """The :class:`JobSpec` for a ``train_distill.plan_train`` plan; reads, writes nothing.

    *expected_outputs* are file or directory names relative to the plan's output directory;
    each must come back (and match its reported sha256) after a zero exit."""
    argv = [str(a) for a in plan["argv"]]
    run_dir = str(plan.get("run_dir") or _flag_value(argv, "--run-dir"))
    staged = Path(os.path.abspath(run_dir))
    portable = ["python", *argv[1:]]
    tokens: dict[str, str] = {}
    external: dict[str, Path] = {}
    inputs: list[StagedFile] = []
    for flag, role in _PATH_FLAGS.items():
        value = _flag_value(argv, flag)
        if value is None:
            continue
        rel, local, is_external = _portable(value, role, staged)
        portable[argv.index(flag) + 1] = rel
        tokens[role] = rel
        if is_external:
            external[rel] = local
        if flag in _INPUT_FLAGS:
            inputs.extend(_input_files(rel, local, role))
    lock = staged / dp.LOCK_REL
    if lock.is_file() and not any(i.path == dp.LOCK_REL for i in inputs):
        inputs.append(StagedFile(dp.LOCK_REL, sha256_path(lock), "prereg_lock"))
    out_root = tokens.get("output_dir", ".")
    outputs = tuple(n if out_root == "." else f"{out_root}/{n}" for n in expected_outputs)
    env = {str(k): str(v) for k, v in (plan.get("env") or {}).items()}
    return JobSpec(
        stage=str(plan.get("stage") or "train"),
        name=name or f"distill-{plan.get('stage') or 'train'}",
        argv=tuple(portable),
        inputs=tuple(inputs),
        outputs=outputs,
        environment=environment or JobEnvironment(python=argv[0]),
        limits=limits,
        binding=LocalBinding(
            argv=tuple(argv), env=env, run_dir=run_dir, staged=staged, external=external
        ),
    )


def stage_inputs(spec: JobSpec, dest: Path | str) -> Path:
    """Copy every staged input to ``dest/<path>`` and re-check its sha256 (for a non-local
    executor's upload folder). A missing or changed input refuses; nothing absolute leaks."""
    dest = Path(dest)
    for item in spec.inputs:
        src = spec.local_path(item.path)
        if item.sha256 is None or not src.is_file():
            raise _refuse(f"staged input {item.path} does not exist", "build it before submitting")
        target = dest / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        if sha256_path(target) != item.sha256:
            raise _refuse(f"staged input {item.path} changed since the job spec was built")
    return dest


# --------------------------------------------------------------------- seam checks


def _side_ids(path: Path) -> set[str]:
    return set(ddataset.load_sealed(path).ids)


def protected_digests(spec: JobSpec, extra: Iterable[str] = ()) -> tuple[set[str], list[str]]:
    """sha256 of every protected record id the seam knows, and where they came from: the
    dataset stage's held-out, test, fit and selection sides under the staged folder, the heal
    spec's ``protected_id_sha256`` and any digests the caller passes."""
    from jev_factory.backbones.pointer.train_distill import id_digest  # train_distill imports us

    digests = {str(d) for d in extra}
    sources = ["caller"] if digests else []
    for side in PROTECTED_SIDES:
        path = spec.binding.staged / DATASET_REL / f"{side}.jsonl"
        if path.is_file():
            digests |= {id_digest(i) for i in _side_ids(path)}
            sources.append(f"{DATASET_REL}/{side}.jsonl")
    for item in spec.inputs:
        if item.role != "heal" or item.sha256 is None:
            continue
        try:
            heal = json.loads(spec.local_path(item.path).read_text(encoding="utf-8"))
            ids = heal.get("protected_id_sha256") if isinstance(heal, dict) else None
        except (OSError, ValueError):
            ids = None
        if isinstance(ids, list):
            digests |= {str(d) for d in ids}
            sources.append(item.path)
    return digests, sources


def staged_record_ids(spec: JobSpec) -> set[str]:
    """Every ``record_id`` / ``id`` in the staged JSONL inputs."""
    ids: set[str] = set()
    for item in spec.inputs:
        if item.sha256 is None or not item.path.endswith(".jsonl"):
            continue
        text = spec.local_path(item.path).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                ids.update(str(row[k]) for k in ("record_id", "id") if row.get(k) is not None)
    return ids


def check_protected(spec: JobSpec, extra: Iterable[str] = ()) -> dict[str, Any]:
    """Refuse (exit 1, ids counted, never printed) a staged record id on a protected side."""
    from jev_factory.backbones.pointer.train_distill import id_digest

    protected, sources = protected_digests(spec, extra)
    ids = staged_record_ids(spec)
    hits = sorted(i for i in ids if id_digest(i) in protected)
    if hits:
        raise _refuse(
            f"the staged inputs hold {len(hits)} protected record id(s) (held-out, test, fit "
            "or selection); nothing was submitted to any executor",
            "rebuild the training set from the train side only; protected sides never train",
        )
    return {"staged_ids": len(ids), "protected_ids": len(protected), "sources": sources}


def _script_flags(folder: Path, files: list[tuple[str, str]]) -> bool:
    """``scripts/scan-secrets.py``'s verdict, through the upload stage's own runner."""
    from jev_factory.distill import bundle as dbundle  # bundle -> quant -> train_distill -> us

    return dbundle._script_flags(folder, files)


def _credential_hits(files: list[tuple[str, str]]) -> list[tuple[str, int]]:
    """The upload stage's in-process scans: credentials, endpoints, hostnames/private IPs."""
    hits = []
    for rel, text in files:
        found = (*scan.scan_credentials(rel, text), *scan.scan_endpoints(rel, text))
        hits.extend((rel, f.line) for f in found)
        hits.extend((rel, number) for number, _ in scan.private_hosts(text))
    return hits


def _scanned_inputs(spec: JobSpec) -> list[tuple[str, Path, str]]:
    out = []
    for item in spec.inputs:
        if item.role in _UNSCANNED_ROLES or item.sha256 is None:
            continue
        local = spec.local_path(item.path)
        try:
            out.append((item.path, local, local.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue  # a binary input; the trainer's inputs here are JSON / JSONL text
    return out


def check_scans(spec: JobSpec) -> dict[str, Any]:
    """Refuse (exit 1, file and line only) on a secret, endpoint or hostname/private IP in
    the staged inputs, the job spec or the environment, or a secret-named variable."""
    files = _scanned_inputs(spec)
    env_lines = [f"{k}={v}" for k, v in {**spec.binding.env, **spec.environment.variables}.items()]
    texts = [(rel, text) for rel, _, text in files]
    texts.append(("<job spec>", json.dumps(spec.to_json(), indent=1, sort_keys=True)))
    texts.append(("<environment>", "\n".join(env_lines)))
    lines = [f"secret/host: {rel}:{n}" for rel, n in sorted(set(_credential_hits(texts)))]
    names = sorted(
        k for k in {**spec.binding.env, **spec.environment.variables} if _SECRET_NAME_RE.search(k)
    )
    lines += [f"secret-named variable: {k}" for k in names]
    by_anchor: dict[str, list[tuple[str, str]]] = {}
    for _, local, _ in files:
        by_anchor.setdefault(local.anchor, []).append((str(local.relative_to(local.anchor)), ""))
    if any(_script_flags(Path(anchor), group) for anchor, group in by_anchor.items()):
        lines.append("secret: staged inputs (scan-secrets.py hit; line unavailable)")
    if lines:
        raise _refuse(
            "the job's staged inputs or environment failed the pre-submit scan "
            "(file:line only):\n  " + "\n  ".join(lines),
            "remove the flagged values; secrets go by env-var name, hosts stay on this box",
        )
    return {"files": len(files)}


def preflight(spec: JobSpec, protected: Iterable[str] = ()) -> dict[str, Any]:
    """Both seam checks, before any executor sees the job."""
    return {"protected_check": check_protected(spec, protected), "scan": check_scans(spec)}


# --------------------------------------------------------------------- executors


@dataclass(frozen=True)
class JobHandle:
    """A submitted job: which executor holds it and its job id."""

    executor: str
    job_id: str


@dataclass(frozen=True)
class JobResult:
    """A finished job: its exit code, each output's sha256 as the job side computed it, and
    the tails the caller reports (never a secret: the job has none)."""

    returncode: int
    outputs: Mapping[str, str] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    log: Sequence[str] = ()


class Executor(Protocol):
    """Where a job runs. Takes only the job spec; the seam does the checks and verification."""

    name: str

    def submit(self, spec: JobSpec) -> JobHandle:
        """Start (or run) the job; return its handle."""
        ...  # pragma: no cover

    def fetch(self, handle: JobHandle, spec: JobSpec) -> JobResult:
        """Wait for the job, place its outputs at ``spec.local_path``, report their sha256."""
        ...  # pragma: no cover


def output_digests(spec: JobSpec) -> dict[str, str]:
    """``relative path -> sha256`` of every expected output present on this box (a directory
    output is listed per file)."""
    found: dict[str, str] = {}
    for rel in spec.outputs:
        local = spec.local_path(rel)
        if local.is_dir():
            found.update({f"{rel}/{sub}": d for sub, d in hub.digests(local).items()})
        elif local.is_file():
            found[rel] = sha256_path(local) or ""
    return found


class LocalExecutor:
    """Today's execution, unchanged: ``gpu.run_gpu_stage`` in place, synchronously."""

    name = "local"

    def __init__(self) -> None:
        self._done: dict[str, Any] = {}

    def submit(self, spec: JobSpec) -> JobHandle:
        b = spec.binding
        done = gpu.run_gpu_stage(
            spec.stage,
            b.run_dir,
            list(b.argv),
            allow_foreign=spec.limits.allow_foreign,
            memory_max=spec.limits.memory_max,
            memory_floor=spec.limits.memory_floor,
            env={**os.environ, **b.env},
            timeout=spec.limits.timeout,
        )
        job_id = f"local-{len(self._done) + 1}"
        self._done[job_id] = done
        return JobHandle(self.name, job_id)

    def fetch(self, handle: JobHandle, spec: JobSpec) -> JobResult:
        done = self._done.pop(handle.job_id)
        rc = int(done.returncode)
        return JobResult(
            returncode=rc,
            outputs=output_digests(spec) if rc == 0 else {},
            stdout=done.stdout or "",
            stderr=done.stderr or "",
        )


def verify_outputs(spec: JobSpec, result: JobResult) -> dict[str, str]:
    """Recompute every expected output here and compare with the executor's report."""
    local = output_digests(spec)
    problems = []
    for rel in spec.outputs:
        mine = {k: v for k, v in local.items() if k == rel or k.startswith(rel + "/")}
        theirs = {k: v for k, v in result.outputs.items() if k == rel or k.startswith(rel + "/")}
        if not mine or not theirs:
            problems.append(f"{rel}: missing")
            continue
        for key in sorted(set(mine) | set(theirs)):
            if key not in mine or key not in theirs:
                problems.append(f"{key}: missing")
            elif mine[key] != theirs[key]:
                problems.append(f"{key}: sha256 differs")
    if problems:
        raise CliError(
            EXIT_ENV_ERROR,
            "the job's outputs failed verification:\n  " + "\n  ".join(problems),
            "re-run the job; an output that is missing or changed in transit is never used",
        )
    return local


def _tail(text: str) -> list[str]:
    return (text or "").strip().splitlines()[-5:]


def run_job(
    spec: JobSpec, executor: Executor | None = None, *, protected: Iterable[str] = ()
) -> dict[str, Any]:
    """Check, submit, fetch and verify one job; return its job manifest.

    A non-zero exit returns the manifest unverified (``verified`` false) for the caller to
    report; a zero exit with a missing or changed output raises."""
    executor = executor or LocalExecutor()
    checks = preflight(spec, protected)
    handle = executor.submit(spec)
    result = executor.fetch(handle, spec)
    manifest: dict[str, Any] = {
        "executor": executor.name,
        "job_id": handle.job_id,
        "spec": spec.to_json(),
        **checks,
        "returncode": int(result.returncode),
        "outputs": {},
        "verified": False,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "log_tail": list(result.log)[-5:],
    }
    if result.returncode == 0:
        manifest["outputs"] = verify_outputs(spec, result)
        manifest["verified"] = True
    return manifest


def record_job(stage_manifest: Mapping[str, Any], job_manifest: Mapping[str, Any]) -> dict:
    """A copy of *stage_manifest* with *job_manifest* appended under ``jobs``."""
    out = dict(stage_manifest)
    out["jobs"] = [*stage_manifest.get("jobs", []), dict(job_manifest)]
    return out


# --------------------------------------------------------------------- job files


def _job_paths(jobdir: Path, name: str) -> tuple[Path, Path, Path, Path]:
    jobdir = Path(jobdir)
    return (
        jobdir / f"{name}.pid",
        jobdir / f"{name}.job",
        jobdir / f"{name}.done",
        jobdir / f"{name}.log",
    )


def write_job_id(jobdir: Path, name: str, job_id: str) -> Path:
    """Record a non-local job's id in ``<name>.job`` (and clear an old done marker)."""
    _, job_f, done_f, _ = _job_paths(jobdir, name)
    job_f.parent.mkdir(parents=True, exist_ok=True)
    done_f.unlink(missing_ok=True)
    job_f.write_text(f"{job_id}\n", encoding="utf-8")
    return job_f


def mark_done(jobdir: Path, name: str, returncode: int) -> Path:
    """Write the job's real exit code to ``<name>.done`` atomically, as detach's wrapper does."""
    _, _, done_f, _ = _job_paths(jobdir, name)
    tmp = done_f.with_suffix(".done.tmp")
    tmp.write_text(f"{int(returncode)}\n", encoding="utf-8")
    os.replace(tmp, done_f)
    return done_f


def append_log(jobdir: Path, name: str, line: str) -> Path:
    """Append one line to the job's ``<name>.log``."""
    _, _, _, log_f = _job_paths(jobdir, name)
    log_f.parent.mkdir(parents=True, exist_ok=True)
    with log_f.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
    return log_f


def job_status(jobdir: Path, name: str) -> dict[str, Any]:
    """state running | done | died | absent, from ``<name>.pid`` (a detached local job, via
    :func:`jev_factory.factory.detach.job_status`) or ``<name>.job`` (a non-local job)."""
    pid_f, job_f, done_f, _ = _job_paths(jobdir, name)
    if pid_f.is_file() or not job_f.is_file():
        return {**detach.job_status(Path(jobdir), name), "job_id": None}
    rc = detach._read_rc(done_f)
    return {
        "state": "done" if rc is not None else "running",
        "pid": None,
        "job_id": job_f.read_text(encoding="utf-8").strip(),
        "rc": rc,
        "progress": detach.read_progress(Path(jobdir), name),
    }
