"""The build pipeline: every step of a jev-like scorer build as a registered stage.

This replaces nvsh's ``scripts/lfm-finetune/pipeline.sh``: its shell stages
with no stage state become ordered, pure-Python stages on the stage engine
(:mod:`jev_factory.factory.stages`), each with declared inputs and outputs, a
manifest per run and staleness. The shell script's guards ("die if the prior
artifact is missing", "refuse without FINAL=1", "a build that was never
quantized is refused", ...) are re-expressed as stage checks and tested in
``tests/test_factory_pipeline.py``.

Stages, in order (:data:`STAGE_ORDER`); each one's documented sub-steps are in
:data:`SUBSTEPS`::

    config, preregister, seed, teachers-pilot, draft-heldout, draft-eval, split,
    snapshot, baseline, augment, targeted, assemble, train,
    select (calibrate -> gate fit -> probe -> rule), quantize, heal, recalibrate,
    measure-final, edge-check, bundle, dataset-bundle, upload, release-gate

Running a stage
---------------

A stage runs through :func:`run` with a :class:`RunContext` (the validated
Domain, the resolved run config and the injectable :class:`Services`):

* without ``apply`` nothing is written: :func:`plan` describes what the stage
  would read and write and why it is (or is not) stale -- the dry run;
* with ``apply`` the work dir is checked (never inside a git worktree), the run
  lock is held for the whole stage (:class:`~jev_factory.factory.lock.RunLock`;
  the measurement code itself never takes it), and the stage engine runs it
  and writes ``manifests/<stage>.json``. A fresh stage is a no-op.

Knobs are JSON values recorded in the manifest; every stage has documented
defaults (:data:`DEFAULT_KNOBS`) and unknown knobs are refused. Services that
need a GPU, the teacher gateway, llama.cpp or the network (training, the
held-out drafter, measurement, probing, the hub, the release gate) are seams on
:class:`Services`, so every stage is testable with stubs; the defaults run the
real thing (training and measurement under the training environment's python,
``train_py``, with this package importable through ``PYTHONPATH``).

Invariants kept here
--------------------

* the pre-registration is required (hash-checked) before ``train``, ``select``
  and :func:`decide`; a changed file needs the run's deviation id;
* ``select`` fits calibration and the gate on the **fit fold only** and applies
  the pre-registered rule mechanically (:mod:`jev_factory.decide.rules`);
  measurement writes pre-gate predictions and the fitted ``gate.json`` is
  applied here (``select``, ``recalibrate``, ``measure-final``);
* ``recalibrate`` re-fits calibration and the gate on the deployed quantized
  build's **own** validation predictions;
* the test side and the sealed held-out set are measured once
  (:mod:`jev_factory.measure.once` refuses a second run without a deviation);
* training data is chosen by the frozen sha256, never by modification time;
* heal is one round only; upload is private-first and needs the operator to
  name the repository; ``edge-check`` only records what the operator ran on
  the edge device and never connects anywhere.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import io
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import jev_factory
from jev_factory import __version__
from jev_factory.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from jev_factory.core import calibration as calib
from jev_factory.core import gate as gate_mod
from jev_factory.core import metrics, split, sweep_gate
from jev_factory.core.predictions import Prediction, read_predictions, write_predictions
from jev_factory.decide import records, rules
from jev_factory.domain.model import Domain
from jev_factory.domain.validate import DomainError, load_domain
from jev_factory.factory import prereg as prereg_mod
from jev_factory.factory.config import RunConfig
from jev_factory.factory.lock import RunLock
from jev_factory.factory.stages import (
    COMPLETE,
    DEFAULT_REGISTRY,
    Registry,
    Stage,
    read_manifest,
    run_stage,
    sha256_path,
    staleness,
)
from jev_factory.factory.workroot import resolve_work_root

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/pipeline.sh",
    "commit": "9debdc6",
    "adaptations": [
        "the shell script is not imported: its stage contract (lines 1-160, the STAGES"
        " list at 162-164) becomes ordered stage-engine stages with manifests",
        "its guards (die-if-missing sequencing, FINAL=1 before upload, stock-copy and"
        " greedy decoding before a measure, never-quantized builds, the frozen training"
        " set) become stage checks re-expressed from tests/test_lfm_finetune_pipeline.py",
        "train-scorer's mtime choice of the training file (line 648) is replaced by the"
        " frozen sha256 (data.assemble.select_frozen)",
        "Track A stages (skills, augment-skills, measure-skills, train) are dropped;"
        " the hard-coded hub prefix (lines 500, 508) comes from the run config/Domain",
        "added stages the script never had: preregister, teachers-pilot, draft-heldout,"
        " draft-eval, snapshot, select, recalibrate, edge-check, dataset-bundle (d14),"
        " release-gate",
        "d17 (2026-10-08): restructured for SonarCloud code quality (cognitive "
        "complexity split into private helpers, plus lint-level cleanups); behaviour "
        "unchanged, pinned by tests/test_complexity_*.py and differential checks against "
        "the imported version",
    ],
    "licence": "Apache-2.0",
}

# ---------------------------------------------------------------------------
# The stage list
# ---------------------------------------------------------------------------

STAGE_ORDER: tuple[str, ...] = (
    "config",
    "preregister",
    "seed",
    "teachers-pilot",
    "draft-heldout",
    "draft-eval",
    "split",
    "snapshot",
    "baseline",
    "augment",
    "targeted",
    "assemble",
    "train",
    "select",
    "quantize",
    "heal",
    "recalibrate",
    "measure-final",
    "edge-check",
    "bundle",
    "dataset-bundle",
    "upload",
    "release-gate",
)

#: Each stage's documented sub-steps, in the order the stage runs them.
SUBSTEPS: dict[str, tuple[str, ...]] = {
    "config": ("validate-domain", "check-work-root", "record-config"),
    "preregister": ("validate-bars", "hash-into-run"),
    "seed": ("validate-seed-corpus", "copy-seed"),
    "teachers-pilot": ("check-teacher-licences", "pilot-draft", "two-reviewer-check", "yield"),
    "draft-heldout": ("non-teacher-draft", "two-reviewer-check", "seal", "import-sealed"),
    "draft-eval": ("draft-eval-pool", "dedupe", "two-reviewer-check"),
    "split": ("grouped-split", "fit-selection-folds", "missing-candidate-slice"),
    "snapshot": ("grounding-snapshot",),
    "baseline": ("stock-copy", "measure-stock", "stock-values"),
    "augment": ("generate", "correct", "review"),
    "targeted": ("targeted-recipes",),
    "assemble": ("merge", "leakage-check", "scorer-rows", "freeze"),
    "train": (
        "prereg-check",
        "frozen-data-by-sha256",
        "train-scorer",
        "merge-check",
        "stage-cache",
        "measure-val",
    ),
    "select": ("calibrate", "gate-fit", "probe", "rule"),
    "quantize": ("gguf-q4_k_m", "measure-quant", "heal-trigger"),
    "heal": ("heal-once", "re-quantize", "heal-check"),
    "recalibrate": ("calibrate", "gate-fit"),
    "measure-final": (
        "measure-test",
        "measure-heldout",
        "missing-candidate-slice",
        "apply-gate",
        "probe",
        "bars",
    ),
    "edge-check": ("record-operator-results",),
    "bundle": ("model-bundle", "scan"),
    "dataset-bundle": ("train-set-used", "dataset-bundle", "scan"),
    "upload": ("private-upload", "fetch-back-verify"),
    "release-gate": ("evals-run",),
}

# Work-dir layout (paths relative to the run's work dir).
CONFIG_FILE = "config.json"
PREREG_FILE = "prereg.json"
PREREG_LOCK = prereg_mod.LOCK_NAME
SEED_FILE = "seed/seed.json"
PILOT_DIR = "pilot"
PILOT_YIELDS = "pilot/yields.json"
HELDOUT_FILE = "heldout/held-out.json"
HELDOUT_SUMMARY = "heldout/summary.json"
POOL_FILE = "pool/draft.json"
SPLIT_DIR = "splits"
TRAIN_SPLIT = "splits/train.json"
VAL_SPLIT = "splits/val.json"
TEST_SPLIT = "splits/test.json"
FOLDS_FILE = "splits/folds.json"
MC_FOLDS_FILE = "splits/folds-mc.json"
SNAPSHOT_FILE = "snapshot/ground-snapshot.json"
BASELINE_SUMMARY = "baseline/summary.json"
ACCEPTED_FILE = "aug/accepted.jsonl"
REJECTED_FILE = "aug/rejected.jsonl"
AUGMENT_COUNTS = "aug/counts.json"
TARGETED_SUMMARY = "aug/targeted.json"
SUPPLEMENT_FILE = "aug/supplement.json"
DATA_DIR = "data"
FREEZE_FILE = "data/freeze.json"
TRAIN_REQUEST = "train-request.json"
RUNS_DIR = "runs"
CANDIDATES_DIR = "candidates"
TRAIN_SUMMARY = "train/summary.json"
SELECT_DIR = "select"
SELECTION_FILE = "select/selection.json"
QUANT_SUMMARY = "quant/summary.json"
HEAL_SUMMARY = "heal/summary.json"
DEPLOYED_DIR = "deployed"
DEPLOYED_CALIBRATION = "deployed/calibration.json"
DEPLOYED_GATE = "deployed/gate.json"
DEPLOYED_BUILD = "deployed/build.json"
FINAL_REPORT = "final/report.json"
FINAL_PARTIAL = "final/report.partial.json"
EDGE_RESULTS = "edge/operator-results.json"
EDGE_RECORD = "edge/edge-check.json"
BUNDLE_RECORD = "bundle/record.json"
DATASET_RECORD = "dataset/record.json"
DATASET_TRAIN = "dataset/train-used.json"
UPLOAD_RECORD = "upload/upload.json"
RELEASE_GATE_RESULT = "release-gate/result.json"
DECISIONS_FILE = records.DEFAULT_NAME
JOBS_DIR = "jobs"

#: Predictions of one measurement live alone in their own directory.
VAL_DIR = "val"
VAL_MC_DIR = "val-mc"
PREDICTIONS_GLOB = "*.predictions.jsonl"

MEASURE_MODULE = "jev_factory.measure.run"
PROBE_MODULE = "jev_factory.measure.probe"
EVALS_MODULE = "jev_factory.evals"

#: The default gate grid: every combination is swept on the fit fold.
DEFAULT_GATE_GRID: dict[str, list[float | None]] = {
    "escalate": [None, 0.2, 0.3, 0.5],
    "ro_floor": [None, 0.5],
    "ro_margin": [None, 0.1, 0.2],
    "ro_max_entropy": [None],
    "mut_floor": [None, 0.5, 0.7, 0.9],
    "mut_margin": [None, 0.2],
    "mut_max_entropy": [None],
}

#: Every stage's knobs and their defaults; a knob not listed here is refused.
DEFAULT_KNOBS: dict[str, dict[str, Any]] = {
    "config": {},
    "preregister": {},
    "seed": {},
    "teachers-pilot": {"per_op": 2, "per_reason": 2, "explain": 4, "max_batch": 8},
    "draft-heldout": {
        "per_op": 3,
        "escalate": 16,
        "explain": 16,
        "review": True,
        "import_from": None,
        "import_sha256": None,
    },
    "draft-eval": {
        "per_op": 20,
        "per_reason": 10,
        "explain": 30,
        "max_batch": 8,
        "only_reasons": None,
    },
    "split": {"val_size": 150, "test_size": 150, "fold_seed": 53, "version": "v2"},
    "snapshot": {},
    "baseline": {"probe": True},
    "augment": {"per_source": 2, "decide_by": "reviewer_b"},
    "targeted": {"recipes": [], "per_recipe": 5, "decide_by": "reviewer_b"},
    "assemble": {
        "reasons": False,
        "randomize_labels": True,
        "perm_seed": 0,
        "missing_candidate_rate": 0.3,
    },
    "train": {"candidates": {}},
    "select": {"gate_grid": DEFAULT_GATE_GRID, "bootstrap_resamples": 1000},
    "quantize": {"candidate": None, "awq": False},
    "heal": {},
    "recalibrate": {"gate_grid": DEFAULT_GATE_GRID, "bootstrap_resamples": 1000},
    "measure-final": {
        "heldout": True,
        "probe": True,
        "bootstrap_resamples": 1000,
    },
    "edge-check": {},
    "bundle": {"repo_suffix": None, "data_summary": None},
    "dataset-bundle": {"repo_suffix": None},
    "upload": {"repo_suffix": None, "repo_type": "model", "dataset_repo_suffix": None},
    "release-gate": {"manifest": None, "no_deepeval": False},
}

#: Run-config keys each stage reads: a change to one makes that stage stale.
STAGE_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "config": ("base", "base_rev", "hub_prefix", "licence"),
    "teachers-pilot": ("teacher_generator_model", "teacher_reviewer_a_model"),
    "draft-eval": ("seed", "teacher_generator_model", "teacher_reviewer_b_model"),
    "draft-heldout": ("seed",),
    "split": ("seed",),
    "snapshot": ("ground_snapshot",),
    "baseline": ("base", "base_rev", "measure_ctx"),
    "augment": ("teacher_generator_model", "teacher_reviewer_b_model"),
    "targeted": ("seed", "teacher_generator_model", "teacher_reviewer_b_model"),
    "train": ("base", "base_rev"),
    "quantize": ("measure_ctx",),
    "measure-final": ("measure_ctx",),
    "bundle": ("hub_prefix", "licence"),
    "dataset-bundle": ("hub_prefix", "licence", "issue_refs"),
    "upload": ("hub_prefix",),
}


# ---------------------------------------------------------------------------
# Context and services
# ---------------------------------------------------------------------------


def _package_root() -> Path:
    """The directory that holds the ``jev_factory`` package (what ``PYTHONPATH`` needs)."""
    return Path(jev_factory.__file__).resolve().parent.parent


def with_package_path(environ: Mapping[str, str]) -> dict[str, str]:
    """*environ* with this package importable first on ``PYTHONPATH``.

    Training and measurement run as ``python -m jev_factory...`` under another
    environment's python (``train_py``); that interpreter finds this package
    through ``PYTHONPATH``, never through an install into it.
    """
    env = dict(environ)
    parts = [str(_package_root())]
    parts += [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != parts[0]]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def default_run_module(
    module: str, argv: Sequence[str], *, python: str, env: Mapping[str, str]
) -> int:
    """Run ``python -m module argv`` (a fixed list, no shell); return its exit code."""
    proc = subprocess.run(  # nosec B603 - fixed argv, operator-configured interpreter
        [python, "-m", module, *argv], env=dict(env), check=False
    )
    return proc.returncode


@contextlib.contextmanager
def default_serve(ctx: "RunContext", model: Path, name: str, run_dir: Path) -> Iterator[str]:
    """Serve *model* (a ``.gguf``) on the run config's port; yield its base URL; stop it."""
    from jev_factory.measure import serve

    settings = serve.with_name(serve.ServeSettings.from_config(ctx.config, run_dir), name)
    port = int(ctx.config.get("measure_port") or 18060)
    serve.start(model, port, settings)
    try:
        serve.wait(port, settings)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        serve.stop(port, settings)


@dataclass
class Services:
    """Every side-effecting seam a stage uses. ``None`` means the real default."""

    #: ``TeacherClient`` for pilot/draft/augment/targeted (default: from the run config).
    teacher_client: Any = None
    #: ``(seed) -> (drafter, snapshot)`` for the sealed held-out (default: in-process model).
    drafter: Callable[[int], tuple[Callable[[str, str], str], str]] | None = None
    #: ``(module, argv, *, python, env) -> rc``: measurement, the probe, the release gate.
    run_module: Callable[..., int] = default_run_module
    #: The runner ``backbones.causal_lm.train.run_plan`` applies commands with.
    train_runner: Callable[..., Any] | None = None
    #: The ``RunFn`` quantize applies commands with (default: the guarded GPU runner).
    quant_run: Callable[..., Any] | None = None
    #: ``(ctx, model, name, run_dir)`` context manager yielding a served base URL.
    serve: Callable[..., Any] = default_serve
    #: The hub client module (default: ``huggingface_hub``, imported only on upload).
    hub: Any = None
    #: Environment the stages read secrets from and pass to subprocesses.
    environ: Mapping[str, str] | None = None
    #: ``() -> YYYY-MM-DD``.
    today: Callable[[], str] | None = None


@dataclass
class RunContext:
    """What every stage runs against: the Domain, the run config and the services."""

    domain: Domain
    domain_ref: str
    config: RunConfig
    deviation_id: str | None = None
    allow_foreign_gpu: bool = False
    services: Services = field(default_factory=Services)

    @classmethod
    def load(cls, domain_ref: str, config: RunConfig, **kw: Any) -> "RunContext":
        """Validate the domain named by *domain_ref* (a dotted module or a JSON file)."""
        try:
            domain = load_domain(domain_ref)
        except DomainError as exc:
            raise CliError(EXIT_USER_ERROR, f"invalid domain {domain_ref!r}: {exc}") from exc
        return cls(domain=domain, domain_ref=domain_ref, config=config, **kw)

    def env(self) -> dict[str, str]:
        base = os.environ if self.services.environ is None else self.services.environ
        return with_package_path(base)

    def today(self) -> str:
        if self.services.today is not None:
            return self.services.today()
        import datetime

        return datetime.date.today().isoformat()


_CURRENT: contextvars.ContextVar[RunContext] = contextvars.ContextVar("jev_pipeline_context")


def _ctx() -> RunContext:
    try:
        return _CURRENT.get()
    except LookupError:
        raise CliError(
            EXIT_USER_ERROR,
            "a pipeline stage ran without a run context",
            "run stages through jev_factory.factory.pipeline.run",
        ) from None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _err(message: str, remediation: str = "", code: int = EXIT_USER_ERROR) -> CliError:
    return CliError(code, message, remediation)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise _err(f"cannot read {path}: {exc.strerror}", "run the stage that writes it") from None
    except ValueError as exc:
        raise _err(f"{path} is not valid JSON: {exc}") from None


def _write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    digest = sha256_path(Path(path))
    if digest is None:
        raise _err(f"{path} does not exist")
    return digest


def _rel(workdir: Path, path: Path) -> str:
    try:
        return Path(path).relative_to(workdir).as_posix()
    except ValueError:
        return str(path)


def _quiet(func: Callable[[], Any]) -> Any:
    """Call *func* with its stdout captured (library mains print counts)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func()


def one_predictions_file(directory: Path) -> Path:
    """The single ``*.predictions.jsonl`` a measurement left in *directory*."""
    found = sorted(Path(directory).glob(PREDICTIONS_GLOB))
    if len(found) != 1:
        raise _err(
            f"{directory}: expected exactly one {PREDICTIONS_GLOB}, found {len(found)}",
            "measure into an empty directory (one measurement per directory)",
        )
    return found[0]


def base_snapshot(config: RunConfig) -> Path:
    """The base model's snapshot in the Hugging Face cache (``hf_cache``/hub layout)."""
    base = str(config.require("base"))
    if "/" not in base:
        raise _err(f"base {base!r} is not an <org>/<name> hub id")
    org, name = base.split("/", 1)
    cache = Path(config.require("hf_cache"))
    return cache / "hub" / f"models--{org}--{name}" / "snapshots" / str(config.require("base_rev"))


def _prereg(workdir: Path, ctx: RunContext) -> tuple[prereg_mod.Prereg, str]:
    """The registered pre-registration (refusing a changed file) and its locked sha256."""
    registered = prereg_mod.require_registered(workdir / PREREG_FILE, workdir, ctx.deviation_id)
    lock = _read_json(workdir / PREREG_LOCK)
    return registered, str(lock["sha256"])


def _candidate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise _err(f"candidate name {name!r} is not usable as a directory name")
    return name


# ---------------------------------------------------------------------------
# Measurement, probing, calibration and the gate (shared by several stages)
# ---------------------------------------------------------------------------


def _python(ctx: RunContext) -> str:
    return str(ctx.config.require("train_py"))


def measure(
    workdir: Path,
    *,
    split_file: Path,
    out_dir: Path,
    label: str,
    model: str,
    revision: str,
    scorer: Sequence[str],
    missing_candidate: bool = False,
    final: bool = False,
    acceptance: bool = False,
    calibration: Path | None = None,
) -> Path:
    """One ``jev_factory.measure.run`` call into its own *out_dir*; return its predictions.

    Measurement writes **pre-gate** predictions (the argmax decision); a stage
    applies the fitted gate afterwards.
    """
    ctx = _ctx()
    out_dir = Path(out_dir)
    if out_dir.exists() and not (final or acceptance):
        shutil.rmtree(out_dir)  # a validation re-measure starts clean
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ["--domain", ctx.domain_ref, "--run-dir", str(workdir), "--split", str(split_file)]
    argv += ["--model", model, "--revision", revision, "--label", label]
    argv += ["--predictions", str(out_dir), "--out", str(out_dir / f"{label}.md")]
    argv += ["--progress-dir", str(workdir / JOBS_DIR)]
    argv += ["--ground-snapshot", str(workdir / SNAPSHOT_FILE), *scorer]
    if missing_candidate:
        argv += ["--slice", "missing-candidate"]
    if final:
        argv.append("--final")
    if acceptance:
        argv.append("--acceptance")
    if (final or acceptance) and ctx.deviation_id:
        # A recorded deviation permits the second sealed measurement and its new page.
        argv += ["--deviation", ctx.deviation_id, "--force"]
    if calibration is not None:
        argv += ["--calibration", str(calibration)]
    if ctx.allow_foreign_gpu:
        argv.append("--allow-foreign-gpu")
    rc = ctx.services.run_module(MEASURE_MODULE, argv, python=_python(ctx), env=ctx.env())
    if rc != 0:
        raise _err(f"measure {label} exited {rc}", "read the measure output above", EXIT_ENV_ERROR)
    return one_predictions_file(out_dir)


def in_process_scorer() -> list[str]:
    return ["--scorer", "in-process"]


def served_gguf_scorer(gguf: Path, tokenizer: Path) -> list[str]:
    ctx = _ctx()
    cfg = ctx.config
    return [
        "--scorer",
        "served",
        "--serve",
        str(gguf),
        "--llama-server",
        str(cfg.require("llama_server")),
        "--tokenizer",
        str(tokenizer),
        "--port",
        str(cfg.get("measure_port") or 18060),
        "--ctx",
        str(cfg.get("measure_ctx") or 2048),
    ]


def probe(
    split_file: Path,
    out: Path,
    *,
    model: str,
    revision: str,
    per_entry: int,
    scorer: Sequence[str] = ("--scorer", "in-process"),
    final: bool = False,
    progress_dir: Path | None = None,
) -> dict:
    """The permutation probe over *split_file*; its report (with ``pooled_change_rate``)."""
    ctx = _ctx()
    argv = ["--domain", ctx.domain_ref, "--split", str(split_file), "--out", str(out)]
    argv += ["--per-entry", str(per_entry), "--model", model, "--revision", revision, *scorer]
    if progress_dir is not None:
        argv += ["--progress-dir", str(progress_dir)]
    if final:
        argv.append("--final")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rc = ctx.services.run_module(PROBE_MODULE, argv, python=_python(ctx), env=ctx.env())
    if rc != 0:
        raise _err(f"the permutation probe exited {rc}", "", EXIT_ENV_ERROR)
    return _read_json(out)


def _grid(spec: Mapping[str, Any]) -> list[gate_mod.Thresholds]:
    unknown = sorted(set(spec) - set(DEFAULT_GATE_GRID))
    if unknown:
        raise _err(f"unknown gate grid knob(s): {', '.join(unknown)}")
    values = {k: list(spec.get(k, DEFAULT_GATE_GRID[k])) for k in DEFAULT_GATE_GRID}
    return sweep_gate.build_threshold_grid(**values)


def gate_objective(mc_bar: float) -> Callable[[list[dict]], int]:
    """The fit-fold gate choice: fewest wrong mutating (0 first), then missing-candidate
    escalation at or above *mc_bar*, then the most right proposals; ties keep grid order."""

    def key(item: tuple[int, dict]) -> tuple:
        index, row = item
        rate = row["missing_candidate_escalation_recall"]["rate"]
        meets = rate is not None and rate >= mc_bar - 1e-9
        return (row["wrong_mutating"]["total"], not meets, -row["right_proposals"]["n"], index)

    def choose(rows: list[dict]) -> int:
        return min(enumerate(rows), key=key)[0]

    return choose


def mc_folds(folds: Mapping[str, Any], mc_ids: set[str]) -> dict[str, list[str]]:
    """The folds extended by each fold id's ``-nocand`` twin (the missing-candidate slice)."""
    out: dict[str, list[str]] = {}
    for key in ("fit_ids", "selection_ids"):
        ids = [str(i) for i in folds.get(key) or []]
        out[key] = ids + [f"{i}-nocand" for i in ids if f"{i}-nocand" in mc_ids]
    return out


def _wrong_mutating_ids(lines: Sequence[Prediction], domain: Domain) -> list[str]:
    return sorted(
        p.id
        for p in lines
        if metrics.compute([p], domain, bootstrap_resamples=0)["wrong_mutating"]["total"] > 0
    )


def _escalated(p: Prediction) -> bool:
    return p.outcome in ("escalate", "abstain_uncertain")


def _failing(p: Prediction, domain: Domain) -> bool:
    kind = metrics.expect_kind(p.expected)
    one = metrics.compute([p], domain, bootstrap_resamples=0)
    if one["wrong_mutating"]["total"] > 0:
        return True
    if kind == "operation":
        return one["right_proposals"]["n"] == 0
    if kind == "escalate":
        return not _escalated(p)
    return p.outcome != "explain"


def _rate(n: int, total: int, what: str) -> float:
    if total <= 0:
        raise _err(f"no {what} lines to measure a rate on")
    return n / total


def calibrate_and_gate(
    val_path: Path,
    mc_path: Path,
    folds: Mapping[str, Any],
    domain: Domain,
    *,
    grid: Sequence[gate_mod.Thresholds],
    mc_bar: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Fit calibration then the gate on the **fit fold**; judge the selection fold.

    Calibration: temperature then vector, fitted on the fit fold; the vector is
    kept only if it lowers selection-fold ECE (:func:`calib.select_calibration`).
    The gate is swept on the calibrated fit-fold lines and their missing-candidate
    twins (:func:`sweep_gate.fit_gate`, which never reads a selection-fold line).
    """
    val = read_predictions(val_path)
    mc = read_predictions(mc_path)
    fitted = calib.fit_params_from_predictions(val, folds, source=str(val_path))
    chosen = calib.select_calibration(val, fitted, folds)
    chosen["fold"] = "fit"
    cal_val = calib.apply_to_predictions(val, chosen)
    cal_mc = calib.apply_to_predictions(mc, chosen)
    both = mc_folds(folds, {p.id for p in mc})
    thresholds, reports = sweep_gate.fit_gate(
        cal_val + cal_mc, both, grid, domain, gate_objective(mc_bar)
    )
    selection = set(both["selection_ids"])
    sel_val = [sweep_gate.redecide(p, thresholds, domain) for p in cal_val if p.id in selection]
    sel_mc = [sweep_gate.redecide(p, thresholds, domain) for p in cal_mc if p.id in selection]
    if not sel_val:
        raise _err("no validation prediction is in the selection fold")
    computed = metrics.compute(sel_val, domain, bootstrap_resamples=bootstrap_resamples)
    mc_computed = metrics.compute(sel_mc, domain, bootstrap_resamples=0)
    right = computed["right_proposals"]
    confidences = [max(p.candidates.values()) for p in sel_val if p.candidates]
    return {
        "calibration": chosen,
        "thresholds": thresholds,
        "sweep": reports,
        "selection": sel_val,
        "selection_mc": sel_mc,
        "metrics": computed,
        "numbers": {
            "wrong_mutating": computed["wrong_mutating"]["total"],
            "wrong_mutating_mc": mc_computed["wrong_mutating"]["total"],
            "right_proposals": _rate(right["n"], right["N"], "selection-fold operation"),
            "ece": computed["calibration"]["ece"],
            "mc_escalation": _rate(
                sum(1 for p in sel_mc if _escalated(p)), len(sel_mc), "missing-candidate"
            ),
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "abstain_recall": computed["abstention"]["recall"],
        },
    }


def gate_document(thresholds: gate_mod.Thresholds, **provenance: Any) -> dict[str, Any]:
    """``gate.json``: the thresholds (``Thresholds.from_json`` reads it) plus how they were fit."""
    return {**thresholds.to_json(), "fit": {"fold": "fit", **provenance}}


def _selection_split(workdir: Path, folds: Mapping[str, Any], out: Path) -> Path:
    """The validation side restricted to the selection fold (what select probes)."""
    raw = _read_json(workdir / VAL_SPLIT)
    keep = set(folds.get("selection_ids") or [])
    doc = {**raw, "entries": [e for e in raw.get("entries", []) if e.get("id") in keep]}
    return _write_json(out, doc)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_config(workdir: Path, knobs: dict[str, Any]) -> None:
    """Validate the domain and the work root; record the resolved config (no secret values)."""
    ctx = _ctx()
    resolve_work_root(workdir)
    domain = load_domain(ctx.domain)
    _write_json(
        workdir / CONFIG_FILE,
        {
            "jev_factory_version": __version__,
            "domain": domain.name,
            "domain_ref": ctx.domain_ref,
            "surface_sha256": domain.surface_sha256(),
            "operations": len(domain.operations),
            "mutating": sum(1 for op in domain.operations if not op.read_only),
            "config": ctx.config.to_manifest(),
        },
    )


def stage_preregister(workdir: Path, knobs: dict[str, Any]) -> None:
    """Validate ``prereg.json`` and hash it into the run (``prereg.lock.json``).

    A lock that already names this very file is kept; a changed file is refused
    (a changed registration is a recorded deviation, never a re-register).
    """
    path = workdir / PREREG_FILE
    lock = workdir / PREREG_LOCK
    if lock.exists() and _read_json(lock).get("sha256") == _sha(path):
        prereg_mod.load(path)
        return
    prereg_mod.register(path, workdir)


def stage_seed(workdir: Path, knobs: dict[str, Any]) -> None:
    """Check the domain's seed corpus against the Domain and copy it into the run."""
    from jev_factory.measure.corpus import load_raw

    domain = _ctx().domain
    try:
        corpus = domain.load_seed_corpus()
    except ValueError as exc:
        raise _err(str(exc), "give the domain a valid seed corpus") from None
    raw = _read_json(Path(domain.seed_corpus))  # type: ignore[arg-type]
    loaded = load_raw(raw, domain)
    if loaded.problems:
        raise _err(
            f"seed corpus has {len(loaded.problems)} invalid entr(y/ies): {loaded.problems[0]}",
            "fix the seed corpus entries against the domain",
        )
    out = workdir / SEED_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(domain.seed_corpus, out)  # type: ignore[arg-type]
    by_kind: dict[str, int] = {}
    for entry in loaded.entries:
        kind = split.expectation_kind(entry.expect)
        by_kind[kind] = by_kind.get(kind, 0) + 1
    _write_json(
        workdir / "seed/summary.json",
        {"entries": len(corpus.entries), "by_kind": by_kind, "sha256": _sha(out)},
    )


def _teacher_client() -> Any:
    ctx = _ctx()
    if ctx.services.teacher_client is not None:
        return ctx.services.teacher_client
    from jev_factory.data.teachers import TeacherClient

    return TeacherClient.from_config(ctx.config)


_PILOT_ID = re.compile(r"^(?P<tag>.+?)(?:-(?P<label>rejected|error))?-\d{3}$")


def pilot_yields(review_rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, dict]:
    """Per drafted class: accepted and reviewed counts (errors are not reviews)."""
    prefix = f"s{seed}-eval-"
    out: dict[str, dict[str, int]] = {}
    for row in review_rows:
        entry_id = str(row.get("id", ""))
        match = _PILOT_ID.match(entry_id.removeprefix(prefix))
        if match is None:
            continue
        tally = out.setdefault(match["tag"], {"accepted": 0, "reviewed": 0, "errors": 0})
        if match["label"] == "error":
            tally["errors"] += 1
            continue
        tally["reviewed"] += 1
        tally["accepted"] += 0 if match["label"] == "rejected" else 1
    return out


def stage_teachers_pilot(workdir: Path, knobs: dict[str, Any]) -> None:
    """A small teacher pilot: draft, two-reviewer check, and each class's review yield.

    The operator reads the pilot (about 20 kept and 20 rejected); a class whose
    yield is under 30% makes ``jev decide`` say "fix the grader first".
    """
    from jev_factory.data import draft

    ctx = _ctx()
    client = _teacher_client()
    out = workdir / PILOT_DIR
    seed = int(ctx.config.get("seed") or 0)
    result = draft.run_draft(
        ctx.domain,
        out,
        seed,
        int(knobs["per_op"]),
        int(knobs["per_reason"]),
        int(knobs["explain"]),
        client,
        max_batch=int(knobs["max_batch"]),
        jobdir=workdir / JOBS_DIR / "pilot",
    )
    rows = [json.loads(line) for line in (out / "review.jsonl").read_text().splitlines() if line]
    yields = pilot_yields(rows, seed)
    _write_json(
        workdir / PILOT_YIELDS,
        {
            "classes": {
                name: {**t, "rate": (t["accepted"] / t["reviewed"]) if t["reviewed"] else 0.0}
                for name, t in sorted(yields.items())
            },
            "kept": result["kept"],
            "rejects": result["rejects"],
        },
    )


def stage_draft_heldout(workdir: Path, knobs: dict[str, Any]) -> None:
    """The sealed held-out set: drafted by a non-teacher model, reviewed, written read-only.

    ``import_from`` + ``import_sha256`` instead copies an already sealed file,
    hash-verified (a migrated sealed set is only ever copied, never redrafted).
    Only counts and sha256 are recorded; no entry text is ever printed.
    """
    from jev_factory.data import draft

    ctx = _ctx()
    out = workdir / HELDOUT_FILE
    if out.exists():
        raise _err(
            f"{out} already exists: the held-out set is sealed",
            "a redraft of a sealed set is a recorded deviation, never a rerun",
        )
    if knobs.get("import_from"):
        source = Path(knobs["import_from"])
        wanted = knobs.get("import_sha256")
        if not wanted or _sha(source) != wanted:
            raise _err(
                "an imported sealed held-out set must match its import_sha256",
                "pass the sha256 recorded when the set was sealed",
            )
        doc = _read_json(source)
    else:
        if ctx.services.drafter is not None:
            drafter, snapshot = ctx.services.drafter(int(ctx.config.get("seed") or 0))
        else:
            drafter, snapshot = draft.load_qwen_drafter(int(ctx.config.get("seed") or 0))
        raw_dir = workdir / "heldout" / "raw"
        drafted = draft.draft_heldout(
            ctx.domain,
            raw_dir,
            int(ctx.config.get("seed") or 0),
            drafter,
            snapshot=snapshot,
            per_op=int(knobs["per_op"]),
            escalate=int(knobs["escalate"]),
            explain=int(knobs["explain"]),
        )
        source = Path(drafted["path"])
        if knobs.get("review", True):
            reviewed_dir = workdir / "heldout" / "reviewed"
            draft.run_review(ctx.domain, source, reviewed_dir, _teacher_client())
            source = reviewed_dir / "draft.json"
        doc = _read_json(source)
    draft.write_sealed(out, doc)
    _write_json(workdir / HELDOUT_SUMMARY, draft.summarize_sealed(out))


def stage_draft_eval(workdir: Path, knobs: dict[str, Any]) -> None:
    """Draft a fresh eval pool with the teachers (resumable per item) into ``pool/``."""
    from jev_factory.data import draft

    ctx = _ctx()
    only = knobs.get("only_reasons")
    draft.run_draft(
        ctx.domain,
        workdir / "pool",
        int(ctx.config.get("seed") or 0),
        int(knobs["per_op"]),
        int(knobs["per_reason"]),
        int(knobs["explain"]),
        _teacher_client(),
        only_reasons=list(only) if only else None,
        max_batch=int(knobs["max_batch"]),
        jobdir=workdir / JOBS_DIR / "draft-eval",
    )


def stage_split(workdir: Path, knobs: dict[str, Any]) -> None:
    """Grouped split of the eval pool (seed train-only), seeded fit/selection folds, MC slice."""
    from jev_factory.measure.slices import missing_candidate_slice

    ctx = _ctx()
    out = workdir / SPLIT_DIR
    argv = ["--corpus", str(workdir / POOL_FILE), "--train-only", str(workdir / SEED_FILE)]
    argv += ["--out-dir", str(out), "--seed", str(ctx.config.get("seed") or 0)]
    argv += ["--val-size", str(knobs["val_size"]), "--test-size", str(knobs["test_size"])]
    argv += ["--fold-seed", str(knobs["fold_seed"]), "--version", str(knobs["version"])]
    try:
        rc = _quiet(lambda: split.main(argv, domain=ctx.domain))
    except SystemExit as exc:  # argparse's parser.error
        raise _err(f"split refused its inputs (exit {exc.code})", "see the message above") from None
    if rc != 0:
        raise _err(f"split exited {rc}")
    folds = _read_json(workdir / FOLDS_FILE)
    val = _read_json(workdir / VAL_SPLIT)
    header = val["header"] if isinstance(val.get("header"), str) else json.dumps(val["header"])
    mc = missing_candidate_slice({"header": header, "entries": val["entries"]}, ctx.domain.names())
    _write_json(workdir / MC_FOLDS_FILE, mc_folds(folds, {e["id"] for e in mc["entries"]}))


def stage_snapshot(workdir: Path, knobs: dict[str, Any]) -> None:
    """The one grounding snapshot every measurement uses (or the configured one, checked)."""
    from jev_factory.measure import snapshot as snap

    ctx = _ctx()
    out = workdir / SNAPSHOT_FILE
    configured = ctx.config.get("ground_snapshot")
    try:
        if configured:
            snap.load_snapshot(Path(configured), ctx.domain)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(configured, out)
            return
        world = _read_json(workdir / SEED_FILE).get("world")
        doc, _counts = snap.build_snapshot(
            ctx.domain,
            [workdir / TRAIN_SPLIT, workdir / VAL_SPLIT, workdir / TEST_SPLIT],
            source="jev snapshot stage",
            created=ctx.today(),
            base_world=world,
        )
    except snap.SnapshotError as exc:
        raise _err(str(exc)) from None
    _write_json(out, doc)


def _stock_values(val: Sequence[Prediction], mc: Sequence[Prediction], domain: Domain) -> dict:
    computed = metrics.compute(val, domain, bootstrap_resamples=0)
    right = computed["right_proposals"]
    return {
        "right_proposals": _rate(right["n"], right["N"], "operation"),
        "ece": computed["calibration"]["ece"],
        "wrong_mutating": computed["wrong_mutating"]["total"] / max(len(val), 1),
        "wrong_mutating_count": computed["wrong_mutating"]["total"],
        "mc_escalation": _rate(sum(1 for p in mc if _escalated(p)), len(mc), "missing-candidate"),
    }


PROBE_FILE = "probe.json"


def stage_baseline(workdir: Path, knobs: dict[str, Any]) -> None:
    """Measure the stock base model (a greedy stock copy) on the validation side.

    Its selection-fold values are what the pre-registration's stock column
    cites (the record id written here), measured like every candidate.
    """
    from jev_factory.backbones.causal_lm import gen_config

    ctx = _ctx()
    stock = workdir / "stock"
    if gen_config.check(stock) is not None:
        snapshot_dir = base_snapshot(ctx.config)
        if not snapshot_dir.is_dir():
            raise _err(
                f"no base snapshot at {snapshot_dir}",
                "download the base model into hf_cache at base_rev first",
                EXIT_ENV_ERROR,
            )
        gen_config.stock_copy(snapshot_dir, stock, force=True)
    problem = gen_config.check(stock)
    if problem is not None:
        raise _err(f"the stock copy does not decode greedily: {problem}")
    revision = str(ctx.config.require("base_rev"))
    out = workdir / "baseline"
    val_path = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=out / VAL_DIR,
        label="baseline-val",
        model=str(stock),
        revision=revision,
        scorer=in_process_scorer(),
    )
    mc_path = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=out / VAL_MC_DIR,
        label="baseline-val-mc",
        model=str(stock),
        revision=revision,
        scorer=in_process_scorer(),
        missing_candidate=True,
    )
    folds = _read_json(workdir / FOLDS_FILE)
    both = mc_folds(folds, {p.id for p in read_predictions(mc_path)})
    sel = set(both["selection_ids"])
    val, mc = read_predictions(val_path), read_predictions(mc_path)
    values = _stock_values(
        [p for p in val if p.id in sel], [p for p in mc if p.id in sel], ctx.domain
    )
    if knobs.get("probe", True):
        report = probe(
            _selection_split(workdir, folds, out / "val-selection.json"),
            out / PROBE_FILE,
            model=str(stock),
            revision=revision,
            per_entry=10,
            progress_dir=workdir / JOBS_DIR,
        )
        values["permutation_change"] = metrics_rate(report)
    body = {
        "model": str(ctx.config.get("base")),
        "revision": revision,
        "fold": "selection",
        "stock": values,
        "predictions": {
            "val": {"path": _rel(workdir, val_path), "sha256": _sha(val_path)},
            "val_mc": {"path": _rel(workdir, mc_path), "sha256": _sha(mc_path)},
        },
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    _write_json(workdir / BASELINE_SUMMARY, {**body, "record_id": f"baseline-{digest[:12]}"})


def metrics_rate(report: Mapping[str, Any]) -> float:
    rate = report.get("pooled_change_rate")
    if not isinstance(rate, (int, float)):
        raise _err("the permutation probe scored no trial", "check the probe's inputs")
    return float(rate)


def stage_augment(workdir: Path, knobs: dict[str, Any]) -> None:
    """Teacher variations of every train-side entry (resumable; ``per_source: 0`` drafts none)."""
    from jev_factory.data import augment
    from jev_factory.factory.detach import ItemLedger

    ctx = _ctx()
    accepted, rejected = workdir / ACCEPTED_FILE, workdir / REJECTED_FILE
    accepted.parent.mkdir(parents=True, exist_ok=True)
    per_source = int(knobs["per_source"])
    counts: dict[str, Any] = {"per_source": per_source}
    if per_source > 0:
        seeds = augment.load_seeds(workdir / TRAIN_SPLIT, ctx.domain, side="train")
        ledger = ItemLedger(workdir / JOBS_DIR, "augment", len(seeds) * per_source)
        result = augment.run_augment(
            [workdir / TRAIN_SPLIT],
            ctx.domain,
            _teacher_client(),
            accepted,
            rejected,
            per_source,
            side="train",
            workers=int(ctx.config.get("workers") or 1),
            decide_by=str(knobs["decide_by"]),
            ledger=ledger,
        )
        counts.update(result.as_dict())
    for path in (accepted, rejected):
        path.touch()
    _write_json(workdir / AUGMENT_COUNTS, counts)


def stage_targeted(workdir: Path, knobs: dict[str, Any]) -> None:
    """Targeted recipes (a failure class is a data request); no recipes drafts nothing."""
    from jev_factory.data import targeted

    ctx = _ctx()
    recipes = tuple(knobs.get("recipes") or ())
    summary: dict[str, Any] = {"recipes": list(recipes), "supplement": None}
    if recipes:
        result = targeted.run(
            workdir / TRAIN_SPLIT,
            workdir / SUPPLEMENT_FILE,
            recipes,
            int(knobs["per_recipe"]),
            int(ctx.config.get("seed") or 0),
            ctx.domain,
            _teacher_client(),
            exclude=[workdir / VAL_SPLIT, workdir / TEST_SPLIT],
            decide_by=str(knobs["decide_by"]),
            review_out=workdir / "aug" / "targeted-review.jsonl",
        )
        summary.update(supplement=SUPPLEMENT_FILE, result=result)
    _write_json(workdir / TARGETED_SUMMARY, summary)


def stage_assemble(workdir: Path, knobs: dict[str, Any]) -> None:
    """Merge the train side, variations and supplement; leak-check; build and freeze."""
    from jev_factory.data.assemble import AssembleConfig, assemble

    ctx = _ctx()
    summary = _read_json(workdir / TARGETED_SUMMARY)
    supplement = workdir / summary["supplement"] if summary.get("supplement") else None
    cfg = AssembleConfig(
        reasons=bool(knobs["reasons"]),
        randomize_labels=bool(knobs["randomize_labels"]),
        perm_seed=int(knobs["perm_seed"]),
        missing_candidate_rate=float(knobs["missing_candidate_rate"]),
    )
    assemble(
        workdir / DATA_DIR,
        domain=ctx.domain,
        split=workdir / TRAIN_SPLIT,
        variations=[workdir / ACCEPTED_FILE],
        supplement=supplement,
        exclude=[workdir / VAL_SPLIT, workdir / TEST_SPLIT],
        protected=[workdir / VAL_SPLIT, workdir / TEST_SPLIT, workdir / HELDOUT_FILE],
        cfg=cfg,
        deviation_id=ctx.deviation_id,
    )


def _train_one(workdir: Path, name: str, hyper: Mapping[str, Any]) -> dict[str, Any]:
    from jev_factory.backbones.causal_lm import stage_cache
    from jev_factory.backbones.causal_lm import train as trainer
    from jev_factory.data.assemble import select_frozen

    ctx = _ctx()
    cfg = ctx.config
    run_dir = workdir / RUNS_DIR / name
    frozen = select_frozen(workdir / FREEZE_FILE, deviation_id=ctx.deviation_id)
    plan = trainer.plan_train(
        run_dir=run_dir,
        domain=ctx.domain_ref,
        prereg_path=workdir / PREREG_FILE,
        lock_dir=workdir,
        freeze=workdir / FREEZE_FILE,
        base=str(cfg.require("base")),
        revision=cfg.get("base_rev"),
        python=_python(ctx),
        val=workdir / VAL_SPLIT,
        deviation_id=ctx.deviation_id,
        **dict(hyper),
    )
    # A finished run is reused only for the same request: training data, base, revision
    # and every hyperparameter. A changed recipe (e.g. more epochs) retrains from clean.
    request = json.loads(
        json.dumps(
            {
                "data_sha256": plan.data_sha256,
                "base": plan.base,
                "revision": plan.revision,
                "hyperparameters": plan.hyperparameters,
            },
            sort_keys=True,
            default=str,
        )
    )
    request_path = run_dir / TRAIN_REQUEST
    log_path = run_dir / "train-log.json"
    trained = (
        (run_dir / trainer.MERGE_REPORT).is_file()
        and log_path.is_file()
        and _read_json(log_path).get("train_sha256") == frozen.sha256
        and request_path.is_file()
        and _read_json(request_path) == request
    )
    if not trained:
        for stale in (*trainer.RUN_OUTPUTS, TRAIN_REQUEST):
            (run_dir / stale).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "merged", ignore_errors=True)
        trainer.run_plan(
            plan,
            apply=True,
            allow_foreign=ctx.allow_foreign_gpu,
            memory_max=cfg.get("train_memory_max"),
            memory_floor=str(cfg.get("train_memory_floor") or "8G"),
            watchdog_seconds=int(cfg.get("train_watchdog_seconds") or 5),
            gpu_memory_gb=cfg.get("train_gpu_memory_gb"),
            env=ctx.env(),
            runner=ctx.services.train_runner,
        )
        _write_json(request_path, request)
    merged = run_dir / "merged"
    revision = stage_cache.revision_of(merged)
    if cfg.get("hf_cache"):
        from jev_factory.release.hub import effective_prefix

        revision = stage_cache.stage(
            merged,
            f"{effective_prefix(cfg, ctx.domain)}{name}",
            Path(cfg["hf_cache"]),
            base_snapshot(cfg),
        )
    (run_dir / "revision").write_text(revision + "\n", encoding="utf-8")
    cand = workdir / CANDIDATES_DIR / name
    val = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=cand / VAL_DIR,
        label=f"val-{name}",
        model=str(merged),
        revision=revision,
        scorer=in_process_scorer(),
    )
    val_mc = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=cand / VAL_MC_DIR,
        label=f"val-mc-{name}",
        model=str(merged),
        revision=revision,
        scorer=in_process_scorer(),
        missing_candidate=True,
    )
    return {
        "run": _rel(workdir, run_dir),
        "train_sha256": frozen.sha256,
        "merge_report_sha256": _sha(run_dir / trainer.MERGE_REPORT),
        "revision": revision,
        "val": _rel(workdir, val),
        "val_mc": _rel(workdir, val_mc),
    }


def stage_train(workdir: Path, knobs: dict[str, Any]) -> None:
    """Train every pre-registered candidate on the frozen set, one GPU job at a time.

    The pre-registration is checked first; each run is a LoRA scorer training
    then a verified merge (``train.run_plan``), the merged checkpoint is staged
    into the HF cache when ``hf_cache`` is set, and the candidate is measured
    on the validation side (full and missing-candidate slice, pre-gate).
    """
    ctx = _ctx()
    registered, _ = _prereg(workdir, ctx)
    per = knobs.get("candidates") or {}
    stray = sorted(set(per) - set(registered.candidates))
    if stray:
        raise _err(
            f"knobs for candidate(s) not pre-registered: {', '.join(stray)}",
            "an out-of-plan run is a recorded deviation",
        )
    done = {
        name: _train_one(workdir, _candidate_name(name), per.get(name, {}))
        for name in registered.candidates
    }
    _write_json(workdir / TRAIN_SUMMARY, {"candidates": done})


def _train_fields(workdir: Path, name: str) -> dict[str, Any]:
    log_path = workdir / RUNS_DIR / name / "train-log.json"
    if not log_path.is_file():
        return {}
    log = _read_json(log_path)
    hyper = log.get("hyperparameters") or {}
    history = log.get("history") or []
    out: dict[str, Any] = {}
    if isinstance(hyper.get("epochs"), int):
        out["epochs"] = hyper["epochs"]
    if isinstance(hyper.get("batch"), int):
        out["batch_size"] = hyper["batch"]
    if isinstance(log.get("train_examples"), int) and log["train_examples"] > 0:
        out["train_examples"] = log["train_examples"]
    if history and isinstance(history[-1].get("loss"), (int, float)):
        out["train_loss"] = float(history[-1]["loss"])
    return out


SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class _SelectPlan:
    """What every candidate of one select run is calibrated, gated and probed against."""

    ctx: RunContext
    registered: prereg_mod.Prereg
    folds: Any
    grid: Any
    mc_bar: float
    out: Path
    selection_split: Path
    knobs: dict[str, Any]


def _candidate_predictions(workdir: Path, name: str) -> tuple[Path, Path] | None:
    """A candidate's validation and missing-candidate predictions, or None if either is absent."""
    cand = workdir / CANDIDATES_DIR / _candidate_name(name)
    try:
        return one_predictions_file(cand / VAL_DIR), one_predictions_file(cand / VAL_MC_DIR)
    except CliError:
        return None


def _write_select_fit(
    workdir: Path, here: Path, fitted: dict[str, Any], val_path: Path, val_sha: str, mc_bar: float
) -> None:
    _write_json(here / "calibration.json", fitted["calibration"])
    _write_json(
        here / "gate.json",
        gate_document(
            fitted["thresholds"],
            predictions=_rel(workdir, val_path),
            predictions_sha256=val_sha,
            objective="0 wrong mutating, then missing-candidate escalation >= the bar,"
            " then most right proposals",
            mc_bar=mc_bar,
        ),
    )
    _write_json(here / "sweep.json", fitted["sweep"])
    write_predictions(here / "selection.gated.predictions.jsonl", fitted["selection"])


def _probe_candidate(workdir: Path, plan: _SelectPlan, name: str, here: Path) -> dict[str, Any]:
    run_dir = workdir / RUNS_DIR / name
    revision_file = run_dir / "revision"
    revision = revision_file.read_text().strip() if revision_file.is_file() else "unknown"
    return probe(
        plan.selection_split,
        here / PROBE_FILE,
        model=str(run_dir / "merged"),
        revision=revision,
        per_entry=plan.registered.perms_per_entry,
        progress_dir=workdir / JOBS_DIR,
    )


def _failure_classes(
    workdir: Path, selection: Sequence[Prediction], domain: Domain
) -> dict[str, int]:
    """Count the selection-fold failures per entry class."""
    classes: dict[str, int] = {}
    entry_class = {
        e["id"]: e.get("class") or split.expectation_kind(e["expect"])
        for e in _read_json(workdir / VAL_SPLIT)["entries"]
    }
    for p in selection:
        if _failing(p, domain):
            cls = entry_class.get(p.id, metrics.expect_kind(p.expected))
            classes[cls] = classes.get(cls, 0) + 1
    return classes


def _select_candidate(
    workdir: Path, plan: _SelectPlan, name: str, val_path: Path, mc_path: Path
) -> rules.CandidateSummary:
    """Calibrate, fit the gate on the fit fold and probe one candidate; write its summary."""
    fitted = calibrate_and_gate(
        val_path,
        mc_path,
        plan.folds,
        plan.ctx.domain,
        grid=plan.grid,
        mc_bar=plan.mc_bar,
        bootstrap_resamples=int(plan.knobs["bootstrap_resamples"]),
    )
    here = plan.out / name
    val_sha = _sha(val_path)
    _write_select_fit(workdir, here, fitted, val_path, val_sha, plan.mc_bar)
    report = _probe_candidate(workdir, plan, name, here)
    classes = _failure_classes(workdir, fitted["selection"], plan.ctx.domain)
    numbers = fitted["numbers"]
    if numbers["ece"] is None:
        raise _err(f"{name}: no selection-fold line has a distribution; ECE is undefined")
    doc = {
        "name": name,
        **{k: v for k, v in numbers.items() if v is not None},
        "permutation_change": metrics_rate(report),
        "failure_classes": classes,
        "gate_fold": "fit",
        "gate_predictions_sha256": val_sha,
        "predictions_sha256": val_sha,
        **_train_fields(workdir, name),
    }
    summary_path = _write_json(here / SUMMARY_FILE, doc)
    return rules.load_summary(summary_path)


def stage_select(workdir: Path, knobs: dict[str, Any]) -> None:
    """Per candidate: calibrate -> gate fit (fit fold) -> probe; then the pre-registered rule.

    Writes ``select/<candidate>/{calibration,gate,sweep,probe,summary}.json`` and
    ``select/selection.json`` (the rule's trail and verdict, via
    :func:`jev_factory.decide.rules.decide`). The decision *record* is appended
    by :func:`decide` (``jev decide``), never here, so a rerun adds nothing.
    """
    ctx = _ctx()
    registered, prereg_sha = _prereg(workdir, ctx)
    folds = _read_json(workdir / FOLDS_FILE)
    grid = _grid(knobs.get("gate_grid") or {})
    mc_bar = registered.bars["mc_escalation"].bar
    out = workdir / SELECT_DIR
    selection_split = _selection_split(workdir, folds, out / "val-selection.json")
    plan = _SelectPlan(ctx, registered, folds, grid, mc_bar, out, selection_split, knobs)
    summaries: list[rules.CandidateSummary] = []
    missing: list[str] = []
    for name in registered.candidates:
        paths = _candidate_predictions(workdir, name)
        if paths is None:
            missing.append(name)
            continue
        summaries.append(_select_candidate(workdir, plan, name, *paths))
    if missing:
        raise _err(
            f"no validation predictions for pre-registered candidate(s): {', '.join(missing)}",
            "train (and measure) every pre-registered candidate first",
        )
    decision = rules.decide(registered, summaries, prereg_sha256=prereg_sha)
    winner, trail = rules.select(registered, summaries)
    _write_json(
        workdir / SELECTION_FILE,
        {
            "winner": winner.name if winner else None,
            "verdict": decision.verdict,
            "params": decision.params,
            "reasons": decision.reasons,
            "trail": trail,
            "prereg_sha256": prereg_sha,
            "rule_version": rules.RULE_VERSION,
        },
    )


def _quant_summary(lines: Sequence[Prediction], domain: Domain):
    from jev_factory.backbones.causal_lm.quantize import QuantSummary

    computed = metrics.compute(lines, domain, bootstrap_resamples=0)
    right = computed["right_proposals"]
    return QuantSummary(
        right_pct=100.0 * _rate(right["n"], right["N"], "operation"),
        wrong_mutating_ids=frozenset(_wrong_mutating_ids(lines, domain)),
    )


def _quantize_run(workdir: Path, name: str, run_dir: Path, *, awq: bool) -> dict[str, Any]:
    """Quantize *run_dir*'s merged checkpoint and measure the Q4_K_M build on val (+MC)."""
    from jev_factory.backbones.causal_lm import quantize as quant

    ctx = _ctx()
    cfg = ctx.config
    merged = run_dir / "merged"
    if not (merged / "config.json").is_file():
        raise _err(
            f"no trained run at {run_dir} (no merged/config.json)",
            "train (and merge) the candidate before quantizing it",
        )
    work = workdir / "quant" / name
    plan = quant.plan_quantize(
        config=cfg,
        model_dir=merged,
        train=workdir / TRAIN_SPLIT,
        val=workdir / VAL_SPLIT,
        test=workdir / TEST_SPLIT,
        work_dir=work,
        with_awq=awq,
    )
    runner = ctx.services.quant_run or quant.gpu_run(
        work,
        memory_max=str(cfg.require("train_memory_max")),
        memory_floor=str(cfg.get("train_memory_floor") or "8G"),
        watchdog_seconds=int(cfg.get("train_watchdog_seconds") or 5),
        allow_foreign=ctx.allow_foreign_gpu,
        env=ctx.env(),
    )
    quant.run_quantize(plan, apply=True, run=runner, env=ctx.env())
    gguf = work / quant.GGUF_Q4_K_M
    scorer = served_gguf_scorer(gguf, merged)
    val = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=work / VAL_DIR,
        label=f"val-{name}-q4",
        model=f"{name}.q4_k_m",
        revision=f"sha256:{_sha(gguf)[:16]}",
        scorer=scorer,
    )
    val_mc = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=work / VAL_MC_DIR,
        label=f"val-mc-{name}-q4",
        model=f"{name}.q4_k_m",
        revision=f"sha256:{_sha(gguf)[:16]}",
        scorer=scorer,
        missing_candidate=True,
    )
    return {
        "candidate": name,
        "run": _rel(workdir, run_dir),
        "gguf": _rel(workdir, gguf),
        "gguf_sha256": _sha(gguf),
        "val": _rel(workdir, val),
        "val_mc": _rel(workdir, val_mc),
    }


def _heal_check(
    workdir: Path, build: Mapping[str, Any], bf16_val: Path, bf16_mc: Path, heal_rounds: int
) -> dict[str, Any]:
    from jev_factory.backbones.causal_lm import quantize as quant

    domain = _ctx().domain
    bf16 = _quant_summary(read_predictions(bf16_val) + read_predictions(bf16_mc), domain)
    q4 = _quant_summary(
        read_predictions(workdir / build["val"]) + read_predictions(workdir / build["val_mc"]),
        domain,
    )
    trigger = quant.heal_trigger(
        bf16,
        q4,
        log=workdir / "quant" / "heal-trigger-log.jsonl",
        candidate=str(build["candidate"]),
        heal_rounds=heal_rounds,
    )
    return {
        "bf16": {
            "right_pct": bf16.right_pct,
            "wrong_mutating_ids": sorted(bf16.wrong_mutating_ids),
        },
        "quant": {"right_pct": q4.right_pct, "wrong_mutating_ids": sorted(q4.wrong_mutating_ids)},
        "heal_needed": trigger is not None,
        "trigger": None if trigger is None else asdict(trigger),
    }


def stage_quantize(workdir: Path, knobs: dict[str, Any]) -> None:
    """Q4_K_M the chosen candidate, measure it on val, and evaluate the heal trigger."""
    ctx = _ctx()
    selection = _read_json(workdir / SELECTION_FILE)
    winner = selection.get("winner")
    name = knobs.get("candidate") or winner
    # Naming the winner explicitly is no way around the verdict: without a deviation id,
    # only a ship_candidate verdict's winner is quantized.
    if not ctx.deviation_id:
        if selection.get("verdict") != "ship_candidate" or not winner:
            raise _err(
                f"select's verdict is {selection.get('verdict')!r}, not ship_candidate",
                "run `jev decide` and follow its verdict, or record a deviation and pass its id",
            )
        if name != winner:
            raise _err(
                f"quantizing {name!r}, not the rule's winner {winner!r}",
                "a choice against the rule is a recorded deviation: pass its id",
            )
    if not name:
        raise _err("no candidate to quantize", "pass candidate=<name> with the deviation id")
    name = _candidate_name(str(name))
    build = _quantize_run(workdir, name, workdir / RUNS_DIR / name, awq=bool(knobs["awq"]))
    cand = workdir / CANDIDATES_DIR / name
    check = _heal_check(
        workdir,
        build,
        one_predictions_file(cand / VAL_DIR),
        one_predictions_file(cand / VAL_MC_DIR),
        heal_rounds=0,
    )
    _write_json(workdir / QUANT_SUMMARY, {"build": build, **check})


def stage_heal(workdir: Path, knobs: dict[str, Any]) -> None:
    """One heal round when quantize's trigger fired (1 epoch, lr 5e-5, same frozen set).

    Without a trigger the quantized build is the deployed one. After a heal the
    healed run is quantized and checked again; a second trigger is refused and
    logged (heal is one round only): the build is dropped, a human decides.
    """
    from jev_factory.backbones.causal_lm import train as trainer

    ctx = _ctx()
    cfg = ctx.config
    quantized = _read_json(workdir / QUANT_SUMMARY)
    build = quantized["build"]
    if not quantized.get("heal_needed"):
        _write_json(workdir / HEAL_SUMMARY, {"healed": False, "deployed": build})
        return
    name = str(build["candidate"])
    base_run = workdir / build["run"]
    healed_name = f"{name}-heal"
    heal_dir = workdir / RUNS_DIR / healed_name
    plan = trainer.plan_heal(
        run_dir=heal_dir,
        base_run=base_run,
        domain=ctx.domain_ref,
        prereg_path=workdir / PREREG_FILE,
        lock_dir=workdir,
        freeze=workdir / FREEZE_FILE,
        python=_python(ctx),
        val=workdir / VAL_SPLIT,
        deviation_id=ctx.deviation_id,
    )
    trainer.run_plan(
        plan,
        apply=True,
        allow_foreign=ctx.allow_foreign_gpu,
        memory_max=cfg.get("train_memory_max"),
        memory_floor=str(cfg.get("train_memory_floor") or "8G"),
        watchdog_seconds=int(cfg.get("train_watchdog_seconds") or 5),
        gpu_memory_gb=cfg.get("train_gpu_memory_gb"),
        env=ctx.env(),
        runner=ctx.services.train_runner,
    )
    merged = heal_dir / "merged"
    from jev_factory.backbones.causal_lm import stage_cache

    revision = stage_cache.revision_of(merged)
    cand = workdir / CANDIDATES_DIR / healed_name
    bf16_val = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=cand / VAL_DIR,
        label=f"val-{healed_name}",
        model=str(merged),
        revision=revision,
        scorer=in_process_scorer(),
    )
    bf16_mc = measure(
        workdir,
        split_file=workdir / VAL_SPLIT,
        out_dir=cand / VAL_MC_DIR,
        label=f"val-mc-{healed_name}",
        model=str(merged),
        revision=revision,
        scorer=in_process_scorer(),
        missing_candidate=True,
    )
    healed = _quantize_run(workdir, healed_name, heal_dir, awq=False)
    healed["candidate"] = name  # the heal log counts rounds per original candidate
    check = _heal_check(workdir, healed, bf16_val, bf16_mc, heal_rounds=1)
    _write_json(
        workdir / HEAL_SUMMARY,
        {"healed": True, "heal_of": name, "deployed": healed, "check": check},
    )


def stage_recalibrate(workdir: Path, knobs: dict[str, Any]) -> None:
    """Re-fit calibration and the gate on the **deployed** build's own validation predictions."""
    ctx = _ctx()
    registered, _ = _prereg(workdir, ctx)
    deployed = _read_json(workdir / HEAL_SUMMARY)["deployed"]
    val_path, mc_path = workdir / deployed["val"], workdir / deployed["val_mc"]
    folds = _read_json(workdir / FOLDS_FILE)
    fitted = calibrate_and_gate(
        val_path,
        mc_path,
        folds,
        ctx.domain,
        grid=_grid(knobs.get("gate_grid") or {}),
        mc_bar=registered.bars["mc_escalation"].bar,
        bootstrap_resamples=int(knobs["bootstrap_resamples"]),
    )
    val_sha = _sha(val_path)
    _write_json(workdir / DEPLOYED_CALIBRATION, fitted["calibration"])
    _write_json(
        workdir / DEPLOYED_GATE,
        gate_document(
            fitted["thresholds"],
            predictions=_rel(workdir, val_path),
            predictions_sha256=val_sha,
            build_sha256=deployed["gguf_sha256"],
            mc_bar=registered.bars["mc_escalation"].bar,
        ),
    )
    _write_json(
        workdir / DEPLOYED_BUILD,
        {**deployed, "selection": dict(fitted["numbers"])},
    )


def _final_side(
    lines: Sequence[Prediction], thresholds: gate_mod.Thresholds, resamples: int
) -> tuple[list[Prediction], dict]:
    domain = _ctx().domain
    gated = [sweep_gate.redecide(p, thresholds, domain) for p in lines]
    return gated, metrics.compute(gated, domain, bootstrap_resamples=resamples)


def _bars(registered: prereg_mod.Prereg, side: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for name, bar in registered.bars.items():
        value = side.get(name)
        out[name] = {
            "bar": bar.bar,
            "value": value,
            "met": None if value is None else bar.met(value),
            "n": side.get("n"),
        }
    return out


def stage_measure_final(workdir: Path, knobs: dict[str, Any]) -> None:
    """The one final measurement of the deployed build: test, held-out, probe, bars.

    Calibration is applied by the measurement, the fitted gate here. Every bar
    is compared with its n and bootstrap CI against the hashed pre-registration;
    a miss is reported as a miss (a bar with no value is reported as unmeasured).
    Each of the test and held-out sides is measured in full and as its
    missing-candidate slice, each pair once with no deviation id (the once-ledger
    is keyed by side and slice, deviation d9); the final ``mc_escalation`` bar is
    computed from the slice. A second measurement of a pair needs a deviation id
    (the once-ledger refuses otherwise).
    """
    ctx = _ctx()
    registered, prereg_sha = _prereg(workdir, ctx)
    build = _read_json(workdir / DEPLOYED_BUILD)
    gguf = workdir / build["gguf"]
    if not gguf.is_file() or _sha(gguf) != build.get("gguf_sha256"):
        raise _err(
            f"the deployed build {build.get('gguf')} is missing or changed since it was quantized",
            "run quantize (and heal, recalibrate) for this build first",
        )
    merged = workdir / build["run"] / "merged"
    thresholds = gate_mod.Thresholds.from_json(_read_json(workdir / DEPLOYED_GATE))
    scorer = served_gguf_scorer(gguf, merged)
    model = f"{build['candidate']}.q4_k_m"
    revision = f"sha256:{build['gguf_sha256'][:16]}"
    resamples = int(knobs["bootstrap_resamples"])
    sides = [("test", workdir / TEST_SPLIT, {"final": True})]
    if knobs.get("heldout", True) and (workdir / HELDOUT_FILE).is_file():
        sides.append(("held-out", workdir / HELDOUT_FILE, {"acceptance": True}))
    report: dict[str, Any] = {"build": build, "prereg_sha256": prereg_sha, "sides": {}}
    # The sealed sides are measured once. When a run died after measuring them (the probe's
    # server failed, say), the partial report carries their numbers and a retry without a
    # deviation resumes from it instead of asking for a second sealed measurement.
    partial_path = workdir / FINAL_PARTIAL
    partial = _read_json(partial_path) if partial_path.is_file() else None
    if (
        partial is not None
        and not ctx.deviation_id
        and partial.get("build") == build
        and partial.get("prereg_sha256") == prereg_sha
        and set(partial.get("sides", {})) == {side for side, _, _ in sides}
    ):
        report = partial
        sides = []
    for side, path, flag in sides:
        full = measure(
            workdir,
            split_file=path,
            out_dir=workdir / "final" / side,
            label=f"final-{side}",
            model=model,
            revision=revision,
            scorer=scorer,
            calibration=workdir / DEPLOYED_CALIBRATION,
            **flag,
        )
        mc = measure(
            workdir,
            split_file=path,
            out_dir=workdir / "final" / f"{side}-mc",
            label=f"final-{side}-mc",
            model=model,
            revision=revision,
            scorer=scorer,
            calibration=workdir / DEPLOYED_CALIBRATION,
            missing_candidate=True,
            **flag,
        )
        gated, computed = _final_side(read_predictions(full), thresholds, resamples)
        gated_mc = _final_side(read_predictions(mc), thresholds, 0)[0]
        write_predictions(workdir / "final" / f"{side}.gated.predictions.jsonl", gated)
        right = computed["right_proposals"]
        values = {
            "n": len(gated),
            "right_proposals": (right["n"] / right["N"]) if right["N"] else None,
            "right_proposals_ci": right["ci"],
            "wrong_mutating": computed["wrong_mutating"]["total"] / max(len(gated), 1),
            "wrong_mutating_count": computed["wrong_mutating"]["total"],
            "ece": computed["calibration"]["ece"],
            "ece_ci": computed["calibration"]["ece_ci"],
            "mc_escalation": (
                sum(1 for p in gated_mc if _escalated(p)) / len(gated_mc) if gated_mc else None
            ),
            "mc_n": len(gated_mc),
        }
        report["sides"][side] = {**values, "bars": _bars(registered, values)}
    if sides:
        _write_json(partial_path, report)
    if knobs.get("probe", True):
        with ctx.services.serve(ctx, gguf, model, workdir / "final" / "serve") as base_url:
            probe_report = probe(
                workdir / TEST_SPLIT,
                workdir / "final" / PROBE_FILE,
                model=model,
                revision=revision,
                per_entry=registered.perms_per_entry,
                scorer=["--scorer", "served", "--base-url", base_url, "--tokenizer", str(merged)],
                final=True,
                progress_dir=workdir / JOBS_DIR,
            )
        change = metrics_rate(probe_report)
        test = report["sides"]["test"]
        test["permutation_change"] = change
        test["bars"] = _bars(registered, test)
    _write_json(workdir / FINAL_REPORT, report)
    partial_path.unlink(missing_ok=True)


EDGE_REQUIRED = ("device", "build_sha256", "decisions", "latency_ms", "operator_approval")


def stage_edge_check(workdir: Path, knobs: dict[str, Any]) -> None:
    """Record the operator's edge-device run of the same Q4_K_M build; never connects anywhere.

    The operator runs the deployed build on the edge device (an AGX Orin),
    with their approval recorded for stopping any resident model server, and
    writes ``edge/operator-results.json``: ``device`` (a description, not an
    address), ``build_sha256``, ``decisions`` (counts per outcome, or the
    metrics), ``latency_ms`` and ``operator_approval``. This stage checks the
    record names the deployed build and stores it; no host is ever guessed.
    """
    results = _read_json(workdir / EDGE_RESULTS)
    if not isinstance(results, dict):
        raise _err(f"{EDGE_RESULTS} must be a JSON object")
    missing = [k for k in EDGE_REQUIRED if k not in results]
    if missing:
        raise _err(f"{EDGE_RESULTS} is missing: {', '.join(missing)}")
    build = _read_json(workdir / DEPLOYED_BUILD)
    if results["build_sha256"] != build.get("gguf_sha256"):
        raise _err(
            "the edge results name another build than the deployed Q4_K_M",
            "run the deployed build on the edge device and record its sha256",
        )
    if results["operator_approval"] is not True:
        raise _err(
            "the edge run has no recorded operator approval",
            "the operator approves stopping any resident server on the device",
        )
    _write_json(
        workdir / EDGE_RECORD,
        {"results": results, "results_sha256": _sha(workdir / EDGE_RESULTS), "build": build},
    )


def _drafted_supplement(workdir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """The targeted supplement's entries keyed by id with the teachers that drafted them
    (``teachers``/``decided_by``), and its rejected review rows."""
    summary = workdir / TARGETED_SUMMARY
    supplement_rel = _read_json(summary).get("supplement") if summary.is_file() else None
    if not supplement_rel:
        return {}, []
    supplement = _read_json(workdir / supplement_rel)
    meta = supplement.get("targeted", {})
    rows = {
        str(entry["id"]): {
            "teachers": meta.get("teachers", {}),
            "decided_by": meta.get("decide_by"),
        }
        for entry in supplement.get("entries", [])
    }
    review = workdir / "aug" / "targeted-review.jsonl"
    rejected = []
    if review.is_file():
        from jev_factory.release import dataset_bundle as ds

        rejected = [row for row in ds.load_jsonl(review) if not row.get("accepted")]
    return rows, rejected


def _bundle_teachers(workdir: Path, scorer_train: Path, *, apache_only: bool) -> Any:
    """Which teachers wrote the synthetic rows that the frozen training set actually holds.

    Augment variations come from ``aug/accepted.jsonl``; targeted entries from the
    supplement's ``targeted.teachers``. ``None`` when the training set holds neither, so
    the card's "no synthetic variations" sentence is then true."""
    from jev_factory.release import dataset_bundle as ds

    rows: dict[str, dict[str, Any]] = {}
    accepted = workdir / ACCEPTED_FILE
    if accepted.is_file():
        rows.update({row["id"]: row for row in ds.load_jsonl(accepted)})
    rows.update(_drafted_supplement(workdir)[0])
    trained = [{"id": str(e["id"])} for e in _read_json(scorer_train).get("entries", [])]
    used = [entry for entry in trained if entry["id"] in rows]
    if not used:
        return None
    try:
        return ds.teacher_summary(
            used, rows, {}, apache_only=apache_only, is_variation=lambda _id: True
        )
    except ValueError as exc:
        raise _err(str(exc), "every synthetic training row must name its teachers") from None


def stage_bundle(workdir: Path, knobs: dict[str, Any]) -> None:
    """The model bundle of the deployed build: calibration.json, gate.json, scorer-train.json.

    The repository suffix is the operator's choice (``repo_suffix``); the prefix
    comes from the run config or the Domain. The finished folder is scanned.
    """
    from jev_factory.data.assemble import select_frozen
    from jev_factory.release import bundle as rb
    from jev_factory.release import scan
    from jev_factory.release.hub import effective_prefix

    ctx = _ctx()
    suffix = knobs.get("repo_suffix")
    if not suffix:
        raise _err("bundle needs repo_suffix", "the operator names the repository suffix")
    repo = f"{effective_prefix(ctx.config, ctx.domain)}{suffix}"
    build = _read_json(workdir / DEPLOYED_BUILD)
    freeze = _read_json(workdir / FREEZE_FILE)
    frozen = select_frozen(workdir / FREEZE_FILE, deviation_id=ctx.deviation_id)
    summary = knobs.get("data_summary") or (
        f"{freeze.get('counts', {}).get('entries', 'n/a')} scorer training rows frozen at"
        f" sha256 {frozen.sha256[:16]}; `scorer-train.json` in this repository is that set."
    )
    teachers = _bundle_teachers(
        workdir, frozen.path, apache_only=rb.licence_of(ctx.domain, ctx.config) == rb.APACHE_LICENCE
    )
    reports = sorted((workdir / "final").glob("*/final-*.md"))
    edge = workdir / "edge" / "report.md"
    if edge.is_file():
        reports.append(edge)
    out = workdir / "bundle" / str(suffix)
    merged = workdir / build["run"] / "merged"
    try:
        revision = rb.build_model_bundle(
            domain=ctx.domain,
            config=ctx.config,
            base_snapshot=base_snapshot(ctx.config),
            repo=repo,
            run=str(build["candidate"]),
            results=reports,
            data_summary=summary,
            out=out,
            files=rb.BundleFiles(
                calibration=workdir / DEPLOYED_CALIBRATION,
                gate=workdir / DEPLOYED_GATE,
                scorer_train=frozen.path,
            ),
            payload=rb.ModelPayload(merged=merged, kind="gguf", gguf=workdir / build["gguf"]),
            teachers=teachers,
        )
    except rb.BundleError as exc:
        raise _err(str(exc)) from None
    scanned = scan.write_scan(out)
    _write_json(
        workdir / BUNDLE_RECORD,
        {"repo": repo, "folder": _rel(workdir, out), "revision": revision, "scan": scanned},
    )


def _train_set_used(workdir: Path, frozen_path: Path) -> Path:
    """The train-side entries the frozen scorer set was built from, as one split file.

    The merge is re-derived from the files the freeze recorded (split, variations,
    supplement, excluded sides) and kept to the ids the frozen rows hold, so the dataset
    publishes exactly the entries the model trained on (derived ``-nocand`` rows are
    the same entries with their candidate removed)."""
    from jev_factory.data.assemble import merged_entries
    from jev_factory.release.dataset_bundle import load_jsonl

    freeze = _read_json(workdir / FREEZE_FILE)
    files = freeze.get("files", {})

    def frozen_file(key: str) -> Path:
        path = Path(files[key]["path"])
        path = path if path.is_absolute() else workdir / path
        if _sha(path) != files[key]["sha256"]:
            raise _err(
                f"{path} changed since the freeze", "re-run assemble before the dataset bundle"
            )
        return path

    variations = [
        row
        for key in sorted(k for k in files if k.startswith("variations_"))
        for row in load_jsonl(frozen_file(key))
    ]
    supplement = _read_json(frozen_file("supplement")) if "supplement" in files else None
    merged, _ = merged_entries(
        _read_json(frozen_file("split")),
        variations,
        supplement,
        [_read_json(workdir / VAL_SPLIT), _read_json(workdir / TEST_SPLIT)],
    )
    trained = {str(e["id"]) for e in _read_json(frozen_path).get("entries", [])}
    entries = [e for e in merged["entries"] if str(e["id"]) in trained]
    out = workdir / DATASET_TRAIN
    _write_json(out, {"header": merged.get("header", ""), "entries": entries})
    return out


def stage_dataset_bundle(workdir: Path, knobs: dict[str, Any]) -> None:
    """The dataset bundle of the run: the validation and test sides, the train set actually
    used (with each synthetic row's teachers), rejected counts, ``scorer-train.json``, the
    deployed ``calibration.json`` and ``gate.json`` and the base model's LICENSE (deviation
    d14). The sealed held-out set is never published. The finished folder is scanned."""
    from jev_factory.data.assemble import select_frozen
    from jev_factory.release import bundle as rb
    from jev_factory.release import scan
    from jev_factory.release.dataset_bundle import DatasetSources
    from jev_factory.release.hub import effective_prefix

    ctx = _ctx()
    suffix = knobs.get("repo_suffix")
    if not suffix:
        raise _err("dataset-bundle needs repo_suffix", "the operator names the dataset repository")
    repo = f"{effective_prefix(ctx.config, ctx.domain)}{suffix}"
    model = _read_json(workdir / BUNDLE_RECORD)
    frozen = select_frozen(workdir / FREEZE_FILE, deviation_id=ctx.deviation_id)
    train = _train_set_used(workdir, frozen.path)
    drafted, rejected_review = _drafted_supplement(workdir)
    rejected = [workdir / REJECTED_FILE] if (workdir / REJECTED_FILE).is_file() else []
    if rejected_review:
        review = workdir / "dataset" / "targeted-rejected.jsonl"
        review.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected_review),
            encoding="utf-8",
        )
        rejected.append(review)
    out = workdir / "dataset" / str(suffix)
    if out.exists():
        shutil.rmtree(out)
    try:
        counts = rb.build_dataset_bundle(
            domain=ctx.domain,
            config=ctx.config,
            sources=DatasetSources(
                splits=workdir / "splits",
                train_augmented=train,
                accepted=workdir / ACCEPTED_FILE,
                rejected=rejected,
            ),
            licence_file=base_snapshot(ctx.config) / "LICENSE",
            role_models={},
            out=out,
            files=rb.BundleFiles(
                calibration=workdir / DEPLOYED_CALIBRATION,
                gate=workdir / DEPLOYED_GATE,
                scorer_train=frozen.path,
            ),
            repo=repo,
            run=str(_read_json(workdir / DEPLOYED_BUILD)["candidate"]),
            model_repos=[model["repo"]],
            supplement_teachers=drafted,
        )
    except (rb.BundleError, ValueError) as exc:
        raise _err(str(exc)) from None
    scanned = scan.write_scan(out)
    _write_json(
        workdir / DATASET_RECORD,
        {"repo": repo, "folder": _rel(workdir, out), "counts": counts, "scan": scanned},
    )


def stage_upload(workdir: Path, knobs: dict[str, Any]) -> None:
    """Upload the bundle **privately**, fetch it back and compare every sha256.

    Refuses unless the operator named the repository (``repo_suffix``, which
    must be the bundled one). Nothing here ever makes a repository public.
    """
    from jev_factory.release import hub

    ctx = _ctx()
    record = _read_json(workdir / BUNDLE_RECORD)
    suffix = knobs.get("repo_suffix")
    if not suffix:
        raise _err("upload needs repo_suffix", "the operator names the repository to upload to")
    repo = f"{hub.effective_prefix(ctx.config, ctx.domain)}{suffix}"
    if repo != record["repo"]:
        raise _err(f"the bundle was built for {record['repo']}, not {repo}")
    dataset = _read_json(workdir / DATASET_RECORD)
    data_suffix = knobs.get("dataset_repo_suffix")
    if not data_suffix:
        raise _err(
            "upload needs dataset_repo_suffix",
            "the operator names the dataset repository to upload to",
        )
    data_repo = f"{hub.effective_prefix(ctx.config, ctx.domain)}{data_suffix}"
    if data_repo != dataset["repo"]:
        raise _err(f"the dataset bundle was built for {dataset['repo']}, not {data_repo}")
    results = {}
    for name, folder, target, repo_type in (
        ("model", record["folder"], repo, str(knobs["repo_type"])),
        ("dataset", dataset["folder"], data_repo, "dataset"),
    ):
        try:
            results[name] = hub.upload(
                bundle=workdir / folder,
                repo=target,
                repo_type=repo_type,
                config=ctx.config,
                domain=ctx.domain,
                apply=True,
                hub=ctx.services.hub,
                environ=ctx.services.environ,
            )
        except hub.UploadError as exc:
            raise _err(str(exc)) from None
    _write_json(workdir / UPLOAD_RECORD, {**results["model"], "dataset": results["dataset"]})


def stage_release_gate(workdir: Path, knobs: dict[str, Any]) -> None:
    """Run the release gate (``python -m jev_factory.evals run``) over an operator manifest."""
    ctx = _ctx()
    manifest = knobs.get("manifest")
    if not manifest:
        raise _err("release-gate needs a manifest", "set the release-gate knob manifest")
    if not Path(manifest).is_file():
        raise _err(f"no release-gate manifest at {manifest}", "point the knob at the manifest")
    run_dir = workdir / "release-gate" / "run"
    argv = ["run", "--manifest", str(manifest), "--run-dir", str(run_dir), "--json"]
    if knobs.get("no_deepeval"):
        argv.append("--no-deepeval")
    env = ctx.env()
    rc = ctx.services.run_module(EVALS_MODULE, argv, python=sys.executable, env=env)
    _write_json(
        workdir / RELEASE_GATE_RESULT,
        {"exit_code": rc, "run_dir": _rel(workdir, run_dir), "manifest_sha256": _sha(manifest)},
    )
    if rc != 0:
        raise _err(
            f"the release gate exited {rc}",
            "`python -m jev_factory.evals status --run-dir` shows why; `continue` resumes",
            EXIT_ENV_ERROR,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _stage(
    name: str,
    func: Callable[[Path, dict[str, Any]], None],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    deps: tuple[str, ...],
) -> Stage:
    doc = (func.__doc__ or "").strip().splitlines()[0]
    steps = " -> ".join(SUBSTEPS[name])
    return Stage(name, func, inputs, outputs, deps, f"{doc} [{steps}]")


def _stages() -> list[Stage]:
    s = _stage
    return [
        s("config", stage_config, (), (CONFIG_FILE,), ()),
        s("preregister", stage_preregister, (PREREG_FILE,), (PREREG_LOCK,), ("config",)),
        s("seed", stage_seed, (), (SEED_FILE,), ("config",)),
        s("teachers-pilot", stage_teachers_pilot, (SEED_FILE,), (PILOT_YIELDS,), ("seed",)),
        s("draft-heldout", stage_draft_heldout, (SEED_FILE,), (HELDOUT_FILE,), ("seed",)),
        s(
            "draft-eval",
            stage_draft_eval,
            (SEED_FILE,),
            (POOL_FILE,),
            ("seed", "teachers-pilot"),
        ),
        s(
            "split",
            stage_split,
            (POOL_FILE, SEED_FILE),
            (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, FOLDS_FILE, MC_FOLDS_FILE),
            ("seed", "draft-eval"),
        ),
        s(
            "snapshot",
            stage_snapshot,
            (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED_FILE),
            (SNAPSHOT_FILE,),
            ("split",),
        ),
        s(
            "baseline",
            stage_baseline,
            (VAL_SPLIT, FOLDS_FILE, SNAPSHOT_FILE),
            (BASELINE_SUMMARY,),
            ("snapshot",),
        ),
        s(
            "augment",
            stage_augment,
            (TRAIN_SPLIT,),
            (ACCEPTED_FILE, AUGMENT_COUNTS),
            ("split",),
        ),
        s(
            "targeted",
            stage_targeted,
            (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT),
            (TARGETED_SUMMARY,),
            ("split",),
        ),
        s(
            "assemble",
            stage_assemble,
            (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, HELDOUT_FILE, ACCEPTED_FILE, TARGETED_SUMMARY),
            (FREEZE_FILE,),
            ("split", "draft-heldout", "augment", "targeted"),
        ),
        s(
            "train",
            stage_train,
            (PREREG_LOCK, FREEZE_FILE, VAL_SPLIT, SNAPSHOT_FILE),
            (TRAIN_SUMMARY,),
            ("preregister", "assemble", "snapshot"),
        ),
        s(
            "select",
            stage_select,
            (PREREG_LOCK, VAL_SPLIT, FOLDS_FILE, CANDIDATES_DIR),
            (SELECTION_FILE,),
            ("train",),
        ),
        s(
            "quantize",
            stage_quantize,
            (SELECTION_FILE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT),
            (QUANT_SUMMARY,),
            ("select",),
        ),
        s("heal", stage_heal, (QUANT_SUMMARY,), (HEAL_SUMMARY,), ("quantize",)),
        s(
            "recalibrate",
            stage_recalibrate,
            (HEAL_SUMMARY, FOLDS_FILE, PREREG_LOCK),
            (DEPLOYED_CALIBRATION, DEPLOYED_GATE, DEPLOYED_BUILD),
            ("heal",),
        ),
        s(
            "measure-final",
            stage_measure_final,
            (DEPLOYED_BUILD, DEPLOYED_CALIBRATION, DEPLOYED_GATE, TEST_SPLIT),
            (FINAL_REPORT,),
            ("recalibrate",),
        ),
        s(
            "edge-check",
            stage_edge_check,
            (EDGE_RESULTS, DEPLOYED_BUILD),
            (EDGE_RECORD,),
            ("measure-final",),
        ),
        s(
            "bundle",
            stage_bundle,
            (DEPLOYED_CALIBRATION, DEPLOYED_GATE, DEPLOYED_BUILD, FREEZE_FILE, FINAL_REPORT),
            (BUNDLE_RECORD,),
            ("measure-final", "edge-check"),
        ),
        s(
            "dataset-bundle",
            stage_dataset_bundle,
            (
                BUNDLE_RECORD,
                FREEZE_FILE,
                VAL_SPLIT,
                TEST_SPLIT,
                DEPLOYED_CALIBRATION,
                DEPLOYED_GATE,
            ),
            (DATASET_RECORD,),
            ("bundle",),
        ),
        s(
            "upload",
            stage_upload,
            (BUNDLE_RECORD, DATASET_RECORD),
            (UPLOAD_RECORD,),
            ("bundle", "dataset-bundle"),
        ),
        s(
            "release-gate",
            stage_release_gate,
            (BUNDLE_RECORD,),
            (RELEASE_GATE_RESULT,),
            ("upload",),
        ),
    ]


def build_registry(registry: Registry | None = None) -> Registry:
    """Register every pipeline stage, in :data:`STAGE_ORDER`, into *registry* (a new one)."""
    reg = registry if registry is not None else Registry()
    for stage in _stages():
        if stage.name not in reg.stages:
            reg.register(stage)
    return reg


#: The pipeline registers into the engine's default registry (what ``jev run`` lists).
REGISTRY = build_registry(DEFAULT_REGISTRY)


def stage_names() -> list[str]:
    return REGISTRY.names()


def get_stage(name: str) -> Stage:
    """The named stage, or a CliError that lists every stage (nothing runs)."""
    try:
        return REGISTRY.get(name)
    except KeyError:
        raise _err(
            f"unknown stage {name!r}",
            "stages, in order: " + ", ".join(REGISTRY.names()),
        ) from None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def effective_knobs(name: str, knobs: Mapping[str, Any] | None, ctx: RunContext) -> dict:
    """Defaults, then *knobs*; plus the domain surface and the config keys the stage reads."""
    get_stage(name)
    given = dict(knobs or {})
    unknown = sorted(set(given) - set(DEFAULT_KNOBS[name]))
    if unknown:
        known = ", ".join(sorted(DEFAULT_KNOBS[name])) or "none"
        raise _err(f"unknown knob(s) for {name}: {', '.join(unknown)}", f"known knobs: {known}")
    merged = json.loads(json.dumps({**DEFAULT_KNOBS[name], **given}))
    merged["_domain"] = {"name": ctx.domain.name, "surface_sha256": ctx.domain.surface_sha256()}
    merged["_config"] = {k: ctx.config.get(k) for k in STAGE_CONFIG_KEYS.get(name, ())}
    return merged


def plan(
    workdir: Path, name: str, ctx: RunContext, knobs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The dry run: what *name* reads and writes and whether it would run. Writes nothing."""
    stage = get_stage(name)
    workdir = Path(workdir)
    effective = effective_knobs(name, knobs, ctx)
    inputs = {rel: sha256_path(workdir / rel) for rel in stage.inputs}
    # Other stages are judged by the knobs they last ran with; this one by *knobs*.
    recorded = {
        other: (read_manifest(workdir, other) or {}).get("knobs", {}) for other in REGISTRY.names()
    }
    reason = staleness(workdir, name, {**recorded, name: effective}, registry=REGISTRY)
    manifest = read_manifest(workdir, name)
    return {
        "stage": name,
        "applied": False,
        "summary": stage.summary,
        "substeps": list(SUBSTEPS[name]),
        "deps": list(stage.deps),
        "inputs": inputs,
        "missing_inputs": [rel for rel, digest in inputs.items() if digest is None],
        "outputs": list(stage.outputs),
        "knobs": effective,
        "stale": reason,
        "would_run": manifest is None or manifest.get("status") != COMPLETE or reason is not None,
        "note": "dry run: nothing was written; pass --apply to run the stage",
    }


def _guarded(func: Callable[[Path, dict[str, Any]], None]) -> Callable[[Path, dict], None]:
    def call(workdir: Path, knobs: dict[str, Any]) -> None:
        try:
            func(workdir, knobs)
        except CliError as exc:
            hint = f" (hint: {exc.remediation})" if exc.remediation else ""
            raise CliError(exc.code, exc.message + hint, exc.remediation) from None

    return call


def run(
    workdir: Path,
    name: str,
    ctx: RunContext,
    knobs: Mapping[str, Any] | None = None,
    *,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run stage *name* in *workdir*: a dry run (:func:`plan`) unless *apply*.

    With *apply* the work dir must be outside any git worktree, the run lock
    is held for the whole stage, and the stage engine writes its manifest (a
    fresh stage is skipped unless *force*). Returns the manifest.
    """
    if not apply:
        return plan(workdir, name, ctx, knobs)
    stage = get_stage(name)
    workdir = resolve_work_root(workdir)
    effective = effective_knobs(name, knobs, ctx)
    guarded = replace(stage, func=_guarded(stage.func))
    one = Registry()
    for dep_name in REGISTRY.names():
        one.register(guarded if dep_name == name else REGISTRY.get(dep_name))
    token = _CURRENT.set(ctx)
    try:
        with RunLock(workdir):
            return run_stage(workdir, name, effective, registry=one, force=force)
    finally:
        _CURRENT.reset(token)


# ---------------------------------------------------------------------------
# jev decide: the rule, recorded
# ---------------------------------------------------------------------------


def _yields(workdir: Path) -> list[rules.ClassYield]:
    path = workdir / PILOT_YIELDS
    if not path.is_file():
        return []
    digest = _sha(path)
    classes = _read_json(path).get("classes") or {}
    return [
        rules.ClassYield(name, int(t["accepted"]), int(t["reviewed"]), str(path), digest)
        for name, t in sorted(classes.items())
        if int(t.get("reviewed", 0)) > 0
    ]


def _quant_check(workdir: Path) -> rules.QuantCheck | None:
    heal_path, quant_path = workdir / HEAL_SUMMARY, workdir / QUANT_SUMMARY
    if heal_path.is_file() and _read_json(heal_path).get("healed"):
        doc, path, rounds = _read_json(heal_path)["check"], heal_path, 1
        candidate = _read_json(heal_path)["heal_of"]
    elif quant_path.is_file():
        doc, path, rounds = _read_json(quant_path), quant_path, 0
        candidate = doc["build"]["candidate"]
    else:
        return None
    return rules.QuantCheck(
        candidate=str(candidate),
        bf16_right=doc["bf16"]["right_pct"] / 100.0,
        quant_right=doc["quant"]["right_pct"] / 100.0,
        bf16_wrong_mutating_ids=frozenset(doc["bf16"]["wrong_mutating_ids"]),
        quant_wrong_mutating_ids=frozenset(doc["quant"]["wrong_mutating_ids"]),
        heal_rounds=rounds,
        source=str(path),
        sha256=_sha(path),
    )


def decide(
    workdir: Path,
    ctx: RunContext,
    *,
    reference: Path | None = None,
    hard_stops: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply the pre-registered rule to the run's evidence and append one decision record.

    Evidence: every candidate summary ``select`` wrote, the teacher pilot's
    yields, and the quantize/heal check when present. The record (via
    :func:`jev_factory.decide.records.append`) cites every metric with the
    sha256 of the file it came from. Returns the record.
    """
    workdir = Path(workdir)
    registered, prereg_sha = _prereg(workdir, ctx)
    summaries = [
        rules.load_summary(workdir / SELECT_DIR / name / SUMMARY_FILE)
        for name in registered.candidates
        if (workdir / SELECT_DIR / name / SUMMARY_FILE).is_file()
    ]
    if not summaries:
        raise _err("no candidate summaries in this run", "run the select stage first")
    decision = rules.decide(
        registered,
        summaries,
        prereg_sha256=prereg_sha,
        reference=rules.load_summary(reference) if reference else None,
        yields=_yields(workdir),
        quant=_quant_check(workdir),
        hard_stops=hard_stops,
    )
    with RunLock(workdir):
        return records.append(
            workdir / DECISIONS_FILE, run=workdir.name, **decision.record_fields()
        )


__all__ = [
    "DEFAULT_KNOBS",
    "REGISTRY",
    "RunContext",
    "SUBSTEPS",
    "STAGE_ORDER",
    "Services",
    "build_registry",
    "decide",
    "get_stage",
    "plan",
    "run",
    "stage_names",
]
