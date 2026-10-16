"""The distillery's stage registry: ``jev distill <stage>`` over the per-stage modules.

Every distill stage is a registered :class:`~jev_factory.factory.stages.Stage` in its **own**
:class:`~jev_factory.factory.stages.Registry` (:data:`REGISTRY`), separate from the jev-tool
build's (:data:`jev_factory.factory.pipeline.REGISTRY`, what ``jev run`` lists), so the two
surfaces never mix and ``jev run`` is unchanged. This module only wires: each stage function
calls the module that owns the work and changes none of its behaviour.

Layout of a run's work dir (``<work>``)::

    <work>/distill/                  every distill module's own outputs (``distill/...``)
    <work>/distill/manifests/<s>.json  the per-stage manifest of stage ``s`` (the stage engine)
    <work>/jobs/distill/<s>.{pid,log,done}  a detached ``jev distill <s> --apply --detach``
                                     (and the localhost servers a stage starts)

so ``jev status <work>`` reads the distill manifests beside the build's and lists the detached
jobs under ``jobs/`` as it always has.

Stages, in data-dependency order (:data:`STAGE_ORDER`)::

    preregister, dataset, contamination, teacher-label, audit, corrections-draft, assemble,
    benchmark, train, quantize, gate, final, edge, bundle, upload

Gates wired here (t17): teacher-label holds the teacher-label lock for its whole run (released in
a ``finally``), so final and edge refuse while it labels; benchmark reproduces the stock model's
JevBench w4096 result and measures it on the probe set and the fit/selection folds, and train
refuses (fail closed) without that reproduction record unless a deviation id is given; final
needs the contamination stage's report and reads the gold ledger, the probe set and the bf16
calibration; edge measures the shipped quant on a board through a 127.0.0.1 tunnel against the
quant stage's selection-fold predictions of the same bundle.

Running a stage goes through :func:`run`: without ``apply`` nothing is written (:func:`plan`
reports what the stage reads and writes, whether it is stale, the run-config keys it needs that
are unset and, for stages whose module has its own dry run, that plan); with ``apply`` the work
dir must be outside any git worktree, the run lock (shared with ``jev run``) is held, and the
stage engine runs the stage and writes its manifest. Operator review of drafted corrections stays
in ``jev review --corrections``; this registry never approves anything.

Side-effecting seams (teacher and drafter clients, the token counter, scorers, measurers,
builders, the final's models, the hub) are fields of :class:`Services`; ``None`` means the real
default built from the run config. Stdlib only at import time.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import assemble as dassemble
from jev_factory.distill import audit as daudit
from jev_factory.distill import final as dfinal
from jev_factory.distill import gate as dgate
from jev_factory.distill import prereg as dprereg
from jev_factory.distill import quant as dquant
from jev_factory.distill import train as dtrain
from jev_factory.distill.runspec import GpuLimits, TrainerEnv
from jev_factory.factory.config import RunConfig
from jev_factory.factory.lock import RunLock
from jev_factory.factory.stages import (
    COMPLETE,
    Registry,
    Stage,
    read_manifest,
    run_stage,
    sha256_path,
    staleness,
)
from jev_factory.factory.workroot import resolve_work_root

#: Every distill module writes under ``<work>/distill``; the stage engine runs there too.
DISTILL_DIR = "distill"
#: Detached distill jobs and the servers a stage starts: ``<work>/jobs/distill``.
JOBS_REL = "jobs/distill"

# Paths relative to the stage engine's work dir (``<work>/distill``).
PREREG_FILE = "prereg.json"
PREREG_LOCK = "prereg.lock.json"
DATASET_DIR = "dataset"
DATASET_MANIFEST = "dataset/manifest.json"
CONTAMINATION_FILE = "contamination.json"
TEACHER_DIR = "teacher"
TEACHER_LABELS = "teacher/teacher_labels.jsonl"
TEACHER_MANIFEST = "teacher/teacher_manifest.json"
AUDIT_FILE = "teacher_audit.json"
CORRECTIONS_FILE = "corrections/corrections.jsonl"
CORRECTIONS_DRAFT = "corrections/draft.json"
TRAIN_SET = "assemble/train_set.jsonl"
ASSEMBLE_MANIFEST = "assemble/assemble_manifest.json"
BENCHMARK_FILE = "benchmark/benchmark.json"
SELECT_SUMMARY = "select/summary.json"
QUANT_MANIFEST = "quant/quant_manifest.json"
QUANT_DECISION = "quant/format_decision.json"
QUANT_CALIBRATION = "quant/calibration.json"
GATE_FILE = "gate/gate.json"
FINAL_FILE = "final/final.json"
FINAL_REPORT = "final/report.json"
BUNDLE_DIR = "bundle"
BUNDLE_MODEL = "bundle/model"
BUNDLE_DATASET = "bundle/dataset"
UPLOAD_FILE = "upload/upload.json"
STOCK_REPRODUCTION = "benchmark/stock_reproduction.json"
STOCK_MEASUREMENTS = "benchmark/stock_measurements.json"
QUANT_MEASUREMENTS = "quant/measurements.json"
QUANT_PREDICTIONS = "quant/deployed_predictions.json"
SELECT_CALIBRATION = "select/calibration.json"
EDGE_DIR = "edge"

STAGE_ORDER = (
    "preregister",
    "dataset",
    "contamination",
    "teacher-label",
    "audit",
    "corrections-draft",
    "assemble",
    "benchmark",
    "train",
    "quantize",
    "gate",
    "final",
    "edge",
    "bundle",
    "upload",
)

#: Every stage's knobs and their defaults; a knob not listed is refused.
DEFAULT_KNOBS: dict[str, dict[str, Any]] = {
    "preregister": {},
    "dataset": {
        "seed": 27,
        "max_tokens": 4096,
        "held_out_fraction": 0.10,
        "train_fraction": 0.70,
        "fit_fraction": 0.15,
    },
    "contamination": {},
    "teacher-label": {"joint": True},
    "audit": {"floor": None, "role": "clef", "labels": None},
    "corrections-draft": {"author": None, "reason": "weak"},
    "assemble": {"teacher_filter": "all", "alt_labels": None},
    "benchmark": {},
    "train": {},
    "quantize": {
        "formats": None,
        "heal_mix": None,
        "gating": None,
        "heal_repeat": None,
    },
    "gate": {},
    "final": {"clef_calibration": None, "unverified_gold": False},
    "edge": {"label": "nano", "served_model": "distill-edge-student"},
    "bundle": {},
    "upload": {"model_repo": None, "dataset_repo": None},
}

#: Run-config keys each stage reads (recorded with its knobs: a change makes it stale).
STAGE_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "dataset": ("distill_records", "distill_student_python", "distill_pin_cache"),
    "teacher-label": (
        "distill_teacher_url",
        "distill_teacher_model",
        "distill_teacher_repo",
        "distill_teacher_revision",
        "distill_teacher_build_id",
        "distill_teacher_engine",
        "distill_teacher_key_env",
        "distill_teacher_model_header",
    ),
    "corrections-draft": ("distill_alt_teacher_url", "distill_alt_teacher_model"),
    "benchmark": (
        "distill_teacher_url",
        "distill_teacher_model",
        "distill_stock_url",
        "distill_stock_model",
        "distill_jevbench_tasks",
        "distill_probe_set",
        "distill_base_revision",
    ),
    "train": (
        "distill_pin_cache",
        "distill_base_revision",
        "distill_decider_revision",
        "distill_decider_repo",
        "distill_trainer_python",
        "distill_student_python",
        "distill_student_lockfile",
        "distill_student_port",
        "train_memory_max",
        "train_memory_floor",
    ),
    "quantize": (
        "distill_base_revision",
        "distill_decider_repo",
        "distill_trainer_python",
        "distill_student_python",
        "distill_student_lockfile",
        "distill_student_port",
        "train_memory_max",
        "train_memory_floor",
        "distill_probe_set",
    ),
    "gate": (
        "distill_decider_repo",
        "distill_student_python",
        "distill_student_lockfile",
        "distill_student_port",
    ),
    "final": (
        "distill_teacher_url",
        "distill_teacher_model",
        "distill_stock_url",
        "distill_stock_model",
        "distill_decider_repo",
        "distill_student_python",
        "distill_student_lockfile",
        "distill_student_port",
        "distill_agx_url",
        "distill_gold_ledger",
        "distill_probe_set",
    ),
    "edge": ("distill_edge_url", "distill_edge_memory_cap", "distill_probe_set"),
    "upload": ("hub_prefix",),
}

#: Stages with no side effect beyond their own outputs still dry-run; these call a module's
#: own dry run (``apply=False``) in :func:`plan`, which reads and never writes.
_MODULE_PLANS = ("train", "quantize", "gate", "final", "edge", "upload")


# ---------------------------------------------------------------------------
# Context and services
# ---------------------------------------------------------------------------


@dataclass
class Services:
    """Every side-effecting seam a distill stage uses. ``None`` means the real default."""

    #: ``(text) -> tokens`` for the dataset's length filter (default: the student tokenizer).
    token_counter: Callable[[str], int] | None = None
    #: ``ask(record) -> CallResult`` for teacher-label (default: SystemOneClient on the teacher).
    teacher_client: Any = None
    #: ``(record) -> {qid: target}`` for corrections-draft (default: the alternative teacher).
    drafter: Callable[[Any], dict[str, dict[str, Any]]] | None = None
    #: ``{"clef"|"stock": Target}`` for benchmark (default: the configured URLs).
    benchmark_targets: Mapping[str, Any] | None = None
    #: ``{name: client}`` for benchmark (default: SystemOneClient per target).
    benchmark_clients: Mapping[str, Any] | None = None
    #: GPU memory probe for benchmark/final (``None`` here = nvidia-smi).
    memory_probe: Callable[[], float | None] | None = None
    #: train: the candidate scorer (default: ``train.served_scorer``).
    scorer: Any = None
    #: quantize: the format builders (default: ``quant.default_builders``).
    builders: Mapping[str, Any] | None = None
    #: quantize/gate: the measurer (default: ``quant.served_measurer``).
    measurer: Any = None
    #: quantize: the heal trainer (default: the module's own).
    trainer: Any = None
    #: final: ``{name: FinalModel}`` for clef, stock, student_bf16 and student (default: from
    #: config; student is the shipped quant pre-served on the AGX Orin at distill_agx_url).
    final_models: Mapping[str, Any] | None = None
    #: edge: ``ask(record) -> CallResult`` on the board (default: SystemOneClient on the URL).
    edge_client: Any = None
    #: edge: ``get(path) -> JSON`` for ``/v1/bundle`` and ``/v1/stats`` (default: HTTP GET).
    edge_get: Callable[[str], Mapping[str, Any]] | None = None
    #: upload: the hub client module (default: huggingface_hub, imported on upload).
    hub: Any = None
    #: Environment secrets are read from (default: os.environ).
    environ: Mapping[str, str] | None = None
    #: ``() -> YYYY-MM-DD``.
    today: Callable[[], str] | None = None


@dataclass
class DistillContext:
    """What every distill stage runs against: the work root, the run config, the services."""

    work: Path
    config: RunConfig
    deviation_id: str | None = None
    allow_foreign_gpu: bool = False
    services: Services = field(default_factory=Services)

    def today(self) -> str:
        if self.services.today is not None:
            return self.services.today()
        return _dt.date.today().isoformat()


_CURRENT: contextvars.ContextVar[DistillContext] = contextvars.ContextVar("jev_distill_context")


def _ctx() -> DistillContext:
    try:
        return _CURRENT.get()
    except LookupError:
        raise CliError(
            EXIT_USER_ERROR,
            "a distill stage ran without a run context",
            "run stages through jev_factory.distill.pipeline.run",
        ) from None


def _err(message: str, remediation: str = "") -> CliError:
    return CliError(EXIT_USER_ERROR, message, remediation)


def stage_dir(work: Path) -> Path:
    """The stage engine's work dir for a run: ``<work>/distill``."""
    return Path(work) / DISTILL_DIR


def jobs_dir(work: Path) -> Path:
    return Path(work) / JOBS_REL


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _err(f"{path}: unreadable ({type(exc).__name__})", "run its stage first") from None


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------


def _prereg_path(ctx: DistillContext) -> Path:
    return ctx.work / dprereg.PREREG_REL


def _dataset_dir(ctx: DistillContext) -> Path:
    return stage_dir(ctx.work) / DATASET_DIR


def _folds(ctx: DistillContext) -> dict[str, Path]:
    ds = _dataset_dir(ctx)
    return {"fit": ds / "fit.jsonl", "selection": ds / "selection.jsonl"}


def _sealed(ctx: DistillContext) -> list[Any]:
    """The sealed sides as ids and sha256 only, so fold checks can refuse them."""
    from jev_factory.distill.dataset import load_sealed

    ds = _dataset_dir(ctx)
    return [load_sealed(ds / name) for name in ("held_out.jsonl", "test.jsonl")]


def _train_records(ctx: DistillContext) -> list[Any]:
    from jev_factory.distill.dataset import read_records

    return read_records(_dataset_dir(ctx) / "train.jsonl")


def _base_revision(ctx: DistillContext) -> str:
    from jev_factory.distill import pins as dpins

    return str(ctx.config.get("distill_base_revision") or dpins.DEFAULT_BASE_REVISION)


def _student_env(ctx: DistillContext) -> Any:
    from jev_factory.distill.envs import EnvSpec

    return EnvSpec(
        role="student",
        lockfile_hash=str(ctx.config.require("distill_student_lockfile")),
        python=str(ctx.config.require("distill_student_python")),
    )


def _measurer(ctx: DistillContext) -> Any:
    if ctx.services.measurer is not None:
        return ctx.services.measurer
    return dquant.served_measurer(
        env=_student_env(ctx),
        jobdir=jobs_dir(ctx.work),
        port=int(ctx.config.get("distill_student_port")),
        decider_repo=ctx.config.get("distill_decider_repo"),
        allow_foreign=ctx.allow_foreign_gpu,
    )


def _memory_probe(ctx: DistillContext) -> Any:
    if ctx.services.memory_probe is not None:
        return ctx.services.memory_probe
    from jev_factory.distill import benchmark as bench

    return bench.nvidia_smi_used_mib


# ---------------------------------------------------------------------------
# Stages (each: (stage_workdir=<work>/distill, knobs) -> None)
# ---------------------------------------------------------------------------


def stage_preregister(workdir: Path, knobs: dict[str, Any]) -> None:
    """Validate the distill pre-registration and hash it into its lock."""
    ctx = _ctx()
    dprereg.lock(_prereg_path(ctx), ctx.work)


def stage_dataset(workdir: Path, knobs: dict[str, Any]) -> None:
    """Build the run dataset: licence allowlist, length filter, grouped and sealed sides."""
    from jev_factory.distill import dataset, pins
    from jev_factory.distill.records import Record

    ctx = _ctx()
    source = Path(ctx.config.require("distill_records"))
    lines = source.read_text(encoding="utf-8").splitlines()
    records = [Record.from_json(json.loads(line)) for line in lines if line.strip()]
    counter = ctx.services.token_counter
    if counter is None:
        cache = Path(ctx.config.require("distill_pin_cache"))
        counter = dataset.subprocess_token_counter(
            str(ctx.config.require("distill_student_python")),
            cache / pins.cache_dirname(pins.BASE_REPO),
        )
    built = dataset.build_dataset(
        records,
        token_counter=counter,
        seed=int(knobs["seed"]),
        max_tokens=int(knobs["max_tokens"]),
        held_out_fraction=float(knobs["held_out_fraction"]),
        train_fraction=float(knobs["train_fraction"]),
        fit_fraction=float(knobs["fit_fraction"]),
    )
    dataset.write_dataset(built, Path(workdir) / DATASET_DIR)


def stage_contamination(workdir: Path, knobs: dict[str, Any]) -> None:
    """Mark every dataset source clean, v21-evaluated or overlapping against upstream data."""
    from jev_factory.distill import contamination
    from jev_factory.distill.dataset import read_records

    ctx = _ctx()
    ds = _dataset_dir(ctx)
    names = list(_read_json(ds / "manifest.json").get("licences") or {})
    # hf ids come from the training-side records; the sealed sides are not read here
    recs = [r for side in ("train", "fit", "selection") for r in read_records(ds / f"{side}.jsonl")]
    report = contamination.check_sources(names, records=recs)
    contamination.write_report(report, workdir)


def _teacher_config(ctx: DistillContext, knobs: Mapping[str, Any]) -> Any:
    """The teacher's provenance and how to reach it (the key by env-var NAME only)."""
    from jev_factory.distill import teacher

    req, get = ctx.config.require, ctx.config.get
    return teacher.TeacherConfig(
        base_url=str(req("distill_teacher_url")),
        model=str(req("distill_teacher_model")),
        repo_id=str(req("distill_teacher_repo")),
        revision=str(req("distill_teacher_revision")),
        build_id=str(req("distill_teacher_build_id")),
        engine=str(req("distill_teacher_engine")),
        joint=bool(knobs["joint"]),
        auth_env=get("distill_teacher_key_env") or None,
        model_header=get("distill_teacher_model_header") or None,
    )


def stage_teacher_label(workdir: Path, knobs: dict[str, Any]) -> None:
    """Soft-label the train side with the teacher, resumable from its response cache."""
    from jev_factory.distill import edge as dedge

    ctx = _ctx()
    cfg = _teacher_config(ctx, knobs)
    dedge.acquire_teacher_lock(ctx.work)  # final and edge refuse while it is held
    try:
        _label(ctx, Path(workdir), cfg)
    finally:
        dedge.release_teacher_lock(ctx.work)


def _label(ctx: DistillContext, workdir: Path, cfg: Any) -> None:
    from jev_factory.distill import teacher

    client = ctx.services.teacher_client
    if client is None:
        from jev_factory.distill.systemone import SystemOneClient

        client = SystemOneClient(
            cfg.base_url,
            model=cfg.model,
            timeout=cfg.timeout,
            auth_env=cfg.auth_env,
            model_header=cfg.model_header,
        )
    records = _train_records(ctx)
    total = len(records) if cfg.joint else sum(len(r.questions) for r in records)
    cache = Path(workdir) / TEACHER_DIR / teacher.CACHE_DIR
    cached = len(list(cache.glob("*.json"))) if cache.is_dir() else 0
    progress = _CountedClient(client, jobs_dir(ctx.work), "teacher-label", total, cached)
    teacher.label_dataset(_dataset_dir(ctx), Path(workdir) / TEACHER_DIR, cfg, client=progress)


class _CountedClient:
    """Wraps a teacher client and writes ``<name>.progress.json`` after every request.

    ``jev status`` shows it beside the stage. A resumed run starts from the number of cached
    (fully answered) requests, so done/total approximates the labelled share of the train side.
    """

    def __init__(self, client: Any, jobdir: Path, name: str, total: int, done: int) -> None:
        from jev_factory.factory.detach import Progress

        self.client = client
        self.progress = Progress(jobdir, name, max(total, 0), min(done, max(total, 0)))

    def ask(self, record: Mapping[str, Any]) -> Any:
        result = self.client.ask(record)
        self.progress.update(min(self.progress.done + 1, self.progress.total))
        return result


def _audit_floor(reg: dprereg.DistillPrereg, knobs: Mapping[str, Any]) -> float:
    if knobs.get("floor") is not None:
        return float(knobs["floor"])
    grid = list(reg.knob_grid.get("teacher_weak_floor") or [])
    if len(grid) == 1:
        return float(grid[0])
    raise _err(
        f"the teacher audit needs a floor; the pre-registered grid has {grid}",
        "pass --knob floor=<one of the pre-registered teacher_weak_floor values>",
    )


def stage_audit(workdir: Path, knobs: dict[str, Any]) -> None:
    """Audit the teacher's train-side soft labels against gold, per source."""
    ctx = _ctx()
    reg = dprereg.require_locked(_prereg_path(ctx), ctx.work, ctx.deviation_id)
    role = str(knobs["role"])
    if knobs.get("labels"):
        labels = Path(knobs["labels"])
    elif role == "clef":
        labels = Path(workdir) / TEACHER_LABELS
    else:
        raise _err(
            f"the {role!r} audit needs that teacher's label file",
            "pass --knob labels=<path to its teacher_labels.jsonl>",
        )
    daudit.run_audit(
        _train_records(ctx), labels, ctx.work, reg, floor=_audit_floor(reg, knobs), role=role
    )


def _weak_rows(ctx: DistillContext) -> list[Any]:
    table = daudit.require_audit(ctx.work, "clef")
    weak = {s["source"] for s in table.get("sources") or () if s.get("teacher_weak")}
    return [r for r in _train_records(ctx) if r.source in weak]


def stage_corrections_draft(workdir: Path, knobs: dict[str, Any]) -> None:
    """Draft proposed corrections for teacher-weak rows; the operator reviews them."""
    from jev_factory.distill import corrections as dcorr
    from jev_factory.review import corrections as review

    ctx = _ctx()
    out = Path(workdir) / CORRECTIONS_FILE
    if review.review_path(out).is_file():
        raise _err(
            f"{out} already has operator review decisions; redrafting would orphan them",
            "keep the reviewed corrections, or record a deviation and move them aside",
        )
    rows = _weak_rows(ctx)
    drafter = ctx.services.drafter
    if drafter is None and rows:
        from jev_factory.distill.systemone import SystemOneClient

        drafter = dcorr.systemone_drafter(
            SystemOneClient(
                str(ctx.config.require("distill_alt_teacher_url")),
                model=str(ctx.config.require("distill_alt_teacher_model")),
            )
        )
    author = knobs.get("author") or f"model:{ctx.config.get('distill_alt_teacher_model')}"
    if rows and drafter is not None:
        protected = dassemble._protected_entries(_dataset_dir(ctx))  # noqa: SLF001
        result = dcorr.draft_corrections(
            rows, drafter, protected, author=author, date=ctx.today(), reason=knobs["reason"]
        )
    else:
        result = dcorr.DraftResult()
    out.parent.mkdir(parents=True, exist_ok=True)
    dcorr.write_corrections(out, result.proposed)
    _write_json(
        Path(workdir) / CORRECTIONS_DRAFT,
        {
            "stage": "distill-corrections-draft",
            "weak_rows": len(rows),
            "proposed": len(result.proposed),
            "refused": [{"record_id": rid, "reason": why} for rid, why in result.refused],
            "author": author,
            "corrections": str(out),
            "review": f"jev review --corrections {out}",
        },
    )


def stage_assemble(workdir: Path, knobs: dict[str, Any]) -> None:
    """Assemble one target per (record, question) by precedence, leakage-check and freeze."""
    ctx = _ctx()
    reg = dprereg.require_locked(_prereg_path(ctx), ctx.work, ctx.deviation_id)
    corrections = Path(workdir) / CORRECTIONS_FILE
    dassemble.assemble(
        ctx.work,
        _dataset_dir(ctx),
        prereg=reg,
        teacher_filter=str(knobs["teacher_filter"]),
        clef_labels=Path(workdir) / TEACHER_LABELS,
        alt_labels=knobs.get("alt_labels") or None,
        corrections_path=corrections if corrections.is_file() else None,
        deviation_id=ctx.deviation_id,
    )


def _probe_records(ctx: DistillContext) -> list[Any]:
    """The operator's private probe set (``distill_probe_set``), refused if it touches a sealed
    side; empty when unset."""
    path = ctx.config.get("distill_probe_set")
    if not path:
        return []
    held_out, test = _sealed(ctx)
    return dfinal.load_probe(path, {"held_out": held_out, "test": test})


def _clef_client(ctx: DistillContext) -> Any:
    """Clef's /v1/systemone client with the bearer key (by env-var name) and model header,
    as teacher-label sends them; without them the gateway answers 401."""
    from jev_factory.distill.systemone import SystemOneClient

    get = ctx.config.get
    return SystemOneClient(
        str(ctx.config.require("distill_teacher_url")),
        model=str(ctx.config.require("distill_teacher_model")),
        auth_env=get("distill_teacher_key_env") or None,
        model_header=get("distill_teacher_model_header") or None,
    )


def stage_benchmark(workdir: Path, knobs: dict[str, Any]) -> None:
    """Zero-shot benchmark of Clef and stock, plus the stock JevBench w4096 reproduction."""
    from jev_factory.distill import benchmark as bench

    ctx = _ctx()
    targets = ctx.services.benchmark_targets
    if targets is None:
        req = ctx.config.require
        targets = {
            "clef": bench.Target(
                "clef", str(req("distill_teacher_url")), str(req("distill_teacher_model"))
            ),
            "stock": bench.Target(
                "stock", str(req("distill_stock_url")), str(req("distill_stock_model"))
            ),
        }
    tasks_path = Path(ctx.config.require("distill_jevbench_tasks"))
    tasks = bench.load_jevbench(tasks_path)  # refused before any model is served
    probe = _probe_records(ctx)
    clients = ctx.services.benchmark_clients
    if clients is None and ctx.services.benchmark_targets is None:
        clients = {"clef": _clef_client(ctx)}
    result = bench.run_benchmark(
        list(targets.values()),
        _folds(ctx),
        sealed=_sealed(ctx),
        memory_probe=_memory_probe(ctx),
        clients=clients,
    )
    _write_json(Path(workdir) / BENCHMARK_FILE, result)
    stock = targets["stock"]
    client = (ctx.services.benchmark_clients or {}).get("stock")
    reproduction = bench.reproduce_stock(
        stock,
        tasks,
        base_revision=_base_revision(ctx),
        tasks_path=tasks_path,
        memory_probe=_memory_probe(ctx),
        client=client,
    )
    bench.write_reproduction(ctx.work, reproduction)
    sets = {name: bench.load_fold(path) for name, path in _folds(ctx).items()}
    if probe:
        sets[bench.PROBE_SET] = probe
    measured = bench.measure_stock(
        stock, sets, memory_probe=_memory_probe(ctx), client=client, sealed=_sealed(ctx)
    )
    measured["probe_set"] = None if not probe else {"count": len(probe)}
    _write_json(ctx.work / bench.STOCK_SETS_REL, measured)


def _pins(ctx: DistillContext) -> Any:
    from jev_factory.distill import pins as dpins

    return dpins.default_pins(
        decider_revision=str(ctx.config.require("distill_decider_revision")),
        base_revision=_base_revision(ctx),
    )


def _trainer_env(ctx: DistillContext) -> TrainerEnv:
    return TrainerEnv(
        python=str(ctx.config.require("distill_trainer_python")),
        decider_repo=ctx.config.require("distill_decider_repo"),
    )


def _gpu_limits(ctx: DistillContext) -> GpuLimits:
    return GpuLimits(
        memory_max=ctx.config.get("train_memory_max"),
        memory_floor=str(ctx.config.get("train_memory_floor") or "8G"),
        allow_foreign=ctx.allow_foreign_gpu,
    )


def _train_kwargs(ctx: DistillContext) -> dict[str, Any]:
    return {
        "prereg_path": _prereg_path(ctx),
        "run_dir": ctx.work,
        "cache_dir": ctx.config.require("distill_pin_cache"),
        "pins": _pins(ctx),
        "folds": _folds(ctx),
        "trainer_env": _trainer_env(ctx),
        "sealed": _sealed(ctx),
        "deviation_id": ctx.deviation_id,
        "limits": _gpu_limits(ctx),
    }


def _require_reproduction(ctx: DistillContext) -> None:
    """Train refuses without the stock reproduction record (fail closed) or on a base
    mismatch, unless a deviation id is given."""
    from jev_factory.distill import benchmark as bench

    bench.require_base_match(ctx.work, ctx.deviation_id, required=True)


def stage_train(workdir: Path, knobs: dict[str, Any]) -> None:
    """Train the pre-registered knob grid, calibrate on fit, select on the selection fold."""
    ctx = _ctx()
    _require_reproduction(ctx)
    scorer = ctx.services.scorer
    if scorer is None:
        scorer = dtrain.served_scorer(
            env=_student_env(ctx),
            jobdir=jobs_dir(ctx.work),
            port=int(ctx.config.get("distill_student_port")),
            base_revision=_base_revision(ctx),
            allow_foreign=ctx.allow_foreign_gpu,
        )
    dtrain.run_grid(**_train_kwargs(ctx), apply=True, scorer=scorer)


def _quant_kwargs(ctx: DistillContext, knobs: Mapping[str, Any]) -> dict[str, Any]:
    memory_floor = str(ctx.config.get("train_memory_floor") or "8G")
    builders = ctx.services.builders
    if builders is None:
        builders = dquant.default_builders(
            _base_revision(ctx),
            python=ctx.config.get("distill_student_python"),
            decider_repo=ctx.config.get("distill_decider_repo"),
            memory_max=ctx.config.get("train_memory_max"),
            memory_floor=memory_floor,
            allow_foreign=ctx.allow_foreign_gpu,
        )
    kwargs: dict[str, Any] = {
        "prereg_path": _prereg_path(ctx),
        "run_dir": ctx.work,
        "folds": _folds(ctx),
        "trainer_env": _trainer_env(ctx),
        "hooks": dquant.QuantHooks(builders=builders, trainer=ctx.services.trainer),
        "sealed": _sealed(ctx),
        "deviation_id": ctx.deviation_id,
        "limits": _gpu_limits(ctx),
        "probe": _probe_records(ctx),
    }
    quant_knobs = dquant.QuantKnobs()
    if knobs.get("formats"):
        quant_knobs = replace(quant_knobs, formats=tuple(knobs["formats"]))
    if knobs.get("heal_mix") is not None:
        quant_knobs = replace(quant_knobs, heal_mix=knobs["heal_mix"])
    if knobs.get("gating") is not None:
        quant_knobs = replace(quant_knobs, gating=_gating(ctx, knobs["gating"]))
    if knobs.get("heal_repeat") is not None:
        quant_knobs = replace(quant_knobs, heal_repeat=knobs["heal_repeat"])
    kwargs["knobs"] = quant_knobs
    return kwargs


#: ``gating`` knob values: the guard floors a format (d12) or is reported only (d14).
GATING_VALUES = ("clean", "clean-report")


def _gating(ctx: DistillContext, value: Any) -> Any:
    """Deviation d12: the quant rules read the contamination stage's clean sources only; the
    guard (every other source) fails a format only below stock's guard accuracy minus
    ``GUARD_TOLERANCE_POINTS`` on the selection fold. ``clean-report`` (deviation d14) reports
    the guard beside stock's and never fails or heals a format on it."""
    if value not in GATING_VALUES:
        raise _err(
            f"knob gating must be one of {list(GATING_VALUES)}, got {value!r}",
            "pass --knob gating=clean (with --deviation-id <id>) or leave it unset",
        )
    clean = _read_json(stage_dir(ctx.work) / CONTAMINATION_FILE).get("clean") or []
    if not clean:
        raise _err("the contamination report names no clean source", "run contamination first")
    gating = dquant.Gating(sources=frozenset(clean))
    selection = dtrain._folds(_folds(ctx), _sealed(ctx))["selection"]  # noqa: SLF001
    _domain, guard = gating.split({"selection": selection})
    if not guard["selection"]:
        return gating
    bench = _read_json(stage_dir(ctx.work) / BENCHMARK_FILE)
    rows = [
        r
        for r in ((bench.get("models") or {}).get("stock") or {}).get("predictions") or []
        if r.get("fold") == "selection"
    ]
    stock = dquant.raw_accuracy(guard["selection"], rows)
    if stock["accuracy"] is None or stock["problems"]:
        raise _err(
            f"stock's guard accuracy on the selection fold is unusable ({stock})",
            "rerun the benchmark stage",
        )
    floor = stock["accuracy"] - dquant.GUARD_TOLERANCE_POINTS / 100.0
    return dquant.Gating(
        sources=frozenset(clean),
        guard_floor=floor if value == "clean" else None,
        guard_stock=stock["accuracy"],
    )


def stage_quantize(workdir: Path, knobs: dict[str, Any]) -> None:
    """Spike the deployed formats, heal once if needed, recalibrate the deployed student."""
    ctx = _ctx()
    kwargs = _quant_kwargs(ctx, knobs)
    kwargs["hooks"] = replace(kwargs["hooks"], measurer=_measurer(ctx))
    dquant.run_quant(**kwargs, apply=True)


def _gate_kwargs(ctx: DistillContext) -> dict[str, Any]:
    return {
        "prereg_path": _prereg_path(ctx),
        "folds": _folds(ctx),
        "sealed": _sealed(ctx),
        "deviation_id": ctx.deviation_id,
    }


def stage_gate(workdir: Path, knobs: dict[str, Any]) -> None:
    """Fit the deployed student's per-kind answer/abstain gate on the fit fold."""
    ctx = _ctx()
    dgate.run_gate(ctx.work, **_gate_kwargs(ctx), apply=True, measurer=_measurer(ctx))


def _bf16_artifact(ctx: DistillContext) -> Path:
    """The bf16 winner's export the quant stage measured every format against.

    It is the selected (pre-heal) distilled student even after a heal: quantize measures a
    healed build's agreement against these same predictions (``reference=`` in
    ``quant._heal_round``), so final's quant-vs-bf16 bar and quantize's agreement share one
    reference. (A healed build's *accuracy* drop is measured against the heal round's own
    bf16; see the delivery summary's follow-up.)"""
    path = stage_dir(ctx.work) / QUANT_MEASUREMENTS
    bf16 = ((_read_json(path).get("formats") or {}).get(dquant.BF16)) or {}
    if not bf16.get("available") or not bf16.get("artifact"):
        raise _err(f"{path}: no bf16 artifact was built", "run the distill quantize stage first")
    artifact = Path(str(bf16["artifact"]))
    return artifact if artifact.is_absolute() else ctx.work / artifact


def _bf16_final_model(ctx: DistillContext) -> Any:
    """The distilled bf16 student, served from its own export on the Spark (127.0.0.1)."""
    from jev_factory.backbones.pointer import serve as pserve
    from jev_factory.distill import benchmark as bench

    artifact = _bf16_artifact(ctx)
    env, jobdir = _student_env(ctx), jobs_dir(ctx.work)
    port = int(ctx.config.get("distill_student_port"))
    base_url = f"http://{pserve.LOOPBACK}:{port}"
    name = "distill-final-student-bf16"
    target = bench.Target(
        name=dfinal.dreport.STUDENT_BF16,
        base_url=base_url,
        model=name,
        start=lambda: pserve.launch(jobdir, name, env, artifact, port=port),
        ready=lambda: pserve.wait_ready(base_url, timeout=600.0, jobdir=jobdir, name=name),
        stop=lambda: dtrain._stop_job(jobdir, name),  # noqa: SLF001
    )
    return dfinal.FinalModel(target=target, client=pserve.PointerClient(base_url, model=name))


def _agx_final_model(ctx: DistillContext) -> Any:
    """The shipped quant, pre-served on the AGX Orin behind a 127.0.0.1 tunnel (no start/stop).

    ``/v1/bundle`` and ``/v1/stats`` are read through the same tunnel (FinalModel.get), so the
    final attests the bundle that answers before any request."""
    from jev_factory.backbones.pointer import serve as pserve
    from jev_factory.distill import benchmark as bench
    from jev_factory.distill import edge as dedge

    base_url = _localhost_url(ctx, "distill_agx_url")
    name = "distill-final-student"
    return dfinal.FinalModel(
        target=bench.Target(name=dfinal.dreport.STUDENT, base_url=base_url, model=name),
        client=pserve.PointerClient(base_url, model=name),
        get=lambda path: dedge._get_json(base_url, path),  # noqa: SLF001 - localhost only
    )


def _localhost_url(ctx: DistillContext, key: str) -> str:
    from jev_factory.measure.preflight import require_localhost

    url = str(ctx.config.require(key))
    try:
        require_localhost(url)
    except ValueError as exc:
        raise _err(
            f"{key}: {exc}", "open a tunnel and set its http://127.0.0.1:<port> URL"
        ) from None
    return url


def _final_models(ctx: DistillContext) -> Mapping[str, Any]:
    """Clef, stock, the bf16 student (served here) and the shipped quant (on the AGX Orin)."""
    if ctx.services.final_models is not None:
        return ctx.services.final_models
    from jev_factory.distill import benchmark as bench

    req = ctx.config.require
    report = dfinal.dreport
    return {
        report.CLEF: dfinal.FinalModel(
            bench.Target(
                report.CLEF, str(req("distill_teacher_url")), str(req("distill_teacher_model"))
            ),
            client=_clef_client(ctx),
        ),
        report.STOCK: dfinal.FinalModel(
            bench.Target(
                report.STOCK, str(req("distill_stock_url")), str(req("distill_stock_model"))
            )
        ),
        report.STUDENT_BF16: _bf16_final_model(ctx),
        report.STUDENT: _agx_final_model(ctx),
    }


def _contamination_manifest(ctx: DistillContext) -> dict[str, Any]:
    """The upstream-sources manifest the contamination stage matched against; the final
    refuses to run before that stage has (its report is the gate)."""
    from jev_factory.distill import contamination

    report = stage_dir(ctx.work) / CONTAMINATION_FILE
    if not report.is_file():
        raise _err(
            f"{report}: the contamination check has not run",
            "run: jev distill contamination --work <dir> --apply",
        )
    return contamination.load_manifest()


def _final_inputs(ctx: DistillContext, knobs: Mapping[str, Any] | None = None) -> Any:
    calibration = stage_dir(ctx.work) / SELECT_CALIBRATION
    unverified = (knobs or {}).get("unverified_gold", False)
    if not isinstance(unverified, bool):
        raise _err(
            f"knob unverified_gold must be true or false, got {unverified!r}",
            "pass --knob unverified_gold=true (with --deviation <id>) or leave it unset",
        )
    return dfinal.FinalInputs(
        gold_ledger=ctx.config.get("distill_gold_ledger") or None,
        probe=ctx.config.get("distill_probe_set") or None,
        bf16_calibration=calibration if calibration.is_file() else None,
        contamination_manifest=_contamination_manifest(ctx),
        unverified_gold=unverified,
    )


def _final_kwargs(ctx: DistillContext, knobs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prereg_path": _prereg_path(ctx),
        "run_dir": ctx.work,
        "dataset_dir": _dataset_dir(ctx),
        "deviation_id": ctx.deviation_id,
        "clef_calibration": knobs.get("clef_calibration") or None,
        "memory_probe": _memory_probe(ctx),
        "inputs": _final_inputs(ctx, knobs),
    }


def stage_final(workdir: Path, knobs: dict[str, Any]) -> None:
    """Score Clef, stock, the bf16 student and the shipped quant once, on test and held-out."""
    ctx = _ctx()
    dfinal.run_final(**_final_kwargs(ctx, knobs), models=_final_models(ctx), apply=True)


_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_memory_cap(value: Any) -> int | None:
    """``distill_edge_memory_cap`` in bytes: an integer, or digits with a K/M/G/T suffix
    (base 1024, as systemd's ``MemoryMax=`` reads it); ``None`` when unset."""
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    unit = text[-1] if text and text[-1] in _SIZE_UNITS else ""
    digits = text[: len(text) - len(unit)]
    if not digits.isdigit() or int(digits) <= 0:
        raise _err(
            f"distill_edge_memory_cap: {value!r} is not a memory size",
            "set bytes, or digits with K, M, G or T (base 1024), e.g. 4G",
        )
    return int(digits) * _SIZE_UNITS[unit]


def _edge_kwargs(ctx: DistillContext, knobs: Mapping[str, Any]) -> dict[str, Any]:
    cap = parse_memory_cap(ctx.config.get("distill_edge_memory_cap"))
    if cap is None:
        raise _err(
            "the edge stage needs the expected cgroup memory cap",
            "set distill_edge_memory_cap (the MemoryMax the board's server runs under)",
        )
    probe = ctx.config.require("distill_probe_set")
    return {
        "run_dir": ctx.work,
        "dataset_dir": _dataset_dir(ctx),
        "selection": _folds(ctx)["selection"],
        "probe": probe,
        "base_url": _localhost_url(ctx, "distill_edge_url"),
        "served_model": str(knobs["served_model"]),
        "label": str(knobs["label"]),
        "expected_cap": cap,
        "client": ctx.services.edge_client,
        "get": ctx.services.edge_get,
    }


def stage_edge(workdir: Path, knobs: dict[str, Any]) -> None:
    """Measure the shipped quant on a board (selection fold and probe set), through a tunnel."""
    from jev_factory.distill import edge as dedge

    ctx = _ctx()
    dedge.run_edge(**_edge_kwargs(ctx, knobs), apply=True)


def stage_bundle(workdir: Path, knobs: dict[str, Any]) -> None:
    """Write the private model bundle and the Apache-2.0/MIT dataset bundle."""
    from jev_factory.distill import bundle

    ctx = _ctx()
    bundle.build_bundles(
        ctx.work,
        Path(workdir) / BUNDLE_DIR,
        prereg_path=_prereg_path(ctx),
        dataset_dir=_dataset_dir(ctx),
        deviation_id=ctx.deviation_id,
        deviations=[ctx.deviation_id] if ctx.deviation_id else [],
    )


def _upload_kwargs(ctx: DistillContext, knobs: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in ("model_repo", "dataset_repo") if not knobs.get(k)]
    if missing:
        raise _err(
            f"upload needs the repositories named: {', '.join(missing)}",
            "pass --knob model_repo=<prefix...> --knob dataset_repo=<prefix...> (private-first)",
        )
    base = stage_dir(ctx.work)
    return {
        "bundles": {"model": base / BUNDLE_MODEL, "dataset": base / BUNDLE_DATASET},
        "model_repo": str(knobs["model_repo"]),
        "dataset_repo": str(knobs["dataset_repo"]),
        "config": ctx.config,
        "hub_client": ctx.services.hub,
        "environ": ctx.services.environ,
    }


def stage_upload(workdir: Path, knobs: dict[str, Any]) -> None:
    """Upload both bundles privately, fetch them back and verify every sha256."""
    from jev_factory.distill import bundle

    ctx = _ctx()
    result = bundle.upload_bundles(**_upload_kwargs(ctx, knobs), apply=True)
    _write_json(Path(workdir) / UPLOAD_FILE, result)


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
    return Stage(name, func, inputs, outputs, deps, doc)


def _stages() -> list[Stage]:
    s = _stage
    teacher_out = (TEACHER_LABELS, TEACHER_MANIFEST)
    return [
        s("preregister", stage_preregister, (PREREG_FILE,), (PREREG_LOCK,), ()),
        s("dataset", stage_dataset, (), (DATASET_DIR,), ()),
        s(
            "contamination",
            stage_contamination,
            (DATASET_MANIFEST,),
            (CONTAMINATION_FILE,),
            ("dataset",),
        ),
        s("teacher-label", stage_teacher_label, (DATASET_MANIFEST,), teacher_out, ("dataset",)),
        s(
            "audit",
            stage_audit,
            (PREREG_FILE, TEACHER_LABELS),
            (AUDIT_FILE,),
            ("preregister", "teacher-label"),
        ),
        s(
            "corrections-draft",
            stage_corrections_draft,
            (AUDIT_FILE,),
            (CORRECTIONS_FILE, CORRECTIONS_DRAFT),
            ("audit",),
        ),
        s(
            "assemble",
            stage_assemble,
            (TEACHER_LABELS, AUDIT_FILE, CORRECTIONS_FILE),
            (TRAIN_SET, ASSEMBLE_MANIFEST),
            ("audit", "corrections-draft"),
        ),
        s(
            "benchmark",
            stage_benchmark,
            (DATASET_MANIFEST,),
            (BENCHMARK_FILE, STOCK_REPRODUCTION, STOCK_MEASUREMENTS),
            ("dataset",),
        ),
        s(
            "train",
            stage_train,
            (PREREG_FILE, TRAIN_SET),
            (SELECT_SUMMARY,),
            ("assemble",),
        ),
        s(
            "quantize",
            stage_quantize,
            (SELECT_SUMMARY,),
            (QUANT_MANIFEST, QUANT_DECISION, QUANT_CALIBRATION, QUANT_PREDICTIONS),
            ("train",),
        ),
        s("gate", stage_gate, (QUANT_CALIBRATION,), (GATE_FILE,), ("quantize",)),
        s(
            "final",
            stage_final,
            (QUANT_CALIBRATION, GATE_FILE, CONTAMINATION_FILE),
            (FINAL_FILE, FINAL_REPORT),
            ("contamination", "gate"),
        ),
        s("edge", stage_edge, (FINAL_FILE, QUANT_PREDICTIONS), (EDGE_DIR,), ("final",)),
        s(
            "bundle",
            stage_bundle,
            (FINAL_REPORT, GATE_FILE),
            (BUNDLE_MODEL, BUNDLE_DATASET),
            ("final",),
        ),
        s("upload", stage_upload, (BUNDLE_MODEL, BUNDLE_DATASET), (UPLOAD_FILE,), ("bundle",)),
    ]


def build_registry(registry: Registry | None = None) -> Registry:
    """Register every distill stage, in :data:`STAGE_ORDER`, into *registry* (a new one)."""
    reg = registry if registry is not None else Registry()
    for stage in _stages():
        reg.register(stage)
    return reg


#: The distillery's own registry: never the engine's default (what ``jev run`` lists).
REGISTRY = build_registry()


def stage_names() -> list[str]:
    return REGISTRY.names()


def get_stage(name: str) -> Stage:
    """The named distill stage, or a CliError that lists every stage (nothing runs)."""
    try:
        return REGISTRY.get(name)
    except KeyError:
        raise _err(
            f"unknown distill stage {name!r}", "stages, in order: " + ", ".join(REGISTRY.names())
        ) from None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _external(name: str, ctx: DistillContext) -> dict[str, Any]:
    """sha256 of the files a stage reads from outside its declared inputs (staleness)."""
    if name == "dataset" and ctx.config.get("distill_records"):
        return {"distill_records": sha256_path(Path(ctx.config["distill_records"]))}
    if name == "train":  # not a declared input: a deviation id may stand in for it
        return {"stock_reproduction": sha256_path(stage_dir(ctx.work) / STOCK_REPRODUCTION)}
    if name == "assemble":
        from jev_factory.review import corrections as review

        path = review.review_path(stage_dir(ctx.work) / CORRECTIONS_FILE)
        return {"corrections_review": sha256_path(path)}
    return {}


def effective_knobs(name: str, knobs: Mapping[str, Any] | None, ctx: DistillContext) -> dict:
    """Defaults, then *knobs*; plus the config keys and the external files the stage reads."""
    get_stage(name)
    given = dict(knobs or {})
    unknown = sorted(set(given) - set(DEFAULT_KNOBS[name]))
    if unknown:
        known = ", ".join(sorted(DEFAULT_KNOBS[name])) or "none"
        raise _err(
            f"unknown knob(s) for distill {name}: {', '.join(unknown)}", f"known knobs: {known}"
        )
    merged = json.loads(json.dumps({**DEFAULT_KNOBS[name], **given}))
    merged["_config"] = {k: ctx.config.get(k) for k in STAGE_CONFIG_KEYS.get(name, ())}
    external = _external(name, ctx)
    if external:
        merged["_external"] = external
    return merged


def _module_plan(name: str, ctx: DistillContext, knobs: Mapping[str, Any]) -> dict[str, Any]:
    """The owning module's own dry run (``apply=False``: reads, never writes)."""
    if name == "train":
        _require_reproduction(ctx)
        return dtrain.run_grid(**_train_kwargs(ctx), apply=False)
    if name == "quantize":
        return dquant.run_quant(**_quant_kwargs(ctx, knobs), apply=False)
    if name == "gate":
        return dgate.run_gate(ctx.work, **_gate_kwargs(ctx), apply=False)
    if name == "final":
        models = _final_models(ctx)
        return dfinal.run_final(**_final_kwargs(ctx, knobs), models=models, apply=False)
    if name == "edge":
        from jev_factory.distill import edge as dedge

        return dedge.run_edge(**_edge_kwargs(ctx, knobs), apply=False)
    from jev_factory.distill import bundle

    return bundle.upload_bundles(**_upload_kwargs(ctx, knobs), apply=False)


def plan(
    work: Path, name: str, ctx: DistillContext, knobs: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The dry run: what distill stage *name* reads and writes and whether it would run."""
    stage = get_stage(name)
    workdir = stage_dir(work)
    effective = effective_knobs(name, knobs, ctx)
    inputs = {rel: sha256_path(workdir / rel) for rel in stage.inputs}
    recorded = {
        other: (read_manifest(workdir, other) or {}).get("knobs", {}) for other in REGISTRY.names()
    }
    reason = staleness(workdir, name, {**recorded, name: effective}, registry=REGISTRY)
    manifest = read_manifest(workdir, name)
    out: dict[str, Any] = {
        "stage": name,
        "verb": "distill",
        "applied": False,
        "summary": stage.summary,
        "deps": list(stage.deps),
        "inputs": inputs,
        "missing_inputs": [rel for rel, digest in inputs.items() if digest is None],
        "outputs": list(stage.outputs),
        "knobs": effective,
        "unset_config": [
            k for k in STAGE_CONFIG_KEYS.get(name, ()) if ctx.config.get(k) in (None, "")
        ],
        "stale": reason,
        "would_run": manifest is None or manifest.get("status") != COMPLETE or reason is not None,
        "note": "dry run: nothing was written; pass --apply to run the stage",
    }
    if name in _MODULE_PLANS:
        from jev_factory.release.hub import UploadError

        token = _CURRENT.set(ctx)
        try:
            out["module_plan"] = _module_plan(name, ctx, effective)
        except CliError as exc:
            out["module_plan_blocked"] = {"message": exc.message, "remediation": exc.remediation}
        except UploadError as exc:
            out["module_plan_blocked"] = {
                "message": str(exc),
                "remediation": "fix what the message names (repo prefix or bundle folders)",
            }
        finally:
            _CURRENT.reset(token)
    return out


def _guarded(func: Callable[[Path, dict[str, Any]], None]) -> Callable[[Path, dict], None]:
    def call(workdir: Path, knobs: dict[str, Any]) -> None:
        try:
            func(workdir, knobs)
        except CliError as exc:
            hint = f" (hint: {exc.remediation})" if exc.remediation else ""
            raise CliError(exc.code, exc.message + hint, exc.remediation) from None

    return call


def run(
    work: Path,
    name: str,
    ctx: DistillContext,
    knobs: Mapping[str, Any] | None = None,
    *,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run distill stage *name* for the run at *work*: a dry run (:func:`plan`) unless *apply*.

    With *apply* the work root must be outside any git worktree and the run lock is held for
    the whole stage; the stage engine writes ``<work>/distill/manifests/<name>.json``. A fresh
    stage is skipped unless *force*. Returns the manifest.
    """
    if not apply:
        return plan(work, name, ctx, knobs)
    get_stage(name)
    work = resolve_work_root(work)
    ctx = replace(ctx, work=work)
    effective = effective_knobs(name, knobs, ctx)
    one = Registry()
    for other in REGISTRY.names():
        st = REGISTRY.get(other)
        one.register(replace(st, func=_guarded(st.func)) if other == name else st)
    token = _CURRENT.set(ctx)
    try:
        with RunLock(work):
            stage_dir(work).mkdir(parents=True, exist_ok=True)
            return run_stage(stage_dir(work), name, effective, registry=one, force=force)
    finally:
        _CURRENT.reset(token)
