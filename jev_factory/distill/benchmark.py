"""Zero-shot benchmark of the teacher (Clef) and stock v21 on the fit/selection folds (t16).

Each model is served in turn (one GPU job at a time): ``start`` brings the server up, every
record of the requested folds is sent through :class:`SystemOneClient`, the answers are kept
as *predictions records* with per-call latency, and ``stop`` always runs before the next model
starts. Peak memory is the maximum of the samples taken from a probe (default: ``nvidia-smi``
used memory) before, during and after the run, on the same box for every model.

Only training-side folds are accepted. A :class:`SealedSide`, a held-out or test file, a record
whose split is ``test`` or whose id is in a sealed side is refused (fail closed): the sealed
sides are touched once, in the final run only. Stdlib only; servers are localhost only.
"""

from __future__ import annotations

import math
import statistics
import subprocess  # nosec B404 - fixed argv, no shell
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jev_factory.cli._errors import EXIT_USER_ERROR, CliError
from jev_factory.distill import metrics
from jev_factory.distill.dataset import TEST_SPLIT, SealedSide, read_records
from jev_factory.distill.records import Record
from jev_factory.distill.systemone import CallResult, SystemOneClient

FOLDS = ("fit", "selection")
SEALED_NAMES = ("held_out", "heldout", "held-out", "test")
SAMPLE_INTERVAL = 0.2

MemoryProbe = Callable[[], "float | None"]


def _refuse(message: str, remediation: str = "") -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def nvidia_smi_used_mib() -> float | None:
    """Total GPU memory in use (MiB) from ``nvidia-smi``, or ``None`` when it is unavailable.

    On unified-memory boxes ``nvidia-smi`` may report no per-GPU figure; that is ``None``,
    recorded as unmeasured rather than zero.
    """
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    values = []
    for line in out.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return sum(values) if values else None


# ---------------------------------------------------------------------------
# Input guard
# ---------------------------------------------------------------------------


def load_fold(path: Path | str) -> list[Record]:
    """Read one fit/selection jsonl, refusing a held-out or test file by name or content."""
    path = Path(path)
    stem = path.stem.lower()
    if stem in SEALED_NAMES or any(stem.startswith(n + ".") for n in SEALED_NAMES):
        raise _refuse(
            f"{path.name}: sealed side refused; only the fit and selection folds are benchmarked",
            "pass fit.jsonl and selection.jsonl",
        )
    return check_records(read_records(path), origin=str(path))


def check_records(
    records: Iterable[Record], *, sealed: Sequence[SealedSide] = (), origin: str = "records"
) -> list[Record]:
    """Return *records* as a list, or refuse if any is a sealed-side record."""
    out = list(records)
    sealed_ids = {i for side in sealed for i in side.ids}
    for rec in out:
        if rec.split == TEST_SPLIT:
            raise _refuse(
                f"{origin}: record {rec.id!r} has split 'test'; test records are refused",
                "benchmark the fit and selection folds only",
            )
        if rec.id in sealed_ids:
            raise _refuse(f"{origin}: record {rec.id!r} is in a sealed side; refused")
    return out


def collect_folds(
    folds: Mapping[str, Any], *, sealed: Sequence[SealedSide] = ()
) -> dict[str, list[Record]]:
    """Resolve ``{fold: path | records}`` to records; only ``fit`` and ``selection`` allowed."""
    out: dict[str, list[Record]] = {}
    for name, value in folds.items():
        if name not in FOLDS:
            raise _refuse(
                f"fold {name!r} refused; only {FOLDS} may be benchmarked",
                "held-out and test are touched once, in the final run only",
            )
        if isinstance(value, SealedSide):
            raise _refuse(f"fold {name!r} is a sealed side; refused")
        if isinstance(value, (str, Path)):
            recs = check_records(load_fold(value), sealed=sealed, origin=str(value))
        else:
            recs = check_records(value, sealed=sealed, origin=f"fold {name!r}")
        out[name] = recs
    if not out:
        raise _refuse("no folds given", "pass fit and/or selection")
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100), ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[rank - 1]


def latency_summary(latencies: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(latencies),
        "median_s": statistics.median(latencies) if latencies else None,
        "p95_s": percentile(latencies, 95),
        "max_s": max(latencies) if latencies else None,
    }


class _PeakSampler:
    """Track the maximum of a memory probe, sampled in a thread and on demand."""

    def __init__(self, probe: MemoryProbe | None, interval: float = SAMPLE_INTERVAL):
        self.probe = probe
        self.interval = interval
        self.peak: float | None = None
        self.baseline: float | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        if self.probe is None:
            return
        try:
            value = self.probe()
        except Exception:  # noqa: BLE001 - a broken probe must not abort the benchmark
            value = None
        if value is None:
            return
        self.samples += 1
        self.peak = value if self.peak is None else max(self.peak, value)

    def __enter__(self) -> _PeakSampler:
        self.sample()
        self.baseline = self.peak
        if self.probe is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample()

    def report(self) -> dict[str, Any]:
        return {
            "probe": (
                "nvidia-smi memory.used (MiB)"
                if self.probe is nvidia_smi_used_mib
                else ("custom" if self.probe else None)
            ),
            "baseline_mib": self.baseline,
            "peak_mib": self.peak,
            "samples": self.samples,
            "measured": self.peak is not None,
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One model to benchmark: where it listens and how to bring it up and down."""

    name: str
    base_url: str
    model: str
    start: Callable[[], Any] | None = None
    stop: Callable[[], Any] | None = None
    ready: Callable[[], Any] | None = None


def _prediction_row(
    target: str, fold: str, rec: Record, result: CallResult, latency: float
) -> dict[str, Any]:
    return {
        "model": target,
        "fold": fold,
        "record_id": rec.id,
        "latency_s": latency,
        "error": result.error,
        "response_sha256": result.response_sha256,
        "questions": {
            q.qid: {
                "kind": q.kind,
                "distribution": q.distribution,
                "p_true": q.p_true,
                "error": q.error,
            }
            for q in result.questions
        },
    }


def wire_record(rec: Record) -> dict[str, Any]:
    """The SystemOne-shaped mapping for *rec*; gold, source and licence are not included.

    Choice criteria go as ``{option: description}``, score criteria as the ordered level
    descriptions (the server answers by level index) and noul criteria as ``None``.
    """
    questions: dict[str, Any] = {}
    for qid, q in rec.questions.items():
        if q.type == "choice":
            criteria: Any = {name: desc for name, desc in q.criteria}
        elif q.type == "score":
            criteria = [desc for _name, desc in q.criteria]
        else:
            criteria = None
        item: dict[str, Any] = {"type": q.type, "criteria": criteria}
        if q.instructions is not None:
            item["instructions"] = q.instructions
        questions[qid] = item
    return {"id": rec.id, "state": rec.state, "questions": questions}


def benchmark_model(
    target: Target,
    folds: Mapping[str, Sequence[Record]],
    *,
    memory_probe: MemoryProbe | None = nvidia_smi_used_mib,
    client: SystemOneClient | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Start *target*, score every record, stop it; return predictions, latency and memory."""
    client = client or SystemOneClient(target.base_url, model=target.model)
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    try:
        if target.start is not None:
            target.start()
        if target.ready is not None:
            target.ready()
        with _PeakSampler(memory_probe) as peak:
            for fold, records in folds.items():
                for rec in records:
                    t0 = clock()
                    result = client.ask(wire_record(rec))
                    elapsed = clock() - t0
                    latencies.append(elapsed)
                    predictions.append(_prediction_row(target.name, fold, rec, result, elapsed))
                    peak.sample()
    finally:
        if target.stop is not None:
            target.stop()
    failed = sum(1 for p in predictions if p["error"] or _has_error(p))
    return {
        "model": target.name,
        "served_model": target.model,
        "base_url": target.base_url,
        "n_records": len(predictions),
        "n_failed": failed,
        "latency": latency_summary(latencies),
        "memory": peak.report(),
        "predictions": predictions,
    }


def _has_error(row: Mapping[str, Any]) -> bool:
    return any(q["error"] for q in row["questions"].values())


def run_benchmark(
    targets: Sequence[Target],
    folds: Mapping[str, Any],
    *,
    sealed: Sequence[SealedSide] = (),
    memory_probe: MemoryProbe | None = nvidia_smi_used_mib,
    clients: Mapping[str, SystemOneClient] | None = None,
) -> dict[str, Any]:
    """Benchmark each target in sequence (never two at once) on the same folds.

    *folds* maps ``fit`` / ``selection`` to a jsonl path or records; anything sealed is refused
    before any server is started.
    """
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise _refuse("duplicate target names", "give each model a distinct name")
    resolved = collect_folds(folds, sealed=sealed)
    out: dict[str, Any] = {
        "folds": {k: len(v) for k, v in resolved.items()},
        "fold_ids": {k: [r.id for r in v] for k, v in resolved.items()},
        "models": {},
    }
    for target in targets:  # sequential: a target is stopped before the next one starts
        out["models"][target.name] = benchmark_model(
            target,
            resolved,
            memory_probe=memory_probe,
            client=(clients or {}).get(target.name),
        )
    return out


def pred_items(
    records: Sequence[Record], rows: Sequence[Mapping[str, Any]]
) -> list[metrics.PredItem]:
    """Gold-bearing metric items for the answered questions of *rows* (errors are skipped)."""
    by_id = {r.id: r for r in records}
    items: list[metrics.PredItem] = []
    for row in rows:
        rec = by_id.get(row["record_id"])
        if rec is None:
            continue
        for qid, ans in row["questions"].items():
            if ans["error"] or qid not in rec.questions:
                continue
            gold = rec.gold[qid]
            names = rec.questions[qid].names
            if ans["kind"] == "noul":
                items.append(metrics.noul_item(ans["p_true"], gold_true=_gold_true(gold)))
                continue
            probs = list(ans["distribution"].values())  # offered order (wire order == names)
            index = names.index(gold.label) if gold.label in names else None
            maker = metrics.choice_item if ans["kind"] == "choice" else metrics.score_item
            kw = "gold_index" if ans["kind"] == "choice" else "gold_level"
            items.append(maker(_normalise(probs), **{kw: index}))
    return items


def _gold_true(gold) -> bool | None:
    p = gold.probability_true
    return None if p is None else p >= 0.5


def _normalise(probs: Sequence[float]) -> list[float]:
    total = math.fsum(probs)
    return [p / total for p in probs]
