"""Zero-shot benchmark of the teacher (Clef) and the stock model on the fit/selection folds (t16).

Each model is served in turn (one GPU job at a time): ``start`` brings the server up, every
record of the requested folds is sent through :class:`SystemOneClient`, the answers are kept
as *predictions records* with per-call latency, and ``stop`` always runs before the next model
starts. Peak memory is the maximum of the samples taken from a probe (default: ``nvidia-smi``
used memory) before, during and after the run, on the same box for every model. On unified-memory
boxes (DGX Spark GB10) ``nvidia-smi`` reports ``[N/A]``; the fallback is then the summed resident
set (``VmRSS`` from ``/proc``) of the :class:`Target`'s served process and its descendants. The
report names the probe that produced the figure (``nvidia-smi`` or ``proc-rss``) and stays
``measured: false`` when neither works.

Only training-side folds are accepted. A :class:`SealedSide`, a held-out or test file, a record
whose split is ``test`` or whose id is in a sealed side is refused (fail closed): the sealed
sides are touched once, in the final run only. Stdlib only; servers are localhost only.
"""

from __future__ import annotations

import math
import re
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
SEALED_NAMES = ("train", "held_out", "heldout", "held-out", "test")
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


PROBE_NVIDIA = "nvidia-smi"
PROBE_PROC = "proc-rss"
PROC_ROOT = Path("/proc")
_VMRSS = re.compile(r"^VmRSS:\s+(\d+)\s*kB", re.MULTILINE)


def _ppid(stat_text: str) -> int | None:
    """Parent pid from ``/proc/<pid>/stat``; the comm field may hold spaces and parentheses."""
    try:
        return int(stat_text.rsplit(")", 1)[1].split()[1])
    except (IndexError, ValueError):
        return None


def proc_rss_mib(pid: int, proc_root: Path | str = PROC_ROOT) -> float | None:
    """Sum of ``VmRSS`` (MiB) over process *pid* and all its descendants, from ``/proc``.

    ``None`` when *pid* does not exist or nothing could be read.
    """
    root = Path(proc_root)
    children: dict[int, list[int]] = {}
    try:
        entries = [p for p in root.iterdir() if p.name.isdigit()]
    except OSError:
        return None
    for entry in entries:
        try:
            parent = _ppid((entry / "stat").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if parent is not None:
            children.setdefault(parent, []).append(int(entry.name))
    if not (root / str(pid)).is_dir():
        return None
    total_kb, seen, todo, found = 0, set(), [pid], False
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        todo.extend(children.get(cur, ()))
        try:
            text = (root / str(cur) / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _VMRSS.search(text)
        if m:
            total_kb += int(m.group(1))
            found = True
    return total_kb / 1024.0 if found else None


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

    def __init__(
        self,
        probe: MemoryProbe | None,
        interval: float = SAMPLE_INTERVAL,
        fallback: MemoryProbe | None = None,
    ):
        self.probe = probe
        self.fallback = fallback
        self.peak_probe: str | None = None
        self.interval = interval
        self.peak: float | None = None
        self.baseline: float | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read(probe: MemoryProbe | None) -> float | None:
        if probe is None:
            return None
        try:
            return probe()
        except Exception:  # noqa: BLE001 - a broken probe must not abort the benchmark
            return None

    def _label(self) -> str:
        return PROBE_NVIDIA if self.probe is nvidia_smi_used_mib else "custom"

    def sample(self) -> None:
        if self.probe is None:
            return
        value, label = self._read(self.probe), self._label()
        if value is None:
            value, label = self._read(self.fallback), PROBE_PROC
        if value is None:
            return
        self.samples += 1
        if self.peak is None or value > self.peak:
            self.peak, self.peak_probe = value, label

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
            "probe": self.peak_probe,
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
    #: Pid of the served process (or a callable returning it, read at each sample), for the
    #: ``/proc`` RSS fallback when the GPU probe has no figure.
    pid: int | Callable[[], int | None] | None = None


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
            criteria: Any = dict(q.criteria)
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
    proc_root: Path | str = PROC_ROOT,
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
        with _PeakSampler(memory_probe, fallback=_rss_probe(target, proc_root)) as peak:
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


def _rss_probe(target: Target, proc_root: Path | str) -> MemoryProbe | None:
    if target.pid is None:
        return None

    def probe() -> float | None:
        pid = target.pid() if callable(target.pid) else target.pid
        return None if pid is None else proc_rss_mib(pid, proc_root)

    return probe


def _has_error(row: Mapping[str, Any]) -> bool:
    return any(q["error"] for q in row["questions"].values())


def run_benchmark(
    targets: Sequence[Target],
    folds: Mapping[str, Any],
    *,
    sealed: Sequence[SealedSide] = (),
    memory_probe: MemoryProbe | None = nvidia_smi_used_mib,
    clients: Mapping[str, SystemOneClient] | None = None,
    proc_root: Path | str = PROC_ROOT,
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
            proc_root=proc_root,
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
            items.append(_pred_item(rec, qid, ans))
    return items


def _pred_item(rec: Record, qid: str, ans: Mapping[str, Any]) -> metrics.PredItem:
    """The metric item for one answered question of *rec*."""
    gold = rec.gold[qid]
    names = rec.questions[qid].names
    if ans["kind"] == "noul":
        return metrics.noul_item(ans["p_true"], gold_true=_gold_true(gold))
    probs = list(ans["distribution"].values())  # offered order (wire order == names)
    index = names.index(gold.label) if gold.label in names else None
    maker = metrics.choice_item if ans["kind"] == "choice" else metrics.score_item
    kw = "gold_index" if ans["kind"] == "choice" else "gold_level"
    return maker(_normalise(probs), **{kw: index})


def _gold_true(gold) -> bool | None:
    p = gold.probability_true
    return None if p is None else p >= 0.5


def _normalise(probs: Sequence[float]) -> list[float]:
    total = math.fsum(probs)
    return [p / total for p in probs]


# ---------------------------------------------------------------------------
# Stock reproduction (t13): does the served stock 2610 reproduce the card's JevBench w4096?
# ---------------------------------------------------------------------------

#: The 2610 card's ``eval/jevbench-w4096`` result: 180 right of 231 public tasks (0.7792).
CARD_N = 231
CARD_CORRECT = 180
CARD_ACCURACY = CARD_CORRECT / CARD_N
#: ``run_meta.json`` ``tasks_sha256`` of the card's task file (informational: our task file is
#: the operator's :class:`Record` conversion, so its hash is recorded, not required to match).
CARD_TASKS_SHA256 = "771ea8ac252af155606eeb09cad920b32b326c5c2b1e9c9af868be00fe8e2fad"
#: The card publishes no interval for 180/231; this stage defines it as the two-sided 95%
#: Wilson score interval of the *measured* correct/n (z = 1.96), a closed form that needs no
#: resampling and behaves at the edges. Seed-to-seed spread on the card (179.7 +- 3.8 of 231)
#: is context, not part of the test.
CI_METHOD = "wilson-95"
WILSON_Z = 1.959964
REPRODUCTION_REL = "benchmark/stock_reproduction.json"
STOCK_SETS_REL = "benchmark/stock_measurements.json"
PROBE_SET = "probe"
JEVBENCH_SET = "jevbench"
#: Sets whose per-record rows are never written out (the Clef probe set is private).
PRIVATE_SETS = (PROBE_SET,)


def wilson_interval(correct: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Two-sided Wilson score interval for ``correct`` of ``n`` (``(0.0, 1.0)`` for no data)."""
    if n <= 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _top1_hit(rec: Record, qid: str, ans: Mapping[str, Any]) -> bool | None:
    """Whether the answer's argmax is the gold, or ``None`` when it cannot be judged."""
    gold = rec.gold.get(qid)
    if gold is None or ans["error"] or qid not in rec.questions:
        return None
    if ans["kind"] == "noul":
        want = _gold_true(gold)
        return None if want is None or ans["p_true"] is None else (ans["p_true"] >= 0.5) == want
    dist = ans["distribution"]
    if not dist or gold.label not in dist:
        return None
    return max(dist, key=lambda k: dist[k]) == gold.label


def top1_summary(records: Sequence[Record], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Top-1 accuracy over the gold-bearing questions answered in *rows*, with a Wilson CI.

    A question the server failed on counts as wrong (the card's 180/231 counts failures
    against the model too); a question with no gold is left out of both counts.
    """
    by_id = {r.id: r for r in records}
    correct = n = 0
    for row in rows:
        rec = by_id.get(row["record_id"])
        if rec is None:
            continue
        for qid, ans in row["questions"].items():
            hit = _top1_hit(rec, qid, ans)
            if hit is None and ans["error"] and qid in rec.gold:
                hit = False
            if hit is None:
                continue
            n += 1
            correct += int(hit)
    lo, hi = wilson_interval(correct, n)
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n if n else None,
        "ci_low": lo if n else None,
        "ci_high": hi if n else None,
        "ci_method": CI_METHOD,
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jevbench(path: Path | str) -> list[Record]:
    """Read the JevBench public tasks (231, :class:`Record` jsonl), refusing a sealed file name."""
    path = Path(path)
    stem = path.stem.lower()
    if stem in SEALED_NAMES or any(stem.startswith(n + ".") for n in SEALED_NAMES):
        raise _refuse(f"{path.name}: sealed side refused as a JevBench task file")
    return check_records(read_records(path), origin=str(path))


def reproduce_stock(
    target: Target,
    tasks: Sequence[Record],
    *,
    base_revision: str,
    tasks_path: Path | str | None = None,
    memory_probe: MemoryProbe | None = None,
    client: SystemOneClient | None = None,
) -> dict[str, Any]:
    """Serve stock 2610 on the pinned base, score JevBench w4096 and compare with the card.

    Records n, accuracy and the Wilson CI. ``base_mismatch`` is true when that CI excludes the
    card's 0.7792 (180/231): the served base is then not the one the card was measured on, and
    :func:`require_base_match` makes ``train`` refuse until a deviation is recorded. A task set
    whose size differs from the card's 231 is refused outright (it is not the same benchmark).
    """
    if len(tasks) != CARD_N:
        raise _refuse(
            f"JevBench has {len(tasks)} tasks, the card's w4096 run has {CARD_N}",
            "pass the 231 public tasks; a different set is not a reproduction",
        )
    if not base_revision:
        raise _refuse("reproduction needs the pinned base revision", "pass base_revision")
    res = benchmark_model(target, {JEVBENCH_SET: tasks}, memory_probe=memory_probe, client=client)
    summary = top1_summary(tasks, res["predictions"])
    lo, hi = summary["ci_low"], summary["ci_high"]
    out = {
        "stage": "distill-stock-reproduction",
        "model": target.name,
        "served_model": target.model,
        "base_revision": base_revision,
        "tasks_sha256": _file_sha256(Path(tasks_path)) if tasks_path else None,
        "card_tasks_sha256": CARD_TASKS_SHA256,
        "card": {"n": CARD_N, "n_correct": CARD_CORRECT, "accuracy": CARD_ACCURACY},
        **summary,
        "n_failed": res["n_failed"],
        "latency": res["latency"],
        "memory": res["memory"],
        "base_mismatch": not (lo <= CARD_ACCURACY <= hi),
    }
    return out


def write_reproduction(run_dir: Path | str, result: Mapping[str, Any]) -> Path:
    """Persist the reproduction record where :func:`require_base_match` looks for it."""
    import json

    path = Path(run_dir) / REPRODUCTION_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_base_match(run_dir: Path | str, deviation_id: str | None = None) -> None:
    """Refuse (train) when the recorded reproduction marks a base mismatch, absent a deviation.

    No record on disk is not a refusal here (the benchmark stage has not run in a bare train
    call); a corrupt record is, because it cannot be trusted either way.
    """
    import json

    path = Path(run_dir) / REPRODUCTION_REL
    if not path.is_file():
        return
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        mismatch = bool(rec["base_mismatch"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _refuse(
            f"{path}: unreadable stock reproduction record ({exc})",
            "rerun the benchmark stage",
        ) from exc
    if mismatch and not (deviation_id or "").strip():
        raise _refuse(
            "stock 2610 did not reproduce the card's JevBench w4096 result (base-mismatch) and no deviation id was given: "
            f"{rec.get('n_correct')}/{rec.get('n')} with CI "
            f"[{rec.get('ci_low'):.4f}, {rec.get('ci_high'):.4f}] "
            f"excludes {CARD_ACCURACY:.4f}",
            "record a deviation and pass its id (--deviation <id>), or fix the served base",
        )


def measure_stock(
    target: Target,
    sets: Mapping[str, Sequence[Record]],
    *,
    memory_probe: MemoryProbe | None = None,
    client: SystemOneClient | None = None,
    sealed: Sequence[SealedSide] = (),
) -> dict[str, Any]:
    """Measure the stock model itself on the Clef probe set and the Jetson-ops fit/selection folds.

    *sets* maps ``probe`` / ``fit`` / ``selection`` to records; every figure comes from this
    server session, none is copied from elsewhere. Sealed records are refused. The probe set is
    private, so its per-record rows are left out of the result (counts and accuracy only).
    """
    unknown = set(sets) - {PROBE_SET, *FOLDS}
    if unknown:
        raise _refuse(f"unknown stock measurement sets {sorted(unknown)}")
    if not sets:
        raise _refuse("no sets given", "pass probe and/or fit and selection")
    checked = {k: check_records(v, sealed=sealed, origin=f"set {k!r}") for k, v in sets.items()}
    res = benchmark_model(target, checked, memory_probe=memory_probe, client=client)
    out_sets: dict[str, Any] = {}
    for name, recs in checked.items():
        rows = [p for p in res["predictions"] if p["fold"] == name]
        out_sets[name] = top1_summary(recs, rows)
    public = [p for p in res["predictions"] if p["fold"] not in PRIVATE_SETS]
    return {
        "stage": "distill-stock-measurement",
        "model": target.name,
        "sets": out_sets,
        "n_failed": res["n_failed"],
        "latency": res["latency"],
        "memory": res["memory"],
        "predictions": public,
    }
