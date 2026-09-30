"""Stage engine: a registry of pure-Python stages, manifests and staleness.

Replaces ``pipeline.sh``'s "die if the prior artifact is missing" sequencing.
A stage declares its inputs (paths relative to the work dir), its outputs and
its upstream stages. Each run writes ``<workdir>/manifests/<stage>.json``.
A stage is *stale* when it never completed, when any recorded input or knob
differs from the current one, when an output changed or vanished, or when any
upstream stage is stale. Re-running a fresh stage is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jev_factory import __version__

MANIFEST_DIR = "manifests"
COMPLETE = "complete"
FAILED = "failed"


@dataclass(frozen=True)
class Stage:
    name: str
    func: Callable[[Path, dict[str, Any]], None]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    summary: str = ""


@dataclass
class Registry:
    stages: dict[str, Stage] = field(default_factory=dict)

    def register(self, stage: Stage) -> Stage:
        if stage.name in self.stages:
            raise ValueError(f"stage already registered: {stage.name}")
        self.stages[stage.name] = stage
        return stage

    def get(self, name: str) -> Stage:
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(f"unknown stage: {name}") from None

    def names(self) -> list[str]:
        """Stage names in dependency (topological) order."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str, trail: tuple[str, ...]) -> None:
            if n in seen:
                return
            if n in trail:
                raise ValueError(f"stage dependency cycle: {' -> '.join(trail + (n,))}")
            for d in self.get(n).deps:
                visit(d, trail + (n,))
            seen.add(n)
            order.append(n)

        for n in self.stages:
            visit(n, ())
        return order

    def downstream(self, name: str) -> list[str]:
        """Every stage that transitively depends on ``name``, in order."""
        self.get(name)
        out: set[str] = set()
        for n in self.names():
            if any(d == name or d in out for d in self.stages[n].deps):
                out.add(n)
        return [n for n in self.names() if n in out]


DEFAULT_REGISTRY = Registry()


def register_stage(stage: Stage) -> Stage:
    return DEFAULT_REGISTRY.register(stage)


def list_stages(registry: Registry | None = None) -> list[dict[str, Any]]:
    reg = registry or DEFAULT_REGISTRY
    return [
        {
            "stage": n,
            "deps": list(reg.stages[n].deps),
            "inputs": list(reg.stages[n].inputs),
            "outputs": list(reg.stages[n].outputs),
            "summary": reg.stages[n].summary,
        }
        for n in reg.names()
    ]


def sha256_path(path: Path) -> str | None:
    """sha256 of a file, or of a directory tree; None if it does not exist."""
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(f.relative_to(path).as_posix().encode() + b"\0")
            h.update((sha256_path(f) or "").encode() + b"\0")
        return h.hexdigest()
    return None


def manifest_path(workdir: Path, name: str) -> Path:
    return Path(workdir) / MANIFEST_DIR / f"{name}.json"


def read_manifest(workdir: Path, name: str) -> dict[str, Any] | None:
    p = manifest_path(workdir, name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _hashes(workdir: Path, rels: tuple[str, ...]) -> dict[str, str | None]:
    return {r: sha256_path(Path(workdir) / r) for r in rels}


def _own_staleness(reg: Registry, workdir: Path, name: str, knobs: dict[str, Any]) -> str | None:
    stage = reg.get(name)
    man = read_manifest(workdir, name)
    if man is None:
        return "no manifest"
    if man.get("status") != COMPLETE:
        return f"last run {man.get('status')}"
    if man.get("inputs") != _hashes(workdir, stage.inputs):
        return "input sha256 changed"
    if man.get("outputs") != _hashes(workdir, stage.outputs):
        return "output sha256 changed or missing"
    if man.get("knobs") != knobs:
        return "knobs changed"
    return None


def staleness(
    workdir: Path,
    name: str,
    knobs: dict[str, dict[str, Any]] | None = None,
    registry: Registry | None = None,
) -> str | None:
    """Why ``name`` is stale (own cause or an upstream stage), else None."""
    reg = registry or DEFAULT_REGISTRY
    knobs = knobs or {}
    workdir = Path(workdir)
    for n in reg.names():
        reason = _own_staleness(reg, workdir, n, knobs.get(n, {}))
        if reason is None:
            for d in reg.stages[n].deps:
                if d in _stale_cache(reg, workdir, knobs):
                    reason = f"upstream stage {d} is stale"
                    break
        if n == name:
            return reason
    raise KeyError(f"unknown stage: {name}")


def _stale_cache(reg: Registry, workdir: Path, knobs: dict[str, Any]) -> set[str]:
    stale: set[str] = set()
    for n in reg.names():
        if _own_staleness(reg, workdir, n, knobs.get(n, {})) or any(
            d in stale for d in reg.stages[n].deps
        ):
            stale.add(n)
    return stale


def stale_stages(
    workdir: Path,
    knobs: dict[str, dict[str, Any]] | None = None,
    registry: Registry | None = None,
) -> list[str]:
    reg = registry or DEFAULT_REGISTRY
    stale = _stale_cache(reg, Path(workdir), knobs or {})
    return [n for n in reg.names() if n in stale]


def run_stage(
    workdir: Path,
    name: str,
    knobs: dict[str, Any] | None = None,
    registry: Registry | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one stage unless it is fresh; return its manifest.

    The returned manifest carries ``skipped: True`` (not persisted) when the
    run was a no-op. Upstream stages are not run here; callers sequence them.
    """
    reg = registry or DEFAULT_REGISTRY
    workdir = Path(workdir)
    stage = reg.get(name)
    knobs = dict(knobs or {})
    if not force and _own_staleness(reg, workdir, name, knobs) is None:
        return {**(read_manifest(workdir, name) or {}), "skipped": True}
    inputs = _hashes(workdir, stage.inputs)
    missing = [r for r, h in inputs.items() if h is None]
    started = time.time()
    status, rc, error = COMPLETE, 0, None
    if missing:
        status, rc, error = FAILED, 2, f"missing inputs: {', '.join(missing)}"
    else:
        try:
            stage.func(workdir, knobs)
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            status, rc, error = FAILED, 1, f"{type(exc).__name__}: {exc}"
    man = {
        "stage": name,
        "jev_factory_version": __version__,
        "inputs": inputs,
        "outputs": _hashes(workdir, stage.outputs),
        "knobs": knobs,
        "status": status,
        "started": started,
        "finished": time.time(),
        "rc": rc,
    }
    if error:
        man["error"] = error
    path = manifest_path(workdir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    return man
