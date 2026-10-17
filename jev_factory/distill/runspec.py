"""Grouped inputs for the distill ``run_*`` stage functions.

The train, quant and final stages take many keyword-only inputs; the ones that always travel
together are grouped here as frozen dataclasses, so a stage signature stays short and the
pipeline builds each group once:

* :class:`Bootstrap` -- the bootstrap CI seed and resample count;
* :class:`GpuLimits` -- the memory cap, the memory floor, the foreign-process override and the
  timeout of a capped GPU run (:mod:`jev_factory.factory.gpu`);
* :class:`TrainerEnv` -- the external trainer venv's python and the strands-decider checkout;
* :class:`StudentSource` -- the pin cache, the verified student pins and the student profile;
* :class:`JobRoute` -- the executor a training job runs on, the job seam and the outputs the
  seam hash-verifies on return.

Each stage unpacks its groups into the same local names it used before, so manifests and
knobs record identical keys and values. Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jev_factory.core import metrics as core_metrics
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile


@dataclass(frozen=True)
class Bootstrap:
    """Seed and resample count for every bootstrap CI a stage computes."""

    seed: int = core_metrics.DEFAULT_BOOTSTRAP_SEED
    resamples: int = core_metrics.DEFAULT_BOOTSTRAP_RESAMPLES


@dataclass(frozen=True)
class GpuLimits:
    """Limits for a capped GPU run; ``memory_max`` is required once a stage applies."""

    memory_max: str | None = None
    memory_floor: str = "8G"
    allow_foreign: bool = False
    timeout: float | None = None


@dataclass(frozen=True)
class TrainerEnv:
    """The external trainer environment: its python and the strands-decider checkout."""

    python: str
    decider_repo: Path | str


@dataclass(frozen=True)
class StudentSource:
    """Where the student starts from: the pin cache, its verified pins (a
    ``pins.StudentPins``; ``None`` is refused) and the student profile."""

    cache_dir: Path | str | None
    pins: Any
    profile: StudentProfile = DEFAULT_2610


@dataclass(frozen=True)
class JobRoute:
    """How a training job runs: *executor* (``None``: the local executor), through the *jobs*
    seam (``None``: :mod:`jev_factory.distill.executor`); *expected_outputs* are hash-verified
    on return."""

    executor: Any = None
    expected_outputs: Sequence[str] = ()
    jobs: Any = None
