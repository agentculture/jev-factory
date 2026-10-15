"""Grouped inputs for the distill ``run_*`` stage functions.

The train, quant and final stages take many keyword-only inputs; the ones that always travel
together are grouped here as frozen dataclasses, so a stage signature stays short and the
pipeline builds each group once:

* :class:`Bootstrap` -- the bootstrap CI seed and resample count;
* :class:`GpuLimits` -- the memory cap, the memory floor, the foreign-process override and the
  timeout of a capped GPU run (:mod:`jev_factory.factory.gpu`);
* :class:`TrainerEnv` -- the external trainer venv's python and the strands-decider checkout.

Each stage unpacks its groups into the same local names it used before, so manifests and
knobs record identical keys and values. Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jev_factory.core import metrics as core_metrics


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
