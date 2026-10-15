"""The per-backbone adapter protocol.

The invariant contract is a bounded candidate set, then a calibrated distribution over it, then
the deterministic gate, then grounding outside the model. Everything backbone-specific lives
behind this protocol: how a request and its candidates are rendered, how a distribution is read,
how training is planned, how adapters are merged and verified, how a build is exported, and which
quantizations are worth trying. Stages that only need a distribution must not import an adapter
(``tests/test_backbone_seam.py``).

Pure stdlib: adapters import their heavy dependencies lazily.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Backbone(Protocol):
    """What a model family supplies to the factory."""

    #: Registry name, e.g. ``causal_lm`` or ``pointer``.
    name: str

    def render(self, domain: Any, request: str, offered: Sequence[str]) -> Any:
        """Render one request and its offered candidates into the family's model input."""
        ...

    def predict_distributions(
        self, model: Any, rendered: Sequence[Any], offered: Sequence[Sequence[str]]
    ) -> list[Mapping[str, float]]:
        """Raw probability over each row's offered candidates (before calibration)."""
        ...

    def train_plan(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """A dry-run description of the training this family would perform."""
        ...

    def merge_verify(self, adapter_dir: str, base: str) -> Mapping[str, Any]:
        """Merge trained adapters into the base and verify the result; report what was checked."""
        ...

    def export(self, model_dir: str, out_dir: str, *, apply: bool = False) -> Mapping[str, Any]:
        """Export a deployable build (dry-run unless *apply*)."""
        ...

    def quantize_candidates(self, config: Mapping[str, Any]) -> Sequence[str]:
        """The quantizations worth measuring for this family, in preference order."""
        ...
