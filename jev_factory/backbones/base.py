"""The per-backbone adapter protocol.

The invariant contract is a bounded candidate set, then a calibrated distribution over it, then
the deterministic gate, then grounding outside the model. Everything backbone-specific lives
behind this protocol: how a request and its candidates are rendered, how a distribution is read,
how training is planned, how adapters are merged and verified, how a build is exported, and which
quantizations are worth trying. Stages that only need a distribution must not import an adapter
(``tests/test_backbone_seam.py``).

The distillery reaches its student only through this protocol and the registry
(``registry.get(profile.backbone)``; deviation d2, ``tests/test_distill_backbone_registry.py``):
the bundle layout and its file names, the serve command and serving helpers, the distill
trainer's plan and run, and the deployed-format builder. The facet hooks (:meth:`layout`,
:meth:`server`, :meth:`trainer`, :meth:`quantizer`) return the family's own module for that
concern; a family with no distill path raises ``NotImplementedError``.

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

    # ---- the distillery's hooks (deviation d2) -------------------------------------------

    def layout(self) -> Any:
        """The bundle-layout facet: reads, writes and names the family's bundle files."""
        ...

    def config_files(self) -> Mapping[str, str]:
        """File names: ``served_config``, ``manifest``, ``provenance`` and ``train_config``."""
        ...

    def serve_argv(
        self, python: str, bundle: Any, *, port: int, host: str = "127.0.0.1"
    ) -> list[str]:
        """The command that serves a verified local *bundle* on *host*:*port*."""
        ...

    def server(self) -> Any:
        """The serving facet: launch, readiness, job status and the wire client."""
        ...

    def plan_distill_train(self, **kwargs: Any) -> Mapping[str, Any]:
        """The distill trainer's dry-run plan (its subprocess command and hyperparameters)."""
        ...

    def run_distill_train(self, plan: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        """Run (or dry-run) a :meth:`plan_distill_train` plan through the caller's job seam."""
        ...

    def trainer(self) -> Any:
        """The distill-trainer facet: knob names, checkpoint outputs and heal helpers."""
        ...

    def quantizer(self) -> Any:
        """The deployed-format facet: the weight budget, builds, attestation and artifacts."""
        ...
