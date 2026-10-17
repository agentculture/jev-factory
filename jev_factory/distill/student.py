"""The distillery's one way to its student backbone: ``registry.get(profile.backbone)``.

No distill module imports a backbone package directly (deviation d2,
``tests/test_distill_backbone_registry.py``); each asks this helper for the backbone its
profile names and uses the :class:`~jev_factory.backbones.base.Backbone` hooks. Modules with
no profile in scope use the built-in :data:`~jev_factory.distill.profile.DEFAULT_2610`.
"""

from __future__ import annotations

from typing import Any

from jev_factory.backbones import registry
from jev_factory.distill.profile import DEFAULT_2610, StudentProfile


def backbone(profile: StudentProfile | None = None) -> Any:
    """The registered backbone *profile* names (default: the built-in 2610 profile's)."""
    return registry.get((profile or DEFAULT_2610).backbone)
