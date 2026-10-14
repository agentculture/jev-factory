"""Backbone registry: names mapped to lazily imported adapters.

Registering a backbone records only where it lives; nothing is imported until :func:`get`.
"""

from __future__ import annotations

import importlib
from typing import Any

#: name -> (module, attribute). The attribute is the adapter object or class.
_LOADERS: dict[str, tuple[str, str]] = {
    "causal_lm": ("jev_factory.backbones.causal_lm", "BACKBONE"),
    "pointer": ("jev_factory.backbones.pointer", "BACKBONE"),
}


def names() -> tuple[str, ...]:
    """Registered backbone names."""
    return tuple(sorted(_LOADERS))


def get(name: str) -> Any:
    """Import and return the adapter registered as *name*."""
    try:
        module, attr = _LOADERS[name]
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; registered: {', '.join(names())}") from None
    return getattr(importlib.import_module(module), attr)
