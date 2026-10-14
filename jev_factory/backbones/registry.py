"""Backbone registry: names mapped to lazily imported adapters.

Registering a backbone records only where it lives; nothing is imported until :func:`get`.
"""

from __future__ import annotations

import importlib
from typing import Any


class BackboneUnavailable(LookupError):
    """A registered backbone's module is not present in this tree."""


#: name -> (module, attribute). The attribute is the adapter object or class.
_LOADERS: dict[str, tuple[str, str]] = {
    "causal_lm": ("jev_factory.backbones.causal_lm_adapter", "BACKBONE"),
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
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            raise BackboneUnavailable(
                f"backbone {name!r} is registered but its module {module!r} is missing"
            ) from exc
        raise
    return getattr(mod, attr)
