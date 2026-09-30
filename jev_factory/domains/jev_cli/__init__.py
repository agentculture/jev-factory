"""The jev-CLI domain: a jev-tool model's candidates are the ``jev`` CLI's own verbs.

``DOMAIN`` is generated from the argparse tree on first access (see
:mod:`jev_factory.domains.jev_cli.generate`), so ``load_domain
("jev_factory.domains.jev_cli")`` and ``jev init jev_factory.domains.jev_cli``
work like any other domain module without importing the CLI at package import.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "DOMAIN":
        from jev_factory.domains.jev_cli.generate import generate_domain

        domain = generate_domain()
        globals()["DOMAIN"] = domain
        return domain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
