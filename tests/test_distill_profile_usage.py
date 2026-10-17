"""No distill module reads a student-specific pin constant outside the allow-list (t11).

The student pins live in the profile (``profile.py``). Reading ``BASE_REPO``, ``DECIDER_REPO``,
``CARD_*`` or ``STOCK_LORA_TARGETS`` directly bypasses a run's own profile.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jev_factory.distill as distill_pkg

DISTILL = Path(distill_pkg.__file__).parent
FORBIDDEN_EXACT = {"BASE_REPO", "DECIDER_REPO", "STOCK_LORA_TARGETS"}
FORBIDDEN_PREFIX = "CARD_"

#: Modules that define or alias the constants.
DEFINING = {"pins.py", "profile.py", "prereg.py", "benchmark.py"}
#: Modules that still read them directly. t12 (the pipeline takes the profile) emptied it; it
#: never grows.
KNOWN_PENDING: set[str] = set()


def _forbidden(name: str) -> bool:
    return name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIX)


def _reads(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and _forbidden(node.attr):
            out.append((node.lineno, node.attr))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if _forbidden(node.id):
                out.append((node.lineno, node.id))
        elif isinstance(node, ast.ImportFrom):
            out += [(node.lineno, a.name) for a in node.names if _forbidden(a.name)]
    return out


def test_no_distill_module_reads_the_pins_directly():
    offenders = {}
    for path in sorted(DISTILL.glob("*.py")):
        if path.name in DEFINING or path.name in KNOWN_PENDING:
            continue
        hits = _reads(path)
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def test_known_pending_is_empty_and_the_pipeline_reads_no_pin_constant():
    assert KNOWN_PENDING == set()
    assert _reads(DISTILL / "pipeline.py") == []
