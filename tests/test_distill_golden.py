"""Byte-identity of the ``jev distill`` stage manifests against main's golden files (t1).

``tests/fixtures/distill_golden/`` holds the pre-registration file and the 15 stage manifests
the e2e fakes (``tests/test_distill_e2e.py``) write at e3345d1, normalised only as listed in
``tests/distill_golden_support.py``. Later changes must leave them byte-identical under the
default config. To refresh after an intended change, run
``JEV_UPDATE_GOLDEN=1 uv run pytest tests/test_distill_golden.py`` and record a deviation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.distill_golden_support import generate
from tests.test_distill_e2e import STAGES, env  # noqa: F401  (env: the e2e fixture)

GOLDEN = Path(__file__).parent / "fixtures" / "distill_golden"


def test_golden_files_cover_prereg_and_every_stage():
    names = {p.relative_to(GOLDEN).as_posix() for p in GOLDEN.rglob("*.json")}
    assert names == {"prereg.json", *(f"manifests/{s}.json" for s in STAGES)}


def test_e2e_manifests_are_byte_identical_to_the_golden_files(env):  # noqa: F811
    produced = generate(env)
    if os.environ.get("JEV_UPDATE_GOLDEN"):
        for name, text in produced.items():
            target = GOLDEN / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        pytest.skip("golden files rewritten")
    differing = [
        name
        for name, text in produced.items()
        if (GOLDEN / name).read_bytes() != text.encode("utf-8")
    ]
    assert not differing, f"golden files differ: {differing}"
