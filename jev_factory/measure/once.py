"""The sealed sides are measured once: an append-only ledger of final and held-out runs.

The test side (a ``--final`` run) and the sealed held-out set (an
``--acceptance`` run) are touched **once**, in the final run only. nvsh
counted earlier final pages in the report (``_count_finals``) but never
refused a second one; here a second measurement of either side without a
recorded deviation id is refused, so the rule is enforced in code rather than
by reading the report.

The ledger is keyed by ``(side, slice)``, the slice being ``full`` or
``missing-candidate`` (deviation d9): the final run measures each side in full
and its missing-candidate slice, each pair once. A record written before
slices existed has no ``slice`` and reads as ``full``.

The ledger is ``<run dir>/measure/once-ledger.jsonl``, one JSON record per
measurement of a sealed side: side, slice, the split file's sha256 (never its
text), label, date, status and the deviation id when there is one. It is
append-only. A record is written only when the run **measured** something
(some model's start-up succeeded and entries were scored). A run in which
every start-up failed, or that was refused before any model started, writes
no record, so it never blocks the real run (nvsh ledger P71, #57); a run
whose server died mid-run did score entries and is recorded.

While a sealed-side measurement runs it holds an exclusive ``flock`` on
``<run dir>/measure/.once.lock``, so two final runs can never overlap.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

NVSH_PROVENANCE = {
    "upstream": "scripts/lfm-finetune/measure.py",
    "commit": "9debdc6",
    "adaptations": [
        "FINAL_MARKER / NOT_MEASURED_MARKER / _count_finals / measured_nothing"
        " (lines 216-222, 2767-2782, 2865-2870): counting earlier final pages becomes an"
        " append-only once-ledger that refuses a second final or held-out measurement"
        " without a deviation id; a run that measured nothing is never recorded (P71)",
        "new: an exclusive flock keeps two sealed-side measurements from overlapping",
    ],
    "licence": "Apache-2.0",
}

#: The two sealed sides, each measured once.
TEST = "test"
HELD_OUT = "held-out"
SIDES = (TEST, HELD_OUT)

#: What of a side a run scored: all of it, or its missing-candidate slice.
FULL = "full"
MISSING_CANDIDATE = "missing-candidate"
SLICES = (FULL, MISSING_CANDIDATE)

#: Record statuses: the run scored entries, cleanly or not.
MEASURED = "measured"
FAILED_MID_RUN = "failed-mid-run"

LEDGER_NAME = "once-ledger.jsonl"
LOCK_NAME = ".once.lock"

_DEVIATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class OnceError(Exception):
    """A second sealed-side measurement without a deviation id, or one already running."""


@dataclass(frozen=True)
class OnceRecord:
    side: str
    split_sha256: str
    label: str
    date: str
    status: str
    deviation: str | None = None
    slice: str = FULL

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "split_sha256": self.split_sha256,
            "label": self.label,
            "date": self.date,
            "status": self.status,
            "deviation": self.deviation,
            "slice": self.slice,
        }


def check_deviation_id(deviation: str | None) -> str | None:
    """*deviation* when it is a plausible deviation id (or ``None``); ``ValueError`` otherwise."""
    if deviation is None:
        return None
    if not _DEVIATION_RE.match(deviation):
        raise ValueError(f"--deviation {deviation!r} is not a deviation id (e.g. d7)")
    return deviation


class OnceLedger:
    """The ledger of one run directory (see the module docstring)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir) / "measure"
        self.path = self.dir / LEDGER_NAME
        self.lock_path = self.dir / LOCK_NAME
        self._lock_fd: int | None = None

    def records(self) -> list[OnceRecord]:
        """Every record, oldest first; a torn last line is ignored."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        found = []
        for line in lines:
            try:
                raw = json.loads(line)
                found.append(OnceRecord(**raw))
            except (ValueError, TypeError):
                continue
        return found

    def measured(self, side: str, slice: str = FULL) -> list[OnceRecord]:  # noqa: A002
        return [r for r in self.records() if r.side == side and r.slice == slice]

    def check(self, side: str, deviation: str | None, slice: str = FULL) -> int:  # noqa: A002
        """Refuse a second measurement of (*side*, *slice*) without *deviation*.

        Returns the earlier count of that pair.
        """
        if side not in SIDES:
            raise ValueError(f"{side!r} is not a sealed side")
        if slice not in SLICES:
            raise ValueError(f"{slice!r} is not a measured slice")
        earlier = self.measured(side, slice)
        if earlier and deviation is None:
            first = earlier[0]
            raise OnceError(
                f"the {side} side ({slice} slice) of this run was already measured"
                f" ({len(earlier)} time(s);"
                f" first as {first.label!r} on {first.date}, status {first.status});"
                " it is measured once"
            )
        return len(earlier)

    def append(self, record: OnceRecord) -> None:
        """Append *record* (the ledger is never rewritten)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- the exclusive lock held for a sealed-side run --

    def __enter__(self) -> OnceLedger:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise OnceError(
                f"another sealed-side measurement of this run holds {self.lock_path}"
            ) from None
        self._lock_fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
