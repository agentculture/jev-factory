"""Record-id digests: how protected record ids are named without being written down."""

from __future__ import annotations

import hashlib
from typing import Any


def id_digest(record_id: Any) -> str:
    """The sha256 of a record id: how a heal spec names its protected ids."""
    return hashlib.sha256(str(record_id).encode("utf-8")).hexdigest()
