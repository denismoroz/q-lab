"""Overlays: pieces that change WHEN and HOW MUCH any strategy holds (docs/OVERLAYS.md).

Importing this package registers the overlays that ship with q-lab; the wrapper is
`qlab.strategies.overlays:Overlaid`. A new overlay is one new module that calls `register`.
"""

from __future__ import annotations

from qlab.strategies.overlays import book, position  # noqa: F401  (registration)
from qlab.strategies.overlays.core import (
    BOOK,
    POSITION,
    BookOverlay,
    Context,
    Overlaid,
    Overlay,
    PositionOverlay,
    chain_warmup_days,
    get,
    names,
    parse_chain,
    register,
)

__all__ = ["BOOK", "POSITION", "BookOverlay", "Context", "Overlaid", "Overlay", "PositionOverlay",
           "chain_warmup_days", "get", "names", "parse_chain", "register"]
