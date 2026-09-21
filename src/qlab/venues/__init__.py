"""Venue capability registry (docs/TASKS.md, task T13).

`qlab evaluate`'s preflight stage (rules/2026-09-20.1.yaml) needs three
metrics no backtest can derive — `venue_supported`, `data_forward_available`,
`atomic_execution` — because they are facts about infrastructure, not about
the market. This package supplies them from a small, cited, per-venue
config (`venues/<id>.yaml` at the repo root, see `qlab.venues.config`) via
pure derivation functions (`qlab.venues.derive`) that `qlab.pipeline.evaluate`
calls. A venue with no config file, or a field left unset in it, yields an
absent metric — never a guessed one.
"""

from __future__ import annotations

from qlab.venues.config import VenueConfig, load_venue
from qlab.venues.derive import (
    atomic_execution,
    data_forward_available,
    derive_venue_metrics,
    venue_supported,
)

__all__ = [
    "VenueConfig",
    "atomic_execution",
    "data_forward_available",
    "derive_venue_metrics",
    "load_venue",
    "venue_supported",
]
