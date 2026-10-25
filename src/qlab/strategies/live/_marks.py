"""The price a spot leg is valued at, shared by the live drivers that hold spot
against a perpetual short (`qlab.strategies.live.frab`, `.bv2`).

The panel says which convention its spot columns follow
(`panel.meta["spot_prices"]`, `qlab.data.snapshot.SPOT_PRICES`):

- ``perp`` (the default, also for a panel that does not say): the spot column
  is the perp's price on the bars the spot printed, so the leg is valued at the
  perp's price. This is the stand's behaviour from before the option existed
  and must not change.
- ``own``: the spot column holds the spot market's own closes. The leg is
  valued at the last close the spot printed up to the bar -- a thin token has
  no print on most hours, and a position is worth its last trade until the next
  one. Never the perp's price, and never a later print: carrying forward looks
  only backward.
"""

from __future__ import annotations

import numpy as np

from qlab.harness.panel import MarketPanel


def spot_marks(panel: MarketPanel, perp: str, spot: str) -> np.ndarray:
    """Per-bar price to value `spot`'s units at (module docstring). NaN before
    the spot's first print in ``own`` mode: nothing is held there."""
    if panel.meta.get("spot_prices", "perp") != "own":
        return panel.prices[perp].to_numpy(dtype=float)
    return panel.prices[spot].ffill().to_numpy(dtype=float)


__all__ = ["spot_marks"]
