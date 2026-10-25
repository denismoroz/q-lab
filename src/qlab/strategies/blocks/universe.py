"""Universe blocks: which coins may be considered at each bar.

A universe block returns a boolean table (bars x coins): True means the coin is
allowed at that bar. It may use only data up to and including that bar.

- `all`: every coin is allowed.
- `top_volume`: the `n` coins with the largest trailing average dollar volume
  (volume times price, averaged over the last `window_days` days). Only coins
  that are tradeable, priced and have a fully known window at that bar compete
  for the places; ties go to the coin listed first in the panel.
"""

from __future__ import annotations

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block, positive_int, positive_number
from qlab.strategies.blocks._util import bars


def all_coins(panel: MarketPanel) -> pd.DataFrame:
    """Every coin is in the universe at every bar."""
    return pd.DataFrame(True, index=panel.prices.index, columns=panel.prices.columns)


def top_volume(panel: MarketPanel, *, n: int, window_days: float) -> pd.DataFrame:
    """The `n` coins with the highest trailing mean dollar volume at each bar."""
    window = bars(panel.prices.index, window_days)
    dollar_volume = panel.volume * panel.prices
    average = dollar_volume.rolling(window, min_periods=window).mean()
    competing = panel.tradeable.astype(bool) & panel.prices.notna() & average.notna()
    rank = average.where(competing).rank(axis=1, method="first", ascending=False)
    return rank.le(n)


UNIVERSES: dict[str, Block] = {
    "all": Block(all_coins, {}),
    "top_volume": Block(top_volume, {"n": positive_int, "window_days": positive_number}),
}
