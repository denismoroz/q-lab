"""Signal blocks: one score per coin per bar. A higher score is more wanted long.

All windows are in calendar days and use only data up to and including the bar.
A score is NaN while its window is not fully known, and such a coin is not
traded.

- `momentum`: the price change over the last `lookback_days` days (what rose keeps rising).
- `reversal`: minus that change (what rose falls back).
- `funding`: minus the average funding rate over the last `window_days` days. A coin
  with high funding gets a low score and is shorted, because the short side
  receives the funding.
- `low_vol`: minus the standard deviation of bar returns over the last
  `window_days` days (calm coins score higher).
"""

from __future__ import annotations

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block, positive_number
from qlab.strategies.blocks._util import bar_returns, bars


def momentum(panel: MarketPanel, *, lookback_days: float) -> pd.DataFrame:
    """Trailing return over `lookback_days`."""
    lookback = bars(panel.prices.index, lookback_days)
    return panel.prices / panel.prices.shift(lookback) - 1.0


def reversal(panel: MarketPanel, *, lookback_days: float) -> pd.DataFrame:
    """Minus the trailing return over `lookback_days`."""
    return -momentum(panel, lookback_days=lookback_days)


def funding(panel: MarketPanel, *, window_days: float) -> pd.DataFrame:
    """Minus the trailing mean funding over `window_days` (all bars must be known)."""
    window = bars(panel.prices.index, window_days)
    return -panel.funding.rolling(window, min_periods=window).mean()


def low_vol(panel: MarketPanel, *, window_days: float) -> pd.DataFrame:
    """Minus the population std (ddof=0, as in trend) of bar returns over `window_days`."""
    window = bars(panel.prices.index, window_days)
    return -bar_returns(panel.prices).rolling(window, min_periods=window).std(ddof=0)


SIGNALS: dict[str, Block] = {
    "momentum": Block(momentum, {"lookback_days": positive_number}),
    "reversal": Block(reversal, {"lookback_days": positive_number}),
    "funding": Block(funding, {"window_days": positive_number}),
    "low_vol": Block(low_vol, {"window_days": positive_number}),
}
