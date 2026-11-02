"""Signal blocks: one score per coin per bar. A higher score is more wanted long.

All windows are in calendar days and use only data up to and including the bar.
A score is NaN while its window is not fully known, and such a coin is not
traded.

- `momentum`: the price change over the last `lookback_days` days (what rose keeps rising).
- `reversal`: minus that change (what rose falls back).
- `funding`: minus the average funding rate over the last `window_days` days. A coin
  with high funding gets a low score and is shorted, because the short side
  receives the funding.
- `funding_tail`: where the current funding stands among the coin's own previous
  `history_window_days` days (the current bar excluded; ties count half). +1 in the low tail
  (position <= `tail_fraction`), -1 in the high tail, 0 between. The sign follows the module
  rule (low funding is wanted long); the exit is the first bar back inside the tails.
- `low_vol`: minus the standard deviation of bar returns over the last
  `window_days` days (calm coins score higher).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block, fraction_up_to, positive_number
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


def _funding_position(panel: MarketPanel, *, history_window_days: float) -> pd.DataFrame:
    """Where the current funding stands among the coin's previous `history_window_days`
    (the current bar excluded): (below + 0.5 * equal) / N, in [0, 1]. Ties count half, so a
    rate that has not moved sits at 0.5 and is not unusual. NaN unless the bar and all N
    previous bars are known."""
    window = bars(panel.prices.index, history_window_days)

    def position(values: np.ndarray) -> float:
        past, current = values[:-1], values[-1]
        return float((past < current).sum() + 0.5 * (past == current).sum()) / len(past)

    return panel.funding.rolling(window + 1, min_periods=window + 1).apply(position, raw=True)


def funding_tail(panel: MarketPanel, *, history_window_days: float,
                 tail_fraction: float) -> pd.DataFrame:
    """+1 where the position is <= `tail_fraction` (unusually low funding), -1 where it is
    >= 1 - `tail_fraction`, 0 between, NaN where the position is unknown."""
    position = _funding_position(panel, history_window_days=history_window_days)
    low = position <= tail_fraction + 1e-9
    high = position >= 1.0 - tail_fraction - 1e-9
    return (low.astype(float) - high.astype(float)).where(position.notna())


def low_vol(panel: MarketPanel, *, window_days: float) -> pd.DataFrame:
    """Minus the population std (ddof=0, as in trend) of bar returns over `window_days`."""
    window = bars(panel.prices.index, window_days)
    return -bar_returns(panel.prices).rolling(window, min_periods=window).std(ddof=0)


SIGNALS: dict[str, Block] = {
    "momentum": Block(momentum, {"lookback_days": positive_number}),
    "reversal": Block(reversal, {"lookback_days": positive_number}),
    "funding": Block(funding, {"window_days": positive_number}),
    "funding_tail": Block(funding_tail, {"history_window_days": positive_number,
                                         "tail_fraction": fraction_up_to(0.5)}),
    "low_vol": Block(low_vol, {"window_days": positive_number}),
}
