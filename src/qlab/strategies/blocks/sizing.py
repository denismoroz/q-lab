"""Sizing blocks: scale the raw portfolio weights.

- `equal`: multiply every weight by `gross` (gross 1 keeps the portfolio as it is).
- `vol_target`: size each coin so it moves about `target_daily` per day:
  multiply its weight by `target_daily` divided by its daily volatility (the
  std of bar returns over `window_days`, put in daily units by the square root of
  bars per day, as trend does). A coin with unknown or zero volatility gets 0.
  Then, if the total of absolute weights exceeds `leverage_cap`, all weights are
  scaled down together so the total equals `leverage_cap`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block, positive_number
from qlab.strategies.blocks._util import bar_returns, bars


def equal(weights: pd.DataFrame, panel: MarketPanel, *, gross: float) -> pd.DataFrame:
    """Scale all weights by `gross`."""
    return weights * gross


def vol_target(
    weights: pd.DataFrame,
    panel: MarketPanel,
    *,
    window_days: float,
    target_daily: float,
    leverage_cap: float,
) -> pd.DataFrame:
    """Inverse-volatility sizing to `target_daily`, then a gross cap."""
    index = panel.prices.index
    window = bars(index, window_days)
    vol = bar_returns(panel.prices).rolling(window, min_periods=window).std(ddof=0)
    bars_per_day = pd.Timedelta(days=1) / index.to_series().diff().median()
    vol = vol * np.sqrt(bars_per_day)

    sized = weights * target_daily / vol
    sized = sized.where(vol > 0, 0.0).fillna(0.0)

    gross = sized.abs().sum(axis=1)
    scale = pd.Series(1.0, index=index)
    over = gross > leverage_cap
    scale[over] = leverage_cap / gross[over]
    return sized.mul(scale, axis=0)


SIZINGS: dict[str, Block] = {
    "equal": Block(equal, {"gross": positive_number}),
    "vol_target": Block(
        vol_target,
        {
            "window_days": positive_number,
            "target_daily": positive_number,
            "leverage_cap": positive_number,
        },
    ),
}
