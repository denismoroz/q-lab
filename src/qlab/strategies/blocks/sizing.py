"""Sizing blocks: scale the raw portfolio weights.

- `equal`: multiply every weight by `gross` (gross 1 keeps the portfolio as it is).
- `vol_target`: size each coin so it moves about `target_daily` per day:
  multiply its weight by `target_daily` divided by its daily volatility (the
  std of bar returns over `window_days`, put in daily units by the square root of
  bars per day, as trend does). A coin with unknown or zero volatility gets 0.
  Then, if the total of absolute weights exceeds `leverage_cap`, all weights are
  scaled down together so the total equals `leverage_cap`.
- `cap_weight`: weight the coins of each side by market capitalisation instead of
  equally, keeping each side's sum: a coin's weight becomes
  `side_sum * cap / (sum of caps of that side's coins)`, then everything is multiplied
  by `gross`. The caps are the latest CoinMarketCap weekly snapshot known at the bar
  (a snapshot dated D is visible from the first bar after D; see `market_caps`). A held
  coin whose cap is unknown (outside the snapshot's top 200, or before the first
  snapshot) gets 0 and the rest of its side takes its share; if a side that held coins
  has no coin with a known cap, the whole row is flat (a long/short book is never left
  one-sided). With no cap data at all a `ValueError` says what to run. A spec requests
  caps just by naming this block: `sizing: {block: cap_weight, gross: 1.0}` (it needs
  the snapshots stored by `qlab data coin-attributes`; nothing else in the spec).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.data.sources import coinmarketcap
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


def market_caps(panel: MarketPanel) -> pd.DataFrame:
    """Market capitalisation (USD) per bar and column, as known AT that bar; NaN = unknown.

    Source, in order: `panel.meta["market_cap"]` when a caller attached a frame (it must
    already be point-in-time, i.e. what was known at each bar); otherwise the stored
    CoinMarketCap weekly snapshots (`coinmarketcap.DEFAULT_STORE`, filled by
    `qlab data coin-attributes --start <date>`), mapped to the panel's columns and made
    visible from the first bar after each snapshot date by `coinmarketcap.as_of_frame`.
    Raises ValueError when neither exists: an unknown cap is never replaced by a guess.
    """
    index, columns = panel.prices.index, panel.prices.columns
    attached = panel.meta.get("market_cap")
    if attached is not None:
        if not isinstance(attached, pd.DataFrame):
            raise ValueError("panel.meta['market_cap'] must be a DataFrame of bars x columns")
        return attached.reindex(index=index, columns=columns).astype(float)
    history = coinmarketcap.load_history(coinmarketcap.DEFAULT_STORE)
    if history.empty:
        raise ValueError(
            "cap_weight needs market caps and the panel carries none (panel.meta['market_cap']) "
            "and no CoinMarketCap snapshots are stored: the spec requests caps by using "
            "sizing 'cap_weight'; fetch the snapshots with "
            "`qlab data coin-attributes --start <date>`"
        )
    return coinmarketcap.as_of_frame(history, index, list(columns), "market_cap_usd")


def cap_weight(weights: pd.DataFrame, panel: MarketPanel, *, gross: float) -> pd.DataFrame:
    """Re-weight each side of the book by market cap, keep each side's sum, scale by `gross`."""
    cap = market_caps(panel)
    cap = cap.where(cap > 0)  # a zero or negative cap is as unknown as a missing one
    out = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    flat = pd.Series(False, index=weights.index)
    for sign, side in ((1.0, weights.clip(lower=0.0)), (-1.0, (-weights).clip(lower=0.0))):
        total = side.sum(axis=1)
        held_cap = cap.where(side > 0)
        cap_sum = held_cap.sum(axis=1, min_count=1)
        out += sign * held_cap.div(cap_sum, axis=0).mul(total, axis=0).fillna(0.0)
        flat |= (total > 0) & cap_sum.isna()
    return out.mask(flat, 0.0) * gross


SIZINGS: dict[str, Block] = {
    "equal": Block(equal, {"gross": positive_number}),
    "cap_weight": Block(cap_weight, {"gross": positive_number}),
    "vol_target": Block(
        vol_target,
        {
            "window_days": positive_number,
            "target_daily": positive_number,
            "leverage_cap": positive_number,
        },
    ),
}
