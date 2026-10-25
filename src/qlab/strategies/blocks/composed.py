"""`Composed`: a strategy assembled from a spec of ready blocks.

The spec names a universe, a signal, a portfolio rule and a sizing rule, plus
how often to rebalance and how much price history a coin needs. See
`docs/BLOCKS.md` for the meaning of each block.

Semantics fixed here (the blocks themselves are in the sibling modules):

- Only perpetual columns trade; any column ending with `SPOT_COLUMN_SUFFIX`
  gets weight 0 always.
- A coin is eligible at bar t when it is tradeable at t, its price is known at
  t, it has at least `min_history_days` (converted to bars with `periods_for`)
  non-NaN price bars up to and including t, it is in the universe at t, and its
  signal score at t is not NaN.
- Weights are decided only on rebalance rows: row 0, then each row at least
  `rebalance_days` (converted to bars) after the previous rebalance row. They
  are carried forward in between, with cells that are not tradeable at a bar
  set to 0. Before row 0 decides anything the book is flat.
- Every block reads data up to and including the bar only, so the stand's
  look-ahead guard holds by construction.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import (
    non_negative_number,
    parse_block,
    positive_number,
)
from qlab.strategies.blocks._util import bars
from qlab.strategies.blocks.portfolio import PORTFOLIOS
from qlab.strategies.blocks.signals import SIGNALS
from qlab.strategies.blocks.sizing import SIZINGS
from qlab.strategies.blocks.universe import UNIVERSES

REQUIRED_KEYS = (
    "universe",
    "signal",
    "portfolio",
    "sizing",
    "rebalance_days",
    "min_history_days",
)


def rebalance_rows(index: pd.DatetimeIndex, rebalance_days: float) -> np.ndarray:
    """Boolean mask of rebalance rows: row 0, then every `rebalance_days` of bars."""
    every = bars(index, rebalance_days)
    mask = np.zeros(len(index), dtype=bool)
    last: int | None = None
    for i in range(len(index)):
        if last is None or i - last >= every:
            mask[i] = True
            last = i
    return mask


def _perp_panel(panel: MarketPanel, columns: list) -> MarketPanel:
    return MarketPanel(
        snapshot_id=panel.snapshot_id,
        prices=panel.prices[columns],
        funding=panel.funding[columns],
        tradeable=panel.tradeable[columns],
        meta=panel.meta,
        volume=panel.volume[columns],
        high=panel.high[columns],
    )


class Composed:
    """A strategy built from universe / signal / portfolio / sizing blocks.

    Required `params` (no defaults; a missing or unknown key raises ValueError):

        universe: {rule: all} | {rule: top_volume, n, window_days}
        signal: {block: momentum | reversal, lookback_days}
                | {block: funding | low_vol, window_days}
        portfolio: {block: long_short | long_only, top_n}
                   | {block: time_series_sign}
        sizing: {block: equal, gross}
                | {block: vol_target, window_days, target_daily, leverage_cap}
        rebalance_days: float
        min_history_days: float
    """

    name = "composed"
    # Every duration is in calendar days and volatility is put in daily units,
    # so the same params mean the same on these bars (checked in test_composed).
    valid_intervals = ("1d", "4h", "1h")

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [key for key in REQUIRED_KEYS if key not in params]
        if missing:
            raise ValueError(f"composed: missing param(s) {missing}")
        unknown = sorted(set(params) - set(REQUIRED_KEYS), key=str)
        if unknown:
            raise ValueError(f"composed: unknown param(s) {unknown}")

        universe_fn, universe_kw = parse_block("universe", params["universe"], UNIVERSES, "rule")
        signal_fn, signal_kw = parse_block("signal", params["signal"], SIGNALS, "block")
        portfolio_fn, portfolio_kw = parse_block(
            "portfolio", params["portfolio"], PORTFOLIOS, "block"
        )
        sizing_fn, sizing_kw = parse_block("sizing", params["sizing"], SIZINGS, "block")
        rebalance_days = positive_number("rebalance_days", params["rebalance_days"])
        min_history_days = non_negative_number("min_history_days", params["min_history_days"])

        all_columns = panel.prices.columns
        index = panel.prices.index
        perp_columns = [c for c in all_columns if not str(c).endswith(SPOT_COLUMN_SUFFIX)]
        if not perp_columns:
            return pd.DataFrame(0.0, index=index, columns=all_columns)
        perp = _perp_panel(panel, perp_columns)

        scores = signal_fn(perp, **signal_kw).replace([np.inf, -np.inf], np.nan)
        history_ok = perp.prices.notna().cumsum() >= bars(index, min_history_days)
        eligible = (
            perp.tradeable.astype(bool)
            & perp.prices.notna()
            & history_ok
            & universe_fn(perp, **universe_kw).astype(bool)
            & scores.notna()
        )

        raw = portfolio_fn(scores, eligible, **portfolio_kw)
        sized = sizing_fn(raw, perp, **sizing_kw)

        undecided = np.flatnonzero(~rebalance_rows(index, rebalance_days))
        carried = sized.copy()
        carried.iloc[undecided] = np.nan
        carried = carried.ffill().fillna(0.0)
        carried = carried.where(perp.tradeable.astype(bool), 0.0)

        weights = pd.DataFrame(0.0, index=index, columns=all_columns)
        weights[perp_columns] = carried
        return weights
