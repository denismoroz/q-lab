"""Regime detectors a STRATEGY may trade on (owner, 2026-10-03: «давай
разделим 2 задачи — использование в стратегии и тестировании стратегии»).

Different object from `qlab.regimes`, which describes the market for testing
and may know the whole history. A detector here:

- uses only data before the bar it labels (tested against tampered futures);
- is a parameter of the strategy that uses it, declared and cited like any
  other (`qlab.pipeline.sources`);
- is judged twice: by how well it predicts the coming market on BTC alone,
  with no strategy involved, and by what it does for the strategy, through
  the ordinary stand (selection period, forward test, noise).

Measured 2026-10-03 on BTC 2019-2026 (docs/REGIMES.md): the direction of the
last 7-90 days predicts the next 7 or 30 days' direction regime no better than
chance (32-37% against 33%); the level of volatility predicts the next 30
days' volatility regime at 44-46%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.regimes import WINDOW_DAYS, market_closes


def direction_terciles(closes: pd.Series, window: int = WINDOW_DAYS) -> pd.Series:
    """Labels a strategy may TRADE on: the same terciles, but each day's
    thresholds come only from the trailing returns of the days before it
    (an expanding window). Unknown until `window` trailing returns exist
    before the day. `label_days` is the hindsight version, for description."""
    trailing = closes / closes.shift(window) - 1.0
    past = trailing.shift(1)
    low = past.expanding(min_periods=window).quantile(1 / 3)
    high = past.expanding(min_periods=window).quantile(2 / 3)
    labels = pd.Series(np.where(trailing >= high, "bull",
                                np.where(trailing < low, "bear", "flat")),
                       index=closes.index, dtype=object)
    return labels.where(trailing.notna() & low.notna())



causal_labels = direction_terciles  # name used before the split (2026-10-02)

__all__ = ["causal_labels", "direction_terciles", "market_closes"]
