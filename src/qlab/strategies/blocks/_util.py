"""Small helpers shared by the blocks: calendar days to bars, bar returns."""

from __future__ import annotations

import pandas as pd

from qlab.strategies._periods import periods_for


def bars(index: pd.DatetimeIndex, days: float) -> int:
    """Number of panel bars spanning `days` calendar days (at least 1)."""
    return periods_for(index, pd.Timedelta(days=days))


def bar_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple return of each bar; NaN wherever either price is unknown."""
    return prices.pct_change(fill_method=None)
