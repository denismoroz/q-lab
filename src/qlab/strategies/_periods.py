"""Shared helper: convert a calendar duration into a number of panel rows.

Every lookback/threshold this package transcribes from research is stated in
calendar time (e.g. "14 days", "12 hours"), but `MarketPanel` makes no
promise about bar spacing -- the same strategy may run over an hourly or a
daily panel. `periods_for` converts a `pd.Timedelta` into an integer row
count using the panel's OWN median timestamp spacing -- exactly the
principle `qlab.harness.metrics.periods_per_year` already uses to infer
annualisation from the actual data instead of a hardcoded 365. This is not a
parameter choice: it is how many rows the documented calendar duration
corresponds to on THIS panel.
"""

from __future__ import annotations

import pandas as pd


def periods_for(index: pd.DatetimeIndex, duration: pd.Timedelta) -> int:
    """Number of panel rows spanning `duration`, from the index's median row spacing.

    Rounds to the nearest integer number of rows and floors at 1 -- a
    documented duration shorter than one panel bar still costs at least one
    bar (there is no such thing as "zero bars of patience").

    Raises:
        ValueError: fewer than 2 timestamps (no spacing to infer), or a
            non-positive median spacing (duplicate/unsorted index).
    """
    if len(index) < 2:
        raise ValueError("cannot infer row spacing from fewer than 2 timestamps")
    deltas = index.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(0):
        raise ValueError(f"non-positive median spacing between timestamps: {median_delta}")
    periods = round(duration / median_delta)
    return max(1, int(periods))
