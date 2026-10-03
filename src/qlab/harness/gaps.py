"""Bars a position is held through without being re-marked (docs/TASKS.md T36).

A bar with no price is a bar on which nobody traded, or none was recorded:
one cannot ENTER there, and the data layer keeps it untradeable. But a
position opened earlier does not vanish because an hour went by without a
trade -- it simply cannot be re-marked until the next print. Treating such a
bar as "must be flat" forced every book out of the market for it, and the
stand's complete-book window broke on it: one hour without spot trades on
Binance (BTC and ETH, 2023-03-24 12:00 UTC, a halt of spot trading) cut
Bv2's judged history from 2019-09 to 2023-03 and with it the bull market of
2020-2021 and the bear market of 2022.

A bar is a HOLDABLE GAP for an instrument when:

- it has no price, and
- the instrument has a price on some earlier bar and on some later bar of the
  panel -- before its listing and after its delisting nothing is held
  through, the position ends with the last tradeable bar as before; and
- its funding there is known, unless the instrument structurally has none
  (a spot market): an unknown funding rate is never a zero one
  (`qlab.harness.accrual`), so a perp with an unknown settlement is still
  not held through.

On a holdable gap the harness holds the previous bar's weights, whatever the
strategy says (it may not trade there), and marks the position at the last
known price; the move across the gap is realised at the first bar with a
price again. That is what a holder experiences, and what frab's live engines
do when a price is missing: they skip the tick and keep their positions.

Bars with a price that a POLICY makes untradeable (the liquidity filter, a
universe rule) are not gaps: there the strategy is meant to be flat and may
exit, and holding would override it.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def holdable_gaps(prices: pd.DataFrame, funding: pd.DataFrame,
                  no_funding: Iterable[str] = ()) -> pd.DataFrame:
    """Boolean frame shaped like `prices`: True on a holdable gap (module
    docstring)."""
    priced = prices.notna()
    seen_before = priced.cummax()
    seen_after = priced.iloc[::-1].cummax().iloc[::-1]
    funding_known = funding.reindex_like(prices).notna()
    structural = [c for c in no_funding if c in prices.columns]
    if structural:
        funding_known[structural] = True
    return ~priced & seen_before & seen_after & funding_known


def held_through(weights: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """`weights` with every holdable-gap bar replaced by the weight held
    before the gap (several gap bars in a row hold the same weight)."""
    gaps = gaps.reindex_like(weights).fillna(False).astype(bool)
    if not gaps.to_numpy().any():
        return weights
    return weights.where(~gaps, weights.where(~gaps).ffill().fillna(0.0))


def marked_through(prices: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """`prices` with every holdable-gap bar marked at the last known price."""
    gaps = gaps.reindex_like(prices).fillna(False).astype(bool)
    if not gaps.to_numpy().any():
        return prices
    return prices.where(~gaps, prices.ffill())


__all__ = ["held_through", "holdable_gaps", "marked_through"]
