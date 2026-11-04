"""Which coins a "top K by traded value" rule can ever pick (docs/COIN_SCOPE.md).

A spec whose coins are chosen by code (`params.universe`, rule `top_liquid`,
`qlab.strategies.live._bv2_mechanisms`) must load the data of every coin that was ever among the
first K, or the rule would be choosing among a list somebody wrote with the future in mind. This
module makes that list from the venue's daily traded value, with the same definition as
`qlab.knowable_coins`: the mean daily traded value (close * volume) of the previous `window_days`
days, a full window (a coin listed inside it does not compete), ranked at each day without looking
at the day itself. Pure functions over a table; nothing is read or written here.
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd


def trailing_rank(volume_usd: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Each day's place (1 = largest) of every coin by the mean daily traded value of the previous
    `window_days` days; NaN where the coin has no full window. The day itself is not read."""
    value = volume_usd.copy()
    value.columns = [str(c).removesuffix("-SPOT").upper() for c in value.columns]
    value = value.loc[:, ~value.columns.duplicated()]
    full = pd.date_range(value.index.min(), value.index.max(), freq="D")
    trailing = (value.reindex(full).rolling(window_days, min_periods=window_days).mean()
                .shift(1))
    return trailing.rank(axis=1, ascending=False, method="min")


def top_k(volume_usd: pd.DataFrame, k: int, window_days: int = 30,
          among: Collection[str] | None = None) -> pd.DataFrame:
    """True where the coin is among the first `k` that day. With `among`, only those coins
    compete (the coins that have a spot leg to hold)."""
    table = volume_usd
    if among is not None:
        keep = {c.upper() for c in among}
        table = volume_usd[[c for c in volume_usd.columns if str(c).upper() in keep]]
    return trailing_rank(table, window_days) <= k


def coins_ever_in_top(volume_usd: pd.DataFrame, k: int, window_days: int = 30,
                      among: Collection[str] | None = None) -> list[str]:
    """The coins that were among the first `k` on at least one day, sorted by name."""
    ever = top_k(volume_usd, k, window_days, among).any()
    return sorted(ever[ever].index)


__all__ = ["coins_ever_in_top", "top_k", "trailing_rank"]
