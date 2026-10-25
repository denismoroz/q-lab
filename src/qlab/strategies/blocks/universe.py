"""Universe blocks: which coins may be considered at each bar.

A universe block returns a boolean table (bars x coins): True means the coin is
allowed at that bar. It may use only data up to and including that bar.

- `all`: every coin is allowed.
- `top_volume`: the `n` coins with the largest trailing average dollar volume
  (volume times price, averaged over the last `window_days` days). Only coins
  that are tradeable, priced and have a fully known window at that bar compete
  for the places; ties go to the coin listed first in the panel.
- `asset_class`: only instruments whose asset class AT THAT BAR is one of `classes`
  (docs/ASSET_CLASSES.md): a kind such as "stock" or "fund", or a ranking group such as
  "stock:US". The class comes from the point-in-time table of the deployment
  (`qlab.data.asset_classes`); an instrument the table does not know, has not listed yet,
  has already delisted, or could not classify is outside every class. Off unless a spec
  names this block, so existing specs are unchanged.
"""

from __future__ import annotations

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block, positive_int, positive_number
from qlab.strategies.blocks._util import bars


def all_coins(panel: MarketPanel) -> pd.DataFrame:
    """Every coin is in the universe at every bar."""
    return pd.DataFrame(True, index=panel.prices.index, columns=panel.prices.columns)


def top_volume(panel: MarketPanel, *, n: int, window_days: float) -> pd.DataFrame:
    """The `n` coins with the highest trailing mean dollar volume at each bar."""
    window = bars(panel.prices.index, window_days)
    dollar_volume = panel.volume * panel.prices
    average = dollar_volume.rolling(window, min_periods=window).mean()
    competing = panel.tradeable.astype(bool) & panel.prices.notna() & average.notna()
    rank = average.where(competing).rank(axis=1, method="first", ascending=False)
    return rank.le(n)


def _class_names(label: str, value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise ValueError(f"{label} must be a non-empty list of class names, got {value!r}")
    return [str(v) for v in value]


def asset_class(panel: MarketPanel, *, classes: list[str]) -> pd.DataFrame:
    """True where the instrument's class at that bar is one of `classes`."""
    from qlab.data import asset_classes

    table = asset_classes.load_table()
    return table.member_mask(panel.prices.index, panel.prices.columns, classes)


UNIVERSES: dict[str, Block] = {
    "all": Block(all_coins, {}),
    "top_volume": Block(top_volume, {"n": positive_int, "window_days": positive_number}),
    "asset_class": Block(asset_class, {"classes": _class_names}),
}
