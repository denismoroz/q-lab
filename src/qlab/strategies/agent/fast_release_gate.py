"""Block `fast_release_gate`: Bv2's hedge signal released (or also switched on) by the coin's own
short-horizon return (cards/bv2-fast-release-by-own-trend.yaml).

What the card asks. Bv2 hedges a coin (a short of its contract for the same amount) while the
coin's 14-day or 30-day return is not above zero, and takes the hedge off only after "no hedge
needed" has held `sticky_exit_hours` hours in a row. The owner's idea: the hedge is released more
often, by a shorter horizon of the SAME coin (not BTC): when the coin's return over the last
`fast_window_hours` hourly bars is above zero, the hour's "hedge needed" signal is dropped even
if the slow rule still says the coin is falling.

What the code does. For every coin and every hourly bar it builds the raw "hedge needed" signal
the live book builds (`frab.strategy.b2.book._signals`: not enough history -> no hedge; hedge
unless both the short and the long window are above `hedge_threshold`), then changes it:

- `fast_window_hours` == 0: nothing changes; the run is `LiveBv2` itself.
- otherwise the fast return is `price_now / price_{fast_window_hours bars ago} - 1`, read from the
  coin's own closes up to and including the bar (the stand's alignment rule), never later.
  With `fast_entry` == 0 the signal is the slow signal AND NOT (fast return > 0); with
  `fast_entry` == 1 it is (fast return <= 0) and the slow rule is not looked at. A fast return
  that cannot be computed yet (too little history) counts as no hedge, like the book's own rule.

The changed raw signal then goes through the book's sticky exit (`sticky_exit_hours`, applied as
in `LiveBv2`'s external wish) and replaces the book's own wish for the next bar, exactly as the
`hedge_by` mechanism of `qlab.strategies.live.bv2` does. Sizing, margin and headroom, the ratchet,
costs and the rest of the book are Bv2's, unchanged; the hedge may come back in the same hour the
signal does, there is no re-entry ban. Windows count the coin's priced bars, as the book's closes
do, and the fast return is read at the price the spot leg is valued at (`spot_marks`).

Parameters (read from `params`; the rest are `LiveBv2`'s):

- `fast_window_hours`: how many hourly bars back the coin's short return looks; 0 switches the
  block off.
- `fast_entry`: 0: the short return only releases the hedge; 1: the short return alone switches
  the hedge on and off, and the slow rule does not take part.

Not combined with `universe`, `hedge_by` or deposits when the block is on: each of them changes
the coin list or the wish this block replaces, and the run refuses instead of running something
else.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.panel import MarketPanel
from qlab.strategies.live import bv2 as _bv2
from qlab.strategies.live._marks import spot_marks

_REFUSED_WITH_BLOCK = ("universe", "hedge_by", "deposit_usd", "deposit_every")


def _lagged_return(values: np.ndarray, priced: np.ndarray, bars: int) -> np.ndarray:
    """`value / value `bars` priced bars earlier - 1`, per bar; NaN where unpriced or too early.
    Uses only bars up to the one it is written at."""
    out = np.full(len(values), np.nan)
    at = np.flatnonzero(priced)
    series = values[at]
    if len(series) > bars:
        out[at[bars:]] = series[bars:] / series[:-bars] - 1.0
    return out


def release_signal(perp: np.ndarray, spot: np.ndarray, p, fast_window_hours: int,
                   fast_entry: int) -> np.ndarray:
    """The wish for the bar after each bar, one coin: 1.0 hedge, 0.0 not, NaN where the perp has
    no price (the book is not stepped there). `p` is the book's `B2Params`."""
    priced = ~np.isnan(perp)
    short = _lagged_return(perp, priced, int(p.short_window_days) * 24)
    long_ = _lagged_return(perp, priced, int(p.long_window_days) * 24)
    short_up = short > p.hedge_threshold  # NaN compares False, as `mom_short is not None and ...`
    long_up = long_ > p.hedge_threshold
    slow = ~np.isnan(long_) & ~(short_up & long_up)  # no long history -> no hedge
    fast = _lagged_return(spot, priced, fast_window_hours)
    fast_up = fast > 0.0
    if fast_entry == 1:
        raw = ~np.isnan(fast) & ~fast_up
    else:
        raw = slow & ~fast_up
    wish = np.full(len(perp), np.nan)
    wish[priced] = _bv2._sticky(raw[priced].astype(float), int(p.sticky_exit_hours))
    return wish


class _FastDriver(_bv2._Driver):
    """Bv2's driver with the wish of every coin handed in instead of read from `hedge_by`."""

    def __init__(self, panel, p, wishes: dict[str, np.ndarray]) -> None:
        self._wishes = wishes
        super().__init__(panel, p, None, None, None, None)

    def _external(self, coin: str):
        return self._wishes[coin]


class FastReleaseGate(_bv2.LiveBv2):
    """See the module docstring. A `LiveBv2` whose hedge wish is the gated signal."""

    name = "bv2-fast-release-gate"
    valid_intervals = ("1h",)

    def _run(self, panel: MarketPanel, params: Mapping[str, object]) -> _bv2.LiveBv2Run:
        params = dict(params)
        for key in ("fast_window_hours", "fast_entry"):
            if key not in params:
                raise ValueError(f"FastReleaseGate needs params.{key}")
        window, entry = params.pop("fast_window_hours"), params.pop("fast_entry")
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(f"fast_window_hours must be an integer >= 0, got {window!r}")
        if entry not in (0, 1) or isinstance(entry, bool):
            raise ValueError(f"fast_entry must be 0 or 1, got {entry!r}")
        if window == 0:
            return super()._run(panel, params)
        refused = [k for k in _REFUSED_WITH_BLOCK if params.get(k) is not None]
        if refused:
            raise ValueError(f"fast_release_gate cannot run together with {refused}")
        for key in _REFUSED_WITH_BLOCK:
            params.pop(key, None)
        _bv2._require_frab_support(params)
        p = _bv2.B2Params.from_dict(params)
        wishes = {}
        for coin in p.coins:
            spot_col = f"{coin}{SPOT_COLUMN_SUFFIX}"
            for col in (coin, spot_col):
                if col not in panel.prices.columns:
                    raise ValueError(f"Bv2 needs column {col!r}; this panel has none")
            wishes[coin] = release_signal(
                panel.prices[coin].to_numpy(dtype=float), spot_marks(panel, coin, spot_col),
                p, window, int(entry))
        return _FastDriver(panel, p, wishes).go()


__all__ = ["FastReleaseGate", "release_signal"]
