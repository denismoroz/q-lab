"""FRAB perp spread: one contract short, the same coin's contract on another venue long.

Driver (catalogue mechanism ``cross-venue-funding-spread``): two venues price the
funding of the same coin differently. Holding the contract that pays more short and
the one that pays less long, in equal size, collects the difference and cancels the
price, whichever way it moves. The position is held until the spread reverses or
falls under an exit bar, not rebalanced on a schedule, which is why this is a
strategy of its own and not a `Composed` block.

Interface shape: carry with a limited number of slots and a minimum position size
(`qlab.harness.strategy.Strategy` docstring, shape 2), like `FrabFundingHarvest`.

The panel carries the two legs as two columns of one panel: the primary venue's coin
(``BTC``) and the other venue's contract (``BTC@ASTER``, see
`qlab.data.sources.base.second_venue_column`).

Rules a spec must respect:

- Fees. Both legs turn over equally, so ``costs.taker_fee_bps`` of a spec is the
  mean of the two venues' fees.
- The signal window. ``signal_window_hours`` must be a whole number of the SLOWEST
  venue's settlement periods. On an hourly panel a venue that settles every 8 hours
  reads 0 between marks, so a window that is not a whole number of its periods sees
  a different number of settlements from one bar to the next, and the spread jumps
  for no reason.

What is not modelled, and a spec must list it in ``unexpressed_mechanisms``: money
moved between venues, separate collateral on each venue, and the liquidation of
each leg on its own.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.harness.metrics import periods_per_year
from qlab.harness.panel import MarketPanel
from qlab.strategies._periods import periods_for


class FrabPerpSpread:
    """Carry: short the venue that pays more, long the one that pays less.

    Required `params` keys (no defaults: every number that shapes a result is
    supplied, and cited, by the spec):

        venue_a_columns / venue_b_columns: Mapping[str, str]
            coin -> panel column of that coin on venue A / venue B. Direction
            ``+1`` is short A, long B; ``-1`` is the mirror.
        max_slots: int
            How many coins are held at once.
        entry_spread_apr: float
            A coin is entered when ``abs(spread)`` is above this (annualised).
        exit_spread_apr: float
            A held coin is closed when its spread, in the held direction, falls
            below this. Must be under ``entry_spread_apr``.
        signal_window_hours: float
            Window of the smoothed funding of each leg (see the module docstring
            for how it must relate to the venues' settlement periods).
        leg_share_of_slot: float
            The share of a slot each leg takes, as a weight of the book.
        min_leg_notional_usd / book_capital_usd:
            The venue floor of a leg and the book it is a fraction of. When a
            slot's leg is under the floor no slot can open.
    """

    name = "frab-perp-spread"
    valid_intervals = ("1h",)
    # Declared (docs/CAPITAL_BASE.md): positions are sized in dollars over a constant budget,
    # as in FRAB; what is earned stays on the account.
    reinvests_profit = False

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        a_columns: Mapping[str, str] = params["venue_a_columns"]
        b_columns: Mapping[str, str] = params["venue_b_columns"]
        max_slots: int = params["max_slots"]
        entry_spread_apr: float = params["entry_spread_apr"]
        exit_spread_apr: float = params["exit_spread_apr"]
        signal_window_hours: float = params["signal_window_hours"]
        leg_share_of_slot: float = params["leg_share_of_slot"]
        min_leg_notional_usd: float = params["min_leg_notional_usd"]
        book_capital_usd: float = params["book_capital_usd"]

        if set(a_columns) != set(b_columns):
            raise ValueError("venue_a_columns and venue_b_columns must name the same coins")
        if set(a_columns.values()) & set(b_columns.values()):
            raise ValueError("a column cannot be a leg on both venues")
        if leg_share_of_slot <= 0:
            raise ValueError(f"leg_share_of_slot must be > 0, got {leg_share_of_slot}")
        if exit_spread_apr >= entry_spread_apr:
            raise ValueError(
                f"exit_spread_apr ({exit_spread_apr}) must be under entry_spread_apr "
                f"({entry_spread_apr}): otherwise a position closes the bar it opens"
            )
        coins = list(a_columns)

        index = panel.prices.index
        ppy = periods_per_year(index)
        window = periods_for(index, pd.Timedelta(hours=signal_window_hours))

        # Annualised difference of the smoothed funding of the two legs. An unknown
        # rate anywhere in a window makes the spread unknown, never zero.
        spread = pd.DataFrame(index=index, columns=coins, dtype=float)
        ok = pd.DataFrame(index=index, columns=coins, dtype=bool)
        for coin in coins:
            a, b = a_columns[coin], b_columns[coin]
            smoothed_a = panel.funding[a].rolling(window, min_periods=window).mean()
            smoothed_b = panel.funding[b].rolling(window, min_periods=window).mean()
            spread[coin] = (smoothed_a - smoothed_b) * ppy
            ok[coin] = panel.tradeable[a] & panel.tradeable[b]

        # Fixed share per slot, as in `FrabFundingHarvest`: an idle slot leaves capital
        # idle. A share under the venue floor means no slot can ever open.
        share = leg_share_of_slot / max_slots
        weights = pd.DataFrame(0.0, index=index, columns=panel.prices.columns)
        if share < min_leg_notional_usd / book_capital_usd:
            return weights

        held: dict[str, int] = {}
        for ts in index:
            row = {c: spread.at[ts, c] for c in coins}
            row_ok = {c: bool(ok.at[ts, c]) for c in coins}

            held = {
                c: d
                for c, d in held.items()
                if row_ok[c] and pd.notna(row[c]) and d * row[c] >= exit_spread_apr
            }

            open_slots = max_slots - len(held)
            if open_slots > 0:
                candidates = sorted(
                    (
                        c
                        for c in coins
                        if c not in held
                        and row_ok[c]
                        and pd.notna(row[c])
                        and abs(row[c]) > entry_spread_apr
                    ),
                    key=lambda c: abs(row[c]),
                    reverse=True,
                )
                for c in candidates[:open_slots]:
                    held[c] = 1 if row[c] > 0 else -1

            for c, direction in held.items():
                weights.loc[ts, a_columns[c]] = -direction * share
                weights.loc[ts, b_columns[c]] = direction * share

        return weights
