"""Strategy B v2 driven by CALLING the live engine's own book
(`frab.strategy.b2.book`), not by re-deriving it.

The transcription `qlab.strategies.bv2` leaves out what the live book does
with money: the margin pool, top-ups when a hedge bleeds, liquidation of the
hedge on a bar's high, and the refill of spot after a hedge closes
(specs/bv2.yaml, `unexpressed_mechanisms`). Bv2 is therefore not evaluable
through it. The live book is a full accounting state machine that returns
fills, not weights, so it cannot be called like trend's signal function; this
module drives it bar by bar and READS weights off its state (docs/PLAN.md,
stage 4, "переходник для стратегий со своим счётом").

## How weights are read

Weights are read on decision bars only -- a fill, a book start, a pause on
missing data -- and held in between (see `run`). Each coin has its own
`CoinBook` with `capital / len(coins)`, exactly as the live engine runs them.
After every bar is stepped, a coin's exposures are

    spot column  (`<COIN>-SPOT`):  +units_spot * P  (+ carry_units * P)
    perp column  (`<COIN>`):        -short_size * P  (- carry_units * P)

and every exposure is divided by the TOTAL equity of all books at that bar
(a coin whose book has not started yet holds its share as cash). The harness
then does its own accounting on those weights -- costs, funding accrual, P&L
-- as for any strategy, which keeps the independent check (the one that
caught the funding sign) and lets matched noise perturb these weights like
any other book's.

## What the live book does that the harness then accounts differently

These are fidelity gaps of the accounting, not missing mechanisms -- every
decision is the live book's own:

- **Liquidation price.** The book closes a breached short at its liquidation
  level within the bar; the harness sees the hedge weight drop to zero at
  that bar's close. `run` returns the book's own equity curve so the two can
  be compared (`docs/BV2_LIVE.md`).
- **Fees.** The book charges frab's own taker fees on its fills; the harness
  charges the spec's costs on weight changes.
- **Spot price.** The live paper book values spot at the perp price; the
  harness prices the spot leg on the real spot column.

## Rules this driver keeps

- **Bar highs are required.** The liquidation check runs on the bar's high;
  an unknown high is not the close (`qlab.data.panel.MarketPanel.high`). A bar
  with an unknown high on a coin whose book is running raises.
- **A coin's book starts on the first bar where both its legs are
  tradeable AND the panel already holds the history the live engine loads
  at startup** (`HISTORY_BARS + WARMUP_BARS` perp closes), with the
  sticky-exit state rebuilt from the preceding bars exactly as its
  `_warm_signals` does. For an established coin whose earlier hourly
  history the panel lacks (the venue serves only its last 5000 candles),
  this is what the live engine would see; starting earlier would leave the
  30-day momentum blind for a month and the hedge unable to switch on. A
  coin genuinely younger than that history would start earlier live.
- **A bar where a leg is not tradeable is skipped,** as the live engine waits
  on missing data; the book is not stepped and holds nothing in the harness
  for that bar.
- **Signals use the perp's closes,** as the live engine does (it fetches only
  the perp candles), over at most `HISTORY_BARS` of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.panel import MarketPanel
from qlab.strategies.live._loader import import_frab

_book = import_frab("frab.strategy.b2.book")
_params_module = import_frab("frab.strategy.b2.params")
B2Params = _params_module.B2Params

# frab/strategy/b2/engine.py:28-29. The engine module itself is not importable
# here (it pulls in the database and the exchange client); these two constants
# are its only inputs to the book.
HISTORY_BARS = 740
WARMUP_BARS = 48


@dataclass(frozen=True)
class LiveBv2Run:
    """Weights for the harness plus the books' own view of the same run."""

    weights: pd.DataFrame
    book_equity: pd.Series  # sum over coins of the live books' own equity
    liquidations: int


class LiveBv2:
    """See module docstring. `params` is handed unchanged to
    `B2Params.from_dict`, so frab's own validation applies."""

    name = "bv2-live"
    valid_intervals = ("1h",)  # the live book counts its windows in hourly bars

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveBv2Run:
        p = B2Params.from_dict(dict(params))
        index = panel.prices.index
        n = len(index)
        columns = panel.prices.columns
        exposure = pd.DataFrame(0.0, index=index, columns=columns)
        equity = np.zeros(n)
        liquidations = 0
        # Bars on which some book actually did something: a fill, a start, or
        # a pause/resume on missing data. Weights are re-read only there.
        decided = np.zeros(n, dtype=bool)

        for coin in p.coins:
            perp, spot = coin, f"{coin}{SPOT_COLUMN_SUFFIX}"
            for col in (perp, spot):
                if col not in columns:
                    raise ValueError(f"Bv2 needs column {col!r}; this panel has none")
            px = panel.prices[perp].to_numpy(dtype=float)
            fr = panel.funding[perp].to_numpy(dtype=float)
            hi = panel.high[perp].to_numpy(dtype=float)
            ok = (
                panel.tradeable[perp].to_numpy(dtype=bool)
                & panel.tradeable[spot].to_numpy(dtype=bool)
                & ~np.isnan(px)
                & ~np.isnan(fr)
            )

            coin_equity = np.full(n, p.book_capital)
            valid = np.flatnonzero(~np.isnan(px))  # closes the live engine would hold
            history_before = np.cumsum(~np.isnan(px)) - (~np.isnan(px)).astype(int)
            started = np.flatnonzero(ok & (history_before >= HISTORY_BARS + WARMUP_BARS))
            if started.size:
                i0 = int(started[0])
                book = _book.CoinBook.new(coin, p)
                closes = px[valid]
                # Funding history feeds only the carry signal; an unknown rate
                # reads as 0 there, exactly like the live engine's
                # `funding.get(h, 0.0)`. A stepped bar's own rate is never
                # unknown (`ok` excludes it).
                rates = np.nan_to_num(fr[valid])
                pos = {int(row): k for k, row in enumerate(valid)}

                k0 = pos[i0]
                for k in range(max(0, k0 - WARMUP_BARS), k0):
                    _book.advance_signals(
                        book,
                        list(closes[max(0, k - HISTORY_BARS + 9) : k + 1]),
                        list(rates[max(0, k - 8) : k + 1]),
                        p,
                    )
                _book.start_book(book, bar_ms=_ms(index[i0]), price=px[i0], params=p)
                decided[i0] = True

                last_equity = book.equity(px[i0])
                for i in range(i0, n):
                    if not ok[i]:
                        coin_equity[i] = last_equity
                        decided[i] = True
                        if i + 1 < n:
                            decided[i + 1] = True
                        continue
                    if np.isnan(hi[i]):
                        raise ValueError(
                            f"bar high unknown for {perp} at {index[i]}: the live book's "
                            "liquidation check needs it, and the close is not a substitute"
                        )
                    k = pos[i]
                    events = _book.step(
                        book,
                        bar_ms=_ms(index[i]),
                        price=float(px[i]),
                        funding_rate=float(fr[i]),
                        closes=list(closes[max(0, k - HISTORY_BARS + 9) : k + 1]),
                        funding_hist=list(rates[max(0, k - 8) : k + 1]),
                        params=p,
                        high=float(hi[i]),
                    )
                    liquidations += sum(1 for e in events if e.get("kind") == "liquidation")
                    if events:
                        decided[i] = True
                    price = float(px[i])
                    # The carry sleeve holds spot long and perp short in equal
                    # units: `carry_units` with the margin model, the fixed
                    # `carry_target` notional without it (book.py, step 5).
                    if not book.carry_on:
                        carry = 0.0
                    elif p.margin_enabled:
                        carry = book.carry_units
                    else:
                        carry = book.carry_target / price
                    exposure.iat[i, columns.get_loc(spot)] = (book.units_spot + carry) * price
                    exposure.iat[i, columns.get_loc(perp)] = -(book.hedge_units() + carry) * price
                    last_equity = book.equity(price)
                    coin_equity[i] = last_equity
            equity += coin_equity

        # The books hold UNITS; as fractions of equity they drift every bar
        # with price although nothing trades. Handing the drift to the harness
        # would book a trade every hour (costs on turnover that never
        # happened) and make every bar a "decision" for matched noise, which
        # then redraws hourly and fails structural matching. So weights are
        # read at decision bars -- fills, starts, data pauses -- and held in
        # between: the harness sees the strategy's own decisions.
        read = exposure.div(equity, axis=0)
        weights = read.where(pd.Series(decided, index=index), np.nan).ffill().fillna(0.0)
        weights = weights.where(panel.tradeable, 0.0)
        return LiveBv2Run(
            weights=weights,
            book_equity=pd.Series(equity, index=index),
            liquidations=liquidations,
        )


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


__all__ = ["HISTORY_BARS", "LiveBv2", "LiveBv2Run", "WARMUP_BARS"]
