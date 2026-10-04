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
- **Spot price.** The live paper book values spot at the perp price, and so
  does the panel (spot legs are marked at their own perp, 2026-10-02).

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
  on missing data; the book is not stepped. If the leg merely has no price
  there (docs/TASKS.md T36, `qlab.harness.gaps`), both legs are held through
  it, as live; otherwise (an unknown funding rate) the book holds nothing in
  the harness for that bar.
- **The hedge may come from a regime detector** (`hedge_by`, docs/TASKS.md
  T39): the book's own wish for the next bar is replaced by the stored
  predictions; everything else -- sticky state, sizes, fills -- stays frab's.
  Without `hedge_by` nothing changes.
- **Signals use the perp's closes,** as the live engine does (it fetches only
  the perp candles), over at most `HISTORY_BARS` of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.gaps import holdable_gaps
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


def _external_hedge(hedge_by: Mapping[str, object], index: pd.DatetimeIndex,
                    coins: list[str], sticky_hours: int) -> dict[str, np.ndarray]:
    """The hedge wish for the bar after each bar, per coin: 1.0 hedge, 0.0
    not, NaN before the first prediction (the book's own rule then applies).

    Two sources (docs/TASKS.md T39):

    - `{predictions: <name>, hedge_when: [...]}` -- the market regime
      detector (`qlab.regime_detect`), one daily wish for every coin: hedge
      when the latest prediction is one of `hedge_when`;
    - `{meta: <name>}` -- Bv2's own level 2 (`qlab.meta_hedge`), an hourly
      wish per coin. Its wish passes through the book's sticky exit, as the
      book's own rule does (book.py `advance_signals`): on at once, off only
      after `sticky_exit_hours` hours in a row without it.

    Predictions are stamped by the close they were made at, from data up to
    it and models trained on labels known a month earlier; an hourly bar
    labelled by its open closes an hour later and may use every prediction
    stamped by then. The look-ahead guard cannot see into the stored files --
    their causality is tested in `qlab.test_regime_detect` and
    `qlab.test_meta_hedge`."""
    keys = set(hedge_by)
    if "meta" in keys:
        if keys != {"meta"}:
            extra = sorted(keys - {"meta"})
            raise ValueError(f"hedge_by with meta takes nothing else, not {extra}")
        from qlab.meta_label import load_decisions

        decisions = load_decisions(str(hedge_by["meta"]))
        out = {}
        for coin in coins:
            if coin not in decisions.columns:
                raise ValueError(f"level-2 decisions {hedge_by['meta']!r} have no column {coin!r}")
            raw = decisions[coin].astype(float).reindex(index).to_numpy()
            out[coin] = _sticky(raw, sticky_hours)
        return out
    from qlab.regime_detect import load_predictions

    unknown = keys - {"predictions", "hedge_when"}
    if unknown:
        raise ValueError(f"hedge_by takes predictions and hedge_when, not {sorted(unknown)}")
    when = list(hedge_by.get("hedge_when") or [])
    if not when or not set(when) <= {"bull", "flat", "bear"}:
        raise ValueError(f"hedge_by.hedge_when must list regimes (bull, flat, bear): {when}")
    predictions = load_predictions(str(hedge_by["predictions"]))
    wish = predictions["label"].isin(when).astype(float)
    closes = index + pd.Timedelta(hours=1)
    market = wish.reindex(closes, method="ffill").to_numpy(dtype=float)
    return {coin: market for coin in coins}


def _sticky(raw: np.ndarray, hours: int) -> np.ndarray:
    """The book's sticky exit applied to an external wish (NaN passes)."""
    out = raw.copy()
    on, off = False, 0
    for i, w in enumerate(raw):
        if np.isnan(w):
            on, off = False, 0
            continue
        if w:
            on, off = True, 0
        elif on:
            off += 1
            if off >= hours:
                on = False
        out[i] = 1.0 if on else 0.0
    return out


class LiveBv2:
    """See module docstring. `params` is handed unchanged to
    `B2Params.from_dict`, so frab's own validation applies."""

    name = "bv2-live"
    valid_intervals = ("1h",)  # the live book counts its windows in hourly bars

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveBv2Run:
        params = dict(params)
        hedge_by = params.pop("hedge_by", None)
        p = B2Params.from_dict(params)
        index = panel.prices.index
        externals = (_external_hedge(hedge_by, index, list(p.coins), int(p.sticky_exit_hours))
                     if hedge_by else None)
        n = len(index)
        columns = panel.prices.columns
        exposure = pd.DataFrame(0.0, index=index, columns=columns)
        gaps = holdable_gaps(panel.prices, panel.funding,
                             panel.meta.get("no_funding_instruments", ()))
        equity = np.zeros(n)
        liquidations = 0
        # Bars on which some book actually did something: a fill, a start, or
        # a pause/resume on missing data. Weights are re-read only there.
        decided = np.zeros(n, dtype=bool)

        for coin in p.coins:
            external = externals[coin] if externals is not None else None
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
            # A bar some leg is held through (no price, docs/TASKS.md T36):
            # the live engine waits with both legs open, and the harness holds
            # them -- the book is not stepped and keeps its last exposure.
            held = (
                (panel.tradeable[perp] | gaps[perp]).to_numpy(dtype=bool)
                & (panel.tradeable[spot] | gaps[spot]).to_numpy(dtype=bool)
                & (gaps[perp] | gaps[spot]).to_numpy(dtype=bool)
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
                if external is not None and i0 > 0 and not np.isnan(external[i0 - 1]):
                    book.hedge_prev = bool(external[i0 - 1])
                _book.start_book(book, bar_ms=_ms(index[i0]), price=px[i0], params=p)
                decided[i0] = True

                last_equity = book.equity(px[i0])
                for i in range(i0, n):
                    if not ok[i] and held[i] and i > i0:
                        coin_equity[i] = last_equity
                        for col in (spot, perp):
                            j = columns.get_loc(col)
                            exposure.iat[i, j] = exposure.iat[i - 1, j]
                        continue
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
                    if external is not None and not np.isnan(external[i]):
                        # The hedge for the next bar comes from the regime
                        # detector instead of the book's own 14/30-day rule
                        # (`hedge_by`, docs/REGIME_DETECT.md).
                        book.hedge_prev = bool(external[i])
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
