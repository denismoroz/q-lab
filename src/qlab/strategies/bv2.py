"""Strategy B v2: spot majors + trend-timed perp hedge + ratchet.

Driver: staying long a small basket of large-cap spot through the long
stretches it isn't trending down, while a perp short neutralises price risk
exactly when trailing momentum turns negative. The edge is the asymmetry
between "ride the trend" (bull) and "flip to hedged/cash-like exposure"
(bear) -- NOT carry income. See
`funding-rate-arbitrage/research/GRAVEYARD_REVIEW_2026_09.md` lines 36-40:
"Идея -- <<держи и страхуй>>. Держишь спот шести монет ... Когда цена
уходит в нисходящий тренд (моментум за 14 или 30 дней становится
отрицательным), открываешь шорт-перп на весь объём спота -- это страховка.
Когда тренд разворачивается вверх, шорт снимаешь. Плюс <<храповик>>: если
спот вырос выше порога, излишек продаёшь в кэш."

What makes this mechanism stop paying: the market chops sideways instead of
trending, so the hedge flips on and off at its own cost with no compensating
trend to protect or ride --
`funding-rate-arbitrage/research/strategy_b_v2/LIMITS.md` lines 64-67:
"Топтание на месте съедает доход ... 2025-26 (пила с отскоками) -- худшие
окна за 6 лет, до -12%."

Interface shape: spot holding with a conditionally applied hedge leg
(`qlab.harness.strategy.Strategy` docstring, shape 3).

Scope note -- this transcribes the VALIDATED B v2 configuration
(`frab/strategy/b2/params.py`'s `B2Params`, whose own docstring says
"Defaults ARE the validated configuration; change them only with a new
forward test") for the spot + conditional hedge + ratchet mechanics only.
Left OUT, and out of scope for this task's brief (spot + conditional hedge
+ ratchet, not a funding-carry engine):
  - the live engine's funding-carry sleeve on the idle cash reserve
    (`carry_enabled`, `carry_fraction`, `carry_entry_apr`,
    `carry_exit_hours`) -- `research/strategy_b_v2/REPORT.md` line 54
    calls it "по сути FRAB на тех же монетах" (i.e. a second copy of the
    FRAB carry strategy bolted onto B's reserve, not part of B's own
    directional-hedge mechanism);
  - the margin/liquidation/top-up model in `frab/strategy/b2/book.py`
    (initial margin, maintenance margin, `margin_rebalance` top-ups,
    liquidation) -- capital/margin sizing is a preflight/exchange-capacity
    concern (`docs/PLAN.md`, M2), not a target-weights decision;
  - the "refill" step that rebuys spot back up to target once trend turns
    up again after the hedge closes with spot below target
    (`frab/strategy/b2/book.py` step 3, lines 331-344). Without margin
    modelling, spot units are never sold down while hedged in this
    simplified version (only the ratchet trims them, and only while
    unhedged), so this module's spot WEIGHT is allowed to float below
    target after a hedge closes with the coin down, rather than being
    bought back up to target on the next up-trend confirmation.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies._periods import periods_for


class BStrategyV2:
    """Spot + conditionally-applied perp hedge, with a growth ratchet.

    Required `params` keys (no defaults -- see module docstring for where
    each is documented):

        spot_columns / hedge_columns: Mapping[str, str]
            coin -> panel column name for that coin's spot / perp-hedge
            leg. Run input, not a strategy parameter.
        spot_share: float
            `B2Params.spot_share` -- "Share of each coin book held as spot;
            the rest is a cash reserve." default 0.5
            (`frab/strategy/b2/params.py:17`). Coin books are equal-weight
            ("Total paper capital split equally across coin books",
            `params.py:15`; corroborated by `REPORT.md` line 15: "веса
            монет поровну | обратно-волатильные веса хуже на проверке") --
            so each coin's spot TARGET weight, as a fraction of the whole
            book, is `spot_share / len(spot_columns)`.
        hedge_threshold: float
            `B2Params.hedge_threshold` -- "Hedge ON unless both 14d and 30d
            returns exceed this threshold." default 0.0
            (`frab/strategy/b2/params.py:19-20`).
        mom_short_days / mom_long_days: int
            The two momentum lookbacks -- `frab/strategy/b2/book.py:26`:
            "H14, H30, CARRY_WINDOW = 14 * 24, 30 * 24, 8" (hours; 14 and
            30 DAYS), and `REPORT.md` line 17: "Хедж: шорт перпа на весь
            спот, пока 14- ИЛИ 30-дневный доход не выше 0%".
        sticky_exit_hours: int
            `B2Params.sticky_exit_hours` -- "Exit the hedge only after the
            signal has been OFF this many hours in a row." default 12
            (`frab/strategy/b2/params.py:21-22`); `REPORT.md` line 12:
            "липкий выход 12ч -- хедж больше не мерцает на часовом шуме".
        ratchet_threshold: float
            `B2Params.ratchet_threshold` -- "Sell spot back to its target
            when it grows above target * (1 + threshold)." default 0.50
            (`frab/strategy/b2/params.py:23-24`); `REPORT.md` line 14:
            "«храповик» 50% (было 20%) | реже продаёт рост".
    """

    name = "bv2"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        spot_columns: Mapping[str, str] = params["spot_columns"]
        hedge_columns: Mapping[str, str] = params["hedge_columns"]
        if set(spot_columns) != set(hedge_columns):
            raise ValueError("spot_columns and hedge_columns must name the same coins")
        coins = list(spot_columns)

        spot_share: float = params["spot_share"]
        hedge_threshold: float = params["hedge_threshold"]
        mom_short_days: float = params["mom_short_days"]
        mom_long_days: float = params["mom_long_days"]
        sticky_exit_hours: float = params["sticky_exit_hours"]
        ratchet_threshold: float = params["ratchet_threshold"]

        index = panel.prices.index
        short_periods = periods_for(index, pd.Timedelta(days=mom_short_days))
        long_periods = periods_for(index, pd.Timedelta(days=mom_long_days))
        sticky_periods = periods_for(index, pd.Timedelta(hours=sticky_exit_hours))
        spot_weight_target = spot_share / len(coins)

        weights = pd.DataFrame(0.0, index=index, columns=panel.prices.columns)

        for coin in coins:
            spot_col, hedge_col = spot_columns[coin], hedge_columns[coin]
            price = panel.prices[spot_col]
            tradeable_ok = panel.tradeable[spot_col] & panel.tradeable[hedge_col]

            mom_short = price.pct_change(periods=short_periods)
            mom_long = price.pct_change(periods=long_periods)

            # `signals_at`, `frab/strategy/b2/book.py:217-235`:
            #   up = mom14 is not None and mom14 > threshold and mom30 > threshold
            #   raw_hedge = not up
            #   if mom30 is None: raw_hedge = False  # not enough history -> no hedge
            up = (mom_short > hedge_threshold) & (mom_long > hedge_threshold)
            raw_hedge = (~up) & mom_long.notna()

            # Sticky-exit latch -- `advance_signals`, `book.py:238-252`:
            # turns on immediately on any raw wish, turns off only after
            # `sticky_periods` consecutive bars with the wish off. This is a
            # small per-bar state machine with no vectorised equivalent, so
            # it is a plain loop (panels here are backtest-sized, not
            # live-tick-sized).
            hedge_state = pd.Series(False, index=index)
            sticky_on = False
            off_streak = 0
            for ts, raw in raw_hedge.items():
                if bool(raw):
                    sticky_on, off_streak = True, 0
                elif sticky_on:
                    off_streak += 1
                    if off_streak >= sticky_periods:
                        sticky_on = False
                hedge_state.loc[ts] = sticky_on

            # Spot weight: floats with price (units held are constant
            # between trades, so dollar value tracks price -- `book.py`'s
            # `book_equity`), except the ratchet trims it back to
            # `spot_weight_target` whenever it would exceed
            # `spot_weight_target * (1 + ratchet_threshold)` -- "(only
            # while unhedged)" (`book.py` line 346-349: "ratchet: sell spot
            # growth above target (only while unhedged)"). Not tradeable ->
            # treated as flat, and the growth baseline resets once it comes
            # back (mirrors a fresh spot buy at whatever the price is then).
            spot_weight = pd.Series(0.0, index=index)
            reset_price: float | None = None
            for ts in index:
                p = price.loc[ts]
                if not bool(tradeable_ok.loc[ts]) or pd.isna(p):
                    spot_weight.loc[ts] = 0.0
                    reset_price = None
                    continue
                if reset_price is None:
                    reset_price = float(p)
                current = spot_weight_target * (p / reset_price)
                if not hedge_state.loc[ts] and current > spot_weight_target * (
                    1 + ratchet_threshold
                ):
                    reset_price = float(p)
                    current = spot_weight_target
                spot_weight.loc[ts] = current

            # Hedge shorts exactly the current spot units (`OpeningShortState`:
            # hedge qty = spot_qty at the moment the hedge opens, held fixed
            # while hedged) -- since spot units are also fixed while hedged,
            # the hedge's weight at any bar equals the spot weight at that
            # SAME bar, sign-flipped.
            hedge_weight = (-spot_weight).where(hedge_state & tradeable_ok, 0.0)

            weights[spot_col] = spot_weight
            weights[hedge_col] = hedge_weight

        return weights
