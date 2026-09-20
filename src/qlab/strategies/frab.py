"""FRAB funding harvest: spot long against a perp short, ranked slots.

Driver: perp funding paid by the crowd's directional (mostly long) leverage
on the shorted instrument, collected by holding spot long against an
equal-notional perp short (delta-neutral). See the live engine's own
description, `frab/strategy/two_phase/strategy.py` lines 6-12 ("thin
orchestrator for two-phase dynamic funding-rate arb ... Params sourced from
research/two_phase_dynamic_stability.py 'Candidate C'"), and the project
README, `funding-rate-arbitrage/README.md` line 3: "Async paper-trading
platform for Hyperliquid funding-harvest strategy (Strategy A)".

What makes this mechanism stop paying: funding on every candidate
compresses toward (or through) zero -- the edge is explicitly tied to how
"hot" HL funding runs. `research/GRAVEYARD_REVIEW_2026_09.md` frames the
whole cross-exchange-spread reassessment on whether "funding на HL снова
разогреется" (line 145), and shows FRAB's own correlated cousin fading
season over season (+18% -> +3% net-on-notional, lines 99-104) -- i.e. the
crowd stops paying up to hold leveraged perp longs, and there is nothing
left to harvest.

Interface shape: carry with a limited number of slots and a minimum
position size (`qlab.harness.strategy.Strategy` docstring, shape 2).

Scope note -- this transcribes the RANK / SLOT / MIN-SIZE mechanics of the
carry shape exactly, but deliberately leaves out the live engine's
per-position two-phase breakeven/patience state machine
(`phase1_negative_patience`, `phase1_breakeven_cap_hours`,
`neg_stop_threshold_apr`, `neg_stop_patience_hours` -- all in
`frab/strategy/two_phase/params.py`): those track PER-POSITION history
(hours since THIS position individually crossed breakeven, hours it has
personally been negative) that has no expression as a stateless function of
`(panel, params) -> weights`. What IS implemented is exactly what
`Strategy`'s carry-shape paragraph describes and what
`TwoPhaseParams`/`EntryEvaluator` actually do for entry and sizing: rank
live candidates by a smoothed, annualised funding signal, enter the top
`max_slots` whose signal clears `entry_threshold_apr`, hold until it drops
below `exit_threshold_apr`, size every open slot IDENTICALLY (the live
engine's `footprint` is a fixed `budget_cap_usdc / concurrency_cap`, never
redivided by how many slots happen to be filled -- see
`TwoPhaseParams.compute_footprint`), and drop (never shrink) a slot whose
fixed share would fall under the venue's minimum order.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from qlab.harness.metrics import periods_per_year
from qlab.harness.panel import MarketPanel
from qlab.strategies._periods import periods_for


class FrabFundingHarvest:
    """Carry: spot long + perp short, up to `max_slots` at a time, ranked by funding.

    Required `params` keys (no defaults -- see module docstring and
    `q-lab` task T12 report for exactly where each one is documented):

        spot_columns / perp_columns: Mapping[str, str]
            coin -> panel column name for that coin's spot / perp leg. Run
            input (which instruments this backtest covers), not a strategy
            parameter -- same role as choosing a universe for any strategy.
        max_slots: int
            `TwoPhaseParams.concurrency_cap` --
            "concurrency_cap: int = 3  # K: max simultaneous open positions"
            (`frab/strategy/two_phase/params.py:27`).
        entry_threshold_apr: float
            `TwoPhaseParams.entry_threshold_apr` --
            "entry_threshold_apr: float = 0.10  # entry when smoothed signal > this"
            (`frab/strategy/two_phase/params.py:21`).
        exit_threshold_apr: float
            `TwoPhaseParams.phase2_exit_threshold` --
            "phase2_exit_threshold: float = -0.10  # exit (phase2) when signal < this"
            (`frab/strategy/two_phase/params.py:22`).
        signal_window_hours: int
            `TwoPhaseParams.signal_window_hours` --
            "signal_window_hours: int = 12  # rolling MA window (funding ticks)"
            (`frab/strategy/two_phase/params.py:26`), consumed exactly this
            way by `SignalComputer.compute`
            (`frab/strategy/two_phase/evaluators/signal.py:29-58`): mean of
            the last `signal_window_hours` raw funding rates, times an
            intervals-per-year annualisation factor. Here that factor is the
            panel's OWN `periods_per_year` (see `qlab.harness.metrics`)
            rather than a hardcoded exchange tick rate, since the panel
            doesn't promise hourly bars.
        min_leg_notional_usd: float
            The venue floor a leg must clear to be opened at all --
            "_MIN_LEG = 12.0  # HL ~$10 min order + slippage buffer"
            (`frab/strategy/xsmom/params.py:141`), corroborated by
            `research/GRAVEYARD_REVIEW_2026_09.md:25`: "FRAB ограничен
            размером позиции ($12), а не числом слотов".
        book_capital_usd: float
            The book capital this run assumes, used only to convert
            `min_leg_notional_usd` into a weight-space floor
            (`min_leg_notional_usd / book_capital_usd`). This is a RUN
            input, not a strategy rule -- like `min_leg_notional` in
            `qlab.harness.metrics.min_capital_usd`, a dollar floor is
            meaningless without knowing the capital it is a fraction of, so
            it has no default here either.
    """

    name = "frab"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        spot_columns: Mapping[str, str] = params["spot_columns"]
        perp_columns: Mapping[str, str] = params["perp_columns"]
        if set(spot_columns) != set(perp_columns):
            raise ValueError("spot_columns and perp_columns must name the same coins")
        coins = list(spot_columns)

        max_slots: int = params["max_slots"]
        entry_threshold_apr: float = params["entry_threshold_apr"]
        exit_threshold_apr: float = params["exit_threshold_apr"]
        signal_window_hours: float = params["signal_window_hours"]
        min_leg_notional_usd: float = params["min_leg_notional_usd"]
        book_capital_usd: float = params["book_capital_usd"]

        index = panel.prices.index
        ppy = periods_per_year(index)
        window = periods_for(index, pd.Timedelta(hours=signal_window_hours))

        # Annualised smoothed funding signal per coin -- mirrors
        # `SignalComputer.compute`: mean of the last `window` raw funding
        # rates, times periods-per-year. An unknown funding rate ANYWHERE in
        # the window makes the smoothed signal unknown too: rolling `mean`
        # with `min_periods=window` needs a full window of non-NaN values,
        # and `panel.funding`'s NaN ("unknown") must never be read as zero
        # (`qlab.data.panel` module docstring).
        apr = pd.DataFrame(index=index, columns=coins, dtype=float)
        tradeable_ok = pd.DataFrame(index=index, columns=coins, dtype=bool)
        for coin in coins:
            spot_col, perp_col = spot_columns[coin], perp_columns[coin]
            smoothed = panel.funding[perp_col].rolling(window, min_periods=window).mean()
            apr[coin] = smoothed * ppy
            tradeable_ok[coin] = panel.tradeable[spot_col] & panel.tradeable[perp_col]

        # Fixed per-slot share -- `TwoPhaseParams.compute_footprint`:
        # `budget_cap_usdc / concurrency_cap`, the SAME for every slot
        # regardless of how many are actually filled (an idle slot just
        # leaves capital idle, it never inflates the others' size). If that
        # fixed share can't clear the venue floor, no slot can ever open --
        # this is the "drop, don't under-size" rule applied up front, since
        # every slot has an identical size by construction.
        share = 1.0 / max_slots
        min_position_size = min_leg_notional_usd / book_capital_usd
        slots_affordable = share >= min_position_size

        weights = pd.DataFrame(0.0, index=index, columns=panel.prices.columns)
        if not slots_affordable:
            return weights

        held: set[str] = set()
        for ts in index:
            row_apr = {c: apr.at[ts, c] for c in coins}
            row_ok = {c: bool(tradeable_ok.at[ts, c]) for c in coins}

            # Drop: signal unknown, below the exit bar, or a leg went
            # non-tradeable.
            held = {
                c
                for c in held
                if row_ok[c] and pd.notna(row_apr[c]) and row_apr[c] >= exit_threshold_apr
            }

            # Fill remaining slots from ranked, tradeable candidates that
            # clear the entry bar -- `EntryEvaluator.evaluate`:
            # "candidates.sort(key=lambda x: -x[1]); ... candidates[:slots]"
            # (`frab/strategy/two_phase/evaluators/entry.py:125-126`).
            open_slots = max_slots - len(held)
            if open_slots > 0:
                candidates = sorted(
                    (
                        c
                        for c in coins
                        if c not in held
                        and row_ok[c]
                        and pd.notna(row_apr[c])
                        and row_apr[c] > entry_threshold_apr
                    ),
                    key=lambda c: row_apr[c],
                    reverse=True,
                )
                held |= set(candidates[:open_slots])

            for c in held:
                weights.loc[ts, spot_columns[c]] = share
                weights.loc[ts, perp_columns[c]] = -share

        return weights
