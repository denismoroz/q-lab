"""FRAB -- the two-phase funding harvest that trades live money -- driven by
CALLING frab's own decision functions (`frab.engine.two_phase_signals`), not
by re-deriving them.

Owner, 2026-10-24: «а почему http://localhost:5890/ideas/frab - пустой»;
«делай все три по порядку». The transcription `qlab.strategies.frab` leaves
out the two-phase breakeven machine and the negative-funding stop
(specs/frab.yaml, `unexpressed_mechanisms`), so FRAB was not evaluable at all.

The live strategy (`frab.strategy.two_phase.TwoPhaseStrategy`) is async and
lives on a database and an exchange client; it cannot run on the stand. But
every decision it makes is one of frab's pure functions, and this module calls
exactly those, with the state the live states keep in `state_data`:

- signal: the mean of the last `signal_window_hours` hourly funding rates,
  times 8760 (evaluators/signal.py, Hyperliquid's hourly interval); unknown
  until that many rates exist;
- each hour, in the live order (`on_hour_tick`): funding accrues on open
  positions, then exits are evaluated, then entries (strategy.py);
- entry (evaluators/entry.py): coins of the universe without a position whose
  signal passes `decide_entry`, strongest first, into the slots left by
  `concurrency_cap` and by `budget_cap_usdc` in footprints of
  `budget_cap_usdc / concurrency_cap`;
- at the open (states/opening_short.py): spot notional `compute_size_for`
  (the slot over 1 + margin_buffer_factor / the coin's leverage), the short
  in equal units, `position_min_hold_hours` from `compute_position_min_hold`,
  `total_fees_paid` = notional x (PERP_TAKER + SPOT_TAKER) x 2;
- phase 1 (states/pre_breakeven.py): latch to phase 2 once accrued funding
  covers the fees, otherwise `decide_pre_breakeven` with the hourly income at
  the current signal; phase 2 (states/post_breakeven.py):
  `decide_post_breakeven` with the giveback from the funding peak;
  `update_consec_negative` in both.

## What is not the live engine

- **Fills.** A position opens and closes at the hour it is decided, at that
  hour's close; live it takes the minutes the state machine needs.
- **The universe** is today's (`coin_registry`: active and validated); the
  live registry changed over time (HYPE and PURR were added 2026-06-01).
- **Failures and retries** (a leg that fails to fill, the one-hour entry
  cooldown after a FAILED position) do not happen here.
- **Margin.** The margin watchdog's top-ups are not modelled; the slot already
  reserves margin x `margin_buffer_factor`.

## How weights are read

As for Bv2 (`qlab.strategies.live.bv2`): a position holds units; its spot
column is +units x the spot's price and its perp column -units x the perp's
price (the same price by default; under `data.spot_prices: own` the spot's own
last close, `qlab.strategies.live._marks`), divided by the
strategy's committed budget (`budget_cap_usdc`, the capital the live engine
sizes from; it does not compound). Weights are read only on bars where a
position opens or closes and held in between, so the harness books the
strategy's own trades, not hourly drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.panel import MarketPanel
from qlab.strategies.live._loader import import_frab
from qlab.strategies.live._marks import spot_marks

_signals = import_frab("frab.engine.two_phase_signals")
_constants = import_frab("frab.constants")

HOURS_PER_YEAR = 8760  # evaluators/signal.py: Hyperliquid funds hourly


@dataclass(frozen=True)
class FrabParams:
    """The fields of frab's `TwoPhaseParams` this driver uses, all required:
    its module cannot be imported here (its package pulls in the exchange),
    so no default of frab's is assumed silently -- the spec writes each one."""

    entry_threshold_apr: float
    phase2_exit_threshold: float
    base_min_hold_hours: int
    cap_min_hold_hours: int
    safety_mult: float
    signal_window_hours: int
    concurrency_cap: int
    budget_cap_usdc: float
    margin_buffer_factor: float
    phase1_negative_patience: int
    phase1_breakeven_cap_hours: int
    neg_stop_threshold_apr: float
    neg_stop_patience_hours: int
    phase2_giveback_fee_mult: float

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> FrabParams:
        names = cls.__dataclass_fields__
        missing = sorted(set(names) - set(d))
        if missing:
            raise ValueError(f"FRAB params missing: {missing}")
        return cls(**{k: d[k] for k in names})


@dataclass
class _Position:
    coin: str
    opened: int
    units: float
    size_usdc: float
    min_hold: int
    total_fees: float
    gross_funding: float = 0.0
    consec_negative: int = 0
    post: bool = False
    peak_funding: float = 0.0
    closed: int | None = None
    exit: str = ""
    # What a venue hook keeps for the position (`LiveFrabHooks`); the plain
    # driver never reads them.
    entry_px: float = 0.0
    cash: float = 0.0


@dataclass(frozen=True)
class LiveFrabRun:
    weights: pd.DataFrame
    positions: list[_Position] = field(default_factory=list)
    # How many coins could be entered on each bar (both legs priced and
    # tradeable, a signal known) and how many positions were open after it.
    eligible: pd.Series | None = None
    held: pd.Series | None = None
    # Costs that are not a rate on turnover, as a share of the budget per bar
    # (`LiveFrabHooks.extra_cost_usd`); None when the driver has none.
    extra_costs: pd.Series | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


class LiveFrabHooks:
    """What a driver adds around the live decisions without changing them: the
    fee the engine counts a round trip at, and what happens to a position's
    money. This one does nothing, which is `LiveFrab` as it has always run."""

    fee_rate: float = 0.0
    n_bars: int = 0

    def can_enter(self, coin: str, i: int) -> bool:
        """A further condition on opening `coin` on bar i, beyond the panel's
        tradeable mask and the signal. Always true here."""
        return True

    def on_open(self, pos: _Position, i: int) -> None:
        """A position was opened on bar i (after the engine sized it)."""

    def on_close(self, pos: _Position, i: int) -> None:
        """A position was closed on bar i by the engine's decision."""

    def on_bar(self, i: int, open_: dict[str, _Position], done: list[_Position]) -> bool:
        """End of bar i, after exits and entries. May resize or close positions
        (closed ones move from `open_` to `done`); True if it changed any."""
        return False

    def extra_cost_usd(self) -> np.ndarray | None:
        """Dollar costs per bar that the harness's rate on turnover does not
        cover, or None."""
        return None

    def diagnostics(self) -> dict[str, float]:
        return {}


def size_for(params: FrabParams, leverage: float) -> float:
    """`TwoPhaseParams.compute_size_for` (frab/strategy/two_phase/params.py),
    with the coin's leverage handed in (live it comes from the coin registry)."""
    slot = params.budget_cap_usdc / params.concurrency_cap
    return slot / (1 + params.margin_buffer_factor / leverage)


def signal_at(rates: np.ndarray, i: int, window: int) -> float | None:
    """The smoothed annual signal from the `window` rates up to row i."""
    if i + 1 < window:
        return None
    last = rates[i + 1 - window : i + 1]
    if np.isnan(last).any():
        return None
    return float(last.mean()) * HOURS_PER_YEAR


class LiveFrab:
    """See module docstring. `params` holds `FrabParams`' fields (the live
    `params_json`, plus `phase2_giveback_fee_mult` which the live row leaves at
    frab's default) and `universe` and `leverage`: the coin registry's active,
    validated coins and their leverage."""

    name = "frab-live"
    valid_intervals = ("1h",)  # the live engine decides on hourly ticks
    # Declared (docs/CAPITAL_BASE.md): the stand's budget is a constant divisor of dollar
    # positions (`budget_cap_usdc`); the result stays on the account, free, and is not put back
    # into positions. `LiveFrabBinanceSpot` and `LiveFrabAllBinance` inherit it.
    reinvests_profit = False

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveFrabRun:
        params = dict(params)
        universe = [str(c) for c in params.pop("universe")]
        leverage = {str(k): float(v) for k, v in dict(params.pop("leverage")).items()}
        p = FrabParams.from_dict(params)
        return self._simulate(panel, p, universe, leverage, None)

    def _simulate(self, panel: MarketPanel, p: FrabParams, universe: list[str],
                  leverage: dict[str, float],
                  make_hooks: Callable[[list[str], dict[str, np.ndarray], dict[str, np.ndarray],
                                        dict[str, np.ndarray], dict[str, np.ndarray],
                                        dict[str, np.ndarray]],
                                       LiveFrabHooks] | None) -> LiveFrabRun:
        """The live decisions over `universe`. `make_hooks` (given the coins and
        their perp prices, spot marks, bar highs, funding rates and the masks of
        bars on which both legs can be traded) supplies the venue's own handling
        of money; None is the plain driver."""
        index = panel.prices.index
        columns = panel.prices.columns
        n = len(index)
        coins = [c for c in universe if c in columns and f"{c}{SPOT_COLUMN_SUFFIX}" in columns]
        missing = sorted(set(universe) - set(coins))
        if missing:
            raise ValueError(f"FRAB's universe needs perp and spot columns; missing: {missing}")
        px = {c: panel.prices[c].to_numpy(dtype=float) for c in coins}
        # What the spot leg is worth per unit: the perp's price by default, the
        # spot's own last close under `data.spot_prices: own` (`_marks`).
        sp = {c: spot_marks(panel, c, f"{c}{SPOT_COLUMN_SUFFIX}") for c in coins}
        rates = {c: panel.funding[c].to_numpy(dtype=float) for c in coins}
        ok = {c: (panel.tradeable[c] & panel.tradeable[f"{c}{SPOT_COLUMN_SUFFIX}"]).to_numpy(bool)
                 & ~np.isnan(px[c]) & ~np.isnan(sp[c]) for c in coins}
        if make_hooks is None:
            hooks = LiveFrabHooks()
            hooks.fee_rate = _constants.PERP_TAKER + _constants.SPOT_TAKER
        else:
            highs = {c: panel.high[c].to_numpy(dtype=float) for c in coins}
            hooks = make_hooks(coins, px, sp, highs, rates, ok)
        fee_rate = hooks.fee_rate
        eligible = np.zeros(n)
        held_count = np.zeros(n)

        exposure = np.zeros((n, len(columns)))
        col = {c: (columns.get_loc(c), columns.get_loc(f"{c}{SPOT_COLUMN_SUFFIX}")) for c in coins}
        decided = np.zeros(n, dtype=bool)
        open_: dict[str, _Position] = {}
        done: list[_Position] = []
        footprint = p.budget_cap_usdc / p.concurrency_cap

        for i in range(n):
            # 1) funding on open shorts: the short receives the rate on its notional
            for c, pos in open_.items():
                if not np.isnan(rates[c][i]) and not np.isnan(px[c][i]):
                    pos.gross_funding += pos.units * px[c][i] * rates[c][i]
            signal = {c: signal_at(rates[c], i, p.signal_window_hours) for c in coins}
            eligible[i] = sum(1 for c in coins if ok[c][i] and signal[c] is not None)
            # 2) exits, phase by phase, as the live states evaluate them
            for c in list(open_):
                pos = open_[c]
                if not ok[c][i]:
                    continue  # the live engine waits on missing data
                s = signal[c]
                held = i - pos.opened
                pos.consec_negative = _signals.update_consec_negative(
                    prev_consec_negative=pos.consec_negative, smoothed_signal_annual=s)
                if not pos.post:
                    if pos.gross_funding >= pos.total_fees:
                        pos.post = True  # the latch; phase 2 evaluates from the next hour
                        pos.peak_funding = pos.gross_funding
                        continue
                    income = pos.size_usdc * s / HOURS_PER_YEAR if s is not None and s > 0 else 0.0
                    decision = _signals.decide_pre_breakeven(
                        smoothed_signal_annual=s, hours_in_position=held,
                        position_min_hold_hours=pos.min_hold,
                        gross_funding_so_far=pos.gross_funding, total_fees_paid=pos.total_fees,
                        consec_negative_hours=pos.consec_negative,
                        current_hourly_income_quote=income,
                        phase1_negative_patience=p.phase1_negative_patience,
                        phase1_breakeven_cap_hours=p.phase1_breakeven_cap_hours,
                        neg_stop_threshold=p.neg_stop_threshold_apr,
                        neg_stop_patience=p.neg_stop_patience_hours)
                else:
                    pos.peak_funding = max(pos.peak_funding, pos.gross_funding)
                    decision = _signals.decide_post_breakeven(
                        smoothed_signal_annual=s, hours_in_position=held,
                        position_min_hold_hours=pos.min_hold,
                        phase2_exit_threshold=p.phase2_exit_threshold,
                        funding_giveback=pos.peak_funding - pos.gross_funding,
                        total_fees_paid=pos.total_fees,
                        giveback_fee_mult=p.phase2_giveback_fee_mult)
                if decision != _signals.TwoPhaseDecision.NONE:
                    pos.closed, pos.exit = i, str(decision)
                    done.append(open_.pop(c))
                    hooks.on_close(pos, i)
                    decided[i] = True
            # 3) entries into the slots left
            slots = min(p.concurrency_cap - len(open_),
                        int((p.budget_cap_usdc - len(open_) * footprint) // footprint))
            if slots > 0:
                wanted = [(c, signal[c]) for c in coins
                          if c not in open_ and ok[c][i] and hooks.can_enter(c, i)
                          and _signals.decide_entry(
                              smoothed_signal_annual=signal[c],
                              entry_threshold=p.entry_threshold_apr)
                          == _signals.TwoPhaseDecision.OPEN]
                wanted.sort(key=lambda x: -x[1])
                for c, s in wanted[:slots]:
                    size = size_for(p, leverage[c])
                    open_[c] = _Position(
                        coin=c, opened=i, units=size / sp[c][i], size_usdc=size,
                        min_hold=_signals.compute_position_min_hold(
                            entry_signal_annual=s, safety_mult=p.safety_mult,
                            base_min_hold_hours=p.base_min_hold_hours,
                            cap_min_hold_hours=p.cap_min_hold_hours),
                        total_fees=size * fee_rate * 2, entry_px=px[c][i])
                    hooks.on_open(open_[c], i)
                    decided[i] = True
            if hooks.on_bar(i, open_, done):
                decided[i] = True
            held_count[i] = len(open_)
            for c, pos in open_.items():
                j_perp, j_spot = col[c]
                price = px[c][i] if not np.isnan(px[c][i]) else px[c][pos.opened]
                spot_price = sp[c][i] if not np.isnan(sp[c][i]) else sp[c][pos.opened]
                exposure[i, j_spot] = pos.units * spot_price
                exposure[i, j_perp] = -pos.units * price

        read = pd.DataFrame(exposure / p.budget_cap_usdc, index=index, columns=columns)
        weights = read.where(pd.Series(decided, index=index), np.nan).ffill().fillna(0.0)
        # A leg with no price is held through, as live (qlab.harness.gaps);
        # anything else untradeable holds nothing in the harness.
        from qlab.harness.gaps import holdable_gaps

        gaps = holdable_gaps(panel.prices, panel.funding,
                             panel.meta.get("no_funding_instruments", ()))
        weights = weights.where(panel.tradeable | gaps, 0.0)
        extra_usd = hooks.extra_cost_usd()
        extra = (None if extra_usd is None
                 else pd.Series(extra_usd / p.budget_cap_usdc, index=index))
        return LiveFrabRun(
            weights=weights, positions=done + list(open_.values()),
            eligible=pd.Series(eligible, index=index), held=pd.Series(held_count, index=index),
            extra_costs=extra, diagnostics=hooks.diagnostics())


__all__ = ["HOURS_PER_YEAR", "FrabParams", "LiveFrab", "LiveFrabRun", "signal_at", "size_for"]
