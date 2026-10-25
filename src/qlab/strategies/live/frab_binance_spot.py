"""FRAB's live decisions with the spot leg bought on Binance and the short held
on Hyperliquid -- a different strategy from FRAB (owner, 2026-10-25: «можно в
qlab запусить frab на более чем 4-5 монетах. я знаю что HL ограничен в
количестве spot pairs. я бы хотел прогнать вариант когда спот открывается на
binance а хеджируется на HL. - поидее это новая стратегия.»).

The decisions are `qlab.strategies.live.frab.LiveFrab`'s, call for call: the
same signal, the same two-phase breakeven machine, the same slots, the same
sizing. What changes is where the money is and what that does:

- **Prices.** The panel is mixed (`data.spot_source: binance`): perp closes,
  bar highs and funding are Hyperliquid's, the spot leg is valued at Binance's
  own USDT close divided by Binance's own USDCUSDT close, hour by hour
  (`qlab.data.snapshot`). Nothing is marked at the perp's price: the basis
  between the venues is a return of this strategy. A coin is eligible on a bar
  only if its perp and its Binance spot pair were both trading then.
- **Fees.** The engine counts a round trip at Hyperliquid's perp taker plus
  Binance's spot taker (`perp_taker_fee`, `spot_taker_fee`), not at frab's
  Hyperliquid-spot rate, so its breakeven gate waits for the fees this
  strategy really pays.
- **Margin on Hyperliquid, spot gains on Binance.** The slot's money is the
  live split: spot notional `size` on Binance, the rest of the slot
  (`slot - size`, frab's `margin_buffer_factor`) as USDC margin on Hyperliquid.
  The spot is not collateral there. All shorts share one cross-margin account
  (as frab's watchdog assesses it): equity = margin cash + funding received +
  unrealised short P&L; maintenance = notional x `maintenance_over_initial` /
  the coin's max leverage (Hyperliquid: half of the initial margin at max
  leverage). When the ratio of equity to maintenance (what is already on its
  way counted in) falls below `top_up_trigger`, the strategy restores it to
  `healthy_ratio` the only way it can without idle cash: it sells the same
  share of the spot and buys back the same share of the short in every open
  position (so the book stays hedged), and withdraws the spot proceeds to
  Hyperliquid, where they arrive `transfer_delay_hours` later.
- **Moving money costs a flat fee.** Entry: the margin goes Binance ->
  Hyperliquid (Binance's USDC withdrawal fee on Arbitrum One, at least its
  minimum withdrawal). Each top-up: the same fee. Close: what is left of the
  position's margin goes back, Hyperliquid's withdrawal fee. These are dollars,
  not a rate on turnover, so they reach the harness as `extra_costs`
  (`qlab.harness.run.run_backtest(extra_cost=...)`), as a share of the budget.
- **Liquidation.** On every bar a position was alive for, the account is
  re-assessed at the bars' HIGHS; if equity is at or below maintenance there,
  every short is closed (counted in the diagnostics). The harness then sees the
  hedge weights drop at that bar's close, not at the liquidation price, and no
  liquidation penalty is charged: the number of liquidations is reported so a
  nonzero count is visible.

- **How money moves** (`transfers`, `TRANSFERS`): `batched` (the main spec; the owner's
  «каждая площадка держит свои деньги»), `when_needed` and `on_close`; the sizes of a batch and
  of idle money are declared parameters (`batch_share`, `idle_share`).
- **Two rules against broken pairs**: frab's own 3% parity between spot and perp at entry
  (`entry_parity_tolerance`), and a held pair that cannot be traded for `max_wait_hours` in a row is
  closed at the last marks (`pair_lost`).
- **Both legs on one venue** (`LiveFrabAllBinance`): the same ledger with Binance's USDⓈ-M
  futures as the short and free, instant transfers between the user's own wallets (fee 0, delay 0).

## What is not modelled

- **Entry delay and the entry transfer's timing.** The margin for a new
  position is on Hyperliquid at the moment of entry (a deposit made ahead of
  the decision); only the top-up path has the declared delay.
- **Idle money.** The stand's budget is a constant divisor; margin cash waiting
  in transit or left on Hyperliquid earns nothing, and Binance cash is not
  tracked (the slot's footprint is the budget's `1 / concurrency_cap`; a
  minimum withdrawal larger than a small margin is paid out of that footprint).
- **Failures and retries** of a transfer, a leg that fails to fill, lot sizes
  and Hyperliquid's own minimum deposit.
- **Liquidation penalty and the partial liquidations** of an account over a
  size threshold.
- **Maximum leverage and maintenance rates are today's** (the meta table read
  on 2026-10-25), applied to the whole history: Hyperliquid does not serve
  them as they were.
- **Hourly drift.** Like every live driver here, weights are read only on the
  bars where a position opens, closes or is resized.

## The holder's account

`LiveFrabRun.diagnostics` also carries the numbers of the ledger (top-ups,
dollars moved, lowest ratio seen, liquidations) so the report can say how often
the margin path mattered.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.live.frab import (
    FrabParams,
    LiveFrab,
    LiveFrabHooks,
    LiveFrabRun,
    _Position,
)

TRANSFERS = ("on_close", "when_needed", "batched")
"""How money moves between the two venues (``transfers``):

- ``on_close``: a closed position's margin goes back to Binance at once, one
  withdrawal fee per close (the costliest way, kept as an end of the sensitivity).
- ``when_needed``: it stays on Hyperliquid as the account's free margin, covers the
  next entry and every short's cushion, and goes back only when a spot purchase finds
  no cash on Binance; a margin short of an entry is deposited by that entry's amount.
- ``batched`` (the owner, 2026-10-25: «а если без вывода спота. просто купили - продали.
  оставили там USDC. вывод когда наберется сумма либо реинвестирование»): each venue
  keeps its own money. Spot is bought from Binance cash and sold back to it; the
  proceeds stay there and fund the next purchases. Hyperliquid's margin is deposited
  in batches of at least `batch_share` of the budget, only when its free margin cannot
  cover an entry; a top-up first moves Binance cash (a batch) and sells spot only for
  what the cash does not cover; money idle on Hyperliquid above `idle_share` of the
  budget is withdrawn down to one batch, as far as the account's margin ratio allows."""

ALL_PAIRED = "all_paired"
"""`universe: all_paired`: every coin of the panel that has a perp column and a
spot column. Whether it can be entered on a bar is the panel's `tradeable`."""

QLAB_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class SplitVenueParams:
    """The parameters of the money handling, all required: nothing is assumed
    silently. Sources are cited in the spec (docs/FRAB_BINANCE_SPOT.md)."""

    spot_taker_fee: float
    perp_taker_fee: float
    transfer_delay_hours: int
    binance_withdrawal_fee_usdc: float
    binance_withdrawal_min_usdc: float
    hl_withdrawal_fee_usdc: float
    top_up_trigger: float
    healthy_ratio: float
    maintenance_over_initial: float
    leverage_table: str
    transfers: str
    batch_share: float
    idle_share: float
    entry_parity_tolerance: float
    max_wait_hours: int

    @classmethod
    def split(cls, d: Mapping[str, object]) -> tuple[SplitVenueParams, dict[str, object]]:
        """The venue part of `d` and the rest (frab's own fields)."""
        names = cls.__dataclass_fields__
        missing = sorted(set(names) - set(d))
        if missing:
            raise ValueError(f"FRAB-on-Binance params missing: {missing}")
        venue = cls(**{k: d[k] for k in names})
        if not 1.0 < venue.top_up_trigger < venue.healthy_ratio:
            raise ValueError("need 1 < top_up_trigger < healthy_ratio")
        if venue.transfer_delay_hours < 0:
            raise ValueError("transfer_delay_hours cannot be negative (0: instant)")
        if venue.transfers not in TRANSFERS:
            raise ValueError(f"transfers must be one of {TRANSFERS}")
        if not 0 <= venue.batch_share <= venue.idle_share:
            raise ValueError("need 0 <= batch_share <= idle_share")
        return venue, {k: v for k, v in d.items() if k not in names}


def read_leverage_table(path: str | Path) -> dict[str, float]:
    """``coin -> Hyperliquid max leverage`` from a venue-facts CSV (columns
    ``coin``, ``max_leverage``), relative paths from the repo root."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = QLAB_ROOT / resolved
    with resolved.open(newline="", encoding="utf-8") as handle:
        return {row["coin"]: float(row["max_leverage"]) for row in csv.DictReader(handle)}


class _SplitVenueMargin(LiveFrabHooks):
    """Hyperliquid margin ledger with the spot on Binance. See the module
    docstring for the rules; every transfer and its fee is recorded here."""

    def __init__(self, *, p: FrabParams, v: SplitVenueParams, coins: list[str],
                 px: dict[str, np.ndarray], sp: dict[str, np.ndarray],
                 highs: dict[str, np.ndarray], rates: dict[str, np.ndarray],
                 leverage: dict[str, float], max_leverage: dict[str, float], n: int,
                 ok: dict[str, np.ndarray]) -> None:
        self.p, self.v = p, v
        self.ok = ok
        self.lost: dict[str, int] = {}  # consecutive bars a held pair could not be traded
        self.px, self.sp, self.highs, self.rates = px, sp, highs, rates
        self.leverage = leverage
        self.maint = {c: v.maintenance_over_initial / max_leverage[c] for c in coins}
        self.fee_rate = v.perp_taker_fee + v.spot_taker_fee
        self.n_bars = n
        self.extra_usd = np.zeros(n)
        self.footprint = p.budget_cap_usdc / p.concurrency_cap
        # Money already on Hyperliquid that belongs to no position (a top-up
        # that arrived), and what is on its way: (arrival bar, dollars).
        self.free = 0.0
        # The cash on Binance (starts as the whole budget; a bought spot, a margin
        # deposit and a withdrawal leave it, a sold spot and a withdrawal home enter it).
        self.cash = p.budget_cap_usdc
        self.pending: list[tuple[int, float]] = []
        self.n_open = 0  # positions open now; slots not filled yet will need cash
        self.batch = v.batch_share * p.budget_cap_usdc
        self.idle_cap = v.idle_share * p.budget_cap_usdc
        self.stats = {"top_ups": 0.0, "top_up_usd": 0.0, "liquidations": 0.0,
                      "transfer_fees_usd": 0.0, "min_ratio": float("inf"),
                      "deposits": 0.0, "returns": 0.0, "cash_shortfalls": 0.0,
                      "forced_spot_sales": 0.0, "idle_withdrawals": 0.0, "transfers": 0.0,
                      "pairs_lost": 0.0}

    # -- helpers ---------------------------------------------------------

    def _fee(self, i: int, usd: float) -> None:
        """One transfer between the venues happened at bar i and cost `usd`."""
        self.extra_usd[i] += usd
        self.stats["transfer_fees_usd"] += usd
        self.stats["transfers"] += 1

    def _funding(self, pos: _Position, i: int) -> float:
        rate, price = self.rates[pos.coin][i], self.px[pos.coin][i]
        if pos.opened >= i or np.isnan(rate) or np.isnan(price):
            return 0.0
        return pos.units * price * rate

    def _price(self, c: str, i: int, pos: _Position) -> float:
        price = self.px[c][i]
        return price if not np.isnan(price) else self.px[c][pos.opened]

    # -- the engine's events ---------------------------------------------

    def can_enter(self, coin: str, i: int) -> bool:
        """frab's own parity rule (coin_discovery.py, SPOT_PERP_PARITY_TOLERANCE): the
        spot must be within the tolerance of the perp at the moment of entry. A pair
        whose legs are far apart is a broken pair (a delisting spot quoted at a fifth of
        its perp), not a hedge."""
        spot, perp = self.sp[coin][i], self.px[coin][i]
        return bool(abs(spot / perp - 1.0) <= self.v.entry_parity_tolerance)

    def on_open(self, pos: _Position, i: int) -> None:
        self.lost[pos.coin] = 0
        margin = self.footprint - pos.size_usdc
        # Free margin already on Hyperliquid covers the position first; only the rest
        # is deposited from Binance, at least the minimum withdrawal.
        drawn = min(self.free, margin)
        self.free -= drawn
        short = margin - drawn
        deposit = 0.0
        if short > 1e-12:
            deposit = max(short, self.v.binance_withdrawal_min_usdc)
            if self.v.transfers == "batched":
                # A batch, not the entry's amount: as much as the cash on Binance allows
                # after this entry's own spot purchase.
                # ... and after the cash that slots not yet filled will need for their own
                # spot and margin: money meant for a later entry is not spare.
                reserve = (self.p.concurrency_cap - self.n_open - 1) * self.footprint
                spare = (self.cash - pos.size_usdc * (1 + self.v.spot_taker_fee) - deposit
                         - self.v.binance_withdrawal_fee_usdc - reserve)
                deposit += min(max(self.batch - deposit, 0.0), max(spare, 0.0))
        if deposit:
            self.free += deposit - short  # the part above the margin joins the free margin
            self._fee(i, self.v.binance_withdrawal_fee_usdc)
            self.stats["deposits"] += 1
        needed = (pos.size_usdc * (1 + self.v.spot_taker_fee) + deposit
                  + (self.v.binance_withdrawal_fee_usdc if deposit else 0.0))
        if self.cash < needed and self.free > 0:
            # Cash on Binance is short of this purchase: bring home from Hyperliquid
            # only what it needs.
            brought = min(self.free, needed - self.cash)
            self.free -= brought
            self.cash += brought - self.v.hl_withdrawal_fee_usdc
            self._fee(i, self.v.hl_withdrawal_fee_usdc)
            self.stats["returns"] += 1
        if self.cash < needed:
            self.stats["cash_shortfalls"] += 1
        self.cash -= needed
        self.n_open += 1
        pos.cash = margin - pos.units * pos.entry_px * self.v.perp_taker_fee

    def on_close(self, pos: _Position, i: int) -> None:
        price = self._price(pos.coin, i, pos)
        pos.cash += (self._funding(pos, i) + pos.units * (pos.entry_px - price)
                     - pos.units * price * self.v.perp_taker_fee)
        self.cash += pos.units * self.sp[pos.coin][i] * (1 - self.v.spot_taker_fee)
        self.n_open -= 1
        self.free += pos.cash  # the position's margin joins the account's free margin
        if self.v.transfers == "on_close" and pos.cash > 0:
            self.free -= pos.cash
            self.cash += pos.cash - self.v.hl_withdrawal_fee_usdc
            self._fee(i, self.v.hl_withdrawal_fee_usdc)
            self.stats["returns"] += 1
        if self.free < 0:
            # A loss beyond the margin is made good from Binance's cash, by a transfer.
            transfer = max(-self.free, self.v.binance_withdrawal_min_usdc)
            self.cash -= transfer + self.v.binance_withdrawal_fee_usdc
            self.free += transfer
            self._fee(i, self.v.binance_withdrawal_fee_usdc)

    # -- the margin path --------------------------------------------------

    def _close_all(self, open_: dict[str, _Position], done: list[_Position], i: int,
                   reason: str) -> None:
        for c in list(open_):
            pos = open_.pop(c)
            pos.closed, pos.exit = i, reason
            done.append(pos)
            self.on_close(pos, i)

    def _arrive(self, i: int) -> None:
        self.free += sum(a for t, a in self.pending if t <= i)
        self.pending = [(t, a) for t, a in self.pending if t > i]

    def on_bar(self, i: int, open_: dict[str, _Position], done: list[_Position]) -> bool:
        # A held pair that cannot be traded for longer than `max_wait_hours` (a leg
        # delisted, a print that never came back) is closed at the last marks: the live
        # engine waits on missing data, but not for ever.
        dropped = False
        for c in list(open_):
            self.lost[c] = 0 if self.ok[c][i] else self.lost.get(c, 0) + 1
            if self.lost[c] > self.v.max_wait_hours:
                pos = open_.pop(c)
                pos.closed, pos.exit = i, "pair_lost"
                done.append(pos)
                self.on_close(pos, i)
                self.stats["pairs_lost"] += 1
                dropped = True
        return self._margin_path(i, open_, done) or dropped

    def _margin_path(self, i: int, open_: dict[str, _Position], done: list[_Position]) -> bool:
        # The funding the position earned this bar goes to its cash. A position
        # the engine closed this bar was credited in `on_close`.
        for pos in open_.values():
            pos.cash += self._funding(pos, i)
        if not open_:
            self._arrive(i)
            return self._withdraw_idle(i, 0.0, 0.0)

        # 1) liquidation, at the highs of the bar the positions lived through
        alive = [pos for pos in open_.values() if pos.opened < i]
        if alive:
            equity_hi = self.free + sum(
                pos.cash + pos.units * (pos.entry_px - self._high(pos, i)) for pos in alive)
            maint_hi = sum(pos.units * self._high(pos, i) * self.maint[pos.coin] for pos in alive)
            if maint_hi > 0 and equity_hi <= maint_hi:
                self.stats["liquidations"] += 1
                self._close_all(open_, done, i, "liquidation")
                self.free = 0.0
                return True

        # What was sent a bar ago reaches the account by this bar's close, not
        # before the high that bar printed.
        self._arrive(i)

        # 2) the ratio at the close, with what is on its way counted in
        equity = self.free + sum(
            pos.cash + pos.units * (pos.entry_px - self._price(pos.coin, i, pos))
            for pos in open_.values())
        maint = sum(pos.units * self._price(pos.coin, i, pos) * self.maint[pos.coin]
                    for pos in open_.values())
        incoming = sum(a for _, a in self.pending)
        if maint <= 0:
            return False
        ratio = (equity + incoming) / maint
        self.stats["min_ratio"] = min(self.stats["min_ratio"], ratio)
        if ratio >= self.v.top_up_trigger:
            return self._withdraw_idle(i, equity, maint)

        # 3) top up. Under `batched` the cash on Binance goes first, in a batch of at
        # least `batch`; whatever it does not cover is made by selling the same share q
        # of every spot and buying back q of every short (the book stays hedged), and
        # the proceeds are sent too. q solves (equity + cash + q*A) / ((1-q)*maint) =
        # healthy. One transfer carries both, so one fee.
        from_cash = 0.0
        if self.v.transfers == "batched" and self.cash >= self.v.binance_withdrawal_min_usdc:
            deficit = self.v.healthy_ratio * maint - (equity + incoming)
            from_cash = min(max(deficit, self.batch), self.cash)
            self.cash -= from_cash
        proceeds_all = sum(pos.units * self.sp[pos.coin][i] * (1 - self.v.spot_taker_fee)
                           for pos in open_.values())
        q = (self.v.healthy_ratio * maint - (equity + incoming + from_cash)) / (
            proceeds_all + self.v.healthy_ratio * maint)
        q = min(max(q, 0.0), 1.0)
        sent = 0.0
        for pos in open_.values():
            price = self._price(pos.coin, i, pos)
            cut = q * pos.units
            pos.cash += cut * (pos.entry_px - price) - cut * price * self.v.perp_taker_fee
            sent += cut * self.sp[pos.coin][i] * (1 - self.v.spot_taker_fee)
            pos.units -= cut
            pos.size_usdc *= 1 - q
        self.pending.append(
            (i + self.v.transfer_delay_hours,
             from_cash + sent - self.v.binance_withdrawal_fee_usdc))
        self._arrive(i)  # a delay of 0 (an instant transfer) lands at once
        self._fee(i, self.v.binance_withdrawal_fee_usdc)
        self.stats["top_ups"] += 1
        self.stats["top_up_usd"] += from_cash + sent
        if q > 1e-12:
            self.stats["forced_spot_sales"] += 1
        return q > 1e-12  # positions change only when spot was sold

    def _withdraw_idle(self, i: int, equity: float, maint: float) -> bool:
        """`batched`: money idle on Hyperliquid above the cap goes home down to one
        batch, as far as the account keeps the healthy margin ratio without it."""
        if self.v.transfers != "batched" or self.free <= self.idle_cap:
            return False
        allowed = self.free - self.batch
        if maint > 0:
            allowed = min(allowed, equity - self.v.healthy_ratio * maint)
        if allowed <= self.v.hl_withdrawal_fee_usdc:
            return False
        self.free -= allowed
        self.cash += allowed - self.v.hl_withdrawal_fee_usdc
        self._fee(i, self.v.hl_withdrawal_fee_usdc)
        self.stats["returns"] += 1
        self.stats["idle_withdrawals"] += 1
        return False

    def _high(self, pos: _Position, i: int) -> float:
        high = self.highs[pos.coin][i]
        if np.isnan(high):
            if not np.isnan(self.px[pos.coin][i]):
                raise ValueError(
                    f"{pos.coin}: the bar at index {i} has a close but no high; the margin "
                    "check runs on the high and the close is not a substitute")
            return self._price(pos.coin, i, pos)  # a bar with no price: held through
        return high

    def extra_cost_usd(self) -> np.ndarray:
        return self.extra_usd

    def diagnostics(self) -> dict[str, float]:
        out = dict(self.stats)
        out["binance_cash_end"] = self.cash
        out["hl_free_margin_end"] = self.free
        if out["min_ratio"] == float("inf"):
            out["min_ratio"] = float("nan")
        return out


class LiveFrabBinanceSpot(LiveFrab):
    """See the module docstring. `params` holds `FrabParams`' fields, the
    `SplitVenueParams` fields, `universe` (a list of coins or ``all_paired``)
    and `leverage` (the coin registry's values for the coins it has; every other
    coin is sized at its Hyperliquid max leverage from `leverage_table`)."""

    name = "frab-binance-spot"
    valid_intervals = ("1h",)

    def _check_panel(self, panel: MarketPanel) -> None:
        if panel.meta.get("spot_source") is None or panel.meta.get("spot_prices") != "own":
            raise ValueError(
                "FRAB-on-Binance runs on a mixed panel (data.spot_source with "
                "data.spot_prices: own); this panel's spot legs are not another venue's")

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights

    def extra_costs(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.Series:
        run = self.run(panel, params)
        assert run.extra_costs is not None
        return run.extra_costs

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveFrabRun:
        self._check_panel(panel)
        params = dict(params)
        universe = params.pop("universe")
        live_leverage = {str(k): float(v) for k, v in dict(params.pop("leverage")).items()}
        venue, rest = SplitVenueParams.split(params)
        p = FrabParams.from_dict(rest)
        table = read_leverage_table(venue.leverage_table)
        leverage = {**table, **live_leverage}
        if universe == ALL_PAIRED:
            wanted = [c for c in panel.prices.columns
                      if not str(c).endswith("-SPOT") and f"{c}-SPOT" in panel.prices.columns]
        else:
            wanted = [str(c) for c in universe]
        # A coin we cannot size or assess (no leverage in the table) cannot be entered.
        coins = [c for c in wanted if c in leverage and c in table]
        unsized = sorted(set(wanted) - set(coins))
        n = len(panel.prices.index)

        def make_hooks(cs, px, sp, highs, rates, ok):
            return _SplitVenueMargin(p=p, v=venue, coins=cs, px=px, sp=sp, highs=highs,
                                     rates=rates, leverage=leverage, max_leverage=table, n=n,
                                     ok=ok)

        result = self._simulate(panel, p, coins, leverage, make_hooks)
        result.diagnostics["coins_without_leverage"] = float(len(unsized))
        result.diagnostics["coins_in_universe"] = float(len(coins))
        return result


class LiveFrabAllBinance(LiveFrabBinanceSpot):
    """The same decisions with BOTH legs on Binance (owner, 2026-10-25: «да, добавляй
    вариант всё на binance»): the spot bought on Binance's spot market, the short on its
    USDⓈ-M futures, the signal and the accrual from Binance's funding. No money crosses
    venues, so the venue params are Binance's: taker fees of the spot market and of the
    futures, margin from the futures wallet, and transfers between a user's own spot
    and futures wallets taken as free and instant (`binance_withdrawal_fee_usdc` 0,
    `transfer_delay_hours` 0; the pages read in docs/FRAB_BINANCE_SPOT.md describe the
    transfer and name no fee, and the free-and-instant claim is not confirmed by
    Binance's own text: a declared assumption).

    The margin account is the plain mode: spot holdings are NOT collateral for the
    short (Portfolio Margin / unified account, which can count them, is a what-if,
    not modelled). Maintenance is `maintenance_over_initial` / max leverage, both read
    from the same flat values Binance's exchangeInfo publishes for every symbol
    (`seed/venue_facts/2026-10-25-binance_um_margin.csv`): per-symbol leverage
    brackets are only served to a signed request, so this is a flat 20x / 2.5%.

    Funding on Binance settles every 8 hours (or 4, or 1): the panel carries each
    settlement on its bar and zero between, so the mean of the last 8 hourly values
    times 8760 is the annualised rate, which is the signal frab's engine reads."""

    name = "frab-all-binance"

    def _check_panel(self, panel: MarketPanel) -> None:
        if (panel.meta.get("venue") != "binance" or panel.meta.get("spot_source") is not None
                or panel.meta.get("spot_prices") != "own"):
            raise ValueError(
                "FRAB-all-on-Binance runs on a Binance panel with Binance's own spot "
                "(data.source: binance, include_spot, data.spot_prices: own)")


__all__ = ["ALL_PAIRED", "LiveFrabAllBinance", "LiveFrabBinanceSpot", "SplitVenueParams",
           "read_leverage_table"]
