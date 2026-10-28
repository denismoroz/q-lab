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

    spot column  (`<COIN>-SPOT`):  +units_spot * S  (+ carry_units * S)
    perp column  (`<COIN>`):        -short_size * P  (- carry_units * P)

where P is the perp's price and S the price the spot leg is valued at: P by
default, the spot's own last close under `data.spot_prices: own`
(`qlab.strategies.live._marks`).

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
  does the panel by default (spot legs are marked at their own perp,
  2026-10-02). Under `data.spot_prices: own` the panel carries the spot's own
  closes and the weights value the spot leg at them, so the spot-perp gap
  enters the harness's returns; the book itself, and `book_equity`, stay on
  the perp's price (frab's code has one price per coin).

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

## Mechanisms beyond the live book's own rules (all off by default)

The owner's ideas for Bv2 (night/research.yaml, `strategies.bv2.owner_ideas`) need things the
production books do not do. Each is a separate switch in the spec's `params`; with every switch
off the driver is the one described above, and `test_bv2_mechanisms.py` proves it equals the
pre-switch driver (`_bv2_reference.py`) to the last bit.

- **A coin list that changes over time** -- `params.universe`
  (`_bv2_mechanisms.UniverseRule`): `{rule: top_liquid, n, rebalance, by, window_days,
  candidates}`. At the first bar of each rebalance period the `n` coins with the largest trailing
  dollar volume (`by`: the spot's or the perp's) are chosen from data of bars BEFORE it, among
  coins that can open a book at that bar (both legs tradeable and priced, the startup history
  present: a coin without a perp that bar cannot be hedged, so it is not held at all, never
  held unhedged). A coin that leaves the list has its book closed at that bar's close: the hedge
  bought back and the spot sold, at frab's costs. A coin that joins starts a new book at once.
  Capital: the strategy has `n` equal slots. A staying book is not touched (its size drifts from
  an equal share by the book's own rules -- the ratchet trims growth); the money of the books that
  left, plus any idle cash, is split equally over the free slots, one share per joining book. A
  slot nobody fills (fewer than `n` coins can open a book) stays cash.
- **One margin pool for all coins** -- frab's `shared_margin` (needs a frab with it; the driver
  refuses to run without rather than silently use one pool per coin): the books' USDC backs all
  the shorts, a hedge may use the pool's free margin, the top-up trigger and the liquidation are
  asked of the sum. A pool liquidation closes every short of the pool, at the price where the
  pool's value meets its maintenance margin if all coins moved from their last close to their
  bar high together (the worst case the bar's data allows; the bar's path is unknown).
- **Top-ups that arrive late** -- frab's `topup_delay_hours`: a top-up that falls due at bar t
  takes effect at bar t + delay, sized to the losses as they stand then; the shorts are exposed
  meanwhile and can be liquidated. The number of hours is the spec's to give and to cite.
- **New money paid in** -- `params.deposit_usd` every `params.deposit_every` (both or neither):
  the money is split over the books that can take it, each invests it as its sizing says (spot
  bought at frab's cost, the rest to the HL reserve, a hedged book extends its short). A deposit
  is not return: the weights are fractions of equity, so the harness's return is time-weighted by
  construction, and the run reports `book_return`, the books' own time-weighted hourly return,
  which excludes the deposit (`(equity - deposit) / previous equity - 1`).

What the driver does NOT express stays out of the spec's reach: the carry sleeve under a pool, a
universe or deposits (refused), the top-up paid in cash from the cold wallet instead of by
selling the coin, the departing coin handed back instead of sold.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.gaps import holdable_gaps
from qlab.harness.panel import MarketPanel
from qlab.strategies.live._bv2_mechanisms import (
    close_book,
    deposit_into_book,
    parse_universe,
    period_starts,
    select_universe,
)
from qlab.strategies.live._loader import import_frab
from qlab.strategies.live._marks import spot_marks

_book = import_frab("frab.strategy.b2.book")
_params_module = import_frab("frab.strategy.b2.params")
B2Params = _params_module.B2Params

# frab/strategy/b2/engine.py:28-29. The engine module itself is not importable
# here (it pulls in the database and the exchange client); these two constants
# are its only inputs to the book.
HISTORY_BARS = 740
WARMUP_BARS = 48

#: B2Params fields that exist only in a frab checkout with the research mechanisms.
#: B2Params.from_dict drops keys it does not know, so a spec asking for one against an older frab
#: would silently run without it -- `_require_frab_support` refuses instead.
_FRAB_MECHANISM_PARAMS = ("shared_margin", "topup_delay_hours", "short_window_days",
                         "long_window_days", "partial_hedge_share")


@dataclass(frozen=True)
class LiveBv2Run:
    """Weights for the harness plus the books' own view of the same run."""

    weights: pd.DataFrame
    # sum over coins of the live books' own equity, idle cash and deposits included
    book_equity: pd.Series
    liquidations: int
    # What follows is filled by every run of this driver; the defaults are for code that builds
    # a run by hand.
    book_return: pd.Series | None = None  # time-weighted hourly return of book_equity
    deposits: pd.Series | None = None  # new money paid in at each bar (zero without deposits)
    in_universe: pd.DataFrame | None = None  # per coin: a slot is held (or waits) at the bar
    counts: Mapping[str, int] = field(default_factory=dict)  # fills by kind, joins, leaves
    log: tuple[dict, ...] = ()  # joins, leaves, deposits, liquidations: times and amounts


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


def _require_frab_support(params: Mapping[str, object]) -> None:
    unsupported = [k for k in _FRAB_MECHANISM_PARAMS
                   if k in params and k not in B2Params.__dataclass_fields__]
    if unsupported:
        raise ValueError(
            f"params {unsupported} need a newer frab checkout (shared margin pool and delayed "
            "top-ups: d06f53b; signal windows and the partial hedge: 7304000): the checkout "
            "in use ignores them, which would run a different strategy than the spec says. "
            "Point QLAB_FRAB_SRC at a checkout that has them."
        )


class _Feed:
    """One coin's data as its book sees it: the arrays the book is stepped on.

    `can_start` marks the bars from which a book may open -- both legs tradeable with a price and
    a known funding rate, and the panel already holds the history the live engine loads at
    startup."""

    def __init__(self, panel: MarketPanel, coin: str, gaps: pd.DataFrame,
                 columns: pd.Index) -> None:
        perp, spot = coin, f"{coin}{SPOT_COLUMN_SUFFIX}"
        for col in (perp, spot):
            if col not in columns:
                raise ValueError(f"Bv2 needs column {col!r}; this panel has none")
        self.coin = coin
        self.jp, self.js = columns.get_loc(perp), columns.get_loc(spot)
        self.px = panel.prices[perp].to_numpy(dtype=float)
        self.spot_px = spot_marks(panel, perp, spot)
        self.fr = panel.funding[perp].to_numpy(dtype=float)
        self.hi = panel.high[perp].to_numpy(dtype=float)
        self.ok = (
            panel.tradeable[perp].to_numpy(dtype=bool)
            & panel.tradeable[spot].to_numpy(dtype=bool)
            & ~np.isnan(self.px)
            & ~np.isnan(self.fr)
        )
        # A bar some leg is held through (no price, docs/TASKS.md T36):
        # the live engine waits with both legs open, and the harness holds
        # them -- the book is not stepped and keeps its last exposure.
        self.held = (
            (panel.tradeable[perp] | gaps[perp]).to_numpy(dtype=bool)
            & (panel.tradeable[spot] | gaps[spot]).to_numpy(dtype=bool)
            & (gaps[perp] | gaps[spot]).to_numpy(dtype=bool)
        )
        priced = ~np.isnan(self.px)
        history_before = np.cumsum(priced) - priced.astype(int)
        self.can_start = self.ok & (history_before >= HISTORY_BARS + WARMUP_BARS)
        # A universe decided at bar t+1 from bar t: the coin can open a book there if it is
        # tradeable at t and will have the startup history by t+1 (a count of past bars).
        self.selectable = self.ok & (history_before + priced >= HISTORY_BARS + WARMUP_BARS)
        valid = np.flatnonzero(priced)  # closes the live engine would hold
        self.pos = np.cumsum(priced) - 1  # a priced bar's place among them
        self.closes = self.px[valid]
        # Funding history feeds only the carry signal; an unknown rate
        # reads as 0 there, exactly like the live engine's
        # `funding.get(h, 0.0)`. A stepped bar's own rate is never
        # unknown (`ok` excludes it).
        self.rates = np.nan_to_num(self.fr[valid])


class _Slot:
    """A share of the strategy's money tied to one coin: cash until its book starts, then it."""

    def __init__(self, feed: _Feed, cash: float, external: np.ndarray | None) -> None:
        self.feed = feed
        self.cash = cash
        self.external = external
        self.book = None
        self.params = None
        self.i0 = -1  # bar the book started at
        self.last_equity = cash
        self.stepped = False  # the book was stepped on the current bar

    def value(self) -> float:
        return self.last_equity if self.book is not None else self.cash


class _Driver:
    """Steps every coin's book bar by bar and reads the weights (module docstring)."""

    def __init__(self, panel: MarketPanel, p, hedge_by, rule, deposit_usd, deposit_every) -> None:
        self.panel, self.p, self.rule = panel, p, rule
        self.index = panel.prices.index
        self.n = len(self.index)
        self.columns = panel.prices.columns
        self.coins = list(p.coins)
        self.exposure = np.zeros((self.n, len(self.columns)))
        self.equity = np.zeros(self.n)
        self.deposits = np.zeros(self.n)
        self.in_universe = np.zeros((self.n, len(self.columns)), dtype=bool)
        # Bars on which some book actually did something: a fill, a start, or
        # a pause/resume on missing data. Weights are re-read only there.
        self.decided = np.zeros(self.n, dtype=bool)
        self.liquidations = 0
        self.counts: Counter = Counter()
        self.log: list[dict] = []
        self.deposit_usd = deposit_usd
        self.deposit_bar = (None if deposit_every is None
                            else period_starts(self.index, deposit_every))
        self.pool = _book.MarginPool() if getattr(p, "shared_margin", False) else None

        externals = (_external_hedge(hedge_by, self.index, self.coins, int(p.sticky_exit_hours))
                     if hedge_by else None)
        gaps = holdable_gaps(panel.prices, panel.funding,
                             panel.meta.get("no_funding_instruments", ()))
        self.feeds = {c: _Feed(panel, c, gaps, self.columns) for c in self.coins}
        self.externals = externals
        if rule is None:
            # The coin list is fixed: every coin has its share from the first bar.
            self.idle = 0.0
            self.slots = [_Slot(self.feeds[c], p.book_capital, self._external(c))
                          for c in self.coins]
            self.rebalance = None
            self.selected = None
        else:
            self.idle = float(p.capital_usd)
            self.slots = []
            selectable = pd.DataFrame({c: self.feeds[c].selectable for c in self.coins},
                                      index=self.index)
            self.selected = select_universe(panel, rule, self.coins, selectable).to_numpy()
            self.rebalance = period_starts(self.index, rule.rebalance)

    def _external(self, coin: str):
        return self.externals[coin] if self.externals is not None else None

    # ── the run ────────────────────────────────────────────────────────────

    def go(self) -> LiveBv2Run:
        for i in range(self.n):
            if self.rebalance is not None and self.rebalance[i]:
                self._rebalance(i)
            if self.pool is not None:
                self._pool_liquidation(i)
            for slot in self.slots:
                self._advance(slot, i)
            if self.pool is not None:
                self._pool_top_up(i)
            if self.deposit_bar is not None and self.deposit_bar[i]:
                self._deposit(i)
            self.equity[i] = self._record(i)
        return self._result()

    def _result(self) -> LiveBv2Run:
        index, columns = self.index, self.columns
        # The books hold UNITS; as fractions of equity they drift every bar
        # with price although nothing trades. Handing the drift to the harness
        # would book a trade every hour (costs on turnover that never
        # happened) and make every bar a "decision" for matched noise, which
        # then redraws hourly and fails structural matching. So weights are
        # read at decision bars -- fills, starts, data pauses -- and held in
        # between: the harness sees the strategy's own decisions.
        exposure = pd.DataFrame(self.exposure, index=index, columns=columns)
        read = exposure.div(self.equity, axis=0)
        weights = read.where(pd.Series(self.decided, index=index), np.nan).ffill().fillna(0.0)
        weights = weights.where(self.panel.tradeable, 0.0)
        equity = pd.Series(self.equity, index=index)
        previous = equity.shift(1)
        deposits = pd.Series(self.deposits, index=index)
        return LiveBv2Run(
            weights=weights,
            book_equity=equity,
            liquidations=self.liquidations,
            book_return=(equity - deposits) / previous - 1.0,
            deposits=deposits,
            in_universe=pd.DataFrame(self.in_universe, index=index, columns=columns),
            counts=dict(self.counts),
            log=tuple(self.log),
        )

    # ── one slot, one bar ──────────────────────────────────────────────────

    def _advance(self, slot: _Slot, i: int) -> None:
        f, index = slot.feed, self.index
        slot.stepped = False
        started_now = False
        if slot.book is None:
            if not f.can_start[i]:
                return
            self._start(slot, i)
            started_now = True
        if not started_now:
            if not f.ok[i] and f.held[i] and i > slot.i0:
                for j in (f.js, f.jp):
                    self.exposure[i, j] = self.exposure[i - 1, j]
                return
            if not f.ok[i]:
                self.decided[i] = True
                if i + 1 < self.n:
                    self.decided[i + 1] = True
                return
        if np.isnan(f.hi[i]):
            raise ValueError(
                f"bar high unknown for {f.coin} at {index[i]}: the live book's "
                "liquidation check needs it, and the close is not a substitute"
            )
        k = f.pos[i]
        extra = {"pool": self.pool} if self.pool is not None else {}
        events = _book.step(
            slot.book,
            bar_ms=_ms(index[i]),
            price=float(f.px[i]),
            funding_rate=float(f.fr[i]),
            closes=list(f.closes[max(0, k - HISTORY_BARS + 9) : k + 1]),
            funding_hist=list(f.rates[max(0, k - 8) : k + 1]),
            params=slot.params,
            high=float(f.hi[i]),
            **extra,
        )
        if slot.external is not None and not np.isnan(slot.external[i]):
            # The hedge for the next bar comes from the regime
            # detector instead of the book's own 14/30-day rule
            # (`hedge_by`, docs/REGIME_DETECT.md).
            slot.book.hedge_prev = bool(slot.external[i])
        self._count(events, i, f.coin)
        slot.stepped = True

    def _start(self, slot: _Slot, i: int) -> None:
        f, p, index = slot.feed, self.p, self.index
        params = replace(p, coins=(f.coin,), capital_usd=slot.cash)
        book = _book.CoinBook.new(f.coin, params)
        k0 = f.pos[i]
        for k in range(max(0, k0 - WARMUP_BARS), k0):
            _book.advance_signals(
                book,
                list(f.closes[max(0, k - HISTORY_BARS + 9) : k + 1]),
                list(f.rates[max(0, k - 8) : k + 1]),
                params,
            )
        ext = slot.external
        if ext is not None and i > 0 and not np.isnan(ext[i - 1]):
            book.hedge_prev = bool(ext[i - 1])
        _book.start_book(book, bar_ms=_ms(index[i]), price=f.px[i], params=params)
        self.decided[i] = True
        slot.book, slot.params, slot.i0 = book, params, i
        slot.last_equity = book.equity(f.px[i])
        if self.pool is not None:
            self.pool.add(book, float(f.px[i]))

    def _record(self, i: int) -> float:
        """Read the exposures of the books stepped on bar i; the strategy's equity at bar i."""
        total = 0.0
        for slot in self.slots:
            f, book = slot.feed, slot.book
            if slot.stepped:
                price = float(f.px[i])
                # The spot leg's own value per unit; NaN only before the
                # spot's first print, which `ok` excludes.
                spot_price = float(f.spot_px[i])
                # The carry sleeve holds spot long and perp short in equal
                # units: `carry_units` with the margin model, the fixed
                # `carry_target` notional without it (book.py, step 5).
                if not book.carry_on:
                    carry = 0.0
                elif slot.params.margin_enabled:
                    carry = book.carry_units
                else:
                    carry = book.carry_target / price
                self.exposure[i, f.js] = (book.units_spot + carry) * spot_price
                self.exposure[i, f.jp] = -(book.hedge_units() + carry) * price
                slot.last_equity = book.equity(price)
            elif self.pool is not None and book is not None:
                # not stepped, but a pool top-up may have touched it at its last price
                slot.last_equity = book.equity(self.pool.marks[f.coin])
            if book is not None or self.rule is not None:
                self.in_universe[i, f.jp] = True
                self.in_universe[i, f.js] = True
            total += slot.value()
        if self.idle:
            total += self.idle
        return total

    def _count(self, events: list[dict], i: int, coin: str | None = None) -> None:
        for e in events:
            self.counts[e["kind"]] += 1
            if e.get("limited_by_margin"):
                self.counts["hedge_limited"] += 1  # opened smaller than the spot: margin ran short
            if e["kind"] == "liquidation":
                self.liquidations += 1
                self.log.append({"time": self.index[i], "kind": "liquidation",
                                 "coin": e.get("coin", coin), "price": e["price"],
                                 "loss": e["loss"]})
        if events:
            self.decided[i] = True

    # ── a shared margin pool ───────────────────────────────────────────────

    def _pool_liquidation(self, i: int) -> None:
        pool = self.pool
        highs: dict[str, float] = {}
        for slot in self.slots:
            f = slot.feed
            if slot.book is None:
                continue
            if f.ok[i]:
                if np.isnan(f.hi[i]):
                    raise ValueError(
                        f"bar high unknown for {f.coin} at {self.index[i]}: the live book's "
                        "liquidation check needs it, and the close is not a substitute"
                    )
                pool.marks[f.coin] = float(f.px[i])
                highs[f.coin] = float(f.hi[i])
            else:
                highs[f.coin] = pool.marks[f.coin]  # no data this bar: the last close stands
        events = _book.pool_liquidate_if_breached(pool, highs=highs, bar_ms=_ms(self.index[i]))
        self._count(events, i)

    def _pool_top_up(self, i: int) -> None:
        events = _book.pool_top_up(self.pool, bar_ms=_ms(self.index[i]), params=self.p)
        self._count(events, i)

    # ── a coin list that changes ───────────────────────────────────────────

    def _rebalance(self, i: int) -> None:
        index = self.index
        chosen = [c for c, flag in zip(self.coins, self.selected[i], strict=True) if flag]
        stay = [s for s in self.slots if s.feed.coin in chosen]
        for slot in self.slots:
            if slot.feed.coin not in chosen:
                self._leave(slot, i)
        self.slots = stay
        held = {s.feed.coin for s in stay}
        joiners = [c for c in chosen if c not in held]
        free = self.rule.n - len(stay)
        if joiners and free > 0:
            share = self.idle / free
            for coin in joiners[:free]:
                self.slots.append(_Slot(self.feeds[coin], share, self._external(coin)))
                self.idle -= share
                self.counts["universe_join"] += 1
                self.log.append({"time": index[i], "kind": "join", "coin": coin, "capital": share})
                self.decided[i] = True

    def _leave(self, slot: _Slot, i: int) -> None:
        f, book = slot.feed, slot.book
        if book is None:
            proceeds = slot.cash  # never started: its money was waiting
        else:
            price = float(f.px[i]) if not np.isnan(f.px[i]) else float(book.last_price)
            before = book.equity(price)
            proceeds = close_book(book, price, slot.params.slippage)
            if self.pool is not None:
                self.pool.remove(f.coin)
            self.log.append({"time": self.index[i], "kind": "leave", "coin": f.coin, "price": price,
                             "equity_before": before, "proceeds": proceeds})
            self.counts["universe_leave"] += 1
            self.decided[i] = True
        self.idle += proceeds

    # ── new money ──────────────────────────────────────────────────────────

    def _deposit(self, i: int) -> None:
        takers = [s for s in self.slots if s.book is None or s.stepped]
        self.deposits[i] += self.deposit_usd
        self.counts["deposit"] += 1
        self.decided[i] = True
        if not takers:
            self.idle += self.deposit_usd
            self.log.append({"time": self.index[i], "kind": "deposit", "to": "idle",
                             "amount": self.deposit_usd})
            return
        share = self.deposit_usd / len(takers)
        for slot in takers:
            if slot.book is None:
                slot.cash += share
                self.log.append({"time": self.index[i], "kind": "deposit",
                                 "coin": slot.feed.coin, "amount": share, "to": "waiting"})
                continue
            price = float(slot.feed.px[i])
            done = deposit_into_book(slot.book, share, price, slot.params, pool=self.pool)
            slot.last_equity = slot.book.equity(price)
            self.log.append({"time": self.index[i], "kind": "deposit", "coin": slot.feed.coin,
                             **done})


_RUN_CACHE: dict[tuple, LiveBv2Run] = {}
_RUN_CACHE_SIZE = 16


def _run_key(panel: MarketPanel, params: Mapping[str, object]) -> tuple:
    """What a run depends on: the panel's numbers and the parameters. The weights and the base
    (`LiveBv2.own_base`) of one run come from one pass of the driver, which on a six-year hourly
    panel takes minutes; a run is a pure function of this key, so a second asker gets the first's
    result."""
    import hashlib
    import json

    digest = hashlib.sha1()
    for frame in (panel.prices, panel.funding, panel.high, panel.volume, panel.tradeable):
        if frame is not None:
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    index = panel.prices.index
    return (panel.snapshot_id, len(index), str(index[0]), str(index[-1]),
            hash(tuple(panel.prices.columns)), digest.hexdigest(),
            json.dumps(dict(panel.meta), sort_keys=True, default=str),
            json.dumps(dict(params), sort_keys=True, default=str))


class LiveBv2:
    """See module docstring. `params` is handed unchanged to
    `B2Params.from_dict`, so frab's own validation applies -- except the keys
    the stand itself reads (`hedge_by`, `universe`, `deposit_usd`,
    `deposit_every`)."""

    name = "bv2-live"
    valid_intervals = ("1h",)  # the live book counts its windows in hourly bars
    reinvests_profit = False
    """Declared (docs/CAPITAL_BASE.md): the book's position sizes are set in dollars when it
    is created and the ratchet moves everything above them into cash, which never goes back
    to work; carry on free money is off in the production parameters. A percent of the
    strategy is therefore a sum of dollars over the money it started with, never a
    compounding of shares."""

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights.copy()

    def own_base(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.Series:
        """The book's equity at each bar over the capital the strategy started with
        (`capital_usd`): the number the stand's weights, which are shares of that equity, have
        to be multiplied by to become shares of the strategy's own money. Idle cash the ratchet
        set aside is in the equity, so the base grows while the positions stay the size they
        were set (docs/CAPITAL_BASE.md).

        Refused (`CapitalBaseRefused`) with deposits: the capital would grow with every
        payment, and what 100% then is belongs to the owner."""
        from qlab.harness.strategy import CapitalBaseRefused

        if params.get("deposit_usd") is not None or params.get("deposit_every") is not None:
            raise CapitalBaseRefused(
                "Bv2 with deposits (deposit_usd): the strategy's capital grows with every "
                "payment, so a percent of it has no single base; the owner has not decided "
                "what 100% is (docs/CAPITAL_BASE.md)")
        rest = {k: v for k, v in params.items()
                if k not in ("hedge_by", "universe", "deposit_usd", "deposit_every")}
        capital = float(B2Params.from_dict(rest).capital_usd)
        return (self.run(panel, params).book_equity / capital).rename("capital_base")

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveBv2Run:
        key = _run_key(panel, params)
        if key not in _RUN_CACHE:
            if len(_RUN_CACHE) >= _RUN_CACHE_SIZE:
                _RUN_CACHE.pop(next(iter(_RUN_CACHE)))
            _RUN_CACHE[key] = self._run(panel, params)
        return _RUN_CACHE[key]

    def _run(self, panel: MarketPanel, params: Mapping[str, object]) -> LiveBv2Run:
        params = dict(params)
        hedge_by = params.pop("hedge_by", None)
        universe = params.pop("universe", None)
        deposit_usd = params.pop("deposit_usd", None)
        deposit_every = params.pop("deposit_every", None)
        _require_frab_support(params)
        rule = None
        if universe is not None:
            rule = parse_universe(universe)
            if "coins" in params:
                raise ValueError("params.coins and params.universe both name the coins: "
                                 "with a universe the candidates are universe.candidates")
            candidates = rule.coins(panel)
            if not candidates:
                raise ValueError("params.universe: no candidate has both a perp and a spot column")
            params["coins"] = candidates
        if (deposit_usd is None) != (deposit_every is None):
            raise ValueError("deposit_usd and deposit_every go together: an amount and how often")
        if deposit_usd is not None and (
            isinstance(deposit_usd, bool) or not isinstance(deposit_usd, (int, float))
            or deposit_usd <= 0
        ):
            raise ValueError(f"deposit_usd must be a positive amount, got {deposit_usd!r}")
        p = B2Params.from_dict(params)
        if (rule is not None or deposit_usd is not None) and p.carry_enabled:
            raise ValueError("a changing coin list and deposits are not modelled with the carry "
                             "sleeve: set carry_enabled false")
        if getattr(p, "shared_margin", False) and (
            "pool" not in inspect.signature(_book.step).parameters
        ):
            raise ValueError("this frab checkout's book cannot step in a shared pool")
        amount = None if deposit_usd is None else float(deposit_usd)
        return _Driver(panel, p, hedge_by, rule, amount, deposit_every).go()


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


__all__ = ["HISTORY_BARS", "LiveBv2", "LiveBv2Run", "WARMUP_BARS"]
