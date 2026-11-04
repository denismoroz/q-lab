"""How a strategy works on different money (owner, 2026-11-03).

The owner, looking at the ideas that passed and wait for a decision: «всем им нужно около 40к -
вот это жесть. а на меньших деньгах - не работает? можно добавить как работает на 1к 3к? почему
вот прям 40к нужно», and then «а если меньшее количество позиций открывать - только вот на эту
сумму».

The strict number (`min_capital_usd`, docs/MIN_CAPITAL.md) answers one question: from what sum
EVERY position the strategy ever wanted fits the venue's minimum order. It is set by one rare tiny
position, and it says nothing about what the strategy earns on less. This module is the READING
that does: the same book run on a ladder of sums, the way the venue would really treat it. It
decides nothing and moves no route. Step 4 of the ladder keeps the strict rule for runs judged
before rules 2026-11-04.2; from that version it reads the «fewer» book on the strategy's own money,
measured at evaluation time on the judged part (`qlab.money_fit`, the same models as here).

Two execution models, for a strategy whose code only produces target weights (trend, XSMOM, the
generated ideas; nothing on the stand skips an order below the venue's minimum, so the stand's
return is the same at every sum):

* `skip` -- the book wants what it wants and the venue refuses what it refuses. A target
  position whose dollar size at the sum is under the instrument's minimum order is not opened; a
  change of an existing position smaller than the minimum is not sent (the position stays as it
  was); closing a position is always sent. Reduce-only orders are exempt from the minimum on
  both venues, but how far differs (`VenueConfig.reduce_only_exempt`): Hyperliquid exempts only
  exactly closing a position, Binance every reduce-only order (so a trim is sent on Binance and
  waits on Hyperliquid). Costs are the stand's own, charged on the position change that was sent.
* `fewer` -- the strategy deliberately holds FEWER positions at this sum. At each bar the target
  positions whose dollar size is under the minimum are dropped, the smallest first, and their
  weight is given to the remaining ones of the same side in proportion, so the book stays
  invested at the strategy's own gross exposure on each side (a hedged book keeps its hedge);
  no position grows past the largest the strategy itself ever takes. Then the same order rule
  as `skip`. This is a DIFFERENT strategy from the measured one (the docstring of
  `qlab.harness.metrics.min_capital_usd`): if it looks good, its own spec is judged by the rules.
  Nothing here creates it.

A strategy that drives its own book (`placed_orders`: Bv2, FRAB, spot with a hedge) already does
what the venue asks inside frab's code, with its money in its params. It is not modelled: its
own code is run again at each sum, as `qlab.capital_floor` does, and the reading is what the book
did. Its `fewer` row is absent: the book itself decides what it opens.

Every figure is on the strategy's own money (docs/CAPITAL_BASE.md): the sum is the strategy's
capital, a percent is a share of it. The stand's weights are rebuilt into executed weights, which
go through the stand's own `run_backtest`, so the costs, the funding and the price gaps are the
stand's, and with no minimum the model returns exactly the stand's numbers (a test).

One thing is circular: the dollar size of an order depends on the money at work, which depends on
the profit already made. The sizing money is taken from the unconstrained book first and
refined by running the constrained book three times; the passes agree to the fourth digit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlab.harness.metrics import ZERO_WEIGHT_TOL, cap_key, capital_metrics
from qlab.harness.run import run_backtest

#: The sums of the ladder; the strategy's own money is added to it (owner, 2026-11-03).
SUMS: tuple[float, ...] = (1_000.0, 3_000.0, 10_000.0)
#: The sum a book-driving strategy is run on to get "the book that never meets a minimum".
REFERENCE_BOOK_USD = 100_000.0
MODES = ("skip", "fewer", "book")
#: Where the readings are kept, one file per trial; a stored reading is reused only for the same
#: trial, the same own money and the same method.
STORE = Path("data/money_ladders")
METHOD_VERSION = "2026-11-03.1"
#: How many times the constrained book is re-run to settle the money at work.
PASSES = 3
#: A fraction of a cent: an order of exactly the minimum is not refused for floating-point dust.
TOLERANCE_USD = 1e-9
METRIC_PREFIX = "ml_"


# --------------------------------------------------------------------------
# The execution models (pure, on arrays)


def execute_orders(targets: np.ndarray, sizing: np.ndarray, minimums: np.ndarray, *,
                   trims_allowed: bool) -> np.ndarray:
    """The positions the book really holds when the venue refuses orders under the minimum.

    `targets` (bars x instruments) are weights, shares of the sizing money; `sizing` (bars) is the
    dollars those shares are of at each bar; `minimums` (instruments) is each instrument's
    minimum order in dollars; `trims_allowed`: whether the venue lets a partial reduce-only
    order through below the minimum (Binance) or only exactly closing a position (Hyperliquid).

    At each bar, for each instrument, with `p` the position held and `t` the target:

    * `t` is flat: the position is closed (always sent);
    * `p` is flat: opened at `t` when `|t| * sizing >= minimum`, else stays flat;
    * the sign of `t` is opposite to `p`: closed (always), and the opposite position is opened
      at `t` when it fits, else the book is flat;
    * same side: moved to `t` when the change is at least the minimum; a smaller change is not
      sent, the position stays as it was -- except a reduction on a venue that lets a trim of
      any size through.
    """
    bars, n = targets.shape
    held = np.zeros(n)
    out = np.empty_like(targets, dtype=float)
    for i in range(bars):
        target, money = targets[i], sizing[i]
        flat_target = np.abs(target) <= ZERO_WEIGHT_TOL
        flat_held = np.abs(held) <= ZERO_WEIGHT_TOL
        fits = np.abs(target) * money >= minimums - TOLERANCE_USD
        new = held.copy()
        new[flat_target] = 0.0
        live = ~flat_target
        opening = live & flat_held
        flipping = live & ~flat_held & (np.sign(target) != np.sign(held))
        new[opening] = np.where(fits, target, 0.0)[opening]
        new[flipping] = np.where(fits, target, 0.0)[flipping]
        same = live & ~flat_held & ~flipping
        big = np.abs(target - held) * money >= minimums - TOLERANCE_USD
        trim = trims_allowed & (np.abs(target) < np.abs(held))
        send = same & (big | trim)
        new[send] = target[send]
        held = new
        out[i] = held
    return out


def fewer_positions(targets: np.ndarray, sizing: np.ndarray, minimums: np.ndarray,
                    cap: float | None = None) -> np.ndarray:
    """The targets when the strategy holds only the positions that fit the sum (pure).

    At each bar the positions whose dollar size is under their minimum are dropped, the smallest
    first, one at a time; after each drop the survivors of each side (longs, shorts) are scaled
    in proportion so that the side keeps the gross exposure the strategy asked for, and no
    position is scaled past `cap` (the largest weight the strategy itself ever takes; the rest is
    left in cash). A side left without positions is left in cash. The check is then repeated on
    the scaled sizes: a position that grew past its minimum stays.
    """
    out = targets.astype(float).copy()
    sizes = np.abs(out) * sizing[:, None]
    bad_rows = np.flatnonzero(((np.abs(out) > ZERO_WEIGHT_TOL)
                               & (sizes < minimums - TOLERANCE_USD)).any(axis=1))
    for i in bad_rows:
        original = targets[i].astype(float)
        money = sizing[i]
        side = np.sign(original)
        gross = {s: np.abs(original[side == s]).sum() for s in (-1.0, 1.0)}
        alive = np.abs(original) > ZERO_WEIGHT_TOL
        row = original.copy()
        while True:
            size = np.abs(row) * money
            bad = alive & (size < minimums - TOLERANCE_USD)
            if not bad.any():
                break
            smallest = np.flatnonzero(bad)[np.argmin(size[bad])]
            alive[smallest] = False
            row = np.zeros_like(original)
            for s in (-1.0, 1.0):
                mine = alive & (side == s)
                total = np.abs(original[mine]).sum()
                if total > 0:
                    scaled = np.abs(original[mine]) * (gross[s] / total)
                    if cap is not None:
                        scaled = np.minimum(scaled, cap)
                    row[mine] = s * scaled
        out[i] = row
    return out


# --------------------------------------------------------------------------
# The readings


@dataclass
class Row:
    """One sum, one mode: what the strategy did with this money."""

    usd: float
    mode: str  # skip | fewer | book
    own: bool = False  # the sum is the strategy's own money
    available: bool = True
    why_not: str | None = None
    ann_return: float | None = None  # on the strategy's own money
    max_dd: float | None = None
    orders_wanted: int | None = None
    orders_sent: int | None = None
    not_sent_share: float | None = None  # of the orders the unconstrained book wanted
    not_sent_volume_share: float | None = None  # the same, of the volume those orders trade
    missing_position_share: float | None = None  # held by the unconstrained book, not here
    avg_positions: float | None = None
    exposure_kept: float | None = None  # gross exposure held over the gross exposure wanted
    correlation: float | None = None  # daily returns against the unconstrained book
    below_minimum_share: float | None = None  # wanted positions under the minimum here


@dataclass
class Strict:
    """What sets the strict minimum: the one position that needs the most money."""

    min_capital_usd: float | None = None
    instrument: str | None = None
    day: str | None = None
    weight: float | None = None  # the position's share of the strategy's money
    minimum_order: float | None = None
    from_orders: bool = False
    venue: str | None = None
    positions_over_own: int | None = None  # held positions that need more than the own money
    positions_total: int | None = None


@dataclass
class MoneyLadder:
    """The reading of one trial on every sum (docs/MONEY_LADDER.md)."""

    trial_id: int
    idea_id: str
    venue: str
    interval: str
    window: tuple[str, str]
    own_usd: float | None
    basis: str  # weights | book
    reference: dict[str, float | None]
    rows: list[Row]
    strict: Strict
    note: str = ""
    method: str = METHOD_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MoneyLadder:
        return cls(**{**data, "window": tuple(data["window"]),
                      "rows": [Row(**r) for r in data["rows"]], "strict": Strict(**data["strict"])})

    def row(self, usd: float, mode: str) -> Row | None:
        return next((r for r in self.rows if r.usd == usd and r.mode == mode), None)

    def metrics(self) -> dict[str, float]:
        """The reading as trial metrics: `ml_<sum>_<mode>_<name>` (sums without cents), numbers
        only, an undefined one left out."""
        out: dict[str, float] = {}
        for r in self.rows:
            if not r.available:
                continue
            base = f"{METRIC_PREFIX}{int(r.usd)}_{r.mode}_"
            for name in ("ann_return", "max_dd", "not_sent_share", "not_sent_volume_share",
                         "missing_position_share", "avg_positions", "exposure_kept", "correlation",
                         "below_minimum_share"):
                value = getattr(r, name)
                if value is not None and np.isfinite(value):
                    out[base + name] = float(value)
        for name in ("ann_return", "max_dd", "avg_positions"):
            value = self.reference.get(name)
            if value is not None and np.isfinite(value):
                out[f"{METRIC_PREFIX}reference_{name}"] = float(value)
        if self.own_usd is not None:
            out[f"{METRIC_PREFIX}own_usd"] = float(self.own_usd)
        return out


def daily(returns: pd.Series) -> pd.Series:
    """Per-bar returns compounded into UTC days."""
    return (1.0 + returns).groupby(returns.index.normalize()).prod() - 1.0


def correlation(mine: pd.Series, other: pd.Series) -> float | None:
    both = pd.concat([daily(mine), daily(other)], axis=1, join="inner").dropna()
    if len(both) < 2 or both.iloc[:, 0].std() <= 0 or both.iloc[:, 1].std() <= 0:
        return None
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def order_counts(wanted_weights: np.ndarray, held: np.ndarray
                 ) -> tuple[int, int, float | None, float | None]:
    """(orders the unconstrained book wanted, of them sent, share not sent, share of the traded
    volume not traded). A wanted order is a
    change of the unconstrained book's own position at a bar; it is sent when the constrained
    book changed that instrument at the same bar. Orders the constrained book sent at other
    bars (catching up) are not counted as wanted."""
    prev_w = np.vstack([np.zeros((1, wanted_weights.shape[1])), wanted_weights[:-1]])
    prev_h = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    wanted = np.abs(wanted_weights - prev_w) > ZERO_WEIGHT_TOL
    sent = np.abs(held - prev_h) > ZERO_WEIGHT_TOL
    n_wanted = int(wanted.sum())
    n_sent = int((wanted & sent).sum())
    volume = np.abs(wanted_weights - prev_w)
    total = float(volume[wanted].sum())
    unsent = float(volume[wanted & ~sent].sum())
    return (n_wanted, n_sent, None if n_wanted == 0 else 1.0 - n_sent / n_wanted,
            None if total == 0 else unsent / total)


def read_row(usd: float, mode: str, *, own: bool, result, reference_result, held: np.ndarray,
             wanted: np.ndarray, minimums: np.ndarray | None, money: np.ndarray | None
             ) -> Row:
    """One row from a constrained run (`result`), the unconstrained run (`reference_result`) and
    the positions held / wanted (bars x instruments)."""
    caps = capital_metrics(result)
    n_wanted, n_sent, not_sent, not_sent_volume = order_counts(wanted, held)
    wanted_cells = np.abs(wanted) > ZERO_WEIGHT_TOL
    held_cells = np.abs(held) > ZERO_WEIGHT_TOL
    missing = (None if not wanted_cells.any()
               else float((wanted_cells & ~held_cells).sum() / wanted_cells.sum()))
    gross_wanted = float(np.abs(wanted).sum())
    below = None
    if minimums is not None and money is not None and wanted_cells.any():
        below = float((wanted_cells & (np.abs(wanted) * money[:, None] < minimums)).sum()
                      / wanted_cells.sum())
    return Row(
        usd=float(usd), mode=mode, own=own,
        ann_return=_finite(caps.get(cap_key("ann_return_net"))),
        max_dd=_finite(caps.get(cap_key("max_dd"))),
        orders_wanted=n_wanted, orders_sent=n_sent, not_sent_share=not_sent,
        not_sent_volume_share=not_sent_volume,
        missing_position_share=missing, avg_positions=float(held_cells.sum(axis=1).mean()),
        exposure_kept=None if gross_wanted == 0 else float(np.abs(held).sum() / gross_wanted),
        correlation=correlation(result.net_return, reference_result.net_return),
        below_minimum_share=below)


def _finite(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


# --------------------------------------------------------------------------
# One trial


def sizing_money(usd: float, base: pd.Series | None, index: pd.DatetimeIndex,
                 equity: np.ndarray | None) -> np.ndarray:
    """The dollars the book's weights are shares of at each bar: the strategy's money times its
    own base when it declares one (`own_base`), the money at work (`equity`, the book's
    compounded equity as a multiple of the start) for a book that puts its profit back, else the
    money itself."""
    if base is not None:
        return usd * base.reindex(index).ffill().bfill().fillna(1.0).to_numpy(dtype=float)
    if equity is not None:
        return usd * equity
    return np.full(len(index), float(usd))


def start_of_bar_equity(net_return: pd.Series) -> np.ndarray:
    """The book's equity at the START of each bar as a multiple of the money it started with."""
    growth = (1.0 + net_return).cumprod().to_numpy(dtype=float)
    return np.concatenate([[1.0], np.maximum(growth[:-1], 1e-9)])


def run_weights(part_panel, frame: pd.DataFrame, costs, extra_cost, capital_kwargs):
    return run_backtest(part_panel, frame, costs, part_panel.funding, extra_cost=extra_cost,
                        **capital_kwargs)


def constrained_run(usd: float, mode: str, *, part_panel, weights: pd.DataFrame, costs,
                    extra_cost, capital_kwargs: Mapping[str, Any], base: pd.Series | None,
                    reinvested: bool, minimums: np.ndarray, trims_allowed: bool,
                    reference_result) -> tuple[Any, np.ndarray, np.ndarray]:
    """(the stand's result of the executed book, its positions, the sizing money) at `usd`.

    The executed weights go through `run_backtest`, so costs, funding and gaps are the stand's.
    The sizing money is settled by running the book `PASSES` times."""
    target = weights.to_numpy(dtype=float)
    index = weights.index
    equity = start_of_bar_equity(reference_result.net_return.reindex(index).fillna(0.0)) \
        if reinvested and base is None else None
    cap = float(np.abs(target).max()) if target.size else None
    result, held, money = None, None, None
    for _ in range(PASSES):
        money = sizing_money(usd, base, index, equity)
        goals = fewer_positions(target, money, minimums, cap) if mode == "fewer" else target
        held = execute_orders(goals, money, minimums, trims_allowed=trims_allowed)
        result = run_weights(part_panel, pd.DataFrame(held, index=index, columns=weights.columns),
                             costs, extra_cost, capital_kwargs)
        if equity is None:
            break
        equity = start_of_bar_equity(result.net_return.reindex(index).fillna(0.0))
    return result, held, money


def strict_origin(weights: pd.DataFrame, minimums: np.ndarray, *, from_orders: bool,
                  venue: str, own_usd: float | None) -> Strict:
    """The one position that sets the strict minimum: the cell with the largest
    `minimum order / share of the money` (the formula of `min_capital_usd`)."""
    cells = np.abs(weights.to_numpy(dtype=float))
    held = np.where(np.isnan(cells) | (cells <= ZERO_WEIGHT_TOL), np.nan, cells)
    if np.isnan(held).all():
        return Strict(from_orders=from_orders, venue=venue)
    need = minimums[np.newaxis, :] / held
    flat = np.nanargmax(need)
    row, col = np.unravel_index(flat, need.shape)
    over = None if own_usd is None else int(np.nansum(need > own_usd))
    return Strict(min_capital_usd=float(need[row, col]), instrument=str(weights.columns[col]),
                  day=str(weights.index[row].date()), weight=float(held[row, col]),
                  minimum_order=float(minimums[col]), from_orders=from_orders, venue=venue,
                  positions_over_own=over, positions_total=int(np.sum(~np.isnan(held))))


def ladder_sums(own_usd: float | None, sums: Sequence[float] = SUMS) -> list[tuple[float, bool]]:
    """The sums to read, ascending, each with whether it is the strategy's own money (a sum
    that equals the own money is read once and marked)."""
    out = {float(s): False for s in sums}
    if own_usd:
        out[float(own_usd)] = True
    return sorted(out.items())


def measure_weights(*, trial_id: int, idea_id: str, venue: str, interval: str,
                    part_panel, weights: pd.DataFrame, costs, extra_cost, capital,
                    minimums: np.ndarray, trims_allowed: bool, own_usd: float | None,
                    sums: Sequence[float] = SUMS) -> MoneyLadder:
    """The reading of a strategy that only produces weights (pure of the registry)."""
    capital_kwargs = capital.kwargs
    reference = run_weights(part_panel, weights, costs, extra_cost, capital_kwargs)
    target = weights.to_numpy(dtype=float)
    ref_caps = capital_metrics(reference)
    rows: list[Row] = []
    for usd, own in ladder_sums(own_usd, sums):
        for mode in ("skip", "fewer"):
            result, held, money = constrained_run(
                usd, mode, part_panel=part_panel, weights=weights, costs=costs,
                extra_cost=extra_cost, capital_kwargs=capital_kwargs, base=capital.base,
                reinvested=capital.reinvested, minimums=minimums, trims_allowed=trims_allowed,
                reference_result=reference)
            rows.append(read_row(usd, mode, own=own, result=result, reference_result=reference,
                                 held=held, wanted=target, minimums=minimums, money=money))
    index = weights.index
    window = (str(index[0].date()), str(index[-1].date())) if len(index) else ("", "")
    return MoneyLadder(
        trial_id=trial_id, idea_id=idea_id, venue=venue, interval=interval, window=window,
        own_usd=own_usd, basis="weights",
        reference={"ann_return": _finite(ref_caps.get(cap_key("ann_return_net"))),
                   "max_dd": _finite(ref_caps.get(cap_key("max_dd"))),
                   "avg_positions": float((np.abs(target) > ZERO_WEIGHT_TOL).sum(axis=1).mean())},
        rows=rows, strict=strict_origin(weights, minimums, from_orders=False, venue=venue,
                                        own_usd=own_usd))


def measure_book(*, trial_id: int, idea_id: str, venue: str, interval: str, window: tuple[str, str],
                 own_usd: float | None, run_at, sums: Sequence[float] = SUMS) -> MoneyLadder:
    """The reading of a strategy that drives its own book: `run_at(usd)` runs its code with that
    money and returns `(result, weights, orders)` on the judged rows (as in
    `qlab.capital_floor`); the reference is the same book on `REFERENCE_BOOK_USD`."""
    reference, ref_weights, ref_orders = run_at(REFERENCE_BOOK_USD)
    wanted = ref_weights.to_numpy(dtype=float)
    ref_caps = capital_metrics(reference)
    rows: list[Row] = []
    for usd, own in ladder_sums(own_usd, sums):
        result, weights, orders = run_at(usd)
        held = weights.reindex_like(ref_weights).fillna(0.0).to_numpy(dtype=float)
        row = read_row(usd, "book", own=own, result=result, reference_result=reference,
                       held=held, wanted=wanted, minimums=None, money=None)
        # The book's own count: orders that opened a leg, and the ones it wanted and did not
        # place for being under the minimum (its own rule, `PlacedOrders.skipped`).
        if orders is not None:
            opened = int((orders.sizes.abs() > ZERO_WEIGHT_TOL).to_numpy().sum())
            skipped = int(orders.skipped.sum())
            row.orders_sent, row.orders_wanted = opened, opened + skipped
            row.not_sent_share = None if opened + skipped == 0 else skipped / (opened + skipped)
        rows.append(row)
    for usd, own in ladder_sums(own_usd, sums):
        rows.append(Row(usd=usd, mode="fewer", own=own, available=False,
                        why_not="книга сама решает, какие позиции открывать"))
    return MoneyLadder(
        trial_id=trial_id, idea_id=idea_id, venue=venue, interval=interval, window=window,
        own_usd=own_usd, basis="book",
        reference={"ann_return": _finite(ref_caps.get(cap_key("ann_return_net"))),
                   "max_dd": _finite(ref_caps.get(cap_key("max_dd"))),
                   "avg_positions": float((np.abs(wanted) > ZERO_WEIGHT_TOL).sum(axis=1).mean())},
        rows=rows, strict=Strict(venue=venue, from_orders=True),
        note="книга сама не ставит заявки меньше минимума; сравнение с той же книгой на "
             f"${REFERENCE_BOOK_USD:,.0f}")


# --------------------------------------------------------------------------
# From the registry


def book_runner(code_ref: str, params: Mapping[str, Any], panel, part_panel, costs, lo: int,
                hi: int):
    """`run_at(usd)` for `measure_book`: the strategy's own code run with every money key of its
    params set to `usd` (`qlab.capital_floor.with_money`), backtested on rows `lo..hi`; returns
    `(result, weights, orders)` of those rows."""
    from qlab.capital_floor import with_money
    from qlab.harness.strategy import capital_setup, own_costs, own_orders
    from qlab.pipeline.evaluate import resolve_strategy

    def run_at(usd: float):
        money_params, _ = with_money(params, usd)
        strategy = resolve_strategy(code_ref)
        weights = strategy.target_weights(panel, money_params)
        extra = own_costs(strategy, panel, money_params)
        setup = capital_setup(strategy, panel, money_params)
        placed = own_orders(strategy, panel, money_params)
        w = weights.iloc[lo:hi + 1]
        result = run_backtest(part_panel, w, costs, part_panel.funding,
                              extra_cost=extra, **setup.kwargs)
        if placed is not None:
            placed = type(placed)(sizes=placed.sizes.iloc[lo:hi + 1],
                                  skipped=placed.skipped.iloc[lo:hi + 1])
        return result, w, placed

    return run_at


def measure_trial(session, trial_id: int, own_usd: float | None = None) -> MoneyLadder:
    """The reading of one stored trial: its spec and data re-created (`qlab.anomalies.rerun`),
    the judged rows, the venue it is judged for. `own_usd` is the idea's own money when the
    ledgers know it (a second stage has its first stage's), else the spec's. Raises
    `RunUnavailable` when the trial cannot be re-run."""
    from qlab.anomalies.rerun import load_run
    from qlab.capital_floor import with_money
    from qlab.harness.strategy import capital_setup, own_orders
    from qlab.pipeline.evaluate import resolve_strategy
    from qlab.venues.config import leg_floors, load_venue

    run = load_run(session, trial_id)
    spec = run.spec
    venue_id = run.judged_for
    venue = load_venue(venue_id)
    columns = list(run.weights.columns)
    floors = leg_floors(venue, columns)
    minimums = np.array([(floors or {}).get(str(c), spec.min_leg_notional) for c in columns],
                        dtype=float)
    trims = venue is not None and venue.reduce_only_exempt == "any_reduction"
    own = own_usd or spec.strategy_capital_usd
    part_panel, part_weights = run.part()
    lo, hi = run.judged
    probe = own_orders(resolve_strategy(spec.code_ref), run.panel, spec.params)
    _, has_money = with_money(spec.params, 1.0)
    if probe is not None and has_money:
        run_at = book_runner(spec.code_ref, spec.params, run.panel, part_panel, run.costs, lo, hi)
        index = part_weights.index
        out = measure_book(trial_id=trial_id, idea_id=spec.idea_id, venue=venue_id,
                           interval=spec.data.interval,
                           window=(str(index[0].date()), str(index[-1].date())), own_usd=own,
                           run_at=run_at)
        out.strict = _strict_of_orders(probe, lo, hi, minimums, columns, venue_id, own, run)
        return out
    capital = capital_setup(resolve_strategy(spec.code_ref), run.panel, spec.params)
    if capital.refused is not None:
        capital = type(capital)(None, capital.reinvested)
    out = measure_weights(
        trial_id=trial_id, idea_id=spec.idea_id, venue=venue_id, interval=spec.data.interval,
        part_panel=part_panel, weights=part_weights, costs=run.costs, extra_cost=run.extra_cost,
        capital=capital, minimums=minimums, trims_allowed=trims, own_usd=own)
    if probe is not None:
        out.note = ("в параметрах стратегии нет денег: заявки книги не пересчитать, "
                    "ступени читаются по весам")
    return out


def _strict_of_orders(probe, lo, hi, minimums, columns, venue_id, own, run) -> Strict:
    """The strict minimum of a book that declares its orders: the order that opened a leg with
    the least money (docs/MIN_CAPITAL.md)."""
    sizes = probe.sizes.iloc[lo:hi + 1]
    if not (sizes.abs() > ZERO_WEIGHT_TOL).to_numpy().any():
        sizes = probe.sizes
    return strict_origin(sizes, minimums, from_orders=True, venue=venue_id, own_usd=own)


# --------------------------------------------------------------------------
# Words (plain Russian, written by code from the numbers)


def _usd(value: float) -> str:
    return "$" + f"{value:,.0f}".replace(",", " ")


def _pct(value: float | None, digits: int = 1, sign: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value * 100:+.{digits}f}" if sign else f"{value * 100:.{digits}f}"
    return text.replace("-", "−") + "%"


def venue_ru(venue: str) -> str:
    return {"binance": "Binance", "hyperliquid": "Hyperliquid"}.get(venue, venue)


def why_strict_ru(ladder: MoneyLadder) -> str:
    """One plain line: which position sets the strict minimum, and on which day."""
    s = ladder.strict
    if s.min_capital_usd is None or s.instrument is None:
        return "Что задаёт строгий минимум, не определено: ни одной позиции в прогоне."
    what = "заявка, открывшая позицию," if s.from_orders else "позиция"
    tail = ""
    if s.positions_over_own is not None and s.positions_total:
        tail = (f" Таких позиций, которым нужно больше денег стратегии, "
                f"{s.positions_over_own} из {s.positions_total} за весь прогон.")
    return (f"Строгий минимум {_usd(s.min_capital_usd)} задаёт одна позиция: {s.instrument} на "
            f"{venue_ru(s.venue or ladder.venue)}, минимальный ордер {_usd(s.minimum_order or 0)}, "
            f"{what} {s.weight * 100:.3g}% денег стратегии {s.day}: чтобы она была не меньше "
            f"минимума, нужно {_usd(s.min_capital_usd)}.{tail}")


def sentence_ru(ladder: MoneyLadder, usd: float) -> str | None:
    """«на $3 000 не отправлено 0.3% ордеров, доход +4.8% в год против +4.9% без ограничения»."""
    row = ladder.row(usd, "skip") or ladder.row(usd, "book")
    if row is None or not row.available or row.ann_return is None:
        return None
    ref = ladder.reference.get("ann_return")
    sent = ""
    if row.not_sent_share is not None:
        sent = f"не отправлено {_pct(row.not_sent_share)} ордеров"
        if row.not_sent_volume_share is not None and ladder.basis == "weights":
            sent += f" ({_pct(row.not_sent_volume_share)} объёма торговли)"
        sent += ", "
    against = (f"у той же книги на {_usd(REFERENCE_BOOK_USD)}" if ladder.basis == "book"
               else "без ограничения")
    return (f"на {_usd(usd)} {sent}доход {_pct(row.ann_return, sign=True)} в год против "
            f"{_pct(ref, sign=True)} {against}")


def money_line_ru(ladder: MoneyLadder, usd: float | None = None) -> str | None:
    """The sentence for the ladder's money step: at the strategy's own money, else at $3 000."""
    target = usd if usd is not None else (ladder.own_usd or 3_000.0)
    return sentence_ru(ladder, target)


def _cell(row: Row | None) -> str:
    if row is None or not row.available or row.ann_return is None:
        return "—"
    missing = ("" if row.missing_position_share is None
               else f", позиций нет {_pct(row.missing_position_share, 0)}")
    return f"{_pct(row.ann_return, 1, True)} / {_pct(row.max_dd, 0)}{missing}"


def report_lines(ladders: Sequence[Any]) -> list[str]:
    """The morning report's table for the ideas that passed every rule and wait for the owner
    (statuses validated, paper, live): on each venue the strict minimum and what the strategy did
    on $1 000, $3 000, $10 000 and its own money, in the two ways of executing. A reading."""
    from qlab.ladder import GATE_STATUSES

    waiting = {s.value for s in GATE_STATUSES}
    rows: list[str] = []
    for found in ladders:
        if found.status not in waiting:
            continue
        for money in found.steps[3].money:
            reading = money.money_ladder
            if reading is None:
                continue
            for mode in ("skip", "fewer", "book"):
                parts = [_cell(reading.row(usd, mode)) for usd in SUMS]
                own = reading.own_usd
                own_cell = "—" if own is None or float(own) in SUMS \
                    else _cell(reading.row(float(own), mode))
                parts.append(own_cell if own_cell == "—" else f"{_usd(own)}: {own_cell}")
                if all(p == "—" for p in parts):
                    continue
                label = {"skip": "пропускаем отказ площадки", "fewer": "меньше позиций",
                         "book": "книга сама"}[mode]
                strict = reading.strict.min_capital_usd
                rows.append(f"| {found.idea_id} | {venue_ru(reading.venue)} | "
                            f"{'—' if strict is None else _usd(strict)} | {label} | "
                            + " | ".join(parts) + " |")
    if not rows:
        return []
    ref_note = ("Ячейка: доход в год / худшее падение на деньгах стратегии, и доля дней-позиций, "
                "которых нет по сравнению с книгой без ограничения. «Свои деньги» — сумма "
                "стратегии, если она не 1 000, 3 000 или 10 000 (иначе она стоит в своём столбце).")
    title = "### Как работают на разных деньгах идеи, ждущие решения (`docs/MONEY_LADDER.md`)"
    return ["", title, "",
            "Показание, не вердикт. С правил 2026-11-04.2 ступень 4 судит по книге «меньше "
            "позиций» на деньгах самой стратегии на последней трети данных (её считает оценка "
            "прогона; держит не меньше 70% дохода книги без ограничений и совпадает с ней день в "
            "день не меньше чем на 0.90 — помещается); прогоны по старым правилам судятся строгим "
            "минимумом, как раньше. " + ref_note, "",
            "| идея | площадка | строгий минимум | как исполняем | $1 000 | $3 000 | $10 000 | "
            "свои деньги |", "|---|---|---|---|---|---|---|---|", *rows]


def format_text(ladder: MoneyLadder) -> str:
    """The reading as plain text for the terminal."""
    ref = ladder.reference
    out = [f"{ladder.idea_id} on {venue_ru(ladder.venue)}, run {ladder.trial_id}, "
           f"{ladder.window[0]} .. {ladder.window[1]}, {ladder.interval} bars",
           why_strict_ru(ladder), "",
           f"{'money':>9} {'how':<7} {'per year':>9} {'worst fall':>10} {'positions':>9} "
           f"{'orders unsent':>13} {'missing':>8} {'corr':>5}",
           f"{'unbound':>9} {'':<7} {_pct(ref.get('ann_return'), 1, True):>9} "
           f"{_pct(ref.get('max_dd')):>10} {ref.get('avg_positions') or 0:>9.1f}"]
    for r in ladder.rows:
        if not r.available:
            continue
        corr = "—" if r.correlation is None else f"{r.correlation:.2f}"
        out.append(f"{_usd(r.usd) + ('*' if r.own else ''):>9} {r.mode:<7} "
                   f"{_pct(r.ann_return, 1, True):>9} {_pct(r.max_dd):>10} "
                   f"{r.avg_positions or 0:>9.1f} {_pct(r.not_sent_share):>13} "
                   f"{_pct(r.missing_position_share):>8} {corr:>5}")
    out.append("* the strategy's own money. skip: the venue refuses what is under its minimum; "
               "fewer: only the positions that fit are held (a different strategy).")
    if ladder.note:
        out.append(ladder.note)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# The store (one file per trial) and the night's stage


def store_path(trial_id: int, root: Path | None = None) -> Path:
    return (root or Path(".")) / STORE / f"{trial_id}.json"


def _key(trial_id: int) -> str:
    return hashlib.sha256(f"{METHOD_VERSION}|{trial_id}|{SUMS}".encode()).hexdigest()


def load_stored(trial_id: int, own_usd: float | None, root: Path | None = None
                ) -> MoneyLadder | None:
    """The stored reading of this trial, when it was made by this method on this own money."""
    try:
        saved = json.loads(store_path(trial_id, root).read_text(encoding="utf-8"))
        if saved.get("key") != _key(trial_id):
            return None
        found = MoneyLadder.from_dict(saved["ladder"])
        # another own money (the spec's number moved) is another reading
        return found if own_usd is None or found.own_usd == own_usd else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_stored(ladder: MoneyLadder, root: Path | None = None) -> Path:
    path = store_path(ladder.trial_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": _key(ladder.trial_id),
                                "ladder": ladder.to_dict()}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def read_trial(session, trial_id: int, own_usd: float | None, root: Path | None = None
               ) -> MoneyLadder | None:
    """The stored reading, else None: a page never measures (a measurement is a minute of work)."""
    return load_stored(trial_id, own_usd, root)


def for_idea(session, idea_id: str, ledgers, root: Path | None = None
             ) -> dict[str, MoneyLadder]:
    """Venue -> stored reading of the idea's latest run on that venue (steps 2 and 3 of its
    ladder). A venue with no run or no stored reading is absent."""
    from qlab import ladder as ladder_mod

    found = ladder_mod.build_ladder(session, idea_id, ledgers)
    if found is None:
        return {}
    own = ladder_mod.strategy_money(idea_id, ledgers)
    out: dict[str, MoneyLadder] = {}
    for step in found.steps[1:3]:
        if step.run is None or step.venue is None:
            continue
        stored = read_trial(session, step.run.trial_id, own, root)
        if stored is not None:
            out[step.venue] = stored
    return out


def write_metrics(session, ladder: MoneyLadder) -> None:
    """Keep the reading on the trial itself: `ml_*` keys beside the stand's metrics. Only these
    keys are touched; the trial's verdict numbers are not."""
    from qlab.registry.models import Trial

    trial = session.get(Trial, ladder.trial_id)
    if trial is None or trial.metrics is None:
        return
    kept = {k: v for k, v in trial.metrics.items() if not k.startswith(METRIC_PREFIX)}
    trial.metrics = {**kept, **ladder.metrics()}


def measure_and_store(session, trial_id: int, own_usd: float | None, root: Path | None = None,
                      *, write: bool = True) -> MoneyLadder:
    ladder = measure_trial(session, trial_id, own_usd)
    save_stored(ladder, root)
    if write:
        write_metrics(session, ladder)
    return ladder


def _metrics_current(session, stored: MoneyLadder) -> bool:
    """Whether the trial carries exactly the `ml_*` keys of the stored reading."""
    from qlab.registry.models import Trial

    trial = session.get(Trial, stored.trial_id)
    if trial is None:
        return True
    have = {k: v for k, v in (trial.metrics or {}).items() if k.startswith(METRIC_PREFIX)}
    return have == stored.metrics()


def money_ladder_stage(day: date, end: date, state: dict, *, only: Sequence[str] | None = None,
                       max_per_night: int = 12, stop_at=None, root: Path | None = None) -> None:
    """The night's stage: for every open idea whose latest run on a venue has no reading yet,
    measure it (a trial shared by an idea and its second stage is measured once), keep it in
    `data/money_ladders/` and on the trial. Code only, no agent, no tokens. At most
    `max_per_night` trials a night; the rest wait and the state says how many."""
    from qlab import ladder as ladder_mod
    from qlab import night
    from qlab.registry.db import session_scope

    block = state.setdefault("money_ladder", {"entries": [], "pending": 0})
    block["entries"], block["pending"] = [], 0
    ledgers = ladder_mod.load_ledgers(root or Path("."))
    with session_scope() as session:
        ids = list(only) if only else ladder_mod.open_idea_ids(session)
        todo: dict[int, tuple[str, float | None]] = {}
        for idea_id in ids:
            found = ladder_mod.build_ladder(session, idea_id, ledgers)
            if found is None:
                continue
            own = ladder_mod.strategy_money(idea_id, ledgers)
            for step in found.steps[1:3]:
                if step.run is None:
                    continue
                tid = step.run.trial_id
                if tid in todo:
                    continue
                stored = None if only else load_stored(tid, own, root)
                if stored is None or not _metrics_current(session, stored):
                    todo[tid] = (idea_id, own)
        for tid, (idea_id, own) in todo.items():
            stored = None if only else load_stored(tid, own, root)
            if stored is None and len(block["entries"]) >= max_per_night:
                block["pending"] += 1
                continue
            if stored is None and stop_at is not None and night.past_stop(
                    stop_at, state, f"money_ladder:{tid}"):
                block["pending"] += 1
                continue
            try:
                if stored is not None:  # measured before: only the trial's metrics are missing
                    write_metrics(session, stored)
                    found = stored
                else:
                    found = measure_and_store(session, tid, own, root)
                session.commit()
                block["entries"].append({"idea": idea_id, "trial": tid, "measured": True,
                                         "sums": [r.usd for r in found.rows if r.mode != "fewer"]})
            except Exception as exc:  # noqa: BLE001 - a stated reason, not a failed night
                session.rollback()
                block["entries"].append({"idea": idea_id, "trial": tid, "measured": False,
                                         "why": f"{type(exc).__name__}: {exc}"[:300]})
    night.save_state(day, state)


__all__ = ["MODES", "REFERENCE_BOOK_USD", "SUMS", "METHOD_VERSION", "MoneyLadder", "Row", "Strict",
           "execute_orders", "fewer_positions", "for_idea", "load_stored", "measure_trial",
           "measure_weights", "money_ladder_stage", "money_line_ru", "sentence_ru",
           "why_strict_ru"]
