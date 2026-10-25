"""The pieces of the Bv2 adapter that are about money moving between books, not about one book's
own rules (`qlab.strategies.live.bv2` drives them):

- a coin list that changes over time (`UniverseRule`, `select_universe`): the coins held at each
  rebalance are chosen from data known before it;
- closing a book whose coin leaves the list (`close_book`);
- paying new money into a running book (`deposit_into_book`);
- calendar periods ("monthly", "30d") for rebalances and deposits (`period_starts`).

Every function here is deterministic and reads only what it is handed. The costs are frab's own
(`SPOT_TAKER`, `PERP_TAKER` and the book's slippage), the same the live book charges its fills.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks.universe import top_volume
from qlab.strategies.live._loader import import_frab

_constants = import_frab("frab.constants")
_book = import_frab("frab.strategy.b2.book")

#: How a coin is ranked. The held asset is the spot (it sits in the cold wallet), so its own
#: market's liquidity is the natural measure; the perp's is offered for comparison.
VOLUME_MEASURES = ("spot_dollar_volume", "perp_dollar_volume")

_NAMED_PERIODS = {"weekly": "W", "monthly": "M", "quarterly": "Q"}
_DAYS = re.compile(r"^(\d+)d$")


def period_starts(index: pd.DatetimeIndex, period: str) -> np.ndarray:
    """True on the first bar of each calendar period (never on bar 0).

    `period` is "weekly", "monthly", "quarterly" (UTC calendar) or "<N>d": every N days from the
    first bar of the panel."""
    if not isinstance(period, str):
        raise ValueError(f"period must be a string like 'monthly' or '30d', got {period!r}")
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    if period in _NAMED_PERIODS:
        label = np.asarray(naive.to_period(_NAMED_PERIODS[period]).asi8)
    else:
        found = _DAYS.match(period)
        if not found or int(found.group(1)) < 1:
            raise ValueError(
                f"period {period!r}: use weekly, monthly, quarterly or '<N>d', N whole days"
            )
        label = np.asarray((naive - naive[0]) // pd.Timedelta(days=int(found.group(1))))
    starts = np.zeros(len(index), dtype=bool)
    starts[1:] = label[1:] != label[:-1]
    return starts


@dataclass(frozen=True)
class UniverseRule:
    """`params.universe` of a spec: which coins the book holds, re-chosen every `rebalance`.

    `n`, `rebalance`, `by`, `window_days` and `candidates` have no defaults: each is a choice the
    spec has to make and cite. `candidates` is the list of coins the choice is made from (each
    needs a perp column `X` and a spot column `X-SPOT`), or "panel" for every coin of the panel
    that has both."""

    n: int
    rebalance: str
    by: str
    window_days: float
    candidates: tuple[str, ...] | None  # None: every coin of the panel with both legs

    def coins(self, panel: MarketPanel) -> list[str]:
        if self.candidates is not None:
            return list(self.candidates)
        columns = set(panel.prices.columns)
        return [c for c in panel.prices.columns
                if not c.endswith("-SPOT") and f"{c}-SPOT" in columns]


_UNIVERSE_KEYS = {"rule", "n", "rebalance", "by", "window_days", "candidates"}


def parse_universe(raw: object) -> UniverseRule:
    if not isinstance(raw, Mapping):
        raise ValueError(f"params.universe must be a mapping, got {raw!r}")
    unknown = set(raw) - _UNIVERSE_KEYS
    if unknown:
        raise ValueError(
            f"params.universe: unknown keys {sorted(unknown)} (known: {sorted(_UNIVERSE_KEYS)})"
        )
    missing = _UNIVERSE_KEYS - set(raw)
    if missing:
        raise ValueError(
            f"params.universe needs {sorted(missing)}: no default is assumed for any of them"
        )
    if raw["rule"] != "top_liquid":
        raise ValueError(f"params.universe.rule {raw['rule']!r}: only 'top_liquid' is implemented")
    n = raw["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"params.universe.n must be a whole number of coins, 1 or more, got {n!r}")
    if raw["by"] not in VOLUME_MEASURES:
        raise ValueError(f"params.universe.by {raw['by']!r}: one of {VOLUME_MEASURES}")
    window = raw["window_days"]
    if isinstance(window, bool) or not isinstance(window, (int, float)) or window <= 0:
        raise ValueError(
            f"params.universe.window_days must be a positive number of days, got {window!r}"
        )
    probe = pd.date_range("2020-01-01", periods=3, freq="1D", tz="UTC")
    period_starts(probe, raw["rebalance"])  # raises on a period it does not know
    cands = raw["candidates"]
    if cands == "panel":
        candidates = None
    elif isinstance(cands, (list, tuple)) and cands and all(isinstance(c, str) for c in cands):
        if len(set(cands)) != len(cands) or any(c.endswith("-SPOT") for c in cands):
            raise ValueError("params.universe.candidates: bare coin names, each once")
        candidates = tuple(cands)
    else:
        raise ValueError("params.universe.candidates must be a list of coins or 'panel'")
    return UniverseRule(n=n, rebalance=raw["rebalance"], by=str(raw["by"]),
                        window_days=float(window), candidates=candidates)


def select_universe(panel: MarketPanel, rule: UniverseRule, coins: list[str],
                    selectable: pd.DataFrame) -> pd.DataFrame:
    """For each bar `t`, the coins the rule would hold if it rebalanced at `t`: True for the `n`
    coins with the largest trailing mean dollar volume among those `selectable` at bar t-1,
    using nothing after bar t-1.

    Reuses the point-in-time arithmetic of `qlab.strategies.blocks.universe.top_volume` (trailing
    window of `window_days`, the window must be fully known, ties to the coin listed first) on the
    volume of the measure `rule.by`, restricted to coins that can open a book: `selectable[c]` at
    bar t-1 says both legs are tradeable and priced there and the startup history will be present
    at bar t. A coin with no perp cannot be hedged and so is not a Bv2 coin: it is left out,
    never held unhedged.

    The row for bar t is the table's row t-1 -- the decision is taken at the close of bar t with
    what was known at the close of bar t-1, like every decision of the book itself (it reads
    signals of the previous bar)."""
    leg = (lambda c: f"{c}-SPOT") if rule.by == "spot_dollar_volume" else (lambda c: c)
    cols = [leg(c) for c in coins]
    index = panel.prices.index
    prices = panel.prices[cols].set_axis(coins, axis=1)
    sub = MarketPanel(
        snapshot_id=str(panel.snapshot_id),
        prices=prices,
        funding=pd.DataFrame(np.nan, index=index, columns=coins),
        tradeable=selectable[coins].astype(bool),
        meta={},
        volume=panel.volume[cols].set_axis(coins, axis=1),
    )
    ranked = top_volume(sub, n=rule.n, window_days=rule.window_days)
    return ranked.shift(1, fill_value=False).astype(bool)


def close_book(book, price: float, slippage: float) -> float:
    """Close a coin book at `price`: the hedge bought back, the spot sold, at frab's costs.
    Returns the cash left in the book (its whole equity, minus the fees)."""
    if getattr(book, "carry_units", 0.0) or book.carry_on:
        raise ValueError("closing a book with a carry sleeve is not modelled")
    perp_cost = _constants.PERP_TAKER + slippage
    spot_cost = _constants.SPOT_TAKER + slippage
    if book.in_pos:
        realized = book.short_size * (book.entry_price - price)
        book.cash += realized
        book.short_realized += realized
        fee = book.short_size * price * perp_cost
        book.cash -= fee
        book.perp_fees += fee
        book.short_size, book.entry_price, book.in_pos = 0.0, 0.0, False
    sale = book.units_spot * price
    fee = sale * spot_cost
    book.cash += sale - fee
    book.spot_fees += fee
    book.units_spot = 0.0
    book.topup_due_ms = None
    return float(book.cash)


def deposit_into_book(book, amount: float, price: float, params, *, pool=None) -> dict:
    """Pay `amount` of new money into a running book and invest it the way the book is sized.

    The book's targets are those of a book started with `capital + amount`: the extra spot is
    bought at `price` (frab's spot cost), the rest joins the HL reserve. A hedged book extends its
    short by the new spot's units as far as the margin allows (the pool's free margin when it is
    in a pool) and the order is at least the minimum; units that cannot be hedged stay
    unhedged until the next hedge cycle. Returns what was done."""
    spot_cost = _constants.SPOT_TAKER + params.slippage
    perp_cost = _constants.PERP_TAKER + params.slippage
    sized = replace(params, coins=(book.coin,), capital_usd=book.capital + amount)
    target_spot, _, _ = sized.book_sizes(book.coin)
    buy = max(target_spot - book.position_size, 0.0)
    if buy < params.min_order_usd or buy <= 0.0:
        buy = 0.0
    fee = buy * spot_cost
    book.capital += amount
    book.cash += amount - buy - fee
    book.reserve += amount - buy
    book.position_size += buy
    book.units_spot += buy / price
    book.spot_fees += fee
    hedged_units = 0.0
    if buy and book.in_pos:
        free = pool.free_margin() if pool is not None else book.free_margin(price)
        room = max(free, 0.0) / (price * (1.0 / book.leverage + perp_cost))
        units = min(buy / price, room)
        if units * price >= params.min_order_usd and units > 0.0:
            total = book.short_size + units
            book.entry_price = (book.short_size * book.entry_price + units * price) / total
            book.short_size = total
            hedge_fee = units * price * perp_cost
            book.cash -= hedge_fee
            book.perp_fees += hedge_fee
            hedged_units = units
    return {"amount": amount, "spot_bought_usd": buy, "fees": fee, "hedged_units": hedged_units}
