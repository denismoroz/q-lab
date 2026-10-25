"""A live strategy's OWN equity: its budget plus what the strategy earned.

Owner, 2026-10-25, about FRAB: «стратегия работает на 90 и должна считать все на 90 43 - это то
что еще доступно на счету - оно не должно никак фигурировать в расчетах»; «нужно ставить
различие - деньги на счету и деньги доступные для стратегии».

Two different sums of money, never mixed:

* **account money** -- everything on the exchange account (`equity_snapshots.total_equity`):
  the positions' coins and margin plus free cash that belongs to no position, and every
  deposit and withdrawal. It is NOT a denominator for any percent in q-lab.
* **strategy money** -- what the strategy may use (`budget_cap_usdc` for FRAB) plus the profit
  and loss the strategy itself made. Every percent of a live strategy is computed on this.

The series built here is

    strategy_equity(t) = budget + P&L(t)  -  P&L(since)

and P&L is the sum over the strategy's legs of

* **trading result** of each leg, from its own fills: cash flows of the fills plus the net
  holding marked at the price of hour t (a closed leg is its realised result; an open leg
  adds the unrealised one);
* **funding** received on the legs (`funding_accruals`; the short perp earns it);
* minus the **fees** of the fills (`fills.fee`).

A deposit to the account, a withdrawal, an internal transfer between the spot and perp wallets
and the free cash change none of these, so none of them can move the series. COLLATERAL rows
are bookkeeping (a reservation of cash) and are ignored; a position with no fills (a FAILED
one) has no legs here at all.

**Dust.** A spot sell is floored to the exchange's lot, so a closed spot leg keeps up to one lot
of its coin (BTC 0.00001, SOL 0.01, HYPE 0.01, ...). That coin is still the strategy's money
and moves with the price, so a closed leg that leaves a holding stays in the series, marked
like an open one; without it a perp short sized to cover the dust would look like a naked
short. The dust is taken from the fill records, which is not exact: over 2026-10-10..25 the
account itself (its spot value less the open legs) held about $2 of dust where the fills say
about $3.7 (the BTC dust of the fills is probably gone from the wallet), so the trading result
carries an uncertainty of about 0.05 on the $90. docs/FRAB_LIVE.md has the comparison.

Everything is computed from raw rows exported read-only from the production host
(`export_queries`), so this logic is tested here, not trusted from the host's own counters.

**Marks.** Hourly prices come from the host's own `prices` table (`mark`, the perp's price),
which the host thins to one row per coin and clock hour after seven days (`_prune_prices`,
`scripts/archive_and_thin_prices.py`, 2026-10-25): exactly the hourly grid needed, from
2026-05-28. The row used for the boundary t is the freshest one before t. The table's `spot`
column is empty on the host, so the spot leg is marked at the perp's price too: the same rule
as the stand's default (`data.spot_prices: perp`, docs/FRAB_LIVE.md). The spot-perp price gap
is therefore not in the series; docs/FRAB_LIVE.md says how big it was. A boundary at which a
coin with an open leg has no price younger than `MARK_MAX_AGE_MS` is dropped from the series
and reported, never filled with a guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000
MARK_MAX_AGE_MS = 2 * HOUR_MS
"""A price older than this at the hour boundary is "missing". The host stores the freshest row
of each clock hour, so one missing hour is tolerated and two are not."""

LEGS_FILE = "frab_legs.csv"
FILLS_FILE = "frab_fills.csv"
FUNDING_FILE = "frab_funding.csv"
MARKS_FILE = "frab_marks.csv"
BUDGET_FILE = "frab_budget.csv"
INPUT_FILES = (LEGS_FILE, FILLS_FILE, FUNDING_FILE, MARKS_FILE, BUDGET_FILE)

BUY_SIDE = "long"
SELL_SIDE = "short"
"""`fills.side`: "long" buys, "short" sells -- for the spot leg and for the perp leg alike
(the perp short opens with a "short" fill and closes with a "long" one)."""

COLLATERAL = "COLLATERAL"
BUDGET_KEY = "budget_cap_usdc"


class StrategyPnlError(ValueError):
    """The strategy's result cannot be computed honestly from the exported rows."""


# ---------------------------------------------------------------------------
# Read-only export (SELECT only)
# ---------------------------------------------------------------------------


LEFTOVER_QTY = "1e-9"


def _leg_ids_sql(strategy_id: int, since_ms: int) -> str:
    """The strategy's legs that matter from `since_ms` on, COLLATERAL excluded: legs alive at or
    after it, and closed legs that still leave a holding (a spot sell is floored to the lot, so
    a closed spot leg keeps up to one lot of the coin, which stays the strategy's and moves
    with the price). A leg that closed flat before `since_ms` cannot change the result."""
    leftover = ("abs((select coalesce(sum(case when fl.side='long' then fl.qty else -fl.qty "
                f"end), 0.0) from fills fl where fl.position_id=p.id))>{LEFTOVER_QTY}")
    return (f"select p.id from positions p join farb_positions f on f.id=p.farb_position_id "
            f"where f.strategy_id={int(strategy_id)} and p.instrument<>'{COLLATERAL}' "
            f"and (p.closed_at is null or p.closed_at>={int(since_ms)} or {leftover})")


def export_queries(strategy_id: int, since_ms: int) -> list[tuple[str, str]]:
    """(file name, SELECT) pairs for the production host's sqlite, read-only. No quote
    characters other than single ones and no `$`: the statement travels inside a double-quoted
    remote shell command (`qlab.night._ssh_query`)."""
    sid, since = int(strategy_id), int(since_ms)
    legs = _leg_ids_sql(sid, since)
    return [
        (LEGS_FILE,
         "select p.id as position_id, p.farb_position_id, p.exchange_id, p.coin, p.instrument, "
         "p.side, p.qty, p.entry_price, p.opened_at, p.closed_at, p.status, "
         "f.state as farb_state from positions p "
         "join farb_positions f on f.id=p.farb_position_id "
         f"where p.id in ({legs}) order by p.id"),
        (FILLS_FILE,
         "select id as fill_id, position_id, ts_ms, side, qty, price, fee from fills "
         f"where position_id in ({legs}) order by position_id, ts_ms, id"),
        (FUNDING_FILE,
         "select position_id, ts_ms, amount from funding_accruals "
         f"where position_id in ({legs}) order by position_id, ts_ms"),
        (MARKS_FILE,
         "select exchange_id, coin, ts_ms, mark, spot from (select exchange_id, coin, ts_ms, "
         "mark, spot, row_number() over (partition by exchange_id, coin, ts_ms/3600000 "
         "order by ts_ms desc) as rn from prices "
         f"where ts_ms>={since - 3 * HOUR_MS} and coin in "
         f"(select p.coin from positions p where p.id in ({legs}))) "
         "where rn=1 order by exchange_id, coin, ts_ms"),
        (BUDGET_FILE,
         f"select id as strategy_id, params_json from strategies where id={sid}"),
    ]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PnlInputs:
    """The raw rows of one strategy: legs, fills, funding payments, hourly marks, and the
    budget in force."""

    legs: pd.DataFrame
    fills: pd.DataFrame
    funding: pd.DataFrame
    marks: pd.DataFrame
    budget: float


def budget_from_params(raw: str) -> float:
    """`budget_cap_usdc` out of a strategy's `params_json` text."""
    params = json.loads(raw)
    value = params.get(BUDGET_KEY)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise StrategyPnlError(f"{BUDGET_KEY} in the strategy's params is {value!r}, "
                               f"not a positive number: the strategy's money is unknown")
    return float(value)


def load_inputs(folder: Path) -> PnlInputs:
    """Read the five export files of one night from `folder`."""
    budget_rows = pd.read_csv(folder / BUDGET_FILE)
    if len(budget_rows) != 1:
        raise StrategyPnlError(f"{BUDGET_FILE} must hold exactly one strategy row, "
                               f"has {len(budget_rows)}")
    return PnlInputs(
        legs=pd.read_csv(folder / LEGS_FILE),
        fills=pd.read_csv(folder / FILLS_FILE),
        funding=pd.read_csv(folder / FUNDING_FILE),
        marks=pd.read_csv(folder / MARKS_FILE),
        budget=budget_from_params(str(budget_rows["params_json"].iloc[0])))


def _traded_legs(inputs: PnlInputs) -> pd.DataFrame:
    """The legs that count: not COLLATERAL (bookkeeping), and with at least one fill (a leg
    that never filled holds nothing)."""
    legs = inputs.legs[inputs.legs["instrument"] != COLLATERAL]
    filled = set(inputs.fills["position_id"])
    return legs[legs["position_id"].isin(filled)]


def ceil_hour(ts_ms: int) -> int:
    return -(-int(ts_ms) // HOUR_MS) * HOUR_MS


# ---------------------------------------------------------------------------
# The series
# ---------------------------------------------------------------------------


def _marks_at(marks: pd.DataFrame, exchange_id: int, coin: str, grid: np.ndarray,
              *, uses_spot: bool) -> np.ndarray:
    """The price of (exchange, coin) at each boundary of `grid`: the freshest stored row at or
    before it, NaN when that row is older than `MARK_MAX_AGE_MS` or there is none. A spot leg
    uses the row's `spot` when the host stored one, else the `mark`."""
    rows = marks[(marks["exchange_id"] == exchange_id) & (marks["coin"] == coin)]
    rows = rows.sort_values("ts_ms")
    price = rows["mark"].astype(float)
    if uses_spot and "spot" in rows.columns:
        price = rows["spot"].astype(float).where(rows["spot"].notna(), price)
    if rows.empty:
        return np.full(len(grid), np.nan)
    ts = rows["ts_ms"].to_numpy(dtype=np.int64)
    at = np.searchsorted(ts, grid, side="right") - 1
    out = np.where(at >= 0, price.to_numpy()[np.clip(at, 0, None)], np.nan)
    age = grid - ts[np.clip(at, 0, None)]
    return np.where((at >= 0) & (age <= MARK_MAX_AGE_MS), out, np.nan)


def _leg_trading_pnl(fills: pd.DataFrame, prices: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """One leg's trading result at every boundary: the cash flow of its fills so far plus the
    net holding marked at that boundary's price. NaN where a holding has no price."""
    ts = fills["ts_ms"].to_numpy(dtype=np.int64)
    sign = np.where(fills["side"] == SELL_SIDE, 1.0, -1.0)
    qty = fills["qty"].to_numpy(dtype=float)
    cash = np.concatenate([[0.0], np.cumsum(sign * qty * fills["price"].to_numpy(dtype=float))])
    held = np.concatenate([[0.0], np.cumsum(-sign * qty)])
    at = np.searchsorted(ts, grid, side="right")
    holding = held[at]
    flat = np.abs(holding) < 1e-12
    return np.where(flat, cash[at], cash[at] + holding * prices)


def _cumulative(rows: pd.DataFrame, column: str, grid: np.ndarray) -> np.ndarray:
    """Sum of `column` over rows with ts_ms <= each boundary."""
    if rows.empty:
        return np.zeros(len(grid))
    ordered = rows.sort_values("ts_ms")
    ts = ordered["ts_ms"].to_numpy(dtype=np.int64)
    total = np.concatenate([[0.0], np.cumsum(ordered[column].to_numpy(dtype=float))])
    return total[np.searchsorted(ts, grid, side="right")]


def strategy_equity(inputs: PnlInputs, *, since_ms: int, until_ms: int | None = None
                    ) -> tuple[pd.DataFrame, list[int]]:
    """The strategy's own equity at every full hour from `since_ms` (rounded up to the hour) to
    `until_ms` (default: the last hour boundary the marks reach).

    Returns (frame, dropped): the frame has `ts_ms`, `strategy_equity`, and the parts
    (`budget`, `trading_pnl`, `funding`, `fees`, all since the first row), `dropped` the
    boundaries left out because a price was missing. The first row equals the budget."""
    t0 = ceil_hour(since_ms)
    legs = _traded_legs(inputs)
    if until_ms is None:
        if inputs.marks.empty:
            raise StrategyPnlError("no hourly prices were exported: nothing to mark the legs at")
        until_ms = (int(inputs.marks["ts_ms"].max()) // HOUR_MS) * HOUR_MS
    if until_ms <= t0:
        raise StrategyPnlError("the exported rows end before the first hour of the period")
    grid = np.arange(t0, until_ms + 1, HOUR_MS, dtype=np.int64)

    trading = np.zeros(len(grid))
    for leg in legs.itertuples():
        fills = inputs.fills[inputs.fills["position_id"] == leg.position_id]
        prices = _marks_at(inputs.marks, int(leg.exchange_id), str(leg.coin), grid,
                           uses_spot=leg.instrument == "SPOT")
        trading = trading + _leg_trading_pnl(fills, prices, grid)

    leg_ids = set(legs["position_id"])
    funding = _cumulative(inputs.funding[inputs.funding["position_id"].isin(leg_ids)],
                          "amount", grid)
    fees = _cumulative(inputs.fills[inputs.fills["position_id"].isin(leg_ids)], "fee", grid)

    frame = pd.DataFrame({"ts_ms": grid, "trading_pnl": trading - trading[0],
                          "funding": funding - funding[0], "fees": fees - fees[0]})
    frame["budget"] = inputs.budget
    frame["strategy_equity"] = (frame["budget"] + frame["trading_pnl"] + frame["funding"]
                                - frame["fees"])
    dropped = [int(t) for t in frame.loc[frame["strategy_equity"].isna(), "ts_ms"]]
    if dropped and int(frame["ts_ms"].iloc[0]) in dropped:
        raise StrategyPnlError("no price at the first hour of the period: the base of the "
                               "series is unknown")
    return frame.dropna(subset=["strategy_equity"]).reset_index(drop=True), dropped


def position_breakdown(inputs: PnlInputs, at_ms: int) -> pd.DataFrame:
    """Each farb position's result SINCE IT OPENED, as of `at_ms`: funding, fees paid so far,
    the perp leg's and the spot leg's trading result. For comparing with the host's own
    per-position numbers."""
    legs = _traded_legs(inputs)
    grid = np.array([at_ms], dtype=np.int64)
    rows = []
    for farb_id, group in legs.groupby("farb_position_id"):
        parts = {"farb_position_id": int(farb_id), "coin": str(group["coin"].iloc[0]),
                 "state": str(group["farb_state"].iloc[0]), "perp_pnl": 0.0, "spot_pnl": 0.0,
                 "funding": 0.0, "fees": 0.0}
        for leg in group.itertuples():
            fills = inputs.fills[inputs.fills["position_id"] == leg.position_id]
            prices = _marks_at(inputs.marks, int(leg.exchange_id), str(leg.coin), grid,
                               uses_spot=leg.instrument == "SPOT")
            pnl = float(_leg_trading_pnl(fills, prices, grid)[0])
            parts["perp_pnl" if leg.instrument == "PERP" else "spot_pnl"] += pnl
            parts["funding"] += float(_cumulative(
                inputs.funding[inputs.funding["position_id"] == leg.position_id],
                "amount", grid)[0])
            parts["fees"] += float(_cumulative(fills, "fee", grid)[0])
        rows.append(parts)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# From the export folder to the file the decay detector reads
# ---------------------------------------------------------------------------


def build_series_file(folder: Path, *, filename: str, strategy_id: int, since_ms: int,
                      expected_budget: float) -> dict:
    """Read the night's export files in `folder`, build the strategy's equity series and write
    it to `folder / filename` (columns `strategy_id`, `ts_ms`, `strategy_equity`, and the parts).

    The budget found in the database must equal `expected_budget`
    (`strategy_capital_usd` in night/paper_books.yaml): if the owner changed it, every percent
    would be on a different base from the one written down, so the series is refused until the
    two agree. Returns a summary for the report."""
    inputs = load_inputs(folder)
    if abs(inputs.budget - expected_budget) > 1e-9:
        raise StrategyPnlError(
            f"the strategy's budget in the database is {inputs.budget:g}, but "
            f"night/paper_books.yaml says strategy_capital_usd {expected_budget:g}: "
            f"percents would be on two different bases; the owner aligns them")
    frame, dropped = strategy_equity(inputs, since_ms=since_ms)
    frame.insert(0, "strategy_id", int(strategy_id))
    frame.to_csv(folder / filename, index=False)
    return {"rows": len(frame), "dropped": len(dropped), "budget": inputs.budget,
            "last_equity": float(frame["strategy_equity"].iloc[-1])}


__all__ = [
    "BUDGET_FILE", "FILLS_FILE", "FUNDING_FILE", "INPUT_FILES", "LEGS_FILE", "MARKS_FILE",
    "PnlInputs", "StrategyPnlError", "budget_from_params", "build_series_file", "ceil_hour",
    "export_queries", "load_inputs", "position_breakdown", "strategy_equity",
]
