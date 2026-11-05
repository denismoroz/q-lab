"""Bv3 as an income strategy (owner 2026-11-05: «как сделать то что я хотел в bv3 - не страховка - а
заработок»; docs/research/bv3-income-*-2026-11-05.md). Pure functions; the runner is
scripts/research/bv3_income.py.

- Step 0, where the money comes from: `decompose` splits a book's result on the strategy's money
  into the spot's move unhedged and hedged (at the book's target size), the hedge's price P&L,
  funding, the spot's deviation from its target, and fees. The parts add up to the stand's
  `capital_return` (tested).
- Step 1, a hedge that pays: `carry_wish` (the production wish OR a funding condition).
- Step 2, relative strength: `weakest_wish` (hedge the weakest coins of the basket each day).
- Step 3, rebalancing premium: `rebalanced_book` (an equal-weight spot basket kept at its targets
  by a band or a schedule, with a deep tail hedge).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.gaps import held_through, holdable_gaps, marked_through
from qlab.harness.panel import MarketPanel

PARTS = ("spot_unhedged", "spot_hedged", "hedge_price", "funding", "spot_drift", "fees")


def decompose(panel: MarketPanel, weights: pd.DataFrame, equity: pd.Series, capital: float,
              cost: pd.Series, targets: Mapping[str, float]) -> pd.DataFrame:
    """Per bar, the book's result as shares of `capital` split into `PARTS`.

    `weights`, `equity` (the book's balance) and `cost` (the stand's cost, a share of equity) are a
    run's; `targets[coin]` is the coin's spot target in dollars (frab's `position_size`), counted
    on the bars the coin's spot is held. The harness's own conventions are kept: a weight at t
    earns the move to t+1, bars without a price are held through, the short receives positive
    funding (`qlab.harness.accrual`)."""
    keep = panel.prices.index[:-1]
    gaps = holdable_gaps(panel.prices, panel.funding, panel.meta.get("no_funding_instruments", ()))
    w = held_through(weights, gaps).loc[keep]
    prices = marked_through(panel.prices, gaps)
    ret = prices.pct_change().shift(-1).loc[keep].fillna(0.0)
    fund = panel.funding.shift(-1).loc[keep]
    eq = equity.reindex(keep).ffill().bfill()
    out = {p: np.zeros(len(keep)) for p in PARTS}
    for coin, target in targets.items():
        spot = f"{coin}{SPOT_COLUMN_SUFFIX}"
        if spot not in w.columns:
            continue
        ws, wp = w[spot].to_numpy(), w[coin].to_numpy()
        rs, rp = ret[spot].to_numpy(), ret[coin].to_numpy()
        spot_usd = ws * eq.to_numpy()
        perp_usd = wp * eq.to_numpy()
        held = ws > 1e-12
        tgt = np.where(held, float(target), 0.0)
        hedged = wp < -1e-12
        out["spot_unhedged"] += tgt * rs * ~hedged
        out["spot_hedged"] += tgt * rs * hedged
        out["spot_drift"] += (spot_usd - tgt) * rs
        out["hedge_price"] += perp_usd * rp
        out["funding"] += np.where(np.abs(wp) > 1e-12, -perp_usd * np.nan_to_num(fund[coin]), 0.0)
        # a spot market has no funding (NaN on a real panel); a synthetic one may carry a rate
        out["funding"] += np.where(held, -spot_usd * np.nan_to_num(fund[spot]), 0.0)
    out["fees"] = -(cost.reindex(keep).fillna(0.0) * eq).to_numpy()
    return pd.DataFrame(out, index=keep) / capital


def summarise_parts(parts: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """% of the strategy's money per year, each part, before and from `cut`."""
    out = {}
    for name, sel in (("first", parts.index < cut), ("last", parts.index >= cut)):
        seg = parts[sel]
        years = len(seg) / 8760
        out[name] = {p: float(100 * seg[p].sum() / years) for p in parts.columns}
        out[name]["total"] = float(100 * seg.sum(axis=1).sum() / years)
    return out


# ── step 1 ─────────────────────────────────────────────────────────────────


def funding_annual(funding: pd.Series, hours: int) -> np.ndarray:
    """The funding settled over the last `hours` bars, as a yearly rate (bars up to t)."""
    f = funding.fillna(0.0)
    return (f.rolling(hours, min_periods=hours).sum() * (8760 / hours)).to_numpy()


def carry_wish(base_wish: np.ndarray, enter: np.ndarray, stay: np.ndarray) -> np.ndarray:
    """The production wish, or a carry hedge: on where `enter`, kept while `stay`, off otherwise.
    NaN in `base_wish` (unpriced bar) passes through."""
    out = base_wish.copy()
    on = False
    for i in range(len(out)):
        if np.isnan(base_wish[i]):
            continue
        if not on and enter[i]:
            on = True
        elif on and not stay[i]:
            on = False
        if on:
            out[i] = 1.0
    return out


# ── step 2 ─────────────────────────────────────────────────────────────────


def weakest_wish(returns: pd.DataFrame, members: pd.DataFrame, share: float) -> pd.DataFrame:
    """Day x coin: hedge the weakest `share` of the day's members by their past return (read at
    the previous day's close; ties to the coin listed first). At least one coin is hedged when
    the basket has two or more members."""
    out = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    for day in returns.index:
        m = members.loc[day] if day in members.index else None
        if m is None:
            continue
        r = returns.loc[day][m.reindex(returns.columns).fillna(False).to_numpy(dtype=bool)].dropna()
        if len(r) < 2:
            continue
        k = max(1, int(np.floor(share * len(r) + 1e-9)))
        weakest = r.sort_values(kind="stable").index[:k]
        out.loc[day, weakest] = True
    return out


# ── step 3 ─────────────────────────────────────────────────────────────────


def rebalanced_book(prices: pd.DataFrame, funding: pd.DataFrame, members: pd.DataFrame,
                    slot_usd: float, capital: float, band: float | None, weekly: bool,
                    tail: float | None, spot_cost: float, perp_cost: float,
                    sticky: int = 12) -> pd.Series:
    """Hourly result (share of `capital`, summed, never compounded) of an equal-weight spot basket.

    Each member coin is bought at `slot_usd`; it is traded back to `slot_usd` when its value leaves
    [1 - band, 1 + band] x slot (`band`) or at the first bar of each week (`weekly`); a coin that
    leaves the members is sold, a joiner bought. `tail`: a short of the coin's whole spot opens
    when its 30-day return falls below -tail and closes after `sticky` bars in a row above 0.
    Costs: `spot_cost` per dollar of spot traded, `perp_cost` per dollar of short opened, closed or
    resized at a rebalance. Funding on the short as the stand counts it (the short receives a
    positive rate). `members` is hourly (bool, same shape as `prices`); prices are the perps' and
    the spot is marked at them, as the Bv2 book."""
    idx = prices.index
    px = prices.ffill().to_numpy()
    fu = funding.reindex_like(prices).fillna(0.0).to_numpy()
    mem = members.reindex_like(prices).fillna(False).to_numpy(dtype=bool)
    n, k = px.shape
    mom = np.full((n, k), np.nan)
    if tail is not None:
        mom[720:] = px[720:] / px[:-720] - 1.0
    week = np.r_[True, (idx[1:].to_period("W") != idx[:-1].to_period("W"))] if weekly else None
    units = np.zeros(k)
    short = np.zeros(k)
    on = np.zeros(k, dtype=bool)
    calm = np.zeros(k, dtype=int)
    out = np.zeros(n)
    for t in range(n - 1):
        p = px[t]
        cost = 0.0
        for j in range(k):
            ok = mem[t, j] and not np.isnan(p[j])
            if not ok:
                if units[j]:
                    cost += abs(units[j]) * p[j] * spot_cost if not np.isnan(p[j]) else 0.0
                    units[j] = 0.0
                if short[j]:
                    cost += short[j] * p[j] * perp_cost if not np.isnan(p[j]) else 0.0
                    short[j], on[j] = 0.0, False
                continue
            value = units[j] * p[j]
            if units[j] == 0.0:
                units[j] = slot_usd / p[j]
                cost += slot_usd * spot_cost
            elif (band is not None and abs(value / slot_usd - 1.0) > band) or (
                    week is not None and week[t]):
                cost += abs(value - slot_usd) * spot_cost
                units[j] = slot_usd / p[j]
            if tail is not None and not np.isnan(mom[t, j]):
                if not on[j] and mom[t, j] < -tail:
                    on[j], calm[j] = True, 0
                elif on[j]:
                    calm[j] = calm[j] + 1 if mom[t, j] > 0 else 0
                    if calm[j] >= sticky:
                        on[j] = False
            want = units[j] if on[j] else 0.0
            if abs(want - short[j]) * p[j] > 1e-9:
                cost += abs(want - short[j]) * p[j] * perp_cost
                short[j] = want
        nxt = px[t + 1]
        move = np.nan_to_num((nxt - p) * (units - short))
        fund = np.nan_to_num(short * p * fu[t + 1])
        out[t] = (move.sum() + fund.sum() - cost) / capital
    return pd.Series(out, index=idx)


__all__ = ["PARTS", "carry_wish", "decompose", "funding_annual", "rebalanced_book",
           "summarise_parts", "weakest_wish"]
