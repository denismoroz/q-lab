"""Why one Bv2 twin earns +2% on Binance and +8% on Hyperliquid over the same 173 days
(docs/research/bv2-binance-vs-hl-2026-10-28.md).

Reads the two data snapshots of runs 105587 (Binance) and 105788 (Hyperliquid) and re-runs
the stand's own code on them. CPU only: no agent, no network, no trial written.

    uv run python scripts/research/bv2_binance_vs_hl.py
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import yaml

from qlab import decay
from qlab.data.snapshot import load_snapshot
from qlab.diagnose import equity_from_returns
from qlab.harness.costs import CostModel
from qlab.harness.run import run_backtest
from qlab.pipeline.evaluate import (
    _slice_rows,
    complete_book_window,
    cut_truncated_history,
    resolve_strategy,
)
from qlab.registry.db import session_scope
from qlab.strategies.live.bv2 import LiveBv2
from qlab.strategies.retune import expand_grid

SPECS = {
    "binance": ("specs/research/bv2/bv2-research-headroom-causal.yaml",
                "7bf96a95f3da5376557476b59742f7b5fea17d0197ccdc5b4744defbd72c6f35"),
    "hyperliquid": ("specs/generated/stage2/bv2-research-headroom-causal-hl.yaml",
                    "bee49766b3c3096b1ad0e210b9f90e6496248cf601612d98e040ac65182acdb3"),
}
CAPITAL = 257.0
COINS = ("BTC", "ETH", "SOL", "AVAX")


def total(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cost_model(spec: dict) -> CostModel:
    return CostModel(taker_fee_bps=spec["costs"]["taker_fee_bps"],
                     slippage_bps=spec["costs"]["slippage_bps"])


def load(venue: str):
    path, snapshot_id = SPECS[venue]
    spec = yaml.safe_load(open(path, encoding="utf-8"))
    with session_scope() as session:
        panel = load_snapshot(snapshot_id, session=session)
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    return spec, panel


def judged_rows(spec: dict, panel) -> tuple[int, int]:
    """The rows the stand judges (the same steps as `evaluate_spec`)."""
    coverage = complete_book_window(panel, spec["required_instruments"])
    first, last = 0, len(panel.prices.index) - 1
    if coverage.coverage < 1.0 and coverage.window is not None:
        first, last = coverage.window
    judged = int(panel.prices.index.searchsorted(pd.Timestamp(spec["judged_from"], tz="UTC")))
    return max(first, min(judged, last)), last


def main() -> None:
    spec_b, panel_b = load("binance")
    spec_h, panel_h = load("hyperliquid")
    first_h, last_h = judged_rows(spec_h, panel_h)
    w0, w1 = panel_h.prices.index[first_h], panel_h.prices.index[last_h]
    print(f"window of the Hyperliquid stage: {w0} .. {w1} ({last_h - first_h + 1} bars)")

    # 1. The twins as the stand runs them, cut to that window.
    runs = {}
    for venue, spec, panel in (("binance", spec_b, panel_b), ("hyperliquid", spec_h, panel_h)):
        strategy = resolve_strategy(spec["code_ref"])
        weights = strategy.target_weights(panel, spec["params"])
        result = run_backtest(panel, weights, cost_model(spec), panel.funding)
        runs[venue] = (strategy._last, result)
        window = result.net_return.loc[w0:w1]
        print(f"{venue}: twin on the window {total(window):+.4f}; "
              f"price {result.gross_return.loc[w0:w1].sum():+.4f} "
              f"funding {result.accrual.loc[w0:w1].sum():+.4f} "
              f"fees {-result.cost.loc[w0:w1].sum():+.4f}; choices:")
        print(strategy._last.choices.loc["2026-04-01":].to_string())

    # 2. The same two candidates, each alone, in dollars and as a share of $257.
    candidates = expand_grid(spec_b["params"]["base_params"], spec_b["params"]["grid"])
    per_candidate = {}
    for venue, spec, panel in (("binance", spec_b, panel_b), ("hyperliquid", spec_h, panel_h)):
        for name, params in candidates.items():
            book = LiveBv2().run(panel, params)
            result = run_backtest(panel, book.weights, cost_model(spec), panel.funding)
            equity = book.book_equity.reindex(result.net_return.index)
            per_candidate[(venue, name)] = (book, result, equity)
            sl = (equity.index >= w0) & (equity.index <= w1)
            dollars = {
                label: float((series[sl] * equity[sl]).sum())
                for label, series in (("net", result.net_return), ("price", result.gross_return),
                                      ("funding", result.accrual), ("fees", -result.cost))}
            held = book.weights.reindex(equity.index)
            coins_held = sum(float((held[f"{c}-SPOT"] * equity)[sl].mean()) for c in COINS)
            mean_equity = float(equity[sl].mean())
            net_usd = dollars["net"]
            print(f"{venue:12s} {name:10s} equity {mean_equity:8.1f}  coins held {coins_held:6.1f}"
                  f"  net ${net_usd:6.2f} = {net_usd / CAPITAL:+.2%} of ${CAPITAL:.0f}"
                  f" ({net_usd / mean_equity:+.2%} of its equity); "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in dollars.items() if k != "net"))
    # The Hyperliquid twin's monthly choices (out of the market, then "experiment", then "base")
    # applied to either venue's candidates, in dollars of $257: the choice and the venue apart.
    june = pd.Timestamp("2026-06-01", tz="UTC")
    october = pd.Timestamp("2026-10-01", tz="UTC")
    for venue in ("binance", "hyperliquid"):
        pieces = {}
        for label, name, lo, hi in (("out of the market", None, w0, june),
                                    ("experiment", "experiment", june, october),
                                    ("base", "base", october, w1 + pd.Timedelta(hours=1))):
            if name is None:
                pieces[label] = 0.0
                continue
            _, result, equity = per_candidate[(venue, name)]
            rows = (equity.index >= lo) & (equity.index < hi)
            pieces[label] = float((result.net_return * equity)[rows].sum())
        print(f"{venue}: Hyperliquid's choices in dollars " + ", ".join(
            f"{k} {v:+.2f}" for k, v in pieces.items()) + f" -> {sum(pieces.values()):+.2f} = "
            f"{sum(pieces.values()) / CAPITAL:+.2%} of ${CAPITAL:.0f}")
        for name in candidates:
            _, result, equity = per_candidate[(venue, name)]
            for lo, hi in ((w0, june), (june, october), (october, w1 + pd.Timedelta(hours=1))):
                rows = (equity.index >= lo) & (equity.index < hi)
                print(f"    {name:10s} {lo.date()} .. {hi.date()}: "
                      f"{float((result.net_return * equity)[rows].sum()):+.2f}")
    print("book equity of the Binance base candidate at year ends:",
          per_candidate[("binance", "base")][2].groupby(
              per_candidate[("binance", "base")][2].index.year).last().round(0).to_dict())

    # The facts under the money: coin prices, the rate paid for holding a short, how long the
    # insurance (a short on the perpetual) was on. Prices are the perpetual's on both venues:
    # the bought coin is marked at its perpetual (`data.spot_prices: perp`).
    days = (w1 - w0).total_seconds() / 86400
    for venue, panel in (("binance", panel_b), ("hyperliquid", panel_h)):
        moves = {c: float(panel.prices.loc[w1, c] / panel.prices.loc[w0, c] - 1) for c in COINS}
        rate = {c: float(panel.funding.loc[w0:w1, c].sum() / days * 365) for c in COINS}
        base_weights = per_candidate[(venue, "base")][0].weights.loc[w0:w1]
        on = {c: float((base_weights[c] < -1e-9).mean()) for c in COINS}
        rounded = {"price move": (moves, 4), "yearly rate paid to a short": (rate, 4),
                   "share of hours with the insurance on": (on, 3)}
        for label, (values, digits) in rounded.items():
            print(f"{venue}: {label} {({c: round(v, digits) for c, v in values.items()})}")

    # 3. The Binance data cut to the Hyperliquid panel's own start, books fresh: the control.
    cut = int(panel_b.prices.index.searchsorted(panel_h.prices.index[0]))
    fresh = _slice_rows(panel_b, cut, len(panel_b.prices.index) - 1)
    before = fresh.prices.index < w0  # Hyperliquid has no AVAX spot before its first print
    tradeable, prices = fresh.tradeable.copy(), fresh.prices.copy()
    tradeable.loc[before, "AVAX-SPOT"] = False
    prices.loc[before, "AVAX-SPOT"] = np.nan
    fresh = dataclasses.replace(fresh, tradeable=tradeable, prices=prices)
    strategy = resolve_strategy(spec_b["code_ref"])
    weights = strategy.target_weights(fresh, spec_b["params"])
    for label, costs in (("Binance fees", cost_model(spec_b)),
                         ("Hyperliquid fees", cost_model(spec_h))):
        result = run_backtest(fresh, weights, costs, fresh.funding)
        print(f"Binance data, fresh books from {fresh.prices.index[0]}, {label}: "
              f"{total(result.net_return.loc[w0:w1]):+.4f}")

    # 4. The agreement check: Hyperliquid's result among Binance's 173-day stretches.
    # As `qlab.pipeline.confirm` reads stage 1's history: the stretch in which the whole book
    # existed, from the day its parameters were fixed.
    book_first, _ = complete_book_window(panel_b, spec_b["required_instruments"]).window
    net = runs["binance"][1].net_return.iloc[book_first:]
    net = net[net.index >= pd.Timestamp(spec_b["params_fixed_at"], tz="UTC")]
    window_days = 173
    closes = decay.daily_closes(equity_from_returns(decay.trim_flat_start(net)))
    stretches = np.array([w.total for w in decay.rolling_windows(closes, window_days)])
    hl_total = total(runs["hyperliquid"][1].net_return.loc[w0:w1])
    print(f"as the stand does it (percent of the book's growing equity): {len(stretches)} "
          f"stretches, Hyperliquid's {hl_total:+.4f} is better than "
          f"{decay.percentile_of(hl_total, stretches):.1f}%")

    # The same stretches with the percent taken of the starting $257 (money of the twin's
    # chosen candidate in dollars, a rebuild: each candidate keeps its own book).
    choices = runs["binance"][0].choices
    index = per_candidate[("binance", "base")][1].net_return.index
    pnl = pd.Series(0.0, index=index)
    edges = list(choices.index) + [index[-1] + pd.Timedelta(hours=1)]
    for i, start in enumerate(choices.index):
        name = choices.iloc[i]
        if not isinstance(name, str):
            continue
        _, result, equity = per_candidate[("binance", name)]
        rows = (index >= start) & (index < edges[i + 1])
        pnl[rows] = (result.net_return * equity)[rows]
    pnl = pnl[pnl.index >= net.index[0]]
    cumulative = pnl.cumsum() / CAPITAL
    days = (cumulative.index - pd.Timedelta(milliseconds=1)).floor("D")
    daily = cumulative.groupby(days).last()
    every_day = pd.date_range(daily.index[0], daily.index[-1], freq="D", tz="UTC")
    values = daily.reindex(every_day).ffill().to_numpy()
    fixed = np.array([values[i + window_days] - values[i]
                      for i in range(len(values) - window_days)])
    print(f"percent of the starting ${CAPITAL:.0f}: Hyperliquid's {hl_total:+.4f} is better than "
          f"{decay.percentile_of(hl_total, fixed):.1f}% of the stretches")
    judged = pnl[pnl.index >= "2025-06-01"]
    days_judged = len(judged) / 24
    share = judged.sum() / CAPITAL
    print(f"Binance twin judged from 2025-06-01 on ${CAPITAL:.0f}: {share:+.2%} "
          f"over {days_judged:.0f} days = {share * 365 / days_judged:+.1%} a year; "
          f"2025: {judged[:'2025-12-31'].sum() / CAPITAL:+.2%}, "
          f"2026: {judged['2026-01-01':].sum() / CAPITAL:+.2%}")


if __name__ == "__main__":
    main()
