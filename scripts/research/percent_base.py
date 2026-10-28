"""What the stand's percent is a percent OF, and what it becomes on the strategy's capital
(docs/research/percent-base-2026-10-28.md).

For each long run the decisions rest on, this script

1. re-runs the stand's own code on the run's own data snapshot, on the judged part
   (the same steps as `evaluate_spec`), and checks that it reproduces the recorded
   `n_periods` and `ann_return_net` of the trial;
2. reads off the dollars behind the stand's return series: the stand's `net_return` of a
   bar is a SHARE of the book's equity at that bar, so the dollars of the bar are
   `net_return * equity` (for Bv2 the equity of the book the code chose, idle cash
   included);
3. computes the same summaries twice: as the stand does (returns compounded bar to bar,
   every share taken of the equity that has grown) and on the strategy's capital (dollars
   of the bar over the capital the book started with, summed, never compounded).

CPU only: no agent, no network. Nothing is written to the registry: pass a COPY of the
registry in QLAB_DB (the script only reads it) and the snapshots are read from
data/snapshots.

    QLAB_DB=/path/to/copy.db uv run python scripts/research/percent_base.py [--only LABEL ...]

The result of each case is cached in a pickle (`--cache`), so the table can be printed again
without the re-run.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import decay
from qlab.data.snapshot import load_snapshot
from qlab.diagnose import equity_from_returns
from qlab.harness.costs import CostModel
from qlab.harness.metrics import (
    COUNTED_YEAR_DAYS,
    compute_metrics,
    periods_per_year,
    year_breakdown,
)
from qlab.harness.run import run_backtest
from qlab.harness.strategy import own_costs
from qlab.pipeline.evaluate import (
    _slice_rows,
    complete_book_window,
    cut_truncated_history,
    resolve_strategy,
    split_at_fixed_date,
)
from qlab.pipeline.spec import load_spec
from qlab.registry.db import get_db_path, session_scope
from qlab.strategies.live.bv2 import LiveBv2
from qlab.strategies.retune import expand_grid

DEFAULT_CACHE = Path("percent_base_cache.pkl")


@dataclass(frozen=True)
class Case:
    """One run: its label, the trial it reproduces, the spec file, the capital."""

    label: str
    trial_id: int
    spec_path: str
    capital: float  # the book's strategy capital in dollars (night/paper_books.yaml)
    kind: str  # "bv2" (the book keeps cash), "stateless" (weights are shares of equity),
    # "budget" (weights are dollars over a fixed budget: FRAB)
    overrides: tuple[tuple[str, object], ...] = ()  # spec fields the recorded run had
    recorded_run: bool = True  # False: the trial only lends its data snapshot


CASES = (
    Case("Bv2 whole Binance (94487)", 94487,
         "specs/research/bv2/bv2-research-shared-pool-causal.yaml", 257.0, "bv2",
         (("judged_from", None),)),
    Case("Bv2 twin Binance, 513 d (105587)", 105587,
         "specs/research/bv2/bv2-research-headroom-causal.yaml", 257.0, "bv2"),
    Case("Bv2 twin Hyperliquid, 173 d (105788)", 105788,
         "specs/generated/stage2/bv2-research-headroom-causal-hl.yaml", 257.0, "bv2"),
    Case("Bv2 prod config Binance, 513 d (101165)", 101165, "specs/bv2-binance.yaml", 257.0,
         "bv2"),
    Case("Bv2 prod config Binance, fresh book 2023-04 .. 2026-10 (data of 101165)", 101165,
         "specs/bv2-binance.yaml", 257.0, "bv2",
         (("params_fixed_at", None), ("params_fit_from", None)), recorded_run=False),
    Case("Bv2 prod config Hyperliquid, 173 d (100361)", 100361, "specs/bv2-live.yaml", 257.0,
         "bv2"),
    Case("XSMOM retune, Binance (85838)", 85838,
         "specs/research/xsmom/xsmom-research-nobull-retune.yaml", 1.0, "stateless"),
    Case("XSMOM twin, Binance (92276)", 92276,
         "specs/research/xsmom/xsmom-research-nobull-causal.yaml", 1.0, "stateless"),
    Case("XSMOM twin, Hyperliquid (92477)", 92477,
         "specs/generated/stage2/xsmom-research-nobull-causal-hl.yaml", 1.0, "stateless"),
    Case("Trend twin, Binance (93080)", 93080, "specs/trend-binance-cap-gated-causal.yaml",
         1.0, "stateless"),
    Case("Trend capped and gated, whole Binance (snapshot of 100964)", 100964,
         "specs/trend-binance-cap-gated.yaml", 1.0, "stateless",
         (("params_fixed_at", None), ("params_fit_from", None)), recorded_run=False),
    Case("FRAB stand run, 17 d (100562)", 100562, "specs/frab-live.yaml", 90.0, "budget"),
    Case("FRAB retuned vs default, 179 d (104984)", 104984,
         "specs/research/frab/frab-research-live-vs-default-retune.yaml", 90.0, "budget"),
    Case("frab-binance-spot-400, 17 d (104180)", 104180,
         "specs/research/frab-binance-spot/frab-binance-spot-400.yaml", 400.0, "budget"),
)


@dataclass
class CaseResult:
    case: Case
    recorded: dict[str, float]
    standard: dict[str, float]  # compute_metrics + year_breakdown on the stand's returns
    net_return: pd.Series  # the stand's judged-part returns (shares of equity)
    dollars: pd.Series  # dollars of each bar
    equity: pd.Series  # the dollars the share is a share of, at the decision bar
    coins_held: pd.Series | None  # Bv2: dollars in coins at the decision bar
    judged: tuple[pd.Timestamp, pd.Timestamp]


def recorded_metrics(trial_id: int) -> tuple[str, dict[str, float]]:
    """The trial's snapshot id and metrics, read-only."""
    path = get_db_path()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        snapshot_id, metrics = con.execute(
            "select snapshot_id, metrics from trial where id = ?", (trial_id,)).fetchone()
    finally:
        con.close()
    import json

    return snapshot_id, json.loads(metrics)


def judged_rows(spec, panel) -> tuple[int, int]:
    """The rows `evaluate_spec` judges: the longest complete-book stretch, from
    `judged_from`, then the forward part when the spec splits fit and forward."""
    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    first, last = 0, len(panel.prices.index) - 1
    if coverage is not None and coverage.coverage < 1.0 and coverage.window is not None:
        first, last = coverage.window
    if spec.judged_from is not None:
        first = max(first, min(
            int(panel.prices.index.searchsorted(pd.Timestamp(spec.judged_from, tz="UTC"))),
            last))
    split = split_at_fixed_date(panel.prices.index, first, last, spec.params_fixed_at,
                                spec.params_fit_from)
    if spec.selects_causally:
        split = dataclasses.replace(split, selection=None)
    judged = split.forward if split.forward is not None else split.selection
    return judged if judged is not None else (first, last)


def bv2_dollars(strategy, panel, spec, result) -> tuple[pd.Series, pd.Series]:
    """Equity and coins held, in dollars at each decision bar, of the book whose weights the
    stand used: a Retune twin trades the candidate it chose that month, a plain Bv2 its own."""
    params = dict(spec.params)
    if "grid" in params:
        candidates = expand_grid(params["base_params"], params["grid"])
        runs = {name: LiveBv2().run(panel, p) for name, p in candidates.items()}
        choices = strategy._last.choices
        index = result.net_return.index
        equity = pd.Series(np.nan, index=index)
        coins = pd.Series(np.nan, index=index)
        edges = list(choices.index) + [panel.prices.index[-1] + pd.Timedelta(hours=1)]
        for i, start in enumerate(choices.index):
            name = choices.iloc[i]
            rows = (index >= start) & (index < edges[i + 1])
            if not isinstance(name, str):
                continue  # nothing chosen: no position, a zero return
            book = runs[name]
            equity[rows] = book.book_equity.reindex(index)[rows]
            coins[rows] = _coin_dollars(book, index)[rows]
        return equity, coins
    book = LiveBv2().run(panel, params)
    index = result.net_return.index
    return book.book_equity.reindex(index), _coin_dollars(book, index)


def _coin_dollars(book, index) -> pd.Series:
    spot = [c for c in book.weights.columns if c.endswith("-SPOT")]
    held = book.weights.reindex(index)[spot].clip(lower=0).sum(axis=1)
    return held * book.book_equity.reindex(index)


def run_case(case: Case) -> CaseResult:
    snapshot_id, recorded = recorded_metrics(case.trial_id)
    spec = load_spec(case.spec_path)
    if case.overrides:
        spec = spec.model_copy(update=dict(case.overrides))
    with session_scope() as session:
        panel = load_snapshot(snapshot_id, session=session)
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    lo, hi = judged_rows(spec, panel)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps)
    extra_cost = own_costs(strategy, panel, spec.params)
    part = panel if (lo, hi) == (0, len(panel.prices.index) - 1) else _slice_rows(panel, lo, hi)
    part_weights = weights if part is panel else weights.iloc[lo:hi + 1]
    result = run_backtest(part, part_weights, costs, part.funding, extra_cost=extra_cost)
    standard = compute_metrics(result)
    standard.update(year_breakdown(result.net_return))
    r = result.net_return

    coins = None
    if case.kind == "bv2":
        equity, coins = bv2_dollars(strategy, panel, spec, result)
        equity, coins = equity.reindex(r.index), coins.reindex(r.index)
        dollars = (r * equity).fillna(0.0)
    elif case.kind == "budget":
        equity = pd.Series(case.capital, index=r.index)  # the share is a share of the budget
        dollars = r * case.capital
    else:
        # Weights are shares of the book's equity. The stand lets that equity grow with the
        # profit (compounding); on the strategy's capital the same shares stay shares of the
        # capital the book started with (the live XSMOM engine sizes from a fixed
        # `budget_cap`, frab/strategy/xsmom/params.py).
        equity = pd.Series(case.capital, index=r.index)
        dollars = r * case.capital
    return CaseResult(case, recorded, standard, r, dollars, equity, coins,
                      (r.index[0], r.index[-1]))


# ---- the two ways to read the same dollars -------------------------------------------------

def capital_metrics(rho: pd.Series) -> dict[str, float]:
    """The summaries on the strategy's capital: `rho` is each bar's dollars over the capital,
    summed, never compounded. Same keys and conventions as the stand's, where they exist:
    the annualisation factor of the full series, the reduced bar count as the base for the
    'without the best ...' summaries, years counted from `COUNTED_YEAR_DAYS` days."""
    n = len(rho)
    ppy = periods_per_year(rho.index)
    total = float(rho.sum())
    cum = rho.cumsum()
    peak = np.maximum(cum.cummax(), 0.0)
    out = {
        "total": total,
        "ann": total * ppy / n,
        "sharpe": float(rho.mean() / rho.std(ddof=1) * np.sqrt(ppy)),
        "max_dd_pp": float((cum - peak).min()),  # fall in points of the capital
        "max_dd_vs_peak": float(((1.0 + cum) / (1.0 + peak) - 1.0).min()),
    }
    daily = rho.resample("1D").sum() if ppy > 366.0 * 1.01 else rho
    n_best = round(len(daily) * 0.01)
    rest = daily.drop(daily.nlargest(n_best).index) if n_best > 0 else daily
    out["ex_best_1pct"] = float(rest.sum() * periods_per_year(daily.index) / len(rest))

    counted: dict[int, float] = {}
    years: dict[int, tuple[float, int]] = {}
    for year, bars in rho.groupby(rho.index.year):
        days = int(bars.index.normalize().nunique())
        years[int(year)] = (float(bars.sum()), days)
        if days >= COUNTED_YEAR_DAYS:
            counted[int(year)] = float(bars.sum())
    out["years"] = years  # type: ignore[assignment]
    if counted:
        out["years_positive_share"] = sum(v > 0 for v in counted.values()) / len(counted)
    if len(counted) >= 2:
        best = max(counted, key=counted.get)  # type: ignore[arg-type]
        rest_year = rho[rho.index.year != best]
        out["ex_best_year"] = float(rest_year.sum() * ppy / len(rest_year))
        out["best_year"] = best
        if total > 0:
            out["best_year_share"] = counted[best] / total
    return out


def worst_episodes(curve: pd.Series, *, ratio: bool, top: int = 3) -> list[tuple[float, str, str]]:
    """The deepest falls of a curve that starts at 0 (capital points) or 1 (ratio to the peak):
    (depth, peak day, trough day), deepest first, each bar in one episode only."""
    values = curve.copy()
    out = []
    for _ in range(top):
        peak = values.cummax()
        fall = (values / peak - 1.0) if ratio else (values - peak)
        if fall.min() >= 0:
            break
        trough = fall.idxmin()
        top_at = values[:trough].idxmax()
        recovery = values[trough:][values[trough:] >= values[top_at]]
        end = recovery.index[0] if len(recovery) else values.index[-1]
        out.append((float(fall.min()), str(top_at.date()), str(trough.date())))
        values = values.drop(values[top_at:end].index)
    return out


def stand_years(standard: dict[str, float]) -> dict[int, tuple[float, int]]:
    return {int(k.split("_")[1]): (v, int(standard[f"year_{k.split('_')[1]}_days"]))
            for k, v in standard.items() if k.startswith("year_") and k.endswith("_return")}


def additive_metrics(r: pd.Series) -> dict[str, float]:
    """The stand's own shares, summed instead of compounded: what compounding alone does."""
    return capital_metrics(r)


def reproduce(res: CaseResult) -> str:
    rec, std = res.recorded, res.standard
    if not res.case.recorded_run:
        return (f"not a recorded run (the trial only lends its snapshot): "
                f"n_periods {int(std['n_periods'])}, ann {std['ann_return_net']:+.4f}")
    ok_n = int(rec["n_periods"]) == int(std["n_periods"])
    diff = abs(rec["ann_return_net"] - std["ann_return_net"])
    return (f"n_periods {int(std['n_periods'])} (recorded {int(rec['n_periods'])})"
            f"{'' if ok_n else ' MISMATCH'}; ann {std['ann_return_net']:+.4f} "
            f"(recorded {rec['ann_return_net']:+.4f}, diff {diff:.4f})")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def describe(res: CaseResult) -> None:
    c, r = res.case, res.net_return
    print(f"\n=== {c.label} ===")
    print("reproduces:", reproduce(res))
    days = (r.index[-1] - r.index[0]).total_seconds() / 86400 + 1 / 24
    print(f"judged {r.index[0]} .. {r.index[-1]} ({days:.0f} days)")
    print(f"equity the share is of: first {res.equity.dropna().iloc[0]:.1f}, "
          f"mean {res.equity.mean():.1f}, last {res.equity.dropna().iloc[-1]:.1f}")
    if res.coins_held is not None:
        print(f"coins held: mean ${res.coins_held.mean():.1f}, last "
              f"${res.coins_held.dropna().iloc[-1]:.1f}")
    std = res.standard
    cap = capital_metrics(res.dollars / c.capital)
    add = additive_metrics(r)
    print("                     stand(compound)   shares summed   on capital")
    print(f"total               {pct(std['total_return_net']):>14} {pct(add['total']):>14} "
          f"{pct(cap['total']):>14}")
    print(f"per year            {pct(std['ann_return_net']):>14} {pct(add['ann']):>14} "
          f"{pct(cap['ann']):>14}")
    print(f"Sharpe              {std['sharpe_net']:>14.2f} {add['sharpe']:>14.2f} "
          f"{cap['sharpe']:>14.2f}")
    print(f"max drawdown        {pct(std['max_dd']):>14} {pct(add['max_dd_pp']):>14} "
          f"{pct(cap['max_dd_pp']):>14}   (vs peak equity: {pct(cap['max_dd_vs_peak'])})")
    print(f"ex best 1% days     {pct(std['ann_return_net_ex_best_1pct']):>14} "
          f"{pct(add['ex_best_1pct']):>14} {pct(cap['ex_best_1pct']):>14}")
    print(f"ex best year        {pct(std.get('ann_return_net_ex_best_year', float('nan'))):>14} "
          f"{pct(add.get('ex_best_year', float('nan'))):>14} "
          f"{pct(cap.get('ex_best_year', float('nan'))):>14}")
    print(f"best year share     {pct(std.get('best_year_share_of_profit', float('nan')), 0):>14} "
          f"{pct(add.get('best_year_share', float('nan')), 0):>14} "
          f"{pct(cap.get('best_year_share', float('nan')), 0):>14}")
    print(f"years positive      {pct(std.get('years_positive_share', float('nan')), 0):>14} "
          f"{'':>14} {pct(cap.get('years_positive_share', float('nan')), 0):>14}")
    s_years, c_years = stand_years(std), cap["years"]
    print("by year (days): stand | on capital")
    for y in sorted(c_years):
        print(f"  {y} ({c_years[y][1]:3d} d)  {pct(s_years[y][0]):>9} | {pct(c_years[y][0]):>9}")
    print(f"dollars: sum {res.dollars.sum():+.2f}")
    if c.kind == "bv2":
        stand_curve = (1.0 + r).cumprod()
        cap_curve = (res.dollars / c.capital).cumsum()
        print("deepest falls, stand (of the grown equity): " + "; ".join(
            f"{pct(d)} {a}..{b}" for d, a, b in worst_episodes(stand_curve, ratio=True)))
        print("deepest falls, in points of the capital:    " + "; ".join(
            f"{pct(d)} {a}..{b}" for d, a, b in worst_episodes(cap_curve, ratio=False)))


def agreement(cache: dict[str, CaseResult]) -> None:
    """The stage-2 agreement check (`qlab.pipeline.confirm`): where the Hyperliquid result
    falls among the same-length stretches of the Binance twin's history, read the stand's way
    (compounded, shares of the grown equity) and on the strategy's capital."""
    binance = next(c for c in CASES if c.trial_id == 105587)
    hl = cache["Bv2 twin Hyperliquid, 173 d (105788)"]
    snapshot_id, _ = recorded_metrics(binance.trial_id)
    spec = load_spec(binance.spec_path)
    with session_scope() as session:
        panel = load_snapshot(snapshot_id, session=session)
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps)
    result = run_backtest(panel, weights, costs, panel.funding)
    first, _ = complete_book_window(panel, spec.required_instruments).window
    equity, _ = bv2_dollars(strategy, panel, spec, result)
    net = result.net_return.iloc[first:]
    keep = net.index >= pd.Timestamp(spec.params_fixed_at, tz="UTC")
    net, dollars = net[keep], (net * equity.reindex(net.index)).fillna(0.0)[keep]
    start = np.flatnonzero(net.to_numpy() != 0)[0]  # decay.trim_flat_start
    net, dollars = net.iloc[start:], dollars.iloc[start:]
    window = 173

    closes = decay.daily_closes(equity_from_returns(net))
    stand = np.array([w.total for w in decay.rolling_windows(closes, window)])
    hl_stand = float((1.0 + hl.net_return).prod() - 1.0)

    cumulative = dollars.cumsum() / binance.capital
    ends = cumulative.copy()
    ends.index = ends.index + pd.Timedelta(hours=1)
    days = (ends.index - pd.Timedelta(milliseconds=1)).floor("D")
    daily = ends.groupby(days).last()
    full = pd.date_range(daily.index[0], daily.index[-1], freq="D", tz="UTC")
    values = daily.reindex(full).ffill().to_numpy()
    fixed = np.array([values[i + window] - values[i] for i in range(len(values) - window)])
    hl_fixed = float(hl.dollars.sum() / binance.capital)

    print(f"\n=== agreement of the two stages, windows of {window} days ===")
    print(f"Binance history {net.index[0].date()} .. {net.index[-1].date()}: "
          f"{len(stand)} windows ({len(fixed)} on capital)")
    print(f"as the stand does it:  Hyperliquid {hl_stand:+.4f}; Binance windows "
          f"median {np.median(stand):+.4f}, p10 {np.percentile(stand, 10):+.4f}, "
          f"p90 {np.percentile(stand, 90):+.4f}; better than "
          f"{decay.percentile_of(hl_stand, stand):.1f}%")
    print(f"on the capital:        Hyperliquid {hl_fixed:+.4f}; Binance windows "
          f"median {np.median(fixed):+.4f}, p10 {np.percentile(fixed, 10):+.4f}, "
          f"p90 {np.percentile(fixed, 90):+.4f}; better than "
          f"{decay.percentile_of(hl_fixed, fixed):.1f}%")
    last_year = fixed[-365:]
    print(f"Binance windows that start in the last 365 days, on capital: median "
          f"{np.median(last_year):+.4f}; Hyperliquid better than "
          f"{decay.percentile_of(hl_fixed, last_year):.1f}% of them")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="substrings of case labels")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--fresh", action="store_true", help="ignore the cache")
    parser.add_argument("--agreement", action="store_true",
                        help="also print the stage-2 agreement check on both bases")
    args = parser.parse_args()
    cache_path = Path(args.cache)
    cache: dict[str, CaseResult] = {}
    if cache_path.exists() and not args.fresh:
        cache = pickle.loads(cache_path.read_bytes())
    for case in CASES:
        if args.only and not any(s in case.label for s in args.only):
            continue
        if case.label not in cache:
            cache[case.label] = run_case(case)
            cache_path.write_bytes(pickle.dumps(cache))
        describe(cache[case.label])
    if args.agreement:
        agreement(cache)


if __name__ == "__main__":
    main()
