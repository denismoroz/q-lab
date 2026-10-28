# ruff: noqa: E501  (a table-printing analysis script: long f-strings are the output format)
"""Where the money of `xsmom-research-nobull-causal` comes from, coin by coin and day by day
(docs/research/xsmom-anomaly-2026-10-28.md).

The run: cross-sectional momentum, 20 legs, honest universe, a regime gate, numbers re-chosen
monthly from the past only; Binance 2020-01-01..2026-10-25 (trial 108000) and its second
stage on Hyperliquid 2025-01-01..2026-10-25 (trial 108201). It earns +36% a year on Binance
and almost all of 2026's +313% comes from ten days. This script

1. re-runs the stored spec on the stored snapshot (the same steps as `qlab.forward_series.replay`)
   and checks that it reproduces the stored daily series point by point;
2. splits every day's net return into coins: weight held, price move, funding, costs;
3. lists the coins behind the best days (and behind 2026-06-15..2026-09-15), with their traded
   volume, listing and delisting facts and whether Hyperliquid listed them;
4. checks the prints of those coin-days (jump reversed next day, flat lines, volume, a second
   venue, a second source);
5. re-runs the book under counterfactual rules fixed in advance (a volume line, no coin in the
   last N days before its delisting, a cap on one coin's share of a day, only coins Hyperliquid
   lists, positions left to drift between rebalances instead of being held at constant weight);
6. looks at the other Binance-history runs of the registry for the same coins.

CPU only: no agent, no network. Nothing is written to the registry or to the data folders
(read-only). Needs the registry (`data/qlab.db`, or `QLAB_DB`) and the snapshots in
`data/snapshots`.

    uv run python scripts/research/xsmom_anomaly.py [--only SECTION ...]

Sections: reproduce, decompose, events, spells, checks, counterfactuals, others.  Everything is printed
as plain tables; the doc quotes them.

"Series date" below is the stand's decision date t (the date in data/forward_series): the weight
is decided at the close of UTC day t and earns the move of UTC day t+1. "Market day" is t+1.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)  # the registry and the snapshots are found by relative paths
sys.path.insert(0, str(ROOT / "src"))

from qlab.data.snapshot import load_snapshot  # noqa: E402
from qlab.harness.costs import CostModel  # noqa: E402
from qlab.harness.gaps import held_through, holdable_gaps, marked_through  # noqa: E402
from qlab.harness.metrics import (  # noqa: E402
    ann_return_net_ex_best_1pct,
    compute_metrics,
    year_breakdown,
)
from qlab.harness.panel import MarketPanel  # noqa: E402
from qlab.harness.run import RunResult, run_backtest  # noqa: E402
from qlab.harness.strategy import own_costs  # noqa: E402
from qlab.pipeline.evaluate import (  # noqa: E402
    _slice_rows,
    cut_truncated_history,
    judged_rows,
    resolve_strategy,
)
from qlab.pipeline.spec import load_spec  # noqa: E402
from qlab.registry.db import session_scope  # noqa: E402
from qlab.rules import loader  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

# --------------------------------------------------------------------------------------
# What is analysed

BINANCE = {
    "label": "binance",
    "trial": 108000,
    "spec": "specs/research/xsmom/xsmom-research-nobull-causal.yaml",
    "snapshot": "b5b98b2af1cad42968ac53c158c59e3d77c3851f76a9631f481cb78ac384ccf7",
}
HYPERLIQUID = {
    "label": "hyperliquid",
    "trial": 108201,
    "spec": "specs/generated/stage2/xsmom-research-nobull-causal-hl.yaml",
    "snapshot": "d9f232c66b6d275ca703b8f73ecf20c7990432454d66bc14f77f039b2c1532d1",
}
WINDOW = ("2026-06-15", "2026-09-15")  # the months the owner's real XSMOM traded live: growth
# from the stored point of 06-15 to the stored point of 09-15, i.e. series dates 06-16..09-15
RAW_BINANCE = Path("data/raw/binance/1d")
RAW_HL = Path("data/raw/hyperliquid/1d")
CMC_DIR = Path("data/raw/coinmarketcap/historical")

# Counterfactual rules, FIXED BEFORE LOOKING AT THEIR RESULTS (the doc states why):
VOLUME_LINES_USD = (1e6, 5e6)  # 1M: the floor frab's live curation uses (MIN_VOL_USD);
# 5M: five times stricter. Two more lines are run only as a dose check (not as candidate rules):
VOLUME_LINES_DOSE_USD = (50e6, 250e6)
DELIST_DAYS = 30  # no coin in the 30 days before its delisting (a month of announcement time)
CAP_SHARES = (0.02, 0.01)  # one coin-day may not add or take more than 2% / 1% of the book
CAP_GAIN_ONLY = 0.02  # pessimist's version: gains capped at 2%, losses left as they were


# --------------------------------------------------------------------------------------
# Re-running a stored run


@dataclass
class Run:
    label: str
    trial: int
    spec: object
    panel: MarketPanel  # the whole panel (the strategy sees all of it)
    weights: pd.DataFrame  # the strategy's weights on the whole panel
    lo: int  # judged rows lo..hi of the panel
    hi: int
    part: MarketPanel  # the judged slice
    part_weights: pd.DataFrame
    costs: CostModel
    extra_cost: pd.Series | None
    result: RunResult  # the backtest of the judged slice


def _judge(panel: MarketPanel, spec, ruleset):
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    rows = judged_rows(spec, panel, None, ruleset)
    return panel, rows


def load_run(cfg: dict, *, panel: MarketPanel | None = None) -> Run:
    """The stored spec on the stored snapshot, run as `qlab.forward_series.replay` runs it."""
    spec = load_spec(Path(cfg["spec"]))
    if panel is None:
        with session_scope() as session:
            panel = load_snapshot(cfg["snapshot"], session=session)
    ruleset = loader.load_latest()
    panel, rows = _judge(panel, spec, ruleset)
    return run_spec(cfg["label"], cfg["trial"], spec, panel, rows, ruleset)


def run_spec(label: str, trial: int, spec, panel: MarketPanel, rows, ruleset) -> Run:
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps, slippage_bps=spec.costs.slippage_bps)
    extra = own_costs(strategy, panel, spec.params)
    lo, hi = rows.judged
    part = _slice_rows(panel, lo, hi)
    part_w = weights.iloc[lo : hi + 1]
    result = run_backtest(part, part_w, costs, part.funding, extra_cost=extra)
    return Run(label, trial, spec, panel, weights, lo, hi, part, part_w, costs, extra, result)


def rerun_on(base: Run, panel: MarketPanel, tag: str) -> Run:
    """The same spec and the same judged rows on a changed panel (a counterfactual). The
    snapshot id gets a suffix: `Retune` caches by (snapshot id, index, columns, params) and
    would otherwise return the base run's answer for a panel whose `tradeable` differs."""
    changed = dataclasses.replace(panel, snapshot_id=f"{panel.snapshot_id}:{tag}")
    ruleset = loader.load_latest()
    rows = dataclasses.replace(
        judged_rows(base.spec, changed, None, ruleset), judged=(base.lo, base.hi)
    )
    return run_spec(base.label, base.trial, base.spec, changed, rows, ruleset)


# --------------------------------------------------------------------------------------
# Splitting a run into coins


@dataclass
class Parts:
    """Per decision date t (rows) and instrument (columns); all shares of the book's equity."""

    w: pd.DataFrame  # weight held over (t, t+1] (after the harness holds through price gaps)
    ret: pd.DataFrame  # price move of that interval
    price: pd.DataFrame  # w * ret
    funding: pd.DataFrame  # -w * funding rate of the interval (a long pays a positive rate)
    cost: pd.DataFrame  # |change of weight| * (fee + slippage)
    net: pd.Series  # sum of the three, equals the harness's net return

    @property
    def contribution(self) -> pd.DataFrame:
        return self.price + self.funding - self.cost


def decompose(part: MarketPanel, weights: pd.DataFrame, costs: CostModel) -> Parts:
    """The harness's own arithmetic (`qlab.harness.run.run_backtest`) kept per instrument."""
    keep = part.prices.index[:-1]
    gaps = holdable_gaps(part.prices, part.funding, part.meta.get("no_funding_instruments", ()))
    wh = held_through(weights, gaps)
    pr = marked_through(part.prices, gaps)
    ret = pr.pct_change().shift(-1).loc[keep]
    wk = wh.loc[keep]
    prev = wh.shift(1).fillna(0.0).loc[keep]
    price = wk * ret
    fund = -(wk * part.funding.shift(-1).loc[keep].fillna(0.0))
    cost = (wk - prev).abs() * (costs.total_bps / 1e4)
    net = price.sum(axis=1) + fund.sum(axis=1) - cost.sum(axis=1)
    return Parts(wk, ret, price.fillna(0.0), fund, cost, net)


def summary(net: pd.Series) -> dict[str, float]:
    """The stand's own metrics on a daily net-return series (indexed by decision date)."""
    stub = RunResult("", net, net, net * 0.0, net * 0.0, net * 0.0)
    m = compute_metrics(stub)
    out = {
        "ann": m["ann_return_net"],
        "total": m["total_return_net"],
        "max_dd": m["max_dd"],
        "ann_ex_best_1pct": ann_return_net_ex_best_1pct(net),
        "sharpe": m["sharpe_net"],
    }
    for key, value in year_breakdown(net).items():
        if key.startswith("year_") and key.endswith("_return"):
            out[key.split("_")[1]] = value
    return out


def window_return(net: pd.Series, a: str = WINDOW[0], b: str = WINDOW[1]) -> float:
    idx = pd.DatetimeIndex(net.index).tz_convert("UTC").normalize()
    sel = (idx > pd.Timestamp(a, tz="UTC")) & (idx <= pd.Timestamp(b, tz="UTC"))
    return float((1.0 + net[sel]).prod() - 1.0)


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{digits}f}%"


# --------------------------------------------------------------------------------------
# Facts about coins


def contract_facts(coin: str, venue: str = "binance") -> dict:
    """First/last print and delisting flag of a perpetual, from the raw file's sidecar."""
    folder, start = (RAW_BINANCE, "2019-09-08") if venue == "binance" else (RAW_HL, "2025-01-01")
    paths = sorted(folder.glob(f"{coin}__{start}__*.json"))
    if not paths:
        return {}
    return json.loads(paths[-1].read_text(encoding="utf-8"))


_HL_CACHE: dict = {}


def hl_panel() -> MarketPanel:
    if "p" not in _HL_CACHE:
        with session_scope() as session:
            _HL_CACHE["p"] = load_snapshot(HYPERLIQUID["snapshot"], session=session)
    return _HL_CACHE["p"]


def hl_name(binance_coin: str, hl_columns: set[str]) -> str | None:
    """The Hyperliquid name of a Binance contract: the same ticker, or `k` + ticker for a
    Binance `1000`-prefixed contract (1000PEPE <-> kPEPE)."""
    if binance_coin in hl_columns:
        return binance_coin
    if binance_coin.startswith("1000") and "k" + binance_coin[4:] in hl_columns:
        return "k" + binance_coin[4:]
    return None


def hl_listed_mask(panel: MarketPanel) -> pd.DataFrame:
    """True where Hyperliquid has a price for the same coin at that bar (2025-01-01 on: the
    Hyperliquid snapshot's own window)."""
    hl = hl_panel()
    cols = set(hl.prices.columns)
    out = pd.DataFrame(False, index=panel.prices.index, columns=panel.prices.columns)
    common = panel.prices.index.intersection(hl.prices.index)
    for coin in panel.prices.columns:
        name = hl_name(coin, cols)
        if name is not None:
            out.loc[common, coin] = hl.prices.loc[common, name].notna().to_numpy()
    return out


def usd_volume(panel: MarketPanel) -> pd.DataFrame:
    """Traded volume of the bar in dollars (the panel stores base units; docs: recomputed)."""
    return panel.volume * panel.prices


# --------------------------------------------------------------------------------------
# Section 1: reproduce


def section_reproduce(runs: dict[str, Run]) -> None:
    print("\n=== 1. REPRODUCTION OF THE STORED DAILY SERIES ===")
    for cfg in (BINANCE, HYPERLIQUID):
        run = runs[cfg["label"]]
        stored = json.loads(Path(f"data/forward_series/{cfg['trial']}.json").read_text())
        pts = pd.DataFrame(stored["points"][1:], columns=["date", "strategy", "hold"])
        pts["date"] = pd.to_datetime(pts["date"], utc=True)
        pts = pts.set_index("date")
        days = pd.DatetimeIndex(run.result.net_return.index).tz_convert("UTC").normalize()
        daily = (1.0 + run.result.net_return).groupby(days).prod() - 1.0
        mine = (1.0 + daily).cumprod() - 1.0
        diff = (mine - pts["strategy"]).abs()
        print(
            f"{cfg['label']:12s} trial {cfg['trial']}: {len(mine)} days, stored {len(pts)}; "
            f"max |difference| of the cumulative return = {diff.max():.2e}; "
            f"total {mine.iloc[-1]:+.6f} vs stored {stored['total_return']:+.6f}"
        )
        s = summary(run.result.net_return)
        print(
            f"   ann {pct(s['ann'])} (stored {pct(stored['ann_return'])}), worst fall "
            f"{pct(s['max_dd'])} (stored {pct(stored['max_dd'])}), without best 1% of days "
            f"{pct(s['ann_ex_best_1pct'])}"
        )
        years = {k: v for k, v in s.items() if k.isdigit()}
        print("   years: " + ", ".join(f"{y} {pct(v, 0)}" for y, v in years.items()))
        print(f"   window {WINDOW[0]}..{WINDOW[1]}: {pct(window_return(run.result.net_return))}")
        net = run.result.net_return
        for a_, b_ in (("2020-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")):
            sub = net[(net.index >= pd.Timestamp(a_, tz="UTC")) & (net.index <= pd.Timestamp(b_, tz="UTC"))]
            if len(sub) > 1:
                tot = float((1 + sub).prod() - 1)
                ann = (1 + tot) ** (365.25 / len(sub)) - 1
                print(f"   {a_}..{min(b_, str(net.index[-1].date()))}: total {pct(tot, 0)}, {len(sub)} days, annual {pct(ann)}")
        hold = pts["hold"]
        a, b = (pd.Timestamp(x, tz="UTC") for x in WINDOW)
        h = (1 + hold[b]) / (1 + hold[a]) - 1
        print(f"   holding the same coins (stored hold line) over the window: {pct(h)}")


# --------------------------------------------------------------------------------------
# Section 2/3: decomposition and events


def top_coin_rows(run: Run, parts: Parts, days: list[pd.Timestamp], k: int = 4) -> pd.DataFrame:
    usd = usd_volume(run.part)
    idx = run.part.prices.index
    hl_mask = hl_listed_mask(run.part) if run.label == "binance" else None
    rows = []
    for t in days:
        pos = idx.get_loc(t)
        nxt = idx[pos + 1]
        contrib = parts.contribution.loc[t]
        order = contrib[parts.w.loc[t].abs() > 0].abs().sort_values(ascending=False).index[:k]
        for coin in order:
            facts = contract_facts(coin, run.label)
            vol_prior = usd[coin].iloc[max(0, pos - 29) : pos + 1].median()
            first_seen = pd.Timestamp(facts["first_seen"]) if facts else None
            rows.append(
                {
                    "series_date": t.date(),
                    "day_net": parts.net.loc[t],
                    "coin": coin,
                    "side": "long" if parts.w.loc[t, coin] > 0 else "short",
                    "weight": parts.w.loc[t, coin],
                    "move": parts.ret.loc[t, coin],
                    "price": parts.price.loc[t, coin],
                    "funding": parts.funding.loc[t, coin],
                    "usd_vol_M": usd.loc[nxt, coin] / 1e6,
                    "prior30_median_vol_M": vol_prior / 1e6,
                    "age_days": (t - first_seen).days
                    if first_seen is not None and run.label == "binance" else np.nan,
                    "delisted": facts.get("is_delisted") if facts else None,
                    "last_print": str(facts.get("last_seen", ""))[:10] if facts else "",
                    "hl_listed": (None if hl_mask is None else bool(hl_mask.loc[t, coin])),
                }
            )
    return pd.DataFrame(rows)


def fmt_rows(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in ("day_net", "weight", "move", "price", "funding"):
        if c in d:
            d[c] = d[c].map(lambda x: f"{100 * x:+.2f}%")
    for c in ("usd_vol_M", "prior30_median_vol_M"):
        if c in d:
            d[c] = d[c].map(lambda x: f"{x:,.1f}")
    return d.to_string(index=False)


def best_days(net: pd.Series, year: int, n: int = 10) -> list[pd.Timestamp]:
    sel = net[net.index.year == year]
    return list(sel.sort_values(ascending=False).index[:n])


def section_decompose(runs: dict[str, Run]) -> dict[str, Parts]:
    print("\n=== 2. WHAT THE DAILY RETURN IS MADE OF ===")
    out = {}
    for label, run in runs.items():
        parts = decompose(run.part, run.part_weights, run.costs)
        out[label] = parts
        gap = (parts.net - run.result.net_return).abs().max()
        print(
            f"{label}: decomposition equals the harness's net return to {gap:.1e}; sums over "
            f"the run (shares of the book, added up): price {parts.price.to_numpy().sum():+.3f}, "
            f"funding {parts.funding.to_numpy().sum():+.3f}, costs {-parts.cost.to_numpy().sum():+.3f}"
        )
        gross = parts.w.abs().sum(axis=1)
        held = (parts.w.abs() > 0).sum(axis=1)
        print(
            f"   gross exposure: median {gross.median():.2f}; legs held: median {held.median():.0f}, "
            f"max weight of a leg {parts.w.abs().max().max():.3f}"
        )
        by_year = pd.DataFrame({"price": parts.price.sum(axis=1), "funding": parts.funding.sum(axis=1),
                                "costs": -parts.cost.sum(axis=1)}).groupby(parts.price.index.year).sum()
        print("   by year (added shares of the book): " + "; ".join(
            f"{y} price {r.price:+.3f} funding {r.funding:+.3f} costs {r.costs:+.3f}" for y, r in by_year.iterrows()))
        f_top = parts.funding.stack().sort_values(ascending=False)
        print(f"   funding: the 10 biggest funding coin-days add {f_top.head(10).sum():+.3f} of {f_top.sum():+.3f}")
        netexp = parts.w.sum(axis=1)
        print(f"   net exposure (longs minus shorts): min {netexp.min():+.3f}, max {netexp.max():+.3f}, "
              f"days with a net exposure above 0.001: {int((netexp.abs() > 0.001).sum())}")
        long_p = (parts.price.where(parts.w > 0, 0.0)).to_numpy().sum()
        short_p = (parts.price.where(parts.w < 0, 0.0)).to_numpy().sum()
        print(f"   price part, long legs {long_p:+.3f}, short legs {short_p:+.3f} (added shares)")
    return out


MOVE_BANDS = (0.0, 0.10, 0.25, 0.50, 1.00, np.inf)


def move_band_table(p: Parts) -> None:
    """Where the money comes from by the size of the coin's own move on the day: coin-days
    (a leg held on a day) sorted by |move| of the coin; added shares of the book."""
    held = p.w.abs() > 0
    stacked = pd.DataFrame({
        "move": p.ret.where(held).stack(),
        "price": p.price.where(held).stack(),
        "funding": p.funding.where(held).stack(),
        "side": np.sign(p.w.where(held).stack()),
    }).dropna(subset=["move"])
    stacked["band"] = pd.cut(stacked["move"].abs(), MOVE_BANDS, right=False)
    total = (stacked["price"] + stacked["funding"]).sum()
    print("   by the size of the coin's own move that day (|move|), legs held:")
    print("   band            coin-days   share   price+funding long / short (added shares)   share of total")
    for band, g in stacked.groupby("band", observed=True):
        lo, hi = band.left, band.right
        name = f"{lo:.0%}..{hi:.0%}" if np.isfinite(hi) else f"{lo:.0%}+"
        gl, gs = g[g.side > 0], g[g.side < 0]
        pf = (g["price"] + g["funding"]).sum()
        print(f"   {name:12s} {len(g):9d}  {len(g) / len(stacked):6.2%}   "
              f"{(gl['price'] + gl['funding']).sum():+8.3f} / {(gs['price'] + gs['funding']).sum():+8.3f}"
              f"   {pf / total:7.0%}")
    ext = stacked[stacked["move"].abs() >= 0.5]
    print(f"   coin-days with |move| >= 50%: {len(ext)} ({len(ext) / len(stacked):.3%} of all), "
          f"they add {(ext['price'] + ext['funding']).sum():+.3f} of {total:+.3f} "
          f"({(ext['price'] + ext['funding']).sum() / total:.0%}); all others {total - (ext['price'] + ext['funding']).sum():+.3f}")


def thin_exposure(run: Run, p: Parts, line: float) -> tuple[int, int, float, float]:
    """(coin-days held with day-before dollar volume under `line`, all coin-days held, what the
    thin ones add, what all add), price + funding, added shares."""
    usd_prev = usd_volume(run.part).shift(1).reindex(p.w.index)
    held = p.w.abs() > 0
    pf = (p.price + p.funding).where(held)
    thin = held & (usd_prev < line)
    add = float(np.nansum(pf.where(thin).to_numpy()))
    return int(thin.to_numpy().sum()), int(held.to_numpy().sum()), add, float(np.nansum(pf.to_numpy()))


def volume_exposure(run: Run, p: Parts) -> None:
    """How much of the book sat in thin coins: legs held on a day whose day-before dollar volume
    was under a line, and what those coin-days added."""
    for line in (1e6, 5e6, 50e6):
        n, n_all, add, total = thin_exposure(run, p, line)
        print(f"   coin-days held with day-before volume under ${line / 1e6:g}M: {n} ({n / n_all:.1%}); "
              f"they add {add:+.3f} of {total:+.3f} (price + funding, added shares)")


def concentration_table(run: Run, parts: Parts) -> None:
    move_band_table(parts)
    volume_exposure(run, parts)
    c = parts.contribution
    stacked = c.stack()
    total = stacked.sum()
    top = stacked.sort_values(ascending=False)
    n_cd = int((parts.w.abs() > 0).to_numpy().sum())
    print(f"   coin-days held: {n_cd}; sum of all contributions {total:+.3f} (added shares)")
    for k in (10, 20, 50, 100):
        print(
            f"   best {k:3d} coin-days add {top.iloc[:k].sum():+.3f} = {top.iloc[:k].sum() / total:.0%} "
            f"of that sum; the worst {k} take {stacked.sort_values().iloc[:k].sum():+.3f}"
        )
    per_coin = c.sum().sort_values(ascending=False)
    print("   best coins over the whole run (sum of contributions): " + ", ".join(
        f"{k} {v:+.3f}" for k, v in per_coin.head(12).items()))
    print("   worst coins: " + ", ".join(f"{k} {v:+.3f}" for k, v in per_coin.tail(8).items()))
    by_year = c.groupby(c.index.year).sum()
    for y in by_year.index:
        row = by_year.loc[y].sort_values(ascending=False)
        print(f"   {y}: best coins " + ", ".join(f"{k} {v:+.3f}" for k, v in row.head(6).items())
              + f" | all coins {row.sum():+.3f}")


def funding_check(run: Run, p: Parts, n: int = 3) -> None:
    """Recompute the biggest funding coin-days from the raw settlements of the venue, with the
    sign convention of the stand (a positive rate is paid by longs)."""
    f = p.funding.stack().sort_values(ascending=False).head(n)
    folder = RAW_BINANCE if run.label == "binance" else RAW_HL
    start = "2019-09-08" if run.label == "binance" else "2025-01-01"
    for (t, coin), value in f.items():
        paths = sorted(folder.glob(f"{coin}__{start}__*.parquet"))
        if not paths:
            continue
        raw = pd.read_parquet(paths[-1])["funding"].dropna()
        window = raw[(raw.index > t) & (raw.index <= t + pd.Timedelta(days=1))]
        by_hand = -p.w.loc[t, coin] * float(window.sum())
        print(f"   funding check {coin} {t.date()}: stand {value:+.5f}, by hand from {len(window)} raw settlements "
              f"(rates {', '.join(f'{x * 1e4:+.0f}bp' for x in window.head(6))}): -weight x sum = {by_hand:+.5f}")


def section_events(runs: dict[str, Run], parts: dict[str, Parts]) -> None:
    print("\n=== 3. THE COINS BEHIND THE BEST DAYS ===")
    for label in ("binance", "hyperliquid"):
        run, p = runs[label], parts[label]
        funding_check(run, p)
        print(f"\n--- {label}: concentration ---")
        concentration_table(run, p)
    run, p = runs["binance"], parts["binance"]
    for year in (2026, 2025):
        days = best_days(p.net, year)
        print(f"\n--- binance: ten best days of {year} (series date; the move is of the next UTC day) ---")
        print(fmt_rows(top_coin_rows(run, p, days)))
        hl_flag = hl_listed_mask(run.part)
        tops = [p.contribution.loc[t].idxmax() for t in days]
        not_hl = sum(not bool(hl_flag.loc[t, c]) for t, c in zip(days, tops, strict=True))
        print(f"   the coin that gave the most on each of the ten days: {', '.join(tops)}; "
              f"not listed on Hyperliquid at the time: {not_hl} of 10")
        net_wo = p.net.drop(days)
        sub = net_wo[net_wo.index.year == year]
        sub_all = p.net[p.net.index.year == year]
        print(
            f"   {year} with them {pct((1 + sub_all).prod() - 1, 0)}; without these ten days "
            f"{pct((1 + sub).prod() - 1, 0)}; the ten days add up (compounded) "
            f"{pct((1 + p.net[days]).prod() - 1, 0)}"
        )
    run, p = runs["hyperliquid"], parts["hyperliquid"]
    for year in (2026, 2025):
        days = best_days(p.net, year, 6)
        print(f"\n--- hyperliquid: six best days of {year} ---")
        print(fmt_rows(top_coin_rows(run, p, days, k=3)))
    # the window
    a, b = (pd.Timestamp(x, tz="UTC") for x in WINDOW)
    for label in ("binance", "hyperliquid"):
        run, p = runs[label], parts[label]
        sel = p.contribution[(p.contribution.index > a) & (p.contribution.index <= b)]
        print(f"\n--- {label}: window {WINDOW[0]}..{WINDOW[1]}: {pct(window_return(p.net))} ---")
        tot = sel.sum().sort_values(ascending=False)
        wsel = p.w.loc[sel.index]
        hl_cols = set(hl_panel().prices.columns)
        def flag(k: str, label: str = label, hl_cols: set = hl_cols) -> str:
            return "" if label != "binance" or hl_name(k, hl_cols) is None else " [on HL]"

        print("   best coins: " + ", ".join(f"{k} {v:+.3f}{flag(k)}" for k, v in tot.head(10).items()))
        if label == "binance":
            print(f"   of the 10 best coins, listed on Hyperliquid: {sum(hl_name(k, hl_cols) is not None for k in tot.head(10).index)}; "
                  f"the best 10 coins add {tot.head(10).sum():+.3f} of {tot.sum():+.3f}")
        print("   worst coins: " + ", ".join(f"{k} {v:+.3f}" for k, v in tot.tail(6).items()))
        print(f"   all coins add {tot.sum():+.3f}; the best 3 coins add {tot.head(3).sum():+.3f}")
        days = list(p.net[sel.index].sort_values(ascending=False).index[:6])
        if label == "binance":
            print("   best days in the window:")
            print(fmt_rows(top_coin_rows(run, p, days, k=3)))
        else:
            print("   best days in the window:")
            for t in days:
                c = sel.loc[t][wsel.loc[t].abs() > 0].abs().sort_values(ascending=False).index[:3]
                print("    ", t.date(), pct(p.net[t]), "; ", ", ".join(
                    f"{k} w{p.w.loc[t, k]:+.2f} move {pct(p.ret.loc[t, k], 0)}" for k in c))


# --------------------------------------------------------------------------------------
# Section 3b: how the big coin-days were entered, and what constant weights do to them


def spells(parts: Parts, panel: MarketPanel) -> pd.DataFrame:
    """Every continuous holding of one coin on one side: entry bar, last held bar, the stand's
    price part (sum of weight x daily move: the weight is put back to its target every day),
    and what the same coin would give as a plain holding bought at the entry close at the entry
    weight and not touched (the move of the coin from entry close to the close after the last
    held bar, times that weight)."""
    idx = parts.w.index
    price_all = panel.prices
    rows = []
    for coin in parts.w.columns:
        w = parts.w[coin].to_numpy()
        if not np.any(w):
            continue
        sign = np.sign(w)
        start = None
        for i in range(len(w) + 1):
            cur = sign[i] if i < len(w) else 0.0
            if start is not None and (cur != sign[start] or i == len(w)):
                end = i - 1  # last held bar
                w0 = w[start]
                p0 = price_all[coin].iloc[price_all.index.get_loc(idx[start])]
                p1 = price_all[coin].iloc[price_all.index.get_loc(idx[end]) + 1]
                stand = float(parts.price[coin].iloc[start : end + 1].sum())
                fund = float(parts.funding[coin].iloc[start : end + 1].sum())
                hold = float(w0 * (p1 / p0 - 1.0)) if p0 and not np.isnan(p0) and not np.isnan(p1) else np.nan
                rows.append((coin, idx[start], idx[end], end - start + 1, w0, p1 / p0 - 1.0,
                             stand, hold, fund))
                start = None
            if start is None and cur != 0.0 and i < len(w):
                start = i
    return pd.DataFrame(rows, columns=["coin", "entry", "last", "days", "w0", "coin_move",
                                       "stand_price", "plain_hold", "funding"])


def section_spells(runs: dict[str, Run], parts: dict[str, Parts]) -> pd.DataFrame:
    print("\n=== 3b. HOLDING SPELLS: THE STAND'S CONSTANT WEIGHT AGAINST A PLAIN HOLDING ===")
    run, p = runs["binance"], parts["binance"]
    sp = spells(p, run.panel)
    sp["diff"] = sp["stand_price"] - sp["plain_hold"]
    print(f"{len(sp)} spells; the stand's price part {sp.stand_price.sum():+.3f}, the same "
          f"spells held plainly {sp.plain_hold.sum():+.3f} (added shares; funding {sp.funding.sum():+.3f} "
          "is the same in both)")
    for y in range(2020, 2027):
        z = sp[sp["entry"].dt.year == y]
        print(f"   spells entered in {y}: stand {z.stand_price.sum():+.3f}, plain holding "
              f"{z.plain_hold.sum():+.3f}")
    longs, shorts = sp[sp.w0 > 0], sp[sp.w0 < 0]
    print(f"   long spells: stand {longs.stand_price.sum():+.3f}, plain {longs.plain_hold.sum():+.3f}; "
          f"short spells: stand {shorts.stand_price.sum():+.3f}, plain {shorts.plain_hold.sum():+.3f}")
    top = sp.sort_values("stand_price", ascending=False).head(15).copy()
    print("   the 15 spells with the biggest stand result:")
    show = top.copy()
    show["entry"] = show["entry"].dt.date
    show["last"] = show["last"].dt.date
    for c in ("w0", "coin_move", "stand_price", "plain_hold", "funding", "diff"):
        show[c] = show[c].map(lambda x: f"{100 * x:+.1f}%")
    print(show.to_string(index=False))
    return sp


def entry_table(run: Run, p: Parts, rows: pd.DataFrame) -> pd.DataFrame:
    """For listed coin-days: how long the leg had been held before the move, and what the coin
    had done before it was bought."""
    idx = p.w.index
    price = run.panel.prices
    out = []
    for _, r in rows.iterrows():
        t, coin = pd.Timestamp(r["t"]), r["coin"]
        w = p.w[coin].to_numpy()
        i = idx.get_loc(t)
        j = i
        while j > 0 and w[j - 1] * w[i] > 0:
            j -= 1
        pe = price.index.get_loc(idx[j])
        pt = price.index.get_loc(t)
        out.append({
            "series_date": t.date(), "coin": coin, "entered": idx[j].date(),
            "days_held_before": i - j,
            "coin_move_since_entry": price[coin].iloc[pt] / price[coin].iloc[pe] - 1.0,
            "coin_14d_at_entry": price[coin].iloc[pe] / price[coin].iloc[pe - 14] - 1.0 if pe >= 14 else np.nan,
            "coin_60d_at_entry": price[coin].iloc[pe] / price[coin].iloc[pe - 60] - 1.0 if pe >= 60 else np.nan,
            "max_day_move_in_prior_7d_of_entry": price[coin].pct_change().iloc[max(pe - 6, 1) : pe + 1].max(),
            "move_that_day": p.ret.loc[t, coin],
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Section 4: data checks on the big coin-days


BIG_COIN_DAY = 0.03  # a coin-day that adds at least 3% of the book in one day


def big_coin_days(p: Parts, threshold: float = BIG_COIN_DAY) -> pd.DataFrame:
    c = p.contribution.stack()
    big = c[c >= threshold].reset_index()
    big.columns = ["t", "coin", "contribution"]
    return big.sort_values("contribution", ascending=False).reset_index(drop=True)


def raw_binance(coin: str) -> pd.DataFrame | None:
    paths = sorted(RAW_BINANCE.glob(f"{coin}__2019-09-08__*.parquet"))
    return pd.read_parquet(paths[-1]) if paths else None


def hourly_binance(coin: str) -> pd.DataFrame | None:
    paths = sorted(Path("data/raw/binance/1h").glob(f"{coin}__*.parquet"))
    return pd.read_parquet(paths[-1]) if paths else None


def cmc_series(symbol: str) -> pd.DataFrame:
    rows = []
    for f in sorted(CMC_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d["rows"]:
            if r["symbol"] == symbol:
                rows.append((pd.Timestamp(d["date"], tz="UTC"), r["cmcRank"], r["price_usd"],
                             r["volume_24h_usd"], r["market_cap_usd"]))
    return pd.DataFrame(rows, columns=["date", "rank", "price", "volume24", "mcap"]).set_index("date")


def section_checks(runs: dict[str, Run], parts: dict[str, Parts]) -> pd.DataFrame:
    print("\n=== 4. DATA CHECKS ON THE BIG COIN-DAYS ===")
    run, p = runs["binance"], parts["binance"]
    big = big_coin_days(p)
    price = run.panel.prices
    usd = usd_volume(run.panel)
    high = run.panel.high
    idx = price.index
    hl = hl_panel()
    hl_cols = set(hl.prices.columns)
    rows = []
    for _, r in big.iterrows():
        t, coin = r["t"], r["coin"]
        pos = idx.get_loc(t)
        day = pos + 1  # the market day
        px = price[coin]
        nxt = px.iloc[day + 1] / px.iloc[day] - 1.0 if day + 1 < len(idx) else np.nan
        after3 = px.iloc[day + 3] / px.iloc[day] - 1.0 if day + 3 < len(idx) else np.nan
        flat10 = int((px.iloc[pos - 9 : pos + 1].diff() == 0).sum())
        prior_med = usd[coin].iloc[max(0, pos - 29) : pos + 1].median()
        raw = raw_binance(coin)
        trades = np.nan
        if raw is not None and idx[day] in raw.index:
            trades = raw.loc[idx[day], "trade_count"]
        h = high[coin].iloc[day]
        name = hl_name(coin, hl_cols)
        hl_move = np.nan
        if name is not None and idx[day] in hl.prices.index and idx[pos] in hl.prices.index:
            a, b = hl.prices[name].loc[idx[pos]], hl.prices[name].loc[idx[day]]
            hl_move = b / a - 1.0 if pd.notna(a) and pd.notna(b) else np.nan
        rows.append({
            "series_date": t.date(), "coin": coin, "contribution": r["contribution"],
            "move": p.ret.loc[t, coin], "next_day_move": nxt, "move_3d_later": after3,
            "flat_days_in_prior10": flat10,
            "usd_vol_M": usd[coin].iloc[day] / 1e6,
            "vol_x_prior30_median": usd[coin].iloc[day] / prior_med if prior_med else np.nan,
            "trades_M": trades / 1e6,
            "high_over_close": (h / px.iloc[day]) if pd.notna(h) else np.nan,
            "hl_same_day_move": hl_move,
        })
    df = pd.DataFrame(rows)
    d = df.copy()
    for c in ("contribution", "move", "next_day_move", "move_3d_later", "hl_same_day_move"):
        d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{100 * x:+.0f}%")
    for c in ("usd_vol_M", "vol_x_prior30_median", "trades_M"):
        d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:,.1f}")
    d["high_over_close"] = d["high_over_close"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    by_year = df.groupby(pd.to_datetime([str(d) for d in df["series_date"]]).year).size()
    print(f"{len(df)} coin-days that add >= {BIG_COIN_DAY:.0%} of the book in one day (by year: "
          + ", ".join(f"{y} {n}" for y, n in by_year.items()) + "); the 30 biggest:")
    print(d.head(30).to_string(index=False))
    mv = df[df["move"] > 0.5]
    print(
        f"\nsummary over all {len(df)}: flat closes in the 10 bars before: "
        f"{int((df.flat_days_in_prior10 > 0).sum())} coin-days; price move below +10%: "
        f"{int((df['move'] < 0.10).sum())} (these are gains of a long in a ~+20% move or a short in a fall); "
        f"moves of +50% or more: {len(mv)}"
    )
    print(
        f"  of the {len(mv)} moves of +50% or more: day's dollar volume at least $100M in "
        f"{int((mv.usd_vol_M >= 100).sum())}, at least $1B in {int((mv.usd_vol_M >= 1000).sum())}, "
        f"median {mv.usd_vol_M.median():,.0f}M; volume at least 10x the prior 30-day median in "
        f"{int((mv.vol_x_prior30_median >= 10).sum())}; the next day gave back at least half "
        f"({'next_day_move <= -0.5'}) in {int((mv.next_day_move <= -0.5).sum())}; "
        f"on a second venue (Hyperliquid) in {int(mv.hl_same_day_move.notna().sum())}"
    )
    pair = df.dropna(subset=["hl_same_day_move"])
    if len(pair):
        print("  Binance move vs Hyperliquid move of the same coin and day (all listed pairs among the big coin-days):")
        print("   " + "; ".join(f"{r.coin} {r.series_date}: {100 * r.move:+.0f}% / {100 * r.hl_same_day_move:+.0f}%"
                                 for r in pair.itertuples()))
    # the same checks over EVERY coin-day held, not only the big ones
    held = p.w.abs() > 0
    usd_day = usd_volume(run.panel).shift(-1).reindex(p.w.index)  # the market day's dollars
    zero_move = int(((p.ret == 0) & held).to_numpy().sum())
    low_vol = int(((usd_day < 1e5) & held).to_numpy().sum())
    big_move = (p.ret.abs() >= 0.5) & held
    nxt = run.panel.prices.pct_change().shift(-2).reindex(p.w.index)
    reversed_ = int((big_move & (nxt.abs() >= 0.5) & (np.sign(nxt) != np.sign(p.ret))).to_numpy().sum())
    zero = ((p.ret == 0) & held)
    print("\nLegs held at exactly zero move (a frozen price): by year " + ", ".join(
        f"{y} {int(n)}" for y, n in zero.sum(axis=1).groupby(zero.index.year).sum().items())
        + "; by coin " + ", ".join(f"{k} {int(v)}" for k, v in zero.sum().sort_values(ascending=False).head(6).items())
        + f"; what they add: {float(np.nansum((p.price + p.funding).where(zero).to_numpy())):+.4f}")
    print(f"All {int(held.to_numpy().sum())} coin-days held: exactly zero move {zero_move}; market-day dollar volume under $100k "
          f"{low_vol}; moves of 50% or more {int(big_move.to_numpy().sum())}, of which followed the next day by a move of "
          f"50% or more the other way {reversed_}")
    # entries
    print("\nHow the 20 biggest were entered (leg held how long before the move; what the coin had done):")
    et = entry_table(run, p, big.head(20))
    e = et.copy()
    for c in ("coin_move_since_entry", "coin_14d_at_entry", "coin_60d_at_entry",
              "max_day_move_in_prior_7d_of_entry", "move_that_day"):
        e[c] = e[c].map(lambda x: "" if pd.isna(x) else f"{100 * x:+.0f}%")
    print(e.to_string(index=False))
    print(f"  bought after a day of +40% or more in the 7 days before the purchase: "
          f"{int((et.max_day_move_in_prior_7d_of_entry >= 0.40).sum())} of {len(et)}; "
          f"coin up +100% or more in 14 days at the purchase: {int((et.coin_14d_at_entry >= 1.0).sum())} of {len(et)}; "
          f"coin up less than +20% between the purchase and the day of the move: "
          f"{int((et.coin_move_since_entry < 0.20).sum())} of {len(et)}")

    # a second source: CoinMarketCap weekly snapshots (top 200 by capitalisation)
    print("\nSecond source, CoinMarketCap weekly top-200 snapshots (price at the snapshot date vs the Binance close of the same label; 24h volume there vs Binance perpetual):")
    for sym in ("SIREN", "RAVE"):
        cm = cmc_series(sym)
        if cm.empty:
            continue
        b = price[sym].reindex(cm.index)
        bv = (usd[sym].reindex(cm.index)) / 1e6
        cmp_ = pd.DataFrame({"cmc_rank": cm["rank"], "cmc_price": cm["price"], "binance_close": b,
                             "ratio": b / cm["price"], "cmc_vol_M": cm["volume24"] / 1e6,
                             "binance_perp_vol_M": bv}).round(4)
        print(f"  {sym}: {len(cm)} snapshots, ratio Binance/CMC median {cmp_['ratio'].median():.4f}, "
              f"min {cmp_['ratio'].min():.4f}, max {cmp_['ratio'].max():.4f}")
        print(cmp_.head(14).to_string())
    # the hourly file (only a few coins have one)
    print("\nHourly bars of the same contract (only some contracts have a file):")
    seen = False
    for r in big.itertuples():
        hr = hourly_binance(r.coin)
        if hr is None:
            continue
        day = idx[idx.get_loc(r.t) + 1]
        piece = hr.loc[day : day + pd.Timedelta(hours=23)].dropna(subset=["price"])
        if len(piece) < 24:
            continue
        seen = True
        daily_close = price[r.coin].loc[day]
        print(f"  {r.coin} market day {day.date()}: hourly bars {len(piece)}, last hourly close {piece['price'].iloc[-1]:.5f} "
              f"vs daily close {daily_close:.5f}; hourly volume sum / daily volume {piece['volume'].sum() / run.panel.volume[r.coin].loc[day]:.3f}; "
              f"highest hourly high {piece['high'].max():.5f} ({piece['high'].max() / daily_close:.2f} x the close)")
    if not seen:
        print("  none of the big coin-days has hourly bars on disk")
    return df


# --------------------------------------------------------------------------------------
# Section 5: counterfactuals


def drift_run(part: MarketPanel, weights: pd.DataFrame, costs: CostModel,
              every_bar: bool = False, stand_costs: bool = False) -> pd.Series:
    """The same book left to drift between rebalances.

    The stand holds each target weight at its target on every bar (the move of a bar is
    weight x move, whatever the weight has become): a coin that has doubled is, in effect,
    sold back to its target every day for free. Here a position is bought at a bar where the
    target weight changes (a rebalance, a listing, a delisting, a monthly switch of the chosen
    candidate), then left alone: its value grows with the coin, funding is paid on the value,
    and the next rebalance pays the cost on everything that has to be traded to get back to the
    target, drift included. `every_bar=True` puts the targets back on every bar and must
    reproduce the stand exactly when `stand_costs=True` (cost on the change of the TARGET only,
    as the stand charges; checked in `section_counterfactuals`); with `stand_costs=False` it
    also pays for trading the drift back, which the stand never does."""
    keep = part.prices.index[:-1]
    gaps = holdable_gaps(part.prices, part.funding, part.meta.get("no_funding_instruments", ()))
    wh = held_through(weights, gaps)
    pr = marked_through(part.prices, gaps)
    ret = pr.pct_change().shift(-1).loc[keep].fillna(0.0).to_numpy()
    fund = part.funding.shift(-1).loc[keep].fillna(0.0).to_numpy()
    w = wh.loc[keep].to_numpy()
    prev = wh.shift(1).fillna(0.0).loc[keep].to_numpy()
    changed = np.abs(w - prev).max(axis=1) > 1e-12
    if every_bar:
        changed[:] = True
    bps = costs.total_bps / 1e4
    pos = np.zeros(w.shape[1])
    equity = 1.0
    out = np.empty(len(keep))
    for i in range(len(keep)):
        cost = 0.0
        if changed[i]:
            target = w[i] * equity
            # the position the previous bar left (drifted), not the previous target
            traded = np.abs(w[i] - prev[i]).sum() * equity if stand_costs else np.abs(target - pos).sum()
            cost = traded * bps
            pos = target
        pnl = float((pos * ret[i]).sum()) - float((pos * fund[i]).sum()) - cost
        out[i] = pnl / equity
        equity += pnl
        pos = pos * (1.0 + ret[i])
        if equity <= 0:
            out[i + 1 :] = -1.0
            break
    return pd.Series(out, index=keep, name="net_return")


def liquidity_mask(panel: MarketPanel, line_usd: float) -> pd.DataFrame:
    from qlab.data.snapshot import _liquidity_eligible_mask

    return _liquidity_eligible_mask(panel.prices, panel.volume, pd.Timedelta(days=1), line_usd)


def delisting_mask(panel: MarketPanel, days: int) -> pd.DataFrame:
    """False from `days` days before the last print of a delisted contract."""
    mask = pd.DataFrame(True, index=panel.prices.index, columns=panel.prices.columns)
    for coin in panel.prices.columns:
        facts = contract_facts(coin)
        if facts.get("is_delisted"):
            cut = pd.Timestamp(facts["last_seen"]) - pd.Timedelta(days=days)
            mask.loc[mask.index >= cut, coin] = False
    return mask


def with_tradeable(panel: MarketPanel, allowed: pd.DataFrame, tag: str) -> MarketPanel:
    return dataclasses.replace(panel, snapshot_id=f"{panel.snapshot_id}:{tag}",
                               tradeable=panel.tradeable & allowed.reindex_like(panel.tradeable).fillna(False))


def cap_net(p: Parts, cap: float, gain_only: bool = False) -> pd.Series:
    c = p.price + p.funding
    clipped = c.clip(upper=cap) if gain_only else c.clip(lower=-cap, upper=cap)
    return clipped.sum(axis=1) - p.cost.sum(axis=1)


def after(net: pd.Series, start: str) -> pd.Series:
    return net[pd.DatetimeIndex(net.index).tz_convert("UTC") >= pd.Timestamp(start, tz="UTC")]


def cf_row(name: str, net: pd.Series, extra: str = "") -> dict:
    s = summary(net)
    row = {"case": name, "ann": s["ann"], "worst_fall": s["max_dd"], "ex_best_1pct": s["ann_ex_best_1pct"],
           "window": window_return(net)}
    for y in range(2020, 2027):
        if str(y) in s:
            row[str(y)] = s[str(y)]
    row["total"] = s["total"]
    return row


def print_cf(rows: list[dict]) -> None:
    df = pd.DataFrame(rows).set_index("case")
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{100 * x:+.0f}%" if abs(x) >= 1 else f"{100 * x:+.1f}%")
    print(out.to_string())


def section_counterfactuals(runs: dict[str, Run], parts: dict[str, Parts]) -> None:
    print("\n=== 5. COUNTERFACTUALS (rules fixed in advance, see the constants at the top) ===")
    base, pb = runs["binance"], parts["binance"]
    # the drift engine must reproduce the stand when told to put the targets back every bar
    daily = drift_run(base.part, base.part_weights, base.costs, every_bar=True, stand_costs=True)
    print(f"check: drift engine with targets put back every bar (stand's costs) differs from the stand by "
          f"{(daily - base.result.net_return).abs().max():.1e} at most")
    daily_paid = drift_run(base.part, base.part_weights, base.costs, every_bar=True)
    print(f"the stand's implicit daily re-balancing, paid for at its own cost rate, would cost "
          f"{(base.result.net_return - daily_paid).sum():.3f} of the book over the run (added shares), "
          f"{(base.result.net_return - daily_paid).mean() * 365.25:.1%} a year")
    cases: list[tuple[str, pd.Series]] = [("base (as stored)", base.result.net_return)]

    def add(name: str, run: Run) -> Run:
        cases.append((name, run.result.net_return))
        return run

    for line in VOLUME_LINES_USD + VOLUME_LINES_DOSE_USD:
        tag = f"vol{int(line / 1e6)}M"
        label = f"A. day-before volume >= ${line / 1e6:g}M" + (" (dose check)" if line in VOLUME_LINES_DOSE_USD else "")
        add(label, rerun_on(base, with_tradeable(base.panel, liquidity_mask(base.panel, line), tag), tag))
    add(f"B. nothing in the {DELIST_DAYS} days before a delisting",
        rerun_on(base, with_tradeable(base.panel, delisting_mask(base.panel, DELIST_DAYS), "delist"), "delist"))
    for cap in CAP_SHARES:
        cases.append((f"C. one coin-day at most +-{cap:.0%} of the book", cap_net(pb, cap)))
    cases.append((f"C. gains of one coin-day at most +{CAP_GAIN_ONLY:.0%} (losses left)", cap_net(pb, CAP_GAIN_ONLY, True)))
    cases.append(("E. positions left to drift between rebalances", drift_run(base.part, base.part_weights, base.costs)))
    cases.append(("F. the stand's daily re-balancing paid for (4.4 bp on the drift traded)", daily_paid))
    for k in (5, 10):
        gross = pb.price.sum(axis=1) + pb.funding.sum(axis=1)
        cases.append((f"G. costs {k} times higher ({k * base.costs.total_bps:.0f} bp per unit traded)",
                      gross - k * pb.cost.sum(axis=1)))

    print("\nWhole run 2020-01-01..2026-10-25 (ann = annual return; worst fall; years; best 1% of days taken out; the window of the live months)")
    print_cf([cf_row(n, s) for n, s in cases])

    # D: only coins Hyperliquid lists, from 2025-01-01 (before it the Hyperliquid snapshot has no data)
    print("\nD. only coins Hyperliquid lists at the time (the rule applies from 2025-01-01; before it all coins, as in the base)")
    hl = hl_listed_mask(base.panel)
    allow = hl | (base.panel.prices.index < pd.Timestamp("2025-01-01", tz="UTC"))[:, None]
    allowed = pd.DataFrame(allow, index=base.panel.prices.index, columns=base.panel.prices.columns)
    hl_run = rerun_on(base, with_tradeable(base.panel, allowed, "hlonly"), "hlonly")
    from_2025 = "2025-01-01"
    rows = [cf_row("base, 2025-01-01..", after(base.result.net_return, from_2025)),
            cf_row("D. Binance prices, Hyperliquid-listed coins only", after(hl_run.result.net_return, from_2025)),
            cf_row("D + E (and left to drift)", after(drift_run(hl_run.part, hl_run.part_weights, hl_run.costs), from_2025)),
            cf_row("the stored Hyperliquid run (HL prices, funding, fee 4.5 bp)", runs["hyperliquid"].result.net_return)]
    print_cf(rows)
    held = (hl_run.part_weights.abs() > 0).to_numpy().sum() / max(1, (base.part_weights.abs() > 0).to_numpy().sum())
    print(f"   coin-days held with the Hyperliquid rule: {held:.0%} of the base's (the rule moves the book to other coins)")

    # the same rules on the Hyperliquid run itself
    hl_base, hl_parts = runs["hyperliquid"], parts["hyperliquid"]
    print("\nThe stored Hyperliquid run (HL prices, funding and fee) under the same rules (A: volume lines; C: caps; E: drift)")
    hl_cases = [("HL base (as stored)", hl_base.result.net_return)]
    for line in VOLUME_LINES_USD:
        tag = f"hlvol{int(line / 1e6)}M"
        r = rerun_on(hl_base, with_tradeable(hl_base.panel, liquidity_mask(hl_base.panel, line), tag), tag)
        hl_cases.append((f"A. day-before volume >= ${line / 1e6:g}M", r.result.net_return))
    for cap in CAP_SHARES:
        hl_cases.append((f"C. one coin-day at most +-{cap:.0%} of the book", cap_net(hl_parts, cap)))
    hl_cases.append(("E. positions left to drift", drift_run(hl_base.part, hl_base.part_weights, hl_base.costs)))
    print_cf([cf_row(n, s) for n, s in hl_cases])

    # all together
    print("\nAll rules together (stress): volume >= $5M and no delisting month and Hyperliquid-listed (2025+) and drift")
    comb_allowed = liquidity_mask(base.panel, 5e6) & delisting_mask(base.panel, DELIST_DAYS) & allowed
    comb = rerun_on(base, with_tradeable(base.panel, comb_allowed, "all"), "all")
    rows = [cf_row("base, 2025-01-01..", after(base.result.net_return, from_2025)),
            cf_row("A5M + B + D (positions held at constant weight)", after(comb.result.net_return, from_2025)),
            cf_row("A5M + B + D + E (left to drift)", after(drift_run(comb.part, comb.part_weights, comb.costs), from_2025))]
    print_cf(rows)
    return None


# --------------------------------------------------------------------------------------
# Section 6: other Binance-history runs of the registry

#: (label, trial id): the latest judged run of each Binance-history idea that is alive or was
#: the nearest relative of this one; read from the registry (see the doc).
OTHER_RUNS = (
    ("trend-binance-cap-gated-causal (bench)", 93080),
    ("bv2-research-headroom-causal (validated)", 106794),
    ("bv2-research-partial-hedge-causal (validated)", 107196),
    ("bv2-research-short-window-causal (validated)", 107598),
    ("bv2-binance (bench)", 101165),
    ("xsmom-research-nobull-retune (retired, older snapshot)", 85838),
    ("gen-xsmom-borri-capweight (rejected)", 79982),
    ("xsmom-research-nobull-retune-hl (validated, Hyperliquid data)", 95090),
    ("xsmom-research-nobull (bench, Hyperliquid data, numbers chosen by hand)", 105386),
)


def load_trial_run(trial_id: int) -> tuple[Run, dict]:
    """A stored trial re-run on its stored snapshot, as `qlab.forward_series.replay` does."""
    from qlab.forward_series import spec_file_for
    from qlab.pipeline.evaluate import complete_book_window
    from qlab.registry.models import Trial

    with session_scope() as session:
        trial = session.get(Trial, trial_id)
        spec_path = spec_file_for(session, trial.spec_id)
        panel = load_snapshot(trial.snapshot_id, session=session)
        metrics = dict(trial.metrics or {})
    if spec_path is None:
        raise RuntimeError(f"no spec file reproduces the stored spec of trial {trial_id}")
    spec = load_spec(spec_path)
    ruleset = loader.load_latest()
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)
    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    rows = judged_rows(spec, panel, coverage, ruleset)
    return run_spec(spec.idea_id, trial_id, spec, panel, rows, ruleset), metrics


def extreme_split(p: Parts, threshold: float = 0.5) -> tuple[float, float, int]:
    held = p.w.abs() > 0
    move = p.ret.where(held).abs()
    pf = p.price + p.funding
    ext = (pf.where(move >= threshold, 0.0)).to_numpy().sum()
    rest = (pf.where(~(move >= threshold), 0.0)).to_numpy().sum()
    return float(ext), float(rest), int((move >= threshold).to_numpy().sum())


def section_others(runs: dict[str, Run], parts: dict[str, Parts]) -> None:
    print("\n=== 6. OTHER BINANCE-HISTORY RUNS: THE SAME COINS ON THEIR BEST DAYS? ===")
    main = runs["binance"]
    p_main = parts["binance"]
    pump = set()
    for y in (2025, 2026):
        for t in best_days(p_main.net, y):
            pump.add(p_main.contribution.loc[t].idxmax())
    print("coins that gave the best coin-day of the main run's ten best days of 2025 and 2026: " + ", ".join(sorted(pump)))
    results = [("MAIN xsmom-research-nobull-causal", main, p_main, None)]
    # plain xsmom books on the same snapshot (the two candidates of the chooser, each alone)
    for name, gate in (("xsmom, no gate (candidate 'always')", None),
                       ("xsmom, held only in flat/bear (candidate 'only_flat_bear')", ("flat", "bear"))):
        spec = main.spec
        base_params = dict(spec.params["base_params"])
        strategy = resolve_strategy(base_params["inner_code_ref"])
        w = strategy.target_weights(main.panel, dict(base_params["inner_params"]))
        if gate is not None:
            gated = resolve_strategy("qlab.strategies.regime_gate:RegimeGate")
            w = gated.target_weights(main.panel, {"inner_code_ref": base_params["inner_code_ref"],
                                                  "inner_params": dict(base_params["inner_params"]),
                                                  "trade_in": list(gate)})
        part_w = w.iloc[main.lo : main.hi + 1]
        res = run_backtest(main.part, part_w, main.costs, main.part.funding)
        r = dataclasses.replace(main, weights=w, part_weights=part_w, result=res)
        results.append((name, r, decompose(r.part, r.part_weights, r.costs), None))
    for label, trial_id in OTHER_RUNS:
        try:
            run, metrics = load_trial_run(trial_id)
        except Exception as exc:  # noqa: BLE001 - report and go on
            print(f"{label}: could not be re-run ({type(exc).__name__}: {exc})")
            continue
        results.append((label, run, decompose(run.part, run.part_weights, run.costs), metrics))
    print(f"\n{'run':72s} {'recorded ann':>12s} {'rerun ann':>10s} {'rerun total':>12s} {'rec. total':>11s}  "
          "extreme coin-days (|move|>=50%): n, added, the rest added | thin (<$1M) coin-days: share, added of all")
    for label, run, p, metrics in results:
        s = summary(run.result.net_return)
        rec = metrics or {}
        ext, rest, n = extreme_split(p)
        n_thin, n_all, add_thin, add_all = thin_exposure(run, p, 1e6)
        print(f"{label:72s} {pct(rec.get('ann_return_net')) if rec else '':>12s} {pct(s['ann']):>10s} "
              f"{pct(s['total'], 0):>12s} {pct(rec.get('total_return_net'), 0) if rec else '':>11s}  "
              f"{n:6d} {ext:+9.3f} {rest:+9.3f} | {n_thin / n_all:6.1%} {add_thin:+8.3f} of {add_all:+.3f}")
    print("\nBest ten days of each run and the coin behind them (top coin by contribution and its move that day):")
    for label, _run, p, _metrics in results:
        net = p.net
        days = list(net.sort_values(ascending=False).index[:10])
        cells = []
        n_pump = n_ext = 0
        for t in days:
            c = p.contribution.loc[t]
            coin = c.abs().idxmax()
            mv = p.ret.loc[t, coin]
            n_ext += int(abs(mv) >= 0.5)
            n_pump += int(coin in pump)
            cells.append(f"{t.date()} {pct(net[t], 0)} {coin}({pct(mv, 0)})")
        s = summary(net)
        print(f"- {label}: ann {pct(s['ann'])}; ten best days {pct((1 + net[days]).prod() - 1, 0)} compounded; "
              f"top coin moved >=50% on {n_ext} of 10; top coin is one of the main run's pump coins on {n_pump} of 10")
        print("    " + "; ".join(cells))



# --------------------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", default=None, help="sections to run")
    args = parser.parse_args()
    want = set(args.only) if args.only else None

    def on(name: str) -> bool:
        return want is None or name in want

    runs = {"binance": load_run(BINANCE), "hyperliquid": load_run(HYPERLIQUID)}
    if on("reproduce"):
        section_reproduce(runs)
    parts = section_decompose(runs)
    if on("events"):
        section_events(runs, parts)
    if on("spells"):
        section_spells(runs, parts)
    if on("checks"):
        section_checks(runs, parts)
    if on("counterfactuals"):
        section_counterfactuals(runs, parts)
    if on("others"):
        section_others(runs, parts)


if __name__ == "__main__":
    main()
