"""Follow-up tasks of the hedge-trigger study (owner 2026-11-05: «запускай все по порядку»).

    uv run python scripts/research/hedge_followup.py <task> --work DIR

`--work` is the folder of `hedge_trigger_study.py prepare` (its set_a.pkl / set_b.pkl); results
go to `DIR/followup-<task>.json`. Each task's protocol is committed before its run
(docs/research/hedge-trigger-<n>-*-protocol-2026-11-05.md).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hedge_trigger_study as hts  # noqa: E402

from qlab import hedge_followup as hf  # noqa: E402
from qlab import hedge_study as hs  # noqa: E402

CUT = hts.CUT
START, END = hts.START, hts.END


def load(work: Path, name: str) -> dict:
    main = sys.modules["__main__"]
    main.CoinData = hts.CoinData  # the pickles were written by the study run as __main__
    with open(work / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def _btc(work: Path) -> pd.Series:
    data = load(work, "set_a")
    return data["panel"].prices["BTC"]


def in_set_days(cd) -> set:
    at = cd.index[cd.stepped][cd.mask]
    return set(at.tz_convert("UTC").tz_localize(None).floor("D"))


# ── task 1 ──────────────────────────────────────────────────────────────


def task1(work: Path) -> dict:
    btc = _btc(work)
    regime = hf.regime_of_days(btc)
    out: dict = {"cut": str(CUT.date())}
    cut = CUT.tz_localize(None)
    for name in ("set_a", "set_b"):
        coins = load(work, name)["coins"]
        tables = {}
        for cd in coins:
            close = hf.daily_closes(cd.spot_own)
            days = in_set_days(cd)
            for h1 in hf.HORIZONS:
                for h2 in hf.HORIZONS:
                    p = hf.momentum_pairs(close, h1, h2)
                    p = p[p.index.isin(days)].dropna()
                    p["coin"] = cd.coin
                    p["regime"] = regime.reindex(p.index).to_numpy()
                    p["end"] = p.index + pd.Timedelta(days=h2)
                    tables.setdefault((h1, h2), []).append(p)
        res = {}
        for (h1, h2), parts in tables.items():
            t = pd.concat(parts)
            for part in ("first", "last"):
                tp = t[t["end"] < cut] if part == "first" else t[t.index >= cut]
                for reg in ("all", "bull", "flat", "bear"):
                    tr = tp if reg == "all" else tp[tp["regime"] == reg]
                    if len(tr) < 50:
                        continue
                    x, y = tr["past"].to_numpy(), tr["next"].to_numpy()
                    c = hf.block_bootstrap(tr.index.to_numpy(),
                                           lambda rows, x=x, y=y: hf.corr(x[rows], y[rows]),
                                           draws=400)
                    # rank correlation: not driven by a few collapses (LUNA, FTT); added after the
                    # Pearson numbers showed it (memo, «Отступление от протокола»)
                    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                    rk = hf.block_bootstrap(tr.index.to_numpy(),
                                            lambda rows, rx=rx, ry=ry: hf.corr(rx[rows], ry[rows]),
                                            draws=400)
                    res[f"{h1}->{h2} {part} {reg}"] = {
                        "corr": c, "rank": rk, "sign_excess": hf.sign_excess(x, y),
                        "n": int(len(tr))}
        # variance ratios, per coin pooled by mean, on the in-set days
        vr = {}
        for part in ("first", "last"):
            for reg in ("all", "bull", "flat", "bear"):
                for h in (3, 7, 14, 30):
                    vals = []
                    for cd in coins:
                        close = hf.daily_closes(cd.spot_own)
                        sel = close.index < cut if part == "first" else close.index >= cut
                        cl = close[sel]
                        if reg != "all":
                            cl = cl.where(regime.reindex(cl.index) == reg)
                        days = in_set_days(cd)
                        cl = cl.where(cl.index.isin(days))
                        if cl.notna().sum() > 200:
                            vals.append(hf.variance_ratio(cl, h))
                    vr[f"VR{h} {part} {reg}"] = float(np.nanmedian(vals)) if vals else None
        # the rule's own reading: next return after "hedge" vs "no hedge"
        rule = {}
        for h2 in (7, 14):
            parts = []
            for cd in coins:
                close = hf.daily_closes(cd.spot_own)
                lc = np.log(close)
                hedge = ~((lc - lc.shift(14) > 0) & (lc - lc.shift(30) > 0))
                hedge = hedge.where((lc - lc.shift(30)).notna())
                t = pd.DataFrame({"hedge": hedge, "next": lc.shift(-h2) - lc}).dropna()
                t = t[t.index.isin(in_set_days(cd))]
                t["regime"] = regime.reindex(t.index).to_numpy()
                t["end"] = t.index + pd.Timedelta(days=h2)
                parts.append(t)
            t = pd.concat(parts)
            for part in ("first", "last"):
                tp = t[t["end"] < cut] if part == "first" else t[t.index >= cut]
                for reg in ("all", "bull", "flat", "bear"):
                    tr = tp if reg == "all" else tp[tp["regime"] == reg]
                    h = tr["hedge"].to_numpy(dtype=bool)
                    nx = tr["next"].to_numpy()

                    def diff(rows, h=h, nx=nx):
                        a, b = nx[rows][h[rows]], nx[rows][~h[rows]]
                        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan

                    rule[f"next{h2} {part} {reg}"] = {
                        "diff": hf.block_bootstrap(tr.index.to_numpy(), diff, draws=400),
                        "mean_hedge": float(nx[h].mean()) if h.any() else None,
                        "mean_free": float(nx[~h].mean()) if (~h).any() else None,
                        "share_hedge": float(h.mean()), "n": int(len(tr))}
        out[name] = {"pairs": res, "vr": vr, "rule": rule}
        print(name, "done", flush=True)
    return out


# ── task 2 ──────────────────────────────────────────────────────────────

SET_CAPITAL = {"set_a": 257.0, "set_b": hts.PER_COIN_CAPITAL * hts.TOP_K}
YEARS = {"first": (CUT - pd.Timestamp(START, tz="UTC")).days / 365,
         "last": (pd.Timestamp(END, tz="UTC") - CUT).days / 365}


def production_records(data: dict, name: str, gates=None, wishes=None, extra=None):
    """{coin: BookRecord} of the production book (set a: one book; set b: each coin's own)."""
    params = {**hts._params(), **(extra or {})}
    coins = data["coins"]
    if name == "set_a":
        rec = hs.run_book(data["panel"], params, gates=gates, wishes=wishes)
        return {cd.coin: rec for cd in coins}
    p_one = {k: v for k, v in params.items() if k != "coins"}
    out = {}
    for cd in coins:
        g = {cd.coin: gates[cd.coin]} if gates and cd.coin in gates else None
        w = {cd.coin: wishes[cd.coin]} if wishes and cd.coin in wishes else None
        out[cd.coin] = hs.run_book(hts._sub_panel(data["panel"], cd.coin),
                                   {**p_one, "coins": [cd.coin],
                                    "capital_usd": hts.PER_COIN_CAPITAL}, gates=g, wishes=w)
    return out


def all_episodes(data: dict, recs: dict, regime_days: pd.Series) -> pd.DataFrame:
    rows = []
    for cd in data["coins"]:
        ep = hs.episodes(recs[cd.coin], cd.coin)
        if not len(ep):
            continue
        in_set = pd.Series(False, index=cd.index)
        in_set.iloc[cd.stepped[cd.mask]] = True
        ep = ep[in_set.reindex(ep["start"]).to_numpy(dtype=bool)]
        ep = hs.label_events(ep.assign(kind="on", liquidation=False), cd.spot_own)
        day = ep["start"].dt.tz_convert(None).dt.floor("D")
        ep["regime"] = regime_days.reindex(day).to_numpy()
        ep["coin"] = cd.coin
        rows.append(ep)
    ep = pd.concat(rows, ignore_index=True)
    ep["part"] = np.where(ep["start"] < CUT, "first", "last")
    return ep


def summarise_episodes(ep: pd.DataFrame, capital: float, years: float) -> dict:
    if not len(ep):
        return {"n": 0}
    pct = 100.0 / capital
    m = ep["money"].sort_values(ascending=False)
    total = float(m.sum())
    false = ep[ep["ret_7d"] >= 0]
    return {
        "n": int(len(ep)), "per_year": len(ep) / years,
        "hours_median": float(ep["hours"].median()), "hours_mean": float(ep["hours"].mean()),
        "share_losing": float((ep["money"] < 0).mean()),
        "money_pct_year": total * pct / years,
        "price_pct_year": float(ep["price"].sum()) * pct / years,
        "funding_pct_year": float(ep["funding"].sum()) * pct / years,
        "fees_pct_year": float(ep["fees"].sum()) * pct / years,
        "top5_pct_year": float(m.head(5).sum()) * pct / years,
        "top10_pct_year": float(m.head(10).sum()) * pct / years,
        "rest_after_top10_pct_year": float(m.iloc[10:].sum()) * pct / years,
        "worst10_pct_year": float(m.tail(10).sum()) * pct / years,
        "long_7d_pct_year": float(ep.loc[ep["hours"] > 168, "money"].sum()) * pct / years,
        "short_7d_pct_year": float(ep.loc[ep["hours"] <= 168, "money"].sum()) * pct / years,
        "share_short_7d": float((ep["hours"] <= 168).mean()),
        "book_fee_per_switch_pct": float(ep["fees"].mean()) * pct,
        "stand_fee_per_switch_pct": float((2 * hts.SWITCH_COST * ep["notional"]).mean()) * pct,
        "false_n": int(len(false)),
        "false_mean_money_pct": float(false["money"].mean()) * pct if len(false) else None,
        "false_mean_price_pct": float(false["price"].mean()) * pct if len(false) else None,
        "false_mean_fees_pct": float(false["fees"].mean()) * pct if len(false) else None,
        "true_mean_money_pct": float(ep.loc[ep["ret_7d"] < 0, "money"].mean()) * pct,
        "notional_mean_usd": float(ep["notional"].mean()),
    }


def task2(work: Path) -> dict:
    regime_days = hf.regime_of_days(_btc(work))
    out: dict = {}
    for name in ("set_a", "set_b"):
        data = load(work, name)
        recs = production_records(data, name)
        ep = all_episodes(data, recs, regime_days)
        cap = SET_CAPITAL[name]
        res = {}
        for part in ("first", "last"):
            e = ep[ep["part"] == part]
            res[part] = summarise_episodes(e, cap, YEARS[part])
            res[f"{part} by coin"] = {c: summarise_episodes(g, cap, YEARS[part])
                                      for c, g in e.groupby("coin")}
            res[f"{part} by regime"] = {r: summarise_episodes(g, cap, YEARS[part])
                                        for r, g in e.groupby("regime")}
            res[f"{part} top10"] = e.nlargest(10, "money")[
                ["coin", "start", "hours", "money", "ret_7d", "regime"]].assign(
                money=lambda x, cap=cap: x["money"] * 100 / cap).to_dict("records")
        out[name] = res
        ep.to_csv(work / f"episodes-{name}.csv", index=False)
        print(name, "done", flush=True)
    return out


# ── task 3 ──────────────────────────────────────────────────────────────

BAND_GRID = [(x, y, h, c) for x in (0.0, 0.02, 0.05, 0.10) for y in (0.0, 0.02, 0.05, 0.10)
             for h in (0, 24, 72, 168) for c in (0, 24, 72, 168)]


def _band_worker(args):
    perp, stepped = args
    out = np.zeros((len(BAND_GRID), len(stepped)), dtype=bool)
    for k, (x, y, h, c) in enumerate(BAND_GRID):
        out[k] = hf.band_wish(perp, x, y, h, c)[stepped] > 0.5
    return out


def wish_to_hedged(wish_stepped: np.ndarray, start: bool) -> np.ndarray:
    out = np.zeros(len(wish_stepped), dtype=bool)
    if len(out):
        out[0] = start
        out[1:] = wish_stepped[:-1]
    return out


def proxy_curves(coins, hedged_of, n_variants: int, n_bars: int):
    """Per variant, the set's mean over coins of (spot return + hedge money) per bar on the full
    index, the count of coins per bar, and the number of switch-ons per variant and part."""
    total = np.zeros((n_variants, n_bars), dtype=np.float64)
    count = np.zeros(n_bars, dtype=np.int32)
    ons = np.zeros((n_variants, 2))
    for k_coin, cd in enumerate(coins):
        hedged_all = hedged_of(k_coin, cd)  # (n_variants, len(stepped))
        px = cd.perp
        ret = np.zeros(len(px))
        ret[:-1] = px[1:] / px[:-1] - 1.0
        ret = np.nan_to_num(ret)
        sel = cd.mask
        pos = cd.stepped[sel]
        count[pos] += 1
        first = cd.index[cd.stepped] < CUT
        for v in range(n_variants):
            h = hedged_all[v]
            m = hs.hedge_money(px, cd.funding, h, hts.SWITCH_COST)
            total[v, pos] += (ret + m)[sel]
            flips = np.r_[False, h[1:] & ~h[:-1]] & sel
            ons[v, 0] += (flips & first).sum()
            ons[v, 1] += (flips & ~first).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return mean, count, ons


def curve_stats(mean: np.ndarray, count: np.ndarray, index: pd.DatetimeIndex, ons: np.ndarray,
                regime: np.ndarray | None = None) -> list[dict]:
    first = index < CUT
    coin_years = {"first": count[first].sum() / 8760, "last": count[~first].sum() / 8760}
    out = []
    for v in range(mean.shape[0]):
        row = {}
        for part, sel in (("first", first), ("last", ~first)):
            seg = mean[v, sel]
            row[f"money_{part}"] = 100 * seg.sum() / YEARS[part]
            row[f"dd_{part}"] = 100 * hf.max_drawdown(seg)
            k_part = 0 if part == "first" else 1
            row[f"ons_per_coin_year_{part}"] = ons[v, k_part] / coin_years[part]
            if regime is not None and part == "first":
                for reg in ("bull", "bear", "flat"):
                    rs = sel & (regime == reg)
                    yrs = rs.sum() / 8760
                    row[f"money_first_{reg}"] = 100 * mean[v, rs].sum() / yrs if yrs else np.nan
        out.append(row)
    return out


def choose_constrained(rows: list[dict], names: list[str], allowed) -> str:
    """Most money on the first 2/3 among `allowed` variants whose first-2/3 drawdown is not
    deeper than production's (names[0] is production)."""
    prod = rows[0]
    best, best_val = None, -np.inf
    for k, r in enumerate(rows):
        if not allowed(k) or r["dd_first"] < prod["dd_first"]:
            continue
        if r["money_first"] > best_val:
            best, best_val = k, r["money_first"]
    return best


def _book_params(name: str, coins_all: list[str], extra=None) -> tuple[dict, float]:
    params = {**hts._params(), **(extra or {})}
    if name == "set_a":
        return params, float(params["capital_usd"])
    capital = hts.PER_COIN_CAPITAL * hts.TOP_K
    return ({**{k: v for k, v in params.items() if k != "coins"}, "capital_usd": capital,
             "universe": {"rule": "top_liquid", "n": hts.TOP_K, "rebalance": "1d",
                          "by": "perp_dollar_volume", "window_days": hts.WINDOW_DAYS,
                          "candidates": coins_all}}, capital)


def task3(work: Path) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    out: dict = {"grid": BAND_GRID}
    for name in ("set_a", "set_b"):
        data = load(work, name)
        coins, panel = data["coins"], data["panel"]
        index = panel.prices.index
        perps = {cd.coin: panel.prices[cd.coin].to_numpy(dtype=float) for cd in coins}
        with ProcessPoolExecutor(max_workers=12) as pool:
            wishes = list(pool.map(_band_worker, [(perps[cd.coin], cd.stepped) for cd in coins]))
        print(name, "wishes", flush=True)

        def hedged_of(k_coin, cd, wishes=wishes):
            start = bool(cd.in_pos[0]) if len(cd.in_pos) else False
            return np.array([wish_to_hedged(w, start) for w in wishes[k_coin]])

        mean, count, ons = proxy_curves(coins, hedged_of, len(BAND_GRID), len(index))
        np.savez_compressed(work / f"band-{name}.npz", mean=mean.astype(np.float32), count=count,
                            ons=ons)
        rows = curve_stats(mean, count, index, ons)
        names = [f"x{x:g} y{y:g} h{h} c{c}" for x, y, h, c in BAND_GRID]
        picks = {
            "best": choose_constrained(rows, names, lambda k: True),
            "band only": choose_constrained(rows, names, lambda k: BAND_GRID[k][2:] == (0, 0)),
            "pause only": choose_constrained(rows, names, lambda k: BAND_GRID[k][:2] == (0, 0)),
        }
        print(name, {k: names[v] for k, v in picks.items()}, flush=True)
        coins_all = [cd.coin for cd in coins]
        bparams, capital = _book_params(name, coins_all)
        replay = {"production": hts._book_money(panel, bparams, capital, coins_all=coins_all)[0]}
        for label, k in picks.items():
            x, y, h, c = BAND_GRID[k]
            w = {coin: hf.band_wish(perps[coin], x, y, h, c) for coin in coins_all}
            replay[label] = {"variant": names[k],
                             **hts._book_money(panel, bparams, capital, wishes=w,
                                               coins_all=coins_all)[0]}
            print(name, label, json.dumps(replay[label]), flush=True)
        out[name] = {"rows": [{"variant": n, **r} for n, r in zip(names, rows, strict=True)],
                     "picks": {k: names[v] for k, v in picks.items()}, "replay": replay}
    return out


# ── task 4 ──────────────────────────────────────────────────────────────

GRADED = [("graded", d, None) for d in (0.02, 0.05, 0.10, 0.20, 0.30)] + \
    [("downside", q, s) for q in (0.5, 0.7) for s in (0.3, 0.5)]


def book_run(panel, bparams, capital, gates=None, wishes=None, shares=None, coins_all=None):
    """Like `hedge_trigger_study._book_money`, with shares, and the capital return series."""
    from qlab.harness.run import run_backtest

    nan = np.full(len(panel.prices.index), np.nan)
    if wishes:
        wishes = {c: wishes.get(c, nan) for c in coins_all}
    if shares:
        shares = {c: shares.get(c, nan) for c in coins_all}
    rec = hs.run_book(panel, bparams, gates=gates or None, wishes=wishes or None,
                      shares=shares or None)
    base = (rec.run.book_equity / capital).rename("capital_base")
    res = run_backtest(panel, rec.run.weights, hts.COSTS, panel.funding, capital_base=base,
                       capital_reinvested=False)
    r = res.capital_return
    out = {}
    for part, sel in (("first", r.index < CUT), ("last", r.index >= CUT)):
        out[f"cap_ann_{part}"] = float(r[sel].sum() * 8760 / int(sel.sum()) * 100)
        out[f"cap_max_dd_{part}"] = 100 * hf.max_drawdown(r[sel].to_numpy())
        out[f"fees_year_{part}"] = float((res.cost * base.reindex(res.cost.index).ffill())[sel]
                                         .sum() * 100 / YEARS[part])
        ons = 0
        for c in rec.stepped.columns:
            ev = hs.events(rec, c)
            if len(ev):
                d = ev["decision"]
                ons += int(((ev["kind"] == "on") & ((d < CUT) if part == "first"
                                                     else (d >= CUT))).sum())
        out[f"ons_{part}"] = ons
    return out, r


def downside_pct(perp: pd.Series) -> pd.Series:
    r = np.log(perp.ffill()).diff()
    dvol = np.sqrt((r.clip(upper=0.0) ** 2).rolling(720, min_periods=360).mean())
    return hs.past_percentile(dvol)


def task4(work: Path) -> dict:
    out: dict = {"grid": GRADED}
    for name in ("set_a", "set_b"):
        data = load(work, name)
        coins, panel = data["coins"], data["panel"]
        coins_all = [cd.coin for cd in coins]
        returns = {}
        rows = {}
        bparams, capital = _book_params(name, coins_all)
        rows["production"], returns["production"] = book_run(panel, bparams, capital,
                                                             coins_all=coins_all)
        pct = {c: downside_pct(panel.prices[c]).to_numpy() for c in coins_all}
        for kind, a, b in GRADED:
            label = f"{kind} {a:g}" + (f" share {b:g}" if b is not None else "")
            if kind == "graded":
                bp, _ = _book_params(name, coins_all, {"graded_hedge_depth": a})
                rows[label], returns[label] = book_run(panel, bp, capital, coins_all=coins_all)
            else:
                bp, _ = _book_params(name, coins_all, {"partial_hedge_share": 0.5})
                shares = {c: np.where(np.isnan(pct[c]), 1.0, np.where(pct[c] < a, b, 1.0))
                          for c in coins_all}
                rows[label], returns[label] = book_run(panel, bp, capital, shares=shares,
                                                       coins_all=coins_all)
            print(name, label, json.dumps(rows[label]), flush=True)
        prod = rows["production"]
        allowed = [k for k in rows if k != "production"
                   and rows[k]["cap_max_dd_first"] >= prod["cap_max_dd_first"]]
        pick = max(allowed, key=lambda k: rows[k]["cap_ann_first"]) if allowed else "production"
        pd.DataFrame(returns).to_parquet(work / f"graded-returns-{name}.parquet")
        out[name] = {"rows": rows, "pick": pick}
        print(name, "pick", pick, flush=True)
    return out


# ── task 5 ──────────────────────────────────────────────────────────────

PURGE = pd.Timedelta(days=15)  # the regime label reads 15 days ahead


def _regime_sums(per_bar: np.ndarray, weight: np.ndarray, index, regime) -> dict:
    """Money per year (per unit of `weight`-year: coin-years or calendar years) on the first 2/3
    (purged) by regime, on the first 2/3 whole, and on the last third."""
    first = index < CUT - PURGE
    out = {}
    for key, sel in (("bull", first & (regime == "bull")), ("bear", first & (regime == "bear")),
                     ("first", index < CUT), ("last", index >= CUT)):
        w = weight[sel].sum() / 8760
        out[key] = float(100 * per_bar[sel].sum() / w) if w else np.nan
    return out


def _scores(table: dict[str, dict]) -> pd.DataFrame:
    t = pd.DataFrame(table).T
    p = t.loc["production"]
    d = t - p
    return pd.DataFrame({"old": d["first"], "robust": np.minimum(d["bull"], d["bear"]),
                         "last": d["last"], "bull": d["bull"], "bear": d["bear"]})


def task5(work: Path) -> dict:
    btc = _btc(work)
    out: dict = {}
    trials = pd.read_csv(hts.TRIALS)
    for name in ("set_a", "set_b"):
        data = load(work, name)
        coins, panel = data["coins"], data["panel"]
        index = panel.prices.index
        regime = hf.regime_of_hours(btc, index)
        coins_all = [cd.coin for cd in coins]
        res = {}
        # first study: hedge money per unit of position, per coin-year
        variants = {v.name: v for v in hs.declared_variants()}
        table = {}
        for vname, v in [("production", None), *variants.items()]:
            total = np.zeros(len(index))
            weight = np.zeros(len(index))
            for cd in coins:
                h = hts.variant_hedged(v, cd)
                m = hs.hedge_money(cd.perp, cd.funding, h, hts.SWITCH_COST)
                pos = cd.stepped[cd.mask]
                total[pos] += m[cd.mask]
                weight[pos] += 1
            table[vname] = _regime_sums(total, weight, index, regime)
        s1 = _scores(table).drop(index="production")
        eligible = trials[trials["set"] == name].set_index("variant")["eligible"]
        s1["eligible"] = eligible.reindex(s1.index).fillna(False).astype(bool)
        s1["kind"] = ["trigger" if variants[k].kind == "trigger" else "filter" for k in s1.index]
        res["study1"] = {"spearman_old": hf.spearman(s1["old"], s1["last"]),
                         "spearman_robust": hf.spearman(s1["robust"], s1["last"])}
        picks = {}
        for kind in ("filter", "trigger"):
            sub = s1[(s1["kind"] == kind) & s1["eligible"]]
            picks[kind] = sub["robust"].idxmax()
        bparams, capital = _book_params(name, coins_all)
        prod_book, _ = book_run(panel, bparams, capital, coins_all=coins_all)
        res["production_book"] = prod_book
        res["study1"]["picks"] = {}
        for kind, vname in picks.items():
            g, w = hts._full_gates(variants[vname], coins)
            book, _ = book_run(panel, bparams, capital, gates=g, wishes=w, coins_all=coins_all)
            res["study1"]["picks"][kind] = {"variant": vname, **s1.loc[vname].to_dict(),
                                            "book": book}
            print(name, "study1", kind, vname, json.dumps(book), flush=True)
        s1.to_csv(work / f"robust-study1-{name}.csv")
        # task 3: the simplified book per unit of position
        z = np.load(work / f"band-{name}.npz")
        mean, count = z["mean"].astype(np.float64), z["count"]
        names = [f"x{x:g} y{y:g} h{h} c{c}" for x, y, h, c in BAND_GRID]
        weight = (count > 0).astype(float)
        table = {("production" if k == 0 else names[k]): _regime_sums(mean[k], weight, index,
                                                                      regime)
                 for k in range(len(names))}
        s3 = _scores(table).drop(index="production")
        res["task3"] = {"spearman_old": hf.spearman(s3["old"], s3["last"]),
                        "spearman_robust": hf.spearman(s3["robust"], s3["last"])}
        k3 = s3["robust"].idxmax()
        x, y, h, c = BAND_GRID[names.index(k3)]
        w = {coin: hf.band_wish(panel.prices[coin].to_numpy(dtype=float), x, y, h, c)
             for coin in coins_all}
        book, _ = book_run(panel, bparams, capital, wishes=w, coins_all=coins_all)
        res["task3"]["pick"] = {"variant": k3, **s3.loc[k3].to_dict(), "book": book}
        print(name, "task3", k3, json.dumps(book), flush=True)
        # task 4: the real book on the strategy's money
        r4 = pd.read_parquet(work / f"graded-returns-{name}.parquet")
        table = {}
        for col in r4.columns:
            r = r4[col].reindex(index).fillna(0.0).to_numpy()
            table[col] = _regime_sums(r, np.ones(len(index)), index, regime)
        s4 = _scores(table).drop(index="production")
        res["task4"] = {"spearman_old": hf.spearman(s4["old"], s4["last"]),
                        "spearman_robust": hf.spearman(s4["robust"], s4["last"]),
                        "pick": {"variant": s4["robust"].idxmax(),
                                 **s4.loc[s4["robust"].idxmax()].to_dict()},
                        "table": s4.to_dict("index")}
        out[name] = res
    return out


# ── task 6 ──────────────────────────────────────────────────────────────

SPOT_COST = (10.0 + 0.9) / 1e4


def simple_book_money(cd, perp_full: pd.Series, book: str) -> np.ndarray:
    """Per stepped bar, money per unit of position of one simple book (protocol, task 6)."""
    px = cd.perp
    ret = np.zeros(len(px))
    ret[:-1] = px[1:] / px[:-1] - 1.0
    ret = np.nan_to_num(ret)
    fund = np.zeros(len(px))
    fund[:-1] = np.nan_to_num(cd.funding[1:])
    at = cd.index[cd.stepped]
    new_day = np.r_[True, at[1:].floor("D") != at[:-1].floor("D")]
    if book == "bv2":
        return ret + hs.hedge_money(px, cd.funding, cd.in_pos, hts.SWITCH_COST)
    if book.startswith("hedge"):
        share = float(book.split()[1])
        # a constant share kept by a daily trade of about share * |day's move|
        day_move = pd.Series(np.abs(ret)).groupby(np.cumsum(new_day)).transform("sum").to_numpy()
        cost = np.where(new_day, share * day_move * hts.SWITCH_COST, 0.0)
        return ret + share * (-ret + fund) - cost
    if book == "spot":
        return ret
    k = float(book.split()[1])
    expo = hf.downside_vol_exposure(perp_full, scale=k)[cd.stepped]
    change = np.abs(np.diff(expo, prepend=expo[0]))
    return expo * ret - change * SPOT_COST


def task6(work: Path) -> dict:
    books = ["spot", "hedge 0.3", "hedge 0.5", "hedge 1.0", "downside 0.75", "downside 1.0",
             "downside 1.5", "bv2"]
    out: dict = {}
    for name in ("set_a", "set_b"):
        data = load(work, name)
        coins, panel = data["coins"], data["panel"]
        index = panel.prices.index
        first = index < CUT
        res = {}
        for book in books:
            total = np.zeros(len(index))
            count = np.zeros(len(index))
            for cd in coins:
                m = simple_book_money(cd, panel.prices[cd.coin], book)
                pos = cd.stepped[cd.mask]
                total[pos] += m[cd.mask]
                count[pos] += 1
            mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            row = {}
            for part, sel in (("first", first), ("last", ~first)):
                row[f"money_{part}"] = float(100 * mean[sel].sum() / YEARS[part])
                row[f"dd_{part}"] = float(100 * hf.max_drawdown(mean[sel]))
            res[book] = row
        for row in res.values():
            for part in ("first", "last"):
                gain = row[f"dd_{part}"] - res["spot"][f"dd_{part}"]
                give = res["spot"][f"money_{part}"] - row[f"money_{part}"]
                row[f"price_{part}"] = float(give / gain) if gain > 0.5 else None
        out[name] = res
        print(name, json.dumps(res), flush=True)
    return out


TASKS = {"1": task1, "2": task2, "3": task3, "4": task4, "5": task5, "6": task6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = TASKS[a.task](a.work)
    out["seconds"] = round(time.time() - t0)
    (a.work / f"followup-{a.task}.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", a.work / f"followup-{a.task}.json", out["seconds"], "s")


if __name__ == "__main__":
    main()
