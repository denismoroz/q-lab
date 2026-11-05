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


TASKS = {"1": task1, "2": task2, "3": task3}


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
