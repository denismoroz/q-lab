"""Waves on minutes, trend on the higher timeframe (docs/research/waves-0-protocol-2026-11-05.md).

    uv run python scripts/research/waves.py --work DIR [--minutes DIR_OF_1M]

Reads the minute archive (read only) and DIR/set_a.pkl (the hourly panel of the hedge studies).
Writes DIR/waves.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hedge_followup as hfu  # noqa: E402
import hedge_trigger_study as hts  # noqa: E402

from qlab import bv3_parts as bp  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402
from qlab import waves as wv  # noqa: E402

CUT = hts.CUT
COINS = ("BTC", "ETH", "SOL", "AVAX")
DEFS = [(x, n) for x in (0.15, 0.25, 0.35) for n in (7, 14)]
MAIN = (0.25, 7)
WARM = pd.Timedelta(days=30)
GBM_START = pd.Timestamp("2022-01-01", tz="UTC")

VARIANTS = (["dc k1", "dc k2", "dc k4", "dc majority"] + ["structure k2", "structure k4",
            "structure k8"] + ["atrous j7", "atrous j8", "atrous j9"]
            + ["jump 2", "jump 5", "jump 10", "jump 20"] + ["gbm 0.3", "gbm 0.4", "gbm 0.5"])


def hourly_from_minutes(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """The value at the last minute of each hour (bar labelled by its open), carried forward."""
    return series.reindex(index + pd.Timedelta(minutes=59), method="ffill").to_numpy()


def coin_features(coin: str, minutes_root: Path, index: pd.DatetimeIndex,
                  btc_logc: np.ndarray | None) -> dict:
    m = wv.load_minutes(minutes_root, coin)
    close = m["close"]
    sig = wv.daily_sigma(close).to_numpy()
    px = close.to_numpy()
    out: dict = {"start": close.index[0]}
    dcs = {}
    for k in (1, 2, 4, 8):
        dc = wv.directional_change(px, k * sig)
        dcs[k] = dc
        st = pd.Series(dc["state"], index=close.index)
        out[f"dc{k}"] = hourly_from_minutes(st, index)
        # granularity: state changes when read every 15 minutes and every hour
        q15 = st.resample("15min").last()
        h1 = st.resample("1h").last()
        years = (close.index[-1] - close.index[0]).days / 365
        out[f"flips15_k{k}"] = float((q15.diff().abs() > 0).sum() / years)
        out[f"flips60_k{k}"] = float((h1.diff().abs() > 0).sum() / years)
        out[f"overshoot_k{k}"] = (float(np.median(dc["overshoot"])) if len(dc["overshoot"])
                                  else None)
        out[f"waves_per_year_k{k}"] = float(len(dc["events"]) / years)
        if k in (2, 4, 8):
            s = wv.structure_state(px, dc, len(px))
            out[f"structure{k}"] = hourly_from_minutes(pd.Series(s, index=close.index), index)
        if k in (1, 2, 4):
            ev = dc["events"]
            pos = np.searchsorted(ev, np.arange(len(px)), side="right") - 1
            since = np.where(pos >= 0, np.arange(len(px)) - ev[np.maximum(pos, 0)], np.nan)
            out[f"since{k}"] = hourly_from_minutes(pd.Series(np.log1p(since / 60.0),
                                                             index=close.index), index)
            if k == 2:
                p_e = np.where(pos >= 0, px[ev[np.maximum(pos, 0)]], np.nan)
                th_e = np.where(pos >= 0, (k * sig)[ev[np.maximum(pos, 0)]], np.nan)
                d_e = np.where(pos >= 0, dc["dirs"][np.maximum(pos, 0)], 0)
                ov = d_e * np.log(px / p_e) / th_e
                out["overshoot_now"] = hourly_from_minutes(pd.Series(ov, index=close.index),
                                                           index)
    # hourly aggregates from minutes
    lr = np.log(close).diff()
    hour = close.index.floor("h")
    agg = pd.DataFrame({"tb": m["taker_buy_base"].to_numpy(), "v": m["volume"].to_numpy(),
                        "rv": (lr ** 2).to_numpy()}, index=hour).groupby(level=0).sum()
    agg = agg.reindex(index)
    for w in (24, 168):
        tb = agg["tb"].rolling(w, min_periods=w // 2).sum()
        v = agg["v"].rolling(w, min_periods=w // 2).sum()
        out[f"imb{w}"] = (tb / v - 0.5).to_numpy()
        out[f"rv{w}"] = np.sqrt(agg["rv"].rolling(w, min_periods=w // 2).sum()).to_numpy()
    logc = np.log(hourly_from_minutes(close, index))
    logc[index < close.index[0]] = np.nan
    out["logc"] = logc
    layers = wv.causal_atrous(logc, 9)
    for j in range(5, 10):
        c = layers[j - 1]
        out[f"slope{j}"] = c - np.r_[np.full(24, np.nan), c[:-24]]
    for d in (1, 7, 30):
        out[f"ret{d}"] = logc - np.r_[np.full(24 * d, np.nan), logc[:-24 * d]]
    sig_h = hourly_from_minutes(pd.Series(sig, index=close.index), index)
    out["sig"] = sig_h
    if btc_logc is not None:
        out["btc_ret7"] = btc_logc - np.r_[np.full(168, np.nan), btc_logc[:-168]]
    else:
        out["btc_ret7"] = out["ret7"]
    return out


FEATURES = ["dc1", "dc2", "dc4", "since1", "since2", "since4", "overshoot_now", "structure4",
            "imb24", "imb168", "rv24", "rv168", "slope5", "slope6", "slope7", "slope8", "slope9",
            "ret1", "ret7", "ret30", "btc_ret7"]


def gbm_probabilities(feats: dict, closes: dict, index: pd.DatetimeIndex) -> tuple[dict, dict]:
    """Walk-forward monthly: train on hours whose big-fall label is already known on the training
    day, predict the next month. Returns per-coin probabilities and sample-size notes."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = {c: np.column_stack([feats[c][f] for f in FEATURES]) for c in feats}
    prob = {c: np.full(len(index), np.nan) for c in feats}
    starts = pd.date_range(GBM_START, index[-1], freq="MS")
    notes = {"retrains": 0, "falls_first": None, "falls_last": None, "rows_last": None}
    day = index.tz_convert("UTC").tz_localize(None).floor("D")
    for t0 in starts:
        rows_x, rows_y, n_falls = [], [], 0
        for c in feats:
            falls, pivot = wv.confirmed_falls(closes[c], *MAIN, t0.tz_localize(None)
                                              - pd.Timedelta(days=1))
            if pivot is None:
                continue
            n_falls += len(falls)
            keep = (day <= pivot) & (index >= feats[c]["start"] + WARM)
            y = np.zeros(len(index), dtype=bool)
            for p, tr in falls:
                y |= (day >= p) & (day <= tr)
            rows_x.append(X[c][keep])
            rows_y.append(y[keep])
        if not rows_x:
            continue
        xt, yt = np.vstack(rows_x), np.concatenate(rows_y)
        if yt.sum() < 100:
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=500, random_state=7)
        model.fit(xt, yt)
        t1 = t0 + pd.offsets.MonthBegin(1)
        sel = (index >= t0) & (index < t1)
        for c in feats:
            prob[c][sel] = model.predict_proba(X[c][sel])[:, 1]
        notes["retrains"] += 1
        if notes["falls_first"] is None:
            notes["falls_first"] = n_falls
        notes["falls_last"], notes["rows_last"] = n_falls, int(len(yt))
    return prob, notes


def down_state(v: str, f: dict, prob: np.ndarray | None) -> np.ndarray:
    kind = v.split()[0]
    with np.errstate(invalid="ignore"):
        if v == "dc majority":
            return (f["dc1"] + f["dc2"] + f["dc4"]) < 0
        if kind == "dc":
            return f[f"dc{v.split()[1][1:]}"] < 0
        if kind == "structure":
            return f[f"structure{v.split()[1][1:]}"] < 0
        if kind == "atrous":
            j = int(v.split()[1][1:])
            return np.nan_to_num(f[f"slope{j}"], nan=0.0) < 0
        if kind == "jump":
            sig = f["sig"]
            feats = np.column_stack([
                np.where(f["dc2"] == 0, np.nan, f["dc2"]).astype(float),
                np.clip(f["ret1"] / sig, -3, 3),
                np.clip(f["ret7"] / (sig * np.sqrt(7)), -3, 3)])
            return wv.online_jump(feats, float(v.split()[1])) < 0
        if kind == "gbm":
            return np.nan_to_num(prob, nan=0.0) > float(v.split()[1])
    raise ValueError(v)


def yardstick(down: dict, px: dict, falls: dict, index, evaluable: dict, regime=None) -> dict:
    """The higher-timeframe metrics over the coins (protocol), per part of the window."""
    out = {}
    for part in ("first", "last"):
        tp = fh = fp = oh = 0.0
        losses, false, hours = [], 0, 0.0
        cont = {"down7": [0, 0], "down14": [0, 0], "up7": [0, 0], "up14": [0, 0]}
        per_coin = {}
        for c in down:
            sel = evaluable[c] & ((index < CUT) if part == "first" else (index >= CUT))
            d = down[c].astype(bool)
            fpos = falls[c]
            fm = bp.in_falls(len(index), fpos)
            a, b, e, g = bp.phase_metric(d, fm, sel)
            tp, fh, fp, oh = tp + a, fh + b, fp + e, oh + g
            ls, fs = bp.entry_metric(d, px[c], fpos, sel)
            losses += ls
            false += fs
            hours += sel.sum()
            pc = {}
            for h in (7, 14):
                fwd = np.r_[px[c][24 * h:] / px[c][:-24 * h] - 1.0, np.full(24 * h, np.nan)]
                ok = sel & ~np.isnan(fwd)
                dn, up = ok & d, ok & ~d
                cont[f"down{h}"][0] += int((fwd[dn] < 0).sum())
                cont[f"down{h}"][1] += int(dn.sum())
                cont[f"up{h}"][0] += int((fwd[up] > 0).sum())
                cont[f"up{h}"][1] += int(up.sum())
                pc[f"down{h}"] = float((fwd[dn] < 0).mean()) if dn.any() else None
            per_coin[c] = pc
        cy = hours / 8760
        out[part] = {"tpr": tp / fh if fh else None, "fpr": fp / oh if oh else None,
                     "J": (tp / fh - fp / oh) if fh and oh else None,
                     "delay": float(np.mean(losses)) if losses else None, "falls": len(losses),
                     "false_per_coin_year": false / cy if cy else None,
                     **{f"cont_{k}": v[0] / v[1] if v[1] else None for k, v in cont.items()},
                     "per_coin": per_coin}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    a = ap.parse_args()
    t0 = time.time()
    data = hfu.load(a.work, "set_a")
    panel = data["panel"]
    index = panel.prices.index
    feats = {}
    btc_logc = None
    for c in COINS:
        feats[c] = coin_features(c, a.minutes, index, btc_logc)
        if c == "BTC":
            btc_logc = feats[c]["logc"]
        print(c, "features", round(time.time() - t0), "s", flush=True)
    px = {c: panel.prices[c].ffill().to_numpy(dtype=float) for c in COINS}
    closes = {c: hf.daily_closes(panel.prices[c]).dropna() for c in COINS}
    evaluable = {c: (index >= feats[c]["start"] + WARM) & ~np.isnan(px[c]) for c in COINS}
    falls_all = {d: {c: bp.fall_positions(bp.zigzag_falls(closes[c], *d), index) for c in COINS}
                 for d in DEFS}
    prob, notes = gbm_probabilities(feats, closes, index)
    print("gbm", notes, round(time.time() - t0), "s", flush=True)
    downs = {}
    prod = {}
    for c in COINS:
        priced = ~np.isnan(panel.prices[c].to_numpy(dtype=float))
        p = panel.prices[c].to_numpy(dtype=float)
        s14, s30 = hf.lagged(p, priced, 336), hf.lagged(p, priced, 720)
        m = np.fmin(np.where(np.isnan(s14), -np.inf, s14), s30)
        prod[c] = ~np.isnan(s30) & (m <= 0)
    downs["production"] = prod
    for v in VARIANTS:
        downs[v] = {c: down_state(v, feats[c], prob[c]) for c in COINS}
    gbm_eval = {c: evaluable[c] & ~np.isnan(prob[c]) for c in COINS}
    res = {"notes": notes, "rows": {}, "granularity": {}, "overshoot": {}}
    for v, d in downs.items():
        ev = gbm_eval if v.startswith("gbm") else evaluable
        res["rows"][v] = {"main": yardstick(d, px, falls_all[MAIN], index, ev),
                          "sens": {f"X{x} N{n}": {p: yardstick(d, px, falls_all[(x, n)], index,
                                                                ev)[p]["J"]
                                                  for p in ("first", "last")}
                                   for x, n in DEFS if (x, n) != MAIN}}
        if v.startswith("gbm"):
            res["rows"][v]["production_same_hours"] = yardstick(prod, px, falls_all[MAIN],
                                                                index, ev)
    for c in COINS:
        res["granularity"][c] = {k: feats[c][k] for k in feats[c] if k.startswith(
            ("flips", "waves_per_year"))}
        res["overshoot"][c] = {k: feats[c][k] for k in feats[c] if k.startswith("overshoot_k")}
    fam = {}
    for v in VARIANTS:
        fam.setdefault(v.split()[0], []).append(v)
    res["family_pick"] = {f: max(vs, key=lambda v: res["rows"][v]["main"]["first"]["J"] or -9)
                          for f, vs in fam.items()}
    best = max(res["family_pick"].values(), key=lambda v: res["rows"][v]["main"]["first"]["J"])
    res["best"] = best
    print("picks", res["family_pick"], "best", best, flush=True)
    # into the book
    params = hts._params()
    bparams, capital = hfu._book_params("set_a", list(COINS))
    out_book = {}
    base, r0 = hfu.book_run(panel, bparams, capital, coins_all=list(COINS))
    base["p_last"] = earning_test.metrics_of(r0[r0.index >= CUT], additive=True).get(
        earning_test.P_KEY)
    out_book["production"] = base
    for label, parts in (("as is", bp.Parts()), ("with F caps", bp.Parts(min_hold=168,
                                                                         cooldown=24))):
        wishes = {}
        for c in COINS:
            phase = np.where(evaluable[c] if not best.startswith("gbm") else gbm_eval[c],
                             downs[best][c], prod[c])
            wishes[c] = bp.machine(panel.prices[c].to_numpy(dtype=float), phase, None, None,
                                   parts)
        res_b, r = hfu.book_run(panel, bparams, capital, wishes=wishes, coins_all=list(COINS))
        res_b["p_last"] = earning_test.metrics_of(r[r.index >= CUT], additive=True).get(
            earning_test.P_KEY)
        out_book[label] = res_b
        print("book", label, json.dumps(res_b), flush=True)
    res["book"] = out_book
    del params
    res["seconds"] = round(time.time() - t0)
    (a.work / "waves.json").write_text(json.dumps(res, indent=1, default=str))
    print("written", res["seconds"], "s")


if __name__ == "__main__":
    main()
