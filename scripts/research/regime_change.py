"""Regime and regime change instead of the next price (docs/research/regime-change-protocol-2026-11-07.md).

    uv run python scripts/research/regime_change.py data --work DIR
    uv run python scripts/research/regime_change.py oracle --work DIR      # ORACLE: reads the future
    uv run python scripts/research/regime_change.py detect --work DIR
    uv run python scripts/research/regime_change.py planted --work DIR
    uv run python scripts/research/regime_change.py book --work DIR        # only if a detector passed

Writes DIR/frames.pkl, DIR/oracle.json, DIR/detect.json, DIR/planted.json, DIR/book.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import regime_change as rc

RAW = Path("/Users/d/prj/q-lab/data/raw")
MINUTES = RAW / "binance" / "1m" / "futures"
H1 = RAW / "binance" / "1h"
STEM = "__2019-09-08__2026-11-06.parquet"
START, END = "2019-09-08", "2026-11-05 23:00"
FIRST0 = pd.Timestamp("2021-01-01", tz="UTC")
CUT = pd.Timestamp("2024-06-12", tz="UTC")
COINS = ("BTC", "ETH", "SOL", "AVAX")
CAPITALS = (1000.0, 64.25)
PROCS = int(os.environ.get("RC_PROCS", "8"))
RETRAIN_DAYS = 91
GRIDS = {"1d": 24, "4h": 4}  # grid name -> hours per step
X_MAIN = 0.25
X_ALL = (0.15, 0.25, 0.35)


def _index() -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC"), freq="1h")


def grid_rows(index: pd.DatetimeIndex, step: int) -> np.ndarray:
    """Positions of the hourly rows whose bar closes on the grid (row T closes at T + 1 h)."""
    return np.flatnonzero((index.hour + 1) % step == 0) if step < 24 else np.flatnonzero(
        index.hour == 23)


# ── data ─────────────────────────────────────────────────────────────────


def build_frames(work: Path) -> None:
    t0 = time.time()
    index = _index()
    frames = {}
    for coin in COINS:
        perp = pd.read_parquet(H1 / f"{coin}{STEM}")
        spot = pd.read_parquet(H1 / f"{coin}-SPOT{STEM}")
        perp.index = perp.index.as_unit("ns")
        spot.index = spot.index.as_unit("ns")
        px = perp["price"].reindex(index)
        first = px.first_valid_index()
        px = px.ffill(limit=48)
        sp = spot["price"].reindex(index).ffill(limit=48)
        fr = pd.DataFrame(index=index)
        fr["logc"] = np.log(px)
        fr["basis"] = np.log(px) - np.log(sp)
        fr["funding"] = perp["funding"].reindex(index)
        fr["qv"] = (perp["volume"] * perp["price"]).reindex(index)
        mins = []
        for f in sorted((MINUTES / coin).glob("*.parquet")):
            m = pd.read_parquet(f, columns=["open_time", "close", "quote_volume", "taker_buy_quote"])
            mins.append(m)
        m = pd.concat(mins).drop_duplicates("open_time").set_index("open_time").sort_index()
        m.index = m.index.as_unit("ns")
        lr = np.log(m["close"]).diff()
        hour = m.index.floor("h")
        agg = pd.DataFrame({"rv": (lr ** 2).groupby(hour).sum(),
                            "tq": m["taker_buy_quote"].groupby(hour).sum(),
                            "mq": m["quote_volume"].groupby(hour).sum(),
                            "nmin": lr.groupby(hour).count()})
        agg = agg.reindex(index)
        hourly_sq = fr["logc"].diff() ** 2
        fr["rv"] = np.where(agg["nmin"].fillna(0) >= 50, agg["rv"], hourly_sq)
        fr["tq"] = agg["tq"]
        fr["mq"] = agg["mq"]
        fr.loc[fr.index < first, :] = np.nan
        frames[coin] = fr
        print(coin, "frame", first, round(time.time() - t0), "s", flush=True)
    with open(work / "frames.pkl", "wb") as fh:
        pickle.dump(frames, fh)


def load_frames(work: Path) -> dict[str, pd.DataFrame]:
    with open(work / "frames.pkl", "rb") as fh:
        return pickle.load(fh)


def permuted_frames(frames: dict, seed: int) -> dict:
    """The day-permuted history (the random-walk control), each coin on its own."""
    rng = np.random.default_rng(seed)
    out = {}
    for coin, fr in frames.items():
        valid = fr["logc"].notna().to_numpy()
        first = int(np.flatnonzero(valid)[0])
        first = first + (-first) % 24  # the first full UTC day (index starts at 00:00)
        part = fr.iloc[first:].copy()
        part["logc"] = part["logc"].ffill()
        perm = rc.permute_days(part, rng)
        new = fr.copy()
        new.iloc[:first] = np.nan
        new.iloc[first:] = perm.to_numpy()
        out[coin] = new
    return out


# ── features ─────────────────────────────────────────────────────────────


def _past_pct(x: pd.Series, min_days: int = 90) -> pd.Series:
    from qlab import hedge_study as hs

    return hs.past_percentile(x, min_days)


def hourly_features(frames: dict) -> dict[str, pd.DataFrame]:
    """The declared features at every hour, from hours up to and including it (protocol, 3)."""
    out = {}
    logc_all = pd.DataFrame({c: f["logc"] for c, f in frames.items()})
    below = pd.DataFrame({c: (logc_all[c] < np.log(np.exp(logc_all[c]).rolling(720, min_periods=600)
                                                   .mean())).where(logc_all[c].notna())
                          for c in frames})
    share_below = below.astype(float).mean(axis=1)
    btc = frames["BTC"]["logc"]
    btc_r30 = btc - btc.shift(720)
    btc_dd90 = btc - btc.rolling(2160, min_periods=1800).max()

    def mp_(w):
        return max(int(w * 0.8), 1)

    for coin, f in frames.items():
        lc = f["logc"]
        p = np.exp(lc)
        F = {}
        for h in (1, 4, 24, 72, 168, 336, 720, 2160, 4320):
            F[f"r_{h}h"] = lc - lc.shift(h)
        rv = f["rv"]
        for w in (4, 24, 168, 720):
            F[f"lrv_{w}h"] = 0.5 * np.log(rv.rolling(w, min_periods=mp_(w)).sum() * 8760 / w + 1e-12)
        F["vol_ratio"] = F["lrv_168h"] - F["lrv_720h"]
        F["vov"] = F["lrv_24h"].rolling(720, min_periods=600).std()
        for w in (720, 2160, 8760):
            F[f"dd_{w}h"] = lc - lc.rolling(w, min_periods=mp_(w)).max()
        for w in (720, 2160):
            F[f"rise_{w}h"] = lc - lc.rolling(w, min_periods=mp_(w)).min()
        fu = f["funding"].fillna(0.0).where(lc.notna())
        F["fund_1d"] = fu.rolling(24, min_periods=24).sum() * 365
        F["fund_7d"] = fu.rolling(168, min_periods=168).sum() * 365 / 7
        fund_30 = fu.rolling(720, min_periods=720).sum() * 365 / 30
        F["fund_trend"] = F["fund_7d"] - fund_30
        F["fund_pct"] = _past_pct(F["fund_7d"])
        b = f["basis"]
        F["basis_1d"] = b.rolling(24, min_periods=12).mean()
        F["basis_7d"] = b.rolling(168, min_periods=84).mean()
        F["basis_pct"] = _past_pct(F["basis_7d"])
        tq, mq = f["tq"], f["mq"]
        for w, nm in ((24, "1d"), (168, "7d")):
            F[f"taker_{nm}"] = 2 * tq.rolling(w, min_periods=mp_(w)).sum() / mq.rolling(
                w, min_periods=mp_(w)).sum() - 1
        qv = f["qv"]
        F["volume_trend"] = np.log(qv.rolling(168, min_periods=120).sum()
                                   / (qv.rolling(2160, min_periods=1800).sum() * 168 / 2160))
        ma200 = p.rolling(4800, min_periods=4000).mean()
        ma50 = p.rolling(1200, min_periods=1000).mean()
        F["p_ma200"] = lc - np.log(ma200)
        F["ma50_ma200"] = np.log(ma50) - np.log(ma200)
        d = lc.diff()
        g = d.clip(lower=0).rolling(336, min_periods=300).mean()
        lo = (-d).clip(lower=0).rolling(336, min_periods=300).mean()
        F["rsi_14d"] = g / (g + lo)
        e12 = p.ewm(span=12 * 24, min_periods=12 * 24).mean()
        e26 = p.ewm(span=26 * 24, min_periods=26 * 24).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9 * 24, min_periods=9 * 24).mean()
        F["macd_hist"] = (macd - sig) / p
        ma20 = p.rolling(480, min_periods=400).mean()
        sd20 = p.rolling(480, min_periods=400).std()
        F["bb_pos"] = (p - ma20) / (2 * sd20)
        F["btc_r30"] = btc_r30
        F["btc_dd90"] = btc_dd90
        F["share_below30"] = share_below
        df = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
        df[lc.isna()] = np.nan
        out[coin] = df
    return out


UNSUP_INPUT = ("r_24h", "r_168h", "r_720h", "lrv_168h", "lrv_720h")


# ── labels on a grid ─────────────────────────────────────────────────────


def grid_series(frames: dict, step: int) -> dict:
    """Per coin: grid rows (hourly positions) from the coin's first valid grid point, and the
    grid closes (ffilled)."""
    index = next(iter(frames.values())).index
    rows = grid_rows(index, step)
    out = {}
    for coin, f in frames.items():
        lc = f["logc"].to_numpy()
        r = rows[~np.isnan(lc[rows])]
        r = r[r >= r[0]]
        v = pd.Series(lc[rows]).ffill().to_numpy()
        keep = rows >= r[0]
        out[coin] = {"rows": rows[keep], "logc": v[keep], "time": index[rows[keep]]}
    return out


def labels_for(g: dict, x: float) -> dict:
    v = np.exp(g["logc"])
    state, _ = rc.r1_state(v, x)
    return {"state": state, "legs": rc.r1_legs(v, x)}


# ── detectors ────────────────────────────────────────────────────────────


def retrain_times(end: pd.Timestamp) -> list[pd.Timestamp]:
    out, t = [], FIRST0
    while t <= end:
        out.append(t)
        t = t + pd.Timedelta(days=RETRAIN_DAYS)
    return out


def rules_on(g: dict, step: int) -> dict[str, np.ndarray]:
    spd = 24 // step
    lc = g["logc"]
    out = {"rule bv2 14/30": rc.rule_bv2(lc, spd), "rule ma50<ma200": rc.rule_ma_cross(lc, spd)}
    for x in (0.10, 0.15, 0.25):
        out[f"rule drawdown {int(x * 100)}%"] = rc.rule_drawdown(np.exp(lc), x)
    return out


def unsupervised_on(feats: pd.DataFrame, g: dict, seed: int) -> dict[str, np.ndarray]:
    """HMM (2, 3 states) and jump models (lambda grid) on the daily grid, refit every 91 days on
    the coin's whole past (>= 365 days), filtered / online states only."""
    X_all = feats.iloc[g["rows"]][list(UNSUP_INPUT)].to_numpy()
    times = g["time"]
    valid = ~np.isnan(X_all).any(1)
    n = len(times)
    names = ["hmm 2", "hmm 3"] + [f"jump lambda {lam:g}" for lam in (10, 30, 100, 300)]
    out = {nm: np.full(n, np.nan) for nm in names}
    rng = np.random.default_rng(seed)
    prev = {}
    first_valid = int(np.flatnonzero(valid)[0])
    T = retrain_times(times[-1] + pd.Timedelta(days=1))
    close_t = times + pd.Timedelta(hours=1)
    for j, tj in enumerate(T):
        nxt = T[j + 1] if j + 1 < len(T) else times[-1] + pd.Timedelta(days=2)
        train = valid & (close_t <= tj)
        if train.sum() < 365:
            continue
        mu, sd = X_all[train].mean(0), X_all[train].std(0) + 1e-9
        Z = (X_all - mu) / sd
        seg_from = first_valid
        upto = int(np.searchsorted(close_t, nxt, side="left"))  # rows with close < next retrain
        pred = (close_t > tj) & (close_t <= nxt) if j > 0 else (close_t <= nxt)
        # rows before the first fit stay unknown; from the first fit, states run from its time
        pred &= close_t > tj
        Zs = Z[seg_from:upto]
        okz = ~np.isnan(Zs).any(1)
        Zs = np.where(okz[:, None], Zs, 0.0)
        Xtr = Z[train]
        i30 = UNSUP_INPUT.index("r_720h")
        for K in (2, 3):
            m = rc.hmm_fit(Xtr, K, rng, starts=5 if f"hmm {K}" not in prev else 2,
                           init=prev.get(f"hmm {K}"))
            prev[f"hmm {K}"] = m
            post = rc.hmm_filtered(m, Zs)
            down = int(np.argmin(m.mu[:, i30]))
            st = (post[:, down] > 0.5).astype(float)
            full = np.full(n, np.nan)
            full[seg_from:upto] = st
            out[f"hmm {K}"][pred] = full[pred]
        for lam in (10, 30, 100, 300):
            C = rc.jump_fit(Xtr, 2, lam, rng, starts=10)
            s, _ = rc.jump_online(Zs, C, lam)
            down = int(np.argmin(C[:, i30]))
            full = np.full(n, np.nan)
            full[seg_from:upto] = (s == down).astype(float)
            out[f"jump lambda {lam:g}"][pred] = full[pred]
    return out


SUP_TARGETS = ["T0"] + [f"T{k}" for k in (3, 7, 14)] + [f"T{k}-peak" for k in (3, 7, 14)]


def target_known(v: np.ndarray, target: str, spd: int) -> np.ndarray:
    if target == "T0":
        return rc.r1_known(v, X_MAIN)
    k = int(target[1:].split("-")[0])
    kinds = ("peak",) if target.endswith("peak") else ("peak", "trough")
    return rc.switch_known(v, X_MAIN, k * spd, kinds)


def target_gap_steps(target: str, spd: int) -> int:
    return 14 * spd if target == "T0" else int(target[1:].split("-")[0]) * spd


def _fit_predict(model: str, Xf, yf, Xv, yv, Xall_tr, yall_tr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    if model == "logistic":
        med = np.nanmedian(Xall_tr, 0)
        med = np.where(np.isnan(med), 0.0, med)

        def prep(A, mu=None, sd=None):
            A = np.where(np.isnan(A), med, A)
            return A if mu is None else (A - mu) / sd

        Af = prep(Xf)
        mu, sd = Af.mean(0), Af.std(0) + 1e-9
        best_c, best_l = None, np.inf
        for C in (0.01, 0.1, 1.0):
            m = LogisticRegression(C=C, max_iter=2000).fit((Af - mu) / sd, yf)
            loss = log_loss(yv, m.predict_proba(prep(Xv, mu, sd))[:, 1], labels=[0, 1])
            if loss < best_l:
                best_c, best_l = C, loss
        Aall = prep(Xall_tr)
        mu, sd = Aall.mean(0), Aall.std(0) + 1e-9
        m = LogisticRegression(C=best_c, max_iter=2000).fit((Aall - mu) / sd, yall_tr)
        p_tr = m.predict_proba((Aall - mu) / sd)[:, 1]
        return m.predict_proba(prep(Xte, mu, sd))[:, 1], p_tr, yall_tr, {"C": best_c}
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=25,
                                       early_stopping=False, warm_start=True, random_state=0)
    best_it, best_l, bad = 0, np.inf, 0
    while m.max_iter <= 500:
        m.fit(Xf, yf)
        loss = log_loss(yv, m.predict_proba(Xv)[:, 1], labels=[0, 1])
        if loss < best_l - 1e-6:
            best_l, best_it, bad = loss, m.max_iter, 0
        else:
            bad += 1
            if bad >= 3:
                break
        m.max_iter += 25
    best_it = max(best_it, 25)

    def at(A):
        for k, pr in enumerate(m.staged_predict_proba(A), start=1):
            if k == best_it:
                return pr[:, 1]
        return m.predict_proba(A)[:, 1]

    return at(Xte), at(Xf), yf, {"trees": best_it}


def supervised_walk(feats: dict, gs: dict, step: int, target: str, model: str,
                    planted: dict | None = None) -> dict:
    """The pooled walk-forward of one target and model on one grid: out-of-sample probabilities
    per coin (NaN before 2021), plus the in-sample/out-of-sample record of each retrain."""
    spd = 24 // step
    coins = list(gs)
    cols = [c for c in next(iter(feats.values())).columns]
    blocks = {}
    for i, c in enumerate(coins):
        g = gs[c]
        X = feats[c].iloc[g["rows"]][cols].to_numpy()
        oh = np.zeros((len(X), len(coins)))
        oh[:, i] = 1.0
        if planted is not None:
            X = np.column_stack([X, planted[c]])
        blocks[c] = {"X": np.column_stack([X, oh]), "close": g["time"] + pd.Timedelta(hours=1),
                     "v": np.exp(g["logc"])}
    out = {c: np.full(len(blocks[c]["X"]), np.nan) for c in coins}
    T = retrain_times(max(b["close"][-1] for b in blocks.values()))
    gap = target_gap_steps(target, spd)
    record = []
    for j, tj in enumerate(T):
        nxt = T[j + 1] if j + 1 < len(T) else tj + pd.Timedelta(days=10_000)
        Xs, ys, ts = [], [], []
        for c in coins:
            b = blocks[c]
            n_known = int(np.searchsorted(b["close"], tj, side="right"))
            if n_known < 30 * spd:
                continue
            y = target_known(b["v"][:n_known], target, spd)
            keep = ~np.isnan(y)
            Xs.append(b["X"][:n_known][keep])
            ys.append(y[keep])
            ts.append(b["close"][:n_known][keep].asi8)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        t = np.concatenate(ts)
        cut_v = np.quantile(t, 0.8)
        gap_ns = gap * step * 3600 * 10**9
        fit_m = t < cut_v - gap_ns
        val_m = t >= cut_v
        if len(np.unique(y[fit_m])) < 2 or len(np.unique(y[val_m])) < 2:
            continue
        te_parts = {}
        for c in coins:
            b = blocks[c]
            sel = (b["close"] > tj) & (b["close"] <= nxt)
            if sel.any():
                te_parts[c] = sel
        if not te_parts:
            continue
        Xte = np.vstack([blocks[c]["X"][s] for c, s in te_parts.items()])
        p_te, p_tr, y_tr, info = _fit_predict(model, X[fit_m], y[fit_m], X[val_m], y[val_m],
                                              X, y, Xte)
        k = 0
        for c, s in te_parts.items():
            m_ = int(s.sum())
            out[c][s] = p_te[k:k + m_]
            k += m_
        record.append({"t": str(tj.date()), "n_train": int(len(y)), "pos_share": float(y.mean()),
                       "auc_in": rc.auc(p_tr, y_tr), **info})
    return {"pred": out, "record": record}


def geometry_baseline(feats: dict, gs: dict, step: int, target: str) -> dict:
    """The shape-only reference for Tk: logistic on drawdown from the 90-day high and rise from
    the 90-day low."""
    small = {c: f[["dd_2160h", "rise_2160h"]] for c, f in feats.items()}
    return supervised_walk(small, gs, step, target, "logistic")


# ── the yardstick over coins and periods ─────────────────────────────────


def periods_of(time: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    close = time + pd.Timedelta(hours=1)
    return {"first": (close > FIRST0) & (close < CUT), "last": close >= CUT}


def measure_states(gs: dict, labs: dict, dets: dict[str, dict], step: int, common: dict
                   ) -> dict:
    """{detector: {period: pooled yardstick}} on the common mask."""
    res = {}
    for name, per_coin in dets.items():
        res[name] = {}
        for per in ("first", "last"):
            parts = []
            for c, g in gs.items():
                mask = common[c] & periods_of(g["time"])[per]
                parts.append(rc.yardstick(g["logc"], labs[c]["state"], per_coin[c],
                                          labs[c]["legs"], mask, step))
            res[name][per] = rc.pool(parts)
    return res


def common_mask(gs: dict, labs: dict, dets: dict[str, dict]) -> dict:
    out = {}
    for c, g in gs.items():
        m = ~np.isnan(labs[c]["state"]) & ((g["time"] + pd.Timedelta(hours=1)) > FIRST0)
        for per_coin in dets.values():
            m &= ~np.isnan(per_coin[c])
        out[c] = m
    return out


def past_rank(p: np.ndarray, min_obs: int) -> np.ndarray:
    s = pd.Series(p)
    r = s.expanding(min_periods=min_obs).rank(pct=True).to_numpy()
    r[np.isnan(p)] = np.nan
    return r


def measure_scores(gs: dict, preds: dict, targets_full: dict, step: int, mask: dict,
                   null_draws: int = 200, seed: int = 0) -> dict:
    """AUC, block-permutation null and precision at the top 10% / 20% (past-only ranks)."""
    spd = 24 // step
    out = {}
    for per in ("first", "last"):
        S, Y, R = [], [], []
        for c, g in gs.items():
            m = mask[c] & periods_of(g["time"])[per] & ~np.isnan(preds[c]) & ~np.isnan(
                targets_full[c])
            S.append(preds[c][m])
            Y.append(targets_full[c][m])
            R.append(past_rank(preds[c], 90 * spd)[m])
        s, y, r = np.concatenate(S), np.concatenate(Y), np.concatenate(R)
        null = rc.block_null_auc(S, Y, 30 * spd, null_draws, seed)
        row = {"auc": rc.auc(s, y), "null_p95": float(np.quantile(null, 0.95)),
               "null_mean": float(null.mean()), "base_rate": float(y.mean()), "n": int(len(y))}
        for top in (0.10, 0.20):
            sel = r >= 1 - top
            row[f"precision_top{int(top * 100)}"] = float(y[sel].mean()) if sel.any() else np.nan
        out[per] = row
    return out


# ── phases ───────────────────────────────────────────────────────────────

_G: dict = {}


def _detect_one(job):
    """One full run of the detection machine on the real (seed None) or a permuted history."""
    kind, seed, grid = job
    frames = _G["frames"] if seed is None else permuted_frames(_G["frames"], 10_000 + seed)
    feats = hourly_features(frames)
    step = GRIDS[grid]
    gs = grid_series(frames, step)
    labs = {c: labels_for(g, X_MAIN) for c, g in gs.items()}
    t0 = time.time()
    out = {"seed": seed, "grid": grid, "kind": kind}
    if kind == "states":
        dets = {}
        rules = {c: rules_on(gs[c], step) for c in gs}
        for nm in rules[COINS[0]]:
            dets[nm] = {c: rules[c][nm] for c in gs}
        if grid == "1d":
            un = {c: unsupervised_on(feats[c], gs[c], 7 + (seed or 0)) for c in gs}
            for nm in un[COINS[0]]:
                dets[nm] = {c: un[c][nm] for c in gs}
        for model in ("logistic", "gbm"):
            w = supervised_walk(feats, gs, step, "T0", model)
            dets[f"{model} T0"] = {c: (w["pred"][c] > 0.5).astype(float)
                                   * np.where(np.isnan(w["pred"][c]), np.nan, 1.0) for c in gs}
            out[f"{model} T0 record"] = w["record"]
            out[f"{model} T0 auc"] = measure_scores(
                gs, w["pred"], {c: labs[c]["state"] for c in gs}, step,
                common_mask(gs, labs, {}), null_draws=50 if seed is not None else 200)
        common = common_mask(gs, labs, dets)
        out["states"] = measure_states(gs, labs, dets, step, common)
        if seed is None:
            out["common_from"] = {c: str(gs[c]["time"][np.flatnonzero(common[c])[0]].date())
                                  for c in gs}
            out["state_series"] = {nm: {c: d[c].tolist() for c in gs} for nm, d in dets.items()}
            out["grid_time"] = {c: [str(t) for t in gs[c]["time"]] for c in gs}
    else:  # switch targets
        target, model = kind.split("|")
        full = {}
        for c, g in gs.items():
            full[c] = target_known(np.exp(g["logc"]), target, 24 // step)
        mask = {c: (gs[c]["time"] + pd.Timedelta(hours=1)) > FIRST0 for c in gs}
        if model == "geometry":
            w = geometry_baseline(feats, gs, step, target)
        else:
            w = supervised_walk(feats, gs, step, target, model)
        out["scores"] = measure_scores(gs, w["pred"], full, step, mask,
                                       null_draws=50 if seed is not None else 200)
        out["record"] = w["record"]
    out["seconds"] = round(time.time() - t0)
    print(kind, grid, seed, out["seconds"], "s", flush=True)
    return out


def run_detect(work: Path, rw_states: int, rw_sup: int) -> None:
    t0 = time.time()
    _G["frames"] = load_frames(work)
    jobs = []
    for grid in GRIDS:
        jobs.append(("states", None, grid))
        for tgt in SUP_TARGETS[1:]:
            for model in ("logistic", "gbm", "geometry"):
                jobs.append((f"{tgt}|{model}", None, grid))
    for grid in GRIDS:
        for s in range(rw_states if grid == "1d" else rw_sup):
            jobs.append(("states", s, grid))
        for s in range(rw_sup):
            for tgt in SUP_TARGETS[1:]:
                for model in ("logistic", "gbm", "geometry"):
                    jobs.append((f"{tgt}|{model}", s, grid))
    # long jobs first
    jobs.sort(key=lambda j: (j[0] != "states", j[2] != "4h", "gbm" not in j[0]))
    ctx = mp.get_context("fork")
    with ctx.Pool(PROCS) as pool:
        got = pool.map(_detect_one, jobs, chunksize=1)
    out = {"jobs": [list(map(str, j)) for j in jobs], "results": got,
           "seconds": round(time.time() - t0)}
    with open(work / "detect.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print("detect written", out["seconds"], "s", flush=True)


# ── oracle ───────────────────────────────────────────────────────────────


def money(rec, panel, capital: float) -> tuple[dict, pd.Series]:
    """The stand's money of a recorded book (forecast_in_book.money of the earlier branch, the
    first period here starting 2021-01-01)."""
    from qlab import earning_test as et
    from qlab import hedge_study as hs
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest

    base = (rec.run.book_equity / capital).rename("capital_base")
    res = run_backtest(panel, rec.run.weights, CostModel(taker_fee_bps=7.5, slippage_bps=0.9),
                       panel.funding, capital_base=base, capital_reinvested=False)
    r = res.capital_return
    out: dict = {}
    for part, sel in (("first", (r.index >= FIRST0) & (r.index < CUT)), ("last", r.index >= CUT)):
        years = int(sel.sum()) / hs.HOURS_PER_YEAR
        out[f"cap_ann_{part}"] = float(r[sel].sum() / years * 100)
        curve = r[sel].cumsum()
        out[f"cap_max_dd_{part}"] = float((curve - curve.cummax().clip(lower=0)).min() * 100)
        out[f"p_{part}"] = et.metrics_of(r[sel], additive=True).get(et.P_KEY)
        ons = 0
        lo, hi = (FIRST0, CUT) if part == "first" else (CUT, r.index[-1] + pd.Timedelta(hours=1))
        for c in rec.stepped.columns:
            ev = hs.events(rec, c)
            if len(ev):
                d = ev["decision"]
                ons += int(((ev["kind"] == "on") & (d >= lo) & (d < hi)).sum())
        out[f"ons_per_year_{part}"] = ons / years / len(rec.stepped.columns)
        hedged = rec.in_pos.iloc[:, 0][sel]
        out[f"hedged_share_{part}"] = float(hedged.mean())
    out["liquidations"] = int(rec.run.liquidations)
    return out, r


def to_hourly(n_hours: int, rows: np.ndarray, label: np.ndarray) -> np.ndarray:
    """A grid label applied to the hourly rows from its grid row on (the hedge over the hours
    after each grid close)."""
    out = np.full(n_hours, np.nan)
    pos = np.searchsorted(rows, np.arange(n_hours), side="right") - 1
    ok = pos >= 0
    out[ok] = label[pos[ok]]
    return out


def oracle_wishes(frames: dict) -> dict:
    """{coin: {variant: hourly wish}} of the exact oracle variants O1..O9."""
    index = next(iter(frames.values())).index
    n = len(index)
    out = {}
    for coin, fr in frames.items():
        W = {}
        lc_h = fr["logc"].ffill().to_numpy()
        for grid, step in GRIDS.items():
            g = grid_series({coin: fr}, step)[coin]
            v = np.exp(g["logc"])
            for x in X_ALL:
                st, _ = rc.r1_state(v, x, fill_tail=True)
                W[f"R1 {int(x * 100)}% {grid}"] = to_hourly(n, g["rows"], st)
            if grid == "4h":
                r2 = rc.r2_class(lc_h, g["rows"])
                r3 = rc.r3_class(lc_h, index, g["rows"])
                W["R2 trend down"] = to_hourly(n, g["rows"], np.where(np.isnan(r2), np.nan,
                                                                     (r2 == -2).astype(float)))
                W["R2 not trend up"] = to_hourly(n, g["rows"], np.where(np.isnan(r2), np.nan,
                                                                       (r2 != 2).astype(float)))
                r1 = W["R1 25% 4h"]
                r3h = to_hourly(n, g["rows"], r3)
                W["R1 down and R3 not calm"] = np.where(np.isnan(r1) | np.isnan(r3h), np.nan,
                                                        ((r1 == 1) & (r3h >= 1)).astype(float))
                W["R1 down or R3 turbulent"] = np.where(np.isnan(r1) | np.isnan(r3h), np.nan,
                                                        ((r1 == 1) | (r3h == 2)).astype(float))
                W["R3 turbulent"] = np.where(np.isnan(r3h), np.nan, (r3h == 2).astype(float))
        out[coin] = W
    return out


EXACT = ["R1 25% 4h", "R1 15% 4h", "R1 35% 4h", "R1 25% 1d", "R2 trend down", "R2 not trend up",
         "R1 down and R3 not calm", "R1 down or R3 turbulent", "R3 turbulent"]
DELAYS = (0, 1, 3, 7, 14)
FALSE_RATES = (0, 2, 4, 8)
SEEDS = 5


def _book_job(job):
    from qlab import hedge_study as hs

    book, wish_key = job
    panel, params, capital = _G["books"][book]
    coin = params["coins"][0]
    if wish_key is None:
        rec = hs.run_book(panel, params)
    else:
        rec = hs.run_book(panel, params, wishes={coin: _G["wishes"][wish_key]})
    m, r = money(rec, panel, capital)
    hedged = rec.in_pos[coin].to_numpy().astype(float)
    return m, (r.to_numpy() if wish_key is None else None), (hedged if wish_key is None else None)


def run_oracle(work: Path) -> None:
    import yaml

    from qlab import hedge_study as hs

    t0 = time.time()
    frames = load_frames(work)
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    W = oracle_wishes(frames)
    books, wishes = {}, {}
    for coin in COINS:
        sp = hs.load_panel([coin], START, END, RAW)
        assert sp.panel.prices.index.equals(frames[coin].index), coin
        for cap in CAPITALS:
            books[f"{coin}@{cap:g}"] = (sp.panel, {**params, "coins": [coin], "capital_usd": cap},
                                        cap)
        for k, w in W[coin].items():
            wishes[f"{coin}|{k}"] = w
        base = W[coin]["R1 25% 4h"]
        for d in DELAYS:
            for f in FALSE_RATES:
                for s in range(SEEDS if f > 0 else 1):
                    rng = np.random.default_rng(100_000 + 1000 * COINS.index(coin) + 100 * d
                                                + 10 * f + s)
                    wishes[f"{coin}|delay {d} false {f} seed {s}"] = rc.flip_noise(
                        rc.delay_wish(base, 24 * d), f, rng)
    _G.update(books=books, wishes=wishes)
    print("wishes ready", len(wishes), round(time.time() - t0), "s", flush=True)
    jobs, names = [], []
    for book in books:
        coin = book.split("@")[0]
        jobs.append((book, None))
        names.append(f"{book} production")
        for k in wishes:
            if k.startswith(f"{coin}|"):
                jobs.append((book, k))
                names.append(f"{book} {k.split('|', 1)[1]}")
    ctx = mp.get_context("fork")
    with ctx.Pool(PROCS) as pool:
        got = pool.map(_book_job, jobs, chunksize=2)
    res = {n: g[0] for n, g in zip(names, got, strict=True)}
    prod_hedged = {n.split()[0]: g[2].tolist() for n, g in zip(names, got, strict=True)
                   if g[2] is not None}
    out = {"label": "ORACLE DIAGNOSTIC: reads the future by design; never a strategy",
           "results": res, "seconds": round(time.time() - t0), "runs": len(jobs)}
    (work / "oracle.json").write_text(json.dumps(out, indent=1, default=str))
    with open(work / "prod_hedged.pkl", "wb") as fh:
        pickle.dump(prod_hedged, fh)
    print("oracle written", len(jobs), out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["data", "oracle", "detect"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rw-states", type=int, default=20)
    ap.add_argument("--rw-sup", type=int, default=10)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.phase == "data":
        build_frames(a.work)
    elif a.phase == "oracle":
        run_oracle(a.work)
    else:
        run_detect(a.work, a.rw_states, a.rw_sup)


if __name__ == "__main__":
    main()
