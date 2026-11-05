"""Forecast pilot: feature selection on the first period only
(docs/research/forecast-pilot-selection-protocol-2026-11-05.md).

    uv run python scripts/research/forecast_select.py --work DIR [--minutes DIR_OF_1M]

Writes DIR/forecast-select.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import forecast_pilot as fp  # noqa: E402

from qlab import forecast as fc  # noqa: E402

FOLDS = pd.date_range("2022-01-01", "2024-01-01", freq="6MS", tz="UTC")
SFI_TARGETS = [("reg", 24), ("reg", 168), ("cls", (24, "fall", 1.0)), ("cls", (24, "rise", 1.0))]


def _gbm(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
    return cls(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=1000,
               random_state=0)


def _score(kind, model, x, y):
    if kind == "reg":
        return fc.pearson(model.predict(x), y)
    return fc.auc(model.predict_proba(x)[:, 1], y)


def _day_permute(rows_tau, rng):
    """A permutation of row positions that moves whole days together."""
    day = rows_tau // fp.DAY
    uniq = np.unique(day)
    order = rng.permutation(len(uniq))
    blocks = [np.flatnonzero(day == u) for u in uniq]
    perm = np.concatenate([blocks[k] for k in order])
    return perm[: len(rows_tau)] if len(perm) >= len(rows_tau) else np.r_[
        perm, np.arange(len(perm), len(rows_tau))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    a = ap.parse_args()
    t0 = time.time()
    D = fp.prepare_data(a.work, a.minutes, "std")
    tau, X, names, d = D["tau"], D["X"], D["names"], D["d"]
    rng = np.random.default_rng(13)
    out: dict = {"names": names}
    # 1. clusters on the first period
    first = (tau < fp.CUT) & (tau >= d["start"])
    sub = X[first][::6]
    rho = spearmanr(sub, nan_policy="omit").statistic
    rho = np.nan_to_num(np.asarray(rho), nan=0.0)
    dist = 1 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(z, t=0.3, criterion="distance")
    clusters = {int(c): [names[i] for i in np.flatnonzero(lab == c)] for c in np.unique(lab)}
    cols_of = {c: np.flatnonzero(lab == c) for c in np.unique(lab)}
    out["clusters"] = clusters
    print("clusters", len(clusters), round(time.time() - t0), "s", flush=True)

    targets = [("reg", h) for h in fp.H] + [("cls", (h, s, k)) for h in fp.H
                                            for s in ("fall", "rise") for k in fp.KS]

    def y_of(t):
        return D["targets"][t[1]] if t[0] == "reg" else D["labels"][t[1]]

    def horizon(t):
        return t[1] if t[0] == "reg" else t[1][0]

    def tname(t):
        return f"z {t[1]}h" if t[0] == "reg" else f"{t[1][0]}h {t[1][1]} k={t[1][2]}"

    mda: dict = {}
    stab: dict = {}
    for t in targets:
        y, h = y_of(t), horizon(t)
        per_fold = {c: [] for c in cols_of}
        for f0 in FOLDS:
            f1 = min(f0 + pd.DateOffset(months=6), pd.Timestamp(fp.CUT, tz="UTC"))
            tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= f0.value) & ~np.isnan(y) & (
                tau >= d["start"])
            va = (tau >= f0.value) & (tau + h * fp.HOUR <= f1.value) & ~np.isnan(y)
            if t[0] == "cls" and np.nansum(y[tr]) < 50:
                continue
            m = _gbm(t[0]).fit(X[tr], y[tr])
            xv, yv = X[va], y[va]
            base = _score(t[0], m, xv, yv)
            perm = _day_permute(tau[va], rng)
            for c, cols in cols_of.items():
                xp = xv.copy()
                xp[:, cols] = xv[perm][:, cols]
                per_fold[c].append(base - _score(t[0], m, xp, yv))
        mda[tname(t)] = {int(c): float(np.mean(v)) if v else None for c, v in per_fold.items()}
        stab[tname(t)] = {int(c): [bool(x > 0) for x in v] for c, v in per_fold.items()}
        print("mda", tname(t), round(time.time() - t0), "s", flush=True)
    out["mda"], out["stability"] = mda, stab

    # 2b. shadow test at the last first-period retrain (2024-01)
    f0 = FOLDS[-1]
    shadow = {}
    for t in targets:
        y, h = y_of(t), horizon(t)
        tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= f0.value) & ~np.isnan(y) & (
            tau >= d["start"])
        va = (tau >= f0.value) & (tau + h * fp.HOUR <= fp.CUT) & ~np.isnan(y)
        if t[0] == "cls" and np.nansum(y[tr]) < 50:
            continue
        wins = {c: 0 for c in cols_of}
        for _run in range(5):
            perm_tr = _day_permute(tau[tr], rng)
            perm_va = _day_permute(tau[va], rng)
            xs_tr = np.column_stack([X[tr], X[tr][perm_tr]])
            xs_va = np.column_stack([X[va], X[va][perm_va]])
            m = _gbm(t[0]).fit(xs_tr, y[tr])
            base = _score(t[0], m, xs_va, y[va])
            perm = _day_permute(tau[va], rng)
            for c, cols in cols_of.items():
                xp = xs_va.copy()
                xp[:, cols] = xs_va[perm][:, cols]
                real = base - _score(t[0], m, xp, y[va])
                xq = xs_va.copy()
                sc = cols + X.shape[1]
                xq[:, sc] = xs_va[perm][:, sc]
                fake = base - _score(t[0], m, xq, y[va])
                wins[c] += int(real > fake)
        shadow[tname(t)] = {int(c): w for c, w in wins.items()}
        print("shadow", tname(t), round(time.time() - t0), "s", flush=True)
    out["shadow_wins_of_5"] = shadow
    selected = {}
    for tn in shadow:
        selected[tn] = [int(c) for c in cols_of if shadow[tn][int(c)] >= 4
                        and (mda[tn][int(c)] or 0) > 0]
    out["selected"] = selected

    # 2c. single-cluster importance and per-feature MDA for four targets
    sfi, feat_mda = {}, {}
    for t in [(k, v) for k, v in SFI_TARGETS]:
        y, h = y_of(t), horizon(t)
        scores = {c: [] for c in cols_of}
        fm = np.zeros(X.shape[1])
        nf = 0
        for f0 in FOLDS:
            f1 = min(f0 + pd.DateOffset(months=6), pd.Timestamp(fp.CUT, tz="UTC"))
            tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= f0.value) & ~np.isnan(y) & (
                tau >= d["start"])
            va = (tau >= f0.value) & (tau + h * fp.HOUR <= f1.value) & ~np.isnan(y)
            for c, cols in cols_of.items():
                m = _gbm(t[0]).fit(X[tr][:, cols], y[tr])
                scores[c].append(_score(t[0], m, X[va][:, cols], y[va]))
            m = _gbm(t[0]).fit(X[tr], y[tr])
            base = _score(t[0], m, X[va], y[va])
            perm = _day_permute(tau[va], rng)
            for j in range(X.shape[1]):
                xp = X[va].copy()
                xp[:, j] = X[va][perm, j]
                fm[j] += base - _score(t[0], m, xp, y[va])
            nf += 1
        sfi[tname(t)] = {int(c): float(np.mean(v)) for c, v in scores.items()}
        top = np.argsort(-fm)[:10]
        feat_mda[tname(t)] = [(names[j], float(fm[j] / nf)) for j in top]
        print("sfi", tname(t), round(time.time() - t0), "s", flush=True)
    out["sfi"], out["feature_mda_top10"] = sfi, feat_mda

    # 3. retrain on the selected clusters: gbm (leaf 1000) walk-forward, both periods
    retrains = pd.date_range(fp.START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
    comp = {}
    for t in targets:
        tn = tname(t)
        if tn not in selected:
            continue
        cols = np.concatenate([cols_of[c] for c in selected[tn]]) if selected[tn] else \
            np.array([], dtype=int)
        y, h = y_of(t), horizon(t)
        pred_all = np.full(len(tau), np.nan)
        pred_sel = np.full(len(tau), np.nan)
        for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
            tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= q0.value) & ~np.isnan(y) & (
                tau >= d["start"])
            sel = (tau >= q0.value) & (tau < q1.value)
            m = _gbm(t[0]).fit(X[tr], y[tr])
            pred_all[sel] = m.predict(X[sel]) if t[0] == "reg" else m.predict_proba(X[sel])[:, 1]
            if len(cols):
                m2 = _gbm(t[0]).fit(X[tr][:, cols], y[tr])
                pred_sel[sel] = (m2.predict(X[sel][:, cols]) if t[0] == "reg"
                                 else m2.predict_proba(X[sel][:, cols])[:, 1])
        row = {}
        for part, mask in (("first", (tau >= fp.START.value) & (tau < fp.CUT)),
                           ("last", tau >= fp.CUT + h * fp.HOUR + fp.DAY)):
            for lbl, p in (("all", pred_all), ("selected", pred_sel)):
                ok = mask & ~np.isnan(p) & ~np.isnan(y)
                pp, yy, dd = p[ok], y[ok], D["dates"][ok]
                if not ok.any():
                    row[f"{lbl} {part}"] = None
                    continue
                stat = ((lambda r, pp=pp, yy=yy: fc.pearson(pp[r], yy[r])) if t[0] == "reg"
                        else (lambda r, pp=pp, yy=yy: fc.auc(pp[r], yy[r])))
                row[f"{lbl} {part}"] = fp._boot(dd, stat)
        row["n_selected_clusters"] = len(selected[tn])
        comp[tn] = row
        print("retrain", tn, round(time.time() - t0), "s", flush=True)
    out["selected_vs_all"] = comp
    out["seconds"] = round(time.time() - t0)
    (a.work / "forecast-select.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s")


if __name__ == "__main__":
    main()
