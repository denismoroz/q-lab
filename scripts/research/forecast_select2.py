"""Forecast pilot 2: descriptive feature selection on BTC
(docs/research/forecast-pilot-2-addendum-2026-11-05.md, section 2).

    uv run python scripts/research/forecast_select2.py --work DIR [--variant X]

The variant defaults to the first-period choice in DIR/forecast2.json. Writes
DIR/forecast2-select.json.
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
import forecast_pilot2 as fp2  # noqa: E402

from qlab import forecast as fc  # noqa: E402
from qlab import forecast_inputs as fi  # noqa: E402

FOLDS = pd.date_range("2022-01-01", "2024-01-01", freq="6MS", tz="UTC")
TARGETS = [("reg", 4, None), ("reg", 24, None)] + [
    ("cls", h, s) for h in (1, 4) for s in ("size", "fall", "rise")]


def _gbm(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
    return cls(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=1000,
               random_state=0)


def _predict(kind, model, x):
    return model.predict(x) if kind == "reg" else model.predict_proba(x)[:, 1]


def _score(kind, p, y):
    return fc.pearson(p, y) if kind == "reg" else fc.auc(p, y)


def tname(t):
    return f"z {t[1]}h" if t[0] == "reg" else f"{t[1]}h {t[2]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    ap.add_argument("--variant")
    a = ap.parse_args()
    t0 = time.time()
    variant = a.variant or json.loads((a.work / "forecast2.json").read_text())["choice"][
        "variant"]
    V = fp2.build(a.work, a.minutes, variant)
    tau, d = V["tau"], V["d"]
    extra, extra_names = fi.calendar_and_size(V["logc"], tau)
    X = np.column_stack([V["X"], extra])
    names = V["names"] + extra_names
    rng = np.random.default_rng(17)
    out: dict = {"variant": variant, "names": names}

    def y_of(t):
        if t[0] == "reg":
            return V["targets"][t[1]]
        z = V["targets"][t[1]]
        if t[2] == "size":
            return np.where(np.isnan(z), np.nan, (np.abs(z) > fp2.K).astype(float))
        return V["labels"][(t[1], t[2])]

    # 1. clusters on the first period
    first = (tau < fp.CUT) & (tau >= d["start"])
    rho = np.nan_to_num(np.asarray(spearmanr(X[first][::6], nan_policy="omit").statistic),
                        nan=0.0)
    dist = 1 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform(dist, checks=False), method="average"), t=0.3,
                   criterion="distance")
    cols_of = {int(c): np.flatnonzero(lab == c) for c in np.unique(lab)}
    out["clusters"] = {c: [names[i] for i in cols] for c, cols in cols_of.items()}
    print("clusters", len(cols_of), round(time.time() - t0), "s", flush=True)

    def masks(h, f0, f1, y):
        tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= f0.value) & ~np.isnan(y) & (
            tau >= d["start"])
        va = (tau >= f0.value) & (tau + h * fp.HOUR <= f1.value) & ~np.isnan(y)
        return tr, va

    def fold_end(f0):
        return min(f0 + pd.DateOffset(months=6), pd.Timestamp(fp.CUT, tz="UTC"))

    # 2. clustered importance by permutation (other day + hour shift), per fold
    mda, stab = {}, {}
    for t in TARGETS:
        y, h = y_of(t), t[1]
        per = {c: [] for c in cols_of}
        for f0 in FOLDS:
            tr, va = masks(h, f0, fold_end(f0), y)
            m = _gbm(t[0]).fit(X[tr], y[tr])
            xv, yv = X[va], y[va]
            base = _score(t[0], _predict(t[0], m, xv), yv)
            src = fi.other_day_source(tau[va], rng)
            for c, cols in cols_of.items():
                xp = xv.copy()
                xp[:, cols] = xv[src][:, cols]
                per[c].append(base - _score(t[0], _predict(t[0], m, xp), yv))
        mda[tname(t)] = {c: float(np.mean(v)) for c, v in per.items()}
        stab[tname(t)] = {c: int(sum(x > 0 for x in v)) for c, v in per.items()}
        print("mda", tname(t), round(time.time() - t0), "s", flush=True)
    out["mda"], out["positive_folds_of_5"] = mda, stab

    # 3. shadow test at the last first-period fold (2024-01), 5 runs
    shadow, selected = {}, {}
    f0 = FOLDS[-1]
    for t in TARGETS:
        y, h = y_of(t), t[1]
        tr, va = masks(h, f0, pd.Timestamp(fp.CUT, tz="UTC"), y)
        wins = {c: 0 for c in cols_of}
        for _ in range(5):
            xs_tr = np.column_stack([X[tr], X[tr][fi.other_day_source(tau[tr], rng)]])
            xs_va = np.column_stack([X[va], X[va][fi.other_day_source(tau[va], rng)]])
            m = _gbm(t[0]).fit(xs_tr, y[tr])
            base = _score(t[0], _predict(t[0], m, xs_va), y[va])
            src = fi.other_day_source(tau[va], rng)
            for c, cols in cols_of.items():
                xp = xs_va.copy()
                xp[:, cols] = xs_va[src][:, cols]
                real = base - _score(t[0], _predict(t[0], m, xp), y[va])
                xq = xs_va.copy()
                sc = cols + X.shape[1]
                xq[:, sc] = xs_va[src][:, sc]
                fake = base - _score(t[0], _predict(t[0], m, xq), y[va])
                wins[c] += int(real > fake)
        shadow[tname(t)] = wins
        selected[tname(t)] = [c for c in cols_of if wins[c] >= 4 and mda[tname(t)][c] > 0]
        print("shadow", tname(t), round(time.time() - t0), "s", flush=True)
    out["shadow_wins_of_5"], out["selected"] = shadow, selected

    # 4. single-cluster importance: a model on one cluster only, per fold
    sfi = {}
    for t in TARGETS:
        y, h = y_of(t), t[1]
        per = {c: [] for c in cols_of}
        for f0 in FOLDS:
            tr, va = masks(h, f0, fold_end(f0), y)
            for c, cols in cols_of.items():
                m = _gbm(t[0]).fit(X[tr][:, cols], y[tr])
                per[c].append(_score(t[0], _predict(t[0], m, X[va][:, cols]), y[va]))
        sfi[tname(t)] = {c: {"mean": float(np.mean(v)), "folds": [float(x) for x in v]}
                         for c, v in per.items()}
        print("sfi", tname(t), round(time.time() - t0), "s", flush=True)
    out["sfi"] = sfi

    # 5. retrain on the selected clusters vs all features, both periods
    retrains = pd.date_range(fp.START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
    comp = {}
    for t in TARGETS:
        tn = tname(t)
        y, h = y_of(t), t[1]
        cols = (np.concatenate([cols_of[c] for c in selected[tn]]) if selected[tn]
                else np.array([], dtype=int))
        p_all, p_sel = np.full(len(tau), np.nan), np.full(len(tau), np.nan)
        for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
            tr = (tau + h * fp.HOUR + (h * fp.HOUR + fp.DAY) <= q0.value) & ~np.isnan(y) & (
                tau >= d["start"])
            sel = (tau >= q0.value) & (tau < q1.value)
            p_all[sel] = _predict(t[0], _gbm(t[0]).fit(X[tr], y[tr]), X[sel])
            if len(cols):
                p_sel[sel] = _predict(t[0], _gbm(t[0]).fit(X[tr][:, cols], y[tr]),
                                      X[sel][:, cols])
        row = {"n_selected_clusters": len(selected[tn])}
        for part, mask in fp2.parts(tau, h).items():
            for lbl, p in (("all", p_all), ("selected", p_sel)):
                ok = mask & ~np.isnan(p) & ~np.isnan(y)
                if not ok.any():
                    row[f"{lbl} {part}"] = None
                    continue
                pp, yy, dd = p[ok], y[ok], V["dates"][ok]
                row[f"{lbl} {part}"] = fp._boot(
                    dd, lambda r, pp=pp, yy=yy, k=t[0]: _score(k, pp[r], yy[r]))
        comp[tn] = row
        print("retrain", tn, round(time.time() - t0), "s", flush=True)
    out["selected_vs_all"] = comp
    n_cl = len(cols_of)
    out["trials"] = {"configurations": len(TARGETS) * (1 + 5 + n_cl + 2),
                     "fits": len(TARGETS) * (len(FOLDS) + 5 + n_cl * len(FOLDS) + 20)}
    out["seconds"] = round(time.time() - t0)
    (a.work / "forecast2-select.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s")


if __name__ == "__main__":
    main()
