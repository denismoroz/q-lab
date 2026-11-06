"""Check 4 of the three checks: library feature selection INSIDE each walk-forward training window
(docs/research/three-checks-selection-addendum-2026-11-06.md).

    uv run python scripts/research/three_select.py --work DIR [--procs 14]

Writes DIR/three-select.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
H = (1, 4, 24)
SELECTORS = ("l1", "sfm", "rfecv", "boruta", "mi10")
NULL = 10
HOUR = 3_600_000_000_000
DAY = 24 * HOUR
BLOCK = 8 * 7 * DAY


def _std(X, m):
    mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
    return np.nan_to_num((X - mu) / sd)


def _gbm(kind, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
    return cls(max_depth=3, max_iter=1000, learning_rate=0.05, min_samples_leaf=1000,
               random_state=0, early_stopping=True, n_iter_no_change=10,
               validation_fraction=None, **kw)


def select(name, kind, X, y, m, tau, h, gap, rng):
    """Feature columns kept by selector `name`, from the training rows `m` only."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.feature_selection import (
        RFECV,
        SelectFromModel,
        mutual_info_classif,
        mutual_info_regression,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV, Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TimeSeriesSplit

    from qlab import forecast as fc
    from qlab import forecast_inputs as fi

    rows = np.flatnonzero(m)
    gap_rows = int(h + 24)
    if name in ("l1", "rfecv"):
        xs = _std(X, m)[rows][::2]
        ys = y[rows][::2]
        g2 = max(1, gap_rows // 2)
        if name == "l1":
            if kind == "reg":
                est = LassoCV(cv=TimeSeriesSplit(4, gap=g2), n_alphas=20, max_iter=2000,
                              random_state=0).fit(xs, ys)
                coef = est.coef_
            else:
                est = LogisticRegressionCV(Cs=8, cv=TimeSeriesSplit(4, gap=g2), penalty="l1",
                                           solver="liblinear", scoring="roc_auc",
                                           max_iter=200).fit(xs, ys)
                coef = est.coef_[0]
            return np.flatnonzero(np.abs(coef) > 0)
        est = Ridge(alpha=10) if kind == "reg" else LogisticRegression(C=0.1, max_iter=300)
        scoring = (make_scorer(lambda a, b: fc.pearson(np.asarray(b), np.asarray(a)))
                   if kind == "reg" else "roc_auc")
        r = RFECV(est, step=4, cv=TimeSeriesSplit(3, gap=g2), scoring=scoring,
                  min_features_to_select=1).fit(xs, ys)
        return np.flatnonzero(r.support_)
    if name == "sfm":
        b0 = tau[rows].max() - BLOCK
        blk = m & (tau >= b0)
        fit = m & (tau + gap <= b0)
        mod = _gbm(kind).fit(X[fit], y[fit], X_val=X[blk], y_val=y[blk])
        imp = permutation_importance(mod, X[blk], y[blk], n_repeats=1, random_state=0,
                                     scoring="r2" if kind == "reg" else "roc_auc")
        sfm = SelectFromModel(mod, prefit=True, threshold="mean",
                              importance_getter=lambda _e: imp.importances_mean)
        return np.flatnonzero(sfm.get_support())
    xm = np.where(np.isnan(X[rows]), np.nanmedian(X[rows], 0), X[rows])
    ym = y[rows]
    if name == "mi10":
        sub = slice(None, None, 4)
        f = mutual_info_regression if kind == "reg" else mutual_info_classif
        mi = f(xm[sub], ym[sub], n_neighbors=3, random_state=0)
        return np.sort(np.argsort(-mi)[:10])
    # boruta: own implementation of the shadow principle (BorutaPy is not installed)
    hits = np.zeros(X.shape[1])
    tau_r = tau[rows]
    for rnd in range(5):
        src = fi.other_day_source(tau_r, rng)
        shadow = xm[src]
        pick = np.random.default_rng(rnd).choice(len(rows), size=min(10_000, len(rows)),
                                                 replace=False)
        both = np.column_stack([xm[pick], shadow[pick]])
        et = (ExtraTreesRegressor if kind == "reg" else ExtraTreesClassifier)(
            n_estimators=50, max_depth=6, random_state=rnd, n_jobs=1).fit(both, ym[pick])
        imp = et.feature_importances_
        hits += imp[: X.shape[1]] > imp[X.shape[1]:].max()
    return np.flatnonzero(hits >= 3)


def walk(name, model, kind, X, tau, start, h, y, rng=None, center=False):
    """Walk-forward with selection inside each training window; predictions and the kept sets."""
    import forecast_pilot3 as f3
    from sklearn.linear_model import LogisticRegression, Ridge

    from qlab import forecast_inputs as fi

    gap = h * HOUR + (h * HOUR + DAY)
    p = np.full(len(tau), np.nan)
    kept = []
    sel_rng = np.random.default_rng(7)
    for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
        m = (tau + gap <= q0.value) & ~np.isnan(y) & (tau >= start)
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 1000:
            continue
        yy = y if rng is None else fi.shuffled_labels(y, m, tau, "other day", rng)
        cols = select(name, kind, X, yy, m, tau, h, gap, sel_rng)
        kept.append([int(c) for c in cols])
        if not len(cols):
            p[sel] = 0.0  # no features kept: no knowledge
            continue
        Xc = X[:, cols]
        if model in ("ridge", "logit"):
            xs = _std(Xc, m)
            mod = (Ridge(alpha=10) if model == "ridge"
                   else LogisticRegression(C=0.1, max_iter=500)).fit(xs[m], yy[m])
            out = mod.predict(xs[sel]) if kind == "reg" else mod.predict_proba(xs[sel])[:, 1]
        else:
            b0 = q0.value - gap - BLOCK
            blk, fit = m & (tau >= b0), m & (tau + gap <= b0)
            mod = _gbm(kind).fit(Xc[fit], yy[fit], X_val=Xc[blk], y_val=yy[blk])
            out = mod.predict(Xc[sel]) if kind == "reg" else mod.predict_proba(Xc[sel])[:, 1]
        if center:
            out = out - np.mean(yy[m])
        p[sel] = out
    return p, kept


def _job(args):
    work, coin, fam, h = args
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    t0 = time.time()
    rec3 = json.loads((work / "forecast3.json").read_text())
    extra3 = json.loads((work / "forecast3-extra.json").read_text())
    V = fp2.build(work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
    tau, X, start, names = V["tau"], V["X"], V["d"]["start"], V["names"]
    parts = fp2.parts(tau, h)
    if fam == "z":
        y, kind, center = V["targets"][h], "reg", False
        model = rec3["table"][f"{coin} {h}h"]["direction"]["model"]
    else:
        fwd = fc.forward_return(V["logc"], h)
        y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        kind, center = "cls", True
        model = extra3["sign"][f"{coin} {h}h"]["model"]

    def score(p):
        out = {}
        for part, mk in parts.items():
            ok = mk & ~np.isnan(p) & ~np.isnan(y)
            out[part] = fc.pearson(p[ok], y[ok]) if kind == "reg" else fc.auc(p[ok], y[ok])
        return out

    res = {"model": model}
    for name in SELECTORS:
        p, kept = walk(name, model, kind, X, tau, start, h, y, center=center)
        real = score(p)
        null = []
        for k in range(NULL):
            rng = np.random.default_rng(600_000 + 1000 * COINS.index(coin) + 10 * h
                                        + (fam == "sign") * 5 + k * 100_000)
            q, _ = walk(name, model, kind, X, tau, start, h, y, rng, center=center)
            null.append(score(q))
        counts = np.zeros(len(names))
        for kc in kept:
            counts[kc] += 1
        row = {"kept_per_window": [[names[c] for c in kc] for kc in kept],
               "n_kept": [len(kc) for kc in kept],
               "survival": {names[j]: int(counts[j]) for j in np.argsort(-counts)
                            if counts[j] > 0},
               "windows": len(kept)}
        for part in ("first", "last"):
            d = np.array([n[part] for n in null])
            row[part] = {"real": real[part], "null_max": float(np.nanmax(d)),
                         "null_mean": float(np.nanmean(d))}
        row["pass"] = bool(all(row[p_]["real"] > row[p_]["null_max"] for p_ in ("first", "last")))
        res[name] = row
    res["seconds"] = round(time.time() - t0)
    return f"{coin} {fam} {h}h", res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(a.work, c, f, h) for c in COINS for f in ("z", "sign") for h in H]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        got = pool.map(_job, jobs)
    out = dict(got)
    out["seconds"] = round(time.time() - t0)
    (a.work / "three-select.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
