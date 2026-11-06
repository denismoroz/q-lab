"""Time-ordered early stopping for the pilot-3 boosting forecasts
(docs/research/fix-stopping-protocol-2026-11-06.md).

    uv run python scripts/research/fix_stopping.py run --work DIR [--procs 12]

Each half-year retrain holds out the last `block_weeks` of its training window (after the same
gap as before the check), stops on that block's loss (10 trees without improvement, at most
1 000), and keeps the stopped model. Writes DIR/fix-stopping.json and
DIR/fix-stopping-forecasts.npz (the new walk-forward forecasts and 10 shuffled ones per series).
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
DRAWS = 50
KEEP_PLACEBOS = 10
BLOCKS = (8, 16)


def _gbm(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
    return cls(max_depth=3, max_iter=1000, learning_rate=0.05, min_samples_leaf=1000,
               random_state=0, early_stopping=True, n_iter_no_change=10, scoring="loss",
               validation_fraction=None)


def walk_ts(kind, X, tau, start, h, y, rng=None, center=False, block_weeks=8, curves=False):
    """Half-yearly walk-forward with the time-ordered stop; predictions and per-retrain rows."""
    import forecast_pilot3 as f3

    from qlab import forecast as fc
    from qlab import forecast_inputs as fi

    gap = h * f3.HOUR + (h * f3.HOUR + f3.DAY)
    block = block_weeks * 7 * f3.DAY
    p = np.full(len(tau), np.nan)
    rows = []
    for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
        m = (tau + gap <= q0.value) & ~np.isnan(y) & (tau >= start)
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 1000:
            continue
        yy = y if rng is None else fi.shuffled_labels(y, m, tau, "other day", rng)
        b0 = q0.value - gap - block
        blk = m & (tau >= b0)
        fit = m & (tau + gap <= b0)
        mod = _gbm(kind).fit(X[fit], yy[fit], X_val=X[blk], y_val=yy[blk])

        def pred(x, mod=mod):
            return mod.predict(x) if kind == "reg" else mod.predict_proba(x)[:, 1]

        def score(a, b):
            return fc.pearson(a, b) if kind == "reg" else fc.auc(a, b)

        out = pred(X[sel])
        if center:
            out = out - np.mean(yy[fit])
        p[sel] = out
        row = {"retrain": str(q0.date()), "fit_rows": int(fit.sum()), "block_rows": int(blk.sum()),
               "trees": int(mod.n_iter_), "hit_max": bool(mod.n_iter_ >= 1000),
               "in_sample": score(pred(X[fit]), yy[fit]), "block": score(pred(X[blk]), yy[blk]),
               "next_half_year": score(out, y[sel])}
        if curves:
            row["train_loss"] = [round(-float(v), 6) for v in mod.train_score_]
            row["block_loss"] = [round(-float(v), 6) for v in mod.validation_score_]
        rows.append(row)
    return p, rows


def series_list(work: Path):
    rec3 = json.loads((work / "forecast3.json").read_text())
    extra3 = json.loads((work / "forecast3-extra.json").read_text())
    out = []
    for coin in COINS:
        for h in H:
            zm = rec3["table"][f"{coin} {h}h"]["direction"]["model"]
            sm = extra3["sign"][f"{coin} {h}h"]["model"]
            out.append((coin, "z", h, zm))
            out.append((coin, "sign", h, sm))
    return out, rec3


def _job(args):
    """One boosting series: the real walk on both blocks, 50 shuffled draws on 8 weeks."""
    work, coin, fam, h = args
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    t0 = time.time()
    rec3 = json.loads((work / "forecast3.json").read_text())
    V = fp2.build(work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
    tau, X, start = V["tau"], V["X"], V["d"]["start"]
    parts = fp2.parts(tau, h)
    ci = f3.COINS.index(coin)
    if fam == "z":
        y, kind, center = V["targets"][h], "reg", False
        idx = list(f3._targets(V)).index(("reg", h, None))

        def seed(k):
            return 10_000 * (1 + ci) + 100 * k + 7 * idx
    else:
        fwd = fc.forward_return(V["logc"], h)
        y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        kind, center = "cls", True

        def seed(k):
            return 50_000 * (1 + ci) + 100 * k + h
    ysig = np.where(np.isnan(y), np.nan, np.where(y > 0, 1.0, -1.0)) if fam == "sign" else None

    def measures(p):
        r = {}
        for part, mk in parts.items():
            ok = mk & ~np.isnan(p) & ~np.isnan(y)
            r[part] = (fc.pearson(p[ok], y[ok]) if kind == "reg" else fc.auc(p[ok], y[ok]))
            if fam == "sign":
                r[f"hit_{part}"] = fc.hit_rate(p[ok], ysig[ok])
        return r

    res: dict = {"coin": coin, "family": fam, "h": h, "real": {}, "null": []}
    arrays = {}
    for bw in BLOCKS:
        p, rows = walk_ts(kind, X, tau, start, h, y, center=center, block_weeks=bw,
                          curves=(bw == 8))
        res["real"][f"{bw}w"] = {"measures": measures(p), "per_retrain": rows}
        if bw == 8:
            arrays["real"] = p
    for k in range(DRAWS):
        q, rows = walk_ts(kind, X, tau, start, h, y, np.random.default_rng(seed(k)),
                          center=center, block_weeks=8)
        res["null"].append({**measures(q), "trees": [r["trees"] for r in rows]})
        if k < KEEP_PLACEBOS:
            arrays[f"placebo{k}"] = q
    arrays["tau"] = tau
    res["seconds"] = round(time.time() - t0)
    return res, arrays


def run(work: Path, procs: int) -> None:
    t0 = time.time()
    series, rec3 = series_list(work)
    boost = [(coin, fam, h) for coin, fam, h, mdl in series if mdl in ("gbm", "gbc")]
    print("boosting series", boost, flush=True)
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_job, [(work, *b) for b in boost])
    out: dict = {"boosting": {}, "unchanged": [f"{c} {f} {h}h {m}" for c, f, h, m in series
                                               if m not in ("gbm", "gbc")]}
    npz = {}
    for (coin, fam, h), (res, arrays) in zip(boost, got, strict=True):
        key = f"{fam}_{coin}_{h}"
        out["boosting"][key] = res
        for k, v in arrays.items():
            npz[f"{key}_{k}"] = v
    np.savez_compressed(work / "fix-stopping-forecasts.npz", **npz)
    out["seconds"] = round(time.time() - t0)
    (work / "fix-stopping.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    run(a.work, a.procs)


if __name__ == "__main__":
    main()
