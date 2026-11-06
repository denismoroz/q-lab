"""A planted signal of known strength through the full forecast pipeline — diagnostics
(docs/research/planted-signal-protocol-2026-11-06.md).

    uv run python scripts/research/planted_signal.py --work DIR [--procs 14]

Writes DIR/planted.json.
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

COINS = ("BTC", "AVAX")
H = (1, 4)
LEVELS = (0.0, 0.02, 0.05, 0.10)
NET_LEVELS = (0.0, 0.05, 0.10)
SEED = 20261106
DRAWS = 50
SELECTORS = ("l1", "sfm", "rfecv", "boruta", "mi10")
SELECT_MODEL = {("BTC", 1): "ridge", ("BTC", 4): "ridge", ("AVAX", 1): "ridge",
                ("AVAX", 4): "gbm"}


def planted_columns(n_features: int) -> list[int]:
    return [int(j) for j in np.random.default_rng(SEED).choice(n_features, 3, replace=False)]


def standardised(f: np.ndarray) -> np.ndarray:
    s = (f - np.nanmean(f)) / np.nanstd(f)
    return np.nan_to_num(s)


def coefficient(s: np.ndarray, z: np.ndarray, rho: float) -> float:
    """c with corr(s, z + c s) = rho over the rows where z is known (bisection; 0 for rho 0)."""
    if rho == 0:
        return 0.0
    ok = ~np.isnan(z)
    a, b = s[ok], z[ok]

    def corr(c):
        return float(np.corrcoef(a, b + c * a)[0, 1])

    lo, hi = -1.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if corr(mid) < rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _build(work: Path, coin: str):
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    return fp2.build(work, f3.MINUTES, "e", coin)


def _job(args):
    work, coin, h, j, rho = args
    import fix_stopping as fs
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3
    import three_select as tsel

    from qlab import forecast as fc

    t0 = time.time()
    V = _build(work, coin)
    X, tau, start, names = V["X"], V["tau"], V["d"]["start"], V["names"]
    z = V["targets"][h]
    s = standardised(X[:, j])
    c = coefficient(s, z, rho)
    y = z + c * s
    parts = fp2.parts(tau, h)

    def score(p):
        return {part: f3._point("reg", p, y, mk) for part, mk in parts.items()}

    ok = ~np.isnan(y)
    out = {"coin": coin, "h": h, "feature": names[j], "column": j, "rho": rho, "c": c,
           "realised_full_ic": float(np.corrcoef(s[ok], y[ok])[0, 1]),
           "realised_ic_by_part": {part: f3._point("reg", s, y, mk) for part, mk in parts.items()}}
    ridge = f3._models()["ridge"]
    seed0 = 700_000 + 1000 * COINS.index(coin) + 10 * h + 100 * j
    for mname in ("ridge", "gbm"):
        def walk(rng, mname=mname):
            if mname == "ridge":
                return f3._walk(*ridge, X, tau, start, h, y, rng)
            return fs.walk_ts("reg", X, tau, start, h, y, rng, block_weeks=8)[0]
        real = score(walk(None))
        null = [score(walk(np.random.default_rng(seed0 + k))) for k in range(DRAWS)]
        row = {}
        for part in ("first", "last"):
            d = np.array([n[part] for n in null])
            row[part] = {"oos": real[part], "null_q95": float(np.nanquantile(d, 0.95)),
                         "pct": float(np.mean(d < real[part]))}
        out[mname] = row
    # the five selectors as configured in check 4, with that check's model for the series
    model = SELECT_MODEL[(coin, h)]
    gap = h * f3.HOUR + (h * f3.HOUR + f3.DAY)
    sel_rows = {}
    for name in SELECTORS:
        kept, infos = [], []
        sel_rng = np.random.default_rng(7)
        p = np.full(len(tau), np.nan)
        for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
            m = (tau + gap <= q0.value) & ~np.isnan(y) & (tau >= start)
            sel = (tau >= q0.value) & (tau < q1.value)
            if not sel.any() or m.sum() < 1000:
                continue
            info: dict = {}
            cols = tsel.select(name, "reg", X, y, m, tau, h, gap, sel_rng, info)
            kept.append([int(x) for x in cols])
            infos.append({k: v for k, v in info.items() if k != "estimator"} | (
                {"estimator": info["estimator"]} if len(infos) == 0 and "estimator" in info
                else {}))
            if not len(cols):
                p[sel] = 0.0
                continue
            Xc = X[:, cols]
            if model == "ridge":
                xs = tsel._std(Xc, m)
                from sklearn.linear_model import Ridge

                p[sel] = Ridge(alpha=10).fit(xs[m], y[m]).predict(xs[sel])
            else:
                b0 = q0.value - gap - tsel.BLOCK
                blk, fit = m & (tau >= b0), m & (tau + gap <= b0)
                p[sel] = tsel._gbm("reg").fit(Xc[fit], y[fit], X_val=Xc[blk],
                                              y_val=y[blk]).predict(Xc[sel])
        sel_rows[name] = {"planted_kept_windows": int(sum(j in k for k in kept)),
                          "windows": len(kept), "n_kept": [len(k) for k in kept],
                          "oos": score(p), "fitted": infos}
    out["selectors"] = {"model": model, **sel_rows}
    out["seconds"] = round(time.time() - t0)
    del fc
    return out


def _net_job(args):
    work, coin, j, rho = args
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3
    import three_capacity as tc

    V = _build(work, coin)
    X, tau = V["X"], V["tau"]
    s = standardised(X[:, j])
    ys = {}
    cs = {}
    for h in H:
        cs[h] = coefficient(s, V["targets"][h], rho)
        ys[h] = V["targets"][h] + cs[h] * s
    tc._D.update(X=X, seq=V["seq"].astype(np.float32), tau=tau,
                 coin=np.zeros(len(tau), dtype=int), start_ok=tau >= V["d"]["start"], y=ys,
                 retrains=f3.RETRAINS, cut=f3.fp.CUT)
    p, rows = tc.walk_net(0)
    res = {"coin": coin, "column": j, "rho": rho, "c": {str(h): c for h, c in cs.items()},
           "epochs": [r["epochs_run"] for r in rows], "best_epoch": [r["best_epoch"] for r in rows]}
    for k, h in enumerate(H):
        parts = fp2.parts(tau, h)
        res[f"z{h}h"] = {part: f3._point("reg", p[:, k], ys[h], mk) for part, mk in parts.items()}
        res[f"z{h}h"]["in_sample"] = float(np.mean([r[f"in_sample_{h}h"] for r in rows]))
        res[f"z{h}h"]["block"] = float(np.mean([r[f"block_{h}h"] for r in rows]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    V = _build(a.work, "BTC")
    cols = planted_columns(len(V["names"]))
    print("planted columns", [(c, V["names"][c]) for c in cols], flush=True)
    jobs = [(a.work, c, h, j, r) for c in COINS for h in H for j in cols for r in LEVELS]
    net_jobs = [(a.work, c, cols[0], r) for c in COINS for r in NET_LEVELS]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs) as pool:
        nets_async = pool.map_async(_net_job, net_jobs)
        got = pool.map(_job, jobs)
        nets = nets_async.get()
    out = {"planted_columns": {str(c): V["names"][c] for c in cols}, "seed": SEED,
           "runs": got, "nets": nets, "seconds": round(time.time() - t0)}
    (a.work / "planted.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
