"""The proof table of the forecast models' protections — diagnostics, not candidates
(docs/research/forecast-proof-table-protocol-2026-11-06.md).

    uv run python scripts/research/forecast_proof.py --work DIR

Writes DIR/proof.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_pilot as fp  # noqa: E402
import forecast_pilot2 as fp2  # noqa: E402
import forecast_pilot3 as f3  # noqa: E402

from qlab import forecast as fc  # noqa: E402

SERIES = {("BTC", 4): "ridge", ("BTC", 24): "gbm", ("AVAX", 4): "gbm", ("AVAX", 24): "gbm"}
ALPHAS = (0.1, 1, 10, 100, 1000, 10000)
LEAVES = (20, 100, 300, 1000, 3000, 10000)
ROUNDS = (10, 25, 50, 100, 200, 300, 500, 750, 1000)


def _ridge(alpha):
    from sklearn.linear_model import Ridge

    return lambda: Ridge(alpha=alpha)


def _gbm(leaf, **kw):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=kw.get("max_iter", 200),
                                                 learning_rate=0.05, min_samples_leaf=leaf,
                                                 random_state=0,
                                                 early_stopping=kw.get("early_stopping", "auto"))


def walk(make, scaled, X, tau, start, h, y, purge=True):
    """Half-yearly walk-forward; returns predictions and the in-sample IC per retrain."""
    p = np.full(len(tau), np.nan)
    ins, kept = [], []
    for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
        gap = (h * f3.HOUR + (h * f3.HOUR + f3.DAY)) if purge else 0
        m = (tau + gap <= q0.value) & ~np.isnan(y) & (tau >= start)
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 1000:
            continue
        if scaled:
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            xt, xp = np.nan_to_num((X[m] - mu) / sd), np.nan_to_num((X[sel] - mu) / sd)
        else:
            xt, xp = X[m], X[sel]
        mod = make().fit(xt, y[m])
        ins.append(fc.pearson(mod.predict(xt), y[m]))
        if hasattr(mod, "do_early_stopping_"):
            kept.append(int(mod.n_iter_))
        p[sel] = mod.predict(xp)
    return p, ins, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    from sklearn.model_selection import KFold

    out: dict = {"label": "diagnostics, not candidates"}
    Vs = {c: fp2.build(a.work, f3.MINUTES, "e", c) for c in ("BTC", "AVAX")}
    for (coin, h), chosen in SERIES.items():
        V = Vs[coin]
        X, tau, start, y = V["X"], V["tau"], V["d"]["start"], V["targets"][h]
        parts = fp2.parts(tau, h)

        def ic(p, part, y=y, parts=parts):
            return f3._point("reg", p, y, parts[part])

        row: dict = {"chosen": chosen}
        # 1. regularisation curves
        reg = {"ridge": {}, "gbm": {}}
        for alpha in ALPHAS:
            p, ins, _ = walk(_ridge(alpha), True, X, tau, start, h, y)
            reg["ridge"][str(alpha)] = {"in_sample": float(np.mean(ins)), "first": ic(p, "first"),
                                        "last": ic(p, "last")}
        for leaf in LEAVES:
            p, ins, kept = walk(_gbm(leaf), False, X, tau, start, h, y)
            reg["gbm"][str(leaf)] = {"in_sample": float(np.mean(ins)), "first": ic(p, "first"),
                                     "last": ic(p, "last"), "trees_kept_mean": float(np.mean(kept))}
        row["regularisation"] = reg
        print(coin, h, "regularisation", round(time.time() - t0), "s", flush=True)
        # 2. training length: boosting without early stopping, first-period retrains
        curve = {r: {"in_sample": [], "next": []} for r in ROUNDS}
        kept = []
        for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
            if q0.value >= fp.CUT:
                break
            m = (tau + h * f3.HOUR + (h * f3.HOUR + f3.DAY) <= q0.value) & ~np.isnan(y) & (
                tau >= start)
            sel = (tau >= q0.value) & (tau < min(q1.value, fp.CUT)) & ~np.isnan(y)
            mod = _gbm(1000, max_iter=1000, early_stopping=False)().fit(X[m], y[m])
            kept.append(int(_gbm(1000)().fit(X[m], y[m]).n_iter_))
            for k, (pi, po) in enumerate(zip(mod.staged_predict(X[m]), mod.staged_predict(X[sel]),
                                             strict=True), start=1):
                if k in curve:
                    curve[k]["in_sample"].append(fc.pearson(pi, y[m]))
                    curve[k]["next"].append(fc.pearson(po, y[sel]))
        row["training_length"] = {
            "rounds": {str(r): {"in_sample": float(np.mean(v["in_sample"])),
                                "next_half_year": float(np.mean(v["next"]))}
                       for r, v in curve.items()},
            "early_stop_kept_per_retrain": kept}
        print(coin, h, "training length", round(time.time() - t0), "s", flush=True)
        # 3. purge and embargo
        make = _ridge(10) if chosen == "ridge" else _gbm(1000)
        scaled = chosen == "ridge"
        p_with, _, _ = walk(make, scaled, X, tau, start, h, y, purge=True)
        p_without, _, _ = walk(make, scaled, X, tau, start, h, y, purge=False)
        rows = np.flatnonzero(parts["first"] & ~np.isnan(y) & (tau >= start))
        pk = np.full(len(tau), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(rows):
            itr, ite = rows[tr], rows[te]
            if scaled:
                mu, sd = np.nanmean(X[itr], 0), np.nanstd(X[itr], 0) + 1e-9
                mod = make().fit(np.nan_to_num((X[itr] - mu) / sd), y[itr])
                pk[ite] = mod.predict(np.nan_to_num((X[ite] - mu) / sd))
            else:
                pk[ite] = make().fit(X[itr], y[itr]).predict(X[ite])
        row["purge"] = {"with_first": ic(p_with, "first"), "with_last": ic(p_with, "last"),
                        "without_first": ic(p_without, "first"),
                        "without_last": ic(p_without, "last"),
                        "random_5fold_first": ic(pk, "first")}
        print(coin, h, "purge", round(time.time() - t0), "s", flush=True)
        # 4. planted leak: the next hour's return as a feature
        nxt = fc.forward_return(V["logc"], 1)
        p_leak, _, _ = walk(make, scaled, np.column_stack([X, nxt]), tau, start, h, y)
        row["planted_leak"] = {"with_leak_first": ic(p_leak, "first"),
                               "with_leak_last": ic(p_leak, "last"),
                               "real_first": ic(p_with, "first"), "real_last": ic(p_with, "last")}
        # 5. the null distribution of pilot 3
        r3 = json.loads((a.work / f"forecast3-{coin}-e.json").read_text())
        key = f"z {h}h {chosen}"
        null = {}
        for part in ("first", "last"):
            d = np.asarray(r3["null"][key][part])
            real = r3["real"][key][part][0]
            null[part] = {"real": real, "null_mean": float(np.nanmean(d)),
                          "null_q95": float(np.nanquantile(d, 0.95)),
                          "null_max": float(np.nanmax(d)), "pct": float(np.mean(d < real))}
        row["null"] = null
        out[f"{coin} {h}h"] = row
        print(coin, h, "done", round(time.time() - t0), "s", flush=True)
    out["seconds"] = round(time.time() - t0)
    (a.work / "proof.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s")


if __name__ == "__main__":
    main()
