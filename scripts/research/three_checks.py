"""Three checks: lead-lag features, information without a model, model capacity
(docs/research/three-checks-protocol-2026-11-06.md).

    uv run python scripts/research/three_checks.py leadlag-prep --work DIR
    uv run python scripts/research/three_checks.py leadlag-run --work DIR [--procs 12]
    uv run python scripts/research/three_checks.py mi --work DIR [--procs 12]
    uv run python scripts/research/three_checks.py capacity --work DIR

Writes DIR/three-*.json (and DIR/three-leadlag-<coin>.npz).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ALTS = ("ETH", "SOL", "AVAX")
COINS = ("BTC", "ETH", "SOL", "AVAX")
TARGETS = ("z15", "z1", "z4", "s15", "s1", "s4")
HOURS = {"z15": 0.25, "z1": 1, "z4": 4, "s15": 0.25, "s1": 1, "s4": 4}
DRAWS = 50
LAGS = (1, 5, 15, 60)


# ── 1. lead-lag ──────────────────────────────────────────────────────────


def _pure_leadlag(alt: pd.DataFrame, btc: pd.DataFrame) -> dict:
    """Per year: corr(BTC's last-k-minute return, the alt's next-k-minute return) and the alt's
    own corr(last k, next k), on non-overlapping k-minute steps of a regular minute grid."""
    grid = pd.date_range(max(alt.index[0], btc.index[0]), min(alt.index[-1], btc.index[-1]),
                         freq="min")
    la = np.log(alt["close"].reindex(grid).to_numpy())
    lb = np.log(btc["close"].reindex(grid).to_numpy())
    years = grid.year.to_numpy()
    out = {}
    for k in LAGS:
        t = np.arange(k, len(grid) - k, k)
        past_b = lb[t] - lb[t - k]
        past_a = la[t] - la[t - k]
        nxt_a = la[t + k] - la[t]
        row = {}
        for y in np.unique(years[t]):
            s = years[t] == y
            ok = s & ~np.isnan(past_b) & ~np.isnan(nxt_a) & ~np.isnan(past_a)
            if ok.sum() < 1000:
                continue
            row[str(y)] = {"btc_leads": float(np.corrcoef(past_b[ok], nxt_a[ok])[0, 1]),
                           "own_past": float(np.corrcoef(past_a[ok], nxt_a[ok])[0, 1]),
                           "n": int(ok.sum())}
        out[f"k{k}"] = row
    return out


def leadlag_prep(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc
    from qlab import forecast_inputs as fi

    t0 = time.time()
    rec3 = json.loads((work / "forecast3.json").read_text())
    mins = {c: fp2.load_minutes(f3.MINUTES, c) for c in COINS}
    pure = {}
    for coin in ALTS:
        V = fp2.build(work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
        alt = mins[coin]
        pos = V["pos"]
        lead_x, lead_names = fi.leader_features(alt, mins["BTC"], pos, "BTC")
        if coin in ("SOL", "AVAX"):
            x2, n2 = fi.leader_features(alt, mins["ETH"], pos, "ETH")
            lead_x, lead_names = np.column_stack([lead_x, x2]), lead_names + n2
        close, open_ns = alt["close"].to_numpy(), alt.index.asi8
        pos_ok = np.where(V["d"]["g60"]["ok"], pos, -1)
        z15, _ = fi.minute_z_target(close, open_ns, pos_ok, 15)
        logm = np.log(close.astype(np.float64))
        p = np.clip(pos, 0, None)
        f15 = np.full(len(pos), np.nan)
        ok = (pos_ok >= 0) & (pos + 15 < len(close))
        i = np.flatnonzero(ok)
        good = open_ns[p[i] + 15] == open_ns[p[i]] + 15 * fi.MINUTE
        f15[i[good]] = logm[p[i[good]] + 15] - logm[p[i[good]]]
        tg = {"z15": z15, "z1": V["targets"][1], "z4": V["targets"][4]}
        for h, key in ((0.25, "s15"), (1, "s1"), (4, "s4")):
            fwd = f15 if h == 0.25 else fc.forward_return(V["logc"], int(h))
            tg[key] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        np.savez_compressed(work / f"three-leadlag-{coin}.npz", X=V["X"], L=lead_x, tau=V["tau"],
                            start=np.array([V["d"]["start"]]), **tg)
        (work / f"three-leadlag-{coin}-names.json").write_text(json.dumps(
            {"base": V["names"], "lead": lead_names}))
        pure[coin] = _pure_leadlag(alt, mins["BTC"])
        print(coin, "prep", round(time.time() - t0), "s", flush=True)
    (work / "three-leadlag-pure.json").write_text(json.dumps(pure, indent=1))


def _ll_job(args):
    work, coin, target = args
    import fix_stopping as fs
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    d = np.load(work / f"three-leadlag-{coin}.npz")
    tau, start, y = d["tau"], int(d["start"][0]), d[target]
    h = HOURS[target]
    kind = "reg" if target.startswith("z") else "cls"
    parts = fp2.parts(tau, h)
    sets = {"base": d["X"], "base+lead": np.column_stack([d["X"], d["L"]])}
    models = f3._models()
    lin = models["ridge"] if kind == "reg" else models["logit"]
    center = kind == "cls"
    res = {}
    seed0 = 200_000 + 10_000 * ALTS.index(coin) + 100 * TARGETS.index(target)

    def score(p):
        out = {}
        for part, mk in parts.items():
            ok = mk & ~np.isnan(p) & ~np.isnan(y)
            out[part] = (fc.pearson(p[ok], y[ok]) if kind == "reg" else fc.auc(p[ok], y[ok]))
        return out

    for sname, X in sets.items():
        for mname in ("linear", "boosting"):
            def walk(rng, X=X, mname=mname):
                if mname == "linear":
                    make, scaled, k_ = lin
                    return f3._walk(make, scaled, k_, X, tau, start, h, y, rng, center=center)
                return fs.walk_ts(kind, X, tau, start, h, y, rng, center=center,
                                  block_weeks=8)[0]
            real = score(walk(None))
            null = [score(walk(np.random.default_rng(seed0 + k))) for k in range(DRAWS)]
            row = {}
            for part in ("first", "last"):
                dd = np.array([n[part] for n in null])
                row[part] = {"real": real[part], "null_q95": float(np.nanquantile(dd, 0.95)),
                             "pct": float(np.mean(dd < real[part]))}
            row["pass"] = bool(all(row[p]["real"] > row[p]["null_q95"] for p in ("first", "last")))
            res[f"{sname} {mname}"] = row
    return coin, target, res


def leadlag_run(work: Path, procs: int) -> None:
    t0 = time.time()
    jobs = [(work, c, t) for c in ALTS for t in TARGETS]
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_ll_job, jobs)
    out = {f"{c} {t}": r for c, t, r in got}
    out["seconds"] = round(time.time() - t0)
    (work / "three-leadlag.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


# ── 2. information without a model ──────────────────────────────────────


def _mi_job(args):
    work, coin, h, what = args
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_regression

    from qlab import forecast_inputs as fi

    rec3 = json.loads((work / "forecast3.json").read_text())
    V = fp2.build(work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
    tau, X, names = V["tau"], V["X"], V["names"]
    z = V["targets"][h]
    y = np.abs(z) if what == "size" else z
    rows = np.flatnonzero((tau >= f3.RETRAINS[0].value) & ~np.isnan(y))[::8]
    xs = X[rows]
    med = np.nanmedian(xs, 0)
    xs = np.where(np.isnan(xs), med, xs)
    ys = y[rows]
    real = mutual_info_regression(xs, ys, n_neighbors=3, random_state=0)
    rng = np.random.default_rng(300_000 + 1000 * COINS.index(coin) + 10 * h + (what == "size"))
    null = np.array([mutual_info_regression(
        xs, ys[fi.other_day_source(tau[rows], rng)], n_neighbors=3, random_state=0)
        for _ in range(DRAWS)])
    # clusters on the first period, as the selection of pilot 3
    first = (tau < f3.fp.CUT) & (tau >= V["d"]["start"])
    rho = np.nan_to_num(np.asarray(spearmanr(X[first][::6], nan_policy="omit").statistic),
                        nan=0.0)
    dist = 1 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    lab = fcluster(linkage(squareform(dist, checks=False), method="average"), t=0.3,
                   criterion="distance")
    feats = []
    for j, n in enumerate(names):
        feats.append({"feature": n, "mi": float(real[j]), "null_mean": float(null[:, j].mean()),
                      "null_q95": float(np.quantile(null[:, j], 0.95)),
                      "pct": float(np.mean(null[:, j] < real[j])), "cluster": int(lab[j])})
    clusters = []
    for c in np.unique(lab):
        cols = np.flatnonzero(lab == c)
        r, nl = real[cols].max(), null[:, cols].max(axis=1)
        clusters.append({"cluster": int(c), "features": [names[j] for j in cols],
                         "mi": float(r), "null_q95": float(np.quantile(nl, 0.95)),
                         "pct": float(np.mean(nl < r))})
    top = sorted(range(len(names)), key=lambda j: (-(real[j] - null[:, j].mean())
                                                    / (null[:, j].std() + 1e-12)))[:3]
    binned = {}
    for j in top:
        dec = pd.qcut(xs[:, j], 10, labels=False, duplicates="drop")
        binned[names[j]] = [{"decile": int(b), "target_mean": float(ys[dec == b].mean()),
                             "z_mean": float(z[rows][dec == b].mean())}
                            for b in np.unique(dec)]
    return f"{coin} {what} {h}h", {"rows": int(len(rows)), "features": feats,
                                   "clusters": clusters, "binned_top3": binned,
                                   "total_mi_real": float(real.sum()),
                                   "total_mi_null_mean": float(null.sum(1).mean()),
                                   "total_mi_null_q95": float(np.quantile(null.sum(1), 0.95))}


def mi_run(work: Path, procs: int) -> None:
    t0 = time.time()
    jobs = [(work, c, h, w) for c in COINS for h in (1, 4, 24) for w in ("direction", "size")]
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_mi_job, jobs)
    out = dict(got)
    out["seconds"] = round(time.time() - t0)
    (work / "three-mi.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["leadlag-prep", "leadlag-run", "mi", "capacity"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.phase == "leadlag-prep":
        leadlag_prep(a.work)
    elif a.phase == "leadlag-run":
        leadlag_run(a.work, a.procs)
    elif a.phase == "mi":
        mi_run(a.work, a.procs)
    else:
        import three_capacity

        three_capacity.run(a.work)


if __name__ == "__main__":
    main()
