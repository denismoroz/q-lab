"""The feature-parameter sweep (docs/research/param-sweep-protocol-2026-11-06.md).

    uv run python scripts/research/param_sweep.py features --work DIR
    uv run python scripts/research/param_sweep.py map --work DIR
    uv run python scripts/research/param_sweep.py model --work DIR [--procs 14]

Writes DIR/sweep-<coin>.npz, DIR/sweep-map.json, DIR/sweep-model.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
TARGETS = ("sign15m", "sign1h", "sign4h", "sign24h", "zw15m", "zw1h", "zw4h", "zw24h")
HOURS = {"15m": 0.25, "1h": 1, "4h": 4, "24h": 24}
DRAWS = 50
SEED = 20261106
HOUR = 3_600_000_000_000
DAY = 24 * HOUR
FIRST0 = pd.Timestamp("2022-01-01", tz="UTC").value
CUT = pd.Timestamp("2024-06-12", tz="UTC").value


def horizon(t: str) -> float:
    return HOURS[t.replace("sign", "").replace("zw", "")]


def parts(tau, h):
    return {"first": (tau >= FIRST0) & (tau < CUT), "last": tau >= CUT + h * HOUR + DAY}


# ── features and targets ───────────────────────────────────────────────


def make_features(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc
    from qlab import forecast_inputs as fi
    from qlab import param_features as pf

    for coin in COINS:
        t0 = time.time()
        with open(work / "ml" / f"{coin}.pkl", "rb") as fh:
            d = pickle.load(fh)
        g = d["g60"]
        tau, pos, valid = g["tau"], g["pos"], g["valid"]
        m = fp2.load_minutes(f3.MINUTES, coin)
        S, names, fams = pf.sweep_features(m, pos, tau)
        close, open_ns = m["close"].to_numpy(), m.index.asi8
        p = np.clip(pos, 0, None)
        logc = np.where(valid, np.log(close[p]), np.nan)
        tg = {}
        for lab, h in HOURS.items():
            if h < 1:
                z, _ = fi.minute_z_target(close, open_ns, np.where(valid, pos, -1), 15)
                logm = np.log(close.astype(np.float64))
                fwd = np.full(len(pos), np.nan)
                ok = valid & (pos + 15 < len(close))
                i = np.flatnonzero(ok)
                good = open_ns[p[i] + 15] == open_ns[p[i]] + 15 * fi.MINUTE
                fwd[i[good]] = logm[p[i[good]] + 15] - logm[p[i[good]]]
            else:
                z, _ = fc.z_target(logc, int(h))
                fwd = fc.forward_return(logc, int(h))
            tg[f"zw{lab}"] = fc.winsorised(z, 3.0)
            tg[f"sign{lab}"] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        np.savez_compressed(work / f"sweep-{coin}.npz", S=S, tau=tau,
                            start=np.array([d["start"]]), **tg)
        (work / "sweep-names.json").write_text(json.dumps({"names": names, "families": fams}))
        print(coin, S.shape, round(time.time() - t0), "s", flush=True)


# ── step 1: the univariate map ────────────────────────────────────────


def _ranks(x):
    from scipy.stats import rankdata

    r = rankdata(x, axis=0)
    r = (r - r.mean(0)) / (r.std(0) + 1e-12)
    return r


def _map_job(args):
    work, coin, target, planted = args
    from qlab import forecast_inputs as fi

    z = np.load(work / f"sweep-{coin}.npz")
    S, tau, start = z["S"], z["tau"], int(z["start"][0])
    y = z[target].copy()
    if planted is not None:
        y = y + planted["c"] * planted["s"]
    h = horizon(target)
    out = {}
    for part, mk in parts(tau, h).items():
        rows = np.flatnonzero(mk & ~np.isnan(y) & (tau >= start))
        X = S[rows]
        med = np.nanmedian(X, 0)
        X = np.where(np.isnan(X), med, X)
        F = _ranks(X)
        yy = y[rows]
        n = len(rows)
        real = F.T @ _ranks(yy) / n
        rng = np.random.default_rng(SEED + 1000 * COINS.index(coin) + TARGETS.index(target)
                                    + (part == "last") * 100)
        null = np.array([F.T @ _ranks(yy[fi.other_day_source(tau[rows], rng)]) / n
                         for _ in range(DRAWS)])
        mu, sd = null.mean(0), null.std(0) + 1e-12
        from scipy.stats import norm

        pval = 2 * norm.sf(np.abs((real - mu) / sd))
        if target.startswith("sign"):
            fwd_sign = np.where(yy > 0.5, 1.0, -1.0)
        else:
            fwd_sign = np.sign(yy)
        out[part] = {"rank_ic": real.tolist(), "null_mean": mu.tolist(), "null_sd": sd.tolist(),
                     "p": pval.tolist(), "n": int(n), "median": med.tolist(),
                     "fwd_sign": None}
        out[part]["_fwd"] = fwd_sign
        out[part]["_X"] = X
    # hit rate: sign(setting - its first-period median) x sign of its first-period rank IC
    med1 = np.asarray(out["first"]["median"])
    sgn = np.sign(np.asarray(out["first"]["rank_ic"]))
    for part in out:
        X, fs = out[part].pop("_X"), out[part].pop("_fwd")
        call = np.sign(X - med1) * sgn
        ok = (call != 0) & (fs != 0)
        out[part]["hit"] = [float(np.mean((call[:, j] == fs)[ok[:, j]])) for j in range(X.shape[1])]
        out[part].pop("fwd_sign")
    return coin, target, out


def planted_spec(work: Path):
    from scipy.stats import spearmanr

    z = np.load(work / "sweep-BTC.npz")
    names = json.loads((work / "sweep-names.json").read_text())["names"]
    j = int(np.random.default_rng(SEED).integers(len(names)))
    x = z["S"][:, j]
    s = np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x))
    y = z["zw1h"]
    ok = ~np.isnan(y)
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if spearmanr(s[ok], (y + mid * s)[ok]).statistic < 0.05:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    return {"column": j, "name": names[j], "c": c, "s": s,
            "realised_rank_ic": float(spearmanr(s[ok], (y + c * s)[ok]).statistic)}


def run_map(work: Path, procs: int) -> None:
    t0 = time.time()
    plant = planted_spec(work)
    jobs = [(work, c, t, None) for c in COINS for t in TARGETS] + [(work, "BTC", "zw1h", plant)]
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_map_job, jobs)
    names = json.loads((work / "sweep-names.json").read_text())
    out: dict = {"names": names["names"], "families": names["families"], "maps": {},
                 "planted": {k: v for k, v in plant.items() if k != "s"}}
    for (coin, target, res), job in zip(got, jobs, strict=True):
        key = f"{coin} {target}" + (" planted" if job[3] is not None else "")
        out["maps"][key] = res
    # Benjamini-Hochberg at 5% over all real tests, per period
    real_keys = [k for k in out["maps"] if not k.endswith("planted")]
    for part in ("first", "last"):
        pv = np.concatenate([out["maps"][k][part]["p"] for k in real_keys])
        order = np.argsort(pv)
        m_ = len(pv)
        thr = 0.05 * (np.arange(1, m_ + 1)) / m_
        passed = pv[order] <= thr
        cutoff = pv[order][np.flatnonzero(passed).max()] if passed.any() else -1.0
        out[f"bh_cutoff_{part}"] = float(cutoff)
        for k in out["maps"]:
            out["maps"][k][part]["bh"] = [bool(p_ <= cutoff) for p_ in out["maps"][k][part]["p"]]
    for v in out["maps"].values():
        f, la = v["first"], v["last"]
        v["holds"] = [bool(f["bh"][j] and la["bh"][j] and np.sign(f["rank_ic"][j])
                           == np.sign(la["rank_ic"][j])) for j in range(len(f["p"]))]
    out["seconds"] = round(time.time() - t0)
    (work / "sweep-map.json").write_text(json.dumps(out))
    print("written", out["seconds"], "s", flush=True)


# ── step 2: first-period choice, model, last-third judgement ──────────


def _model_job(args):
    work, coin, target, planted = args
    import forecast_pilot as fp
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    mp_ = json.loads((work / "sweep-map.json").read_text())
    fams = mp_["families"]
    key = f"{coin} {target}" + (" planted" if planted is not None else "")
    ric1 = np.asarray(mp_["maps"][key]["first"]["rank_ic"])
    chosen = []
    for fam in dict.fromkeys(fams):
        cols = [j for j, f in enumerate(fams) if f == fam]
        chosen.append(max(cols, key=lambda j: abs(ric1[j])))
    z = np.load(work / f"sweep-{coin}.npz")
    S, tau, start = z["S"], z["tau"], int(z["start"][0])
    y = z[target].copy()
    if planted is not None:
        y = y + planted["c"] * planted["s"]
    h = horizon(target)
    X = S[:, chosen]
    kind = "cls" if target.startswith("sign") else "reg"
    make = f3._models()["logit" if kind == "cls" else "ridge"]
    p = f3._walk(*make, X, tau, start, h, y, center=(kind == "cls"))
    rev = S[:, mp_["names"].index("reversal_1h")]
    pr = parts(tau, h)
    dates = pd.to_datetime(tau, utc=True).tz_localize(None).to_numpy()

    def ric(q, mk):
        ok = mk & ~np.isnan(q) & ~np.isnan(y)
        return fc.spearman(q[ok], y[ok])

    res = {"chosen": [mp_["names"][j] for j in chosen]}
    null = []
    for k in range(DRAWS):
        rng = np.random.default_rng(SEED + 50_000 + 1000 * COINS.index(coin) + 10 * k
                                    + TARGETS.index(target))
        q = f3._walk(*make, X, tau, start, h, y, rng, center=(kind == "cls"))
        null.append({part: ric(q, mk) for part, mk in pr.items()})
    for part, mk in pr.items():
        d_ = np.array([n[part] for n in null])
        ok = mk & ~np.isnan(p) & ~np.isnan(y) & ~np.isnan(rev)
        pp, rr, yy = p[ok], rev[ok], y[ok]
        diff = fp._boot(dates[ok], lambda r, pp=pp, rr=rr, yy=yy: fc.spearman(pp[r], yy[r])
                        - fc.spearman(rr[r], yy[r]))
        res[part] = {"model": ric(p, mk), "null_q95": float(np.nanquantile(d_, 0.95)),
                     "pct": float(np.mean(d_ < ric(p, mk))), "reversal_1h": ric(rev, mk),
                     "model_minus_reversal": diff,
                     "hit": fc.hit_rate(pp, np.where(yy > (0.5 if kind == "cls" else 0.0),
                                                     1.0, -1.0))}
    return key, res


def run_model(work: Path, procs: int) -> None:
    t0 = time.time()
    plant = planted_spec(work)
    jobs = [(work, c, t, None) for c in COINS for t in TARGETS] + [(work, "BTC", "zw1h", plant)]
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_model_job, jobs)
    out = dict(got)
    out["seconds"] = round(time.time() - t0)
    (work / "sweep-model.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["features", "map", "model"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    {"features": lambda: make_features(a.work), "map": lambda: run_map(a.work, a.procs),
     "model": lambda: run_model(a.work, a.procs)}[a.phase]()


if __name__ == "__main__":
    main()
