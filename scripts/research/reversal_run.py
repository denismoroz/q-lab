"""The reversal detector's walk-forward: train on small-bar events, apply unchanged on large-bar
events, measure against simple rules, the large-bar model, a shuffled-label null and a planted
signal (docs/research/reversal-detector-protocol-2026-11-06.md). Called from
`reversal_detector.py run`. Writes DIR/reversal.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
PAIRS = ("0.5_1.5", "0.5_2.5", "1_3", "1_5")
LARGE = ("1h", "4h", "1d")
DAY = 86_400_000_000_000
BLOCK = 56 * DAY
RETRAINS = pd.date_range("2022-01-01", "2026-11-01", freq="6MS", tz="UTC")
FIRST0 = pd.Timestamp("2022-01-01", tz="UTC").value
CUT = pd.Timestamp("2024-06-12", tz="UTC").value
DRAWS = 50
CV_ROWS = 20_000
_W: dict = {}


def load(work: Path, tf: str, key: str, side: int, every: int = 1) -> dict:
    parts = []
    for i, c in enumerate(COINS):
        z = np.load(work / f"rev-{c}-{tf}.npz")
        s = z[f"{key}_side"] == side
        oh = np.zeros((int(s.sum()), len(COINS)), dtype=np.float32)
        oh[:, i] = 1
        d = {"X": np.column_stack([z[f"{key}_X"][s], oh]), "y": z[f"{key}_label"][s],
             "t": z[f"{key}_time"][s], "res": z[f"{key}_resolve"][s],
             "coin": np.full(int(s.sum()), i), "step_pct": z[f"{key}_step_pct"][s]}
        if every > 1:
            d = {k: v[::every] for k, v in d.items()}
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def shuffle_labels(D: dict, m: np.ndarray, rng) -> np.ndarray:
    from qlab import forecast_inputs as fi

    y = D["y"].copy()
    for i in range(len(COINS)):
        idx = np.flatnonzero(m & (D["coin"] == i))
        idx = idx[np.argsort(D["t"][idx], kind="stable")]
        if len(idx) > 10:
            y[idx] = D["y"][idx[fi.other_day_source(D["t"][idx], rng)]]
    return y


def _std(X, m):
    mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
    return lambda A: np.nan_to_num((A - mu) / sd)


def fit_predict(model, D, apply: dict, rng=None, fixed=None):
    """Walk-forward on the training events D; predictions for each applied event set; per
    retrain the chosen settings and the in-sample / block AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import TimeSeriesSplit

    from qlab import forecast as fc

    preds = {k: np.full(len(A["y"]), np.nan) for k, A in apply.items()}
    rows = []
    for r, (q0, q1) in enumerate(zip(RETRAINS[:-1], RETRAINS[1:], strict=True)):
        m = (D["res"] + DAY <= q0.value)
        if m.sum() < 2000:
            continue
        y = D["y"] if rng is None else shuffle_labels(D, m, rng)
        order = np.flatnonzero(m)[np.argsort(D["t"][m], kind="stable")]
        if model == "logit":
            tr = _std(D["X"], m)
            if fixed is None:
                sub = np.sort(np.random.default_rng(r).choice(order, size=min(CV_ROWS, len(order)),
                                                             replace=False))
                sub = sub[np.argsort(D["t"][sub], kind="stable")]
                cv = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), l1_ratios=[0.0, 1.0],
                                          cv=TimeSeriesSplit(3, gap=200), solver="saga",
                                          scoring="roc_auc", max_iter=200, tol=1e-3).fit(
                    tr(D["X"][sub]), y[sub])
                C, l1 = float(np.ravel(cv.C_)[0]), float(np.ravel(cv.l1_ratio_)[0])
            else:
                C, l1 = fixed[r]
            if l1 == 0.0:
                mod = LogisticRegression(C=C, max_iter=500)
            else:
                mod = LogisticRegression(C=C, l1_ratio=l1, solver="saga", max_iter=300,
                                         tol=1e-3)
            mod.fit(tr(D["X"][order]), y[order])
            setting = (C, l1)
            ins = fc.auc(mod.predict_proba(tr(D["X"][order]))[:, 1], y[order])
            blk_auc = float("nan")

            def pr(A, mod=mod, tr=tr):
                return mod.predict_proba(tr(A))[:, 1]
        else:
            if fixed is None:
                b0 = D["t"][order].max() - BLOCK
                blk = order[D["t"][order] >= b0]
                fit = order[D["res"][order] + DAY <= b0]
                mod = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=1000, learning_rate=0.05, min_samples_leaf=1000,
                    random_state=0, early_stopping=True, n_iter_no_change=10,
                    validation_fraction=None).fit(D["X"][fit], y[fit], X_val=D["X"][blk],
                                                  y_val=y[blk])
                setting = int(mod.n_iter_)
                blk_auc = fc.auc(mod.predict_proba(D["X"][blk])[:, 1], y[blk])
                fit_rows = fit
            else:
                setting = fixed[r]
                mod = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=max(1, setting), learning_rate=0.05,
                    min_samples_leaf=1000, random_state=0, early_stopping=False).fit(
                    D["X"][order], y[order])
                blk_auc, fit_rows = float("nan"), order
            ins = fc.auc(mod.predict_proba(D["X"][fit_rows])[:, 1], y[fit_rows])

            def pr(A, mod=mod):
                return mod.predict_proba(A)[:, 1]
        thr = float(np.quantile(pr(D["X"][order]), 0.8))
        for k, A in apply.items():
            sel = (A["t"] >= q0.value) & (A["t"] < q1.value)
            if sel.any():
                preds[k][sel] = pr(A["X"][sel])
        rows.append({"retrain": str(q0.date()), "setting": setting, "in_sample": ins,
                     "block": blk_auc, "threshold": thr, "train_events": int(m.sum())})
    return preds, rows


def measure(A, p, thr_by_time=None):
    from qlab import forecast as fc

    out = {}
    for part, mk in (("first", (A["t"] >= FIRST0) & (A["t"] < CUT)), ("last", A["t"] >= CUT)):
        ok = mk & ~np.isnan(p)
        out[part] = {"auc": fc.auc(p[ok], A["y"][ok]), "n": int(ok.sum()),
                     "base_rate": float(A["y"][ok].mean()) if ok.any() else float("nan")}
        if thr_by_time is not None:
            thr = thr_by_time[ok]
            flag = p[ok] > thr
            yy = A["y"][ok]
            out[part].update(precision=float(yy[flag].mean()) if flag.any() else float("nan"),
                             recall=float(flag[yy == 1].mean()) if (yy == 1).any() else
                             float("nan"), flagged=float(flag.mean()),
                             step_pct_flagged=float(np.nanmean(A["step_pct"][ok][flag]))
                             if flag.any() else float("nan"))
    return out


def thresholds(A, rows):
    thr = np.full(len(A["t"]), np.nan)
    for row in rows:
        q0 = pd.Timestamp(row["retrain"], tz="UTC")
        q1 = q0 + pd.DateOffset(months=6)
        sel = (A["t"] >= q0.value) & (A["t"] < q1.value)
        thr[sel] = row["threshold"]
    return thr


def bv2_rule(work: Path, A: dict, side: int) -> np.ndarray:
    """Bv2's own 14/30-day hedge wish in force at the event: 1 for a «top» when the hedge is
    wanted, for a «bottom» when it is not."""
    score = np.full(len(A["t"]), np.nan)
    for i, c in enumerate(COINS):
        w = _W["wish"][c]
        sel = A["coin"] == i
        pos = np.searchsorted(_W["close_ns"][c], A["t"][sel], side="right") - 1
        ok = pos >= 0
        v = np.full(int(sel.sum()), np.nan)
        v[ok] = w[pos[ok]]
        score[sel] = v if side == 1 else 1 - v
    return score


def _job(args):
    work, key, side, model, train_tf = args
    t0 = time.time()
    D = load(work, train_tf, key, side, every=3 if train_tf == "5m" else 1)
    apply = {tf: load(work, tf, key, side) for tf in ("15m",) + LARGE}
    preds, rows = fit_predict(model, D, apply)
    res = {"key": key, "side": side, "model": model, "train_tf": train_tf, "per_retrain": rows,
           "train_events": int(len(D["y"])), "tfs": {}}
    for tf, A in apply.items():
        res["tfs"][tf] = measure(A, preds[tf], thresholds(A, rows))
    res["seconds"] = round(time.time() - t0)
    return res


def _direct_job(args):
    work, key, side, model, tf = args
    D = load(work, tf, key, side)
    preds, rows = fit_predict(model, D, {tf: D})
    return {"key": key, "side": side, "model": model, "tf": tf,
            "metrics": measure(D, preds[tf], thresholds(D, rows)),
            "in_sample": float(np.nanmean([r["in_sample"] for r in rows]))}


def _null_job(args):
    work, key, side, model, k, fixed, planted = args
    D = load(work, "15m", key, side)
    apply = {tf: load(work, tf, key, side) for tf in LARGE}
    if planted is not None:
        for A in [D, *apply.values()]:
            A["y"] = planted_labels(A, planted)
    rng = None if k < 0 else np.random.default_rng(800_000 + 1000 * k + side)
    preds, _ = fit_predict(model, D, apply, rng=rng, fixed=None if planted else fixed)
    out = {tf: measure(A, preds[tf]) for tf, A in apply.items()}
    if planted is not None:
        from qlab import forecast as fc

        for tf, A in apply.items():
            s = A["X"][:, planted["column"]]
            ok = ~np.isnan(s)
            out[tf]["oracle_auc"] = fc.auc(s[ok], A["y"][ok])
    return k, out


def planted_labels(A, planted):
    s = A["X"][:, planted["column"]].astype(float)
    s = np.nan_to_num((s - planted["mu"]) / planted["sd"])
    p = 1 / (1 + np.exp(-(planted["a"] + planted["beta"] * s)))
    rng = np.random.default_rng(int(A["t"][0] % 1_000_000))
    return (rng.random(len(p)) < p).astype(float)


def run(work: Path, procs: int) -> None:
    import forecast_pilot2  # noqa: F401  (loads the qlab path)

    from qlab import hedge_followup as hf
    from qlab import hedge_study as hs

    t0 = time.time()
    wish, close_ns = {}, {}
    for c in COINS:
        sp = hs.load_panel([c], "2019-09-08", "2026-10-30", Path("/Users/d/prj/q-lab/data/raw"))
        px = sp.panel.prices[c]
        wish[c] = hf.band_wish(px.to_numpy(dtype=float))
        close_ns[c] = (px.index + pd.Timedelta(hours=1)).as_unit("ns").asi8
    _W.update(wish=wish, close_ns=close_ns)
    ctx = mp.get_context("fork")
    jobs = [(work, k, s, m, "15m") for k in PAIRS for s in (1, -1) for m in ("logit", "gbm")]
    jobs += [(work, k, s, "gbm", "5m") for k in PAIRS for s in (1, -1)]
    with ctx.Pool(procs) as pool:
        main = pool.map(_job, jobs)
    print("main", len(main), round(time.time() - t0), "s", flush=True)
    # simple rules on the large bars
    rules = {}
    for k in PAIRS:
        for s in (1, -1):
            for tf in LARGE:
                A = load(work, tf, k, s)
                rules[f"{k} {s} {tf}"] = {"step_pct": measure(A, A["step_pct"]),
                                          "bv2": measure(A, bv2_rule(work, A, s))}
    direct_jobs = [(work, k, s, m, tf) for k in PAIRS for s in (1, -1) for m in ("logit", "gbm")
                   for tf in LARGE]
    with ctx.Pool(procs) as pool:
        direct = pool.map(_direct_job, direct_jobs)
    print("direct", round(time.time() - t0), "s", flush=True)
    # the first-period choice per side: the 15-minute-trained config with the best mean
    # first-period AUC over the large bars
    chosen = {}
    for s in (1, -1):
        cands = [r for r in main if r["side"] == s and r["train_tf"] == "15m"]
        best = max(cands, key=lambda r: np.nanmean([r["tfs"][tf]["first"]["auc"]
                                                    for tf in LARGE]))
        chosen[s] = best
    null_jobs = []
    for s, r in chosen.items():
        fixed = [row["setting"] for row in r["per_retrain"]]
        null_jobs += [(work, r["key"], s, r["model"], k, fixed, None) for k in range(DRAWS)]
    # the planted signal: labels from a known dependence on rsi_4 (AUC about 0.6 by design)
    from qlab import reversal as rv

    col = list(rv.FEATURE_NAMES).index("rsi_4")
    D0 = load(work, "15m", chosen[1]["key"], 1)
    sx = D0["X"][:, col].astype(float)
    planted = {"column": col, "name": "rsi_4", "mu": float(np.nanmean(sx)),
               "sd": float(np.nanstd(sx)), "a": float(np.log(D0["y"].mean()
                                                            / (1 - D0["y"].mean()))),
               "beta": 0.6}
    null_jobs.append((work, chosen[1]["key"], 1, chosen[1]["model"], -1, None, planted))
    with ctx.Pool(procs) as pool:
        nulls = pool.map(_null_job, null_jobs)
    print("nulls", round(time.time() - t0), "s", flush=True)
    null_by_side = {1: [], -1: []}
    for (job, (_k, out)) in zip(null_jobs, nulls, strict=True):
        if job[6] is None:
            null_by_side[job[2]].append(out)
    planted_out = nulls[-1][1]
    verdict = {}
    for s, r in chosen.items():
        row = {"config": f"{r['key']} {r['model']}"}
        for tf in LARGE:
            cell = {}
            for part in ("first", "last"):
                auc = r["tfs"][tf][part]["auc"]
                d = np.array([n[tf][part]["auc"] for n in null_by_side[s]])
                rl = rules[f"{r['key']} {s} {tf}"]
                cell[part] = {"auc": auc, "null_q95": float(np.nanquantile(d, 0.95)),
                              "pct": float(np.mean(d < auc)),
                              "rule_step_pct": rl["step_pct"][part]["auc"],
                              "rule_bv2": rl["bv2"][part]["auc"]}
                cell[part]["pass"] = bool(auc > cell[part]["null_q95"]
                                          and auc > max(0.5, cell[part]["rule_step_pct"],
                                                        cell[part]["rule_bv2"]))
            cell["pass"] = cell["first"]["pass"] and cell["last"]["pass"]
            row[tf] = cell
        verdict[str(s)] = row
    out = {"main": main, "rules": rules, "direct": direct, "verdict": verdict,
           "planted": {k: v for k, v in planted.items()} | {"result": planted_out},
           "trials": {"main": len(main) * 4, "direct": len(direct),
                      "null_draws": sum(1 for j in null_jobs if j[6] is None)},
           "seconds": round(time.time() - t0)}
    (work / "reversal.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s", flush=True)
