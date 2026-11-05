"""One-coin forecast pilot: BTC, demeaned log-return targets and fall/rise classes
(docs/research/forecast-pilot-protocol-2026-11-05.md and the classes addendum).

    uv run python scripts/research/forecast_pilot.py --work DIR [--minutes DIR_OF_1M]

Needs DIR/ml/BTC.pkl from `ml_money.py prepare`. Writes DIR/forecast.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qlab import forecast as fc
from qlab import hedge_followup as hf
from qlab import ml_money as mm

H = (1, 4, 24, 168)
KS = (0.5, 1.0)
CUT = pd.Timestamp("2024-06-12", tz="UTC").value
START = pd.Timestamp("2022-01-01", tz="UTC")
HOUR = 3_600_000_000_000
DAY = 24 * HOUR


def _boot(dates, stat):
    return hf.block_bootstrap(dates, stat, block_days=30, draws=300)


def prepare_data(work: Path, minutes: Path, features: str) -> dict:
    """The decision grid, features, targets and classes of one feature set (the names of
    the features in `names`)."""
    with open(work / "ml" / "BTC.pkl", "rb") as fh:
        d = pickle.load(fh)
    g = d["g60"]
    close = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in
                       sorted((minutes / "futures" / "BTC").glob("*.parquet"))])
    close = close.drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
    tau, pos = g["tau"], g["pos"]
    logc = np.where(g["ok"], np.log(close.to_numpy()[np.clip(pos, 0, None)]), np.nan)
    X = np.column_stack([g["static"], g["summ"]]).astype(np.float64)
    if features != "base":
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        import waves as wr

        if features.startswith("ema"):
            extra = fc.ema_features(np.exp(logc))
        else:  # the standard set on hourly bars built from the minutes
            m = pd.concat([pd.read_parquet(f, columns=["open_time", "high", "low", "close",
                                                       "volume"])
                           for f in sorted((minutes / "futures" / "BTC").glob("*.parquet"))])
            m = m.drop_duplicates("open_time").set_index("open_time").sort_index()
            hb = m.groupby(m.index.floor("h")).agg(
                {"high": "max", "low": "min", "close": "last", "volume": "sum"})
            hb.index = pd.DatetimeIndex(hb.index).as_unit("ns")
            labels_h = pd.to_datetime(tau - HOUR, utc=True)  # the hour that closes at tau
            hb = hb.reindex(pd.DatetimeIndex(labels_h).as_unit("ns"))
            extra = fc.standard_features(hb)
        cols = [wr.FEATURES.index(f) for f in ("ret1", "ret7", "ret30", "btc_ret7")]
        if features in ("ema", "std"):
            X = np.column_stack([X, extra])
        else:  # the indicators and the longer returns only
            X = np.column_stack([extra, g["static"][:, cols]]).astype(np.float64)
    dates = pd.to_datetime(tau, utc=True).tz_localize(None).to_numpy()
    targets, pasts = {}, {}
    for h in H:
        targets[h], pasts[h] = fc.z_target(logc, h)
    labels = {}
    for h in H:
        for k in KS:
            z = targets[h]
            labels[(h, "fall", k)] = np.where(np.isnan(z), np.nan, (z < -k).astype(float))
            labels[(h, "rise", k)] = np.where(np.isnan(z), np.nan, (z > k).astype(float))
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import waves as wr

    base_names = list(wr.FEATURES) + ["min_ret", "min_range", "min_volume", "min_imbalance",
                                      "min_trades"]
    long_names = ["ret1", "ret7", "ret30", "btc_ret7"]
    names = {"base": base_names, "ema": base_names + list(fc.STANDARD_NAMES[:13]),
             "ema_only": list(fc.STANDARD_NAMES[:13]) + long_names,
             "std": base_names + list(fc.STANDARD_NAMES),
             "std_only": list(fc.STANDARD_NAMES) + long_names}[features]
    return {"d": d, "tau": tau, "pos": pos, "X": X, "dates": dates, "targets": targets,
            "pasts": pasts, "labels": labels, "names": names}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    ap.add_argument("--features", choices=["base", "ema", "ema_only", "std", "std_only"],
                    default="base")
    a = ap.parse_args()
    t0 = time.time()
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    D = prepare_data(a.work, a.minutes, a.features)
    d, tau, pos, X, dates = D["d"], D["tau"], D["pos"], D["X"], D["dates"]
    targets, pasts, labels = D["targets"], D["pasts"], D["labels"]
    retrains = pd.date_range(START, pd.Timestamp("2026-11-01", tz="UTC"), freq="6MS")
    preds: dict = {}
    rng = np.random.default_rng(5)

    def train_mask(h, t_end, y):
        return (tau + h * HOUR + (h * HOUR + DAY) <= t_end) & ~np.isnan(y) & (
            tau >= d["start"])

    def shuffled(y, mask):
        """Labels permuted in blocks of one day inside the training rows."""
        idx = np.flatnonzero(mask)
        day = tau[idx] // DAY
        blocks = [idx[day == k] for k in np.unique(day)]
        order = rng.permutation(len(blocks))
        vals = np.concatenate([y[blocks[k]] for k in order])
        out = y.copy()
        out[idx] = vals[: len(idx)]
        return out

    def put(key, sel, values):
        preds.setdefault(key, np.full(len(tau), np.nan))[sel] = values

    # ── tabular models ──
    for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
        t_end = q0.value
        sel = (tau >= q0.value) & (tau < q1.value)
        for h in H:
            y = targets[h]
            m = train_mask(h, t_end, y)
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            xs = np.nan_to_num((X - mu) / sd)
            for alpha in (1, 10, 100):
                put(("ridge", h, alpha), sel, Ridge(alpha=alpha).fit(xs[m], y[m]).predict(xs[sel]))
            for leaf in (200, 1000):
                gb = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                   min_samples_leaf=leaf, random_state=0)
                put(("gbm", h, leaf), sel, gb.fit(X[m], y[m]).predict(X[sel]))
            ys = shuffled(y, m)
            put(("ridge shuffled", h, 10), sel,
                Ridge(alpha=10).fit(xs[m], ys[m]).predict(xs[sel]))
            gb = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=1000, random_state=0)
            put(("gbm shuffled", h, 1000), sel, gb.fit(X[m], ys[m]).predict(X[sel]))
            for side in ("fall", "rise"):
                for k in KS:
                    lab = labels[(h, side, k)]
                    mk = train_mask(h, t_end, lab)
                    if np.nansum(lab[mk]) < 50:
                        continue
                    for C in (0.1, 1.0):
                        lr = LogisticRegression(C=C, max_iter=500).fit(xs[mk], lab[mk])
                        put(("logit", h, side, k, C), sel, lr.predict_proba(xs[sel])[:, 1])
                    for leaf in (200, 1000):
                        gc = HistGradientBoostingClassifier(
                            max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=leaf,
                            random_state=0).fit(X[mk], lab[mk])
                        put(("gbc", h, side, k, leaf), sel, gc.predict_proba(X[sel])[:, 1])
                    ls = shuffled(lab, mk)
                    put(("logit shuffled", h, side, k, 1.0), sel, LogisticRegression(
                        C=1.0, max_iter=500).fit(xs[mk], ls[mk]).predict_proba(xs[sel])[:, 1])
                    put(("gbc shuffled", h, side, k, 1000), sel, HistGradientBoostingClassifier(
                        max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=1000,
                        random_state=0).fit(X[mk], ls[mk]).predict_proba(X[sel])[:, 1])
        print("tabular", q0.date(), round(time.time() - t0), "s", flush=True)

    # ── TCN: one net for the 4 regressions, one for the 16 classes; 3 seeds ──
    keys_c = [(h, s, k) for h in H for s in ("fall", "rise") for k in KS]
    torch.set_num_threads(8)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        srng = np.random.default_rng(seed)
        for kind in ("reg", "cls"):
            n_out = len(H) if kind == "reg" else len(keys_c)
            net = fc.ForecastTCN(X.shape[1], n_out)
            for q0, q1 in zip(retrains[:-1], retrains[1:], strict=True):
                t_end = q0.value
                Y = np.column_stack([targets[h] for h in H] if kind == "reg"
                                    else [labels[k] for k in keys_c])
                # a row trains an output only if that output's target ended in time
                horizons = np.array(list(H) if kind == "reg" else [k[0] for k in keys_c])
                ok = (tau[:, None] + horizons[None, :] * HOUR + (horizons[None, :] * HOUR + DAY)
                      <= t_end) & ~np.isnan(Y) & (tau[:, None] >= d["start"])
                rows = np.flatnonzero(ok.any(axis=1))[::4]
                mu, sd = np.nanmean(X[rows], 0), np.nanstd(X[rows], 0) + 1e-9
                xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                for _ in range(2):
                    order = srng.permutation(rows)
                    for b in range(0, len(order), 256):
                        r = order[b:b + 256]
                        w = torch.from_numpy(mm.windows(d["ch"], np.clip(pos[r], 0, None)))
                        out = net(w, torch.from_numpy(xs[r]))
                        yt = torch.from_numpy(np.nan_to_num(Y[r]).astype(np.float32))
                        mt = torch.from_numpy(ok[r].astype(np.float32))
                        if kind == "reg":
                            loss = ((out - yt) ** 2 * mt).sum() / mt.sum().clamp(min=1)
                        else:
                            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                                out, yt, reduction="none") * mt).sum() / mt.sum().clamp(min=1)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                sel = np.flatnonzero((tau >= q0.value) & (tau < q1.value))
                with torch.no_grad():
                    outs = []
                    for b in range(0, len(sel), 2048):
                        r = sel[b:b + 2048]
                        w = torch.from_numpy(mm.windows(d["ch"], np.clip(pos[r], 0, None)))
                        o = net(w, torch.from_numpy(xs[r]))
                        outs.append(o.numpy() if kind == "reg" else torch.sigmoid(o).numpy())
                    o = np.vstack(outs) if outs else np.zeros((0, n_out))
                for j in range(n_out):
                    key = (("tcn", H[j], seed) if kind == "reg"
                           else ("tcn cls", *keys_c[j], seed))
                    put(key, sel, o[:, j])
        print("tcn seed", seed, round(time.time() - t0), "s", flush=True)

    # ── measures ──
    def parts(h):
        emb = h * HOUR + DAY
        first = (tau >= START.value) & (tau < CUT)
        last = tau >= CUT + emb
        return {"first": first, "last": last}

    res = {"regression": {}, "classes": {}, "seconds": None}
    for h in H:
        y = targets[h]
        rows = {}
        cands = {f"ridge a={a_}": preds[("ridge", h, a_)] for a_ in (1, 10, 100)}
        cands.update({f"gbm leaf={lf}": preds[("gbm", h, lf)] for lf in (200, 1000)})
        tcn = np.nanmean(np.stack([preds[("tcn", h, s)] for s in (0, 1, 2)]), 0)
        cands["tcn (3 seeds)"] = tcn
        cands["shuffled ridge"] = preds[("ridge shuffled", h, 10)]
        cands["shuffled gbm"] = preds[("gbm shuffled", h, 1000)]
        cands["previous-period return"] = pasts[h]
        pr = parts(h)
        for name, p in cands.items():
            row = {}
            for part, sel in pr.items():
                ok = sel & ~np.isnan(p) & ~np.isnan(y)
                pp, yy, dd = p[ok], y[ok], dates[ok]
                row[part] = {
                    "n": int(ok.sum()),
                    "ic": _boot(dd, lambda r, pp=pp, yy=yy: fc.pearson(pp[r], yy[r])),
                    "rank_ic": _boot(dd, lambda r, pp=pp, yy=yy: fc.spearman(pp[r], yy[r])),
                    "hit": _boot(dd, lambda r, pp=pp, yy=yy: fc.hit_rate(pp[r], yy[r])),
                    "r2": _boot(dd, lambda r, pp=pp, yy=yy: fc.r2_vs_zero(pp[r], yy[r])),
                }
            rows[name] = row
        seeds_ic = [fc.pearson(preds[("tcn", h, s)][pr["last"]], y[pr["last"]]) for s in (0, 1, 2)]
        rows["tcn (3 seeds)"]["seed_ic_last"] = seeds_ic
        seeds_ic1 = [fc.pearson(preds[("tcn", h, s)][pr["first"]], y[pr["first"]])
                     for s in (0, 1, 2)]
        rows["tcn (3 seeds)"]["seed_ic_first"] = seeds_ic1
        # the choice per family on the first period
        rows["chosen"] = {fam: max([n for n in rows if n.startswith(fam)],
                                   key=lambda n: rows[n]["first"]["ic"][0])
                          for fam in ("ridge", "gbm")}
        res["regression"][h] = rows
        print("regression", h, round(time.time() - t0), "s", flush=True)
    for h in H:
        for side in ("fall", "rise"):
            for k in KS:
                lab = labels[(h, side, k)]
                pr = parts(h)
                if ("logit", h, side, k, 1.0) not in preds:
                    continue
                cands = {f"logit C={c}": preds[("logit", h, side, k, c)] for c in (0.1, 1.0)}
                cands.update({f"gbm leaf={lf}": preds[("gbc", h, side, k, lf)]
                              for lf in (200, 1000)})
                cands["tcn (3 seeds)"] = np.nanmean(np.stack(
                    [preds[("tcn cls", h, side, k, s)] for s in (0, 1, 2)]), 0)
                cands["shuffled logit"] = preds[("logit shuffled", h, side, k, 1.0)]
                cands["shuffled gbm"] = preds[("gbc shuffled", h, side, k, 1000)]
                past = pasts[h]
                # previous-period return as a score, its sign chosen on the first period
                s1 = fc.auc(past[pr["first"]], lab[pr["first"]])
                cands["previous-period return"] = past if s1 >= 0.5 else -past
                rows = {}
                for name, p in cands.items():
                    row = {}
                    base_train = np.nanmean(lab[(tau < START.value) & (tau >= d["start"])])
                    # operating point: flag the same share of hours as the event's base rate
                    # in the training years (before 2022)
                    thr = np.nanquantile(p[pr["first"]], 1 - base_train)
                    for part, sel in pr.items():
                        ok = sel & ~np.isnan(p) & ~np.isnan(lab)
                        pp, yy, dd = p[ok], lab[ok], dates[ok]
                        flag = pp > thr
                        tp = (flag & (yy == 1)).sum()
                        cal = []
                        if name not in ("previous-period return",):
                            edges = np.quantile(pp, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                                m_ = (pp >= lo) & (pp <= hi)
                                cal.append((float(pp[m_].mean()), float(yy[m_].mean())))
                        row[part] = {
                            "n": int(ok.sum()), "base_rate": float(yy.mean()),
                            "auc": _boot(dd, lambda r, pp=pp, yy=yy: fc.auc(pp[r], yy[r])),
                            "precision": float(tp / max(1, flag.sum())),
                            "recall": float(tp / max(1, (yy == 1).sum())),
                            "calibration": cal}
                    rows[name] = row
                rows["chosen"] = {fam: max([n for n in rows if n.startswith(fam)],
                                           key=lambda n: rows[n]["first"]["auc"][0])
                                  for fam in ("logit", "gbm")}
                res["classes"][f"{h}h {side} k={k}"] = rows
        print("classes", h, round(time.time() - t0), "s", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["features"] = a.features
    name = "forecast.json" if a.features == "base" else f"forecast-{a.features}.json"
    (a.work / name).write_text(json.dumps(res, indent=1, default=str))
    print("written", res["seconds"], "s")


if __name__ == "__main__":
    main()
