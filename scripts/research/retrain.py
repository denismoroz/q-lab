"""Retraining more often, on shorter windows, and more complex nets
(docs/research/retrain-protocol-2026-11-07.md). Run with `nice`.

    nice -n 15 uv run python scripts/research/retrain.py data --work DIR
    nice -n 15 uv run python scripts/research/retrain.py tab --work DIR [--procs 10]
    nice -n 15 uv run python scripts/research/retrain.py nets --work DIR [--procs 10]

Writes DIR/rt-data.npz, DIR/rt-seq.npy, DIR/retrain-tab.json, DIR/retrain-nets.json.
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

COINS = ("BTC", "ETH", "SOL", "AVAX")
TARGETS = ("sign15m", "sign1h", "sign4h", "zw15m", "zw1h", "zw4h")
HOURS = {"15m": 0.25, "1h": 1, "4h": 4}
HOUR = 3_600_000_000_000
DAY = 24 * HOUR
WEEK = 7 * DAY
BLOCK = 8 * WEEK
FIRST0 = pd.Timestamp("2022-01-01", tz="UTC").value
CUT = pd.Timestamp("2024-06-12", tz="UTC").value
END = "2026-11-01"
SCHEDULES = {"6M": pd.date_range("2022-01-01", END, freq="6MS", tz="UTC"),
             "1M": pd.date_range("2022-01-01", END, freq="MS", tz="UTC"),
             "1W": pd.date_range("2022-01-03", END, freq="W-MON", tz="UTC")}
WINDOWS = {"expanding": None, "12m": 365 * DAY, "6m": 182 * DAY, "3m": 91 * DAY}
DRAWS = 50
NET_DRAWS = 20  # owner, 2026-11-07: cores free today; declared 10 raised before the nets ran
_D: dict = {}


def horizon(t: str) -> float:
    return HOURS[t.replace("sign", "").replace("zw", "")]


# ── data ────────────────────────────────────────────────────────────────


def make_data(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    t0 = time.time()
    built = {c: fp2.build(work, f3.MINUTES, "c", c) for c in COINS}
    names = list(built["ETH"]["names"])  # ETH drops no column; BTC's btc_ret7 is its ret7
    X, seq, tau, coin, start_ok = [], [], [], [], []
    tg = {t: [] for t in TARGETS}
    for i, c in enumerate(COINS):
        V = built[c]
        z = np.load(work / f"sweep-{c}.npz")
        assert np.array_equal(z["tau"], V["tau"]), c
        cols = []
        for n in names:
            src = n if n in V["names"] else ("ret7" if n == "btc_ret7" else None)
            cols.append(V["X"][:, V["names"].index(src)] if src else np.full(len(V["tau"]),
                                                                             np.nan))
        oh = np.zeros((len(V["tau"]), len(COINS)))
        oh[:, i] = 1
        X.append(np.column_stack([z["S"], np.column_stack(cols), oh]).astype(np.float32))
        seq.append(V["seq"].astype(np.float32))
        tau.append(V["tau"])
        coin.append(np.full(len(V["tau"]), i))
        start_ok.append(V["tau"] >= int(z["start"][0]))
        for t in TARGETS:
            tg[t].append(z[t])
    np.save(work / "rt-seq.npy", np.concatenate(seq))
    np.savez_compressed(work / "rt-data.npz", X=np.vstack(X), tau=np.concatenate(tau),
                        coin=np.concatenate(coin), start_ok=np.concatenate(start_ok),
                        **{t: np.concatenate(v) for t, v in tg.items()})
    print("data", np.vstack(X).shape, round(time.time() - t0), "s", flush=True)


def _load(work: Path, seq: bool = False) -> None:
    z = np.load(work / "rt-data.npz")
    _D.update({k: z[k] for k in z.files})
    if seq:
        _D["seq"] = np.load(work / "rt-seq.npy", mmap_mode="r")


def shuffled(y: np.ndarray, m: np.ndarray, rng) -> np.ndarray:
    from qlab import forecast_inputs as fi

    out = y.copy()
    for i in range(len(COINS)):
        idx = np.flatnonzero(m & (_D["coin"] == i))
        idx = idx[np.argsort(_D["tau"][idx], kind="stable")]
        if len(idx) > 50:
            out[idx] = y[idx[fi.other_day_source(_D["tau"][idx], rng)]]
    return out


def parts(h: float) -> dict:
    tau = _D["tau"]
    return {"first": (tau >= FIRST0) & (tau < CUT), "last": tau >= CUT + h * HOUR + DAY}


def measure(p: np.ndarray, target: str) -> dict:
    from qlab import forecast as fc

    y = _D[target]
    h = horizon(target)
    sgn = np.where(y > (0.5 if target.startswith("sign") else 0.0), 1.0, -1.0)
    out = {}
    for part, mk in parts(h).items():
        row = {}
        for i, c in enumerate(["pooled", *COINS]):
            sel = mk & ~np.isnan(p) & ~np.isnan(y)
            if c != "pooled":
                sel = sel & (_D["coin"] == i - 1)
            row[c] = {"rank_ic": fc.spearman(p[sel], y[sel]),
                      "hit": fc.hit_rate(p[sel], sgn[sel])}
        out[part] = row
    return out


# ── linear and boosting ────────────────────────────────────────────────


def walk_tab(model: str, target: str, sched: str, window: str, rng=None, fixed_y=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    X, tau, ok0 = _D["X"], _D["tau"], _D["start_ok"]
    y0 = _D[target] if fixed_y is None else fixed_y
    h = horizon(target)
    gap = h * HOUR + (h * HOUR + DAY)
    dates = SCHEDULES[sched]
    win = WINDOWS[window]
    p = np.full(len(tau), np.nan)
    gaps = []
    logit = LogisticRegression(C=0.1, max_iter=300, warm_start=True)
    for q0, q1 in zip(dates[:-1], dates[1:], strict=True):
        m = (tau + gap <= q0.value) & ok0 & ~np.isnan(y0)
        if win is not None:
            m &= tau >= q0.value - win
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any() or m.sum() < 500:
            continue
        y = y0 if rng is None else shuffled(y0, m, rng)
        if model in ("logit", "ridge"):
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            xs = np.nan_to_num((X[m] - mu) / sd)
            xp = np.nan_to_num((X[sel] - mu) / sd)
            if model == "ridge":
                mod = Ridge(alpha=10).fit(xs, y[m])
                p[sel] = mod.predict(xp)
            else:
                mod = logit.fit(xs, y[m])
                p[sel] = mod.predict_proba(xp)[:, 1] - y[m].mean()
        else:
            b0 = q0.value - gap - BLOCK
            blk, fit = m & (tau >= b0), m & (tau + gap <= b0)
            if blk.sum() < 200 or fit.sum() < 500:
                continue
            mod = HistGradientBoostingRegressor(
                max_depth=3, max_iter=1000, learning_rate=0.05, min_samples_leaf=1000,
                random_state=0, early_stopping=True, n_iter_no_change=10,
                validation_fraction=None).fit(X[fit], y[fit], X_val=X[blk], y_val=y[blk])
            p[sel] = mod.predict(X[sel])
            gaps.append(int(mod.n_iter_))
    return p, gaps


def _tab_job(args):
    work, model, target, sched, window, k, planted = args
    if not _D:
        _load(work)
    y = None
    if planted is not None:
        y = _D[target] + planted["c"] * planted["s"]
    rng = None if k < 0 else np.random.default_rng(1_100_000 + 1000 * k + TARGETS.index(target))
    p, trees = walk_tab(model, target, sched, window, rng, fixed_y=y)
    if planted is not None:
        saved = _D[target]
        _D[target] = y
        res = measure(p, target)
        _D[target] = saved
    else:
        res = measure(p, target)
    return {"model": model, "target": target, "sched": sched, "window": window, "draw": k,
            "planted": planted is not None, "trees": trees, "m": res}


def run_tab(work: Path, procs: int) -> None:
    t0 = time.time()
    ctx = mp.get_context("spawn")
    jobs = []
    for t in TARGETS:
        lin = "logit" if t.startswith("sign") else "ridge"
        for s in SCHEDULES:
            for w in WINDOWS:
                jobs.append((work, lin, t, s, w, -1, None))
                if s != "1W":
                    jobs.append((work, "gbm", t, s, w, -1, None))
    with ctx.Pool(procs) as pool:
        grid = pool.map(_tab_job, jobs)
    print("grid", len(jobs), round(time.time() - t0), "s", flush=True)
    chosen = {}
    for t in TARGETS:
        for fam in ("linear", "gbm"):
            cands = [g for g in grid if g["target"] == t
                     and ((g["model"] == "gbm") == (fam == "gbm"))]
            chosen[f"{t} {fam}"] = max(cands, key=lambda g: g["m"]["first"]["pooled"]["rank_ic"])
    _load(work)
    from scipy.stats import spearmanr

    names_s = json.loads((work / "sweep-names.json").read_text())["names"]
    j = int(np.random.default_rng(20261107).integers(len(names_s)))
    s = _D["X"][:, j].astype(float)
    s = np.nan_to_num((s - np.nanmean(s)) / np.nanstd(s))
    y = _D["zw1h"]
    ok = ~np.isnan(y)
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if spearmanr(s[ok], (y + mid * s)[ok]).statistic < 0.05:
            lo = mid
        else:
            hi = mid
    planted = {"column": j, "name": names_s[j], "c": (lo + hi) / 2, "s": s}
    jobs2 = []
    for g in chosen.values():
        jobs2 += [(work, g["model"], g["target"], g["sched"], g["window"], k, None)
                  for k in range(DRAWS)]
    pz = chosen["zw1h linear"]
    jobs2.append((work, pz["model"], "zw1h", pz["sched"], pz["window"], -1, planted))
    with ctx.Pool(procs) as pool:
        rest = pool.map(_tab_job, jobs2)
    print("nulls", len(jobs2), round(time.time() - t0), "s", flush=True)
    out = {"grid": grid, "chosen": {k: {kk: g[kk] for kk in ("model", "target", "sched",
                                                             "window")}
                                    for k, g in chosen.items()},
           "null": [r for r in rest if not r["planted"]],
           "planted": {"column": j, "name": names_s[j], "c": planted["c"],
                       "result": [r for r in rest if r["planted"]][0]["m"]},
           "seconds": round(time.time() - t0)}
    (work / "retrain-tab.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s", flush=True)


# ── nets ────────────────────────────────────────────────────────────────


def make_net(arch: str, n_static: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class Conv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers, c_in = [], 5
            for d in (1, 2, 4, 8, 16):
                layers += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(c_in, 64, 3, dilation=d),
                           nn.ReLU(), nn.Dropout(0.2)]
                c_in = 64
            self.enc = nn.Sequential(*layers)

        def forward(self, w):
            return self.enc(w)[:, :, -1]

    class GRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(5, 160, batch_first=True)
            self.drop = nn.Dropout(0.2)

        def forward(self, w):
            out, _ = self.rnn(w.transpose(1, 2))
            return self.drop(out[:, -1])

    class Trans(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(5, 96)
            self.pos = nn.Parameter(torch.zeros(1, 96, 96))
            layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=0.2, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(96))

        def forward(self, w):
            x = self.inp(w.transpose(1, 2)) + self.pos
            return self.enc(x, mask=self.mask, is_causal=True)[:, -1]

    enc, width = {"conv": (Conv, 64), "gru": (GRU, 160), "transformer": (Trans, 96)}[arch]

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = enc()
            self.static = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(0.2))
            self.head = nn.Sequential(nn.Linear(width + 64, 64), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(64, len(TARGETS)))

        def forward(self, w, s):
            return self.head(torch.cat([self.seq(w), self.static(s)], -1))

    return Net()


def _net_job(args):
    work, arch, seed, k = args
    import torch

    torch.set_num_threads(1)
    if not _D:
        _load(work, seq=True)
    t0 = time.time()
    X, tau, seq = _D["X"], _D["tau"], _D["seq"]
    Y = np.column_stack([np.where(np.isnan(_D[t]), np.nan, 2 * _D[t] - 1)
                         if t.startswith("sign") else _D[t] for t in TARGETS]).astype(np.float32)
    gap = 4 * HOUR + (4 * HOUR + DAY)
    ok0 = _D["start_ok"] & ~np.isnan(Y).all(1)
    rng = None if k < 0 else np.random.default_rng(1_200_000 + 1000 * k)
    pA = np.full(Y.shape, np.nan, dtype=np.float32)
    pB = np.full(Y.shape, np.nan, dtype=np.float32)
    curves = []
    g = np.random.default_rng(seed)

    def batches(net, rows, Yt, opt, epochs, xs):
        for _ in range(epochs):
            net.train()
            order = g.permutation(rows)
            for b in range(0, len(order), 512):
                r = np.sort(order[b:b + 512])
                yt = torch.from_numpy(np.nan_to_num(Yt[r]))
                mt = torch.from_numpy((~np.isnan(Yt[r])).astype(np.float32))
                out = net(torch.from_numpy(np.asarray(seq[r])), torch.from_numpy(xs[r]))
                loss = ((out - yt) ** 2 * mt).sum() / mt.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(net, rows, xs):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(rows), 4096):
                r = rows[b:b + 4096]
                out.append(net(torch.from_numpy(np.asarray(seq[r])),
                               torch.from_numpy(xs[r])).numpy())
        return np.vstack(out) if out else np.zeros((0, Y.shape[1]), dtype=np.float32)

    def loss_on(net, rows, Yt, xs):
        pr, yy = predict(net, rows, xs), Yt[rows]
        okk = ~np.isnan(yy)
        return float(np.mean((pr[okk] - yy[okk]) ** 2))

    halves = SCHEDULES["6M"]
    weeks = SCHEDULES["1W"]
    for q0, q1 in zip(halves[:-1], halves[1:], strict=True):
        m = (tau + gap <= q0.value) & ok0
        Yt = Y.copy()
        if rng is not None:
            for j in range(Y.shape[1]):
                Yt[:, j] = shuffled(Y[:, j], m & ~np.isnan(Y[:, j]), rng)
        b0 = q0.value - gap - BLOCK
        fit = np.flatnonzero(m & (tau + gap <= b0))[::4]
        blk = np.flatnonzero(m & (tau >= b0))
        mu, sd = np.nanmean(X[fit], 0), np.nanstd(X[fit], 0) + 1e-9
        xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
        net = make_net(arch, X.shape[1], seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, state, bad, hist, tr_hist = np.inf, None, 0, [], []
        for _ep in range(8):
            batches(net, fit, Yt, opt, 1, xs)
            vl = loss_on(net, blk, Yt, xs)
            hist.append(vl)
            tr_hist.append(loss_on(net, fit[::8], Yt, xs))
            if vl < best - 1e-6:
                best, bad = vl, 0
                state = {kk: v.clone() for kk, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        net.load_state_dict(state)
        sel = np.flatnonzero((tau >= q0.value) & (tau < q1.value))
        pA[sel] = predict(net, sel, xs)
        curves.append({"retrain": str(q0.date()), "block_loss": hist, "train_loss": tr_hist,
                       "best_epoch": int(np.argmin(hist)) + 1})
        # weekly warm-start fine-tuning inside this half-year
        opt2 = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
        wk = [w for w in weeks if q0 <= w < q1]
        for w0, w1 in zip(wk, [*wk[1:], q1], strict=True):
            recent = np.flatnonzero((tau + gap <= w0.value) & (tau >= w0.value - 4 * WEEK) & ok0)
            past = np.flatnonzero((tau + gap <= w0.value) & ok0)
            if len(recent) > 50:
                replay = g.choice(past, size=min(len(past), len(recent)), replace=False)
                batches(net, np.r_[recent, replay], Yt, opt2, 2, xs)
            sw = np.flatnonzero((tau >= w0.value) & (tau < w1.value))
            if len(sw):
                pB[sw] = predict(net, sw, xs)
    out = {"arch": arch, "seed": seed, "draw": k, "curves": curves if k < 0 else None,
           "params": int(sum(p.numel() for p in make_net(arch, X.shape[1], 0).parameters())),
           "seconds": round(time.time() - t0)}
    for regime, P in (("half-yearly", pA), ("weekly fine-tune", pB)):
        out[regime] = {t: measure(P[:, j].astype(float), t) for j, t in enumerate(TARGETS)}
    if k < 0:
        np.save(work / f"rt-net-{arch}-{seed}.npy", np.stack([pA, pB]))
    return out


def run_nets(work: Path, procs: int, archs) -> None:
    t0 = time.time()
    ctx = mp.get_context("spawn")
    jobs = [(work, a, s, -1) for a in archs for s in (0, 1, 2)]
    with ctx.Pool(procs) as pool:
        real = pool.map(_net_job, jobs)
    print("nets", round(time.time() - t0), "s", flush=True)
    # seed-averaged predictions per arch and regime, and the first-period choice
    _load(work)
    avg = {}
    for a in archs:
        P = np.nanmean(np.stack([np.load(work / f"rt-net-{a}-{s}.npy") for s in (0, 1, 2)]), 0)
        for r, regime in enumerate(("half-yearly", "weekly fine-tune")):
            avg[f"{a} {regime}"] = {t: measure(P[r][:, j].astype(float), t)
                                    for j, t in enumerate(TARGETS)}
    best = max(avg, key=lambda k: np.nanmean([avg[k][t]["first"]["pooled"]["rank_ic"]
                                             for t in TARGETS]))
    arch = best.split()[0]
    with ctx.Pool(procs) as pool:
        nulls = pool.map(_net_job, [(work, arch, 0, k) for k in range(NET_DRAWS)])
    out = {"seeds": real, "averaged": avg, "chosen": best, "null": nulls,
           "seconds": round(time.time() - t0)}
    (work / "retrain-nets.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["data", "tab", "nets"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--archs", default="conv,gru,transformer")
    a = ap.parse_args()
    if a.phase == "data":
        make_data(a.work)
    elif a.phase == "tab":
        run_tab(a.work, a.procs)
    else:
        run_nets(a.work, a.procs, a.archs.split(","))


if __name__ == "__main__":
    main()
