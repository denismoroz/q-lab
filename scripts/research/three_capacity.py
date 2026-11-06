"""Check 3 of the three checks: model capacity on one pooled model for the four coins
(docs/research/three-checks-protocol-2026-11-06.md): ridge, boosting with the time-ordered stop,
and a larger causal convolution net with dropout and the same stop; z return at 1 h and 4 h.

Called from `three_checks.py capacity`. Writes DIR/three-capacity.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

COINS = ("BTC", "ETH", "SOL", "AVAX")
H = (1, 4)
HOUR = 3_600_000_000_000
DAY = 24 * HOUR
BLOCK = 8 * 7 * DAY
NULL_LIN = 50
NULL_NET = 5  # shrunk from the declared 10: the net would take over an hour (declared rule)
SEEDS = (0, 1, 2)
_D: dict = {}


def load(work: Path) -> dict:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    Vs = {c: fp2.build(work, f3.MINUTES, "e", c) for c in COINS}
    X, seq, tau, coin_of, start_ok, ys = [], [], [], [], [], {h: [] for h in H}
    for i, c in enumerate(COINS):
        V = Vs[c]
        oh = np.zeros((len(V["tau"]), len(COINS)))
        oh[:, i] = 1
        X.append(np.column_stack([V["X"], oh]))
        seq.append(V["seq"].astype(np.float32))
        tau.append(V["tau"])
        coin_of.append(np.full(len(V["tau"]), i))
        start_ok.append(V["tau"] >= V["d"]["start"])
        for h in H:
            ys[h].append(V["targets"][h])
    return {"X": np.vstack(X), "seq": np.concatenate(seq), "tau": np.concatenate(tau),
            "coin": np.concatenate(coin_of), "start_ok": np.concatenate(start_ok),
            "y": {h: np.concatenate(v) for h, v in ys.items()}, "retrains": f3.RETRAINS,
            "cut": f3.fp.CUT}


def shuffled(y, m, rng):
    from qlab import forecast_inputs as fi

    out = y.copy()
    for i in range(len(COINS)):
        mc = m & (_D["coin"] == i)
        idx = np.flatnonzero(mc)
        order = np.argsort(_D["tau"][idx], kind="stable")
        idx = idx[order]
        src = fi.other_day_source(_D["tau"][idx], rng)
        out[idx] = y[idx[src]]
    return out


def parts_of(h):
    tau, cut = _D["tau"], _D["cut"]
    return {"first": (tau >= _D["retrains"][0].value) & (tau < cut),
            "last": tau >= cut + h * HOUR + DAY}


def _ic(p, y, mask):
    from qlab import forecast as fc

    ok = mask & ~np.isnan(p) & ~np.isnan(y)
    return fc.pearson(p[ok], y[ok]) if ok.sum() > 100 else float("nan")


def walk_tab(model, h, rng=None):
    """Pooled walk-forward of ridge ('ridge') or the time-stopped boosting ('gbm')."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    from qlab import forecast as fc

    X, tau, y0 = _D["X"], _D["tau"], _D["y"][h]
    gap = h * HOUR + (h * HOUR + DAY)
    p = np.full(len(tau), np.nan)
    rows = []
    for q0, q1 in zip(_D["retrains"][:-1], _D["retrains"][1:], strict=True):
        m = (tau + gap <= q0.value) & ~np.isnan(y0) & _D["start_ok"]
        sel = (tau >= q0.value) & (tau < q1.value)
        if not sel.any():
            continue
        y = y0 if rng is None else shuffled(y0, m, rng)
        if model == "ridge":
            mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0) + 1e-9
            mod = Ridge(alpha=10).fit(np.nan_to_num((X[m] - mu) / sd), y[m])
            p[sel] = mod.predict(np.nan_to_num((X[sel] - mu) / sd))
            ins = fc.pearson(mod.predict(np.nan_to_num((X[m] - mu) / sd)), y[m])
            rows.append({"in_sample": ins})
        else:
            b0 = q0.value - gap - BLOCK
            blk, fit = m & (tau >= b0), m & (tau + gap <= b0)
            mod = HistGradientBoostingRegressor(
                max_depth=3, max_iter=1000, learning_rate=0.05, min_samples_leaf=1000,
                random_state=0, early_stopping=True, n_iter_no_change=10,
                validation_fraction=None).fit(X[fit], y[fit], X_val=X[blk], y_val=y[blk])
            p[sel] = mod.predict(X[sel])
            rows.append({"in_sample": fc.pearson(mod.predict(X[fit]), y[fit]),
                         "block": fc.pearson(mod.predict(X[blk]), y[blk]),
                         "trees": int(mod.n_iter_)})
    return p, rows


class Net:
    """A causal convolution encoder over the 96 dollar bars (5 layers x 64 channels, dilations
    1..16, dropout) joined with the hourly features, two outputs (z 1 h, z 4 h)."""

    @staticmethod
    def make(n_static: int, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers, c_in = [], 5
                for d in (1, 2, 4, 8, 16):
                    layers += [nn.ConstantPad1d((2 * d, 0), 0.0),
                               nn.Conv1d(c_in, 64, 3, dilation=d), nn.ReLU(), nn.Dropout(0.2)]
                    c_in = 64
                self.conv = nn.Sequential(*layers)
                self.static = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(0.2))
                self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(64, len(H)))

            def forward(self, w, s):
                return self.head(torch.cat([self.conv(w)[:, :, -1], self.static(s)], -1))

        return M()


def walk_net(seed: int, rng=None, epochs: int = 8, patience: int = 2, stride: int = 4):
    import torch

    from qlab import forecast as fc

    torch.set_num_threads(1)
    X, seq, tau = _D["X"], _D["seq"], _D["tau"]
    Y = np.column_stack([_D["y"][h] for h in H])
    hz = np.array(H)
    p = np.full((len(tau), len(H)), np.nan)
    rows = []
    for q0, q1 in zip(_D["retrains"][:-1], _D["retrains"][1:], strict=True):
        gap = hz.max() * HOUR + (hz.max() * HOUR + DAY)
        m = (tau + gap <= q0.value) & _D["start_ok"] & ~np.isnan(Y).all(1)
        sel = np.flatnonzero((tau >= q0.value) & (tau < q1.value))
        if not len(sel):
            continue
        Yt = Y.copy()
        if rng is not None:
            for j in range(len(H)):
                Yt[:, j] = shuffled(Y[:, j], m & ~np.isnan(Y[:, j]), rng)
        b0 = q0.value - gap - BLOCK
        fit = np.flatnonzero(m & (tau + gap <= b0))[::stride]
        blk = np.flatnonzero(m & (tau >= b0))
        mu, sd = np.nanmean(X[fit], 0), np.nanstd(X[fit], 0) + 1e-9
        xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
        net = Net.make(X.shape[1], seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = np.random.default_rng(seed)

        def predict(idx, net=net, xs=xs):
            net.eval()
            out = []
            with torch.no_grad():
                for b in range(0, len(idx), 4096):
                    r = idx[b:b + 4096]
                    out.append(net(torch.from_numpy(seq[r]), torch.from_numpy(xs[r])).numpy())
            return np.vstack(out)

        def loss_on(idx, predict=predict, Yt=Yt):
            pr, yy = predict(idx), Yt[idx]
            ok = ~np.isnan(yy)
            return float(np.mean((pr[ok] - yy[ok]) ** 2))

        best, best_state, bad, hist = np.inf, None, 0, []
        for _ep in range(epochs):
            net.train()
            order = g.permutation(fit)
            for b in range(0, len(order), 512):
                r = order[b:b + 512]
                yt = torch.from_numpy(np.nan_to_num(Yt[r]).astype(np.float32))
                mt = torch.from_numpy((~np.isnan(Yt[r])).astype(np.float32))
                out = net(torch.from_numpy(seq[r]), torch.from_numpy(xs[r]))
                loss = ((out - yt) ** 2 * mt).sum() / mt.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vl = loss_on(blk)
            hist.append(vl)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        p[sel] = predict(sel)
        pf = predict(fit)
        pb = predict(blk)
        rows.append({"epochs_run": len(hist), "best_epoch": int(np.argmin(hist)) + 1,
                     "block_loss": hist,
                     **{f"in_sample_{h}h": fc.pearson(pf[:, j], Yt[fit, j])
                        for j, h in enumerate(H)},
                     **{f"block_{h}h": fc.pearson(pb[:, j], Yt[blk, j]) for j, h in enumerate(H)}})
    return p, rows


def _tab_job(args):
    model, h, k = args
    rng = None if k < 0 else np.random.default_rng(400_000 + 1000 * h + k)
    p, rows = walk_tab(model, h, rng)
    pr = parts_of(h)
    return model, h, k, {part: _ic(p, _D["y"][h], mk) for part, mk in pr.items()}, (
        rows if k < 0 else None), (p if k < 0 else None)


def _net_job(args):
    seed, k = args
    rng = None if k < 0 else np.random.default_rng(500_000 + k)
    p, rows = walk_net(seed, rng)
    return seed, k, p, rows


def _init(work: Path) -> None:
    """Each worker loads the data itself (spawned, not forked: forking after torch and OpenMP
    were loaded hung the first run)."""
    _D.update(load(work))


def run(work: Path) -> None:
    t0 = time.time()
    _D.update(load(work))
    print("loaded", len(_D["tau"]), round(time.time() - t0), "s", flush=True)
    ctx = mp.get_context("spawn")
    jobs = [(m, h, k) for m in ("ridge", "gbm") for h in H for k in range(-1, NULL_LIN)]
    with ctx.Pool(12, initializer=_init, initargs=(work,)) as pool:
        tab = pool.map(_tab_job, jobs, chunksize=4)
    print("tables", round(time.time() - t0), "s", flush=True)
    out: dict = {"tabular": {}, "net": {}}
    for m in ("ridge", "gbm"):
        for h in H:
            real = next(t for t in tab if t[0] == m and t[1] == h and t[2] < 0)
            null = [t[3] for t in tab if t[0] == m and t[1] == h and t[2] >= 0]
            row = {"per_retrain": real[4]}
            for part in ("first", "last"):
                d = np.array([n[part] for n in null])
                row[part] = {"oos": real[3][part], "null_q95": float(np.nanquantile(d, 0.95)),
                             "pct": float(np.mean(d < real[3][part]))}
                for i, c in enumerate(COINS):
                    row[part][c] = _ic(real[5], _D["y"][h], parts_of(h)[part] & (_D["coin"] == i))
            out["tabular"][f"{m} z{h}h"] = row
    net_jobs = [(s, -1) for s in SEEDS] + [(0, k) for k in range(NULL_NET)]
    with ctx.Pool(len(net_jobs), initializer=_init, initargs=(work,)) as pool:
        nets = pool.map(_net_job, net_jobs)
    print("nets", round(time.time() - t0), "s", flush=True)
    real_p = np.nanmean(np.stack([n[2] for n in nets if n[1] < 0]), 0)
    for j, h in enumerate(H):
        pr = parts_of(h)
        row = {"per_retrain_seed0": [n[3] for n in nets if n[1] < 0][0],
               "params": None}
        for part, mk in pr.items():
            null = np.array([_ic(n[2][:, j], _D["y"][h], mk) for n in nets if n[1] >= 0])
            real = _ic(real_p[:, j], _D["y"][h], mk)
            row[part] = {"oos": real, "null_max": float(np.nanmax(null)),
                         "null_q95": float(np.nanquantile(null, 0.95)),
                         "pct": float(np.mean(null < real)),
                         "seeds": [_ic(n[2][:, j], _D["y"][h], mk) for n in nets if n[1] < 0]}
            for i, c in enumerate(COINS):
                row[part][c] = _ic(real_p[:, j], _D["y"][h], mk & (_D["coin"] == i))
        out["net"][f"z{h}h"] = row
    net = Net.make(_D["X"].shape[1], 0)
    out["net_params"] = int(sum(p.numel() for p in net.parameters()))
    out["seconds"] = round(time.time() - t0)
    (work / "three-capacity.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


__all__ = ["run"]
