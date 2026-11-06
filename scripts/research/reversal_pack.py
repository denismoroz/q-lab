"""The reversal-specific feature pack for the reversal detector
(docs/research/reversal-features-protocol-2026-11-06.md).

    uv run python scripts/research/reversal_pack.py prep --work DIR       # the four coins
    uv run python scripts/research/reversal_pack.py prep-rw --work DIR    # the random walks
    uv run python scripts/research/reversal_pack.py run --work DIR [--procs 12]

Writes DIR/rv2-<coin>-<tf>.npz and DIR/reversal-pack.json.
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
RW = ("RW1", "RW2", "RW3", "RW4")
TFS = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
NEXT = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": None}
PAIRS = ((0.5, 1.5), (0.5, 2.5), (1.0, 3.0), (1.0, 5.0))
RAW = Path("/Users/d/prj/q-lab/data/raw")
MIN = RAW / "binance" / "1m"
LARGE = ("1h", "4h", "1d")
DRAWS = 50


def _minutes(coin: str, market: str = "futures", cols=None) -> pd.DataFrame:
    cols = cols or ["open_time", "open", "high", "low", "close", "volume", "trades",
                    "taker_buy_base"]
    m = pd.concat([pd.read_parquet(f, columns=cols)
                   for f in sorted((MIN / market / coin).glob("*.parquet"))])
    m = m.drop_duplicates("open_time").set_index("open_time").sort_index()
    m.index = pd.DatetimeIndex(m.index).as_unit("ns")
    return m


def _funding(coin: str):
    from qlab import hedge_study as hs

    sp = hs.load_panel([coin], "2019-09-08", "2026-10-30", RAW)
    f = sp.panel.funding[coin]
    settle = f[f.index.hour % 8 == 0]
    pct = settle.rolling(270, min_periods=30).rank(pct=True)
    return settle.index.as_unit("ns").asi8, settle.to_numpy(), pct.to_numpy()


def _build(name: str, m: pd.DataFrame, work: Path, funding, spot_close, btc_moves):
    from qlab import reversal as rv

    bars, sig, bfs, states = {}, {}, {}, {}
    for tf, mins in TFS.items():
        b = rv.make_bars(m, mins)
        c = b["close"].to_numpy()
        bars[tf], sig[tf] = b, rv.bar_sigma(c)
        bfs[tf] = rv.bar_features(b, sig[tf])
    basis_min = (None if spot_close is None
                 else (m["close"] / spot_close.reindex(m.index) - 1).dropna())
    moves = {}
    for tf, b in bars.items():
        lc = pd.Series(np.log(b["close"].to_numpy()), index=b.index)
        ss = pd.Series(sig[tf], index=b.index)
        moves[tf] = pd.DataFrame(np.column_stack([(lc - lc.shift(k)) / (ss * np.sqrt(k))
                                                  for k in (4, 16)]), index=b.index)
    for x, big in PAIRS:
        key = f"{x:g}_{big:g}"
        evs = {tf: rv.wave_events(bars[tf]["close"].to_numpy(), sig[tf], x, big) for tf in TFS}
        for tf in TFS:
            states[(tf, key)] = (evs[tf]["state_leg"], evs[tf]["state_move"])
        for tf, b in bars.items():
            ev = evs[tf]
            c = b["close"].to_numpy()
            lab, res, timeout = rv.first_passage(c, ev, big)
            t = ev["bar"].astype(int)
            end = b["end"].to_numpy()
            extra = (np.full((len(b), 2), np.nan) if btc_moves is None
                     else btc_moves[tf].reindex(b.index).to_numpy())
            X = np.column_stack([rv.event_features(bfs[tf], ev), extra[t]])
            fund = None
            if funding is not None:
                f_ns, f_rate, f_pct = funding
                fund = (rv.bar_series(f_rate, f_ns, end), rv.bar_series(f_pct, f_ns, end))
            basis = None
            if basis_min is not None:
                rule = "D" if TFS[tf] == 1440 else f"{TFS[tf]}min"
                bb = basis_min.groupby(basis_min.index.floor(rule)).last()
                bb.index = pd.DatetimeIndex(bb.index).as_unit("ns")
                basis = bb.reindex(b.index).to_numpy()
            nxt = NEXT[tf]
            mtf = np.full((len(t), 4), np.nan)
            if nxt is not None:
                nb = bars[nxt]
                j = np.searchsorted(nb["end"].to_numpy(), end[t], side="right") - 1
                ok = j >= 0
                leg, move = evs[nxt]["state_leg"], evs[nxt]["state_move"]
                rsi8 = bfs[nxt][:, rv.FEATURE_NAMES.index("rsi_8")]
                ema16 = bfs[nxt][:, rv.FEATURE_NAMES.index("ema_dist_16")]
                jj = j[ok]
                mtf[ok] = np.column_stack([(leg[jj] == ev["side"][ok]).astype(float), move[jj],
                                           rsi8[jj], ema16[jj]])
            P = rv.reversal_pack(b, ev, sig[tf], bfs[tf], fund, basis, mtf)
            np.savez_compressed(
                work / f"rv2-{name}-{tf}-{key}.npz", X=X.astype(np.float32),
                P=P.astype(np.float32), side=ev["side"], label=lab, time=end[t],
                resolve=end[res], timeout=timeout,
                step_pct=(np.exp(ev["step_sigma"] * ev["sigma"]) - 1) * 100,
                step_sigma=ev["step_sigma"])
    return moves


def prep(work: Path) -> None:
    t0 = time.time()
    btc = None
    for coin in COINS:
        m = _minutes(coin)
        spot = _minutes(coin, "spot", ["open_time", "close"])["close"]
        moves = _build(coin, m, work, _funding(coin), spot, btc)
        if coin == "BTC":
            btc = moves
        print(coin, round(time.time() - t0), "s", flush=True)


def prep_rw(work: Path) -> None:
    """Four Gaussian random walks of minutes, each with a real coin's minute volatility and minute
    timestamps; volume, trades, taker flow, funding and spot absent (NaN)."""
    t0 = time.time()
    btc = None
    for k, coin in enumerate(COINS):
        idx = _minutes(coin, cols=["open_time", "close"])
        r = np.log(idx["close"]).diff().dropna()
        sd = float(r.clip(r.quantile(0.001), r.quantile(0.999)).std())
        rng = np.random.default_rng(20261106 + k)
        c = 100 * np.exp(np.cumsum(rng.normal(0, sd, len(idx))))
        o = np.r_[c[0], c[:-1]]
        nan = np.full(len(c), np.nan)
        m = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                          "close": c, "volume": nan, "trades": nan, "taker_buy_base": nan},
                         index=idx.index)
        moves = _build(RW[k], m, work, None, None, btc)
        if k == 0:
            btc = moves
        print(RW[k], round(time.time() - t0), "s", flush=True)


# ── the models ──────────────────────────────────────────────────────────


def cols_of(feature_set: str) -> tuple[list[int], list[int]] | None:
    """(base columns, pack columns) of a feature set: base, full, base+<group>, full-<group>."""
    from qlab import reversal as rv

    n_base = len(rv.FEATURE_NAMES) + 2
    base = list(range(n_base))
    pack = list(range(len(rv.PACK_NAMES)))
    if feature_set == "base":
        return base, []
    if feature_set == "full":
        return base, pack
    group = feature_set.split("+")[-1].split("-")[-1]
    gcols = [rv.PACK_NAMES.index(n) for n in rv.PACK_GROUPS[group]]
    if feature_set.startswith("base+"):
        return base, gcols
    return base, [j for j in pack if j not in gcols]


def load(work: Path, names, tf: str, key: str, side: int, feature_set: str) -> dict:
    bcols, pcols = cols_of(feature_set)
    parts = []
    for i, c in enumerate(names):
        z = np.load(work / f"rv2-{c}-{tf}-{key}.npz")
        s = z["side"] == side
        oh = np.zeros((int(s.sum()), len(names)), dtype=np.float32)
        oh[:, i] = 1
        X = np.column_stack([z["X"][s][:, bcols], z["P"][s][:, pcols], oh])
        parts.append({"X": X, "y": z["label"][s], "t": z["time"][s], "res": z["resolve"][s],
                      "coin": np.full(int(s.sum()), i), "step_pct": z["step_pct"][s],
                      "step_sigma": z["step_sigma"][s]})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def earliness(A: dict, p: np.ndarray, target: float = 0.8) -> dict:
    """At the threshold where the first period's precision reaches `target` (the lowest such
    score), the last period's flagged events: precision, share, travel from the extreme."""
    import reversal_run as rr

    first = (A["t"] >= rr.FIRST0) & (A["t"] < rr.CUT) & ~np.isnan(p)
    last = (A["t"] >= rr.CUT) & ~np.isnan(p)
    order = np.argsort(-p[first])
    yy = A["y"][first][order]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    ok = np.flatnonzero(prec >= target)
    if not len(ok):
        return {}
    thr = p[first][order][ok.max()]
    out = {}
    for part, mk in (("first", first), ("last", last)):
        flag = mk & (p >= thr)
        out[part] = {"precision": float(A["y"][flag].mean()) if flag.any() else float("nan"),
                     "flagged": float(flag.sum() / max(1, mk.sum())),
                     "travel_sigma": float(np.nanmean(A["step_sigma"][flag])),
                     "travel_pct": float(np.nanmean(A["step_pct"][flag]))}
    return out


def _job(args):
    work, names, key, side, model, fs, rng_k, fixed, planted = args
    import reversal_run as rr

    D = load(work, names, "15m", key, side, fs)
    apply = {tf: load(work, names, tf, key, side, fs) for tf in ("15m",) + LARGE}
    # columns that are empty in the training events (the random walk has no volume, taker flow,
    # funding or spot) are left out: the boosting cannot bin an all-missing column
    keep = ~np.all(np.isnan(D["X"]), axis=0)
    if not keep.all():
        D["X"] = D["X"][:, keep]
        for A in apply.values():
            A["X"] = A["X"][:, keep]
    if planted is not None:
        for A in [D, *apply.values()]:
            A["y"] = rr.planted_labels(A, planted)
    rng = None if rng_k < 0 else np.random.default_rng(900_000 + 1000 * rng_k + side)
    preds, rows = rr.fit_predict(model, D, apply, rng=rng, fixed=fixed)
    res = {"key": key, "side": side, "model": model, "set": fs, "names": list(names),
           "per_retrain": rows, "tfs": {}}
    for tf, A in apply.items():
        res["tfs"][tf] = rr.measure(A, preds[tf])
        if rng_k < 0 and planted is None and tf in LARGE:
            res["tfs"][tf]["earliness"] = earliness(A, preds[tf])
            res["tfs"][tf]["overshoot_rule"] = rr.measure(A, A["step_sigma"])
            res["tfs"][tf]["overshoot_earliness"] = earliness(A, A["step_sigma"])
        if planted is not None:
            from qlab import forecast as fc

            s = A["X"][:, planted["column"]]
            ok = ~np.isnan(s)
            res["tfs"][tf]["oracle_auc"] = fc.auc(s[ok], A["y"][ok])
    return res


def run(work: Path, procs: int) -> None:
    from qlab import reversal as rv

    t0 = time.time()
    ctx = mp.get_context("spawn")
    keys = [f"{x:g}_{b:g}" for x, b in PAIRS]
    jobs = [(work, COINS, k, s, m, fs, -1, None, None) for k in keys for s in (1, -1)
            for m in ("gbm", "logit") for fs in ("base", "full")]
    with ctx.Pool(procs) as pool:
        main = pool.map(_job, jobs)
    print("main", round(time.time() - t0), "s", flush=True)
    chosen = {}
    for s in (1, -1):
        cands = [r for r in main if r["side"] == s and r["set"] == "full"]
        chosen[s] = max(cands, key=lambda r: np.nanmean([r["tfs"][tf]["first"]["auc"]
                                                         for tf in ("1h", "4h")]))
    jobs2 = []
    for s, r in chosen.items():
        for g in rv.PACK_GROUPS:
            for fs in (f"base+{g}", f"full-{g}"):
                jobs2.append((work, COINS, r["key"], s, "gbm", fs, -1, None, None))
        for m in ("gbm", "logit"):
            for fs in ("base", "full"):
                jobs2.append((work, RW, r["key"], s, m, fs, -1, None, None))
        fixed = [row["setting"] for row in r["per_retrain"]]
        jobs2 += [(work, COINS, r["key"], s, r["model"], "full", k, fixed, None)
                  for k in range(DRAWS)]
    col = len(rv.FEATURE_NAMES) + 2 + rv.PACK_NAMES.index("rsi_div_8")
    D0 = load(work, COINS, "15m", chosen[1]["key"], 1, "full")
    sx = D0["X"][:, col].astype(float)
    planted = {"column": col, "name": "rsi_div_8", "mu": float(np.nanmean(sx)),
               "sd": float(np.nanstd(sx)),
               "a": float(np.log(D0["y"].mean() / (1 - D0["y"].mean()))), "beta": 0.6}
    jobs2.append((work, COINS, chosen[1]["key"], 1, chosen[1]["model"], "full", -1, None,
                  planted))
    with ctx.Pool(procs) as pool:
        rest = pool.map(_job, jobs2)
    print("rest", round(time.time() - t0), "s", flush=True)
    out = {"main": main, "chosen": {str(s): f"{r['key']} {r['model']}" for s, r in chosen.items()},
           "ablation": [r for r, j in zip(rest, jobs2, strict=True)
                        if j[1] == COINS and j[6] < 0 and j[8] is None],
           "random_walk": [r for r, j in zip(rest, jobs2, strict=True) if j[1] == RW],
           "null": [{"side": j[3], "tfs": r["tfs"]} for r, j in zip(rest, jobs2, strict=True)
                    if j[6] >= 0],
           "planted": {k: v for k, v in planted.items()} | {"result": rest[-1]["tfs"]},
           "seconds": round(time.time() - t0)}
    (work / "reversal-pack.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prep", "prep-rw", "run"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    {"prep": lambda: prep(a.work), "prep-rw": lambda: prep_rw(a.work),
     "run": lambda: run(a.work, a.procs)}[a.phase]()


if __name__ == "__main__":
    main()
