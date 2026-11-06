"""ORACLE DIAGNOSTIC — reads the future by design, never a strategy
(docs/research/oracle-protocol-2026-11-06.md): the true pilot-3 targets, and z degraded to a
given correlation with noise, driving the timing and size uses inside the live Bv2 book.

    uv run python scripts/research/oracle_book.py series --work DIR
    uv run python scripts/research/oracle_book.py books --work DIR

Writes DIR/oracle-series.npz and DIR/oracle.json.
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
import forecast_in_book as fib  # noqa: E402
import forecast_size as fsz  # noqa: E402

COINS = ("BTC", "AVAX", "ETH")
CAPITALS = (fib.PER_COIN, 1000.0)
Z_H = (4, 24, 168, 336)
SB_H = (4, 24)
ICS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03)
SEEDS = 5
BARRIER_K = 1.0
MINUTES = Path("/Users/d/prj/q-lab/data/raw/binance/1m")


def make_series(work: Path) -> None:
    import forecast_pilot2 as fp2

    from qlab import forecast as fc
    from qlab import forecast_inputs as fi
    from qlab import hedge_study as hs

    t0 = time.time()
    out: dict = {}
    for coin in COINS:
        sp = hs.load_panel([coin], fib.START, fib.END, fib.RAW)
        index = sp.panel.prices.index
        logc = np.log(sp.panel.prices[coin].to_numpy(dtype=float))
        for h in Z_H:
            z, _ = fc.z_target(logc, h)
            out[f"{coin}_z{h}"] = z
        for h in SB_H:
            fwd = fc.forward_return(logc, h)
            out[f"{coin}_sign{h}"] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        m = fp2.load_minutes(MINUTES, coin)
        open_ns = m.index.asi8
        last_min = (index + pd.Timedelta(minutes=59)).as_unit("ns").asi8
        pos = np.searchsorted(open_ns, last_min)
        ok = (pos < len(open_ns)) & (open_ns[np.clip(pos, 0, len(open_ns) - 1)] == last_min)
        pos = np.where(ok, pos, -1)
        tau = last_min + fi.MINUTE
        for h in SB_H:
            width = BARRIER_K * fi.past_h_sigma(logc, h)
            out[f"{coin}_barrier{h}"] = fi.triple_barrier(m["close"].to_numpy(), open_ns, pos,
                                                          tau, h, width)
        print(coin, "series", round(time.time() - t0), "s", flush=True)
    np.savez_compressed(work / "oracle-series.npz", **out)


_G: dict = {}


def _discrete_pct(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sign":
        return x.astype(float)  # down 0, up 1
    return np.where(np.isnan(x), np.nan, (x + 1.0) / 2.0)  # down 0, time out 0.5, up 1


def _run(job):
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    book, use, spec = job
    panel, params, capital = _G["books"][book]
    coin = params["coins"][0]
    if use == "prod":
        rec = hs.run_book(panel, params)
    else:
        key, kind, x, y = spec
        pct = _G["pct"][key]
        n = len(pct)
        if use == "timing":
            gate = ft.DelayGate(pct, entry=kind in ("entry", "both"),
                                exit=kind in ("exit", "both"), q=x, wait=int(y))
            rec = ft.run_book_delay(panel, params, {coin: gate})
        else:
            if kind == "a":
                gate = ft.SizeGate(np.clip(1.0 - x * (pct - 0.5) * 2.0, y, 1.0), None)
            else:
                gate = ft.SizeGate(np.full(n, np.nan), np.nan_to_num(pct, nan=1.0) <= x, beta=y)
            rec = ft.run_book_size(panel, params, {coin: gate})
    m, _ = fib.money(rec, panel, capital)
    m["unsent_resize_total"] = int(rec.counts.get("order_unsent_resize", 0))
    return m


def run_books(work: Path, procs: int) -> None:
    import yaml

    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    t0 = time.time()
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    s = np.load(work / "oracle-series.npz")
    books, pct, real_ic = {}, {}, {}
    for coin in COINS:
        sp = hs.load_panel([coin], fib.START, fib.END, fib.RAW)
        for cap in CAPITALS:
            books[f"{coin}@{cap:g}"] = (sp.panel, {**params, "coins": [coin], "capital_usd": cap},
                                        cap)
        for h in Z_H:
            z = s[f"{coin}_z{h}"]
            pct[f"{coin} z{h} oracle"] = ft.past_percentile(z)
            zs = z / np.nanstd(z)
            for ic in ICS:
                for seed in range(SEEDS):
                    rng = np.random.default_rng(1000 * seed + int(ic * 100) + h)
                    noise = rng.normal(size=len(z)) * np.sqrt(max(1.0 / ic ** 2 - 1.0, 0.0))
                    f = zs + noise
                    ok = ~np.isnan(f) & ~np.isnan(z)
                    real_ic[f"{coin} z{h} ic{ic} s{seed}"] = float(np.corrcoef(f[ok], z[ok])[0, 1])
                    pct[f"{coin} z{h} ic{ic} s{seed}"] = ft.past_percentile(f)
        for h in SB_H:
            pct[f"{coin} sign{h} oracle"] = _discrete_pct(s[f"{coin}_sign{h}"], "sign")
            pct[f"{coin} barrier{h} oracle"] = _discrete_pct(s[f"{coin}_barrier{h}"], "barrier")
    _G.update(books=books, pct=pct)
    print("series ready", len(pct), round(time.time() - t0), "s", flush=True)

    oracle_keys = ([f"z{h}" for h in Z_H] + [f"sign{h}" for h in SB_H]
                   + [f"barrier{h}" for h in SB_H])
    jobs, names = [], []
    for book in books:
        coin = book.split("@")[0]
        jobs.append((book, "prod", None))
        names.append(f"{book} production")
        for ok_ in oracle_keys:
            for kind, q, w in fib.GRID:
                jobs.append((book, "timing", (f"{coin} {ok_} oracle", kind, q, w)))
                names.append(f"{book} timing {ok_} {kind} {q} {w}")
            for kind, x, y in fsz.GRID:
                jobs.append((book, "size", (f"{coin} {ok_} oracle", kind, x, y)))
                names.append(f"{book} size {ok_} {kind} {x} {y}")
    ctx = mp.get_context("fork")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs, chunksize=4)
    res = dict(zip(names, got, strict=True))
    print("oracle grid", len(jobs), round(time.time() - t0), "s", flush=True)

    # best variant under the exact z oracle on the first period, per book, horizon and use
    best = {}
    for book in books:
        prod = res[f"{book} production"]
        for h in Z_H:
            for use in ("timing", "size"):
                cands = [n for n in res if n.startswith(f"{book} {use} z{h} ")
                         and res[n]["cap_max_dd_first"] >= prod["cap_max_dd_first"] - 1.0]
                best[f"{book} {use} z{h}"] = (max(cands, key=lambda n: res[n]["cap_ann_first"])
                                              if cands else None)
    jobs2, names2 = [], []
    for key, n in best.items():
        if n is None:
            continue
        book, use, zh = key.split()
        coin = book.split("@")[0]
        _, _, _, _, kind, x, y = n.split()
        x, y = float(x), float(y)
        for ic in ICS:
            for seed in range(SEEDS):
                jobs2.append((book, use, (f"{coin} {zh} ic{ic} s{seed}", kind, x, y)))
                names2.append(f"{key} ic{ic} s{seed}")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs2, chunksize=4)
    res.update(dict(zip(names2, got, strict=True)))
    print("degraded", len(jobs2), round(time.time() - t0), "s", flush=True)

    curve = {}
    for key, n in best.items():
        if n is None:
            continue
        book = key.split()[0]
        prod = res[f"{book} production"]
        row = {"variant": n,
               "exact_gain_first": res[n]["cap_ann_first"] - prod["cap_ann_first"],
               "exact_gain_last": res[n]["cap_ann_last"] - prod["cap_ann_last"]}
        for ic in ICS:
            gf = [res[f"{key} ic{ic} s{s_}"]["cap_ann_first"] - prod["cap_ann_first"]
                  for s_ in range(SEEDS)]
            gl = [res[f"{key} ic{ic} s{s_}"]["cap_ann_last"] - prod["cap_ann_last"]
                  for s_ in range(SEEDS)]
            dl = [res[f"{key} ic{ic} s{s_}"]["cap_max_dd_last"] - prod["cap_max_dd_last"]
                  for s_ in range(SEEDS)]
            row[f"ic{ic}"] = {"gain_first": [float(np.mean(gf)), float(min(gf)), float(max(gf))],
                              "gain_last": [float(np.mean(gl)), float(min(gl)), float(max(gl))],
                              "dd_change_last_mean": float(np.mean(dl))}
        curve[key] = row
    out = {"label": "ORACLE DIAGNOSTIC: reads the future by design; never a strategy",
           "results": res, "best_under_exact_oracle": best, "curve": curve,
           "realised_ic": real_ic,
           "trials_oracle": {"grid": len(jobs) - len(books), "productions": len(books),
                             "degraded": len(jobs2)},
           "seconds": round(time.time() - t0)}
    (work / "oracle.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", out["seconds"], "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["series", "books"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.phase == "series":
        make_series(a.work)
    else:
        run_books(a.work, a.procs)


if __name__ == "__main__":
    main()
