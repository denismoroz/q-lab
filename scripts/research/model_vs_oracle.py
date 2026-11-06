"""Model vs oracle on the check data — descriptive
(docs/research/model-vs-oracle-protocol-2026-11-06.md).

    uv run python scripts/research/model_vs_oracle.py --work DIR

Writes DIR/model-vs-oracle.json.
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

COINS = ("BTC", "AVAX")
H = (4, 24)
_G: dict = {}


def _run(job):
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    coin, pct_key, kind, q, w = job
    panel, params = _G["books"][coin]
    if pct_key is None:
        rec = hs.run_book(panel, params)
    else:
        gate = ft.DelayGate(_G["pct"][pct_key], entry=kind in ("entry", "both"),
                            exit=kind in ("exit", "both"), q=q, wait=int(w))
        rec = ft.run_book_delay(panel, params, {coin: gate})
    return rec.in_pos[coin].to_numpy()


def main() -> None:
    import yaml

    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    W = a.work
    osr = np.load(W / "oracle-series.npz")
    old = np.load(W / "inbook-forecasts.npz")
    new = np.load(W / "fix-stopping-forecasts.npz")
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    books = {}
    for coin in COINS:
        sp = hs.load_panel([coin], fib.START, fib.END, fib.RAW)
        books[coin] = (sp.panel, {**params, "coins": [coin], "capital_usd": 1000.0})
    index = books["BTC"][0].prices.index
    assert books["AVAX"][0].prices.index.equals(index)
    parts = {"first": (index >= fib.FIRST0) & (index < fib.CUT), "last": index >= fib.CUT}

    models = {}
    for coin in COINS:
        for h in H:
            models[(coin, h, "chosen")] = fib.on_panel(index, old[f"z_{coin}_{h}_tau"],
                                                       old[f"z_{coin}_{h}_real"])
            key = f"z_{coin}_{h}"
            if f"{key}_real" in new:
                models[(coin, h, "fixed stop")] = fib.on_panel(index, new[f"{key}_tau"],
                                                               new[f"{key}_real"])
    out: dict = {"label": "descriptive; the oracle reads the future by design", "tables": {}}
    pct = {}
    for (coin, h, ver), f in models.items():
        z = osr[f"{coin}_z{h}"]
        fp = ft.past_percentile(f)
        pct[f"{coin} {h} {ver}"] = fp
        pct[f"{coin} {h} oracle"] = ft.past_percentile(z)
        res = {}
        for part, sel in parts.items():
            ok = sel & ~np.isnan(f) & ~np.isnan(z) & ~np.isnan(fp)
            ff, zz, pp = f[ok], z[ok], fp[ok]
            r = {"hours": int(ok.sum()), "ic": float(np.corrcoef(ff, zz)[0, 1]),
                 "sign_agree": float(np.mean(np.sign(ff) == np.sign(zz))),
                 "sd_model": float(np.std(ff)), "sd_oracle": float(np.std(zz))}
            dz = pd.qcut(zz, 10, labels=False) + 1
            r["by_oracle_decile"] = [
                {"decile": int(d), "oracle_mean_z": float(zz[dz == d].mean()),
                 "model_mean_pct": float(pp[dz == d].mean()),
                 "model_bottom10": float(np.mean(pp[dz == d] <= 0.1)),
                 "model_top10": float(np.mean(pp[dz == d] >= 0.9)),
                 "sign_agree": float(np.mean(np.sign(ff[dz == d]) == np.sign(zz[dz == d])))}
                for d in range(1, 11)]
            dm = pd.qcut(ff, 10, labels=False, duplicates="drop") + 1
            r["by_model_decile"] = [
                {"decile": int(d), "model_mean": float(ff[dm == d].mean()),
                 "oracle_mean_z": float(zz[dm == d].mean()),
                 "strong_falls": float(np.mean(zz[dm == d] < -1)),
                 "strong_rises": float(np.mean(zz[dm == d] > 1))}
                for d in sorted(set(dm))]
            r["base_strong_falls"] = float(np.mean(zz < -1))
            r["base_strong_rises"] = float(np.mean(zz > 1))
            res[part] = r
        out["tables"][f"{coin} z {h}h {ver}"] = res
    print("tables", round(time.time() - t0), "s", flush=True)

    # (5) the month of the last third with BTC's biggest fall
    px = books["BTC"][0].prices["BTC"]
    last_px = px[index >= fib.CUT]
    monthly = last_px.groupby(last_px.index.tz_convert("UTC").strftime("%Y-%m")).agg(["first",
                                                                                     "last"])
    change = monthly["last"] / monthly["first"] - 1
    month = str(change.idxmin())
    in_month = np.asarray(index.tz_convert("UTC").strftime("%Y-%m") == month)
    series = {"month": month, "btc_change": float(change.min()),
              "hours": [str(t) for t in index[in_month]]}
    for (coin, h, ver), f in models.items():
        series[f"{coin} z {h}h {ver}"] = [None if np.isnan(x) else round(float(x), 4)
                                          for x in f[in_month]]
    for coin in COINS:
        for h in H:
            series[f"{coin} oracle z {h}h"] = [None if np.isnan(x) else round(float(x), 4)
                                               for x in osr[f"{coin}_z{h}"][in_month]]
        series[f"{coin} price"] = [round(float(x), 4) for x in
                                   books[coin][0].prices[coin].to_numpy()[in_month]]
    out["month"] = series

    # (6) hedge decisions: production, the oracle's best timing variant, the model on it
    best = json.loads((W / "oracle.json").read_text())["best_under_exact_oracle"]
    _G.update(books=books, pct=pct)
    jobs, names = [], []
    for coin in COINS:
        jobs.append((coin, None, None, None, None))
        names.append(f"{coin} production")
        for h in H:
            v = best[f"{coin}@1000 timing z{h}"]
            _, _, _, kind, q, w = v.split()
            for ver in ("oracle", "chosen", "fixed stop"):
                if f"{coin} {h} {ver}" in pct:
                    jobs.append((coin, f"{coin} {h} {ver}", kind, float(q), float(w)))
                    names.append(f"{coin} {h} {ver} | {kind} {q} {w}")
    with mp.get_context("fork").Pool(min(len(jobs), 12)) as pool:
        got = pool.map(_run, jobs)
    state = dict(zip(names, got, strict=True))
    dec = {}
    for coin in COINS:
        prod = state[f"{coin} production"]
        for h in H:
            okey = next(n for n in state if n.startswith(f"{coin} {h} oracle"))
            orc = state[okey]
            for ver in ("chosen", "fixed stop"):
                mk = [n for n in state if n.startswith(f"{coin} {h} {ver} |")]
                if not mk:
                    continue
                mod = state[mk[0]]
                row = {"variant": okey.split("| ")[1]}
                for part, sel in parts.items():
                    changed = sel & (orc != prod)
                    on = changed & orc        # the oracle hedged where production did not
                    off = changed & ~orc      # the oracle released where production hedged
                    row[part] = {
                        "oracle_changed_hours": int(changed.sum()),
                        "share_model_same": float(np.mean(mod[changed] == orc[changed]))
                        if changed.any() else None,
                        "oracle_extra_hedge_hours": int(on.sum()),
                        "share_model_also_hedged": float(np.mean(mod[on])) if on.any() else None,
                        "oracle_released_hours": int(off.sum()),
                        "share_model_also_released": float(np.mean(~mod[off]))
                        if off.any() else None,
                        "model_changed_hours": int((sel & (mod != prod)).sum()),
                        "model_changes_against_oracle": int((sel & (mod != prod)
                                                             & (mod != orc)).sum())}
                dec[f"{coin} z {h}h {ver}"] = row
    out["hedge_decisions"] = dec
    out["seconds"] = round(time.time() - t0)
    (W / "model-vs-oracle.json").write_text(json.dumps(out, indent=1))
    print("written", out["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
