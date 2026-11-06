"""The pre-declared check of the frozen extra-hedge scheme on SOL and ETH, BTC as a reading
(docs/research/avax-scheme-check-protocol-2026-11-06.md).

    uv run python scripts/research/scheme_check.py forecasts --work DIR
    uv run python scripts/research/scheme_check.py books --work DIR

Writes DIR/scheme-forecasts.npz and DIR/scheme-check.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_in_book as fib  # noqa: E402
import forecast_size as fsz  # noqa: E402

CAPITAL = 1000.0
H = 24
RULE = ("b", 0.1, 0.5)  # extra hedge of 0.5 while unhedged when the percentile <= 0.1
PLACEBOS = 10


def make_forecasts(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    rec3 = json.loads((work / "forecast3.json").read_text())
    out = {}
    check = {}
    for coin in ("SOL", "ETH"):
        V = fp2.build(work, f3.MINUTES, rec3["choice"][coin]["variant"], coin)
        name = rec3["table"][f"{coin} {H}h"]["direction"]["model"]
        assert name == "ridge", (coin, name)
        make, scaled, kind = f3._models()[name]
        y = V["targets"][H]
        p = f3._walk(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], H, y)
        parts = fp2.parts(V["tau"], H)
        rec = rec3["table"][f"{coin} {H}h"]["direction"]
        check[coin] = {"recorded": [rec["first"]["ic"][0], rec["last"]["ic"][0]],
                       "remade": [f3._point("reg", p, y, parts[k]) for k in ("first", "last")]}
        out[f"z_{coin}_{H}_real"] = p
        idx = list(f3._targets(V)).index(("reg", H, None))
        for k in range(PLACEBOS):
            rng = np.random.default_rng(10_000 * (1 + f3.COINS.index(coin)) + 100 * k + 7 * idx)
            out[f"z_{coin}_{H}_placebo{k}"] = f3._walk(make, scaled, kind, V["X"], V["tau"],
                                                      V["d"]["start"], H, y, rng)
        out[f"z_{coin}_{H}_tau"] = V["tau"]
        print(coin, check[coin], flush=True)
    old = np.load(work / "size-forecasts.npz")
    for k in ["real", "tau"] + [f"placebo{i}" for i in range(PLACEBOS)]:
        out[f"zold_BTC_{H}_{k}"] = old[f"z_BTC_{H}_{k}"]
    new = np.load(work / "fix-stopping-forecasts.npz")
    for k in ["real", "tau"] + [f"placebo{i}" for i in range(PLACEBOS)]:
        out[f"znew_BTC_{H}_{k}"] = new[f"z_BTC_{H}_{k}"]
    np.savez_compressed(work / "scheme-forecasts.npz", **out)
    (work / "scheme-forecast-check.json").write_text(json.dumps(check, indent=1))


def run_books(work: Path, procs: int) -> None:
    import yaml

    from qlab import earning_test as et
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    t0 = time.time()
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    fz = np.load(work / "scheme-forecasts.npz")
    books, pct = {}, {}
    for coin in ("SOL", "ETH", "BTC"):
        sp = hs.load_panel([coin], fib.START, fib.END, fib.RAW)
        books[coin] = (sp.panel, {**params, "coins": [coin], "capital_usd": CAPITAL}, CAPITAL)
    index = books["BTC"][0].prices.index
    for b in books:
        assert books[b][0].prices.index.equals(index), b
    series = {"SOL": [f"z_SOL_{H}"], "ETH": [f"z_ETH_{H}"],
              "BTC": [f"zold_BTC_{H}", f"znew_BTC_{H}"]}
    for keys in series.values():
        for key in keys:
            tau = fz[f"{key}_tau"]
            for tag in ["real"] + [f"placebo{k}" for k in range(PLACEBOS)]:
                pct[f"{key}_{tag}"] = ft.past_percentile(fib.on_panel(index, tau,
                                                                      fz[f"{key}_{tag}"]))
    fsz._G.update(books=books, pct=pct, n=len(index))
    jobs, names = [], []
    for coin, keys in series.items():
        jobs.append((coin, {}, True))
        names.append(f"{coin} production")
        for key in keys:
            for tag in ["real"] + [f"placebo{k}" for k in range(PLACEBOS)]:
                jobs.append((coin, {coin: (f"{key}_{tag}", *RULE)}, False))
                names.append(f"{key} {tag}")
    with mp.get_context("fork").Pool(procs) as pool:
        got = pool.map(fsz._run, jobs)
    res = {n: g[0] for n, g in zip(names, got, strict=True)}
    ret = {n: g[1] for n, g in zip(names, got, strict=True)}
    judge = {}
    for coin, keys in series.items():
        prod = f"{coin} production"
        for key in keys:
            row = {}
            for part, sel_of in (("first", lambda r: (r.index >= fib.FIRST0) & (r.index < fib.CUT)),
                                 ("last", lambda r: r.index >= fib.CUT)):
                sel = sel_of(ret[prod])
                years = sel.sum() / hs.HOURS_PER_YEAR
                gain = ret[f"{key} real"][sel] - ret[prod][sel]
                plac = [float((ret[f"{key} placebo{k}"][sel] - ret[prod][sel]).sum() / years * 100)
                        for k in range(PLACEBOS)]
                g = float(gain.sum() / years * 100)
                p = et.metrics_of(gain, additive=True).get(et.P_KEY)
                dd_ok = (res[f"{key} real"][f"cap_max_dd_{part}"]
                         >= res[prod][f"cap_max_dd_{part}"] - 1.0)
                row[part] = {"gain": g, "p": p, "placebo": plac, "drawdown_ok": bool(dd_ok),
                             "pass": bool(g > max(plac) and p is not None and p <= 0.05
                                          and dd_ok)}
            judge[key] = row
    out = {"rule": "extra 0.5 hedge while Bv2 is unhedged when the 1-day z forecast's past "
                   "percentile <= 0.1 (12 h hold, band 0.25), 1000 USD per coin",
           "results": res, "judge": judge, "seconds": round(time.time() - t0)}
    (work / "scheme-check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(judge, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["forecasts", "books"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.phase == "forecasts":
        make_forecasts(a.work)
    else:
        run_books(a.work, a.procs)


if __name__ == "__main__":
    main()
