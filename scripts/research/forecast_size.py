"""The pilot-3 forecasts setting the hedge SIZE inside the live Bv2 book, which still decides
on/off (docs/research/forecast-size-protocol-2026-11-06.md).

    uv run python scripts/research/forecast_size.py forecasts --work DIR
    uv run python scripts/research/forecast_size.py books --work DIR

`forecasts` adds placebo draws 5..9 to the direction forecasts of DIR/inbook-forecasts.npz and
makes the size-baseline forecast (and 10 shuffled-label ones); `books` runs the variants, chooses
on the first period per coin and family, judges on the last third. Writes DIR/size.json.
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

PLACEBOS = 10
COINS = ("BTC", "AVAX", "ETH")
HORIZONS = {"BTC": (4, 24), "AVAX": (4, 24), "ETH": (4,)}
GRID = ([("a", a, s) for a in (0.5, 1.0) for s in (0.5, 0.75)]
        + [("b", q, b) for q in (0.1, 0.25) for b in (0.25, 0.5)])
GRID_SIZE = [("c1", s, None) for s in (0.5, 0.75)] + [("c2", t, 0.5) for t in (0.9, 0.75)]


def make_forecasts(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    t0 = time.time()
    rec3 = json.loads((work / "forecast3.json").read_text())
    extra3 = json.loads((work / "forecast3-extra.json").read_text())
    old = dict(np.load(work / "inbook-forecasts.npz"))
    out = dict(old)
    models = f3._models()
    Vz = {c: fp2.build(work, f3.MINUTES, "e", c) for c in COINS}
    Vs = {c: fp2.build(work, f3.MINUTES, rec3["choice"][c]["variant"], c) for c in COINS}
    for (fam, coin, h), (_src, name) in fib.SERIES.items():
        tag = f"{fam}_{coin}_{h}"
        if fam == "z":
            V = Vz[coin]
            make, scaled, kind = models[name]
            y = V["targets"][h]
            idx = list(f3._targets(V)).index(("reg", h, None))
            for k in range(5, PLACEBOS):
                rng = np.random.default_rng(10_000 * (1 + f3.COINS.index(coin)) + 100 * k
                                            + 7 * idx)
                out[f"{tag}_placebo{k}"] = f3._walk(make, scaled, kind, V["X"], V["tau"],
                                                    V["d"]["start"], h, y, rng)
        else:
            V = Vs[coin]
            make, scaled, kind = models[extra3["sign"][f"{coin} {h}h"]["model"]]
            fwd = fc.forward_return(V["logc"], h)
            up = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
            for k in range(5, PLACEBOS):
                rng = np.random.default_rng(50_000 * (1 + f3.COINS.index(coin)) + 100 * k + h)
                out[f"{tag}_placebo{k}"] = f3._walk(make, scaled, kind, V["X"], V["tau"],
                                                    V["d"]["start"], h, up, rng, center=True)
        print(tag, "placebos 5-9", round(time.time() - t0), "s", flush=True)
    # the size baseline: P(strong fall) + P(strong rise), the family stronger on the first period
    check = {}
    for coin in COINS:
        V = Vs[coin]
        r3 = json.loads((work / f"forecast3-{coin}-{rec3['choice'][coin]['variant']}.json")
                        .read_text())
        for h in HORIZONS[coin]:
            fam = max(("logit", "gbc"), key=lambda f_: np.mean(
                [r3["size"][f"{h}h {s} size {f_}"]["first"][0] for s in ("fall", "rise")]))
            make, scaled, kind = models[fam]
            tag = f"size_{coin}_{h}"
            labels = [V["labels"][(h, s)] for s in ("fall", "rise")]
            ps = [f3._walk(make, scaled, kind, V["B"], V["tau"], V["d"]["start"], h, y)
                  for y in labels]
            out[f"{tag}_real"] = ps[0] + ps[1]
            got = [f3._point("cls", ps[j], labels[j], fp2.parts(V["tau"], h)["first"])
                   for j in range(2)]
            want = [r3["size"][f"{h}h {s} size {fam}"]["first"][0] for s in ("fall", "rise")]
            check[tag] = {"family": fam, "recorded_first_auc": want, "remade_first_auc": got}
            for k in range(PLACEBOS):
                q = []
                for j, y in enumerate(labels):
                    rng = np.random.default_rng(70_000 * (1 + f3.COINS.index(coin)) + 100 * k
                                                + 2 * h + j)
                    q.append(f3._walk(make, scaled, kind, V["B"], V["tau"], V["d"]["start"], h,
                                      y, rng))
                out[f"{tag}_placebo{k}"] = q[0] + q[1]
            out[f"{tag}_tau"] = V["tau"]
            print(tag, check[tag], round(time.time() - t0), "s", flush=True)
    np.savez_compressed(work / "size-forecasts.npz", **out)
    (work / "size-forecast-check.json").write_text(json.dumps(check, indent=1))


# ── books ────────────────────────────────────────────────────────────────

_G: dict = {}


def _gate_of(spec):
    """spec = (series, kind, x, y) or None (the production-equivalent gate)."""
    from qlab import forecast_timing as ft

    n = _G["n"]
    if spec is None:
        return ft.SizeGate(np.full(n, np.nan), None)
    series, kind, x, y = spec
    pct = _G["pct"][series]
    if kind == "a":
        inside = np.clip(1.0 - x * (pct - 0.5) * 2.0, y, 1.0)
        return ft.SizeGate(inside, None)
    if kind == "b":
        return ft.SizeGate(np.full(n, np.nan), np.nan_to_num(pct, nan=1.0) <= x, beta=y)
    if kind == "c1":
        return ft.SizeGate(x + (1.0 - x) * pct, None)
    return ft.SizeGate(np.full(n, np.nan), np.nan_to_num(pct, nan=0.0) >= x, beta=y)


def _size_changes(rec, panel) -> pd.Series:
    """Per bar and coin summed: the short's size changed while the hedge stayed open."""
    total = pd.Series(0.0, index=rec.in_pos.index)
    for c in rec.in_pos.columns:
        px = panel.prices[c].ffill()
        units = (rec.parts["short_usd"][c] / px).fillna(0.0)
        held = rec.in_pos[c] & rec.in_pos[c].shift(1, fill_value=False)
        moved = (units.diff().abs() > 1e-9 * units.abs().clip(lower=1e-9)) & held
        total += moved.astype(float)
    return total


def _run(job):
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    book, spec, production = job
    panel, params, capital = _G["books"][book]
    if production:
        rec = hs.run_book(panel, params)
    else:
        rec = ft.run_book_size(panel, params, {c: _gate_of(spec.get(c)) for c in params["coins"]})
    m, r = fib.money(rec, panel, capital)
    ch = _size_changes(rec, panel)
    for part, sel in (("first", (ch.index >= fib.FIRST0) & (ch.index < fib.CUT)),
                      ("last", ch.index >= fib.CUT)):
        years = sel.sum() / hs.HOURS_PER_YEAR
        m[f"size_changes_per_coin_year_{part}"] = float(ch[sel].sum() / years
                                                         / len(params["coins"]))
    m["unsent_resize_total"] = int(rec.counts.get("order_unsent_resize", 0))
    m["unsent_open_total"] = int(rec.counts.get("order_unsent_open", 0))
    return m, r


def run_books(work: Path, procs: int, per_coin: float = fib.PER_COIN) -> None:
    import yaml

    from qlab import earning_test as et
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    t0 = time.time()
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    fz = np.load(work / "size-forecasts.npz")
    books = {}
    for coin in COINS:
        sp = hs.load_panel([coin], fib.START, fib.END, fib.RAW)
        books[coin] = (sp.panel, {**params, "coins": [coin], "capital_usd": per_coin}, per_coin)
    sp2 = hs.load_panel(["BTC", "AVAX"], fib.START, fib.END, fib.RAW)
    books["BTC+AVAX"] = (sp2.panel, {**params, "coins": ["BTC", "AVAX"],
                                      "capital_usd": 2 * per_coin}, 2 * per_coin)
    index = books["BTC"][0].prices.index
    for b in books:
        assert books[b][0].prices.index.equals(index), b
    pct = {}
    for coin in COINS:
        for fam in ("z", "sign", "size"):
            for h in HORIZONS[coin]:
                base = f"{fam}_{coin}_{h}"
                tau = fz[f"{base}_tau"]
                for tag in ["real"] + [f"placebo{k}" for k in range(PLACEBOS)]:
                    pct[f"{base}_{tag}"] = ft.past_percentile(fib.on_panel(index, tau,
                                                                           fz[f"{base}_{tag}"]))
    _G.update(books=books, pct=pct, n=len(index))
    print("panels and percentiles", round(time.time() - t0), "s", flush=True)

    ctx = mp.get_context("fork")
    jobs, names = [], []
    for b in list(COINS) + ["BTC+AVAX"]:
        jobs.append((b, {}, True))
        names.append(f"{b} production")
        jobs.append((b, {}, False))
        names.append(f"{b} share-mode base")
    for coin in COINS:
        for h in HORIZONS[coin]:
            for fam in ("z", "sign"):
                for kind, x, y in GRID:
                    jobs.append((coin, {coin: (f"{fam}_{coin}_{h}_real", kind, x, y)}, False))
                    names.append(f"{coin} {fam} {h}h {kind} {x} {y}")
            for kind, x, y in GRID_SIZE:
                jobs.append((coin, {coin: (f"size_{coin}_{h}_real", kind, x, y)}, False))
                names.append(f"{coin} size {h}h {kind} {x} {y}")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs)
    res = {n: g[0] for n, g in zip(names, got, strict=True)}
    ret = {n: g[1] for n, g in zip(names, got, strict=True)}
    print("variants", len(jobs), round(time.time() - t0), "s", flush=True)

    choice: dict = {fam: {} for fam in ("z", "sign", "size")}
    for fam in choice:
        for coin in COINS:
            prod = res[f"{coin} production"]
            cands = [n for n in res if n.startswith(f"{coin} {fam} ")
                     and res[n]["cap_max_dd_first"] >= prod["cap_max_dd_first"] - 1.0
                     and res[n]["cap_ann_first"] > prod["cap_ann_first"]]
            choice[fam][coin] = (max(cands, key=lambda n: res[n]["cap_ann_first"]) if cands
                                 else None)
    print("choice", choice, flush=True)

    def spec_of(name, tag):
        coin, fam, hh, kind, x, y = name.split()
        return coin, (f"{fam}_{coin}_{hh[:-1]}_{tag}", kind, float(x),
                      None if y == "None" else float(y))

    jobs2, names2, two_of = [], [], {}
    for fam in choice:
        for coin, ch in choice[fam].items():
            if ch is None:
                continue
            for k in range(PLACEBOS):
                c, sp = spec_of(ch, f"placebo{k}")
                jobs2.append((coin, {c: sp}, False))
                names2.append(f"{coin} {fam} placebo{k}")
        two = {c: choice[fam][c] for c in ("BTC", "AVAX") if choice[fam][c]}
        two_of[fam] = two
        if two:
            jobs2.append(("BTC+AVAX", dict(spec_of(n, "real") for n in two.values()), False))
            names2.append(f"BTC+AVAX {fam} chosen")
            for k in range(PLACEBOS):
                jobs2.append(("BTC+AVAX", dict(spec_of(n, f"placebo{k}") for n in two.values()),
                              False))
                names2.append(f"BTC+AVAX {fam} placebo{k}")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs2)
    for n, g in zip(names2, got, strict=True):
        res[n], ret[n] = g
    print("placebos and two-coin books", len(jobs2), round(time.time() - t0), "s", flush=True)

    judge: dict = {}
    for fam in choice:
        for book in list(COINS) + ["BTC+AVAX"]:
            chosen = (choice[fam].get(book) if book != "BTC+AVAX"
                      else (f"BTC+AVAX {fam} chosen" if two_of[fam] else None))
            key = f"{fam} {book}"
            if chosen is None:
                judge[key] = {"chosen": None}
                continue
            prod = f"{book} production"
            last = ret[prod].index >= fib.CUT
            years = last.sum() / hs.HOURS_PER_YEAR
            gain = ret[chosen][last] - ret[prod][last]
            g_ann = float(gain.sum() / years * 100)
            g_p = et.metrics_of(gain, additive=True).get(et.P_KEY)
            plac = [float((ret[f"{book} {fam} placebo{k}"][last] - ret[prod][last]).sum()
                          / years * 100) for k in range(PLACEBOS)]
            dd_ok = res[chosen]["cap_max_dd_last"] >= res[prod]["cap_max_dd_last"] - 1.0
            judge[key] = {"chosen": chosen, "gain_ann_last": g_ann, "gain_p_last": g_p,
                          "placebo_gain_ann_last": plac, "drawdown_ok": bool(dd_ok),
                          "pass": bool(g_ann > max(plac) and dd_ok),
                          "p_below_0.05": bool(g_p is not None and g_p < 0.05)}
    base_check = {b: {"production_last": res[f"{b} production"]["cap_ann_last"],
                      "share_mode_last": res[f"{b} share-mode base"]["cap_ann_last"],
                      "production_first": res[f"{b} production"]["cap_ann_first"],
                      "share_mode_first": res[f"{b} share-mode base"]["cap_ann_first"]}
                  for b in books}
    n_var = sum(1 for n in names if " production" not in n and "share-mode" not in n)
    out = {"results": res, "choice": choice, "judge": judge, "base_check": base_check,
           "trials": {"variants": n_var, "productions": len(books),
                      "placebos": sum(1 for n in names2 if "placebo" in n),
                      "two_coin_chosen": sum(int(bool(v)) for v in two_of.values())},
           "seconds": round(time.time() - t0)}
    out["per_coin_usd"] = per_coin
    name = "size.json" if per_coin == fib.PER_COIN else f"size-{per_coin:g}.json"
    (work / name).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"choice": choice, "judge": judge, "base_check": base_check}, indent=1,
                     default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["forecasts", "books"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--capital", type=float, default=fib.PER_COIN, help="money per coin, USD")
    a = ap.parse_args()
    if a.phase == "forecasts":
        make_forecasts(a.work)
    else:
        run_books(a.work, a.procs, a.capital)


if __name__ == "__main__":
    main()
