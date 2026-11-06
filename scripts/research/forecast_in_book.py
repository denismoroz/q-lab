"""The pilot-3 direction forecast as a timing hint inside the live Bv2 book
(docs/research/forecast-in-book-protocol-2026-11-06.md).

    uv run python scripts/research/forecast_in_book.py forecasts --work DIR
    uv run python scripts/research/forecast_in_book.py books --work DIR

`forecasts` re-makes the chosen walk-forward forecasts (and 5 shuffled-label ones per series) and
checks them against pilot 3's recorded IC; `books` runs the production book and the declared
variants, chooses on the first period, judges on the last third. Writes DIR/inbook.json.
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

RAW = Path("/Users/d/prj/q-lab/data/raw")
START, END = "2019-09-08", "2026-10-30"
FIRST0 = pd.Timestamp("2022-01-01", tz="UTC")
CUT = pd.Timestamp("2024-06-12", tz="UTC")
PER_COIN = 257.0 / 4
# (family, coin, horizon) -> (source, model): the pilot-3 first-period choices; the sign
# family's model is read from pilot 3's records (`sign_model`)
SERIES = {("z", "BTC", 4): ("own", "ridge"), ("z", "BTC", 24): ("own", "gbm"),
          ("z", "AVAX", 4): ("own", "gbm"), ("z", "AVAX", 24): ("own", "gbm"),
          ("z", "ETH", 4): ("pooled", "ridge"),
          ("sign", "BTC", 4): ("own", None), ("sign", "BTC", 24): ("own", None),
          ("sign", "AVAX", 4): ("own", None), ("sign", "AVAX", 24): ("own", None),
          ("sign", "ETH", 4): ("own", None)}
FAMILIES = ("z", "sign")
PLACEBO_DRAWS = 5
GRID = [(kind, q, w) for kind in ("entry", "exit", "both") for q in (0.5, 0.25)
        for w in (24, 72)]


# ── forecasts ────────────────────────────────────────────────────────────


def make_forecasts(work: Path) -> None:
    import forecast_pilot2 as fp2
    import forecast_pilot3 as f3

    from qlab import forecast as fc

    t0 = time.time()
    rec3 = json.loads((work / "forecast3.json").read_text())
    pooled3 = json.loads((work / "forecast3-pooled.json").read_text())
    out: dict = {}
    check: dict = {}
    extra3 = json.loads((work / "forecast3-extra.json").read_text())
    Vs = {c: fp2.build(work, f3.MINUTES, "e", c) for c in f3.COINS}
    Vsign = {c: fp2.build(work, f3.MINUTES, rec3["choice"][c]["variant"], c)
             for c in ("BTC", "AVAX", "ETH")}
    models = f3._models()
    for (fam, coin, h), (src, name) in SERIES.items():
        tag = f"{fam}_{coin}_{h}"
        if fam == "sign":
            V = Vsign[coin]
            row = extra3["sign"][f"{coin} {h}h"]
            name = row["model"]
            make, scaled, kind = models[name]
            fwd = fc.forward_return(V["logc"], h)
            up = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
            p = f3._walk(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, up,
                         center=True)
            parts = fp2.parts(V["tau"], h)
            got = [f3._point("cls", p, up, parts[k]) for k in ("first", "last")]
            check[tag] = {"model": name, "recorded_auc": [row["first"]["auc"][0],
                                                         row["last"]["auc"][0]],
                          "remade_auc": got}
            out[f"{tag}_real"] = p
            for k in range(PLACEBO_DRAWS):
                rng = np.random.default_rng(50_000 * (1 + f3.COINS.index(coin)) + 100 * k + h)
                out[f"{tag}_placebo{k}"] = f3._walk(make, scaled, kind, V["X"], V["tau"],
                                                    V["d"]["start"], h, up, rng, center=True)
            out[f"{tag}_tau"] = V["tau"]
            print(tag, check[tag], round(time.time() - t0), "s", flush=True)
            continue
        V = Vs[coin]
        make, scaled, kind = models[name]
        y = V["targets"][h]
        t = ("reg", h, None)
        if src == "own":
            p = f3._walk(make, scaled, kind, V["X"], V["tau"], V["d"]["start"], h, y)
            want = rec3["table"][f"{coin} {h}h"]["direction"]
            assert want["model"] == name, (coin, h, want["model"])
            want_ic = [want["first"]["ic"][0], want["last"]["ic"][0]]
        else:
            p = pooled_walk(Vs, coin, h, make, scaled, kind)
            r = pooled3["real"][f"{coin} z {h}h {name}"]
            want_ic = [r["first"][0], r["last"][0]]
        parts = fp2.parts(V["tau"], h)
        got = [f3._point("reg", p, y, parts[k]) for k in ("first", "last")]
        check[tag] = {"model": name, "recorded_ic": want_ic, "remade_ic": got}
        out[f"{tag}_real"] = p
        # placebo: the pilot-3 null draws 0..4 of the coin's own model on the same input
        targets = list(f3._targets(V))
        for k in range(PLACEBO_DRAWS):
            rng = np.random.default_rng(10_000 * (1 + f3.COINS.index(coin)) + 100 * k
                                        + 7 * targets.index(t))
            out[f"{tag}_placebo{k}"] = f3._walk(make, scaled, kind, V["X"], V["tau"],
                                                V["d"]["start"], h, y, rng)
        out[f"{tag}_tau"] = V["tau"]
        print(tag, check[tag], round(time.time() - t0), "s", flush=True)
    np.savez_compressed(work / "inbook-forecasts.npz", **out)
    (work / "inbook-forecast-check.json").write_text(json.dumps(check, indent=1))


def pooled_walk(Vs: dict, coin: str, h: int, make, scaled, kind) -> np.ndarray:
    """The pilot-3 pooled model's walk-forward predictions for one coin."""
    import forecast_pilot3 as f3

    coins = f3.COINS
    rows = []
    for i, c in enumerate(coins):
        oh = np.zeros((len(Vs[c]["tau"]), len(coins)))
        oh[:, i] = 1
        rows.append(np.column_stack([Vs[c]["X"], oh]))
    X = np.vstack(rows)
    tau = np.concatenate([Vs[c]["tau"] for c in coins])
    start_ok = np.concatenate([Vs[c]["tau"] >= Vs[c]["d"]["start"] for c in coins])
    coin_of = np.concatenate([[i] * len(Vs[c]["tau"]) for i, c in enumerate(coins)])
    y = np.concatenate([Vs[c]["targets"][h] for c in coins])
    p = np.full(len(tau), np.nan)
    for q0, q1 in zip(f3.RETRAINS[:-1], f3.RETRAINS[1:], strict=True):
        m = (tau + h * f3.HOUR + (h * f3.HOUR + f3.DAY) <= q0.value) & ~np.isnan(y) & start_ok
        sel = (tau >= q0.value) & (tau < q1.value)
        if sel.any():
            p[sel] = f3._fit_predict(make, scaled, kind, X, m, y, sel)
    return p[coin_of == coins.index(coin)]


# ── books ────────────────────────────────────────────────────────────────

_G: dict = {}  # filled before forking: panels, params, percentiles


def on_panel(index: pd.DatetimeIndex, tau: np.ndarray, f: np.ndarray) -> np.ndarray:
    """The forecast for the hourly bar opened at T: the one made at T + 1 h."""
    want = (index + pd.Timedelta(hours=1)).as_unit("ns").asi8
    pos = np.searchsorted(tau, want)
    ok = (pos < len(tau)) & (tau[np.clip(pos, 0, len(tau) - 1)] == want)
    out = np.full(len(index), np.nan)
    out[ok] = f[pos[ok]]
    return out


def money(rec, panel, capital: float) -> tuple[dict, pd.Series]:
    from qlab import earning_test as et
    from qlab import hedge_study as hs
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest

    base = (rec.run.book_equity / capital).rename("capital_base")
    res = run_backtest(panel, rec.run.weights, CostModel(taker_fee_bps=7.5, slippage_bps=0.9),
                       panel.funding, capital_base=base, capital_reinvested=False)
    r = res.capital_return
    fee = (res.cost * base.reindex(res.cost.index).ffill())
    out: dict = {}
    for part, sel in (("first", (r.index >= FIRST0) & (r.index < CUT)),
                      ("last", r.index >= CUT)):
        n = int(sel.sum())
        years = n / hs.HOURS_PER_YEAR
        out[f"cap_ann_{part}"] = float(r[sel].sum() / years * 100)
        curve = r[sel].cumsum()
        out[f"cap_max_dd_{part}"] = float((curve - curve.cummax().clip(lower=0)).min() * 100)
        out[f"fees_year_{part}"] = float(fee[sel].sum() * 100 / years)
        out[f"p_{part}"] = et.metrics_of(r[sel], additive=True).get(et.P_KEY)
        ons = 0
        lo, hi = (FIRST0, CUT) if part == "first" else (CUT, r.index[-1] + pd.Timedelta(hours=1))
        for c in rec.stepped.columns:
            ev = hs.events(rec, c)
            if len(ev):
                d = ev["decision"]
                ons += int(((ev["kind"] == "on") & (d >= lo) & (d < hi)).sum())
        out[f"ons_per_coin_year_{part}"] = ons / years / len(rec.stepped.columns)
    return out, r


def _run(job):
    """One book: job = (book, {coin: (series, kind, q, wait)} or {})."""
    from qlab import forecast_timing as ft

    book, spec = job
    panel, params, capital = _G["books"][book]
    gates = {}
    for coin, (series, kind, q, wait) in spec.items():
        gates[coin] = ft.DelayGate(_G["pct"][series], entry=kind in ("entry", "both"),
                                   exit=kind in ("exit", "both"), q=q, wait=wait)
    rec = ft.run_book_delay(panel, params, gates)
    m, r = money(rec, panel, capital)
    return m, r.to_numpy(), r.index.asi8


def run_books(work: Path, procs: int) -> None:
    import yaml

    from qlab import earning_test as et
    from qlab import forecast_timing as ft
    from qlab import hedge_study as hs

    t0 = time.time()
    params = dict(yaml.safe_load(Path("specs/bv2-binance.yaml").read_text())["params"])
    fz = np.load(work / "inbook-forecasts.npz")
    books = {}
    for coin in ("BTC", "AVAX", "ETH"):
        sp = hs.load_panel([coin], START, END, RAW)
        books[coin] = (sp.panel, {**params, "coins": [coin], "capital_usd": PER_COIN}, PER_COIN)
    sp2 = hs.load_panel(["BTC", "AVAX"], START, END, RAW)
    books["BTC+AVAX"] = (sp2.panel, {**params, "coins": ["BTC", "AVAX"],
                                      "capital_usd": 2 * PER_COIN}, 2 * PER_COIN)
    pct = {}
    for (fam, coin, h) in SERIES:
        index = books[coin][0].prices.index
        base = f"{fam}_{coin}_{h}"
        tau = fz[f"{base}_tau"]
        for tag in ["real"] + [f"placebo{k}" for k in range(PLACEBO_DRAWS)]:
            pct[f"{base}_{tag}"] = ft.past_percentile(on_panel(index, tau, fz[f"{base}_{tag}"]))
    # the two-coin book shares the index of BTC's panel
    idx2 = books["BTC+AVAX"][0].prices.index
    assert idx2.equals(books["BTC"][0].prices.index), "the two-coin index differs from BTC's"
    _G.update(books=books, pct=pct)
    print("panels and percentiles", round(time.time() - t0), "s", flush=True)

    ctx = mp.get_context("fork")
    jobs = [(coin, {}) for coin in ("BTC", "AVAX", "ETH")]
    names = [f"{coin} production" for coin in ("BTC", "AVAX", "ETH")]
    for (fam, coin, h) in SERIES:
        for kind, q, w in GRID:
            jobs.append((coin, {coin: (f"{fam}_{coin}_{h}_real", kind, q, w)}))
            names.append(f"{coin} {fam} {h}h {kind} q={q} w={w}")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs)
    res = {n: g[0] for n, g in zip(names, got, strict=True)}
    ret = {n: pd.Series(g[1], index=pd.DatetimeIndex(g[2], tz="UTC"))
           for n, g in zip(names, got, strict=True)}
    print("variants", len(jobs), round(time.time() - t0), "s", flush=True)

    # choice on the first period, per coin
    choice: dict = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        for coin in ("BTC", "AVAX", "ETH"):
            prod = res[f"{coin} production"]
            cands = [n for n in res if n.startswith(f"{coin} {fam} ")
                     and res[n]["cap_max_dd_first"] >= prod["cap_max_dd_first"] - 1.0
                     and res[n]["cap_ann_first"] > prod["cap_ann_first"]]
            choice[fam][coin] = (max(cands, key=lambda n: res[n]["cap_ann_first"]) if cands
                                 else None)
    print("choice", choice, flush=True)

    def spec_of(name):
        _, fam, hh, kind, q, w = name.split()
        return fam, int(hh[:-1]), kind, float(q[2:]), int(w[2:])

    def series_spec(name, tag):
        fam, h, kind, q, w = spec_of(name)
        return (f"{fam}_{name.split()[0]}_{h}_{tag}", kind, q, w)

    # placebos of each chosen variant, and the two-coin book
    jobs2, names2 = [("BTC+AVAX", {})], ["BTC+AVAX production"]
    two_of = {}
    for fam in FAMILIES:
        for coin, ch in choice[fam].items():
            if ch is None:
                continue
            for k in range(PLACEBO_DRAWS):
                jobs2.append((coin, {coin: series_spec(ch, f"placebo{k}")}))
                names2.append(f"{coin} {fam} placebo{k}")
        two = {c: choice[fam][c] for c in ("BTC", "AVAX") if choice[fam][c]}
        two_of[fam] = two
        if two:
            jobs2.append(("BTC+AVAX", {c: series_spec(n, "real") for c, n in two.items()}))
            names2.append(f"BTC+AVAX {fam} chosen")
            for k in range(PLACEBO_DRAWS):
                jobs2.append(("BTC+AVAX", {c: series_spec(n, f"placebo{k}")
                                           for c, n in two.items()}))
                names2.append(f"BTC+AVAX {fam} placebo{k}")
    with ctx.Pool(procs) as pool:
        got = pool.map(_run, jobs2)
    for n, g in zip(names2, got, strict=True):
        res[n] = g[0]
        ret[n] = pd.Series(g[1], index=pd.DatetimeIndex(g[2], tz="UTC"))
    print("placebos and two-coin book", len(jobs2), round(time.time() - t0), "s", flush=True)

    # judgement on the last third
    judge: dict = {}
    for fam, book in [(f, b) for f in FAMILIES for b in ("BTC", "AVAX", "ETH", "BTC+AVAX")]:
        chosen = (choice[fam].get(book) if book != "BTC+AVAX" else
                  (f"BTC+AVAX {fam} chosen" if two_of[fam] else None))
        key = f"{fam} {book}"
        if chosen is None:
            judge[key] = {"chosen": None}
            continue
        prod = f"{book} production"
        last = ret[prod].index >= CUT
        gain = ret[chosen][last] - ret[prod][last]
        g_ann = float(gain.sum() / (last.sum() / hs.HOURS_PER_YEAR) * 100)
        g_p = et.metrics_of(gain, additive=True).get(et.P_KEY)
        plac = []
        for k in range(PLACEBO_DRAWS):
            pk = ret[f"{book} {fam} placebo{k}"][last] - ret[prod][last]
            plac.append(float(pk.sum() / (last.sum() / hs.HOURS_PER_YEAR) * 100))
        dd_ok = res[chosen]["cap_max_dd_last"] >= res[prod]["cap_max_dd_last"] - 1.0
        judge[key] = {"chosen": chosen, "gain_ann_last": g_ann, "gain_p_last": g_p,
                       "placebo_gain_ann_last": plac, "drawdown_ok": bool(dd_ok),
                       "pass": bool(g_ann > 0 and g_p is not None and g_p < 0.05 and dd_ok
                                    and all(g_ann > x for x in plac))}
    out = {"results": res, "choice": choice, "judge": judge,
           "trials": {"variants": len(GRID) * len(SERIES), "productions": 4,
                      "placebos": sum(1 for n in names2 if "placebo" in n),
                      "two_coin_chosen": sum(int(bool(v)) for v in two_of.values())},
           "seconds": round(time.time() - t0)}
    (work / "inbook.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"choice": choice, "judge": judge}, indent=1, default=str))


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
