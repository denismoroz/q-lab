"""Bv3 as income (owner 2026-11-05: «как сделать то что я хотел в bv3 - не страховка - а
заработок»).

    uv run python scripts/research/bv3_income.py <step> --work DIR

`--work`: the folder of `hedge_trigger_study.py prepare` (set_a.pkl / set_b.pkl). Results go to
`DIR/bv3-income-<step>.json`. Each step's protocol is committed before its run
(docs/research/bv3-income-<n>-*-protocol-2026-11-05.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hedge_followup as hfu  # noqa: E402
import hedge_trigger_study as hts  # noqa: E402

from qlab import bv3_income as bi  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402
from qlab import hedge_study as hs  # noqa: E402
from qlab.harness.run import run_backtest  # noqa: E402
from qlab.strategies.live.bv2 import B2Params  # noqa: E402

CUT = hts.CUT
PURGE = pd.Timedelta(days=15)
FAST = [(w, s) for w in (12, 24) for s in (0, 12)]
SPOT_COST = (10.0 + 0.9) / 1e4  # Binance spot taker + slippage, per dollar traded


def stand(panel, run, capital):
    """The stand's result of a book run on the strategy's money (summed hours)."""
    base = (run.book_equity / capital).rename("capital_base")
    return run_backtest(panel, run.weights, hts.COSTS, panel.funding, capital_base=base,
                        capital_reinvested=False)


def measures(r: pd.Series, regime: np.ndarray | None = None) -> dict:
    """Yearly money, worst drawdown (points of the strategy's money), the earning test on the
    last third; by regime inside the first 2/3 (purged) when `regime` is given."""
    out = {}
    for part, sel in (("first", r.index < CUT), ("last", r.index >= CUT)):
        seg = r[sel]
        out[f"ann_{part}"] = float(100 * seg.sum() * 8760 / len(seg))
        out[f"dd_{part}"] = float(100 * hf.max_drawdown(seg.to_numpy()))
    test = earning_test.metrics_of(r[r.index >= CUT], additive=True)
    out["p_last"] = test.get(earning_test.P_KEY)
    if regime is not None:
        first = (r.index < CUT - PURGE)
        for reg in ("bull", "bear"):
            sel = first & (regime[: len(r)] == reg)
            out[f"ann_first_{reg}"] = float(100 * r[sel].sum() * 8760 / max(1, sel.sum()))
    return out


def targets_of(name: str, coins: list[str]) -> dict[str, float]:
    params = hts._params()
    if name == "set_a":
        p = B2Params.from_dict(params)
        return {c: p.book_sizes(c)[0] for c in coins}
    rest = {k: v for k, v in params.items() if k != "coins"}
    return {c: B2Params.from_dict({**rest, "coins": [c], "capital_usd": hts.PER_COIN_CAPITAL})
            .book_sizes(c)[0] for c in coins}


def fast_run(panel, bparams, a_window, sticky):
    from qlab.strategies.agent.fast_release_gate import FastReleaseGate

    return FastReleaseGate().run(panel, {**bparams, "fast_window_hours": a_window,
                                         "fast_entry": 0, "sticky_exit_hours": sticky})


def step0(work: Path) -> dict:
    out: dict = {"cut": str(CUT.date())}
    for name in ("set_a", "set_b"):
        data = hfu.load(work, name)
        panel = data["panel"]
        coins = [cd.coin for cd in data["coins"]]
        bparams, capital = hfu._book_params(name, coins)
        targets = targets_of(name, coins)
        res = {}
        rec = hs.run_book(panel, bparams)
        st = stand(panel, rec.run, capital)
        parts = bi.decompose(panel, rec.run.weights, rec.run.book_equity, capital, st.cost,
                             targets)
        gap = float((parts.sum(axis=1) - st.capital_return).abs().max())
        res["production"] = {"parts": bi.summarise_parts(parts, CUT),
                             "measures": measures(st.capital_return), "max_gap": gap}
        fast = {}
        for w, s in FAST:
            run = fast_run(panel, bparams, w, s)
            stf = stand(panel, run, capital)
            fast[f"{w}h sticky {s}"] = (run, stf, measures(stf.capital_return))
            print(name, w, s, json.dumps(fast[f"{w}h sticky {s}"][2]), flush=True)
        best = max(fast, key=lambda k: fast[k][2]["ann_first"])
        run, stf, m = fast[best]
        partsf = bi.decompose(panel, run.weights, run.book_equity, capital, stf.cost, targets)
        res["fast_release"] = {"all": {k: v[2] for k, v in fast.items()}, "best": best,
                               "parts": bi.summarise_parts(partsf, CUT), "measures": m}
        out[name] = res
        print(name, json.dumps(res["production"]), flush=True)
    return out


# ── common for steps 1-3 ────────────────────────────────────────────────


def _ons(rec) -> dict:
    out = {"first": 0, "last": 0}
    for c in rec.stepped.columns:
        ev = hs.events(rec, c)
        if len(ev):
            on = ev[ev["kind"] == "on"]
            out["first"] += int((on["decision"] < CUT).sum())
            out["last"] += int((on["decision"] >= CUT).sum())
    return out


def evaluate_book(name, data, wishes=None):
    """The live book (optionally with external wishes), through the stand: measures, the
    decomposition, switch-ons and the capital return series."""
    panel = data["panel"]
    coins = [cd.coin for cd in data["coins"]]
    bparams, capital = hfu._book_params(name, coins)
    if wishes:
        nan = np.full(len(panel.prices.index), np.nan)
        wishes = {c: wishes.get(c, nan) for c in coins}
    rec = hs.run_book(panel, bparams, wishes=wishes or None)
    st = stand(panel, rec.run, capital)
    parts = bi.decompose(panel, rec.run.weights, rec.run.book_equity, capital, st.cost,
                         targets_of(name, coins))
    return {"parts": bi.summarise_parts(parts, CUT), "ons": _ons(rec)}, st.capital_return


def choose_and_report(rows: dict, returns: dict, regime: np.ndarray) -> dict:
    """The first-2/3 choice by money, the worst-regime choice, and every row's measures."""
    prod = returns["production"]
    table = {}
    for k, r in returns.items():
        m = measures(r, regime)
        table[k] = {**m, **rows[k]}
    pm = table["production"]
    names = [k for k in table if k != "production"]
    by_money = max(names, key=lambda k: table[k]["ann_first"])

    def worst(k):
        return min(table[k]["ann_first_bull"] - pm["ann_first_bull"],
                   table[k]["ann_first_bear"] - pm["ann_first_bear"])

    by_regime = max(names, key=worst)
    for k in names:
        table[k]["worst_regime_delta"] = worst(k)
    del prod
    return {"table": table, "pick_money": by_money, "pick_regime": by_regime}


def _run_grid(name, data, wish_of, labels, work, step):
    btc = hfu._btc(work)
    regime = hf.regime_of_hours(btc, data["panel"].prices.index)
    rows, returns = {}, {}
    rows["production"], returns["production"] = evaluate_book(name, data)
    for label in labels:
        rows[label], returns[label] = evaluate_book(name, data, wish_of(label))
        print(name, step, label, json.dumps(measures(returns[label])), flush=True)
    pd.DataFrame(returns).to_parquet(work / f"bv3-{step}-returns-{name}.parquet")
    return choose_and_report(rows, returns, regime)


# ── step 1: carry hedge ─────────────────────────────────────────────────

CARRY = [(kind, th, e) for kind, th in (("level", 0.10), ("level", 0.20), ("level", 0.30),
                                        ("level", 0.50), ("pct", 0.8), ("pct", 0.9))
         for e in (0.0, 0.05)]


def step1(work: Path) -> dict:
    out: dict = {"grid": CARRY}
    for name in ("set_a", "set_b"):
        data = hfu.load(work, name)
        panel = data["panel"]
        coins = [cd.coin for cd in data["coins"]]
        pre = {}
        for c in coins:
            f = panel.funding[c]
            f24, f72 = bi.funding_annual(f, 24), bi.funding_annual(f, 72)
            pct = hs.past_percentile(pd.Series(f24, index=f.index)).to_numpy()
            base = hf.band_wish(panel.prices[c].to_numpy(dtype=float))
            pre[c] = (base, f24, f72, pct)
        labels = [f"{k} {th:g} stay>{e:g}" for k, th, e in CARRY]

        def wish_of(label, pre=pre, labels=labels):
            kind, th, e = CARRY[labels.index(label)]
            w = {}
            for c, (base, f24, f72, pct) in pre.items():
                x = f24 if kind == "level" else pct
                with np.errstate(invalid="ignore"):
                    enter = np.nan_to_num(x, nan=-np.inf) >= th
                    stay = np.nan_to_num(f72, nan=-np.inf) > e
                w[c] = bi.carry_wish(base, enter, stay)
            return w

        out[name] = _run_grid(name, data, wish_of, labels, work, "1")
    return out


# ── step 2: relative strength ───────────────────────────────────────────

STRENGTH = [(w, h, mode) for w in (7, 14, 30) for h in (0.25, 0.5) for mode in ("alone", "and")]


def _members(name, data, days):
    coins = [cd.coin for cd in data["coins"]]
    if name == "set_a":
        close = pd.DataFrame({c: hf.daily_closes(data["panel"].prices[c]) for c in coins})
        return close.notna().reindex(days).fillna(False)
    top = data["top"]
    return top.reindex(index=days, columns=coins).fillna(False).astype(bool)


def step2(work: Path) -> dict:
    out: dict = {"grid": STRENGTH}
    for name in ("set_a", "set_b"):
        data = hfu.load(work, name)
        panel = data["panel"]
        index = panel.prices.index
        coins = [cd.coin for cd in data["coins"]]
        close = pd.DataFrame({c: hf.daily_closes(panel.prices[c]) for c in coins})
        members = _members(name, data, close.index)
        hour_day = index.tz_convert("UTC").tz_localize(None).floor("D")
        base = {c: hf.band_wish(panel.prices[c].to_numpy(dtype=float)) for c in coins}
        priced = {c: panel.prices[c].notna().to_numpy() for c in coins}
        labels = [f"W{w} share{h:g} {m}" for w, h, m in STRENGTH]

        def wish_of(label, close=close, members=members, hour_day=hour_day, base=base,
                    priced=priced, labels=labels):
            w_days, h, mode = STRENGTH[labels.index(label)]
            past = (close / close.shift(w_days) - 1.0).shift(1)  # known at the day's start
            weak = bi.weakest_wish(past, members, h)
            flags = weak.reindex(hour_day).fillna(False).to_numpy(dtype=bool)
            out_w = {}
            for k, c in enumerate(close.columns):
                f = flags[:, k]
                if mode == "and":
                    f = f & (np.nan_to_num(base[c]) > 0.5)
                out_w[c] = np.where(priced[c], f.astype(float), np.nan)
            return out_w

        out[name] = _run_grid(name, data, wish_of, labels, work, "2")
    return out


# ── step 3: rebalancing premium ─────────────────────────────────────────

REBAL = [(b, wk, d) for b, wk in ((0.05, False), (0.10, False), (0.20, False), (None, True))
         for d in (None, 0.15, 0.25)]


def step3(work: Path) -> dict:
    out: dict = {"grid": REBAL}
    btc = hfu._btc(work)
    for name in ("set_a", "set_b"):
        data = hfu.load(work, name)
        panel = data["panel"]
        index = panel.prices.index
        coins = [cd.coin for cd in data["coins"]]
        targets = targets_of(name, coins)
        slot = float(np.mean(list(targets.values())))
        capital = 257.0 if name == "set_a" else hts.PER_COIN_CAPITAL * hts.TOP_K
        both = panel.tradeable[coins].to_numpy() & panel.tradeable[
            [f"{c}-SPOT" for c in coins]].to_numpy()
        mem = pd.DataFrame(both, index=index, columns=coins)
        if name == "set_b":
            days = index.tz_convert("UTC").tz_localize(None).floor("D")
            top = data["top"].reindex(index=days, columns=coins).fillna(False).to_numpy(bool)
            mem = mem & top
        prices, funding = panel.prices[coins], panel.funding[coins]
        regime = hf.regime_of_hours(btc, index)
        returns = {}
        returns["production"] = evaluate_book(name, data)[1]
        returns["hold, no rebalance"] = bi.rebalanced_book(
            prices, funding, mem, slot, capital, None, False, None, SPOT_COST, hts.SWITCH_COST)
        rows = {"production": {}, "hold, no rebalance": {}}
        for b, wk, d in REBAL:
            label = ("weekly" if wk else f"band {b:g}") + (f" tail {d:g}" if d else " no tail")
            returns[label] = bi.rebalanced_book(prices, funding, mem, slot, capital, b, wk, d,
                                                SPOT_COST, hts.SWITCH_COST)
            rows[label] = {}
            print(name, "3", label, json.dumps(measures(returns[label])), flush=True)
        pd.DataFrame(returns).to_parquet(work / f"bv3-3-returns-{name}.parquet")
        res = choose_and_report(rows, returns, regime)
        res["slot_usd"], res["capital"] = slot, capital
        # the reference row is not a trial: never picked
        for key in ("pick_money", "pick_regime"):
            if res[key] == "hold, no rebalance":
                names = [k for k in res["table"] if k not in ("production", "hold, no rebalance")]
                res[key] = max(names, key=lambda k: res["table"][k]["ann_first"]
                               if key == "pick_money" else res["table"][k]["worst_regime_delta"])
        out[name] = res
    return out


STEPS = {"0": step0, "1": step1, "2": step2, "3": step3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = STEPS[a.step](a.work)
    out["seconds"] = round(time.time() - t0)
    path = a.work / f"bv3-income-{a.step}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("written", path, out["seconds"], "s")


if __name__ == "__main__":
    main()
