"""Follow-up tasks of the hedge-trigger study (owner 2026-11-05: «запускай все по порядку»).

    uv run python scripts/research/hedge_followup.py <task> --work DIR

`--work` is the folder of `hedge_trigger_study.py prepare` (its set_a.pkl / set_b.pkl); results
go to `DIR/followup-<task>.json`. Each task's protocol is committed before its run
(docs/research/hedge-trigger-<n>-*-protocol-2026-11-05.md).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hedge_trigger_study as hts  # noqa: E402

from qlab import hedge_followup as hf  # noqa: E402

CUT = hts.CUT
START, END = hts.START, hts.END


def load(work: Path, name: str) -> dict:
    main = sys.modules["__main__"]
    main.CoinData = hts.CoinData  # the pickles were written by the study run as __main__
    with open(work / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def _btc(work: Path) -> pd.Series:
    data = load(work, "set_a")
    return data["panel"].prices["BTC"]


def in_set_days(cd) -> set:
    at = cd.index[cd.stepped][cd.mask]
    return set(at.tz_convert("UTC").tz_localize(None).floor("D"))


# ── task 1 ──────────────────────────────────────────────────────────────


def task1(work: Path) -> dict:
    btc = _btc(work)
    regime = hf.regime_of_days(btc)
    out: dict = {"cut": str(CUT.date())}
    cut = CUT.tz_localize(None)
    for name in ("set_a", "set_b"):
        coins = load(work, name)["coins"]
        tables = {}
        for cd in coins:
            close = hf.daily_closes(cd.spot_own)
            days = in_set_days(cd)
            for h1 in hf.HORIZONS:
                for h2 in hf.HORIZONS:
                    p = hf.momentum_pairs(close, h1, h2)
                    p = p[p.index.isin(days)].dropna()
                    p["coin"] = cd.coin
                    p["regime"] = regime.reindex(p.index).to_numpy()
                    p["end"] = p.index + pd.Timedelta(days=h2)
                    tables.setdefault((h1, h2), []).append(p)
        res = {}
        for (h1, h2), parts in tables.items():
            t = pd.concat(parts)
            for part in ("first", "last"):
                tp = t[t["end"] < cut] if part == "first" else t[t.index >= cut]
                for reg in ("all", "bull", "flat", "bear"):
                    tr = tp if reg == "all" else tp[tp["regime"] == reg]
                    if len(tr) < 50:
                        continue
                    x, y = tr["past"].to_numpy(), tr["next"].to_numpy()
                    c = hf.block_bootstrap(tr.index.to_numpy(),
                                           lambda rows, x=x, y=y: hf.corr(x[rows], y[rows]),
                                           draws=400)
                    # rank correlation: not driven by a few collapses (LUNA, FTT); added after the
                    # Pearson numbers showed it (memo, «Отступление от протокола»)
                    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                    rk = hf.block_bootstrap(tr.index.to_numpy(),
                                            lambda rows, rx=rx, ry=ry: hf.corr(rx[rows], ry[rows]),
                                            draws=400)
                    res[f"{h1}->{h2} {part} {reg}"] = {
                        "corr": c, "rank": rk, "sign_excess": hf.sign_excess(x, y),
                        "n": int(len(tr))}
        # variance ratios, per coin pooled by mean, on the in-set days
        vr = {}
        for part in ("first", "last"):
            for reg in ("all", "bull", "flat", "bear"):
                for h in (3, 7, 14, 30):
                    vals = []
                    for cd in coins:
                        close = hf.daily_closes(cd.spot_own)
                        sel = close.index < cut if part == "first" else close.index >= cut
                        cl = close[sel]
                        if reg != "all":
                            cl = cl.where(regime.reindex(cl.index) == reg)
                        days = in_set_days(cd)
                        cl = cl.where(cl.index.isin(days))
                        if cl.notna().sum() > 200:
                            vals.append(hf.variance_ratio(cl, h))
                    vr[f"VR{h} {part} {reg}"] = float(np.nanmedian(vals)) if vals else None
        # the rule's own reading: next return after "hedge" vs "no hedge"
        rule = {}
        for h2 in (7, 14):
            parts = []
            for cd in coins:
                close = hf.daily_closes(cd.spot_own)
                lc = np.log(close)
                hedge = ~((lc - lc.shift(14) > 0) & (lc - lc.shift(30) > 0))
                hedge = hedge.where((lc - lc.shift(30)).notna())
                t = pd.DataFrame({"hedge": hedge, "next": lc.shift(-h2) - lc}).dropna()
                t = t[t.index.isin(in_set_days(cd))]
                t["regime"] = regime.reindex(t.index).to_numpy()
                t["end"] = t.index + pd.Timedelta(days=h2)
                parts.append(t)
            t = pd.concat(parts)
            for part in ("first", "last"):
                tp = t[t["end"] < cut] if part == "first" else t[t.index >= cut]
                for reg in ("all", "bull", "flat", "bear"):
                    tr = tp if reg == "all" else tp[tp["regime"] == reg]
                    h = tr["hedge"].to_numpy(dtype=bool)
                    nx = tr["next"].to_numpy()

                    def diff(rows, h=h, nx=nx):
                        a, b = nx[rows][h[rows]], nx[rows][~h[rows]]
                        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan

                    rule[f"next{h2} {part} {reg}"] = {
                        "diff": hf.block_bootstrap(tr.index.to_numpy(), diff, draws=400),
                        "mean_hedge": float(nx[h].mean()) if h.any() else None,
                        "mean_free": float(nx[~h].mean()) if (~h).any() else None,
                        "share_hedge": float(h.mean()), "n": int(len(tr))}
        out[name] = {"pairs": res, "vr": vr, "rule": rule}
        print(name, "done", flush=True)
    return out


TASKS = {"1": task1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = TASKS[a.task](a.work)
    out["seconds"] = round(time.time() - t0)
    (a.work / f"followup-{a.task}.json").write_text(json.dumps(out, indent=1, default=str))
    print("written", a.work / f"followup-{a.task}.json", out["seconds"], "s")


if __name__ == "__main__":
    main()
