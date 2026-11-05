"""Waves: the coverage-first, money-weighted threshold rule for the boosted wave model
(docs/research/waves-3-coverage-protocol-2026-11-05.md).

    uv run python scripts/research/waves_cov.py --work DIR [--minutes DIR_OF_1M]

Writes DIR/waves-cov.json.
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
import bv3_income as b3  # noqa: E402
import hedge_followup as hfu  # noqa: E402
import waves as wr  # noqa: E402

from qlab import bv3_parts as bp  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402
from qlab import waves as wv  # noqa: E402

TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
WEIGHTS = {"W1": ("to_trough", 0.0), "W2": ("to_trough", 0.35), "W3": ("hour", 0.0)}
FLOORS = (0.80, 0.90)
PURGE = pd.Timedelta(days=15)


def table(prob, px, falls, index, evaluable, regime, money_scale):
    """Per tau and subset: covered weight (each weighting), fall-hour coverage, false share, false
    alarms per coin-year."""
    cut = wr.CUT
    subsets = {"first": index < cut, "last": index >= cut,
               "bull": (index < cut - PURGE) & (regime == "bull"),
               "bear": (index < cut - PURGE) & (regime == "bear")}
    w = {c: {k: wv.fall_weights(px[c], falls[c], kind, depth) * money_scale[c]
             for k, (kind, depth) in WEIGHTS.items()} for c in px}
    out = {}
    for tau in TAUS:
        row = {}
        for name, sub in subsets.items():
            acc = {k: [0.0, 0.0] for k in WEIGHTS}
            tp = fh = fp = oh = 0.0
            starts = 0
            hours = 0.0
            for c in px:
                sel = evaluable[c] & sub
                d = np.nan_to_num(prob[c], nan=0.0) > tau
                fm = bp.in_falls(len(index), falls[c])
                for k in WEIGHTS:
                    acc[k][0] += float((w[c][k] * d)[sel].sum())
                    acc[k][1] += float(w[c][k][sel].sum())
                a, b, e, g = bp.phase_metric(d, fm, sel)
                tp, fh, fp, oh = tp + a, fh + b, fp + e, oh + g
                on = np.r_[False, d[1:] & ~d[:-1]]
                starts += int((on & ~fm & sel).sum())
                hours += sel.sum()
            row[name] = {**{f"cov_{k}": (v[0] / v[1] if v[1] else None) for k, v in acc.items()},
                         "tpr": tp / fh if fh else None, "fpr": fp / oh if oh else None,
                         "false_per_coin_year": starts / (hours / 8760) if hours else None}
        out[tau] = row
    return out


def pick(tab: dict, weight: str, floor: float, worst: bool) -> float:
    key = f"cov_{weight}"
    if not worst:
        ok = [t for t in TAUS if (tab[t]["first"][key] or 0) >= floor]
        if ok:
            return min(ok, key=lambda t: tab[t]["first"]["fpr"])
        return max(TAUS, key=lambda t: tab[t]["first"][key] or 0)
    ok = [t for t in TAUS if min(tab[t]["bull"][key] or 0, tab[t]["bear"][key] or 0) >= floor]
    if ok:
        return min(ok, key=lambda t: max(tab[t]["bull"]["fpr"], tab[t]["bear"]["fpr"]))
    return max(TAUS, key=lambda t: min(tab[t]["bull"][key] or 0, tab[t]["bear"][key] or 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    a = ap.parse_args()
    t0 = time.time()
    data = hfu.load(a.work, "set_a")
    panel = data["panel"]
    index = panel.prices.index
    feats, btc_logc = {}, None
    for c in wr.COINS:
        feats[c] = wr.coin_features(c, a.minutes, index, btc_logc)
        if c == "BTC":
            btc_logc = feats[c]["logc"]
    px = {c: panel.prices[c].ffill().to_numpy(dtype=float) for c in wr.COINS}
    closes = {c: hf.daily_closes(panel.prices[c]).dropna() for c in wr.COINS}
    evaluable = {c: (index >= feats[c]["start"] + wr.WARM) & ~np.isnan(px[c]) for c in wr.COINS}
    prob, notes = wr.gbm_probabilities(feats, closes, index)
    print("gbm", notes, round(time.time() - t0), "s", flush=True)
    ev = {c: evaluable[c] & ~np.isnan(prob[c]) for c in wr.COINS}
    falls = {c: bp.fall_positions(bp.zigzag_falls(closes[c], *wr.MAIN), index) for c in wr.COINS}
    regime = hf.regime_of_hours(panel.prices["BTC"], index)
    targets = b3.targets_of("set_a", list(wr.COINS))
    scale = {c: targets[c] / 257.0 for c in wr.COINS}
    tab = table(prob, px, falls, index, ev, regime, scale)
    # production on the same hours, for reference
    prod = {}
    for c in wr.COINS:
        p = panel.prices[c].to_numpy(dtype=float)
        priced = ~np.isnan(p)
        s14, s30 = hf.lagged(p, priced, 336), hf.lagged(p, priced, 720)
        m = np.fmin(np.where(np.isnan(s14), -np.inf, s14), s30)
        prod[c] = ~np.isnan(s30) & (m <= 0)
    prod_prob = {c: prod[c].astype(float) for c in wr.COINS}
    prod_tab = table(prod_prob, px, falls, index, ev, regime, scale)[0.5]
    picks = {}
    for worst in (False, True):
        for k in WEIGHTS:
            for f in FLOORS:
                picks[f"{k} F{f:g}{' worst' if worst else ''}"] = pick(tab, k, f, worst)
    print("picks", picks, flush=True)
    bparams, capital = hfu._book_params("set_a", list(wr.COINS))
    books = {}
    for tau in sorted(set(picks.values())):
        for label, parts in (("as is", bp.Parts()),
                             ("with F caps", bp.Parts(min_hold=168, cooldown=24))):
            wishes = {}
            for c in wr.COINS:
                phase = np.where(ev[c], np.nan_to_num(prob[c]) > tau, prod[c])
                wishes[c] = bp.machine(panel.prices[c].to_numpy(dtype=float), phase, None, None,
                                       parts)
            rb, r = hfu.book_run(panel, bparams, capital, wishes=wishes,
                                 coins_all=list(wr.COINS))
            rb["p_last"] = earning_test.metrics_of(r[r.index >= wr.CUT], additive=True).get(
                earning_test.P_KEY)
            books[f"tau {tau} {label}"] = rb
            print("book", tau, label, json.dumps(rb), flush=True)
    res = {"notes": notes, "table": {str(t): v for t, v in tab.items()},
           "production_same_hours": prod_tab, "picks": picks, "books": books,
           "seconds": round(time.time() - t0)}
    (a.work / "waves-cov.json").write_text(json.dumps(res, indent=1, default=str))
    print("written", res["seconds"], "s")


if __name__ == "__main__":
    main()
