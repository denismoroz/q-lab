"""Waves: the small sequence model (docs/research/waves-1-sequence-protocol-2026-11-05.md).

    uv run python scripts/research/waves_seq.py --work DIR [--minutes DIR_OF_1M]

Writes DIR/waves-seq.json.
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
import waves as wr  # noqa: E402

from qlab import bv3_parts as bp  # noqa: E402
from qlab import earning_test  # noqa: E402
from qlab import hedge_followup as hf  # noqa: E402
from qlab import waves as wv  # noqa: E402
from qlab import waves_tcn as wt  # noqa: E402

SEQ_FEATURES = ["dc1", "dc2", "dc4", "imb24", "rv24", "ret1", "slope7", "slope8", "overshoot_now"]
GBM_BEST_FIRST_J = 0.395  # from waves.json, the gate of the addendum protocol


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
    print("features", round(time.time() - t0), "s", flush=True)
    px = {c: panel.prices[c].ffill().to_numpy(dtype=float) for c in wr.COINS}
    closes = {c: hf.daily_closes(panel.prices[c]).dropna() for c in wr.COINS}
    evaluable = {c: (index >= feats[c]["start"] + wr.WARM) & ~np.isnan(px[c]) for c in wr.COINS}
    n = len(index)
    gap = wt.WINDOW
    offsets, blocks = {}, []
    for k, c in enumerate(wr.COINS):
        offsets[c] = k * (n + gap)
        blocks.append(np.column_stack([feats[c][f] for f in SEQ_FEATURES]))
        blocks.append(np.zeros((gap, len(SEQ_FEATURES))))
    big = np.vstack(blocks)
    day = index.tz_convert("UTC").tz_localize(None).floor("D")
    prob = {c: np.full(n, np.nan) for c in wr.COINS}
    starts = pd.date_range(wr.GBM_START, index[-1], freq="6MS")
    notes = {"retrains": 0, "train_rows": []}
    for t_start in starts:
        ends, ys = [], []
        for c in wr.COINS:
            falls, pivot = wv.confirmed_falls(closes[c], *wr.MAIN,
                                              t_start.tz_localize(None) - pd.Timedelta(days=1))
            if pivot is None:
                continue
            keep = (day <= pivot) & (index >= feats[c]["start"] + wr.WARM + pd.Timedelta(
                hours=wt.WINDOW))
            y = np.zeros(n, dtype=bool)
            for p, tr in falls:
                y |= (day >= p) & (day <= tr)
            pos = np.flatnonzero(keep)[::6]
            ends.append(offsets[c] + pos)
            ys.append(y[pos])
        e, yv = np.concatenate(ends), np.concatenate(ys)
        mu = np.nanmean(big[e], axis=0)
        sd = np.nanstd(big[e], axis=0) + 1e-9
        x = (big - mu) / sd
        model = wt.fit(x, e, yv)
        t_end = t_start + pd.DateOffset(months=6)
        sel = np.flatnonzero((index >= t_start) & (index < t_end))
        for c in wr.COINS:
            prob[c][sel] = wt.predict(model, x, offsets[c] + sel)
        notes["retrains"] += 1
        notes["train_rows"].append(int(len(e)))
        print("retrain", t_start.date(), len(e), round(time.time() - t0), "s", flush=True)
    falls = {c: bp.fall_positions(bp.zigzag_falls(closes[c], *wr.MAIN), index) for c in wr.COINS}
    seq_eval = {c: evaluable[c] & ~np.isnan(prob[c]) for c in wr.COINS}
    prod = {}
    for c in wr.COINS:
        p = panel.prices[c].to_numpy(dtype=float)
        priced = ~np.isnan(p)
        s14, s30 = hf.lagged(p, priced, 336), hf.lagged(p, priced, 720)
        m = np.fmin(np.where(np.isnan(s14), -np.inf, s14), s30)
        prod[c] = ~np.isnan(s30) & (m <= 0)
    res = {"notes": notes, "rows": {},
           "production_same_hours": wr.yardstick(prod, px, falls, index, seq_eval)}
    for tau in (0.3, 0.4, 0.5):
        d = {c: np.nan_to_num(prob[c], nan=0.0) > tau for c in wr.COINS}
        res["rows"][f"tcn {tau}"] = wr.yardstick(d, px, falls, index, seq_eval)
    pick = max(res["rows"], key=lambda k: res["rows"][k]["first"]["J"] or -9)
    res["pick"] = pick
    print("pick", pick, res["rows"][pick]["first"]["J"], flush=True)
    if (res["rows"][pick]["first"]["J"] or -9) > GBM_BEST_FIRST_J:
        tau = float(pick.split()[1])
        bparams, capital = hfu._book_params("set_a", list(wr.COINS))
        res["book"] = {}
        for label, parts in (("as is", bp.Parts()),
                             ("with F caps", bp.Parts(min_hold=168, cooldown=24))):
            wishes = {}
            for c in wr.COINS:
                phase = np.where(seq_eval[c], np.nan_to_num(prob[c]) > tau, prod[c])
                wishes[c] = bp.machine(panel.prices[c].to_numpy(dtype=float), phase, None, None,
                                       parts)
            rb, r = hfu.book_run(panel, bparams, capital, wishes=wishes,
                                 coins_all=list(wr.COINS))
            rb["p_last"] = earning_test.metrics_of(r[r.index >= wr.CUT], additive=True).get(
                earning_test.P_KEY)
            res["book"][label] = rb
            print("book", label, json.dumps(rb), flush=True)
    res["seconds"] = round(time.time() - t0)
    (a.work / "waves-seq.json").write_text(json.dumps(res, indent=1, default=str))
    print("written", res["seconds"], "s")


if __name__ == "__main__":
    main()
