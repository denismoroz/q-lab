"""The reversal detector: trained on 15-minute (and 5-minute) bars, applied unchanged on 1 h / 4 h /
1 d bars (docs/research/reversal-detector-protocol-2026-11-06.md).

    uv run python scripts/research/reversal_detector.py prep --work DIR
    uv run python scripts/research/reversal_detector.py run --work DIR [--procs 12]

Writes DIR/rev-<coin>-<tf>.npz and DIR/reversal.json.
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

COINS = ("BTC", "ETH", "SOL", "AVAX")
TFS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
PAIRS = ((0.5, 1.5), (0.5, 2.5), (1.0, 3.0), (1.0, 5.0))
EXTRA = ["btc_ret_4", "btc_ret_16"]
MINUTES = Path("/Users/d/prj/q-lab/data/raw/binance/1m")


def prep(work: Path) -> None:
    import forecast_pilot2 as fp2

    from qlab import reversal as rv

    t0 = time.time()
    counts: dict = {}
    btc_feats: dict = {}
    for coin in COINS:
        m = fp2.load_minutes(MINUTES, coin)
        for tf, mins in TFS.items():
            b = rv.make_bars(m, mins)
            c = b["close"].to_numpy()
            s = rv.bar_sigma(c)
            bf = rv.bar_features(b, s)
            lc = pd.Series(np.log(c), index=b.index)
            ss = pd.Series(s, index=b.index)
            mv = np.column_stack([(lc - lc.shift(k)) / (ss * np.sqrt(k)) for k in (4, 16)])
            if coin == "BTC":
                btc_feats[tf] = pd.DataFrame(mv, index=b.index)
                extra = np.full((len(b), 2), np.nan)
            else:
                extra = btc_feats[tf].reindex(b.index).to_numpy()
            out = {"end": b["end"].to_numpy(), "close": c}
            for x, big in PAIRS:
                ev = rv.wave_events(c, s, x, big)
                lab, res, timeout = rv.first_passage(c, ev, big)
                t = ev["bar"].astype(int)
                X = np.column_stack([rv.event_features(bf, ev), extra[t]])
                key = f"{x:g}_{big:g}"
                out[f"{key}_X"] = X.astype(np.float32)
                out[f"{key}_side"] = ev["side"]
                out[f"{key}_label"] = lab
                out[f"{key}_time"] = b["end"].to_numpy()[t]
                out[f"{key}_resolve"] = b["end"].to_numpy()[res]
                out[f"{key}_timeout"] = timeout
                out[f"{key}_step_pct"] = (np.exp(ev["step_sigma"] * ev["sigma"]) - 1) * 100
                out[f"{key}_sigma"] = ev["sigma"]
                counts[f"{coin} {tf} {key}"] = {
                    "events": int(len(lab)), "top_rate": float(lab[ev["side"] == 1].mean()),
                    "bottom_rate": float(lab[ev["side"] == -1].mean()),
                    "timeout": float(timeout.mean()),
                    "mean_step_pct": float(np.nanmean(out[f"{key}_step_pct"]))}
            np.savez_compressed(work / f"rev-{coin}-{tf}.npz", **out)
            print(coin, tf, len(b), "bars", round(time.time() - t0), "s", flush=True)
    (work / "rev-counts.json").write_text(json.dumps(counts, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prep", "run"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.phase == "prep":
        prep(a.work)
    else:
        import reversal_run

        reversal_run.run(a.work, a.procs)


if __name__ == "__main__":
    main()
