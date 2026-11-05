"""Step 0 of the money-trained model: the passport of the minute data
(docs/research/ml-15m-2-discipline-protocol-2026-11-05.md).

    uv run python scripts/research/ml_passport.py --out FILE [--minutes DIR_OF_1M] [--raw DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from qlab import data_passport as dp

COINS = ("BTC", "ETH", "SOL", "AVAX")
COLS = ["open_time", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"]


def _cached(raw: Path, interval: str, inst: str) -> pd.DataFrame | None:
    stem = re.compile(rf"^{re.escape(inst)}__(\d{{4}}-\d{{2}}-\d{{2}})__(\d{{4}}-\d{{2}}-\d{{2}})$")
    found = []
    for p in (raw / "binance" / interval).glob(f"{inst}__*.parquet"):
        m = stem.match(p.stem)
        if m:
            found.append((m[1], [-int(x) for x in m[2].split("-")], p))
    best = min(found, key=lambda r: (r[0], r[1])) if found else None  # earliest start, latest end
    return pd.read_parquet(best[2], columns=["price", "volume"]) if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=Path, default=Path("/Users/d/prj/q-lab/data/raw/binance/1m"))
    ap.add_argument("--raw", type=Path, default=Path("/Users/d/prj/q-lab/data/raw"))
    a = ap.parse_args()
    t0 = time.time()
    out: dict = {"instruments": {}, "hashes": {}}
    closes = {}
    for market in ("futures", "spot"):
        for c in COINS:
            files = sorted((a.minutes / market / c).glob("*.parquet"))
            h = hashlib.sha256()
            for f in files:
                h.update(f.name.encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
            out["hashes"][f"{market}/{c}"] = {"files": len(files), "sha256": h.hexdigest()}
            raw = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
            rec = dp.passport(raw)
            df = raw.drop_duplicates("open_time").set_index("open_time").sort_index()
            inst = c if market == "futures" else f"{c}-SPOT"
            hourly = _cached(a.raw, "1h", inst)
            if hourly is not None:
                rec["vs_cache_1h"] = dp.cross_check(df, hourly, "h")
            if market == "futures":
                daily = _cached(a.raw, "1d", inst)
                if daily is not None:
                    rec["vs_cache_1d"] = dp.cross_check(df, daily, "D")
            closes[(market, c)] = df["close"]
            out["instruments"][f"{market}/{c}"] = rec
            print(market, c, json.dumps({k: rec[k] for k in ("rows", "missing_share", "gaps",
                                                              "duplicates", "spikes")}),
                  round(time.time() - t0), "s", flush=True)
    out["spot_vs_perp"] = {c: dp.align(closes[("futures", c)], closes[("spot", c)])
                           for c in COINS}
    fails = []
    for name, rec in out["instruments"].items():
        if rec["missing_share"] > 0.01:
            fails.append(f"{name}: missing {rec['missing_share']:.4f}")
        if rec["non_monotonic"] or rec["duplicates"]:
            fails.append(f"{name}: {rec['duplicates']} duplicates, {rec['non_monotonic']} "
                         "non-monotonic (before de-duplication on read)")
        cc = rec.get("vs_cache_1h")
        if cc and cc["mismatch_share"] is not None and cc["mismatch_share"] > 0.001:
            fails.append(f"{name}: {cc['mismatch_share']:.4f} of hours disagree with the cache")
    out["stop_rule_hits"] = fails
    out["seconds"] = round(time.time() - t0)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print("stop rule hits:", fails)


if __name__ == "__main__":
    main()
