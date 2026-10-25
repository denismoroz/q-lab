"""The candidate coins of the Bv2 "most liquid coins" specs (specs/bv2-owner-ideas/).

The stand's universe rule chooses among the instruments the panel holds, so the panel must hold
every coin that could have been among the most liquid at some date. The list is built from data
dated at each step, not from today's ranking:

1. Superset: coins that were among the 40 with the largest trailing 30-day dollar volume of their
   USDT-margined PERPETUAL on Binance at the first day of any month 2023-05 .. 2026-09 (the daily
   snapshot of Binance's whole contract archive, delisted contracts included).
2. Their SPOT market's daily quote volume (api.binance.com, `<COIN>USDT`, 1d klines). A coin whose
   spot ticker differs from its perp's (1000PEPE) has no spot column in the stand's Binance source
   and cannot be a Bv2 coin there.
3. Candidates: coins that were among the 10 with the largest trailing 30-day SPOT dollar volume at
   `MIN_WEEKS` or more of the weekly dates 2023-05-01 .. 2026-09-01.

A coin that was in the top 10 on fewer weekly dates (a listing spike) is not a candidate; at a
rebalance where it would have ranked, the next coin takes its place. That is the one place the
list departs from the rule, and it is the price of fetching 43 coins and not 75.

    uv run python scripts/research/bv2_candidates.py <daily Binance snapshot dir> <out.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

MIN_WEEKS = 4
TOP_PERP = 40
TOP_SPOT = 10
WINDOW_DAYS = 30


def perp_superset(snapshot: Path) -> list[str]:
    prices = pd.read_parquet(snapshot / "prices.parquet")
    volume = pd.read_parquet(snapshot / "volume.parquet")
    average = (volume * prices).rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean()
    coins: set[str] = set()
    for month in pd.date_range("2023-05-01", "2026-09-01", freq="MS", tz="UTC"):
        row = average.loc[: month - pd.Timedelta(days=1)].iloc[-1].dropna()
        coins |= set(row.sort_values(ascending=False).index[:TOP_PERP])
    return sorted(c for c in coins if c.isalpha() and c.isupper())


def spot_dollar_volume(coins: list[str]) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    start = int(pd.Timestamp("2023-03-25", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-09-21", tz="UTC").timestamp() * 1000)
    with httpx.Client(timeout=30.0) as client:
        for coin in coins:
            rows, cursor = [], start
            while cursor < end:
                resp = client.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": f"{coin}USDT", "interval": "1d", "startTime": cursor,
                            "endTime": end, "limit": 1000},
                )
                if resp.status_code == 400:  # no such spot pair
                    break
                resp.raise_for_status()
                klines = resp.json()
                if not klines:
                    break
                rows += [(pd.Timestamp(k[0], unit="ms", tz="UTC"), float(k[7])) for k in klines]
                cursor = klines[-1][0] + 86_400_000
                time.sleep(0.1)
            if rows:
                series[coin] = pd.Series(dict(rows))
    return pd.DataFrame(series).sort_index()


def candidates(dollar_volume: pd.DataFrame) -> dict[str, int]:
    average = dollar_volume.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean()
    weeks: dict[str, int] = {}
    for day in pd.date_range("2023-05-01", "2026-09-01", freq="7D", tz="UTC"):
        row = average.loc[: day - pd.Timedelta(days=1)].iloc[-1].dropna()
        for coin in row.sort_values(ascending=False).index[:TOP_SPOT]:
            weeks[coin] = weeks.get(coin, 0) + 1
    ranked = sorted(weeks.items(), key=lambda kv: (-kv[1], kv[0]))
    return {c: n for c, n in ranked if n >= MIN_WEEKS}


if __name__ == "__main__":
    snapshot, out = Path(sys.argv[1]), Path(sys.argv[2])
    volume = spot_dollar_volume(perp_superset(snapshot))
    chosen = candidates(volume)
    out.write_text(json.dumps({"candidates": list(chosen), "weeks_in_top10": chosen}, indent=1))
    print(len(chosen), list(chosen))
