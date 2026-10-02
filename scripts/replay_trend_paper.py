"""Replay the paper trend book on q-lab's own data and compare, hour by hour,
with what the production engine wrote (docs/TREND_LIVE_REPLAY.md).

Drives frab's live trend book (`frab.strategy.trend.book`) and signal code
(`frab.strategy.trend.signals`) exactly as `frab/strategy/trend/engine.py`
does: every closed hour applies funding and the liquidation check; at
`rebalance_hour_utc` (and on the very first bar) it rebalances to
`target_weights` of the daily closes already closed by that hour, scaled by
`book_vol_scale` of the book's own daily equity. Hourly closes and funding
come from the T33 recorder's store; daily closes are fetched from
Hyperliquid's free /info endpoint.

Export the paper side first (read-only on the production host):

    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 \\
      "file:data/frab.db?mode=ro" "select params_json from strategies where id=5"' \\
      > trend_params.json
    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \\
      "file:data/frab.db?mode=ro" "select * from trend_equity where strategy_id=5 \\
      order by ts_ms"' > trend_equity.csv

Usage:

    uv run python scripts/replay_trend_paper.py --paper trend_equity.csv \\
        --params trend_params.json
"""

import argparse
import json

import httpx
import pandas as pd

from qlab.strategies.live._loader import import_frab

book_mod = import_frab("frab.strategy.trend.book")
signals = import_frab("frab.strategy.trend.signals")
TrendParams = import_frab("frab.strategy.trend.params").TrendParams

HOUR_MS, DAY_MS = 3_600_000, 86_400_000


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def daily_candles(coin: str, start_ms: int, end_ms: int) -> list[tuple[int, int, float]]:
    """(open_ms, close_ms, close) of every daily candle Hyperliquid serves in range."""
    payload = {"type": "candleSnapshot",
               "req": {"coin": coin, "interval": "1d", "startTime": start_ms, "endTime": end_ms}}
    candles = httpx.post("https://api.hyperliquid.xyz/info", json=payload, timeout=30).json()
    return sorted((int(c["t"]), int(c["T"]), float(c["c"])) for c in candles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="trend_equity export from frab.db")
    parser.add_argument("--params", required=True, help="strategies.params_json of the trend book")
    parser.add_argument("--events", help="trend_events export from frab.db, to compare fills")
    parser.add_argument("--out", help="write the hour-by-hour comparison here (csv)")
    parser.add_argument("--record-dir", default="data/recorded/hyperliquid")
    args = parser.parse_args()

    params = TrendParams.from_dict(json.load(open(args.params)))
    paper = pd.read_csv(args.paper).sort_values("ts_ms")
    first = int(paper.ts_ms.iloc[0]) - HOUR_MS  # candle open of the first paper row

    hourly, funding, daily = {}, {}, {}
    last_common = None
    for coin in params.coins:
        candles = pd.read_parquet(f"{args.record_dir}/candles/1h/{coin}.parquet")
        hourly[coin] = {_ms(t): v for t, v in candles["close"].items()}
        newest = max(hourly[coin])
        last_common = newest if last_common is None else min(last_common, newest)
        f = pd.read_parquet(f"{args.record_dir}/funding/{coin}.parquet")
        funding[coin] = {(_ms(t) // HOUR_MS) * HOUR_MS: v for t, v in f["funding_rate"].items()}
        daily[coin] = daily_candles(coin, first - (params.history_days + 5) * DAY_MS,
                                   int(paper.ts_ms.iloc[-1]) + DAY_MS)
    last = min(int(paper.ts_ms.iloc[-1]) - HOUR_MS, last_common)

    book = book_mod.TrendBook.new(params)
    book_mod.start_book(book, bar_ms=first - HOUR_MS, params=params)
    daily_eq: list[float] = []
    events, rows = [], []
    for h in range(first, last + HOUR_MS, HOUR_MS):
        prices = {c: hourly[c].get(h, book.prices.get(c)) for c in params.coins}
        prices = {c: p for c, p in prices.items() if p}
        rates = {c: funding[c].get(h, 0.0) for c in params.coins}
        weights = sigs = scale = None
        if (h // HOUR_MS) % 24 == params.rebalance_hour_utc or h == first:
            # The engine fetches daily candles from `history_days` before the
            # tick's hour and keeps those already closed at `h`.
            window_start = h - params.history_days * DAY_MS
            closes = {c: [px for t_open, t_close, px in daily[c]
                          if t_open >= window_start and t_close <= h] for c in params.coins}
            scale = signals.book_vol_scale(daily_eq, params)
            weights = signals.target_weights(closes, params, size_scale=scale)
            sigs = {c: signals.ensemble_signal(v, params) for c, v in closes.items()
                    if len(v) >= params.min_history_days}
            weights = {c: w for c, w in weights.items() if c in prices}
        events += book_mod.step(book, bar_ms=h, prices=prices, funding=rates, params=params,
                                weights=weights, signals=sigs, size_scale=scale)
        if (h + HOUR_MS) % DAY_MS == 0:
            daily_eq.append(book.equity(prices))
        rows.append({"ts_ms": h + HOUR_MS, "replay_equity": book.equity(prices),
                     "replay_legs": book.legs(), "replay_fees": book.fees,
                     "replay_funding": book.funding_total})

    rep = pd.DataFrame(rows).merge(paper[["ts_ms", "equity", "legs", "fees", "funding_total"]],
                                   on="ts_ms")
    rep.index = pd.to_datetime(rep.ts_ms, unit="ms", utc=True)
    if args.out:
        rep.to_csv(args.out)
    d = rep[["equity", "replay_equity"]].resample("1D").last().pct_change().dropna()
    print(f"matched hours {len(rep)} ({rep.index[0]} .. {rep.index[-1]})")
    print(f"paper  {rep.equity.iloc[0]:.2f} -> {rep.equity.iloc[-1]:.2f} "
          f"({rep.equity.iloc[-1] / rep.equity.iloc[0] - 1:+.2%})")
    print(f"replay {rep.replay_equity.iloc[0]:.2f} -> {rep.replay_equity.iloc[-1]:.2f} "
          f"({rep.replay_equity.iloc[-1] / rep.replay_equity.iloc[0] - 1:+.2%})")
    print(f"max |hourly equity diff| {abs(rep.equity - rep.replay_equity).max():.4f}")
    print(f"daily return correlation {d.equity.corr(d.replay_equity):.4f} (n={len(d)})")
    print(f"legs paper/replay at end {rep.legs.iloc[-1]}/{rep.replay_legs.iloc[-1]}; "
          f"fees {rep.fees.iloc[-1]:.2f}/{rep.replay_fees.iloc[-1]:.2f}; "
          f"funding {rep.funding_total.iloc[-1]:.2f}/{rep.replay_funding.iloc[-1]:.2f}")
    diff = (rep.equity - rep.replay_equity).abs()
    print("largest hourly diffs:")
    print(diff.sort_values().tail(5).round(4).to_string())
    print(f"hours with |diff| > $0.01: {(diff > 0.01).sum()}")
    print(f"fills replay {sum(e['kind'] != 'fund' for e in events)}")
    if args.events:
        pe = pd.read_csv(args.events)
        pe = pe[pe.kind != "fund"].assign(bar_ms=lambda x: x.ts_ms - HOUR_MS)
        re = pd.DataFrame([e for e in events if e["kind"] != "fund"])
        m = pe.merge(re, on=["bar_ms", "coin", "kind"], how="outer", suffixes=("_paper", "_replay"),
                     indicator=True)
        side = m._merge.value_counts()
        print(f"fills matched {side['both']}, paper only {side['left_only']}, "
              f"replay only {side['right_only']}")
        both = m[m._merge == "both"]
        qty = ((both.qty_paper - both.qty_replay).abs() / both.qty_paper.abs()).max()
        px = ((both.price_paper - both.price_replay).abs() / both.price_paper).max()
        print(f"max |qty rel diff| {qty:.2e}, max |price rel diff| {px:.2e}")


if __name__ == "__main__":
    main()
