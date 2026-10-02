"""Replay the paper Bv2 books on q-lab's own recorded data and compare, hour by
hour, with what the production engine wrote (docs/BV2_LIVE.md, "Сверка с бумагой").

Drives frab's live book (`frab.strategy.b2.book`) exactly as its engine does --
same startup history (740 + 48 hourly closes), same warm-up, same start bar,
perp closes and highs, funding floored to the hour and read as 0 when absent --
over `data/recorded/hyperliquid/` (the T33 recorder's store).

Export the paper side first (read-only on the production host):

    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \
      "file:data/frab.db?mode=ro" "select strategy_id, ts_ms, coin, price, equity, \
      book_equity, cash, spot_value, short_pnl, hedge_on from b2_equity \
      order by strategy_id, coin, ts_ms"' > b2_equity.csv

Usage:

    uv run python scripts/replay_bv2_paper.py --paper b2_equity.csv [--b2-sized-with-carry]

`--b2-sized-with-carry` reproduces how book 3 (`b2`) was actually created on
2026-09-13: with carry still enabled, so its persisted position sizes are the
carry-on ones, while it has been stepped with carry off ever since.
"""
import argparse

import pandas as pd

from qlab.strategies.live._loader import import_frab

book_mod = import_frab("frab.strategy.b2.book")
B2Params = import_frab("frab.strategy.b2.params").B2Params

HIST, WARM, HOUR_MS = 740, 48, 3_600_000  # frab/strategy/b2/engine.py:28-29
# The production b2 configuration (frab.db strategies.params_json, read 2026-10-01).
BASE = {
    "coins": ["BTC", "ETH", "SOL", "AVAX"],
    "capital_usd": 257.0,
    "spot_share": 0.5,
    "hedge_threshold": 0.0,
    "sticky_exit_hours": 12,
    "ratchet_threshold": 0.5,
    "carry_enabled": False,
    "carry_fraction": 0.6,
    "carry_entry_apr": 0.1,
    "carry_exit_hours": 24,
    "slippage": 0.0005,
    "min_order_usd": 10.0,
    "margin_enabled": True,
    "short_leverage": {"BTC": 3.0, "ETH": 2.0, "SOL": 1.5, "AVAX": 1.5},
    "default_leverage": 1.0,
    "maint_margin_rate": {"BTC": 0.0125, "ETH": 0.02, "SOL": 0.025, "AVAX": 0.05},
    "default_maint_margin_rate": 0.05,
    "margin_buffer": 0.1,
    "hedge_margin_headroom": 1.0,
    "rebalance_at_im_share": 0.5,
    "mode": "paper",
}
BOOKS = ((3, 1.0), (4, 1.5))  # strategy id on prod, hedge_margin_headroom


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def replay_coin(coin, params, sizing_params, first, last, record_dir):
    candles = pd.read_parquet(f"{record_dir}/candles/1h/{coin}.parquet")
    funding_frame = pd.read_parquet(f"{record_dir}/funding/{coin}.parquet")
    closes = {_ms(t): v for t, v in candles["close"].items()}
    highs = {_ms(t): v for t, v in candles["high"].items()}
    funding = {(_ms(t) // HOUR_MS) * HOUR_MS: v for t, v in funding_frame["funding_rate"].items()}

    hours = range(first - (HIST + WARM) * HOUR_MS, last + HOUR_MS, HOUR_MS)
    avail = [h for h in hours if h in closes]
    last = min(last, max(avail))  # our record ends at the last recorder run
    px = [closes[h] for h in avail]
    fr = [funding.get(h, 0.0) for h in avail]
    pos = {h: i for i, h in enumerate(avail)}

    def window(i):
        return px[max(0, i - HIST + 9) : i + 1], fr[max(0, i - 8) : i + 1]

    book = book_mod.CoinBook.new(coin, sizing_params)
    i0 = pos[first]
    for i in range(max(0, i0 - WARM), i0):
        book_mod.advance_signals(book, *window(i), params)
    book_mod.start_book(book, bar_ms=first, price=px[i0], params=params)
    rows = []
    for h in range(first, last + HOUR_MS, HOUR_MS):
        i = pos[h]
        closes_i, funding_i = window(i)
        book_mod.step(
            book, bar_ms=h, price=px[i], funding_rate=fr[i], closes=closes_i,
            funding_hist=funding_i, params=params, high=highs.get(h),
        )
        rows.append({"ts_ms": h + HOUR_MS, "replay_equity": book.equity(px[i]),
                     "replay_price": px[i]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="b2_equity export from frab.db")
    parser.add_argument("--record-dir", default="data/recorded/hyperliquid")
    parser.add_argument("--b2-sized-with-carry", action="store_true")
    args = parser.parse_args()
    paper = pd.read_csv(args.paper)

    for strategy_id, headroom in BOOKS:
        params = B2Params.from_dict({**BASE, "hedge_margin_headroom": headroom})
        carry_sizing = args.b2_sized_with_carry and strategy_id == 3
        sizing = B2Params.from_dict(
            {**BASE, "hedge_margin_headroom": headroom, "carry_enabled": carry_sizing}
        )
        merged = []
        for coin in params.coins:
            pc = paper[(paper.strategy_id == strategy_id) & (paper.coin == coin)]
            first = int(pc.ts_ms.min()) - HOUR_MS  # candle open of the first paper row
            last = int(pc.ts_ms.max()) - HOUR_MS
            rep = replay_coin(coin, params, sizing, first, last, args.record_dir)
            m = rep.merge(pc[["ts_ms", "equity", "price"]], on="ts_ms")
            m["coin"] = coin
            merged.append(m)
        m = pd.concat(merged)
        mismatched = int((abs(m.replay_price - m.price) > 1e-9).sum())
        total = m.groupby("ts_ms")[["equity", "replay_equity"]].sum()
        total.index = pd.to_datetime(total.index, unit="ms", utc=True)
        daily = total.resample("1D").last().pct_change().dropna()

        change = total.iloc[-1] / total.iloc[0] - 1
        print(f"book {strategy_id} (headroom {headroom}, sized with carry: {carry_sizing})")
        print(f"  matched hours {len(m)}, price mismatches {mismatched}")
        print(f"  paper  {change['equity']:+.2%}   replay {change['replay_equity']:+.2%}")
        print(f"  max |hourly equity diff| {abs(total.equity - total.replay_equity).max():.4f}")
        corr = daily.equity.corr(daily.replay_equity)
        print(f"  daily return correlation {corr:.4f} (n={len(daily)})")


if __name__ == "__main__":
    main()
