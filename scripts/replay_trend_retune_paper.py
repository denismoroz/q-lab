"""Replay the paper book `trend_retune_3k` (frab strategy id 6) on q-lab's own data and
compare, hour by hour, with what the production engine wrote.

`trend_retune_*` is the weekly re-tuned trend book: ten candidates (coin-selection rule x
lookback set) are simulated every day, and once a week the one with the best Sharpe so far is
the one traded; coins come from code (top 25 by market cap or by volume), not from a list.
This script drives frab's OWN code, the way `frab/strategy/trend/retune_engine.py` does:

1. the candidates' past from 2025-01-01 up to the launch, by frab's `retune_backfill`
   (`iter_replay` -> `advance_day`) on the stand's cached daily candles
   (`data/raw/hyperliquid/1d`), the funding settlements in the same cache, and the
   CoinMarketCap rankings of q-lab's store (`qlab.data.cap_history`, byte-identical to
   frab's `deploy/trend_retune_cap_history.json` when checked on 2026-10-25);
2. the book: started at the launch hour and sized to the last historical bar's weights, then
   `retune.advance_day` on every daily bar after it and `book.step` every hour, on the
   recorder's hourly closes and funding (`data/recorded/hyperliquid`).

One input is NOT q-lab's: from the first live week frab ranks by CoinGecko's market caps of
that day, which nobody else can reproduce. The cap-rule universe of every live bar is read
from frab's own log (`retune_day` events in `trend_events`, `--events`); the volume-rule
universe is recomputed from the data and compared with the log. A live bar with no logged
universe stops the replay there, and the block says so.

Printed: the live block `qlab.paper_rule` reads (from the book's first row; with under
`MIN_REPLAY_DAYS` complete days the correlation is not computed, and the block says so), then
lines of two checks that are not paper: the replayed history against frab's weekly `retune`
events (the state at launch), and, with `--events`, the fills and the universes.

Export the paper side first (read-only on the production host):

    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 \\
      "file:data/frab.db?mode=ro" "select params_json from strategies where id=6"' \\
      > trend_retune_params.json
    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \\
      "file:data/frab.db?mode=ro" "select * from trend_equity where strategy_id=6 \\
      order by ts_ms"' > trend_retune_equity.csv
    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \\
      "file:data/frab.db?mode=ro" "select * from trend_events where strategy_id=6 \\
      order by ts_ms"' > trend_retune_events.csv

Usage:

    uv run python scripts/replay_trend_retune_paper.py --paper trend_retune_equity.csv \\
        --params trend_retune_params.json --events trend_retune_events.csv \\
        [--strategy-id 6] [--out replay_trend_retune.csv]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from qlab.data.cap_history import build_cap_history, hyperliquid_perp_names
from qlab.data.sources import coinmarketcap
from qlab.paper_replay import (
    DAY_MS,
    DEFAULT_RAW_DAILY_DIR,
    DEFAULT_RECORD_DIR,
    HOUR_MS,
    compare_curves,
    comparison_lines,
    daily_from_hourly,
    daily_from_raw,
    extend_daily,
    funding_from_raw,
    latest_raw_file,
    merge_funding,
    read_recorded_hourly,
)
from qlab.strategies.live._loader import import_frab

book_mod = import_frab("frab.strategy.trend.book")
retune = import_frab("frab.strategy.trend.retune")
retune_params = import_frab("frab.strategy.trend.retune_params")
backfill = import_frab("frab.strategy.trend.retune_backfill")
cap_history = import_frab("frab.strategy.trend.cap_history")
selector = import_frab("frab.strategy.trend.coin_selector")

SCORE_TOLERANCE = 1e-6
NOT_FILLS = ("fund", "retune", "retune_day", "backfill")


def listed_perps(record_dir: Path) -> set[str]:
    """Perps Hyperliquid listed when the recorder last looked (delisted ones excluded), the
    set frab's live ticks use; the recorder keeps the meta of every run."""
    newest = sorted((record_dir / "meta").glob("*.json"))[-1]
    universe = json.loads(newest.read_text())["universe"]
    return {u["name"] for u in universe if not u.get("isDelisted")}


def load_history(raw_dir: Path, record_dir: Path, since_ms: int, through_ms: int, listed: set[str]
                 ) -> tuple[dict, dict, dict, int]:
    """(daily candles of every perp the stand cached, funding settlements, recorded hourly
    data of the listed perps, newest hour with a candle for most perps).

    Daily candles: the stand's cache, extended with days built from the recorded hourly
    candles (the stand's cache ends at its last night). Funding: the cache plus the
    recorder's settlements."""
    daily, funding, hourly = {}, {}, {}
    for coin in sorted({p.name.split("__")[0] for p in raw_dir.glob("*.parquet")}):
        if coin.endswith("-SPOT"):
            continue
        file = latest_raw_file(raw_dir, coin)
        frame = pd.read_parquet(file)
        daily[coin] = {o: v for o, v in daily_from_raw(frame).items() if o >= since_ms}
        funding[coin] = funding_from_raw(frame)
    newest = []
    for coin in sorted(listed):
        if not (record_dir / "candles" / "1h" / f"{coin}.parquet").is_file():
            continue
        rec = read_recorded_hourly(record_dir, coin)
        hourly[coin] = rec
        newest.append(max(rec.close))
        last_day = (through_ms // DAY_MS) * DAY_MS - DAY_MS
        extra = daily_from_hourly(rec, since_ms, last_day)
        daily[coin] = extend_daily(daily.get(coin, {}), extra)
        funding[coin] = merge_funding(funding.get(coin, []), rec.funding_records)
    newest.sort()
    return daily, funding, hourly, newest[len(newest) // 2]


def cap_history_from_store(store: Path, raw_dir: Path, start: str) -> object:
    """frab's `CapHistory` from q-lab's own CoinMarketCap snapshots."""
    history = coinmarketcap.load_history(store)
    names = hyperliquid_perp_names(raw_dir)
    doc = build_cap_history(history, names, date.fromisoformat(start))
    return cap_history.CapHistory.from_dict(doc)


def logged_universes(events: pd.DataFrame) -> dict[int, dict[str, list[str]]]:
    """`retune_day` universes of frab's live bars, by decision bar."""
    out = {}
    for row in events[events.kind == "retune_day"].itertuples():
        details = json.loads(row.details_json)
        out[int(details["bar_ms"]) if "bar_ms" in details else int(row.ts_ms) - HOUR_MS] = details[
            "universes"]
    return out


def check_history(result, events: pd.DataFrame) -> list[str]:
    """The replayed weekly choices against frab's `retune` events of the backfill."""
    theirs = {}
    for row in events[events.kind == "retune"].itertuples():
        details = json.loads(row.details_json)
        if details.get("phase") == "history":
            theirs[details["week"]] = details
    mine = {e["week"]: e for e in result.events}
    same = off = 0
    worst = 0.0
    for week, details in theirs.items():
        ours = mine.get(week)
        if ours is None:
            continue
        same += ours["choice"] == details["choice"]
        for name, score in details["scores"].items():
            other = ours["scores"].get(name)
            if score is None and other is None:
                continue
            if score is None or other is None:
                off += 1
                continue
            worst = max(worst, abs(score - other))
            off += abs(score - other) > SCORE_TOLERANCE
    return [f"  history before launch: frab {len(theirs)} weekly re-tunings, replay {len(mine)}, "
            f"same choice {same}, scores off by more than {SCORE_TOLERANCE:g}: {off} "
            f"(largest gap {worst:.2e}); last choice {result.state.chosen}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="trend_equity export of the retune books")
    parser.add_argument("--strategy-id", type=int, default=6)
    parser.add_argument("--params", required=True, help="strategies.params_json of the book")
    parser.add_argument("--events", help="trend_events export of the book (universes, fills)")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DAILY_DIR)
    parser.add_argument("--cmc-store", type=Path, default=coinmarketcap.DEFAULT_STORE)
    parser.add_argument("--out", help="write the hour-by-hour totals here (csv)")
    args = parser.parse_args()

    params = retune_params.RetuneParams.from_dict(json.loads(Path(args.params).read_text()))
    trend = params.trend
    paper = pd.read_csv(args.paper)
    paper = paper[paper.strategy_id == args.strategy_id].sort_values("ts_ms")
    if paper.empty:
        raise SystemExit(f"no rows of strategy {args.strategy_id} in {args.paper}")
    events = pd.read_csv(args.events) if args.events else None
    first = int(paper.ts_ms.iloc[0]) - HOUR_MS          # the bar the book was started and sized on
    last_paper = int(paper.ts_ms.iloc[-1]) - HOUR_MS
    since = params.backfill_start_ms

    listed = listed_perps(args.record_dir)
    daily, funding, hourly, recorded_through = load_history(
        args.raw_dir, args.record_dir, since, last_paper, listed)
    last = min(last_paper, recorded_through)
    cap = cap_history_from_store(args.cmc_store, args.raw_dir, params.backfill_start)

    # 1) the candidates' past, by frab's own backfill
    hist = backfill.IndexedHistory(daily, funding, since)
    bars = backfill.history_bars(since, first)
    state = retune.RetuneState.new(params)
    result = backfill.ReplayResult(state=state, last=None)
    for _, outcome, universes in backfill.iter_replay(hist, cap, params, bars, state):
        backfill.absorb(result, outcome, universes)
    launch = result.last

    # 2) the live bars: universes (cap rule from frab's log), then advance_day
    live_series = {c: d for c, d in daily.items() if c in listed}
    logged = logged_universes(events) if events is not None else {}
    outcomes: dict[int, object] = {}
    universe_checks = [0, 0]  # volume universes recomputed, of which equal to the log
    stopped_at = None
    for bar in range(first, last + HOUR_MS, HOUR_MS):
        if (bar // HOUR_MS) % 24 != trend.rebalance_hour_utc:
            continue
        if bar not in logged:
            stopped_at = bar
            last = bar - HOUR_MS
            break
        day0 = retune.build_day(live_series, {}, bar, None)
        universes = selector.select_universes(
            cap_ranked=logged[bar]["cap"], closes_count={c: len(v) for c, v in day0.closes.items()},
            tradeable=set(day0.price_now), volume_usd=day0.volume_usd,
            k_cap=params.top_k_by_market_cap, k_volume=params.top_k_by_volume,
            min_history=trend.min_history_days if params.top_k_requires_history else None)
        universe_checks[0] += 1
        universe_checks[1] += universes["volume"] == logged[bar]["volume"]
        state.cap_snapshot = {"source": "paper_log"}
        day = retune.build_day(live_series, funding, bar, state.last_bar_ms)
        outcomes[bar] = retune.advance_day(state, day, params, universes)
    if launch is not None and first not in outcomes and launch.chosen:
        outcomes[first] = retune.BarOutcome(weights=launch.weights, signals=launch.signals,
                                            size_scale=launch.size_scale, chosen=launch.chosen,
                                            refit=False, events=[])

    # 3) the book, hour by hour, as the engine's tick does
    book = book_mod.TrendBook.new(trend)
    fills = book_mod.start_book(book, bar_ms=first - HOUR_MS, params=trend)
    rows = []
    for h in range(first, last + HOUR_MS, HOUR_MS):
        out = outcomes.get(h)
        held = set(book.units) | (set(out.weights) if out else set())
        prices = {c: hourly[c].close.get(h, book.prices.get(c)) for c in held if c in hourly}
        prices = {c: p for c, p in prices.items() if p}
        rates = {c: hourly[c].funding.get(h, 0.0) for c in held if c in hourly}
        weights = signals = scale = None
        if out is not None:
            weights = {c: w for c, w in out.weights.items() if c in prices}
            signals, scale = out.signals, out.size_scale
        fills += book_mod.step(book, bar_ms=h, prices=prices, funding=rates, params=trend,
                               weights=weights, signals=signals, size_scale=scale)
        rows.append({"ts_ms": h + HOUR_MS, "replay_equity": book.equity(prices),
                     "replay_legs": book.legs(), "replay_fees": book.fees,
                     "replay_funding": book.funding_total})

    replay = pd.DataFrame(rows).merge(
        paper[["ts_ms", "equity", "legs", "fees", "funding_total"]], on="ts_ms")
    replay.index = pd.to_datetime(replay.ts_ms, unit="ms", utc=True)
    live = compare_curves(replay.equity, replay.replay_equity)

    print(f"replay of strategy {args.strategy_id}: history {len(bars)} daily bars to "
          f"{pd.Timestamp(bars[-1], unit='ms', tz='UTC'):%Y-%m-%d}, replay through "
          f"{pd.Timestamp(last, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC "
          f"(paper's last bar {pd.Timestamp(last_paper, unit='ms', tz='UTC'):%Y-%m-%d %H:%M})")
    if stopped_at is not None:
        print(f"  replay stops before the bar of {pd.Timestamp(stopped_at, unit='ms', tz='UTC')}: "
              "frab's log has no universe for it yet")
    for line in comparison_lines(live, prices=False, money=True):
        print(line)
    if live.hours >= 2:
        end = replay.iloc[-1]
        print(f"  legs paper/replay at end {end.legs}/{end.replay_legs}; "
              f"fees {end.fees:.2f}/{end.replay_fees:.2f}; "
              f"funding {end.funding_total:.2f}/{end.replay_funding:.2f}")
    for line in check_history(result, events if events is not None else pd.DataFrame(
            columns=["kind", "details_json"])):
        print(line)
    if events is not None:
        pe = events[~events.kind.isin(NOT_FILLS)].assign(bar_ms=lambda x: x.ts_ms - HOUR_MS)
        pe = pe[pe.bar_ms <= last]
        re = pd.DataFrame([e for e in fills if e["kind"] not in NOT_FILLS])
        if not re.empty and not pe.empty:
            m = pe.merge(re, on=["bar_ms", "coin", "kind"], how="outer",
                         suffixes=("_paper", "_replay"), indicator=True)
            side = m._merge.value_counts()
            both = m[m._merge == "both"]
            qty = ((both.qty_paper - both.qty_replay).abs() / both.qty_paper.abs()).max()
            px = ((both.price_paper - both.price_replay).abs() / both.price_paper).max()
            print(f"  fills matched {side.get('both', 0)}, paper only {side.get('left_only', 0)}, "
                  f"replay only {side.get('right_only', 0)}; max |qty rel diff| {qty:.2e}, "
                  f"max |price rel diff| {px:.2e}")
        print(f"  live bars after launch: {len(outcomes) - (first in outcomes)}; volume "
              f"universes equal to frab's log {universe_checks[1]} of {universe_checks[0]}")
    if args.out:
        out = replay[["ts_ms", "equity", "replay_equity"]].assign(strategy_id=args.strategy_id)
        out[["strategy_id", "ts_ms", "equity", "replay_equity"]].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
