"""Replay the paper book `b2_retune` (frab strategy id 8) on q-lab's own recorded data and
compare, hour by hour, with what the production engine wrote.

`b2_retune` is the monthly re-tuned Bv2 book: two candidates (hedge margin headroom 1.0 and
1.5) run continuously, and on the first bar of every month the one with the better Sharpe so
far is the one traded (stand: `bv2-research-headroom-retune`). This script drives frab's OWN
code (`frab.strategy.b2.retune.advance`, the function its engine calls every closed hour) over
`data/recorded/hyperliquid/` from the history start in the book's parameters, exactly as the
engine's first tick did at launch -- so the replay is in the book's state at launch by
construction, and the hours after it are compared with the paper rows.

Two separate checks are printed:

* **history** (rows up to the launch point): the replay of the book's backfilled past against
  what frab wrote at launch, and the monthly choices against frab's `retune` events. This is
  the check that the replay reproduces frab's state at launch; it is NOT paper and the paper
  rule does not read it.
* **live** (from `--live-from`, the book's `realised_from`): the block `qlab.paper_rule`
  reads. With under `MIN_REPLAY_DAYS` complete days the correlation is not computed and the
  block says so.

Export the paper side first (read-only on the production host):

    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \\
      "file:data/frab.db?mode=ro" "select strategy_id, ts_ms, coin, price, equity, \\
      book_equity, cash, spot_value, short_pnl, hedge_on from b2_equity \\
      order by strategy_id, coin, ts_ms"' > b2_equity.csv
    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 \\
      "file:data/frab.db?mode=ro" "select params_json from strategies where id=8"' \\
      > b2_retune_params.json
    ssh dis@10.8.0.5 'cd ~/prj/funding-rate-arbitrage && sqlite3 -header -csv \\
      "file:data/frab.db?mode=ro" "select ts_ms, details_json from b2_events \\
      where strategy_id=8 and kind=\\"retune\\" order by ts_ms"' > b2_retune_events.csv

Usage:

    uv run python scripts/replay_b2_retune_paper.py --paper b2_equity.csv \\
        --params b2_retune_params.json --events b2_retune_events.csv \\
        --live-from 2026-10-25T14:00:00+00:00 [--out replay_b2_retune.csv]

`--out` writes the hour-by-hour totals (strategy_id, ts_ms, equity, replay_equity) of every
hour both sides have, history included, which `qlab.paper_check` cuts to its window.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from qlab.paper_replay import (
    DEFAULT_RECORD_DIR,
    HOUR_MS,
    compare_curves,
    comparison_lines,
    cut_live,
    on_grid,
    read_recorded_hourly,
)
from qlab.strategies.live._loader import import_frab

retune = import_frab("frab.strategy.b2.retune")
retune_params = import_frab("frab.strategy.b2.retune_params")
retune_history = import_frab("frab.strategy.b2.retune_history")

BOOK_ID = 8
SCORE_TOLERANCE = 1e-6  # a re-tuning score is a Sharpe near 0.01; agree to this much


def build_series(params, record_dir: Path, through_ms: int) -> dict:
    """frab's `CoinSeries` of every coin on the hourly grid from the book's history start,
    from the recorder's store; a hole is NaN, as in frab's own history store."""
    origin = params.history_start_ms
    series = {}
    for coin in params.coins:
        rec = read_recorded_hourly(record_dir, coin)
        last = min(through_ms, max(rec.close))
        series[coin] = retune_history.CoinSeries(
            coin, origin, on_grid(rec.close, origin, last), on_grid(rec.high, origin, last),
            on_grid(rec.funding, origin, last))
    return series


def run_replay(params, series: dict, last_bar_ms: int) -> tuple[pd.DataFrame, list[dict], int]:
    """Step frab's `advance` over every bar from the history start; returns the published
    per-coin rows, the monthly `retune` events and the last bar actually stepped."""
    steppable = [s.ready_through_ms() for s in series.values()]
    if any(r is None for r in steppable):
        raise SystemExit("the recorded store has no steppable bar for some coin")
    last = min(last_bar_ms, *steppable)
    state = retune.RetuneState.new(params)
    rows, choices = [], []
    for bar in range(params.history_start_ms, last + HOUR_MS, HOUR_MS):
        out = retune.advance(state, params, series, bar)
        rows += out.equity_rows
        choices += [e for e in out.events if e["kind"] == "retune"]
    return pd.DataFrame(rows), choices, last


def total_curve(frame: pd.DataFrame, column: str) -> pd.Series:
    """The book's equity by timestamp: the sum of its coins, a timestamp where a coin is
    missing dropped rather than summed short."""
    grouped = frame.groupby("ts_ms")[column].agg(["sum", "count"])
    grouped = grouped[grouped["count"] == grouped["count"].max()]["sum"].sort_index()
    grouped.index = pd.to_datetime(grouped.index, unit="ms", utc=True)
    return grouped.astype(float)


def check_choices(replay_choices: list[dict], events_csv: Path) -> list[str]:
    """The replay's monthly choices against frab's `retune` events."""
    paper = pd.read_csv(events_csv)
    theirs = [(int(r.ts_ms), json.loads(r.details_json)) for r in paper.itertuples()]
    mine = {e["refit"]: e for e in replay_choices}
    agree = score_gap = 0
    worst = 0.0
    for _, details in theirs:
        ours = mine.get(details["refit"])
        if ours is None:
            continue
        agree += ours["choice"] == details["choice"]
        for name, score in details["scores"].items():
            a, b = score["score"], ours["scores"].get(name, {}).get("score")
            if a is None and b is None:
                continue
            if a is None or b is None:
                score_gap += 1
                continue
            worst = max(worst, abs(a - b))
            score_gap += abs(a - b) > SCORE_TOLERANCE
    return [f"  history re-tunings: frab {len(theirs)}, replay {len(mine)}, same choice {agree}, "
            f"scores off by more than {SCORE_TOLERANCE:g}: {score_gap} "
            f"(largest gap {worst:.2e})"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="b2_equity export from frab.db")
    parser.add_argument("--params", required=True, help="strategies.params_json of b2_retune")
    parser.add_argument("--events", help="b2_events export (kind retune) of the book")
    parser.add_argument("--live-from", required=True,
                        help="the book's realised_from: the launch point, ISO with offset")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--out", help="write the hour-by-hour totals here (csv)")
    args = parser.parse_args()

    live_from = pd.Timestamp(datetime.fromisoformat(args.live_from))
    params = retune_params.B2RetuneParams.from_dict(json.loads(Path(args.params).read_text()))
    paper = pd.read_csv(args.paper)
    paper = paper[paper.strategy_id == BOOK_ID]
    if paper.empty:
        raise SystemExit(f"no rows of strategy {BOOK_ID} in {args.paper}")
    last_paper_bar = int(paper.ts_ms.max()) - HOUR_MS

    series = build_series(params, args.record_dir, last_paper_bar)
    replay, choices, last = run_replay(params, series, last_paper_bar)

    merged = replay.merge(paper[["ts_ms", "coin", "equity", "price"]], on=["ts_ms", "coin"],
                          suffixes=("_replay", "_paper"))
    merged["live"] = pd.to_datetime(merged.ts_ms, unit="ms", utc=True) >= live_from
    live_rows = merged[merged.live]
    mismatched = int((abs(live_rows.price_replay - live_rows.price_paper) > 1e-9).sum())

    paper_curve = total_curve(merged.rename(columns={"equity_paper": "e"}), "e")
    replay_curve = total_curve(merged.rename(columns={"equity_replay": "e"}), "e")
    history = pd.concat({"p": paper_curve, "r": replay_curve}, axis=1).dropna()
    old = history[history.index < live_from]
    live = compare_curves(cut_live(paper_curve, live_from), cut_live(replay_curve, live_from),
                          price_mismatches=mismatched)

    print(f"book {BOOK_ID} (b2_retune: headroom 1.0 or 1.5 chosen monthly; "
          f"live from {live_from:%Y-%m-%d %H:%M} UTC)")
    print(f"  replay stepped through {pd.Timestamp(last, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC "
          f"(paper's last bar {pd.Timestamp(last_paper_bar, unit='ms', tz='UTC'):%Y-%m-%d %H:%M})")
    for line in comparison_lines(live):
        print(line)
    if not old.empty:
        diff = (old.p - old.r).abs()
        rel = float((diff / old.p).max())
        paper_total = old.p.iloc[-1] / old.p.iloc[0] - 1
        replay_total = old.r.iloc[-1] / old.r.iloc[0] - 1
        print(f"  history before launch: {len(old)} hours both sides, paper {paper_total:+.2%} "
              f"replay {replay_total:+.2%}, max |hourly equity diff| {diff.max():.4f} "
              f"({rel:.2%} of equity)")
    if args.events:
        for line in check_choices(choices, Path(args.events)):
            print(line)
    if choices:
        print(f"  history last choice (refit {choices[-1]['refit']}): {choices[-1]['choice']}")
    if args.out:
        out = pd.concat({"equity": paper_curve, "replay_equity": replay_curve}, axis=1).dropna()
        out = out.assign(strategy_id=BOOK_ID, ts_ms=[int(t.timestamp() * 1000) for t in out.index])
        out[["strategy_id", "ts_ms", "equity", "replay_equity"]].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
