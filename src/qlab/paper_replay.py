"""Shared pieces of the replay scripts of the re-tuned paper books
(`scripts/replay_b2_retune_paper.py`, `scripts/replay_trend_retune_paper.py`).

The scripts drive frab's own live code on q-lab's own recorded data and compare, hour by
hour, with the equity the production engine wrote. What is common to both lives here, as
plain functions without frab or the network:

* reading q-lab's recorded hourly store (`data/recorded/hyperliquid`) as millisecond
  dictionaries keyed the way frab's engines key them (funding floored to the hour), and the
  stand's cached daily candles (`data/raw/hyperliquid/1d`) extended by daily candles built
  from the recorded hourly ones;
* comparing a curve with its replay: matched hours, total returns, the largest hourly
  difference and the correlation of daily returns;
* the correlation is computed on COMPLETE UTC days (midnight to midnight) and only from
  `MIN_REPLAY_DAYS` of them: before that the block says so plainly and prints no figure,
  because two or three days correlate by luck. `qlab.paper_rule` reads the block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qlab.paper_rule import MIN_REPLAY_DAYS

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
DEFAULT_RECORD_DIR = Path("data/recorded/hyperliquid")
DEFAULT_RAW_DAILY_DIR = Path("data/raw/hyperliquid/1d")


def ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def floor_hour(value_ms: int) -> int:
    return value_ms - value_ms % HOUR_MS


# ── recorded hourly store ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RecordedHourly:
    """One coin's recorded hourly candles and funding, keyed by the hour's open (ms)."""

    close: dict[int, float]
    high: dict[int, float]
    volume: dict[int, float]
    funding: dict[int, float]                  # floored to the hour, as frab's engines key it
    funding_records: list[tuple[int, float]]   # (settlement ms, rate) as the venue stamped them


def read_recorded_hourly(record_dir: Path, coin: str) -> RecordedHourly:
    """The recorder's candles and funding of `coin`. A funding settlement stamped a few ms
    after the hour belongs to that hour's bar (frab floors it the same way); when two fall in
    one hour the first stays, as in frab's history store."""
    candles = pd.read_parquet(record_dir / "candles" / "1h" / f"{coin}.parquet")
    funding_frame = pd.read_parquet(record_dir / "funding" / f"{coin}.parquet")
    funding: dict[int, float] = {}
    records: list[tuple[int, float]] = []
    for stamp, rate in funding_frame["funding_rate"].items():
        funding.setdefault(floor_hour(ms(stamp)), float(rate))
        records.append((ms(stamp), float(rate)))
    return RecordedHourly(
        close={ms(t): float(v) for t, v in candles["close"].items()},
        high={ms(t): float(v) for t, v in candles["high"].items()},
        volume={ms(t): float(v) for t, v in candles["volume"].items()},
        funding=funding,
        funding_records=records,
    )


def on_grid(values: dict[int, float], origin_ms: int, last_ms: int) -> list[float]:
    """`values` on the hourly grid `origin_ms .. last_ms` (inclusive), NaN where unknown."""
    n = (last_ms - origin_ms) // HOUR_MS + 1
    out = [math.nan] * n
    for stamp, value in values.items():
        i = (stamp - origin_ms) // HOUR_MS
        if 0 <= i < n and (stamp - origin_ms) % HOUR_MS == 0:
            out[i] = value
    return out


# ── daily candles: the stand's cache, extended by the recorder's hourly ones ─

Daily = dict[int, tuple[float, float | None]]  # day open (ms) -> (close, base volume)


def daily_from_raw(frame: pd.DataFrame) -> Daily:
    """Daily candles from a cached raw frame (`data/raw/hyperliquid/1d/<coin>__*.parquet`:
    `price` is the close; the funding settlements are rows of their own without a price)."""
    rows = frame[frame["price"].notna()]
    return {ms(t): (float(price), None if volume != volume else float(volume))
            for t, price, volume in zip(rows.index, rows["price"], rows["volume"], strict=True)}


def funding_from_raw(frame: pd.DataFrame) -> list[tuple[int, float]]:
    """Funding settlements `(ms, rate)` from a cached raw frame."""
    rows = frame["funding"].dropna()
    return [(ms(t), float(v)) for t, v in rows.items()]


def daily_from_hourly(rec: RecordedHourly, first_day_ms: int, last_day_ms: int) -> Daily:
    """Daily candles built from the recorded hourly ones, for every COMPLETE UTC day in
    `first_day_ms .. last_day_ms` (all 24 hourly candles present): the close of the last
    hour, the sum of the hourly base volumes. On the days both exist this equals the
    venue's own daily candle (checked on 2026-10-25: five coins, 231 days each, closes
    identical, volumes to 1e-16)."""
    out: Daily = {}
    for day in range(first_day_ms, last_day_ms + DAY_MS, DAY_MS):
        hours = [day + h * HOUR_MS for h in range(24)]
        if all(h in rec.close for h in hours):
            out[day] = (rec.close[hours[-1]], sum(rec.volume.get(h, 0.0) for h in hours))
    return out


def extend_daily(daily: Daily, extra: Daily) -> Daily:
    """`daily` plus the days of `extra` it does not have (the stand's cache wins where both do)."""
    return {**extra, **daily}


def merge_funding(*sources: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Funding settlements of several sources, one per timestamp (the earlier source wins),
    sorted by time."""
    seen: dict[int, float] = {}
    for source in sources:
        for stamp, rate in source:
            seen.setdefault(stamp, rate)
    return sorted(seen.items())


def latest_raw_file(raw_dir: Path, coin: str) -> Path | None:
    """The newest cached daily file of `coin` (`<coin>__<start>__<end>.parquet`; the end date
    sorts as text), None when the stand never fetched it."""
    files = sorted(raw_dir.glob(f"{coin}__*.parquet"))
    return files[-1] if files else None


# ── comparing a replay with the paper book ──────────────────────────────────

def daily_returns(curve: pd.Series) -> pd.Series:
    """Returns from one UTC midnight to the next, for COMPLETE days only. `curve` is indexed
    by UTC timestamps; a day with either midnight missing contributes nothing."""
    at_midnight = curve[(curve.index.hour == 0) & (curve.index.minute == 0)]
    at_midnight = at_midnight[~at_midnight.index.duplicated(keep="last")].sort_index()
    gap = at_midnight.index.to_series().diff()
    returns = at_midnight.pct_change()
    return returns[gap == pd.Timedelta(days=1)].dropna()


@dataclass(frozen=True)
class Comparison:
    """A paper curve against its replay over the same hours."""

    hours: int                  # points in the compared curve (the launch point included)
    paper_return: float | None
    replay_return: float | None
    max_abs_diff: float | None
    days: int                   # complete UTC days with a return on both sides
    correlation: float | None   # None when `days` is under the minimum or a side is constant
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    paper_first: float | None = None
    paper_last: float | None = None
    replay_first: float | None = None
    replay_last: float | None = None
    price_mismatches: int | None = None


def compare_curves(paper: pd.Series, replay: pd.Series, *, min_days: int = MIN_REPLAY_DAYS,
                   price_mismatches: int | None = None) -> Comparison:
    """Compare two equity curves (UTC-indexed) on the hours both have."""
    both = pd.concat({"paper": paper, "replay": replay}, axis=1).dropna().sort_index()
    if both.empty:
        return Comparison(0, None, None, None, 0, None, price_mismatches=price_mismatches)
    paper_ret = replay_ret = None
    if len(both) >= 2 and both.paper.iloc[0] and both.replay.iloc[0]:
        paper_ret = float(both.paper.iloc[-1] / both.paper.iloc[0] - 1)
        replay_ret = float(both.replay.iloc[-1] / both.replay.iloc[0] - 1)
    daily = pd.concat({"paper": daily_returns(both.paper), "replay": daily_returns(both.replay)},
                      axis=1).dropna()
    corr = None
    if len(daily) >= max(min_days, 2) and daily.paper.std() > 0 and daily.replay.std() > 0:
        value = daily.paper.corr(daily.replay)
        corr = None if value != value else float(value)
    return Comparison(
        hours=len(both), paper_return=paper_ret, replay_return=replay_ret,
        max_abs_diff=float((both.paper - both.replay).abs().max()), days=len(daily),
        correlation=corr, first=both.index[0], last=both.index[-1],
        paper_first=float(both.paper.iloc[0]), paper_last=float(both.paper.iloc[-1]),
        replay_first=float(both.replay.iloc[0]), replay_last=float(both.replay.iloc[-1]),
        price_mismatches=price_mismatches)


def cut_live(curve: pd.Series, live_from: pd.Timestamp) -> pd.Series:
    """The part of a curve from the launch point on (the launch row itself is the base)."""
    return curve[curve.index >= live_from]


def _correlation_line(c: Comparison, min_days: int) -> str:
    if c.correlation is not None:
        return f"daily return correlation {c.correlation:.4f} (n={c.days})"
    why = (f"n={c.days} complete UTC days, at least {min_days} needed" if c.days < min_days
           else "one side has no variation")
    return f"daily return correlation not computed: {why}"


def comparison_lines(c: Comparison, *, min_days: int = MIN_REPLAY_DAYS, prices: bool = True,
                     money: bool = False) -> list[str]:
    """The block body `qlab.paper_rule.parse_replay_lines` reads, indented two spaces.
    Every figure is of the LIVE part. Where there is none, or too little, the lines say so
    and print no number the rule could take for a match.

    `money`: the totals as the trend script prints them (`paper 2999.64 -> 2999.40 (-0.01%)`),
    otherwise as the Bv2 script does (`paper  -0.01%   replay -0.01%`)."""
    head = f"  matched hours {c.hours}"
    if prices:
        head += f", price mismatches {c.price_mismatches or 0}"
    lines = [head]
    if c.hours < 2:
        return lines + [
            "  paper and replay totals: not yet, no live hour has closed after the launch point",
            f"  daily return correlation not computed: n=0 complete UTC days, "
            f"at least {min_days} needed",
        ]
    if money:
        lines.append(f"  paper  {c.paper_first:.2f} -> {c.paper_last:.2f} ({c.paper_return:+.2%})")
        lines.append(f"  replay {c.replay_first:.2f} -> {c.replay_last:.2f} "
                     f"({c.replay_return:+.2%})")
    else:
        lines.append(f"  paper  {c.paper_return:+.2%}   replay {c.replay_return:+.2%}")
    lines.append(f"  max |hourly equity diff| {c.max_abs_diff:.4f}")
    lines.append(f"  {_correlation_line(c, min_days)}")
    return lines
