"""Daily recorder: keep the market data a venue will stop serving (docs/TASKS.md T33).

Hyperliquid serves no candles at all for a delisted instrument of a HIP-3
deployment -- all 19 delisted `xyz` instruments return nothing for any
range, though each traded within the last year (docs/STAGE3_XYZ_BAB.md).
And it serves only about the last 5000 candles of any interval, so hourly
history on the main market reaches back roughly seven months
(docs/TASKS.md T25). The past cannot be recovered; this module stops losing
the future: run daily, it appends every closed candle and every funding
settlement the venue serves now to a store of our own, before the venue
drops them.

## Store layout (under `store_dir`, default `data/recorded/`)

    <source>/candles/<interval>/<instrument>.parquet   open/high/low/close/volume/trade_count,
                                                         indexed by candle OPEN time (UTC)
    <source>/funding/<instrument>.parquet               funding rate, indexed by settlement time
    <source>/meta/<YYYY-MM-DD>.json                    the venue's listing as seen that day:
                                                         isDelisted, growthMode, maxLeverage, ...

Full candles are kept, not just the close the backtest panels use: the bar
high is what a liquidation check needs (Bv2), and the open/low cost nothing
extra to keep. The daily listing snapshot records, among other things,
`growthMode` -- the fee discount the API shows only for today
(docs/STAGE3_XYZ_BAB.md), which this turns into a history from now on.

## Rules

- **Only closed candles.** A candle whose period has not ended when the
  recorder runs is not written; the next run writes it closed.
- **Append and overwrite by timestamp, never delete.** A re-fetched candle
  replaces the stored one with the same open time (the venue's last word);
  rows the venue no longer serves are kept -- that is the point.
- **Catch-up, not a fixed window.** Each run fetches from the last stored
  candle onward, so a missed day (laptop asleep) costs nothing as long as the
  instrument is still served when the next run happens.
- **Funding starts where candles start.** On an instrument's first run the
  funding fetch begins at its first recorded candle, not at the venue's
  beginning of time: main-market funding is fully served historically and
  does not need rescuing; the record only needs to cover what it pairs with.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd
import structlog

from qlab.data.sources import hyperliquid as hl
from qlab.data.sources.base import INTERVAL_TO_TIMEDELTA, InstrumentHistory

logger = structlog.get_logger()

DEFAULT_STORE_DIR = Path("data/recorded")
_CANDLE_COLUMNS = ("open", "high", "low", "close", "volume", "trade_count")
# How far back a first run asks for candles. The venue answers with what it
# still has (about 5000 bars); asking further back costs one request.
_FIRST_RUN_FROM = pd.Timestamp("2015-01-01", tz="UTC")


def source_dex(source: str) -> str | None:
    """`hyperliquid` -> None, `hyperliquid-<dex>` -> `<dex>`; anything else is
    not a source this recorder knows how to read."""
    if source == hl.VENUE:
        return None
    prefix = f"{hl.VENUE}-"
    if source.startswith(prefix) and source[len(prefix) :] in hl.HIP3_DEXES:
        return source[len(prefix) :]
    raise ValueError(f"recorder does not support source {source!r}")


@dataclass
class RecordReport:
    source: str
    listed: int = 0
    delisted: int = 0
    new_candles: dict[str, int] = field(default_factory=dict)
    new_funding: int = 0
    failed: list[str] = field(default_factory=list)


def _candle_path(store: Path, source: str, interval: str, instrument: str) -> Path:
    return store / source / "candles" / interval / f"{instrument}.parquet"


def _funding_path(store: Path, source: str, instrument: str) -> Path:
    return store / source / "funding" / f"{instrument}.parquet"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def _write_merged(path: Path, old: pd.DataFrame | None, new: pd.DataFrame) -> int:
    """Merge `new` over `old` by index (new wins on equal timestamps), write
    atomically, return how many timestamps were not in `old`."""
    if new.empty:
        return 0
    added = len(new.index) if old is None else len(new.index.difference(old.index))
    merged = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, engine="pyarrow", index=True)
    tmp.replace(path)
    return added


def fetch_closed_candles(
    client: httpx.Client, coin: str, interval: str, start: pd.Timestamp, now: pd.Timestamp
) -> pd.DataFrame:
    """Every candle the venue serves for `coin` from `start`, full OHLCV, with
    the still-open candle (period not ended at `now`) dropped."""
    step = INTERVAL_TO_TIMEDELTA[interval]
    rows: dict[pd.Timestamp, tuple] = {}
    cursor = start
    while cursor <= now:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": hl._to_ms(cursor),
                "endTime": hl._to_ms(now),
            },
        }
        candles = hl._post(client, payload)
        if not candles:
            break
        last = cursor
        for c in candles:
            opened = pd.Timestamp(int(c["t"]), unit="ms", tz="UTC")
            if opened + step > now:
                continue  # still forming
            rows[opened] = (
                float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]),
                float(c["v"]), int(c["n"]),
            )
            last = max(last, opened)
        if len(candles) < hl._MAX_CANDLES_PER_PAGE:
            break
        cursor = last + step
    index = pd.DatetimeIndex(sorted(rows), name="timestamp")
    frame = pd.DataFrame([rows[t] for t in index], index=index, columns=list(_CANDLE_COLUMNS))
    return frame.astype({"trade_count": "int64"}) if len(frame) else frame


def record(
    source: str,
    intervals: Sequence[str],
    *,
    store_dir: Path = DEFAULT_STORE_DIR,
    now: pd.Timestamp | None = None,
) -> RecordReport:
    """One recorder run for `source` (see module docstring)."""
    dex = source_dex(source)
    intervals = list(dict.fromkeys(intervals))  # a repeated interval is one interval
    for interval in intervals:
        if interval not in INTERVAL_TO_TIMEDELTA:
            raise ValueError(f"unknown interval {interval!r}")
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    report = RecordReport(source=source, new_candles={i: 0 for i in intervals})

    with httpx.Client() as client:
        payload: dict[str, str] = {"type": "metaAndAssetCtxs"}
        if dex is not None:
            payload["dex"] = dex
        meta, ctxs = hl._post(client, payload)
        listing = [
            {**entry, "ctx": ctx} for entry, ctx in zip(meta["universe"], ctxs, strict=False)
        ]
        meta_path = store_dir / source / "meta" / f"{now.date().isoformat()}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps({"recorded_at": now.isoformat(), "universe": listing}, indent=1),
            encoding="utf-8",
        )

        live = [e["name"] for e in meta["universe"] if not e.get("isDelisted")]
        report.listed = len(live)
        report.delisted = len(meta["universe"]) - len(live)

        for coin in live:
            try:
                first_candle: pd.Timestamp | None = None
                for interval in intervals:
                    path = _candle_path(store_dir, source, interval, coin)
                    old = _read(path)
                    start = _FIRST_RUN_FROM if old is None or old.empty else old.index.max()
                    new = fetch_closed_candles(client, coin, interval, start, now)
                    report.new_candles[interval] += _write_merged(path, old, new)
                    stored = _read(path)
                    if stored is not None and not stored.empty:
                        earliest = stored.index.min()
                        first_candle = (
                            earliest if first_candle is None else min(first_candle, earliest)
                        )

                if first_candle is None:
                    continue
                fpath = _funding_path(store_dir, source, coin)
                old_f = _read(fpath)
                f_start = first_candle if old_f is None or old_f.empty else old_f.index.max()
                series = hl.fetch_funding(client, coin, f_start, now)
                new_f = series.rename("funding_rate").to_frame()
                new_f.index.name = "timestamp"
                report.new_funding += _write_merged(fpath, old_f, new_f)
            except Exception as exc:  # noqa: BLE001 - one bad instrument must not stop the run
                logger.warning(
                    "recorder_instrument_failed", source=source, coin=coin, error=str(exc)
                )
                report.failed.append(coin)

    logger.info(
        "recorder_run",
        source=source,
        listed=report.listed,
        delisted=report.delisted,
        new_candles=report.new_candles,
        new_funding=report.new_funding,
        failed=len(report.failed),
    )
    return report


def load_recorded_history(
    source: str,
    instrument: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    store_dir: Path = DEFAULT_STORE_DIR,
    is_delisted: bool,
) -> InstrumentHistory | None:
    """What this recorder kept for `instrument` within `[start, end]`, in the
    shape a source fetcher returns -- or None if nothing was recorded there.
    `qlab.data.snapshot.build_snapshot` asks this for a delisted instrument
    the venue no longer serves, before declaring its history erased."""
    candles = _read(_candle_path(store_dir, source, interval, instrument))
    if candles is None:
        return None
    candles = candles.loc[(candles.index >= start) & (candles.index <= end)]
    if candles.empty:
        return None
    funding_frame = _read(_funding_path(store_dir, source, instrument))
    funding = (
        funding_frame["funding_rate"]
        if funding_frame is not None
        else pd.Series(dtype=float)
    )
    return InstrumentHistory(
        instrument=instrument,
        prices=candles["close"],
        funding=funding,
        volume=candles["volume"],
        trade_count=candles["trade_count"],
        first_seen=candles.index.min(),
        last_seen=candles.index.max(),
        is_delisted=is_delisted,
    )


__all__ = [
    "DEFAULT_STORE_DIR",
    "RecordReport",
    "fetch_closed_candles",
    "load_recorded_history",
    "record",
    "source_dex",
]
