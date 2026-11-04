"""Shared types and helpers for market-data source fetchers.

Both ``hyperliquid.py`` and ``binance.py`` return the same shape of output
(a mapping of instrument -> ``InstrumentHistory``) so ``qlab.data.snapshot``
can assemble a ``MarketPanel`` without caring which venue the data came
from.
"""

from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Fixed delay between successive HTTP requests to a free public endpoint.
# A whole-universe fetch is hundreds of instruments and several paginated
# calls each, which is well past the "handful of instruments" this was first
# sized for — that combination is what produced 429s in practice, so the
# delay is paced for the bulk case and the retry policy below carries the
# rest.
REQUEST_DELAY_SECONDS = 0.5

# A rate-limited endpoint needs to be waited out, not hammered: five tries
# capped at eight seconds cannot outlast a per-minute limit window.
RATE_LIMIT_MAX_ATTEMPTS = 8
RATE_LIMIT_MAX_WAIT_SECONDS = 60.0

INTERVAL_TO_TIMEDELTA: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

# Column-naming convention for a coin's spot market, shared by every source
# fetcher and by `qlab.data.snapshot`/`qlab.data.panel`: a perpetual keeps
# its bare ticker (``BTC``), the SAME coin's spot market is that ticker plus
# this suffix (``BTC-SPOT``). One convention, one place, so a panel can hold
# both instruments for the same coin as two unambiguous columns (exactly
# what a spot-vs-perp pair trade needs) without any module inventing its
# own spelling.
SPOT_COLUMN_SUFFIX = "-SPOT"

SECOND_VENUE_SEPARATOR = "@"
"""``<COIN>@<VENUE>`` names a perpetual of the same coin on a venue other than the
panel's source (upper case, so a requested name is found again by
`_find_matching_snapshot`, which upper-cases what it is given)."""


def second_venue_column(coin: str, venue: str) -> str:
    return f"{coin}{SECOND_VENUE_SEPARATOR}{venue.upper()}"


def parse_second_venue(column: str) -> tuple[str, str] | None:
    """``(coin, venue)`` of a second-venue column, the venue lower-cased as in
    `_SOURCES`; None for a column without the separator."""
    coin, separator, venue = column.rpartition(SECOND_VENUE_SEPARATOR)
    if not separator or not coin or not venue:
        return None
    return coin, venue.lower()


@dataclass(frozen=True)
class InstrumentHistory:
    """Raw per-instrument history as returned by a source fetcher.

    ``prices``/``funding`` are raw series indexed by UTC timestamps as
    returned by the venue — ``funding`` at the venue's *native* settlement
    frequency, not yet resampled onto the panel's chosen interval (that is
    ``snapshot.py``'s job, via ``align_funding_to_index``).

    ``volume`` (docs/TASKS.md, T27): the candle's own traded-volume field,
    in BASE-ASSET units (e.g. BTC, not USD), same index as ``prices`` —
    every source fetcher parses it from the exact same candle response it
    already parses price from, so there is no separate network cost. USD
    volume is deliberately NOT stored here or anywhere downstream
    (``qlab.data.panel.MarketPanel`` included): it is always recomputed as
    ``volume * price`` at the point of use (see
    `qlab.harness.capacity.book_daily_volume_usd`), because ``prices`` gets
    cleaned after this point (``snapshot.py`` NaNs out corrupted prints via
    `detect_bad_price_bars`) — storing a USD figure here would freeze it
    against a price that a later step might still correct, and the two
    would silently drift out of sync.

    ``trade_count``: the candle's fill count (``"n"`` on Hyperliquid,
    ``numberOfTrades`` on Binance), same index as ``prices``. Kept for raw
    fidelity — it costs nothing to retain since it comes from the same
    candle row as price/volume — but nothing downstream consumes it yet;
    it exists so a future sanity check (e.g. "large volume, implausibly few
    fills") has the data available without a second fetch.

    Genuine zero volume (a bar with no trades) and UNKNOWN volume are
    different facts and must stay distinguishable end to end — this is the
    same class of error `qlab.harness.accrual` already guards for funding
    (an unknown rate is not a zero rate). A raw fetch never produces an
    unknown-volume bar for a timestamp it actually returns (the venue
    always includes ``v``/``n`` on every candle row); "unknown" only
    happens where a raw cache entry predates volume support at all — see
    `load_cached_history`, which detects that case explicitly and forces a
    re-fetch rather than returning a frame with a fabricated zero.

    ``first_seen``/``last_seen`` bound the instrument's tradeable life
    within the requested window. ``is_delisted`` says whether that life
    ended because the instrument was actually delisted (so
    ``tradeable`` should read False after ``last_seen``) as opposed to the
    requested window simply ending while it was still listed (so
    ``tradeable`` stays True through the end of the panel).

    ``has_funding`` distinguishes two reasons ``funding`` can be NaN, which
    ``snapshot.py``'s tradeability rule must not treat the same way:

    - ``True`` (the default — every perpetual): a NaN bar is a genuine gap
      in the venue's settlement feed. Holding through it is not honestly
      simulable, so ``snapshot.py`` marks that bar untradeable.
    - ``False`` (a spot market, which never pays or charges funding by
      construction): ``funding`` is empty/all-NaN for the instrument's
      entire life, and that is the correct, expected shape of the data —
      not a gap. Applying the same "unknown funding -> untradeable" rule
      here would make every spot bar untradeable, which is exactly the trap
      this field exists to avoid (see docs/TASKS.md, T17).
    """

    instrument: str
    prices: pd.Series
    funding: pd.Series
    volume: pd.Series
    trade_count: pd.Series
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    is_delisted: bool
    has_funding: bool = True
    # The bar's high (Bv2's liquidation check runs on it). None when the
    # source or an older cache entry did not provide it: unknown, never the
    # close (a short is liquidated by the move INSIDE the bar).
    high: pd.Series | None = None


# Where per-instrument raw history is cached between attempts. Collecting a
# whole universe is thousands of requests over roughly an hour; without this,
# a single 500 on the last one throws the hour away. It is not an
# optimisation — at that request count a clean run is the unlikely outcome,
# so an unresumable fetch never finishes at all.
DEFAULT_RAW_CACHE_DIR = Path("data/raw")


def _cache_paths(
    cache_dir: Path, source: str, instrument: str, interval: str, start, end
) -> tuple[Path, Path]:
    stem = f"{instrument}__{pd.Timestamp(start).date()}__{pd.Timestamp(end).date()}"
    folder = cache_dir / source / interval
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def load_cached_history(
    cache_dir: Path, source: str, instrument: str, interval: str, start, end
) -> InstrumentHistory | None:
    """Return a previously fetched history, or None. Never raises: a corrupt
    or half-written cache entry is treated as absent and refetched.

    A cache entry written before volume support existed (docs/TASKS.md,
    T27) has ``price``/``funding`` columns only. That is detected here
    EXPLICITLY (``"volume" not in frame.columns``) and treated exactly like
    a missing file -- return None so `fetch_universe_resumable` re-fetches
    from the venue. The alternative -- backfilling a missing ``volume``
    column with 0.0 -- would silently turn "we never asked the venue" into
    "the venue reported no trading", which is a different, false claim
    (see `InstrumentHistory`'s docstring: unknown volume is not zero
    volume, the same class of bug `qlab.harness.accrual` guards against for
    funding). This check must be an explicit column test, not incidental
    reliance on `frame["volume"]` raising inside the broad `except`
    below -- that would happen to work today but silently stop working the
    moment this function's error handling changes shape.
    """
    frame_path, meta_path = _cache_paths(cache_dir, source, instrument, interval, start, end)
    if not frame_path.exists() or not meta_path.exists():
        return None
    try:
        frame = pd.read_parquet(frame_path)
        if "volume" not in frame.columns or "trade_count" not in frame.columns:
            return None
        meta = json.loads(meta_path.read_text())
        prices = frame["price"].dropna()
        funding = frame["funding"].dropna()
        volume = frame["volume"].dropna()
        trade_count = frame["trade_count"].dropna().astype("int64")
        high = frame["high"].dropna() if "high" in frame.columns else None
        return InstrumentHistory(
            instrument=instrument,
            prices=prices,
            funding=funding,
            volume=volume,
            trade_count=trade_count,
            first_seen=pd.Timestamp(meta["first_seen"]),
            last_seen=pd.Timestamp(meta["last_seen"]),
            is_delisted=bool(meta["is_delisted"]),
            # Older cache entries (written before spot support existed)
            # have no "has_funding" key at all -- they are all perp
            # fetches, so True is the correct backfill, not a guess.
            has_funding=bool(meta.get("has_funding", True)),
            high=high,
        )
    except Exception:
        return None


def store_cached_history(
    cache_dir: Path, source: str, interval: str, start, end, history: InstrumentHistory
) -> None:
    """Persist one instrument's history. Written frame-first, metadata last,
    so an interrupted write leaves an entry that load_cached_history rejects
    rather than one it trusts."""
    frame_path, meta_path = _cache_paths(
        cache_dir, source, history.instrument, interval, start, end
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "price": history.prices,
        "funding": history.funding,
        "volume": history.volume,
        "trade_count": history.trade_count,
    }
    if history.high is not None:
        columns["high"] = history.high
    frame = pd.DataFrame(columns)
    frame.to_parquet(frame_path)
    meta_path.write_text(
        json.dumps(
            {
                "first_seen": history.first_seen.isoformat(),
                "last_seen": history.last_seen.isoformat(),
                "is_delisted": history.is_delisted,
                "has_funding": history.has_funding,
            }
        )
    )


def fetch_universe_resumable(
    instruments,
    start,
    end,
    interval: str,
    *,
    source: str,
    fetch_one: Callable[[str], InstrumentHistory],
    on_missing: str = "raise",
    cache_dir: Path | str = DEFAULT_RAW_CACHE_DIR,
    accept_cached: Callable[[InstrumentHistory], bool] | None = None,
    fetch_range: Callable[[str, pd.Timestamp, pd.Timestamp], InstrumentHistory] | None = None,
    funding_end_pad: pd.Timedelta | None = None,
) -> dict[str, InstrumentHistory]:
    """Fetch each instrument, reusing anything already cached from an earlier
    attempt and persisting each success as it lands.

    `funding_end_pad`, when given, lets a request be answered from an entry cached from the
    same start to a LATER end (`_cut_later_cache`): the source fetches candles up to `end` and
    funding up to `end + funding_end_pad`, so the later entry holds the same bars and the same
    settlements plus the ones after, and cutting it gives what a fetch would. Only daily bars
    qualify (a source that serves a capped number of recent candles may have served the old
    entry more history than it would serve now). Without it, an experiment on a window ending
    before the nightly one fetched every instrument again -- two hours for 222 instruments of
    Hyperliquid funding (night of 2026-10-29).

    `fetch_range(instrument, start, end)`, when given, lets a request whose
    end moved forward extend an entry cached to an earlier end instead of
    refetching it whole (`_extend_earlier_cache`).

    `accept_cached`, when given, decides whether a cached entry is complete
    enough to reuse; a rejected one is fetched again (and its replacement
    overwrites it). Used for entries written before a field existed -- bar
    highs, here -- when the source can supply it.

    `on_missing` decides what an instrument with no data in range means, and
    that depends on where the list came from: `"raise"` for a hand-written
    list, where it is a typo; `"skip"` for a venue-discovered one, where it
    just means the instrument did not exist during the window.
    """
    if on_missing not in {"raise", "skip"}:
        raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")
    cache_dir = Path(cache_dir)

    def one(instrument: str) -> InstrumentHistory | None:
        cached = load_cached_history(cache_dir, source, instrument, interval, start, end)
        if cached is not None and (accept_cached is None or accept_cached(cached)):
            return cached
        assembled = _assemble_from_cache(cache_dir, source, instrument, interval, start, end,
                                         fetch_range, accept_cached, funding_end_pad)
        if assembled is not None:
            history, used = assembled
            store_cached_history(cache_dir, source, interval, start, end, history)
            _drop_superseded(cache_dir, source, instrument, interval, start, end)
            _drop_contained(cache_dir, source, instrument, interval, start, end, used)
            return history
        try:
            history = fetch_one(instrument)
        except ValueError:
            if on_missing == "raise":
                raise
            return None
        store_cached_history(cache_dir, source, interval, start, end, history)
        return history

    names = list(instruments)
    workers = min(fetch_workers(source), len(names))
    if workers > 1:
        # Each instrument is its own pair of requests and its own cache files, so they can
        # overlap; the answer is the same and keeps the order of `instruments`.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qlab-fetch") as pool:
            found = list(pool.map(one, names))
    else:
        found = [one(name) for name in names]
    return {name: history for name, history in zip(names, found, strict=True)
            if history is not None}


def _merge_series(old: pd.Series | None, new: pd.Series | None, cut: pd.Timestamp
                  ) -> pd.Series | None:
    if old is None and new is None:
        return None
    old_part = old[old.index < cut] if old is not None else pd.Series(dtype=float)
    new_part = new if new is not None else pd.Series(dtype=float)
    merged = pd.concat([old_part, new_part])
    # A fetch from `cut` can still return entries stamped just before it
    # (funding settlements land a few ms around their mark): the newer one
    # wins, and no timestamp appears twice.
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def merge_histories(old: InstrumentHistory, new: InstrumentHistory, cut: pd.Timestamp
                    ) -> InstrumentHistory:
    """`old` up to `cut`, `new` from `cut` on (the newer fetch wins where they
    overlap: the old entry's last bar may have been written before it
    closed)."""
    high = None
    if old.high is not None or new.high is not None:
        high = _merge_series(old.high if old.high is not None else old.prices * float("nan"),
                             new.high if new.high is not None else new.prices * float("nan"),
                             cut)
    return InstrumentHistory(
        instrument=old.instrument,
        prices=_merge_series(old.prices, new.prices, cut),
        funding=_merge_series(old.funding, new.funding, cut),
        volume=_merge_series(old.volume, new.volume, cut),
        trade_count=_merge_series(old.trade_count, new.trade_count, cut),
        first_seen=min(old.first_seen, new.first_seen),
        last_seen=max(old.last_seen, new.last_seen),
        is_delisted=new.is_delisted,
        has_funding=old.has_funding,
        high=high,
    )


def _drop_superseded(cache_dir, source, instrument, interval, start, end) -> None:
    """Remove entries of the same instrument and start cached to an earlier
    end: the extended entry holds everything they did (snapshots, which keep
    their own copy of the data, are untouched)."""
    folder = Path(cache_dir) / source / interval
    prefix = f"{instrument}__{pd.Timestamp(start).date()}__"
    keep = str(pd.Timestamp(end).date())
    for path in folder.glob(f"{glob.escape(prefix)}*"):
        stem = path.name[len(prefix):].split(".")[0]
        if stem < keep:
            path.unlink()


# --------------------------------------------------------------------------
# Any cached window of an instrument answers a request (owner, 2026-11-04: "why re-download the
# same data all the time -- why can't we use what is already in the base?"). A file is keyed by
# (instrument, start, end), but what it HOLDS is every bar of that window the venue had, so a
# request is answered by the union of the instrument's entries plus a fetch of what the union
# does not cover: a head before the earliest entry, a gap between two entries, a tail after the
# latest one.


@dataclass(frozen=True)
class CachedWindow:
    """One cache entry of an instrument, as its file name and metadata say."""

    start: pd.Timestamp
    end: pd.Timestamp
    is_delisted: bool = False


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _day(value) -> pd.Timestamp:
    return _utc(value).normalize()


def cached_windows(
    cache_dir: Path | str, source: str, instrument: str, interval: str, *, verify: bool = False
) -> list[CachedWindow]:
    """The instrument's cache entries, earliest end last. By file name alone (fast, the estimate
    of a fetch); with `verify`, an entry must also have both its files, readable metadata and the
    volume and trade-count columns (an entry from before volume support is unknown volume, not
    zero, and is refetched -- see `load_cached_history`), and carries its `is_delisted`."""
    folder = Path(cache_dir) / source / interval
    out = []
    for path in folder.glob(f"{glob.escape(instrument)}__*.parquet"):
        parts = path.stem.rsplit("__", 2)
        if len(parts) != 3 or parts[0] != instrument:
            continue
        try:
            first, last = _day(parts[1]), _day(parts[2])
        except ValueError:
            continue
        delisted = False
        if verify:
            try:
                meta = json.loads(path.with_suffix(".json").read_text())
                names = pq.read_schema(path).names
            except (OSError, ValueError):
                continue
            if "volume" not in names or "trade_count" not in names:
                continue
            delisted = bool(meta.get("is_delisted", False))
        out.append(CachedWindow(first, last, delisted))
    return sorted(out, key=lambda w: (w.end, w.start))


# A source that serves only its most recent candles (Hyperliquid, `SERVED_CANDLES_CAP`): an old
# hourly entry may hold bars a fetch would not return now, so a window ending before an entry's
# end is fetched, not cut from it. Daily bars are not capped.
_CAPPED_SOURCE_PREFIX = "hyperliquid"


def _cut_later_allowed(source: str, interval: str) -> bool:
    return interval == "1d" or not source.startswith(_CAPPED_SOURCE_PREFIX)


def missing_ranges(
    cache_dir: Path | str, source: str, instrument: str, interval: str, start, end, *,
    verify: bool = False, windows: list[CachedWindow] | None = None,
    cut_later: bool | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The parts of [start, end] that no cached window of the instrument covers, as the ranges
    to fetch. A window covers from the START it was requested from, not from its first bar: an
    entry requested from S whose first bar is later proves the venue had nothing in between, so
    a request from before S needs a head only for [start, S). A range that begins where an entry
    ends begins a day earlier (the entry's last bars may have been written before they closed).
    A delisted instrument has no tail to fetch. Nothing cached: the whole of [start, end]."""
    start_ts, end_ts = _utc(start), _utc(end)
    first, last = start_ts.normalize(), end_ts.normalize()
    if windows is None:
        windows = cached_windows(cache_dir, source, instrument, interval, verify=verify)
    if cut_later is None:
        cut_later = _cut_later_allowed(source, interval)
    usable = [w for w in windows if cut_later or w.end <= last]
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor, from_entry = first, False
    for w in sorted(usable, key=lambda w: (w.start, w.end)):
        if w.start > cursor:
            if w.start >= last:
                break
            ranges.append(_range(cursor, w.start, first, last, start_ts, end_ts, from_entry))
            cursor, from_entry = w.start, False
        if w.end > cursor:
            cursor, from_entry = w.end, True
    if cursor < last:
        delisted_tail = from_entry and any(
            w.is_delisted for w in usable if w.end == cursor)
        if not delisted_tail:
            ranges.append(_range(cursor, last, first, last, start_ts, end_ts, from_entry))
    return ranges


def _range(a, b, first, last, start_ts, end_ts, from_entry):
    begin = a - pd.Timedelta(days=1) if from_entry else a
    begin = start_ts if begin <= first else begin
    return begin, end_ts if b >= last else b


def missing_share(
    cache_dir: Path | str, source: str, instrument: str, interval: str, start, end, *,
    verify: bool = False,
) -> float:
    """What share of [start, end] has to be fetched, 0..1 (by days): 0 when the cache covers the
    window, 1 when it holds nothing of the instrument. A day-long refetch before an entry's end
    counts as missing, so a window cached to yesterday still costs a little. File names alone by
    default, so a whole listing is priced quickly (`verify` also reads the metadata)."""
    total = (_day(end) - _day(start)).days
    ranges = missing_ranges(cache_dir, source, instrument, interval, start, end, verify=verify)
    if not ranges:
        return 0.0
    if total <= 0:
        return 1.0
    days = sum(max(1, (_day(b) - _day(a)).days) for a, b in ranges)
    return min(1.0, days / total)


def _stack(parts: list[pd.Series | None]) -> pd.Series:
    """Series joined in order, the LAST one winning where a timestamp is in more than one (the
    later entry or the fresh fetch: a bar written before it closed is replaced)."""
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    merged = pd.concat(parts)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def _assemble_from_cache(
    cache_dir, source, instrument, interval, start, end, fetch_range, accept_cached,
    funding_end_pad,
) -> tuple[InstrumentHistory, list[CachedWindow]] | None:
    """The instrument's history for [start, end] built from its cache entries of ANY window plus
    a fetch of the missing ranges (`missing_ranges`), cut to the request: bars from `start` to
    `end`, funding to `end + funding_end_pad`. Also the plain cases: a window ending earlier is
    extended by its gap, one ending later is cut. None when nothing usable is cached, a range is
    missing and there is no `fetch_range`, or nothing of the instrument is left in the window
    (the fetch then says what that means: a typo, or a coin that was not listed yet).

    An entry ending after `end` is cut from only when the source says how its funding window ends
    (`funding_end_pad`) and does not serve a capped number of recent candles at this interval
    (`_cut_later_allowed`)."""
    cut_later = funding_end_pad is not None and _cut_later_allowed(source, interval)
    first_day, last_day = _day(start), _day(end)
    loaded: list[tuple[CachedWindow, InstrumentHistory]] = []
    for window in cached_windows(cache_dir, source, instrument, interval, verify=True):
        if (window.end > last_day and not cut_later) or window.end < first_day \
                or window.start > last_day:
            continue
        cached = load_cached_history(
            cache_dir, source, instrument, interval, window.start, window.end)
        if cached is None or (accept_cached is not None and not accept_cached(cached)):
            continue
        loaded.append((window, cached))
    if not loaded:
        return None
    windows = [w for w, _ in loaded]
    ranges = missing_ranges(cache_dir, source, instrument, interval, start, end,
                            windows=windows, cut_later=cut_later)
    if ranges and fetch_range is None:
        return None
    fresh: list[tuple[pd.Timestamp, InstrumentHistory]] = []
    for a, b in ranges:
        try:
            fresh.append((b, fetch_range(instrument, a, b)))
        except ValueError:
            continue  # nothing in that range (not listed yet, or delisted since)
    pieces = [h for _, h in loaded] + [h for _, h in sorted(fresh, key=lambda f: f[0])]
    tail = [h for b, h in fresh if b >= _utc(end)]
    newest = tail[-1] if tail else loaded[-1][1]

    start_ts, end_ts = _utc(start), _utc(end)
    prices = _stack([p.prices for p in pieces])
    # A window requested from the same start keeps every bar a fetch returned (a source pads its
    # window by a bar); only a request that begins later than the earliest window is cut at its
    # start.
    cut_low = start_ts.normalize() > min(w.start for w in windows)
    keep = prices.index <= end_ts
    if cut_low:
        keep &= prices.index >= start_ts
    prices = prices[keep]
    if prices.empty:
        return None

    def bars(series: list[pd.Series | None]) -> pd.Series:
        return _stack(series).reindex(prices.index).dropna()

    funding = _stack([p.funding for p in pieces])
    if funding_end_pad is not None:
        funding = funding[funding.index <= end_ts + funding_end_pad]
    if cut_low:
        funding = funding[funding.index >= start_ts - pd.Timedelta(days=1)]
    high = None
    if any(p.high is not None for p in pieces):
        high = bars([p.high if p.high is not None else p.prices * float("nan")
                     for p in pieces])
    history = InstrumentHistory(
        instrument=instrument,
        prices=prices,
        funding=funding,
        volume=bars([p.volume for p in pieces]),
        trade_count=bars([p.trade_count for p in pieces]).astype("int64"),
        first_seen=max(min(p.first_seen for p in pieces), prices.index.min()),
        last_seen=prices.index.max(),
        is_delisted=newest.is_delisted,
        has_funding=pieces[0].has_funding,
        high=high,
    )
    return history, windows


def _drop_contained(cache_dir, source, instrument, interval, start, end,
                    used: list[CachedWindow]) -> None:
    """Remove the entries just merged into the one stored for [start, end] that lie wholly
    inside its window: it holds everything they did, so a copy is not kept for every start ever
    asked. Entries reaching beyond the window stay (they hold more), and so does everything of
    Hyperliquid, whose hourly fetch reads another interval's entry by its exact name
    (`reusable_funding`). Snapshots keep their own copy of the data and point at no file here."""
    if source.startswith(_CAPPED_SOURCE_PREFIX):
        return
    first, last = _day(start), _day(end)
    for window in used:
        if window.start >= first and window.end <= last and (window.start, window.end) != (
                first, last):
            for path in _entry_files(cache_dir, source, instrument, interval, window):
                path.unlink(missing_ok=True)


def _entry_files(cache_dir, source, instrument, interval, window: CachedWindow) -> list[Path]:
    return list(_cache_paths(cache_dir, source, instrument, interval, window.start, window.end))




# Instruments of a source fetched at the same time (`QLAB_FETCH_WORKERS`, default 1: one after
# another, as before). Only for the sources named here: Binance's limits allow a few requests a
# second, while Hyperliquid is already paced at its own limit by one (a second request in flight
# would only be answered with a 429 and a minute's wait).
FETCH_WORKERS_ENV = "QLAB_FETCH_WORKERS"
THREADED_SOURCES = frozenset({"binance"})


def fetch_workers(source: str) -> int:
    """How many instruments of `source` are fetched at once."""
    if source not in THREADED_SOURCES:
        return 1
    try:
        return max(1, int(os.environ.get(FETCH_WORKERS_ENV, "1")))
    except ValueError:
        return 1


def polite_sleep() -> None:
    time.sleep(REQUEST_DELAY_SECONDS)


def _is_retryable(exc: BaseException) -> bool:
    """Retry what waiting can fix, and nothing else.

    A transport error or a 5xx is the venue or the network having a bad
    moment. A 429 is us asking too fast — also worth waiting out, and the
    reason the backoff below reaches a full minute. Every other 4xx is a
    malformed request, i.e. a bug on our side: retrying it burns rate limit
    and hides the defect, so it propagates immediately.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.HTTPError)


def http_retry():
    """Shared tenacity retry policy for free public endpoints."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=RATE_LIMIT_MAX_WAIT_SECONDS),
        retry=retry_if_exception(_is_retryable),
    )


def align_funding_to_index(
    funding_raw: pd.Series, index: pd.DatetimeIndex, native_interval: pd.Timedelta
) -> pd.Series:
    """Resample a raw, native-frequency funding series onto ``index``.

    Each output label ``t`` gets the funding rate for the half-open period
    ``(prev_t, t]``, matching the ``MarketPanel.funding`` contract ("rate
    for the period ending at that index label"):

    - If the panel bar is *coarser than or equal to* the native settlement
      interval (e.g. daily bars over hourly Hyperliquid funding), the bucket
      must contain *exactly* the expected number of native settlements
      (``round(bar / native)``); a short or empty bucket is NaN, never a
      partial sum silently passed off as the full period's funding.
    - If the panel bar is *finer* than the native interval (e.g. hourly bars
      over Binance's 8h funding), a settlement's value lands on the bar
      ending at its native mark. A bar between marks is 0.0 -- no
      settlement can occur there, a structural zero, not a filled gap (until
      2026-10-02 it was NaN, which made holding through any hour between
      marks impossible: Bv2 on Binance's hourly history). A bar ending ON a
      mark with no settlement is NaN: a settlement was due and is missing.

    A due settlement that is missing is NaN in both cases; a missing value is
    never zero-filled.

    Real venues settle a few milliseconds *after* the nominal mark
    (Hyperliquid's ``fundingHistory`` timestamps typically land 20-40ms past
    the hour, not exactly on it) — under a strict ``(prev, t]`` window that
    jitter would push a settlement into the *next* bucket instead of the one
    it actually belongs to. Raw timestamps are dropped to the minute and
    rounded up to ``native_interval`` first, so a settlement is attributed to
    the native period it closes, not shifted by clock jitter.
    """
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    if funding_raw.empty:
        return pd.Series(float("nan"), index=index, dtype=float)

    funding_raw = funding_raw.copy()
    # A settlement closes the interval that ends at it, so it belongs to the
    # native slot ENDING at or after it: clock jitter is dropped to the minute,
    # then the time is rounded UP to the slot boundary (a time on the grid
    # stays where it is). Several settlements in one slot -- a Binance
    # contract on 4-hourly or hourly funding, common since 2023 -- are that
    # slot's funding together: summed, not counted as extra slots (which made
    # every such day read as incomplete).
    funding_raw.index = funding_raw.index.floor("min").ceil(native_interval)
    funding_raw = funding_raw.groupby(level=0).sum().sort_index()
    bar = index[1] - index[0] if len(index) > 1 else native_interval

    values: list[float] = []
    prev = index[0] - bar
    for ts in index:
        window = funding_raw[(funding_raw.index > prev) & (funding_raw.index <= ts)]
        if bar >= native_interval:
            expected = round(bar / native_interval)
            complete = expected >= 1 and len(window) == expected
            value = float(window.sum()) if complete else float("nan")
        elif len(window):
            value = float(window.sum())
        elif (ts - pd.Timestamp(0, tz="UTC")) % native_interval == pd.Timedelta(0):
            value = float("nan")  # a settlement was due at this mark and is missing
        else:
            value = 0.0  # between marks no settlement can occur: a true zero
        values.append(value)
        prev = ts

    return pd.Series(values, index=index, dtype=float)


# A single-bar move this large is not observed on any real, actively-traded
# instrument at this granularity: Bitcoin's worst CLOSE-to-close daily move
# on record is roughly -37% (2020-03-12, the COVID crash), and that is the
# most liquid crypto asset there is -- a thinner instrument can be more
# volatile still, but not by an order of magnitude on genuine trading. 0.80
# (80%) sits well above that with a wide safety margin for illiquid alts,
# while remaining far below the moves this check exists to catch: live
# Hyperliquid spot data (docs/TASKS.md, T17) showed BTC-SPOT (@142, UBTC/
# USDC) jumping -98.8% in one bar, and other thin pairs (BERA-SPOT,
# MON-SPOT, TRUMP-SPOT) jumping 1000%+, all at the exact boundary where a
# repeating placeholder price gives way to real trading (see
# `detect_bad_price_bars`). A real move that large would itself be
# economically implausible to simulate holding through, so treating it as
# corrupted rather than as a genuine return is the conservative, honest
# choice either way.
MAX_PLAUSIBLE_BAR_MOVE = 0.80


def detect_bad_price_bars(
    prices: pd.Series,
    trade_count: pd.Series | None = None,
    reference: pd.Series | None = None,
) -> pd.Series:
    """Flag bars whose price looks like a corrupted print rather than real
    price discovery (docs/TASKS.md, T17) -- a data-quality gate applied
    during collection, not a strategy-time filter, so a bad bar never
    reaches a panel (or a signal computed over it) silently.

    Observed live on Hyperliquid's free spot API: a newly created spot pair
    with no real liquidity yet returns a repeating, non-zero PLACEHOLDER
    "close" (Hyperliquid's UBTC/USDC pair printed a constant 6969696 then
    7979573 for 11 daily bars in Feb 2025 -- about 82x BTC's real price and
    completely flat -- before jumping to a real ~$97.6k print the moment
    actual trading began). Perp candles are fetched from the same endpoint
    shape and could in principle show the same artifact for a newly listed
    contract, so this check is applied to every instrument, not just spot.

    Two independent signals, either one flags a bar; a bar with no
    predecessor or successor to compare against is never flagged (nothing
    to detect an anomaly against):

    Revised again 2026-10-02 after the owner pointed out that FRAB trades
    AVAX spot live: what the placeholder and a merely quiet market have in
    common is a candle with NO trades, which repeats the last price by
    construction. The placeholder's interior was exactly that (UBTC/USDC,
    Feb 2025: 5 tiny trades at 6969696, then four days of zero-trade candles
    at the same price), and so are most hours of Hyperliquid's AVAX spot
    token (UAVAX in the API, AVAX spot in the venue's interface), which FRAB
    bought live on 2026-08-19 at 6.3883 against a 6.3648 perp. The
    difference is the LEVEL, not the repetition. So:

    - constancy ignores zero-trade bars when `trade_count` is known: a
      quiet hour is not a frozen quote;
    - `reference` (the perp of the same asset, for a spot column) flags a
      bar whose price is more than `MAX_PLAUSIBLE_BAR_MOVE` away from it --
      the placeholder sat 70x above BTC's perp, a real spot token cannot.

    - constancy: part of a RUN of three or more identical closes; every
      bar of such a run is flagged, the first included. Two identical closes
      in a row are NOT enough (revised 2026-10-02): the original premise --
      "a real market, quoted at full float precision, essentially never
      prints the same close twice" -- is false on Hyperliquid, which rounds
      prices to five significant figures (BTC to a whole dollar). On hourly
      bars BTC repeats its close on about 0.7% of consecutive pairs (32 of
      ~4700 hours, 2026), and flagging them knocked real bars out of every
      hourly panel, breaking held positions in the harness. A run of three
      needs two such coincidences in a row -- about one bar in 20,000 at
      that rate -- while the placeholder this check exists for held one
      price for 5 and 6 bars at a time.
    - implausible jump: see `MAX_PLAUSIBLE_BAR_MOVE` for the threshold and
      its justification.

    Returns a boolean Series aligned to `prices.index` (True = suspect).
    Does not itself decide what to do with a flagged bar -- see
    `qlab.data.snapshot._build_frames_from_histories`, which NaNs the price
    and clears `tradeable` there.
    """
    if prices.empty:
        return pd.Series(False, index=prices.index, dtype=bool)

    prev = prices.shift(1)
    traded = prices.notna()
    if trade_count is not None:
        traded &= trade_count.reindex(prices.index).fillna(0) > 0
    # Runs are counted over traded bars only: a zero-trade bar neither joins
    # nor breaks a run of repeated prints.
    traded_prices = prices.where(traded)
    run_source = traded_prices.dropna()
    run_id = (run_source != run_source.shift(1)).cumsum()
    run_length = run_id.map(run_id.value_counts()).reindex(prices.index)
    # The run test only stands in for data that cannot tell a frozen quote
    # from a quiet or pinned market: caches written before trade counts were
    # kept. With trade counts, every repeated close found so far has been
    # real -- AVAX spot at 9.00 three hours running, ETH spot at 1879.5 three
    # hours with ~250 trades each, low-priced perps (HMSTR, MEME, NOT) on a
    # coarse tick -- and the placeholder this test was written for is caught
    # by its level instead (below) and by the jump test.
    if trade_count is None:
        constant = (run_length >= 3) & traded
    else:
        constant = pd.Series(False, index=prices.index)
    if reference is not None:
        ratio = prices / reference.reindex(prices.index)
        constant |= (ratio - 1.0).abs() > MAX_PLAUSIBLE_BAR_MOVE

    pct_change = (prices - prev).abs() / prev.abs()
    big_jump = pct_change > MAX_PLAUSIBLE_BAR_MOVE
    if trade_count is not None:
        # A jump between two bars that both had trades is price discovery,
        # not a corrupted print: LUNA fell 96% in a day on 2022-05-11 with
        # millions of trades, and the jump test erased the crash from the
        # panel (found on Binance's long history, 2026-10-02). The jump test
        # stays for bars without trades or without a known count.
        big_jump &= ~(traded & traded.shift(1, fill_value=False))

    return (constant | big_jump).fillna(False)


__all__ = [
    "InstrumentHistory",
    "INTERVAL_TO_TIMEDELTA",
    "SPOT_COLUMN_SUFFIX",
    "SECOND_VENUE_SEPARATOR",
    "second_venue_column",
    "parse_second_venue",
    "MAX_PLAUSIBLE_BAR_MOVE",
    "align_funding_to_index",
    "detect_bad_price_bars",
    "fetch_universe_resumable",
    "load_cached_history",
    "merge_histories",
    "store_cached_history",
    "http_retry",
    "polite_sleep",
    "REQUEST_DELAY_SECONDS",
]
