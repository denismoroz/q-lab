"""Shared types and helpers for market-data source fetchers.

Both ``hyperliquid.py`` and ``binance.py`` return the same shape of output
(a mapping of instrument -> ``InstrumentHistory``) so ``qlab.data.snapshot``
can assemble a ``MarketPanel`` without caring which venue the data came
from.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
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
    frame = pd.DataFrame(
        {
            "price": history.prices,
            "funding": history.funding,
            "volume": history.volume,
            "trade_count": history.trade_count,
        }
    )
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
) -> dict[str, InstrumentHistory]:
    """Fetch each instrument, reusing anything already cached from an earlier
    attempt and persisting each success as it lands.

    `on_missing` decides what an instrument with no data in range means, and
    that depends on where the list came from: `"raise"` for a hand-written
    list, where it is a typo; `"skip"` for a venue-discovered one, where it
    just means the instrument did not exist during the window.
    """
    if on_missing not in {"raise", "skip"}:
        raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")
    cache_dir = Path(cache_dir)
    histories: dict[str, InstrumentHistory] = {}
    for instrument in instruments:
        cached = load_cached_history(cache_dir, source, instrument, interval, start, end)
        if cached is not None:
            histories[instrument] = cached
            continue
        try:
            history = fetch_one(instrument)
        except ValueError:
            if on_missing == "raise":
                raise
            continue
        store_cached_history(cache_dir, source, interval, start, end, history)
        histories[instrument] = history
    return histories


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
      over Binance's 8h funding), a bucket can hold at most one settlement.
      That settlement's value is used at the bar where it lands; every other
      bar is NaN. This is not a "missing value" — no settlement occurs
      between marks — but callers aggregating over multiple bars must
      `sum(skipna=True)` rather than treat each NaN bar as a zero-cost bar.

    Either way, a bucket with **no data present** is NaN. Funding is never
    zero-filled here.

    Real venues settle a few milliseconds *after* the nominal mark
    (Hyperliquid's ``fundingHistory`` timestamps typically land 20-40ms past
    the hour, not exactly on it) — under a strict ``(prev, t]`` window that
    jitter would push a settlement into the *next* bucket instead of the one
    it actually belongs to. Raw timestamps are floored to ``native_interval``
    first so a settlement is attributed by the period it belongs to, not by
    clock jitter.
    """
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)
    if funding_raw.empty:
        return pd.Series(float("nan"), index=index, dtype=float)

    funding_raw = funding_raw.copy()
    funding_raw.index = funding_raw.index.floor(native_interval)
    funding_raw = funding_raw.sort_index()
    bar = index[1] - index[0] if len(index) > 1 else native_interval

    values: list[float] = []
    prev = index[0] - bar
    for ts in index:
        window = funding_raw[(funding_raw.index > prev) & (funding_raw.index <= ts)]
        if bar >= native_interval:
            expected = round(bar / native_interval)
            complete = expected >= 1 and len(window) == expected
            value = float(window.sum()) if complete else float("nan")
        else:
            value = float(window.iloc[0]) if len(window) == 1 else float("nan")
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


def detect_bad_price_bars(prices: pd.Series) -> pd.Series:
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

    - constancy: identical to the bar immediately BEFORE or AFTER it. No
      magnitude to calibrate -- a real market, quoted at full float
      precision (not rounded to cents), essentially never prints the exact
      same close on two consecutive bars. Checking both directions catches
      the FIRST bar of a repeated run too, not just the second bar onward
      (the run 6969696, 6969696, 6969696 needs to flag all three, not just
      the last two). A genuinely pegged asset can trip this on an ordinary
      day; marking one flat bar untradeable there is the same conservative
      trade-off `snapshot.py` already makes for a funding gap -- excluding
      the affected bar, not guessing, and not dropping the whole
      instrument.
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
    nxt = prices.shift(-1)
    constant = (prices == prev) | (prices == nxt)

    pct_change = (prices - prev).abs() / prev.abs()
    big_jump = pct_change > MAX_PLAUSIBLE_BAR_MOVE

    return (constant | big_jump).fillna(False)


__all__ = [
    "InstrumentHistory",
    "INTERVAL_TO_TIMEDELTA",
    "SPOT_COLUMN_SUFFIX",
    "MAX_PLAUSIBLE_BAR_MOVE",
    "align_funding_to_index",
    "detect_bad_price_bars",
    "fetch_universe_resumable",
    "load_cached_history",
    "store_cached_history",
    "http_retry",
    "polite_sleep",
    "REQUEST_DELAY_SECONDS",
]
