"""Shared types and helpers for market-data source fetchers.

Both ``hyperliquid.py`` and ``binance.py`` return the same shape of output
(a mapping of instrument -> ``InstrumentHistory``) so ``qlab.data.snapshot``
can assemble a ``MarketPanel`` without caring which venue the data came
from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Conservative fixed delay between successive HTTP requests to a free public
# endpoint. Both Hyperliquid and Binance publish per-IP rate limits well
# above this; a fixed delay is simpler than per-venue token-bucket
# accounting and comfortably avoids 429s for the request volumes this
# research tool makes (a handful of instruments, occasional fetches).
REQUEST_DELAY_SECONDS = 0.25

INTERVAL_TO_TIMEDELTA: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass(frozen=True)
class InstrumentHistory:
    """Raw per-instrument history as returned by a source fetcher.

    ``prices``/``funding`` are raw series indexed by UTC timestamps as
    returned by the venue — ``funding`` at the venue's *native* settlement
    frequency, not yet resampled onto the panel's chosen interval (that is
    ``snapshot.py``'s job, via ``align_funding_to_index``).

    ``first_seen``/``last_seen`` bound the instrument's tradeable life
    within the requested window. ``is_delisted`` says whether that life
    ended because the instrument was actually delisted (so
    ``tradeable`` should read False after ``last_seen``) as opposed to the
    requested window simply ending while it was still listed (so
    ``tradeable`` stays True through the end of the panel).
    """

    instrument: str
    prices: pd.Series
    funding: pd.Series
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    is_delisted: bool


def polite_sleep() -> None:
    time.sleep(REQUEST_DELAY_SECONDS)


def http_retry():
    """Shared tenacity retry policy for flaky/rate-limited public endpoints:
    exponential backoff, 5 attempts, only on network/HTTP errors — a bug in
    our own request-building must never be silently retried away."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
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


__all__ = [
    "InstrumentHistory",
    "INTERVAL_TO_TIMEDELTA",
    "align_funding_to_index",
    "http_retry",
    "polite_sleep",
    "REQUEST_DELAY_SECONDS",
]
