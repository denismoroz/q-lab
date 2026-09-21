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


__all__ = [
    "InstrumentHistory",
    "INTERVAL_TO_TIMEDELTA",
    "align_funding_to_index",
    "http_retry",
    "polite_sleep",
    "REQUEST_DELAY_SECONDS",
]
