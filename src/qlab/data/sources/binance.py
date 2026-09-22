"""Binance USDT-margined perpetual futures fetcher — free public REST API,
no API key.

Docs: https://binance-docs.github.io/apidocs/futures/en/en/

Uses ``/fapi/v1/klines`` (candles), ``/fapi/v1/fundingRate`` (funding
history) and ``/fapi/v1/exchangeInfo`` (listing date via ``onboardDate``).

Binance does not expose a historical *delisting* date for a symbol once it
drops out of ``exchangeInfo`` — a delisted symbol simply disappears from the
current listing. So, same as the Hyperliquid fetcher: ``is_delisted`` is
"not currently present/TRADING in exchangeInfo", and the delisting boundary
within the requested window is approximated by the last candle actually
returned (``last_seen``), following the point-in-time-via-price-presence
approach used in funding-rate-arbitrage's crypto cross-sectional research
(research/cross_sectional/crypto/survivorship.py).

**A survivorship-free universe cannot be built from Binance's free API at
all.** Point-in-time ``tradeable`` for an instrument you already asked for
is honest, but ``exchangeInfo`` only ever lists what's TRADING *now* — dead
perpetuals aren't in it, and there's no free historical-listing endpoint to
recover them from. So there is no way to discover "every contract that was
ever listed" the way `hyperliquid.describe_universe` can. `discover_universe`
therefore returns ``None`` here, not an empty list and not the current
survivor set — callers must fall back to an explicit instrument list, which
is honestly tagged ``universe_complete=False`` (see `qlab.data.panel`).
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pandas as pd
import structlog

from qlab.data.sources.base import (
    INTERVAL_TO_TIMEDELTA,
    InstrumentHistory,
    fetch_universe_resumable,
    http_retry,
    polite_sleep,
)

BASE_URL = "https://fapi.binance.com"
VENUE = "binance"

_MAX_KLINES_PER_PAGE = 1500
_MAX_FUNDING_PER_PAGE = 1000

FUNDING_NATIVE_INTERVAL = pd.Timedelta(hours=8)

logger = structlog.get_logger()


def _get(client: httpx.Client, path: str, params: dict) -> object:
    @http_retry()
    def _do() -> object:
        resp = client.get(f"{BASE_URL}{path}", params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    result = _do()
    polite_sleep()
    return result


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _symbol(instrument: str) -> str:
    """q-lab instrument names are bare coin tickers (e.g. ``"BTC"``); Binance
    USDT-M perpetuals are quoted symbols (e.g. ``"BTCUSDT"``)."""
    instrument = instrument.upper()
    return instrument if instrument.endswith("USDT") else f"{instrument}USDT"


def fetch_exchange_info(client: httpx.Client) -> dict[str, dict]:
    """Return ``{symbol: {"status": str, "onboard_date": Timestamp | None}}``
    from ``/fapi/v1/exchangeInfo``. Only symbols currently known to Binance
    appear here — a delisted symbol is simply absent."""
    data = _get(client, "/fapi/v1/exchangeInfo", {})
    out: dict[str, dict] = {}
    for s in data["symbols"]:
        onboard_ms = s.get("onboardDate")
        onboard_date = pd.Timestamp(int(onboard_ms), unit="ms", tz="UTC") if onboard_ms else None
        out[s["symbol"]] = {"status": s.get("status"), "onboard_date": onboard_date}
    return out


def describe_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> None:
    """See `discover_universe` and the module docstring: Binance's free API
    has no way to list delisted contracts, so there is no honest universe
    description to hand back here. Always returns ``None``."""
    return None


def discover_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> None:
    """Always returns ``None`` — Binance's free ``exchangeInfo`` only lists
    currently-TRADING symbols, so a full (survivors + delisted) universe
    cannot be discovered from it. See the module docstring. Callers must
    pass an explicit instrument list to `fetch_universe`/`build_snapshot`,
    which then must be marked ``universe_complete=False``."""
    return None


_KLINE_FRAME_COLUMNS = ("price", "volume", "trade_count")


def _empty_kline_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_KLINE_FRAME_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame.astype({"price": float, "volume": float, "trade_count": "int64"})


def fetch_candles(
    client: httpx.Client, instrument: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Page through ``/fapi/v1/klines`` and return a DataFrame indexed by the
    UTC candle-open timestamp, with columns ``price`` (close, ``k[4]``),
    ``volume`` (base-asset volume, ``k[5]`` -- NOT USD, see
    `qlab.data.sources.base.InstrumentHistory`) and ``trade_count``
    (``k[8]``, ``numberOfTrades``). Standard Binance futures kline array
    shape: ``[openTime, open, high, low, close, volume, closeTime,
    quoteAssetVolume, numberOfTrades, takerBuyBaseVolume,
    takerBuyQuoteVolume, ignore]``. All three parsed columns come off the
    exact same row already being paged through for price -- no extra
    request.
    """
    symbol = _symbol(instrument)
    step = INTERVAL_TO_TIMEDELTA[interval]
    cursor = start
    rows: dict[pd.Timestamp, tuple[float, float, int]] = {}

    while cursor <= end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": _to_ms(cursor),
            "endTime": _to_ms(end),
            "limit": _MAX_KLINES_PER_PAGE,
        }
        klines = _get(client, "/fapi/v1/klines", params)
        if not klines:
            break

        max_ts = cursor
        for k in klines:
            ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
            if ts > end:
                continue
            rows[ts] = (float(k[4]), float(k[5]), int(k[8]))
            if ts > max_ts:
                max_ts = ts

        if len(klines) < _MAX_KLINES_PER_PAGE:
            break
        cursor = max_ts + step

    if not rows:
        return _empty_kline_frame()

    index = pd.DatetimeIndex(sorted(rows), name="timestamp")
    frame = pd.DataFrame(
        [rows[ts] for ts in index], index=index, columns=list(_KLINE_FRAME_COLUMNS)
    )
    frame["trade_count"] = frame["trade_count"].astype("int64")
    return frame


def fetch_funding(
    client: httpx.Client, instrument: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Page through ``/fapi/v1/fundingRate`` and return a funding-rate
    Series (a fraction) indexed by UTC settlement time."""
    symbol = _symbol(instrument)
    cursor = start
    rows: dict[pd.Timestamp, float] = {}

    while cursor <= end:
        params = {
            "symbol": symbol,
            "startTime": _to_ms(cursor),
            "endTime": _to_ms(end),
            "limit": _MAX_FUNDING_PER_PAGE,
        }
        entries = _get(client, "/fapi/v1/fundingRate", params)
        if not entries:
            break

        max_ts = cursor
        for e in entries:
            ts = pd.Timestamp(int(e["fundingTime"]), unit="ms", tz="UTC")
            if ts > end:
                continue
            rows[ts] = float(e["fundingRate"])
            if ts > max_ts:
                max_ts = ts

        if len(entries) < _MAX_FUNDING_PER_PAGE:
            break
        cursor = max_ts + pd.Timedelta(milliseconds=1)

    return pd.Series(rows, dtype=float).sort_index()


def fetch_instrument_history(
    client: httpx.Client,
    instrument: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_info: dict[str, dict],
) -> InstrumentHistory:
    candles = fetch_candles(client, instrument, interval, start, end)
    if candles.empty:
        raise ValueError(f"binance: no kline data for {instrument!r} in [{start}, {end}]")
    prices = candles["price"]
    funding = fetch_funding(client, instrument, start, end)

    symbol = _symbol(instrument)
    info = exchange_info.get(symbol)
    is_delisted = info is None or info.get("status") != "TRADING"

    onboard_date = info["onboard_date"] if info else None
    first_seen = prices.index.min()
    if onboard_date is not None and onboard_date > start:
        first_seen = max(onboard_date, first_seen)

    return InstrumentHistory(
        instrument=instrument,
        prices=prices,
        funding=funding,
        volume=candles["volume"],
        trade_count=candles["trade_count"],
        first_seen=first_seen,
        last_seen=prices.index.max(),
        is_delisted=is_delisted,
    )


def fetch_universe(
    instruments: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    *,
    on_missing: str = "raise",
) -> dict[str, InstrumentHistory]:
    """Fetch price + funding history for each requested instrument.

    `on_missing` behaves as in the hyperliquid source: `"raise"` for a
    hand-written list, where an instrument with no data means a typo;
    `"skip"` for a venue-discovered list, where it just means the
    instrument did not exist during the window.
    """
    with httpx.Client() as client:
        exchange_info = fetch_exchange_info(client)
        return fetch_universe_resumable(
            instruments,
            start,
            end,
            interval,
            source=VENUE,
            fetch_one=lambda symbol: fetch_instrument_history(
                client, symbol, interval, start, end, exchange_info
            ),
            on_missing=on_missing,
        )


__all__ = [
    "VENUE",
    "FUNDING_NATIVE_INTERVAL",
    "fetch_exchange_info",
    "describe_universe",
    "discover_universe",
    "fetch_candles",
    "fetch_funding",
    "fetch_instrument_history",
    "fetch_universe",
]
