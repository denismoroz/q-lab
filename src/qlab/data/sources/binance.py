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

**The full universe comes from Binance's public data archive**
(``data.binance.vision``, 2026-10-02). ``exchangeInfo`` lists only what
Binance still knows (TRADING, plus SETTLING for some delisted contracts); the
archive keeps a monthly kline file for EVERY USDT-margined contract ever
listed, dead ones included, so its listing is the survivorship-free set.
Prices and funding still come from the REST API, which -- checked on SRM,
TOMO, HNT, ANT, BTS -- serves delisted contracts' history too; an instrument
without candles inside a window is skipped by `fetch_universe(on_missing=
"skip")`, as for Hyperliquid. Until 2026-10-02 this module returned ``None``
here and documented the universe as impossible to build; the archive was
not known then.
"""

from __future__ import annotations

import re
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


ARCHIVE_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_PREFIX = "data/futures/um/monthly/klines/"
_ARCHIVE_ENTRY = re.compile(r"<Prefix>" + re.escape(ARCHIVE_PREFIX) + r"([^/<]+)/</Prefix>")
_NEXT_MARKER = re.compile(r"<NextMarker>([^<]+)</NextMarker>")


def archive_symbols(client: httpx.Client) -> list[str]:
    """Every symbol with a monthly kline folder in the public archive -- the
    contracts Binance ever listed, delisted ones included."""
    symbols: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": ARCHIVE_PREFIX}
        if marker:
            params["marker"] = marker
        resp = client.get(ARCHIVE_LIST_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        symbols += _ARCHIVE_ENTRY.findall(resp.text)
        found = _NEXT_MARKER.search(resp.text)
        if not found:
            return sorted(set(symbols))
        marker = found.group(1)
        polite_sleep()


def _usdt_perpetual(symbol: str) -> str | None:
    """The instrument name of a USDT-margined perpetual (``BTCUSDT`` ->
    ``BTC``); None for a dated quarterly (``BTCUSDT_250328``) or another
    quote asset."""
    if "_" in symbol or not symbol.endswith("USDT") or symbol == "USDT":
        return None
    return symbol[: -len("USDT")]


def describe_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
                      ) -> list[tuple[str, bool]]:
    """``[(instrument, is_delisted)]`` for every USDT perpetual Binance ever
    listed: the archive's set plus anything ``exchangeInfo`` knows; delisted
    = not TRADING now. Raises if the archive cannot be read -- a partial list
    would silently reintroduce survivorship."""
    with httpx.Client() as client:
        info = fetch_exchange_info(client)
        names = set(archive_symbols(client)) | set(info)
    out = {}
    for symbol in names:
        instrument = _usdt_perpetual(symbol)
        if instrument:
            out[instrument] = info.get(symbol, {}).get("status") != "TRADING"
    return sorted(out.items())


def discover_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
                      ) -> list[str]:
    """Every instrument of `describe_universe`; those without candles inside
    the window are skipped by `fetch_universe(on_missing="skip")`."""
    return [instrument for instrument, _ in describe_universe(as_of_range)]


_KLINE_FRAME_COLUMNS = ("price", "volume", "trade_count")


def _empty_kline_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_KLINE_FRAME_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame.astype({"price": float, "volume": float, "trade_count": "int64"})


ARCHIVE_FILES_URL = "https://data.binance.vision"


def _archive_keys(client: httpx.Client, prefix: str) -> list[str]:
    keys: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        resp = client.get(ARCHIVE_LIST_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        page = re.findall(r"<Key>([^<]+\.zip)</Key>", resp.text)
        keys += page
        if "<IsTruncated>true</IsTruncated>" not in resp.text or not page:
            return keys
        marker = page[-1]


def _archive_rows(client: httpx.Client, key: str) -> list[list[str]]:
    """Rows of one archive CSV; a header row (newer files) is dropped."""
    import csv
    import io
    import zipfile

    resp = client.get(f"{ARCHIVE_FILES_URL}/{key}", timeout=60.0)
    resp.raise_for_status()
    polite_sleep()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    return [row for row in csv.reader(io.StringIO(text)) if row and row[0][:1].isdigit()]


def _months(start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    return {p.strftime("%Y-%m") for p in pd.period_range(start.tz_localize(None),
                                                        end.tz_localize(None), freq="M")}


def archive_candles(client: httpx.Client, instrument: str, interval: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """Candles from the public archive's monthly kline files -- for a contract
    the REST API no longer knows (it answers 400, "Invalid symbol")."""
    symbol = _symbol(instrument)
    months = _months(start, end)
    rows: dict[pd.Timestamp, tuple[float, float, int]] = {}
    for key in _archive_keys(client, f"{ARCHIVE_PREFIX}{symbol}/{interval}/"):
        if key[-11:-4] not in months:
            continue
        for k in _archive_rows(client, key):
            ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
            if start <= ts <= end:
                rows[ts] = (float(k[4]), float(k[5]), int(k[8]))
    if not rows:
        return _empty_kline_frame()
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(_KLINE_FRAME_COLUMNS))
    frame.index.name = "timestamp"
    return frame.sort_index()


def archive_funding(client: httpx.Client, instrument: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.Series:
    """Funding from the archive's monthly fundingRate files (calc_time,
    funding_interval_hours, last_funding_rate)."""
    symbol = _symbol(instrument)
    months = _months(start, end)
    rows: dict[pd.Timestamp, float] = {}
    prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    for key in _archive_keys(client, prefix):
        if key[-11:-4] not in months:
            continue
        for r in _archive_rows(client, key):
            ts = pd.Timestamp(int(r[0]), unit="ms", tz="UTC")
            if start <= ts <= end:
                rows[ts] = float(r[2])
    return pd.Series(rows, dtype=float).sort_index()


def _unknown_symbol(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code == 400


def fetch_candles(
    client: httpx.Client, instrument: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """REST candles; for a contract the API no longer knows, the archive's."""
    try:
        return _rest_candles(client, instrument, interval, start, end)
    except httpx.HTTPStatusError as exc:
        if not _unknown_symbol(exc):
            raise
        return archive_candles(client, instrument, interval, start, end)


def _rest_candles(
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
    """REST funding; for a contract the API no longer knows, the archive's."""
    try:
        return _rest_funding(client, instrument, start, end)
    except httpx.HTTPStatusError as exc:
        if not _unknown_symbol(exc):
            raise
        return archive_funding(client, instrument, start, end)


def _rest_funding(
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
    "archive_symbols",
    "fetch_exchange_info",
    "describe_universe",
    "discover_universe",
    "fetch_candles",
    "fetch_funding",
    "fetch_instrument_history",
    "fetch_universe",
]
