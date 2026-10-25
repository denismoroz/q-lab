"""Aster USDT-margined perpetual futures fetcher -- free public REST API, no
API key.

Docs: https://github.com/asterdex/api-docs (V1 futures API; the V3 API is the
signed, account-level one and is not needed for market data).

Aster's futures API is Binance-compatible: the same ``/fapi/v1/klines`` array
shape, ``/fapi/v1/fundingRate`` records, ``/fapi/v1/exchangeInfo`` and the
extra ``/fapi/v1/fundingInfo``. What differs from Binance is the market, not
the wire format, and four of those differences matter to a data source:

- **Funding interval is per symbol and can change.** ``fundingInfo`` on
  2026-10-25: 361 symbols settle every 4 hours, 331 every 8, 103 hourly and 3
  every 2, with caps from +-0.3% to +-3% (most +-2%). Aster documents that it
  may change a contract's interval in volatile markets. The panel's native
  interval is the 8 hours of the default; settlements of a shorter interval
  are summed into their 8-hour slot (`align_funding_to_index`), exactly as
  for the Binance contracts that moved to 4 and 1 hours.
- **0.0001 is the funding floor, not a gap.** The rate is the premium plus
  ``clamp(0.01% - premium, +-0.05%)`` per 8 hours, so it sits at exactly the
  0.01% interest rate whenever the premium is small: a third of BTCUSDT's
  daily funding sums to exactly 3 x 0.0001. Only the first ten records
  (2021-08-27 to 08-30) are a constant before the formula starts. Kept as
  published.
- **A contract that stops trading keeps emitting candles.** TONUSDT and
  PAPERUSDT (status SETTLING) return flat, zero-volume, zero-trade bars up to
  today, and a delisted FTTUSDT keeps a funding feed a year past its last
  candle. For a contract that is not TRADING, the history is cut at its last
  bar WITH trades and the funding at that same moment, so a dead contract is
  not a flat price with a zero rate that a strategy could hold.
- **The delisted universe cannot be listed.** ``exchangeInfo`` shows what
  Aster still knows: TRADING, and SETTLING for contracts being wound down
  (36 on 2026-10-25). A contract delisted earlier is gone from it, yet its
  history is still served by name (LUNAUSDT, FTTUSDT, TOMOUSDT answer with
  candles back to 2022-23). No archive like Binance's ``data.binance.vision``
  exists for Aster. So `describe_universe`/`discover_universe` return
  ``None`` -- the documented answer for "this source cannot expose a complete
  point-in-time universe" -- and a snapshot needs an explicit instrument list
  (``universe_complete=False``). `best_effort_universe` narrows the gap by
  probing every name Binance ever listed against Aster (360 names not in
  ``exchangeInfo`` on 2026-10-25, 57 of them still served: LUNA, FTT, TOMO and
  other dead contracts), but it can never prove it found them all (a contract
  only Aster listed is invisible), so it is a list-maker for a person, not a
  basis for ``universe_complete``.

Only ``<COIN>USDT`` perpetuals are covered. Aster also lists USD1-, USDC- and
U-quoted contracts (32 on 2026-10-25); they are a different quote asset and
are left out rather than mixed into a USDT panel. There is no spot support.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pandas as pd
import structlog

from qlab.data.sources import binance as binance_source
from qlab.data.sources.base import (
    DEFAULT_RAW_CACHE_DIR,
    INTERVAL_TO_TIMEDELTA,
    InstrumentHistory,
    fetch_universe_resumable,
    http_retry,
    polite_sleep,
)

BASE_URL = "https://fapi.asterdex.com"
VENUE = "aster"

_MAX_KLINES_PER_PAGE = 1500
_MAX_FUNDING_PER_PAGE = 1000

FUNDING_NATIVE_INTERVAL = pd.Timedelta(hours=8)

# Before the venue's first contract (ETHUSDT's first candle is 2021-09-07), so a
# probe with this start answers with a symbol's very first candle if it has one.
_PROBE_START_MS = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)

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
    """q-lab instrument names are bare coin tickers (``"BTC"``); Aster USDT
    perpetuals are ``"BTCUSDT"``."""
    instrument = instrument.upper()
    return instrument if instrument.endswith("USDT") else f"{instrument}USDT"


def _instrument(symbol: str) -> str | None:
    """``BTCUSDT`` -> ``BTC``; None for a name that is not ``<COIN>USDT``."""
    if "_" in symbol or not symbol.endswith("USDT") or symbol == "USDT":
        return None
    return symbol[: -len("USDT")]


def fetch_exchange_info(client: httpx.Client) -> dict[str, dict]:
    """``{symbol: {"status", "onboard_date", "contract_type", "quote"}}`` from
    ``/fapi/v1/exchangeInfo``. A contract delisted long ago is absent."""
    data = _get(client, "/fapi/v1/exchangeInfo", {})
    out: dict[str, dict] = {}
    for s in data["symbols"]:
        onboard_ms = s.get("onboardDate")
        out[s["symbol"]] = {
            "status": s.get("status"),
            "onboard_date": pd.Timestamp(int(onboard_ms), unit="ms", tz="UTC")
            if onboard_ms else None,
            "contract_type": s.get("contractType"),
            "quote": s.get("quoteAsset"),
        }
    return out


def _is_usdt_perpetual(symbol: str, info: dict) -> bool:
    return (
        _instrument(symbol) is not None
        and info.get("contract_type") == "PERPETUAL"
        and info.get("quote") == "USDT"
    )


def listed_universe(client: httpx.Client) -> list[tuple[str, bool]]:
    """``[(instrument, is_delisted)]`` for the USDT perpetuals ``exchangeInfo``
    knows: TRADING ones are live, SETTLING (and any other status) delisted.
    Survivors plus the contracts being wound down -- NOT every contract ever
    listed (see the module docstring)."""
    info = fetch_exchange_info(client)
    return sorted(
        (_instrument(symbol), meta["status"] != "TRADING")
        for symbol, meta in info.items()
        if _is_usdt_perpetual(symbol, meta)
    )


def describe_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
                      ) -> list[tuple[str, bool]] | None:
    """None: Aster cannot list its delisted contracts (module docstring), so
    no universe can honestly be called complete."""
    return None


def discover_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
                      ) -> list[str] | None:
    """None, for the reason `describe_universe` gives. `build_snapshot` then
    asks for an explicit instrument list and marks the snapshot
    ``universe_complete=False``."""
    return None


def _probe_cache_path(cache_dir: Path | str) -> Path:
    return Path(cache_dir) / VENUE / "_probe.json"


def _is_invalid_symbol(exc: httpx.HTTPStatusError) -> bool:
    """HTTP 400 with code -1121: Aster does not know the name at all."""
    if exc.response.status_code != 400:
        return False
    try:
        return exc.response.json().get("code") == -1121
    except ValueError:
        return False


def _first_candle_ms(client: httpx.Client, symbol: str) -> int | None:
    """Open time of the symbol's first daily candle, or None if Aster has never
    heard of the name (HTTP 400, "Invalid symbol")."""
    params = {"symbol": symbol, "interval": "1d", "startTime": _PROBE_START_MS, "limit": 1}
    try:
        rows = _get(client, "/fapi/v1/klines", params)
    except httpx.HTTPStatusError as exc:
        if _is_invalid_symbol(exc):
            return None
        raise
    return int(rows[0][0]) if rows else None


def probe_symbols(
    client: httpx.Client, symbols: Sequence[str], cache_dir: Path | str = DEFAULT_RAW_CACHE_DIR
) -> dict[str, int | None]:
    """``{symbol: first daily candle (ms) or None}`` for every symbol asked,
    asking the venue only about names not probed before. The answers are kept
    in ``<cache_dir>/aster/_probe.json`` and written as they arrive, so an
    interrupted probe resumes; a "no" is kept too (a contract Aster never
    listed does not appear later, and a new listing shows in ``exchangeInfo``
    on its own)."""
    path = _probe_cache_path(cache_dir)
    known: dict[str, int | None] = {}
    if path.is_file():
        try:
            known = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            known = {}
    for symbol in symbols:
        if symbol in known:
            continue
        known[symbol] = _first_candle_ms(client, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(known, sort_keys=True))
    return {symbol: known[symbol] for symbol in symbols}


def best_effort_universe(cache_dir: Path | str = DEFAULT_RAW_CACHE_DIR) -> list[tuple[str, bool]]:
    """``[(instrument, is_delisted)]``: the listed universe plus every USDT
    perpetual Binance ever listed that Aster still serves candles for.

    A list-maker for a person writing an explicit ``instruments`` list, not a
    claim of completeness: a contract only Aster listed and later removed is
    not found, and the probe is one request per unknown name (360 on
    2026-10-25, about 10 minutes) the first time and cached after.
    """
    with httpx.Client() as client:
        listed = dict(listed_universe(client))
        candidates = sorted(
            f"{inst}USDT"
            for inst in map(_instrument, binance_source.archive_symbols(client))
            if inst and inst not in listed
        )
        probed = probe_symbols(client, candidates, cache_dir)
    out = dict(listed)
    for symbol, first_ms in probed.items():
        if first_ms is not None:
            out[_instrument(symbol)] = True
    return sorted(out.items())


_KLINE_FRAME_COLUMNS = ("price", "volume", "trade_count", "high")


def _empty_kline_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_KLINE_FRAME_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame.astype({"price": float, "volume": float, "trade_count": "int64"})


def fetch_candles(
    client: httpx.Client, instrument: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Page through ``/fapi/v1/klines``: a frame indexed by UTC candle-open
    time with ``price`` (close), ``volume`` (BASE-asset units, not USD),
    ``trade_count`` and ``high`` -- the Binance array shape ``[openTime, open,
    high, low, close, volume, closeTime, quoteVolume, trades, ...]``."""
    symbol = _symbol(instrument)
    step = INTERVAL_TO_TIMEDELTA[interval]
    cursor = start
    rows: dict[pd.Timestamp, tuple[float, float, int, float]] = {}

    while cursor <= end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": _to_ms(cursor),
            "endTime": _to_ms(end),
            "limit": _MAX_KLINES_PER_PAGE,
        }
        try:
            klines = _get(client, "/fapi/v1/klines", params)
        except httpx.HTTPStatusError as exc:
            # A name Aster has never heard of is "no data" -- a typo for a
            # hand-written list, ordinary absence for a discovered one -- and
            # `fetch_instrument_history` turns it into the ValueError that
            # `on_missing` decides about. Any other 4xx is a bug of ours.
            if _is_invalid_symbol(exc):
                break
            raise
        if not klines:
            break

        max_ts = cursor
        for k in klines:
            ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
            if ts > end:
                continue
            rows[ts] = (float(k[4]), float(k[5]), int(k[8]), float(k[2]))
            max_ts = max(max_ts, ts)

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
    """Page through ``/fapi/v1/fundingRate`` (ascending, ``startTime`` and
    ``endTime`` inclusive, at most 1000 per page): the funding rate (a
    fraction per settlement) indexed by UTC settlement time, at whatever
    interval the contract settled at the time. ``start`` must be a real
    timestamp: ``startTime=0`` is read as absent and returns the newest
    records."""
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
            max_ts = max(max_ts, ts)

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
        raise ValueError(f"aster: no kline data for {instrument!r} in [{start}, {end}]")

    info = exchange_info.get(_symbol(instrument))
    is_delisted = info is None or info.get("status") != "TRADING"

    if is_delisted:
        # A contract that stopped trading keeps emitting flat, zero-trade
        # bars (module docstring): its life ends at the last bar with trades.
        traded = candles.index[candles["trade_count"] > 0]
        if len(traded) == 0:
            raise ValueError(f"aster: no traded bar for {instrument!r} in [{start}, {end}]")
        candles = candles.loc[: traded.max()]

    prices = candles["price"]
    funding = fetch_funding(client, instrument, start, end)
    if is_delisted:
        # Funding records stamped after the last bar belong to no trading.
        funding = funding[funding.index <= prices.index.max() + INTERVAL_TO_TIMEDELTA[interval]]

    onboard_date = info["onboard_date"] if info else None
    first_seen = prices.index.min()
    if onboard_date is not None and onboard_date > start:
        first_seen = max(onboard_date, first_seen)

    return InstrumentHistory(
        instrument=instrument,
        prices=prices,
        funding=funding,
        volume=candles["volume"],
        high=candles["high"],
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
    """Price, volume and funding history for each requested instrument.

    `on_missing` behaves as for the other sources: ``"raise"`` for a
    hand-written list (an instrument with no data is a typo), ``"skip"`` for
    a discovered one. Cached under ``data/raw/aster/<interval>/`` and resumed
    (`fetch_universe_resumable`).
    """
    with httpx.Client() as client:
        exchange_info = fetch_exchange_info(client)
        return fetch_universe_resumable(
            instruments,
            start,
            end,
            interval,
            source=VENUE,
            fetch_one=lambda name: fetch_instrument_history(
                client, name, interval, start, end, exchange_info
            ),
            fetch_range=lambda name, a, b: fetch_instrument_history(
                client, name, interval, a, b, exchange_info
            ),
            on_missing=on_missing,
        )


__all__ = [
    "VENUE",
    "FUNDING_NATIVE_INTERVAL",
    "best_effort_universe",
    "describe_universe",
    "discover_universe",
    "fetch_candles",
    "fetch_exchange_info",
    "fetch_funding",
    "fetch_instrument_history",
    "fetch_universe",
    "listed_universe",
    "probe_symbols",
]
