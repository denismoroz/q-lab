"""Hyperliquid perpetuals fetcher — free public ``/info`` endpoint, no API key.

Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint

Only read-only market-data request types are used:
``metaAndAssetCtxs`` (current listing status), ``candleSnapshot`` (mark/close
candles) and ``fundingHistory`` (hourly funding settlements).

Point-in-time tradeability is inferred the same way as the prior art in
funding-rate-arbitrage/research/cross_sectional/crypto/survivorship.py: HL's
free API exposes no historical delisting *date*, only a current
``isDelisted`` flag, so an instrument is considered listed for exactly the
span its candle feed actually covers (``first_seen``..``last_seen``), and
``is_delisted`` tells ``snapshot.py`` whether "after last_seen" means
"delisted" (tradeable=False) or just "request window ended while still
listed" (tradeable=True through the end of the panel).
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pandas as pd
import structlog

from qlab.data.sources.base import (
    INTERVAL_TO_TIMEDELTA,
    InstrumentHistory,
    http_retry,
    polite_sleep,
)

BASE_URL = "https://api.hyperliquid.xyz/info"
VENUE = "hyperliquid"

# HL caps candleSnapshot/fundingHistory responses; a page shorter than this
# means we've reached the tail of the available history.
_MAX_CANDLES_PER_PAGE = 5000
_MAX_FUNDING_PER_PAGE = 500

FUNDING_NATIVE_INTERVAL = pd.Timedelta(hours=1)

logger = structlog.get_logger()


def _post(client: httpx.Client, payload: dict) -> object:
    @http_retry()
    def _do() -> object:
        resp = client.post(BASE_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    result = _do()
    polite_sleep()
    return result


def _to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def fetch_meta(client: httpx.Client) -> dict[str, dict]:
    """Return ``{instrument: {"is_delisted": bool, "max_leverage": int}}``
    from ``metaAndAssetCtxs``."""
    meta, _asset_ctxs = _post(client, {"type": "metaAndAssetCtxs"})
    return {
        entry["name"]: {
            "is_delisted": bool(entry.get("isDelisted", False)),
            "max_leverage": entry.get("maxLeverage"),
        }
        for entry in meta["universe"]
    }


def describe_universe(
    as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[tuple[str, bool]]:
    """Return ``[(instrument, is_delisted), ...]`` for HL's full perpetual
    universe — survivors *and* delisted names alike, sorted by instrument.

    This is what makes a Hyperliquid panel able to be survivorship-free:
    ``metaAndAssetCtxs`` exposes ``isDelisted`` for every perp HL has ever
    listed, current or dead, so nothing needs to be hand-picked. HL's free
    API has no historical listing *snapshot* though (no "give me the
    universe as it stood on date X"), so ``as_of_range`` is accepted only
    for interface symmetry with other sources' ``discover_universe`` and is
    not used to filter — the full CURRENT universe is always returned,
    which already includes every past delisting.
    """
    with httpx.Client() as client:
        meta = fetch_meta(client)
    return sorted((name, info["is_delisted"]) for name, info in meta.items())


def discover_universe(as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> list[str]:
    """Return just the instrument names from `describe_universe` — the full
    point-in-time-safe universe (survivors + delisted) that
    ``build_snapshot`` uses by default when no explicit instrument list is
    given, so a caller never has to (and never accidentally does) hand-pick
    survivors."""
    return [name for name, _is_delisted in describe_universe(as_of_range)]


def fetch_candles(
    client: httpx.Client, coin: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Page through ``candleSnapshot`` and return a close-price Series
    indexed by the UTC candle-open timestamp."""
    step = INTERVAL_TO_TIMEDELTA[interval]
    cursor = start
    rows: dict[pd.Timestamp, float] = {}

    while cursor <= end:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": _to_ms(cursor),
                "endTime": _to_ms(end),
            },
        }
        candles = _post(client, payload)
        if not candles:
            break

        max_ts = cursor
        for c in candles:
            ts = pd.Timestamp(int(c["t"]), unit="ms", tz="UTC")
            if ts > end:
                continue
            rows[ts] = float(c["c"])
            if ts > max_ts:
                max_ts = ts

        if len(candles) < _MAX_CANDLES_PER_PAGE:
            break
        cursor = max_ts + step

    return pd.Series(rows, dtype=float).sort_index()


def fetch_funding(
    client: httpx.Client, coin: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Page through ``fundingHistory`` and return a funding-rate Series (a
    fraction, e.g. ``0.0000125`` for 1.25bp) indexed by UTC settlement time."""
    cursor = start
    rows: dict[pd.Timestamp, float] = {}

    while cursor <= end:
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": _to_ms(cursor),
            "endTime": _to_ms(end),
        }
        entries = _post(client, payload)
        if not entries:
            break

        max_ts = cursor
        for e in entries:
            ts = pd.Timestamp(int(e["time"]), unit="ms", tz="UTC")
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
    coin: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    meta: dict[str, dict],
) -> InstrumentHistory:
    prices = fetch_candles(client, coin, interval, start, end)
    if prices.empty:
        raise ValueError(f"hyperliquid: no candle data for {coin!r} in [{start}, {end}]")
    funding = fetch_funding(client, coin, start, end)

    is_delisted = bool(meta.get(coin, {}).get("is_delisted", False))
    return InstrumentHistory(
        instrument=coin,
        prices=prices,
        funding=funding,
        first_seen=prices.index.min(),
        last_seen=prices.index.max(),
        is_delisted=is_delisted,
    )


def fetch_universe(
    instruments: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, interval: str
) -> dict[str, InstrumentHistory]:
    """Fetch price + funding history for each requested instrument.

    Raises if a requested instrument has no candle data at all in range: a
    typo'd or never-listed instrument must fail loudly rather than silently
    produce an all-NaN column that looks like "listed but never traded".
    """
    with httpx.Client() as client:
        meta = fetch_meta(client)
        return {
            coin: fetch_instrument_history(client, coin, interval, start, end, meta)
            for coin in instruments
        }


__all__ = [
    "VENUE",
    "FUNDING_NATIVE_INTERVAL",
    "fetch_meta",
    "fetch_candles",
    "fetch_funding",
    "fetch_instrument_history",
    "fetch_universe",
    "describe_universe",
    "discover_universe",
]
