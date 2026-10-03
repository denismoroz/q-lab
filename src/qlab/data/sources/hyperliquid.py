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
from pathlib import Path

import httpx
import pandas as pd
import structlog

from qlab.data.sources.base import (
    DEFAULT_RAW_CACHE_DIR,
    INTERVAL_TO_TIMEDELTA,
    SPOT_COLUMN_SUFFIX,
    InstrumentHistory,
    fetch_universe_resumable,
    http_retry,
    load_cached_history,
    polite_sleep,
)

BASE_URL = "https://api.hyperliquid.xyz/info"
VENUE = "hyperliquid"

# HIP-3 deployments (builder-deployed perp markets on the same HyperCore
# engine and account) that q-lab can read. Every request type used here takes
# an optional `dex` and otherwise behaves identically; instrument names come
# back already prefixed (`xyz:AAPL`), so candles and funding need nothing new.
# Only `xyz` is listed: of the nine deployments measured 2026-09-22, it is
# the one live market (cards/README.md); the others are dead or tiny.
HIP3_DEXES: tuple[str, ...] = ("xyz",)


def hip3_venue(dex: str) -> str:
    """The `source` name a spec uses for one HIP-3 deployment, e.g.
    `hyperliquid-xyz`. Kept separate from plain `hyperliquid` because the
    universe, the fee schedule and the execution support all differ."""
    return f"{VENUE}-{dex}"

# HL caps candleSnapshot/fundingHistory responses; a page shorter than this
# means we've reached the tail of the available history.
_MAX_CANDLES_PER_PAGE = 5000

# The venue serves only the most recent candles of an instrument -- this many
# per interval (Hyperliquid info-endpoint docs on candleSnapshot: "Only the
# most recent 5000 candles are available"). Measured 2026-10-01: on 1h bars
# 200 of 222 main-market instruments span exactly the last ~5000 hours
# although their daily history goes back years. Older bars do not exist at
# the source, which the panel would otherwise read as "not yet listed"
# (`qlab.data.snapshot`, `history_truncated`).
SERVED_CANDLES_CAP = 5000
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


def fetch_meta(client: httpx.Client, dex: str | None = None) -> dict[str, dict]:
    """Return ``{instrument: {"is_delisted": bool, "max_leverage": int}}``
    from ``metaAndAssetCtxs`` -- of the main perp market, or of HIP-3
    deployment `dex` when given. A deployment's list includes its delisted
    instruments too (19 of 129 for `xyz` on 2026-10-01), so the same
    survivorship-free argument as for the main market holds."""
    payload: dict[str, str] = {"type": "metaAndAssetCtxs"}
    if dex is not None:
        payload["dex"] = dex
    meta, _asset_ctxs = _post(client, payload)
    return {
        entry["name"]: {
            "is_delisted": bool(entry.get("isDelisted", False)),
            "max_leverage": entry.get("maxLeverage"),
        }
        for entry in meta["universe"]
    }


def describe_universe(
    as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    *,
    dex: str | None = None,
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
        meta = fetch_meta(client, dex)
    return sorted((name, info["is_delisted"]) for name, info in meta.items())


def discover_universe(
    as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    *,
    dex: str | None = None,
) -> list[str]:
    """Return just the instrument names from `describe_universe` — the full
    point-in-time-safe universe (survivors + delisted) that
    ``build_snapshot`` uses by default when no explicit instrument list is
    given, so a caller never has to (and never accidentally does) hand-pick
    survivors."""
    return [name for name, _is_delisted in describe_universe(as_of_range, dex=dex)]


_CANDLE_FRAME_COLUMNS = ("price", "volume", "trade_count", "high")


def _empty_candle_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(_CANDLE_FRAME_COLUMNS))
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame.astype({"price": float, "volume": float, "trade_count": "int64", "high": float})


def fetch_candles(
    client: httpx.Client, coin: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Page through ``candleSnapshot`` and return a DataFrame indexed by the
    UTC candle-open timestamp, with columns:

    - ``price``: the candle close (``"c"``) -- the only field the pre-T27
      version of this function returned.
    - ``volume`` (docs/TASKS.md, T27): the candle's base-asset volume
      (``"v"``), e.g. BTC for the BTC perp -- NOT USD. See
      `qlab.data.sources.base.InstrumentHistory`'s docstring for why USD
      volume is deliberately computed downstream (``volume * price``) and
      never stored as its own column.
    - ``trade_count``: the candle's fill count (``"n"``) -- kept for raw
      fidelity, not consumed by anything yet.

    All three come off the exact same candle row already being paged
    through for price, so parsing them adds no extra request.
    """
    step = INTERVAL_TO_TIMEDELTA[interval]
    cursor = start
    rows: dict[pd.Timestamp, tuple[float, float, int, float]] = {}

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
            rows[ts] = (float(c["c"]), float(c["v"]), int(c["n"]), float(c["h"]))
            if ts > max_ts:
                max_ts = ts

        if len(candles) < _MAX_CANDLES_PER_PAGE:
            break
        cursor = max_ts + step

    if not rows:
        return _empty_candle_frame()

    index = pd.DatetimeIndex(sorted(rows), name="timestamp")
    frame = pd.DataFrame(
        [rows[ts] for ts in index], index=index, columns=list(_CANDLE_FRAME_COLUMNS)
    )
    frame["trade_count"] = frame["trade_count"].astype("int64")
    return frame


def has_history_before(coin: str, end: pd.Timestamp, *, dex: str | None = None) -> bool:
    """Does the venue serve ANY daily candle for `coin` up to `end`?

    Used for one question only: a delisted instrument that returned no data
    for a requested window -- did it live entirely before the window
    (ordinary point-in-time truth), or does the venue no longer serve its
    history at all? On the main market the first holds: every delisted perp
    missing from a 2025-2026 panel still serves candles ending in 2024. On the
    `xyz` HIP-3 deployment the second holds: all 19 delisted instruments
    return zero candles for any range (checked 2026-10-01), although the
    deployment only launched in October 2025, so each of them traded inside
    any window that covers it. `dex` is accepted for symmetry with the other
    source functions; instrument names already carry the deployment prefix.
    """
    del dex
    with httpx.Client() as client:
        candles = fetch_candles(client, coin, "1d", pd.Timestamp("2015-01-01", tz="UTC"), end)
    return not candles.empty


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
    funding: pd.Series | None = None,
) -> InstrumentHistory:
    """`funding`, when given, is used instead of fetching it -- see
    `reusable_funding`."""
    candles = fetch_candles(client, coin, interval, start, end)
    if candles.empty:
        raise ValueError(f"hyperliquid: no candle data for {coin!r} in [{start}, {end}]")
    prices = candles["price"]
    volume = candles["volume"]
    trade_count = candles["trade_count"]
    # Funding is fetched wider than the price range on purpose. A bar labelled
    # `t` carries the funding for the half-open period `(t - bar, t]`, so the
    # first bar's bucket needs settlements from before `start`, and the last
    # bar's needs the settlement landing exactly on `end` — which a
    # `[start, end)` fetch leaves out. Without the padding both edge buckets
    # come back short, `align_funding_to_index` correctly refuses to pass off a
    # partial sum as the period's funding, and the harness then refuses to run
    # on a NaN rate. The data is there; only the window was too narrow.
    bar = pd.Timedelta(interval)
    if funding is None:
        funding = fetch_funding(
            client, coin, start - bar, end + FUNDING_NATIVE_INTERVAL
        )

    is_delisted = bool(meta.get(coin, {}).get("is_delisted", False))
    return InstrumentHistory(
        instrument=coin,
        prices=prices,
        funding=funding,
        volume=volume,
        trade_count=trade_count,
        first_seen=prices.index.min(),
        last_seen=prices.index.max(),
        is_delisted=is_delisted,
        high=candles["high"],
    )


def reusable_funding(
    source: str,
    coin: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache_dir: Path = DEFAULT_RAW_CACHE_DIR,
) -> pd.Series | None:
    """Funding for `coin` over this window, already cached by a fetch at
    ANOTHER interval -- or None (docs/TASKS.md T20).

    Funding settles hourly whatever the bar size, so the series is the same
    for a 1h, 4h or 1d panel of the same window; only the raw cache key
    differed. Refetching it per interval cost ~30 requests per instrument
    (about 2.5 hours for the main market) to download an identical series.

    Only a cache at an interval AT LEAST as long as the requested one is
    reused: each fetch pads its start by one of its own bars (the first bar's
    funding bucket reaches back one bar), so a 1d cache covers what a 4h or
    1h panel needs, while a 1h cache would leave a 4h panel's first bucket
    three hours short.
    """
    bar = INTERVAL_TO_TIMEDELTA[interval]
    for other, other_bar in sorted(INTERVAL_TO_TIMEDELTA.items(), key=lambda kv: -kv[1]):
        # The same interval counts too: an entry refetched only because it
        # predates bar highs still holds exactly the funding needed.
        if other_bar < bar:
            continue
        cached = load_cached_history(cache_dir, source, coin, other, start, end)
        if cached is not None and cached.has_funding and not cached.funding.empty:
            return cached.funding
    return None


def fetch_universe(
    instruments: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    *,
    on_missing: str = "raise",
    dex: str | None = None,
) -> dict[str, InstrumentHistory]:
    """Fetch price + funding history for each requested instrument.

    `dex` selects a HIP-3 deployment: its listing metadata is read instead
    of the main market's, and its raw cache lives under its own source name
    (`hip3_venue(dex)`) so the two can never be mixed up on disk.

    `on_missing` decides what an instrument with no candle data in range
    means, and that depends entirely on where the list came from:

    - `"raise"` (default) for a hand-written list. A typo'd or never-listed
      ticker must fail loudly rather than silently become an all-NaN column
      that looks like "listed but never traded".
    - `"skip"` for a list discovered from the venue itself. There are no
      typos there, and a coin whose whole tradeable life falls outside the
      window is ordinary point-in-time truth, not an error. Raising on it
      forces the caller to widen the range to the venue's entire history
      just to ask about one year.
    """
    with httpx.Client() as client:
        meta = fetch_meta(client, dex)
        return fetch_universe_resumable(
            instruments,
            start,
            end,
            interval,
            source=VENUE if dex is None else hip3_venue(dex),
            fetch_one=lambda coin: fetch_instrument_history(
                client, coin, interval, start, end, meta,
                funding=reusable_funding(
                    VENUE if dex is None else hip3_venue(dex), coin, interval, start, end
                ),
            ),
            fetch_range=lambda coin, a, b: fetch_instrument_history(
                client, coin, interval, a, b, meta
            ),
            on_missing=on_missing,
            # Entries written before bar highs were kept are refetched (the
            # candles only -- funding comes back from the cache above).
            accept_cached=lambda history: history.high is not None,
        )


# --------------------------------------------------------------------------
# Spot markets (docs/TASKS.md, T17). Hyperliquid quotes spot pairs against
# USDC and names most of them with an anonymous "@<index>" symbol rather
# than a human ticker (`spotMeta`'s `universe[i].name`) -- only a handful of
# canonical pairs (PURR/USDC, HYPE/USDC, ...) get a real "BASE/QUOTE" name.
# The base TOKEN name, however, is human-readable (`spotMeta`'s
# `tokens[i].name`) and for every major coin is that coin's ticker with a
# leading "U" ("Unit" bridge token: UBTC, UETH, USOL, ...) -- confirmed
# against funding-rate-arbitrage's own
# `src/frab/exchanges/hyperliquid/tokens.py` and
# `research/venue_refresh_2026_06/probe_spot_availability.py`, and against
# HL's live `spotMeta` response. `fetch_spot_meta` reverses that mapping:
# coin ticker -> the "@N" (or named) pair symbol `candleSnapshot` accepts as
# `coin` for the spot market.
#
# Hyperliquid's free API has no `fundingHistory`-equivalent for spot (a spot
# `fundingHistory` request returns `null`, not an error) and no historical
# delisting flag either (`spotMeta`'s universe entries carry no `isDelisted`
# field, unlike `metaAndAssetCtxs`'s perp `universe`) -- both are genuine,
# permanent limitations of the free API, not bugs here.


def fetch_spot_meta(client: httpx.Client, perp_instruments: Sequence[str]) -> dict[str, str]:
    """Return ``{coin: hl_pair_symbol}`` for every ``coin`` in
    `perp_instruments` that also has a USDC-quoted spot market on
    Hyperliquid, e.g. ``{"BTC": "@142"}``.

    Most of Hyperliquid's ~500 spot tokens (HFUN, LICK, MANLET, ...) are
    unrelated community/meme coins with no perp counterpart at all, and are
    silently excluded here -- this is exactly the boundary the
    ``<COIN>-SPOT`` column-naming convention (``SPOT_COLUMN_SUFFIX``) needs
    to stay unambiguous, since it only ever names a coin that ALSO has a
    perp column in the same panel. A coin's spot pair is looked up by an
    exact token-name match first (native tokens like HYPE, PURR), then by
    stripping a leading ``U``/``u`` (bridge tokens like UBTC, UETH, USOL).
    """
    meta = _post(client, {"type": "spotMeta"})
    token_name_by_index: dict[int, str] = {
        t["index"]: t["name"] for t in meta.get("tokens", []) if isinstance(t.get("index"), int)
    }
    usdc_indices = {idx for idx, name in token_name_by_index.items() if name == "USDC"}

    base_token_to_pair: dict[str, str] = {}
    for pair in meta.get("universe", []):
        toks = pair.get("tokens") or []
        if len(toks) != 2 or toks[1] not in usdc_indices:
            continue
        base_name = token_name_by_index.get(toks[0])
        if base_name:
            base_token_to_pair[base_name] = pair["name"]

    result: dict[str, str] = {}
    for coin in perp_instruments:
        for candidate in (coin, f"U{coin}", f"u{coin}"):
            if candidate in base_token_to_pair:
                result[coin] = base_token_to_pair[candidate]
                break
    return result


def describe_spot_universe(
    as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[tuple[str, bool]]:
    """Return ``[(spot_column_name, is_delisted), ...]`` for every
    Hyperliquid spot market resolvable to a coin that also has a perp
    market, e.g. ``[("BTC-SPOT", False), ...]`` -- the spot analogue of
    `describe_universe`. ``as_of_range`` is accepted for interface symmetry
    only, same reasoning as `describe_universe`.

    ``is_delisted`` is always ``False``: Hyperliquid's ``spotMeta`` exposes
    no historical-delisting flag for spot pairs (unlike
    ``metaAndAssetCtxs`` for perps) -- a genuine, permanent gap in the free
    API. A spot market the venue quietly stopped listing therefore reads as
    "listed through the end of the panel" rather than "delisted", same
    fallback semantics as a perp whose candle feed simply ends within a
    requested window (see `InstrumentHistory`).
    """
    with httpx.Client() as client:
        meta = fetch_meta(client)
        pair_by_coin = fetch_spot_meta(client, list(meta))
    return sorted((f"{coin}{SPOT_COLUMN_SUFFIX}", False) for coin in pair_by_coin)


def discover_spot_universe(
    as_of_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[str]:
    """Return just the ``<COIN>-SPOT`` column names from
    `describe_spot_universe` -- the spot analogue of `discover_universe`."""
    return [name for name, _is_delisted in describe_spot_universe(as_of_range)]


def fetch_spot_instrument_history(
    client: httpx.Client,
    coin: str,
    pair_symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> InstrumentHistory:
    """Fetch one coin's spot candle history and return it as an
    ``InstrumentHistory`` named ``<coin>-SPOT`` with ``has_funding=False``.

    No funding fetch is attempted: a spot market never pays or charges
    funding by construction (Hyperliquid's `fundingHistory` answers `null`
    for a spot `coin`, not an error), so `funding` is an empty series and
    `has_funding=False` tells `snapshot.py` that is the CORRECT shape, not a
    gap to flag untradeable -- see `InstrumentHistory.has_funding` and
    docs/TASKS.md T17.
    """
    candles = fetch_candles(client, pair_symbol, interval, start, end)
    if candles.empty:
        raise ValueError(
            f"hyperliquid: no spot candle data for {coin!r} ({pair_symbol}) "
            f"in [{start}, {end}]"
        )
    prices = candles["price"]
    return InstrumentHistory(
        instrument=f"{coin}{SPOT_COLUMN_SUFFIX}",
        prices=prices,
        funding=pd.Series(dtype=float),
        volume=candles["volume"],
        trade_count=candles["trade_count"],
        first_seen=prices.index.min(),
        last_seen=prices.index.max(),
        # See describe_spot_universe: HL's free API exposes no historical
        # delisting flag for spot pairs, so "still listed" is the honest
        # fallback, same as a perp whose feed just ends mid-window.
        is_delisted=False,
        has_funding=False,
        high=candles["high"],
    )


def fetch_spot_universe(
    coins: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    *,
    on_missing: str = "raise",
) -> dict[str, InstrumentHistory]:
    """Fetch spot candle history for each requested coin, keyed by its
    ``<coin>-SPOT`` column name. The spot analogue of `fetch_universe`,
    reusing the exact same pacing/retry (`http_retry`/`polite_sleep`, via
    `_post`/`fetch_candles`) and the same resumable per-instrument cache
    (`fetch_universe_resumable`) -- no second fetch path with its own rules.

    `on_missing` has the same meaning as `fetch_universe`'s: `"raise"` for a
    hand-written coin list (a typo, or a coin with genuinely no spot market,
    must fail loudly), `"skip"` for a venue-discovered list (already
    filtered to coins `fetch_spot_meta` resolved, so this should rarely
    trigger in practice).
    """
    with httpx.Client() as client:
        meta = fetch_meta(client)
        pair_by_coin = fetch_spot_meta(client, list(meta))

        def _fetch_range(column: str, a: pd.Timestamp, b: pd.Timestamp) -> InstrumentHistory:
            coin = column[: -len(SPOT_COLUMN_SUFFIX)]
            pair_symbol = pair_by_coin.get(coin)
            if pair_symbol is None:
                raise ValueError(
                    f"hyperliquid: {coin!r} has no USDC-quoted spot market on "
                    "Hyperliquid (no matching base token in spotMeta)"
                )
            return fetch_spot_instrument_history(client, coin, pair_symbol, interval, a, b)

        columns = [f"{coin}{SPOT_COLUMN_SUFFIX}" for coin in coins]
        return fetch_universe_resumable(
            columns,
            start,
            end,
            interval,
            source=VENUE,
            fetch_one=lambda column: _fetch_range(column, start, end),
            fetch_range=_fetch_range,
            on_missing=on_missing,
            accept_cached=lambda history: history.high is not None,
        )


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
    "fetch_spot_meta",
    "fetch_spot_instrument_history",
    "fetch_spot_universe",
    "describe_spot_universe",
    "discover_spot_universe",
]
