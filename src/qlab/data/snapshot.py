"""Build and load ``MarketPanel`` snapshots (docs/PLAN.md, M2 "слой данных";
docs/REGISTRY.md, the ``data_snapshot`` table).

``build_snapshot`` fetches from a source, assembles the four aligned
frames (``prices``, ``funding``, ``tradeable``, ``volume`` — docs/TASKS.md,
T27), writes them under ``data/snapshots/<sha256>/`` as parquet, and
inserts the matching ``data_snapshot`` registry row. ``load_snapshot`` reads
a snapshot back from disk (using the registry row for provenance) with no
network access — this is the path a harness run always takes, so a trial's
inputs never depend on a source's uptime.

Determinism: the snapshot id is::

    sha256(canonical_manifest_json + prices_parquet + funding_parquet
           + tradeable_parquet + volume_parquet)

The manifest freezes every parameter that can change the *result*: source,
instruments (deduplicated and sorted), start, end, interval, and
``universe_complete``. Wall-clock metadata (``fetched_at``) deliberately
never enters the manifest or the hash — it is recorded only on the registry
row and in ``MarketPanel.meta``, so re-running the exact same request at a
different time still produces the same snapshot id. Columns are always
sorted alphabetically before hashing and writing, for the same reason.

``universe_complete`` is part of the manifest (and therefore the hash) even
though it doesn't change a single byte of ``prices``/``funding``/
``tradeable``: two requests that happen to fetch the identical instrument
list — one via `discover_universe` (the honest full universe), one typed by
hand — must NOT collide on the same snapshot id. They are different claims
about the data's provenance, and `honest_universe` needs to be able to tell
them apart even when, by coincidence, they cover exactly the same coins.

``no_funding_instruments`` (docs/TASKS.md, T17) is also part of the
manifest: it lists the columns (spot markets) whose all-NaN ``funding`` is
structural, not a settlement gap — see `qlab.data.sources.base
.InstrumentHistory.has_funding`. Unlike ``universe_complete`` this DOES
follow mechanically from the fetched data (it doesn't need a separate
provenance claim), but it must still be persisted in the manifest, not just
computed at build time: `load_snapshot` reconstructs ``meta`` from
``manifest.json`` alone, with no network access, so anything the harness
needs from ``meta`` after a save/load round trip has to live there.

Survivorship bias has two independent entry points, and this module closes
both. `qlab.data.panel.MarketPanel.tradeable` handles the first: an
instrument that IS in the panel must read False before it listed and after
it delisted. This module's `instruments=None` default handles the second,
which `tradeable` alone cannot: an instrument that was never REQUESTED at
all doesn't show up as a "delisted" column, it just silently isn't there. A
hand-typed ``--instruments BTC,ETH,SOL`` is the tickers a person remembers —
i.e. the survivors — and no amount of per-column point-in-time correctness
fixes a universe that was hand-picked before the fetch even started. See
`docs/PLAN.md` and funding-rate-arbitrage's XSMOM stress test
(research/cross_sectional/crypto/survivorship.py), which measured exactly
this defect at 0.45 Sharpe.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from qlab.data.panel import MarketPanel
from qlab.data.sources import binance as binance_source
from qlab.data.sources import hyperliquid as hyperliquid_source
from qlab.data.sources.base import (
    SPOT_COLUMN_SUFFIX,
    InstrumentHistory,
    align_funding_to_index,
    detect_bad_price_bars,
)
from qlab.registry import repo
from qlab.registry.db import session_scope
from qlab.registry.models import DataSnapshot

logger = structlog.get_logger()

DEFAULT_SNAPSHOTS_DIR = Path("data/snapshots")

_INTERVAL_TO_PANDAS_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

_SOURCES = {
    hyperliquid_source.VENUE: {
        "fetch": hyperliquid_source.fetch_universe,
        "funding_native_interval": hyperliquid_source.FUNDING_NATIVE_INTERVAL,
        "discover_universe": hyperliquid_source.discover_universe,
        "describe_universe": hyperliquid_source.describe_universe,
        # Spot support (docs/TASKS.md, T17) is Hyperliquid-only for now --
        # these three keys are absent from binance's spec below, and every
        # call site treats their absence as "this source has no spot
        # markets" via `.get(...)`, never as an error.
        "fetch_spot": hyperliquid_source.fetch_spot_universe,
        "discover_spot_universe": hyperliquid_source.discover_spot_universe,
        "describe_spot_universe": hyperliquid_source.describe_spot_universe,
    },
    binance_source.VENUE: {
        "fetch": binance_source.fetch_universe,
        "funding_native_interval": binance_source.FUNDING_NATIVE_INTERVAL,
        "discover_universe": binance_source.discover_universe,
        "describe_universe": binance_source.describe_universe,
    },
}


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _canonical_manifest(
    source: str,
    instruments: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    universe_complete: bool,
    no_funding_instruments: list[str],
    min_daily_volume_usd: float | None,
) -> dict:
    return {
        "source": source,
        "instruments": instruments,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "universe_complete": universe_complete,
        # Which columns structurally never carry funding (spot markets) --
        # part of the manifest (and therefore the snapshot id) for the same
        # reason `universe_complete` is: `load_snapshot` reconstructs
        # `meta` from this file alone, with no network access, so the fact
        # must be persisted here to survive a save/load round trip.
        "no_funding_instruments": no_funding_instruments,
        # The point-in-time liquidity filter applied while building
        # `tradeable` (docs/TASKS.md, T27, gap 2), or None if none was
        # requested. Already baked into `tradeable_bytes` below (a
        # different threshold produces different tradeable bytes and
        # therefore a different id on its own), but it is recorded
        # explicitly here too so `load_snapshot` can hand it back in
        # `meta` without a network call, the same reasoning as
        # `no_funding_instruments`.
        "min_daily_volume_usd": min_daily_volume_usd,
    }


def _manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _dataframe_bytes(df: pd.DataFrame) -> bytes:
    """Canonical parquet bytes for hashing and on-disk storage: fixed
    compression codec, index written as a named column so it round-trips
    through ``pd.read_parquet`` as the same tz-aware DatetimeIndex."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="zstd", index=True)
    return buf.getvalue()


def _build_frames_from_histories(
    histories: dict[str, InstrumentHistory],
    full_index: pd.DatetimeIndex,
    funding_native_interval: pd.Timedelta,
    *,
    min_daily_volume_usd: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble and return ``(prices, funding, tradeable, volume)`` from raw
    per-instrument histories.

    ``min_daily_volume_usd`` (docs/TASKS.md, T27, gap 2), when given, adds a
    POINT-IN-TIME liquidity exclusion on top of the existing listing/
    funding-gap/bad-price exclusions already applied to ``tradeable`` below
    — see `_liquidity_eligible_mask` for the exact trailing-window
    arithmetic and why it cannot look ahead.
    """
    bar_interval = (
        full_index[1] - full_index[0] if len(full_index) > 1 else funding_native_interval
    )
    instruments = sorted(histories)
    price_cols: dict[str, pd.Series] = {}
    funding_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}
    tradeable_cols: dict[str, pd.Series] = {}

    for coin in instruments:
        hist = histories[coin]
        price_cols[coin] = hist.prices.reindex(full_index)
        volume_cols[coin] = hist.volume.reindex(full_index)
        funding_cols[coin] = align_funding_to_index(
            hist.funding, full_index, funding_native_interval
        )

        listed = full_index >= hist.first_seen
        still_listed = (not hist.is_delisted) | (full_index <= hist.last_seen)
        tradeable = pd.Series(listed & still_listed, index=full_index)

        # A bar whose funding rate is unknown is a bar we cannot honestly
        # simulate holding through, so it is not tradeable. Venues do drop
        # settlements — Hyperliquid has gaps of a few hours on some coins —
        # and the alternatives are both wrong: treating the gap as zero
        # funding hands a carry strategy a free ride, and dropping the
        # instrument entirely re-introduces the survivorship this module
        # exists to prevent. Excluding just the affected bars keeps the coin
        # in the universe and leaves the strategy unable to hold it exactly
        # where the data cannot support the claim.
        #
        # Only when the panel's bar is coarser than or equal to the venue's
        # settlement interval, where NaN means "incomplete". On a finer panel
        # (hourly bars over 8h funding) most bars are NaN by construction and
        # this test would make everything untradeable.
        #
        # AND only when `hist.has_funding` is True. An instrument that
        # structurally never pays funding (a spot market, see
        # `InstrumentHistory.has_funding`) is all-NaN here by construction,
        # not because a settlement was dropped -- applying this rule to it
        # would mark every spot bar untradeable, which is the exact trap
        # docs/TASKS.md T17 warns about, not an honest data gap.
        if bar_interval >= funding_native_interval and hist.has_funding:
            tradeable &= funding_cols[coin].notna()

        # Quote sanity (docs/TASKS.md, T17): a corrupted print -- a
        # repeating placeholder price, or an implausible one-bar jump --
        # must never enter the panel as if it were a real observation. See
        # `detect_bad_price_bars` for what triggered this and why: live
        # Hyperliquid spot data for a newly created pair returns a constant
        # non-zero "close" with no real trading behind it until the pair
        # actually starts trading, at which point the price jumps to a
        # realistic level -- which without this check reads as a huge, and
        # entirely fictitious, one-bar return. The bad bar is BOTH NaN'd out
        # (so no signal computed over `prices` can see it either) and
        # cleared from `tradeable` (so a strategy cannot hold through it) --
        # the same "exclude just the affected bars" policy as the funding
        # gap above, not a guess and not a whole-instrument drop.
        bad_price = detect_bad_price_bars(price_cols[coin])
        if bad_price.any():
            logger.warning(
                "bad_price_bars_excluded",
                instrument=coin,
                n_bars=int(bad_price.sum()),
                first=str(price_cols[coin].index[bad_price][0]),
                last=str(price_cols[coin].index[bad_price][-1]),
            )
            price_cols[coin] = price_cols[coin].where(~bad_price)
            volume_cols[coin] = volume_cols[coin].where(~bad_price)
            tradeable &= ~bad_price

        tradeable_cols[coin] = tradeable

    prices = pd.DataFrame(price_cols, index=full_index)[instruments]
    prices.index.name = "timestamp"
    funding = pd.DataFrame(funding_cols, index=full_index)[instruments]
    funding.index.name = "timestamp"
    volume = pd.DataFrame(volume_cols, index=full_index)[instruments]
    volume.index.name = "timestamp"
    tradeable = pd.DataFrame(tradeable_cols, index=full_index)[instruments].astype(bool)
    tradeable.index.name = "timestamp"

    if min_daily_volume_usd is not None:
        tradeable &= _liquidity_eligible_mask(prices, volume, bar_interval, min_daily_volume_usd)

    return prices, funding, tradeable, volume


# HL's own live curation (`funding-rate-arbitrage/research/cross_sectional/
# crypto/cryptodata.py`, `MIN_VOL_USD = 1_000_000  # >= $1M 24h notional
# volume on HL`) filters on the venue's own rolling **24-HOUR** notional
# volume figure (`dayNtlVlm` from `metaAndAssetCtxs`) -- not a multi-day
# average, and not a number invented here. `_LIQUIDITY_WINDOW` translates
# that same 24h span onto whatever bar interval the panel actually uses: at
# a daily bar it is exactly 1 trailing bar (the previous day), at an hourly
# bar it is the trailing 24 bars summed. This is a straight unit conversion
# of the live engine's own metric, not a new threshold.
_LIQUIDITY_WINDOW = pd.Timedelta(days=1)


def _liquidity_eligible_mask(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    bar_interval: pd.Timedelta,
    min_daily_volume_usd: float,
) -> pd.DataFrame:
    """Point-in-time liquidity eligibility (docs/TASKS.md, T27, gap 2):
    ``True`` at ``(t, instrument)`` iff the TRAILING 24h of USD volume
    ending strictly BEFORE ``t`` (never including ``t`` itself) was at
    least ``min_daily_volume_usd``.

    This is what makes the filter honest and non-look-ahead: an
    instrument's eligibility at bar ``t`` depends only on volume observed
    at bars before ``t``, exactly the same "decide using only information
    available up to and including the PREVIOUS decision point" discipline
    `qlab.harness.strategy`'s alignment rule already requires of a
    strategy's own weights. Filtering on volume measured at or after ``t``
    -- in particular on the window's OWN endpoint or on "today's" volume --
    would be look-ahead of the exact shape T21 already found for XSMOM's
    survivorship (a fixed universe curated with 2026-06 information and
    applied backward to 2025-01): a coin that is thin now and liquid later
    must read ineligible NOW regardless of what it becomes.

    Arithmetic: ``bars = round(_LIQUIDITY_WINDOW / bar_interval)`` trailing
    bars of ``volume * prices`` are summed with `pandas.DataFrame.rolling`
    (``min_periods=bars``, so an incomplete window -- not enough history
    yet -- is NaN, never a partial sum passed off as a full day's volume,
    same "never silently pass off a short bucket" rule
    `qlab.data.sources.base.align_funding_to_index` already applies to
    funding), then shifted forward by one bar so the sum ending at ``t-1``
    is compared at row ``t`` -- never the sum ending at ``t`` itself. NaN
    (unknown -- either not enough history yet, or volume itself unknown)
    is treated as INELIGIBLE, not as a free pass: an unproven instrument
    does not default to tradeable.

    Returns a boolean DataFrame the same shape as ``prices``, intended to
    be AND-ed into ``tradeable`` — this narrows an already-listed
    instrument's tradeable span, it never widens it and never touches
    which instruments are IN the panel at all (see `build_snapshot`'s
    ``min_daily_volume_usd`` docstring for why that distinction matters for
    `universe_complete`/`point_in_time_universe`).
    """
    bars = round(_LIQUIDITY_WINDOW / bar_interval)
    bars = max(bars, 1)
    volume_usd = volume * prices
    trailing = volume_usd.rolling(window=bars, min_periods=bars).sum().shift(1)
    return (trailing >= min_daily_volume_usd).fillna(False)


def _top_k_by_volume_mask(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    bar_interval: pd.Timedelta,
    k: int,
) -> pd.DataFrame:
    """Point-in-time "the k most traded instruments", the ranked counterpart
    of `_liquidity_eligible_mask`.

    WHY A RANK AND NOT ONLY A THRESHOLD. A threshold changes how MANY
    instruments survive as the market's overall liquidity moves, and book
    breadth is not a neutral detail: the same idea on 20 legs and on 148
    behaved like two different strategies in `docs/XSMOM_T21.md`, and a
    comparison that varies the selection rule AND the leg count at once
    measures neither. Holding `k` fixed is what makes "the live engine's
    hand-written 25-coin list" and "the 25 most traded coins as of each
    bar" answer the same question with one difference between them: WHEN
    the choice was made.

    Same causality as the threshold mask and for the same reason: the
    ranking at bar ``t`` uses the trailing 24h of USD volume ending
    strictly BEFORE ``t``. An instrument with unknown trailing volume (not
    enough history, or volume itself unknown) cannot be ranked and is
    excluded, never ranked last-but-included.

    Ties are broken by `rank(method="first")`, i.e. by column order, so the
    mask is deterministic. Exact ties in a 24h dollar volume are
    vanishingly unlikely on real data; the determinism matters for
    reproducibility, not for fairness between instruments.
    """
    bars = max(round(_LIQUIDITY_WINDOW / bar_interval), 1)
    volume_usd = volume * prices
    trailing = volume_usd.rolling(window=bars, min_periods=bars).sum().shift(1)
    ranks = trailing.rank(axis=1, ascending=False, method="first")
    return (ranks <= k).fillna(False)


def describe_universe(source: str, *, include_spot: bool = False) -> list[tuple[str, bool]] | None:
    """Return ``[(instrument, is_delisted), ...]`` for `source`'s full
    point-in-time universe, or ``None`` if the source's free API cannot
    expose delisted names at all (see that source's `discover_universe`
    docstring — this is a real, permanent limitation for Binance, not a
    bug). Used by both `build_snapshot`'s ``instruments=None`` default and
    the ``qlab data universe`` CLI command.

    ``include_spot=True`` additionally merges in the source's spot markets
    (``<COIN>-SPOT`` names, see `qlab.data.sources.base.SPOT_COLUMN_SUFFIX``)
    when the source supports them (Hyperliquid; absent from ``_SOURCES`` for
    a source that doesn't, e.g. Binance — silently a no-op there, not an
    error, since "no spot support" is a fact about the venue, same as "no
    survivorship-free perp universe"). Defaults to ``False`` so a plain
    `describe_universe(source)` call — in particular `build_snapshot`'s own
    internal use for the PERP-only discovery step — never triggers an extra
    network round trip it didn't ask for.
    """
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(_SOURCES)}")
    spec = _SOURCES[source]
    perp = spec["describe_universe"]()
    if perp is None or not include_spot:
        return perp
    describe_spot = spec.get("describe_spot_universe")
    if describe_spot is None:
        return perp
    return sorted(perp + describe_spot())


def _resolve_casing(source_spec, instruments, start_ts, end_ts) -> list[str]:
    """Map hand-typed tickers onto the venue's own spelling, case-insensitively.

    Venue casing is not cosmetic: Hyperliquid's thousand-multiple contracts
    are kBONK, kPEPE, kSHIB, and the uppercase spelling is not an alias for
    them — it is an unknown instrument the API answers with a 500.
    """
    try:
        discovered = source_spec["discover_universe"]((start_ts, end_ts)) or []
    except Exception:
        discovered = []
    by_lower = {name.lower(): name for name in discovered}
    return sorted({by_lower.get(i.lower(), i) for i in instruments})


def build_snapshot(
    source: str,
    instruments: Sequence[str] | None,
    start: object,
    end: object,
    interval: str,
    *,
    include_spot: bool = False,
    min_daily_volume_usd: float | None = None,
    snapshots_dir: Path | str = DEFAULT_SNAPSHOTS_DIR,
    session: Session | None = None,
) -> MarketPanel:
    """Fetch, assemble, persist and register a ``MarketPanel`` snapshot.

    ``min_daily_volume_usd`` (docs/TASKS.md, T27, gap 2) is an optional
    POINT-IN-TIME liquidity floor: an instrument reads ``tradeable=False``
    at any bar where its trailing 24h of USD volume (observed strictly
    BEFORE that bar — see `_liquidity_eligible_mask`) fell short of this
    threshold. It exists because `qlab.harness.metrics.min_capital_usd`
    only ever checks a venue's minimum ORDER size, which a thin memecoin
    clears just as easily as BTC — the exact gap that let a 20-leg XSMOM
    book of GRIFFAIN/FARTCOIN/MERL/VVV/ZORA/PURR-type names read as
    "affordable" up to $120k, a size at which each leg is fully
    unexecutable (docs/XSMOM_T21.md, "Ревизия").

    This does NOT touch which instruments are IN the panel, and does NOT
    affect ``universe_complete``/``instruments`` — an instrument excluded
    here for being thin during some span is not the same claim as one that
    was never requested at all (that is what ``universe_complete`` is
    about), so a liquidity-filtered snapshot with a discovered instrument
    list still reads ``universe_complete=True``, exactly like an unfiltered
    one. The filter only ever narrows ``tradeable``, the same mechanism
    already used for delisting, funding gaps and bad-price bars above.

    ``instruments=None`` (the recommended default) fetches the source's full
    discovered universe — survivors and delisted names alike — via
    `discover_universe`, and the resulting panel is marked
    ``meta["universe_complete"] = True``. Passing an explicit instrument
    list is a manual, potentially survivorship-biased selection (the
    tickers a person remembers), so it is always marked
    ``universe_complete=False``, even if it happens to name every
    instrument the source has. Raises if ``instruments`` is omitted and the
    source cannot discover a full universe from its free API (Binance) —
    there is no honest way to fetch "everything" there, so the caller must
    pass an explicit list and accept ``universe_complete=False``.

    ``include_spot=True`` additionally discovers and fetches the source's
    spot markets (``<COIN>-SPOT`` columns, see
    `qlab.data.sources.base.SPOT_COLUMN_SUFFIX`) alongside the perp universe
    — required by any strategy holding spot against a perp leg (docs/TASKS.md,
    T17). Only valid together with ``instruments=None``: naming an explicit
    instrument list is already a full, literal statement of what to fetch,
    including any ``-SPOT`` columns the caller wants (typed exactly, e.g.
    ``"BTC-SPOT"``) — ``include_spot`` would either be redundant or silently
    add columns nobody asked for, so it raises instead of guessing. Raises if
    the source has no spot fetcher registered (only Hyperliquid does today).

    ``universe_complete`` composes across both halves rather than reporting
    on the perp side alone: it is ``True`` only when the perp universe AND
    (if ``include_spot``) the spot universe were BOTH fully discovered. A
    complete perp universe says nothing about the spot side — they are
    fetched from different venue endpoints with independent (and, for spot,
    permanently weaker — see `describe_spot_universe`) delisting visibility
    — so claiming completeness from the perp half alone would be exactly the
    kind of unearned "honest_universe" pass this flag exists to prevent.
    When ``include_spot=False`` the spot half was never requested at all and
    doesn't factor in, so behaviour for existing (perp-only) callers is
    unchanged.

    Idempotent: because the id is a pure function of the request (see module
    docstring), re-running with identical inputs recomputes the same id and
    then skips re-writing files / re-inserting the registry row if they
    already exist — it still re-fetches from the source, since the id can
    only be known once the data is in hand (no way to look up "have we
    already fetched this?" from parameters alone without a network round
    trip anyway).
    """
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(_SOURCES)}")
    if interval not in _INTERVAL_TO_PANDAS_FREQ:
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of {sorted(_INTERVAL_TO_PANDAS_FREQ)}"
        )
    if include_spot and instruments is not None:
        raise ValueError(
            "include_spot is only valid when instruments is omitted (discovered universe); "
            "for a hand-picked list, name the '-SPOT' columns you want directly in "
            "`instruments` instead (e.g. ['BTC', 'BTC-SPOT'])"
        )
    if min_daily_volume_usd is not None and min_daily_volume_usd <= 0:
        raise ValueError(f"min_daily_volume_usd must be > 0, got {min_daily_volume_usd}")

    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start {start_ts} is after end {end_ts}")

    source_spec = _SOURCES[source]

    if instruments is None:
        discovered = source_spec["discover_universe"]((start_ts, end_ts))
        if discovered is None:
            raise ValueError(
                f"{source!r} does not expose a full point-in-time universe via its free "
                "API (delisted instruments aren't listed anywhere) — pass an explicit "
                "instruments list; the resulting snapshot will be marked "
                "universe_complete=False and will fail the honest_universe rule"
            )
        # The venue's own spelling is authoritative and must survive intact.
        # Hyperliquid's thousand-multiple contracts are named with a lowercase
        # k — kBONK, kPEPE, kSHIB — and asking for "KBONK" returns a 500, not
        # a not-found. Upper-casing a discovered name therefore does not
        # normalise it, it invents an instrument that does not exist.
        instruments_sorted = sorted(set(discovered))
        perp_complete = True

        spot_complete = True
        if include_spot:
            discover_spot = source_spec.get("discover_spot_universe")
            if discover_spot is None:
                raise ValueError(f"{source!r} does not support spot markets")
            discovered_spot = discover_spot((start_ts, end_ts))
            if discovered_spot is None:
                spot_complete = False
            else:
                instruments_sorted = sorted(set(instruments_sorted) | set(discovered_spot))

        # See this function's docstring: completeness is the AND of both
        # halves, not the perp half alone.
        universe_complete = perp_complete and spot_complete
    else:
        # A hand-written list may be typed in any case, so resolve it against
        # the venue's spelling where discovery is available; anything that
        # does not resolve is passed through as typed, so a genuine typo still
        # fails loudly rather than being silently rewritten.
        instruments_sorted = _resolve_casing(source_spec, instruments, start_ts, end_ts)
        universe_complete = False

    if not instruments_sorted:
        raise ValueError("instruments must not be empty")

    # An instrument with no data means different things depending on where the
    # list came from: a typo in a hand-written one, ordinary point-in-time
    # truth in a discovered one. See each source's fetch_universe docstring.
    #
    # Perp and spot columns are dispatched to different fetchers (see
    # SPOT_COLUMN_SUFFIX) regardless of which branch above produced the
    # list -- so `instruments=["BTC", "BTC-SPOT"]` routes correctly even
    # without `include_spot` (which only controls AUTO-discovery).
    on_missing = "skip" if universe_complete else "raise"
    perp_instruments = [i for i in instruments_sorted if not i.endswith(SPOT_COLUMN_SUFFIX)]
    spot_coins = [
        i[: -len(SPOT_COLUMN_SUFFIX)] for i in instruments_sorted if i.endswith(SPOT_COLUMN_SUFFIX)
    ]

    histories: dict[str, InstrumentHistory] = {}
    if perp_instruments:
        histories.update(
            source_spec["fetch"](
                perp_instruments, start_ts, end_ts, interval, on_missing=on_missing
            )
        )
    if spot_coins:
        fetch_spot = source_spec.get("fetch_spot")
        if fetch_spot is None:
            raise ValueError(f"{source!r} does not support spot markets (requested {spot_coins})")
        histories.update(fetch_spot(spot_coins, start_ts, end_ts, interval, on_missing=on_missing))

    if not histories:
        raise ValueError(
            f"no instrument had data in [{start_ts}, {end_ts}] at interval {interval!r}"
        )
    instruments_sorted = sorted(histories)

    full_index = pd.date_range(start_ts, end_ts, freq=_INTERVAL_TO_PANDAS_FREQ[interval], tz="UTC")
    if len(full_index) == 0:
        raise ValueError(f"empty index for range [{start_ts}, {end_ts}] at interval {interval!r}")

    prices, funding, tradeable, volume = _build_frames_from_histories(
        histories,
        full_index,
        source_spec["funding_native_interval"],
        min_daily_volume_usd=min_daily_volume_usd,
    )

    no_funding_instruments = sorted(
        name for name, hist in histories.items() if not hist.has_funding
    )

    manifest = _canonical_manifest(
        source,
        instruments_sorted,
        start_ts,
        end_ts,
        interval,
        universe_complete,
        no_funding_instruments,
        min_daily_volume_usd,
    )
    manifest_bytes = _manifest_bytes(manifest)
    prices_bytes = _dataframe_bytes(prices)
    funding_bytes = _dataframe_bytes(funding)
    tradeable_bytes = _dataframe_bytes(tradeable)
    volume_bytes = _dataframe_bytes(volume)

    snapshot_id = hashlib.sha256(
        manifest_bytes
        + b"\0"
        + prices_bytes
        + b"\0"
        + funding_bytes
        + b"\0"
        + tradeable_bytes
        + b"\0"
        + volume_bytes
    ).hexdigest()

    snapshot_dir = Path(snapshots_dir) / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    _write_if_absent(snapshot_dir / "prices.parquet", prices_bytes)
    _write_if_absent(snapshot_dir / "funding.parquet", funding_bytes)
    _write_if_absent(snapshot_dir / "tradeable.parquet", tradeable_bytes)
    _write_if_absent(snapshot_dir / "volume.parquet", volume_bytes)
    _write_if_absent(snapshot_dir / "manifest.json", manifest_bytes)

    fetched_at = datetime.now(UTC)

    def _persist(sess: Session) -> None:
        if sess.get(DataSnapshot, snapshot_id) is None:
            repo.add_data_snapshot(
                sess,
                id=snapshot_id,
                source=source,
                # DataSnapshot.instruments is a JSON *mapping* column
                # (registry.repo.add_data_snapshot does `dict(instruments)`),
                # so the sorted instrument list is stored as {name: True}
                # rather than a bare list.
                instruments=dict.fromkeys(instruments_sorted, True),
                range_start=start_ts.date(),
                range_end=end_ts.date(),
                path=str(snapshot_dir),
                rows=len(full_index),
                fetched_at=fetched_at,
            )

    if session is not None:
        _persist(session)
    else:
        with session_scope() as sess:
            _persist(sess)

    meta = {
        "venue": source,
        "interval": interval,
        "fetched_at": fetched_at.isoformat(),
        "range_start": start_ts.isoformat(),
        "range_end": end_ts.isoformat(),
        "instruments": instruments_sorted,
        "universe_complete": universe_complete,
        "no_funding_instruments": no_funding_instruments,
        "min_daily_volume_usd": min_daily_volume_usd,
    }
    return MarketPanel(
        snapshot_id=snapshot_id,
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        volume=volume,
        meta=meta,
    )


def _write_if_absent(path: Path, data: bytes) -> None:
    if not path.exists():
        path.write_bytes(data)


def load_snapshot(snapshot_id: str, *, session: Session | None = None) -> MarketPanel:
    """Load a previously built snapshot from disk. No network access — only
    the local parquet files and the local registry row are read."""

    def _load(sess: Session) -> MarketPanel:
        row = sess.get(DataSnapshot, snapshot_id)
        if row is None:
            raise ValueError(f"no data_snapshot row for id {snapshot_id!r}")

        snapshot_dir = Path(row.path)
        manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))

        prices = pd.read_parquet(snapshot_dir / "prices.parquet")
        funding = pd.read_parquet(snapshot_dir / "funding.parquet")
        tradeable = pd.read_parquet(snapshot_dir / "tradeable.parquet").astype(bool)

        volume_path = snapshot_dir / "volume.parquet"
        if volume_path.exists():
            volume = pd.read_parquet(volume_path)
        else:
            # A snapshot built before volume support existed (docs/TASKS.md,
            # T27) has no volume.parquet on disk at all. NaN ("unknown"),
            # not 0.0 ("no trading") -- see `InstrumentHistory`'s docstring
            # for why those are different facts. `MarketPanel`'s own
            # `volume=None` default does exactly this backfill, but it is
            # done explicitly here (not just left to the default) so this
            # branch documents WHY a legacy snapshot ends up that way.
            volume = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)

        meta = {
            "venue": row.source,
            "interval": manifest["interval"],
            "fetched_at": row.fetched_at.isoformat(),
            "range_start": row.range_start.isoformat(),
            "range_end": row.range_end.isoformat(),
            "instruments": manifest["instruments"],
            "universe_complete": manifest["universe_complete"],
            # `.get(..., [])`: a snapshot built before spot support existed
            # has no such key in its manifest.json -- absence there means
            # "no instrument was flagged", which is the correct read for an
            # all-perp snapshot, not a data-loss error.
            "no_funding_instruments": manifest.get("no_funding_instruments", []),
            # `.get(..., None)`: a snapshot built before T27 has no such key
            # -- None correctly reads as "no liquidity filter was ever
            # applied", not a data-loss error.
            "min_daily_volume_usd": manifest.get("min_daily_volume_usd"),
        }
        return MarketPanel(
            snapshot_id=snapshot_id,
            prices=prices,
            funding=funding,
            tradeable=tradeable,
            volume=volume,
            meta=meta,
        )

    if session is not None:
        return _load(session)
    with session_scope() as sess:
        return _load(sess)


__all__ = ["build_snapshot", "load_snapshot", "describe_universe", "DEFAULT_SNAPSHOTS_DIR"]
