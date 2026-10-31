"""What it costs to fetch an idea's hourly candles, and whether a night can afford it.

Night of 2026-10-31: three ideas of the queue of closed ideas each needed the hourly candles of
every Hyperliquid contract (about an hour), and a restamp needed them once more for a window with
another end; the stages that took them ran until the stop time and the agent stages never started.
Later the same day one restamp took 80 minutes while this module's estimate said six: the estimate
priced every named instrument alike (4.4 s, a figure that divided the hour by the number of
FILES, which are two per column), whatever the kind of column and whatever the cache held.

The estimate is now counted in requests. Every column is fetched by paging a venue's endpoints, and
a request costs a measured time (`night/hyperliquid_fetch.yaml`); how many requests a column needs
follows from the venue's page sizes, the window and what the raw cache already holds
(`qlab.data.sources.base.fetch_universe_resumable`):

* a column cached for exactly this window costs nothing;
* a column cached from the same start to an earlier end costs only the gap (about two requests for
  a perpetual, one for a spot market) -- the nightly extension;
* any other column is fetched whole: a Hyperliquid perpetual is candles (5000 a page, only the last
  5000 hours are served) plus funding (500 a page), a spot market is candles only; a Binance
  perpetual is klines (1500 a page) plus funding (1000 a page, one per eight hours), a spot market
  is klines (1000 a page).

A snapshot that answers the request exactly costs no fetch at all. What the fetch estimate cannot
see is the computing after it: the same idea took 80 minutes on data that was already cached. So
the caller may pass the time the idea took last time (`total`), and the estimate is the larger of
the two. A stage that would not finish an idea within its own time limit does not start it, and
the report says «данные слишком тяжелы за ночь»; a hand run with `--allow-heavy` runs it. Reads
the registry and the file names of the raw cache; fetches nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

CONFIG = Path("night/hyperliquid_fetch.yaml")
HEAVY_RU = "данные слишком тяжелы за ночь"
SPOT_SUFFIX = "-SPOT"

# Page sizes of the endpoints (`qlab.data.sources.hyperliquid`, `qlab.data.sources.binance`).
HL_CANDLE_PAGE = 5000  # and the venue serves no more than the last 5000 candles
HL_FUNDING_PAGE = 500
BINANCE_KLINE_PAGE = 1500
BINANCE_SPOT_KLINE_PAGE = 1000
BINANCE_FUNDING_PAGE = 1000
BINANCE_FUNDING_HOURS = 8  # one settlement per eight hours
HL_FUNDING_PADDING_HOURS = 2  # the window is fetched one bar wider on both sides


@dataclass(frozen=True)
class FetchCost:
    """The measured numbers of `night/hyperliquid_fetch.yaml`."""

    hl_seconds_per_request: float
    binance_seconds_per_request: float
    #: Columns assumed for a spec that takes a venue's whole listing when the raw cache does not
    #: show how many there are.
    hl_listing_perps: int
    hl_listing_spots: int
    binance_listing_perps: int
    binance_listing_spots: int


_KEYS = ("hl_seconds_per_request", "binance_seconds_per_request", "hl_listing_perps",
         "hl_listing_spots", "binance_listing_perps", "binance_listing_spots")


def load_config(path: Path = CONFIG) -> FetchCost:
    """`night/hyperliquid_fetch.yaml`; a missing key is an error (no silent default makes a heavy
    fetch look cheap)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        seconds = [float(raw[k]) for k in _KEYS[:2]]
        counts = [int(raw[k]) for k in _KEYS[2:]]
    except KeyError as missing:
        raise ValueError(f"{path}: missing key {missing}") from None
    return FetchCost(*seconds, *counts)


@dataclass(frozen=True)
class Estimate:
    """`minutes` the idea is expected to take (the larger of its data fetch and, when known, the
    time it took last time); `cached` says nothing is to be fetched; `fetch_minutes` is the fetch
    alone."""

    minutes: float
    cached: bool
    why: str
    fetch_minutes: float | None = None

    @property
    def fetch(self) -> float:
        return self.minutes if self.fetch_minutes is None else self.fetch_minutes


# --------------------------------------------------------------------------------------------
# Requests of one column


def hours_between(start: date, end: date) -> int:
    return max(0, (end - start).days * 24)


def requests_whole(venue: str, spot: bool, hours: int) -> int:
    """Requests to fetch one column of `hours` hourly bars from nothing."""
    if venue == "binance":
        if spot:
            return hours // BINANCE_SPOT_KLINE_PAGE + 1
        funding_rows = hours // BINANCE_FUNDING_HOURS + 2
        return hours // BINANCE_KLINE_PAGE + 1 + funding_rows // BINANCE_FUNDING_PAGE + 1
    # Hyperliquid: a full page of candles asks for one more; the venue serves the last 5000 only.
    candles = 2 if hours >= HL_CANDLE_PAGE else 1
    if spot:
        return candles
    return candles + (hours + HL_FUNDING_PADDING_HOURS) // HL_FUNDING_PAGE + 1


def requests_gap(venue: str, spot: bool, gap_hours: int) -> int:
    """Requests to extend a column cached to an earlier end by `gap_hours` (the refetch starts a
    day before the old end, which the caller has added)."""
    return requests_whole(venue, spot, gap_hours)


# --------------------------------------------------------------------------------------------
# What the raw cache holds


def _cache_dir() -> Path:
    from qlab.data.sources.base import DEFAULT_RAW_CACHE_DIR

    return Path(DEFAULT_RAW_CACHE_DIR)


def _stems(folder: Path) -> dict[str, list[str]]:
    """The cached columns of a folder: ``{column: [end date, ...]}`` per start date, as
    ``{f"{column}__{start}": [end, ...]}``, read from the file names alone."""
    found: dict[str, list[str]] = {}
    if not folder.is_dir():
        return found
    for path in folder.glob("*.parquet"):
        parts = path.stem.split("__")
        if len(parts) == 3:
            found.setdefault(f"{parts[0]}__{parts[1]}", []).append(parts[2])
    return found


def _listing(folder: Path) -> tuple[set[str], set[str]]:
    perps, spots = set(), set()
    if folder.is_dir():
        for path in folder.glob("*.parquet"):
            name = path.stem.split("__")[0]
            (spots if name.endswith(SPOT_SUFFIX) else perps).add(name)
    return perps, spots


@dataclass
class _Tally:
    """Columns of one venue and kind by what they need."""

    whole: int = 0
    gap: int = 0
    held: int = 0
    requests: int = 0


def _count(venue: str, interval: str, columns: Sequence[str], start: date, end: date,
           cache_dir: Path) -> dict[bool, _Tally]:
    """Tally the named columns of `venue` (False: perpetuals, True: spot) against the cache."""
    stems = _stems(cache_dir / venue / interval)
    hours = hours_between(start, end)
    out = {False: _Tally(), True: _Tally()}
    for column in columns:
        spot = column.endswith(SPOT_SUFFIX)
        tally = out[spot]
        ends = stems.get(f"{column}__{start.isoformat()}", [])
        if end.isoformat() in ends:
            tally.held += 1
            continue
        earlier = sorted(e for e in ends if e < end.isoformat())
        if earlier:
            last = date.fromisoformat(earlier[-1])
            gap = hours_between(last, end) + 24  # the refetch begins a day before the old end
            tally.gap += 1
            tally.requests += requests_gap(venue, spot, gap)
            continue
        tally.whole += 1
        n = requests_whole(venue, spot, hours)
        if venue == "hyperliquid" and not spot and _funding_reusable(
                cache_dir, venue, interval, column, start, end):
            n -= (hours + HL_FUNDING_PADDING_HOURS) // HL_FUNDING_PAGE + 1
        tally.requests += n
    return out


def _funding_reusable(cache_dir: Path, venue: str, interval: str, column: str, start: date,
                      end: date) -> bool:
    """Funding of this exact window already cached by a fetch at a longer bar
    (`qlab.data.sources.hyperliquid.reusable_funding`): the series is the same whatever the bar."""
    for other in ("1d", "4h"):
        if other == interval:
            continue
        if (cache_dir / venue / other / f"{column}__{start}__{end}.parquet").exists():
            return True
    return False


# --------------------------------------------------------------------------------------------
# The estimate


def _plan(data, cost: FetchCost, cache_dir: Path) -> tuple[list[str], list[str], str, str]:
    """The perpetual columns, the spot columns, the perpetuals' venue and the spot venue of a
    spec's data request. A spec naming no instruments takes the venue's whole listing: the columns
    the raw cache has seen, filled up to the configured size when the cache has seen fewer."""
    venue = data.source
    spot_venue = data.spot_source or venue
    if data.instruments is not None:
        perps = [n for n in data.instruments if not n.endswith(SPOT_SUFFIX)]
        spots = [n for n in data.instruments if n.endswith(SPOT_SUFFIX)]
    else:
        held, _ = _listing(cache_dir / venue / data.interval)
        listed = cost.hl_listing_perps if venue == "hyperliquid" else cost.binance_listing_perps
        perps = sorted(held) + [f"?{i}" for i in range(max(0, listed - len(held)))]
        spots = []
    if data.include_spot:
        _, held_spots = _listing(cache_dir / spot_venue / data.interval)
        if spot_venue == "binance":
            # Binance has a spot market for nearly every coin: one spot leg per perpetual.
            wanted = {f"{p}{SPOT_SUFFIX}" for p in perps}
        elif data.instruments is None:
            wanted = set(held_spots)
        else:
            # Hyperliquid has spot for few coins: the ones the cache has seen.
            wanted = {f"{p}{SPOT_SUFFIX}" for p in perps} & held_spots
        spots = sorted({*spots, *wanted})
        if data.instruments is None and spot_venue == "hyperliquid":
            short = max(0, cost.hl_listing_spots - len(spots))
            spots += [f"?{i}{SPOT_SUFFIX}" for i in range(short)]
    return perps, spots, venue, spot_venue


def _venue_seconds(venue: str, cost: FetchCost) -> float:
    if venue == "binance":
        from qlab.data.sources.base import fetch_workers

        return cost.binance_seconds_per_request / fetch_workers("binance")
    return cost.hl_seconds_per_request


def _describe(label: str, tally: dict[bool, _Tally]) -> str | None:
    parts = []
    for spot, word in ((False, "контрактов"), (True, "спотов")):
        t = tally[spot]
        if t.whole + t.gap + t.held:
            parts.append(f"{t.whole + t.gap + t.held} {word} ({t.whole} целиком, {t.gap} докачать, "
                         f"{t.held} в кэше)")
    return f"{label}: {', '.join(parts)}" if parts else None


def estimate(spec, session, cost: FetchCost, *, cache_dir: Path | None = None) -> Estimate:
    """About how long the data request of `spec` takes to fetch (hourly bars only: a daily panel
    is quick). Hyperliquid or Binance; the spot legs may come from the other venue."""
    from qlab.pipeline.evaluate import resolve_cached_panel

    data = spec.data
    if data.source not in ("hyperliquid", "binance") or data.interval == "1d":
        return Estimate(0.0, True, "daily or not a venue with a fetch estimate: quick")
    if resolve_cached_panel(session, spec) is not None:
        return Estimate(0.0, True, "a snapshot answers the request")
    cache_dir = cache_dir or _cache_dir()
    perps, spots, venue, spot_venue = _plan(data, cost, cache_dir)
    seconds = 0.0
    by_venue: dict[str, dict[bool, _Tally]] = {}
    for source, names in ((venue, perps), (spot_venue, spots)):
        if not names:
            continue
        tally = _count(source, data.interval, names, data.start, data.end, cache_dir)
        seconds += sum(t.requests for t in tally.values()) * _venue_seconds(source, cost)
        merged = by_venue.setdefault(source, {False: _Tally(), True: _Tally()})
        for spot, t in tally.items():
            m = merged[spot]
            m.whole, m.gap, m.held, m.requests = (
                m.whole + t.whole, m.gap + t.gap, m.held + t.held, m.requests + t.requests)
    lines = [text for source, tally in by_venue.items()
             if (text := _describe("Hyperliquid" if source == "hyperliquid" else "Binance", tally))]
    minutes = seconds / 60
    cached = seconds == 0.0
    why = ("часовые свечи: " + "; ".join(lines) + (
        ", всё в кэше" if cached else f", около {minutes:.0f} минут подвоза"))
    return Estimate(minutes, cached, why, fetch_minutes=minutes)


def total(estimates: Sequence[Estimate], last_seconds: float | None) -> Estimate:
    """One estimate for an item that runs several data requests one after another, and the time
    the item took last time when that is known: the fetch cannot be seen from the time, nor the
    computing from the fetch, so the item is expected to take the larger of the two."""
    fetch = sum(e.fetch for e in estimates)
    last = None if last_seconds is None else last_seconds / 60
    why = "; ".join(e.why for e in estimates) or "нечего подвозить"
    if last is not None and last > fetch:
        why = f"в прошлый раз идея заняла {last:.0f} минут, подвоз данных сейчас {fetch:.0f}; {why}"
    return Estimate(max(fetch, last or 0.0), all(e.cached for e in estimates), why,
                    fetch_minutes=fetch)


def last_seconds(ledger_entry, step_id: str, name: str) -> float | None:
    """How long the item took the last time: the stage's own ledger entry (written by every run,
    by hand too), else the journal's record of the item. None when it was never done."""
    seconds = (ledger_entry or {}).get("seconds")
    if isinstance(seconds, int | float) and not isinstance(seconds, bool) and seconds > 0:
        return float(seconds)
    try:
        from qlab.nightflow import journal
        from qlab.registry.db import session_scope

        with session_scope() as session:
            return journal.last_item_seconds(session, step_id, name)
    except Exception:  # noqa: BLE001 - a history that cannot be read is no history
        return None


def heavy_reason(est: Estimate, limit_minutes: float) -> str | None:
    """Why the item is too heavy for a stage with `limit_minutes` for the whole night, or None."""
    if est.minutes <= limit_minutes:
        return None
    return (f"{HEAVY_RU}: оценка {est.minutes:.0f} минут при пределе этапа на всю ночь "
            f"{limit_minutes:g} ({est.why}); вручную: --allow-heavy")


__all__ = ["CONFIG", "Estimate", "FetchCost", "HEAVY_RU", "estimate", "heavy_reason",
           "hours_between", "last_seconds", "load_config", "requests_gap", "requests_whole",
           "total"]
