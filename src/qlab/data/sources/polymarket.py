"""Polymarket resolved binary markets: price histories into the raw store -- free public
REST APIs, no key, no account, read-only (docs/OUTCOME_MARKETS.md).

This module READS public market data. It never signs, never places or cancels an order and
holds no wallet: q-lab places no orders (CLAUDE.md). Polymarket is NOT a venue the owner may
trade on -- its geoblock page names Belarus ("Close-Only on Frontend and API",
https://docs.polymarket.com/api-reference/geoblock, read 2026-10-25) -- so this is a research
source for the study of how binary-outcome prices relate to outcomes, nothing more. Whether
the lab may read the public data of a venue closed to the owner is the owner's call and is
asked in docs/OUTCOME_MARKETS.md.

Two public services are used (https://docs.polymarket.com/api-reference/introduction):

- Gamma ``GET /markets?closed=true`` lists markets with their metadata, including the
  resolution (``outcomePrices`` = ``["1", "0"]`` when Yes won) and ``closedTime``. A page is
  at most 100 markets (a larger ``limit`` is silently cut to 100 -- checked 2026-10-25), so
  paging is by ``offset`` until a short page comes back.
- CLOB ``GET /prices-history?market=<token id>`` returns ``{"history": [{"t", "p"}]}`` for an
  outcome token. Checked on 2026-10-25 against a market resolved in November 2024:

  * ``interval=max&fidelity=1440`` returns the whole life of a RESOLVED market at one point
    a day;
  * ``interval=max`` with a finer fidelity (1..360 minutes) returns an EMPTY history for a
    resolved market, and so do ``interval=1w|1d|all``;
  * a ``startTs``/``endTs`` window works at any fidelity but only up to about two weeks
    (``'startTs' and 'endTs' interval is too long`` above ~15 days at fidelity 60), so
    hourly history is fetched in chunks of `HOURLY_CHUNK_DAYS`.

What the data is NOT, so nobody trusts it more than it deserves:

- What ``p`` is (last trade, mid-quote, a volume-weighted price) is not documented on the
  endpoint's page ("does not describe what p represents"). It is stored as ``price`` and
  must not be read as an executable price.
- There is no order-book history. A price here is not a quote one could have hit; spread
  and depth at the time are unknown.
- Every market the Gamma list returns is a market that RESOLVED. A strategy that picks
  contracts by price at time t knows nothing of the markets that were voided or never
  resolved; the survivorship of this list is a property of the source (Gamma has no
  "annulled" status in the fields used here, and `parse_market` drops everything that did not
  end in a clean 1/0).
- After resolution a price walks to 0 or 1. `prices_before_resolution` cuts a series at the
  market's ``closedTime`` -- the only point-in-time-honest way to use it. ``closedTime`` can
  precede ``endDate`` (a market can resolve early), so the cut is on ``closedTime``.
- A sample listed by volume is a sample of liquid markets; `collect_resolved` says in its
  meta file how the list was drawn.

Store layout (`data/raw/` is git-ignored, like the other raw caches):

    <cache_dir>/polymarket/<interval>/<condition id>.parquet   Yes-token price, UTC index
    <cache_dir>/polymarket/<interval>/<condition id>.json      the market as listed + fetch facts

Frame first, metadata last: an entry without its ``.json`` is treated as absent and refetched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import structlog

from qlab.data.sources.base import DEFAULT_RAW_CACHE_DIR, http_retry, polite_sleep

log = structlog.get_logger()

SOURCE = "polymarket"
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
USER_AGENT = "qlab-research/0.1 (read-only market data)"

GAMMA_PAGE_SIZE = 100  # the server's own cap; a larger `limit` is cut to this
DAILY_FIDELITY_MINUTES = 1440
HOURLY_FIDELITY_MINUTES = 60
# The window limit is somewhere between 15 and 20 days at fidelity 60 (2026-10-25); 12 days
# keeps clear of it.
HOURLY_CHUNK_DAYS = 12


@dataclass(frozen=True)
class ResolvedMarket:
    """A binary (Yes/No) Polymarket market that ended in a clean 1/0, as the list gave it."""

    condition_id: str
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    resolved_yes: bool
    start: pd.Timestamp | None  # ``startDate``: when trading was meant to begin
    end_date: pd.Timestamp | None  # ``endDate``: the scheduled end
    closed_time: pd.Timestamp  # when the market was closed -- the point-in-time cut
    volume_usd: float
    neg_risk: bool  # one of several mutually exclusive outcomes of an event
    order_min_size: float | None
    tick_size: float | None


def _ts(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _json_list(value: object) -> list | None:
    """Gamma sends lists as JSON text inside JSON (``"outcomes": "[\\"Yes\\", \\"No\\"]"``)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def parse_market(raw: dict) -> ResolvedMarket | None:
    """The market if it is a binary Yes/No market that resolved cleanly (Yes 1 / No 0 or the
    reverse), else None. A 50/50 payout, an unresolved market, a categorical one or one
    without CLOB token ids (the old AMM markets) is not usable for a price-vs-outcome study
    and is dropped rather than guessed at."""
    outcomes = _json_list(raw.get("outcomes"))
    prices = _json_list(raw.get("outcomePrices"))
    tokens = _json_list(raw.get("clobTokenIds"))
    if outcomes != ["Yes", "No"] or not prices or not tokens:
        return None
    if len(prices) != 2 or len(tokens) != 2:
        return None
    try:
        settled = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    if settled not in ([1.0, 0.0], [0.0, 1.0]):
        return None
    if raw.get("umaResolutionStatus") not in (None, "resolved"):
        return None
    condition_id = raw.get("conditionId")
    closed = _ts(raw.get("closedTime"))
    if not condition_id or closed is None:
        return None
    return ResolvedMarket(
        condition_id=str(condition_id),
        market_id=str(raw.get("id", "")),
        question=str(raw.get("question", "")),
        yes_token_id=str(tokens[0]),
        no_token_id=str(tokens[1]),
        resolved_yes=settled == [1.0, 0.0],
        start=_ts(raw.get("startDate")),
        end_date=_ts(raw.get("endDate")),
        closed_time=closed,
        volume_usd=float(raw.get("volumeNum") or 0.0),
        neg_risk=bool(raw.get("negRisk", False)),
        order_min_size=_float_or_none(raw.get("orderMinSize")),
        tick_size=_float_or_none(raw.get("orderPriceMinTickSize")),
    )


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _get(client: httpx.Client, url: str, params: dict) -> object:
    @http_retry()
    def _do() -> object:
        resp = client.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    result = _do()
    polite_sleep()
    return result


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT})


def iter_closed_markets(
    client: httpx.Client, *, end_after: pd.Timestamp, end_before: pd.Timestamp,
    min_volume_usd: float, order: str = "volumeNum", ascending: bool = False,
) -> Iterator[dict]:
    """Raw closed markets whose scheduled end is in the window, paged until a short page."""
    offset = 0
    while True:
        page = _get(client, f"{GAMMA_URL}/markets", {
            "closed": "true", "limit": GAMMA_PAGE_SIZE, "offset": offset,
            "end_date_min": end_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": end_before.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "volume_num_min": min_volume_usd, "order": order,
            "ascending": str(ascending).lower()})
        if not isinstance(page, list):
            raise RuntimeError(f"Gamma /markets answered with {type(page).__name__}: {page!r:.200}")
        yield from page
        if len(page) < GAMMA_PAGE_SIZE:
            return
        offset += GAMMA_PAGE_SIZE


def _history_frame(history: object) -> pd.Series:
    if not isinstance(history, list) or not history:
        return pd.Series(dtype="float64", name="price")
    frame = pd.DataFrame(history)
    index = pd.to_datetime(frame["t"], unit="s", utc=True)
    series = pd.Series(frame["p"].astype("float64").to_numpy(), index=index, name="price")
    return series[~series.index.duplicated(keep="last")].sort_index()


def fetch_daily_history(client: httpx.Client, token_id: str) -> pd.Series:
    """One point a day for the whole life of a resolved market (empty if the venue serves none)."""
    data = _get(client, f"{CLOB_URL}/prices-history", {
        "market": token_id, "interval": "max", "fidelity": DAILY_FIDELITY_MINUTES})
    return _history_frame(data.get("history") if isinstance(data, dict) else None)


def fetch_hourly_history(
    client: httpx.Client, token_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Hourly points in `HOURLY_CHUNK_DAYS` windows (the endpoint refuses long windows)."""
    parts: list[pd.Series] = []
    cursor = start
    step = pd.Timedelta(days=HOURLY_CHUNK_DAYS)
    while cursor < end:
        stop = min(cursor + step, end)
        data = _get(client, f"{CLOB_URL}/prices-history", {
            "market": token_id, "startTs": int(cursor.timestamp()), "endTs": int(stop.timestamp()),
            "fidelity": HOURLY_FIDELITY_MINUTES})
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"prices-history refused {cursor}..{stop}: {data['error']}")
        parts.append(_history_frame(data.get("history") if isinstance(data, dict) else None))
        cursor = stop
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.Series(dtype="float64", name="price")
    merged = pd.concat(parts)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def prices_before_resolution(prices: pd.Series, closed_time: pd.Timestamp) -> pd.Series:
    """The prices strictly before the market closed. After that a price is the payout."""
    return prices[prices.index < closed_time]


# --------------------------------------------------------------------------
# Store


def _paths(cache_dir: Path, interval: str, condition_id: str) -> tuple[Path, Path]:
    folder = cache_dir / SOURCE / interval
    return folder / f"{condition_id}.parquet", folder / f"{condition_id}.json"


def store_market(
    cache_dir: Path, interval: str, market: ResolvedMarket, prices: pd.Series, *,
    fetched_at: datetime, selection: dict,
) -> None:
    frame_path, meta_path = _paths(cache_dir, interval, market.condition_id)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_frame("price").to_parquet(frame_path)
    meta = {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
            for k, v in asdict(market).items()}
    meta.update({
        "interval": interval, "points": int(len(prices)),
        "first_point": prices.index[0].isoformat() if len(prices) else None,
        "last_point": prices.index[-1].isoformat() if len(prices) else None,
        "fetched_at": fetched_at.astimezone(UTC).isoformat(), "selection": selection,
        "price_meaning": "unspecified by the venue's documentation (see module docstring)",
    })
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_market(cache_dir: Path, interval: str, condition_id: str) -> tuple[dict, pd.Series] | None:
    """(meta, Yes-token price series) of a stored market, or None if absent or half-written."""
    frame_path, meta_path = _paths(cache_dir, interval, condition_id)
    if not frame_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta, pd.read_parquet(frame_path)["price"]
    except Exception:
        return None


def collect_resolved(
    client: httpx.Client, *, cache_dir: Path = DEFAULT_RAW_CACHE_DIR,
    end_after: pd.Timestamp, end_before: pd.Timestamp, min_volume_usd: float,
    max_markets: int, interval: str = "1d", now: datetime | None = None,
) -> dict:
    """Fetch and store up to `max_markets` resolved binary markets ending in the window, the
    largest by volume first. Resumable: a market already stored is skipped (and counted, not
    refetched). Returns counts and the date range of what is now in the store for this draw."""
    if interval not in ("1d", "1h"):
        raise ValueError(f"interval must be '1d' or '1h', got {interval!r}")
    now = now or datetime.now(UTC)
    selection = {
        "list": "gamma /markets closed=true", "end_after": end_after.isoformat(),
        "end_before": end_before.isoformat(), "min_volume_usd": min_volume_usd,
        "order": "volumeNum descending", "max_markets": max_markets,
        "note": "a list ordered by volume: liquid markets only, not a random sample",
    }
    listed = skipped_unusable = fetched = already = empty = 0
    stored: list[ResolvedMarket] = []
    for raw in iter_closed_markets(client, end_after=end_after, end_before=end_before,
                                   min_volume_usd=min_volume_usd):
        listed += 1
        market = parse_market(raw)
        if market is None:
            skipped_unusable += 1
            continue
        if len(stored) >= max_markets:
            break
        if load_market(cache_dir, interval, market.condition_id) is not None:
            already += 1
            stored.append(market)
            continue
        if interval == "1d":
            prices = fetch_daily_history(client, market.yes_token_id)
        else:
            start = market.start or market.closed_time - pd.Timedelta(days=90)
            prices = fetch_hourly_history(client, market.yes_token_id, start,
                                          market.closed_time + pd.Timedelta(hours=1))
        if prices.empty:
            empty += 1
            log.warning("polymarket_no_history", condition_id=market.condition_id,
                        question=market.question[:80])
            continue
        store_market(cache_dir, interval, market, prices, fetched_at=now, selection=selection)
        fetched += 1
        stored.append(market)
    return {
        "listed": listed, "unusable_dropped": skipped_unusable, "fetched": fetched,
        "already_stored": already, "no_history": empty, "in_store": len(stored),
        "yes_won": sum(m.resolved_yes for m in stored),
        "closed_first": min((m.closed_time for m in stored), default=None),
        "closed_last": max((m.closed_time for m in stored), default=None),
        "volume_usd_total": sum(m.volume_usd for m in stored),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end-after", required=True, help="scheduled end on or after (YYYY-MM-DD)")
    ap.add_argument("--end-before", required=True, help="scheduled end on or before (YYYY-MM-DD)")
    ap.add_argument("--min-volume-usd", type=float, required=True)
    ap.add_argument("--max-markets", type=int, required=True)
    ap.add_argument("--interval", choices=("1d", "1h"), default="1d")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    args = ap.parse_args(argv)
    with make_client() as client:
        summary = collect_resolved(
            client, cache_dir=args.cache_dir, end_after=_ts(args.end_after),
            end_before=_ts(args.end_before) + pd.Timedelta(hours=23, minutes=59, seconds=59),
            min_volume_usd=args.min_volume_usd, max_markets=args.max_markets,
            interval=args.interval)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
