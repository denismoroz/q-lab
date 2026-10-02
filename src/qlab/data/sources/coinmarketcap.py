"""Point-in-time coin attributes from CoinMarketCap's weekly historical
snapshots (docs/COIN_ATTRIBUTES.md).

Owner, 2026-10-02: "как сделать так, чтобы фреймворк умел выбирать такие вот
варианты, как сейчас выбран с 25 монетами, а не брал торгуемые". The
hand-written trend list reads as "large established projects", and Hyperliquid
serves nothing that says which coin is large or what kind of coin it is: only
prices, volume and funding. A rule that selects like that at a past date needs
the attributes AS THEY WERE on that date.

Why this source. CoinGecko's and CoinPaprika's free APIs refuse market-cap
history older than 365 days (checked 2026-10-02: "Public API users are limited
to querying historical data within the past 365 days"; "Getting daily
historical data before 2025-10-02 ... is not allowed in this plan"). The public
page `coinmarketcap.com/historical/YYYYMMDD/` carries, for the top 200 coins as
of that date, the rank, market cap, 24h volume, circulating supply, the date
the coin was added and its tags. robots.txt allows it ("Allow: /",
"ai-input=yes"). Snapshots are weekly (Sundays).

What is point-in-time and what is not:

- rank, market cap, volume and supply are the snapshot's own (`lastUpdated`
  is the snapshot date);
- `dateAdded` is a historical fact;
- `tags` are attached by the page and may be today's tags rather than the
  snapshot date's; a coin's "memes" or "stablecoin" tag rarely changes, but
  this is a known approximation, recorded here rather than hidden.

A snapshot dated D describes the market at D 00:00 UTC, so a strategy may use
it from D on -- `as_of_frame` makes it visible from the first bar AFTER D.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from qlab.data.sources.base import polite_sleep

DEFAULT_STORE = Path("data/raw/coinmarketcap/historical")
URL = "https://coinmarketcap.com/historical/{:%Y%m%d}/"
USER_AGENT = "Mozilla/5.0 (q-lab research; weekly snapshots, one request at a time)"

# The fields kept per coin; everything else on the page is presentation.
FIELDS = ("cmcRank", "symbol", "name", "slug", "dateAdded", "tags", "circulatingSupply")

_NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def snapshot_dates(start: date, end: date) -> list[date]:
    """Every Sunday in [start, end] -- the dates CoinMarketCap snapshots."""
    first = start + timedelta(days=(6 - start.weekday()) % 7)
    out = []
    d = first
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def _find_key(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def parse_snapshot(html: str) -> list[dict]:
    """The top-200 rows of one historical page, reduced to `FIELDS` plus USD
    `market_cap_usd`, `volume_24h_usd` and `price_usd`.

    Raises ValueError when the page does not carry the listing -- a changed
    page layout must fail loudly, never read as an empty market.
    """
    match = _NEXT_DATA.search(html)
    if match is None:
        raise ValueError("no __NEXT_DATA__ block: the page layout changed")
    state = _find_key(json.loads(match.group(1)), "initialState")
    if isinstance(state, str):
        state = json.loads(state)
    rows = (((state or {}).get("cryptocurrency") or {}).get("listingHistorical") or {}).get("data")
    if not rows:
        raise ValueError("no listingHistorical rows: the page layout changed")
    out = []
    for row in rows:
        usd = (row.get("quote") or {}).get("USD") or {}
        if "cmcRank" not in row or "symbol" not in row or "marketCap" not in usd:
            continue
        kept = {field: row.get(field) for field in FIELDS}
        kept["tags"] = list(row.get("tags") or [])
        kept["market_cap_usd"] = usd.get("marketCap")
        kept["volume_24h_usd"] = usd.get("volume24h")
        kept["price_usd"] = usd.get("price")
        out.append(kept)
    if not out:
        raise ValueError("listingHistorical rows carry no rank/symbol/market cap")
    return out


def fetch_snapshot(client: httpx.Client, day: date) -> list[dict]:
    response = client.get(URL.format(day), headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    polite_sleep()
    return parse_snapshot(response.text)


def download(start: date, end: date, store: Path = DEFAULT_STORE) -> tuple[int, list[date]]:
    """Fetch every missing Sunday snapshot in [start, end] into `store`, one
    JSON file per date. A snapshot is history and never changes, so a file on
    disk is never fetched again. Returns (new files, dates that failed)."""
    store.mkdir(parents=True, exist_ok=True)
    new, failed = 0, []
    with httpx.Client(follow_redirects=True) as client:
        for day in snapshot_dates(start, end):
            path = store / f"{day.isoformat()}.json"
            if path.is_file():
                continue
            try:
                rows = fetch_snapshot(client, day)
            except (httpx.HTTPError, ValueError):
                failed.append(day)
                continue
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"date": day.isoformat(), "rows": rows}), encoding="utf-8")
            tmp.replace(path)
            new += 1
    return new, failed


def load_history(store: Path = DEFAULT_STORE) -> pd.DataFrame:
    """All stored snapshots, one row per (snapshot date, coin)."""
    frames = []
    for path in sorted(store.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["rows"])
        frame["snapshot"] = pd.Timestamp(payload["date"], tz="UTC")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history["is_meme"] = history["tags"].map(lambda tags: "memes" in tags)
    history["is_stablecoin"] = history["tags"].map(lambda tags: "stablecoin" in tags)
    return history


_UNIT_PREFIXES = ("k", "1000", "1000000", "1M")
"""How venues list a low-priced coin per many units: Hyperliquid kPEPE,
Binance 1000PEPE, 1000000MOG, 1MBABYDOGE."""


def venue_symbol(cmc_symbol: str, names: set[str]) -> str | None:
    """The panel column for a CoinMarketCap symbol, or None: the plain symbol,
    else the same coin listed per many units (`_UNIT_PREFIXES`). Symbols are
    not unique on CoinMarketCap, so callers resolve a symbol to the
    HIGHEST-ranked coin carrying it in each snapshot (`as_of_frame`); a symbol
    reused by a later coin maps to the same column, whose prices end when the
    old contract did."""
    if cmc_symbol in names:
        return cmc_symbol
    for prefix in _UNIT_PREFIXES:
        if f"{prefix}{cmc_symbol}" in names:
            return f"{prefix}{cmc_symbol}"
    return None


def hyperliquid_symbol(cmc_symbol: str, hl_names: set[str]) -> str | None:
    """Kept for callers written before Binance; same as `venue_symbol`."""
    return venue_symbol(cmc_symbol, hl_names)


def as_of_frame(
    history: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str], value: str
) -> pd.DataFrame:
    """`value` (e.g. "cmcRank", "market_cap_usd", "is_meme") per panel bar and
    Hyperliquid column, as known at that bar: each snapshot is visible from the
    first bar strictly after its date, forward-filled until the next one, and
    NaN where no snapshot is known yet or the coin was outside the top 200."""
    names = set(columns)
    rows = history.sort_values(["snapshot", "cmcRank"]).copy()
    rows["hl"] = rows["symbol"].map(lambda s: venue_symbol(str(s), names))
    rows = rows.dropna(subset=["hl"]).drop_duplicates(["snapshot", "hl"], keep="first")
    wide = rows.pivot(index="snapshot", columns="hl", values=value)
    wide = wide.reindex(columns=columns)
    # Visible strictly after the snapshot date.
    wide.index = wide.index + pd.Timedelta(microseconds=1)
    combined = wide.reindex(wide.index.union(index)).sort_index()
    # A coin missing from a later snapshot (fell out of the top 200) must not
    # keep its old value: forward-fill whole snapshots, not single cells.
    snap_id = pd.Series(range(len(wide.index)), index=wide.index)
    last_snap = snap_id.reindex(combined.index).ffill()
    filled = wide.reset_index(drop=True).reindex(last_snap.to_numpy())
    filled.index = combined.index
    return filled.reindex(index)


__all__ = [
    "DEFAULT_STORE",
    "as_of_frame",
    "download",
    "fetch_snapshot",
    "hyperliquid_symbol",
    "venue_symbol",
    "load_history",
    "parse_snapshot",
    "snapshot_dates",
]
