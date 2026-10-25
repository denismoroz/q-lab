"""Weekly market-cap rankings for the paper book's backfill (T52b).

frab's `trend_retune` book rebuilds its candidates' past from point-in-time inputs. The only one it
cannot get itself is the market-cap ranking of the past: CoinGecko's free API refuses history older
than a year, and q-lab keeps CoinMarketCap's weekly snapshots (`qlab.data.sources.coinmarketcap`).
This module reduces those snapshots to what the book needs -- per snapshot date, the ranked list of
Hyperliquid perps -- using EXACTLY the rule q-lab's validated `trend-retune-weekly` ran with
(`qlab.strategies.live.trend._top_k_by_market_cap_mask` over `coinmarketcap.as_of_frame`):

* a CoinMarketCap symbol maps to the Hyperliquid perp of the same name, else to the same coin listed
  per many units (`coinmarketcap.venue_symbol`: kPEPE, 1000PEPE, ...); a coin with no perp is not
  ranked;
* a symbol carried by several coins resolves to the HIGHEST-ranked one, BEFORE any exclusion
  (`as_of_frame`: `drop_duplicates(["snapshot", "hl"])`); so a stablecoin that outranks the real
  coin of the same symbol removes that symbol, as it did in the stand;
* coins carrying an excluded tag (`stablecoin`, the spec's `exclude_tags`) are dropped;
* the rest are ranked by market cap, largest first, ties by perp name (the stand's
  `rank(method="first")` went by panel column order, which is alphabetical).

The history gate (`top_k_requires_history`) and "has a price at the bar" are applied bar by bar by
the consumer, as in the stand; the file only carries the ranking.

A snapshot dated D describes the market at D 00:00 UTC and is visible from the first bar strictly
AFTER D (`as_of_frame`). The consumer owns that rule; the file therefore starts at the last snapshot
strictly before `start`, the one the first bar of the window sees.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from pathlib import Path

import pandas as pd

from qlab.data.sources import coinmarketcap

FORMAT_VERSION = 1
DEFAULT_TOP = 80
DEFAULT_EXCLUDE_TAGS = ("stablecoin",)
SPOT_SUFFIX = "-SPOT"


def hyperliquid_perp_names(raw_dir: Path) -> set[str]:
    """Every Hyperliquid perp the lab ever fetched daily history for, delisted ones included.

    Read from the metadata files next to the cached candles (`<COIN>__<start>__<end>.json`), so no
    network is needed. Spot columns (`-SPOT`) are not perps and are left out. This is the stand's
    own column set: the panel `trend-retune-weekly` ran on held the full discovered universe,
    survivors and delisted names.
    """
    names = set()
    for path in raw_dir.glob("*.json"):
        coin = path.name.split("__", 1)[0]
        if coin and not coin.endswith(SPOT_SUFFIX):
            names.add(coin)
    return names


def ranked_snapshot(
    rows: pd.DataFrame,
    names: Collection[str],
    *,
    top: int = DEFAULT_TOP,
    exclude_tags: Collection[str] = DEFAULT_EXCLUDE_TAGS,
) -> list[list]:
    """`[hl_symbol, market_cap_usd, cmc_rank]` of one snapshot's rows, largest market cap first."""
    known = set(names)
    wanted = set(exclude_tags)
    frame = rows.sort_values("cmcRank", kind="stable").copy()
    frame["hl"] = frame["symbol"].map(lambda s: coinmarketcap.venue_symbol(str(s), known))
    frame = frame.dropna(subset=["hl"]).drop_duplicates("hl", keep="first")
    frame = frame[~frame["tags"].map(lambda tags: bool(wanted & set(tags)))]
    frame = frame[frame["market_cap_usd"].notna()]
    frame = frame.sort_values(["market_cap_usd", "hl"], ascending=[False, True], kind="stable")
    return [
        [str(r.hl), float(r.market_cap_usd), int(r.cmcRank)]
        for r in frame.head(top).itertuples(index=False)
    ]


def build_cap_history(
    history: pd.DataFrame,
    names: Collection[str],
    start: date,
    *,
    top: int = DEFAULT_TOP,
    exclude_tags: Collection[str] = DEFAULT_EXCLUDE_TAGS,
) -> dict:
    """The export document: one entry per snapshot from the last one strictly before `start` onward.

    Deterministic: the same snapshots and names always give the same document (no timestamps), so a
    rerun that finds nothing new writes the same bytes.
    """
    if history.empty:
        raise ValueError("no CoinMarketCap snapshots: run `qlab data coin-attributes` first")
    cutoff = pd.Timestamp(start, tz="UTC")
    dates = sorted(history["snapshot"].unique())
    before = [d for d in dates if d < cutoff]
    keep = ([before[-1]] if before else []) + [d for d in dates if d >= cutoff]
    snapshots = []
    for snap in keep:
        rows = history[history["snapshot"] == snap]
        snapshots.append({
            "date": pd.Timestamp(snap).date().isoformat(),
            "ranked": ranked_snapshot(rows, names, top=top, exclude_tags=exclude_tags),
        })
    return {
        "version": FORMAT_VERSION,
        "source": "coinmarketcap",
        "rule": (
            "hyperliquid perp by symbol (or per-many-units listing); symbol -> highest-ranked coin "
            "BEFORE exclusion; tags excluded: " + ",".join(sorted(exclude_tags))
            + "; ranked by market cap, ties by perp name. A snapshot dated D is visible from the "
            "first bar strictly after D."
        ),
        "start": start.isoformat(),
        "top": top,
        "names": len(set(names)),
        "snapshots": snapshots,
    }


__all__ = [
    "DEFAULT_EXCLUDE_TAGS",
    "DEFAULT_TOP",
    "FORMAT_VERSION",
    "build_cap_history",
    "hyperliquid_perp_names",
    "ranked_snapshot",
]
