"""Export weekly market-cap rankings for frab's `trend_retune` backfill (T52b).

Reads the stored CoinMarketCap snapshots (data/raw/coinmarketcap/historical) and writes one JSON
document: per snapshot date, the ranked list of Hyperliquid perps
`[hl_symbol, market_cap, cmc_rank]`, top 80, stablecoins excluded, with the mapping and the
exclusion q-lab's `trend-retune-weekly` ran with (`qlab.data.cap_history`). No network is
touched.

    uv run python scripts/export_cap_history.py \\
        --out ../funding-rate-arbitrage/deploy/trend_retune_cap_history.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from qlab.data.cap_history import (
    DEFAULT_TOP,
    build_cap_history,
    hyperliquid_perp_names,
)
from qlab.data.sources import coinmarketcap

DEFAULT_RAW_1D = Path("data/raw/hyperliquid/1d")
DEFAULT_OUT = Path("../funding-rate-arbitrage/deploy/trend_retune_cap_history.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2025-01-01",
                        help="window start of the spec (default 2025-01-01)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--store", type=Path, default=coinmarketcap.DEFAULT_STORE)
    parser.add_argument("--hl-raw", type=Path, default=DEFAULT_RAW_1D,
                        help="cached Hyperliquid daily history; its file names are the perps")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    history = coinmarketcap.load_history(args.store)
    names = hyperliquid_perp_names(args.hl_raw)
    if not names:
        parser.error(f"no Hyperliquid perps found under {args.hl_raw}")
    document = build_cap_history(history, names, date.fromisoformat(args.start), top=args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, separators=(",", ":"), sort_keys=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    snaps = document["snapshots"]
    print(f"{len(snaps)} snapshots {snaps[0]['date']} .. {snaps[-1]['date']}, "
          f"{len(names)} perp names, {args.out.stat().st_size} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
