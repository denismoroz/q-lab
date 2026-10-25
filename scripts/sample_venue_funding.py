"""Sample recent funding rates from public APIs of perpetual DEXes and annualise them.

Written 2026-10-25 for docs/FRAB_VENUES.md. Public, key-less GET/POST endpoints only;
no orders, no accounts. Output is a CSV table on stdout (one row per venue x coin).

Endpoints (all read 2026-10-25; interval in brackets):
  hyperliquid  POST api.hyperliquid.xyz/info {"type":"fundingHistory"}  [1 h]
  aster        GET  fapi.asterdex.com/fapi/v1/fundingRate               [1/2/4/8 h per symbol]
  dydx         GET  indexer.dydx.trade/v4/historicalFunding/{ticker}    [1 h]
  backpack     GET  api.backpack.exchange/api/v1/fundingRates           [1 h]
  extended     GET  api.starknet.extended.exchange/api/v1/info/{m}/funding  [1 h]
  edgex        GET  edgex-prod-v2.edgex.exchange/api/v2/public/funding/getFundingRatePage  [4 h]
  apex         GET  omni.apex.exchange/api/v3/history-funding           [1 h]
  orderly      GET  api-evm.orderly.org/v1/public/funding_rate_history  [8 h]
  aevo         GET  api.aevo.xyz/funding-history                        [1 h]
  grvt         POST market-data.grvt.io/full/v1/funding                 [8 h, rate in percent]
  pacifica     GET  api.pacifica.fi/api/v1/funding_rate/history         [1 h]

How the annual figure is computed (the same for every venue):
  every published print is a payment of `rate` (fraction of notional) for one funding
  interval of `h` hours. Over the window the mean hourly rate is
      sum(rate) / sum(h)
  and the annual figure is that times 8760, in percent. A positive number means longs pay
  shorts, i.e. the short-perp leg of FRAB earns it. The interval `h` is the venue's nominal
  one where the docs fix it (hourly venues, Orderly 8 h, GRVT 8 h); for Aster and edgeX it is
  the gap to the previous print, because it differs by symbol and can change over time.
  Windows are counted back from the moment of the run; a venue whose history is shorter than
  the window is reported with the days it actually covers (`days_covered`).

Usage:
  python3 -I scripts/sample_venue_funding.py [--cache DIR] [--coins BTC,ETH,...] [--venues a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
WINDOWS = (30, 90, 365)
DEFAULT_COINS = ("BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "LINK", "AVAX")
UA = {"User-Agent": "qlab-research/0.1 (read-only funding sample)", "Accept": "application/json"}

# A print: (timestamp_ms, rate as a fraction of notional per interval, interval_hours or None)
Print = tuple[int, float, float | None]


def http(url: str, body: dict | None = None, retries: int = 4) -> object:
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(UA)
    if body is not None:
        headers["Content-Type"] = "application/json"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 404):
                raise
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - network noise, retried
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


# --- per-venue fetchers: return prints inside [t0, now], any order ---------------------------


def f_hyperliquid(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    start = t0
    while start < now:
        rows = http(
            "https://api.hyperliquid.xyz/info",
            {"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": now},
        )
        if not rows:
            break
        out += [(r["time"], float(r["fundingRate"]), 1.0) for r in rows]
        nxt = rows[-1]["time"] + 1
        if nxt <= start or len(rows) < 2:
            break
        start = nxt
    return out


def f_aster(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    start = t0
    while start < now:
        rows = http(
            f"https://fapi.asterdex.com/fapi/v1/fundingRate?symbol={coin}USDT"
            f"&startTime={start}&endTime={now}&limit=1000"
        )
        if not rows:
            break
        out += [(r["fundingTime"], float(r["fundingRate"]), None) for r in rows]
        nxt = rows[-1]["fundingTime"] + 1
        if nxt <= start or len(rows) < 1000:
            break
        start = nxt
    return out


def f_dydx(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    before = datetime.fromtimestamp(now / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    while True:
        d = http(
            f"https://indexer.dydx.trade/v4/historicalFunding/{coin}-USD"
            f"?limit=100&effectiveBeforeOrAt={before}"
        )
        rows = d["historicalFunding"]
        if not rows:
            break
        stamps = [
            int(datetime.fromisoformat(r["effectiveAt"].replace("Z", "+00:00")).timestamp() * 1000)
            for r in rows
        ]
        out += [(s, float(r["rate"]), 1.0) for s, r in zip(stamps, rows, strict=True)]
        oldest = min(stamps)
        if oldest <= t0 or len(rows) < 100:
            break
        before = (
            datetime.fromtimestamp((oldest - 1) / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
    return [p for p in out if p[0] >= t0]


def f_backpack(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    offset = 0
    while True:
        rows = http(
            f"https://api.backpack.exchange/api/v1/fundingRates?symbol={coin}_USDC_PERP"
            f"&limit=1000&offset={offset}"
        )
        if not rows:
            break
        stamps = [
            int(
                datetime.fromisoformat(r["intervalEndTimestamp"]).replace(tzinfo=UTC).timestamp()
                * 1000
            )
            for r in rows
        ]
        out += [(s, float(r["fundingRate"]), 1.0) for s, r in zip(stamps, rows, strict=True)]
        if min(stamps) <= t0 or len(rows) < 1000:
            break
        offset += 1000
    return [p for p in out if p[0] >= t0]


def f_extended(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    end = now
    while True:
        d = http(
            f"https://api.starknet.extended.exchange/api/v1/info/{coin}-USD/funding"
            f"?startTime={t0}&endTime={end}&limit=1000"
        )
        rows = d["data"]
        if not rows:
            break
        out += [(r["T"], float(r["f"]), 1.0) for r in rows]
        oldest = min(r["T"] for r in rows)
        if len(rows) < 1000 or oldest <= t0:
            break
        end = oldest - 1
    return out


_EDGEX_IDS: dict[str, str] = {}


def _edgex_ids() -> dict[str, str]:
    if not _EDGEX_IDS:
        meta = http("https://edgex-prod-v2.edgex.exchange/api/v2/public/meta/getMetaData")
        for c in meta["data"]["contractList"]:
            if c["contractName"].endswith("USDC"):
                _EDGEX_IDS[c["contractName"][:-4]] = c["contractId"]
    return _EDGEX_IDS


def f_edgex(coin: str, t0: int, now: int) -> list[Print]:
    cid = _edgex_ids().get(coin)
    if cid is None:
        return []
    out: list[Print] = []
    end = now
    while True:
        d = http(
            "https://edgex-prod-v2.edgex.exchange/api/v2/public/funding/getFundingRatePage"
            f"?contractId={cid}&size=100&filterSettlementFundingRate=true"
            f"&filterBeginTimeInclusive={t0}&filterEndTimeExclusive={end}"
        )
        rows = d["data"]["dataList"]
        if not rows:
            break
        out += [(int(r["fundingTime"]), float(r["fundingRate"]), None) for r in rows]
        oldest = min(int(r["fundingTime"]) for r in rows)
        if len(rows) < 100 or oldest <= t0:
            break
        end = oldest
    return out


def f_apex(coin: str, t0: int, now: int) -> list[Print]:
    # The endpoint serves at most 360 prints (15 days) per query window, so the window end is
    # moved back to the oldest print received until the start of the period is reached.
    out: list[Print] = []
    end = now
    while True:
        window: list[Print] = []
        page = 0
        while True:
            d = http(
                f"https://omni.apex.exchange/api/v3/history-funding?symbol={coin}-USDT&limit=100"
                f"&page={page}&beginTimeInclusive={t0}&endTimeExclusive={end}"
            )
            rows = d["data"].get("historyFunds") or []
            window += [(int(r["fundingTime"]), float(r["rate"]), 1.0) for r in rows]
            if len(rows) < 100:
                break
            page += 1
        if not window:
            break
        out += window
        oldest = min(p[0] for p in window)
        if oldest <= t0 or len(window) < 360:
            break
        end = oldest
    return out


def f_pacifica(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    cursor = None
    while True:
        url = f"https://api.pacifica.fi/api/v1/funding_rate/history?symbol={coin}&limit=4000"
        if cursor:
            url += f"&cursor={cursor}"
        d = http(url)
        rows = d.get("data") or []
        out += [(int(r["created_at"]), float(r["funding_rate"]), 1.0) for r in rows]
        cursor = d.get("next_cursor")
        if (
            not rows
            or not d.get("has_more")
            or not cursor
            or min(int(r["created_at"]) for r in rows) <= t0
        ):
            break
    return [p for p in out if p[0] >= t0]


def f_orderly(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    page = 1
    while True:
        d = http(
            f"https://api-evm.orderly.org/v1/public/funding_rate_history?symbol=PERP_{coin}_USDC"
            f"&start_t={t0}&end_t={now}&page={page}&size=500"
        )
        rows = d["data"]["rows"]
        if not rows:
            break
        out += [(int(r["funding_rate_timestamp"]), float(r["funding_rate"]), 8.0) for r in rows]
        if len(rows) < 500:
            break
        page += 1
    return out


def f_aevo(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    end = now
    while True:
        d = http(
            f"https://api.aevo.xyz/funding-history?instrument_name={coin}-PERP"
            f"&start_time={t0 * 10**6}&end_time={end * 10**6}&limit=50"
        )
        rows = d["funding_history"]
        if not rows:
            break
        out += [(int(r[1]) // 10**6, float(r[2]), 1.0) for r in rows]
        oldest = min(int(r[1]) // 10**6 for r in rows)
        if len(rows) < 50 or oldest <= t0:
            break
        end = oldest
    return out


def f_grvt(coin: str, t0: int, now: int) -> list[Print]:
    out: list[Print] = []
    cursor = None
    while True:
        body = {
            "instrument": f"{coin}_USDT_Perp",
            "start_time": str(t0 * 10**6),
            "end_time": str(now * 10**6),
            "limit": 1000,
        }
        if cursor:
            body["cursor"] = cursor
        d = http("https://market-data.grvt.io/full/v1/funding", body)
        rows = d.get("result", [])
        # GRVT publishes the rate in percent: "0.01" = 0.01 % per 8 h = 0.0001 as a fraction.
        out += [
            (int(r["funding_time"]) // 10**6, float(r["funding_rate"]) / 100.0, 8.0) for r in rows
        ]
        cursor = d.get("next")
        if not cursor or not rows:
            break
    return out


FETCHERS = {
    "hyperliquid": f_hyperliquid,
    "aster": f_aster,
    "dydx": f_dydx,
    "backpack": f_backpack,
    "extended": f_extended,
    "edgex": f_edgex,
    "apex": f_apex,
    "orderly": f_orderly,
    "aevo": f_aevo,
    "grvt": f_grvt,
    "pacifica": f_pacifica,
}


# --- aggregation --------------------------------------------------------------------------


def with_intervals(prints: list[Print]) -> list[tuple[int, float, float]]:
    """Fill missing interval hours from the gap to the previous print (oldest first)."""
    ps = sorted({p[0]: p for p in prints}.values())
    out: list[tuple[int, float, float]] = []
    for i, (ts, rate, h) in enumerate(ps):
        if h is None:
            gap = (
                (ts - ps[i - 1][0]) / HOUR_MS
                if i
                else ((ps[1][0] - ts) / HOUR_MS if len(ps) > 1 else 8.0)
            )
            h = float(min(max(round(gap), 1), 8))
        out.append((ts, rate, h))
    return out


def apr(series: list[tuple[int, float, float]], t0: int) -> float | None:
    sel = [(r, h) for ts, r, h in series if ts >= t0]
    if not sel:
        return None
    return sum(r for r, _ in sel) / sum(h for _, h in sel) * 8760 * 100


def neg_share(series: list[tuple[int, float, float]], t0: int) -> float | None:
    sel = [r for ts, r, _ in series if ts >= t0]
    return 100 * sum(1 for r in sel if r < 0) / len(sel) if sel else None


# Aevo returns 50 prints per call (about 1.2 s each), so a year takes ~3.5 min per coin; its
# window is cut to 120 days. Its 365-day column is then reported over the days it covers.
MAX_DAYS = {"aevo": 120}


def run_venue(venue: str, coins: tuple[str, ...], now: int, cache: Path | None) -> list[dict]:
    rows = []
    t0 = now - MAX_DAYS.get(venue, max(WINDOWS)) * DAY_MS
    for coin in coins:
        key = cache / f"{venue}_{coin}.json" if cache else None
        prints = None
        if key and key.exists():
            prints = [tuple(p) for p in json.loads(key.read_text())]
        if prints is None:
            try:
                prints = FETCHERS[venue](coin, t0, now)
            except Exception as exc:  # noqa: BLE001
                rows.append({"venue": venue, "coin": coin, "error": str(exc)[:80]})
                continue
            if key:
                key.write_text(json.dumps(prints))
        if not prints:
            rows.append({"venue": venue, "coin": coin, "error": "no prints"})
            continue
        series = with_intervals(prints)
        row = {
            "venue": venue,
            "coin": coin,
            "n": len(series),
            "days_covered": round((now - series[0][0]) / DAY_MS, 1),
        }
        for w in WINDOWS:
            v = apr(series, now - w * DAY_MS)
            row[f"apr{w}"] = None if v is None else round(v, 2)
        n90 = neg_share(series, now - 90 * DAY_MS)
        row["neg90_pct"] = None if n90 is None else round(n90, 1)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--coins", default=",".join(DEFAULT_COINS))
    ap.add_argument("--venues", default=",".join(FETCHERS))
    args = ap.parse_args()
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    coins = tuple(args.coins.split(","))
    venues = args.venues.split(",")
    now = int(time.time() * 1000)
    with ThreadPoolExecutor(max_workers=len(venues)) as pool:
        results = list(pool.map(lambda v: run_venue(v, coins, now, args.cache), venues))
    print("venue,coin,n_prints,days_covered,apr30_pct,apr90_pct,apr365_pct,neg90_pct,error")
    for rows in results:
        for r in rows:
            print(
                ",".join(
                    str(x)
                    for x in (
                        r["venue"],
                        r["coin"],
                        r.get("n", ""),
                        r.get("days_covered", ""),
                        r.get("apr30", ""),
                        r.get("apr90", ""),
                        r.get("apr365", ""),
                        r.get("neg90_pct", ""),
                        r.get("error", ""),
                    )
                )
            )
    print(f"# run at {datetime.fromtimestamp(now / 1000, UTC).isoformat()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
