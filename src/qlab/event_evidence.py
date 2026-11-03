"""What followed each kind of venue event, measured from q-lab's own data
(docs/SCOUT.md).

Owner, 2026-11-03, on the scout producing no cards: «нужно разобраться почему он не
приносит - а не брать и душить его», then «делай все 4». The scout used to see one day's
events with no history, so it could only name an event, never say what such an event had
led to before. This module is the history: code, no agent, no tokens. For each kind of
event the venue watcher reports (`qlab.venue_watch`) it takes the events of the past in the
raw caches and says what followed:

- a listing: the first 1, 7 and 30 days -- the move, the funding, the volume and the
  swings against the later weeks;
- a delisting: the days between the removal's announcement and the last trading day (the
  announcement date is not in q-lab's data: `delist_lead_days` is an assumption, written
  into the file, and such a pattern is never `tradeable`);
- a funding extreme (the five highest and five lowest daily funding of a venue-day): how
  long the funding stays, the price move, and what a position would have earned after
  costs -- one that only collects the funding (hedged by another leg) and a plain
  directional one;
- a jump in daily volume (Binance, Hyperliquid) or in open interest (Hyperliquid, from the
  recorder's snapshots): the move that follows and the swings after.

**No looking ahead.** An event of the day D is found from data up to the close of D; what
follows is measured only from data after D (entry at the close of D, so a funding payment
stamped at that very close is not counted). The file is written for a night `day` and
reads no data after the last closed day.

**Costs** are the venue's own row of `night/costs.yaml` (`qlab.costs`): a taker order plus
slippage per leg; a position that holds a hedge leg pays two legs on the way in and two on
the way out.

**Honest limits**, written into the file as well: a cache holds the contracts that still
exist or were delisted late (survivorship), listings cluster in time (so standard errors
are clustered by month of the event), and a cell tested among dozens is a cell: the file
counts the cells and `night/scout.yaml` asks for a large t-value before it wakes anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EVIDENCE_DIR = Path("data/events/evidence")
SCHEMA = 1
MIN_MONTHS = 5  # fewer month-clusters than this: no t-value is given

DEFAULTS: dict = {
    "top": 5,                    # extremes per measure and day, as the venue watcher lists
    "min_universe": 20,          # fewer live contracts that day: no ranking
    "episode_days": 7,           # an event repeating within this many days is the same episode
    "horizons_days": [1, 7],
    "listing_horizons_days": [1, 7, 30],
    "delist_lead_days": 3,
    "liquid_floor_usd": {"binance": 10_000_000, "hyperliquid": 1_000_000},
    "venues": ["binance", "hyperliquid"],
    "listing_venues": ["binance", "hyperliquid", "hyperliquid-xyz"],
    "oi_venues": ["hyperliquid"],
}


# --------------------------------------------------------------------------
# The caches


@dataclass
class Panel:
    """One venue's daily history, wide: a row per UTC day D, a column per contract."""

    venue: str
    price: pd.DataFrame      # the close of day D
    notional: pd.DataFrame   # price x volume of day D, USD
    funding: pd.DataFrame    # funding paid in (D 00:00, D+1 00:00], a fraction
    last_trading_day: dict[str, pd.Timestamp] = field(default_factory=dict)  # delisted only
    cache_start: pd.Timestamp = pd.Timestamp("1970-01-01")  # the first day the cache could hold


def _split_name(path: Path) -> tuple[str, str, str]:
    symbol, start, end = path.stem.split("__")
    return symbol, start, end


def widest_files(folder: Path) -> dict[str, Path]:
    """One file per contract: the one that reaches the latest day, the earliest start among
    those (the caches keep several windows of the same contract)."""
    best: dict[str, tuple[str, str, Path]] = {}
    for path in sorted(folder.glob("*.parquet")):
        symbol, start, end = _split_name(path)
        held = best.get(symbol)
        if held is None or (end, held[1]) > (held[0], start):
            best[symbol] = (end, start, path)
    return {s: v[2] for s, v in best.items()}


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index


def load_panel(root: Path, venue: str, last_day: date, *, folder: str = "1d") -> Panel:
    """The daily history of a venue from `<root>/data/raw/<venue>/<folder>`, cut at
    `last_day` (the last closed day). Price is the close of the bar stamped D 00:00 UTC."""
    files = widest_files(root / "data" / "raw" / venue / folder)
    if not files:
        raise FileNotFoundError(f"no cached daily history for {venue} under {root}/data/raw")
    prices, notionals, fundings, last_trading = {}, {}, {}, {}
    starts = []
    cut = pd.Timestamp(last_day)
    for symbol, path in files.items():
        if symbol.endswith("-SPOT"):
            continue  # the spot leg's history (the hedge leg's data), not a perpetual
        frame = pd.read_parquet(path)
        starts.append(pd.Timestamp(_split_name(path)[1]))
        bars = frame.reindex(columns=["price", "volume"]).dropna(subset=["price"])
        bars = bars[bars["price"] > 0]
        if bars.empty:
            continue
        bars.index = _naive(bars.index).normalize()
        bars = bars[~bars.index.duplicated(keep="last")]
        prices[symbol] = bars["price"]
        notionals[symbol] = bars["price"] * bars["volume"].fillna(0.0)
        paid = frame["funding"].dropna()
        if len(paid):
            stamps = _naive(paid.index).floor("h")
            day = (stamps - pd.Timedelta(1, "ns")).floor("D")
            fundings[symbol] = pd.Series(paid.to_numpy(), index=day).groupby(level=0).sum()
        meta = path.with_suffix(".json")
        if meta.is_file() and json.loads(meta.read_text()).get("is_delisted"):
            # a removed contract keeps a flat settlement price for days: its last trading
            # day is the last with a traded value
            traded = notionals[symbol][notionals[symbol] > 0]
            last_trading[symbol] = traded.index.max() if len(traded) else bars.index.max()
    index = pd.date_range(min(s.index.min() for s in prices.values()), cut, freq="D")
    price = pd.DataFrame(prices).reindex(index)
    notional = pd.DataFrame(notionals).reindex(index)
    funding = pd.DataFrame(fundings).reindex(index).reindex(columns=price.columns)
    # a day with prices but no funding row at all stays NaN: it is no zero.
    return Panel(venue, price, notional, funding,
                 {s: d for s, d in last_trading.items() if d < cut - pd.Timedelta(days=1)},
                 min(starts))


# --------------------------------------------------------------------------
# Statistics


def summarize(values: pd.Series, dates: pd.Series) -> dict:
    """n, mean, median and hit rate of a net result (a fraction), the t-value of the mean
    with standard errors clustered by month of the event, and the mean in each half of the
    sample by date. Percent, rounded."""
    keep = values.notna().to_numpy()
    x = pd.Series(values.to_numpy()[keep])
    d = pd.Series(pd.to_datetime(dates.to_numpy()[keep]))
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    months = d.dt.to_period("M")
    clustered = (x - mean).groupby(months.to_numpy()).sum()
    se = math.sqrt(float((clustered ** 2).sum())) / n
    # With fewer than MIN_MONTHS clusters the clustered error means nothing.
    t = mean / se if se > 0 and months.nunique() >= MIN_MONTHS else float("nan")
    middle = d.sort_values().iloc[n // 2]
    first, second = x[d < middle], x[d >= middle]

    def pct(v: float) -> float | None:
        return None if v != v else round(100 * float(v), 3)

    return {"n": n, "months": int(months.nunique()), "mean_pct": pct(mean),
            "median_pct": pct(x.median()), "hit_rate": round(float((x > 0).mean()), 3),
            "t": None if t != t else round(t, 2),
            "first_half_pct": pct(first.mean()) if len(first) else None,
            "second_half_pct": pct(second.mean()) if len(second) else None}


def _round(value: float, digits: int = 3) -> float | None:
    return None if value is None or value != value else round(float(value), digits)


def _median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return _round(float(np.median(values))) if len(values) else None


# --------------------------------------------------------------------------
# Event finders


def top_events(score: pd.DataFrame, eligible: pd.DataFrame, *, top: int, largest: bool,
               min_universe: int, episode_days: int,
               universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """A boolean frame: the `top` largest (or smallest) scores of each day among the eligible
    contracts, on days when the `universe` (default: the eligible ones) has at least
    `min_universe` contracts, without a repeat of the same contract's event in the
    `episode_days` before."""
    ok = score.where(eligible)
    enough = (eligible if universe is None else universe).sum(axis=1) >= min_universe
    rank = ok.rank(axis=1, ascending=not largest, method="first")
    raw = rank <= top
    raw.loc[~enough.to_numpy()] = False
    earlier = raw.shift(1, fill_value=False).astype(int).rolling(episode_days, min_periods=1).max()
    return raw & (earlier == 0)


def _points(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return np.nonzero(frame.to_numpy())


def _take(frame: pd.DataFrame, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return frame.to_numpy()[rows, cols]


def _forward(panel: Panel, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The move of the close D -> D+k and the funding paid in the k days after D."""
    ret = panel.price.shift(-k) / panel.price - 1
    fund = panel.funding.rolling(k, min_periods=k).sum().shift(-k)
    return ret, fund


def _measure(name: str, position: str, horizon: int, net: np.ndarray, dates: pd.Series,
             **extra) -> dict:
    return {"id": f"{name}@{horizon}d", "position": position, "horizon_days": horizon,
            **summarize(pd.Series(net), dates), **extra}


# --------------------------------------------------------------------------
# Patterns


def _legs(cost_bps: dict[str, float], venue: str) -> float:
    """One leg, in and out: a fraction."""
    return 2 * cost_bps[venue.replace("-xyz", "")] / 1e4


def funding_extremes(panel: Panel, cfg: dict, cost_bps: dict[str, float]) -> list[dict]:
    out = []
    one_leg = _legs(cost_bps, panel.venue)
    floor = cfg["liquid_floor_usd"].get(panel.venue.replace("-xyz", ""), 1e6)
    for universe in ("all", "liquid"):
        eligible = panel.price.notna() & panel.funding.notna()
        if universe == "liquid":
            eligible &= panel.notional >= floor
        for side, label in ((1, "high"), (-1, "low")):
            score = panel.funding * side
            found = top_events(score, eligible & (score > 0), top=cfg["top"], largest=True,
                               min_universe=cfg["min_universe"], episode_days=cfg["episode_days"],
                               universe=eligible)
            rows, cols = _points(found)
            if not len(rows):
                continue
            dates = pd.Series(panel.price.index[rows])
            today = _take(panel.funding, rows, cols)
            measures, profile = [], {}
            for k in cfg["horizons_days"]:
                ret, fund = _forward(panel, k)
                r, f = _take(ret, rows, cols), _take(fund, rows, cols)
                valid = np.isfinite(r) & np.isfinite(f)
                kept = np.where(valid, (f / (k * today)), np.nan)
                profile[f"{k}d"] = {
                    "funding_kept_median": _median(kept[valid]),
                    "same_sign_share": _round(float((kept[valid] > 0).mean())) if valid.any()
                    else None,
                    "price_move_median_pct": _round(100 * float(np.median(r[valid])))
                    if valid.any() else None,
                    "n": int(valid.sum())}
                carry = np.where(valid, side * f - 2 * one_leg, np.nan)
                direction = np.where(valid, side * (f - r) - one_leg, np.nan)
                measures += [
                    _measure("collect_funding_hedged", "perpetual against a hedge leg, funding "
                             "collected, price risk hedged away", k, carry, dates),
                    _measure("directional_perpetual", "the perpetual alone: short after a high "
                             "extreme, long after a low one", k, direction, dates)]
            out.append({
                "key": f"funding-extreme/{panel.venue}/{label}/{universe}",
                "kind": "funding-extreme", "venue": panel.venue,
                "definition": f"the {cfg['top']} {label}est daily funding of the venue-day"
                              + (f" among contracts trading at least ${floor:,.0f} a day"
                                 if universe == "liquid" else "")
                              + f", a repeat within {cfg['episode_days']} days counted once",
                "n_events": len(rows), "first_event": str(dates.min().date()),
                "last_event": str(dates.max().date()), "profile": profile,
                "measures": measures, "tradeable": True})
    return out


def volume_jumps(panel: Panel, cfg: dict, cost_bps: dict[str, float]) -> list[dict]:
    one_leg = _legs(cost_bps, panel.venue)
    floor = cfg["liquid_floor_usd"].get(panel.venue.replace("-xyz", ""), 1e6)
    before = panel.notional.shift(1)
    ratio = panel.notional / before
    day_move = panel.price / panel.price.shift(1) - 1
    out = []
    for universe in ("all", "liquid"):
        eligible = (before > (floor if universe == "liquid" else 0)) & day_move.notna()
        found = top_events(ratio, eligible & (ratio > 1), top=cfg["top"], largest=True,
                           min_universe=cfg["min_universe"], episode_days=cfg["episode_days"],
                           universe=eligible)
        out.append(_move_pattern(
            panel, found, day_move, cfg, one_leg, kind="volume-jump",
            key=f"volume-jump/{panel.venue}/{universe}",
            definition=f"the {cfg['top']} largest rises of the day's traded value against the "
                       "day before"
                       + (f", among contracts that traded at least ${floor:,.0f} the day before"
                          if universe == "liquid" else "")
                       + f", a repeat within {cfg['episode_days']} days counted once"))
    return [p for p in out if p]


def _move_pattern(panel: Panel, found: pd.DataFrame, day_move: pd.DataFrame, cfg: dict,
                  one_leg: float, *, kind: str, key: str, definition: str) -> dict | None:
    rows, cols = _points(found)
    if not len(rows):
        return None
    dates = pd.Series(panel.price.index[rows])
    side = np.sign(_take(day_move, rows, cols))
    absret = panel.price.pct_change().abs()
    after_swing = absret.rolling(7, min_periods=7).mean().shift(-7)
    before_swing = absret.rolling(7, min_periods=7).mean()
    swing = _take(after_swing, rows, cols) / _take(before_swing, rows, cols)
    measures, profile = [], {"swing_after_vs_before_median":
                             _median(swing), "n_swing": int(np.isfinite(swing).sum()),
                             "event_day_move_median_abs_pct":
                             _round(100 * float(np.median(np.abs(_take(day_move, rows, cols)))))}
    for k in cfg["horizons_days"]:
        ret, fund = _forward(panel, k)
        r, f = _take(ret, rows, cols), _take(fund, rows, cols)
        valid = np.isfinite(r) & np.isfinite(f) & (side != 0)
        gross = np.where(valid, side * (r - f), np.nan)
        measures += [
            _measure("follow_the_move", "the perpetual alone, in the direction of the event "
                     "day's move", k, gross - one_leg, dates),
            _measure("fade_the_move", "the perpetual alone, against the direction of the event "
                     "day's move", k, -gross - one_leg, dates)]
    return {"key": key, "kind": kind, "venue": panel.venue, "definition": definition,
            "n_events": len(rows), "first_event": str(dates.min().date()),
            "last_event": str(dates.max().date()), "profile": profile, "measures": measures,
            "tradeable": True}


def listings(panel: Panel, cfg: dict, cost_bps: dict[str, float]) -> list[dict]:
    one_leg = _legs(cost_bps, panel.venue)
    first = panel.price.apply(pd.Series.first_valid_index).dropna()
    # A contract with no funding history at all cannot be measured (its cache lacks it).
    first = first[[s for s in first.index if panel.funding[s].notna().any()]]
    born = first[first > panel.cache_start + pd.Timedelta(days=1)]
    if born.empty:
        return []
    btc = "BTC" if "BTC" in panel.price.columns else None
    horizons = cfg["listing_horizons_days"]
    dates = pd.Series(born.to_numpy())
    rows = np.array([panel.price.index.get_loc(d) for d in born])
    cols = np.array([panel.price.columns.get_loc(s) for s in born.index])
    measures = []
    for k in horizons:
        ret, fund = _forward(panel, k)
        r, f = _take(ret, rows, cols), _take(fund, rows, cols)
        base = ret[btc].to_numpy()[rows] if btc else np.zeros(len(rows))
        valid = np.isfinite(r) & np.isfinite(f)
        measures += [
            _measure("short_perpetual", "short the perpetual at the close of its first day",
                     k, np.where(valid, f - r - one_leg, np.nan), dates),
            _measure("long_perpetual", "long the perpetual at the close of its first day",
                     k, np.where(valid, r - f - one_leg, np.nan), dates)]
        both = valid & np.isfinite(base)
        measures.append(_measure(
            "short_against_btc", "short the new perpetual, long BTC's perpetual (two legs)", k,
            np.where(both, f - (r - base) - 2 * one_leg, np.nan), dates))
    profile = _listing_profile(panel, rows, cols)
    return [{"key": f"listing/{panel.venue}", "kind": "listing", "venue": panel.venue,
             "definition": "a contract's first day with a price in the cache, the cache not "
                           f"starting within a day of it (cache starts {panel.cache_start.date()})",
             "n_events": len(rows), "first_event": str(dates.min().date()),
             "last_event": str(dates.max().date()), "profile": profile, "measures": measures,
             "tradeable": True}]


def _listing_profile(panel: Panel, rows: np.ndarray, cols: np.ndarray) -> dict:
    price = panel.price.to_numpy()
    notional = panel.notional.to_numpy()
    funding = panel.funding.to_numpy()
    swing, volume, fund_ratio, drift = [], [], [], []
    for r, c in zip(rows, cols, strict=True):
        if r + 90 >= len(price):
            continue
        p = price[r:r + 91, c]
        if not np.isfinite(p).all():
            continue
        returns = np.diff(np.log(p))  # returns[i] is the day r+i -> r+i+1
        early, late = returns[0:7], returns[30:90]
        if early.std() > 0 and late.std() > 0:
            swing.append(early.std() / late.std())
        n_early = notional[r + 1:r + 8, c].mean()
        n_late = notional[r + 31:r + 91, c].mean()
        if n_late > 0:
            volume.append(n_early / n_late)
        f_early, f_late = funding[r + 1:r + 8, c], funding[r + 31:r + 91, c]
        if np.isfinite(f_early).all() and np.isfinite(f_late).all():
            fund_ratio.append((f_early.mean(), f_late.mean()))
        drift.append(p[30] / p[0] - 1)
    return {
        "n_with_90_days": len(swing),
        "daily_swing_first_7d_vs_days_31_90_median": _median(np.array(swing)),
        "traded_value_first_7d_vs_days_31_90_median": _median(np.array(volume)),
        "move_first_30d_median_pct": _round(100 * float(np.median(drift))) if drift else None,
        "daily_funding_first_7d_median_pct": _round(
            100 * float(np.median([a for a, _ in fund_ratio]))) if fund_ratio else None,
        "daily_funding_days_31_90_median_pct": _round(
            100 * float(np.median([b for _, b in fund_ratio]))) if fund_ratio else None}


def delistings(panel: Panel, cfg: dict, cost_bps: dict[str, float]) -> list[dict]:
    """Contracts removed from the venue, measured from `delist_lead_days` before their last
    trading day. The announcement date is not in the data: this is a stated assumption, so
    the pattern is described, never `tradeable`."""
    lead = cfg["delist_lead_days"]
    one_leg = _legs(cost_bps, panel.venue)
    rows, cols, ends = [], [], []
    for symbol, last in panel.last_trading_day.items():
        start = last - pd.Timedelta(days=lead)
        if start in panel.price.index and symbol in panel.price.columns \
                and np.isfinite(panel.price.at[start, symbol]):
            rows.append(panel.price.index.get_loc(start))
            cols.append(panel.price.columns.get_loc(symbol))
            ends.append(panel.price.index.get_loc(last))
    if not rows:
        return []
    rows, cols, ends = np.array(rows), np.array(cols), np.array(ends)
    price, fund = panel.price.to_numpy(), panel.funding.to_numpy()
    start_px, end_px = price[rows, cols], price[ends, cols]
    # funding stops with the contract: a day without a payment counts as none
    paid = np.array([np.nansum(fund[r + 1:e + 1, c])
                     for r, e, c in zip(rows, ends, cols, strict=True)])
    r = end_px / start_px - 1
    dates = pd.Series(panel.price.index[rows])
    measures = [
        _measure("short_perpetual_to_last_day", "short the perpetual `delist_lead_days` before "
                 "its last trading day, hold to it", lead, paid - r - one_leg, dates),
        _measure("long_perpetual_to_last_day", "long the perpetual `delist_lead_days` before "
                 "its last trading day, hold to it", lead, r - paid - one_leg, dates)]
    return [{"key": f"delisting/{panel.venue}", "kind": "delisting", "venue": panel.venue,
             "definition": f"contracts the venue removed ({len(rows)} with data), measured from "
                           f"{lead} days before the last trading day to that day",
             "n_events": len(rows), "first_event": str(dates.min().date()),
             "last_event": str(dates.max().date()),
             "profile": {"move_to_last_day_median_pct": _round(100 * float(np.median(r))),
                         "funding_paid_median_pct": _median(paid * 100)},
             "measures": measures, "tradeable": False,
             "assumption": f"the announcement is taken to come {lead} days before the last "
                           "trading day; its real date is not in q-lab's data, so what is "
                           "measured is known only in hindsight"}]


def open_interest_jumps(root: Path, panel: Panel, cfg: dict, cost_bps: dict[str, float],
                        last_day: date) -> list[dict]:
    """Jumps in open interest from the recorder's daily snapshots (Hyperliquid, since
    2026-10-01): a snapshot taken early on day S describes the close of S-1."""
    folder = root / "data" / "recorded" / panel.venue / "meta"
    oi: dict[pd.Timestamp, dict[str, float]] = {}
    for path in sorted(folder.glob("*.json")):
        day = pd.Timestamp(path.stem) - pd.Timedelta(days=1)
        if day > pd.Timestamp(last_day):
            continue
        universe = json.loads(path.read_text()).get("universe", [])
        oi[day] = {e["name"]: float(e["ctx"]["openInterest"]) * float(e["ctx"].get("markPx") or 0)
                   for e in universe if e.get("ctx") and not e.get("isDelisted")
                   and e["ctx"].get("openInterest") is not None}
    if len(oi) < 3:
        return []
    value = pd.DataFrame(oi).T.sort_index().reindex(columns=panel.price.columns)
    value = value.reindex(panel.price.index)
    ratio = value / value.shift(1)
    day_move = panel.price / panel.price.shift(1) - 1
    one_leg = _legs(cost_bps, panel.venue)
    floor = cfg["liquid_floor_usd"].get(panel.venue, 1e6)
    out = []
    for universe in ("all", "liquid"):
        eligible = (value.shift(1) > (floor if universe == "liquid" else 0)) & day_move.notna()
        found = top_events(ratio, eligible & (ratio > 1), top=cfg["top"], largest=True,
                           min_universe=cfg["min_universe"], episode_days=cfg["episode_days"],
                           universe=eligible)
        pattern = _move_pattern(
            panel, found, day_move, cfg, one_leg, kind="oi-jump",
            key=f"oi-jump/{panel.venue}/{universe}",
            definition=f"the {cfg['top']} largest rises of open interest (in USD at the mark) "
                       "against the day before"
                       + (f", among contracts with at least ${floor:,.0f} of it the day before"
                          if universe == "liquid" else "")
                       + f"; snapshots exist from {min(oi).date()}")
        if pattern:
            out.append(pattern)
    return out


# --------------------------------------------------------------------------
# The file


def load_costs(venues: list[str]) -> dict[str, float]:
    """One leg's taker fee plus slippage, in basis points, per venue (`night/costs.yaml`)."""
    from qlab.costs import venue_costs

    out = {}
    for venue in venues:
        row = venue_costs(venue.replace("-xyz", ""))
        if row is None:
            raise ValueError(f"night/costs.yaml has no row for {venue}")
        out[venue.replace("-xyz", "")] = row.taker_fee_bps + row.slippage_bps
    return out


def build(day: date, root: Path, cfg: dict | None = None, *,
          cost_bps: dict[str, float] | None = None, events: list[dict] | None = None) -> dict:
    """The evidence for the night `day`: every pattern, its sample and what followed. A venue
    whose cache is missing is listed under `skipped`, never guessed."""
    cfg = {**DEFAULTS, **(cfg or {})}
    last_day = day - timedelta(days=1)
    venues = sorted({*cfg["venues"], *cfg["listing_venues"], *cfg["oi_venues"]})
    costs = cost_bps or load_costs(venues)
    patterns: list[dict] = []
    skipped: list[str] = []
    panels: dict[str, Panel] = {}
    for venue in venues:
        try:
            panels[venue] = load_panel(root, venue, last_day)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(f"{venue}: {exc}")
    for venue, panel in panels.items():
        if venue in cfg["venues"]:
            patterns += funding_extremes(panel, cfg, costs) + volume_jumps(panel, cfg, costs)
        if venue in cfg["listing_venues"]:
            patterns += listings(panel, cfg, costs)
        if venue == "binance":
            patterns += delistings(panel, cfg, costs)
        if venue in cfg["oi_venues"]:
            patterns += open_interest_jumps(root, panel, cfg, costs, last_day)
    cells = sum(len(p["measures"]) for p in patterns)
    return {
        "schema": SCHEMA, "night": day.isoformat(), "last_data_day": last_day.isoformat(),
        "built_at": datetime.now().replace(microsecond=0).isoformat(),
        "reading": "Every number here is measured by code from q-lab's own caches. Entry is at "
                   "the close of the event's day; only later data enters a measurement. 'net' "
                   "means after the venue's taker fee and slippage (night/costs.yaml). A "
                   "t-value is of the mean, with errors clustered by month of the event.",
        "caveats": [
            "The caches hold contracts that still trade or were delisted late: survivorship "
            "flatters short-lived listings less than it should be trusted.",
            f"{cells} measures were computed in this file. At |t| = 2 about one in twenty "
            "of them looks real by chance alone; the scout is woken only at the larger t of "
            "night/scout.yaml, and a pattern is a lead for a mechanism, not a mechanism.",
            "Events cluster in time (a bull month lists many contracts at once): the t-values "
            "are clustered by month, the halves of the sample are shown, and a pattern that "
            "has one sign in only one half is not stable.",
        ],
        "costs_bps_per_leg": costs,
        "sources": {v: f"data/raw/{v}/1d" for v in panels},
        "patterns": patterns, "skipped": skipped,
        "today": today_events(patterns, events or []),
    }


_KIND_ALIASES = {"funding-extreme": "funding-extreme", "dayNtlVlm-jump": "volume-jump",
                 "openInterest-jump": "oi-jump", "listed": "listing", "delisted": "delisting"}


def today_events(patterns: list[dict], events: list[dict]) -> list[dict]:
    """For each of today's events, the pattern(s) whose history it belongs to."""
    keys = {p["key"]: p for p in patterns}
    out = []
    for e in events:
        kind = _KIND_ALIASES.get(e.get("kind"))
        if not kind:
            continue
        source = e["source"]
        match = [k for k in keys if k == f"{kind}/{source}" or k.startswith(f"{kind}/{source}/")]
        out.append({"source": source, "kind": e["kind"], "instrument": e["instrument"],
                    "detail": e.get("detail", ""), "history": match})
    return out


def path_for(day: date, root: Path = Path(".")) -> Path:
    return root / EVIDENCE_DIR / f"{day.isoformat()}.json"


def write(day: date, evidence: dict, root: Path) -> Path:
    path = path_for(day, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def ensure(day: date, root: Path, cfg: dict | None = None, *, data_root: Path | None = None,
           events: list[dict] | None = None) -> tuple[Path, dict]:
    """Today's evidence file: the one written already, else built and written. `data_root`
    is where the caches are (the checkout itself, unless a measurement points elsewhere)."""
    path = path_for(day, root)
    if path.is_file():
        return path, json.loads(path.read_text(encoding="utf-8"))
    evidence = build(day, data_root or root, cfg, events=events)
    return write(day, evidence, root), evidence


def format_for_log(evidence: dict) -> str:
    """A few lines for the console."""
    lines = [f"evidence for {evidence['night']}: {len(evidence['patterns'])} patterns"]
    for p in evidence["patterns"]:
        measured = [m for m in p["measures"] if m.get("mean_pct") is not None]
        best = max(measured, key=lambda m: m["t"] if m.get("t") is not None else -1e9,
                   default=None)
        if best and best.get("t") is None:
            best = max(measured, key=lambda m: m["mean_pct"])
        if best:
            t = "no t-value (too few months)" if best.get("t") is None else f"t={best['t']}"
            lines.append(f"  {p['key']:<46} n={p['n_events']:<6} best {best['id']} "
                         f"{best['mean_pct']:+.2f}% {t}")
    return "\n".join(lines)


__all__ = ["DEFAULTS", "EVIDENCE_DIR", "Panel", "build", "ensure", "format_for_log",
           "load_panel", "path_for", "summarize", "top_events", "write"]
