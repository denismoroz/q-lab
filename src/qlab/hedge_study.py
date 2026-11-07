"""Where Bv2 switches its hedge on and off, and whether other information picks those moments
better (docs/research/hedge-trigger-protocol-2026-11-04.md; owner, 2026-11-04: «вероятно у всех
этих стратегий - один корень - не верное определение точки включения/выключения хеджирования»).

The study, in four parts, all pure functions over cached data:

1. **Data** (`load_panel`): Binance hourly perpetual + spot from the raw cache, read only
   (`load_cached_history` never writes), assembled by the stand's own frame builder.
2. **Events** (`run_book`): the production Bv2 book is the live frab code driven by
   `qlab.strategies.live.bv2`; `_StudyDriver` is that driver with a recorder: after every bar it
   reads each coin's `in_pos` (the hedge is open), the book's own wish for the next bar, and the
   hedge's money so far by the book's own accounts. `events` turns the hedged state into on/off
   events; nothing about the rule is re-derived.
3. **Variants**: a *filter* vetoes a decision of the book (`gate_wish`, and inside the driver the
   same logic on the live book's wish); a *trigger* replaces the wish (`trigger_wish`, fed to the
   book as an external wish, the mechanism `hedge_by` and `fast_release_gate` use).
4. **Measures**: labels (`label_events`: the spot return after the fill, read only from bars after
   it), the hedge's money per bar on one unit of position (`hedge_money`), and a month-block
   bootstrap of the false-alarm share (`bootstrap_drop`).

Features (`features`) read only bars up to and including the decision bar; a percentile is the
value's place among the coin's own PAST daily samples (`past_percentile`).

Copied unchanged from branch worktree-agent-aabf750358df53596 (commit 8fa3bc3) for the regime-change
study (docs/research/regime-change-protocol-2026-11-07.md): the book replay with an external hedge
wish is reused as is.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.data.snapshot import _build_frames_from_histories
from qlab.data.sources.base import SPOT_COLUMN_SUFFIX, InstrumentHistory, load_cached_history
from qlab.harness.panel import MarketPanel
from qlab.strategies.live import bv2 as _bv2

SOURCE = "binance"
INTERVAL = "1h"
FUNDING_NATIVE = pd.Timedelta(hours=8)
HOURS_PER_YEAR = 24 * 365
HORIZONS_DAYS = (1, 3, 7, 14)
PRIMARY_HORIZON_DAYS = 7
MIN_PAST_DAYS = 90
_STEM = re.compile(r"^(?P<inst>.+)__(?P<start>\d{4}-\d{2}-\d{2})__(?P<end>\d{4}-\d{2}-\d{2})$")


# ── 1. data ──────────────────────────────────────────────────────────────


def cached_ranges(cache_dir: Path, instrument: str) -> list[tuple[str, str]]:
    """The (start, end) date pairs the raw cache holds for one instrument."""
    folder = Path(cache_dir) / SOURCE / INTERVAL
    out = []
    for path in folder.glob(f"{instrument}__*.parquet"):
        match = _STEM.match(path.stem)
        if match and match["inst"] == instrument:
            out.append((match["start"], match["end"]))
    return sorted(out)


def read_history(cache_dir: Path, instrument: str, end: pd.Timestamp) -> InstrumentHistory | None:
    """The cached entry with the earliest start among those reaching `end`; None if none does."""
    ranges = [r for r in cached_ranges(cache_dir, instrument)
              if pd.Timestamp(r[1], tz="UTC") >= end.normalize()]
    if not ranges:
        return None
    start, stop = min(ranges, key=lambda r: (r[0], r[1]))
    return load_cached_history(Path(cache_dir), SOURCE, instrument, INTERVAL, start, stop)


@dataclass(frozen=True)
class StudyPanel:
    panel: MarketPanel  # spot legs marked at the perp's price, as specs/bv2-binance.yaml
    spot_own: pd.DataFrame  # the spot markets' own closes (labels, basis), per coin


def load_panel(coins: Sequence[str], start, end, cache_dir: Path) -> StudyPanel:
    """Both legs of every coin from the raw cache, on an hourly UTC index [start, end]."""
    start_ts, end_ts = _utc(start), _utc(end)
    histories: dict[str, InstrumentHistory] = {}
    for coin in coins:
        for inst in (coin, f"{coin}{SPOT_COLUMN_SUFFIX}"):
            hist = read_history(cache_dir, inst, end_ts)
            if hist is None:
                raise ValueError(f"{inst}: no cached hourly history reaching {end_ts.date()}")
            histories[inst] = hist
    index = pd.date_range(start_ts, end_ts, freq="1h", tz="UTC")
    prices, funding, tradeable, volume = _build_frames_from_histories(
        histories, index, FUNDING_NATIVE, spot_prices="perp")
    own, _, _, _ = _build_frames_from_histories(histories, index, FUNDING_NATIVE,
                                                spot_prices="own")
    high = pd.DataFrame({c: (histories[c].high.reindex(index) if histories[c].high is not None
                             else pd.Series(np.nan, index=index)) for c in prices.columns},
                        index=index)[list(prices.columns)].where(prices.notna())
    spots = [c for c in prices.columns if c.endswith(SPOT_COLUMN_SUFFIX)]
    panel = MarketPanel(snapshot_id="hedge-study", prices=prices, funding=funding,
                        tradeable=tradeable, volume=volume, high=high,
                        meta={"venue": SOURCE, "interval": INTERVAL, "universe_complete": False,
                              "no_funding_instruments": spots})
    spot_own = pd.DataFrame({c: own[f"{c}{SPOT_COLUMN_SUFFIX}"] for c in coins}, index=index)
    return StudyPanel(panel=panel, spot_own=spot_own)


# ── 2. the book, recorded ────────────────────────────────────────────────


@dataclass
class BookRecord:
    """What the book did, per bar and coin (columns = coins)."""

    run: _bv2.LiveBv2Run
    stepped: pd.DataFrame  # the coin's book was stepped on the bar
    in_pos: pd.DataFrame  # the hedge is open after the bar
    own: pd.DataFrame  # the book's own wish for the next bar (before any gate)
    hedge_acc: pd.DataFrame  # hedge money so far, USD, by the book's accounts
    liquidated: pd.DataFrame  # a liquidation closed the hedge on the bar
    counts: dict = field(default_factory=dict)
    # the hedge's money so far by part, USD: "price" (the short's realised P&L), "funding",
    # "fees" (perp fees, margin top-up fees, liquidation penalties); price + funding - fees is
    # `hedge_acc`. "short_usd": the short's notional after the bar (0 when unhedged).
    # "carry_usd": the carry sleeve's notional (0 when off); "liq_dist": the liquidation price
    # over the bar's price, minus 1 (NaN without a short); "carry_funding", "carry_fees": the
    # sleeve's running totals, USD.
    parts: dict = field(default_factory=dict)


class _StudyDriver(_bv2._Driver):
    """Bv2's driver plus a recorder, an optional gate on the book's own wish, and optional
    external wishes (the `hedge_by` mechanism). With neither, the book is the production one."""

    def __init__(self, panel, p, rule, gates=None, wishes=None, shares=None) -> None:
        self._wishes = wishes
        self._gates = gates or {}
        self._shares = shares or {}
        super().__init__(panel, p, None, rule, None, None)
        n, coins = self.n, self.coins
        self._col = {c: k for k, c in enumerate(coins)}
        self.rec_stepped = np.zeros((n, len(coins)), dtype=bool)
        self.rec_in_pos = np.zeros((n, len(coins)), dtype=bool)
        self.rec_own = np.zeros((n, len(coins)), dtype=bool)
        self.rec_acc = np.full((n, len(coins)), np.nan)
        self.rec_parts = {k: np.full((n, len(coins)), np.nan)
                          for k in ("price", "funding", "fees", "short_usd", "carry_usd",
                                    "liq_dist", "carry_funding", "carry_fees")}
        self.rec_liq = np.zeros((n, len(coins)), dtype=bool)
        self._penalty = dict.fromkeys(coins, 0.0)
        self._gate_state = {c: {"own_prev": False, "suppress": False, "hold": False}
                            for c in coins}

    def _external(self, coin: str):
        return None if self._wishes is None else self._wishes[coin]

    def _count(self, events, i, coin=None) -> None:
        for e in events:
            if e["kind"] == "liquidation":
                who = e.get("coin", coin)
                self._penalty[who] += float(e["fee"])
                self.rec_liq[i, self._col[who]] = True
        super()._count(events, i, coin)

    def _advance(self, slot, i: int) -> None:
        super()._advance(slot, i)
        if not slot.stepped:
            return
        coin, book = slot.feed.coin, slot.book
        k = self._col[coin]
        own = bool(book.hedge_prev)
        if coin in self._gates:
            if slot.i0 == i:
                # a book that opens here carries the wish of its warm-up: a start, not a decision
                self._gate_state[coin] = {"own_prev": bool(book.in_pos), "suppress": False,
                                          "hold": False}
            book.hedge_prev = self._gate(coin, own, bool(book.in_pos), i)
        if coin in self._shares and not np.isnan(self._shares[coin][i]):
            # the share of the spot the hedge covers on the next bar (frab's partial-hedge
            # mechanism resizes the short to it; needs partial_hedge_share < 1 in the params)
            book.hedge_share_prev = float(self._shares[coin][i])
        self.rec_stepped[i, k] = True
        self.rec_in_pos[i, k] = bool(book.in_pos)
        self.rec_own[i, k] = own
        fees = book.perp_fees + book.margin_fees + self._penalty[coin]
        self.rec_acc[i, k] = book.short_realized + book.funding_total - fees
        self.rec_parts["price"][i, k] = book.short_realized
        self.rec_parts["funding"][i, k] = book.funding_total
        self.rec_parts["fees"][i, k] = fees
        price = float(slot.feed.px[i])
        self.rec_parts["short_usd"][i, k] = book.short_size * price if book.in_pos else 0.0
        if book.carry_on:
            units = book.carry_units if slot.params.margin_enabled else book.carry_target / price
            self.rec_parts["carry_usd"][i, k] = units * price
        else:
            self.rec_parts["carry_usd"][i, k] = 0.0
        liq = book.liquidation_price()
        self.rec_parts["liq_dist"][i, k] = liq / price - 1.0 if liq is not None else np.nan
        self.rec_parts["carry_funding"][i, k] = book.carry_funding
        self.rec_parts["carry_fees"][i, k] = book.carry_fees

    def _gate(self, coin: str, own: bool, in_pos: bool, i: int) -> bool:
        """The wish handed to the book for the next bar under a veto filter (`gate_wish`)."""
        veto_on, veto_off = self._gates[coin]
        st = self._gate_state[coin]
        if own and not st["own_prev"] and not in_pos and veto_on is not None and veto_on[i]:
            st["suppress"] = True
        if not own:
            st["suppress"] = False
        if (not own and st["own_prev"] and in_pos and veto_off is not None and veto_off[i]):
            st["hold"] = True
        if own or not in_pos:
            st["hold"] = st["hold"] and not own and in_pos
        st["own_prev"] = own
        return (own and not st["suppress"]) or st["hold"]


def run_book(panel: MarketPanel, params: Mapping[str, object],
             gates: Mapping[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
             wishes: Mapping[str, np.ndarray] | None = None,
             shares: Mapping[str, np.ndarray] | None = None) -> BookRecord:
    """Run the live Bv2 book (frab's code) on `panel` and record it bar by bar.

    `gates[coin] = (veto_on, veto_off)`: bool arrays on the panel's index, the filter's condition
    at each bar; `wishes[coin]`: an external wish per bar (1.0/0.0/NaN), as `hedge_by`."""
    params = dict(params)
    universe = params.pop("universe", None)
    rule = None
    if universe is not None:
        from qlab.strategies.live import _bv2_mechanisms as mech

        rule = mech.parse_universe(universe)
        params["coins"] = rule.coins(panel)
    _bv2._require_frab_support(params)
    p = _bv2.B2Params.from_dict(params)
    if shares and not p.partial_hedge_share < 1.0:
        raise ValueError("shares need partial_hedge_share < 1: frab resizes a hedge only then")
    driver = _StudyDriver(panel, p, rule, gates=gates, wishes=wishes, shares=shares)
    run = driver.go()
    index, coins = panel.prices.index, list(p.coins)

    def frame(a):
        return pd.DataFrame(a, index=index, columns=coins)

    return BookRecord(run=run, stepped=frame(driver.rec_stepped), in_pos=frame(driver.rec_in_pos),
                      own=frame(driver.rec_own), hedge_acc=frame(driver.rec_acc),
                      liquidated=frame(driver.rec_liq), counts=dict(run.counts),
                      parts={k: frame(v) for k, v in driver.rec_parts.items()})


def events(record: BookRecord, coin: str) -> pd.DataFrame:
    """The coin's hedge-on and hedge-off events: `fill` (the bar the hedge opened or closed),
    `decision` (the stepped bar before it, when the book's wish turned), `kind` ("on"/"off"),
    `liquidation` (an off forced by the venue, not decided), and for an on its episode's `money`
    (USD, the hedge's own accounts from before the open fill to the close fill) and `end`."""
    stepped = record.stepped[coin].to_numpy()
    at = np.flatnonzero(stepped)
    if len(at) < 2:
        return pd.DataFrame(columns=["fill", "decision", "kind", "liquidation", "money", "end"])
    pos = record.in_pos[coin].to_numpy()[at]
    acc = record.hedge_acc[coin].to_numpy()[at]
    liq = record.liquidated[coin].to_numpy()[at]
    index = record.stepped.index
    rows = []
    open_row = None
    for j in range(1, len(at)):
        if pos[j] and not pos[j - 1]:
            open_row = {"fill": index[at[j]], "decision": index[at[j - 1]], "kind": "on",
                        "liquidation": False, "money": np.nan, "end": pd.NaT, "_acc0": acc[j - 1]}
            rows.append(open_row)
        elif pos[j - 1] and not pos[j]:
            if open_row is not None:
                open_row["money"] = acc[j] - open_row["_acc0"]
                open_row["end"] = index[at[j]]
                open_row = None
            rows.append({"fill": index[at[j]], "decision": index[at[j - 1]], "kind": "off",
                         "liquidation": bool(liq[j]), "money": np.nan, "end": pd.NaT,
                         "_acc0": np.nan})
    out = pd.DataFrame(rows, columns=["fill", "decision", "kind", "liquidation", "money", "end",
                                      "_acc0"])
    return out.drop(columns="_acc0")


def episodes(record: BookRecord, coin: str) -> pd.DataFrame:
    """Each closed hedge episode of the coin: `start` (decision bar of the on), `fill`, `end`
    (the bar it closed), `hours`, and its money by part (USD, the book's accounts, from the bar
    before the open fill to the close fill): `price`, `funding`, `fees`, `money`, plus
    `notional` (the short's notional right after it opened)."""
    st = record.stepped[coin].to_numpy()
    at = np.flatnonzero(st)
    pos = record.in_pos[coin].to_numpy()[at]
    parts = {k: record.parts[k][coin].to_numpy()[at] for k in ("price", "funding", "fees")}
    short = record.parts["short_usd"][coin].to_numpy()[at]
    index = record.stepped.index
    rows, open_j = [], None
    for j in range(1, len(at)):
        if pos[j] and not pos[j - 1]:
            open_j = j
        elif pos[j - 1] and not pos[j] and open_j is not None:
            d = {k: v[j] - v[open_j - 1] for k, v in parts.items()}
            rows.append({"start": index[at[open_j - 1]], "fill": index[at[open_j]],
                         "end": index[at[j]],
                         "hours": (index[at[j]] - index[at[open_j]]) / pd.Timedelta(hours=1),
                         **d, "money": d["price"] + d["funding"] - d["fees"],
                         "notional": short[open_j]})
            open_j = None
    return pd.DataFrame(rows, columns=["start", "fill", "end", "hours", "price", "funding",
                                       "fees", "money", "notional"])


# ── 3. variants ──────────────────────────────────────────────────────────


def gate_wish(own: np.ndarray, veto_on: np.ndarray | None, veto_off: np.ndarray | None,
              start_in_pos: bool = False) -> np.ndarray:
    """The book's own wish (bool per stepped bar) under a veto filter, assuming the hedge is open
    exactly when the previous bar's wish was (the book's rule; `_StudyDriver._gate` asks the live
    book instead). An on whose decision bar has `veto_on` is skipped until the wish next drops;
    an off whose decision bar has `veto_off` keeps the hedge until the wish next rises.
    `start_in_pos`: the hedge is open after the first bar (the book started with a hedge wish)."""
    out = np.zeros(len(own), dtype=bool)
    suppress = hold = False
    # the first bar is the book's start: the hedge it opens there is its warm-up wish
    own_prev = final_prev = bool(start_in_pos)
    for i, w in enumerate(own):
        in_pos = final_prev
        if w and not own_prev and not in_pos and veto_on is not None and veto_on[i]:
            suppress = True
        if not w:
            suppress = False
        if not w and own_prev and in_pos and veto_off is not None and veto_off[i]:
            hold = True
        if w or not in_pos:
            hold = hold and not w and in_pos
        own_prev = w
        final_prev = out[i] = (w and not suppress) or hold
    return out


def trigger_wish(raw: np.ndarray, priced: np.ndarray, sticky_hours: int) -> np.ndarray:
    """An alternative rule's wish for the bar after each bar (1.0/0.0, NaN where unpriced): the
    raw condition through the book's sticky exit, as an external wish of the book."""
    wish = np.full(len(raw), np.nan)
    wish[priced] = _bv2._sticky(np.asarray(raw, dtype=bool)[priced].astype(float), sticky_hours)
    return wish


# ── 4. measures ──────────────────────────────────────────────────────────


def hedge_money(perp: np.ndarray, funding: np.ndarray, hedged: np.ndarray,
                cost: float) -> np.ndarray:
    """Per bar s, what a short of one unit of position earns over (s, s+1] while `hedged[s]`:
    minus the perp's return plus the funding settled at s+1, less `cost` per switch at s."""
    px = pd.Series(perp).ffill().to_numpy()
    ret = np.zeros(len(px))
    ret[:-1] = px[1:] / px[:-1] - 1.0
    fund = np.zeros(len(px))
    fund[:-1] = np.nan_to_num(funding[1:])
    h = hedged.astype(float)
    switch = np.abs(np.diff(h, prepend=0.0))
    return np.nan_to_num(h * (-ret + fund)) - switch * cost


def hedged_from_wish(wish: np.ndarray) -> np.ndarray:
    """The hedge held over (s, s+1] given the wish decided at each bar: the previous bar's."""
    w = np.nan_to_num(np.asarray(wish, dtype=float)) > 0.5
    out = np.zeros(len(w), dtype=bool)
    out[1:] = w[:-1]
    return out


def label_events(ev: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """Each event's spot return over the next 1, 3, 7, 14 days from its fill bar: read at the fill
    and at fill + h days (the last price at or before it, at most a day stale); nothing before
    the fill is read. NaN when the window runs past the data."""
    s = spot.ffill(limit=24)
    out = ev.copy()
    out["fill"] = pd.to_datetime(out["fill"], utc=True)  # an empty table has no dtype
    last = spot.last_valid_index()
    for h in HORIZONS_DAYS:
        later = out["fill"] + pd.Timedelta(days=h)
        p0 = s.reindex(out["fill"]).to_numpy()
        p1 = s.reindex(later).to_numpy()
        ok = (later <= last).to_numpy()
        out[f"ret_{h}d"] = np.where(ok, p1 / p0 - 1.0, np.nan)
    return out


def false_share(ev: pd.DataFrame, kind: str, h: int = PRIMARY_HORIZON_DAYS) -> tuple[float, int]:
    """Share of wrong events among those with a full label: an on followed by no fall
    (return >= 0), an off followed by no rise (return <= 0). Returns (share, count)."""
    sel = ev[(ev["kind"] == kind) & ev[f"ret_{h}d"].notna()]
    if kind == "off":
        sel = sel[~sel["liquidation"].astype(bool)]
    if not len(sel):
        return float("nan"), 0
    r = sel[f"ret_{h}d"]
    wrong = (r >= 0) if kind == "on" else (r <= 0)
    return float(wrong.mean()), int(len(sel))


def bootstrap_drop(base: pd.DataFrame, variant: pd.DataFrame, kind: str = "on",
                   h: int = PRIMARY_HORIZON_DAYS, draws: int = 2000, seed: int = 7
                   ) -> tuple[float, float, float]:
    """(point, low, high): the drop of the false-alarm share from `base` to `variant` and its 95%
    interval, months of the decision drawn with replacement (both tables on the same months)."""
    def wrong(ev):
        sel = ev[(ev["kind"] == kind) & ev[f"ret_{h}d"].notna()]
        if kind == "off":
            sel = sel[~sel["liquidation"].astype(bool)]
        r = sel[f"ret_{h}d"]
        flag = (r >= 0) if kind == "on" else (r <= 0)
        month = sel["decision"].dt.tz_convert(None).dt.to_period("M")
        return pd.DataFrame({"m": month.to_numpy(), "w": flag.to_numpy(dtype=float)}).groupby(
            "m")["w"].agg(["sum", "count"])

    a, b = wrong(base), wrong(variant)
    months = sorted(set(a.index) | set(b.index))
    a, b = a.reindex(months, fill_value=0.0), b.reindex(months, fill_value=0.0)
    point = a["sum"].sum() / a["count"].sum() - b["sum"].sum() / b["count"].sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(months), size=(draws, len(months)))
    av, ac = a["sum"].to_numpy()[pick].sum(1), a["count"].to_numpy()[pick].sum(1)
    bv, bc = b["sum"].to_numpy()[pick].sum(1), b["count"].to_numpy()[pick].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        drops = av / ac - bv / bc
    drops = drops[np.isfinite(drops)]
    return float(point), float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))


def choose(money: Mapping[str, pd.Series], cut: pd.Timestamp,
           eligible: Mapping[str, bool] | None = None) -> tuple[str | None, float]:
    """The variant with the most money on the bars BEFORE `cut` (ties: the first named), among
    the eligible ones; nothing on or after `cut` is read. (None, nan) when none is eligible."""
    best, best_value = None, float("nan")
    for name, series in money.items():
        if eligible is not None and not eligible.get(name, False):
            continue
        value = float(series[series.index < cut].sum())
        if best is None or value > best_value:
            best, best_value = name, value
    return best, best_value


def split_date(start, end) -> pd.Timestamp:
    """The first day of the judged last third: start + 2/3 of the window, floored to a day."""
    s, e = _utc(start), _utc(end)
    return (s + (e - s) * 2 / 3).floor("D")


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# ── features ─────────────────────────────────────────────────────────────


def past_percentile(values: pd.Series, min_days: int = MIN_PAST_DAYS) -> pd.Series:
    """Each bar's value placed among the coin's own daily samples (the day's last value) of all
    PREVIOUS days: the share of them below it. NaN until `min_days` samples exist."""
    v = values.copy()
    days = v.index.floor("D")
    daily = v.groupby(days).last()
    out = np.full(len(v), np.nan)
    past: list[float] = []
    day_codes = days.asi8
    starts = np.searchsorted(day_codes, daily.index.asi8, side="left")
    stops = np.searchsorted(day_codes, daily.index.asi8, side="right")
    arr = v.to_numpy(dtype=float)
    for k, (a, b) in enumerate(zip(starts, stops, strict=True)):
        if len(past) >= min_days:
            seg = arr[a:b]
            ranks = np.searchsorted(np.asarray(past), seg, side="left") / len(past)
            out[a:b] = np.where(np.isnan(seg), np.nan, ranks)
        sample = daily.iloc[k]
        if not np.isnan(sample):
            bisect.insort(past, float(sample))
    return pd.Series(out, index=v.index)


def lagged_return(px: pd.Series, hours: int) -> pd.Series:
    p = px.ffill()
    return p / p.shift(hours) - 1.0


def features(perp: pd.Series, spot_own: pd.Series, funding: pd.Series, volume: pd.Series,
             btc: pd.Series | None) -> pd.DataFrame:
    """The declared information at each bar, from bars up to and including it."""
    px = perp.ffill()
    fund24 = funding.fillna(0.0).rolling(24, min_periods=24).sum() * 365
    basis = (perp / spot_own - 1.0).rolling(24, min_periods=12).mean()
    logret = np.log(px).diff()
    vol = logret.rolling(168, min_periods=120).std()
    dollar = (volume * perp).fillna(0.0)
    day_value = dollar.rolling(24, min_periods=24).sum()
    month_mean = day_value.shift(24).rolling(720, min_periods=600).mean()
    surge = day_value / month_mean
    drawdown = px / px.rolling(720, min_periods=600).max() - 1.0
    out = {
        "fund_level": fund24,
        "fund_pct": past_percentile(fund24),
        "basis_pct": past_percentile(basis),
        "vol_pct": past_percentile(vol),
        "surge_pct": past_percentile(surge.replace([np.inf, -np.inf], np.nan)),
        "drawdown": drawdown,
    }
    for d in (1, 3, 7, 14, 30):
        out[f"own_{d}d"] = lagged_return(perp, d * 24)
    if btc is not None:
        for d in (7, 14, 30):
            out[f"btc_{d}d"] = lagged_return(btc, d * 24)
    return pd.DataFrame(out, index=perp.index)


@dataclass(frozen=True)
class Variant:
    family: str
    kind: str  # "veto_on", "veto_off", "trigger"
    name: str
    column: str | tuple[str, ...]
    side: str  # "above" / "below" / "any_not_up"
    threshold: float

    def condition(self, feats: pd.DataFrame) -> np.ndarray:
        """True where the variant's condition holds; False where the feature is unknown."""
        if self.side == "any_not_up":
            cols = (self.column,) if isinstance(self.column, str) else self.column
            known = np.ones(len(feats), dtype=bool)
            any_down = np.zeros(len(feats), dtype=bool)
            for c in cols:
                x = feats[c].to_numpy(dtype=float)
                known &= ~np.isnan(x)
                any_down |= ~(x > self.threshold)
            return known & any_down
        x = feats[self.column].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            hit = x > self.threshold if self.side == "above" else x < self.threshold
        return hit & ~np.isnan(x)


def declared_variants(with_btc: bool = True) -> list[Variant]:
    """The 116 filters and 53 triggers of the protocol (section 5), in a fixed order."""
    pct = (0.2, 0.4, 0.6, 0.8)
    grids = [("F1", "fund_level", (0.0, 0.10, 0.30)), ("F2", "fund_pct", pct),
             ("F3", "basis_pct", pct), ("F4", "vol_pct", pct), ("F5", "surge_pct", pct),
             ("F6", "drawdown", (-0.025, -0.05, -0.10, -0.20))]
    out: list[Variant] = []
    for kind in ("veto_on", "veto_off"):
        for fam, col, ths in grids:
            for th in ths:
                for side in ("above", "below"):
                    out.append(Variant(fam, kind, f"{kind} {col} {side} {th:g}", col, side, th))
        if with_btc:
            for d in (7, 14, 30):
                for side in ("above", "below"):
                    out.append(Variant("F7", kind, f"{kind} btc_{d}d {side} 0", f"btc_{d}d",
                                       side, 0.0))
        for d in (1, 3, 7):
            for side in ("above", "below"):
                out.append(Variant("F8", kind, f"{kind} own_{d}d {side} 0", f"own_{d}d", side,
                                   0.0))
    for fam, col, ths in grids[:5]:
        for th in ths:
            for side in ("above", "below"):
                out.append(Variant(fam, "trigger", f"hedge while {col} {side} {th:g}", col, side,
                                   th))
    for th in grids[5][2]:
        out.append(Variant("F6", "trigger", f"hedge while drawdown below {th:g}", "drawdown",
                           "below", th))
    if with_btc:
        for d in (7, 14, 30):
            out.append(Variant("F7", "trigger", f"hedge while btc_{d}d not up", (f"btc_{d}d",),
                               "any_not_up", 0.0))
        out.append(Variant("F7", "trigger", "the book's rule on BTC's prices (14d, 30d)",
                           ("btc_14d", "btc_30d"), "any_not_up", 0.0))
    for days in ((1,), (3,), (7,), (1, 7), (3, 14), (7, 30), (1, 3, 7, 14, 30)):
        cols = tuple(f"own_{d}d" for d in days)
        out.append(Variant("F8", "trigger", f"hedge while own return not up on {days} days",
                           cols, "any_not_up", 0.0))
    return out


__all__ = [
    "BookRecord", "StudyPanel", "Variant", "bootstrap_drop", "declared_variants", "episodes",
    "events",
    "false_share", "features", "gate_wish", "hedge_money", "hedged_from_wish", "label_events",
    "load_panel", "past_percentile", "run_book", "split_date", "trigger_wish",
]
