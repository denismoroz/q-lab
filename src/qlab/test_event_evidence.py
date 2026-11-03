"""Tests for the measured history of venue events (`qlab.event_evidence`): synthetic caches
with known paths, so every number is checked by hand; no real data, no network."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from qlab import event_evidence as ev

COSTS = {"binance": 5.0, "hyperliquid": 5.0}  # bps per leg -> 0.001 a round trip of one leg
CFG = {**ev.DEFAULTS, "min_universe": 3, "top": 1, "horizons_days": [1, 7],
       "listing_horizons_days": [1, 7, 30],
       "liquid_floor_usd": {"binance": 1.0, "hyperliquid": 1.0}}


def _panel(days: int = 120, symbols=("A", "B", "C", "HOT"), venue="binance") -> ev.Panel:
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    price = pd.DataFrame(100.0, index=index, columns=list(symbols))
    notional = pd.DataFrame(1_000.0, index=index, columns=list(symbols))
    funding = pd.DataFrame(0.0, index=index, columns=list(symbols))
    return ev.Panel(venue, price, notional, funding, {}, index[0])


# --------------------------------------------------------------------------
# Statistics


def test_summarize_clusters_by_month_and_splits_halves() -> None:
    dates = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03", "2026-03-04",
                                      "2026-04-04", "2026-05-05", "2026-06-06", "2026-07-07"]))
    values = pd.Series([0.02, 0.02, 0.01, 0.03, -0.01, 0.02, 0.04, 0.03])
    out = ev.summarize(values, dates)
    assert out["n"] == 8 and out["months"] == 7
    assert out["mean_pct"] == 2.0
    assert out["hit_rate"] == 0.875
    assert out["first_half_pct"] == 2.0 and out["second_half_pct"] == 2.0
    assert out["t"] is not None and out["t"] > 2


def test_with_few_months_no_t_value_is_given() -> None:
    dates = pd.Series(pd.to_datetime(["2026-01-05"] * 4 + ["2026-02-05"] * 4))
    out = ev.summarize(pd.Series([0.01] * 8), dates)
    assert out["months"] == 2 and out["t"] is None


def test_nan_results_are_left_out_of_a_sample() -> None:
    dates = pd.Series(pd.to_datetime(["2026-01-05", "2026-02-05", "2026-03-05"]))
    assert ev.summarize(pd.Series([0.01, np.nan, 0.03]), dates)["n"] == 2
    assert ev.summarize(pd.Series([np.nan]), dates.iloc[:1]) == {"n": 0}


# --------------------------------------------------------------------------
# Finding events


def test_top_events_count_an_episode_once_and_never_look_ahead() -> None:
    index = pd.date_range("2026-01-01", periods=40, freq="D")
    score = pd.DataFrame(0.0, index=index, columns=["A", "B", "C", "HOT"])
    score.loc[index[10:17], "HOT"] = 5.0  # seven days at the top
    eligible = score.notna()
    found = ev.top_events(score, eligible & (score > 0), top=1, largest=True, min_universe=1,
                          episode_days=7)
    assert list(found.index[found["HOT"]]) == [index[10]]
    # Causal: the verdict of a day does not change when later days are cut away.
    for cut in (11, 14, 25):
        part = ev.top_events(score.iloc[:cut], eligible.iloc[:cut] & (score.iloc[:cut] > 0),
                             top=1, largest=True, min_universe=1, episode_days=7)
        assert part.equals(found.iloc[:cut])


def test_a_day_with_too_small_a_universe_has_no_events() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    score = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5}, index=index)
    found = ev.top_events(score, score.notna(), top=1, largest=True, min_universe=3,
                          episode_days=7)
    assert not found.to_numpy().any()


# --------------------------------------------------------------------------
# What follows a funding extreme


def test_funding_extreme_numbers_after_costs() -> None:
    panel = _panel()
    day = panel.price.index[40]
    panel.funding.loc[panel.price.index[40:47], "HOT"] = 0.003
    panel.price.loc[panel.price.index[41]:, "HOT"] = 99.0  # a one percent fall the day after
    patterns = ev.funding_extremes(panel, CFG, COSTS)
    high = next(p for p in patterns if p["key"] == "funding-extreme/binance/high/all")
    assert high["n_events"] == 1 and high["first_event"] == str(day.date())
    carry = next(m for m in high["measures"] if m["id"] == "collect_funding_hedged@1d")
    directional = next(m for m in high["measures"] if m["id"] == "directional_perpetual@1d")
    # funding of the day after (0.3%) less a round trip of two legs (2 x 0.1%)
    assert carry["mean_pct"] == 0.1
    # short the perpetual: funding 0.3% + price fall 1% - one leg's round trip 0.1%
    assert directional["mean_pct"] == 1.2
    assert high["profile"]["1d"]["funding_kept_median"] == 1.0
    week = next(m for m in high["measures"] if m["id"] == "collect_funding_hedged@7d")
    assert week["mean_pct"] == round(100 * (0.003 * 6 - 0.002), 3)
    assert not [p for p in patterns if p["key"].endswith("/low/all")]  # no negative funding


def test_a_measure_uses_only_data_after_the_event_day() -> None:
    first, second = _panel(), _panel()
    for panel in (first, second):
        panel.funding.loc[panel.price.index[40:47], "HOT"] = 0.003
    # A different future beyond the 7-day window of the event (day 40), and a different past.
    second.price.iloc[48:, :] *= 3
    second.price.iloc[:39, :] *= 2
    a = ev.funding_extremes(first, CFG, COSTS)[0]
    b = ev.funding_extremes(second, CFG, COSTS)[0]
    def pick(p: dict) -> dict:
        return {m["id"]: m["mean_pct"] for m in p["measures"]}

    assert pick(a) == pick(b) and len(pick(a)) == 4


# --------------------------------------------------------------------------
# Listings, volume jumps, delistings


def test_listing_measures_the_days_after_the_first_day() -> None:
    panel = _panel(days=60, symbols=("BTC", "NEW"))
    index = panel.price.index
    panel.price.loc[index[:10], "NEW"] = np.nan
    panel.price.loc[index[10]:, "NEW"] = 100.0
    panel.price.loc[index[11]:, "NEW"] = 90.0       # a tenth lower after the first day
    panel.funding["NEW"] = 0.001
    panel.funding.loc[index[:10], "NEW"] = np.nan
    pattern = ev.listings(panel, CFG, COSTS)[0]
    assert pattern["key"] == "listing/binance" and pattern["n_events"] == 1
    assert pattern["first_event"] == str(index[10].date())
    short = next(m for m in pattern["measures"] if m["id"] == "short_perpetual@1d")
    long_ = next(m for m in pattern["measures"] if m["id"] == "long_perpetual@1d")
    assert short["mean_pct"] == 10.0   # 0.1 fall + 0.001 funding - 0.001 round trip
    assert long_["mean_pct"] == -10.2  # -(0.1) - 0.001 funding - 0.001 round trip
    hedged = next(m for m in pattern["measures"] if m["id"] == "short_against_btc@1d")
    assert hedged["mean_pct"] == round(100 * (0.001 + 0.1 - 0.002), 3)


def test_a_contract_already_in_the_cache_at_its_start_is_no_listing() -> None:
    panel = _panel(days=30, symbols=("BTC", "OLD"))
    assert ev.listings(panel, CFG, COSTS) == []


def test_volume_jump_follow_and_fade_are_mirror_images_apart_from_costs() -> None:
    panel = _panel(days=60)
    index = panel.price.index
    panel.notional.loc[index[30], "HOT"] = 50_000.0   # a fifty-fold jump
    panel.price.loc[index[30]:, "HOT"] = 110.0          # up ten percent that day
    panel.price.loc[index[31]:, "HOT"] = 121.0          # and ten percent more the next
    pattern = ev.volume_jumps(panel, CFG, COSTS)[0]
    follow = next(m for m in pattern["measures"] if m["id"] == "follow_the_move@1d")
    fade = next(m for m in pattern["measures"] if m["id"] == "fade_the_move@1d")
    assert follow["mean_pct"] == 9.9 and fade["mean_pct"] == -10.1


def test_delisting_is_measured_from_an_assumed_announcement_and_never_tradeable() -> None:
    panel = _panel(days=60, symbols=("BTC", "GONE"))
    index = panel.price.index
    panel.last_trading_day["GONE"] = index[50]
    panel.price.loc[index[47]:, "GONE"] = 100.0
    panel.price.loc[index[50]:, "GONE"] = 90.0
    pattern = ev.delistings(panel, CFG, COSTS)[0]
    assert pattern["tradeable"] is False and "assumption" in pattern
    short = pattern["measures"][0]
    assert short["horizon_days"] == 3 and short["mean_pct"] == 9.9


# --------------------------------------------------------------------------
# The caches


def _write(folder, name, start, end, prices, *, funding=None, notional=1000.0, delisted=False,
           volume_from=None):
    folder.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    frame = pd.DataFrame({"price": prices, "volume": notional / np.array(prices), "funding": np.nan,
                          "trade_count": 1.0, "high": np.nan}, index=days)
    if funding is not None:
        stamps = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq="8h")
        rows = pd.DataFrame({"price": np.nan, "volume": np.nan, "funding": funding,
                             "trade_count": np.nan, "high": np.nan}, index=stamps)
        frame = pd.concat([frame, rows]).sort_index()
    stem = f"{name}__{start}__{end}"
    frame.to_parquet(folder / f"{stem}.parquet")
    (folder / f"{stem}.json").write_text(json.dumps({"is_delisted": delisted}))


def test_the_widest_file_of_a_contract_is_read_and_funding_goes_to_the_day_it_ended(
        tmp_path) -> None:
    folder = tmp_path / "data/raw/binance/1d"
    _write(folder, "AAA", "2026-01-01", "2026-01-10", [100.0] * 10, funding=0.0001)
    _write(folder, "AAA", "2026-01-01", "2026-03-01", [100.0] * 60, funding=0.0001)
    _write(folder, "BBB", "2026-01-05", "2026-03-01", [50.0] * 56)
    _write(folder, "AAA-SPOT", "2026-01-01", "2026-03-01", [100.0] * 60)  # a hedge leg's history
    assert ev.widest_files(folder)["AAA"].stem.endswith("2026-03-01")
    panel = ev.load_panel(tmp_path, "binance", date(2026, 3, 1))
    assert set(panel.price.columns) == {"AAA", "BBB"}   # no spot series among the perpetuals
    # An interior day holds the three payments stamped in (00:00 of the day, 00:00 of the next].
    assert abs(panel.funding.loc["2026-01-05", "AAA"] - 3 * 0.0001) < 1e-12
    assert panel.funding["BBB"].isna().all()   # no funding row at all is no zero
    assert panel.cache_start == pd.Timestamp("2026-01-01")


def test_a_removed_contract_trades_until_its_last_day_with_volume(tmp_path) -> None:
    folder = tmp_path / "data/raw/binance/1d"
    _write(folder, "BTC", "2026-01-01", "2026-03-01", [100.0] * 60)
    _write(folder, "GONE", "2026-01-01", "2026-03-01", [100.0] * 60, delisted=True)
    # the last ten days are a flat settlement price with no volume
    path = next(folder.glob("GONE__*.parquet"))
    frame = pd.read_parquet(path)
    frame.iloc[-10:, frame.columns.get_loc("volume")] = 0.0
    frame.to_parquet(path)
    panel = ev.load_panel(tmp_path, "binance", date(2026, 3, 1))
    assert panel.last_trading_day["GONE"] == pd.Timestamp("2026-02-19")


def test_build_writes_every_kind_of_pattern_and_names_what_it_could_not_read(tmp_path) -> None:
    rng = np.random.default_rng(1)
    for venue in ("binance", "hyperliquid"):
        folder = tmp_path / f"data/raw/{venue}/1d"
        for i, name in enumerate(["BTC", "A", "B", "C", "D"]):
            walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
            funding = rng.normal(0.0001 * (i + 1), 0.0001, 3 * 200 + 1)
            _write(folder, name, "2026-01-01", "2026-07-19", list(walk), funding=funding)
        _write(folder, "LATE", "2026-04-01", "2026-07-19", [100.0] * 110, funding=0.0001)
    cfg = {**CFG, "venues": ["binance", "hyperliquid"], "listing_venues": ["binance"],
           "oi_venues": []}
    evidence = ev.build(date(2026, 7, 20), tmp_path, cfg, cost_bps=COSTS,
                        events=[{"source": "binance", "kind": "listed", "instrument": "LATEUSDT",
                                 "detail": ""},
                                {"source": "hyperliquid", "kind": "funding-extreme",
                                 "instrument": "A", "detail": "+10% a year"}])
    kinds = {p["kind"] for p in evidence["patterns"]}
    assert {"funding-extreme", "volume-jump", "listing"} <= kinds
    assert evidence["last_data_day"] == "2026-07-19"
    assert evidence["night"] == "2026-07-20"
    assert evidence["caveats"] and evidence["costs_bps_per_leg"] == COSTS
    today = {(t["kind"], t["instrument"]): t["history"] for t in evidence["today"]}
    assert today[("listed", "LATEUSDT")] == ["listing/binance"]
    assert today[("funding-extreme", "A")] and all(
        k.startswith("funding-extreme/hyperliquid") for k in today[("funding-extreme", "A")])
    path = ev.write(date(2026, 7, 20), evidence, tmp_path)
    assert path == tmp_path / "data/events/evidence/2026-07-20.json"
    again_path, again = ev.ensure(date(2026, 7, 20), tmp_path, cfg)
    assert again_path == path and again["built_at"] == evidence["built_at"]  # read, not rebuilt


def test_a_venue_without_a_cache_is_skipped_and_said_so(tmp_path) -> None:
    folder = tmp_path / "data/raw/binance/1d"
    for name in ("BTC", "A", "B", "C"):
        _write(folder, name, "2026-01-01", "2026-03-01", [100.0] * 60, funding=0.0001)
    cfg = {**CFG, "venues": ["binance"], "listing_venues": ["binance", "hyperliquid-xyz"],
           "oi_venues": []}
    evidence = ev.build(date(2026, 3, 2), tmp_path, cfg, cost_bps={**COSTS})
    assert any(s.startswith("hyperliquid-xyz") for s in evidence["skipped"])
