"""What fetching hourly candles costs (`qlab.hl_fetch`): counted in requests, by kind of column and
by what the raw cache holds. The numbers in the tests are worked by hand from the page sizes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qlab import hl_fetch
from qlab.pipeline.spec import StrategySpec

COST = hl_fetch.FetchCost(hl_seconds_per_request=1.0, binance_seconds_per_request=0.5,
                          hl_listing_perps=10, hl_listing_spots=2, binance_listing_perps=20,
                          binance_listing_spots=20)


def _spec(interval: str = "1h", instruments: list[str] | None = None,
          source: str = "hyperliquid", start: str = "2026-03-08", end: str = "2026-10-30",
          **data_extra) -> StrategySpec:
    data: dict = {"source": source, "interval": interval, "start": start, "end": end, **data_extra}
    if instruments is not None:
        data["instruments"] = instruments
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": {}, "data": data,
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


@pytest.fixture()
def uncached(monkeypatch):
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)


def _cache(root: Path, venue: str, interval: str, *entries: str) -> Path:
    """Empty cache files named like the fetcher's: `COLUMN__START__END`."""
    folder = root / venue / interval
    folder.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (folder / f"{entry}.parquet").write_bytes(b"")
    return root


def test_the_shipped_numbers_are_the_measured_ones() -> None:
    cost = hl_fetch.load_config()
    assert cost.hl_seconds_per_request == 1.3 and cost.binance_seconds_per_request == 0.51
    assert (cost.hl_listing_perps, cost.hl_listing_spots) == (223, 15)


def test_a_missing_key_is_an_error(tmp_path) -> None:
    bad = tmp_path / "f.yaml"
    bad.write_text("hl_seconds_per_request: 1.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        hl_fetch.load_config(bad)


# ---- requests of one column: the arithmetic of the page sizes ----------------------------------


def test_a_whole_hyperliquid_perpetual_is_the_measured_fourteen_requests() -> None:
    """2026-03-08 .. 2026-10-30 is 236 days = 5664 hours. Candles: 5664 >= 5000, a full page asks
    for one more = 2. Funding: (5664 + 2) // 500 = 11 full pages, plus the last = 12. Together
    14, which at 18.4 s measured is 1.31 s a request."""
    assert hl_fetch.hours_between(date(2026, 3, 8), date(2026, 10, 30)) == 5664
    assert hl_fetch.requests_whole("hyperliquid", False, 5664) == 14
    assert hl_fetch.requests_whole("hyperliquid", False, 5520) == 14  # 230 days: 2 + 11 + 1
    assert hl_fetch.requests_whole("hyperliquid", False, 720) == 1 + (722 // 500 + 1)  # 3


def test_a_whole_hyperliquid_spot_market_is_candles_only() -> None:
    assert hl_fetch.requests_whole("hyperliquid", True, 5664) == 2  # 2.6 s measured
    assert hl_fetch.requests_whole("hyperliquid", True, 720) == 1


def test_a_whole_binance_column() -> None:
    """5520 hours. A perpetual: klines 5520 // 1500 + 1 = 4; funding every eight hours is
    5520 // 8 + 2 = 692 rows, 692 // 1000 + 1 = 1 page; 5. A spot market: 5520 // 1000 + 1 = 6.
    394 perpetuals and 406 spot columns = 4406 requests, which the 37.4 measured minutes make
    0.51 s each."""
    assert hl_fetch.requests_whole("binance", False, 5520) == 5
    assert hl_fetch.requests_whole("binance", True, 5520) == 6
    assert 394 * 5 + 406 * 6 == 4406


def test_extending_by_a_day_is_two_requests_for_a_perpetual_and_one_for_a_spot_market() -> None:
    # an entry cached to the day before, the refetch begins a day before that: 48 hours
    assert hl_fetch.requests_gap("hyperliquid", False, 48) == 2
    assert hl_fetch.requests_gap("hyperliquid", True, 48) == 1
    assert hl_fetch.requests_gap("binance", False, 48) == 2


# ---- the estimate of a data request -------------------------------------------------------------


def test_a_cached_snapshot_or_a_daily_panel_costs_nothing(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: object())
    assert hl_fetch.estimate(_spec(), None, COST).cached
    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)
    assert hl_fetch.estimate(_spec("1d"), None, COST).minutes == 0
    assert hl_fetch.estimate(_spec(source="aster"), None, COST).minutes == 0


def test_each_column_costs_by_what_the_cache_holds(uncached, tmp_path) -> None:
    """A: cached for exactly the window = 0. B: cached to 2026-10-29, one day short: the gap is
    (10-30 - 10-29) * 24 + 24 = 48 hours = 2 requests. C: nothing = 14. A-SPOT: nothing = 2.
    18 requests at 1 s = 18 s."""
    root = _cache(tmp_path, "hyperliquid", "1h", "A__2026-03-08__2026-10-30",
                  "B__2026-03-08__2026-10-29")
    est = hl_fetch.estimate(_spec(instruments=["A", "B", "C", "A-SPOT"]), None, COST,
                            cache_dir=root)
    assert est.minutes == pytest.approx(18 / 60) and not est.cached
    assert "3 контрактов (1 целиком, 1 докачать, 1 в кэше)" in est.why
    assert "1 спотов (1 целиком, 0 докачать, 0 в кэше)" in est.why


def test_a_window_that_is_all_in_the_cache_costs_nothing(uncached, tmp_path) -> None:
    root = _cache(tmp_path, "hyperliquid", "1h", "A__2026-03-08__2026-10-30",
                  "A-SPOT__2026-03-08__2026-10-30")
    est = hl_fetch.estimate(_spec(instruments=["A", "A-SPOT"]), None, COST, cache_dir=root)
    assert est.minutes == 0 and est.cached and "всё в кэше" in est.why


def test_the_top10_pool_case_is_priced_by_requests_not_by_names(uncached, tmp_path) -> None:
    """38 perpetuals and 8 spot markets, 2025-01-01 .. 2026-10-30, the real shape of the second
    stage of bv2-research-top10-pool. With every column cached the fetch is 0 (what happened:
    the 80 minutes were the computing, not the fetching). With none cached, a perpetual is 35
    requests (667 days = 16008 hours: candles 2, funding 32 full pages and the last) and a spot
    market 2; the old estimate, 4.4 s a name, said 46 * 4.4 = 202 s whatever the cache."""
    names = [f"P{i}" for i in range(38)] + [f"S{i}-SPOT" for i in range(8)]
    hours = hl_fetch.hours_between(date(2025, 1, 1), date(2026, 10, 30))
    assert hours == 667 * 24 == 16008
    perp = hl_fetch.requests_whole("hyperliquid", False, hours)
    assert perp == 2 + (16008 + 2) // 500 + 1 == 35
    est = hl_fetch.estimate(_spec(instruments=names, start="2025-01-01"), None, COST,
                            cache_dir=tmp_path)
    assert est.minutes == pytest.approx((38 * 35 + 8 * 2) / 60)  # 1346 s: 22.4 minutes
    held = _cache(tmp_path / "held", "hyperliquid", "1h",
                  *[f"{n}__2025-01-01__2026-10-30" for n in names])
    assert hl_fetch.estimate(_spec(instruments=names, start="2025-01-01"), None, COST,
                             cache_dir=held).minutes == 0


def test_funding_cached_by_a_daily_fetch_is_not_asked_for_again(uncached, tmp_path) -> None:
    """The perpetual's funding series is the same at every bar size
    (`reusable_funding`): a whole hourly column then costs only its candles, 14 - 12 = 2."""
    root = _cache(tmp_path, "hyperliquid", "1d", "A__2026-03-08__2026-10-30")
    est = hl_fetch.estimate(_spec(instruments=["A"]), None, COST, cache_dir=root)
    assert est.minutes == pytest.approx(2 / 60)


def test_binance_is_priced_by_its_own_pages_and_divided_among_the_fetch_workers(
        uncached, tmp_path, monkeypatch) -> None:
    """X and Y with their spot legs over 5520 hours: 5 + 5 + 6 + 6 = 22 requests at 0.5 s with two
    workers (QLAB_FETCH_WORKERS=2) = 22 * 0.25 = 5.5 s."""
    monkeypatch.setenv("QLAB_FETCH_WORKERS", "2")
    spec = _spec(source="binance", instruments=["X", "Y"], include_spot=True,
                 start="2026-03-08", end="2026-10-24")
    est = hl_fetch.estimate(spec, None, COST, cache_dir=tmp_path)
    assert est.minutes == pytest.approx(5.5 / 60)
    assert "2 контрактов (2 целиком" in est.why and "2 спотов (2 целиком" in est.why
    monkeypatch.setenv("QLAB_FETCH_WORKERS", "1")
    assert hl_fetch.estimate(spec, None, COST, cache_dir=tmp_path).minutes == pytest.approx(11 / 60)


def test_hyperliquid_perpetuals_held_against_binance_spot_pay_each_venues_rate(
        uncached, tmp_path, monkeypatch) -> None:
    """`spot_source: binance`: the perpetuals A and B cost 14 requests each at 1 s, their Binance
    spot legs 6 each at 0.5 s (one worker): 28 + 6 = 34 s."""
    monkeypatch.setenv("QLAB_FETCH_WORKERS", "1")
    spec = _spec(instruments=["A", "B", "A-SPOT", "B-SPOT"], spot_prices="own",
                 spot_source="binance", end="2026-10-24", start="2026-03-08")
    # 230 days = 5520 hours: HL perp 14, Binance spot 6
    est = hl_fetch.estimate(spec, None, COST, cache_dir=tmp_path)
    assert est.minutes == pytest.approx((2 * 14 * 1.0 + 2 * 6 * 0.5) / 60)
    assert "Hyperliquid: 2 контрактов" in est.why and "Binance: 2 спотов" in est.why


def test_a_whole_listing_is_what_the_cache_has_seen_filled_up_to_the_configured_size(
        uncached, tmp_path) -> None:
    """No instrument list: the three perpetuals the cache knows (cached for the window = 0) and
    seven assumed ones (listing size 10), 14 requests each = 98 s."""
    root = _cache(tmp_path, "hyperliquid", "1h", *[f"K{i}__2026-03-08__2026-10-30"
                                                    for i in range(3)])
    est = hl_fetch.estimate(_spec(), None, COST, cache_dir=root)
    assert est.minutes == pytest.approx(98 / 60)
    assert "10 контрактов (7 целиком, 0 докачать, 3 в кэше)" in est.why


# ---- the item, with the time it took last time -------------------------------------------------


def test_an_item_is_expected_to_take_the_larger_of_its_fetch_and_its_last_time() -> None:
    fetch = hl_fetch.Estimate(10.0, False, "часовые свечи: 10 минут", fetch_minutes=10.0)
    again = hl_fetch.total([fetch], 4800.0)  # 80 minutes the last time
    assert again.minutes == 80 and again.fetch == 10
    assert "в прошлый раз идея заняла 80 минут" in again.why
    assert hl_fetch.total([fetch], 120.0).minutes == 10  # 2 minutes: the fetch is the larger
    assert hl_fetch.total([fetch], None).minutes == 10
    only_history = hl_fetch.total([], 4800.0)
    assert only_history.minutes == 80 and only_history.cached


def test_heavy_means_longer_than_the_whole_stage_may_take() -> None:
    long = hl_fetch.Estimate(80.0, True, "в прошлый раз идея заняла 80 минут")
    reason = hl_fetch.heavy_reason(long, 60)
    assert reason.startswith("данные слишком тяжелы за ночь") and "--allow-heavy" in reason
    assert "оценка 80 минут при пределе этапа на всю ночь 60" in reason
    assert hl_fetch.heavy_reason(hl_fetch.Estimate(12.0, False, "x"), 30) is None
    assert hl_fetch.heavy_reason(hl_fetch.Estimate(0.0, True, "x"), 30) is None


def test_the_last_time_comes_from_the_ledger_then_from_the_journal(monkeypatch) -> None:
    from contextlib import contextmanager

    import qlab.nightflow.journal as journal
    import qlab.registry.db as db

    @contextmanager
    def scope():
        yield object()

    monkeypatch.setattr(db, "session_scope", scope)
    monkeypatch.setattr(journal, "last_item_seconds",
                        lambda session, step, name: 4793.7 if (step, name) == ("restamp", "i")
                        else None)
    assert hl_fetch.last_seconds({"seconds": 12.5}, "restamp", "i") == 12.5
    assert hl_fetch.last_seconds({"seconds": None}, "restamp", "i") == 4793.7
    assert hl_fetch.last_seconds(None, "restamp", "other") is None
