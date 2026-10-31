"""What fetching Hyperliquid's hourly candles costs (`qlab.hl_fetch`)."""

from __future__ import annotations

import pytest

from qlab import hl_fetch
from qlab.pipeline.spec import StrategySpec

COST = hl_fetch.FetchCost(all_contracts_minutes=60.0, seconds_per_instrument=4.4)


def _spec(interval: str = "1h", instruments: list[str] | None = None,
          source: str = "hyperliquid") -> StrategySpec:
    data: dict = {"source": source, "interval": interval, "start": "2026-03-08",
                  "end": "2026-10-30"}
    if instruments is not None:
        data["instruments"] = instruments
    return StrategySpec.model_validate({
        "idea_id": "x", "title": "x", "code_ref": "a:b", "params": {}, "data": data,
        "costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": []})


def test_the_shipped_numbers_are_the_measured_ones() -> None:
    cost = hl_fetch.load_config()
    assert (cost.all_contracts_minutes, cost.seconds_per_instrument) == (60.0, 4.4)


def test_a_missing_key_is_an_error(tmp_path) -> None:
    bad = tmp_path / "f.yaml"
    bad.write_text("all_contracts_minutes: 60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key"):
        hl_fetch.load_config(bad)


def test_a_cached_snapshot_or_a_daily_panel_costs_nothing(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: object())
    assert hl_fetch.estimate(_spec(), None, COST).cached
    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)
    assert hl_fetch.estimate(_spec("1d"), None, COST).minutes == 0
    assert hl_fetch.estimate(_spec(source="binance"), None, COST).minutes == 0


def test_an_uncached_hourly_panel_takes_the_measured_time(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)
    whole = hl_fetch.estimate(_spec(), None, COST)
    assert (whole.minutes, whole.cached) == (60.0, False)
    named = hl_fetch.estimate(_spec(instruments=[f"C{i}" for i in range(100)]), None, COST)
    assert named.minutes == pytest.approx(100 * 4.4 / 60)


def test_heavy_means_longer_than_the_whole_stage_may_take(monkeypatch) -> None:
    long = hl_fetch.Estimate(60.0, False, "часовые свечи всех контрактов площадки, около 60 минут")
    reason = hl_fetch.heavy_reason(long, 30)
    assert reason.startswith("данные слишком тяжелы за ночь") and "--allow-heavy" in reason
    assert hl_fetch.heavy_reason(hl_fetch.Estimate(12.0, False, "x"), 30) is None
    assert hl_fetch.heavy_reason(hl_fetch.Estimate(0.0, True, "x"), 30) is None
