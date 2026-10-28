"""Tests for `qlab.overlay_diagnostics`: the three readings of a forward series that describe
and decide nothing (docs/OVERLAYS.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import overlay_diagnostics as dg


def series(values, start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC"))


# ---- 1. from every possible entry day ---------------------------------------------------------


def test_a_steady_series_has_one_answer_whatever_the_entry_day() -> None:
    spread = dg.entry_day_spread(series([0.01] * 100), 30)
    expected = 1.01 ** 30 - 1
    assert spread["starts"] == 71
    assert spread["worst"] == pytest.approx(expected) and spread["best"] == pytest.approx(expected)
    assert spread["median"] == pytest.approx(expected)
    assert spread["share_positive"] == 1.0


def test_worst_and_best_entry_days_are_found() -> None:
    values = [0.0] * 100
    values[10] = -0.30                    # one bad day
    values[70] = 0.50                     # one good day
    spread = dg.entry_day_spread(series(values), 30)
    assert spread["worst"] == pytest.approx(-0.30)        # every start that holds day 10 only
    assert spread["best"] == pytest.approx(0.50)
    assert spread["median"] == 0.0
    assert 0 < spread["share_positive"] < 0.5


def test_a_series_shorter_than_the_holding_period_says_so() -> None:
    assert dg.entry_day_spread(series([0.01] * 20), 30) is None
    rows = dg.entry_days(series([0.01] * 40), None)
    assert rows["30"]["strategy"]["starts"] == 11 and rows["60"]["strategy"] is None
    assert rows["30"]["hold"] is None


def test_hold_the_same_coins_is_read_the_same_way() -> None:
    rows = dg.entry_days(series([0.01] * 100), series([-0.01] * 100))
    assert rows["90"]["strategy"]["worst"] > 0 > rows["90"]["hold"]["best"]


# ---- 2. does a good period predict the next ---------------------------------------------------


def test_periods_are_whole_blocks_ending_on_the_last_day() -> None:
    returns = dg.period_returns(series([0.01] * 7 + [0.02] * 7 + [0.03] * 7 + [0.0] * 3), 7)
    # 24 days: the first 3 are dropped so that the last block ends on the last day
    assert returns == pytest.approx([1.01 ** 4 * 1.02 ** 3 - 1, 1.02 ** 4 * 1.03 ** 3 - 1,
                                     1.03 ** 4 - 1])


def test_alternating_periods_have_a_negative_correlation_and_are_distinguishable() -> None:
    blocks = [0.01 if k % 2 == 0 else -0.01 for k in range(30)]
    daily = series(sum(([b] * 7 for b in blocks), []))
    week = dg.persistence(daily)["week"]
    assert week["pairs"] == 29 and week["correlation"] == pytest.approx(-1.0)
    assert week["distinguishable"] is True
    assert week["noise_band"] == pytest.approx(2 / np.sqrt(29))


def test_trending_periods_have_a_positive_correlation() -> None:
    blocks = [0.001 * k for k in range(30)]
    daily = series(sum(([b] * 7 for b in blocks), []))
    assert dg.persistence(daily)["week"]["correlation"] > 0.9


def test_noise_is_inside_its_band() -> None:
    daily = series(np.random.default_rng(1).normal(0, 0.01, 7 * 400))
    week = dg.persistence(daily)["week"]
    assert week["pairs"] == 399 and week["distinguishable"] is False
    assert abs(week["correlation"]) < week["noise_band"]


def test_too_few_pairs_say_not_defined() -> None:
    daily = series(np.random.default_rng(2).normal(0, 0.01, 100))
    quarter = dg.persistence(daily)["quarter"]
    assert quarter["periods"] == 1 and quarter["pairs"] == 0
    assert quarter["correlation"] is None and quarter["distinguishable"] is None
    month = dg.persistence(daily)["month"]
    assert month["pairs"] == 2 and month["correlation"] is None   # fewer than eight pairs


# ---- 3. by market state known at the time -----------------------------------------------------


def _closes(n: int = 400, seed: int = 3) -> pd.Series:
    """BTC-like closes indexed by close time (a day after the candle's open)."""
    index = pd.date_range("2023-01-02", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=index)


def test_every_day_has_exactly_one_state_and_the_early_ones_are_unknown() -> None:
    closes = _closes()
    net = series(np.random.default_rng(4).normal(0, 0.01, 300), start="2023-03-01")
    states = dg.market_states(net, closes)
    for group in ("direction", "volatility"):
        assert sum(v["days"] for v in states[group].values()) == 300
    early = series([0.01] * 10, start="2023-01-03")
    assert dg.market_states(early, closes)["direction"]["unknown"]["days"] == 10


def test_a_days_state_does_not_depend_on_what_happens_after_it() -> None:
    closes = _closes()
    index = pd.date_range("2023-06-01", periods=60, freq="1D", tz="UTC")
    before = dg.direction_states(closes, index)
    before_vol = dg.volatility_states(closes, index)
    tampered = closes.copy()
    tampered.loc[tampered.index >= "2023-07-15"] *= 3.0     # a jump in the future of day 45
    assert dg.direction_states(tampered, index).iloc[:44].equals(before.iloc[:44])
    assert dg.volatility_states(tampered, index).iloc[:44].equals(before_vol.iloc[:44])


def test_a_state_holds_the_days_of_that_state_only() -> None:
    closes = _closes(500)
    net = series([0.0] * 200, start="2023-07-01")
    states = dg.direction_states(closes, net.index)
    bull_day = states[states == "bull"].index[0]
    net.loc[bull_day] = 0.10
    result = dg.market_states(net, closes)["direction"]
    assert result["bull"]["total_return"] == pytest.approx(0.10)
    assert result["flat"]["total_return"] == 0.0 and result["bear"]["total_return"] == 0.0
    assert result["bull"]["share_positive"] == pytest.approx(1 / result["bull"]["days"])


def test_without_btc_closes_the_state_split_is_absent() -> None:
    assert dg.market_states(series([0.01] * 40), None) is None


# ---- all together -----------------------------------------------------------------------------


def test_the_readings_are_built_stored_and_worded(tmp_path) -> None:
    net = series(np.random.default_rng(5).normal(0.001, 0.02, 400), start="2023-06-01")
    hold = series(np.random.default_rng(6).normal(0.0005, 0.03, 400), start="2023-06-01")
    payload = dg.build("toy", net, hold, _closes(900), trial_id=12)
    assert payload["window"] == {"start": "2023-06-01", "end": "2024-07-04", "days": 400}
    assert set(payload["entry_days"]) == {"30", "60", "90"}
    assert set(payload["persistence"]) == {"week", "month", "quarter"}
    dg.save(payload, tmp_path)
    assert dg.load("toy", tmp_path)["trial_id"] == 12 and dg.load("none", tmp_path) is None
    text = "\n".join(dg.lines_ru(payload))
    assert "ничего не решает" in text and "Войти в любой день и держать 30 дн." in text
    assert "просто держать те же монеты" in text
    for word in ("long", "short", "bull", "bear", "flat", "drawdown", "Sharpe"):
        assert word not in text                    # plain words, no trading jargon


def test_nothing_to_describe_is_refused() -> None:
    with pytest.raises(ValueError, match="no judged days"):
        dg.build("toy", series([0.01]), None, None)
