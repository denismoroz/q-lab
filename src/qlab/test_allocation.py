"""Tests for `qlab.allocation`."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.allocation import Leg, summary, switch


def _leg(name: str, ret: float, gross: float, days: pd.DatetimeIndex) -> Leg:
    return Leg(name=name, daily_return=pd.Series(ret, index=days),
               daily_gross=pd.Series(gross, index=days), cost_rate=0.001,
               first_active=days[0])


def test_capital_follows_the_regime_known_at_the_start_of_the_day() -> None:
    days = pd.date_range("2025-01-02", periods=4, freq="1D", tz="UTC")
    history = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    legs = {"trend": _leg("trend", 0.01, 1.0, history), "bv2": _leg("bv2", 0.002, 0.5, history)}
    # labels by close time: the label at 01-02 00:00 is known at the start of 01-02
    labels = pd.Series(["bull", "flat", "flat", "bear"],
                       index=pd.date_range("2025-01-02", periods=4, freq="1D", tz="UTC"))
    out, held = switch(legs, {"bull": "trend", "flat": "bv2", "bear": "trend"}, labels, days)
    assert held.tolist() == ["trend", "bv2", "bv2", "trend"]
    # day 1: entering trend from cash costs its gross; day 2: trend out, bv2 in
    assert out.iloc[0] == pytest.approx(0.01 - 0.001 * 1.0)
    assert out.iloc[1] == pytest.approx(0.002 - 0.001 * 1.0 - 0.001 * 0.5)
    assert out.iloc[2] == pytest.approx(0.002)


def test_cash_regime_earns_nothing() -> None:
    days = pd.date_range("2025-01-02", periods=2, freq="1D", tz="UTC")
    legs = {"trend": _leg("trend", 0.01, 1.0, days)}
    labels = pd.Series(["flat", "flat"], index=days)
    out, held = switch(legs, {"bull": "trend", "flat": None}, labels, days)
    assert out.tolist() == [0.0, 0.0] and held.isna().all()
    assert summary(pd.Series([0.01, -0.01, 0.02]))["days"] == 3
