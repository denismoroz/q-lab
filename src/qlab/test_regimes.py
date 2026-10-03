"""Tests for `qlab.regimes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.regimes import RegimeSeries, breakdown, label_days


def _closes(n: int = 400) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(2)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.03, n)), index=index)


def test_terciles_split_the_labelled_days_in_thirds() -> None:
    labels, bear_below, bull_above = label_days(_closes(), window=30)
    known = labels.dropna()
    assert labels.iloc[:30].isna().all()
    assert bear_below < bull_above
    for r in ("bull", "flat", "bear"):
        assert (known == r).mean() == pytest.approx(1 / 3, abs=0.01)


def test_breakdown_splits_returns_by_the_label_of_their_period() -> None:
    days = pd.date_range("2025-01-01", periods=7, freq="1D", tz="UTC")
    labels = pd.Series(["bull", "bull", "bull", "bear", "bear", "bear", "bear"], index=days,
                       dtype=object)
    regimes = RegimeSeries(labels=labels, market_return=pd.Series(0.01, index=days),
                           bear_below=-0.1, bull_above=0.1, source="test")
    # decision times 0..5; the return at t ends at t+1 and takes that day's label
    net = pd.Series([0.01, 0.01, -0.02, -0.02, -0.02, -0.02], index=days[:6])
    out = breakdown(net, regimes, periods_per_year=365)
    assert out["regime_bull_share"] == pytest.approx(2 / 6)
    assert out["regime_bear_share"] == pytest.approx(4 / 6)
    assert out["regime_bull_return"] == pytest.approx(1.01 ** 2 - 1)
    assert out["regime_bear_return"] == pytest.approx(0.98 * 0.98 ** 3 - 1)
    assert out["regime_flat_share"] == 0.0 and "regime_flat_return" not in out


def test_claims_are_checked_only_where_the_run_saw_the_regime() -> None:
    from qlab.regimes import claim_checks

    measured = {"regime_bull_days": 40.0, "regime_bull_return": 0.05,
                "regime_bear_days": 45.0, "regime_bear_return": -0.02,
                "regime_bear_btc_return": -0.20,
                "regime_flat_days": 10.0, "regime_flat_return": 0.01}
    out = claim_checks({"bull": "earns", "bear": "beats_market", "flat": "loses"}, measured)
    assert out == {"regime_bull_claim_met": 1.0, "regime_bear_claim_met": 1.0}  # flat: too few days
    assert claim_checks({"bear": "earns"}, measured) == {"regime_bear_claim_met": 0.0}


def test_causal_labels_never_see_the_future() -> None:
    from qlab.strategies.detectors import causal_labels

    closes = _closes(300)
    labels = causal_labels(closes, window=30)
    cut = 200
    tampered = closes.copy()
    tampered.iloc[cut + 1:] *= np.linspace(1.0, 4.0, len(closes) - cut - 1)
    again = causal_labels(tampered, window=30)
    pd.testing.assert_series_equal(labels.iloc[: cut + 1], again.iloc[: cut + 1])
    # A trailing return exists from day 30; a day needs 30 of them BEFORE it.
    assert labels.iloc[:60].isna().all()
    assert labels.iloc[60:].notna().all()
