"""Level 2 learns a strategy's outcome from the past only (docs/TASKS.md T39)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import meta_label as ml

DAYS = pd.date_range("2020-01-01", periods=900, freq="1D", tz="UTC")


def _inputs(seed: int = 5):
    rng = np.random.default_rng(seed)
    btc = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS)))), index=DAYS)
    p = rng.dirichlet([1, 1, 1], len(DAYS))
    level1 = pd.DataFrame(p, index=DAYS, columns=["p_bull", "p_flat", "p_bear"])
    returns = pd.Series(rng.normal(0.0003, 0.01, len(DAYS)), index=DAYS)
    return level1, btc, returns


def test_an_outcome_is_known_only_once_its_thirty_days_are_realised() -> None:
    _, _, returns = _inputs()
    as_of = DAYS[400]
    known = ml.known_at(returns, as_of)
    assert known.index.max() == as_of - pd.Timedelta(days=ml.HORIZON_DAYS)
    full = ml.outcome(returns)
    pd.testing.assert_series_equal(known, full.loc[known.index])


def test_features_use_only_returns_realised_by_the_close() -> None:
    level1, btc, returns = _inputs()
    cut = DAYS[500]
    changed = returns.copy()
    changed[changed.index >= cut] = 0.5  # a future that would be obvious to see
    a = ml.features(level1, btc, returns)
    b = ml.features(level1, btc, changed)
    pd.testing.assert_frame_equal(a[a.index <= cut], b[b.index <= cut])


def test_decisions_do_not_move_with_the_strategys_future() -> None:
    level1, btc, returns = _inputs()
    start = DAYS[200]
    base = ml.walk_forward(ml.features(level1, btc, returns), returns, start)
    cut = DAYS[600]
    changed = returns.copy()
    changed[changed.index >= cut] *= -3.0
    moved = ml.walk_forward(ml.features(level1, btc, changed), changed, start)
    pd.testing.assert_frame_equal(base[base.index < cut], moved[moved.index < cut])
    month = base.index.to_series().dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    first = month.where(month >= start, start)
    assert ((first - pd.to_datetime(base["trained_until"])).dt.days >= ml.EMBARGO_DAYS).all()


def test_the_gate_trades_only_where_level_two_says_so(monkeypatch) -> None:
    from qlab.harness.panel import MarketPanel
    from qlab.strategies.regime_gate import RegimeGate

    index = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
    stored = pd.DataFrame({"trade": [True, False, True, False]}, index=index)
    monkeypatch.setattr(ml, "load_decisions", lambda name: stored)
    panel = MarketPanel(snapshot_id="g", prices=pd.DataFrame(1.0, index=index, columns=["A"]),
                        funding=pd.DataFrame(0.0, index=index, columns=["A"]),
                        tradeable=pd.DataFrame(True, index=index, columns=["A"]), meta={})
    params = {"inner_code_ref": "qlab.strategies.test_regime_gate:Constant", "inner_params": {},
              "trade_in": ["trade"], "detector": "meta:x"}
    assert RegimeGate().target_weights(panel, params)["A"].tolist() == [0.5, 0.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="trade_in"):
        RegimeGate().target_weights(panel, {**params, "trade_in": ["bull"]})
