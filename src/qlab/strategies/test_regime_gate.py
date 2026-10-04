"""Tests for `qlab.strategies.regime_gate`."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.regime_gate import RegimeGate


class Constant:
    name = "constant"

    def target_weights(self, panel, params):
        return pd.DataFrame(0.5, index=panel.prices.index, columns=panel.prices.columns)


def test_weights_are_kept_only_in_the_declared_regimes(monkeypatch) -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
    labels = pd.Series([None, "bull", "flat", "bear", "flat", "bull"], index=index, dtype=object)
    monkeypatch.setattr("qlab.strategies.detectors.market_closes",
                        lambda: pd.Series(1.0, index=index))
    monkeypatch.setattr("qlab.strategies.detectors.causal_labels", lambda closes: labels)
    panel = MarketPanel(snapshot_id="g", prices=pd.DataFrame(1.0, index=index, columns=["A"]),
                        funding=pd.DataFrame(0.0, index=index, columns=["A"]),
                        tradeable=pd.DataFrame(True, index=index, columns=["A"]), meta={})
    params = {"inner_code_ref": "qlab.strategies.test_regime_gate:Constant", "inner_params": {},
              "trade_in": ["bull", "bear"]}
    w = RegimeGate().target_weights(panel, params)
    assert w["A"].tolist() == [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # unknown regime: no position
    with pytest.raises(ValueError, match="trade_in"):
        RegimeGate().target_weights(panel, {**params, "trade_in": ["sideways"]})


def test_the_gate_can_read_the_walk_forward_detector(monkeypatch) -> None:
    """docs/TASKS.md T39: `detector: predictions:<name>` reads stored
    predictions instead of the trailing-month terciles."""
    import qlab.regime_detect as rd

    index = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
    stored = pd.DataFrame({"label": ["flat", "bull", "flat", "bear"]}, index=index)
    monkeypatch.setattr(rd, "load_predictions", lambda name: stored)
    panel = MarketPanel(snapshot_id="g", prices=pd.DataFrame(1.0, index=index, columns=["A"]),
                        funding=pd.DataFrame(0.0, index=index, columns=["A"]),
                        tradeable=pd.DataFrame(True, index=index, columns=["A"]), meta={})
    params = {"inner_code_ref": "qlab.strategies.test_regime_gate:Constant", "inner_params": {},
              "trade_in": ["bull", "bear"], "detector": "predictions:x"}
    assert RegimeGate().target_weights(panel, params)["A"].tolist() == [0.0, 0.5, 0.0, 0.5]
    with pytest.raises(ValueError, match="unknown detector"):
        RegimeGate().target_weights(panel, {**params, "detector": "magic"})
