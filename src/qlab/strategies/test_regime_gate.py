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
    monkeypatch.setattr("qlab.regimes.market_closes", lambda: pd.Series(1.0, index=index))
    monkeypatch.setattr("qlab.regimes.causal_labels", lambda closes: labels)
    panel = MarketPanel(snapshot_id="g", prices=pd.DataFrame(1.0, index=index, columns=["A"]),
                        funding=pd.DataFrame(0.0, index=index, columns=["A"]),
                        tradeable=pd.DataFrame(True, index=index, columns=["A"]), meta={})
    params = {"inner_code_ref": "qlab.strategies.test_regime_gate:Constant", "inner_params": {},
              "trade_in": ["bull", "bear"]}
    w = RegimeGate().target_weights(panel, params)
    assert w["A"].tolist() == [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # unknown regime: no position
    with pytest.raises(ValueError, match="trade_in"):
        RegimeGate().target_weights(panel, {**params, "trade_in": ["sideways"]})
