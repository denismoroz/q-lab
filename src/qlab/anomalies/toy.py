"""A toy book with known answers, for the anomaly tests (and only for them).

Six instruments on 40 daily bars from 2026-01-01:

* `A`, `C` -- calm coins the book holds long, `B`, `D` -- calm coins it holds short;
* `PUMP` -- held long at 0.1, thin (200 thousand dollars a day) until bar 20, then it triples
  on the bar after bar 20 (the decision at bar 20 earns +200% of the weight: +0.2 of capital)
  and trades five million dollars a day afterwards;
* `DEAD` -- held long at 0.1, last price at bar 30, then no price and not tradeable.

`A` has no volume figure on bars 5 and 6 (unknown, never zero).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from qlab.anomalies.rerun import StoredRun
from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel

BARS = 40
PUMP_BAR = 20
DEAD_LAST_BAR = 30
INSTRUMENTS = ["A", "B", "C", "D", "PUMP", "DEAD"]
WEIGHTS = {"A": 0.1, "B": -0.1, "C": 0.1, "D": -0.1, "PUMP": 0.1, "DEAD": 0.1}


def toy_panel(bars: int = BARS) -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=bars, freq="D", tz="UTC")
    n = np.arange(bars, dtype=float)
    prices = pd.DataFrame({
        "A": 100.0 * 1.002 ** n, "B": 50.0 * 1.001 ** n, "C": 20.0 * 1.0015 ** n,
        "D": 30.0 * 0.999 ** n, "PUMP": np.where(n <= PUMP_BAR, 10.0, 30.0),
        "DEAD": np.where(n <= DEAD_LAST_BAR, 5.0, np.nan)}, index=index)
    funding = pd.DataFrame(0.0001, index=index, columns=INSTRUMENTS)
    funding["DEAD"] = funding["DEAD"].where(n <= DEAD_LAST_BAR + 1)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)
    tradeable["DEAD"] = n <= DEAD_LAST_BAR
    usd = pd.DataFrame({"A": 2e6, "B": 5e6, "C": 3e6, "D": 4e6,
                        "PUMP": np.where(n <= PUMP_BAR, 2e5, 5e6), "DEAD": 5e5}, index=index)
    volume = usd / prices
    volume.loc[index[5:7], "A"] = np.nan
    return MarketPanel(snapshot_id="toy", prices=prices, funding=funding, tradeable=tradeable,
                       volume=volume, meta={"no_funding_instruments": ()})


def toy_weights(panel: MarketPanel) -> pd.DataFrame:
    weights = pd.DataFrame({c: WEIGHTS[c] for c in INSTRUMENTS}, index=panel.prices.index)
    return weights.where(panel.tradeable, 0.0)


class ToyStrategy:
    """The toy book as a strategy: its fixed weights on every instrument that is tradeable."""

    name = "toy"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        return toy_weights(panel)


def toy_run(trial_id: int = 1, idea_id: str = "toy-idea", venue: str = "binance") -> StoredRun:
    panel = toy_panel()
    spec = SimpleNamespace(idea_id=idea_id, code_ref="qlab.anomalies.toy:ToyStrategy",
                           params={}, data=SimpleNamespace(source=venue, interval="1d"))
    return StoredRun(trial_id=trial_id, idea_id=idea_id, spec=spec, panel=panel,
                     weights=toy_weights(panel), judged=(0, len(panel.prices) - 1),
                     costs=CostModel(taker_fee_bps=5.0, slippage_bps=5.0), extra_cost=None,
                     route="paper")
