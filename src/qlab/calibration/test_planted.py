"""Tests for `qlab.calibration.planted`: a planted edge of zero is admitted
about as often as noise, a large one almost always."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.calibration.planted import noise_books, planted_outcomes
from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.rules.schema import Comparator, Rule, RuleSet, Stage

RULES = RuleSet(version="2026-01-01.1", rules=[
    Rule(id="net_edge_positive", stage=Stage.EDGE, metric="ann_return_net",
         comparator=Comparator.GE, threshold=0.04, fatal=True),
    Rule(id="shape_aware_edge", stage=Stage.EDGE, metric="noise_return_percentile",
         comparator=Comparator.GE, threshold=0.99, fatal=True),
])


def _setup():
    rng = np.random.default_rng(1)
    index = pd.date_range("2025-01-01", periods=400, freq="1D", tz="UTC")
    cols = [f"C{i}" for i in range(20)]
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.03, (400, 20)), axis=0),
                          index=index, columns=cols)
    panel = MarketPanel(snapshot_id="planted", prices=prices,
                        funding=pd.DataFrame(0.0, index=index, columns=cols),
                        tradeable=pd.DataFrame(True, index=index, columns=cols), meta={})
    signs = np.sign(rng.normal(size=(400, 20)))
    reference = pd.DataFrame(np.repeat(signs[::10], 10, axis=0)[:400] * 0.05,
                             index=index, columns=cols)
    return panel, reference


def test_zero_edge_is_rarely_admitted_and_a_large_one_almost_always() -> None:
    panel, reference = _setup()
    books = noise_books(panel, reference, CostModel(taker_fee_bps=1.0, slippage_bps=1.0), 100)
    assert len(books) > 50
    out = planted_outcomes(
        panel=panel, reference=reference, books=books, ruleset=RULES, shared_facts={},
        alphas=[0.0, 3.0], window_starts={"full": panel.prices.index[0]},
        min_leg_notional=10.0, deployable_capital_usd=1e9,
    )["full"]
    zero, large = out
    assert zero.admitted / zero.n <= 0.05
    assert large.admitted / large.n >= 0.9
