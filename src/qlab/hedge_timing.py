"""Does a spot-plus-hedge book choose WHEN to hedge better than chance?
(docs/HEDGE_TIMING.md)

Owner, 2026-10-24, reading the assessor on strategy-b-v2: «стратегия не
умеет определять моменты когда нужно включать хеджирование»; on the
proposal to test exactly that -- «делай все три по порядку».

The generic noise books cannot answer it: they are dollar-neutral (half long,
half short by construction), so they never lose with the market, and a book
that holds spot always looks worse than them when the market falls. Here the
comparison keeps everything but the timing:

- the spot leg is the strategy's own, untouched;
- each coin's hedge (its perp weight) is the strategy's own series shifted in
  time by a random offset, circularly within the judged window. The share of
  time hedged, the number of switches and the length of every hedge episode
  stay exactly the strategy's; only WHEN they happen changes.

If the strategy beats nearly all shifted books, its moments are better than
random moments with the same amount of hedging. Two plain books are shown
beside it: the spot alone (never hedged) and the spot always hedged.

A measurement, not a rule: no threshold here decides anything (CLAUDE.md:
verdicts and their thresholds are the owner's).

The returns are on the strategy's own money when the caller passes its `CapitalSetup`
(docs/CAPITAL_BASE.md, owner 2026-10-29): every book of the test is read on the same base (the
strategy's own, fixed by the book as it was), so the two plain alternatives shown beside the
strategy are on the same footing as the strategy's own yearly figure.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.calibration.percentile import noise_percentile
from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.costs import CostModel
from qlab.harness.metrics import capital_metrics, compute_metrics
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest


@dataclass(frozen=True)
class HedgeTimingResult:
    days: float
    own: float
    shifted: list[float]
    spot_only: float
    always_hedged: float
    hedged_share: float

    @property
    def percentile(self) -> float:
        """Share of the shifted books the strategy beats."""
        return noise_percentile(self.own, self.shifted)


def hedge_pairs(weights: pd.DataFrame) -> dict[str, str]:
    """{perp column: its spot column} for every coin the book holds both of."""
    return {c: f"{c}{SPOT_COLUMN_SUFFIX}" for c in weights.columns
            if not c.endswith(SPOT_COLUMN_SUFFIX)
            and f"{c}{SPOT_COLUMN_SUFFIX}" in weights.columns
            and weights[c].abs().sum() > 0 and weights[f"{c}{SPOT_COLUMN_SUFFIX}"].abs().sum() > 0}


def shifted_hedges(weights: pd.DataFrame, start: pd.Timestamp, seed: int) -> pd.DataFrame:
    """The book with each coin's hedge rolled by its own random offset inside
    [start, end]; the spot and everything before `start` untouched."""
    rng = np.random.default_rng(seed)
    out = weights.copy()
    inside = weights.index >= start
    n = int(inside.sum())
    for perp in hedge_pairs(weights):
        k = int(rng.integers(1, n)) if n > 1 else 0
        out.loc[inside, perp] = np.roll(weights.loc[inside, perp].to_numpy(), k)
    return out


def _ann(result: RunResult, start: pd.Timestamp, on_capital: bool = False) -> float:
    keep = result.net_return.index >= start
    part = dataclasses.replace(
        result, net_return=result.net_return[keep], gross_return=result.gross_return[keep],
        turnover=result.turnover[keep], cost=result.cost[keep], accrual=result.accrual[keep],
        capital_return=result.capital_return[keep])
    if on_capital:
        return float(capital_metrics(part)["cap_ann_return"])
    return float(compute_metrics(part)["ann_return_net"])


def hedge_timing_test(panel: MarketPanel, weights: pd.DataFrame, costs: CostModel,
                      start: pd.Timestamp, n: int = 200, seed_offset: int = 0,
                      capital=None) -> HedgeTimingResult:
    """The strategy against `n` books with its hedges shifted in time, over
    [start, the panel's end]. With `capital` (a `qlab.harness.strategy.CapitalSetup` whose base
    is defined) every figure is the yearly return on the strategy's own money."""
    kwargs = capital.kwargs if capital is not None and capital.refused is None else {}
    pairs = hedge_pairs(weights)
    if not pairs:
        raise ValueError("the book holds no coin with both a spot and a perp leg")

    # A weight may sit only where the panel trades it, or where the book's
    # own weight already sat (a gap it holds through, qlab.harness.gaps).
    allowed = (panel.tradeable.reindex(index=weights.index, columns=weights.columns,
                                       fill_value=False).to_numpy(dtype=bool)
               | (weights.to_numpy() != 0))

    def run(w: pd.DataFrame) -> float:
        return _ann(run_backtest(panel, w.where(allowed, 0.0), costs, panel.funding, **kwargs),
                    start, on_capital=bool(kwargs))

    spot_only = weights.copy()
    always = weights.copy()
    for perp, spot in pairs.items():
        spot_only[perp] = 0.0
        always[perp] = -weights[spot]
    inside = weights.index >= start
    hedged = np.mean([(weights.loc[inside, p].abs() > 0).mean() for p in pairs])
    days = (weights.index[-1] - start) / pd.Timedelta(days=1)
    return HedgeTimingResult(
        days=float(days), own=run(weights),
        shifted=[run(shifted_hedges(weights, start, seed_offset + i)) for i in range(n)],
        spot_only=run(spot_only), always_hedged=run(always), hedged_share=float(hedged))


__all__ = ["HedgeTimingResult", "hedge_pairs", "hedge_timing_test", "shifted_hedges"]
