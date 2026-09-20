"""Trading cost model: taker fee + slippage, applied to turnover.

There is deliberately no default `CostModel` anywhere in this package, and
`CostModel` itself has no field defaults: `run_backtest` requires a
`CostModel` instance, so a run that "forgot" to pass costs is a `TypeError`,
not a silent free-to-trade simulation. Same principle as
`qlab.harness.accrual.NO_ACCRUAL` — a book that really has zero costs must
say so explicitly (`CostModel(taker_fee_bps=0.0, slippage_bps=0.0)`), not by
omission.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Cost per unit of turnover, in basis points, both legs required.

    `taker_fee_bps` and `slippage_bps` are summed and applied to turnover
    (per period, `sum(|Δweight|)` across instruments — see
    `qlab.harness.run.run_backtest`) at each period:

        cost[t] = turnover[t] * (taker_fee_bps + slippage_bps) / 1e4

    Turnover already counts the FULL absolute change of each instrument's
    weight (opening and closing are both folded into `|Δweight|`), matching
    `funding-rate-arbitrage/research/cross_sectional/xsec.py`'s convention.
    """

    taker_fee_bps: float
    slippage_bps: float

    def __post_init__(self) -> None:
        if self.taker_fee_bps < 0:
            raise ValueError(f"taker_fee_bps must be >= 0, got {self.taker_fee_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")

    @property
    def total_bps(self) -> float:
        """Total cost rate in basis points (fee + slippage)."""
        return self.taker_fee_bps + self.slippage_bps

    def cost_of_turnover(self, turnover: pd.Series) -> pd.Series:
        """Per-period cost series from a per-period turnover series."""
        return turnover * (self.total_bps / 1e4)
