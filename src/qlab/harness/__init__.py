"""Backtest harness: positions x prices x funding/borrow x costs -> metrics.

See `docs/PLAN.md`, milestone M2. `qlab.harness.panel.MarketPanel` is a
structural stand-in for the shared data-layer contract
(`qlab.data.panel.MarketPanel`, built separately) — see that module's
docstring.
"""

from qlab.harness.accrual import NO_ACCRUAL, AccrualError, compute_accrual
from qlab.harness.costs import CostModel
from qlab.harness.metrics import compute_metrics, min_capital_usd, periods_per_year
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest
from qlab.harness.strategy import (
    ZERO_WEIGHT_TOL,
    Strategy,
    WeightValidationError,
    validate_weights,
)

__all__ = [
    "NO_ACCRUAL",
    "ZERO_WEIGHT_TOL",
    "AccrualError",
    "CostModel",
    "MarketPanel",
    "RunResult",
    "Strategy",
    "WeightValidationError",
    "compute_accrual",
    "compute_metrics",
    "min_capital_usd",
    "periods_per_year",
    "run_backtest",
    "validate_weights",
]
