"""Run a backtest: positions x prices x funding/borrow x costs -> `RunResult`.

Alignment (see `qlab.harness.strategy.Strategy`'s docstring for the contract
weights must satisfy): `weights.loc[t]` is decided using information up to
`t` and is held over the interval `(t, t+1]`. To pair it correctly with the
panel's "value for the period ENDING at this label" convention (both prices'
period return and funding are given that way — see
`qlab.harness.panel.MarketPanel`), this module shifts them one row backward
relative to their own index, so that at row `t` we have the return/funding
that will be realised between `t` and `t+1`:

    price_return_fwd = panel.prices.pct_change().shift(-1)
    funding_fwd      = panel.funding.shift(-1)

The output series (`net_return`, `gross_return`, `turnover`, `cost`,
`accrual`) are indexed by DECISION time `t`, i.e. `panel.prices.index[:-1]`
— the last timestamp is dropped because there is no realised `t -> t+1`
return to pair it with.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qlab.harness.accrual import NO_ACCRUAL, compute_accrual
from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights


@dataclass(frozen=True)
class RunResult:
    """Per-period contributions to a backtest's net return, kept separate
    so each one (gross return, cost, accrual) can be inspected on its own
    rather than taken on faith.

    Every series is indexed by decision time `t` (see module docstring);
    for every `t`:

        net_return.loc[t] == gross_return.loc[t] + accrual.loc[t] - cost.loc[t]
    """

    snapshot_id: str
    net_return: pd.Series
    gross_return: pd.Series
    turnover: pd.Series
    cost: pd.Series
    accrual: pd.Series


def run_backtest(
    panel: MarketPanel,
    weights: pd.DataFrame,
    costs: CostModel,
    accrual: pd.DataFrame | object,
) -> RunResult:
    """Simulate the book described by `weights` on `panel`.

    Args:
        panel: the market data.
        weights: target weights, same index/columns as `panel.prices`.
            Validated with `qlab.harness.strategy.validate_weights` before
            anything else runs.
        costs: required, no default — see `qlab.harness.costs.CostModel`.
        accrual: required, no default — `qlab.harness.accrual.NO_ACCRUAL`,
            or a `DataFrame` shaped like `panel.funding` (funding for the
            period ending at that label; `NaN` = unknown, same contract as
            `MarketPanel.funding`). This function does the forward-shift
            internally before handing it to `compute_accrual` — pass the
            RAW, `panel.funding`-shaped frame, not a pre-shifted one.

    Returns:
        A `RunResult` covering `panel.prices.index[:-1]`.

    Raises:
        qlab.harness.strategy.WeightValidationError: `weights` fails
            `validate_weights` against `panel`.
        ValueError: `panel` has fewer than 2 timestamps (no period over
            which any weight could realise a return).
        TypeError: `accrual` is omitted (`None`).
        qlab.harness.accrual.AccrualError: accrual is requested and `NaN`
            where a position is held.
    """
    if len(panel.prices.index) < 2:
        raise ValueError(
            "panel must have at least 2 timestamps to realise any return "
            f"(got {len(panel.prices.index)})"
        )
    validate_weights(panel, weights)

    # Drop the last timestamp up front: no realised t -> t+1 return exists
    # for it, so nothing below should look at it -- in particular, the
    # accrual NaN-while-held check must not fire on a row that will be
    # discarded anyway (its funding_fwd is unconditionally NaN, being the
    # tail of a `shift(-1)`).
    keep = panel.prices.index[:-1]

    price_return_fwd = panel.prices.pct_change().shift(-1).loc[keep]
    weights_kept = weights.loc[keep]

    prev_weights = weights.shift(1).fillna(0.0).loc[keep]
    turnover = (weights_kept - prev_weights).abs().sum(axis=1)
    cost = costs.cost_of_turnover(turnover)

    gross_return = (weights_kept * price_return_fwd).sum(axis=1)

    if accrual is None:
        raise TypeError(
            "run_backtest(accrual=...) is required. Pass a funding panel shaped "
            "like panel.funding, or qlab.harness.accrual.NO_ACCRUAL. `None` is "
            "not accepted -- see qlab.harness.accrual for why."
        )
    if accrual is NO_ACCRUAL:
        accrual_series = compute_accrual(weights_kept, NO_ACCRUAL)
    else:
        funding_fwd = accrual.reindex_like(panel.funding).shift(-1).loc[keep]
        accrual_series = compute_accrual(weights_kept, funding_fwd)

    net_return = gross_return + accrual_series - cost

    return RunResult(
        snapshot_id=panel.snapshot_id,
        net_return=net_return.rename("net_return"),
        gross_return=gross_return.rename("gross_return"),
        turnover=turnover.rename("turnover"),
        cost=cost.rename("cost"),
        accrual=accrual_series.rename("accrual"),
    )
