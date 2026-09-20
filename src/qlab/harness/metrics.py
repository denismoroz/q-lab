"""Metrics computed from a `RunResult` (and, for `min_capital_usd`, from a
strategy's own weights).

Names match what `rules/2026-09-20.1.yaml` references: `ann_return_net`,
`sharpe_net`, `max_dd`, `turnover` (see `docs/REGISTRY.md`'s `trial.metrics`
field). Annualisation is derived from the ACTUAL median spacing of the
panel's timestamps (never a hardcoded 365/365.25) — see `periods_per_year`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.run import RunResult
from qlab.harness.strategy import ZERO_WEIGHT_TOL

_YEAR = pd.Timedelta(days=365.25)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer the annualisation factor from the median spacing between timestamps.

    Uses the median (not the mean) so a handful of irregular gaps — a
    missing day, an exchange outage — don't skew the whole run's
    annualisation.

    Raises:
        ValueError: fewer than 2 timestamps (no spacing to infer), or a
            non-positive median spacing (duplicate/unsorted index).
    """
    if len(index) < 2:
        raise ValueError("cannot infer periods_per_year from fewer than 2 timestamps")
    deltas = index.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta(0):
        raise ValueError(f"non-positive median spacing between timestamps: {median_delta}")
    return _YEAR / median_delta


def compute_metrics(result: RunResult) -> dict[str, float]:
    """Compute the metrics `rules/<version>.yaml` references, plus context.

    Returns a dict with:
        ann_return_net: geometric-annualised net return —
            `(1 + total_growth) ** (periods_per_year / n_periods) - 1`,
            where `total_growth = (1 + net_return).prod() - 1`. If
            `total_growth <= -1` (the book is wiped out or worse), this is
            `-1.0` rather than raising on a fractional power of a
            non-positive base.
        sharpe_net: `mean(net_return) / std(net_return, ddof=1) *
            sqrt(periods_per_year)`. `NaN` if fewer than 2 periods or the
            return series has zero variance (a flat return series has no
            well-defined Sharpe, not an infinite one).
        max_dd: the most negative drawdown of the net-return equity curve,
            as a fraction `<= 0` (e.g. `-0.35` for a 35% peak-to-trough
            loss). `0.0` if the equity curve never dips below its running
            peak.
        turnover: annualised mean per-period turnover —
            `mean(result.turnover) * periods_per_year`.
        skew: sample skew (Fisher-Pearson, bias-corrected — pandas'
            `Series.skew()`) of `net_return`. `NaN` with fewer than 3
            periods (undefined).
        worst_week: the worst compounded return over any single calendar
            week (`net_return` grouped by `pd.Grouper(freq="W")`, compounded
            within each week via `(1 + r).prod() - 1`, then the minimum
            across weeks).
        n_periods: `len(result.net_return)` — the number of periods this
            run actually produced a realised return for (already excludes
            the final, undecidable period `run_backtest` drops).

    Raises:
        ValueError: `result.net_return` is empty.
    """
    r = result.net_return
    n = len(r)
    if n == 0:
        raise ValueError("cannot compute metrics from an empty net_return series")

    total_growth = float((1.0 + r).prod())

    if n < 2:
        ppy = float("nan")
        ann_return_net = float("nan")
        sharpe_net = float("nan")
        turnover_ann = float("nan")
    else:
        ppy = periods_per_year(r.index)
        ann_return_net = -1.0 if total_growth <= 0.0 else total_growth ** (ppy / n) - 1.0
        std = r.std(ddof=1)
        mean = r.mean()
        sharpe_net = float("nan") if not std or pd.isna(std) else float(mean / std * np.sqrt(ppy))
        turnover_ann = float(result.turnover.mean() * ppy)

    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    skew = float(r.skew()) if n >= 3 else float("nan")

    weekly_growth = (1.0 + r).groupby(pd.Grouper(freq="W")).apply(lambda g: g.prod() - 1.0)
    worst_week = float(weekly_growth.min()) if len(weekly_growth) else float("nan")

    return {
        "ann_return_net": float(ann_return_net),
        "sharpe_net": float(sharpe_net),
        "max_dd": max_dd,
        "turnover": turnover_ann,
        "skew": skew,
        "worst_week": worst_week,
        "n_periods": n,
    }


def min_capital_usd(weights: pd.DataFrame, min_leg_notional: float) -> float:
    """Minimum book capital at which the strategy is executable AS SPECIFIED.

    Deliberately NOT a coverage curve and NOT a "runs from $X with degraded
    tracking" number. A strategy executed with its smallest legs dropped for
    being unaffordable is a DIFFERENT strategy with a different edge, not
    the same one for less money — this is a measured fact, not a style
    choice: in `funding-rate-arbitrage`, the full trend strategy needs
    roughly $2200 to hold every leg it wants; a version squeezed to run from
    roughly $410 by dropping small legs is "no longer better than XSMOM".
    Reporting "runs from $410" would misrepresent which strategy's numbers
    were actually measured. If a smaller, leg-dropping variant is worth
    running, it gets formalised as its OWN spec and goes through its own
    trial with its own numbers — this function does not suggest one, and
    `qlab.harness` deliberately has no "partial coverage" metric next to it.

    Computed from the smallest non-zero `|weight|` the strategy actually
    takes at ANY `(period, instrument)` over the whole `weights` frame — not
    from a nominal/target weighting scheme — so a strategy that is mostly
    large legs with one rare small leg is correctly sized by that rare small
    leg, not by its typical rebalance:

        min_capital_usd = min_leg_notional / min(|weight| for weight != 0)

    Args:
        weights: the full target-weights frame the strategy actually
            produced (e.g. `strategy.target_weights(panel, params)`), same
            shape as the panel. Every non-zero cell across every period is
            considered — a leg that appears in 1 rebalance out of 500 still
            sets the floor.
        min_leg_notional: the venue's minimum tradeable leg size in USD
            (e.g. ~$10 on HL, ~$12 in the live config). Required, no
            default, by the same principle as `CostModel` and
            `qlab.harness.accrual.NO_ACCRUAL`: a capital figure computed
            without knowing the venue's floor is not a real number, so
            omitting this argument is a `TypeError`, not a free pass.

    Raises:
        ValueError: `min_leg_notional` is not positive, or `weights` never
            takes a non-zero position anywhere (the floor is undefined —
            there is no smallest leg).
    """
    if min_leg_notional <= 0:
        raise ValueError(f"min_leg_notional must be > 0, got {min_leg_notional}")

    magnitudes = np.abs(weights.to_numpy(dtype=float))
    magnitudes = magnitudes[~np.isnan(magnitudes)]
    nonzero = magnitudes[magnitudes > ZERO_WEIGHT_TOL]
    if nonzero.size == 0:
        raise ValueError(
            "weights never take a non-zero position anywhere; min_capital_usd "
            "is undefined for an all-flat book"
        )
    smallest_weight = float(nonzero.min())
    return min_leg_notional / smallest_weight
