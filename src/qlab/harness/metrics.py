"""Metrics computed from a `RunResult` (and, for `min_capital_usd`, from a
strategy's own weights).

Names match what `rules/2026-09-20.1.yaml` references: `ann_return_net`,
`sharpe_net`, `max_dd`, `turnover` (see `docs/REGISTRY.md`'s `trial.metrics`
field). Annualisation is derived from the ACTUAL median spacing of the
panel's timestamps (never a hardcoded 365/365.25) — see `periods_per_year`.

`fragility_days`/`fragility_share`/`ann_return_net_ex_best_1pct` (T27, gap
1) measure how much of a run's result is carried by a handful of best days
-- no existing metric here saw a run whose whole result was ten days out of
627 (docs/XSMOM_T21.md, "Ревизия"). No threshold on them lives in this
module or in any `rules/*.yaml` file: measuring and gating are different
jobs (CLAUDE.md, "Никогда не выдумывать числа и пороги"; docs/T27_CONCENTRATION.md
has the measurement and a proposal left for the owner to decide).
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


def fragility_days(net_return: pd.Series) -> int:
    """Smallest number of best days whose removal brings the cumulative net
    return to zero or below.

    Primary concentration metric (docs/T27_CONCENTRATION.md, T27 gap 1)
    because it is PARAMETER-FREE by construction: there is no fraction of
    days to choose, so there is nothing here to tune. Definition: sort
    daily net returns descending, remove them one at a time (best first),
    and report the count at which the remaining ARITHMETIC SUM first stops
    being positive (`<= 0`) -- the same arithmetic-sum convention
    `docs/XSMOM_T21.md`'s "Ревизия" section used to find this failure mode
    (+0.564 total, +0.226 without the best 5 days, -0.009 without the best
    10), not the compounded/geometric total `ann_return_net` uses: a day's
    contribution to "how much of this result is these ten days" is read off
    the plain sum, because compounding would let the removal order itself
    change how much earlier days appear to contribute.

    `0` if the total is already `<= 0` -- nothing to remove; the run is not
    "propped up" by any days, it is simply unprofitable outright.

    Since the total is finite, removing every day drives the remaining sum
    to exactly `total - total == 0` (up to floating-point rounding), so for
    a strictly positive total the count is always found at or before `n`.
    The `RuntimeError` below instead of a sentinel return is defensive
    only: this branch is provably unreachable for a finite positive total,
    so hitting it would mean a bug in this function, not in the data.

    Args:
        net_return: the per-period net return series (`RunResult.net_return`,
            or an equivalent hand-built series for testing).

    Raises:
        RuntimeError: removing every day never brought the cumulative sum to
            `<= 0` -- impossible for a finite positive total; if this is
            ever raised, the bug is here, not in the input.
    """
    values = np.asarray(net_return, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return 0

    sorted_desc = np.sort(values)[::-1]
    remaining = total - np.cumsum(sorted_desc)
    hits = np.flatnonzero(remaining <= 0.0)
    if hits.size == 0:
        raise RuntimeError(
            "removing every day did not bring the cumulative net return to <= 0; "
            "this is impossible for a finite positive total and indicates a bug "
            "in fragility_days, not in the data"
        )
    return int(hits[0]) + 1


def fragility_share(net_return: pd.Series) -> float:
    """`fragility_days(net_return) / len(net_return)` -- the comparable form
    across runs of different length: ten fragile days mean something
    different over 627 bars than over 60 (docs/TASKS.md T27, gap 1).

    Raises:
        ValueError: `net_return` is empty (division by zero periods).
    """
    n = len(net_return)
    if n == 0:
        raise ValueError("cannot compute fragility_share from an empty net_return series")
    return fragility_days(net_return) / n


def effective_profitable_days(net_return: pd.Series) -> float:
    """How many days' worth of profit the run really had, as a number that
    does not move when the whole book is scaled up or down.

        effective = (sum of positive returns)^2 / (sum of squared positive returns)

    The inverse Herfindahl of the positive daily contributions, also called a
    participation ratio. Parameter-free, like `fragility_days`, but unlike it
    also SCALE-FREE: doubling every return leaves this unchanged, because both
    numerator and denominator scale quadratically. A run whose profit is spread
    evenly over `m` days scores exactly `m`; a run whose profit is one day
    scores exactly 1.

    **Why this exists next to `fragility_days`, which came first.** Measured on
    the 50 noise books matched to the admitted XSMOM run (docs/T27_CONCENTRATION.md),
    `fragility_days` has a rank correlation of **+0.90 with the run's own return**
    (+0.97 among the profitable ones): it mostly reports how much profit there
    was to remove, not how concentrated that profit is. A bigger result needs
    more days removed to cancel it, whatever its shape. `effective_profitable_days`
    on the same books correlates **-0.12** with return, so it measures the thing
    its name claims.

    This is the third time in this project that a plausible-looking measure
    turned out to be a proxy for something else -- the concentration sweep that
    was a volatility sort, the survivorship comparison that was a book-width
    comparison, and this. A rule must be built on this metric, not on
    `fragility_days`; `fragility_days` stays because "ten days out of 627 carry
    the whole result" is the sentence a human understands, and it is true, it
    just cannot be compared across runs of different size.

    Returns 0.0 when no day was profitable -- there is no profit to concentrate.
    """
    positive = net_return[net_return > 0].to_numpy(dtype=float)
    if positive.size == 0:
        return 0.0
    return float(positive.sum() ** 2 / (positive**2).sum())


def effective_profitable_days_share(net_return: pd.Series) -> float:
    """`effective_profitable_days / len(net_return)` -- the form comparable
    across runs of different length and different bar size.

    Raises:
        ValueError: `net_return` is empty.
    """
    n = len(net_return)
    if n == 0:
        raise ValueError(
            "cannot compute effective_profitable_days_share from an empty net_return series"
        )
    return effective_profitable_days(net_return) / n


def ann_return_net_ex_best_1pct(net_return: pd.Series) -> float:
    """Annualised net return recomputed with the best 1% of days removed.

    NOT parameter-free -- unlike `fragility_days`, this bakes in a specific
    fraction (1%) chosen because it is the form most readers expect
    ("returns without the best N% of days"), not because 1% is derived from
    anything in this particular run. `fragility_days` is the primary
    concentration metric for exactly this reason: there is no fraction here
    to defend, and this metric is reported alongside it, not instead of it.

    Rounding: `n_best = round(len(net_return) * 0.01)` (Python's
    round-half-to-even). For the admitted XSMOM control
    (`specs/xsmom-honest-20legs.yaml`, trial_id 1248, n=627) this is
    `round(6.27) == 6`, matching the 6 days used in `docs/XSMOM_T21.md`'s
    revision -- floor would give the same answer here, but round is used as
    the more natural reading of "1% of the periods"; the rare exact-`.5`
    tie is broken no more arbitrarily than any other convention would.

    WARNING, verified against the run this metric exists to describe: for
    the admitted XSMOM control the best 1% is 6 days, and the remaining 621
    days still compound to a POSITIVE annualised return. A rule built on
    THIS metric alone would NOT have caught the case that motivated T27 --
    `fragility_days` (== 10 for that same run) is what catches it.

    The `n_best` best days (by `net_return` value; ties broken by pandas'
    `nlargest`, i.e. by original position) are dropped from the series
    entirely -- not zeroed -- and the remaining periods are compounded and
    annualised exactly like `ann_return_net`: the SAME annualisation factor
    (`periods_per_year` of the full, un-truncated index -- the trading
    cadence does not change because a few good days are excluded) but the
    REDUCED period count as the compounding exponent's base, matching
    `ann_return_net`'s own `total_growth ** (ppy / n) - 1` where `n` is
    "however many periods contributed to `total_growth`".

    `NaN` if fewer than 2 periods (annualisation undefined, same as
    `ann_return_net`). `-1.0` if the remaining total growth is `<= 0` (same
    wipeout convention as `ann_return_net`).
    """
    n = len(net_return)
    if n < 2:
        return float("nan")

    ppy = periods_per_year(net_return.index)
    n_best = round(n * 0.01)
    remaining = net_return.drop(net_return.nlargest(n_best).index) if n_best > 0 else net_return
    n_remaining = len(remaining)
    if n_remaining == 0:
        return float("nan")

    total_growth = float((1.0 + remaining).prod())
    return -1.0 if total_growth <= 0.0 else total_growth ** (ppy / n_remaining) - 1.0


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
        fragility_days: see `fragility_days`'s own docstring -- the
            parameter-free concentration metric (docs/TASKS.md T27, gap 1).
        fragility_share: `fragility_days / n_periods`, see `fragility_share`.
        effective_profitable_days: the scale-free concentration measure, and
            the one any RULE must use -- see `effective_profitable_days` for
            why `fragility_days` cannot serve that purpose.
        effective_profitable_days_share: the same divided by `n_periods`.
        ann_return_net_ex_best_1pct: see `ann_return_net_ex_best_1pct`'s own
            docstring -- NOT parameter-free, and verified NOT sufficient on
            its own to catch the concentrated-result case T27 exists for;
            reported for readers who expect this form, `fragility_days`
            remains the primary metric.

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

    fdays = fragility_days(r)

    return {
        "ann_return_net": float(ann_return_net),
        "sharpe_net": float(sharpe_net),
        "max_dd": max_dd,
        "turnover": turnover_ann,
        "skew": skew,
        "worst_week": worst_week,
        "n_periods": n,
        "fragility_days": fdays,
        "fragility_share": fdays / n,
        "ann_return_net_ex_best_1pct": ann_return_net_ex_best_1pct(r),
        "effective_profitable_days": effective_profitable_days(r),
        "effective_profitable_days_share": effective_profitable_days_share(r),
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
