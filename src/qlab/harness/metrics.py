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

`year_breakdown` (owner, 2026-10-27) shows the same failure from the calendar side:
the result year by year, and whether it rests on one lucky year. It is informational
for the same reason (rules 2026-10-27.2, docs/YEARS.md).
"""

from __future__ import annotations

from collections.abc import Mapping

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


def ann_return_net_ex_best_1pct(net_return: pd.Series, *, additive: bool = False) -> float:
    """Annualised net return recomputed with the best 1% of days removed.

    `additive` (docs/CAPITAL_BASE.md): the bars are shares of a constant capital, so days
    are summed instead of compounded and the annual figure is the sum scaled by
    `periods_per_year / n_remaining`, never a power. The default is the stand's own
    compounding.

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
    if len(net_return) >= 2 and periods_per_year(net_return.index) > 366.0 * 1.01:
        # Intraday bars are compounded into days first: "the best 1% of
        # days", as above. On hourly bars the best 1% of HOURS -- 41 of
        # 4,081 -- is where any crypto holding earns: BTC held 2026-05..10
        # read +1.2% a year and -81% without them (fixed 2026-10-24).
        net_return = (net_return.resample("1D").sum() if additive
                      else (1.0 + net_return).resample("1D").prod() - 1.0)
    n = len(net_return)
    if n < 2:
        return float("nan")

    ppy = periods_per_year(net_return.index)
    n_best = round(n * 0.01)
    remaining = net_return.drop(net_return.nlargest(n_best).index) if n_best > 0 else net_return
    n_remaining = len(remaining)
    if n_remaining == 0:
        return float("nan")

    if additive:
        return float(remaining.sum() * ppy / n_remaining)
    total_growth = float((1.0 + remaining).prod())
    return -1.0 if total_growth <= 0.0 else total_growth ** (ppy / n_remaining) - 1.0


COUNTED_YEAR_DAYS = 120
"""A calendar year with at least this many days in the judged part is counted in the
year summaries (owner, 2026-10-27: «давай вот эту проверку с разбивкой добавим в
фреймворк»). A shorter stretch is still listed year by year but is too short to be
called "a year that was good or bad"."""
SHARPE_YEAR_DAYS = 60
"""A year's own Sharpe is reported from this many days on; below it the ratio of a
few weeks' mean to their spread says nothing."""


def year_breakdown(net_return: pd.Series, *, additive: bool = False) -> dict[str, float]:
    """The judged part's result year by year, and three summaries of how evenly the
    years share it. Flat numeric keys, because `trial.metrics` is a flat mapping.

    `additive` (docs/CAPITAL_BASE.md): the bars are shares of the strategy's constant
    capital and profit is not reinvested, so a year's return is the SUM of its bars, its
    fall is measured in points of the capital from the year's running peak (the year
    starts at 0), and "without the best year" sums what remains and scales it by
    `periods_per_year / bars` instead of raising a growth to a power. The default is the
    stand's own compounding; the keys are the same either way.

    Per calendar year `Y` (UTC) present in the series:
        year_Y_return: the year's compounded net return, `(1 + r).prod() - 1`. NOT
            annualised: a partial first or last year shows what it really earned.
        year_Y_days: calendar days with a return in that year (a 282-day 2026 says so).
        year_Y_max_dd: the deepest peak-to-trough fall of the equity curve inside the
            year, which starts from 1.0 on the year's first bar; `<= 0`.
        year_Y_sharpe: only with `SHARPE_YEAR_DAYS` days or more, annualised with the
            whole series' `periods_per_year`.

    Summaries, over the COUNTED years (`COUNTED_YEAR_DAYS` days or more):
        years_counted, years_positive: how many years count and how many of them
            ended above zero.
        years_positive_share: their ratio.
        ann_return_net_ex_best_year: `ann_return_net` recomputed with every bar of
            the best counted year (by `year_Y_return`) removed -- the same convention
            as `ann_return_net_ex_best_1pct`: the full series' annualisation factor,
            the reduced bar count as the compounding base. `-1.0` if what remains
            wipes the book out. Absent with fewer than two counted years: removing
            the only year leaves nothing to annualise.
        best_year_share_of_profit: the best counted year's profit divided by the
            whole series' profit, both as plain sums of the bars' net returns (the
            arithmetic convention of `fragility_days`, which keeps the shares
            additive). Above 1 when the other years together lost money. Absent
            when the total is not positive or with fewer than two counted years.

    A key whose value is undefined is left out rather than set to NaN (the rules
    engine reads an absent metric as unknown). Fewer than two bars: empty dict.
    """
    if len(net_return) < 2:
        return {}
    ppy = periods_per_year(net_return.index)
    out: dict[str, float] = {}
    counted: dict[int, float] = {}
    for year, bars in net_return.groupby(net_return.index.year):
        year = int(year)
        days = int(bars.index.normalize().nunique())
        if additive:
            curve = bars.cumsum()
            out[f"year_{year}_return"] = float(curve.iloc[-1])
            out[f"year_{year}_max_dd"] = float((curve - curve.cummax().clip(lower=0.0)).min())
        else:
            equity = (1.0 + bars).cumprod()
            out[f"year_{year}_return"] = float(equity.iloc[-1] - 1.0)
            out[f"year_{year}_max_dd"] = float(
                (equity / equity.cummax().clip(lower=1.0) - 1.0).min())
        out[f"year_{year}_days"] = days
        std = bars.std(ddof=1)
        if days >= SHARPE_YEAR_DAYS and len(bars) >= 2 and std and not pd.isna(std):
            out[f"year_{year}_sharpe"] = float(bars.mean() / std * np.sqrt(ppy))
        if days >= COUNTED_YEAR_DAYS:
            counted[year] = out[f"year_{year}_return"]
    if not counted:
        return out
    positive = sum(1 for value in counted.values() if value > 0)
    out["years_counted"] = len(counted)
    out["years_positive"] = positive
    out["years_positive_share"] = positive / len(counted)
    if len(counted) < 2:
        return out

    best = max(counted, key=lambda year: counted[year])
    rest = net_return[net_return.index.year != best]
    if additive:
        out["ann_return_net_ex_best_year"] = float(rest.sum() * ppy / len(rest))
    else:
        growth = float((1.0 + rest).prod())
        out["ann_return_net_ex_best_year"] = (
            -1.0 if growth <= 0.0 else growth ** (ppy / len(rest)) - 1.0)
    total = float(net_return.sum())
    if total > 0.0:
        out["best_year_share_of_profit"] = float(
            net_return[net_return.index.year == best].sum()) / total
    return out


def held_coins(weights: pd.DataFrame) -> list[str]:
    """The coins a book holds at any time, a coin traded as spot and perp counted once,
    as spot: what "just hold the same coins" means."""
    from qlab.data.sources.base import SPOT_COLUMN_SUFFIX

    held = [c for c in weights.columns if weights[c].abs().sum() > 0]
    return [c for c in held if c.endswith(SPOT_COLUMN_SUFFIX)
            or f"{c}{SPOT_COLUMN_SUFFIX}" not in held]


def hold_benchmark(prices: pd.DataFrame, weights: pd.DataFrame) -> dict[str, float]:
    """Holding, long and equally weighted, the coins the book holds -- what a
    reader compares a strategy with before anything else (owner, 2026-10-24:
    the assessor read "without its best 1%" as luck with nothing to compare
    it to). A coin traded as spot and perp counts once, as spot. No costs, no
    funding; rebalanced every bar. Empty when the book holds nothing."""
    coins = held_coins(weights)
    if not coins:
        return {}
    r = prices[coins].pct_change().iloc[1:].fillna(0.0).mean(axis=1)
    if len(r) < 2:
        return {}
    growth = float((1.0 + r).prod())
    ppy = periods_per_year(prices.index)
    return {
        "hold_ann_return": -1.0 if growth <= 0 else growth ** (ppy / len(r)) - 1.0,
        "hold_ann_return_ex_best_1pct": ann_return_net_ex_best_1pct(r),
    }


def hold_returns(prices: pd.DataFrame, weights: pd.DataFrame, index: pd.DatetimeIndex
                 ) -> pd.Series | None:
    """The per-bar returns of the same hold-the-coins book, on the harness's own clock:
    the value at decision time `t` is the return realised over `(t, t+1]`, exactly like
    `RunResult.net_return`, so the two series can be laid side by side. None when the book
    holds nothing. Display only -- no rule reads it."""
    coins = held_coins(weights)
    if not coins:
        return None
    forward = prices[coins].pct_change().shift(-1).fillna(0.0).mean(axis=1)
    return forward.reindex(index).fillna(0.0).rename("hold_return")


def compute_metrics(result: RunResult) -> dict[str, float]:
    """Compute the metrics `rules/<version>.yaml` references, plus context.

    Returns a dict with:
        ann_return_net: geometric-annualised net return —
            `(1 + total_growth) ** (periods_per_year / n_periods) - 1`,
            where `total_growth = (1 + net_return).prod() - 1`. If
            `total_growth <= -1` (the book is wiped out or worse), this is
            `-1.0` rather than raising on a fractional power of a
            non-positive base.
        total_return_net: `(1 + net_return).prod() - 1`, the compounded net
            return over the whole window (not annualised). Read by the stage 2
            check against stage 1's same-length segments (docs/TWO_STAGE.md).
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
        "total_return_net": total_growth - 1.0,
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


# ---- the same summaries on the strategy's own money (docs/CAPITAL_BASE.md) -----------------

#: A stand key whose capital twin is not simply `cap_<key>`.
_CAP_RENAMED = {
    "ann_return_net": "cap_ann_return",
    "total_return_net": "cap_total_return",
    "sharpe_net": "cap_sharpe",
    "max_dd": "cap_max_dd",
    "ann_return_net_ex_best_1pct": "cap_ann_return_ex_best_1pct",
    "ann_return_net_ex_best_year": "cap_ann_return_ex_best_year",
    "best_year_share_of_profit": "cap_best_year_share_of_profit",
}
CAP_STAND_KEYS = {v: k for k, v in _CAP_RENAMED.items()}
"""The stand key each non-year `cap_` key is the capital twin of."""
CAP_REINVESTED = "cap_reinvested"
"""1.0 when the run's capital summaries compound (the book reinvests its profit), 0.0 when they
sum (it does not)."""


def cap_key(stand_key: str) -> str:
    """The capital twin of a stand metric key: `ann_return_net` -> `cap_ann_return`,
    `year_2025_return` -> `cap_year_2025_return`, `years_positive_share` ->
    `cap_years_positive_share`."""
    if stand_key in _CAP_RENAMED:
        return _CAP_RENAMED[stand_key]
    return f"cap_{stand_key}"


def stand_key_of(key: str) -> str | None:
    """The stand key a `cap_` key is the twin of; None for a key that is not one."""
    if key in CAP_STAND_KEYS:
        return CAP_STAND_KEYS[key]
    if key.startswith("cap_year_") or key.startswith("cap_years_"):
        return key[len("cap_"):]
    return None


def as_stand_keys(metrics: Mapping[str, object] | None) -> dict[str, object]:
    """The `cap_` entries of `metrics` under the stand's own key names: `cap_year_2025_return`
    becomes `year_2025_return`, `cap_ann_return` becomes `ann_return_net`. What the year view
    and the four year signs read, so they can be asked of the capital summaries with the code
    that reads the stand's. Entries that are not capital twins are left out."""
    out: dict[str, object] = {}
    for key, value in (metrics or {}).items():
        twin = stand_key_of(key)
        if twin is not None:
            out[twin] = value
    return out


def capital_metrics(result: RunResult) -> dict[str, float]:
    """The run's summaries on the strategy's own money, `cap_` keys beside the stand's.

    The series is `result.capital_return`: each bar's dollars over the capital the strategy
    started with (the stand's share times the strategy's `own_base`; just the stand's share
    for a book whose weights already are shares of that capital). Whether profit is
    reinvested is the strategy's declared property (`result.capital_reinvested`):

      * reinvested (a trend book resizes its positions to all its money): compounded, exactly
        the stand's formulas, so on a base of 1 every `cap_` figure equals its stand twin;
      * not reinvested (Bv2 sets cash aside, FRAB and spot run on a constant budget, XSMOM
        live runs on a fixed budget): bars are summed, never compounded. The total is the sum,
        "per year" is the sum scaled by `periods_per_year / n`, the deepest fall is the fall
        of the running profit from its peak in points of the capital (the curve starts at 0,
        so the fall from the start counts), and the "without the best ..." summaries sum what
        remains.

    Keys: `cap_ann_return`, `cap_total_return`, `cap_sharpe`, `cap_max_dd`,
    `cap_ann_return_ex_best_1pct`, `cap_worst_week`, `cap_reinvested`, and the year view under
    `cap_year_*`,
    `cap_years_*`, `cap_ann_return_ex_best_year`, `cap_best_year_share_of_profit` (keys with
    an undefined value are left out, as in `year_breakdown`).

    Raises:
        ValueError: `result.capital_return` is empty.
    """
    r = result.capital_return
    n = len(r)
    if n == 0:
        raise ValueError("cannot compute capital metrics from an empty capital_return series")
    additive = not result.capital_reinvested
    out: dict[str, float] = {CAP_REINVESTED: 0.0 if additive else 1.0}
    growth = float((1.0 + r).prod())
    total = float(r.sum()) if additive else growth - 1.0
    out[cap_key("total_return_net")] = total
    if n < 2:
        out[cap_key("ann_return_net")] = float("nan")
        out[cap_key("sharpe_net")] = float("nan")
    else:
        ppy = periods_per_year(r.index)
        if additive:
            ann = total * ppy / n
        else:
            ann = -1.0 if growth <= 0.0 else growth ** (ppy / n) - 1.0
        std = r.std(ddof=1)
        out[cap_key("ann_return_net")] = float(ann)
        out[cap_key("sharpe_net")] = (
            float("nan") if not std or pd.isna(std) else float(r.mean() / std * np.sqrt(ppy)))
    if additive:
        curve = r.cumsum()
        out[cap_key("max_dd")] = float((curve - curve.cummax().clip(lower=0.0)).min())
    else:
        equity = (1.0 + r).cumprod()
        out[cap_key("max_dd")] = float((equity / equity.cummax() - 1.0).min())
    out[cap_key("ann_return_net_ex_best_1pct")] = ann_return_net_ex_best_1pct(
        r, additive=additive)
    weekly = r.groupby(pd.Grouper(freq="W")).sum() if additive else \
        (1.0 + r).groupby(pd.Grouper(freq="W")).prod() - 1.0
    out[cap_key("worst_week")] = float(weekly.min()) if len(weekly) else float("nan")
    out.update({cap_key(k): v for k, v in year_breakdown(r, additive=additive).items()})
    return out


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

    The same formula is applied to a frame of ORDERS when the strategy drives a
    real book and declares what it sent (`qlab.harness.strategy.own_orders`,
    `qlab.pipeline.evaluate.measured_min_capital`, docs/MIN_CAPITAL.md): the
    weights of such a book read a position a cut left behind as a leg to buy, and
    are shares of a balance that outgrows the money at work, so the number they
    give is not what the venue would be asked for. The frame is then the
    orders that opened a leg, as shares of the strategy's own money.

    Args:
        weights: the full target-weights frame the strategy actually
            produced (e.g. `strategy.target_weights(panel, params)`), same
            shape as the panel; or the frame of opening orders, as shares of
            the strategy's own money. Every non-zero cell across every period is
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
