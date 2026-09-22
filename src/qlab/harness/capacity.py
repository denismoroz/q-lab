"""Liquidity capacity measurement (docs/TASKS.md, T27, gap 2).

Pure functions over a strategy's own weights frame and a `MarketPanel`'s
volume — no I/O, no rules, no verdicts. `qlab.harness.metrics.min_capital_usd`
answers "what capital does the venue's minimum order size require" and stops
there; it has no notion of whether the resulting positions are actually
TRADEABLE against real turnover. That gap is exactly what let a 20-leg XSMOM
book of thin memecoins (GRIFFAIN, FARTCOIN, MERL, VVV, ZORA, PURR, ...) read
as "affordable" all the way up to $120 000, a size at which each leg is a
$6 000 position with no real market behind it (docs/XSMOM_T21.md,
"Ревизия"). This module measures capacity from the panel's own traded
volume instead of guessing a number.

These are MEASUREMENTS, not gates: nothing here returns a verdict, writes to
`rules/`, or bakes in a pass/fail threshold. `capacity_usd`'s `participation`
argument has no default for exactly that reason — see its docstring.

Both functions treat "the instruments the book actually holds" as those
where `|weight| > qlab.harness.strategy.ZERO_WEIGHT_TOL` at some point in
`weights` — the same zero-weight tolerance and "any nonzero cell over the
whole frame counts" convention `qlab.harness.metrics.min_capital_usd` already
uses, so a leg that only appears in one rebalance out of many still counts.

Both functions restrict every per-instrument calculation (volume, turnover)
to the bars where THAT instrument is actually held (`|weight| > tol`), not
its whole lifetime in the panel — measuring "how liquid is BTC in general"
is a different question from "how liquid was BTC while we were actually
holding a position in it", and the second is what a strategy would actually
need to trade against.

``qlab.harness.panel.MarketPanel`` is used for the type hint (not
``qlab.data.panel.MarketPanel``) for the same reason every other harness
module does this — see ``qlab.harness.panel``'s own docstring: it is a
light structural stand-in, and the real ``qlab.data.panel.MarketPanel``
satisfies the same shape by construction (duck typing, not a shared base
class), so either can be passed here.
"""

from __future__ import annotations

import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import ZERO_WEIGHT_TOL


class CapacityError(ValueError):
    """Raised when a book's liquidity/capacity cannot be honestly computed."""


def _check_shape(weights: pd.DataFrame, panel: MarketPanel) -> None:
    if not weights.index.equals(panel.prices.index):
        raise CapacityError(
            "weights index does not match panel.prices index exactly "
            "(different length, order, or timestamps)"
        )
    if not weights.columns.equals(panel.prices.columns):
        raise CapacityError(
            "weights columns do not match panel.prices columns exactly "
            "(different instruments or order)"
        )


def _held_mask(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.abs() > ZERO_WEIGHT_TOL


def _held_instruments(weights: pd.DataFrame) -> list[str]:
    held = _held_mask(weights).any(axis=0)
    return list(held.index[held])


def book_daily_volume_usd(weights: pd.DataFrame, panel: MarketPanel) -> pd.DataFrame:
    """Traded-volume profile of the instruments the book actually holds.

    For every instrument with `|weight| > ZERO_WEIGHT_TOL` at some point in
    `weights`, restricted to exactly the bars it is held (see module
    docstring), this computes ``volume_usd = panel.volume * panel.prices``
    (base-asset volume times price — `panel.volume` is never in USD itself,
    see `qlab.data.panel.MarketPanel`'s own docstring for why that product
    is always recomputed here rather than stored anywhere), then reports:

    - ``avg_abs_weight``: the mean `|weight|` over the held bars — "how much
      of the book sits in this instrument" while it is held.
    - ``n_bars_held``: how many bars the instrument was actually held for.
    - ``n_bars_volume_known``: of those, how many had a non-NaN volume_usd
      (NaN volume means UNKNOWN, not zero — see
      `qlab.data.sources.base.InstrumentHistory`'s docstring — so it is
      excluded from the two statistics below rather than pulling them
      toward zero).
    - ``median_daily_volume_usd``: the TYPICAL day's traded volume while
      held. Useful for "is this instrument normally liquid enough" but easy
      to overstate capacity with, because a handful of unusually liquid
      days can pull a median up even for a name that is illiquid most of
      the time.
    - ``min_daily_volume_usd``: the WORST single day's traded volume while
      held. This is the conservative, "binding" figure: a fixed book that
      must be able to trade on every day it holds a position is constrained
      by its worst day, not its typical one — exactly why `capacity_usd`
      (below) uses this column, not the median, to compute the capital
      ceiling. Reporting both side by side is what makes a wide gap between
      them visible (a name that is usually liquid but occasionally goes
      dry looks very different from one that is consistently thin), which
      the median or the min ALONE would each hide half of.

    Returns a `DataFrame` indexed by instrument (only instruments actually
    held — a column the book never touched is not "thin", it is
    irrelevant), sorted by `avg_abs_weight` descending (the legs the book
    leans on most first). Empty (zero rows, correct dtypes) if `weights` is
    entirely flat -- an all-flat book holds nothing, so there is no
    instrument to profile, not an error.

    Raises:
        CapacityError: `weights`' index/columns don't match `panel.prices`
            exactly.
    """
    _check_shape(weights, panel)
    volume_usd = panel.volume * panel.prices
    held_mask = _held_mask(weights)

    rows: dict[str, dict[str, float]] = {}
    for instrument in _held_instruments(weights):
        mask = held_mask[instrument]
        series = volume_usd.loc[mask, instrument].dropna()
        rows[instrument] = {
            "avg_abs_weight": float(weights.loc[mask, instrument].abs().mean()),
            "n_bars_held": int(mask.sum()),
            "n_bars_volume_known": int(len(series)),
            "median_daily_volume_usd": float(series.median()) if len(series) else float("nan"),
            "min_daily_volume_usd": float(series.min()) if len(series) else float("nan"),
        }

    profile = pd.DataFrame.from_dict(rows, orient="index")
    if not profile.empty:
        profile = profile.sort_values("avg_abs_weight", ascending=False)
    profile.index.name = "instrument"
    return profile


def _bar_interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing between consecutive timestamps -- the same "use the
    median, not the first gap, so one irregular spacing doesn't skew it"
    idiom as `qlab.harness.metrics.periods_per_year`, duplicated locally
    rather than imported so this module has no dependency on
    `qlab.harness.metrics` (out of scope for this task -- see the module
    docstring)."""
    if len(index) < 2:
        raise CapacityError("cannot infer the panel's bar interval from fewer than 2 timestamps")
    deltas = index.to_series().diff().dropna()
    interval = deltas.median()
    if interval <= pd.Timedelta(0):
        raise CapacityError(f"non-positive median bar spacing: {interval}")
    return interval


def capacity_usd(weights: pd.DataFrame, panel: MarketPanel, participation: float) -> float:
    """Book capital at which the strategy's own daily turnover would exceed
    `participation` of the traded volume of its thinnest meaningful leg.

    THE ARITHMETIC, IN FULL (redoable by hand from these five numbers per
    instrument):

    1. For each instrument the book holds (`|weight| > ZERO_WEIGHT_TOL` at
       some point), take `min_daily_volume_usd` from `book_daily_volume_usd`
       -- the worst day's traded volume while the book actually held it.
       Using the min (not the median) makes this the CONSERVATIVE, binding
       figure: see `book_daily_volume_usd`'s docstring for why.
    2. Compute that instrument's own mean per-bar turnover FRACTION while
       held: `weights[instrument].diff().abs()`, restricted to the bars the
       instrument is held, then averaged. This includes the bar a position
       opens or closes (a real trade against real liquidity, not something
       to exclude) and bars in between where the weight doesn't change
       (correctly pulling the mean toward zero for a long, rarely-rebalanced
       hold -- a leg that is opened once and left alone is not turnover-
       constrained by this formula, which is realistic, not a loophole).
    3. Convert step 2's per-BAR fraction to a per-DAY fraction:
       `daily_turnover_fraction = mean_turnover_per_bar * bars_per_day`,
       where `bars_per_day = 1 day / bar_interval` and `bar_interval` is the
       panel's own median timestamp spacing. At a daily-bar panel (every
       spec in `specs/` today) `bars_per_day == 1` and this is a no-op.
    4. At book capital `C`, the dollar amount traded in that leg per day is
       `C * daily_turnover_fraction`. Solve for the `C` at which that equals
       `participation * min_daily_volume_usd` (step 1):

           leg_capacity = participation * min_daily_volume_usd / daily_turnover_fraction

       Worked example: a leg with `min_daily_volume_usd = $100,000`,
       `daily_turnover_fraction = 0.125` (e.g. opened at weight 0.5 and held
       flat for 4 bars: mean |diff| over those 4 held bars is `0.5/4`), and
       `participation = 0.05` gives `leg_capacity = 0.05 * 100_000 / 0.125
       = $40,000`.
    5. `capacity_usd` is the MINIMUM `leg_capacity` over every held
       instrument -- the thinnest meaningful leg is the one that breaks
       first as capital scales up, exactly the same "the smallest leg sets
       the floor" logic `qlab.harness.metrics.min_capital_usd` already uses
       for the venue's minimum-order-size constraint. A leg with no
       measurable turnover (opened once, never traded again in this window)
       contributes no constraint here and is skipped, not treated as
       infinitely thin.

    Args:
        weights: the strategy's full target-weights frame, same shape as
            `panel.prices` (fractions of book notional, per
            `qlab.harness.strategy.Strategy.target_weights`'s contract).
        panel: supplies `prices` and `volume` (base units; USD volume is
            always `volume * prices`, computed here, never stored).
        participation: the assumed maximum fraction of a day's OWN traded
            volume the book is willing to be, e.g. `0.05` for "at most 5% of
            the day's volume in this name". Has NO DEFAULT: a default here
            would be exactly the kind of invented threshold this task
            exists to refuse (docs/TASKS.md, T27, gap 2) -- the caller must
            state it, and the report this feeds
            (`docs/T27_LIQUIDITY.md`) states several side by side rather
            than picking one.

    Returns:
        Capital in USD, or effectively unbounded (a very large float) only
        in the degenerate case where every held leg has literally zero
        turnover -- which cannot happen if `participation` legs exist,
        since that raises instead (see below).

    Raises:
        CapacityError: `weights`' shape doesn't match `panel.prices`;
            `participation <= 0`; the book never holds anything
            (`book_daily_volume_usd` is empty); any held instrument has
            `min_daily_volume_usd` unknown (NaN) -- an unknown volume is
            not a deep one, so this would otherwise silently understate the
            binding constraint (or miss it) exactly the way a spurious
            `fillna(0)` would; or no held instrument has any measurable
            turnover at all (the book opens positions and never trades
            again in this window, so a turnover-based capacity is
            undefined).
    """
    if participation <= 0:
        raise CapacityError(f"participation must be > 0, got {participation}")
    _check_shape(weights, panel)

    profile = book_daily_volume_usd(weights, panel)
    if profile.empty:
        raise CapacityError(
            "weights never take a non-zero position anywhere; capacity_usd is "
            "undefined for an all-flat book"
        )
    unknown = list(profile.index[profile["min_daily_volume_usd"].isna()])
    if unknown:
        raise CapacityError(
            f"volume is unknown (NaN) for held instrument(s) {unknown}; capacity cannot "
            "be honestly computed -- fix the data (re-fetch with volume) or exclude the "
            "instrument from the book. An unknown volume is not a deep one."
        )

    bars_per_day = pd.Timedelta(days=1) / _bar_interval(panel.prices.index)
    turnover = weights.diff().abs()
    held_mask = _held_mask(weights)

    per_leg_capacity: dict[str, float] = {}
    for instrument in profile.index:
        mask = held_mask[instrument]
        mean_turnover_per_bar = turnover.loc[mask, instrument].mean()
        if pd.isna(mean_turnover_per_bar) or mean_turnover_per_bar <= 0:
            continue
        daily_turnover_fraction = mean_turnover_per_bar * bars_per_day
        per_leg_capacity[instrument] = (
            participation
            * profile.loc[instrument, "min_daily_volume_usd"]
            / daily_turnover_fraction
        )

    if not per_leg_capacity:
        raise CapacityError(
            "no held instrument has measurable turnover (every position, once opened, "
            "is never traded again in this window); capacity_usd is undefined"
        )
    return float(min(per_leg_capacity.values()))


__all__ = ["CapacityError", "book_daily_volume_usd", "capacity_usd"]
