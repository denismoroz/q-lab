"""Noise strategy generators for ruleset calibration (docs/TASKS.md, T16).

The whole point of T16 is to measure how often `qlab evaluate`'s ruleset
admits a strategy that is KNOWN to have no edge. That measurement is only
honest if every noise book pays the same kind of bill a real strategy would:
comparable gross exposure, comparable turnover (so it pays comparable
trading costs), and a comparable number of open positions. A book with no
turnover pays no costs and clears `net_edge_positive` for free, which would
make the whole exercise meaningless -- see `check_structural_match` and
`STRUCTURAL_BAND` below.

Four generators, each taking `(reference_weights, panel, *, seed)` and
returning an "unconstrained" (not dollar-neutral) noise book with the same
shape as `reference_weights` (docs/TASKS.md, T16's own list):

    random_weights        -- fresh i.i.d. random weights at every rebalance.
    shuffled_instruments  -- reference's own weights, columns permuted by one
                             fixed seeded permutation (destroys the
                             instrument<->signal link, keeps the
                             cross-sectional weight distribution exactly).
    random_signs          -- reference's own weight MAGNITUDES, independent
                             random sign per (period, instrument).
    bootstrap_time        -- reference's own weight ROWS, resampled with
                             replacement independently at every timestamp
                             (destroys the alignment with realised returns).

`neutralize` turns any of the four into the dollar-neutral series (Σ weights
= 0 per row) docs/TASKS.md calls the "чистый шум отбора инструментов" half
of the two required series; the unconstrained generator output IS the other
half ("шум без ограничения на нетто... включает рыночную бету и carry").
Both series come from the SAME underlying random draw for a given
(generator, seed) pair, differing only in whether the neutrality constraint
was applied afterwards -- this isolates exactly the effect the neutrality
constraint has on the admission rate, which is the headline comparison T16
asks for.

Every generator is a deterministic pure function of `(reference_weights,
panel, seed)` -- `np.random.default_rng(seed)` is the only source of
randomness anywhere in this module, seeded explicitly by the caller.

`NoiseStrategy` wraps a generator (named by string in `params`, per the
`qlab.harness.strategy.Strategy` protocol) so noise books run through the
real `evaluate_spec` exactly like any other spec -- same rules, same costs,
same accrual, same universe (docs/TASKS.md T16's own requirement).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import ZERO_WEIGHT_TOL, validate_weights
from qlab.pipeline.evaluate import resolve_strategy


class StructuralMismatchError(ValueError):
    """A generated (or hand-built) book fails to structurally match its
    reference strategy -- see `check_structural_match`."""


@dataclass(frozen=True, slots=True)
class BookShape:
    """Summary statistics describing a weights frame's trading footprint.

    Used to compare a noise book against the real strategy it must be
    structurally matched to (docs/TASKS.md, T16): comparable gross exposure,
    turnover, and position count. A book that is free on any of these three
    axes -- in particular zero turnover, which pays no trading costs -- would
    make the whole calibration meaningless.
    """

    gross: float
    turnover: float
    positions: float


def compute_shape(weights: pd.DataFrame) -> BookShape:
    """`BookShape` from a weights frame.

    Uses the exact turnover convention `qlab.harness.run.run_backtest` uses:
    `|Δweight|` summed across instruments per period, with the previous
    weight taken as 0 before the first row (`weights.shift(1).fillna(0.0)`).
    This is a read of the same numbers the harness itself will compute --
    not a separate approximation -- so a book that looks structurally sound
    here looks the same way to `evaluate_spec`.
    """
    values = weights.to_numpy(dtype=float)
    gross_per_row = np.abs(values).sum(axis=1)
    prev = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    turnover_per_row = np.abs(values - prev).sum(axis=1)
    positions_per_row = (np.abs(values) > ZERO_WEIGHT_TOL).sum(axis=1)
    return BookShape(
        gross=float(gross_per_row.mean()),
        turnover=float(turnover_per_row.mean()),
        positions=float(positions_per_row.mean()),
    )


# Structural-match tolerance band applied to the ratio (candidate / reference)
# of each of the three `BookShape` fields. Deliberately wide: these are four
# DIFFERENT construction methods (docs/TASKS.md T16 lists them), not copies
# of the reference book, so an exact match is neither expected nor the point.
# What this band exists to catch is the failure mode the task calls out by
# name -- "a book with no turnover pays no costs and passes for free" --  not
# to force every generator onto the reference's exact footprint.
#
# 0.2x-8x was chosen by running all four generators (and their dollar-neutral
# variants) against the real trend strategy on the real daily snapshot and
# observing where the actual ratios fall (see `qlab.calibration.test_noise`
# for the measured numbers) -- it is not tuned to make any particular
# calibration trial pass or fail the SCREENING rules, only to keep every
# generator within the same order of magnitude as the strategy it imitates.
#
# The band is deliberately ASYMMETRIC in what it is really guarding against.
# Trend's own turnover is low relative to its gross exposure (0.10 vs 0.46 on
# the real daily snapshot) because its signal is a slow-moving multi-week
# average -- three of the four generators redraw noise independently every
# rebalance, which is inherently CHOPPIER than a persistent trend signal and
# measured 5x-6.7x trend's own turnover (only `shuffled_instruments`, which
# reuses trend's own day-to-day path verbatim, lands within ~0.8x-1.4x). A
# higher upper bound than the lower one is the honest reflection of that: a
# noisier-than-reference book pays MORE trading cost than the real strategy,
# which makes admission HARDER, not easier -- the opposite of the free-ride
# failure mode this module exists to prevent. The lower bound (and the
# unconditional zero-turnover floor below) is where that failure mode
# actually lives, so it is the one held tight.
STRUCTURAL_BAND = (0.2, 8.0)

# Independent of the band above: a candidate at or below this absolute
# turnover floor is rejected outright, regardless of what the reference's own
# turnover happens to be. Zero turnover is a structural defect in its own
# right (docs/TASKS.md T16's own worked example: "книга без оборота... не
# платит костов"), not merely a ratio that happens to be small.
_MIN_TURNOVER = 1e-9


def check_structural_match(
    candidate: BookShape,
    reference: BookShape,
    *,
    band: tuple[float, float] = STRUCTURAL_BAND,
) -> None:
    """Raise `StructuralMismatchError` unless `candidate` is within `band`
    of `reference` on gross, turnover, and position count -- and unless
    `candidate.turnover` clears the absolute zero-turnover floor.

    Raises:
        StructuralMismatchError: `candidate.turnover <= 1e-9` (a free ride --
            no trading costs are ever paid), or any of the three
            candidate/reference ratios falls outside `band`.
    """
    if candidate.turnover <= _MIN_TURNOVER:
        raise StructuralMismatchError(
            f"book has (near) zero turnover ({candidate.turnover!r}); it would pay no "
            "trading costs and is not a valid noise comparison"
        )
    low, high = band
    for field in ("gross", "turnover", "positions"):
        ref_value = getattr(reference, field)
        cand_value = getattr(candidate, field)
        if ref_value <= 0:
            # Nothing to form a ratio against; the absolute turnover floor
            # above already caught the one case (turnover) where this could
            # hide a real problem.
            continue
        ratio = cand_value / ref_value
        if not (low <= ratio <= high):
            raise StructuralMismatchError(
                f"{field} ratio {ratio:.3f} (candidate={cand_value:.6g}, "
                f"reference={ref_value:.6g}) is outside the structural-match "
                f"band {band}"
            )


def _tradeable_mask(panel: MarketPanel) -> np.ndarray:
    return panel.tradeable.to_numpy(dtype=bool)


def random_weights(reference: pd.DataFrame, panel: MarketPanel, *, seed: int) -> pd.DataFrame:
    """Fresh random weights at every rebalance.

    docs/TASKS.md T16, first bullet: "случайные веса на каждом ребалансе".
    Each row independently reassigns `reference`'s OWN magnitudes for that
    day to a random permutation of that day's own active instrument slots --
    the "which instrument" decision is re-randomised fresh at every single
    rebalance (unlike `shuffled_instruments`, which fixes one permutation
    for the whole run). Reusing `reference`'s per-row magnitudes rather than
    drawing fresh continuous values is deliberate, not a simplification: an
    unconstrained continuous draw (e.g. Gaussian) has density arbitrarily
    close to zero, so across ~600 rows and ~100 names it reliably produces
    at least one near-zero-but-nonzero cell -- which `min_capital_usd`
    (`qlab.harness.metrics`) reads as "this strategy needs an effectively
    infinite book to hold its smallest leg", a metric artifact that would
    swamp every noise trial's `capital_fit` rule for reasons having nothing
    to do with edge. Reusing real magnitudes keeps the noise book's smallest
    leg exactly as large as the real strategy's, by construction.
    """
    rng = np.random.default_rng(seed)
    values = reference.to_numpy(dtype=float)
    result = np.zeros_like(values)
    for t in range(values.shape[0]):
        row = values[t]
        support = np.flatnonzero(np.abs(row) > ZERO_WEIGHT_TOL)
        if support.size == 0:
            continue
        shuffled_positions = rng.permutation(support)
        result[t, shuffled_positions] = row[support]
    return pd.DataFrame(result, index=reference.index, columns=reference.columns)


def shuffled_instruments(
    reference: pd.DataFrame, panel: MarketPanel, *, seed: int
) -> pd.DataFrame:
    """`reference`'s own weights with instrument columns permuted by one
    fixed, seeded permutation applied across the whole run.

    docs/TASKS.md T16, second bullet: "веса настоящей стратегии,
    перемешанные между инструментами (распределение весов сохраняется,
    связь «инструмент-сигнал» разрушается)". Because the SAME permutation is
    used at every timestamp, the cross-sectional weight distribution at
    every row is preserved exactly (same multiset of values every row, so
    gross/turnover/position-count match the reference to the last decimal)
    before the tradeable mask below; only which instrument each weight lands
    on is scrambled.

    The tradeable mask is re-applied here (not inherited from `reference`'s
    own support) because a weight now sits on a DIFFERENT instrument than
    the one whose tradeable status it was originally computed against --
    e.g. an instrument delisted earlier than the one `reference` held on
    that day.
    """
    rng = np.random.default_rng(seed)
    n = reference.shape[1]
    perm = rng.permutation(n)
    values = reference.to_numpy(dtype=float)[:, perm]
    values = np.where(_tradeable_mask(panel), values, 0.0)
    return pd.DataFrame(values, index=reference.index, columns=reference.columns)


def random_signs(reference: pd.DataFrame, panel: MarketPanel, *, seed: int) -> pd.DataFrame:
    """`reference`'s own weight magnitudes with an independent random sign
    drawn per (period, instrument).

    docs/TASKS.md T16, third bullet: "случайные знаки при сохранённых
    модулях весов". Every `|weight|` the real strategy actually took is kept
    exactly, so gross exposure and position count match the reference
    exactly by construction; only the direction (long vs short) of each
    position is randomised, destroying the directional signal (this is
    trend's whole edge: which way is which instrument going).
    """
    rng = np.random.default_rng(seed)
    magnitude = np.abs(reference.to_numpy(dtype=float))
    signs = rng.choice(np.array([-1.0, 1.0]), size=magnitude.shape)
    values = magnitude * signs
    return pd.DataFrame(values, index=reference.index, columns=reference.columns)


def bootstrap_time(reference: pd.DataFrame, panel: MarketPanel, *, seed: int) -> pd.DataFrame:
    """`reference`'s own weight rows, resampled with replacement from the
    PAST only -- for row `t`, the resampled source row `j` satisfies `j <=
    t`, never `j > t`.

    docs/TASKS.md T16, fourth bullet: "bootstrap весов во времени
    (разрушает выравнивание с доходностями)". Each day's book is a real day
    the strategy actually held on or before `t` (so the marginal
    distribution of gross/position-count is close to the reference's own),
    but the alignment between a given day's book and that day's realised
    return is destroyed: day t's weights come from a randomly chosen EARLIER
    (or the same) day, never a later one.

    `j <= t` is not a stylistic choice, it is what keeps this generator
    honest: `qlab.harness.strategy.Strategy`'s own alignment rule requires
    `weights.loc[t]` to be decided using only information available up to
    and including `t`. `reference.loc[j]` for `j <= t` satisfies that by
    construction (the reference strategy itself is causal), so reusing it at
    `t` is a legitimate "what if the strategy had instead held one of its
    own past books today" question. Sampling `j > t` (an earlier version of
    this function did, uniformly over the WHOLE index) would hand `t` a
    weight vector that was computed from -- and for a momentum-style
    reference, ENCODES -- price history strictly after `t`, e.g. a lookback
    return spanning `t+1..t+300`. That is look-ahead: the book would know,
    on day `t`, which way the market trended after `t`. See
    `qlab.calibration.test_noise.test_generator_does_not_use_future_reference_rows`
    for the regression test this exact bug is caught by.

    Row 0 has only one possible source (`j = 0`, itself) -- there is no
    "past" before the first row, so it is not resampled at all; this is the
    same "cannot invent history before the panel starts" boundary every
    lookback-based computation in this codebase already respects.

    The tradeable mask is re-applied for the same reason as
    `shuffled_instruments`: a resampled row's active instruments may not
    have been tradeable on the NEW day they were placed on.
    """
    rng = np.random.default_rng(seed)
    n_rows = reference.shape[0]
    values = reference.to_numpy(dtype=float)
    result = np.empty_like(values)
    for t in range(n_rows):
        j = int(rng.integers(0, t + 1))  # only the past-or-present: j in [0, t]
        result[t] = values[j]
    result = np.where(_tradeable_mask(panel), result, 0.0)
    return pd.DataFrame(result, index=reference.index, columns=reference.columns)


GeneratorFn = Callable[[pd.DataFrame, MarketPanel], pd.DataFrame]

GENERATORS: dict[str, GeneratorFn] = {
    "random_weights": random_weights,
    "shuffled_instruments": shuffled_instruments,
    "random_signs": random_signs,
    "bootstrap_time": bootstrap_time,
}


def reference_min_position(reference: pd.DataFrame) -> float:
    """The smallest non-zero `|weight|` `reference` takes anywhere over the
    whole run -- the same quantity `qlab.harness.metrics.min_capital_usd`
    inverts into a capital requirement. Used as the dust floor `neutralize`
    enforces, so a neutralized noise book is never charged a `min_capital_usd`
    the real strategy itself would never be charged for the same reason.

    Raises:
        ValueError: `reference` never takes a non-zero position anywhere.
    """
    magnitudes = np.abs(reference.to_numpy(dtype=float))
    nonzero = magnitudes[magnitudes > ZERO_WEIGHT_TOL]
    if nonzero.size == 0:
        raise ValueError("reference weights never take a non-zero position anywhere")
    return float(nonzero.min())


def _demean_active_and_rescale(
    values: np.ndarray, active: np.ndarray, orig_gross: np.ndarray
) -> np.ndarray:
    count_active = active.sum(axis=1)
    row_sum = np.where(active, values, 0.0).sum(axis=1)
    mean_active = np.divide(
        row_sum, count_active, out=np.zeros_like(row_sum), where=count_active > 0
    )
    demeaned = np.where(active, values - mean_active[:, None], 0.0)
    new_gross = np.abs(demeaned).sum(axis=1)
    scale = np.divide(orig_gross, new_gross, out=np.zeros_like(orig_gross), where=new_gross > 0)
    return demeaned * scale[:, None]


def neutralize(
    raw: pd.DataFrame, *, min_position_floor: float, max_iterations: int = 8
) -> pd.DataFrame:
    """Make `raw` dollar-neutral (Σ weights = 0 per row).

    docs/TASKS.md T16: "dollar-neutral шум (Σ весов = 0) -- чистый шум
    отбора инструментов". Demeans only the ACTIVE (non-zero) cells of each
    row -- never touching a cell that was zero, so this can never introduce
    exposure on an instrument the generator had already excluded (in
    particular never on a non-tradeable one, preserving
    `qlab.harness.strategy.validate_weights`'s contract) -- then rescales the
    demeaned row back to that row's ORIGINAL gross exposure, so neutralizing
    does not, by itself, change how big the book is.

    Demeaning ~100 continuous magnitudes reliably drags a few of them to
    within a hair of the row's mean, producing dust legs `min_capital_usd`
    would read as "needs an effectively infinite book" -- an artifact of
    exact neutrality on a wide cross-section, not a real capital
    requirement. `min_position_floor` (see `reference_min_position`) is
    enforced the same way a real strategy enforces one (see
    `qlab.harness.strategy.Strategy`'s own docstring on FRAB-style slot
    sizing: "DROPS... any candidate whose resulting weight would fall under
    `min_position_size`, rather than under-sizing it into an unexecutable
    dust position"): any cell that ends up below the floor after demeaning
    is dropped (zeroed), the row is re-demeaned over its remaining active
    cells (dropping a leg unbalances the zero-sum book), and this repeats
    for up to `max_iterations` rounds or until nothing more needs dropping.

    A row with 0 or 1 active instruments left (whether from `raw` itself or
    from the floor above emptying it down to one) cannot be made
    dollar-neutral -- there is no zero-sum book with a single leg -- so it
    ends up all-zero. This is a documented, deliberate edge case, not a
    silent bug: on a 222-name universe it affects only a handful of rows,
    negligible in aggregate.

    Raises:
        ValueError: `min_position_floor` is not positive.
    """
    if min_position_floor <= 0:
        raise ValueError(f"min_position_floor must be > 0, got {min_position_floor}")

    values = raw.to_numpy(dtype=float)
    orig_gross = np.abs(values).sum(axis=1)
    active = np.abs(values) > ZERO_WEIGHT_TOL

    result = _demean_active_and_rescale(values, active, orig_gross)
    for _ in range(max_iterations):
        keep = active & (np.abs(result) >= min_position_floor)
        if np.array_equal(keep, active):
            break
        active = keep
        result = _demean_active_and_rescale(values, active, orig_gross)

    return pd.DataFrame(result, index=raw.index, columns=raw.columns)


class NoiseStrategy:
    """A `qlab.harness.strategy.Strategy` whose weights are noise, built
    from a real reference strategy's own weights (docs/TASKS.md, T16).

    Every parameter needed to reproduce the exact weights lives in `params`
    -- `Strategy`'s own pure-function contract -- so a single
    `NoiseStrategy()` instance, resolved via one fixed `code_ref`, serves
    every noise trial; each trial still only depends on `(panel, params)`,
    exactly like any other strategy `qlab.pipeline.evaluate.evaluate_spec`
    runs.

    Required `params` keys:
        generator: one of `GENERATORS`'s keys.
        seed: int, forwarded verbatim to the generator's `np.random.Generator`.
        neutral: bool -- `True` for the dollar-neutral series, `False` for
            the unconstrained one (docs/TASKS.md T16's "две серии").
        reference_code_ref: `code_ref` of the real strategy this noise book
            must structurally match, e.g.
            `"qlab.strategies.trend:TrendTSMOMEnsemble"`.
        reference_params: the `params` mapping to run the reference strategy
            with -- transcribed verbatim from the reference's own spec file,
            never invented here.

    Raises (from `target_weights`):
        KeyError: a required `params` key is missing.
        ValueError: `generator` is not one of `GENERATORS`.
        qlab.calibration.noise.StructuralMismatchError: the generated book
            does not structurally match the reference (see
            `check_structural_match`) -- this propagates out of
            `target_weights` and is caught by `evaluate_spec`'s own
            error-isolation boundary like any other strategy failure,
            recorded as an ERROR trial rather than a silent bad result.
    """

    name = "noise"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        generator_name = str(params["generator"])
        if generator_name not in GENERATORS:
            raise ValueError(
                f"unknown noise generator {generator_name!r}; expected one of "
                f"{sorted(GENERATORS)}"
            )
        seed = int(params["seed"])  # type: ignore[arg-type]
        neutral = bool(params["neutral"])
        reference_code_ref = str(params["reference_code_ref"])
        reference_params = params["reference_params"]
        if not isinstance(reference_params, Mapping):
            raise TypeError("params['reference_params'] must be a mapping")

        reference_strategy = resolve_strategy(reference_code_ref)
        reference_weights = reference_strategy.target_weights(panel, reference_params)
        validate_weights(panel, reference_weights)

        raw = GENERATORS[generator_name](reference_weights, panel, seed=seed)
        if neutral:
            floor = reference_min_position(reference_weights)
            weights = neutralize(raw, min_position_floor=floor)
        else:
            weights = raw

        check_structural_match(compute_shape(weights), compute_shape(reference_weights))
        return weights


__all__ = [
    "GENERATORS",
    "STRUCTURAL_BAND",
    "BookShape",
    "GeneratorFn",
    "NoiseStrategy",
    "StructuralMismatchError",
    "bootstrap_time",
    "check_structural_match",
    "compute_shape",
    "neutralize",
    "reference_min_position",
    "random_signs",
    "random_weights",
    "shuffled_instruments",
]
