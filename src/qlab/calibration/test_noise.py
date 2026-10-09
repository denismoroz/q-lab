"""Tests for `qlab.calibration.noise` (docs/TASKS.md, T16).

Uses a small, hand-built synthetic panel and reference weights rather than
a real market snapshot -- generator determinism and structural-match
arithmetic are properties of `noise.py`'s own code, not of any particular
strategy or dataset, and a synthetic fixture keeps these tests independent
of whether a real snapshot happens to be on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.calibration.noise import (
    GENERATORS,
    STRUCTURAL_BAND,
    BookShape,
    NoiseStrategy,
    StructuralMismatchError,
    check_structural_match,
    compute_shape,
    decision_rows,
    neutralize,
    reference_min_position,
)
from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import ZERO_WEIGHT_TOL, validate_weights

N_ROWS = 40
INSTRUMENTS = [f"COIN{i}" for i in range(10)]


def _panel(*, delist_first_half: bool = False) -> MarketPanel:
    index = pd.date_range("2025-01-01", periods=N_ROWS, freq="1D", tz="UTC")
    prices = pd.DataFrame(100.0, index=index, columns=INSTRUMENTS)
    funding = pd.DataFrame(0.0001, index=index, columns=INSTRUMENTS)
    tradeable = pd.DataFrame(True, index=index, columns=INSTRUMENTS)
    if delist_first_half:
        # COIN0 is not tradeable for the first half -- exercises the
        # tradeable mask that shuffled_instruments/bootstrap_time re-apply.
        tradeable.iloc[: N_ROWS // 2, 0] = False
    return MarketPanel(
        snapshot_id="test-snap",
        prices=prices,
        funding=funding,
        tradeable=tradeable,
        meta={"universe_complete": True},
    )


def _reference_weights(panel: MarketPanel, seed: int = 7) -> pd.DataFrame:
    """A hand-built stand-in for a real strategy's weights: a rolling mean
    of noise (so it isn't i.i.d. from row to row, like a real momentum
    signal), cross-sectionally rescaled to a fixed gross exposure, and
    masked to `panel.tradeable` -- non-degenerate gross/turnover/position
    count, without depending on any real strategy or dataset."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(len(panel.prices.index), len(INSTRUMENTS)))
    smoothed = (
        pd.DataFrame(raw, index=panel.prices.index, columns=INSTRUMENTS)
        .rolling(5, min_periods=1)
        .mean()
    )
    row_gross = smoothed.abs().sum(axis=1).to_numpy()
    weights = smoothed.to_numpy() / row_gross[:, None] * 0.5
    weights = pd.DataFrame(weights, index=panel.prices.index, columns=INSTRUMENTS)
    return weights.where(panel.tradeable, 0.0)


class FixedReferenceStrategy:
    """A `Strategy` that always returns `_reference_weights(panel)` --
    resolved by dotted path in the `NoiseStrategy` end-to-end tests below,
    exactly the way a real strategy would be."""

    name = "fixed-reference"

    def target_weights(self, panel: MarketPanel, params) -> pd.DataFrame:
        return _reference_weights(panel, seed=int(params.get("seed", 7)))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_generators_are_deterministic_given_a_seed(generator_name):
    panel = _panel()
    reference = _reference_weights(panel)
    generator = GENERATORS[generator_name]

    first = generator(reference, panel, seed=123)
    second = generator(reference, panel, seed=123)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_generators_differ_across_seeds(generator_name):
    panel = _panel()
    reference = _reference_weights(panel)
    generator = GENERATORS[generator_name]

    first = generator(reference, panel, seed=1)
    second = generator(reference, panel, seed=2)

    assert not first.equals(second)


# --------------------------------------------------------------------------
# No look-ahead: a decision at (or before) t must not depend on reference
# rows AFTER t.
#
# This is the regression test for a real bug: an earlier `bootstrap_time`
# sampled `j` uniformly over the WHOLE index for every `t`, so row `t` could
# be filled from `reference.loc[j]` with `j > t` -- a row computed from (and,
# for a momentum-style reference, encoding) price history strictly after
# `t`. That is look-ahead: `weights.loc[t]` must be decidable from
# information available up to and including `t` alone
# (`qlab.harness.strategy.Strategy`'s own alignment rule), and it inflated
# the noise calibration's headline numbers because the leak paid off more
# often than chance would. This test is intentionally generator-agnostic
# (parametrized over every entry in `GENERATORS`) so a FUTURE generator with
# the same defect is caught here too, not only in a bug report.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_generator_does_not_use_future_reference_rows(generator_name):
    panel = _panel()
    reference = _reference_weights(panel)
    generator = GENERATORS[generator_name]
    seed = 17
    cutoff = len(reference.index) // 2

    baseline = generator(reference, panel, seed=seed)

    # Perturb ONLY the reference rows strictly after `cutoff` -- drastically,
    # so any dependence on them is impossible to miss by coincidence.
    perturbed_reference = reference.copy()
    perturbed_reference.iloc[cutoff + 1 :] = (
        perturbed_reference.iloc[cutoff + 1 :] * -13.0 + 0.37
    )

    perturbed = generator(perturbed_reference, panel, seed=seed)

    pd.testing.assert_frame_equal(
        baseline.iloc[: cutoff + 1],
        perturbed.iloc[: cutoff + 1],
        check_exact=True,
        obj=(
            f"{generator_name}: output at/before t={cutoff} changed after perturbing "
            "reference rows AFTER t -- this is look-ahead"
        ),
    )


# --------------------------------------------------------------------------
# Dollar-neutral books sum to zero
# --------------------------------------------------------------------------


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_neutralize_sums_to_zero_per_row(generator_name):
    panel = _panel()
    reference = _reference_weights(panel)
    floor = reference_min_position(reference)
    raw = GENERATORS[generator_name](reference, panel, seed=5)

    neutral = neutralize(raw, min_position_floor=floor)

    row_sums = neutral.to_numpy().sum(axis=1)
    assert np.all(np.abs(row_sums) < 1e-8)


def test_neutralize_never_adds_exposure_on_non_tradeable_instrument():
    panel = _panel(delist_first_half=True)
    reference = _reference_weights(panel)
    floor = reference_min_position(reference)
    raw = GENERATORS["shuffled_instruments"](reference, panel, seed=5)

    neutral = neutralize(raw, min_position_floor=floor)

    validate_weights(panel, neutral)  # raises if any non-tradeable cell is non-zero


# --------------------------------------------------------------------------
# Structural match: turnover/gross/position-count comparable to reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
@pytest.mark.parametrize("neutral", [False, True])
def test_generated_books_are_within_the_structural_band(generator_name, neutral):
    """The stated band is `STRUCTURAL_BAND` = (0.2x, 8.0x): every one of the
    four generators, in both series, must land within 0.2x-8x of the
    reference's own gross exposure, turnover, and position count. The lower
    bound is held tight (a near-zero ratio is the free-ride failure mode
    docs/TASKS.md T16 warns about); the upper bound is wide because three of
    the four generators redraw noise independently at every rebalance, which
    is inherently choppier -- and therefore MORE costly, not less -- than a
    slow-moving real trend signal (see `qlab.calibration.noise`'s own
    module-level comment on `STRUCTURAL_BAND`).
    """
    panel = _panel()
    reference = _reference_weights(panel)
    reference_shape = compute_shape(reference)
    floor = reference_min_position(reference)

    raw = GENERATORS[generator_name](reference, panel, seed=11)
    candidate = neutralize(raw, min_position_floor=floor) if neutral else raw
    candidate_shape = compute_shape(candidate)

    # Does not raise -- this IS the assertion (band stated above).
    check_structural_match(candidate_shape, reference_shape)

    low, high = STRUCTURAL_BAND
    for field in ("gross", "turnover", "positions"):
        ref_value = getattr(reference_shape, field)
        cand_value = getattr(candidate_shape, field)
        ratio = cand_value / ref_value
        assert low <= ratio <= high, (
            f"{generator_name} (neutral={neutral}) {field} ratio {ratio} outside "
            f"the stated band {STRUCTURAL_BAND}"
        )


# --------------------------------------------------------------------------
# A zero-turnover book is rejected as structurally invalid
# --------------------------------------------------------------------------


def test_zero_turnover_book_is_rejected():
    panel = _panel()
    reference = _reference_weights(panel)
    reference_shape = compute_shape(reference)

    # A book that never changes: constant non-zero weight on every
    # instrument, every period pays no cost after the (irrelevant, one-off)
    # first entry -- `check_structural_match`'s absolute zero-turnover floor
    # (`qlab.calibration.noise._MIN_TURNOVER`) exists for exactly this shape,
    # independent of the ratio band, so it is exercised directly here on a
    # hand-built `BookShape` with matching gross/position count but zero
    # turnover -- what "a book with no turnover pays no costs" (docs/TASKS.md
    # T16) means precisely.
    constant_shape = BookShape(
        gross=reference_shape.gross, turnover=0.0, positions=reference_shape.positions
    )

    with pytest.raises(StructuralMismatchError, match="turnover"):
        check_structural_match(constant_shape, reference_shape)

    # Sanity check that a real constant weights frame is at least in the
    # same neighbourhood: after the one-off entry cost on row 1, every later
    # row genuinely contributes zero turnover.
    constant = pd.DataFrame(0.1, index=panel.prices.index, columns=INSTRUMENTS)
    real_shape = compute_shape(constant)
    assert real_shape.turnover < reference_shape.turnover


def test_check_structural_match_rejects_gross_far_outside_band():
    panel = _panel()
    reference = _reference_weights(panel)
    reference_shape = compute_shape(reference)

    # 100x the reference's gross exposure, same turnover/position shape --
    # structurally NOT comparable, must be rejected even though turnover is
    # non-zero.
    oversized = reference * 100.0
    oversized_shape = compute_shape(oversized)

    with pytest.raises(StructuralMismatchError, match="gross"):
        check_structural_match(oversized_shape, reference_shape)


# --------------------------------------------------------------------------
# NoiseStrategy end to end (Strategy protocol contract)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
@pytest.mark.parametrize("neutral", [False, True])
def test_noise_strategy_target_weights_matches_panel_shape(generator_name, neutral):
    panel = _panel()
    strategy = NoiseStrategy()
    params = {
        "generator": generator_name,
        "seed": 42,
        "neutral": neutral,
        "reference_code_ref": "qlab.calibration.test_noise:FixedReferenceStrategy",
        "reference_params": {"seed": 7},
    }

    weights = strategy.target_weights(panel, params)

    assert weights.index.equals(panel.prices.index)
    assert weights.columns.equals(panel.prices.columns)
    validate_weights(panel, weights)  # no NaN, no exposure on non-tradeable cells
    if neutral:
        assert np.all(np.abs(weights.to_numpy().sum(axis=1)) < 1e-8)


def test_noise_strategy_is_deterministic_given_full_params():
    panel = _panel()
    strategy = NoiseStrategy()
    params = {
        "generator": "random_weights",
        "seed": 99,
        "neutral": True,
        "reference_code_ref": "qlab.calibration.test_noise:FixedReferenceStrategy",
        "reference_params": {"seed": 7},
    }

    first = strategy.target_weights(panel, params)
    second = NoiseStrategy().target_weights(panel, params)

    pd.testing.assert_frame_equal(first, second)


def test_noise_strategy_rejects_unknown_generator():
    panel = _panel()
    strategy = NoiseStrategy()
    params = {
        "generator": "not-a-real-generator",
        "seed": 1,
        "neutral": False,
        "reference_code_ref": "qlab.calibration.test_noise:FixedReferenceStrategy",
        "reference_params": {"seed": 7},
    }

    with pytest.raises(ValueError, match="unknown noise generator"):
        strategy.target_weights(panel, params)


def test_reference_min_position_matches_min_capital_floor():
    panel = _panel()
    reference = _reference_weights(panel)
    floor = reference_min_position(reference)

    magnitudes = reference.to_numpy()
    nonzero = np.abs(magnitudes[np.abs(magnitudes) > ZERO_WEIGHT_TOL])
    assert floor == pytest.approx(nonzero.min())


def test_reference_min_position_raises_on_all_flat_book():
    panel = _panel()
    flat = pd.DataFrame(0.0, index=panel.prices.index, columns=INSTRUMENTS)

    with pytest.raises(ValueError, match="never take a non-zero position"):
        reference_min_position(flat)


# --- rebalance cadence is part of the structure (T27 item 5) ---------------


def _weekly_reference(panel: MarketPanel, *, every: int = 7) -> pd.DataFrame:
    """A reference that decides once every `every` bars and holds flat in
    between -- the XSMOM shape, which is what broke the calibration."""
    daily = _reference_weights(panel)
    decided = daily.where(
        pd.Series(np.arange(len(daily.index)) % every == 0, index=daily.index), np.nan
    )
    return decided.ffill().fillna(0.0).where(panel.tradeable, 0.0)


def test_decision_rows_is_all_true_for_an_every_bar_reference() -> None:
    """The no-op guarantee: a reference that rebalances every bar must see
    every generator behave exactly as it did before cadence inheritance
    existed, so docs/CALIBRATION_2026-09-21.1.md stays reproducible."""
    panel = _panel()
    assert decision_rows(_reference_weights(panel), panel).all()


def test_decision_rows_follows_a_weekly_reference() -> None:
    panel = _panel()
    decisions = decision_rows(_weekly_reference(panel, every=7), panel)
    assert decisions[0]
    # Every decision after the first lands on a multiple of 7.
    assert all(i % 7 == 0 for i in np.flatnonzero(decisions))
    assert decisions.sum() == pytest.approx(len(panel.prices.index) / 7, abs=1)


def test_decision_rows_does_not_count_a_forced_delisting_exit() -> None:
    """A weight zeroed because the harness's safety gate dropped a delisted
    instrument is not the strategy choosing to trade. Counting it would let
    a delisting-heavy universe inflate the inferred cadence back towards
    every-bar, which is the failure this whole function exists to prevent."""
    index = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
    cols = ["A", "B"]
    tradeable = pd.DataFrame(True, index=index, columns=cols)
    tradeable.iloc[2:, 0] = False  # A delists at bar 2 and stays gone
    panel = MarketPanel(
        snapshot_id="t",
        prices=pd.DataFrame(100.0, index=index, columns=cols),
        funding=pd.DataFrame(0.0, index=index, columns=cols),
        tradeable=tradeable,
        meta={"universe_complete": True},
    )
    # Constant book; the ONLY change is A being zeroed by the delisting.
    reference = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.0], "B": [-0.5] * 4}, index=index)
    decisions = decision_rows(reference, panel)
    assert decisions[0]
    assert not decisions[1:].any()


@pytest.mark.parametrize("generator_name", ["random_weights", "random_signs", "bootstrap_time"])
def test_generators_inherit_a_weekly_reference_cadence(generator_name) -> None:
    """Before this, the three redrawing generators traded every bar whatever
    the reference did: on the real XSMOM book that came out at 8.69x the
    reference's turnover, past the structural-match band's 8.0 ceiling, and
    150 of 200 trials were rejected before they ran (docs/XSMOM_T21.md)."""
    panel = _panel()
    reference = _weekly_reference(panel, every=7)
    noise = GENERATORS[generator_name](reference, panel, seed=11)

    ref_turnover = reference.diff().abs().sum(axis=1).sum()
    noise_turnover = noise.diff().abs().sum(axis=1).sum()
    assert ref_turnover > 0
    ratio = noise_turnover / ref_turnover
    assert 0.2 < ratio < 8.0, f"{generator_name} turnover ratio {ratio:.2f} outside the band"


@pytest.mark.parametrize("generator_name", ["random_weights", "random_signs", "bootstrap_time"])
def test_generators_hold_flat_between_a_weekly_reference_decisions(generator_name) -> None:
    """Sharper than the turnover band: the noise book must be UNCHANGED on
    every bar the reference held flat, not merely close on aggregate."""
    panel = _panel()
    reference = _weekly_reference(panel, every=7)
    noise = GENERATORS[generator_name](reference, panel, seed=5)
    decisions = decision_rows(reference, panel)
    changes = noise.diff().abs().sum(axis=1).to_numpy()
    held_rows = ~decisions
    held_rows[0] = False
    assert np.allclose(changes[held_rows], 0.0)


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_generators_never_hold_a_non_tradeable_instrument(generator_name) -> None:
    """Holding a book across bars (T27 item 5) can carry a weight onto an
    instrument that has since delisted, which `validate_weights` rejects.
    This killed 50 of 200 XSMOM-matched trials before the mask was applied
    in `random_weights` too."""
    panel = _panel(delist_first_half=True)
    reference = _weekly_reference(panel, every=7)
    noise = GENERATORS[generator_name](reference, panel, seed=3)
    assert not (noise.abs().gt(0.0) & ~panel.tradeable).to_numpy().any()


@pytest.mark.parametrize("generator_name", sorted(GENERATORS))
def test_warmup_rows_stay_flat_and_cadence_inheritance_is_a_no_op_there(generator_name) -> None:
    """The regression that protects docs/CALIBRATION_2026-09-21.1.md.

    trend's reference book is all-zero for its first 150 bars (its
    `min_history_days` warm-up) and changes on every one of the 478 bars
    after that -- measured, not assumed. So its only non-decision rows are
    rows where the book holds NOTHING, and "hold the previous book" and
    "draw a fresh one from an empty support" both produce zeros. Cadence
    inheritance therefore cannot move trend's calibration, and this pins the
    property that makes that true.
    """
    panel = _panel()
    reference = _reference_weights(panel)
    warmup = 5
    reference.iloc[:warmup] = 0.0
    noise = GENERATORS[generator_name](reference, panel, seed=17)
    assert (noise.iloc[:warmup].abs().to_numpy() == 0.0).all()


# --------------------------------------------------------------------------
# Matched fallbacks: when a generator's plain book cannot match the reference
# --------------------------------------------------------------------------


def _rotating_universe():
    """A point-in-time universe like top-25-by-cap on Binance: 60 instruments
    listed in turn, the reference holding the five newest tradeable ones."""
    rows, cols, held = 200, 60, 5
    index = pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC")
    names = [f"C{i:02d}" for i in range(cols)]
    tradeable = np.zeros((rows, cols), dtype=bool)
    for j in range(cols):
        tradeable[j * 3 : j * 3 + 20, j] = True  # each lives 20 days, a new one every 3
    rng = np.random.default_rng(1)
    weights = np.zeros((rows, cols))
    for t in range(rows):
        live = np.flatnonzero(tradeable[t])
        take = live[-held:]
        weights[t, take] = rng.choice([-1.0, 1.0], take.size) * 0.1 * (1 + 0.02 * rng.random())

    def frame(v, dtype=float):
        return pd.DataFrame(v, index=index, columns=names).astype(dtype)

    panel = MarketPanel(snapshot_id="rot", prices=frame(100.0 + np.zeros((rows, cols))),
                        funding=frame(np.zeros((rows, cols))), tradeable=frame(tradeable, bool),
                        meta={})
    return panel, frame(weights)


def test_plain_shuffle_loses_its_book_on_a_rotating_universe_and_the_fallback_keeps_it():
    from qlab.calibration.noise import shuffled_among_tradeable

    panel, reference = _rotating_universe()
    ref = compute_shape(reference)
    plain = compute_shape(GENERATORS["shuffled_instruments"](reference, panel, seed=3))
    assert plain.gross / ref.gross < STRUCTURAL_BAND[0]  # the defect: weights land on dead columns
    book = shuffled_among_tradeable(reference, panel, seed=3)
    validate_weights(panel, book)
    matched = compute_shape(book)
    assert matched.gross == pytest.approx(ref.gross)  # every row keeps the reference's weights
    check_structural_match(matched, ref)
    assert not book.equals(reference)  # and they sit on other instruments


@pytest.mark.parametrize("fallback", ["shuffled_instruments", "random_weights"])
def test_fallbacks_do_not_use_future_reference_rows(fallback):
    from qlab.calibration.noise import MATCHED_FALLBACKS

    panel, reference = _rotating_universe()
    generator = MATCHED_FALLBACKS[fallback]
    cutoff = len(reference.index) // 2
    baseline = generator(reference, panel, seed=17)
    perturbed_reference = reference.copy()
    later = perturbed_reference.iloc[cutoff + 1 :]
    perturbed_reference.iloc[cutoff + 1 :] = (later * -13.0).where(later != 0, 0.0)
    perturbed = generator(perturbed_reference, panel, seed=17)
    pd.testing.assert_frame_equal(baseline.iloc[: cutoff + 1], perturbed.iloc[: cutoff + 1])


def test_random_weights_at_the_reference_rate_trades_far_less_than_a_daily_reshuffle():
    from qlab.calibration.noise import random_weights_at_reference_rate

    panel = _panel()
    reference = _reference_weights(panel)
    plain = compute_shape(GENERATORS["random_weights"](reference, panel, seed=5))
    book = random_weights_at_reference_rate(reference, panel, seed=5)
    validate_weights(panel, book)
    paced = compute_shape(book)
    assert paced.turnover < plain.turnover
    assert paced.gross == pytest.approx(compute_shape(reference).gross)


def test_noise_strategy_falls_back_only_when_the_plain_book_does_not_match(monkeypatch):
    import qlab.calibration.noise as noise

    panel, reference = _rotating_universe()
    monkeypatch.setattr(noise, "_reference_weights", lambda *a, **k: reference)
    params = {"generator": "shuffled_instruments", "seed": 3, "neutral": False,
              "reference_code_ref": "x:y", "reference_params": {}}
    book = noise.NoiseStrategy().target_weights(panel, params)  # no StructuralMismatchError
    assert compute_shape(book).gross == pytest.approx(compute_shape(reference).gross)
    # Where the plain book matches, it is the plain book that is returned.
    fixed = _panel()
    fixed_reference = _reference_weights(fixed)
    monkeypatch.setattr(noise, "_reference_weights", lambda *a, **k: fixed_reference)
    plain = GENERATORS["shuffled_instruments"](fixed_reference, fixed, seed=3)
    pd.testing.assert_frame_equal(noise.NoiseStrategy().target_weights(fixed, params), plain)
