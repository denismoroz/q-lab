"""Tests for the individual blocks: every block on a toy case with a hand-computed result."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qlab.data.sources import coinmarketcap
from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import parse_block
from qlab.strategies.blocks.portfolio import (
    PORTFOLIOS,
    long_only,
    long_only_fraction,
    long_only_quantile,
    long_short,
    long_short_fraction,
    long_short_quantile,
    spot_perp_carry,
    time_series_sign,
)
from qlab.strategies.blocks.signals import SIGNALS, funding, low_vol, momentum, reversal
from qlab.strategies.blocks.sizing import SIZINGS, cap_weight, equal, market_caps, vol_target
from qlab.strategies.blocks.universe import UNIVERSES, all_coins, top_volume

NAN = np.nan


def _panel(prices: dict, volume: dict | None = None, funding_: dict | None = None,
           tradeable: dict | None = None) -> MarketPanel:
    n = len(next(iter(prices.values())))
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    p = pd.DataFrame(prices, index=index, dtype=float)
    return MarketPanel(
        snapshot_id="toy",
        prices=p,
        funding=pd.DataFrame(funding_ if funding_ else 0.0, index=index, columns=p.columns),
        tradeable=pd.DataFrame(tradeable if tradeable else True, index=index, columns=p.columns),
        meta={},
        volume=pd.DataFrame(volume, index=index, columns=p.columns) if volume else None,
    )


def _frame(rows: list[list[float]], columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or list("ABCDEF")[: len(rows[0])]
    index = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=columns, dtype=float)


# ---- signals --------------------------------------------------------------


def test_momentum_is_the_trailing_return_over_the_lookback() -> None:
    panel = _panel({"A": [100, 110, 121], "B": [100, 90, 99]})
    out = momentum(panel, lookback_days=1)
    assert out["A"].isna().iloc[0]
    assert out["A"].iloc[1:].tolist() == pytest.approx([0.10, 0.10])
    assert out["B"].iloc[1:].tolist() == pytest.approx([-0.10, 0.10])
    two = momentum(panel, lookback_days=2)
    assert two["A"].iloc[2] == pytest.approx(0.21)
    assert two["B"].iloc[2] == pytest.approx(-0.01)


def test_reversal_is_minus_momentum() -> None:
    panel = _panel({"A": [100, 110, 121], "B": [100, 90, 99]})
    pd.testing.assert_frame_equal(
        reversal(panel, lookback_days=1), -momentum(panel, lookback_days=1)
    )


def test_funding_score_is_minus_the_trailing_mean_funding() -> None:
    panel = _panel({"A": [1, 1, 1], "B": [1, 1, 1]},
                   funding_={"A": [0.01, 0.03, 0.05], "B": [-0.01, 0.0, NAN]})
    out = funding(panel, window_days=2)
    assert out["A"].isna().iloc[0]
    assert out["A"].iloc[1:].tolist() == pytest.approx([-0.02, -0.04])
    assert out["B"].iloc[1] == pytest.approx(0.005)
    assert np.isnan(out["B"].iloc[2])  # an unknown funding bar is never read as zero


def test_low_vol_is_minus_the_std_of_bar_returns() -> None:
    panel = _panel({"A": [100, 110, 121], "B": [100, 110, 99]})
    out = low_vol(panel, window_days=2)
    assert out["A"].iloc[2] == pytest.approx(0.0, abs=1e-12)  # returns +10%, +10%
    assert out["B"].iloc[2] == pytest.approx(-0.10)  # returns +10%, -10%, population std
    assert out["A"].iloc[:2].isna().all()


# ---- universe -------------------------------------------------------------


def test_all_coins_allows_everything() -> None:
    panel = _panel({"A": [1, 1], "B": [1, 1]})
    assert all_coins(panel).to_numpy().all()


def test_top_volume_ranks_by_trailing_mean_dollar_volume() -> None:
    # Dollar volume = volume * price. Window 2 days, top 1.
    panel = _panel(
        {"A": [10, 10, 10, 10], "B": [1, 1, 1, 1], "C": [1, 1, 1, 1]},
        volume={"A": [1, 1, 1, 1], "B": [5, 5, 20, 20], "C": [4, 4, 4, 30]},
    )
    out = top_volume(panel, n=1, window_days=2)
    assert not out.iloc[0].any()  # window not full yet
    # row 1: mean dollar volume A=10, B=5, C=4 -> A
    assert out.iloc[1].to_dict() == {"A": True, "B": False, "C": False}
    # row 2: A=10, B=12.5, C=4 -> B
    assert out.iloc[2].to_dict() == {"A": False, "B": True, "C": False}
    # row 3: A=10, B=20, C=17 -> B
    assert out.iloc[3].to_dict() == {"A": False, "B": True, "C": False}


def test_top_volume_skips_untradeable_and_unknown_volume() -> None:
    panel = _panel(
        {"A": [1.0] * 3, "B": [1.0] * 3, "C": [1.0] * 3},
        volume={"A": [100, 100, 100], "B": [10, 10, 10], "C": [1, NAN, 1]},
        tradeable={"A": [True, True, False], "B": True, "C": True},
    )
    out = top_volume(panel, n=1, window_days=1)
    assert out["A"].tolist() == [True, True, False]  # A not tradeable at row 2: B takes the place
    assert out["B"].tolist() == [False, False, True]
    assert out["C"].tolist() == [False, False, False]


def test_top_volume_ties_go_to_the_first_listed_coin() -> None:
    panel = _panel({"A": [1.0, 1.0], "B": [1.0, 1.0]}, volume={"A": [5, 5], "B": [5, 5]})
    out = top_volume(panel, n=1, window_days=1)
    assert out.iloc[1].to_dict() == {"A": True, "B": False}


# ---- portfolio ------------------------------------------------------------


def test_long_short_takes_top_and_bottom_with_half_each_side() -> None:
    scores = _frame([[3, 1, 2, 0]])
    eligible = scores.notna()
    out = long_short(scores, eligible, top_n=1)
    assert out.iloc[0].tolist() == [0.5, 0.0, 0.0, -0.5]
    out2 = long_short(scores, eligible, top_n=2)
    assert out2.iloc[0].tolist() == [0.25, -0.25, 0.25, -0.25]
    # B(1) and D(0) are the two lowest, A(3) and C(2) the two highest
    assert out2.iloc[0].tolist() == [0.25, -0.25, 0.25, -0.25]


def test_long_short_with_few_eligible_coins_uses_floor_half() -> None:
    scores = _frame([[3, 1, 2, 5, 4]])
    eligible = pd.DataFrame([[True, True, True, False, False]], index=scores.index,
                            columns=scores.columns)
    out = long_short(scores, eligible, top_n=5)  # 3 eligible -> k = 1
    assert out.iloc[0].tolist() == [0.5, -0.5, 0.0, 0.0, 0.0]
    one = pd.DataFrame([[True, False, False, False, False]], index=scores.index,
                       columns=scores.columns)
    assert long_short(scores, one, top_n=3).abs().sum().sum() == 0.0  # k = 0 -> no position
    none = one & False
    assert long_short(scores, none, top_n=3).abs().sum().sum() == 0.0


def test_long_short_is_dollar_neutral_with_gross_one() -> None:
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(rng.normal(size=(30, 9)), columns=list("ABCDEFGHI"),
                          index=pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"))
    eligible = pd.DataFrame(rng.random((30, 9)) > 0.3, index=scores.index, columns=scores.columns)
    out = long_short(scores, eligible, top_n=3)
    assert (out.to_numpy()[~eligible.to_numpy()] == 0.0).all()
    for _, row in out.iterrows():
        if row.abs().sum() > 0:
            assert row.sum() == pytest.approx(0.0, abs=1e-12)
            assert row.abs().sum() == pytest.approx(1.0)


def test_long_short_ties_break_by_column_order() -> None:
    scores = _frame([[1, 1, 1, 1]])
    out = long_short(scores, scores.notna(), top_n=1)
    assert out.iloc[0].tolist() == [0.5, 0.0, 0.0, -0.5]


def test_long_only_equal_parts_summing_to_one() -> None:
    scores = _frame([[3, 1, 2, 0]])
    out = long_only(scores, scores.notna(), top_n=2)
    assert out.iloc[0].tolist() == [0.5, 0.0, 0.5, 0.0]
    few = long_only(scores, pd.DataFrame([[True, False, True, False]], index=scores.index,
                                         columns=scores.columns), top_n=5)
    assert few.iloc[0].tolist() == [0.5, 0.0, 0.5, 0.0]  # only 2 eligible: 1/2 each
    assert few.iloc[0].sum() == pytest.approx(1.0)


def test_long_short_quantile_takes_top_and_bottom_group() -> None:
    # 10 eligible coins, quantiles=5 -> k = 2: the two best long, the two worst short.
    scores = _frame([[9, 0, 8, 1, 5, 4, 6, 3, 7, 2]], list("ABCDEFGHIJ"))
    out = long_short_quantile(scores, scores.notna(), quantiles=5)
    expected = [0.25, -0.25, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # best: A(9), C(8); worst: B(0), D(1)
    assert out.iloc[0].tolist() == expected
    # quantiles=2 -> k = 5: halves; every coin is held, 0.1 each side
    halves = long_short_quantile(scores, scores.notna(), quantiles=2)
    assert halves.iloc[0].tolist() == pytest.approx(
        [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])


def test_long_short_quantile_floors_the_group_size_and_can_be_empty() -> None:
    scores = _frame([[3, 1, 2, 5, 4, 0, 6]], list("ABCDEFG"))
    # 7 eligible, quantiles=3 -> k = 2 (floor of 7 / 3); the best two long, the worst two short
    out = long_short_quantile(scores, scores.notna(), quantiles=3)
    assert out.iloc[0].tolist() == [0.0, -0.25, 0.0, 0.25, 0.0, -0.25, 0.25]
    # 4 eligible, quantiles=5 -> k = 0: no position
    four = pd.DataFrame([[True, True, True, True, False, False, False]], index=scores.index,
                        columns=scores.columns)
    assert long_short_quantile(scores, four, quantiles=5).abs().sum().sum() == 0.0
    none = four & False
    assert long_short_quantile(scores, none, quantiles=2).abs().sum().sum() == 0.0


def test_long_short_quantile_ineligible_coins_stay_flat_and_book_is_neutral() -> None:
    rng = np.random.default_rng(2)
    scores = pd.DataFrame(rng.normal(size=(30, 12)), columns=list("ABCDEFGHIJKL"),
                          index=pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"))
    eligible = pd.DataFrame(rng.random((30, 12)) > 0.3, index=scores.index, columns=scores.columns)
    out = long_short_quantile(scores, eligible, quantiles=4)
    assert (out.to_numpy()[~eligible.to_numpy()] == 0.0).all()
    for ts, row in out.iterrows():
        if row.abs().sum() > 0:
            assert row.sum() == pytest.approx(0.0, abs=1e-12)
            assert row.abs().sum() == pytest.approx(1.0)
            assert (row > 0).sum() == eligible.loc[ts].sum() // 4


def test_long_short_quantile_ties_break_by_column_order() -> None:
    scores = _frame([[1, 1, 1, 1]])
    out = long_short_quantile(scores, scores.notna(), quantiles=4)  # k = 1
    assert out.iloc[0].tolist() == [0.5, 0.0, 0.0, -0.5]


def test_long_only_quantile_equal_parts_of_the_top_group() -> None:
    scores = _frame([[3, 1, 2, 0, 5, 4]])
    # 6 eligible, quantiles=3 -> k = 2: E(5) and F(4)
    out = long_only_quantile(scores, scores.notna(), quantiles=3)
    assert out.iloc[0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    # quantiles=1 -> every eligible coin, equal parts
    every = long_only_quantile(scores, scores.notna(), quantiles=1)
    assert every.iloc[0].tolist() == pytest.approx([1 / 6] * 6)
    # 7 eligible, quantiles=2 -> k = 3
    seven = _frame([[7, 6, 5, 4, 3, 2, 1]], list("ABCDEFG"))
    assert long_only_quantile(seven, seven.notna(), quantiles=2).iloc[0].tolist() == \
        pytest.approx([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0])
    # fewer eligible coins than quantiles -> k = 0, no position
    two = pd.DataFrame([[True, True, False, False, False, False]], index=scores.index,
                       columns=scores.columns)
    assert long_only_quantile(scores, two, quantiles=3).abs().sum().sum() == 0.0


def test_quantile_blocks_validate_their_group_count() -> None:
    for block, bad in (("long_short_quantile", 1), ("long_short_quantile", 0),
                       ("long_only_quantile", 0), ("long_only_quantile", 2.5),
                       ("long_only_quantile", True)):
        with pytest.raises(ValueError, match="quantiles"):
            parse_block("portfolio", {"block": block, "quantiles": bad}, PORTFOLIOS, "block")
    for block, good in (("long_short_quantile", 2), ("long_only_quantile", 1),
                        ("long_short_quantile", 5.0)):
        parse_block("portfolio", {"block": block, "quantiles": good}, PORTFOLIOS, "block")


def test_time_series_sign_divides_by_the_number_of_eligible_coins() -> None:
    scores = _frame([[2.0, -1.0, 0.0, 5.0]])
    eligible = pd.DataFrame([[True, True, True, False]], index=scores.index,
                            columns=scores.columns)
    out = time_series_sign(scores, eligible)
    assert out.iloc[0].tolist() == pytest.approx([1 / 3, -1 / 3, 0.0, 0.0])
    assert out.abs().sum(axis=1).iloc[0] <= 1.0


# ---- sizing ---------------------------------------------------------------


def test_equal_scales_by_gross() -> None:
    panel = _panel({"A": [1, 1], "B": [1, 1]})
    weights = _frame([[0.5, -0.5]] * 2, ["A", "B"])
    assert equal(weights, panel, gross=2.0).iloc[0].tolist() == [1.0, -1.0]


def test_vol_target_sizes_inverse_to_daily_vol() -> None:
    # A returns +10%, -10% -> population std 0.10; B returns +1%, -1% -> 0.01.
    panel = _panel({"A": [100, 110, 99], "B": [100, 101, 99.99]})
    weights = _frame([[0.5, -0.5]] * 3, ["A", "B"])
    out = vol_target(weights, panel, window_days=2, target_daily=0.02, leverage_cap=100.0)
    # returns of A: .1, -0.1 -> std .1 -> 0.5 * 0.02 / 0.1 = 0.1; B: -0.5 * 0.02 / 0.01 = -1.0
    assert out.iloc[2].to_numpy() == pytest.approx([0.1, -1.0])
    assert (out.iloc[:2] == 0.0).all().all()  # no full window yet: undefined vol -> 0


def test_vol_target_scales_the_whole_book_down_to_the_cap() -> None:
    panel = _panel({"A": [100, 110, 99], "B": [100, 101, 99.99]})
    weights = _frame([[0.5, -0.5]] * 3, ["A", "B"])
    out = vol_target(weights, panel, window_days=2, target_daily=0.02, leverage_cap=0.55)
    assert out.iloc[2].abs().sum() == pytest.approx(0.55)
    assert out.iloc[2].to_numpy() == pytest.approx(np.array([0.1, -1.0]) * 0.55 / 1.1)


def test_vol_target_zero_vol_gives_zero_weight() -> None:
    panel = _panel({"A": [100, 100, 100], "B": [100, 101, 99.99]})
    weights = _frame([[0.5, -0.5]] * 3, ["A", "B"])
    out = vol_target(weights, panel, window_days=2, target_daily=0.02, leverage_cap=100.0)
    assert out["A"].iloc[2] == 0.0


# ---- portfolio by fraction ------------------------------------------------


def test_long_short_fraction_takes_the_top_and_bottom_share() -> None:
    # 10 eligible coins, fraction 0.3 -> k = 3: A(9), C(8), I(7) long; B(0), D(1), J(2) short.
    scores = _frame([[9, 0, 8, 1, 5, 4, 6, 3, 7, 2]], list("ABCDEFGHIJ"))
    out = long_short_fraction(scores, scores.notna(), fraction=0.3)
    third = 0.5 / 3
    expected = [third, -third, third, -third, 0, 0, 0, 0, third, -third]
    assert out.iloc[0].tolist() == pytest.approx(expected)
    assert out.iloc[0].sum() == pytest.approx(0.0, abs=1e-12)
    assert out.iloc[0].abs().sum() == pytest.approx(1.0)


def test_long_short_fraction_floors_the_group_size_and_can_be_empty() -> None:
    scores = _frame([[3, 1, 2, 5, 4, 0, 6]], list("ABCDEFG"))
    # 7 eligible * 0.3 = 2.1 -> k = 2: G(6), D(5) long; F(0), B(1) short
    out = long_short_fraction(scores, scores.notna(), fraction=0.3)
    assert out.iloc[0].tolist() == [0.0, -0.25, 0.0, 0.25, 0.0, -0.25, 0.25]
    # 3 eligible * 0.3 = 0.9 -> k = 0: no position
    three = pd.DataFrame([[True, True, True, False, False, False, False]], index=scores.index,
                         columns=scores.columns)
    assert long_short_fraction(scores, three, fraction=0.3).abs().sum().sum() == 0.0
    none = three & False
    assert long_short_fraction(scores, none, fraction=0.3).abs().sum().sum() == 0.0
    # fraction 0.5 with 5 eligible -> k = 2 per side, the middle coin stays out
    five = _frame([[5, 4, 3, 2, 1]])
    half = long_short_fraction(five, five.notna(), fraction=0.5)
    assert half.iloc[0].tolist() == [0.25, 0.25, 0.0, -0.25, -0.25]


def test_long_short_fraction_is_not_fooled_by_float_noise() -> None:
    # 100 * 0.29 is 28.999999999999996 in floats; the group must still be 29 coins.
    scores = _frame([list(range(100))], [f"C{i:03d}" for i in range(100)])
    out = long_short_fraction(scores, scores.notna(), fraction=0.29)
    assert (out.iloc[0] > 0).sum() == 29
    assert (out.iloc[0] < 0).sum() == 29


def test_long_short_fraction_ineligible_stay_flat_and_ties_break_by_column_order() -> None:
    rng = np.random.default_rng(3)
    scores = pd.DataFrame(rng.normal(size=(30, 12)), columns=list("ABCDEFGHIJKL"),
                          index=pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"))
    eligible = pd.DataFrame(rng.random((30, 12)) > 0.3, index=scores.index, columns=scores.columns)
    out = long_short_fraction(scores, eligible, fraction=0.25)
    assert (out.to_numpy()[~eligible.to_numpy()] == 0.0).all()
    for ts, row in out.iterrows():
        if row.abs().sum() > 0:
            assert row.sum() == pytest.approx(0.0, abs=1e-12)
            assert row.abs().sum() == pytest.approx(1.0)
            assert (row > 0).sum() == int(eligible.loc[ts].sum() * 0.25)
    ties = _frame([[1, 1, 1, 1]])
    assert long_short_fraction(ties, ties.notna(), fraction=0.25).iloc[0].tolist() == \
        [0.5, 0.0, 0.0, -0.5]


def test_long_only_fraction_equal_parts_of_the_top_share() -> None:
    scores = _frame([[3, 1, 2, 0, 5, 4]])
    # 6 eligible * 0.5 = 3 -> E(5), F(4), A(3), a third each
    out = long_only_fraction(scores, scores.notna(), fraction=0.5)
    assert out.iloc[0].tolist() == pytest.approx([1 / 3, 0, 0, 0, 1 / 3, 1 / 3])
    # fraction 1 -> every eligible coin, equal parts
    every = long_only_fraction(scores, scores.notna(), fraction=1.0)
    assert every.iloc[0].tolist() == pytest.approx([1 / 6] * 6)
    # 6 eligible * 0.1 -> k = 0, no position
    assert long_only_fraction(scores, scores.notna(), fraction=0.1).abs().sum().sum() == 0.0
    two = pd.DataFrame([[True, False, True, False, False, False]], index=scores.index,
                       columns=scores.columns)
    assert long_only_fraction(scores, two, fraction=1.0).iloc[0].tolist() == \
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]


def test_fraction_blocks_validate_the_share() -> None:
    for block, bad in (("long_short_fraction", 0), ("long_short_fraction", 0.6),
                       ("long_short_fraction", -0.1), ("long_only_fraction", 1.01),
                       ("long_only_fraction", 0), ("long_only_fraction", True),
                       ("long_only_fraction", float("nan")), ("long_short_fraction", "0.3")):
        with pytest.raises(ValueError, match="portfolio.fraction"):
            parse_block("portfolio", {"block": block, "fraction": bad}, PORTFOLIOS, "block")
    for block, good in (("long_short_fraction", 0.5), ("long_short_fraction", 0.3),
                        ("long_only_fraction", 1.0), ("long_only_fraction", 0.3)):
        parse_block("portfolio", {"block": block, "fraction": good}, PORTFOLIOS, "block")
    with pytest.raises(ValueError, match="top_n"):
        parse_block("portfolio", {"block": "long_short_fraction", "fraction": 0.3, "top_n": 3},
                    PORTFOLIOS, "block")


# ---- carry: a coin against its perpetual ---------------------------------


def test_spot_perp_carry_sells_the_perpetual_of_the_lowest_scores() -> None:
    # Funding signal scores are minus the mean funding, so the LOWEST scores are the
    # highest funding: those perpetuals are shorted (they receive the funding).
    funding_rates = _frame([[0.01, 0.05, -0.02, 0.03]])
    scores = -funding_rates
    out = spot_perp_carry(scores, scores.notna(), top_n=2)
    # B (0.05) and D (0.03) pay the short the most
    assert out.iloc[0].tolist() == [0.0, -0.25, 0.0, -0.25]
    assert out.iloc[0].sum() == pytest.approx(-0.5)
    one = spot_perp_carry(scores, scores.notna(), top_n=1)
    assert one.iloc[0].tolist() == [0.0, -0.5, 0.0, 0.0]


def test_spot_perp_carry_takes_fewer_coins_when_fewer_are_eligible() -> None:
    scores = _frame([[3, 1, 2, 5, 4]])
    eligible = pd.DataFrame([[True, True, False, False, True]], index=scores.index,
                            columns=scores.columns)
    out = spot_perp_carry(scores, eligible, top_n=5)  # 3 eligible: A(3), B(1), E(4)
    assert out.iloc[0].tolist() == pytest.approx([-1 / 6, -1 / 6, 0.0, 0.0, -1 / 6])
    none = eligible & False
    assert spot_perp_carry(scores, none, top_n=2).abs().sum().sum() == 0.0


def test_spot_perp_carry_ties_select_the_last_listed_coins() -> None:
    scores = _frame([[1, 1, 1, 1]])
    out = spot_perp_carry(scores, scores.notna(), top_n=1)
    assert out.iloc[0].tolist() == [0.0, 0.0, 0.0, -0.5]  # as in `_ranks`: first is best


def test_spot_perp_carry_is_registered_as_needing_spot() -> None:
    assert PORTFOLIOS["spot_perp_carry"].needs_spot
    assert not any(b.needs_spot for name, b in PORTFOLIOS.items() if name != "spot_perp_carry")
    parse_block("portfolio", {"block": "spot_perp_carry", "top_n": 3}, PORTFOLIOS, "block")
    with pytest.raises(ValueError, match="portfolio.top_n"):
        parse_block("portfolio", {"block": "spot_perp_carry", "top_n": 0}, PORTFOLIOS, "block")


# ---- cap weighting ----------------------------------------------------------


def _cap_panel(caps: list[list[float]], columns: list[str] | None = None) -> MarketPanel:
    """A panel whose meta carries `caps` (bars x coins) as the market caps known at each bar."""
    cap = _frame(caps, columns)
    panel = _panel({c: [1.0] * len(cap) for c in cap.columns})
    return MarketPanel(snapshot_id="cap", prices=panel.prices, funding=panel.funding,
                       tradeable=panel.tradeable, meta={"market_cap": cap}, volume=panel.volume)


def test_cap_weight_splits_each_side_by_cap_and_keeps_the_side_sum() -> None:
    panel = _cap_panel([[3.0, 1.0, 2.0, 6.0]])
    weights = _frame([[0.25, 0.25, -0.25, -0.25]])
    out = cap_weight(weights, panel, gross=1.0)
    # long side 0.5 split 3:1, short side -0.5 split 2:6
    assert out.iloc[0].tolist() == pytest.approx([0.375, 0.125, -0.125, -0.375])
    scaled = cap_weight(weights, panel, gross=2.0)
    assert scaled.iloc[0].tolist() == pytest.approx([0.75, 0.25, -0.25, -0.75])


def test_cap_weight_handles_long_only_and_carry_sides() -> None:
    panel = _cap_panel([[1.0, 2.0, 7.0]])
    long_only_w = _frame([[0.5, 0.5, 0.0]])
    assert cap_weight(long_only_w, panel, gross=1.0).iloc[0].tolist() == \
        pytest.approx([1 / 3, 2 / 3, 0.0])
    short_only_w = _frame([[0.0, -0.25, -0.25]])  # the perpetual side of a carry book
    assert cap_weight(short_only_w, panel, gross=1.0).iloc[0].tolist() == \
        pytest.approx([0.0, -0.5 * 2 / 9, -0.5 * 7 / 9])


def test_cap_weight_unknown_cap_gets_zero_and_the_side_keeps_its_sum() -> None:
    panel = _cap_panel([[3.0, NAN, 2.0, 2.0]])
    weights = _frame([[1 / 4, 1 / 4, -0.25, -0.25]])  # the long side holds A and B, sum 0.5
    out = cap_weight(weights, panel, gross=1.0)
    assert out.iloc[0].tolist() == pytest.approx([0.5, 0.0, -0.25, -0.25])


def test_cap_weight_flattens_a_row_where_a_held_side_has_no_known_cap() -> None:
    panel = _cap_panel([[NAN, NAN, 2.0, 2.0], [3.0, 1.0, 2.0, 2.0]])
    weights = _frame([[0.25, 0.25, -0.25, -0.25]] * 2)
    out = cap_weight(weights, panel, gross=1.0)
    assert out.iloc[0].tolist() == [0.0, 0.0, 0.0, 0.0]  # never a one-sided "neutral" book
    assert out.iloc[1].abs().sum() == pytest.approx(1.0)
    # a bar with no position stays empty
    empty = _frame([[0.0, 0.0, 0.0, 0.0]] * 2)
    assert cap_weight(empty, panel, gross=1.0).abs().sum().sum() == 0.0
    # non-positive caps count as unknown
    zero = _cap_panel([[0.0, -5.0, 2.0, 2.0]])
    assert cap_weight(weights.iloc[:1], zero, gross=1.0).abs().sum().sum() == 0.0


def test_market_caps_without_any_data_names_what_to_do(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(coinmarketcap, "DEFAULT_STORE", tmp_path / "empty")
    panel = _panel({"A": [1.0, 1.0], "B": [1.0, 1.0]})
    weights = _frame([[0.5, -0.5]] * 2, ["A", "B"])
    with pytest.raises(ValueError, match="cap_weight.*coin-attributes"):
        cap_weight(weights, panel, gross=1.0)
    with pytest.raises(ValueError, match="DataFrame"):
        market_caps(MarketPanel(snapshot_id="x", prices=panel.prices, funding=panel.funding,
                                tradeable=panel.tradeable, meta={"market_cap": [1, 2]},
                                volume=panel.volume))


def _write_snapshot(store, day: str, caps: dict[str, float]) -> None:
    rows = [{"symbol": sym, "cmcRank": rank, "market_cap_usd": cap, "tags": []}
            for rank, (sym, cap) in enumerate(caps.items(), start=1)]
    (store / f"{day}.json").write_text(
        json.dumps({"date": day, "rows": rows}), encoding="utf-8")


def test_cap_weight_reads_stored_snapshots_visible_only_after_their_date(
        tmp_path, monkeypatch) -> None:
    _write_snapshot(tmp_path, "2026-01-03", {"A": 1.0, "B": 3.0})
    _write_snapshot(tmp_path, "2026-01-05", {"A": 9.0, "B": 1.0})
    monkeypatch.setattr(coinmarketcap, "DEFAULT_STORE", tmp_path)
    panel = _panel({"A": [1.0] * 7, "B": [1.0] * 7})  # daily bars 2026-01-01 .. 2026-01-07
    weights = _frame([[0.5, -0.5]] * 7, ["A", "B"])
    caps = market_caps(panel)
    # known nothing on 01-01..01-03; the 01-03 snapshot from 01-04; the 01-05 one from 01-06
    assert caps["A"].isna().tolist()[:3] == [True, True, True]
    assert caps["A"].tolist()[3:5] == [1.0, 1.0] and caps["A"].tolist()[5:] == [9.0, 9.0]
    out = cap_weight(weights, panel, gross=1.0)
    # a one-coin side takes the whole side whatever its cap, but needs a known cap
    assert out["A"].tolist() == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
    assert out["B"].tolist() == [0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5]

# ---- registries / validation ----------------------------------------------


def test_registries_hold_every_block() -> None:
    assert set(UNIVERSES) == {"all", "top_volume"}
    assert set(SIGNALS) == {"momentum", "reversal", "funding", "low_vol"}
    assert set(PORTFOLIOS) == {
        "long_short", "long_only", "long_short_quantile", "long_only_quantile",
        "long_short_fraction", "long_only_fraction", "spot_perp_carry", "time_series_sign"}
    assert set(SIZINGS) == {"equal", "cap_weight", "vol_target"}


@pytest.mark.parametrize(
    "kind, raw, registry, selector, fragment",
    [
        ("signal", {"block": "nope"}, SIGNALS, "block", "nope"),
        ("signal", {"block": "momentum"}, SIGNALS, "block", "lookback_days"),
        ("signal", {"block": "momentum", "lookback_days": 5, "extra": 1},
         SIGNALS, "block", "extra"),
        ("signal", {"lookback_days": 5}, SIGNALS, "block", "block"),
        ("signal", {"block": "momentum", "lookback_days": -1}, SIGNALS, "block", "lookback_days"),
        ("signal", {"block": "momentum", "lookback_days": True}, SIGNALS, "block", "lookback_days"),
        ("portfolio", {"block": "long_short", "top_n": 1.5}, PORTFOLIOS, "block", "top_n"),
        ("universe", {"rule": "top_volume", "n": 3}, UNIVERSES, "rule", "window_days"),
        ("sizing", "equal", SIZINGS, "block", "sizing"),
    ],
)
def test_parse_block_rejects_bad_fragments_naming_the_problem(
    kind: str, raw: object, registry: dict, selector: str, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        parse_block(kind, raw, registry, selector)
