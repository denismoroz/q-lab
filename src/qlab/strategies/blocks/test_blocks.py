"""Tests for the individual blocks: every block on a toy case with a hand-computed result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import parse_block
from qlab.strategies.blocks.portfolio import PORTFOLIOS, long_only, long_short, time_series_sign
from qlab.strategies.blocks.signals import SIGNALS, funding, low_vol, momentum, reversal
from qlab.strategies.blocks.sizing import SIZINGS, equal, vol_target
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


# ---- registries / validation ----------------------------------------------


def test_registries_hold_every_block() -> None:
    assert set(UNIVERSES) == {"all", "top_volume"}
    assert set(SIGNALS) == {"momentum", "reversal", "funding", "low_vol"}
    assert set(PORTFOLIOS) == {"long_short", "long_only", "time_series_sign"}
    assert set(SIZINGS) == {"equal", "vol_target"}


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
