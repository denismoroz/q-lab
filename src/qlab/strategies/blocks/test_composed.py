"""Tests for `Composed`: invariants of the assembled strategy on synthetic panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.lookahead import lookahead_violation
from qlab.harness.panel import MarketPanel
from qlab.harness.strategy import validate_weights
from qlab.strategies.blocks import Composed
from qlab.strategies.blocks.composed import rebalance_rows

COLUMNS = list("ABCDEF")


def _walk_panel(n: int = 120, seed: int = 0, columns: list[str] = COLUMNS) -> MarketPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0, 0.02, (n, len(columns))), axis=0),
        index=index, columns=columns,
    )
    prices.iloc[:15, 5] = np.nan  # the last coin is listed late
    volume = pd.DataFrame(rng.uniform(1e3, 1e5, (n, len(columns))), index=index, columns=columns)
    volume = volume.where(prices.notna())
    funding = pd.DataFrame(rng.normal(1e-4, 1e-4, (n, len(columns))), index=index, columns=columns)
    return MarketPanel(
        snapshot_id="walk", prices=prices, funding=funding, tradeable=prices.notna(),
        meta={}, volume=volume,
    )


def _params(**overrides: object) -> dict:
    params: dict = {
        "universe": {"rule": "all"},
        "signal": {"block": "momentum", "lookback_days": 5},
        "portfolio": {"block": "long_short", "top_n": 2},
        "sizing": {"block": "equal", "gross": 1.0},
        "rebalance_days": 7,
        "min_history_days": 10,
    }
    params.update(overrides)
    return params


def test_output_matches_panel_and_passes_validate_weights() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(panel, _params())
    validate_weights(panel, weights)
    assert weights.index.equals(panel.prices.index)
    assert weights.columns.equals(panel.prices.columns)
    assert not weights.isna().any().any()
    assert weights.abs().sum().sum() > 0


def test_long_short_book_is_dollar_neutral_with_gross_one() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(panel, _params())
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50
    assert held.sum(axis=1).abs().max() < 1e-12
    assert held.abs().sum(axis=1).tolist() == pytest.approx([1.0] * len(held))


def test_long_only_book_sums_to_one() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(
        panel, _params(portfolio={"block": "long_only", "top_n": 3})
    )
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50
    assert held.sum(axis=1).tolist() == pytest.approx([1.0] * len(held))
    assert (weights >= 0).all().all()


def test_time_series_sign_gross_is_at_most_one() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(panel, _params(portfolio={"block": "time_series_sign"}))
    assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-12
    assert weights.abs().sum().sum() > 0


def test_rebalance_rows_first_row_then_every_period() -> None:
    index = pd.date_range("2026-01-01", periods=16, freq="D", tz="UTC")
    mask = rebalance_rows(index, 7)
    assert np.flatnonzero(mask).tolist() == [0, 7, 14]


def test_weights_are_carried_forward_between_rebalances() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(panel, _params(rebalance_days=7, min_history_days=3))
    # Row 0 decides with no history -> flat; the next decisions are rows 7, 14, ...
    assert (weights.iloc[:7] == 0).all().all()
    for start in (7, 14, 21, 28):
        block = weights.iloc[start : start + 7]
        for i in range(1, 7):
            pd.testing.assert_series_equal(block.iloc[i], block.iloc[0], check_names=False)
    assert not weights.iloc[7].equals(weights.iloc[14])  # and it does change at a rebalance


def test_untradeable_cells_are_zeroed_even_inside_a_holding_period() -> None:
    panel = _walk_panel()
    base = Composed().target_weights(panel, _params(min_history_days=3))
    held_coin = base.iloc[8].abs().idxmax()
    assert base.iloc[8][held_coin] != 0.0
    tradeable = panel.tradeable.copy()
    tradeable.loc[tradeable.index[9:12], held_coin] = False
    cut = MarketPanel(snapshot_id="cut", prices=panel.prices, funding=panel.funding,
                      tradeable=tradeable, meta={}, volume=panel.volume)
    weights = Composed().target_weights(cut, _params(min_history_days=3))
    validate_weights(cut, weights)
    assert (weights.loc[tradeable.index[9:12], held_coin] == 0.0).all()
    # carried weight comes back after the gap, unchanged, until the next decision
    assert weights.iloc[12][held_coin] == base.iloc[12][held_coin]


def test_spot_columns_are_ignored() -> None:
    plain = _walk_panel()
    cols = COLUMNS + [f"A{SPOT_COLUMN_SUFFIX}", f"C{SPOT_COLUMN_SUFFIX}"]
    rng = np.random.default_rng(9)
    prices = plain.prices.copy()
    for spot in cols[-2:]:
        prices[spot] = 100 * np.cumprod(1 + rng.normal(0.0, 0.05, len(prices)))
    index = prices.index
    with_spot = MarketPanel(
        snapshot_id="spot", prices=prices,
        funding=plain.funding.reindex(columns=cols),
        tradeable=prices.notna(), meta={},
        volume=plain.volume.reindex(columns=cols),
    )
    base = Composed().target_weights(plain, _params())
    weights = Composed().target_weights(with_spot, _params())
    validate_weights(with_spot, weights)
    assert (weights[cols[-2:]] == 0.0).all().all()
    pd.testing.assert_frame_equal(weights[COLUMNS], base)
    assert weights.index.equals(index)


def test_top_volume_does_not_read_volume_after_the_bar() -> None:
    panel = _walk_panel()
    params = _params(universe={"rule": "top_volume", "n": 3, "window_days": 5})
    base = Composed().target_weights(panel, params)
    volume = panel.volume.copy()
    later = volume.iloc[61:]
    volume.iloc[61:] = later * np.random.default_rng(5).uniform(0.01, 100, later.shape)
    changed = MarketPanel(snapshot_id="v", prices=panel.prices, funding=panel.funding,
                          tradeable=panel.tradeable, meta={}, volume=volume)
    again = Composed().target_weights(changed, params)
    pd.testing.assert_frame_equal(base.iloc[:61], again.iloc[:61])
    assert not base.iloc[61:].equals(again.iloc[61:])  # the later volume does matter later


def test_top_volume_restricts_the_book_to_the_biggest_coins() -> None:
    panel = _walk_panel()
    volume = panel.volume.copy()
    volume["A"] = 1e9
    volume["B"] = 9e8
    volume = volume.where(panel.prices.notna())
    boosted = MarketPanel(snapshot_id="b", prices=panel.prices, funding=panel.funding,
                          tradeable=panel.tradeable, meta={}, volume=volume)
    params = _params(universe={"rule": "top_volume", "n": 2, "window_days": 3},
                     portfolio={"block": "long_short", "top_n": 1})
    weights = Composed().target_weights(boosted, params)
    assert (weights[["C", "D", "E", "F"]] == 0.0).all().all()
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50
    assert held[["A", "B"]].abs().sum(axis=1).tolist() == pytest.approx([1.0] * len(held))


def test_history_gate_counts_known_price_bars() -> None:
    panel = _walk_panel()  # F is listed at row 15 (first price on row 15)
    weights = Composed().target_weights(
        panel,
        _params(
            rebalance_days=1,
            min_history_days=10,
            portfolio={"block": "long_only", "top_n": 6},
            signal={"block": "momentum", "lookback_days": 1},
        ),
    )
    # F has 10 known bars from row 15 + 9 = 24 onward (inclusive count)
    assert (weights["F"].iloc[:24] == 0.0).all()
    assert weights["F"].iloc[24] > 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"signal": {"block": "reversal", "lookback_days": 7}},
        {"signal": {"block": "funding", "window_days": 5}},
        {"signal": {"block": "low_vol", "window_days": 10},
         "portfolio": {"block": "long_only", "top_n": 3}},
        {"universe": {"rule": "top_volume", "n": 4, "window_days": 5}},
        {"portfolio": {"block": "time_series_sign"},
         "sizing": {"block": "vol_target", "window_days": 10, "target_daily": 0.02,
                    "leverage_cap": 2.0}},
        {"universe": {"rule": "top_volume", "n": 5, "window_days": 3},
         "signal": {"block": "funding", "window_days": 3},
         "sizing": {"block": "vol_target", "window_days": 5, "target_daily": 0.01,
                    "leverage_cap": 1.5},
         "rebalance_days": 3},
    ],
)
def test_no_look_ahead_for_block_combinations(overrides: dict) -> None:
    panel = _walk_panel(n=150)
    params = _params(**overrides)
    strategy = Composed()
    weights = strategy.target_weights(panel, params)
    assert weights.abs().sum().sum() > 0
    assert lookahead_violation(strategy, panel, params, weights) is None


def _bar_panels(seed: int = 4) -> tuple[MarketPanel, MarketPanel]:
    """The same random walk on 4h bars and on 1d bars (each daily bar is the 4h bar
    at 00:00 of that day, so lookbacks in days pick the same price pairs)."""
    rng = np.random.default_rng(seed)
    cols = ["X", "Y", "Z", "W"]
    four_h = pd.date_range("2025-01-01", periods=6 * 300, freq="4h", tz="UTC")
    ret = rng.normal(0.0, 0.008, size=(len(four_h), len(cols)))
    fine = (1.0 + pd.DataFrame(ret, index=four_h, columns=cols)).cumprod() * 100.0
    coarse = fine.iloc[::6]

    def panel(prices: pd.DataFrame) -> MarketPanel:
        return MarketPanel(
            snapshot_id="bars", prices=prices,
            funding=pd.DataFrame(0.0, index=prices.index, columns=cols),
            tradeable=pd.DataFrame(True, index=prices.index, columns=cols), meta={},
        )

    return panel(fine), panel(coarse)


def test_same_params_give_the_same_positions_on_4h_and_1d_bars() -> None:
    fine, coarse = _bar_panels()
    params = _params(
        signal={"block": "momentum", "lookback_days": 5}, rebalance_days=1, min_history_days=3,
        portfolio={"block": "long_short", "top_n": 1},
    )
    w_fine = Composed().target_weights(fine, params).iloc[::6]
    w_coarse = Composed().target_weights(coarse, params)
    pd.testing.assert_frame_equal(w_fine.iloc[10:], w_coarse.iloc[10:])


def test_vol_target_position_size_does_not_depend_on_bar_size() -> None:
    fine, coarse = _bar_panels()
    params = _params(
        signal={"block": "momentum", "lookback_days": 5}, rebalance_days=1, min_history_days=40,
        portfolio={"block": "time_series_sign"},
        sizing={"block": "vol_target", "window_days": 30, "target_daily": 0.02,
                "leverage_cap": 1e9},
    )
    w_fine = Composed().target_weights(fine, params)
    w_coarse = Composed().target_weights(coarse, params)
    size_fine = w_fine.abs().sum(axis=1)
    size_coarse = w_coarse.abs().sum(axis=1)
    ratio = size_fine[size_fine > 0].median() / size_coarse[size_coarse > 0].median()
    assert ratio == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda p: p.pop("universe"), "universe"),
        (lambda p: p.pop("rebalance_days"), "rebalance_days"),
        (lambda p: p.pop("min_history_days"), "min_history_days"),
        (lambda p: p.update(surprise=1), "surprise"),
        (lambda p: p["signal"].pop("lookback_days"), "lookback_days"),
        (lambda p: p["signal"].update(block="nope"), "nope"),
        (lambda p: p["signal"].update(window_days=3), "window_days"),
        (lambda p: p["portfolio"].update(block="nope"), "nope"),
        (lambda p: p["portfolio"].pop("top_n"), "top_n"),
        (lambda p: p["sizing"].update(block="nope"), "nope"),
        (lambda p: p["sizing"].pop("gross"), "gross"),
        (lambda p: p["universe"].update(rule="nope"), "nope"),
        (lambda p: p.update(rebalance_days=0), "rebalance_days"),
    ],
)
def test_missing_or_unknown_param_raises_value_error(mutate, fragment: str) -> None:
    params = _params()
    mutate(params)
    with pytest.raises(ValueError, match=fragment):
        Composed().target_weights(_walk_panel(), params)


def test_declares_name_and_intervals() -> None:
    assert Composed.name == "composed"
    assert Composed.valid_intervals == ("1d", "4h", "1h")
