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
        {"portfolio": {"block": "long_short_quantile", "quantiles": 3}},
        {"portfolio": {"block": "long_only_quantile", "quantiles": 2}},
        {"portfolio": {"block": "long_short_fraction", "fraction": 0.4}},
        {"portfolio": {"block": "long_only_fraction", "fraction": 0.5},
         "signal": {"block": "reversal", "lookback_days": 3}},
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


# ---- fraction portfolios ---------------------------------------------------------


def test_fraction_portfolio_holds_the_share_of_eligible_coins_on_each_side() -> None:
    panel = _walk_panel()
    weights = Composed().target_weights(
        panel, _params(portfolio={"block": "long_short_fraction", "fraction": 0.4},
                       min_history_days=3, rebalance_days=1))
    validate_weights(panel, weights)
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50
    assert held.sum(axis=1).abs().max() < 1e-12
    assert held.abs().sum(axis=1).tolist() == pytest.approx([1.0] * len(held))
    # 5 or 6 eligible coins * 0.4 -> 2 per side
    assert set(((held > 0).sum(axis=1)).tolist()) == {2}
    long_only_book = Composed().target_weights(
        panel, _params(portfolio={"block": "long_only_fraction", "fraction": 0.5},
                       min_history_days=3, rebalance_days=1))
    held = long_only_book[long_only_book.abs().sum(axis=1) > 0]
    assert held.sum(axis=1).tolist() == pytest.approx([1.0] * len(held))
    assert (long_only_book >= 0).all().all()


# ---- cap weighting ----------------------------------------------------------------


def _with_caps(panel: MarketPanel, seed: int = 6) -> MarketPanel:
    """`panel` with market caps (known at each bar) in its meta: a stable size per coin."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(1.0, 50.0, size=len(panel.prices.columns))
    caps = pd.DataFrame(np.tile(size, (len(panel.prices), 1)), index=panel.prices.index,
                        columns=panel.prices.columns)
    caps = caps * (1.0 + 0.01 * np.arange(len(caps)))[:, None]  # caps drift with time
    return MarketPanel(snapshot_id=panel.snapshot_id, prices=panel.prices, funding=panel.funding,
                       tradeable=panel.tradeable, meta={"market_cap": caps}, volume=panel.volume)


_CAP_PARAMS = {"portfolio": {"block": "long_short_fraction", "fraction": 0.5},
               "sizing": {"block": "cap_weight", "gross": 1.0}, "rebalance_days": 5}


def test_cap_weight_weights_coins_by_cap_inside_each_side() -> None:
    panel = _with_caps(_walk_panel())
    weights = Composed().target_weights(panel, _params(**_CAP_PARAMS))
    validate_weights(panel, weights)
    caps = panel.meta["market_cap"]
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50
    assert held.sum(axis=1).abs().max() < 1e-12  # neutral
    assert held.abs().sum(axis=1).tolist() == pytest.approx([1.0] * len(held))
    row = held.iloc[0]
    ts = held.index[0]
    for side in (row[row > 0], -row[row < 0]):
        assert side.sum() == pytest.approx(0.5)
        # inside a side, weight is proportional to cap
        ratio = side / caps.loc[ts, side.index]
        assert ratio.to_numpy() == pytest.approx(np.full(len(side), ratio.iloc[0]))
    assert row[row > 0].nunique() > 1  # not equal weights


def test_cap_change_after_t_does_not_change_weights_at_t() -> None:
    panel = _with_caps(_walk_panel(n=150))
    params = _params(**_CAP_PARAMS)
    base = Composed().target_weights(panel, params)
    caps = panel.meta["market_cap"].copy()
    later = caps.iloc[81:]
    caps.iloc[81:] = later * np.random.default_rng(8).uniform(0.05, 20, later.shape)
    changed = MarketPanel(snapshot_id="c", prices=panel.prices, funding=panel.funding,
                          tradeable=panel.tradeable, meta={"market_cap": caps},
                          volume=panel.volume)
    again = Composed().target_weights(changed, params)
    pd.testing.assert_frame_equal(base.iloc[:81], again.iloc[:81])
    assert not base.iloc[81:].equals(again.iloc[81:])  # the later caps do matter later


def test_cap_change_after_t_in_the_stored_snapshots_does_not_change_weights_at_t(
        tmp_path, monkeypatch) -> None:
    import json

    from qlab.data.sources import coinmarketcap

    panel = _walk_panel(n=100)  # daily bars from 2026-01-01
    monkeypatch.setattr(coinmarketcap, "DEFAULT_STORE", tmp_path)

    def snapshot(day: str, scale: float) -> None:
        rows = [{"symbol": c, "cmcRank": i + 1, "market_cap_usd": scale * (10.0 + 7 * i),
                 "tags": []} for i, c in enumerate(COLUMNS)]
        (tmp_path / f"{day}.json").write_text(json.dumps({"date": day, "rows": rows}))

    snapshot("2025-12-28", 1.0)
    snapshot("2026-02-01", 1.0)
    params = _params(**_CAP_PARAMS, min_history_days=3)
    before = Composed().target_weights(panel, params)
    assert before.abs().sum().sum() > 0
    # A later snapshot is added, and an existing future one is rewritten: bars up to and
    # including the snapshot date (2026-03-01 is bar 59) cannot change.
    snapshot("2026-02-01", 5.0)
    rows = [{"symbol": c, "cmcRank": i + 1, "market_cap_usd": 1000.0 - 90 * i, "tags": []}
            for i, c in enumerate(COLUMNS)]
    (tmp_path / "2026-03-01.json").write_text(json.dumps({"date": "2026-03-01", "rows": rows}))
    after = Composed().target_weights(panel, params)
    cut = panel.prices.index.get_loc(pd.Timestamp("2026-03-01", tz="UTC"))
    pd.testing.assert_frame_equal(before.iloc[: cut + 1], after.iloc[: cut + 1])
    assert not before.iloc[cut + 1 :].equals(after.iloc[cut + 1 :])


def test_no_look_ahead_with_cap_weight() -> None:
    panel = _with_caps(_walk_panel(n=150))
    params = _params(**_CAP_PARAMS)
    strategy = Composed()
    weights = strategy.target_weights(panel, params)
    assert weights.abs().sum().sum() > 0
    assert lookahead_violation(strategy, panel, params, weights) is None


def test_cap_weight_without_cap_data_says_how_to_get_it(tmp_path, monkeypatch) -> None:
    from qlab.data.sources import coinmarketcap

    monkeypatch.setattr(coinmarketcap, "DEFAULT_STORE", tmp_path / "none")
    with pytest.raises(ValueError, match="cap_weight.*coin-attributes"):
        Composed().target_weights(_walk_panel(), _params(**_CAP_PARAMS))


# ---- spot_perp_carry: a coin against its perpetual -------------------------------------


SPOT = SPOT_COLUMN_SUFFIX
FUNDING_LEVEL = {"A": 1e-4, "B": 5e-4, "C": 3e-4, "D": -2e-4, "E": 2e-4, "F": 9e-4}


def _carry_panel(n: int = 120, seed: int = 11, spot_coins: str = "ABCDE") -> MarketPanel:
    """The walk panel plus a spot column for `spot_coins`; funding differs by coin so the
    ranking is stable (B, C and E pay a short the most among coins that have spot)."""
    base = _walk_panel(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    prices = base.prices.copy()
    funding = base.funding.copy()
    for coin, level in FUNDING_LEVEL.items():
        funding[coin] = level + rng.normal(0.0, 1e-5, n)
    for coin in spot_coins:
        prices[f"{coin}{SPOT}"] = base.prices[coin] * (1.0 + rng.normal(0.0, 0.001, n))
    funding = funding.reindex(columns=prices.columns)  # spot funding stays NaN
    return MarketPanel(snapshot_id="carry", prices=prices, funding=funding,
                       tradeable=prices.notna(), meta={}, volume=base.volume.reindex(
                           columns=prices.columns))


_CARRY = {"signal": {"block": "funding", "window_days": 5},
          "portfolio": {"block": "spot_perp_carry", "top_n": 2},
          "rebalance_days": 7, "min_history_days": 3}


def test_carry_holds_spot_long_against_perp_short_of_the_highest_funding_coins() -> None:
    panel = _carry_panel()
    weights = Composed().target_weights(panel, _params(**_CARRY))
    validate_weights(panel, weights)
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 80
    # B and C pay a short the most among coins with a spot market (F pays more but has none)
    assert held[["B", "C"]].to_numpy() == pytest.approx(-0.25)
    assert held[[f"B{SPOT}", f"C{SPOT}"]].to_numpy() == pytest.approx(0.25)
    others = [c for c in weights.columns if c not in ("B", "C", f"B{SPOT}", f"C{SPOT}")]
    assert (weights[others] == 0.0).all().all()
    assert held.sum(axis=1).abs().max() < 1e-12
    assert held.abs().sum(axis=1).tolist() == pytest.approx([1.0] * len(held))


def test_carry_gross_scales_with_sizing_and_each_coin_is_market_neutral() -> None:
    panel = _carry_panel()
    weights = Composed().target_weights(
        panel, _params(**_CARRY, sizing={"block": "equal", "gross": 2.0}))
    held = weights[weights.abs().sum(axis=1) > 0]
    assert held.abs().sum(axis=1).tolist() == pytest.approx([2.0] * len(held))
    for coin in "BC":
        assert (held[coin] + held[f"{coin}{SPOT}"]).abs().max() < 1e-12


def test_carry_needs_both_markets_tradeable_and_priced_at_the_decision_bar() -> None:
    panel = _carry_panel()
    base = Composed().target_weights(panel, _params(**_CARRY))
    # Row 14 is a rebalance row (0, 7, 14, ...). B's spot is untradeable there; C's spot has
    # no price there: neither may be chosen, so the next best (E, then A) take their places.
    tradeable = panel.tradeable.copy()
    tradeable.loc[tradeable.index[14], f"B{SPOT}"] = False
    prices = panel.prices.copy()
    prices.loc[prices.index[14], f"C{SPOT}"] = np.nan
    cut = MarketPanel(snapshot_id="cut", prices=prices, funding=panel.funding,
                      tradeable=tradeable.where(prices.notna(), False), meta={},
                      volume=panel.volume)
    weights = Composed().target_weights(cut, _params(**_CARRY))
    validate_weights(cut, weights)
    # the weights decided on row 14 are carried to row 20
    chosen = weights.iloc[15][weights.iloc[15] != 0].index.tolist()
    assert sorted(chosen) == sorted(["E", "A", f"E{SPOT}", f"A{SPOT}"])
    assert base.iloc[15][["B", "C"]].abs().sum() > 0  # and untouched, B and C were the picks


def test_carry_closes_both_legs_when_one_market_stops_trading() -> None:
    panel = _carry_panel()
    base = Composed().target_weights(panel, _params(**_CARRY))
    assert base.iloc[30]["B"] != 0.0  # row 28 decided; rows 29-31 only carry it
    tradeable = panel.tradeable.copy()
    tradeable.loc[tradeable.index[29:32], f"B{SPOT}"] = False  # spot halted, perpetual open
    cut = MarketPanel(snapshot_id="halt", prices=panel.prices, funding=panel.funding,
                      tradeable=tradeable, meta={}, volume=panel.volume)
    weights = Composed().target_weights(cut, _params(**_CARRY))
    validate_weights(cut, weights)
    assert (weights.iloc[29:32][["B", f"B{SPOT}"]] == 0.0).all().all()  # no half-open pair
    assert weights.iloc[32]["B"] == base.iloc[32]["B"]  # and it comes back, unchanged
    assert weights.iloc[32][f"B{SPOT}"] == base.iloc[32][f"B{SPOT}"]


def test_carry_without_spot_columns_asks_for_include_spot() -> None:
    with pytest.raises(ValueError, match="include_spot"):
        Composed().target_weights(_walk_panel(), _params(**_CARRY))


def test_carry_needs_coins_with_a_spot_market() -> None:
    panel = _carry_panel(spot_coins="D")  # only D has spot, and its funding is negative
    weights = Composed().target_weights(panel, _params(**_CARRY))
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 50  # one eligible coin is enough: k = min(2, 1)
    assert held["D"].tolist() == pytest.approx([-0.5] * len(held))
    assert held[f"D{SPOT}"].tolist() == pytest.approx([0.5] * len(held))


def test_spot_columns_stay_flat_for_every_other_portfolio_on_a_spot_panel() -> None:
    panel = _carry_panel()
    spot_columns = [c for c in panel.prices.columns if str(c).endswith(SPOT)]
    for portfolio in ({"block": "long_short", "top_n": 2},
                      {"block": "long_short_fraction", "fraction": 0.4},
                      {"block": "time_series_sign"}):
        weights = Composed().target_weights(panel, _params(portfolio=portfolio))
        assert (weights[spot_columns] == 0.0).all().all()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"signal": {"block": "funding", "window_days": 3}, "rebalance_days": 3},
        {"portfolio": {"block": "spot_perp_carry", "top_n": 1},
         "sizing": {"block": "vol_target", "window_days": 10, "target_daily": 0.02,
                    "leverage_cap": 2.0}},
        {"universe": {"rule": "top_volume", "n": 4, "window_days": 5}},
    ],
)
def test_no_look_ahead_for_carry(overrides: dict) -> None:
    panel = _carry_panel(n=150)
    params = _params(**{**_CARRY, **overrides})
    strategy = Composed()
    weights = strategy.target_weights(panel, params)
    assert weights.abs().sum().sum() > 0
    assert lookahead_violation(strategy, panel, params, weights) is None


def test_carry_with_cap_weight_splits_the_perpetual_side_by_cap_and_mirrors_it_in_spot() -> None:
    panel = _carry_panel()
    caps = pd.DataFrame(np.tile([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                (len(panel.prices), 1))[:, : len(panel.prices.columns)],
                        index=panel.prices.index, columns=panel.prices.columns)
    caps["C"] = 1.0  # B (the larger) is 3 : 1 against C
    with_caps = MarketPanel(snapshot_id="cc", prices=panel.prices, funding=panel.funding,
                            tradeable=panel.tradeable, meta={"market_cap": caps},
                            volume=panel.volume)
    weights = Composed().target_weights(
        with_caps, _params(**_CARRY, sizing={"block": "cap_weight", "gross": 1.0}))
    validate_weights(with_caps, weights)
    held = weights[weights.abs().sum(axis=1) > 0]
    assert len(held) > 80
    assert held["B"].tolist() == pytest.approx([-0.375] * len(held))
    assert held["C"].tolist() == pytest.approx([-0.125] * len(held))
    assert held[f"B{SPOT}"].tolist() == pytest.approx([0.375] * len(held))
    assert held[f"C{SPOT}"].tolist() == pytest.approx([0.125] * len(held))
