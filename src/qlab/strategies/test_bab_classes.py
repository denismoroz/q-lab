"""Tests for the optional asset-class params of `qlab.strategies.bab` (docs/ASSET_CLASSES.md):
off unless asked, point in time, and ranking inside a class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from qlab.data import asset_classes as ac
from qlab.harness.panel import MarketPanel
from qlab.strategies.bab import BettingAgainstBeta, formation_rows, pooled_rank_weights
from qlab.strategies.test_bab import PARAMS, _panel

GROUPS = {
    "A": "stock:US", "B": "stock:US", "C": "stock:US",
    "D": "stock:KR", "E": "stock:KR",
    "F": "fund",
}
KINDS = {"stock:US": "stock", "stock:KR": "stock", "fund": "fund"}


def _class_table(tmp_path, classes: dict[str, str], listed: str = "2025-01-01") -> str:
    """A class table file where `classes` maps instrument -> ranking group key."""
    document = {
        "version": ac.TABLE_FORMAT_VERSION,
        "venue": "test",
        "rule_version": "t",
        "instruments": {
            name: {
                "listed_from": listed,
                "history": [{
                    "valid_from": listed,
                    "asset_class": KINDS[key],
                    "class_key": key,
                }],
            }
            for name, key in classes.items()
        },
    }
    path = tmp_path / "classes.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return str(path)


def test_without_class_params_the_book_is_what_it_was(tmp_path) -> None:
    panel = _panel()
    plain = BettingAgainstBeta().target_weights(panel, PARAMS)
    off = BettingAgainstBeta().target_weights(
        panel, {**PARAMS, "class_table": _class_table(tmp_path, GROUPS)}
    )
    pd.testing.assert_frame_equal(plain, off)


def test_the_class_filter_removes_instruments_outside_the_listed_classes(tmp_path) -> None:
    panel = _panel()
    weights = BettingAgainstBeta().target_weights(
        panel, {**PARAMS, "classes": ["stock"], "class_table": _class_table(tmp_path, GROUPS)}
    )
    held = weights.iloc[formation_rows(panel.prices.index)[-1]]
    assert held["F"] == 0.0  # a fund, not a stock
    assert (held[["A", "B", "C", "D", "E"]] != 0.0).any()


def test_a_listed_class_the_table_does_not_hold_leaves_no_book(tmp_path) -> None:
    panel = _panel()
    weights = BettingAgainstBeta().target_weights(
        panel, {**PARAMS, "classes": ["commodity"], "class_table": _class_table(tmp_path, GROUPS)}
    )
    assert (weights == 0.0).all().all()


def test_instruments_are_out_before_their_listing_in_the_table(tmp_path) -> None:
    panel = _panel()
    late = _class_table(tmp_path, GROUPS, listed="2030-01-01")
    weights = BettingAgainstBeta().target_weights(
        panel, {**PARAMS, "classes": ["stock"], "class_table": late}
    )
    assert (weights == 0.0).all().all()


def test_pooled_ranking_sides_sum_to_one_and_follow_group_size() -> None:
    betas = pd.Series({"A": 0.4, "B": 0.9, "C": 1.4, "D": 0.7, "E": 1.3, "F": 1.0})
    low, high = pooled_rank_weights(betas, pd.Series(GROUPS))
    assert low.sum() == pytest.approx(1.0) and high.sum() == pytest.approx(1.0)
    # F is alone in its group: it is not ranked.
    assert low["F"] == 0.0 and high["F"] == 0.0
    # Inside a group the lowest beta is on the low side and the highest on the high side.
    assert low["A"] > 0 and high["A"] == 0.0 and high["C"] > 0 and low["C"] == 0.0
    assert low["D"] > 0 and high["E"] > 0
    # The three-name group weighs 3/5 of each side, the two-name group 2/5.
    assert low[["A", "B", "C"]].sum() == pytest.approx(0.6)
    assert high[["D", "E"]].sum() == pytest.approx(0.4)


def test_pooled_ranking_with_no_group_of_two_holds_nothing() -> None:
    betas = pd.Series({"A": 0.4, "D": 0.7})
    low, high = pooled_rank_weights(betas, pd.Series({"A": "x", "D": "y"}))
    assert low.sum() == 0.0 and high.sum() == 0.0


def test_rank_within_class_ranks_inside_each_group(tmp_path) -> None:
    panel = _panel()
    table = _class_table(tmp_path, GROUPS)
    pooled = BettingAgainstBeta().target_weights(
        panel, {**PARAMS, "classes": ["stock"], "class_table": table}
    )
    within = BettingAgainstBeta().target_weights(
        panel,
        {**PARAMS, "classes": ["stock"], "rank_within_class": True, "class_table": table},
    )
    t = formation_rows(panel.prices.index)[-1]
    # True betas D=1.2 and E=1.5: inside the KR group D is the low beta, E the high one.
    assert within.iloc[t]["D"] > 0 > within.iloc[t]["E"]
    # Pooled with the US stocks (A .3, B .6, C .9), D and E are both on the high side.
    assert pooled.iloc[t]["D"] < 0
    assert not within.equals(pooled)


def test_class_weights_never_see_the_future(tmp_path) -> None:
    """Scrambling prices after a formation must not move an earlier weight, with the class
    filter and the within-class rank on."""
    panel = _panel()
    params = {
        **PARAMS, "classes": ["stock"], "rank_within_class": True,
        "class_table": _class_table(tmp_path, GROUPS),
    }
    cut = formation_rows(panel.prices.index)[-3]
    base = BettingAgainstBeta().target_weights(panel, params)
    rng = np.random.default_rng(3)
    scrambled = panel.prices.copy()
    scrambled.iloc[cut + 1 :] = rng.uniform(1, 200, scrambled.iloc[cut + 1 :].shape)
    other = MarketPanel(
        snapshot_id="scrambled", prices=scrambled, funding=panel.funding,
        tradeable=panel.tradeable, meta=panel.meta,
    )
    after = BettingAgainstBeta().target_weights(other, params)
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], after.iloc[: cut + 1])
