"""The `asset_class` universe block (docs/ASSET_CLASSES.md): a class filter that is off unless a
spec names it, and that an existing spec cannot reach."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.data import asset_classes as ac
from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks._schema import Block
from qlab.strategies.blocks.composed import Composed
from qlab.strategies.blocks.universe import UNIVERSES, all_coins, asset_class


def _table() -> ac.ClassTable:
    def inst(key: str, kind: str, listed: str, delisted: str | None = None) -> dict:
        entry: dict = {
            "listed_from": listed,
            "history": [{"valid_from": listed, "asset_class": kind, "class_key": key}],
        }
        if delisted:
            entry["delisted_since"] = delisted
        return entry

    return ac.ClassTable({
        "version": ac.TABLE_FORMAT_VERSION,
        "venue": "test",
        "rule_version": "t",
        "instruments": {
            "S1": inst("stock:US", "stock", "2026-01-01"),
            "S2": inst("stock:US", "stock", "2026-01-04"),
            "S3": inst("stock:US", "stock", "2026-01-01", delisted="2026-01-04"),
            "F1": inst("fund", "fund", "2026-01-01"),
        },
    })


def _panel() -> MarketPanel:
    index = pd.date_range("2026-01-01", periods=6, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (6, 5)), axis=0),
        index=index, columns=["S1", "S2", "S3", "F1", "NEW"],
    )
    return MarketPanel(
        snapshot_id="t", prices=prices,
        funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
        tradeable=pd.DataFrame(True, index=index, columns=prices.columns),
        meta={},
    )


@pytest.fixture
def table(monkeypatch) -> ac.ClassTable:
    built = _table()
    monkeypatch.setattr(ac, "load_table", lambda *a, **k: built)
    return built


def test_the_block_is_registered_and_the_default_universe_is_unchanged() -> None:
    assert isinstance(UNIVERSES["asset_class"], Block)
    out = all_coins(_panel())
    assert out.all().all()


def test_only_instruments_in_the_class_at_that_bar_are_allowed(table) -> None:
    out = asset_class(_panel(), classes=["stock"])
    assert out["S1"].all()
    assert out["S2"].tolist() == [False, False, False, True, True, True]  # listed Jan 4
    assert out["S3"].tolist() == [True, True, True, False, False, False]  # delisted Jan 4
    assert not out["F1"].any() and not out["NEW"].any()  # a fund; not in the table


def test_a_key_selects_one_ranking_group(table) -> None:
    assert asset_class(_panel(), classes=["fund"])["F1"].all()
    assert not asset_class(_panel(), classes=["stock:KR"]).any().any()


def test_the_class_list_is_validated_before_anything_runs() -> None:
    check = UNIVERSES["asset_class"].params["classes"]
    assert check("classes", ["stock"]) == ["stock"]
    for bad in ([], "stock", [1], [""]):
        with pytest.raises(ValueError, match="classes"):
            check("classes", bad)


def test_a_composed_strategy_trades_only_the_listed_class(table) -> None:
    index = pd.date_range("2026-01-01", periods=60, freq="1D", tz="UTC")
    rng = np.random.default_rng(2)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (60, 4)), axis=0),
        index=index, columns=["S1", "S2", "F1", "NEW"],
    )
    panel = MarketPanel(
        snapshot_id="t", prices=prices,
        funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
        tradeable=pd.DataFrame(True, index=index, columns=prices.columns),
        meta={},
    )
    params = {
        "universe": {"rule": "asset_class", "classes": ["stock"]},
        "signal": {"block": "momentum", "lookback_days": 5},
        "portfolio": {"block": "long_short", "top_n": 1},
        "sizing": {"block": "equal", "gross": 1.0},
        "rebalance_days": 1,
        "min_history_days": 5,
    }
    weights = Composed().target_weights(panel, params)
    assert (weights[["F1", "NEW"]] == 0.0).all().all()
    assert (weights[["S1", "S2"]] != 0.0).any().any()
