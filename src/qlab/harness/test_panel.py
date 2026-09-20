"""Tests for the MarketPanel structural stand-in."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel


def _frame(index: pd.DatetimeIndex, columns: list[str], value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=index, columns=columns)


def test_market_panel_accepts_matching_shapes() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    cols = ["A", "B"]
    panel = MarketPanel(
        snapshot_id="snap-1",
        prices=_frame(index, cols, 100.0),
        funding=_frame(index, cols, 0.0001),
        tradeable=_frame(index, cols, True),
        meta={},
    )
    assert panel.snapshot_id == "snap-1"
    assert panel.prices.shape == (3, 2)


def test_market_panel_rejects_funding_index_mismatch() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    other_index = pd.date_range("2026-02-01", periods=3, freq="D", tz="UTC")
    cols = ["A", "B"]
    with pytest.raises(ValueError, match="funding index"):
        MarketPanel(
            snapshot_id="snap-1",
            prices=_frame(index, cols),
            funding=_frame(other_index, cols),
            tradeable=_frame(index, cols, True),
            meta={},
        )


def test_market_panel_rejects_tradeable_columns_mismatch() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    cols = ["A", "B"]
    with pytest.raises(ValueError, match="tradeable columns"):
        MarketPanel(
            snapshot_id="snap-1",
            prices=_frame(index, cols),
            funding=_frame(index, cols),
            tradeable=_frame(index, ["A", "C"], True),
            meta={},
        )
