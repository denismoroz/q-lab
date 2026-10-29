"""`measured_min_capital`: the capital facts of one part of a run, from the orders a book sent
when it declares them and from the weights otherwise (docs/MIN_CAPITAL.md)."""

from __future__ import annotations

import pandas as pd
import pytest

from qlab.harness.strategy import PlacedOrders
from qlab.pipeline.evaluate import measured_min_capital

INDEX = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
COLUMNS = ["A", "A-SPOT"]


def _weights(value: float = 0.25) -> pd.DataFrame:
    return pd.DataFrame(value, index=INDEX, columns=COLUMNS)


def _orders(placed: dict[int, float], skipped: dict[int, float] | None = None) -> PlacedOrders:
    sizes = pd.DataFrame(0.0, index=INDEX, columns=COLUMNS)
    for row, share in placed.items():
        sizes.iloc[row, 0] = share
    waits = pd.Series(0.0, index=INDEX)
    for row, count in (skipped or {}).items():
        waits.iloc[row] = count
    return PlacedOrders(sizes=sizes, skipped=waits)


def test_without_orders_the_weights_decide_as_before() -> None:
    assert measured_min_capital(_weights(), _weights(), None, 0, 9, 10.0) == {
        "min_capital_usd": pytest.approx(40.0)}


def test_with_orders_they_decide_and_the_weights_number_stays_beside_them() -> None:
    out = measured_min_capital(_weights(0.001), _weights(0.001),
                               _orders({0: 0.5, 4: 0.2}, {2: 1.0, 7: 2.0}), 0, 9, 10.0)
    assert out["min_capital_usd"] == pytest.approx(50.0)  # $10 over the smaller order, 0.2
    assert out["min_capital_weights_usd"] == pytest.approx(10_000.0)
    assert out["min_capital_from_orders"] == 1.0
    assert out["orders_opening"] == 2.0
    assert out["orders_skipped"] == 3.0


def test_a_part_is_measured_by_its_own_orders_and_counts_its_own_skips() -> None:
    orders = _orders({0: 0.5, 8: 0.2}, {1: 4.0, 8: 1.0})
    early = measured_min_capital(_weights(), _weights().iloc[:5], orders, 0, 4, 10.0)
    late = measured_min_capital(_weights(), _weights().iloc[5:], orders, 5, 9, 10.0)
    assert early["min_capital_usd"] == pytest.approx(20.0) and early["orders_skipped"] == 4.0
    assert late["min_capital_usd"] == pytest.approx(50.0) and late["orders_skipped"] == 1.0


def test_a_part_that_opened_nothing_is_measured_by_the_whole_run() -> None:
    """The capital a strategy needs is the strategy's, not one part's."""
    orders = _orders({0: 0.5, 1: 0.25}, {9: 1.0})
    out = measured_min_capital(_weights(), _weights().iloc[5:], orders, 5, 9, 10.0)
    assert out["min_capital_usd"] == pytest.approx(40.0)
    assert out["orders_opening"] == 0.0
    assert out["orders_skipped"] == 1.0


def test_a_book_that_opened_no_leg_falls_back_to_the_weights() -> None:
    out = measured_min_capital(_weights(), _weights(), _orders({}), 0, 9, 10.0)
    assert out == {"min_capital_usd": pytest.approx(40.0)}


def test_a_book_that_held_nothing_still_raises() -> None:
    flat = _weights(0.0)
    with pytest.raises(ValueError, match="all-flat"):
        measured_min_capital(flat, flat, _orders({0: 0.5}), 0, 9, 10.0)
