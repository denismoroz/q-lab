"""How many bars a spot leg was valued at a carried close (qlab.spot_fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.spot_fallback import fallback_sentence, spot_fallback_counts, spot_fallback_metrics

HOURS = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")


def _panel(meta: dict[str, object], perp: list[float] | None = None) -> MarketPanel:
    prices = pd.DataFrame({
        "A": perp if perp is not None else [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        # prints on bars 1 and 4 only; bar 0 is before the first print
        "A-SPOT": [np.nan, 101.5, np.nan, np.nan, 104.5, np.nan],
        # a second spot that never prints in the window
        "B": 50.0,
        "B-SPOT": np.nan,
    }, index=HOURS)
    return MarketPanel(snapshot_id="m", prices=prices,
                       funding=pd.DataFrame(0.0, index=HOURS, columns=prices.columns),
                       tradeable=prices.notna(), meta=meta)


def _held(*columns: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=HOURS, columns=["A", "A-SPOT", "B", "B-SPOT"])
    for column in columns:
        weights[column] = 0.5
    return weights


def test_carried_bars_are_the_bars_after_the_first_print_without_a_print() -> None:
    # prints: bars 1 and 4 (2). Carried: bars 2, 3 and 5 (3). Bar 0 is before the first print:
    # nothing is held there, it is not a fallback.
    counts = spot_fallback_counts(_panel({"spot_prices": "own"}), _held("A-SPOT"))
    assert counts == {"A-SPOT": (2, 3)}


def test_a_bar_where_the_perp_has_no_price_is_not_counted() -> None:
    # bar 5's perp is gone (delisted): the coin is not alive there, so its missing spot print
    # is not a carried bar. Carried: bars 2 and 3.
    perp = [100.0, 101.0, 102.0, 103.0, 104.0, np.nan]
    counts = spot_fallback_counts(_panel({"spot_prices": "own"}, perp), _held("A-SPOT"))
    assert counts == {"A-SPOT": (2, 2)}


def test_a_spot_the_book_never_holds_is_not_counted() -> None:
    counts = spot_fallback_counts(_panel({"spot_prices": "own"}), _held("A"))
    assert counts == {}
    both = spot_fallback_counts(_panel({"spot_prices": "own"}), _held("A-SPOT", "B-SPOT"))
    assert both == {"A-SPOT": (2, 3), "B-SPOT": (0, 0)}
    assert set(spot_fallback_counts(_panel({"spot_prices": "own"}))) == {"A-SPOT", "B-SPOT"}


def test_metrics_sum_the_counts_and_give_the_share() -> None:
    metrics = spot_fallback_metrics(_panel({"spot_prices": "own"}), _held("A-SPOT"))
    assert metrics == {"spot_prints": 2.0, "spot_carried": 3.0, "spot_carried_share": 0.6}
    both = spot_fallback_metrics(_panel({"spot_prices": "own"}), _held("A-SPOT", "B-SPOT"))
    assert both["spot_prints"] == 2.0 and both["spot_carried"] == 3.0


def test_a_perp_marked_panel_has_no_counts_and_no_metrics() -> None:
    """The identical-numbers guard: a spec that does not ask for own spot prices gains no key."""
    for meta in ({}, {"spot_prices": "perp"}):
        assert spot_fallback_counts(_panel(meta), _held("A-SPOT")) == {}
        assert spot_fallback_metrics(_panel(meta), _held("A-SPOT")) == {}


def test_the_sentence_states_both_counts_and_that_the_perp_price_is_not_used() -> None:
    metrics = spot_fallback_metrics(_panel({"spot_prices": "own"}), _held("A-SPOT"))
    sentence = fallback_sentence(metrics)
    assert sentence is not None
    assert "2 баров по собственной цене" in sentence and "3 (60.0%)" in sentence
    assert "цена контракта под именем спота не подставляется" in sentence
    assert fallback_sentence({}) is None


@pytest.mark.parametrize("kind", ["own"])
def test_the_count_looks_only_at_the_bars_of_the_panel_it_is_given(kind: str) -> None:
    """A part of a run (the judged window) is counted on its own bars: cutting the panel after
    bar 3 leaves the prints {1} and the carried bars {2, 3}."""
    panel = _panel({"spot_prices": kind})
    cut = MarketPanel(snapshot_id="m", prices=panel.prices.iloc[:4],
                      funding=panel.funding.iloc[:4], tradeable=panel.tradeable.iloc[:4],
                      meta=panel.meta)
    assert spot_fallback_counts(cut, _held("A-SPOT").iloc[:4]) == {"A-SPOT": (1, 2)}
