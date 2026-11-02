"""Tests for the `funding_tail` signal and the `time_series_long` portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.agents.cataloguer import known_params
from qlab.harness.panel import MarketPanel
from qlab.strategies.blocks import Composed
from qlab.strategies.blocks._schema import parse_block
from qlab.strategies.blocks.portfolio import PORTFOLIOS, time_series_long
from qlab.strategies.blocks.signals import SIGNALS, _funding_position, funding_tail

NAN = np.nan


def _frame(rows: list[list[float]], columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or list("ABCDEF")[: len(rows[0])]
    index = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index, columns=columns, dtype=float)


def _panel(funding_rows: list[list[float]]) -> MarketPanel:
    funding = _frame(funding_rows)
    prices = pd.DataFrame(100.0, index=funding.index, columns=funding.columns)
    return MarketPanel(snapshot_id="toy", prices=prices, funding=funding,
                       tradeable=pd.DataFrame(True, index=funding.index, columns=funding.columns),
                       meta={}, volume=None)


def _column(values: list[float]) -> list[list[float]]:
    return [[v] for v in values]


_RANK_ROWS = [.02, .03, .04, .05, .01, .06, .045]


def test_position_ranks_the_current_rate_among_the_previous_bars_only() -> None:
    out = _funding_position(_panel(_column(_RANK_ROWS)), history_window_days=4)["A"]
    assert out.iloc[:4].isna().all()
    assert out.iloc[4] == pytest.approx(0.0)
    assert out.iloc[5] == pytest.approx(1.0)
    assert out.iloc[6] == pytest.approx(0.5)  # of .04, .05, .01, .06 two are below .045


def test_position_counts_a_tie_as_half() -> None:
    out = _funding_position(_panel(_column([.01, .02, .02, .03, .02])), history_window_days=4)
    assert out["A"].iloc[4] == pytest.approx(0.5)  # (1 below + 0.5 * 2 equal) / 4


def test_a_rate_that_never_moves_is_in_neither_tail() -> None:
    panel = _panel(_column([.0001] * 6))
    position = _funding_position(panel, history_window_days=4)["A"]
    score = funding_tail(panel, history_window_days=4, tail_fraction=0.25)["A"]
    assert position.iloc[4:].tolist() == [0.5, 0.5]
    assert score.iloc[4:].tolist() == [0.0, 0.0]


def test_tail_scores_plus_one_low_minus_one_high_zero_between() -> None:
    out = funding_tail(_panel(_column(_RANK_ROWS)), history_window_days=4, tail_fraction=0.25)["A"]
    assert out.iloc[:4].isna().all()
    assert out.iloc[4:].tolist() == [1.0, -1.0, 0.0]


def test_tail_boundary_belongs_to_the_tail() -> None:
    rows = [round(0.01 * i, 2) for i in range(1, 11)] + [.095]
    out = funding_tail(_panel(_column(rows)), history_window_days=10, tail_fraction=0.1)["A"]
    assert out.iloc[10] == -1.0  # position 9 / 10 = 1 - tail_fraction
    both = funding_tail(_panel(_column([.01, .02, .03, .04, .025])), history_window_days=4,
                        tail_fraction=0.5)["A"]
    assert both.iloc[4] == 0.0  # position .5 is on both boundaries: the tails cancel


def test_an_unknown_bar_blanks_the_window() -> None:
    out = funding_tail(_panel(_column([.01, NAN, .03, .04, .02, .05])), history_window_days=2,
                       tail_fraction=0.25)["A"]
    assert out.isna().tolist() == [True, True, True, True, False, False]
    assert out.iloc[4:].tolist() == [1.0, -1.0]


def test_tail_reads_only_the_past() -> None:
    full = funding_tail(_panel(_column(_RANK_ROWS)), history_window_days=4, tail_fraction=0.25)
    cut = funding_tail(_panel(_column(_RANK_ROWS[:5])), history_window_days=4, tail_fraction=0.25)
    pd.testing.assert_frame_equal(full.iloc[:5], cut)


def test_time_series_long_takes_positive_scores_in_parts_of_the_eligible() -> None:
    scores = _frame([[1.0, 0.0, -1.0, NAN]])
    eligible = pd.DataFrame([[True, True, True, False]], index=scores.index,
                            columns=scores.columns)
    out = time_series_long(scores, eligible)
    assert out.iloc[0].tolist() == pytest.approx([1 / 3, 0.0, 0.0, 0.0])


def test_time_series_long_holds_while_positive_and_leaves_on_return() -> None:
    scores = _frame(_column([0, 1, 1, 0, 1]))
    out = time_series_long(scores, scores.notna())
    assert out["A"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


def _rebound_panel() -> MarketPanel:
    n = 14
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    funding = pd.DataFrame(0.01, index=index, columns=list("ABC"))
    funding["A"] = [.02 + .001 * i for i in range(n)]
    funding.loc[index[10], "A"] = -.1
    prices = pd.DataFrame(100.0, index=index, columns=funding.columns)
    return MarketPanel(snapshot_id="rebound", prices=prices, funding=funding,
                       tradeable=prices.notna(), meta={}, volume=None)


def _rebound_params(portfolio: str) -> dict:
    return {
        "universe": {"rule": "all"},
        "signal": {"block": "funding_tail", "history_window_days": 5, "tail_fraction": 0.2},
        "portfolio": {"block": portfolio},
        "sizing": {"block": "equal", "gross": 1.0},
        "rebalance_days": 1,
        "min_history_days": 0,
    }


def test_composed_buys_the_low_tail_and_leaves_on_return() -> None:
    weights = Composed().target_weights(_rebound_panel(), _rebound_params("time_series_long"))
    expected = [0.0] * 14
    expected[10] = 1 / 3
    assert weights["A"].tolist() == pytest.approx(expected)
    assert weights[["B", "C"]].eq(0.0).all().all()


def test_composed_time_series_sign_trades_both_tails() -> None:
    weights = Composed().target_weights(_rebound_panel(), _rebound_params("time_series_sign"))
    expected = [0.0] * 5 + [-1 / 3] * 5 + [1 / 3] + [-1 / 3] * 3
    assert weights["A"].tolist() == pytest.approx(expected)
    assert weights[["B", "C"]].eq(0.0).all().all()


def test_the_catalogue_rows_are_parameters_of_the_block() -> None:
    assert {"signal.history_window_days", "signal.tail_fraction"} <= known_params()


def test_the_rule_needs_no_number_outside_the_catalogue() -> None:
    fn, kwargs = parse_block("portfolio", {"block": "time_series_long"}, PORTFOLIOS, "block")
    assert fn is time_series_long
    assert kwargs == {}


def test_tail_parameters_are_validated() -> None:
    good = {"block": "funding_tail", "history_window_days": 180, "tail_fraction": 0.1}
    parse_block("signal", good, SIGNALS, "block")
    for key in good:
        if key != "block":
            with pytest.raises(ValueError, match=key):
                parse_block("signal", {k: v for k, v in good.items() if k != key}, SIGNALS, "block")
    for bad in (0, 0.6, True, float("nan")):
        with pytest.raises(ValueError, match="tail_fraction"):
            parse_block("signal", dict(good, tail_fraction=bad), SIGNALS, "block")
    with pytest.raises(ValueError, match="history_window_days"):
        parse_block("signal", dict(good, history_window_days=0), SIGNALS, "block")
