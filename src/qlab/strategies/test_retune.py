"""Tests for `qlab.strategies.retune`: the grid, the re-tuning calendar, and
above all that every choice uses only returns realised before it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.harness.panel import MarketPanel
from qlab.strategies.retune import Retune, expand_grid, refit_rows


class ConstantSign:
    """Holds `sign` * 0.5 in column A from bar `start` on."""

    name = "constant-sign"

    def target_weights(self, panel, params):
        w = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        w.iloc[int(params.get("start", 0)):, 0] = 0.5 * float(params["sign"])
        return w


def _panel(up_days: int, down_days: int) -> MarketPanel:
    n = up_days + down_days
    index = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    step = np.r_[np.full(up_days, 0.01), np.full(down_days, -0.01)]
    rng = np.random.default_rng(5)
    a = 100 * np.cumprod(1 + step + rng.normal(0, 0.002, n))
    prices = pd.DataFrame({"A": a, "B": 100.0}, index=index)
    return MarketPanel(
        snapshot_id=f"retune-{up_days}-{down_days}", prices=prices,
        funding=pd.DataFrame(0.0, index=index, columns=prices.columns),
        tradeable=pd.DataFrame(True, index=index, columns=prices.columns), meta={},
    )


PARAMS = {
    "inner_code_ref": "qlab.strategies.test_retune:ConstantSign",
    "base_params": {},
    "grid": {"side": {"long": {"sign": 1}, "short": {"sign": -1}}},
    "refit_every": "W",
    "window": "28D",
    "objective": "sharpe_net",
    "score_costs": {"taker_fee_bps": 1.0, "slippage_bps": 1.0},
}


def test_expand_grid_is_the_cartesian_product() -> None:
    grid = {"u": {"a": {"x": 1}, "b": {"x": 2}}, "l": {"p": {"y": 1}, "q": {"y": 2}}}
    out = expand_grid({"z": 0}, grid)
    assert set(out) == {"a/p", "a/q", "b/p", "b/q"}
    assert out["b/q"] == {"z": 0, "x": 2, "y": 2}
    with pytest.raises(ValueError, match="two candidates"):
        expand_grid({}, {"u": {"a": {}}})


def test_refit_rows_mark_the_first_bar_of_each_week() -> None:
    index = pd.date_range("2026-01-01", periods=15, freq="1D", tz="UTC")  # a Thursday
    rows = refit_rows(index, "W")
    assert list(index[rows].day_name()) == ["Monday", "Monday"]


def test_choice_follows_the_trend_with_a_lag_and_never_sees_the_future() -> None:
    panel = _panel(60, 60)
    run = Retune().run(panel, PARAMS)
    chosen = run.choices.dropna()
    assert chosen.loc[:"2026-02-28"].eq("long").all()
    assert chosen.loc["2026-04-15":].eq("short").all()

    # Tamper with everything after a cut: choices made up to the cut must not move.
    cut = pd.Timestamp("2026-02-20", tz="UTC")
    future = panel.prices.copy()
    future.loc[future.index > cut, "A"] *= np.linspace(1.0, 3.0, int((future.index > cut).sum()))
    tampered = MarketPanel(snapshot_id="retune-tampered", prices=future, funding=panel.funding,
                           tradeable=panel.tradeable, meta={})
    again = Retune().run(tampered, PARAMS)
    before = run.choices[run.choices.index <= cut]
    pd.testing.assert_series_equal(before, again.choices[again.choices.index <= cut])
    pd.testing.assert_frame_equal(run.weights.loc[:cut], again.weights.loc[:cut])


def test_no_candidate_is_chosen_before_it_traded_a_whole_period() -> None:
    panel = _panel(60, 0)
    params = {**PARAMS, "base_params": {"start": 30}}
    run = Retune().run(panel, params)
    first = run.choices.dropna().index[0]
    assert first > panel.prices.index[30] + pd.Timedelta(days=6)
    assert (run.weights.loc[: first - pd.Timedelta(days=1)] == 0).all().all()


def test_diagnostics_count_switches() -> None:
    strategy = Retune()
    strategy.run(_panel(60, 60), PARAMS)
    d = strategy.diagnostics()
    assert d["retune_candidates"] == 2
    assert d["retune_switches"] >= 1
    assert 0 < d["retune_top_choice_share"] <= 1
