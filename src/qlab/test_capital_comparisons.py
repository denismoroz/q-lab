"""Comparisons that must be like for like (docs/CAPITAL_BASE.md): stage 1 against stage 2, the
paper book against the stand, the windows of the decay check, the falls of the diagnosis, the
forward chart and the year view -- each on the strategy's own money, summed when the strategy
does not reinvest its profit."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from qlab import capital_text, decay, diagnose, forward_series, paper_check
from qlab.harness.metrics import capital_metrics
from qlab.harness.run import RunResult
from qlab.pipeline import confirm
from qlab.pipeline.test_confirm import _context, _metrics, _review
from qlab.year_view import capital_view, year_rows, year_view, years_line


def _daily(values, start="2026-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


# ---- decay: windows ------------------------------------------------------------------------


def test_summed_windows_are_gains_in_points_of_the_capital_not_ratios_to_a_grown_balance() -> None:
    # +1 point of the capital every day: a summed book gains the same 10 points in every window
    capital = _daily([0.01] * 60)
    additive = decay.rolling_windows(
        decay.daily_closes(diagnose.equity_from_returns(capital, additive=True)), 10,
        additive=True)
    ratio = decay.rolling_windows(
        decay.daily_closes(diagnose.equity_from_returns(capital, additive=True)), 10)
    assert all(w.total == pytest.approx(0.10) for w in additive)
    assert all(w.max_dd == 0.0 for w in additive)
    assert ratio[-1].total < ratio[0].total, "a ratio to the grown balance shrinks later gains"


def test_a_summed_fall_is_in_points_of_the_capital() -> None:
    path = np.array([1.0, 1.4, 1.1, 1.3])
    assert decay.max_drawdown(path, additive=True) == pytest.approx(-0.3)
    assert decay.max_drawdown(path) == pytest.approx(1.1 / 1.4 - 1)
    assert decay.daily_spread(np.array([1.0, 1.1, 1.0, 1.1]), additive=True) == pytest.approx(
        np.std([0.1, -0.1, 0.1], ddof=1))


def _paper_dollars(start: datetime, hours: int, dollars_per_hour: float, capital: float
                   ) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start), periods=hours + 1, freq="h")
    return pd.Series(capital + dollars_per_hour * np.arange(hours + 1), index=index)


def test_the_paper_book_and_the_stand_are_both_in_points_of_the_capital_when_it_sums() -> None:
    """The stand's history earned 0.5 point of the capital a day; the paper book (dollars, $257)
    earned $1.285 a day, the same 0.5 point. On the capital they agree exactly."""
    capital = 257.0
    stand_capital = _daily([0.005] * 60)
    stand_net = _daily([0.005 / (1 + 0.005 * k) for k in range(60)])  # shares of a growing book
    realised = _paper_dollars(datetime(2026, 3, 1, tzinfo=UTC), 24 * 10, 1.285 / 24, capital)
    result = decay.compare(
        idea_id="x", label="X", book="b", live=False, realised=realised,
        stand_net_return=stand_net, stand_capital_return=stand_capital, reinvested=False,
        strategy_capital=capital, bands=[], alert_percentile=None)
    assert result.on_capital and result.strategy_capital == capital
    assert result.ret.realised == pytest.approx(0.05)
    assert result.ret.stand.p50 == pytest.approx(0.05)
    assert (result.ret.stand.p10, result.ret.stand.p90) == pytest.approx((0.05, 0.05))
    assert any("от денег стратегии" in line for line in decay.result_lines(result))

    # the same two books read the old way disagree: the stand's windows are shares of its balance
    old = decay.compare(
        idea_id="x", label="X", book="b", live=False, realised=realised,
        stand_net_return=stand_net, bands=[], alert_percentile=None)
    assert not old.on_capital
    assert old.ret.stand.p50 != pytest.approx(0.05, abs=1e-4) or old.ret.percentile != 50.0


def test_a_summed_strategy_needs_its_capital_to_be_compared() -> None:
    realised = _paper_dollars(datetime(2026, 3, 1, tzinfo=UTC), 48, 0.1, 100.0)
    with pytest.raises(ValueError, match="strategy_capital"):
        decay.compare(idea_id="x", label="X", book="b", live=False, realised=realised,
                      stand_net_return=_daily([0.01] * 40), reinvested=False, bands=[],
                      alert_percentile=None)


def test_a_stand_function_that_knows_nothing_of_the_money_is_read_the_old_way() -> None:
    series = _daily([0.01, 0.02])
    run = decay.StandRun.of(series)
    assert run.net_return is series and run.capital_return is series and run.reinvested is True
    assert decay.StandRun.of(run) is run


# ---- diagnose ------------------------------------------------------------------------------


def test_the_diagnosis_reads_falls_and_months_on_the_capital_when_the_book_sums() -> None:
    values = [0.01] * 40 + [-0.02] * 10 + [0.01] * 40
    net = _daily(values)
    capital = diagnose.diagnose_returns(
        net, idea_id="x", params_fixed_at=None, share=0.2, regimes=None,
        capital_return=net, reinvested=False)
    stand = diagnose.diagnose_returns(net, idea_id="x", params_fixed_at=None, share=0.2,
                                      regimes=None)
    assert capital.on_capital and not stand.on_capital
    assert capital.drawdowns[0].depth == pytest.approx(-0.20)  # 10 days of -2 points
    assert stand.drawdowns[0].depth != pytest.approx(-0.20, abs=1e-3)
    assert capital.worst_months[0].ret < 0 < capital.worst_months[-1].ret + 1
    assert any("points of the capital" in line for line in diagnose.render(capital))


def test_equity_from_returns_adds_when_asked_and_compounds_otherwise() -> None:
    net = _daily([0.10, 0.10])
    assert list(diagnose.equity_from_returns(net, additive=True))[-1] == pytest.approx(1.2)
    assert list(diagnose.equity_from_returns(net))[-1] == pytest.approx(1.21)


# ---- stage 1 against stage 2 ---------------------------------------------------------------


def _bv2_like_stage1(days: int = 900, dollars: float = 0.0005, noise: float = 0.002,
                     seed: int = 5) -> confirm.Stage1History:
    """Stage 1 of a strategy that sets money aside: the stand's shares shrink as the balance
    grows (they are dollars over a balance four times the capital by the end), the capital
    shares do not."""
    index = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    capital_share = pd.Series(np.random.default_rng(seed).normal(dollars, noise, days), index=index)
    balance = 1.0 + 3.0 * np.arange(days) / days  # a balance growing to four times the capital
    return confirm.Stage1History(
        11, capital_share / balance, 0.95, capital_return=capital_share, reinvested=False)


def test_segments_of_a_summed_history_are_gains_in_points_of_the_capital() -> None:
    history = _bv2_like_stage1()
    on_capital = confirm.segments_of(history.capital_return, 100, additive=True)
    stand = confirm.segments_of(history.net_return, 100)
    assert len(on_capital.totals) == len(stand.totals)
    assert np.median(on_capital.totals) > 2 * np.median(stand.totals)
    assert np.median(on_capital.totals) == pytest.approx(0.0005 * 100, rel=0.35)


def test_stage_2_is_set_against_stage_1_on_the_same_base() -> None:
    """A fresh stage 2 book (balance = capital) that earned what stage 1's median segment earned
    on the capital: it sits in the middle on the capital, and far below the middle when set
    against stage 1's diluted shares (the 74 -> 42 of docs/research/percent-base)."""
    history = _bv2_like_stage1()
    median = float(np.median(confirm.segments_of(history.capital_return, 100,
                                                 additive=True).totals))
    metrics = _metrics(total_return_net=median, cap_total_return=median, cap_reinvested=0.0)
    review, found = _review(_context(metrics), history)
    assert found.on_capital and 40 < found.percentile < 60
    assert found.total_return == pytest.approx(median)
    assert review.metrics["stage1_on_capital"] == 1.0
    assert "both on the strategy's own money" in review.routing.reason

    old = dataclass_without_capital(history)
    _, found_old = _review(_context(_metrics(total_return_net=median)), old)
    assert not found_old.on_capital
    assert found_old.percentile > 90, "against diluted shares the same result looks great"


def dataclass_without_capital(history: confirm.Stage1History) -> confirm.Stage1History:
    return confirm.Stage1History(history.trial_id, history.net_return, history.noise_power)


def test_a_run_or_a_history_without_capital_numbers_is_compared_the_old_way() -> None:
    history = _bv2_like_stage1()
    metrics = _metrics(total_return_net=0.05)  # no cap_total_return: a trial from before
    _, found = _review(_context(metrics), history)
    assert not found.on_capital
    # a disagreement about how the money grows is not a comparison either
    metrics = _metrics(total_return_net=0.05, cap_total_return=0.05, cap_reinvested=1.0)
    _, found = _review(_context(metrics), history)
    assert not found.on_capital


def test_the_positivity_rule_still_reads_the_stand_s_total() -> None:
    history = _bv2_like_stage1()
    metrics = _metrics(total_return_net=-0.001, cap_total_return=0.05, cap_reinvested=0.0)
    review, found = _review(_context(metrics), history)
    assert found.status == "not-positive" and found.route == "reject"


# ---- the paper check -----------------------------------------------------------------------


def test_a_forward_total_on_capital_is_linear_when_the_book_does_not_reinvest() -> None:
    assert paper_check.forward_total_capital(0.073, 100, reinvested=False) == pytest.approx(
        0.073 * 100 / 365.25)
    assert paper_check.forward_total_capital(0.50, 100, reinvested=True) == pytest.approx(
        paper_check.forward_total(0.50, 100))


def test_a_window_return_on_capital_is_the_gain_over_the_money_not_over_the_first_row() -> None:
    index = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    curve = pd.Series([400.0, 410.0, 440.0], index=index)  # the book holds 400, the money is 257
    start = datetime(2026, 3, 1, tzinfo=UTC)
    assert paper_check.window_return(curve, start).total == pytest.approx(0.10)
    assert paper_check.window_return(curve, start, capital=257.0).total == pytest.approx(
        40.0 / 257.0)


# ---- the forward chart ---------------------------------------------------------------------


def test_the_forward_series_sums_the_days_of_a_book_that_does_not_reinvest() -> None:
    net = _daily([0.02, -0.01, 0.03, 0.01])
    capital = net * 2.0  # the base was 2: shares of the capital are twice the stand's
    payload = forward_series.build_payload(
        trial_id=1, idea_id="x", venue="binance", interval="1h", net_return=net,
        hold_return=_daily([0.01] * 4), capital_return=capital, reinvested=False,
        recorded_capital_total=0.10)
    assert payload["base"] == "strategy_capital" and payload["reinvested"] is False
    assert payload["total_return"] == pytest.approx(0.10)
    assert payload["matches_trial"] is True
    assert payload["points"][-1][1] == pytest.approx(0.10)
    assert payload["points"][-1][2] == pytest.approx(0.04), "the held coins are summed too"
    assert payload["stand_total_return"] == pytest.approx(1.02 * 0.99 * 1.03 * 1.01 - 1)
    assert payload["ann_return"] == pytest.approx(0.10 * 365.25 / 4)
    assert payload["max_dd"] == pytest.approx(-0.02)


def test_the_forward_series_without_capital_numbers_is_the_old_compounding() -> None:
    net = _daily([0.02, -0.01, 0.03, 0.01])
    payload = forward_series.build_payload(
        trial_id=1, idea_id="x", venue=None, interval="1d", net_return=net, hold_return=None,
        recorded_total=1.02 * 0.99 * 1.03 * 1.01 - 1)
    assert payload["base"] == "stand" and payload["reinvested"] is True
    assert payload["total_return"] == pytest.approx(1.02 * 0.99 * 1.03 * 1.01 - 1)
    assert payload["matches_trial"] is True


# ---- the year view -------------------------------------------------------------------------


def _metrics_with_years() -> dict[str, float]:
    idx = pd.date_range("2022-01-01", periods=1000, freq="D", tz="UTC")
    net = pd.Series(0.0003, index=idx)
    zero = pd.Series(0.0, index=idx)
    result = RunResult("s", net, net, zero, zero, zero, capital_return=net * 3.0,
                       capital_reinvested=False)
    from qlab.harness.metrics import year_breakdown

    out = dict(year_breakdown(net))
    out.update(capital_metrics(result))
    return out


def test_the_year_view_shows_the_capital_twin_beside_the_stand_s() -> None:
    m = _metrics_with_years()
    stand = {r.year: r.ret for r in year_rows(m)}
    capital = {r.year: r.ret for r in year_rows(capital_view(m))}
    assert set(stand) == set(capital) == {2022, 2023, 2024}
    assert capital[2022] == pytest.approx(0.0009 * 365)
    assert capital[2022] > stand[2022]
    view = year_view(m)
    assert view["capital"]["reinvested"] is False
    assert view["capital"]["years"][0]["return"] == pytest.approx(capital[2022])
    line = years_line(m)
    assert "На деньгах стратегии (прибыль не вкладывается обратно, годы складываются)" in line
    assert years_line(m, capital=False).count("На деньгах") == 0
    assert year_view({k: v for k, v in m.items() if not k.startswith("cap_")})["capital"] is None


def test_the_capital_line_is_left_out_when_it_says_nothing_new() -> None:
    idx = pd.date_range("2022-01-01", periods=1000, freq="D", tz="UTC")
    net = pd.Series(0.0003, index=idx)
    zero = pd.Series(0.0, index=idx)
    from qlab.harness.metrics import year_breakdown

    out = dict(year_breakdown(net))
    out.update(capital_metrics(RunResult("s", net, net, zero, zero, zero)))
    assert "На деньгах" not in years_line(out)


# ---- the words -----------------------------------------------------------------------------


def test_the_capital_suffix_is_added_only_where_the_capital_numbers_differ() -> None:
    same = {"ann_return_net": 0.05, "cap_ann_return": 0.05, "max_dd": -0.1, "cap_max_dd": -0.1,
            "cap_reinvested": 1.0}
    assert capital_text.capital_suffix(same) == ""
    assert capital_text.capital_suffix({}) == ""
    differs = {"ann_return_net": 0.02, "cap_ann_return": 0.084, "max_dd": -0.045,
               "cap_max_dd": -0.193, "cap_sharpe": 0.53, "cap_reinvested": 0.0}
    text = capital_text.capital_suffix(differs)
    assert "на деньгах стратегии" in text and "+8.4%" in text and "−19.3%" in text
    assert "считается суммой" in text
