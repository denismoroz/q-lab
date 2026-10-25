"""Tests for `qlab.diagnose`: drawdown evidence computed only before the cut."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from qlab import diagnose
from qlab.harness.panel import MarketPanel
from qlab.regimes import RegimeSeries

DAYS = pd.date_range("2025-01-01", periods=300, freq="1D", tz="UTC")


def _series(blocks: list[tuple[int, float]]) -> pd.Series:
    """Daily returns from (length, daily return) blocks, from 2025-01-01."""
    values = np.concatenate([np.full(n, r) for n, r in blocks])
    return pd.Series(values, index=DAYS[:len(values)])


def _drawdown_series() -> pd.Series:
    # 10 days +1%, 10 days -2% (the deepest), 10 days +3% (recovers on the 7th),
    # then 10 days -1% and 10 days -1% more that never recover.
    return _series([(10, 0.01), (10, -0.02), (10, 0.03), (20, -0.01)])


def test_deepest_drawdowns_find_peak_trough_recovery_and_depth() -> None:
    equity = diagnose.equity_from_returns(_drawdown_series())
    first, second = diagnose.deepest_drawdowns(equity)[:2]
    assert first.peak == date(2025, 1, 11)  # after 10 returns
    assert first.trough == date(2025, 1, 21)
    assert first.recovered == date(2025, 1, 28)  # 0.98**10 * 1.03**7 >= 1
    assert first.depth == pytest.approx(0.98 ** 10 - 1)
    assert second.recovered is None  # the last fall never regains its peak
    assert second.depth == pytest.approx(0.99 ** 20 - 1) or second.depth < 0
    assert [e.depth for e in diagnose.deepest_drawdowns(equity)] == sorted(
        e.depth for e in diagnose.deepest_drawdowns(equity))


def test_only_the_top_three_episodes_are_returned_deepest_first() -> None:
    blocks: list[tuple[int, float]] = []
    for depth in (-0.01, -0.05, -0.02, -0.04, -0.03):
        blocks += [(1, depth), (3, 0.5)]  # a fall of `depth`, then a full recovery
    episodes = diagnose.deepest_drawdowns(diagnose.equity_from_returns(_series(blocks)))
    assert [round(e.depth, 3) for e in episodes] == [-0.05, -0.04, -0.03]
    assert all(e.recovered is not None for e in episodes)


def test_a_rising_curve_has_no_drawdown() -> None:
    assert diagnose.deepest_drawdowns(diagnose.equity_from_returns(_series([(30, 0.01)]))) == []


def test_worst_months_are_calendar_months_of_the_period_end() -> None:
    returns = _series([(31, 0.001), (28, -0.01), (31, 0.002), (30, -0.003), (31, 0.001)])
    months = diagnose.worst_months(returns, n=2)
    assert [m.month for m in months] == ["2025-02", "2025-04"]
    # Jan 31's return ends on Feb 1, Feb 28's ends on Mar 1
    assert months[0].ret == pytest.approx(1.001 * 0.99 ** 27 - 1)
    assert len(diagnose.worst_months(returns, n=5)) == 5


def _regimes(bull_until: int) -> RegimeSeries:
    labels = pd.Series(["bull"] * bull_until + ["bear"] * (len(DAYS) - bull_until),
                       index=DAYS, dtype=object)
    return RegimeSeries(labels=labels, market_return=pd.Series(0.001, index=DAYS),
                        bear_below=-0.1, bull_above=0.1, source="test")


def test_returns_by_regime_use_the_label_of_the_period_end() -> None:
    returns = _series([(20, 0.01), (20, -0.02)])
    rows = {r.regime: r for r in diagnose.regime_returns(returns, _regimes(20), 365.25)}
    # returns 0..18 end on bull days 1..19; return 19 ends on day 20, already bear
    assert rows["bull"].ret == pytest.approx(1.01 ** 19 - 1)
    assert rows["bear"].ret == pytest.approx(1.01 * 0.98 ** 20 - 1)
    assert rows["flat"].ret is None and rows["flat"].days == 0.0
    assert rows["bull"].btc_return == pytest.approx(1.001 ** 19 - 1, abs=1e-3)


def test_the_cut_is_the_specs_params_fixed_at() -> None:
    cut = diagnose.choose_cut(DAYS, params_fixed_at=date(2025, 6, 1), share=1 / 3)
    assert cut.day == date(2025, 6, 1) and cut.row == 151
    assert "params_fixed_at 2025-06-01" in cut.source


def test_without_params_fixed_at_the_cut_leaves_the_held_out_share() -> None:
    cut = diagnose.choose_cut(DAYS, params_fixed_at=None, share=1 / 3)
    assert cut.row == 200 and cut.day == DAYS[200].date()
    assert "holdout_share" in cut.source


def test_a_fixed_date_after_the_window_falls_back_to_the_share() -> None:
    cut = diagnose.choose_cut(DAYS, params_fixed_at=date(2026, 1, 1), share=1 / 3)
    assert cut.row == 200 and "after the last bar" in cut.source


def test_a_causal_choosers_fixed_date_is_not_a_cut() -> None:
    cut = diagnose.choose_cut(DAYS, params_fixed_at=date(2025, 1, 20), share=1 / 3,
                              selects_causally=True)
    assert cut.row == 200 and "warm-up" in cut.source


def test_no_cut_available_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="no cut"):
        diagnose.choose_cut(DAYS, params_fixed_at=None, share=None)


def test_evidence_ignores_everything_after_the_cut() -> None:
    # A crash after the cut must not appear anywhere in the evidence.
    returns = _series([(100, 0.001), (50, -0.01), (50, 0.002), (50, -0.05), (50, -0.05)])
    seen = diagnose.diagnose_returns(returns, idea_id="x", params_fixed_at=date(2025, 7, 20),
                                     share=1 / 3, regimes=_regimes(150))
    assert seen.cut.day == date(2025, 7, 20) and seen.last == date(2025, 7, 20)
    assert seen.periods == 200
    assert seen.held_out_days == (DAYS[299].date() - date(2025, 7, 20)).days
    assert min(e.depth for e in seen.drawdowns) > -0.5  # the -5% blocks are held out
    assert all(m.month <= "2025-07" for m in seen.worst_months)
    by_share = diagnose.diagnose_returns(returns, idea_id="x", params_fixed_at=None,
                                         share=1 / 3, regimes=None)
    assert by_share.cut.row == 200 and by_share.regimes is None
    assert any("no stored regime labels" in n for n in by_share.notes)


def test_a_fixed_date_past_the_window_uses_the_share_so_something_is_held_out() -> None:
    result = diagnose.diagnose_returns(_series([(60, 0.001)]), idea_id="x",
                                       params_fixed_at=date(2030, 1, 1), share=1 / 3,
                                       regimes=None)
    assert result.periods == 40 and result.held_out_days == 19


def test_render_names_the_cut_and_lists_the_three_kinds_of_evidence() -> None:
    result = diagnose.diagnose_returns(_drawdown_series(), idea_id="x", params_fixed_at=None,
                                       share=1 / 3, regimes=_regimes(25))
    text = "\n".join(diagnose.render(result))
    assert f"Cut date: {result.cut.day}" in text and "held out" in text
    assert "Deepest drawdowns" in text and "not recovered by the cut" in text
    assert "Returns by BTC market regime" in text and "Worst calendar months" in text


# ---- diagnose_spec on a fake cached panel ------------------------------------------------


class _Seen:
    """A strategy that remembers the last bar of the panel it was handed."""

    name = "seen"
    last_bar: pd.Timestamp | None = None

    def target_weights(self, panel, params):
        _Seen.last_bar = panel.prices.index[-1]
        return pd.DataFrame(0.5, index=panel.prices.index, columns=panel.prices.columns)


def _panel() -> MarketPanel:
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(np.where(np.arange(300) % 7 < 3, 1.01, 0.995))},
                          index=DAYS)
    return MarketPanel(snapshot_id="fake", prices=prices,
                       funding=pd.DataFrame(0.0, index=DAYS, columns=["A"]),
                       tradeable=pd.DataFrame(True, index=DAYS, columns=["A"]), meta={})


def _spec(**extra):
    from qlab.pipeline.spec import StrategySpec

    return StrategySpec(
        idea_id="fake", title="fake", code_ref="qlab.test_diagnose:_Seen",
        data={"source": "binance", "interval": "1d", "start": date(2025, 1, 1),
              "end": date(2025, 10, 27)},
        costs={"taker_fee_bps": 1.0, "slippage_bps": 1.0}, min_leg_notional=10.0,
        unexpressed_mechanisms=[], **extra)


def test_the_strategy_never_sees_the_held_out_part(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: _panel())
    spec = _spec(params_fixed_at=date(2025, 6, 1), params_fixed_evidence="a test")
    result = diagnose.diagnose_spec(spec, None, share=1 / 3, regimes=_regimes(150))
    assert result is not None and result.cut.day == date(2025, 6, 1)
    assert _Seen.last_bar == pd.Timestamp("2025-05-31", tz="UTC")  # one bar before the cut
    assert result.last == date(2025, 5, 31) and result.periods == 150
    assert result.held_out_days == (DAYS[299].date() - date(2025, 6, 1)).days
    assert result.drawdowns  # a zig-zag price path has drawdowns


def test_the_share_cut_is_used_without_params_fixed_at(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: _panel())
    result = diagnose.diagnose_spec(_spec(), None, share=1 / 3, regimes=None)
    assert result is not None and result.cut.row == 200
    assert _Seen.last_bar == DAYS[199]


def test_an_uncached_panel_is_reported_not_fetched(monkeypatch) -> None:
    import qlab.pipeline.evaluate as evaluate

    monkeypatch.setattr(evaluate, "resolve_cached_panel", lambda session, spec: None)
    assert diagnose.diagnose_spec(_spec(), None, share=1 / 3, regimes=None) is None
