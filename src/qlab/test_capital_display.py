"""The strategy's own money leads everywhere the owner reads numbers (docs/CAPITAL_BASE.md; owner,
2026-10-29: «было бы здорово пофиксить это везде»): the shared headline, the morning report's
lines, the console's API, the regime split, the hedge timing, the overlay comparison, the
anomaly toolkit and the year signs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab import capital_text, night, twin_stage
from qlab.harness.metrics import capital_metrics
from qlab.harness.run import RunResult
from qlab.pipeline import stage2

# Bv2 on Binance as docs/CAPITAL_BASE.md section 5 gives it: the stand divides by a balance about
# four times the money, the money's own figures are +8.4% a year and a fall of 19.3%.
BV2 = {
    "ann_return_net": 0.0202, "sharpe_net": 0.55, "max_dd": -0.0448, "total_return_net": 0.028,
    "noise_return_percentile": 0.99, "noise_power": 0.95,
    "cap_ann_return": 0.0841, "cap_sharpe": 0.53, "cap_max_dd": -0.1934, "cap_total_return": 0.118,
    "cap_ann_return_ex_best_1pct": -0.04, "cap_reinvested": 0.0,
    "noise_cap_return_percentile": 0.98, "noise_cap_power": 0.40,
}
STAND_ONLY = {k: v for k, v in BV2.items() if not k.startswith(("cap_", "noise_cap_"))}


# ---- the headline ---------------------------------------------------------------------------


def test_the_headline_is_the_strategys_money_and_never_mixes_two_bases() -> None:
    h = capital_text.headline(BV2)
    assert h.on_capital and h.reinvested is False
    assert (h.ann, h.sharpe, h.max_dd, h.total, h.noise) == (0.0841, 0.53, -0.1934, 0.118, 0.98)
    old = capital_text.headline(STAND_ONLY)
    assert not old.on_capital and (old.ann, old.max_dd, old.noise) == (0.0202, -0.0448, 0.99)
    # A run that has the capital yearly figure but lacks the capital noise rank shows no rank at
    # all rather than the stand's, which is another base.
    half = {k: v for k, v in BV2.items() if k != "noise_cap_return_percentile"}
    assert capital_text.headline(half).noise is None
    assert capital_text.headline(None).ann is None and not capital_text.headline({}).on_capital


def test_a_part_of_the_run_is_read_on_the_same_base_as_the_whole() -> None:
    m = {**BV2, "fit_ann_return_net": 0.01, "fit_cap_ann_return": 0.05, "fit_cap_sharpe": 0.7,
         "fit_cap_max_dd": -0.1}
    fit = capital_text.headline(m, "fit_")
    assert fit.on_capital and (fit.ann, fit.sharpe, fit.max_dd) == (0.05, 0.7, -0.1)
    assert not capital_text.headline({"fit_ann_return_net": 0.01}, "fit_").on_capital


def test_the_numbers_line_leads_with_the_money_and_the_stand_follows_small() -> None:
    line = capital_text.numbers_ru(BV2)
    assert line.startswith("в год +8.4%, Шарп 0.53, просадка −19.3%, обгоняет шум 98%")
    assert line.endswith("; как считает стенд (от баланса книги): в год +2.0%, Шарп 0.55, "
                         "просадка −4.5%")
    assert line.index("+8.4%") < line.index("как считает стенд")
    # Where the two are the same figure nothing is added.
    same = {"ann_return_net": 0.05, "cap_ann_return": 0.05, "max_dd": -0.1, "cap_max_dd": -0.1,
            "sharpe_net": 1.0, "cap_sharpe": 1.0, "noise_cap_return_percentile": 0.99,
            "cap_reinvested": 1.0}
    assert "стенд" not in capital_text.numbers_ru(same)
    # An older run is shown as the stand counted it, with no claim about the money.
    assert capital_text.numbers_ru(STAND_ONLY).startswith("в год +2.0%, Шарп 0.55, просадка −4.5%")
    assert "обгоняет шум" not in capital_text.numbers_ru(BV2, noise="never")
    assert "обгоняет шум" not in capital_text.numbers_ru({"cap_ann_return": 0.1}, noise="present")


def test_the_base_is_named_in_words() -> None:
    assert "от денег стратегии, прибыль не вкладывается обратно" in capital_text.basis_ru(BV2)
    assert "со сложным процентом" in capital_text.basis_ru({**BV2, "cap_reinvested": 1.0})
    assert "по счёту стенда" in capital_text.basis_ru(STAND_ONLY)
    refused = capital_text.basis_ru({**STAND_ONLY, "cap_base_refused": 1.0})
    assert "основа процентов у этой книги не определена" in refused


# ---- the morning report ---------------------------------------------------------------------


def test_the_watch_table_row_is_on_the_money_and_says_so() -> None:
    m = {**BV2, "forward_days": 120.0, "fit_ann_return_net": 0.01, "fit_cap_ann_return": 0.05,
         "fit_cap_sharpe": 0.7, "cap_regime_bull_return": 0.03, "cap_regime_flat_return": 0.01,
         "cap_regime_bear_return": -0.02, "regime_bull_return": 0.01, "regime_flat_return": 0.0,
         "regime_bear_return": -0.005, "fit_cap_regime_bull_return": 0.04,
         "fit_cap_regime_flat_return": 0.02, "fit_cap_regime_bear_return": -0.01}
    row = night._report_row(m)
    assert row[0] == "+5.0%" and row[1] == "0.70"          # the selection period, on the money
    assert row[3] == "+11.8%"                              # the forward test's total, on the money
    assert row[4:] == ("+4.0%", "+2.0%", "-1.0%")
    stand = {k: v for k, v in m.items() if not k.startswith(("cap_", "fit_cap_", "noise_cap"))}
    assert night._report_row(stand)[0] == "+1.0%"          # an older run: the stand's, as ever


def test_the_stage_lines_lead_with_the_money() -> None:
    line = stage2._numbers_ru(BV2)
    assert line.startswith("в год +8.4%, Шарп 0.53, просадка −19.3%, обгоняет шум 98%")
    assert "как считает стенд (от баланса книги): в год +2.0%" in line
    twin = twin_stage.numbers_ru({**BV2, "forward_days": 513.0})
    assert twin.startswith("в год +8.4%, Шарп 0.53, просадка −19.3%")
    assert "проверка вперёд 513 дн." in twin and "как считает стенд" in twin
    assert twin.index("проверка вперёд") < twin.index("как считает стенд")


def test_the_generators_result_line_leads_with_the_money() -> None:
    from qlab.agents.generator import GeneratedResult, result_line

    line = result_line(GeneratedResult(idea_id="x", mechanism="m", spec_path="specs/x.yaml",
                                       trial_id=7, route="paper", metrics=BV2))
    assert "в год +8.4%" in line and line.index("+8.4%") < line.index("+2.0%")


def test_the_assessors_brief_names_the_money_and_the_prompt_says_which_keys_to_trust() -> None:
    from qlab.agents import assessor

    assert "cap_ann_return" in assessor.SYSTEM_PROMPT and "OWN money" in assessor.SYSTEM_PROMPT
    assert "noise_cap_power" in assessor.FAMILY_KEYS


# ---- the console's API ----------------------------------------------------------------------


def test_the_api_headline_is_the_money_with_the_stand_beside_it() -> None:
    from qlab.api.serializers import headline_json

    h = headline_json({**BV2, "fit_cap_ann_return": 0.05, "fit_cap_sharpe": 0.7,
                       "fit_cap_max_dd": -0.1})
    assert h["on_capital"] and h["reinvested"] is False and h["differs"]
    assert h["ann_return"] == 0.0841 and h["max_dd"] == -0.1934 and h["noise_percentile"] == 0.98
    assert h["stand"]["ann_return"] == 0.0202 and h["stand"]["noise_percentile"] == 0.99
    assert h["fit"] == {"ann_return": 0.05, "sharpe": 0.7, "max_dd": -0.1}
    assert "от денег стратегии" in h["basis"] and not h["base_refused"]
    old = headline_json(STAND_ONLY)
    assert not old["on_capital"] and old["stand"] is None and old["ann_return"] == 0.0202
    assert headline_json(None) is None
    refused = headline_json({"ann_return_net": 0.3, "cap_base_refused": 1.0})
    assert refused["base_refused"] and not refused["on_capital"]


# ---- the numbers behind the lines -----------------------------------------------------------


def _result(values: list[float], base: float, reinvested: bool) -> RunResult:
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC")
    net = pd.Series(values, index=idx)
    zero = pd.Series(0.0, index=idx)
    return RunResult("s", net, net, zero, zero, zero, capital_return=net * base,
                     capital_reinvested=reinvested)


def test_the_worst_week_has_a_twin_on_the_money() -> None:
    values = [0.001] * 20 + [-0.01] * 7 + [0.001] * 20
    result = _result(values, 4.0, reinvested=False)
    weekly = (result.capital_return.groupby(pd.Grouper(freq="W")).sum())
    assert capital_metrics(result)["cap_worst_week"] == pytest.approx(float(weekly.min()))
    stand_week = capital_metrics(_result(values, 4.0, reinvested=True))["cap_worst_week"]
    assert stand_week == pytest.approx(float(((1.0 + result.capital_return)
                                              .groupby(pd.Grouper(freq="W")).prod() - 1.0).min()))


def test_the_regime_split_has_a_twin_on_the_money_summed_when_it_does_not_reinvest() -> None:
    from qlab.regimes import RegimeSeries, breakdown

    idx = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
    labels = pd.Series(["bull"] * 30 + ["flat"] * 30 + ["bear"] * 30, index=idx)
    regimes = RegimeSeries(labels=labels, market_return=pd.Series(0.001, index=idx),
                           bear_below=-0.05, bull_above=0.05, source="test")
    ret = pd.Series(0.001, index=idx)
    stand = breakdown(ret, regimes, 365.25)
    money = breakdown(ret * 4.0, regimes, 365.25, additive=True, prefix="cap_regime_")
    assert "regime_bear_below" in stand and "cap_regime_bear_below" not in money
    assert money["cap_regime_bull_days"] == stand["regime_bull_days"]
    # summed: 29 bars of 0.4 point; the stand compounds 29 bars of 0.1 point
    n = round(money["cap_regime_bull_return"] / 0.004)
    assert money["cap_regime_bull_return"] == pytest.approx(n * 0.004)
    assert money["cap_regime_bull_ann_return"] == pytest.approx(
        money["cap_regime_bull_return"] * 365.25 / n)
    assert money["cap_regime_bear_max_dd"] == 0.0
    assert money["cap_regime_bull_return"] > stand["regime_bull_return"] * 3.5


def test_the_overlay_comparison_sums_the_days_of_a_strategy_that_does_not_reinvest() -> None:
    from qlab import overlay_compare as oc

    daily = pd.Series([0.01, -0.02, 0.01, 0.01], index=pd.date_range("2025-01-01", periods=4,
                                                                      freq="D", tz="UTC"))
    summed = oc.window_metrics(daily, additive=True)
    assert summed["total_return"] == pytest.approx(0.01)
    assert summed["ann_return"] == pytest.approx(0.01 * 365.25 / 4)
    assert summed["max_dd"] == pytest.approx(-0.02)          # points of the capital
    assert oc.window_metrics(daily)["total_return"] != summed["total_return"]
    assert oc._per_day(daily, additive=True).sum() == pytest.approx(daily.sum())
    rows = np.array([[0.01, 0.01], [-0.01, 0.0]])
    assert oc._annual_rows(rows, additive=True) == pytest.approx([0.02 * 365.25 / 2,
                                                                 -0.01 * 365.25 / 2])


def test_the_overlay_table_says_on_which_base_it_was_made() -> None:
    from qlab import overlay_stage

    def row(on_capital):
        comparison = {"days": 400, "base": {"ann_return": 0.08, "max_dd": -0.1, "sharpe": 0.5},
                      "variant": {"ann_return": 0.09, "max_dd": -0.09, "sharpe": 0.6},
                      "delta_sharpe": 0.1, "delta_sharpe_ci": [-0.1, 0.3], "corrected_p": 0.5,
                      "verdict_ru": "неотличимо от случайности", "on_capital": on_capital}
        return {"idea_id": "b-ov-gate", "comparison": comparison}

    def lines(*flags):
        return "\n".join(overlay_stage.table_lines(
            {"base_id": "b", "variants": [row(f) for f in flags]}))

    assert "Числа — в процентах от денег стратегии" in lines(True)
    assert "посчитаны по счёту стенда" in lines(False)
    assert "более ранние сравнения посчитаны по счёту стенда" in lines(True, False)


def test_the_anomaly_toolkit_reports_the_run_on_the_money_with_the_stand_in_a_corner() -> None:
    from qlab.anomalies.toolkit import summarize_result

    out = summarize_result(_result([0.001] * 60, 4.0, reinvested=False))
    assert out["ann_return_net"] == pytest.approx(0.06 * 4 * 365.25 / 60, rel=0.05)
    assert out["on_capital"] and out["reinvested"] is False
    assert out["ann_return_net"] > out["stand"]["ann_return_net"]       # 4x the stand's share
    assert out["total_return_net"] == pytest.approx(60 * 0.001 * 4)     # summed
    assert out["max_dd"] == 0.0
