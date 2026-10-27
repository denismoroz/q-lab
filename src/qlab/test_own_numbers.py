"""Own numbers and own mechanisms (qlab.own_numbers, docs/OWN_NUMBERS.md): owner, 2026-10-26,
«Разрешить с этими четырьмя ограничениями ... - еще я бы довавил правило - придумать свой
механизм основываясь на идее»."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qlab.agents.generator as gen
from qlab import own_numbers
from qlab.harness.panel import MarketPanel
from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import load_spec
from qlab.strategies.retune import Retune, refit_rows

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "docs" / "MECHANISMS.md"
DATA_END = date(2026, 10, 24)
OWN_CITE = "night/own_numbers.yaml, the declared list for {}; owner 2026-10-26"


@pytest.fixture(scope="module")
def mechanisms() -> list[gen.Mechanism]:
    return gen.catalogue(CATALOGUE)


@pytest.fixture(scope="module")
def config() -> dict:
    return own_numbers.load_config()


def _spec(mech: str, name: str, *, window: str, rebalance: str, min_history: str,
          own_block: str = "", extra_sources: str = "") -> str:
    return f"""# mechanism: {mech}
idea_id: {name}
title: Пример для проверки своих чисел
code_ref: qlab.strategies.blocks:Composed
params:
  universe:
    rule: all
  signal:
    block: funding
    window_days: {window}
  portfolio:
    block: time_series_sign
  sizing:
    block: equal
    gross: 1.0
  rebalance_days: {rebalance}
  min_history_days: {min_history}
data:
  source: binance
  interval: 1d
  start: '2019-09-08'
  end: '{DATA_END}'
costs:
  taker_fee_bps: 3.5
  slippage_bps: 0.9
min_leg_notional: 10.0
simultaneous_legs: 1
noise_universe: panel
unexpressed_mechanisms: []
{own_block}sources:
  universe: 'docs/MECHANISMS.md, mechanism {mech}, blocks: universe: all'
  signal: 'docs/MECHANISMS.md, mechanism {mech}, blocks: signal: funding'
  portfolio: 'docs/MECHANISMS.md, mechanism {mech}, blocks: portfolio'
  sizing: 'docs/MECHANISMS.md, mechanism {mech}, equal shares'
  sizing.gross: 'docs/MECHANISMS.md, framework value: gross 1 = the whole capital, no leverage'
{extra_sources}  costs.taker_fee_bps: 'owner 2026-10-25: test fixture'
  costs.slippage_bps: 'owner 2026-10-25: test fixture'
  min_leg_notional: 'owner 2026-10-25: test fixture'
"""


OPEN_SOURCES = (f"  signal.window_days: '{OWN_CITE.format('signal.window_days')}'\n"
                f"  rebalance_days: '{OWN_CITE.format('rebalance_days')}'\n"
                "  min_history_days: 'derived: 30; the longest open window'\n")
OWN_BLOCK = """own_mechanism:
  idea: Short the coins whose perpetual pays a high funding rate and buy those paying little.
  why: Longs who pay a high funding rate overpay for leverage; the carry is earned from them.
  idea_from: A variation of perp-funding-carry in docs/MECHANISMS.md, without its spot leg.
"""


def _check(tmp_path: Path, mechanisms, text: str, name: str) -> list[str]:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    _, problems = gen.check_generated(path, mechanisms=mechanisms, data_end=DATA_END,
                                      reference=None)
    return [p for p in problems if "costs" not in p and "min_leg_notional" not in p
            and "differs_from_source" not in p]


def test_an_empty_row_may_be_left_open_with_the_whole_declared_list(mechanisms, tmp_path):
    text = _spec("perp-carry-crash-timing", "gen-ex-open", window="[3, 7, 14, 30]",
                 rebalance="[1, 7]", min_history="30", extra_sources=OPEN_SOURCES)
    assert _check(tmp_path, mechanisms, text, "gen-ex-open") == []


def test_a_subset_of_the_list_is_refused(mechanisms, tmp_path):
    text = _spec("perp-carry-crash-timing", "gen-ex-subset", window="[7, 14]",
                 rebalance="[1, 7]", min_history="30", extra_sources=OPEN_SOURCES)
    problems = _check(tmp_path, mechanisms, text, "gen-ex-subset")
    assert any("whole declared list" in p for p in problems)


def test_a_row_the_catalogue_fills_cannot_be_left_open(mechanisms, tmp_path):
    # classic-carry gives rebalance_days 30 (a month): no list may replace it.
    text = _spec("classic-carry", "gen-ex-filled", window="[3, 7, 14, 30]",
                 rebalance="[1, 7]", min_history="30", extra_sources=OPEN_SOURCES)
    problems = _check(tmp_path, mechanisms, text, "gen-ex-filled")
    assert any(p.startswith("rebalance_days: classic-carry gives values") for p in problems)


def test_an_own_mechanism_needs_its_idea_and_open_numbers(mechanisms, tmp_path):
    good = _spec("own", "gen-ex-own", window="[3, 7, 14, 30]", rebalance="[1, 7]",
                 min_history="30", own_block=OWN_BLOCK, extra_sources=OPEN_SOURCES)
    assert _check(tmp_path, mechanisms, good, "gen-ex-own") == []
    no_idea = _spec("own", "gen-ex-own2", window="[3, 7, 14, 30]", rebalance="[1, 7]",
                    min_history="30", extra_sources=OPEN_SOURCES)
    assert any("needs own_mechanism" in p for p in _check(tmp_path, mechanisms, no_idea,
                                                           "gen-ex-own2"))
    fixed = _spec("own", "gen-ex-own3", window="7", rebalance="[1, 7]", min_history="30",
                  own_block=OWN_BLOCK, extra_sources=(
                      "  signal.window_days: 'docs/MECHANISMS.md, mechanism own, 7'\n"
                      f"  rebalance_days: '{OWN_CITE.format('rebalance_days')}'\n"
                      "  min_history_days: 'derived: 30; the longest open window'\n"))
    assert any("an own mechanism takes its numbers" in p
               for p in _check(tmp_path, mechanisms, fixed, "gen-ex-own3"))


def test_too_many_candidates_are_refused(config, tmp_path):
    text = _spec("own", "gen-ex-many", window="[3, 7, 14, 30]", rebalance="[1, 7]",
                 min_history="30", own_block=OWN_BLOCK, extra_sources=OPEN_SOURCES)
    path = tmp_path / "gen-ex-many.yaml"
    path.write_text(text, encoding="utf-8")
    spec = load_spec(path)
    assert own_numbers.problems(spec, config) == []
    assert own_numbers.problems(spec, {**config, "max_candidates": 4})[0].startswith("8 candidates")


def test_the_spec_becomes_one_choice_at_the_cut_judged_after_it(config, tmp_path):
    text = _spec("perp-carry-crash-timing", "gen-ex-cut", window="[3, 7, 14, 30]",
                 rebalance="[1, 7]", min_history="30", extra_sources=OPEN_SOURCES)
    path = tmp_path / "gen-ex-cut.yaml"
    path.write_text(text, encoding="utf-8")
    spec = load_spec(path)
    assert len(own_numbers.candidates(spec)) == 8
    cut = own_numbers.cut_date(spec, config)
    span = (spec.data.end - spec.data.start).days
    assert abs((cut - spec.data.start).days - round(span * 2 / 3)) <= 1  # the owner's 2/3
    retune = own_numbers.to_retune(spec, config)
    assert retune.code_ref == own_numbers.RETUNE and retune.selects_causally
    assert retune.params_fixed_at == cut and retune.params["refit_every"] == "once"
    assert retune.params["grid"]["rebalance_days"] == {"rebalance_days=1": {"rebalance_days": 1},
                                                       "rebalance_days=7": {"rebalance_days": 7}}
    problems = [p for p in source_problems(retune) if "costs" not in p
                and "min_leg_notional" not in p]
    assert problems == []
    late = spec.model_copy(update={"params_fixed_at": date(2026, 1, 1),
                                   "params_fixed_evidence": "a paper of 2026"})
    assert own_numbers.cut_date(late, config) == date(2026, 1, 1)


class TwoWindows:
    """A strategy whose `window` param picks one of two coins: the better one before the cut
    is held after it."""

    name = "two-windows"

    def target_weights(self, panel, params):
        w = pd.DataFrame(0.0, index=panel.prices.index, columns=panel.prices.columns)
        w["A" if params["signal"]["window_days"] == 3 else "B"] = 1.0
        return w


def _panel() -> MarketPanel:
    index = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    a = np.r_[np.full(120, 1.01), np.full(80, 0.99)]  # A wins before the cut, loses after
    b = np.r_[np.full(120, 1.0), np.full(80, 1.0)]
    prices = pd.DataFrame({"A": 100 * np.cumprod(a), "B": 100 * np.cumprod(b)}, index=index)
    zero = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    return MarketPanel(prices=prices, funding=zero,
                       tradeable=pd.DataFrame(True, index=index, columns=["A", "B"]),
                       snapshot_id="own-test", meta={})


def test_the_choice_sees_only_data_before_the_cut():
    panel = _panel()
    params = {"inner_code_ref": "qlab.test_own_numbers:TwoWindows",
              "base_params": {"signal": {"window_days": 3}},
              "grid": {"signal.window_days": {"w3": {"signal.window_days": 3},
                                              "w7": {"signal.window_days": 7}}},
              "refit_every": "once", "choose_at": "2024-04-30", "window": "expanding",
              "objective": "total_return_net",
              "score_costs": {"taker_fee_bps": 0.0, "slippage_bps": 0.0}}
    rows = refit_rows(panel.prices.index, "once", "2024-04-30")
    assert rows.sum() == 1 and panel.prices.index[rows][0] == pd.Timestamp("2024-04-30", tz="UTC")
    run = Retune().run(panel, params)
    assert list(run.choices.dropna()) == ["w3"]  # A was the better before the cut
    held = run.weights.loc["2024-05-01":]
    assert (held["A"] == 1.0).all() and (run.weights.loc[:"2024-04-29"].abs().sum().sum() == 0)


def test_a_night_rerun_keeps_the_first_runs_cut_on_a_longer_tail(config, tmp_path):
    # The night of 2026-10-27 passed open lists to the stand as they were; a re-run must judge
    # the same one choice, its cut fixed by the spec file's window, not moved by the new data.
    from qlab.night import rerun_spec, with_end

    text = _spec("perp-carry-crash-timing", "gen-ex-rerun", window="[3, 7, 14, 30]",
                 rebalance="[1, 7]", min_history="30", extra_sources=OPEN_SOURCES)
    path = tmp_path / "gen-ex-rerun.yaml"
    path.write_text(text, encoding="utf-8")
    spec = load_spec(path)
    first = own_numbers.cut_date(spec, config)
    judged = with_end(rerun_spec(spec), date(2027, 6, 1))
    assert judged.code_ref == own_numbers.RETUNE and judged.params_fixed_at == first
    assert judged.params["choose_at"] == first.isoformat() and judged.data.end == date(2027, 6, 1)
    plain = spec.model_copy(update={"params": {**spec.params, "rebalance_days": 1,
                                               "signal": {"block": "funding", "window_days": 7}}})
    assert rerun_spec(plain) is plain
