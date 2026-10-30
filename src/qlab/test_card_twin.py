"""The causal version of an idea from the inbox (docs/IMPLEMENTER.md): the card declares the
grid, and `qlab.twin` builds the spec whose numbers code chooses over exactly that grid, on
Binance's history; the baseline is re-run on the same panel (`same_panel_spec`)."""

from __future__ import annotations

from datetime import date

import pytest

from qlab.pipeline.sources import source_problems
from qlab.pipeline.spec import StrategySpec
from qlab.twin import (
    BINANCE_ARCHIVE_START,
    card_grid,
    causal_twin,
    same_panel_spec,
)

CARD = """idea_id: toy-idea
title: Игрушка
# the numbers the spec cites literally
costs: taker_fee_bps 1.5, slippage_bps 0.5, min_leg_notional 10
params: top_n 3, signal.window_days 7
grid: window_days [7, 14, 30]
"""


def _spec(idea_id: str = "toy-idea", card: str = "card.yaml", **over) -> StrategySpec:
    raw = {
        "idea_id": idea_id, "title": "Игрушка", "code_ref": "qlab.strategies.agent.toy_block:Toy",
        "params": {"signal": {"window_days": 7}, "top_n": 3},
        "data": {"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                 "end": "2026-10-26"},
        "costs": {"taker_fee_bps": 1.5, "slippage_bps": 0.5}, "min_leg_notional": 10.0,
        "unexpressed_mechanisms": [], "params_fixed_at": "2026-10-27",
        "params_fixed_evidence": "written by the implementer agent from the card on 2026-10-27",
        "sources": {"signal.window_days": card, "top_n": card, "costs": card,
                    "min_leg_notional": card}}
    raw.update(over)
    return StrategySpec.model_validate(raw)


AXES = {"signal.window_days": [7, 14, 30]}
NAMED = {"fast": {"signal.window_days": 7}, "slow": {"signal.window_days": 30}}


def _grid(spec=None, **kw):
    kw.setdefault("axes", AXES)
    kw.setdefault("candidates", None)
    return card_grid(spec or _spec(), cite="card.yaml", max_candidates=8, **kw)


# ---- the grid of a card ----------------------------------------------------------------


def test_lists_make_the_product_and_the_spec_carries_the_first_value() -> None:
    grid, why = _grid(axes={"signal.window_days": [7, 14, 30], "top_n": [3, 5]})
    assert why is None and grid.kind == "card"
    assert len(grid.candidates) == 6
    assert grid.base_params["signal"]["window_days"] == 7 and grid.base_params["top_n"] == 3
    names = list(grid.candidates)
    assert names[0] == "window_days=7/top_n=3"
    assert grid.candidates["window_days=30/top_n=5"]["signal"]["window_days"] == 30
    assert grid.sources["grid"].startswith("card.yaml: the grid the card declares, 6 variants")


def test_named_candidates_are_one_axis_of_overrides() -> None:
    grid, why = _grid(axes=None, candidates=NAMED)
    assert why is None and list(grid.candidates) == ["fast", "slow"]
    assert grid.candidates["slow"]["signal"]["window_days"] == 30
    assert grid.base_params["signal"]["window_days"] == 7  # the first candidate's value


@pytest.mark.parametrize("kwargs,words", [
    ({"axes": {"signal.window_days": [7]}}, "at least two"),
    ({"axes": {"signal.nothing": [1, 2]}}, "not parameters"),
    ({"axes": {"signal.window_days": [14, 30]}}, "not the card's first variant"),
    ({"axes": {"signal.window_days": list(range(7, 17))}}, "at most 8"),
    ({"axes": AXES, "candidates": NAMED}, "not both"),
    ({"axes": None, "candidates": {"a": {"signal.window_days": 3}, "b": {"top_n": 4}}},
     "not the first candidate's"),
])
def test_a_grid_that_cannot_be_made_says_why(kwargs, words) -> None:
    grid, why = _grid(**kwargs)
    assert grid is None and words in why


# ---- the causal version over the grid -------------------------------------------------


def test_the_causal_version_chooses_over_exactly_the_cards_grid(tmp_path) -> None:
    card = tmp_path / "card.yaml"
    card.write_text(CARD, encoding="utf-8")
    spec = _spec(card=str(card))
    grid, _ = card_grid(spec, axes=AXES, candidates=None, cite=str(card), max_candidates=8)
    result = causal_twin(spec, grid=grid, honest_from=(date(2023, 1, 1), str(card)),
                         execution_venue="hyperliquid")
    assert result.reason is None and result.kind == "card"
    twin = result.spec
    assert twin.idea_id == "toy-idea-causal" and twin.selects_causally
    assert twin.code_ref.endswith("retune:Retune")
    assert twin.params["inner_code_ref"] == spec.code_ref
    assert list(twin.params["grid"]) == ["signal.window_days"]
    assert result.candidates == ("window_days=7", "window_days=14", "window_days=30")
    assert twin.params["refit_every"] == "M" and twin.params["window"] == "expanding"
    assert twin.params["objective"] == "sharpe_net"
    # judged for Binance itself (docs/VENUE_SPLIT.md): Binance's costs, not the card's
    assert twin.params["score_costs"] == {"taker_fee_bps": 5.0, "slippage_bps": 0.9}
    assert twin.costs.taker_fee_bps == 5.0 and twin.min_leg_notional == 5.0
    assert twin.data.source == "binance" and twin.execution_venue is None
    # the card says its data honestly exists from 2023: the twin starts no earlier
    assert twin.data.start == date(2023, 1, 1)
    assert twin.sources["data.start"].endswith("(data.honest_from = 2023-01-01)")
    assert twin.params_fixed_at > twin.data.start  # the first causal choice, after the warm-up
    assert not source_problems(twin), source_problems(twin)  # every number cites a file


def test_without_honest_from_the_data_starts_at_the_archive() -> None:
    grid, _ = _grid()
    twin = causal_twin(_spec(), grid=grid).spec
    assert twin.data.start == BINANCE_ARCHIVE_START and twin.execution_venue is None


def test_a_hand_picked_list_is_judged_from_the_day_the_card_says_it_was_known() -> None:
    spec = _spec(data={"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                       "end": "2026-10-26", "instruments": ["BTC", "ETH"]})
    grid, _ = _grid(spec)
    listing = ["BTC", "ETH", "BTC-SPOT", "ETH-SPOT"]
    result = causal_twin(spec, grid=grid, venue_names=listing,
                         honest_from=(date(2022, 1, 1), "card.yaml"))
    twin = result.spec
    assert twin.data.instruments == ["BTC", "ETH"]
    assert twin.judged_from == twin.instruments_chosen_at == date(2022, 1, 1)
    assert twin.data.start == BINANCE_ARCHIVE_START  # the earlier history is warm-up
    assert "card.yaml" in twin.sources["instruments_chosen_at"]


def test_a_hand_picked_list_without_the_cards_date_is_dated_by_the_spec_and_refused() -> None:
    spec = _spec(data={"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                       "end": "2026-10-26", "instruments": ["BTC", "ETH"]})
    grid, _ = _grid(spec)
    result = causal_twin(spec, grid=grid, venue_names=["BTC", "ETH"])
    # the spec was written today: the list cannot be called older than that, no day is left
    assert result.spec is None and "on or after the end of the data" in result.reason


# ---- the baseline on the same panel -----------------------------------------------------


def test_the_baseline_runs_on_the_causal_versions_panel_costs_and_judged_part() -> None:
    spec = _spec(data={"source": "hyperliquid", "interval": "1d", "start": "2025-01-01",
                       "end": "2026-10-26", "instruments": ["BTC", "ETH"]})
    grid, _ = _grid(spec)
    twin = causal_twin(spec, grid=grid, venue_names=["BTC", "ETH"],
                       honest_from=(date(2022, 1, 1), "card.yaml")).spec
    baseline = _spec(
        "old-strategy", "old.yaml", code_ref="qlab.strategies.trend:TrendTSMOMEnsemble",
        params={"lookbacks_days": [30]}, costs={"taker_fee_bps": 9.0, "slippage_bps": 9.0},
        min_leg_notional=99.0, sources={"lookbacks_days": "old.yaml", "costs": "old.yaml",
                                        "min_leg_notional": "old.yaml"},
        params_fixed_at="2025-03-01")
    base = same_panel_spec(baseline, twin, idea_id="toy-idea-base", cite="twin.yaml")
    assert base.idea_id == "toy-idea-base"
    assert base.data == twin.data  # the same source, interval, window and instruments
    assert base.costs == twin.costs and base.min_leg_notional == twin.min_leg_notional
    assert base.judged_from == twin.judged_from == date(2022, 1, 1)
    assert base.instruments_chosen_at == twin.instruments_chosen_at
    assert base.execution_venue == twin.execution_venue
    assert base.code_ref == baseline.code_ref and base.params == baseline.params
    assert base.params_fixed_at == date(2025, 3, 1)  # the baseline's own numbers are untouched
    assert base.sources["costs.taker_fee_bps"].startswith("twin.yaml")
    assert "toy-idea-causal" in base.differs_from_source[-1]
