"""The chosen value's own money over the selection part (docs/FORWARD_CHART.md, «подбор»).

Owner, 2026-11-04: «график о том как работает стратегия на периоде доступном по данным». Under
rules 2026-11-04.1 an idea that declares lists of values is a `Retune` that chooses ONCE, at the
cut, on the first 2/3 of its window and holds that value on the last third; an older causal chooser
(`selects_causally`) re-chooses every month and its first 2/3 is a warm-up. Either way the wrapper
holds nothing before its first choice, so the first 2/3 of the whole-window chart was a flat zero
line: 4.5 years that show nothing.

What this module draws there instead is the money of the CHOSEN value itself -- the candidate the
wrapper holds on the first forward bar, run alone on the same panel, with the same costs and on the
same strategy-money basis -- over the selection rows. It is the curve of numbers that code picked by
looking at exactly this stretch, so it flatters the strategy (the console says so); the honest part
is the forward test, which is the run itself and is not touched here.

Display only. Nothing here is read by a rule or by the route: the numbers are written under the
`fit_` names of the selection period's figures (`fit_cap_ann_return`, ...) next to a flag
`fit_chosen_in_sample`, and `qlab.pipeline.evaluate` keeps deciding from the forward part and from
a selection period that does not exist for a causal chooser (no `fit_period` rule is applied).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qlab.harness.costs import CostModel
from qlab.harness.metrics import capital_metrics, compute_metrics
from qlab.harness.panel import MarketPanel
from qlab.harness.run import RunResult, run_backtest
from qlab.harness.strategy import CapitalSetup, capital_setup, own_costs
from qlab.strategies.retune import Retune
from qlab.year_view import is_year_key

FLAG = "fit_chosen_in_sample"
"""The metric that says the `fit_` figures of a run are the chosen value's own, picked on that very
stretch, not figures a rule judged."""


@dataclass(frozen=True)
class ChosenCandidate:
    """The value a `Retune` holds on the first forward bar, run alone over the selection rows."""

    name: str
    inner_code_ref: str
    params: dict[str, Any]
    result: RunResult
    """Over the selection rows only: decision bars `first .. cut - 1`."""
    weights: pd.DataFrame
    """The candidate's weights on the same bars."""
    capital: CapitalSetup
    cut_label: pd.Timestamp
    """The first forward bar."""

    def fit_metrics(self) -> dict[str, float]:
        """The selection part's figures of the chosen value, under the `fit_` names, plus the
        flag. The strategy's own money's figures (`cap_`) are there unless the base is refused."""
        measured = compute_metrics(self.result)
        if self.capital.refused is None:
            measured.update(capital_metrics(self.result))
        out = {f"fit_{k}": v for k, v in measured.items() if not is_year_key(k)}
        out[FLAG] = 1.0
        return out

    def stitch(self, whole_result: RunResult, whole_weights: pd.DataFrame
               ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        """The whole window's `(net_return, capital_return, weights)` with the selection bars of
        the chosen value in place of the wrapper's flat ones; the bars from the cut on are the
        run's own, so the forward part is what it was."""
        def join(chosen: pd.Series, run: pd.Series) -> pd.Series:
            return pd.concat([chosen, run[run.index >= self.cut_label]])

        weights = pd.concat([self.weights.iloc[:-1],
                             whole_weights[whole_weights.index >= self.cut_label]])
        return (join(self.result.net_return, whole_result.net_return),
                join(self.result.capital_return, whole_result.capital_return), weights)

    def chart_bounds(self, bounds: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]]
                     ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """The parts of the window with the warm-up named the selection part it now shows."""
        if "warmup" not in bounds or "selection" in bounds:
            return dict(bounds)
        return {("selection" if name == "warmup" else name): span for name, span in bounds.items()}

    def info(self) -> dict[str, Any]:
        """What the series file keeps about the part: which value, and its figures."""
        return {"name": self.name, "numbers": self.fit_metrics()}


def held_at_cut(strategy: object, panel: MarketPanel, params: Mapping[str, object], cut: int
                ) -> str | None:
    """The name of the candidate a `Retune` holds on row `cut`: its last choice made on or before
    that bar (the single choice of `refit_every: once`, which is made on the first forward bar),
    or None when the strategy is not a `Retune` or held nothing there."""
    if not isinstance(strategy, Retune):
        return None
    choices = strategy.run(panel, params).choices.dropna()
    held = choices[choices.index <= panel.prices.index[cut]]
    return None if held.empty else str(held.iloc[-1])


def chosen_candidate(strategy: object, panel: MarketPanel, params: Mapping[str, object], *,
                     first: int, cut: int, costs: CostModel) -> ChosenCandidate | None:
    """The candidate `strategy` holds on the first forward bar `cut`, run alone over rows
    `first .. cut` (so decision bars `first .. cut - 1`) with `costs`; None when `strategy` is
    not a `Retune` choosing among values, or chose nothing."""
    name = held_at_cut(strategy, panel, params, cut)
    if name is None or cut <= first:
        return None
    from qlab.pipeline.evaluate import _slice_rows, resolve_strategy
    from qlab.strategies.retune import expand_grid

    inner_ref = str(params["inner_code_ref"])
    candidates = expand_grid(dict(params["base_params"]), dict(params["grid"]))  # type: ignore[arg-type]
    cand_params = candidates[name]
    inner = resolve_strategy(inner_ref)
    weights = inner.target_weights(panel, cand_params).where(panel.tradeable, 0.0)
    capital = capital_setup(inner, panel, cand_params)
    sliced = _slice_rows(panel, first, cut)
    part_weights = weights.iloc[first:cut + 1]
    result = run_backtest(sliced, part_weights, costs, sliced.funding,
                          extra_cost=own_costs(inner, panel, cand_params), **capital.kwargs)
    return ChosenCandidate(name=name, inner_code_ref=inner_ref, params=dict(cand_params),
                           result=result, weights=part_weights, capital=capital,
                           cut_label=panel.prices.index[cut])


__all__ = ["FLAG", "ChosenCandidate", "chosen_candidate", "held_at_cut"]
