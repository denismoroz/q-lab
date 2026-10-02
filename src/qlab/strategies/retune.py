"""Regular re-tuning of a strategy's parameters (docs/TASKS.md T38,
docs/RETUNE.md).

Owner, 2026-10-02: "у нас должен быть шаг во фреймворке, который должен
настраивать эти параметры — применять разные комбинации, пересчитывать их раз
в день"; "для каждой стратегии подбор параметров должен быть периодическим —
раз в неделю/день или месяц, а так получается стратегия непонятно что торгует".

`Retune` wraps any strategy. The spec declares a GRID of parameter variants
(candidates), how often to re-tune, over which window of the past to judge,
and by which measure. At the first bar of every re-tuning period the wrapper
scores every candidate on the net returns it has ALREADY realised by that bar,
picks the best, and holds that candidate's weights until the next re-tuning.

Why the output is a forward test by construction: the candidate traded on any
day was chosen from returns realised before that day's decision. The return
earned at decision row t is realised over (t, t+1] (`qlab.harness.run`), so a
choice made at bar tau sees only rows < tau. What is NOT forward is the grid
itself -- the spec author chose which candidates to offer, knowing what they
know -- which is why a retune spec still carries its own `params_fixed_at`
(the day the grid was declared) and is judged by the selection/forward split
like any other (docs/FIT_VS_FORWARD.md).

Rules this wrapper keeps:

- **Scores use the spec's costs.** `score_costs` must repeat the spec's
  `costs` (a cost-blind score would favour whichever candidate trades most).
- **A candidate is eligible only after it has traded through one whole
  re-tuning period** before the decision -- otherwise the first choice after
  a warm-up would be made on a handful of returns. No number is introduced:
  the period is the spec's own `refit_every`.
- **No eligible candidate -> no position** until the next re-tuning; the
  wrapper never falls back to a favourite.
- **Each candidate runs as the inner strategy runs alone.** A candidate with
  path-dependent state (live trend's book-volatility scale) sees its OWN
  equity curve, not the composite's -- a documented approximation.

`diagnostics()` reports, after a run, how many re-tunings there were and how
often the choice changed: a choice that jumps at every re-tuning is chasing
noise, not the market (folded into the trial's metrics by
`qlab.pipeline.evaluate`).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.harness.costs import CostModel
from qlab.harness.panel import MarketPanel
from qlab.harness.run import run_backtest

_REQUIRED = ("inner_code_ref", "base_params", "grid", "refit_every", "window", "objective",
             "score_costs")
_PERIODS = {"D": "D", "W": "W-SUN", "M": "M"}
_OBJECTIVES = ("sharpe_net", "total_return_net")

_CACHE: dict[tuple, RetuneRun] = {}
_CACHE_SIZE = 8


@dataclass(frozen=True)
class RetuneRun:
    weights: pd.DataFrame
    choices: pd.Series  # re-tuning bar -> chosen candidate name (None = no position)
    candidate_returns: pd.DataFrame  # net return per decision row, per candidate


def expand_grid(base: Mapping[str, object], grid: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """{candidate name: params}. `grid` maps each axis to labelled override
    dicts; candidates are the cartesian product, overrides merged onto
    `base` in axis order; the name joins the labels with '/'."""
    if not grid:
        raise ValueError("retune grid is empty")
    axes = list(grid.items())
    for axis, options in axes:
        if not options:
            raise ValueError(f"retune grid axis {axis!r} has no options")
    out = {}
    for combo in itertools.product(*[list(options.items()) for _, options in axes]):
        params = dict(base)
        for _, overrides in combo:
            params.update(dict(overrides))
        out["/".join(label for label, _ in combo)] = params
    if len(out) < 2:
        raise ValueError("retune needs at least two candidates to choose between")
    return out


def refit_rows(index: pd.DatetimeIndex, refit_every: str) -> np.ndarray:
    """True on the first bar of every re-tuning period (calendar day, week
    ending Sunday, or month, UTC). The very first bar is not one: nothing has
    been realised yet."""
    if refit_every not in _PERIODS:
        raise ValueError(f"refit_every must be one of {sorted(_PERIODS)}, got {refit_every!r}")
    periods = index.tz_convert("UTC").tz_localize(None).to_period(_PERIODS[refit_every])
    starts = np.r_[False, np.asarray(periods[1:] != periods[:-1])]
    return starts


def _score(returns: pd.Series, objective: str) -> float:
    if objective == "sharpe_net":
        std = returns.std(ddof=1)
        return float(returns.mean() / std) if std and not np.isnan(std) else float("nan")
    return float((1.0 + returns).prod() - 1.0)


class Retune:
    """See module docstring. Every param is required; none has a default."""

    name = "retune"

    def __init__(self) -> None:
        self._last: RetuneRun | None = None

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        return self.run(panel, params).weights

    def diagnostics(self) -> dict[str, float]:
        if self._last is None:
            return {}
        chosen = self._last.choices.dropna()
        switches = int((chosen != chosen.shift()).iloc[1:].sum()) if len(chosen) > 1 else 0
        top_share = float(chosen.value_counts(normalize=True).iloc[0]) if len(chosen) else 0.0
        return {
            "retune_candidates": float(self._last.candidate_returns.shape[1]),
            "retune_refits": float(len(self._last.choices)),
            "retune_refits_with_choice": float(len(chosen)),
            "retune_switches": float(switches),
            "retune_top_choice_share": top_share,
        }

    def run(self, panel: MarketPanel, params: Mapping[str, object]) -> RetuneRun:
        missing = [key for key in _REQUIRED if key not in params]
        if missing:
            raise ValueError(f"Retune needs params {missing}; none has a default")
        index = panel.prices.index
        key = (
            panel.snapshot_id, str(index[0]), str(index[-1]), len(index),
            hash(tuple(panel.prices.columns)), json.dumps(params, sort_keys=True, default=str),
        )
        if key in _CACHE:
            self._last = _CACHE[key]
            return self._last

        objective = str(params["objective"])
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
        window = str(params["window"])
        lookback = None if window == "expanding" else pd.Timedelta(window)
        costs = CostModel(**dict(params["score_costs"]))  # type: ignore[arg-type]

        from qlab.pipeline.evaluate import resolve_strategy

        inner = resolve_strategy(str(params["inner_code_ref"]))
        candidates = expand_grid(dict(params["base_params"]), dict(params["grid"]))  # type: ignore[arg-type]

        weights_by: dict[str, pd.DataFrame] = {}
        returns_by: dict[str, pd.Series] = {}
        first_active: dict[str, pd.Timestamp | None] = {}
        for name, cand_params in candidates.items():
            w = inner.target_weights(panel, cand_params)
            weights_by[name] = w
            returns_by[name] = run_backtest(panel, w, costs, panel.funding).net_return
            active = w.abs().sum(axis=1) > 0
            first_active[name] = active.idxmax() if bool(active.any()) else None
        returns = pd.DataFrame(returns_by)

        refits = np.flatnonzero(refit_rows(index, str(params["refit_every"])))
        choices: dict[pd.Timestamp, str | None] = {}
        out = np.zeros((len(index), len(panel.prices.columns)))
        for n, row in enumerate(refits):
            tau = index[row]
            previous_refit = index[refits[n - 1]] if n > 0 else None
            past = returns[returns.index < tau]
            if lookback is not None:
                past = past[past.index >= tau - lookback]
            best, best_score = None, -np.inf
            for name in candidates:
                started = first_active[name]
                # Eligible only after trading through one whole period.
                if started is None or previous_refit is None or started >= previous_refit:
                    continue
                score = _score(past[name][past.index >= started], objective)
                if not np.isnan(score) and score > best_score:
                    best, best_score = name, score
            choices[tau] = best
            stop = refits[n + 1] if n + 1 < len(refits) else len(index)
            if best is not None:
                out[row:stop] = weights_by[best].to_numpy(dtype=float)[row:stop]
        weights = pd.DataFrame(out, index=index, columns=panel.prices.columns)
        weights = weights.where(panel.tradeable, 0.0)

        result = RetuneRun(
            weights=weights,
            choices=pd.Series(choices, dtype=object, name="choice"),
            candidate_returns=returns,
        )
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
        self._last = result
        return result


__all__ = ["Retune", "RetuneRun", "expand_grid", "refit_rows"]
