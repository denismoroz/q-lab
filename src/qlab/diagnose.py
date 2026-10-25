"""Where a strategy lost money: drawdown evidence for the researcher
(docs/NIGHT.md, docs/FIT_VS_FORWARD.md).

Owner, 2026-10-25: «данные есть - отлично - проверили - если данных мало -
смотрим где стратегия плохо работает - основные просадки и их пытаемся
исправить». The agreed refinement: a fix designed by looking at drawdowns must be
checked on data the fix did not see. So the evidence is computed ONLY over the part
of the data before a cut, and the part after it stays untouched for judging the fix:

- the cut is the spec's `params_fixed_at` when it has one (the date the parameters
  were chosen: the data after it is the forward test); a causal chooser
  (`selects_causally`) is the exception, because its `params_fixed_at` is only the
  end of its warm-up, not a cut between seen and unseen data;
- otherwise the cut is the point after which the ruleset's `holdout_share` of the
  window lies (the 2/3 : 1/3 split of rules 2026-10-24.1; the share is read from the
  ruleset, never written here).

**This is not a new selection.** `diagnose_spec` re-runs a strategy that is already
recorded, on the panel that is already cached, to describe its history; it chooses
nothing, so it writes no `trial` (CLAUDE.md: a trial is a selection step, counted
in deflation). The strategy is run on the panel TRUNCATED at the cut, so the
held-out part never reaches the strategy, not just the report. It never fetches:
a panel that is not cached yields `None`.

The output is plain data and no judgement: the three deepest drawdowns, the returns
by BTC market regime (`qlab.regimes`; its labels describe the past and use BTC
prices up to 15 days after a day, never the strategy's results) and the five worst
calendar months.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from qlab.regimes import REGIMES, RegimeSeries
from qlab.regimes import breakdown as regime_breakdown

TOP_DRAWDOWNS = 3
WORST_MONTHS = 5


@dataclass(frozen=True)
class Drawdown:
    """One peak-to-recovery episode of the equity curve."""

    peak: date
    trough: date
    recovered: date | None  # None: the curve has not regained the peak by the cut
    depth: float  # negative: -0.2 is a 20% fall


@dataclass(frozen=True)
class RegimeReturn:
    regime: str
    days: float
    ret: float | None  # compounded over the regime's periods; None: too few periods
    max_dd: float | None
    btc_return: float | None


@dataclass(frozen=True)
class MonthReturn:
    month: str  # "2025-03"
    ret: float
    periods: int


@dataclass(frozen=True)
class Cut:
    """The first row held out, where the cut comes from, and the date."""

    row: int
    day: date
    source: str


@dataclass(frozen=True)
class Diagnosis:
    idea_id: str
    first: date | None
    last: date | None  # last day of the evidence, before the cut
    cut: Cut
    periods: int
    held_out_days: int  # calendar days from the cut to the end of the window
    drawdowns: list[Drawdown] = field(default_factory=list)
    regimes: list[RegimeReturn] | None = None  # None: no stored regime labels
    worst_months: list[MonthReturn] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def choose_cut(index: pd.DatetimeIndex, *, params_fixed_at: date | None, share: float | None,
               selects_causally: bool = False) -> Cut:
    """The first held-out row of a window of bars `index`.

    `params_fixed_at` wins when the spec has one and the window has bars on or
    after it (not a causal chooser's warm-up end). Otherwise `share` (the
    ruleset's `holdout_share`) is the held-out part of the window: the cut falls
    after `1 - share` of its bars.
    Raises ValueError when neither is available."""
    n = len(index)
    after_window = False
    if params_fixed_at is not None and not selects_causally:
        row = int(index.searchsorted(pd.Timestamp(params_fixed_at, tz="UTC")))
        if row < n:
            return Cut(row, _day(index, row), f"params_fixed_at {params_fixed_at}")
        after_window = True  # no bar on or after it: nothing of this window would be held out
    if share is None:
        raise ValueError("no cut: the spec has no usable params_fixed_at and the ruleset has no "
                         "holdout_share")
    row = min(max(int(round(n * (1.0 - share))), 1), n)
    why = ("the spec's params_fixed_at is the end of a causal warm-up, not a cut; "
           if selects_causally and params_fixed_at is not None
           else f"the spec's params_fixed_at {params_fixed_at} is after the last bar of this "
                "window, so nothing would be held out; " if after_window
           else "the spec has no params_fixed_at; ")
    return Cut(row, _day(index, row), why + f"the last {share:.0%} of the window is held out "
               "(the ruleset's holdout_share)")


def _day(index: pd.DatetimeIndex, row: int) -> date:
    """The date of bar `row`; past the end, one bar after the last."""
    if row < len(index):
        return index[row].date()
    step = index[-1] - index[-2] if len(index) > 1 else pd.Timedelta(days=1)
    return (index[-1] + step).date()


def equity_from_returns(net_return: pd.Series) -> pd.Series:
    """Equity (start 1.0) at the END of each period, preceded by the start point.
    A return is indexed by decision time t and realised over (t, t+1]."""
    if len(net_return) == 0:
        return pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("1970-01-01", tz="UTC")]))
    step = (net_return.index.to_series().diff().median() if len(net_return) > 1
            else pd.Timedelta(days=1))
    ends = net_return.index + step
    growth = (1.0 + net_return).cumprod()
    return pd.concat([pd.Series([1.0], index=net_return.index[:1]),
                      pd.Series(growth.to_numpy(), index=ends)])


def deepest_drawdowns(equity: pd.Series, top: int = TOP_DRAWDOWNS) -> list[Drawdown]:
    """The `top` deepest peak-to-recovery episodes of an equity curve, deepest
    first. An episode runs from a running maximum until the curve regains it
    (`recovered`) or to the end of the data (`recovered` is None)."""
    values = equity.to_numpy(dtype=float)
    episodes: list[Drawdown] = []
    peak_i = 0
    trough_i = 0
    in_episode = False

    def _close(recovered_i: int | None) -> None:
        episodes.append(Drawdown(
            peak=equity.index[peak_i].date(), trough=equity.index[trough_i].date(),
            recovered=None if recovered_i is None else equity.index[recovered_i].date(),
            depth=float(values[trough_i] / values[peak_i] - 1.0)))

    for i in range(1, len(values)):
        if values[i] >= values[peak_i]:
            if in_episode:
                _close(i)
                in_episode = False
            peak_i = trough_i = i
        elif not in_episode:
            in_episode, trough_i = True, i
        elif values[i] < values[trough_i]:
            trough_i = i
    if in_episode:
        _close(None)
    return sorted(episodes, key=lambda e: e.depth)[:top]


def worst_months(net_return: pd.Series, n: int = WORST_MONTHS) -> list[MonthReturn]:
    """The `n` calendar months with the lowest compounded return, worst first.
    A return belongs to the month in which its period ends."""
    if len(net_return) == 0:
        return []
    equity = equity_from_returns(net_return)
    by_end = pd.Series(net_return.to_numpy(), index=equity.index[1:])
    months = by_end.groupby(by_end.index.strftime("%Y-%m"))
    rows = [MonthReturn(month=str(month), ret=float((1.0 + part).prod() - 1.0), periods=len(part))
            for month, part in months]
    return sorted(rows, key=lambda m: m.ret)[:n]


def regime_returns(net_return: pd.Series, regimes: RegimeSeries, periods_per_year: float
                   ) -> list[RegimeReturn]:
    """The strategy's returns on BTC's bull / flat / bear days
    (`qlab.regimes.breakdown`)."""
    measured = regime_breakdown(net_return, regimes, periods_per_year)
    return [RegimeReturn(
        regime=r, days=measured.get(f"regime_{r}_days", 0.0),
        ret=measured.get(f"regime_{r}_return"), max_dd=measured.get(f"regime_{r}_max_dd"),
        btc_return=measured.get(f"regime_{r}_btc_return")) for r in REGIMES]


def describe(net_return: pd.Series, *, idea_id: str, cut: Cut, held_out_days: int,
             regimes: RegimeSeries | None) -> Diagnosis:
    """The evidence over `net_return`, which must already end before the cut."""
    notes: list[str] = []
    if len(net_return) < 2:
        return Diagnosis(idea_id, None, None, cut, len(net_return), held_out_days,
                         notes=["fewer than two periods before the cut: nothing to describe"])
    equity = equity_from_returns(net_return)
    ppy = float(pd.Timedelta(days=365.25) / net_return.index.to_series().diff().median())
    regime_rows = regime_returns(net_return, regimes, ppy) if regimes is not None else None
    if regimes is None:
        notes.append("no stored regime labels (`qlab regimes build`): regime breakdown omitted")
    return Diagnosis(
        idea_id=idea_id, first=net_return.index[0].date(), last=equity.index[-1].date(),
        cut=cut, periods=len(net_return), held_out_days=held_out_days,
        drawdowns=deepest_drawdowns(equity), regimes=regime_rows,
        worst_months=worst_months(net_return), notes=notes)


def diagnose_returns(net_return: pd.Series, *, idea_id: str, params_fixed_at: date | None,
                     share: float | None, regimes: RegimeSeries | None,
                     selects_causally: bool = False) -> Diagnosis:
    """Cut a whole-window return series and describe the part before the cut.
    The series-level entry point `diagnose_spec` reduces to, usable on any
    return series (the strategy is not run here)."""
    cut = choose_cut(net_return.index, params_fixed_at=params_fixed_at, share=share,
                     selects_causally=selects_causally)
    held_out = (net_return.index[-1].date() - cut.day).days if cut.row < len(net_return) else 0
    return describe(net_return.iloc[:cut.row], idea_id=idea_id, cut=cut,
                    held_out_days=max(held_out, 0), regimes=regimes)


def diagnose_spec(spec, session, *, share: float | None, regimes: RegimeSeries | None
                  ) -> Diagnosis | None:
    """Run the recorded strategy of `spec` on its cached panel, truncated at the
    cut, and describe it. None when the panel is not cached (nothing is
    fetched). No trial is written -- see the module docstring."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.harness.strategy import own_costs
    from qlab.pipeline.evaluate import (
        _slice_rows,
        complete_book_window,
        resolve_cached_panel,
        resolve_strategy,
    )

    panel = resolve_cached_panel(session, spec)
    if panel is None:
        return None
    index = panel.prices.index
    first, last = 0, len(index) - 1
    if spec.required_instruments is not None:
        coverage = complete_book_window(panel, spec.required_instruments)
        if coverage.window is not None:  # the window the verdict is rendered on
            first, last = coverage.window
    cut = choose_cut(index[first:last + 1], params_fixed_at=spec.params_fixed_at, share=share,
                     selects_causally=spec.selects_causally)
    cut_row = first + cut.row  # the first held-out row of the whole panel
    if cut_row - first < 3:
        return Diagnosis(spec.idea_id, None, None, cut, 0, 0,
                         notes=["the cut leaves fewer than three bars before it"])
    seen = _slice_rows(panel, 0, cut_row - 1)  # the held-out part never reaches the strategy
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(seen, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    net_return = run_backtest(seen, weights, costs, seen.funding,
                              extra_cost=own_costs(strategy, seen, spec.params)
                              ).net_return.iloc[first:]
    held_out = (index[last].date() - cut.day).days if cut_row <= last else 0
    return describe(net_return, idea_id=spec.idea_id,
                    cut=Cut(cut_row, cut.day, cut.source), held_out_days=max(held_out, 0),
                    regimes=regimes)


def _pct(value: float | None) -> str:
    return "n/a" if value is None or value != value or np.isinf(value) else f"{value:+.1%}"


def render(diagnosis: Diagnosis) -> list[str]:
    """The evidence as markdown lines for the researcher's context file."""
    d = diagnosis
    lines = [f"Cut date: {d.cut.day} ({d.cut.source}). "
             f"Everything below is computed on the data BEFORE the cut"
             + (f": {d.first} .. {d.last}, {d.periods} periods." if d.first else "."),
             f"The data after the cut ({d.held_out_days} days) is held out."]
    lines += [f"- note: {n}" for n in d.notes]
    if d.drawdowns:
        lines += ["", "Deepest drawdowns (peak -> trough -> recovery, depth):"]
        lines += [f"- {e.peak} -> {e.trough} -> "
                  f"{e.recovered if e.recovered else 'not recovered by the cut'}, "
                  f"{e.depth:.1%}" for e in d.drawdowns]
    if d.regimes is not None:
        lines += ["", "Returns by BTC market regime (the strategy's compounded return on the "
                  "regime's days only; BTC's own return over those same days in brackets):"]
        lines += [f"- {r.regime}: {_pct(r.ret)} over {r.days:.0f} days, worst fall "
                  f"{_pct(r.max_dd)} (BTC {_pct(r.btc_return)})" for r in d.regimes]
    if d.worst_months:
        lines += ["", "Worst calendar months:"]
        lines += [f"- {m.month}: {_pct(m.ret)} ({m.periods} periods)" for m in d.worst_months]
    return lines


__all__ = [
    "Cut",
    "Diagnosis",
    "Drawdown",
    "MonthReturn",
    "RegimeReturn",
    "choose_cut",
    "deepest_drawdowns",
    "describe",
    "diagnose_returns",
    "diagnose_spec",
    "equity_from_returns",
    "regime_returns",
    "render",
    "worst_months",
]
