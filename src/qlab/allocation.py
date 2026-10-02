"""Switching capital between strategies by market regime (docs/REGIMES.md).

Owner, 2026-10-02: «переключение между стратегиями по режиму — делай».

The strategies live on different data -- trend decides daily over a wide coin
list, Bv2 hourly on four coins with spot -- so they do not share one panel.
The switch is therefore made between ACCOUNTS: each strategy (a leg) is run
by the stand on its own panel, exactly as `qlab evaluate` runs it; its net
returns are compounded per UTC day; and each day the capital sits in the leg
assigned to the regime known at the START of that day
(`qlab.regimes.causal_labels`: BTC's 30-day return against terciles of the
days before it). A regime assigned `None` holds cash.

Each switch pays the cost of closing the old leg's book and opening the new
one: gross exposure at the end of the previous day times that leg's own cost
per unit of turnover (taker + slippage from its spec).

What this approximates: every leg's book runs on in the background, so a leg
re-entered after a pause comes back with the positions and the internal state
(trend's book-volatility scale, Bv2's hedge latch) it would have had running
all along -- as if all books ran in paper and only the capital moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.pipeline.spec import StrategySpec


@dataclass(frozen=True)
class Leg:
    name: str
    daily_return: pd.Series  # by UTC day (00:00 of the day the returns end in)
    daily_gross: pd.Series  # gross exposure at the end of the day
    cost_rate: float  # per unit of turnover
    first_active: pd.Timestamp


def leg_from_spec(spec: StrategySpec, session) -> Leg:
    """Run a spec's strategy on its own panel, as the stand does, and fold it
    to daily returns and end-of-day gross exposure."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.pipeline.evaluate import resolve_panel, resolve_strategy

    panel = resolve_panel(session, spec)
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    net = run_backtest(panel, weights, costs, panel.funding).net_return
    index = panel.prices.index
    bar = index[1] - index[0]
    # A return at decision time t is realised over (t, t + bar]: it belongs to
    # the UTC day that period ends in (a period ending exactly at 00:00 closes
    # the previous day).
    day = (net.index + bar - pd.Timedelta(microseconds=1)).normalize()
    daily = (1.0 + net).groupby(day).prod() - 1.0
    gross = weights.abs().sum(axis=1)
    gross_day = gross.groupby((gross.index + bar - pd.Timedelta(microseconds=1)).normalize()).last()
    active = gross > 0
    return Leg(name=spec.idea_id, daily_return=daily, daily_gross=gross_day,
               cost_rate=(spec.costs.taker_fee_bps + spec.costs.slippage_bps) / 1e4,
               first_active=active.idxmax() if bool(active.any()) else index[-1])


def switch(legs: dict[str, Leg], assignment: dict[str, str | None], labels: pd.Series,
           days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Daily returns of the switched account over `days`, and the leg held
    each day. `labels` are causal regime labels indexed by the close time of
    the day they describe; day d uses the label known at its start (d 00:00)."""
    known = labels.dropna()
    regime_at = known.reindex(days, method="ffill")
    held = regime_at.map(lambda r: assignment.get(r) if isinstance(r, str) else None)
    out = np.zeros(len(days))
    previous: str | None = None
    for i, (d, leg_name) in enumerate(zip(days, held, strict=True)):
        r = 0.0
        if leg_name is not None:
            r = float(legs[leg_name].daily_return.get(d, 0.0))
        if leg_name != previous:
            yesterday = d - pd.Timedelta(days=1)
            for name in (previous, leg_name):
                if name is not None:
                    leg = legs[name]
                    gross = leg.daily_gross.asof(yesterday)
                    r -= leg.cost_rate * (0.0 if pd.isna(gross) else float(gross))
            previous = leg_name
        out[i] = r
    return pd.Series(out, index=days), held


def summary(returns: pd.Series) -> dict[str, float]:
    growth = float((1.0 + returns).prod())
    equity = (1.0 + returns).cumprod()
    std = returns.std(ddof=1)
    return {
        "ann_return": growth ** (365 / len(returns)) - 1.0 if growth > 0 else -1.0,
        "sharpe": float(returns.mean() / std * np.sqrt(365)) if std else float("nan"),
        "max_dd": float((equity / equity.cummax() - 1.0).min()),
        "days": float(len(returns)),
    }


__all__ = ["Leg", "leg_from_spec", "summary", "switch"]
