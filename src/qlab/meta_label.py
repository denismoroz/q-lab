"""Two levels: the market's direction, then a decision for one strategy
(docs/TASKS.md T39, docs/REGIME_DETECT.md).

Owner, 2026-10-04: «делай мета-разметку для trend»; «по-моему тут нужно
несколько уровней — направление рынка, а уже второй уровень для стратегии».

- **Level 1** is the market regime detector (`qlab.regime_detect`): stored
  walk-forward probabilities of bull / flat / bear, each made by a model
  trained before its day.
- **Level 2** is learned for ONE strategy: from level 1's probabilities and
  the strategy's own recent state, will the strategy earn over the next
  `HORIZON_DAYS`? The label is the strategy's own outcome -- what the first
  attempt lacked: a detector that guessed the regime better made trend worse,
  because guessing a label half built on the future is not knowing when a
  strategy loses.

The strategy trades on days level 2 says it will earn and holds cash on the
others (`qlab.strategies.regime_gate.RegimeGate`, detector `meta:<name>`).

Discipline, as in level 1 and tested in `test_meta_label.py`:

- a day's label needs the strategy's returns over the next 30 days, so it is
  known 30 days later; models are refit monthly on labels known a full
  `EMBARGO_DAYS` before the month;
- features at a close use level 1's prediction made at that close and the
  strategy's returns realised up to it;
- the strategy's returns are its UNGATED returns on the stand (the harness on
  the spec's own panel): level 2 decides whether to take them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.regime_detect import EMBARGO_DAYS
from qlab.regimes import DEFAULT_DIR, WINDOW_DAYS

HORIZON_DAYS = WINDOW_DAYS
"""Owner, 2026-10-02: «окно 30 дней» -- the regime window, used again as the
horizon the strategy's outcome is judged over."""

MIN_CLASS_DAYS = 30
"""A model is fitted only once each outcome (earned / lost) has this many
training days: the rules' own `regime_coverage_days` (rules/2026-10-02.3.yaml)
-- a regime counts as seen after 30 days of it."""

PREFIX = "meta-"


def strategy_returns(spec_path: Path, session) -> pd.Series:
    """The spec's ungated daily net returns from the stand, re-indexed by the
    CLOSE at which each weight was decided: a daily panel is stamped by candle
    open and its price at t is the close at t + 1 day, so the return at panel
    row t is realised over (close t+1d, close t+2d]."""
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.pipeline.evaluate import resolve_panel, resolve_strategy
    from qlab.pipeline.spec import load_spec

    spec = load_spec(spec_path)
    if spec.data.interval != "1d":
        raise ValueError("level 2 is daily: the spec must be on 1d bars")
    panel = resolve_panel(session, spec)
    weights = resolve_strategy(spec.code_ref).target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    net = run_backtest(panel, weights, costs, panel.funding).net_return
    return net.set_axis(net.index + pd.Timedelta(days=1))


def features(level1: pd.DataFrame, btc: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Per close: level 1's probabilities made at that close, BTC's trailing
    30-day return (what the simple detector reads), and the strategy's own
    return and volatility over the 30 days realised by that close."""
    realised = returns.shift(1)  # the return at close c is realised a day later
    frame = pd.DataFrame({
        "p_bull": level1["p_bull"], "p_flat": level1["p_flat"], "p_bear": level1["p_bear"],
    })
    frame["btc_ret_30"] = (btc / btc.shift(WINDOW_DAYS) - 1.0).reindex(frame.index)
    frame["own_ret_30"] = realised.rolling(WINDOW_DAYS).sum().reindex(frame.index)
    frame["own_vol_30"] = realised.rolling(WINDOW_DAYS).std().reindex(frame.index)
    return frame


def outcome(returns: pd.Series) -> pd.Series:
    """Per close: did the strategy earn over the next `HORIZON_DAYS` (the
    returns decided at that close and the following ones)? NaN until known."""
    ahead = np.log1p(returns)[::-1].rolling(HORIZON_DAYS).sum()[::-1]
    return (ahead > 0).astype(float).where(ahead.notna())


def known_at(returns: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Outcomes known at `as_of`: every return they need was realised by then
    (the last one, decided at close c, is realised at c + 1 day)."""
    seen = returns[returns.index + pd.Timedelta(days=1) <= as_of]
    return outcome(seen).dropna()


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Same as level 1: scikit-learn's defaults, nothing tuned.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def walk_forward(feats: pd.DataFrame, returns: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """P(the strategy earns over the next 30 days) per close from `start`,
    each month's model fitted on outcomes known `EMBARGO_DAYS` before it.
    `trade` is P > 0.5: earn more likely than not."""
    feats = feats.dropna()
    months = pd.date_range(start.normalize(), feats.index[-1], freq="MS", tz="UTC")
    if len(months) == 0 or months[0] > start:
        months = months.insert(0, start)
    rows = []
    for k, month in enumerate(months):
        until = months[k + 1] if k + 1 < len(months) else feats.index[-1] + pd.Timedelta(days=1)
        labels = known_at(returns, month)
        labels = labels[labels.index <= month - pd.Timedelta(days=EMBARGO_DAYS)]
        train = feats.index.intersection(labels.index)
        counts = labels.loc[train].value_counts()
        if len(counts) < 2 or counts.min() < MIN_CLASS_DAYS:
            continue
        model = _model().fit(feats.loc[train], labels.loc[train])
        target = feats[(feats.index >= max(month, start)) & (feats.index < until)]
        if target.empty:
            continue
        p = model.predict_proba(target)[:, list(model.classes_).index(1.0)]
        rows.append(pd.DataFrame({"p_earn": p, "trained_until": train.max()},
                                 index=target.index))
    if not rows:
        raise ValueError("no month had enough of both outcomes in its training data")
    out = pd.concat(rows)
    out["trade"] = out["p_earn"] > 0.5
    return out[["p_earn", "trade", "trained_until"]]


def store(name: str, decisions: pd.DataFrame, manifest: dict,
          directory: Path = DEFAULT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{PREFIX}{name}.parquet"
    decisions.to_parquet(path)
    (directory / f"{PREFIX}{name}.json").write_text(json.dumps(
        {**manifest, "built_at": datetime.now(UTC).isoformat()}, indent=1, default=str),
        encoding="utf-8")
    return path


def load_decisions(name: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    path = directory / f"{PREFIX}{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no level-2 decisions {path}: build them with "
                                "`qlab regimes meta`")
    return pd.read_parquet(path)


__all__ = ["HORIZON_DAYS", "MIN_CLASS_DAYS", "features", "known_at", "load_decisions",
           "outcome", "store", "strategy_returns", "walk_forward"]
