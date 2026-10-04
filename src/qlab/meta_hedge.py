"""Level 2 for Bv2: should this coin's spot be hedged now? (docs/TASKS.md T39,
docs/REGIME_DETECT.md).

Owner, 2026-10-04: «так же можно поступить с bv2 — чтобы принимать решение —
хеджировать или нет — то есть отдельная модель может это говорить»; «да,
делай для bv2».

Same two levels as `qlab.meta_label`, at Bv2's own grain -- per coin and per
hour, because the book decides that way and a single daily market signal
lost to it (the first T39 attempt):

- **level 1**: the market regime probabilities (`qlab.regime_detect`), the
  latest made by each hour's close;
- **level 2**: one model pooled over the coins. At each hourly close it says
  whether a hedge opened now would pay over the next `HORIZON_HOURS`: the
  short perp earns minus the coin's return plus the funding it receives, and
  pays a perp round trip (`round_trip_cost`). The book hedges when that is
  more likely than not; the sticky exit and everything else stay frab's
  (`qlab.strategies.live.bv2`, `hedge_by: {meta: <name>}`).

Discipline as in level 1 (tested in `test_meta_hedge.py`): features use data
up to the hour's close; an hour's label needs `HORIZON_HOURS` more closes and
funding, so it is known only then; models are refit monthly on labels known
`EMBARGO_DAYS` before the month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.regime_detect import EMBARGO_DAYS

HORIZON_HOURS = 14 * 24
"""Bv2's shorter window (frab, src/frab/strategy/b2/book.py: `H14 = 14 * 24`):
the span the book's own rule looks back over, used as the span a hedge must
pay over."""

FEATURE_DAYS = (3, 7, 14, 30)
"""Coin returns over these many days: 14 and 30 are the book's own rule's
inputs (book.py `H14, H30`); 3 and 7 the shorter end of the range measured in
docs/REGIMES.md («the last 7-90 days»), 3 for the start of a fall."""

MIN_CLASS_HOURS = 30 * 24
"""Each outcome must have a regime's worth of training hours before a model
is fitted -- the rules' `regime_coverage_days` (30) in hours."""


def round_trip_cost(params: dict) -> float:
    """Opening and closing the hedge: twice the perp taker fee plus slippage
    (frab, book.py: `perp_cost = PERP_TAKER + params.slippage`)."""
    from qlab.strategies.live._loader import import_frab

    taker = import_frab("frab.constants").PERP_TAKER
    return 2.0 * (taker + float(params["slippage"]))


def coin_features(perp: pd.Series, funding: pd.Series, level1: pd.DataFrame) -> pd.DataFrame:
    """Per hourly bar (indexed by open time; the bar's price is its close, an
    hour later): the coin's returns, volatility, drawdown, funding, and level
    1's probabilities known by the bar's close."""
    hours = {d: d * 24 for d in FEATURE_DAYS}
    frame = pd.DataFrame({f"ret_{d}d": perp / perp.shift(h) - 1.0 for d, h in hours.items()},
                         index=perp.index)
    hourly = np.log(perp).diff()
    frame["vol_7d"] = hourly.rolling(7 * 24).std()
    frame["drawdown_30d"] = perp / perp.rolling(30 * 24).max() - 1.0
    frame["funding_3d"] = funding.rolling(3 * 24, min_periods=1).sum()
    closes = perp.index + pd.Timedelta(hours=1)
    known = level1[["p_bull", "p_bear"]].reindex(closes, method="ffill")
    frame["p_bull"] = known["p_bull"].to_numpy()
    frame["p_bear"] = known["p_bear"].to_numpy()
    return frame


def hedge_pays(perp: pd.Series, funding: pd.Series, cost: float) -> pd.Series:
    """Per bar: would a hedge opened at this bar's close and held
    `HORIZON_HOURS` have paid? The short earns -(price change) and receives
    the funding of the bars after it, minus a round trip. NaN until known."""
    ahead = perp.shift(-HORIZON_HOURS) / perp - 1.0
    paid_funding = funding.fillna(0.0)[::-1].rolling(HORIZON_HOURS).sum()[::-1].shift(-1)
    pnl = -ahead + paid_funding - cost
    return (pnl > 0).astype(float).where(ahead.notna())


def known_at(labels: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Labels known at `as_of`: the close `HORIZON_HOURS` after the bar (the
    bar opening `HORIZON_HOURS` later closes an hour after its open)."""
    done = labels.index + pd.Timedelta(hours=HORIZON_HOURS + 1)
    return labels[(done <= as_of) & labels.notna()]


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def walk_forward(features: dict[str, pd.DataFrame], labels: dict[str, pd.Series],
                 start: pd.Timestamp) -> pd.DataFrame:
    """P(the hedge pays) per bar and coin from `start`; one model pooled over
    the coins, refit monthly on labels known `EMBARGO_DAYS` before the month.
    Columns `<coin>` (hedge wish, bool) and `p_<coin>`, plus trained_until."""
    coins = sorted(features)
    clean = {c: features[c].dropna() for c in coins}
    last = max(f.index[-1] for f in clean.values())
    months = pd.date_range(start.normalize(), last, freq="MS", tz="UTC")
    if len(months) == 0 or months[0] > start:
        months = months.insert(0, start)
    out = []
    for k, month in enumerate(months):
        until = months[k + 1] if k + 1 < len(months) else last + pd.Timedelta(hours=1)
        cutoff = month - pd.Timedelta(days=EMBARGO_DAYS)
        xs, ys = [], []
        for c in coins:
            y = known_at(labels[c], month)
            y = y[y.index <= cutoff]
            rows = clean[c].index.intersection(y.index)
            xs.append(clean[c].loc[rows])
            ys.append(y.loc[rows])
        x, y = pd.concat(xs), pd.concat(ys)
        counts = y.value_counts()
        if len(counts) < 2 or counts.min() < MIN_CLASS_HOURS:
            continue
        model = _model().fit(x.to_numpy(), y.to_numpy())
        one = list(model.classes_).index(1.0)
        block = {}
        for c in coins:
            target = clean[c][(clean[c].index >= max(month, start)) & (clean[c].index < until)]
            if not target.empty:
                block[f"p_{c}"] = pd.Series(model.predict_proba(target.to_numpy())[:, one],
                                            index=target.index)
        if block:
            frame = pd.DataFrame(block)
            frame["trained_until"] = y.index.max()
            out.append(frame)
    if not out:
        raise ValueError("no month had enough of both outcomes in its training data")
    result = pd.concat(out)
    for c in coins:
        result[c] = result[f"p_{c}"] > 0.5
    return result


__all__ = ["HORIZON_HOURS", "coin_features", "hedge_pays", "known_at", "round_trip_cost",
           "walk_forward"]
