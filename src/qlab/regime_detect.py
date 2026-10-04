"""Learning to recognise the CURRENT market regime from the past alone
(docs/TASKS.md T39, docs/REGIME_DETECT.md).

Owner, 2026-10-03: «нам было бы здорово научиться определять тренд для
стратегии — у нас очень много данных и есть чёткий критерий — определение
тренда с заглядыванием вперёд — возможно это ML задача?»; the first target is
Bv2's hedge: «достаточно сильно ошибается с моментом, когда нужно включать
хеджирование». scikit-learn approved 2026-10-03.

The yardstick is the testing description (`qlab.regimes`): a day's state is
BTC's return over the 30 days AROUND it, in terciles. A strategy cannot know
it on the day -- half of its window is the future -- so the task is a
nowcast: from what is known at a day's close, guess the label that the next
fifteen days will give that day.

Discipline (every rule below is tested in `test_regime_detect.py`):

- **Features use only data up to the close they are stamped with** (BTC
  closes, the Binance market's breadth and funding) -- `features`.
- **Targets are rebuilt at every retraining from what was known then**: a
  day's centered label needs closes up to fifteen days after it, and its
  tercile thresholds come only from labels already known -- `known_labels`.
  The official labels use thresholds from the whole history; training on
  them would leak the future's distribution.
- **Walk-forward with a gap**: the model is refit at the start of every
  month on days whose labels were known by then, ending `EMBARGO_DAYS`
  before the month: adjacent days' 30-day labels overlap, so the last
  training days and the first predicted ones would otherwise share
  information -- `walk_forward`.
- **Success is not accuracy** but what the detector does for a strategy on
  the stand (noise, every trial counted, forward test). Accuracy and lag are
  reported to explain the stand's result, never to pick a variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.regimes import DEFAULT_DIR, REGIMES, WINDOW_DAYS

HALF = WINDOW_DAYS // 2
EMBARGO_DAYS = WINDOW_DAYS
"""docs/TASKS.md T39: «с зазором между обучением и проверкой (30-дневные
метки соседних дней перекрываются)»."""

RETURN_WINDOWS = (7, 14, 30, 60, 90)
"""BTC's return over these many days. 7-90: the range measured in
docs/REGIMES.md («the direction of the last 7-90 days»); 14 and 30: Bv2's own
hedge windows (frab, src/frab/strategy/b2/book.py: `H14, H30`)."""

VOL_WINDOWS = (7, 30)
"""Realised volatility over these many days: docs/REGIMES.md found the level
of volatility the one thing the past predicts (44-46% against 33%); 30 is the
regime window, 7 the shortest return window above."""

PREFIX = "ml-"


def return_spans(window: int = WINDOW_DAYS) -> list[int]:
    """`RETURN_WINDOWS` scaled to a regime window (7-90 days for 30 days;
    3-42 for 14)."""
    return sorted({max(1, round(window * w / WINDOW_DAYS)) for w in RETURN_WINDOWS})


def features(btc: pd.Series, market: pd.DataFrame | None = None,
             market_funding: pd.DataFrame | None = None,
             market_tradeable: pd.DataFrame | None = None) -> pd.DataFrame:
    """Past-only features, one row per BTC daily close (indexed by close
    time). `market*` are a whole-market daily panel (prices, funding,
    tradeable), indexed by CLOSE time too; when given they add breadth (the
    share of tradeable coins up over 7 and 30 days) and the market's mean
    funding over the last 7 days."""
    log = np.log(btc)
    out = {f"ret_{w}": btc / btc.shift(w) - 1.0 for w in RETURN_WINDOWS}
    daily = log.diff()
    out.update({f"vol_{w}": daily.rolling(w).std() for w in VOL_WINDOWS})
    out["drawdown_90"] = btc / btc.rolling(90).max() - 1.0
    frame = pd.DataFrame(out, index=btc.index)
    if market is not None:
        alive = market_tradeable if market_tradeable is not None else market.notna()
        for w in (7, 30):
            up = (market / market.shift(w) - 1.0) > 0
            counted = alive & market.shift(w).notna()
            share = (up & counted).sum(axis=1) / counted.sum(axis=1).replace(0, np.nan)
            frame[f"breadth_{w}"] = share.reindex(frame.index)
    if market_funding is not None:
        alive = market_tradeable if market_tradeable is not None else market_funding.notna()
        mean = market_funding.where(alive).mean(axis=1)
        frame["funding_7"] = mean.rolling(7).mean().reindex(frame.index)
    return frame


def known_labels(btc: pd.Series, as_of: pd.Timestamp, window: int = WINDOW_DAYS) -> pd.Series:
    """Centered `window`-day labels of every day whose label was KNOWN at
    `as_of`: the close `window - window // 2` days after it is at or before
    `as_of`, and the tercile thresholds come only from those days."""
    seen = btc[btc.index <= as_of]
    back = window // 2
    around = seen.shift(-(window - back)) / seen.shift(back) - 1.0
    around = around.dropna()
    if around.empty:
        return pd.Series(dtype=object)
    low, high = around.quantile([1 / 3, 2 / 3]).to_list()
    return pd.Series(np.where(around >= high, "bull", np.where(around < low, "bear", "flat")),
                     index=around.index, dtype=object)


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # scikit-learn's defaults (C=1.0, L2), nothing tuned: docs/TASKS.md T39,
    # «модели — простые (логистическая регрессия ...)». max_iter only lets the
    # solver converge on unscaled-looking data; it changes no answer.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def walk_forward(btc: pd.Series, feats: pd.DataFrame, start: pd.Timestamp,
                 window: int = WINDOW_DAYS) -> pd.DataFrame:
    """Predicted probability of each regime for every close from `start`,
    each month's model fitted only on what was known at the month's start
    (module docstring). Columns p_bull, p_flat, p_bear, label, trained_until.

    `window` is the regime's span: 30 days by default (the testing
    description's); a strategy's detector may use a shorter one (owner,
    2026-10-04: «15 дней это очень много для крипты»; 14 days confirmed on
    the 2024-01 .. 2026-09 holdout, scripts/research/detector_windows.py). The
    gap before each month is one window, so adjacent labels never overlap."""
    feats = feats.dropna()
    months = pd.date_range(start.normalize(), feats.index[-1], freq="MS", tz="UTC")
    if len(months) == 0 or months[0] > start:
        months = months.insert(0, start)
    rows = []
    for k, month in enumerate(months):
        until = months[k + 1] if k + 1 < len(months) else feats.index[-1] + pd.Timedelta(days=1)
        labels = known_labels(btc, month, window)
        labels = labels[labels.index <= month - pd.Timedelta(days=window)]
        train = feats.index.intersection(labels.index)
        if set(labels.loc[train]) != set(REGIMES):
            continue  # not every regime seen yet: no model, no prediction
        model = _model().fit(feats.loc[train], labels.loc[train])
        target = feats[(feats.index >= max(month, start)) & (feats.index < until)]
        if target.empty:
            continue
        proba = pd.DataFrame(model.predict_proba(target), index=target.index,
                             columns=[f"p_{c}" for c in model.classes_])
        proba["label"] = [model.classes_[i] for i in proba.to_numpy().argmax(axis=1)]
        proba["trained_until"] = train.max()
        rows.append(proba)
    if not rows:
        raise ValueError("no month had every regime in its training data")
    return pd.concat(rows)[["p_bull", "p_flat", "p_bear", "label", "trained_until"]]


def persistent(proba: pd.DataFrame, penalty: float) -> pd.Series:
    """A regime that does not flicker (docs/TASKS.md T39; owner, 2026-10-04:
    «делай устойчивость режима для trend»).

    The statistical jump model's idea (Shu, Yu, Mulvey, Journal of Asset
    Management 2024, arXiv 2402.05272): a regime path pays each day's cost of
    its state -- here minus the log of level 1's probability -- plus
    `penalty` at every change of state. Online, the day's regime is the last
    state of the cheapest path over the days up to it: the forward recursion
    D_t(s) = cost_t(s) + min(D_{t-1}(s), min_{s' != s} D_{t-1}(s') + penalty),
    regime_t = argmin_s D_t(s). It reads only probabilities made up to the
    day. `penalty` 0 is the most probable regime each day."""
    states = [f"p_{r}" for r in REGIMES]
    cost = -np.log(proba[states].clip(lower=1e-9).to_numpy())
    best = np.zeros(len(states))
    out = []
    for row in cost:
        if np.isnan(row).any():
            out.append(None)
            continue
        switch = best.min() + penalty
        best = row + np.minimum(best, switch)
        out.append(REGIMES[int(best.argmin())])
    return pd.Series(out, index=proba.index, dtype=object)


# --- comparison with simple detectors --------------------------------------

def bv2_rule(btc: pd.Series) -> pd.Series:
    """Bv2's own hedge wish on daily BTC closes: on unless BTC rose over both
    14 and 30 days (frab, src/frab/strategy/b2/book.py, `signals_at`, with
    `hedge_threshold` 0.0 as in production). True = hedge on."""
    up = (btc / btc.shift(14) - 1.0 > 0.0) & (btc / btc.shift(30) - 1.0 > 0.0)
    return (~up).where(btc.shift(30).notna())


@dataclass(frozen=True)
class HedgeScore:
    """A hedge signal against the regime labels: how much of each regime it
    covers, and how late it comes to a fall."""

    hedged_share: float
    bear_hedged: float  # share of bear days hedged (the protection)
    bull_hedged: float  # share of bull days hedged (the cost)
    flat_hedged: float
    median_lag_days: float  # from a bear stretch's first day to the first hedged day in it
    stretches: int


def score_hedge(hedge: pd.Series, labels: pd.Series) -> HedgeScore:
    both = pd.concat({"h": hedge, "l": labels}, axis=1).dropna()
    h = both["h"].astype(bool)
    lab = both["l"]
    lags = []
    run_id = (lab != lab.shift()).cumsum()
    for _, stretch in lab[lab == "bear"].groupby(run_id[lab == "bear"]):
        hit = h.loc[stretch.index]
        if hit.any():
            lags.append((hit.idxmax() - stretch.index[0]) / pd.Timedelta(days=1))
        else:
            lags.append(float(len(stretch)))
    return HedgeScore(
        hedged_share=float(h.mean()),
        bear_hedged=float(h[lab == "bear"].mean()),
        bull_hedged=float(h[lab == "bull"].mean()),
        flat_hedged=float(h[lab == "flat"].mean()),
        median_lag_days=float(np.median(lags)) if lags else float("nan"),
        stretches=len(lags),
    )


def accuracy(predicted: pd.Series, labels: pd.Series) -> dict[str, float]:
    both = pd.concat({"p": predicted, "l": labels}, axis=1).dropna()
    out = {"accuracy": float((both["p"] == both["l"]).mean()), "days": float(len(both))}
    for r in REGIMES:
        is_r = both["l"] == r
        said_r = both["p"] == r
        out[f"{r}_recall"] = float((said_r & is_r).sum() / max(is_r.sum(), 1))
        out[f"{r}_precision"] = float((said_r & is_r).sum() / max(said_r.sum(), 1))
    return out


# --- storage ----------------------------------------------------------------

def store(name: str, predictions: pd.DataFrame, manifest: dict,
          directory: Path = DEFAULT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{PREFIX}{name}.parquet"
    predictions.to_parquet(path)
    (directory / f"{PREFIX}{name}.json").write_text(json.dumps(
        {**manifest, "built_at": datetime.now(UTC).isoformat()}, indent=1, default=str),
        encoding="utf-8")
    return path


def load_predictions(name: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    path = directory / f"{PREFIX}{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no regime predictions {path}: build them with "
                                "`qlab regimes detect`")
    return pd.read_parquet(path)


__all__ = ["EMBARGO_DAYS", "HedgeScore", "accuracy", "bv2_rule", "features", "known_labels",
           "persistent",
           "load_predictions", "score_hedge", "store", "walk_forward"]
