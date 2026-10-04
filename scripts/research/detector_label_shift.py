"""Training labels with less of the future in them (docs/TASKS.md T39; owner,
2026-10-04: «возможно проблема с тем, что происходит позднее предсказывание —
разметку может тоже нужно адаптировать к разным таймфреймам»).

The testing label of a day looks 15 days ahead; a model of the past only
must be late against it. Here the TRAINING label is a 30-bar window with
F bars after the bar (F = 15 centered as now, 10, 5), learned on hourly BTC
with scale-free features (detector_timeframes.py) and applied to daily data.
The yardstick stays the daily testing label -- changing it would only move
the goalposts -- and both accuracy and lag are reported: days from the start
of each bull / bear stretch of 15 days or more to the first day the pair
says so, and how long it keeps saying so after the stretch ends.

Development only (2020-04 .. 2023-12).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from detector_binary import binary_scores, combine  # noqa: E402
from detector_timeframes import DEV, WINDOW, _model, hourly_btc, scale_free  # noqa: E402

from qlab import regime_detect as rd  # noqa: E402
from qlab.regimes import load, market_closes  # noqa: E402

MIN_STRETCH = 15  # half the regime window: shorter stretches are label flicker


def shifted_labels(closes: pd.Series, as_of: pd.Timestamp, future: int) -> pd.Series:
    seen = closes[closes.index <= as_of]
    around = (seen.shift(-future) / seen.shift(WINDOW - future) - 1.0).dropna()
    lo, hi = around.quantile([1 / 3, 2 / 3]).to_list()
    return pd.Series(np.where(around >= hi, "bull", np.where(around < lo, "bear", "flat")),
                     index=around.index)


def walk(hourly: pd.Series, daily_x: pd.DataFrame, target: str, future: int) -> pd.Series:
    train_x = scale_free(hourly).dropna()
    daily_x = daily_x.dropna()
    out = []
    for month in pd.date_range(DEV[0], DEV[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        labels = shifted_labels(hourly, month, future)
        labels = labels[labels.index <= month - pd.Timedelta(days=rd.EMBARGO_DAYS)]
        rows = train_x.index.intersection(labels.index)[::4]
        y = (labels.loc[rows] == target).astype(int)
        part = daily_x[(daily_x.index >= month) & (daily_x.index < min(nxt, DEV[1]))]
        if part.empty or y.nunique() < 2:
            continue
        fit = _model().fit(train_x.loc[rows].to_numpy(), y.to_numpy())
        out.append(pd.Series(fit.predict_proba(part.to_numpy())[:, 1], index=part.index))
    return pd.concat(out)


def lag(signal: pd.Series, labels: pd.Series, regime: str) -> tuple[float, float]:
    days = signal.index.intersection(labels.dropna().index)
    lab = labels.loc[days]
    run = (lab != lab.shift()).cumsum()
    starts, ends = [], []
    for _, s in lab[lab == regime].groupby(run[lab == regime]):
        if len(s) < MIN_STRETCH:
            continue
        seen = signal.reindex(s.index) == regime
        starts.append((seen.idxmax() - s.index[0]).days if seen.any() else len(s))
        after = signal.reindex(pd.date_range(s.index[-1], periods=61, freq="1D",
                                             tz="UTC")[1:]) == regime
        ends.append(int((~after).to_numpy().argmax()) + 1 if (~after).any() else 60)
    return float(np.median(starts)), float(np.median(ends))


def main() -> None:
    daily, hourly = market_closes(), hourly_btc()
    labels = load().labels
    daily_x = scale_free(daily)
    for future in (15, 10, 5):
        up = walk(hourly, daily_x, "bull", future)
        down = walk(hourly, daily_x, "bear", future)
        su, sd = binary_scores(up, labels, "bull"), binary_scores(down, labels, "bear")
        pair = combine(up, down)
        acc = rd.accuracy(pair, labels)["accuracy"]
        bs, be = lag(pair, labels, "bear")
        us, ue = lag(pair, labels, "bull")
        print(f"label: {WINDOW - future} bars back, {future} ahead | UP auc {su['auc']:.3f} | "
              f"DOWN auc {sd['auc']:.3f} | acc {acc:.3f} | bear noticed after {bs:.0f}d, "
              f"held {be:.0f}d past end | bull noticed after {us:.0f}d, held {ue:.0f}d",
              flush=True)


if __name__ == "__main__":
    main()
