"""Kinds of neural networks for the regime detector (docs/TASKS.md T39; owner,
2026-10-04: «а эксперимент с разными видами нейросетей?»).

Trained on hourly BTC (scale-free features, hourly labels -- the lower-
timeframe setup of detector_timeframes.py, where a network has tens of
thousands of bars to learn from) and applied to daily data, UP and DOWN
separately, walking forward month by month on data known before each month.
Training rows are every 4th hour: adjacent hours' 30-bar labels overlap
almost entirely, so the others add little but time, and every variant
(the logistic baseline included) sees the same rows.

Variants: the logistic baseline; feed-forward networks of 1, 2 and 3 hidden
layers on the five summary features; and a network on the SEQUENCE itself --
the last 90 one-bar returns, each in units of the series' volatility.
scikit-learn's MLP with Adam, ReLU and early stopping on a tenth of the
training rows; nothing tuned. Development only (2020-04 .. 2023-12); the
chosen variant is scored once on the holdout (`--holdout NAME`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _record import log  # noqa: E402
from detector_binary import binary_scores, combine  # noqa: E402
from detector_timeframes import (  # noqa: E402
    DEV,
    HOLDOUT,
    VOL_BARS,
    WINDOW,
    bar_labels,
    hourly_btc,
    scale_free,
)

from qlab import regime_detect as rd  # noqa: E402
from qlab.regimes import load, market_closes  # noqa: E402

SEQ = 90
STRIDE = 4


def sequence(closes: pd.Series) -> pd.DataFrame:
    """The last `SEQ` one-bar log returns, each divided by the trailing
    `VOL_BARS`-bar volatility known at the bar."""
    r = np.log(closes).diff()
    sigma = r.rolling(VOL_BARS).std()
    z = r / sigma
    return pd.DataFrame({f"r_{k}": z.shift(k) for k in range(SEQ)}, index=closes.index)


def _variants():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def net(layers):
        return lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=layers, max_iter=500, early_stopping=True, random_state=0))

    return {
        "logistic": ("summary", lambda: make_pipeline(StandardScaler(),
                                                      LogisticRegression(max_iter=2000))),
        "net 16": ("summary", net((16,))),
        "net 64-32": ("summary", net((64, 32))),
        "net 128-64-32": ("summary", net((128, 64, 32))),
        "net on sequence 64-32": ("sequence", net((64, 32))),
    }


def walk(make, train_x: pd.DataFrame, train_closes: pd.Series, daily_x: pd.DataFrame,
         target: str, window) -> pd.Series:
    train_x, daily_x = train_x.dropna(), daily_x.dropna()
    bar = pd.Timedelta(hours=1)
    out = []
    for month in pd.date_range(window[0], window[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        labels = bar_labels(train_closes, month, bar)
        gap = max(pd.Timedelta(days=rd.EMBARGO_DAYS), WINDOW * bar)
        labels = labels[labels.index <= month - gap]
        rows = train_x.index.intersection(labels.index)[::STRIDE]
        y = (labels.loc[rows] == target).astype(int)
        part = daily_x[(daily_x.index >= month) & (daily_x.index < min(nxt, window[1]))]
        if part.empty or y.nunique() < 2:
            continue
        fit = make().fit(train_x.loc[rows].to_numpy(), y.to_numpy())
        out.append(pd.Series(fit.predict_proba(part.to_numpy())[:, 1], index=part.index))
    return pd.concat(out)


def main() -> None:
    daily, hourly = market_closes(), hourly_btc()
    labels = load().labels
    inputs = {"summary": (scale_free(hourly), scale_free(daily)),
              "sequence": (sequence(hourly), sequence(daily))}
    variants = _variants()

    def run(name: str, window) -> None:
        kind, make = variants[name]
        train_x, daily_x = inputs[kind]
        up = walk(make, train_x, hourly, daily_x, "bull", window)
        down = walk(make, train_x, hourly, daily_x, "bear", window)
        su, sd = binary_scores(up, labels, "bull"), binary_scores(down, labels, "bear")
        acc = rd.accuracy(combine(up, down), labels)["accuracy"]
        log("regime-level1", f"1h-trained {name}", window,
            {"up_auc": su["auc"], "down_auc": sd["auc"], "acc": acc},
            "scripts/research/detector_nets.py")
        print(f"{name:<22} | UP auc {su['auc']:.3f} prec {su['precision']:.2f} rec "
              f"{su['recall']:.2f} | DOWN auc {sd['auc']:.3f} prec {sd['precision']:.2f} rec "
              f"{sd['recall']:.2f} | both acc {acc:.3f}", flush=True)

    if "--holdout" in sys.argv:
        run(sys.argv[sys.argv.index("--holdout") + 1], HOLDOUT)
        return
    for name in variants:
        run(name, DEV)


if __name__ == "__main__":
    main()
