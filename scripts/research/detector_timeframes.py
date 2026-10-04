"""Learn the regime on a lower timeframe, use it on the daily one (docs/TASKS.md
T39; owner, 2026-10-04: «можно спуститься на уровень ниже по таймфреймам ...
обучить модель на более мелких таймфреймах — разметку тоже делать исходя из
таймфрейма мелкого. вот и море просто данных. а потом переключить модель на
старший таймфрейм»).

To move a model between timeframes its inputs must not depend on the scale:
an hourly return is a fraction of a daily one. Every feature is therefore a
return over k bars divided by the series' own volatility over that span
(trailing 90-bar volatility of one-bar log returns, times sqrt(k)); the label
on each timeframe is its own -- terciles of the return over the 30 bars
around the bar, thresholds from labels known at the time.

UP and DOWN logistic models (the pair chosen in detector_binary.py) are
trained, walking forward month by month on data known before each month, on
1h, 4h or 1d bars, and always applied to DAILY features and scored against
the daily testing labels. Development only (2020-04 .. 2023-12); the holdout
is scored once for the variant chosen here (`--holdout TF`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qlab.regimes import load, market_closes

sys.path.insert(0, str(Path(__file__).parent))
from _record import log  # noqa: E402
from detector_binary import binary_scores, combine  # noqa: E402

from qlab import regime_detect as rd  # noqa: E402

DEV = (pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"))
BARS = (7, 14, 30, 60, 90)  # the daily set's windows, in bars of each timeframe
VOL_BARS = 90
WINDOW = 30
HALF = WINDOW // 2


def scale_free(closes: pd.Series) -> pd.DataFrame:
    """Return over k bars in units of the series' own volatility over k bars."""
    log = np.log(closes)
    sigma = log.diff().rolling(VOL_BARS).std()
    return pd.DataFrame({f"z_{k}": (log - log.shift(k)) / (sigma * np.sqrt(k)) for k in BARS},
                        index=closes.index)


def bar_labels(closes: pd.Series, as_of: pd.Timestamp, bar: pd.Timedelta) -> pd.Series:
    """Labels of the bars whose 30-bar centered window had closed by `as_of`
    (bars stamped by their close), terciles from those labels only."""
    seen = closes[closes.index <= as_of]
    around = (seen.shift(-HALF) / seen.shift(HALF) - 1.0).dropna()
    if around.empty:
        return pd.Series(dtype=object)
    lo, hi = around.quantile([1 / 3, 2 / 3]).to_list()
    return pd.Series(np.where(around >= hi, "bull", np.where(around < lo, "bear", "flat")),
                     index=around.index)


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def walk(train_closes: pd.Series, bar: pd.Timedelta, daily_feats: pd.DataFrame, target: str,
         window) -> pd.Series:
    train_feats = scale_free(train_closes).dropna()
    daily_feats = daily_feats.dropna()
    out = []
    for month in pd.date_range(window[0], window[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        labels = bar_labels(train_closes, month, bar)
        # The daily detector's gap, at least a whole label window of this timeframe.
        gap = max(pd.Timedelta(days=rd.EMBARGO_DAYS), WINDOW * bar)
        labels = labels[labels.index <= month - gap]
        rows = train_feats.index.intersection(labels.index)
        y = (labels.loc[rows] == target).astype(int)
        part = daily_feats[(daily_feats.index >= month) & (daily_feats.index < min(nxt, window[1]))]
        if part.empty or y.nunique() < 2:
            continue
        fit = _model().fit(train_feats.loc[rows].to_numpy(), y.to_numpy())
        out.append(pd.Series(fit.predict_proba(part.to_numpy())[:, 1], index=part.index))
    return pd.concat(out)


def hourly_btc() -> pd.Series:
    from qlab.pipeline.evaluate import resolve_panel
    from qlab.pipeline.spec import load_spec
    from qlab.registry.db import get_sessionmaker

    session = get_sessionmaker()()
    try:
        panel = resolve_panel(session, load_spec(Path("specs/bv2-binance.yaml")))
    finally:
        session.rollback()
        session.close()
    btc = panel.prices["BTC"].dropna()
    return btc.set_axis(btc.index + pd.Timedelta(hours=1))  # stamped by close


def main() -> None:
    daily = market_closes()
    labels = load().labels
    hourly = hourly_btc()
    frames = {
        "1h": (hourly, pd.Timedelta(hours=1)),
        "4h": (hourly.resample("4h", label="right", closed="right").last().dropna(),
               pd.Timedelta(hours=4)),
        "1d": (daily, pd.Timedelta(days=1)),
    }
    daily_feats = scale_free(daily)

    def run(tf: str, window) -> None:
        closes, bar = frames[tf]
        up = walk(closes, bar, daily_feats, "bull", window)
        down = walk(closes, bar, daily_feats, "bear", window)
        su, sd = binary_scores(up, labels, "bull"), binary_scores(down, labels, "bear")
        acc = rd.accuracy(combine(up, down), labels)["accuracy"]
        log("regime-level1", f"pair trained on {tf}", window,
            {"up_auc": su["auc"], "up_precision": su["precision"], "up_recall": su["recall"],
             "down_auc": sd["auc"], "down_precision": sd["precision"],
             "down_recall": sd["recall"], "acc": acc},
            "scripts/research/detector_timeframes.py", chosen=tf == CHOICE)
        print(f"trained on {tf:<3} ({len(closes):>6} bars) | UP auc {su['auc']:.3f} prec "
              f"{su['precision']:.2f} rec {su['recall']:.2f} | DOWN auc {sd['auc']:.3f} prec "
              f"{sd['precision']:.2f} rec {sd['recall']:.2f} | both acc {acc:.3f}", flush=True)

    if "--holdout" in sys.argv:
        tf = sys.argv[sys.argv.index("--holdout") + 1]
        run(tf, HOLDOUT)
        if tf != "1d":
            run("1d", HOLDOUT)
        return
    for tf in ("1d", "4h", "1h"):
        run(tf, DEV)


# Development result (2026-10-04), read before the holdout was predicted:
# trained on 1d UP 0.808 / DOWN 0.790 auc; on 4h 0.834 / 0.831 (DOWN precision
# 0.78 against 0.62); on 1h 0.834 / 0.823. Chosen: 4h.
CHOICE = "4h"

if __name__ == "__main__":
    main()
