"""A shorter regime window for strategies (docs/TASKS.md T39; owner,
2026-10-04: «за 15 дней — цена на крипту может уйти ниже плинтуса или в
космос — 15 дней это очень много для крипты»).

The testing description keeps its 30-day window (owner, 2026-10-02); a
detector a strategy trades on may use another. For each window W (7, 14, 30
days): labels are terciles of BTC's return over the W days around the day
(the yardstick, on the whole history), features are BTC's returns over W/4,
W/2, W, 2W and 3W days (the 30-day detector's 7-90 scaled), the model is the
stored detector's -- three-class logistic, walked forward monthly on labels
known W days before the month. Reported: accuracy on its own labels, the
ceiling (the trailing half of the window with thresholds fitted in
hindsight), and lag in days at stretches of W/2 days or more.

Development only (2020-04 .. 2023-12).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _record import log  # noqa: E402

from qlab import regime_detect as rd  # noqa: E402
from qlab.regimes import market_closes  # noqa: E402

DEV = (pd.Timestamp("2020-04-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
HOLDOUT = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"))
# Owner, 2026-10-04, after the development table: «делай 14 дней на контроле и в trend».
CHOICE = 14


def centered(closes: pd.Series, window: int) -> pd.Series:
    half = window // 2
    return closes.shift(-(window - half)) / closes.shift(half) - 1.0


def terciles(values: pd.Series) -> pd.Series:
    known = values.dropna()
    lo, hi = known.quantile([1 / 3, 2 / 3]).to_list()
    return pd.Series(np.where(known >= hi, "bull", np.where(known < lo, "bear", "flat")),
                     index=known.index)


def features(btc: pd.Series, window: int) -> pd.DataFrame:
    spans = sorted({max(1, round(window * k / 30)) for k in (7, 14, 30, 60, 90)})
    return pd.DataFrame({f"ret_{s}": btc / btc.shift(s) - 1 for s in spans}, index=btc.index)


def walk(btc: pd.Series, window: int, span=DEV) -> pd.Series:
    feats = features(btc, window).dropna()
    out = []
    for month in pd.date_range(span[0], span[1], freq="MS", tz="UTC"):
        nxt = month + pd.offsets.MonthBegin(1)
        seen = btc[btc.index <= month]
        labels = terciles(centered(seen, window))
        labels = labels[labels.index <= month - pd.Timedelta(days=window)]
        rows = feats.index.intersection(labels.index)
        part = feats[(feats.index >= month) & (feats.index < min(nxt, span[1]))]
        if part.empty or labels.loc[rows].nunique() < 3:
            continue
        fit = rd._model().fit(feats.loc[rows], labels.loc[rows])
        out.append(pd.Series(fit.predict(part), index=part.index))
    return pd.concat(out)


def ceiling(btc: pd.Series, labels: pd.Series, window: int, days: pd.Index) -> float:
    x = (btc / btc.shift(window // 2) - 1).reindex(days)
    lab = labels.reindex(days)
    best = 0.0
    qs = np.linspace(0.05, 0.95, 37)
    for a in qs:
        for b in qs[qs > a]:
            lo, hi = x.quantile(a), x.quantile(b)
            p = np.where(x >= hi, "bull", np.where(x < lo, "bear", "flat"))
            best = max(best, float((p == lab).mean()))
    return best


def lag(signal: pd.Series, labels: pd.Series, regime: str, window: int) -> tuple[float, float]:
    days = signal.index.intersection(labels.index)
    lab = labels.loc[days]
    run = (lab != lab.shift()).cumsum()
    starts, ends = [], []
    for _, s in lab[lab == regime].groupby(run[lab == regime]):
        if len(s) < max(2, window // 2):
            continue
        seen = signal.reindex(s.index) == regime
        starts.append((seen.idxmax() - s.index[0]).days if seen.any() else len(s))
        after = signal.reindex(pd.date_range(s.index[-1], periods=61, freq="1D",
                                             tz="UTC")[1:]) == regime
        ends.append(int((~after).to_numpy().argmax()) + 1 if (~after).any() else 60)
    return float(np.median(starts)), float(np.median(ends))


def main() -> None:
    btc = market_closes()
    holdout = "--holdout" in sys.argv
    for window in ((CHOICE, 30) if holdout else (7, 14, 30)):
        labels = terciles(centered(btc, window))
        pred = walk(btc, window, HOLDOUT if holdout else DEV)
        days = pred.index.intersection(labels.index)
        acc = float((pred.loc[days] == labels.loc[days]).mean())
        bs, be = lag(pred, labels, "bear", window)
        us, ue = lag(pred, labels, "bull", window)
        lab = labels.loc[days]
        run = (lab != lab.shift()).cumsum()
        stretch = lab.groupby(run).size().median()
        top = ceiling(btc, labels, window, days)
        span = HOLDOUT if holdout else DEV
        log("regime-level1", f"window {window}d, returns scaled, 3-class logistic", span,
            {"acc": acc, "ceiling": top, "bear_notice_days": bs, "bear_hold_days": be,
             "bull_notice_days": us, "bull_hold_days": ue},
            "scripts/research/detector_windows.py", chosen=holdout and window == CHOICE)
        print(f"window {window:>2}d | acc {acc:.3f} ceiling {top:.3f}"
              f" | typical stretch {stretch:.0f}d | bear noticed after {bs:.0f}d, held {be:.0f}d"
              f" past end | bull noticed after {us:.0f}d, held {ue:.0f}d", flush=True)


if __name__ == "__main__":
    main()
