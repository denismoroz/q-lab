"""Market regimes -- how a strategy behaved in rising, falling and flat
markets (docs/TASKS.md T25, docs/REGIMES.md).

Owner, 2026-10-02: «делать нарезку по разным режимам — есть бычий режим, есть
медвежий, есть боковик»; decisions the same day: «BTC, терцили, окно 30 дней,
начинай с описания».

**Definition, declared before any strategy is looked at:**

- the market is BTC (Binance BTCUSDT perpetual, daily closes, from its first
  day, 2019-09-08);
- a day's state is BTC's return over the 30 days ending at that day's close;
- the days are split into terciles of that return over the WHOLE history:
  the top third is `bull`, the bottom third `bear`, the middle `flat`.

No threshold is chosen by hand; the data set them, and the manifest records
them so a run can be reproduced.

**This is a description, not a signal.** The thresholds use the whole history,
so a label is known only afterwards -- fine for explaining where a strategy
earns and loses, useless for switching strategies by regime (that needs a
detector built on past data only and is a separate claim to prove). Each
regime holds about a third of the days, so per-regime numbers are noisier
than the whole window's; they are shown, not judged (no rule reads them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 30  # owner, 2026-10-02: «окно 30 дней»
MARKET = "BTC"  # owner, 2026-10-02: «BTC»
REGIMES = ("bull", "flat", "bear")
DEFAULT_DIR = Path("data/regimes")
_FILE = "btc-30d"


@dataclass(frozen=True)
class RegimeSeries:
    """Per-day label (indexed by the day's CLOSE time, UTC) plus the market's
    own daily return, and the thresholds that produced the labels."""

    labels: pd.Series
    market_return: pd.Series
    bear_below: float
    bull_above: float
    source: str


def label_days(closes: pd.Series, window: int = WINDOW_DAYS) -> tuple[pd.Series, float, float]:
    """Tercile labels of the trailing `window`-day return; the first `window`
    days have no label."""
    trailing = closes / closes.shift(window) - 1.0
    known = trailing.dropna()
    bear_below, bull_above = known.quantile([1 / 3, 2 / 3]).to_list()
    labels = pd.Series(np.where(trailing >= bull_above, "bull",
                                np.where(trailing < bear_below, "bear", "flat")),
                       index=closes.index, dtype=object)
    return labels.where(trailing.notna()), float(bear_below), float(bull_above)


def build(store: Path = DEFAULT_DIR) -> RegimeSeries:
    """Fetch BTC's whole daily history from Binance, label it, and store it.
    Only closed days are kept (a candle is closed once its open + 1 day has
    passed)."""
    import httpx

    from qlab.data.sources.binance import fetch_candles

    now = pd.Timestamp(datetime.now(UTC))
    with httpx.Client() as client:
        candles = fetch_candles(client, MARKET, "1d", pd.Timestamp("2019-01-01", tz="UTC"), now)
    closes = candles["price"]
    closes = closes[closes.index + pd.Timedelta(days=1) <= now]
    closes.index = closes.index + pd.Timedelta(days=1)  # label each day by its close time
    labels, bear_below, bull_above = label_days(closes)
    source = (f"Binance {MARKET}USDT perpetual 1d closes {closes.index[0]:%Y-%m-%d}.."
              f"{closes.index[-1]:%Y-%m-%d}, {WINDOW_DAYS}-day return terciles")
    store.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"close": closes, "label": labels})
    frame.to_parquet(store / f"{_FILE}.parquet")
    (store / f"{_FILE}.json").write_text(json.dumps({
        "source": source, "window_days": WINDOW_DAYS, "bear_below": bear_below,
        "bull_above": bull_above, "built_at": now.isoformat(),
    }, indent=1), encoding="utf-8")
    return load(store)


def load(store: Path = DEFAULT_DIR) -> RegimeSeries | None:
    path = store / f"{_FILE}.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    meta = json.loads((store / f"{_FILE}.json").read_text(encoding="utf-8"))
    return RegimeSeries(
        labels=frame["label"],
        market_return=frame["close"].pct_change(),
        bear_below=meta["bear_below"],
        bull_above=meta["bull_above"],
        source=meta["source"],
    )


def breakdown(net_return: pd.Series, regimes: RegimeSeries, periods_per_year: float
              ) -> dict[str, float]:
    """Per-regime metrics of a strategy's net returns (indexed by decision
    time t; each return is realised over (t, t+1]). A return takes the label
    of the last day closed by the end of its period. Keys:
    regime_<r>_share, _days (distinct calendar days), _return (compounded
    over the regime's periods),
    _ann_return, _sharpe, _max_dd, _btc_return (BTC over the same days)."""
    if len(net_return) < 2:
        return {}
    ends = net_return.index[1:].append(pd.DatetimeIndex([net_return.index[-1] + (
        net_return.index[-1] - net_return.index[-2])]))
    labels = regimes.labels.dropna()
    label = pd.Series(labels.reindex(ends, method="ffill").to_numpy(), index=net_return.index)
    out: dict[str, float] = {
        "regime_bear_below": regimes.bear_below, "regime_bull_above": regimes.bull_above,
    }
    days = pd.Series(ends.normalize(), index=net_return.index)
    for r in REGIMES:
        mask = (label == r).to_numpy()
        part = net_return[mask]
        out[f"regime_{r}_share"] = float(mask.mean())
        out[f"regime_{r}_days"] = float(pd.DatetimeIndex(days[mask]).unique().size)
        if len(part) < 2:
            continue
        growth = float((1.0 + part).prod())
        std = part.std(ddof=1)
        equity = (1.0 + part).cumprod()
        out[f"regime_{r}_return"] = growth - 1.0
        out[f"regime_{r}_ann_return"] = (
            growth ** (periods_per_year / len(part)) - 1.0 if growth > 0 else -1.0
        )
        out[f"regime_{r}_sharpe"] = (float(part.mean() / std * np.sqrt(periods_per_year))
                                     if std and not np.isnan(std) else float("nan"))
        out[f"regime_{r}_max_dd"] = float((equity / equity.cummax() - 1.0).min())
        regime_days = pd.DatetimeIndex(days[mask].unique())
        market = regimes.market_return.reindex(regime_days).dropna()
        out[f"regime_{r}_btc_return"] = float((1.0 + market).prod() - 1.0)
    return out


__all__ = ["MARKET", "REGIMES", "WINDOW_DAYS", "RegimeSeries", "breakdown", "build",
           "label_days", "load"]
