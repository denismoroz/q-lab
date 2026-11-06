"""The declared grid of indicator settings for the parameter sweep
(docs/research/param-sweep-protocol-2026-11-06.md): RSI, Bollinger, ATR, EMA distance / slope /
crossovers, taker imbalance, Donchian, past-return reversal and realised volatility, each at
hourly decisions and read only from minutes that ended by the decision (tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.forecast import _rsi, rsi_daily

MINUTE = 60_000_000_000
RSI_H = (4, 8, 14, 28, 56, 100)
RSI_D = (7, 14, 28)
BOLL_N = (10, 20, 50)
BOLL_W = (1.5, 2.0, 2.5)
ATR_N = (7, 14, 28)
EMA = (4, 12, 24, 72, 168, 336, 720)
IMB_MIN = (15, 60, 240, 1440, 4320, 10080)
DONCH = (4, 24, 168, 720)
REV = ((15, "15m"), (60, "1h"), (240, "4h"), (1440, "24h"))
RV_MIN = ((60, "1h"), (1440, "24h"), (10080, "168h"))


def hourly_bars(m: pd.DataFrame, tau: np.ndarray) -> pd.DataFrame:
    """The hour bars that close at each decision `tau` (open = tau - 1 h), from the minutes."""
    hb = m.groupby(m.index.floor("h")).agg({"high": "max", "low": "min", "close": "last",
                                            "volume": "sum"})
    hb.index = pd.DatetimeIndex(hb.index).as_unit("ns")
    labels = pd.DatetimeIndex(pd.to_datetime(tau - 3_600_000_000_000, utc=True)).as_unit("ns")
    return hb.reindex(labels)


def sweep_features(m: pd.DataFrame, pos: np.ndarray, tau: np.ndarray
                   ) -> tuple[np.ndarray, list[str], list[str]]:
    """(decisions x settings, the setting names, each setting's indicator family)."""
    bars = hourly_bars(m, tau)
    c = bars["close"].ffill()
    hi, lo = bars["high"].ffill(), bars["low"].ffill()
    lc = np.log(c)
    cols, names, fams = [], [], []

    def add(x, name, fam):
        cols.append(np.asarray(x, dtype=np.float64))
        names.append(name)
        fams.append(fam)

    for n in RSI_H:
        add((_rsi(c, n) - 50) / 50, f"rsi_{n}h", "rsi")
    for n in RSI_D:
        add((rsi_daily(c, n) - 50) / 50, f"rsi_{n}d", "rsi")
    for n in BOLL_N:
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        for w in BOLL_W:
            add((c - (ma - w * sd)) / (2 * w * sd) - 0.5, f"boll_b_{n}_{w:g}", "bollinger")
            add(2 * w * sd / ma, f"boll_width_{n}_{w:g}", "bollinger")
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    for n in ATR_N:
        add(tr.ewm(alpha=1 / n, adjust=False).mean() / c, f"atr_{n}", "atr")
    le = {n: np.log(c.ewm(span=n, adjust=False).mean()) for n in EMA}
    for n in EMA:
        add(lc - le[n], f"ema_dist_{n}h", "ema")
        add(le[n] - le[n].shift(min(n, 24)), f"ema_slope_{n}h", "ema")
    for i, a in enumerate(EMA):
        for b in EMA[i + 1:]:
            add(le[a] - le[b], f"ema_cross_{a}_{b}", "ema_cross")
    # minute windows at the decision's last minute (inclusive)
    p = np.clip(pos, 0, None)
    cb = np.r_[0.0, np.cumsum(m["taker_buy_base"].to_numpy(dtype=np.float64))]
    cv = np.r_[0.0, np.cumsum(m["volume"].to_numpy(dtype=np.float64))]
    for k in IMB_MIN:
        lo_ = np.clip(p + 1 - k, 0, None)
        vol = cv[p + 1] - cv[lo_]
        with np.errstate(invalid="ignore", divide="ignore"):
            imb = (cb[p + 1] - cb[lo_]) / vol - 0.5
        add(np.where((pos >= k) & (vol > 0), imb, np.nan), f"imbalance_{k}m", "imbalance")
    for n in DONCH:
        hn, ln = hi.rolling(n).max(), lo.rolling(n).min()
        add((c - ln) / (hn - ln) - 0.5, f"donchian_{n}h", "donchian")
    logm = np.log(m["close"].to_numpy(dtype=np.float64))
    for k, lab in REV:
        add(np.where(pos >= k, -(logm[p] - logm[np.clip(p - k, 0, None)]), np.nan),
            f"reversal_{lab}", "reversal")
    r = np.r_[0.0, np.diff(logm)]
    cr, cr2 = np.r_[0.0, np.cumsum(r)], np.r_[0.0, np.cumsum(r * r)]
    for k, lab in RV_MIN:
        lo_ = np.clip(p + 1 - k, 0, None)
        nn = np.maximum(p + 1 - lo_, 1)
        mean = (cr[p + 1] - cr[lo_]) / nn
        var = (cr2[p + 1] - cr2[lo_]) / nn - mean ** 2
        add(np.where(pos >= k, np.sqrt(np.clip(var, 0, None)), np.nan), f"rv_{lab}", "rv")
    x = np.column_stack(cols)
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan), names, fams


__all__ = ["hourly_bars", "sweep_features"]
