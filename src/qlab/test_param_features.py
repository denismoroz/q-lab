"""The sweep's indicator settings read only minutes that ended by the decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab import param_features as pf


def _minutes(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="min", tz="UTC").as_unit("ns")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    v = rng.uniform(1, 10, n)
    return pd.DataFrame({"high": c * 1.001, "low": c * 0.999, "close": c, "volume": v,
                         "taker_buy_base": v * rng.uniform(0.3, 0.7, n)}, index=idx)


def test_the_settings_read_only_minutes_before_the_decision() -> None:
    m = _minutes(40 * 1440)
    tau = m.index.asi8[59::60] + pf.MINUTE  # decisions at the end of each hour
    pos = np.arange(59, len(m), 60)
    x, names, fams = pf.sweep_features(m, pos, tau)
    assert x.shape == (len(tau), len(names)) == (len(tau), 82)
    assert len(set(names)) == 82 and len(set(fams)) == 9
    cut = 30 * 1440 + 7
    m2 = m.copy()
    for col in ("close", "high", "low", "volume", "taker_buy_base"):
        m2.iloc[cut:, m2.columns.get_loc(col)] *= 1.4
    x2, _, _ = pf.sweep_features(m2, pos, tau)
    k = np.searchsorted(pos, cut)  # decisions whose last minute is before the change
    assert np.allclose(x[:k], x2[:k], equal_nan=True)
