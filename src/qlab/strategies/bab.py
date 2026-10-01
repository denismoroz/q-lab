"""Betting against beta (Frazzini & Pedersen) -- long low-beta, short high-beta,
each side scaled to a beta of one, rebalanced every calendar month.

Driver: investors who cannot or may not borrow buy high-beta assets instead
of levering low-beta ones, bid them up, and so high beta comes with low
alpha (cards/xyz-equity-bab.yaml, `driver`). The factor collects that gap
without a view on market direction: at formation it is beta-neutral.

## Where every number comes from

Two versions of the same paper by the same authors are used, and which part
comes from which is the main fidelity fact about this module.

**Portfolio construction -- the published version** (Journal of Financial
Economics 111(1), 2014; text of the authors' PDF, section "Constructing
Betting-Against-Beta Factors", equations 16-17):

    "all securities in an asset class are ranked in ascending order on the
    basis of their estimated beta. The ranked securities are assigned to one
    of two portfolios: low-beta and high-beta. The low (high) beta portfolio
    is comprised of all stocks with a beta below (above) its asset-class
    median"; "securities are weighted by the ranked betas"; "The portfolios
    are rebalanced every calendar month"; "both portfolios are rescaled to
    have a beta of one at portfolio formation."

    z = rank(beta), z_bar = mean(z), k = 2 / sum|z - z_bar|,
    w_H = k (z - z_bar)+,  w_L = k (z - z_bar)-,
    r_BAB = (r_L - r_f) / beta_L - (r_H - r_f) / beta_H,
    beta_L = beta' w_L,  beta_H = beta' w_H.

**Beta estimation -- the working-paper version** (NBER Working Paper 16601,
2010, section "Estimating Betas"):

    "If daily data is available we use 1-year rolling windows and require at
    least 200 observations." "Following Dimson (1979) ... we estimate betas
    as the sum of the slopes in a regression of the asset's excess return of
    the current and prior market excess returns" with "lags up to K = 5
    trading days", then shrink toward the cross-sectional mean with "w = 0.5
    and beta_XS = 1".

Why not the published estimator: it needs "at least 3 years (750 trading
days) of non-missing return data for correlations". The longest history in
the `xyz` deployment is under 360 daily bars, so the published estimator
cannot produce a single beta there. The working-paper estimator is the same
authors' own method for the same factor and fits the data that exists; it is
taken whole, not mixed with the published windows.

Every one of these numbers is a REQUIRED param (no defaults), so a spec
states them and a reader can check them against the quotes above.

## What this module decides that the paper does not

- **Risk-free rate is zero.** A perpetual futures position is unfunded: its
  price return is already an excess return, and its financing cost is the
  funding payment, which `qlab.harness` accrues separately and always.
- **A "year" is 365 daily bars.** `xyz` perps trade every day, weekends
  included (`xyz:XYZ100`: 354 daily candles from 2025-10-13 to 2026-10-01,
  one per calendar day, checked 2026-10-01), so the paper's "1-year rolling
  window" of trading days becomes 365 calendar days here.
- **Formation is the last bar of each calendar month**, with data up to and
  including that bar, held until the next formation. The paper ranks "at the
  end of the previous month"; in q-lab's alignment rule a weight at row t
  earns the return over (t, t+1], so this is the same moment.
- **An instrument that stops being tradeable mid-month drops to zero** and
  the remaining weights are not re-scaled until the next formation.
- **The market proxy itself is excluded from the cross-section**: it is the
  denominator of every beta, not a security being ranked.
- **Fewer than two eligible instruments at a formation means no position**
  until the next formation: there is no median to split.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel

_REQUIRED = (
    "market",
    "window_bars",
    "min_observations",
    "dimson_lags",
    "shrink_weight",
    "shrink_target",
)


def dimson_beta(asset: np.ndarray, market: np.ndarray, lags: int) -> float | None:
    """Sum of slopes of `asset_t` on `market_t, market_{t-1}, ..., market_{t-lags}`
    (with an intercept). `asset[i]` and `market[i]` are returns of the same
    bar; NaN rows are dropped pairwise. Returns None if nothing is estimable.
    """
    n = len(asset)
    if n <= lags:
        return None
    columns = [market[lags - j : n - j] for j in range(lags + 1)]
    y = asset[lags:]
    x = np.column_stack(columns)
    ok = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    if ok.sum() <= lags + 1:
        return None
    design = np.column_stack([np.ones(ok.sum()), x[ok]])
    coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    return float(coef[1:].sum())


def rank_weights(betas: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Equation 16: weights of the low- and high-beta portfolios, each
    summing to one. Ties get average ranks."""
    z = betas.rank(method="average")
    centred = z - z.mean()
    k = 2.0 / centred.abs().sum()
    high = (k * centred).clip(lower=0.0)
    low = (k * -centred).clip(lower=0.0)
    return low, high


def formation_rows(index: pd.DatetimeIndex) -> list[int]:
    """Row positions of the last bar of each calendar month in `index`."""
    months = index.tz_convert(None).to_period("M") if index.tz is not None else index.to_period("M")
    last = pd.Series(np.arange(len(index)), index=index).groupby(months).max()
    return [int(i) for i in last.to_numpy()]


class BettingAgainstBeta:
    """See module docstring. `params` must carry every key in `_REQUIRED`."""

    name = "betting-against-beta"
    # `window_bars`/`min_observations` are counted in bars and the paper's
    # values are for daily data; on other bars they would mean other windows.
    valid_intervals = ("1d",)

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        missing = [key for key in _REQUIRED if key not in params]
        if missing:
            raise ValueError(f"BettingAgainstBeta needs params {missing}; none has a default")
        market = str(params["market"])
        window = int(params["window_bars"])
        min_obs = int(params["min_observations"])
        lags = int(params["dimson_lags"])
        shrink_w = float(params["shrink_weight"])
        shrink_to = float(params["shrink_target"])

        prices = panel.prices
        if market not in prices.columns:
            raise ValueError(f"market proxy {market!r} is not a column of this panel")

        returns = prices.pct_change(fill_method=None)
        candidates = [c for c in prices.columns if c != market]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        held: pd.Series | None = None

        formations = set(formation_rows(prices.index))
        for t in range(len(prices.index)):
            if t in formations:
                held = self._form(returns, panel.tradeable, candidates, market, t,
                                  window, min_obs, lags, shrink_w, shrink_to)
            if held is not None:
                row = held.where(panel.tradeable.iloc[t][held.index], 0.0)
                weights.iloc[t, weights.columns.get_indexer(row.index)] = row.to_numpy()
        return weights

    @staticmethod
    def _form(
        returns: pd.DataFrame,
        tradeable: pd.DataFrame,
        candidates: list[str],
        market: str,
        t: int,
        window: int,
        min_obs: int,
        lags: int,
        shrink_w: float,
        shrink_to: float,
    ) -> pd.Series | None:
        """Weights decided at row `t` from returns of rows `t-window+1 .. t`."""
        first = max(0, t - window + 1)
        # `lags` extra rows before the window feed the lagged market terms of
        # the window's first observations; still entirely at or before `t`.
        lo = max(0, first - lags)
        market_r = returns[market].iloc[lo : t + 1].to_numpy(dtype=float)
        trim = first - lo

        betas: dict[str, float] = {}
        for name in candidates:
            if not bool(tradeable[name].iloc[t]):
                continue
            asset_r = returns[name].iloc[lo : t + 1].to_numpy(dtype=float)
            in_window = asset_r[trim:]
            mkt_in_window = market_r[trim:]
            observed = int((~np.isnan(in_window) & ~np.isnan(mkt_in_window)).sum())
            if observed < min_obs:
                continue
            beta_ts = dimson_beta(asset_r, market_r, lags)
            if beta_ts is None:
                continue
            betas[name] = shrink_w * beta_ts + (1.0 - shrink_w) * shrink_to

        if len(betas) < 2:
            return None
        beta = pd.Series(betas)
        low, high = rank_weights(beta)
        beta_low = float((beta * low).sum())
        beta_high = float((beta * high).sum())
        if beta_low <= 0 or beta_high <= 0:
            # Equation 17 divides by both; a non-positive portfolio beta has
            # no "rescale to one". Hold nothing rather than invert a sign.
            return None
        return low / beta_low - high / beta_high


__all__ = ["BettingAgainstBeta", "dimson_beta", "formation_rows", "rank_weights"]
