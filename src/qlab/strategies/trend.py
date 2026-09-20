"""TSMOM ensemble on crypto perps: directional long/short, per instrument.

Driver: persistent directional price trends (time-series momentum) on each
instrument's OWN history -- paid when an instrument that has been
rising/falling for weeks keeps doing so long enough to outrun transaction
and funding cost, regardless of what any other instrument in the universe is
doing. `funding-rate-arbitrage/research/trend_following/PLAN.md` line 25:
"Directional, не dollar-neutral: позиция per-asset = +1/-1/flat по
СОБСТВЕННОМУ тренду, не по cross-sectional рангу."

What makes this mechanism stop paying: the market chops instead of trending
-- same file, line 28: "Зарабатывает на затяжных движениях, проигрывает в
чопе (whipsaw) -> cost-sensitive". This materialised on fresh, never-seen
data: `research/trend_following/FINDINGS.md`'s 2026-09-14 update, line 107,
"2026-06 -- 2026-09 (свежее) | -1.16 | -32%/год".

Interface shape: cross-sectional long/short with periodic rebalance
(`qlab.harness.strategy.Strategy` docstring, shape 1) -- "cross-sectional"
here means "one independent score per instrument, each sized on its own",
not a market-neutral spread: every instrument's sign comes only from its own
price history, and the book is NOT normalised to zero net exposure.

Scope note -- ports the COMMITTED configuration validated in
`FINDINGS.md` ("Committed = TSMOM-ENSEMBLE (lookbacks 30/60/90/120).
Константы: VOL_TARGET=0.02/день, LEVERAGE_CAP=3.0") and mirrored in the live
paper engine `frab/strategy/trend/{params,signals}.py`: an equal-weight
ensemble of TSMOM signs, inverse-vol sizing to a per-asset daily
volatility target, the book capped by gross leverage, then scaled by
`risk_scale`. Left OUT: the live engine's separate whole-BOOK volatility
targeting (`TrendParams.book_vol_target_ann` / `book_vol_scale`,
`frab/strategy/trend/signals.py:44-62`) -- that layer scales every weight by
a factor computed from the BOOK'S OWN trailing equity curve, which a
`target_weights(panel, params) -> weights` function does not have (weights,
not realised P&L, are its only output); folding it in here would mean
simulating a full return path inside a weights function. It IS documented
and used live -- `TrendParams`'s own docstring calls it "the only knob of
seven that helped on BOTH the fitted and the never-seen window"
(`frab/strategy/trend/params.py:17-19`) -- so if it is ever added it belongs
downstream of `qlab.harness.run`'s realised returns, not here.

There is deliberately no `coins`/universe parameter: the instrument universe
is whatever columns `panel.prices` has. Restricting or expanding it is a
data/spec concern (`docs/PLAN.md`, M2 "интерфейс стратегии"), not a rule
this strategy enforces on itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies._periods import periods_for


class TrendTSMOMEnsemble:
    """Directional TSMOM ensemble, vol-targeted, leverage-capped.

    Required `params` keys (no defaults -- see module docstring for where
    each is documented):

        lookbacks_days: Sequence[int]
            `TrendParams.lookbacks` -- "Signal: sign of the trailing return
            over each lookback (days), averaged." default (30, 60, 90, 120)
            (`frab/strategy/trend/params.py:41-42`); `FINDINGS.md`:
            "Committed = TSMOM-ENSEMBLE (lookbacks 30/60/90/120)".
        vol_window_days: int
            `TrendParams.vol_window`, default 30
            (`frab/strategy/trend/params.py:43`).
        vol_target_daily: float
            `TrendParams.vol_target_daily`, default 0.02
            (`frab/strategy/trend/params.py:46`); `FINDINGS.md`:
            "VOL_TARGET=0.02/день".
        leverage_cap: float
            `TrendParams.leverage_cap`, default 3.0
            (`frab/strategy/trend/params.py:47`); `FINDINGS.md`:
            "LEVERAGE_CAP=3.0".
        risk_scale: float
            `TrendParams.risk_scale`, default 0.2
            (`frab/strategy/trend/params.py:48`), documented in the same
            file's module docstring (lines 8-11) as a deliberate,
            documented paper-scale-down to the ~30% annual vol every number
            in `FINDINGS.md` is quoted at -- "risk_scale 1.0 reproduces the
            research book exactly".
        min_history_days: int
            `TrendParams.min_history_days`, default 151
            (`frab/strategy/trend/params.py:59`): "A coin needs this many
            daily closes before it can be traded (longest lookback + vol
            window)."
    """

    name = "trend"

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        lookbacks_days: Sequence[float] = params["lookbacks_days"]
        vol_window_days: float = params["vol_window_days"]
        vol_target_daily: float = params["vol_target_daily"]
        leverage_cap: float = params["leverage_cap"]
        risk_scale: float = params["risk_scale"]
        min_history_days: float = params["min_history_days"]

        prices = panel.prices
        index = prices.index
        lookback_periods = [periods_for(index, pd.Timedelta(days=d)) for d in lookbacks_days]
        vol_window_periods = periods_for(index, pd.Timedelta(days=vol_window_days))
        min_history_periods = periods_for(index, pd.Timedelta(days=min_history_days))

        # `tsmom_sign` / `ensemble_signal`, `frab/strategy/trend/signals.py:14-27`:
        # sign of the trailing return over each lookback, flat (0) when the
        # history is too short; the ensemble is the equal-weight mean of the
        # per-lookback signs.
        signs = [np.sign(prices.pct_change(periods=lb)).fillna(0.0) for lb in lookback_periods]
        ensemble = sum(signs) / len(signs)

        # `realized_vol`, `signals.py:30-41`: population std (ddof=0) of the
        # trailing `vol_window` daily returns, undefined without a full
        # window.
        returns = prices.pct_change()
        vol = returns.rolling(vol_window_periods, min_periods=vol_window_periods).std(ddof=0)

        raw = ensemble * vol_target_daily / vol
        raw = raw.where(vol > 0, 0.0)  # undefined/zero vol -> no sizeable position
        raw = raw.fillna(0.0)

        # Blanket history gate -- `target_weights`, `signals.py:76-79`:
        # "if len(closes) < params.min_history_days: raw[coin] = 0.0".
        history_count = prices.notna().expanding().sum()
        raw = raw.where(history_count >= min_history_periods, 0.0)

        # Not tradeable now -> no position, and excluded from the
        # leverage-cap computation below (a delisted instrument's phantom
        # weight must not steal cap headroom from live ones).
        raw = raw.where(panel.tradeable, 0.0)

        # `target_weights`, `signals.py:83-85`: scale the WHOLE book down
        # (never up) so gross exposure never exceeds leverage_cap, then
        # apply risk_scale.
        gross = raw.abs().sum(axis=1)
        cap = pd.Series(1.0, index=index)
        over = gross > leverage_cap
        cap[over] = leverage_cap / gross[over]

        weights = raw.mul(cap, axis=0) * risk_scale
        return weights
