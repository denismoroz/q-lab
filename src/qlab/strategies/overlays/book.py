"""Book-level overlays: one number per bar scales or gates the whole book (docs/OVERLAYS.md).

- `market_gate`: hold the strategy only in the market states listed (BTC's 30-day direction,
  `qlab.strategies.detectors`; or the walk-forward detector's stored predictions). The same
  logic `qlab.strategies.regime_gate.RegimeGate` has -- that wrapper keeps working and shares
  these functions.
- `meta_gate`: hold the strategy only on the days the strategy's own level-2 decisions say it
  will earn (`qlab.meta_label`, stored by `qlab regimes meta`). Reachable through
  `RegimeGate(detector="meta:<name>")` as well.
- `vol_target`: scale the book so that the volatility of its recent results does not exceed what
  it has usually been: the target is the median of the strategy's own past volatility, so the
  overlay introduces no number of its own beyond the look-back.
- `own_results`: scale the book by how well it has done lately -- full size while the recent
  results are above zero, a floor when they are below; "step" switches, "smooth" is
  proportional to how clearly the results are above zero; the scale may come back up at once or
  gradually.

None of them reads anything after the bar it scales for: the market states are the ones known
at the bar (`regime_gate.open_at`), the strategy's results are the ones realised before it
(`Context.known_returns`).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import pandas as pd

from qlab.strategies.overlays.core import BookOverlay, Context, register
from qlab.strategies.regime_gate import check_trade_in, open_at, regime_labels

#: Where the standard "clearly above zero" mark comes from: a mean result more than two
#: standard errors above zero is the usual reading of "not luck" (the t-statistic of 2).
SMOOTH_FULL_AT = 2.0
TRADE_IN_SKIP_ONE = {"skip-bull": ["flat", "bear"], "skip-flat": ["bull", "bear"],
                     "skip-bear": ["bull", "flat"]}
"""The market-state family: stay out of ONE state, as the owner put it, «если мы знаем что
стратегия убыточна в определенный режим рынка - не входить»; which one is chosen from the past by
the causal choice, never by looking at the data it is judged on."""


class MarketGate(BookOverlay):
    name = "market_gate"
    short = "gate"
    params = ("detector", "trade_in")

    def check(self, params: Mapping[str, object]) -> None:
        super().check(params)
        check_trade_in(params["trade_in"], str(params["detector"]))

    def warmup_days(self, params: Mapping[str, object]) -> float:
        from qlab.regimes import WINDOW_DAYS

        return float(WINDOW_DAYS)  # the detector needs a month of BTC before it labels a day

    def scale(self, ctx: Context, params: Mapping[str, object]) -> pd.Series:
        detector = str(params["detector"])
        labels = regime_labels(detector)
        on = open_at(ctx.index, labels, check_trade_in(params["trade_in"], detector))
        return pd.Series(on.astype(float), index=ctx.index)

    def grid(self) -> dict[str, dict[str, object]]:
        return {label: {"detector": "direction_terciles", "trade_in": list(states)}
                for label, states in TRADE_IN_SKIP_ONE.items()}


class MetaGate(BookOverlay):
    name = "meta_gate"
    short = "meta"
    params = ("name",)

    def warmup_days(self, params: Mapping[str, object]) -> float:
        from qlab.meta_label import HORIZON_DAYS

        return float(HORIZON_DAYS)

    def scale(self, ctx: Context, params: Mapping[str, object]) -> pd.Series:
        detector = f"meta:{params['name']}"
        on = open_at(ctx.index, regime_labels(detector), {"trade"})
        return pd.Series(on.astype(float), index=ctx.index)

    def grid(self) -> dict[str, dict[str, object]]:
        """Empty: the decisions are learned for one strategy (`qlab regimes meta`), so which
        sets belong to a base is declared in night/overlays.yaml (`meta_decisions`)."""
        return {}

    def grid_for(self, decision_names: list[str]) -> dict[str, dict[str, object]]:
        return {f"meta-{n}": {"name": n} for n in decision_names}


class VolTarget(BookOverlay):
    name = "vol_target"
    short = "vol"
    params = ("lookback_days", "max_scale")
    needs_returns = True

    def check(self, params: Mapping[str, object]) -> None:
        super().check(params)
        if float(params["lookback_days"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("vol_target: lookback_days must be > 0")
        if float(params["max_scale"]) < 1.0:  # type: ignore[arg-type]
            raise ValueError("vol_target: max_scale < 1 would shrink the book at its calmest")

    def warmup_days(self, params: Mapping[str, object]) -> float:
        # A volatility needs the look-back; its median over the past needs as much again.
        return 2.0 * float(params["lookback_days"])  # type: ignore[arg-type]

    def scale(self, ctx: Context, params: Mapping[str, object]) -> pd.Series:
        window = ctx.bars(float(params["lookback_days"]))  # type: ignore[arg-type]
        known = ctx.known_returns()
        vol = known.rolling(window, min_periods=window).std(ddof=0)
        # What it has usually been: the median of the volatilities seen BEFORE this bar.
        usual = vol.shift(1).expanding(min_periods=window).median()
        factor = (usual / vol).where(vol > 1e-12)
        factor = factor.clip(upper=float(params["max_scale"]))  # type: ignore[arg-type]
        return factor.fillna(1.0)  # no evidence yet (or a flat book): leave the book alone

    def grid(self) -> dict[str, dict[str, object]]:
        return {f"d{d}-max{m:g}": {"lookback_days": d, "max_scale": m}
                for d, m in itertools.product((14, 30, 60, 90), (1.0, 2.0))}

    def coarse_grid(self) -> dict[str, dict[str, object]]:
        return {f"d{d}-max1": {"lookback_days": d, "max_scale": 1.0} for d in (30, 90)}


class OwnResults(BookOverlay):
    name = "own_results"
    short = "own"
    params = ("window_days", "floor", "mode", "up_days")
    needs_returns = True

    def check(self, params: Mapping[str, object]) -> None:
        super().check(params)
        if float(params["window_days"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("own_results: window_days must be > 0")
        if not 0.0 <= float(params["floor"]) <= 1.0:  # type: ignore[arg-type]
            raise ValueError("own_results: floor is a share of the book in [0, 1]")
        if params["mode"] not in ("step", "smooth"):
            raise ValueError("own_results: mode is 'step' or 'smooth'")
        if float(params["up_days"]) < 1:  # type: ignore[arg-type]
            raise ValueError("own_results: up_days is at least one day (1 = back up at once)")

    def warmup_days(self, params: Mapping[str, object]) -> float:
        return float(params["window_days"])  # type: ignore[arg-type]

    def scale(self, ctx: Context, params: Mapping[str, object]) -> pd.Series:
        window = ctx.bars(float(params["window_days"]))  # type: ignore[arg-type]
        floor = float(params["floor"])  # type: ignore[arg-type]
        known = ctx.known_returns()
        mean = known.rolling(window, min_periods=window).mean()
        if params["mode"] == "step":
            target = pd.Series(np.where(mean > 0, 1.0, floor), index=known.index)
        else:
            std = known.rolling(window, min_periods=window).std(ddof=1)
            tstat = (mean / (std / np.sqrt(window))).where(std > 1e-12)
            target = (tstat / SMOOTH_FULL_AT).clip(lower=floor, upper=1.0)
        target = target.where(mean.notna(), 1.0).fillna(1.0)
        return _ramp_up(target, up_bars=ctx.bars(float(params["up_days"])),  # type: ignore[arg-type]
                        floor=floor)

    def grid(self) -> dict[str, dict[str, object]]:
        return {f"w{w}-f{f:g}-{m}-up{u}": {"window_days": w, "floor": f, "mode": m, "up_days": u}
                for w, f, m, u in itertools.product((7, 14, 30, 60, 90), (0.0, 0.25, 0.5),
                                                    ("step", "smooth"), (1, 7))}

    def coarse_grid(self) -> dict[str, dict[str, object]]:
        return {f"w{w}-f{f:g}-{m}-up1": {"window_days": w, "floor": f, "mode": m, "up_days": 1}
                for w, f, m in itertools.product((14, 60), (0.0, 0.5), ("step", "smooth"))}


def _ramp_up(target: pd.Series, *, up_bars: int, floor: float) -> pd.Series:
    """The scale follows `target` down at once and up by at most the whole range above the
    floor over `up_bars` bars (`up_bars` of 1 is at once)."""
    if up_bars <= 1:
        return target
    step = max(1.0 - floor, 1e-12) / up_bars
    out = np.empty(len(target))
    aims = target.to_numpy(dtype=float)
    current = 1.0  # the book starts at full size: nothing was known yet
    for i, aim in enumerate(aims):
        current = aim if aim <= current else min(aim, current + step)
        out[i] = current
    return pd.Series(out, index=target.index)


MARKET_GATE = register(MarketGate())
META_GATE = register(MetaGate())
VOL_TARGET = register(VolTarget())
OWN_RESULTS = register(OwnResults())

__all__ = ["MARKET_GATE", "META_GATE", "OWN_RESULTS", "SMOOTH_FULL_AT", "TRADE_IN_SKIP_ONE",
           "VOL_TARGET", "MarketGate", "MetaGate", "OwnResults", "VolTarget"]
