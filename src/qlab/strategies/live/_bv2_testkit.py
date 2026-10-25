"""Synthetic markets and parameters for the Bv2 adapter's tests (`test_bv2_*.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel
from qlab.strategies.live.bv2 import B2Params

HAS_POOL = "shared_margin" in B2Params.__dataclass_fields__

HOURS = 24 * 100
FOUR = ("A", "B", "C", "D")
#: The first 7-day boundary at which a coin has the startup history (788 bars) behind it.
FIRST_REBALANCE = 840


def market(coins=("A", "B", "C"), hours=HOURS, seed=5, dollar_volume=None, spot_volume=None,
           perp_from=None, spot_from=None, listed_from=None, paths=None,
           funding=0.00001) -> MarketPanel:
    """A synthetic hourly panel: per coin a perp `X` and a spot `X-SPOT`; 40 days up, then down.

    `dollar_volume[coin]` is a number or a per-bar array (hourly dollar volume of the perp, and of
    the spot unless `spot_volume` says otherwise); `perp_from` / `spot_from` give the first
    tradeable bar of a leg; `listed_from` the first bar a coin has any price (before it both legs
    are unpriced); `paths[coin]` replaces a coin's price path."""
    index = pd.date_range("2026-01-01", periods=hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.0012, hours)
    drift = np.r_[np.full(24 * 40, 0.0006), np.full(hours - 24 * 40, -0.0009)]
    cols, volume = {}, {}
    dv = {c: 1000.0 * (len(coins) - k) for k, c in enumerate(coins)}
    dv.update(dollar_volume or {})
    sv = {**dv, **(spot_volume or {})}
    for c in coins:
        path = (paths or {}).get(c)
        if path is None:
            path = 100.0 * np.cumprod(1.0 + drift + common + rng.normal(0, 0.0018, hours))
        cols[c] = path
        cols[f"{c}-SPOT"] = path * 1.0005
        volume[c] = np.broadcast_to(np.asarray(dv[c], dtype=float), (hours,)) / path
        volume[f"{c}-SPOT"] = np.broadcast_to(np.asarray(sv[c], dtype=float), (hours,)) / path
    prices = pd.DataFrame(cols, index=index)
    volume = pd.DataFrame(volume, index=index)
    tradeable = pd.DataFrame(True, index=index, columns=prices.columns)
    for c, start in (listed_from or {}).items():
        for col in (c, f"{c}-SPOT"):
            prices.iloc[:start, prices.columns.get_loc(col)] = np.nan
            volume.iloc[:start, volume.columns.get_loc(col)] = np.nan
            tradeable.iloc[:start, tradeable.columns.get_loc(col)] = False
    for c, start in (perp_from or {}).items():
        tradeable.iloc[:start, tradeable.columns.get_loc(c)] = False
    for c, start in (spot_from or {}).items():
        tradeable.iloc[:start, tradeable.columns.get_loc(f"{c}-SPOT")] = False
    return MarketPanel(
        snapshot_id="m",
        prices=prices,
        funding=pd.DataFrame(funding, index=index, columns=prices.columns),
        tradeable=tradeable,
        meta={"universe_complete": True},
        volume=volume,
        high=prices * 1.001,
    )


def params(coins=("A", "B", "C"), capital=300.0, **kw) -> dict:
    """The production b2 settings (frab.db, 2026-10-01) for synthetic coins."""
    base = {
        "coins": list(coins), "capital_usd": capital, "spot_share": 0.5, "hedge_threshold": 0.0,
        "sticky_exit_hours": 12, "ratchet_threshold": 0.5, "carry_enabled": False,
        "slippage": 0.0005, "min_order_usd": 10.0, "margin_enabled": True,
        "short_leverage": {c: 2.0 for c in coins}, "default_leverage": 1.0,
        "maint_margin_rate": {c: 0.05 for c in coins}, "default_maint_margin_rate": 0.05,
        "margin_buffer": 0.1, "hedge_margin_headroom": 1.0, "rebalance_at_im_share": 0.5,
        "mode": "paper",
    }
    base.update(kw)
    return base


UNIVERSE = {"rule": "top_liquid", "n": 2, "rebalance": "7d", "by": "spot_dollar_volume",
            "window_days": 3, "candidates": list(FOUR)}


def universe_params(coins=FOUR, **kw) -> dict:
    universe = {**UNIVERSE, "candidates": list(coins), **kw.pop("universe", {})}
    out = params(coins=coins, **kw)
    out.pop("coins")
    return {**out, "universe": universe}


def spike_from(bar: int, low=400.0, high=5000.0) -> np.ndarray:
    """A dollar-volume path that jumps from `low` to `high` at `bar`."""
    return np.r_[np.full(bar, low), np.full(HOURS - bar, high)]
