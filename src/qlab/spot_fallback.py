"""How many bars a spot leg was valued at a CARRIED close, not at a print of the bar
(`data.spot_prices: own`, docs/SPOT_PRICES.md).

The fallback rule, in one place. Under ``own`` the panel's spot column holds the spot market's
own closes and is NaN on a bar where the market printed nothing (a thin token's hours without a
trade, a print flagged as corrupt). On such a bar the leg is valued at the LAST close the spot
printed before it (`qlab.strategies.live._marks.spot_marks`; the harness holds a position
across the bar and re-marks it at the next print, `qlab.harness.gaps`). Never the perp's price
under the spot's name, and never a later print. A bar before the spot's first print has no
price at all and nothing is held there: it is not a fallback, it is a bar without the leg.

This module only COUNTS those bars, so that a run states how much of its spot leg rests on a
carried close. A perp-marked panel has none: its spot column is the perp's price by design
(`data.spot_prices: perp`), so the counts are empty and a run of such a spec gains no metric.
"""

from __future__ import annotations

import pandas as pd

from qlab.data.sources.base import SPOT_COLUMN_SUFFIX
from qlab.harness.panel import MarketPanel

METRIC_KEYS = ("spot_prints", "spot_carried", "spot_carried_share")


def spot_fallback_counts(panel: MarketPanel,
                         weights: pd.DataFrame | None = None) -> dict[str, tuple[int, int]]:
    """Per spot column: ``(bars with the spot's own close, bars valued at a carried close)``.

    A carried bar is one where the column is NaN, the coin's perp is priced (the coin is alive
    there) and the spot has printed at least once before. ``weights`` restricts the count to the
    spot columns the book holds (a column that is in the panel but never traded says nothing
    about the run); None counts every spot column. Empty for a panel that is not ``own``."""
    if panel.meta.get("spot_prices", "perp") != "own":
        return {}
    counts: dict[str, tuple[int, int]] = {}
    for column in panel.prices.columns:
        name = str(column)
        if not name.endswith(SPOT_COLUMN_SUFFIX):
            continue
        if weights is not None and (name not in weights.columns
                                    or not (weights[name].abs() > 0).any()):
            continue
        spot = panel.prices[column]
        printed = spot.notna()
        if not printed.any():
            counts[name] = (0, 0)
            continue
        perp_name = name[: -len(SPOT_COLUMN_SUFFIX)]
        alive = (panel.prices[perp_name].notna() if perp_name in panel.prices.columns
                 else pd.Series(True, index=spot.index))
        after_first = printed.cummax()
        counts[name] = (int(printed.sum()), int((~printed & alive & after_first).sum()))
    return counts


def spot_fallback_metrics(panel: MarketPanel,
                          weights: pd.DataFrame | None = None) -> dict[str, float]:
    """The counts of `spot_fallback_counts` summed over the spot columns, as metrics a trial
    records. Empty when the panel is not ``own`` or holds no spot column."""
    counts = spot_fallback_counts(panel, weights)
    if not counts:
        return {}
    prints = sum(p for p, _ in counts.values())
    carried = sum(c for _, c in counts.values())
    total = prints + carried
    return {"spot_prints": float(prints), "spot_carried": float(carried),
            "spot_carried_share": carried / total if total else 0.0}


def fallback_sentence(metrics: dict[str, float]) -> str | None:
    """The sentence a report says about the counts, or None for a run that has none."""
    if "spot_carried" not in metrics:
        return None
    carried, prints = int(metrics["spot_carried"]), int(metrics["spot_prints"])
    return (f"цена спота: {prints} баров по собственной цене спота, {carried} "
            f"({metrics['spot_carried_share']:.1%}) по последней известной цене, "
            "подставленной вместо пропуска; цена контракта под именем спота не подставляется")


__all__ = ["METRIC_KEYS", "fallback_sentence", "spot_fallback_counts", "spot_fallback_metrics"]
