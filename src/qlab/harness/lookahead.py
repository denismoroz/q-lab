"""Look-ahead guard for any strategy (docs/PLAN.md stage 5, docs/LOOKAHEAD.md).

A strategy's weight at bar t may use data up to and including t, never after
(`qlab.harness.strategy`, alignment rule). Hand-written strategies carry their
own tests for this (trend, Bv2, re-tuning); code written by an agent from a
card will not. So the pipeline checks it itself, on every candidate run:

1. compute the weights on the panel as given;
2. for each cut bar c, replace every value AFTER c -- prices, bar highs,
   volume, funding -- with a different, plausible path, and compute the
   weights again on that tampered panel;
3. the weights up to and including c must be identical. If any differs, the
   strategy read data from after c to decide at or before c.

What is changed after a cut keeps its shape: every price move after the cut
is reversed and rescaled (x0.5..x2), prices stay positive and keep their NaN
pattern (an instrument not listed stays not listed), so a strategy
that legitimately reacts to listings is not disturbed. Listing itself
(`tradeable`) is left alone: changing it would change which weights are even
allowed, not what the strategy knows.

Data a strategy reads from OUTSIDE the panel (e.g. CoinMarketCap snapshots)
is not tampered here; such sources must be point-in-time by their own tests
(`qlab.data.sources.coinmarketcap.as_of_frame`).

The tampered panel gets its own `snapshot_id`, so a strategy that caches by
snapshot (the re-tuning wrapper does) cannot hand back its untampered result.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from qlab.harness.panel import MarketPanel

DEFAULT_CUT_FRACTIONS: tuple[float, ...] = (1 / 3, 2 / 3)
"""Where the cuts fall, as fractions of the span from the strategy's first
position to the panel's end. Two cuts, so that a
peek of any horizon shorter than a third of the window is caught by at least
one of them; the positions are fixed so a run is reproducible."""

DEFAULT_DECISION_CUTS = 2
"""Extra cuts placed on bars where the strategy's own weights change."""

ATOL = 1e-12
RTOL = 1e-9


def tampered_after(panel: MarketPanel, cut: int, seed: int) -> MarketPanel:
    """`panel` with every price, bar high, volume and funding value after row
    `cut` replaced by a different plausible path; rows up to `cut` untouched."""
    rng = np.random.default_rng(seed)
    after = slice(cut + 1, None)

    prices = panel.prices
    if len(prices.index) - cut - 1 <= 0:
        return panel
    # Every move after the cut is reversed and rescaled: a strategy that reads
    # the direction of a future move sees the opposite one, a strategy that
    # reads its size sees another. Listing gaps (NaN) are kept as they are.
    log_p = np.log(prices)
    moves = log_p.diff().iloc[after]
    scale = rng.uniform(0.5, 2.0, size=moves.shape)
    reversed_moves = (-moves * scale).fillna(0.0).cumsum()
    base = log_p.iloc[cut]
    first_after = log_p.iloc[after].bfill().iloc[0]  # instruments listed after the cut
    base = base.where(base.notna(), first_after)
    new_after = np.exp(reversed_moves.add(base, axis=1)).where(prices.iloc[after].notna())
    prices = prices.copy()
    prices.iloc[after] = new_after.to_numpy()

    funding = panel.funding.copy()
    noise = rng.normal(0.0, 1e-4, size=funding.iloc[after].shape)
    funding.iloc[after] = np.where(funding.iloc[after].isna(), np.nan, noise)

    changes: dict[str, object] = {
        "snapshot_id": f"{panel.snapshot_id}#lookahead-{cut}",
        "prices": prices,
        "funding": funding,
    }
    high = getattr(panel, "high", None)
    if isinstance(high, pd.DataFrame):
        tampered_high = high.copy()
        tampered_high.iloc[after] = prices.iloc[after].to_numpy() * (
            1.0 + np.abs(rng.normal(0.0, 0.01, size=prices.iloc[after].shape))
        )
        tampered_high = tampered_high.where(high.notna())
        changes["high"] = tampered_high
    volume = getattr(panel, "volume", None)
    if isinstance(volume, pd.DataFrame):
        tampered_volume = volume.copy()
        tampered_volume.iloc[after] = volume.iloc[after].to_numpy() * rng.uniform(
            0.1, 10.0, size=volume.iloc[after].shape
        )
        changes["volume"] = tampered_volume
    return dataclasses.replace(panel, **changes)


def lookahead_violation(
    strategy,
    panel: MarketPanel,
    params: Mapping[str, object],
    weights: pd.DataFrame,
    *,
    cut_fractions: Sequence[float] = DEFAULT_CUT_FRACTIONS,
    decision_cuts: int = DEFAULT_DECISION_CUTS,
    seed: int = 0,
) -> str | None:
    """None if `strategy`'s weights up to each cut ignore everything after it;
    otherwise a sentence naming the first cut and the first bar that moved.
    `weights` is the strategy's output on the untouched `panel`."""
    index = panel.prices.index
    n = len(index)
    base = weights.to_numpy(dtype=float)
    held = np.flatnonzero(np.abs(np.nan_to_num(base)).sum(axis=1) > 0)
    if held.size == 0:
        return None  # never holds anything: there is no decision to have peeked
    # Cuts fall inside the span where the strategy holds positions: a cut in
    # a warm-up of zeros would compare zeros with zeros and prove nothing
    # (found on Bv2, whose book holds 4182 of 15313 hourly bars).
    first = int(held[0])
    cuts = [first + int(fraction * (n - 1 - first)) for fraction in cut_fractions]
    # A strategy that trades rarely ignores most bars, so a peek shows only
    # where it acts: also cut ON some of its own decision bars (rows where its
    # weights change), spread over the span -- the decision there may use
    # that bar's data, never the next one's. Bv2 trades ~50 times in ~4000
    # hours and passed with fraction cuts alone.
    changed = np.flatnonzero(
        ~np.isclose(base[1:], base[:-1], atol=ATOL, rtol=RTOL, equal_nan=True).all(axis=1)
    ) + 1
    changed = changed[changed > first]
    if changed.size:
        picks = np.linspace(0, changed.size - 1, num=min(decision_cuts, changed.size))
        cuts += [int(changed[int(round(i))]) for i in picks]
    for k, cut in enumerate(sorted(set(cuts))):
        if cut >= n - 1 or cut < 0:
            continue
        again = strategy.target_weights(tampered_after(panel, cut, seed + k), params)
        again = again.reindex(index=index, columns=weights.columns).to_numpy(dtype=float)
        same = np.isclose(base[: cut + 1], again[: cut + 1], atol=ATOL, rtol=RTOL, equal_nan=True)
        if not same.all():
            row = int(np.flatnonzero(~same.all(axis=1))[0])
            return (
                f"look-ahead: the weights at {index[row]} changed when only data after "
                f"{index[cut]} was changed -- the strategy reads the future"
            )
    return None


__all__ = [
    "DEFAULT_CUT_FRACTIONS",
    "DEFAULT_DECISION_CUTS",
    "lookahead_violation",
    "tampered_after",
]
