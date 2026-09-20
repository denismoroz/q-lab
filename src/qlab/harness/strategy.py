"""Strategy interface: `(panel, params) -> target weights`.

A strategy is a pure function. It reads a `MarketPanel` and a parameter
mapping and returns a weights `DataFrame`, index and columns matching the
panel exactly. It has no other state and no side effects — the same
`(panel, params)` pair always produces the same weights.

Alignment rule (THE defect class this whole harness exists to prevent):

    `weights.loc[t]` MUST be decided using only information available up to
    and including `t` (prices, funding, tradeable status at or before `t`),
    and it earns the return realised over the NEXT interval, from `t` to
    `t+1`.

`qlab.harness.run.run_backtest` enforces the *earning* half of this pairing
mechanically — it shifts price and funding series so `weights.loc[t]` is
always multiplied against the return/funding realised between `t` and
`t+1`, never against `t` or earlier. It CANNOT enforce the *deciding* half:
nothing stops a `target_weights` implementation from reading
`panel.prices.loc[t + 1:]` while computing `weights.loc[t]`. See
`validate_weights` for exactly what it can and cannot catch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd

from qlab.harness.panel import MarketPanel

# Weights at or below this magnitude are treated as "no position" throughout
# the harness: `validate_weights`'s non-tradeable check, and
# `qlab.harness.metrics.min_capital_usd`'s "smallest position actually held"
# floor both use it, so that float dust (1e-17 left over from an arithmetic
# cancellation) is never mistaken for a real position.
ZERO_WEIGHT_TOL = 1e-12


@runtime_checkable
class Strategy(Protocol):
    """A pure function of `(panel, params) -> target weights`.

    Three concrete shapes this interface must express without any
    special-casing (see `docs/PLAN.md`, milestone M2, "интерфейс
    стратегии"). In all three, the interface stays exactly
    `panel, params -> DataFrame`; the "shape" is just how `target_weights`
    happens to build that one DataFrame, not a different method or a
    subclass hierarchy:

      - **Cross-sectional long/short, periodic rebalance** (XSMOM, TSMOM
        ensembles): `target_weights` computes a cross-sectional score panel
        from `panel` (using only data available at each row), turns it into
        dollar-neutral weights on rebalance rows only (every
        `params["rebal_every"]` periods), and carries the last rebalanced
        vector forward (`ffill`-equivalent) on the rows in between. The
        harness sees an ordinary weights frame that happens to repeat
        between rebalances; `run_backtest`'s turnover naturally comes out
        near 0 on those rows without any special "holding period" concept.

      - **Carry with a limited number of slots and a minimum position
        size** (FRAB-like funding harvest): `target_weights` ranks
        instruments by expected carry (e.g. `panel.funding`), keeps only
        the top `params["max_slots"]`, and — this is the part that is easy
        to get wrong — DROPS (zeroes) any candidate whose resulting weight
        would fall under `params["min_position_size"]`, rather than
        under-sizing it into an unexecutable dust position. Slots and
        minimum size are just extra logic inside `target_weights`; they
        don't need new interface surface.

      - **Spot holding with a conditionally applied hedge leg** (Strategy
        B): `target_weights` assigns a (typically constant) weight to the
        spot column and, on rows where some condition computed from the
        panel holds (a funding regime, a vol trigger, ...), an offsetting
        weight to a separate hedge column; on rows where the condition is
        false, the hedge column gets weight 0. Both legs are just columns
        of the same weights `DataFrame` — "conditionally applied" is a
        per-row branch inside `target_weights`, not a different return
        shape.

    Detecting look-ahead is a REVIEWER's job (see `docs/PLAN.md`'s Adversary
    role — the one stage deliberately given a strong judge model, precisely
    because this can't be checked mechanically), not this interface's or
    `validate_weights`'s. A strategy that peeks at `t+1` while computing
    `weights.loc[t]` produces a weights frame that is structurally
    indistinguishable from an honest one — see `validate_weights`.
    """

    name: str

    def target_weights(self, panel: MarketPanel, params: Mapping[str, object]) -> pd.DataFrame:
        """Return target weights, same index/columns as `panel.prices`.

        Weights are fractions of the book's notional (e.g. `0.5` means "half
        the book, long, in this instrument"); nothing here bounds their sum
        to any particular gross or net exposure — book-level exposure limits
        are a preflight/rules concern (`docs/REGISTRY.md`), not enforced by
        this interface or by `validate_weights`.
        """
        ...


class WeightValidationError(ValueError):
    """Raised by `validate_weights` when a weights frame violates the harness contract."""


def validate_weights(panel: MarketPanel, weights: pd.DataFrame) -> None:
    """Reject structurally broken weights before they reach `run_backtest`.

    Checks (each one is a caller bug, not a modelling choice):
      - `weights` has the exact same index and columns as `panel.prices` —
        no silent reindexing, reordering, or subsetting;
      - no NaN weights — an "I don't know" position is not the same thing
        as a flat `0.0` position, so it must never be produced silently;
      - no non-zero weight (beyond `ZERO_WEIGHT_TOL`) on an instrument at a
        timestamp where `panel.tradeable` is `False` (delisted, not yet
        listed, halted).

    Does NOT check for look-ahead bias. See `Strategy`'s docstring for why
    that is structurally uncatchable here — a look-ahead strategy's weights
    pass every one of the checks above.

    Raises:
        WeightValidationError: on any of the violations above.
    """
    if not weights.index.equals(panel.prices.index):
        raise WeightValidationError(
            "weights index does not match panel.prices index exactly "
            "(different length, order, or timestamps)"
        )
    if not weights.columns.equals(panel.prices.columns):
        raise WeightValidationError(
            "weights columns do not match panel.prices columns exactly "
            "(different instruments or order)"
        )

    nan_mask = weights.isna()
    if nan_mask.any().any():
        first_row = nan_mask.any(axis=1).idxmax()
        first_cols = list(nan_mask.columns[nan_mask.loc[first_row]])
        raise WeightValidationError(
            f"weights contain NaN, e.g. at {first_row} in {first_cols}; "
            "a NaN weight is not a decision -- use 0.0 for flat"
        )

    tradeable = panel.tradeable.reindex_like(weights).astype(bool)
    non_tradeable_exposure = weights.where(~tradeable, 0.0).abs() > ZERO_WEIGHT_TOL
    if non_tradeable_exposure.any().any():
        first_row = non_tradeable_exposure.any(axis=1).idxmax()
        first_cols = list(
            non_tradeable_exposure.columns[non_tradeable_exposure.loc[first_row]]
        )
        raise WeightValidationError(
            f"non-zero weight on non-tradeable instrument(s), e.g. at "
            f"{first_row} in {first_cols}"
        )
