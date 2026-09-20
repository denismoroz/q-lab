"""Funding/borrow accrual, with an explicit `NO_ACCRUAL` sentinel.

Modeled directly on `xsec.NO_ACCRUAL`
(`funding-rate-arbitrage/research/cross_sectional/xsec.py`): a run that does
not pass `accrual` at all raises `TypeError`, and simulating a book with NO
held-position cash-flow requires passing the `NO_ACCRUAL` sentinel
EXPLICITLY. Forgetting funding must be impossible; choosing to ignore it
must be visible at the call site. Quoting that module's own rationale for
why `None` was rejected as a default: a crypto book ran for years without
funding in its backtest simply because nobody passed it, and the live book
paid real funding the backtest never saw.

On top of xsec's design, this module adds one more guard the crypto-carry
case needs: if real accrual IS requested and the funding panel has `NaN` for
an instrument the book actually holds, that raises too. An unknown funding
rate is not a zero funding rate — silently `fillna(0.0)`-ing it is exactly
how a carry strategy gets a free ride in a backtest (it would hold straight
through the periods funding is missing, at zero simulated cost).
"""

from __future__ import annotations

import pandas as pd

from qlab.harness.strategy import ZERO_WEIGHT_TOL


class _NoAccrual:
    """Sentinel: caller has explicitly decided this book has no held-position
    cash-flow (funding / swap / borrow). See module docstring."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "qlab.harness.accrual.NO_ACCRUAL"


NO_ACCRUAL = _NoAccrual()


class AccrualError(ValueError):
    """Raised when accrual is requested but cannot be honestly computed."""


def compute_accrual(
    held_weights: pd.DataFrame,
    accrual: pd.DataFrame | _NoAccrual,
) -> pd.Series:
    """Per-period accrual contribution to net return.

    Args:
        held_weights: weights held over each period, already aligned so
            that `held_weights.loc[t]` is exactly the position earning
            `accrual.loc[t]` (see `qlab.harness.run.run_backtest`, which
            builds this alignment from the raw `MarketPanel.funding` with
            the same forward-shift it applies to prices).
        accrual: `NO_ACCRUAL`, or a `DataFrame` the same shape as
            `held_weights` giving the funding/borrow rate (a fraction)
            realised over the SAME period each weight in `held_weights` is
            held. There is no default — omitting this argument entirely
            (passing `None`) is a `TypeError`, matching `xsec.NO_ACCRUAL`.

    Returns:
        A `pd.Series`, one accrual contribution per period, equal to
        `(held_weights * accrual).sum(axis=1)` (all-zero if `NO_ACCRUAL`).
        Sign follows the position: a long (`weight > 0`) with positive
        accrual earns; a short (`weight < 0`) with negative accrual ALSO
        earns (`weight * accrual > 0`) — correct by construction for a
        rate-differential / funding-flow instrument.

    Raises:
        TypeError: `accrual` is `None` (i.e. omitted / forgotten).
        AccrualError: `accrual` is not `NO_ACCRUAL` and is `NaN` at a
            `(period, instrument)` cell where `held_weights` is non-zero.
            An unknown funding rate is not a zero funding rate.
    """
    if accrual is None:
        raise TypeError(
            "compute_accrual(accrual=...) is required. Pass the funding/borrow "
            "panel to apply real accrual, or qlab.harness.accrual.NO_ACCRUAL to "
            "state explicitly that this run has none. `None` is not accepted -- "
            "see this module's docstring for why."
        )
    if isinstance(accrual, _NoAccrual):
        return pd.Series(0.0, index=held_weights.index, name="accrual")

    aligned = accrual.reindex_like(held_weights)
    held_mask = held_weights.abs() > ZERO_WEIGHT_TOL
    missing = aligned.isna() & held_mask
    if missing.any().any():
        first_row = missing.any(axis=1).idxmax()
        first_cols = list(missing.columns[missing.loc[first_row]])
        raise AccrualError(
            f"funding rate is NaN while a position is held, e.g. at {first_row} "
            f"in {first_cols}; an unknown funding rate is not a zero funding "
            "rate -- fix the data or exclude the instrument from the book"
        )
    contribution = (held_weights * aligned.fillna(0.0)).sum(axis=1)
    return contribution.rename("accrual")
