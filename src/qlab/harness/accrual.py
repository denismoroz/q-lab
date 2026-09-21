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

That guard has one deliberate exception (docs/TASKS.md, T17): an instrument
that structurally never pays funding at all — a spot market, held against a
perp short — is NaN in `funding` for its entire life, and that is the
CORRECT shape of the data, not a gap. Raising on it would make holding spot
impossible outright, which is not what the guard is for; the guard exists
to catch a venue silently dropping a settlement it should have reported,
not to forbid an instrument that was never going to report one. Callers
name these columns explicitly via `no_funding_instruments` — nothing here
infers "no funding" from a column's name or shape, since doing so from
inside the accrual check would be exactly the kind of filename heuristic
this task was warned against. Every other column keeps the original,
unweakened guard.
"""

from __future__ import annotations

from collections.abc import Iterable

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
    *,
    no_funding_instruments: Iterable[str] = (),
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
        no_funding_instruments: columns of `held_weights`/`accrual` whose
            missing funding is STRUCTURAL, not a gap — e.g. a spot market,
            which never pays funding by construction (see
            `qlab.data.sources.base.InstrumentHistory.has_funding` and
            `qlab.data.panel.MarketPanel.meta["no_funding_instruments"]`).
            Holding one of these through a `NaN` bar contributes zero
            accrual for that bar rather than raising. Every column NOT
            listed here keeps the unweakened `AccrualError` guard below —
            this parameter narrows WHERE the guard applies, it does not
            loosen what the guard does.

    Returns:
        A `pd.Series`, one accrual contribution per period, equal to
        `(held_weights * accrual).sum(axis=1)` (all-zero if `NO_ACCRUAL`;
        NaN treated as zero contribution only for a column named in
        `no_funding_instruments`, see above).
        Sign follows the position: a long (`weight > 0`) with positive
        accrual earns; a short (`weight < 0`) with negative accrual ALSO
        earns (`weight * accrual > 0`) — correct by construction for a
        rate-differential / funding-flow instrument.

    Raises:
        TypeError: `accrual` is `None` (i.e. omitted / forgotten).
        AccrualError: `accrual` is not `NO_ACCRUAL` and is `NaN` at a
            `(period, instrument)` cell where `held_weights` is non-zero
            and the column is NOT listed in `no_funding_instruments`. An
            unknown funding rate is not a zero funding rate.
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

    # A structurally fundingless instrument's NaN is expected, not a gap --
    # exclude exactly those columns from the "is this a gap" check. Every
    # other column's NaN-while-held is still a gap, unconditionally.
    structural_cols = [c for c in no_funding_instruments if c in missing.columns]
    if structural_cols:
        missing = missing.copy()
        missing.loc[:, structural_cols] = False

    if missing.any().any():
        first_row = missing.any(axis=1).idxmax()
        first_cols = list(missing.columns[missing.loc[first_row]])
        raise AccrualError(
            f"funding rate is NaN while a position is held, e.g. at {first_row} "
            f"in {first_cols}; an unknown funding rate is not a zero funding "
            "rate -- fix the data or exclude the instrument from the book"
        )
    # SIGN CONVENTION, and it is not a detail: a venue's funding rate is the
    # rate LONGS PAY SHORTS, so a positive rate is a cost to a long position
    # and income to a short one. The accrual is therefore MINUS weight times
    # rate, not weight times rate.
    #
    # Getting this backwards does not produce an obviously broken number, it
    # produces a plausible one with the wrong sign, and it inverts the verdict
    # of every strategy whose edge is carry. It was caught only because FRAB —
    # which is spot-long against a perp-short, entering when funding is
    # positive, and which earns real money in production — came out at
    # -7.2%/yr here. Its own entry rule settles the convention: it goes SHORT
    # the perp when funding is HIGH, so a positive rate must pay the short.
    contribution = -(held_weights * aligned.fillna(0.0)).sum(axis=1)
    return contribution.rename("accrual")
