"""Tests for `qlab.venues.derive`: the pure venue-facts + spec -> preflight
metric functions, and the atomicity question this task exists to settle
(docs/TASKS.md T13) -- see `atomic_execution`'s docstring for why "no
uncovered leg remains after a failure" (SCREENING.md's actual wording) is
not the same question as "do all legs fill simultaneously," and why the
difference is exactly what lets FRAB pass without any metric being
guessed.
"""

from __future__ import annotations

from qlab.venues.config import VenueConfig
from qlab.venues.derive import (
    atomic_execution,
    data_forward_available,
    derive_venue_metrics,
    venue_supported,
)

HYPERLIQUID_LIKE = VenueConfig(
    id="hyperliquid",
    title="Hyperliquid",
    execution_adapter=True,
    min_leg_notional_usd=10.0,
    free_public_data=True,
    supports_atomic_multileg=False,
    uncovered_leg_recovery="sequential-leg rollback on entry failure",
)

BINANCE_LIKE = VenueConfig(
    id="binance",
    title="Binance",
    execution_adapter=False,
    min_leg_notional_usd=50.0,
    free_public_data=True,
    supports_atomic_multileg=None,
    uncovered_leg_recovery=None,
)

BARE_VENUE = VenueConfig(id="bare", title="Bare")


# --------------------------------------------------------------------------
# venue_supported
# --------------------------------------------------------------------------


def test_venue_supported_true_from_fixture_config() -> None:
    assert venue_supported(HYPERLIQUID_LIKE) is True


def test_venue_supported_false_from_fixture_config() -> None:
    assert venue_supported(BINANCE_LIKE) is False


def test_venue_supported_none_when_field_unset() -> None:
    assert venue_supported(BARE_VENUE) is None


def test_venue_supported_none_when_venue_missing() -> None:
    assert venue_supported(None) is None


# --------------------------------------------------------------------------
# data_forward_available
# --------------------------------------------------------------------------


def test_data_forward_available_true_when_source_matches_and_free() -> None:
    result = data_forward_available(
        HYPERLIQUID_LIKE, snapshot_source="hyperliquid", venue_id="hyperliquid"
    )
    assert result is True


def test_data_forward_available_false_when_snapshot_source_mismatches_venue() -> None:
    """A backtest run on binance data cannot inherit hyperliquid's
    data-availability fact just because the spec NAMES hyperliquid."""
    result = data_forward_available(
        HYPERLIQUID_LIKE, snapshot_source="binance", venue_id="hyperliquid"
    )
    assert result is False


def test_data_forward_available_none_when_free_public_data_unset() -> None:
    result = data_forward_available(BARE_VENUE, snapshot_source="bare", venue_id="bare")
    assert result is None


def test_data_forward_available_none_when_venue_missing() -> None:
    result = data_forward_available(None, snapshot_source="hyperliquid", venue_id="hyperliquid")
    assert result is None


# --------------------------------------------------------------------------
# atomic_execution
# --------------------------------------------------------------------------


def test_atomic_execution_true_for_single_leg_regardless_of_venue() -> None:
    """`simultaneous_legs=1` is satisfied trivially -- no venue lookup even
    matters, since there is no partner leg a failure could ever strand."""
    assert atomic_execution(None, simultaneous_legs=1) is True
    assert atomic_execution(BARE_VENUE, simultaneous_legs=1) is True
    assert atomic_execution(BINANCE_LIKE, simultaneous_legs=1) is True


def test_atomic_execution_true_when_venue_supports_atomic_fills() -> None:
    atomic_venue = VenueConfig(id="v", title="V", supports_atomic_multileg=True)
    assert atomic_execution(atomic_venue, simultaneous_legs=2) is True


def test_atomic_execution_true_via_named_recovery_mechanism_frab_case() -> None:
    """THE FRAB case: Hyperliquid does not fill spot+perp atomically
    (`supports_atomic_multileg=False`), but the live engine actively
    unwinds a partially-opened entry, so a named `uncovered_leg_recovery`
    satisfies the rule anyway -- matching SCREENING.md's actual wording
    ("не остаётся ли неприкрытая нога", not "fills simultaneously")."""
    assert atomic_execution(HYPERLIQUID_LIKE, simultaneous_legs=2) is True


def test_atomic_execution_false_when_neither_atomic_nor_recovered() -> None:
    no_recovery_venue = VenueConfig(
        id="v", title="V", supports_atomic_multileg=False, uncovered_leg_recovery=None
    )
    assert atomic_execution(no_recovery_venue, simultaneous_legs=2) is False


def test_atomic_execution_false_when_recovery_is_blank_string() -> None:
    """An empty/whitespace-only recovery string does not count as naming a
    mechanism -- same as `None`."""
    blank_recovery_venue = VenueConfig(
        id="v", title="V", supports_atomic_multileg=False, uncovered_leg_recovery="   "
    )
    assert atomic_execution(blank_recovery_venue, simultaneous_legs=2) is False


def test_atomic_execution_none_when_atomicity_unknown_and_no_recovery() -> None:
    """Binance: no execution adapter has ever been built, so atomicity was
    never investigated and no recovery mechanism was ever documented --
    the honest answer is "unknown," not a guessed `False`."""
    assert atomic_execution(BINANCE_LIKE, simultaneous_legs=2) is None


def test_atomic_execution_none_when_venue_missing_and_multi_leg() -> None:
    assert atomic_execution(None, simultaneous_legs=2) is None


def test_atomic_execution_none_when_field_unset_on_bare_venue() -> None:
    assert atomic_execution(BARE_VENUE, simultaneous_legs=2) is None


# --------------------------------------------------------------------------
# derive_venue_metrics
# --------------------------------------------------------------------------


def test_derive_venue_metrics_all_present_for_frab_on_hyperliquid() -> None:
    metrics = derive_venue_metrics(
        HYPERLIQUID_LIKE, snapshot_source="hyperliquid", venue_id="hyperliquid", simultaneous_legs=2
    )
    assert metrics == {
        "venue_supported": 1.0,
        "data_forward_available": 1.0,
        "atomic_execution": 1.0,
    }


def test_derive_venue_metrics_drops_unknown_metrics() -> None:
    """Binance, multi-leg: venue_supported is a known False (kept, as
    0.0 -- a known "no" is still a computed metric); atomic_execution is
    unknown (dropped)."""
    metrics = derive_venue_metrics(
        BINANCE_LIKE, snapshot_source="binance", venue_id="binance", simultaneous_legs=2
    )
    assert metrics == {
        "venue_supported": 0.0,
        "data_forward_available": 1.0,
    }
    assert "atomic_execution" not in metrics


def test_derive_venue_metrics_all_absent_when_venue_missing() -> None:
    metrics = derive_venue_metrics(
        None, snapshot_source="ghost", venue_id="ghost", simultaneous_legs=2
    )
    assert metrics == {}


def test_derive_venue_metrics_single_leg_on_unconfigured_venue() -> None:
    """Even with no venue config at all, a single-leg strategy still gets
    an honest atomic_execution=1.0 -- only venue_supported and
    data_forward_available stay absent."""
    metrics = derive_venue_metrics(
        None, snapshot_source="ghost", venue_id="ghost", simultaneous_legs=1
    )
    assert metrics == {"atomic_execution": 1.0}
