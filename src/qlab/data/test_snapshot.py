"""Tests for qlab.data.snapshot: point-in-time tradeability, snapshot
round-tripping, and id determinism/stability. No network access — sources are
monkeypatched with synthetic InstrumentHistory fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from qlab.data import snapshot as snap
from qlab.data.sources.base import InstrumentHistory
from qlab.registry.models import Base, DataSnapshot

# --------------------------------------------------------------------------
# DB fixture (in-memory, mirrors registry/test_registry.py's pattern)
# --------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()
    engine.dispose()


# --------------------------------------------------------------------------
# Point-in-time tradeable frame (direct, no DB/network involved)
# --------------------------------------------------------------------------


def _ramp_prices(index: pd.DatetimeIndex, base: float = 100.0) -> pd.Series:
    """A tiny, strictly-increasing synthetic price series -- NOT a flat
    constant. `detect_bad_price_bars` (docs/TASKS.md, T17) flags a bar equal
    to its neighbour as a corrupted print, so a genuinely constant test
    fixture (the old default here) would now trip that check and pollute
    every test in this file that isn't actually about quote sanity. The
    ramp is tiny (1 cent/bar on a ~100 base) so it can never itself trigger
    the "implausible jump" half of the check either."""
    return pd.Series(base + 0.01 * np.arange(len(index)), index=index)


_DEFAULT_VOLUME = 1_000_000.0  # generously liquid -- irrelevant unless a test opts into filtering


def _hist(
    name: str,
    index: pd.DatetimeIndex,
    is_delisted: bool,
    *,
    prices: pd.Series | None = None,
    funding: pd.Series | None = None,
    has_funding: bool = True,
    volume: pd.Series | None = None,
) -> InstrumentHistory:
    if prices is None:
        prices = _ramp_prices(index)
    if funding is None:
        funding = pd.Series(0.0001, index=index)
    if volume is None:
        volume = pd.Series(_DEFAULT_VOLUME, index=index)
    return InstrumentHistory(
        instrument=name,
        prices=prices,
        funding=funding,
        volume=volume,
        trade_count=pd.Series(1, index=index, dtype="int64"),
        first_seen=index[0],
        last_seen=index[-1],
        is_delisted=is_delisted,
        has_funding=has_funding,
    )


class TestPointInTimeTradeable:
    def test_listing_and_delisting_bounds_are_respected(self):
        full_index = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        histories = {
            # listed for the entire window
            "BTC": _hist("BTC", full_index, is_delisted=False),
            # lists partway through (first 3 bars are pre-listing)
            "NEWCOIN": _hist("NEWCOIN", full_index[3:], is_delisted=False),
            # delists partway through (is_delisted=True + last_seen bounds it)
            "DEADCOIN": _hist("DEADCOIN", full_index[:4], is_delisted=True),
        }

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(hours=1)
        )

        # BTC: tradeable throughout
        assert tradeable["BTC"].all()

        # NEWCOIN: False before listing, True from listing onward
        assert not tradeable["NEWCOIN"].iloc[:3].any()
        assert tradeable["NEWCOIN"].iloc[3:].all()

        # DEADCOIN: True up to (and including) delisting bar, False after —
        # a panel that read all-True here would be exactly the survivorship
        # bias documented in funding-rate-arbitrage's XSMOM stress test.
        assert tradeable["DEADCOIN"].iloc[:4].all()
        assert not tradeable["DEADCOIN"].iloc[4:].any()

    def test_prices_may_be_nan_where_not_listed(self):
        full_index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        histories = {"NEWCOIN": _hist("NEWCOIN", full_index[2:], is_delisted=False)}
        prices, _funding, _tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(hours=1)
        )
        assert prices["NEWCOIN"].iloc[:2].isna().all()
        assert prices["NEWCOIN"].iloc[2:].notna().all()


class TestFundingGapTradeable:
    """The 6th defect in docs/ACCEPTANCE_M2.md: a bar with unknown funding
    is marked untradeable -- but only for an instrument where "unknown" is
    an actual gap (a perp). T17's trap is applying that same rule to a spot
    market, whose funding is ALWAYS NaN by construction -- see
    `InstrumentHistory.has_funding`."""

    def test_perp_missing_funding_bar_is_not_tradeable(self):
        full_index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
        # A genuine settlement drop is an ABSENT raw entry, not an explicit
        # NaN -- align_funding_to_index (and therefore this test) must model
        # it that way: `pd.Series.sum(skipna=True)` would otherwise turn a
        # single explicit-NaN "complete" bucket into a silent 0.0, not NaN.
        funding = pd.Series(
            0.0001, index=full_index.delete(1)  # settlement at full_index[1] is missing
        )
        histories = {"BTC": _hist("BTC", full_index, is_delisted=False, funding=funding)}

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1)
        )

        assert not tradeable["BTC"].iloc[1]
        assert tradeable["BTC"].iloc[[0, 2, 3]].all()

    def test_spot_all_nan_funding_stays_tradeable(self):
        """A spot column's funding is NaN for its entire life by
        construction -- that must NOT trip the "unknown funding ->
        untradeable" rule, or every spot bar would be unholdable."""
        full_index = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
        all_nan_funding = pd.Series(float("nan"), index=full_index)
        histories = {
            "BTC-SPOT": _hist(
                "BTC-SPOT",
                full_index,
                is_delisted=False,
                funding=all_nan_funding,
                has_funding=False,
            )
        }

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1)
        )

        assert tradeable["BTC-SPOT"].all()


class TestBadPriceBarsExcludedFromTradeable:
    """Regression for the live defect a coordinator review caught: HL's
    UBTC/USDC spot pair (@142) returned a constant, non-zero placeholder
    price (6969696, then 7979573 -- about 82x BTC's real price) for 11
    daily bars before real trading began and the price jumped to a genuine
    ~$97.6k print. Fed through the ORIGINAL `_build_frames_from_histories`
    (pre quote-sanity-check), those bars were tradeable and the placeholder
    -> real transition read as a fictitious ~-98.8% one-day return. This
    must never happen again for ANY instrument, perp or spot."""

    def test_constant_then_jump_bars_are_excluded_from_tradeable(self):
        full_index = pd.date_range("2025-02-03", periods=13, freq="1D", tz="UTC")
        # Exact values observed live against Hyperliquid's free /info API.
        garbage = [6969696.0] * 5 + [7979573.0] * 6
        real = [97578.0, 97597.0]
        prices = pd.Series(garbage + real, index=full_index)
        histories = {"BTC-SPOT": _hist("BTC-SPOT", full_index, is_delisted=False, prices=prices)}

        out_prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1)
        )

        # Every placeholder bar, and the implausible transition bar itself,
        # must read untradeable -- not just "some of them".
        assert not tradeable["BTC-SPOT"].iloc[:12].any()
        # The two genuinely ordinary bars after the transition are fine.
        assert tradeable["BTC-SPOT"].iloc[12]
        # The corrupted print must not linger in `prices` either -- a
        # momentum/signal computation reading `panel.prices` directly (not
        # just position sizing) must not see it.
        assert out_prices["BTC-SPOT"].iloc[:12].isna().all()
        assert out_prices["BTC-SPOT"].iloc[12] == 97597.0

    def test_ordinary_instrument_unaffected(self):
        """A normal, non-corrupted price series must not lose any
        tradeable bars to this check -- it is a targeted defect filter, not
        a general-purpose volatility cap."""
        full_index = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
        prices = pd.Series([100.0, 101.5, 99.0, 103.0, 102.0, 104.5], index=full_index)
        histories = {"ETH": _hist("ETH", full_index, is_delisted=False, prices=prices)}

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1)
        )

        assert tradeable["ETH"].all()


class TestLiquidityFilter:
    """docs/TASKS.md, T27, gap 2: `min_daily_volume_usd` must be strictly
    point-in-time -- eligibility at bar `t` depends only on volume observed
    BEFORE `t`, never at or after it. This is the exact same shape of bug
    as the frozen 32-coin XSMOM universe (docs/XSMOM_T21.md): curating a
    universe with LATER information and applying it backward."""

    def test_thin_early_liquid_late_is_excluded_before_and_on_the_transition_day(self):
        """Adversarial case: an instrument that is thin for the first half
        of the window and liquid for the second half. A look-ahead filter
        would mark it eligible starting the exact day its OWN volume
        crosses the threshold; a correct point-in-time filter can only
        "know" that from the NEXT bar onward, because it may only consult
        volume observed strictly before the bar being judged."""
        full_index = pd.date_range("2025-01-01", periods=6, freq="1D", tz="UTC")
        # price ~100 (tiny ramp, see _ramp_prices) -- volume chosen with a
        # wide margin either side of the $1,000,000 threshold so the ramp's
        # own tiny price drift never matters to which side of it a bar
        # lands on.
        volume = pd.Series(
            [1_000.0, 1_000.0, 1_000.0, 20_000.0, 20_000.0, 20_000.0], index=full_index
        )
        histories = {
            "THIN2LIQUID": _hist("THIN2LIQUID", full_index, is_delisted=False, volume=volume)
        }

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1), min_daily_volume_usd=1_000_000.0
        )

        col = tradeable["THIN2LIQUID"]
        # Day 0: no prior bar exists to judge eligibility from -- unknown
        # is not eligible, same "fail closed" convention as an unfetched
        # funding rate.
        assert not col.iloc[0]
        # Days 1-2: yesterday was thin -> still ineligible.
        assert not col.iloc[1]
        assert not col.iloc[2]
        # Day 3: THIS is the adversarial assertion. Day 3's OWN volume
        # (20_000, volume_usd ~ $2M) already clears the threshold -- a
        # filter that consulted "today's" volume would read this bar
        # eligible. It must not: yesterday (day 2) was still thin
        # ($1,000 volume, ~$100k), so day 3 stays ineligible.
        assert not col.iloc[3]
        # Days 4-5: yesterday was already liquid -> eligible from here on.
        assert col.iloc[4]
        assert col.iloc[5]

    def test_no_filter_when_threshold_omitted(self):
        """Baseline: an instrument thin its entire life is untouched when
        `min_daily_volume_usd` is not given -- the filter is strictly
        opt-in."""
        full_index = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC")
        thin_volume = pd.Series(0.0001, index=full_index)
        histories = {"THIN": _hist("THIN", full_index, is_delisted=False, volume=thin_volume)}

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1)
        )

        assert tradeable["THIN"].all()

    def test_comfortably_liquid_instrument_loses_only_its_first_bar(self):
        """A liquid instrument only ever loses eligibility on the very
        first bar of the whole panel (no prior bar exists yet to prove
        liquidity from) -- the filter doesn't punish an otherwise-liquid
        name for anything beyond that unavoidable warm-up bar."""
        full_index = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
        volume = pd.Series(1_000_000.0, index=full_index)  # price ~100 -> volume_usd ~$1e8
        histories = {"BTC": _hist("BTC", full_index, is_delisted=False, volume=volume)}

        _prices, _funding, tradeable, _volume = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(days=1), min_daily_volume_usd=1_000_000.0
        )

        assert not tradeable["BTC"].iloc[0]
        assert tradeable["BTC"].iloc[1:].all()

    def test_filter_never_touches_universe_complete_or_instrument_list(
        self, session, patched_source, tmp_path
    ):
        """An instrument excluded for being thin is not the same claim as
        one that was never requested -- see build_snapshot's own
        docstring. universe_complete and the instrument list must be
        completely unaffected by this filter."""
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            min_daily_volume_usd=1.0,
            snapshots_dir=tmp_path, session=session,
        )
        assert panel.meta["universe_complete"] is True
        assert set(panel.instruments) == {"BTC", "DEADCOIN", "ETH"}
        assert panel.meta["min_daily_volume_usd"] == 1.0

    def test_filter_threshold_survives_round_trip(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            min_daily_volume_usd=5.0,
            snapshots_dir=tmp_path, session=session,
        )
        loaded = snap.load_snapshot(panel.snapshot_id, session=session)
        assert loaded.meta["min_daily_volume_usd"] == 5.0

    def test_omitted_threshold_round_trips_as_none(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel.meta["min_daily_volume_usd"] is None
        loaded = snap.load_snapshot(panel.snapshot_id, session=session)
        assert loaded.meta["min_daily_volume_usd"] is None

    def test_negative_threshold_rejected(self, session, tmp_path):
        with pytest.raises(ValueError, match="min_daily_volume_usd"):
            snap.build_snapshot(
                "hyperliquid", ["BTC"], "2024-01-01", "2024-01-02", "1h",
                min_daily_volume_usd=-1.0,
                snapshots_dir=tmp_path, session=session,
            )

    def test_different_threshold_changes_snapshot_id(self, session, patched_source, tmp_path):
        """Two otherwise-identical requests with different liquidity floors
        must not collide on snapshot id, mirroring the same reasoning
        `universe_complete` already gets in this module's docstring."""
        panel1 = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            min_daily_volume_usd=1.0,
            snapshots_dir=tmp_path, session=session,
        )
        panel2 = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            min_daily_volume_usd=2.0,
            snapshots_dir=tmp_path, session=session,
        )
        assert panel1.snapshot_id != panel2.snapshot_id


# --------------------------------------------------------------------------
# build_snapshot / load_snapshot
# --------------------------------------------------------------------------


def _fake_fetch_universe(instruments, start, end, interval, *, on_missing="raise"):
    full_index = pd.date_range(start, end, freq="1h", tz="UTC")
    out = {}
    for i, coin in enumerate(instruments):
        idx = full_index[i:]  # stagger listing so instruments aren't identical
        out[coin] = _hist(coin, idx, is_delisted=False)
    return out


def _fake_discover_universe(as_of_range):
    return ["BTC", "DEADCOIN", "ETH"]  # pretends to include a delisted name


def _fake_describe_universe():
    return [("BTC", False), ("DEADCOIN", True), ("ETH", False)]


@pytest.fixture()
def patched_source(monkeypatch):
    monkeypatch.setitem(snap._SOURCES["hyperliquid"], "fetch", _fake_fetch_universe)
    monkeypatch.setitem(snap._SOURCES["hyperliquid"], "discover_universe", _fake_discover_universe)
    monkeypatch.setitem(snap._SOURCES["hyperliquid"], "describe_universe", _fake_describe_universe)
    return "hyperliquid"


class TestBuildAndLoadSnapshot:
    def test_round_trip_is_identical(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid",
            ["BTC", "ETH"],
            "2024-01-01",
            "2024-01-01T05:00:00",
            "1h",
            snapshots_dir=tmp_path,
            session=session,
        )

        loaded = snap.load_snapshot(panel.snapshot_id, session=session)

        assert loaded.snapshot_id == panel.snapshot_id
        # Parquet round-trips values/dtypes exactly but doesn't preserve a
        # pandas DatetimeIndex's inferred `.freq` — irrelevant to equality.
        pd.testing.assert_frame_equal(loaded.prices, panel.prices, check_freq=False)
        pd.testing.assert_frame_equal(loaded.funding, panel.funding, check_freq=False)
        pd.testing.assert_frame_equal(loaded.tradeable, panel.tradeable, check_freq=False)
        assert loaded.meta["venue"] == panel.meta["venue"]
        assert loaded.meta["interval"] == panel.meta["interval"]

    def test_registers_data_snapshot_row(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid",
            ["BTC"],
            "2024-01-01",
            "2024-01-01T02:00:00",
            "1h",
            snapshots_dir=tmp_path,
            session=session,
        )
        row = session.get(DataSnapshot, panel.snapshot_id)
        assert row is not None
        assert row.source == "hyperliquid"
        assert row.instruments == {"BTC": True}
        assert row.rows == len(panel.prices.index)

    def test_id_stable_across_identical_builds(self, session, patched_source, tmp_path):
        panel1 = snap.build_snapshot(
            "hyperliquid", ["BTC", "ETH"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        panel2 = snap.build_snapshot(
            "hyperliquid", ["ETH", "BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel1.snapshot_id == panel2.snapshot_id

    def test_id_differs_when_range_differs(self, session, patched_source, tmp_path):
        panel1 = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        panel2 = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T06:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel1.snapshot_id != panel2.snapshot_id

    def test_id_differs_when_instruments_differ(self, session, patched_source, tmp_path):
        panel1 = snap.build_snapshot(
            "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        panel2 = snap.build_snapshot(
            "hyperliquid", ["ETH"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel1.snapshot_id != panel2.snapshot_id

    def test_unknown_source_raises(self, session, tmp_path):
        with pytest.raises(ValueError, match="unknown source"):
            snap.build_snapshot(
                "coinbase", ["BTC"], "2024-01-01", "2024-01-02", "1h",
                snapshots_dir=tmp_path, session=session,
            )

    def test_load_unknown_id_raises(self, session):
        with pytest.raises(ValueError, match="no data_snapshot row"):
            snap.load_snapshot("does-not-exist", session=session)


# --------------------------------------------------------------------------
# universe_complete: the second half of the survivorship fix. tradeable
# handles instruments that ARE in the panel; universe_complete says whether
# the panel's instrument LIST itself was hand-picked (survivors) or
# discovered (survivors + delisted).
# --------------------------------------------------------------------------


class TestUniverseComplete:
    def test_discovered_universe_is_marked_complete(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel.meta["universe_complete"] is True
        # discover_universe's delisted name must actually be fetched, not
        # silently dropped -- this is the whole point of the flag.
        assert set(panel.instruments) == {"BTC", "DEADCOIN", "ETH"}

    def test_manual_instruments_are_marked_incomplete(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid", ["BTC", "ETH"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel.meta["universe_complete"] is False

    def test_manual_full_list_is_still_marked_incomplete(self, session, patched_source, tmp_path):
        """Naming every instrument by hand is still a manual selection --
        the flag is about provenance, not about whether it happens to match
        the discovered set."""
        panel = snap.build_snapshot(
            "hyperliquid", ["BTC", "DEADCOIN", "ETH"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert panel.meta["universe_complete"] is False

    def test_universe_complete_survives_round_trip(self, session, patched_source, tmp_path):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        loaded = snap.load_snapshot(panel.snapshot_id, session=session)
        assert loaded.meta["universe_complete"] is True

    def test_discovered_and_manual_ids_differ_even_with_same_instruments(
        self, session, patched_source, tmp_path
    ):
        """A discovered full universe and a hand-typed list that happens to
        name the same coins must not collide on snapshot id -- they are
        different claims about provenance, and honest_universe needs to
        tell them apart."""
        discovered = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        manual = snap.build_snapshot(
            "hyperliquid", ["BTC", "DEADCOIN", "ETH"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert discovered.snapshot_id != manual.snapshot_id

    def test_source_without_discovery_requires_explicit_instruments(self, session, tmp_path):
        monkeypatch_result = snap._SOURCES["binance"]["discover_universe"]((None, None))
        assert monkeypatch_result is None  # sanity: binance really has no discovery

        with pytest.raises(ValueError, match="does not expose a full point-in-time universe"):
            snap.build_snapshot(
                "binance", None, "2024-01-01", "2024-01-02", "1h",
                snapshots_dir=tmp_path, session=session,
            )

    def test_describe_universe_dispatches_to_source(self, patched_source):
        described = snap.describe_universe("hyperliquid")
        assert described == [("BTC", False), ("DEADCOIN", True), ("ETH", False)]

    def test_describe_universe_binance_is_none(self):
        assert snap.describe_universe("binance") is None

    def test_describe_universe_unknown_source_raises(self):
        with pytest.raises(ValueError, match="unknown source"):
            snap.describe_universe("coinbase")


class TestMissingInstrumentProvenance:
    """An instrument with no data in range means different things depending on
    where the list came from: a typo in a hand-written one, ordinary
    point-in-time truth in a venue-discovered one."""

    @staticmethod
    def _recording_fetch(seen: dict):
        def fetch(instruments, start, end, interval, *, on_missing="raise"):
            seen["on_missing"] = on_missing
            return _fake_fetch_universe(instruments, start, end, interval)

        return fetch

    def test_discovered_universe_skips_missing_instruments(
        self, session, patched_source, tmp_path, monkeypatch
    ):
        seen: dict = {}
        monkeypatch.setitem(snap._SOURCES["hyperliquid"], "fetch", self._recording_fetch(seen))
        snap.build_snapshot(
            "hyperliquid",
            None,
            "2026-01-01",
            "2026-01-03",
            "1h",
            snapshots_dir=tmp_path,
            session=session,
        )
        assert seen["on_missing"] == "skip"

    def test_manual_list_still_raises_on_missing_instrument(
        self, session, patched_source, tmp_path, monkeypatch
    ):
        seen: dict = {}
        monkeypatch.setitem(snap._SOURCES["hyperliquid"], "fetch", self._recording_fetch(seen))
        snap.build_snapshot(
            "hyperliquid",
            ["BTC", "ETH"],
            "2026-01-01",
            "2026-01-03",
            "1h",
            snapshots_dir=tmp_path,
            session=session,
        )
        assert seen["on_missing"] == "raise"


# --------------------------------------------------------------------------
# Spot markets (docs/TASKS.md, T17): column naming, universe_complete
# composition, and the structural-vs-gap funding distinction end to end
# through build_snapshot. No network -- the spot fetch/discovery functions
# are monkeypatched exactly like the perp ones above.
# --------------------------------------------------------------------------


def _fake_fetch_spot_universe(coins, start, end, interval, *, on_missing="raise"):
    full_index = pd.date_range(start, end, freq="1h", tz="UTC")
    out = {}
    for i, coin in enumerate(coins):
        idx = full_index[i:]  # stagger listing, same trick as _fake_fetch_universe
        column = f"{coin}-SPOT"
        out[column] = InstrumentHistory(
            instrument=column,
            prices=_ramp_prices(idx),
            funding=pd.Series(dtype=float),
            volume=pd.Series(_DEFAULT_VOLUME, index=idx),
            trade_count=pd.Series(1, index=idx, dtype="int64"),
            first_seen=idx[0],
            last_seen=idx[-1],
            is_delisted=False,
            has_funding=False,
        )
    return out


def _fake_discover_spot_universe(as_of_range):
    return ["BTC-SPOT", "ETH-SPOT"]


def _fake_describe_spot_universe(as_of_range=None):
    return [("BTC-SPOT", False), ("ETH-SPOT", False)]


@pytest.fixture()
def patched_source_with_spot(monkeypatch, patched_source):
    monkeypatch.setitem(snap._SOURCES["hyperliquid"], "fetch_spot", _fake_fetch_spot_universe)
    monkeypatch.setitem(
        snap._SOURCES["hyperliquid"], "discover_spot_universe", _fake_discover_spot_universe
    )
    monkeypatch.setitem(
        snap._SOURCES["hyperliquid"], "describe_spot_universe", _fake_describe_spot_universe
    )
    return patched_source


class TestSpotMarkets:
    def test_discovered_universe_with_include_spot_has_both_column_kinds(
        self, session, patched_source_with_spot, tmp_path
    ):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        assert "BTC" in panel.instruments
        assert "BTC-SPOT" in panel.instruments
        assert panel.meta["universe_complete"] is True

    def test_no_include_spot_never_adds_spot_columns(
        self, session, patched_source_with_spot, tmp_path
    ):
        """Baseline regression: a perp-only request must behave exactly as
        before even though `_SOURCES["hyperliquid"]` now also knows how to
        fetch spot -- `include_spot` defaults to False."""
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert set(panel.instruments) == {"BTC", "DEADCOIN", "ETH"}
        assert panel.meta["no_funding_instruments"] == []

    def test_no_funding_instruments_lists_only_spot_columns(
        self, session, patched_source_with_spot, tmp_path
    ):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        assert set(panel.meta["no_funding_instruments"]) == {"BTC-SPOT", "ETH-SPOT"}
        assert "BTC" not in panel.meta["no_funding_instruments"]

    def test_spot_all_nan_funding_bars_are_tradeable_once_listed(
        self, session, patched_source_with_spot, tmp_path
    ):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        first_valid = panel.prices["BTC-SPOT"].first_valid_index()
        assert panel.tradeable.loc[first_valid:, "BTC-SPOT"].all()

    def test_snapshot_id_deterministic_with_spot_included(
        self, session, patched_source_with_spot, tmp_path
    ):
        panel1 = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        panel2 = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        assert panel1.snapshot_id == panel2.snapshot_id

    def test_snapshot_round_trip_preserves_no_funding_instruments(
        self, session, patched_source_with_spot, tmp_path
    ):
        panel = snap.build_snapshot(
            "hyperliquid", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
            include_spot=True, snapshots_dir=tmp_path, session=session,
        )
        loaded = snap.load_snapshot(panel.snapshot_id, session=session)
        assert sorted(loaded.meta["no_funding_instruments"]) == ["BTC-SPOT", "ETH-SPOT"]

    def test_include_spot_requires_discovered_universe(
        self, session, patched_source_with_spot, tmp_path
    ):
        with pytest.raises(ValueError, match="only valid when instruments is omitted"):
            snap.build_snapshot(
                "hyperliquid", ["BTC"], "2024-01-01", "2024-01-01T05:00:00", "1h",
                include_spot=True, snapshots_dir=tmp_path, session=session,
            )

    def test_include_spot_unsupported_source_raises(self, session, tmp_path, monkeypatch):
        monkeypatch.setitem(
            snap._SOURCES,
            "fakevenue",
            {
                "fetch": _fake_fetch_universe,
                "funding_native_interval": pd.Timedelta(hours=1),
                "discover_universe": lambda as_of_range: ["BTC"],
                "describe_universe": lambda: [("BTC", False)],
            },
        )
        with pytest.raises(ValueError, match="does not support spot markets"):
            snap.build_snapshot(
                "fakevenue", None, "2024-01-01", "2024-01-01T05:00:00", "1h",
                include_spot=True, snapshots_dir=tmp_path, session=session,
            )

    def test_manual_instruments_with_spot_column_dispatch_to_spot_fetch(
        self, session, patched_source_with_spot, tmp_path
    ):
        """A hand-typed instrument list naming a `-SPOT` column routes to
        the spot fetcher even without `include_spot` -- that flag only
        controls AUTO-discovery, not routing of an explicit column name."""
        panel = snap.build_snapshot(
            "hyperliquid", ["BTC", "BTC-SPOT"], "2024-01-01", "2024-01-01T05:00:00", "1h",
            snapshots_dir=tmp_path, session=session,
        )
        assert set(panel.instruments) == {"BTC", "BTC-SPOT"}
        assert panel.meta["universe_complete"] is False
        assert panel.meta["no_funding_instruments"] == ["BTC-SPOT"]
