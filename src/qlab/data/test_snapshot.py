"""Tests for qlab.data.snapshot: point-in-time tradeability, snapshot
round-tripping, and id determinism/stability. No network access — sources are
monkeypatched with synthetic InstrumentHistory fixtures.
"""

from __future__ import annotations

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


def _hist(name: str, index: pd.DatetimeIndex, is_delisted: bool) -> InstrumentHistory:
    prices = pd.Series(100.0, index=index)
    funding = pd.Series(0.0001, index=index)
    return InstrumentHistory(
        instrument=name,
        prices=prices,
        funding=funding,
        first_seen=index[0],
        last_seen=index[-1],
        is_delisted=is_delisted,
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

        _prices, _funding, tradeable = snap._build_frames_from_histories(
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
        prices, _funding, _tradeable = snap._build_frames_from_histories(
            histories, full_index, pd.Timedelta(hours=1)
        )
        assert prices["NEWCOIN"].iloc[:2].isna().all()
        assert prices["NEWCOIN"].iloc[2:].notna().all()


# --------------------------------------------------------------------------
# build_snapshot / load_snapshot
# --------------------------------------------------------------------------


def _fake_fetch_universe(instruments, start, end, interval):
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
