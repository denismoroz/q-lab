"""Variants are grouped under their parent strategy (qlab.registry.families)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.registry.families import assign_parents, strategy_key
from qlab.registry.models import AssetClass, Base, Idea, IdeaStatus, Profile, SourceType, Spec

T0 = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _idea(session, idea_id, status, code_ref, params=None, day=0):
    session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                     asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER, status=status,
                     created_at=T0 + timedelta(days=day), updated_at=T0))
    if code_ref:
        session.add(Spec(idea_id=idea_id, version=1, params=params or {}, data_requirements={},
                         rebalance="1d", costs_model={}, code_ref=code_ref, created_at=T0))
    session.flush()


def test_wrappers_are_unwrapped_to_the_strategy_they_run() -> None:
    assert strategy_key("qlab.strategies.live.trend:LiveTrendTSMOMEnsemble", {}) == "trend"
    assert strategy_key("qlab.strategies.trend:TrendTSMOMEnsemble", {}) == "trend"
    gated_retune = {"inner_code_ref": "qlab.strategies.retune:Retune",
                    "inner_params": {"inner_code_ref": "qlab.strategies.live.trend:X",
                                     "base_params": {}}}
    assert strategy_key("qlab.strategies.regime_gate:RegimeGate", gated_retune) == "trend"
    assert strategy_key("qlab.calibration.noise:NoiseStrategy", {}) is None


def test_the_most_established_member_is_the_parent(session) -> None:
    _idea(session, "trend-tsmom-crypto", IdeaStatus.PAPER, "qlab.strategies.trend:T", day=5)
    _idea(session, "trend-live", IdeaStatus.BENCH, "qlab.strategies.live.trend:L", day=1)
    _idea(session, "trend-gated", IdeaStatus.BENCH, "qlab.strategies.regime_gate:RegimeGate",
          {"inner_code_ref": "qlab.strategies.live.trend:L", "inner_params": {}}, day=9)
    _idea(session, "frab", IdeaStatus.LIVE, "qlab.strategies.frab:F")
    _idea(session, "fx-carry", IdeaStatus.REJECTED, None)
    changed = assign_parents(session)
    assert changed == {"trend-live": "trend-tsmom-crypto", "trend-gated": "trend-tsmom-crypto"}
    assert session.get(Idea, "trend-tsmom-crypto").parent_id is None
    assert session.get(Idea, "frab").parent_id is None  # alone in its family
    assert session.get(Idea, "fx-carry").parent_id is None  # no spec, no family
    assert assign_parents(session) == {}  # idempotent


def test_composed_strategies_are_not_grouped_by_module() -> None:
    # Every generated idea runs qlab.strategies.blocks:Composed; grouping by module
    # would fold them all into one strategy.
    from qlab.registry.families import strategy_key

    assert strategy_key("qlab.strategies.blocks:Composed", {"signal": {}}) is None
    assert strategy_key("qlab.strategies.trend:TrendTSMOMEnsemble", {}) == "trend"


def test_a_variant_set_under_another_member_stays_there(session) -> None:
    """The same variant on other data sits under that variant (`qlab idea set-parent`); the
    family pass used to flatten it under the root, and it took a place of its own."""
    _idea(session, "strategy-b-v2", IdeaStatus.PAPER, "qlab.strategies.live.bv2:LiveBv2", day=0)
    _idea(session, "bv2-pool", IdeaStatus.CANDIDATE, "qlab.strategies.live.bv2:LiveBv2", day=1)
    _idea(session, "bv2-pool-hl", IdeaStatus.BENCH, "qlab.strategies.live.bv2:LiveBv2", day=2)
    assign_parents(session)
    session.get(Idea, "bv2-pool-hl").parent_id = "bv2-pool"
    assert assign_parents(session) == {}
    assert session.get(Idea, "bv2-pool-hl").parent_id == "bv2-pool"
    assert session.get(Idea, "bv2-pool").parent_id == "strategy-b-v2"
