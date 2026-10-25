"""Strategies are the roots of parent chains; variants are what hangs below."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qlab.registry.models import AssetClass, Base, Idea, IdeaStatus, Profile, SourceType
from qlab.registry.strategies import (
    active_variants,
    strategy_root,
    strategy_view,
    would_make_cycle,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _ideas(session, rows):
    """rows: (id, status, parent_id); parents are linked after all rows exist."""
    for idea_id, status, _parent in rows:
        session.add(Idea(id=idea_id, title=idea_id, source_type=SourceType.INTERNAL,
                         asset_class=AssetClass.CRYPTO_PERP, profile=Profile.OTHER,
                         status=status))
    session.flush()
    for idea_id, _status, parent in rows:
        session.get(Idea, idea_id).parent_id = parent
    session.flush()


def test_strategy_root_follows_the_chain_to_the_top(session) -> None:
    _ideas(session, [("a", IdeaStatus.PAPER, None), ("b", IdeaStatus.BENCH, "a"),
                     ("c", IdeaStatus.BENCH, "b"), ("lone", IdeaStatus.BENCH, None)])
    assert strategy_root(session, "c") == "a"
    assert strategy_root(session, "b") == "a"
    assert strategy_root(session, "a") == "a"
    assert strategy_root(session, "lone") == "lone"


def test_strategy_root_of_an_unknown_id_is_the_id_itself(session) -> None:
    assert strategy_root(session, "nowhere") == "nowhere"


def test_strategy_root_survives_a_cycle(session) -> None:
    _ideas(session, [("x", IdeaStatus.BENCH, "y"), ("y", IdeaStatus.BENCH, "z"),
                     ("z", IdeaStatus.BENCH, "y"), ("self", IdeaStatus.BENCH, "self")])
    assert strategy_root(session, "x") == "y"  # the smallest id on the loop y -> z -> y
    assert strategy_root(session, "z") == "y"
    assert strategy_root(session, "self") == "self"


def test_active_variants_are_active_ideas_below_the_root_without_the_root(session) -> None:
    _ideas(session, [
        ("root", IdeaStatus.CANDIDATE, None),
        ("v-bench", IdeaStatus.BENCH, "root"), ("v-validated", IdeaStatus.VALIDATED, "root"),
        ("v-deep", IdeaStatus.CANDIDATE, "v-bench"),
        ("v-rejected", IdeaStatus.REJECTED, "root"), ("v-paper", IdeaStatus.PAPER, "root"),
        ("other", IdeaStatus.BENCH, None), ("other-v", IdeaStatus.BENCH, "other")])
    assert active_variants(session, "root") == ["v-bench", "v-deep", "v-validated"]
    assert active_variants(session, "other") == ["other-v"]
    assert active_variants(session, "v-rejected") == []


def test_would_make_cycle(session) -> None:
    _ideas(session, [("a", IdeaStatus.BENCH, None), ("b", IdeaStatus.BENCH, "a"),
                     ("c", IdeaStatus.BENCH, "b"), ("d", IdeaStatus.BENCH, None)])
    assert would_make_cycle(session, "a", "c")
    assert would_make_cycle(session, "a", "a")
    assert not would_make_cycle(session, "d", "c")
    assert not would_make_cycle(session, "c", "a")


def test_strategy_view_for_the_idea_page(session) -> None:
    _ideas(session, [("root", IdeaStatus.PAPER, None), ("v", IdeaStatus.BENCH, "root")])
    assert strategy_view(session, "v") == {"root": "root", "is_variant": True,
                                           "active_variants": []}
    assert strategy_view(session, "root") == {"root": "root", "is_variant": False,
                                              "active_variants": ["v"]}


def test_a_variant_and_its_own_variants_take_one_place() -> None:
    # Owner, 2026-10-25: the Binance run of the retune "должна быть 1 стратегия - просто вариация".
    from qlab.registry.strategies import active_branches

    class _Q:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *_):
            return _Q([(i,) for i, _p, s in ROWS if s in ("candidate", "bench", "validated")])

        def __iter__(self):
            return iter(self.rows)

    class _S:
        def query(self, *cols):
            if len(cols) == 2:
                return _Q([(i, p) for i, p, _s in ROWS])
            return _Q([(i,) for i, _p, _s in ROWS])

    ROWS = [("bv2", None, "paper"), ("retune", "bv2", "validated"),
            ("retune-binance", "retune", "candidate"), ("headroom", "bv2", "bench"),
            ("old", "bv2", "retired"), ("old-child", "old", "bench")]
    assert active_branches(_S(), "bv2") == ["headroom", "old", "retune"]
