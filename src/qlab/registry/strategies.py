"""Strategies and their variants, read off `idea.parent_id`.

Owner, 2026-10-25: the lab counts STRATEGIES in active work, not ideas --
«bv2-research-headroom-retune-binance ... это все таже самая
bv2-research-headroom-retune ... должна быть 1 стратегия - просто вариация».
A strategy is the root of a parent chain; every idea below the root is one of
its variants. (`qlab.registry.families` assigns parents from the code the
specs run; this module only reads the chains, wherever they came from.)
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from qlab.registry.models import Idea, IdeaStatus

# An idea is in active work while it is a candidate, waits on the bench or has
# passed and waits for the owner.
ACTIVE_STATUSES = (IdeaStatus.CANDIDATE, IdeaStatus.BENCH, IdeaStatus.VALIDATED)


def _parents(session: Session) -> dict[str, str | None]:
    return {i: p for i, p in session.query(Idea.id, Idea.parent_id)}


def _root_of(parents: dict[str, str | None], idea_id: str) -> str:
    seen: list[str] = []
    current = idea_id
    while current in parents and parents[current] is not None:
        if current in seen:  # a cycle: the smallest id of the loop stands for the root
            return min(seen[seen.index(current):])
        seen.append(current)
        current = parents[current]  # type: ignore[assignment]
    return current


def strategy_root(session: Session, idea_id: str) -> str:
    """The top of the idea's parent chain -- the strategy it belongs to. An
    unknown id is its own root; a cycle (which `set-parent` refuses to make)
    resolves to the smallest id on the loop instead of hanging."""
    return _root_of(_parents(session), idea_id)


def active_variants(session: Session, root: str) -> list[str]:
    """Active ideas under `root` (the root itself excluded), sorted."""
    parents = _parents(session)
    active = {i for (i,) in session.query(Idea.id).filter(Idea.status.in_(ACTIVE_STATUSES))}
    return sorted(i for i in active if i != root and _root_of(parents, i) == root)


def active_branches(session: Session, root: str) -> list[str]:
    """The root's direct variants that are in active work themselves or have an
    active variant below them, sorted. A variant of a variant is the same
    variant on other data (bv2-research-headroom-retune-binance under
    bv2-research-headroom-retune), so a branch takes one place, not one per idea."""
    parents = _parents(session)
    active = {i for (i,) in session.query(Idea.id).filter(Idea.status.in_(ACTIVE_STATUSES))}
    branches: set[str] = set()
    for idea in active:
        if idea == root or _root_of(parents, idea) != root:
            continue
        current = idea
        while parents.get(current) not in (None, root):
            current = parents[current]  # type: ignore[assignment]
        branches.add(current)
    return sorted(branches)


def would_make_cycle(session: Session, idea_id: str, parent_id: str) -> bool:
    """True if pointing `idea_id` at `parent_id` closes a loop."""
    parents = _parents(session)
    current: str | None = parent_id
    seen: set[str] = set()
    while current is not None and current not in seen:
        if current == idea_id:
            return True
        seen.add(current)
        current = parents.get(current)
    return False


def strategy_view(session: Session, idea_id: str) -> dict:
    """What the idea page shows: the idea's strategy and, on a root, its active
    variants."""
    parents = _parents(session)
    root = _root_of(parents, idea_id)
    return {"root": root, "is_variant": root != idea_id,
            "active_variants": active_variants(session, idea_id) if root == idea_id else []}


__all__ = ["ACTIVE_STATUSES", "active_branches", "active_variants", "strategy_root", "strategy_view",
           "would_make_cycle"]
