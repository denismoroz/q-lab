"""Families of ideas: variants grouped under the strategy they vary
(owner, 2026-10-09: «на скамейке 30 кандидатов — может их как-то
сгруппировать по "родительской" стратегии?»).

Thirty bench ideas were three strategies -- trend, Bv2, XSMOM -- and their
variants: other coin lists, other venues, wrappers that gate or retune them.
A family is decided by code, from what the idea's spec RUNS: the strategy
at the bottom of its wrappers (`inner_code_ref`, as deep as it goes), named
by its module -- `qlab.strategies.live.trend` and `qlab.strategies.trend`
are both "trend". The family's parent is its most established member: live
before paper before the rest, then the oldest. Everyone else in the family
points at it through `idea.parent_id`.

Ideas without a spec (the imported graveyard) and q-lab's own noise books
belong to no family.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from qlab.registry.models import Idea, IdeaStatus, Spec

_NOT_STRATEGIES = ("qlab.calibration.",)
_STATUS_RANK = {IdeaStatus.LIVE: 0, IdeaStatus.PAPER: 1}


def strategy_key(code_ref: str | None, params: object) -> str | None:
    """The module name of the strategy a spec runs, wrappers unwrapped."""
    seen = 0
    while isinstance(params, dict) and params.get("inner_code_ref") and seen < 8:
        code_ref = str(params["inner_code_ref"])
        params = params.get("inner_params") or params.get("base_params") or {}
        seen += 1
    if not code_ref or code_ref.startswith(_NOT_STRATEGIES):
        return None
    module = code_ref.split(":", 1)[0]
    return module.rsplit(".", 1)[-1]


def family_keys(session: Session) -> dict[str, str]:
    """idea id -> strategy key, from each idea's latest spec."""
    out: dict[str, str] = {}
    specs = session.query(Spec).filter(~Spec.code_ref.like("qlab.calibration.%"))
    for spec in specs.order_by(Spec.idea_id, Spec.version):
        key = strategy_key(spec.code_ref, spec.params)
        if key is not None:
            out[spec.idea_id] = key
        else:
            out.pop(spec.idea_id, None)
    return out


def assign_parents(session: Session) -> dict[str, str]:
    """Point every family member at its parent; returns the changes made
    (idea id -> parent id). Idempotent; a parent is never re-chosen away from
    an idea already serving as one unless a more established member exists."""
    keys = family_keys(session)
    by_key: dict[str, list[Idea]] = {}
    for idea_id, key in keys.items():
        idea = session.get(Idea, idea_id)
        if idea is not None:
            by_key.setdefault(key, []).append(idea)
    changed: dict[str, str] = {}
    for members in by_key.values():
        if len(members) < 2:
            continue
        parent = min(members, key=lambda i: (_STATUS_RANK.get(i.status, 2), i.created_at, i.id))
        if parent.parent_id is not None:
            parent.parent_id = None
        for idea in members:
            if idea is not parent and idea.parent_id != parent.id:
                idea.parent_id = parent.id
                changed[idea.id] = parent.id
    session.flush()
    return changed


__all__ = ["assign_parents", "family_keys", "strategy_key"]
