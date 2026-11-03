"""The paper queue ranked by usefulness to q-lab, not by arrival (docs/SCOUT.md).

Owner, 2026-11-03, «делай все 4»: the scout used to be handed three abstracts a night from a
queue of 65-77, whatever they were about. The publication filter (`qlab.publications.filter`)
answers "is this about a mechanism q-lab could test"; this module orders the papers that pass it
by how CLOSE they are to what q-lab has: the filter's strong terms (the objects of the
catalogue of mechanisms), words that name data and venues q-lab holds (funding, perpetual,
liquidation, listing, open interest, candles), against words that name data it lacks (an order
book, tick data, on-chain pools, CFD), and whether the paper's full text can be read without a
person (arXiv). A pure function of the paper and `night/scout.yaml`'s `papers.rank`: the same
paper always gets the same score.

Like the filter, it never reads what a paper claims to earn.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.publications.filter import _pattern
from qlab.registry.models import Publication

DEFAULT_WEIGHTS = {"strong_term": 3, "other_term": 1, "close_to_data": 2, "far_from_data": -3,
                   "full_text_available": 2, "close_terms": [], "far_terms": []}


@dataclass(frozen=True)
class Ranked:
    row: Publication
    score: float
    why: tuple[str, ...]


def usefulness(title: str, abstract: str, matched: dict[str, list[str]], source_id: str,
               weights: dict) -> tuple[float, tuple[str, ...]]:
    """(score, reasons) of one paper. `matched` is the filter's own verdict (group -> terms)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    text = f"{title}. {abstract}".lower()
    score = 0.0
    why: list[str] = []
    strong = matched.get("strong", [])
    other = [*matched.get("crypto", []), *matched.get("mechanism", [])]
    if strong:
        score += w["strong_term"] * len(strong)
        why.append(f"strong terms: {', '.join(strong)}")
    if other:
        score += w["other_term"] * len(other)
    close = [t for t in w["close_terms"] if _pattern(t).search(text)]
    if close:
        score += w["close_to_data"] * len(close)
        why.append(f"near q-lab's data: {', '.join(close)}")
    far = [t for t in w["far_terms"] if _pattern(t).search(text)]
    if far:
        score += w["far_from_data"] * len(far)
        why.append(f"needs data q-lab lacks: {', '.join(far)}")
    if source_id.startswith("arxiv:"):
        score += w["full_text_available"]
        why.append("full text readable (arXiv)")
    return score, tuple(why)


def rank_candidates(session: Session, weights: dict, *, min_abstract_chars: int = 0,
                    skip_ids: frozenset[str] | set[str] = frozenset(),
                    arxiv_only: bool = False) -> list[Ranked]:
    """Every candidate not yet handed to the scout, best first (ties: the newer paper, then id).
    `skip_ids` are papers another agent has read tonight."""
    query = session.query(Publication).filter(
        Publication.candidate.is_(True), Publication.handed_at.is_(None),
        func.length(Publication.abstract) >= min_abstract_chars)
    ranked = []
    for row in query:
        if row.id in skip_ids or (arxiv_only and not row.id.startswith("arxiv:")):
            continue
        score, why = usefulness(row.title, row.abstract, row.matched or {}, row.id, weights)
        ranked.append(Ranked(row, score, why))
    ranked.sort(key=lambda r: (-r.score, -r.row.published.timestamp(), r.row.id))
    return ranked


__all__ = ["DEFAULT_WEIGHTS", "Ranked", "rank_candidates", "usefulness"]
