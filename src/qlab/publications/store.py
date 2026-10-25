"""The `publication` table: what the feed has seen, and what became of it.

Every fetched paper is kept (not only candidates), so the next night reads
only the tail: `since` is the newest stored paper of a source minus an
overlap. A paper is stored once -- by its id, and by its title across
sources (the same paper on arXiv and SSRN).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.publications.filter import Rules, judge
from qlab.publications.items import Item, title_key
from qlab.registry.models import Publication


def _aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def since_for(session: Session, source: str, *, now: datetime, lookback_days: int,
              overlap_days: int) -> datetime:
    """Where a source's next read starts: the newest paper stored for it, less
    the overlap; on a source never read, `lookback_days` back."""
    newest = session.query(func.max(Publication.published)).filter(
        Publication.source == source).scalar()
    if newest is None:
        return now - timedelta(days=lookback_days)
    return max(_aware(newest) - timedelta(days=overlap_days), now - timedelta(days=lookback_days))


def add_new(session: Session, items: list[Item], *, now: datetime) -> list[Publication]:
    """Store the papers not seen before; return the new rows. A paper whose id
    or title is already stored is skipped and the stored row is left as it is."""
    seen_ids = {i for (i,) in session.query(Publication.id)}
    seen_titles = {k for (k,) in session.query(Publication.title_key)}
    added = []
    for item in sorted(items, key=lambda i: i.published):
        key = title_key(item.title)
        if item.id in seen_ids or key in seen_titles:
            continue
        row = Publication(
            id=item.id, source=item.source, title=item.title, title_key=key,
            authors=item.authors, published=item.published, abstract=item.abstract,
            link=item.link, categories=item.categories, first_seen=now, filter_version=None,
            candidate=False, score=0, matched={})
        session.add(row)
        added.append(row)
        seen_ids.add(item.id)
        seen_titles.add(key)
    session.flush()
    return added


def apply_filter(session: Session, rules: Rules) -> list[Publication]:
    """Judge every paper not yet judged under `rules.version` that has not
    gone to the scout; return those the rule marks as candidates and that were
    not candidates before (the night's new candidates)."""
    rows = session.query(Publication).filter(
        Publication.handed_at.is_(None),
        (Publication.filter_version.is_(None)) | (Publication.filter_version != rules.version),
    ).all()
    new_candidates = []
    for row in rows:
        verdict = judge(rules, row.title, row.abstract)
        was_candidate = row.candidate
        row.filter_version = rules.version
        row.candidate = verdict.candidate
        row.score = verdict.score
        row.matched = verdict.matched
        if verdict.candidate and not was_candidate:
            new_candidates.append(row)
    session.flush()
    return new_candidates


def pick_for_scout(session: Session, *, cap: int, min_abstract_chars: int) -> list[Publication]:
    """At most `cap` candidates not yet handed to the scout, with an abstract
    long enough to read: best match first, then the newest."""
    return (session.query(Publication)
            .filter(Publication.candidate.is_(True), Publication.handed_at.is_(None),
                    func.length(Publication.abstract) >= min_abstract_chars)
            .order_by(Publication.score.desc(), Publication.published.desc(), Publication.id)
            .limit(max(0, cap)).all())


def mark_handed(session: Session, rows: list[Publication], *, at: datetime,
                notes: dict[str, str], drafts: dict[str, str]) -> None:
    for row in rows:
        row.handed_at = at
        row.scout_note = notes.get(row.id)
        row.draft = drafts.get(row.id)
    session.flush()


__all__ = ["add_new", "apply_filter", "mark_handed", "pick_for_scout", "since_for"]
