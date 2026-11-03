"""What the registry page adds to each idea row (owner, 2026-11-03).

Four things the stored row does not carry, all derived by code and none of them a verdict:

- `bench`: for a bench idea, which of the three labels «скамейка» stood for
  (`qlab.bench_label`) -- or «причина не определена»;
- `family` / `root_id`: the strategy family, by the root of the parent chain
  (`qlab.registry.strategies`), so a filter by family catches every variant;
- `plain_title`: the short plain title the idea's page opens with (`qlab.plain`,
  docs/PLAIN_PAGE.md);
- `changed_24h`: the status change of the last day, from `stage_transition`, old -> new.

The bench labels need one ladder per bench idea and the titles one plain page per idea: a few
seconds together. They change only when the night writes, so they are kept until the registry's
own counters move (`state_key`) or `TTL_SECONDS` pass, whichever comes first. Read-only throughout.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from qlab.bench_label import BenchReason, bench_block, bench_reasons, status_key
from qlab.registry.models import Idea, Spec, StageTransition, Trial
from qlab.registry.strategies import families_of

log = logging.getLogger(__name__)

TTL_SECONDS = 600
RECENT = timedelta(hours=24)


@dataclass
class _Expensive:
    key: tuple | None = None
    at: float = 0.0
    bench: dict[str, BenchReason] = field(default_factory=dict)
    titles: dict[str, str | None] = field(default_factory=dict)
    title_seen: dict[str, tuple] = field(default_factory=dict)
    """Per idea, what its title was built from (`_title_signature`): an idea whose signature did
    not move keeps its title when the cache is rebuilt, so a night that wrote five runs costs five
    titles."""
    titles_at: float = 0.0


class RegistryView:
    """The per-app holder of the cached part. One per `create_app()`, so a test's registry never
    sees another's cache."""

    def __init__(self) -> None:
        self._cache = _Expensive()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @staticmethod
    def state_key(session: Session) -> tuple:
        """The registry's own counters: they move whenever a night writes a run, a spec, a status
        change or a new idea."""
        return (
            session.query(func.count(Idea.id)).scalar(),
            session.query(func.max(Trial.id)).scalar(),
            session.query(func.max(Spec.id)).scalar(),
            session.query(func.max(StageTransition.id)).scalar(),
        )

    def _expensive(self, session: Session) -> _Expensive:
        key = self.state_key(session)
        with self._lock:
            fresh = (self._cache.key == key and time.monotonic() - self._cache.at < TTL_SECONDS)
            if fresh:
                return self._cache
        now = time.monotonic()
        old = self._cache
        keep = now - old.titles_at < TTL_SECONDS  # the plain texts are files: reread now and then
        built = _Expensive(key=key, at=now, bench=_bench(session), titles_at=old.titles_at)
        if keep:
            built.titles, built.title_seen = dict(old.titles), dict(old.title_seen)
        else:
            built.titles_at = now
        _titles(session, built.titles, built.title_seen)
        with self._lock:
            self._cache = built
        return built

    def warm(self, session: Session) -> None:
        """Build the cached part now, so the first page load does not wait for it."""
        self._expensive(session)

    # ------------------------------------------------------------------

    def bench_reasons(self, session: Session) -> dict[str, BenchReason]:
        return self._expensive(session).bench

    def extras(self, session: Session, *, now: datetime | None = None) -> dict[str, dict[str, Any]]:
        """`{idea_id: {...}}` for every idea: the fields listed in the module docstring."""
        expensive = self._expensive(session)
        families = families_of(session)
        changes = recent_changes(session, now=now)
        out: dict[str, dict[str, Any]] = {}
        for idea_id, status in session.query(Idea.id, Idea.status):
            reason = expensive.bench.get(idea_id)
            root, family = families.get(idea_id, (idea_id, "other"))
            out[idea_id] = {
                "root_id": root,
                "family": family,
                "plain_title": expensive.titles.get(idea_id),
                "bench": bench_block(reason),
                "status_key": status_key(status.value, reason),
                "changed_24h": changes.get(idea_id),
            }
        return out


def _bench(session: Session) -> dict[str, BenchReason]:
    from qlab.ladder import load_ledgers

    try:
        return bench_reasons(session, load_ledgers())
    except Exception:  # noqa: BLE001 - a registry view must not fail over a ledger it cannot read
        log.exception("bench labels not built")
        return {}


def _title_signature(idea: Idea, latest_spec: int | None) -> tuple:
    return (idea.parent_id, idea.title, latest_spec)


def _titles(session: Session, titles: dict[str, str | None], seen: dict[str, tuple]) -> None:
    """Fill `titles` for the ideas whose signature moved (all of them on an empty cache)."""
    from qlab.api.cards import load_cards
    from qlab.plain import plain_page

    latest = dict(session.query(Spec.idea_id, func.max(Spec.id)).group_by(Spec.idea_id))
    cards: dict[str, Any] | None = None
    for idea in session.query(Idea):
        if idea.id.startswith("noise-"):
            continue
        signature = _title_signature(idea, latest.get(idea.id))
        if seen.get(idea.id) == signature and idea.id in titles:
            continue
        if cards is None:
            cards = {c["idea_id"]: c for c in load_cards()}
        try:
            titles[idea.id] = (
                str(plain_page(session, idea, cards=cards).get("title") or "") or None)
        except Exception:  # noqa: BLE001 - one idea's text must not take the list down
            log.exception("plain title of %s not built", idea.id)
            titles[idea.id] = None
        seen[idea.id] = signature


def _aware(value: datetime) -> datetime:
    """SQLite hands datetimes back without their zone; every `at` here was written as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def recent_changes(session: Session, *, now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """Ideas whose status changed in the last 24 hours: `{idea_id: {from, to, at, count}}`.

    `from` is the status before the first change of the window and `to` the one after the last, so
    a chain of changes in one night reads as one step. An idea that ended the day where it began is
    left out: nothing changed for the owner. A new idea (no earlier status) has `from: None`.
    """
    now = now or datetime.now(UTC)
    since = now - RECENT
    # SQLite keeps no zone, so the SQL cut is wide (a day of margin) and the exact one is below
    wide = (since - timedelta(days=1)).replace(tzinfo=None)
    rows = (session.query(StageTransition).filter(StageTransition.at >= wide)
            .order_by(StageTransition.at, StageTransition.id).all())
    window: dict[str, list[StageTransition]] = {}
    for row in rows:
        if _aware(row.at) >= since and _aware(row.at) <= now:
            window.setdefault(row.idea_id, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for idea_id, items in window.items():
        first, last = items[0], items[-1]
        before = first.from_status.value if first.from_status else None
        after = last.to_status.value
        if before == after:
            continue
        out[idea_id] = {"from": before, "to": after, "at": _aware(last.at).isoformat(),
                        "count": len(items)}
    return out


__all__ = ["RECENT", "RegistryView", "TTL_SECONDS", "recent_changes"]
