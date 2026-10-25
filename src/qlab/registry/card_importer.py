"""Importer: load candidate cards (`cards/*.yaml`) into the registry (T32).

Cards stay the format of authorship (`cards/SCHEMA.md`); the registry stays the
source of truth about ideas (`docs/REGISTRY.md`). This is the same arrangement the
graveyard has with `seed/graveyard.yaml` (:mod:`qlab.registry.importer`).

What one card becomes:

- an ``idea`` row (id = ``idea_id``, status ``candidate``);
- a ``driver`` row built from the card's ``driver`` block (id = ``idea_id``), so the
  card's driver can be compared with the graveyard's by query, not by hand;
- a ``candidate -> retired`` transition in the funnel ledger when the card itself
  says the owner cannot trade it (``feasibility.venue_supported: false``, or
  ``feasibility.owner_can_open.all_legs_reachable: false``). Code reads the flag
  the card carries; nothing is judged here.

``prior.graveyard_matches`` is checked against the registry: every entry must be
an idea id or a driver id that exists (the first cards name drivers, the schema
text says ideas, so both are accepted) or an idea of the same batch. A typo no
longer passes silently: an unresolved reference fails the whole import before a
row is written.

An idea that already exists (the card of a strategy that has been in the registry
since before its card) keeps its status, title and every other field; only an
empty ``driver_id`` is filled from the card. Running the import twice changes
nothing the second time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from qlab.registry import repo
from qlab.registry.models import (
    AssetClass,
    Driver,
    Idea,
    IdeaStatus,
    Profile,
    SourceType,
)

DEFAULT_CARDS_DIR = Path("cards")
"""Where the owner-accepted cards live. `cards/drafts/` (the scout's drafts) is a
subdirectory and is never read: a draft becomes a candidate only when the owner
moves it up (docs/NIGHT.md)."""

_DRIVER_FIELDS = ("who_pays", "why", "kill_condition", "observable")


class CardImportError(ValueError):
    """One or more cards failed validation; nothing was written."""


@dataclass(slots=True)
class CardReport:
    """Summary of one :func:`import_cards` call."""

    ideas_inserted: list[str] = field(default_factory=list)
    ideas_existing: list[str] = field(default_factory=list)
    drivers_inserted: list[str] = field(default_factory=list)
    drivers_attached: list[str] = field(default_factory=list)
    retired: list[tuple[str, str]] = field(default_factory=list)
    matches_checked: int = 0


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _problems(raw: Any, path: Path) -> list[str]:
    """Structural problems of one parsed card, one line each."""
    if not isinstance(raw, dict):
        return [f"{path}: not a YAML mapping"]
    found: list[str] = []
    for key in ("idea_id", "title", "plain_summary"):
        if _text(raw.get(key)) is None:
            found.append(f"{path}: `{key}` is missing"
                         + (" -- a card without a plain summary is not accepted"
                            if key == "plain_summary" else ""))
    driver = raw.get("driver")
    if not isinstance(driver, dict):
        found.append(f"{path}: `driver` block is missing")
    else:
        found += [f"{path}: `driver.{name}` is missing"
                  for name in _DRIVER_FIELDS if _text(driver.get(name)) is None]
    matches = (raw.get("prior") or {}).get("graveyard_matches") if isinstance(
        raw.get("prior"), dict) else None
    if matches is not None and (
            not isinstance(matches, list) or not all(isinstance(m, str) for m in matches)):
        found.append(f"{path}: `prior.graveyard_matches` must be a list of ids")
    return found


def read_cards(cards_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every ``*.yaml`` directly in ``cards_dir`` (not in ``drafts/``), validated."""
    if not cards_dir.is_dir():
        raise FileNotFoundError(f"no cards directory at {cards_dir}")
    cards: list[tuple[Path, dict[str, Any]]] = []
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for path in sorted(cards_dir.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        found = _problems(raw, path)
        if not found and raw["idea_id"] in seen:
            found = [f"{path}: idea_id {raw['idea_id']!r} repeats {seen[raw['idea_id']]}"]
        if found:
            problems += found
            continue
        seen[raw["idea_id"]] = path
        cards.append((path, raw))
    if problems:
        raise CardImportError("cards failed validation:\n"
                              + "\n".join(f"  - {p}" for p in problems))
    return cards


def unreachable_reason(card: dict[str, Any]) -> str | None:
    """Why the card says the owner cannot trade it, or None when it does not say so.

    Both forms of the card carry the fact: the first five wrote
    ``feasibility.venue_supported: false`` with a note, the later ones add
    ``feasibility.owner_can_open``. A card that says nothing is not retired.
    """
    feasibility = card.get("feasibility")
    if not isinstance(feasibility, dict):
        return None
    reasons: list[str] = []
    if feasibility.get("venue_supported") is False:
        reasons.append("venue_supported is false"
                       + (f" ({_text(feasibility.get('note'))})" if _text(
                           feasibility.get("note")) else ""))
    can_open = feasibility.get("owner_can_open")
    if isinstance(can_open, dict) and can_open.get("all_legs_reachable") is False:
        legs = "; ".join(str(leg) for leg in can_open.get("unreachable_legs") or [])
        reasons.append("the owner cannot open every leg" + (f": {legs}" if legs else ""))
    return "; ".join(reasons) if reasons else None


def _source_type(card: dict[str, Any]) -> SourceType:
    urls = [str(s.get("url", "")) for s in card.get("sources") or [] if isinstance(s, dict)]
    return SourceType.PAPER if any(u.startswith("http") for u in urls) else SourceType.INTERNAL


def _notes(card: dict[str, Any], path: Path) -> str:
    lines = [f"Card {path} (found {card.get('found_at')}).", _text(card["plain_summary"]) or ""]
    matches = (card.get("prior") or {}).get("graveyard_matches") or []
    if matches:
        lines.append("graveyard_matches: " + ", ".join(matches))
    return "\n".join(lines)


def check_matches(session: Session, cards: Iterable[tuple[Path, dict[str, Any]]]) -> list[str]:
    """Unresolved ``prior.graveyard_matches`` references, one line each."""
    cards = list(cards)
    batch = {card["idea_id"] for _, card in cards}
    problems: list[str] = []
    for path, card in cards:
        for match in (card.get("prior") or {}).get("graveyard_matches") or []:
            if match in batch or session.get(Idea, match) or session.get(Driver, match):
                continue
            problems.append(f"{path}: graveyard match {match!r} is neither an idea nor a "
                            "driver in the registry")
    return problems


def import_cards(session: Session, cards_dir: Path | str = DEFAULT_CARDS_DIR) -> CardReport:
    """Load every card in ``cards_dir`` into the registry (idempotent).

    Raises :class:`CardImportError` before any write when a card is malformed or a
    ``graveyard_matches`` reference does not resolve.
    """
    cards = read_cards(Path(cards_dir))
    unresolved = check_matches(session, cards)
    if unresolved:
        raise CardImportError("cards failed validation:\n"
                              + "\n".join(f"  - {p}" for p in unresolved))

    report = CardReport()
    for path, card in cards:
        idea_id = card["idea_id"]
        report.matches_checked += len((card.get("prior") or {}).get("graveyard_matches") or [])
        driver = card["driver"]
        existing = session.get(Idea, idea_id)

        needs_driver = existing is None or existing.driver_id is None
        if needs_driver and session.get(Driver, idea_id) is None:
            repo.upsert_driver(
                session, id=idea_id, title=_text(card["title"]) or idea_id,
                description=f"{_text(driver['who_pays'])} {_text(driver['why'])}",
                kill_condition=_text(driver["kill_condition"]) or "",
                observable=_text(driver["observable"]) or "")
            report.drivers_inserted.append(idea_id)

        if existing is not None:
            report.ideas_existing.append(idea_id)
            if existing.driver_id is None:
                existing.driver_id = idea_id
                report.drivers_attached.append(idea_id)
            continue

        repo.upsert_idea(
            session, id=idea_id, title=_text(card["title"]) or idea_id,
            source_type=_source_type(card), asset_class=AssetClass.CRYPTO_PERP,
            profile=Profile.OTHER, driver_id=idea_id, claimed_edge=_text(driver["why"]),
            status=IdeaStatus.CANDIDATE, notes=_notes(card, path))
        report.ideas_inserted.append(idea_id)

        reason = unreachable_reason(card)
        if reason is not None:
            repo.set_status(session, idea_id=idea_id, new_status=IdeaStatus.RETIRED,
                            reason=f"card {path}: {reason}")
            report.retired.append((idea_id, reason))
    session.flush()
    return report


__all__ = ["CardImportError", "CardReport", "DEFAULT_CARDS_DIR", "check_matches",
           "import_cards", "read_cards", "unreachable_reason"]
