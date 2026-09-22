"""Loader for candidate cards (`cards/*.yaml`, schema: cards/SCHEMA.md).

Cards are files, not registry rows, so this module reads and normalizes
them for display. It never fills a gap: a `null` field stays `null` and is
labelled as absent by the UI, because cards/SCHEMA.md is explicit that
"нет в источнике" is more honest than invented content.

Location comes from ``QLAB_CARDS`` (default: ``cards``, relative to the
working directory), mirroring how ``QLAB_DB`` locates the database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# The six parts of a card, in the order cards/SCHEMA.md lists them. The UI
# renders exactly these as the card's body; `feasibility` and `prior` are
# the two appended sections and are rendered after them.
CARD_PARTS = ("driver", "signal", "sizing", "protection", "execution", "death")


def get_cards_dir() -> Path:
    """Resolve the cards directory from ``QLAB_CARDS`` (default ``cards``)."""
    return Path(os.environ.get("QLAB_CARDS", "cards"))


def _quotes(section: Any) -> list[dict[str, Any]]:
    """Collect every quote in a section.

    A section may carry more than one quote (`quote`, `quote_venue`, ...);
    cards/SCHEMA.md requires each claim to stand on a verbatim quote with
    its source, so all of them are surfaced, never just the first.
    """
    if not isinstance(section, dict):
        return []
    found: list[dict[str, Any]] = []
    for key, value in section.items():
        if key != "quote" and not key.startswith("quote_"):
            continue
        if not isinstance(value, dict):
            continue
        found.append(
            {
                "key": key,
                "text": value.get("text"),
                "source": value.get("source"),
            }
        )
    return found


def _fields(section: Any) -> dict[str, Any]:
    """The section's own fields, with quotes split out."""
    if not isinstance(section, dict):
        return {}
    return {
        key: value
        for key, value in section.items()
        if key != "quote" and not key.startswith("quote_")
    }


def protection_is_empty(protection: Any) -> bool:
    """True when the card records no tail protection at all.

    This is a reading of what the card says, not a judgement about the
    strategy: all three mechanisms null and `structural` not true means the
    source describes nothing that closes the tails. cards/README.md calls
    such a row a finding in its own right ("пустая строка «защита» — это
    сама по себе находка"), so the UI must show it as a stated absence
    rather than as an empty cell that reads like missing data.
    """
    if not isinstance(protection, dict):
        return True
    mechanisms = (protection.get("crash"), protection.get("pump"), protection.get("costs"))
    return all(m is None for m in mechanisms) and protection.get("structural") is not True


def parse_card(raw: dict[str, Any], *, path: str) -> dict[str, Any]:
    """Normalize one loaded card into the shape the UI renders."""
    parts = {}
    for name in CARD_PARTS:
        section = raw.get(name)
        parts[name] = {
            "present": isinstance(section, dict),
            "fields": _fields(section),
            "quotes": _quotes(section),
        }

    protection = raw.get("protection")
    return {
        "idea_id": raw.get("idea_id"),
        "title": raw.get("title"),
        "found_at": str(raw["found_at"]) if raw.get("found_at") is not None else None,
        "path": path,
        "sources": raw.get("sources") or [],
        "parts": parts,
        "feasibility": raw.get("feasibility"),
        "prior": raw.get("prior"),
        "protection_is_empty": protection_is_empty(protection),
    }


def load_cards(cards_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load every `cards/*.yaml`, sorted by idea_id. Missing directory -> []."""
    directory = cards_dir if cards_dir is not None else get_cards_dir()
    if not directory.is_dir():
        return []

    cards: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            continue
        cards.append(parse_card(raw, path=str(path)))
    cards.sort(key=lambda card: card["idea_id"] or "")
    return cards


def card_summary(card: dict[str, Any]) -> dict[str, Any]:
    """Short form for the card index."""
    driver = card["parts"]["driver"]["fields"]
    return {
        "idea_id": card["idea_id"],
        "title": card["title"],
        "found_at": card["found_at"],
        "who_pays": driver.get("who_pays"),
        "protection_is_empty": card["protection_is_empty"],
        "source_count": len(card["sources"]),
    }


__all__ = [
    "CARD_PARTS",
    "card_summary",
    "get_cards_dir",
    "load_cards",
    "parse_card",
    "protection_is_empty",
]
