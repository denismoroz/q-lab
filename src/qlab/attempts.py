"""Entries of a night stage that is started more than once the same day.

The night's state file keeps the whole day. When the button starts the night again after an
interrupted one, a stage finds the entries of the earlier attempts in its block, next to its own.
Each entry carries the moment it was written (`at`) and the block the moment the current attempt of
the stage began (`attempt_at`), so that the journal and the morning report can tell the stage's
own work from what an earlier attempt left. Nothing here decides anything; it only labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def latest_per_idea(entries: Sequence[Mapping[str, Any]], key: str = "idea") -> list[Mapping]:
    """One entry per idea, the last one written, in the order the ideas first appeared: an idea
    that failed in the morning and was recomputed in the evening is its evening entry."""
    last: dict[str, Mapping] = {}
    for entry in entries:
        last[str(entry.get(key))] = entry
    return list(last.values())


def earlier_attempt_note(entry: Mapping[str, Any], attempt_at: str | None) -> str:
    """` (прошлый запуск этапа этой ночью, HH:MM UTC)` for an entry written before the stage's
    current attempt began, else an empty string. An entry with no stamp (written before the
    stamps existed) is not labelled."""
    at = entry.get("at")
    if not at or not attempt_at or str(at) >= str(attempt_at):
        return ""
    try:
        clock = datetime.fromisoformat(str(at)).astimezone(UTC).strftime("%H:%M")
    except ValueError:
        return ""
    return f" (прошлый запуск этапа этой ночью, {clock} UTC)"


__all__ = ["earlier_attempt_note", "latest_per_idea", "now_iso"]
