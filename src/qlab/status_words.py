"""The words the owner reads for a registry status (owner, 2026-11-04).

The stored status `candidate` stays as it is -- it means "registered, never run" -- but the word
«кандидат» also named the cards of the inbox, so wherever the owner reads it (the console, the
funnel, the morning report) it is «ещё не прогонялась». One place for that rewording; the other
statuses are shown by their own names in each reader's table."""

from __future__ import annotations

NOT_RUN_RU = "ещё не прогонялась"
"""What the stored registry status `candidate` is called wherever the owner reads it."""

_WORDS = {"candidate": NOT_RUN_RU}


def status_ru(status: str) -> str:
    """The owner's word for a stored status; every other status comes back unchanged."""
    return _WORDS.get(status, status)


__all__ = ["NOT_RUN_RU", "status_ru"]
