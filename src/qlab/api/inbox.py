"""The «Входящие» page's data: ideas that are NOT yet in the registry, and the stage each one is at
(owner, 2026-11-04: «у нас есть карточки кандидатов и реестр - и там тоже есть статус кандидат -
очень похоже на дупликацию»).

An idea's path is `ideas/inbox.yaml` -> a card in `cards/` -> `qlab import-cards` puts it into the
registry -> runs. From the moment it is in the registry the registry is its one place: the card is
a section of the idea page and this page only links there. Before that, the stage is read from the
inbox entry by code and nothing here judges an idea:

- ``waiting``      `new`, never taken by the carder -- «ждёт карточника»;
- ``writing``      `new` after at least one attempt -- «карточка пишется / не принята (попытка N)»;
- ``owner``        `needs-owner` -- «вопрос владельцу», the question is the entry's note;
- ``duplicate``    `duplicate` -- «повтор уже сделанного», with what it repeats;
- ``card_ready``   the card is written (`carded` / `implemented`, or a card with no inbox entry)
                   and the idea is not in the registry -- «карточка готова, ждёт реализации»;
- ``registered``   its card's idea is in the registry: a link to the idea page;
- ``closed``       `rejected` by the owner.

Read-only: files and a set of registered ids in, plain dicts out.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qlab import ideas as inbox_module
from qlab.status_words import NOT_RUN_RU

STAGES = ("waiting", "writing", "owner", "duplicate", "card_ready", "registered", "closed")
"""In the order of an idea's path; the page lists the stages in this order."""

STAGE_RU = {
    "waiting": "ждёт карточника",
    "writing": "карточка пишется / не принята",
    "owner": "вопрос владельцу",
    "duplicate": "повтор уже сделанного",
    "card_ready": "карточка готова, ждёт реализации",
    "registered": "уже в реестре",
    "closed": "закрыта владельцем",
}

OWNER_PREFIX = "вопрос владельцу:"
DUPLICATE_PREFIX = "повтор:"


def get_inbox_path() -> Path:
    """``QLAB_INBOX`` (default ``ideas/inbox.yaml``), mirroring ``QLAB_CARDS``."""
    return Path(os.environ.get("QLAB_INBOX", str(inbox_module.INBOX)))


def card_page(idea_id: str, registered: bool) -> str:
    """Where a card is read now: the idea page once the idea is registered, else the inbox."""
    return f"/ideas/{idea_id}" if registered else f"/inbox/{idea_id}"


def _after(prefix: str, notes: str) -> str:
    text = " ".join((notes or "").split())
    return text[len(prefix):].strip() if text.lower().startswith(prefix) else text


def _problems(entry: dict[str, Any]) -> list[str]:
    found = [str(p) for p in entry.get("problems") or []]
    for block in entry.get("review_blocks") or []:
        if isinstance(block, dict):
            found.append(": ".join(str(block[k]) for k in ("kind", "ref", "text") if block.get(k)))
        else:
            found.append(str(block))
    return found


def _card_id(entry: dict[str, Any]) -> str | None:
    """The idea id of the entry's card: the file's name (`cards/<idea_id>.yaml`)."""
    path = entry.get("card")
    return Path(str(path)).stem if path else None


def _item(entry: dict[str, Any], *, cards: dict[str, dict[str, Any]],
          registered: set[str]) -> dict[str, Any]:
    status = entry.get("status")
    card_id = _card_id(entry)
    attempts = int(entry.get("attempts") or 0)
    notes = " ".join(str(entry.get("notes") or "").split())
    out: dict[str, Any] = {
        "id": entry["id"], "date": str(entry.get("date") or ""), "source": entry.get("source"),
        "text": entry.get("text") or "", "status": status, "attempt": attempts,
        "question": None, "repeats": None, "reason": None, "problems": [],
        "idea_id": card_id, "card_title": None, "page": None,
    }
    if status == "rejected":
        stage = "closed"
        out["reason"] = notes or None
    elif card_id and card_id in registered:
        stage = "registered"
    elif status in ("carded", "implemented") or (card_id and card_id in cards):
        stage = "card_ready"
    elif status == "needs-owner":
        stage = "owner"
        out["question"] = _after(OWNER_PREFIX, notes) or None
    elif status == "duplicate":
        stage = "duplicate"
        target, _, reason = _after(DUPLICATE_PREFIX, notes).partition(". ")
        out["repeats"], out["reason"] = target or None, reason or None
    elif attempts > 0:
        stage = "writing"
        out["problems"] = _problems(entry)
    else:
        stage = "waiting"
    out["stage"], out["stage_ru"] = stage, STAGE_RU[stage]
    if stage == "writing":
        out["stage_ru"] = f"{STAGE_RU[stage]} (попытка {attempts})"
    if card_id and stage in ("card_ready", "registered"):
        card = cards.get(card_id)
        out["card_title"] = card["title"] if card else None
        out["page"] = card_page(card_id, stage == "registered")
    return out


def inbox_view(entries: list[dict[str, Any]], cards: list[dict[str, Any]],
               registered: set[str]) -> dict[str, Any]:
    """`{items, counts, stage_ru}`: every inbox idea with its stage, then the cards that have no
    inbox entry (hand-written, the scout's) and are not in the registry yet.

    `counts` has every stage (zero included), so the page and the funnel read the same numbers."""
    by_id = {c["idea_id"]: c for c in cards if c.get("idea_id")}
    items = [_item(e, cards=by_id, registered=registered) for e in entries]
    seen = {i["idea_id"] for i in items if i["idea_id"]}
    for idea_id, card in by_id.items():
        if idea_id in seen or idea_id in registered:
            continue
        items.append({
            "id": None, "date": card.get("found_at") or "", "source": None,
            "text": card.get("plain_summary") or card.get("title") or "", "status": None,
            "attempt": 0, "question": None, "repeats": None, "reason": None, "problems": [],
            "idea_id": idea_id, "card_title": card.get("title"), "stage": "card_ready",
            "stage_ru": STAGE_RU["card_ready"], "page": card_page(idea_id, False),
        })
    order = {s: n for n, s in enumerate(STAGES)}
    items.sort(key=lambda i: (order[i["stage"]], i["date"], i["id"] or i["idea_id"] or ""))
    counts = {s: sum(1 for i in items if i["stage"] == s) for s in STAGES}
    return {"items": items, "counts": counts, "stage_ru": STAGE_RU}


def idea_path(counts: dict[str, int], ideas_by_status: dict[str, int]) -> list[dict[str, Any]]:
    """The funnel's first four steps, one idea counted once: inbox -> card -> in the registry, not
    yet run -> run. `counts` is `inbox_view(...)["counts"]`; `ideas_by_status` is the registry's
    own count (calibration noise left out), so the numbers are those of the inbox page and the
    registry page, not a third reading."""
    not_run = ideas_by_status.get("candidate", 0)
    run = sum(n for status, n in ideas_by_status.items() if status != "candidate")
    return [
        {"key": "inbox", "label_ru": "Во входящих, карточки ещё нет",
         "count": sum(counts.get(s, 0) for s in ("waiting", "writing", "owner", "duplicate"))},
        {"key": "card_ready", "label_ru": "Карточка готова, в реестр ещё не внесена",
         "count": counts.get("card_ready", 0)},
        {"key": "not_run", "label_ru": NOT_RUN_RU, "count": not_run},
        {"key": "run", "label_ru": "В реестре, уже прогонялась", "count": run},
    ]


__all__ = ["NOT_RUN_RU", "STAGES", "STAGE_RU", "card_page", "get_inbox_path", "idea_path",
           "inbox_view"]
