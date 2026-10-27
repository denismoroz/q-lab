"""The ideas inbox (`ideas/inbox.yaml`, docs/IDEAS.md): every idea anybody brings, in the words
it was brought in, and what became of it.

Owner, 2026-10-27: «задача фреймворка генерировать подобные идеи и проверять». The loop
idea -> code -> check -> report starts here: the owner (`qlab idea add`) and the researcher and
the research director (an ```ideas block in their memos) put an idea that may need NEW code into
the inbox; the carder agent (`qlab.agents.carder`) turns the oldest one into a card for the
implementer; the implementer writes the code; the stand checks it.

Rules this module keeps:

- the text of an idea is stored verbatim, never paraphrased and never shortened;
- an idea already in the inbox (the same words, case and spacing aside) is not added twice;
- the status is changed by code only (`qlab.agents.carder`, the owner's commands), never on an
  agent's word.

Statuses: `new` (waits for the carder), `carded` (a card is written and queued for the
implementer), `duplicate` (the same as something already tried or buried), `needs-owner` (the
carder has a question, or refused twice), `implemented` (the implementer took the card),
`rejected` (the owner closed it)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from qlab.pipeline.sources import QLAB_ROOT

INBOX = Path("ideas/inbox.yaml")
STATUSES = ("new", "carded", "duplicate", "needs-owner", "implemented", "rejected")
OWNER = "owner"
MIN_LENGTH = 15  # a memo's idea shorter than this is not an idea
MAX_PER_MEMO = 3  # a memo may offer a few ideas, not a stream of them
HEADER = """\
# The ideas inbox (src/qlab/ideas.py, docs/IDEAS.md). Add with `qlab idea add "..."`; the text
# of an idea is kept verbatim. The carder agent takes the oldest `new` one each night; its
# status is changed by code and by the owner's commands only.
"""
_FENCE = re.compile(r"^```ideas[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else QLAB_ROOT / path


def normal(text: str) -> str:
    """An idea's words as they are compared: case and spacing aside."""
    return re.sub(r"\s+", " ", text).strip().lower()


def source_problem(source: str) -> str | None:
    if source == OWNER or re.fullmatch(r"agent:[a-z][a-z0-9-]*", source):
        return None
    return f"source must be `owner` or `agent:<name>`, got {source!r}"


def load(path: Path | None = None) -> list[dict]:
    path = _abs(path or INBOX)
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("ideas") or [])


def save(ideas: list[dict], path: Path | None = None) -> None:
    path = _abs(path or INBOX)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"ideas": ideas}, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(HEADER + body, encoding="utf-8")


def find(ideas: list[dict], idea_id: str) -> dict | None:
    return next((i for i in ideas if i["id"] == idea_id), None)


def add(ideas: list[dict], text: str, *, source: str, day: date,
        note: str = "") -> tuple[dict, bool]:
    """(entry, added). The words already in the inbox give back their entry, added False."""
    if problem := source_problem(source):
        raise ValueError(problem)
    if not text.strip():
        raise ValueError("an idea needs words")
    key = normal(text)
    for entry in ideas:
        if normal(entry["text"]) == key:
            return entry, False
    entry = {"id": f"idea-{len(ideas) + 1:04d}", "date": day.isoformat(), "source": source,
             "text": text, "status": "new", "card": None, "notes": note, "attempts": 0}
    ideas.append(entry)
    return entry, True


def by_status(ideas: list[dict], *statuses: str) -> list[dict]:
    return [i for i in ideas if i["status"] in statuses]


def oldest_new(ideas: list[dict], *, max_attempts: int) -> dict | None:
    """The idea the carder takes next: the oldest `new` one (inbox order is date order)."""
    return next((i for i in ideas if i["status"] == "new"
                 and int(i.get("attempts") or 0) < max_attempts), None)


def set_status(entry: dict, status: str, note: str | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    entry["status"] = status
    if note is not None:
        entry["notes"] = note


def reject(ideas: list[dict], idea_id: str, why: str) -> dict:
    """The owner closes an idea."""
    entry = find(ideas, idea_id)
    if entry is None:
        raise KeyError(idea_id)
    set_status(entry, "rejected", f"закрыто владельцем: {why.strip()}")
    return entry


def retry(ideas: list[dict], idea_id: str) -> dict:
    """The owner puts an idea back to `new`: the carder takes it again with fresh attempts."""
    entry = find(ideas, idea_id)
    if entry is None:
        raise KeyError(idea_id)
    entry["attempts"] = 0
    set_status(entry, "new", entry.get("notes") or "")
    return entry


# --------------------------------------------------------------------------
# The ```ideas block of an agent's memo


def extract_block(memo_text: str) -> str | None:
    """The body of the memo's first ```ideas fence; None when there is none."""
    match = _FENCE.search(memo_text)
    return match["body"] if match else None


def parse_block(body: str) -> tuple[list[str], list[str]]:
    """(the ideas, what was dropped and why). The block is a YAML list of strings (or of
    mappings with `text`). An idea is kept as written; one too short, not words, or over the
    memo's cap is dropped and said so."""
    try:
        raw = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        return [], [f"the ideas block is not YAML: {exc}"]
    if isinstance(raw, dict) and "ideas" in raw:
        raw = raw["ideas"]
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], ["the ideas block must be a list"]
    kept: list[str] = []
    dropped: list[str] = []
    for item in raw:
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or len(text.strip()) < MIN_LENGTH:
            dropped.append(f"not an idea (a text of at least {MIN_LENGTH} characters): {item!r}")
        elif len(kept) >= MAX_PER_MEMO:
            dropped.append(f"over the cap of {MAX_PER_MEMO} ideas a memo: {text.strip()[:60]!r}")
        else:
            kept.append(text.strip())
    return kept, dropped


def ingest_memo(memo: Path, *, source: str, day: date, path: Path | None = None) -> dict:
    """Put the ideas of an agent's memo into the inbox (idempotent: an idea already there is
    not added again). Returns {"added": [ids], "known": [ids], "dropped": [reasons]}; a memo
    with no block returns all three empty -- the block is optional."""
    out: dict = {"added": [], "known": [], "dropped": []}
    memo_file = _abs(memo)
    if not memo_file.is_file():
        return out
    body = extract_block(memo_file.read_text(encoding="utf-8"))
    if body is None:
        return out
    texts, out["dropped"] = parse_block(body)
    if not texts:
        return out
    ideas = load(path)
    shown = memo_file.relative_to(QLAB_ROOT) if memo_file.is_relative_to(QLAB_ROOT) else memo_file
    for text in texts:
        entry, added = add(ideas, text, source=source, day=day, note=f"из записки {shown}")
        out["added" if added else "known"].append(entry["id"])
    save(ideas, path)
    return out


def report_lines(entries: list[dict]) -> list[str]:
    """One idea per line, for the CLI and the report: id, status, source, the first words."""
    return [f"{e['id']}  {e['status']:<11} {e['source']:<16} {e['date']}  "
            f"{' '.join(e['text'].split())[:90]}" for e in entries]


__all__ = ["HEADER", "INBOX", "STATUSES", "add", "by_status", "extract_block", "find",
           "ingest_memo", "load", "normal", "oldest_new", "parse_block", "reject",
           "report_lines", "retry", "save", "set_status"]
