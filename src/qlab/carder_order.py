"""The order in which the carder takes ideas from the inbox (docs/IDEAS.md, docs/RESEARCHER.md).

Owner, 2026-11-03: «да, делай все четыре». The researcher kept writing, night after night, that a
strategy waited for code that sat in the inbox without a card (the Bv2 fast release, idea-0006,
since 2026-10-28; the FRAB drawdown cut, idea-0022, since 2026-10-31), while the carder took the
OLDEST idea each night -- one that had failed twice and was not needed by anybody. So the order is
decided by CODE, from files, not by an agent:

1. **named as needed** -- an idea id that stands in the newest memo of a researcher strategy or of
   the research director, or in `night/directions.yaml`. Inside the group: the idea ranked highest
   in its memo's list of hypotheses (`Гипотезы ...`, the number of its row or item) first; an idea
   named by more files first; then the older one;
2. **the owner's own ideas** (`source: owner`), oldest first;
3. **the rest**, oldest first.

Every pick carries the reason in plain Russian; the morning report prints it, so the owner sees why
this idea and not another. Only ideas the carder may take at all (`new`, attempts left) are ordered.

Nothing here calls an agent or changes the inbox.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from qlab.pipeline.sources import QLAB_ROOT

MEMO_DIR = Path("docs/research")
DIRECTIONS = Path("night/directions.yaml")
DIRECTOR = "director"
UNRANKED = 99

_IDEA = re.compile(r"idea-\d{4}")
_MEMO = re.compile(r"^(?P<name>.+)-(?P<day>\d{4}-\d{2}-\d{2})\.md$")
_ROW_RANK = re.compile(r"^\s*(?:\|\s*)?(?P<n>\d{1,2})\s*[.|)]")
_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+)$")


@dataclass(frozen=True)
class Mention:
    """Where an idea id was named: `who` (a strategy or `director`), the file, the day of the
    memo, and the rank it has in the memo's list of hypotheses (`UNRANKED` when it only stands in
    the text)."""

    who: str
    file: str
    day: str | None
    rank: int = UNRANKED

    def words(self) -> str:
        who = "руководителя исследований" if self.who == DIRECTOR else \
            f"исследователя {self.who}"
        if self.who == "directions":
            return f"в направлениях руководителя ({self.file})"
        place = (f"пункт {self.rank} списка гипотез" if self.rank != UNRANKED
                 else "в тексте, вне списка гипотез")
        return f"в записке {who} от {self.day} ({place})"


@dataclass(frozen=True)
class Pick:
    """An idea for the carder, with the tier it fell in and why, in plain Russian."""

    idea: dict
    tier: str  # named | owner | age
    why: str
    position: int  # the idea's place in the inbox (older first)
    rank: int = UNRANKED
    mentions: tuple[Mention, ...] = field(default_factory=tuple)


def newest_memos(names: set[str], root: Path | None = None) -> dict[str, Path]:
    """Per name (a researcher strategy, `director`): its newest `<name>-<date>.md` memo. Only the
    exact name counts (`frab-*` would also take `frab-binance-spot-...`)."""
    folder = (root or QLAB_ROOT) / MEMO_DIR
    found: dict[str, tuple[str, Path]] = {}
    for path in folder.glob("*.md") if folder.is_dir() else ():
        match = _MEMO.match(path.name)
        if match and match["name"] in names and match["day"] > found.get(match["name"],
                                                                         ("", path))[0]:
            found[match["name"]] = (match["day"], path)
    return {name: path for name, (_, path) in found.items()}


def memo_mentions(text: str) -> dict[str, int]:
    """Idea id -> its best rank in the memo's hypotheses list (`UNRANKED` when named elsewhere).
    A rank is the number of a table row or a numbered item under a heading with `гипотез` in it."""
    best: dict[str, int] = {}
    in_hypotheses = False
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            in_hypotheses = "гипотез" in heading["title"].lower()
            continue
        ids = _IDEA.findall(line)
        if not ids:
            continue
        row = _ROW_RANK.match(line) if in_hypotheses else None
        rank = int(row["n"]) if row else UNRANKED
        for idea_id in ids:
            best[idea_id] = min(best.get(idea_id, UNRANKED), rank)
    return best


def named_as_needed(strategies: set[str], root: Path | None = None) -> dict[str, list[Mention]]:
    """Idea id -> the places that name it: the newest memo of each researcher strategy, the
    newest memo of the director, and `night/directions.yaml` itself."""
    base = root or QLAB_ROOT
    out: dict[str, list[Mention]] = {}
    for who, path in sorted(newest_memos({*strategies, DIRECTOR}, base).items()):
        day = _MEMO.match(path.name)["day"]
        shown = str(path.relative_to(base))
        for idea_id, rank in memo_mentions(path.read_text(encoding="utf-8")).items():
            out.setdefault(idea_id, []).append(Mention(who, shown, day, rank))
    directions = base / DIRECTIONS
    if directions.is_file():
        for idea_id in sorted(set(_IDEA.findall(directions.read_text(encoding="utf-8")))):
            out.setdefault(idea_id, []).append(Mention("directions", str(DIRECTIONS), None))
    return out


def order(inbox: list[dict], *, max_attempts: int, strategies: set[str],
          root: Path | None = None) -> list[Pick]:
    """The ideas the carder may take, in the order it takes them (module docstring)."""
    named = named_as_needed(strategies, root)
    tiers = {"named": 0, "owner": 1, "age": 2}
    picks: list[Pick] = []
    for position, idea in enumerate(inbox):
        if idea["status"] != "new" or int(idea.get("attempts") or 0) >= max_attempts:
            continue
        mentions = tuple(named.get(idea["id"], ()))
        if mentions:
            where = "; ".join(m.words() for m in mentions)
            rank = min(m.rank for m in mentions)
            lead = "названа как нужная" if rank != UNRANKED else "упомянута"
            picks.append(Pick(idea, "named", f"{lead}: {where}", position, rank, mentions))
        elif idea.get("source") == "owner":
            picks.append(Pick(idea, "owner", "идея самого владельца", position))
        else:
            picks.append(Pick(idea, "age", f"по возрасту (с {idea['date']}); никто не назвал её "
                                           "нужной", position))
    return sorted(picks, key=lambda p: (
        tiers[p.tier], p.rank if p.tier == "named" else 0,
        -len(p.mentions) if p.tier == "named" else 0, p.position))


def age_order(inbox: list[dict], *, max_attempts: int) -> list[Pick]:
    """The carder's order before 2026-11-03 and without `need_first`: the oldest first."""
    return [Pick(i, "age", f"по возрасту (с {i['date']})", n) for n, i in enumerate(inbox)
            if i["status"] == "new" and int(i.get("attempts") or 0) < max_attempts]


__all__ = ["Mention", "Pick", "UNRANKED", "age_order", "memo_mentions", "named_as_needed",
           "newest_memos", "order"]
