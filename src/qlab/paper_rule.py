"""The owner's paper rule (2026-10-25: «как минимум 4-6 недель в paper»).

A book that trades on paper has to run at least four to six weeks before the
owner decides about it. For every paper book in `night/paper_books.yaml` this
module says how long it has been running and whether the stand's replay of the
book matched it; after four weeks with the replay matching, the book is ready
for the owner's decision.

* **Age** -- from the book's first row to its latest in the night's export
  (`data/paper/<day>/`), the same rows `qlab.paper_check` reads.
* **Replay matched** -- the stand replays the paper book through the live code
  on its own data (`scripts/replay_*_paper.py`). It matched when the daily
  returns of replay and paper correlate at least `MIN_REPLAY_CORRELATION`, the
  preregistered bar; the figure comes from the replay's own output lines saved
  in the night's state (`qlab.night.paper_reconciliation`), found by the book's
  `replay_block`. No correlation: unknown, which is not a match.

Paper does not judge profitability -- the stand does. The rule only says when
a decision about the book may be taken; it never changes a status.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.paper_check import equity_curve, load_books

MIN_WEEKS = 4
MAX_WEEKS = 6
"""The owner's range: four to six weeks at least. Four weeks with the replay
matching is when a book becomes ready to decide about; six is the usual end."""

MIN_REPLAY_CORRELATION = 0.80
"""The replay matches the paper book when their daily returns correlate at least
this much: the bar preregistered before the first read of frab.db
(seed/preregistration/2026-09-21-paper-bv2-trend.yaml,
daily_return_correlation_paper_vs_bench_replay, predicted ">= 0.80"; rationale
«XSMOM's clean-execution live-vs-model correlation was +0.86»). No bar on the
difference of total returns was ever declared, so none is used: without a
correlation the match is unknown."""

PAPER_DIR = Path("data/paper")
NIGHT_DIR = Path("data/night")

RULE_TITLE = "Правило владельца: бумага не меньше 4–6 недель"
READY_TEXT = ("по правилу владельца книга готова к решению "
              "(бумага не судит о доходности — судит стенд)")


@dataclass
class ReplayFigures:
    paper: float | None = None
    replay: float | None = None
    correlation: float | None = None

    def matches(self) -> bool | None:
        """True / False by the preregistered correlation bar, None without a
        correlation (the totals alone decide nothing)."""
        if self.correlation is None:
            return None
        return self.correlation >= MIN_REPLAY_CORRELATION


_BV2_RETURNS = re.compile(r"^paper\s+([+-]?\d+(?:\.\d+)?)%\s+replay\s+([+-]?\d+(?:\.\d+)?)%")
_TREND_RETURN = re.compile(r"^(paper|replay)\b.*\(([+-]?\d+(?:\.\d+)?)%\)\s*$")
_CORRELATION = re.compile(r"^daily return correlation\s+(-?\d+(?:\.\d+)?)")
_BOOK = re.compile(r"^book (\d+)\b")


def parse_replay_lines(lines: list[str]) -> dict[str, ReplayFigures]:
    """The replay scripts' output as the night saved it -> figures per block.

    Blocks are named `book <id>` (the Bv2 script prints one per book) and
    `trend`; a line outside a block is ignored. Percentages become fractions."""
    blocks: dict[str, ReplayFigures] = {}
    current: ReplayFigures | None = None
    for raw in lines:
        line = raw.strip()
        if line.startswith("**"):
            current = None
            if line.lower().startswith("**trend"):
                current = blocks.setdefault("trend", ReplayFigures())
            continue
        if book := _BOOK.match(line):
            current = blocks.setdefault(f"book {book.group(1)}", ReplayFigures())
            continue
        if current is None:
            continue
        if m := _BV2_RETURNS.match(line):
            current.paper, current.replay = float(m.group(1)) / 100, float(m.group(2)) / 100
        elif m := _TREND_RETURN.match(line):
            setattr(current, m.group(1), float(m.group(2)) / 100)
        elif m := _CORRELATION.match(line):
            current.correlation = float(m.group(1))
    return blocks


def latest_replay_lines(night_dir: Path | None = None) -> list[str]:
    """The newest saved night's paper-reconciliation lines (empty when none)."""
    folder = night_dir if night_dir is not None else NIGHT_DIR
    for path in sorted(folder.glob("*.json"), reverse=True):
        try:
            lines = json.loads(path.read_text(encoding="utf-8")).get("paper")
        except (OSError, ValueError):
            continue
        if lines:
            return [ln for ln in lines if isinstance(ln, str)]
    return []


def latest_export(root: Path, filename: str) -> Path | None:
    """The newest `<root>/<day>/<filename>`; day folders sort by name."""
    for folder in sorted((p for p in root.glob("*") if p.is_dir()), reverse=True):
        if (folder / filename).is_file():
            return folder / filename
    return None


@dataclass(frozen=True)
class PaperRule:
    idea_id: str
    label: str
    book: str
    first: datetime
    last: datetime
    weeks: float
    replay: str  # "match" | "mismatch" | "unknown"
    ready: bool

    @property
    def whole_weeks(self) -> int:
        return int(self.weeks)

    def line(self) -> str:
        return rule_line(self)

    def to_dict(self) -> dict:
        return {
            "idea_id": self.idea_id, "label": self.label, "book": self.book,
            "first": self.first.isoformat(), "last": self.last.isoformat(),
            "weeks": round(self.weeks, 2), "whole_weeks": self.whole_weeks,
            "min_weeks": MIN_WEEKS, "max_weeks": MAX_WEEKS, "replay": self.replay,
            "ready": self.ready, "text": rule_line(self),
        }


def _weeks_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "неделя"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "недели"
    return "недель"


REPLAY_TEXT = {
    "match": "повтор бумаги на стенде сошёлся",
    "mismatch": "повтор бумаги на стенде не сошёлся",
    "unknown": "повтор бумаги на стенде не посчитан",
}


def rule_line(rule: PaperRule) -> str:
    """The rule's state as one plain Russian line."""
    n = rule.whole_weeks
    text = (f"{rule.label}: на бумаге {n} {_weeks_word(n)} из минимум {MIN_WEEKS}–{MAX_WEEKS} "
            f"(с {rule.first:%d.%m.%Y}); {REPLAY_TEXT[rule.replay]}")
    if rule.ready:
        return f"{text}; {READY_TEXT}"
    if rule.weeks >= MIN_WEEKS:
        return f"{text}; срок вышел, но к решению книга не готова, пока повтор не сойдётся"
    return f"{text}; до решения ещё рано"


def book_age(csv: Path, *, strategy_id: int | None, sum_coins: bool) -> tuple[datetime, datetime]:
    """First and latest row of the book in the export; ValueError when empty."""
    curve = equity_curve(csv, strategy_id=strategy_id, sum_coins=sum_coins)
    if curve.empty:
        raise ValueError(f"no rows for the book in {csv}")
    return curve.index[0].to_pydatetime(), curve.index[-1].to_pydatetime()


def replay_state(session: Session, idea_id: str, cfg: dict,
                 replay_lines: list[str] | None) -> str:
    """"match", "mismatch" or "unknown" for the book's replay."""
    block = cfg.get("replay_block")
    if block and replay_lines:
        verdict = parse_replay_lines(replay_lines).get(block, ReplayFigures()).matches()
        if verdict is not None:
            return "match" if verdict else "mismatch"
    return "unknown"


def paper_rule(session: Session, idea_id: str, cfg: dict, *, root: Path | None = None,
               replay_lines: list[str] | None = None) -> PaperRule | None:
    """One book's age and replay state; None when the book has no export yet."""
    csv = latest_export(root if root is not None else PAPER_DIR, cfg["paper_file"])
    if csv is None:
        return None
    first, last = book_age(csv, strategy_id=cfg.get("strategy_id"),
                           sum_coins=bool(cfg.get("sum_coins")))
    weeks = (last - first).total_seconds() / (7 * 86400)
    replay = replay_state(session, idea_id, cfg, replay_lines)
    return PaperRule(idea_id=idea_id, label=cfg.get("label", idea_id), book=cfg["book"],
                     first=first, last=last, weeks=weeks, replay=replay,
                     ready=weeks >= MIN_WEEKS and replay == "match")


def paper_rules(session: Session, *, books: dict[str, dict] | None = None,
                root: Path | None = None, replay_lines: list[str] | None = None
                ) -> tuple[list[PaperRule], list[str]]:
    """Every mapped book's rule state, and a line for each book that could not
    be computed (one book failing does not stop the others)."""
    books = books if books is not None else load_books()
    rules: list[PaperRule] = []
    problems: list[str] = []
    for idea_id, cfg in books.items():
        label = cfg.get("label", idea_id)
        try:
            rule = paper_rule(session, idea_id, cfg, root=root, replay_lines=replay_lines)
        except Exception as exc:  # noqa: BLE001 - a report line, not a failed night
            problems.append(f"{label}: срок на бумаге не посчитан ({type(exc).__name__}: {exc})")
            continue
        if rule is None:
            problems.append(f"{label}: нет выгрузки бумажной книги, срок не посчитан")
        else:
            rules.append(rule)
    return rules, problems


def report_lines(session: Session, *, books: dict[str, dict] | None = None,
                 root: Path | None = None, replay_lines: list[str] | None = None) -> list[str]:
    """The rule's section of the morning report."""
    rules, problems = paper_rules(session, books=books, root=root, replay_lines=replay_lines)
    return [f"**{RULE_TITLE}**", *[f"    {r.line()}" for r in rules],
            *[f"    {p}" for p in problems]]


def paper_rule_for_idea(session: Session, idea_id: str, *, books: dict[str, dict] | None = None,
                        root: Path | None = None, night_dir: Path | None = None
                        ) -> PaperRule | None:
    """The rule for the idea's own paper book; None for an idea without one.
    Errors reading the export propagate as None too: the idea page must not
    break over a missing file."""
    try:
        books = books if books is not None else load_books()
    except (OSError, KeyError):
        return None
    cfg = books.get(idea_id)
    if cfg is None:
        return None
    try:
        return paper_rule(session, idea_id, cfg, root=root,
                          replay_lines=latest_replay_lines(night_dir))
    except Exception:  # noqa: BLE001
        return None


__all__ = ["MAX_WEEKS", "MIN_WEEKS", "PaperRule", "ReplayFigures", "book_age",
           "latest_export", "latest_replay_lines", "paper_rule", "paper_rule_for_idea",
           "paper_rules", "parse_replay_lines", "report_lines", "rule_line"]
