"""The researcher's memos in the registry (qlab.agents.researcher).

A memo is a markdown file, `docs/research/<strategy>-<date>.md`, to which code
appends the stand's results under "## Эксперименты на стенде". This module
parses those result lines and stores the memo as one `research_note` row per
idea the memo covers, so the console can show it on the idea page and later
analysis does not need to scrape files. Writing is an upsert keyed on
(idea, memo path): running it again changes nothing it should not.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.registry.models import Idea, ResearchNote

MEMO_DIR = Path("docs/research")
SPEC_DIR = Path("specs/research")

# The header run_experiments writes; a memo may carry it more than once (a
# restarted run appends a second section).
EXPERIMENTS_HEADER = "## Эксперименты на стенде"

_RAN = re.compile(r"^- `(?P<idea>[^`]+)` \(прогон (?P<trial>\d+)\): (?P<route>[\w-]+) — ")
_CRASHED = re.compile(r"^- `(?P<idea>[^`]+)`: прогон упал")
_NOT_RUN = re.compile(r"^- `(?P<path>[^`]+\.yaml)`: (?:не запущен|спек не загружается)")
_MEMO_NAME = re.compile(r"^(?P<name>.+)-(?P<day>\d{4}-\d{2}-\d{2})$")


def _experiment_lines(memo: str) -> list[str]:
    """Lines of every stand section: from its header to the next heading."""
    lines: list[str] = []
    inside = False
    for line in memo.splitlines():
        if line.startswith("## "):
            inside = line.startswith(EXPERIMENTS_HEADER)
        elif inside:
            lines.append(line)
    return lines


def spec_index(spec_dir: Path = SPEC_DIR, extra: Iterable[Path | str] = ()) -> dict[str, str]:
    """Experiment idea id -> spec path. Specs are named after their idea, so
    the file stem is the idea id; `extra` (known paths) wins over the scan."""
    index = ({p.stem: p.as_posix() for p in sorted(spec_dir.rglob("*.yaml"))}
             if spec_dir.is_dir() else {})
    index.update({Path(p).stem: Path(p).as_posix() for p in extra})
    return index


def parse_experiments(memo: str, specs: Mapping[str, str] | None = None) -> list[dict]:
    """The stand's result lines of a memo as {"idea_id", "trial_id", "route",
    "spec"}. A run that crashed has route "error" and no trial; a spec that
    was not run (or did not load) has no idea id, route "not-run" and its
    path. `spec` is null where the path cannot be recovered."""
    specs = specs or {}
    out: list[dict] = []
    for line in _experiment_lines(memo):
        if m := _RAN.match(line):
            out.append({"idea_id": m["idea"], "trial_id": int(m["trial"]), "route": m["route"],
                        "spec": specs.get(m["idea"])})
        elif m := _CRASHED.match(line):
            out.append({"idea_id": m["idea"], "trial_id": None, "route": "error",
                        "spec": specs.get(m["idea"])})
        elif m := _NOT_RUN.match(line):
            out.append({"idea_id": None, "trial_id": None, "route": "not-run",
                        "spec": m["path"]})
    return out


def upsert_research_notes(
    session: Session, *, name: str, idea_ids: Iterable[str], memo_path: Path | str,
    memo: str | None = None, at: datetime | None = None, refresh_at: bool = True,
    problems: Iterable[str] = (), tokens: int | None = None, model: str | None = None,
    specs: Mapping[str, str] | None = None,
) -> list[ResearchNote]:
    """One note per idea the memo covers (ideas not in the registry are
    skipped: the foreign key would refuse them). The memo text is read from
    the file unless given. An existing note has its memo, experiments and
    problems refreshed; `at` only when `refresh_at`, and tokens and model only
    when given, so a later import of the same file cannot blank them."""
    path = Path(memo_path)
    text = memo if memo is not None else path.read_text(encoding="utf-8")
    key = path.as_posix()
    experiments = parse_experiments(text, specs if specs is not None else spec_index())
    notes: list[ResearchNote] = []
    for idea_id in dict.fromkeys(idea_ids):
        if session.get(Idea, idea_id) is None:
            continue
        note = (session.query(ResearchNote)
                .filter(ResearchNote.idea_id == idea_id, ResearchNote.memo_path == key).first())
        if note is None:
            note = ResearchNote(idea_id=idea_id, memo_path=key, at=at or datetime.now(UTC),
                                name=name, memo=text, experiments=experiments,
                                problems=list(problems), tokens=tokens, model=model)
            session.add(note)
        else:
            note.name, note.memo, note.experiments = name, text, experiments
            note.problems = list(problems)
            if refresh_at and at is not None:
                note.at = at
            if tokens is not None:
                note.tokens = tokens
            if model is not None:
                note.model = model
        notes.append(note)
    session.flush()
    return notes


@dataclass
class ImportReport:
    imported: list[str] = field(default_factory=list)  # "<file>: <n> ideas"
    skipped: list[str] = field(default_factory=list)  # "<file>: <why>"


def import_memos(session: Session, *, strategies: Mapping[str, Iterable[str]],
                 memo_dir: Path = MEMO_DIR, spec_dir: Path = SPEC_DIR) -> ImportReport:
    """Every `<strategy>-<date>.md` in `memo_dir` into the registry. The
    strategy's idea ids come from `strategies` (night/research.yaml); a file
    of a strategy not there is skipped and reported. The note's date is the
    file name's date at UTC midnight; tokens and model stay null."""
    report = ImportReport()
    specs = spec_index(spec_dir)
    for path in sorted(memo_dir.glob("*.md")):
        m = _MEMO_NAME.match(path.stem)
        if m is None:
            report.skipped.append(f"{path.name}: no date in the file name")
            continue
        if m["name"] not in strategies:
            report.skipped.append(f"{path.name}: strategy {m['name']!r} is not in research.yaml")
            continue
        day = date.fromisoformat(m["day"])
        notes = upsert_research_notes(
            session, name=m["name"], idea_ids=strategies[m["name"]],
            memo_path=memo_dir / path.name, at=datetime(day.year, day.month, day.day, tzinfo=UTC),
            refresh_at=False, specs=specs)
        report.imported.append(f"{path.name}: {len(notes)} ideas")
    return report
