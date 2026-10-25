"""The agents' funnel: does the loop that finds strategies by itself work, and
what does one survivor cost (owner, 2026-10-25: a multi-agent system that
autonomously finds profitable strategies; CLAUDE.md: «цена одного выжившего»).

Everything here is counted from the registry, nothing is estimated:

* **created** -- ideas an agent put into the registry, by who did it. The
  agents leave a mark in `idea.notes` (and the generator and the researcher in
  the id as well); an idea without a mark is the owner's, imported, or made in
  an interactive session, and is not counted:

  - generator    `gen-*`, notes "Generated from docs/MECHANISMS.md (...)"
                 (`qlab.agents.generator.run_generated_one`);
  - researcher   `*-research-*`, notes "Declared by the researcher (...)"
                 (`qlab.agents.researcher.run_experiments`);
  - implementer  notes "Implemented by the implementer agent from ..."
                 (`qlab.night._ensure_idea`);
  - scout        writes only DRAFT cards in `cards/drafts/`; a draft is not an
                 idea until the owner accepts it, so the scout is counted by
                 the drafts the nights recorded (`data/night/<day>.json`).
* **evaluated** -- the idea has a trial with a route: the stand actually ran it.
* **routes** -- where the idea's latest routed trial sent it.
* **survivors** -- ideas that reached the owner: status validated, paper or
  live, or the latest routed trial says `paper` (every rule passed, waiting for
  the owner's decision; the code never moves an idea into paper by itself,
  `qlab.registry.lifecycle`).
* **tokens** -- `token_spend` rows in the period, by stage and by agent, counted
  the way the budget guard counts them (`qlab.budget.guard._total`: input,
  output and both cache kinds). The guard's own window probes are overhead of
  the meter and are shown apart, not charged to the survivors.

The price of one survivor is the agents' tokens in the period divided by the
survivors among the ideas they created in it; with no survivors there is no
price, and the result says so (`None`) instead of dividing by zero.

Pure with respect to its inputs: a session, a period and a folder of night
state files. Nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from qlab.budget.guard import _total as tokens_of
from qlab.registry.models import Idea, IdeaStatus, Spec, TokenSpend, Trial, TrialRoute

AGENT_WORK_START = datetime(2026, 10, 3, tzinfo=UTC)
"""The day the agents started to put ideas into the registry (the researcher's
first experiments, 2026-10-03)."""

NIGHT_DIR = Path("data/night")

NOISE_PREFIX = "noise-"
"""Calibration ideas (random books the stand compares strategies with)."""

SURVIVOR_STATUSES = frozenset({IdeaStatus.VALIDATED, IdeaStatus.PAPER, IdeaStatus.LIVE})

IDEA_SOURCES = ("generator", "researcher", "implementer")
SOURCES = (*IDEA_SOURCES, "scout")

PROBE_AGENT = "probe"
"""The guard's own reading of the subscription windows: not agent work."""

NO_SURVIVORS = "выживших пока нет"


def idea_source(idea_id: str, notes: str | None) -> str | None:
    """Which agent created the idea, from the mark it left; None when no agent
    did (imported, the owner's, or made in an interactive session)."""
    notes = notes or ""
    if notes.startswith("Generated from") or idea_id.startswith("gen-"):
        return "generator"
    if notes.startswith("Declared by the researcher") or "-research-" in idea_id:
        return "researcher"
    if notes.startswith("Implemented by the implementer agent"):
        return "implementer"
    return None


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


@dataclass
class SourceFunnel:
    source: str
    created: int = 0
    evaluated: int = 0
    routes: dict[str, int] = field(default_factory=dict)
    survivors: int = 0
    survivor_ids: list[str] = field(default_factory=list)


@dataclass
class PeriodFunnel:
    label: str
    since: datetime
    until: datetime
    by_source: list[SourceFunnel]
    created: int
    evaluated: int
    routes: dict[str, int]
    survivors: int
    survivor_ids: list[str]
    scout_calls: int
    scout_drafts: int
    tokens_by_stage: dict[str, int]
    tokens_by_agent: dict[str, int]
    tokens_total: int
    probe_tokens: int
    tokens_per_survivor: int | None


@dataclass
class AgentFunnel:
    now: datetime
    periods: list[PeriodFunnel]

    def to_dict(self) -> dict:
        return asdict(self)


def scout_drafts(night_dir: Path, since: datetime, until: datetime) -> int:
    """Drafts the scout wrote on nights inside the period, from the nights'
    saved state (a draft is a card file, not a registry row)."""
    count = 0
    for path in sorted(night_dir.glob("*.json")):
        try:
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
        except ValueError:
            continue
        # A night is dated by the day it ran; the period counts it whole.
        if not (since - timedelta(days=1) < day <= until):
            continue
        try:
            scout = json.loads(path.read_text(encoding="utf-8")).get("scout") or {}
        except (OSError, ValueError):
            continue
        if scout.get("draft"):
            count += 1
    return count


def latest_routed_trials(session: Session, idea_ids: list[str]) -> dict[str, tuple[int, str]]:
    """Idea id -> (trial id, route) of its latest trial with a route."""
    if not idea_ids:
        return {}
    latest: dict[str, tuple[int, TrialRoute]] = {}
    rows = (session.query(Spec.idea_id, Trial.id, Trial.route)
            .join(Trial, Trial.spec_id == Spec.id)
            .filter(Trial.route.isnot(None), Spec.idea_id.in_(idea_ids)).all())
    for idea_id, trial_id, route in rows:
        if idea_id not in latest or trial_id > latest[idea_id][0]:
            latest[idea_id] = (trial_id, route)
    return {idea_id: (trial_id, TrialRoute(route).value)
            for idea_id, (trial_id, route) in latest.items()}


def _latest_routes(session: Session, idea_ids: list[str]) -> dict[str, str]:
    """Idea id -> the route of its latest routed trial."""
    return {idea_id: route for idea_id, (_, route) in
            latest_routed_trials(session, idea_ids).items()}


def period_funnel(session: Session, since: datetime, until: datetime, *, label: str,
                  night_dir: Path | None = None) -> PeriodFunnel:
    """The funnel of ideas the agents created in [since, until], and the
    tokens the agents spent in it."""
    since, until = _utc(since), _utc(until)
    night_dir = night_dir if night_dir is not None else NIGHT_DIR

    made: dict[str, Idea] = {}
    source_of: dict[str, str] = {}
    for idea in session.query(Idea).filter(~Idea.id.startswith(NOISE_PREFIX)).all():
        source = idea_source(idea.id, idea.notes)
        if source is not None and since <= _utc(idea.created_at) <= until:
            made[idea.id] = idea
            source_of[idea.id] = source
    routes = _latest_routes(session, list(made))

    by_source = {name: SourceFunnel(name) for name in SOURCES}
    for idea_id, idea in made.items():
        row = by_source[source_of[idea_id]]
        row.created += 1
        route = routes.get(idea_id)
        if route is not None:
            row.evaluated += 1
            row.routes[route] = row.routes.get(route, 0) + 1
        if idea.status in SURVIVOR_STATUSES or route == TrialRoute.PAPER.value:
            row.survivors += 1
            row.survivor_ids.append(idea_id)

    drafts = scout_drafts(night_dir, since, until)
    by_source["scout"].created = drafts

    spend = [r for r in session.query(TokenSpend).all() if since <= _utc(r.at) <= until]
    by_stage: dict[str, int] = {}
    by_agent: dict[str, int] = {}
    probe = 0
    for r in spend:
        if r.agent == PROBE_AGENT:
            probe += tokens_of(r)
            continue
        by_stage[r.stage] = by_stage.get(r.stage, 0) + tokens_of(r)
        by_agent[r.agent] = by_agent.get(r.agent, 0) + tokens_of(r)
    total = sum(by_agent.values())

    all_routes: dict[str, int] = {}
    for row in by_source.values():
        for route, n in row.routes.items():
            all_routes[route] = all_routes.get(route, 0) + n
    survivor_ids = sorted(i for row in by_source.values() for i in row.survivor_ids)
    return PeriodFunnel(
        label=label, since=since, until=until, by_source=list(by_source.values()),
        created=sum(by_source[name].created for name in IDEA_SOURCES),
        evaluated=sum(r.evaluated for r in by_source.values()), routes=all_routes,
        survivors=len(survivor_ids), survivor_ids=survivor_ids,
        scout_calls=sum(1 for r in spend if r.agent == "scout"), scout_drafts=drafts,
        tokens_by_stage=by_stage, tokens_by_agent=by_agent, tokens_total=total,
        probe_tokens=probe,
        tokens_per_survivor=round(total / len(survivor_ids)) if survivor_ids else None)


def agent_funnel(session: Session, now: datetime | None = None, *,
                 night_dir: Path | None = None) -> AgentFunnel:
    """The last seven days and everything since the agents started."""
    now = _utc(now or datetime.now(UTC))
    return AgentFunnel(now=now, periods=[
        period_funnel(session, now - timedelta(days=7), now, label="last-7-days",
                      night_dir=night_dir),
        period_funnel(session, AGENT_WORK_START, now, label="since-agents-started",
                      night_dir=night_dir),
    ])


# --------------------------------------------------------------------------
# The owner's words
# --------------------------------------------------------------------------

PERIOD_TITLE = {
    "last-7-days": "За последние 7 дней",
    "since-agents-started": "С начала работы агентов (с 03.10.2026)",
}

SOURCE_TITLE = {
    "generator": "генератор",
    "researcher": "исследователь",
    "implementer": "реализатор",
    "scout": "разведчик",
}

ROUTE_TITLE = {
    "paper": "прошла все правила, ждёт решения владельца",
    "shelf": "прошла, не хватает капитала",
    "needs-infrastructure": "прошла, нужен адаптер площадки",
    "needs-forward": "ждёт проверки на новых данных",
    "reject": "отвергнута",
    "not-evaluable": "проверить было нечем",
    "needs-more-data": "не хватило данных для метрики",
    "error": "прогон упал",
}

AGENT_TITLE = {**SOURCE_TITLE, "assessor": "второе мнение", "reviewer": "проверяющий",
               "director": "руководитель"}

STAGE_TITLE = {
    "recheck": "перепроверка",
    "implement": "реализация и прогон",
    "search": "поиск нового",
}


def _n(value: int) -> str:
    return f"{value:,}"


def price_text(tokens_per_survivor: int | None) -> str:
    return (NO_SURVIVORS if tokens_per_survivor is None
            else f"{_n(tokens_per_survivor)} токенов")


def report_lines(period: PeriodFunnel) -> list[str]:
    """One period as plain Russian lines for the morning report."""
    made = ", ".join(f"{SOURCE_TITLE[r.source]} {r.created}"
                     for r in period.by_source if r.source != "scout")
    routes = ", ".join(f"{ROUTE_TITLE.get(route, route)} — {n}" for route, n in
                       sorted(period.routes.items(), key=lambda kv: -kv[1])) or "ни одного исхода"
    agents = ", ".join(f"{AGENT_TITLE.get(a, a)} {_n(t)}" for a, t in
                       sorted(period.tokens_by_agent.items(), key=lambda kv: -kv[1]))
    survivors = (f"{period.survivors} ({', '.join(f'`{i}`' for i in period.survivor_ids)})"
                 if period.survivors else "0")
    return [
        f"**{PERIOD_TITLE.get(period.label, period.label)}**",
        f"- Агенты придумали идей: {period.created} ({made}); "
        f"разведчик написал черновиков карточек: {period.scout_drafts} "
        f"(вызовов {period.scout_calls})",
        f"- Прогнано на стенде: {period.evaluated} из "
        f"{period.created}; исходы: {routes}",
        f"- Дошли до владельца (прошли все правила и ждут его решения): {survivors}",
        f"- Потрачено токенов: {_n(period.tokens_total)}"
        + (f" ({agents})" if agents else "")
        + f"; цена одного выжившего — {price_text(period.tokens_per_survivor)}",
    ]


__all__ = ["AGENT_WORK_START", "NO_SURVIVORS", "AgentFunnel", "PeriodFunnel", "SourceFunnel",
           "agent_funnel", "idea_source", "latest_routed_trials", "period_funnel", "price_text",
           "report_lines", "scout_drafts"]
