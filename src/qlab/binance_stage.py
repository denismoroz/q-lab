"""The Binance stage of every idea born on Hyperliquid: the plan and the hand run
(docs/VENUE_SPLIT.md, docs/TWO_STAGE.md).

Owner, 2026-11-05, on trend-research-flat-choice-hl-ov-gate-enter: «почему вот эта стратегия не
тестировалась на бинансе?», then «да, делай и прогоняй все». The second stage used to go one way
only (Binance -> Hyperliquid); an idea whose own spec is on Hyperliquid's data now gets a Binance
stage too (`qlab.pipeline.stage2` with `night/stage2_binance.yaml`, child `<idea>-bn`). The night
gives it to open ideas (`qlab.waiting.binance_stage_plan`) and to closed ones
(`qlab.second_venue`); this module lists every such idea with what its Binance stage would be --
the coins kept and dropped, the window, whether its data is heavy for a night, about how long it
takes -- and runs the ones named by hand. Code only: no agent, no tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

BINANCE = "binance"
HYPERLIQUID = "hyperliquid"


@dataclass
class PlanRow:
    """One idea born on Hyperliquid and its Binance stage."""

    idea_id: str
    status: str
    closed: bool
    spec_path: str | None = None
    owed: bool = False
    """Whether the Binance stage is owed and can be made (the night or a hand run makes it)."""
    why_not: str | None = None
    """Why the idea gets no Binance stage (or why it cannot be made yet)."""
    child_route: str | None = None
    """The route of the Binance stage's latest run, when it has run."""
    start: str | None = None
    end: str | None = None
    interval: str | None = None
    kept: list[str] | None = None
    """The instruments kept on Binance (None: the strategy picks from Binance's whole listing)."""
    dropped: list[str] = field(default_factory=list)
    dropped_candidates: list[str] = field(default_factory=list)
    fetch_minutes: float | None = None
    fetch_why: str | None = None
    heavy: str | None = None
    compute_minutes: float | None = None
    """About how long the run computes: the idea's own latest Hyperliquid run (`cpu_seconds`, else
    its wall time) scaled by the windows' lengths in days; a rough guide, not a promise."""

    @property
    def minutes(self) -> float | None:
        parts = [m for m in (self.fetch_minutes, self.compute_minutes) if m is not None]
        return sum(parts) if parts else None


def born_on_hyperliquid(session) -> list:
    """The ideas whose own latest routed run is on Hyperliquid's data: not retired, not noise, not
    a second stage. (Ideas on Hyperliquid's equity markets, `hyperliquid-xyz`, are not here:
    Binance lists no such instrument.)"""
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.registry.stages import is_stage2_child, latest_routed

    out = []
    for idea in session.query(Idea).order_by(Idea.id):
        if idea.status == IdeaStatus.RETIRED or idea.id.startswith("noise-") \
                or is_stage2_child(idea):
            continue
        routed = latest_routed(session, idea.id)
        if routed is None or (routed[1].data_requirements or {}).get("source") != HYPERLIQUID:
            continue
        out.append((idea, routed))
    return out


def plan(session, *, end: date, specs: dict[str, list[Path]] | None = None,
         names: list[str] | None = None, cost=None, limit_minutes: float | None = None
         ) -> list[PlanRow]:
    """Every idea born on Hyperliquid with what its Binance stage would be. Reads only (but for
    Binance's listing, kept a month, when an idea names coins and `names` is not given)."""
    from qlab import hl_fetch, night, waiting
    from qlab.pipeline import stage2
    from qlab.pipeline.spec import load_spec
    from qlab.pipeline.venue_status import binance_stage_possible
    from qlab.registry.lifecycle import CLOSED_STATUSES, PRODUCTION_STATUSES
    from qlab.registry.stages import binance_child, child_id_taken, latest_routed

    config = stage2.config_for(BINANCE)
    specs = specs if specs is not None else waiting.spec_files()
    rows: list[PlanRow] = []
    for idea, routed in born_on_hyperliquid(session):
        row = PlanRow(idea.id, idea.status.value, idea.status in CLOSED_STATUSES)
        rows.append(row)
        child = binance_child(session, idea.id)
        child_run = None if child is None else latest_routed(session, child.id)
        if child_run is not None:
            row.child_route = str(getattr(child_run[0].route, "value", child_run[0].route))
        path, skip = waiting._pick_spec(specs, idea.id)
        row.spec_path = None if path is None else path.as_posix()
        if idea.status in PRODUCTION_STATUSES:
            row.why_not = ("книга была в деньгах и выключена (decayed): её проверкой была "
                           "настоящая торговля" if idea.status.value == "decayed" else
                           "книга на бумаге или в деньгах: её проверка — настоящая торговля "
                           "(docs/ONE_CHECK.md)")
            continue
        if path is None:
            row.why_not = skip or "нет файла спека"
            continue
        if child_id_taken(session, idea.id, BINANCE):
            row.why_not = f"имя {idea.id}-bn занято другой идеей"
            continue
        try:
            spec = night.with_end(load_spec(path), end)
        except Exception as exc:  # noqa: BLE001 - a broken file is its own reason
            row.why_not = f"спек не читается: {type(exc).__name__}: {exc}"
            continue
        if stage2.second_venue_of(spec) != BINANCE:
            row.why_not = "проверка идеи и так идёт на истории Binance (docs/ONE_CHECK.md)"
            continue
        if not binance_stage_possible(session, idea.id, routed[1].code_ref):
            only_on = stage2.strategy_sources(spec.code_ref)
            row.why_not = ("у стратегии нет версии на данных Binance (она объявляет только "
                           f"{', '.join(only_on)})" if only_on is not None and BINANCE
                           not in only_on else "этап на Binance сделать нельзя (выключен или "
                           "сбои признаны постоянными)")
            continue
        listed = names
        if listed is None and (spec.data.instruments is not None
                               or spec.required_instruments is not None):
            listed = names = stage2.venue_listing(BINANCE)
        translation = stage2.translate(spec, config, venue_names=listed)
        row.dropped = list(translation.dropped)
        row.dropped_candidates = list(translation.dropped_candidates)
        if translation.spec is None or translation.unavailable:
            row.why_not = "Binance не может выразить спек: " + "; ".join(
                [*translation.problems, *translation.unavailable])
            continue
        child_spec = translation.spec
        row.owed = row.child_route is None
        row.start, row.end = child_spec.data.start.isoformat(), child_spec.data.end.isoformat()
        row.interval = child_spec.data.interval
        row.kept = None if child_spec.data.instruments is None else list(
            child_spec.data.instruments)
        if cost is not None:
            try:
                est = hl_fetch.estimate(child_spec, session, cost)
                row.fetch_minutes, row.fetch_why = est.minutes, est.why
                if limit_minutes is not None:
                    row.heavy = hl_fetch.heavy_reason(est, limit_minutes)
            except Exception as exc:  # noqa: BLE001 - an estimate is advice
                row.fetch_why = f"оценка не сделана: {type(exc).__name__}"
        trial = routed[0]
        seconds = trial.cpu_seconds
        if not seconds and trial.started_at is not None and trial.finished_at is not None:
            seconds = (trial.finished_at - trial.started_at).total_seconds()
        own_days = (spec.data.end - spec.data.start).days or 1
        if isinstance(seconds, int | float) and seconds > 0:
            row.compute_minutes = seconds / 60 * (
                (child_spec.data.end - child_spec.data.start).days / own_days)
    return rows


def plan_lines(rows: list[PlanRow]) -> list[str]:
    """The plan in plain lines (the hand command's dry run)."""
    out = []
    for r in rows:
        state = "закрыта" if r.closed else "открыта"
        head = f"{r.idea_id} ({r.status}, {state})"
        if r.child_route is not None:
            out.append(f"{head}: этап на Binance уже есть, исход {r.child_route}")
            continue
        if r.why_not:
            out.append(f"{head}: не делается — {r.why_not}")
            continue
        coins = ("монеты выбирает сама стратегия из всего списка Binance" if r.kept is None
                 else f"монет {len(r.kept)}: {', '.join(r.kept)}")
        dropped = (f"; на Binance нет и не торгуются: {', '.join(r.dropped)}" if r.dropped else "")
        if r.dropped_candidates:
            dropped += f"; из выбора убраны: {', '.join(r.dropped_candidates)}"
        minutes = "" if r.minutes is None else f"; около {r.minutes:.0f} мин"
        heavy = f"; ТЯЖЕЛО ДЛЯ НОЧИ: {r.heavy}" if r.heavy else ""
        out.append(f"{head}: {r.interval} с {r.start} по {r.end}; {coins}{dropped}{minutes}"
                   f"{heavy}" + (f" ({r.fetch_why})" if r.fetch_why else ""))
    return out


def run(idea_ids: list[str], *, end: date, noise_trials: int = 200,
        allow_heavy: bool = False, day: date | None = None) -> list[dict]:
    """Run the Binance stage of the named ideas born on Hyperliquid. An open idea is settled by
    the two venues as any second stage settles it; a closed one goes through the queue of closed
    ideas (`qlab.second_venue`), so only a pass revives it. Returns one result per idea."""
    from qlab import night, second_venue, waiting
    from qlab.pipeline import stage2
    from qlab.registry.db import session_scope
    from qlab.registry.lifecycle import CLOSED_STATUSES
    from qlab.registry.models import Idea

    config = stage2.config_for(BINANCE)
    specs = waiting.spec_files()
    results: list[dict] = []
    closed: list[str] = []
    for idea_id in idea_ids:
        with session_scope() as session:
            idea = session.get(Idea, idea_id)
            status = None if idea is None else idea.status
        if status is None:
            results.append({"idea": idea_id, "action": "error", "error": "no such idea"})
            continue
        if status in CLOSED_STATUSES:
            closed.append(idea_id)
            continue
        path, skip = waiting._pick_spec(specs, idea_id)
        if path is None:
            results.append({"idea": idea_id, "action": "skip", "why": skip})
            continue
        plan_ = waiting.Plan(idea_id, waiting.BINANCE_STEP, path.as_posix(), None, True,
                             "вручную: этап на Binance для идеи, родившейся на Hyperliquid")
        try:
            results.append(waiting.rerun_stage2(plan_, end, stage1=None,
                                                noise_trials=noise_trials, config=config))
        except Exception as exc:  # noqa: BLE001 - one idea's failure must not stop the others
            results.append({"idea": idea_id, "action": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
    if closed:
        state: dict = {"done": {}}
        night.STATE_DIR = Path("data/restamp/runs")
        second_venue.second_venue_stage(day or date.today(), end, state,
                                        noise_trials=noise_trials, only=closed,
                                        allow_heavy=allow_heavy)
        block = state.get("second_venue") or {}
        seen = set()
        for e in block.get("entries") or []:
            seen.add(e.get("idea"))
            results.append({"idea": e.get("idea"), "action": e.get("action"),
                            "route_after": e.get("route"), "trial": e.get("trial"),
                            "status": e.get("status"), "error": e.get("error"),
                            "why": e.get("why"), "lines": e.get("lines") or [],
                            "outcome": e.get("outcome")})
        refused = block.get("refused") or {}
        for idea_id in closed:
            if idea_id not in seen:
                results.append({"idea": idea_id, "action": "skip",
                                "why": refused.get(idea_id) or "очередь закрытых идей её не "
                                "взяла (уже смотрели на Binance, или отказ в журнале)"})
    return results


__all__ = ["PlanRow", "born_on_hyperliquid", "plan", "plan_lines", "run"]
