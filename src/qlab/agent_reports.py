"""What each agent did night by night and what came of it, for the console's «Агенты» page.

Owner, 2026-10-27: «было бы здоров в qlab - видеть отчет по каждому агенту - что было сделано и
каков результат».

Read only: every row is copied from the night's saved state (`data/night/<day>.json`), which the
stages write as they go -- the same facts the morning report prints, grouped by agent instead of
by night. Nothing here judges: a verdict is the stand's, a status the code's."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qlab.pipeline.sources import QLAB_ROOT

STATE_DIR = QLAB_ROOT / "data" / "night"

ROUTE_RU = {"paper": "на бумагу", "shelf": "на полку: не хватает денег",
            "needs-infrastructure": "нужна инфраструктура", "needs-forward": "ждёт проверки вперёд",
            "needs-more-data": "не хватает данных", "reject": "отказ",
            "not-evaluable": "не оценить", "error": "сбой"}
_VIEW_RU = {"agree": "согласен", "too_harsh": "вердикт слишком строгий",
            "too_lenient": "вердикт слишком мягкий", "unsure": "не уверен"}
_PROGRAMMER_RU = {"queued": "в очереди", "ready": "готово, ждёт вашего слияния",
                  "failed": "не вышло", "blocked": "остановлено, решает владелец",
                  "merged": "слито", "rejected": "отклонено"}
_CATALOGUE_RU = {"queued": "в очереди", "catalogued": "новый механизм в каталоге",
                 "extended": "числа добавлены к механизму", "none": "механизма нет",
                 "failed": "черновик не принят кодом", "blocked": "остановлено проверяющим",
                 "no_text": "текст статьи недоступен", "rejected": "отклонено владельцем"}
_CARDER_RU = {"new": "ждёт карточника (попытка не принята)", "carded": "карточка написана",
              "duplicate": "повтор уже сделанного", "needs-owner": "вопрос к владельцу",
              "implemented": "реализатор взял карточку", "rejected": "закрыто владельцем"}
_ROUTE_IN_LINE = re.compile(r"\):\s*([a-z-]+)\s+—")


@dataclass
class Run:
    day: str
    status: str  # "ran" | "stopped" | "skipped" | "failed"
    did: str
    result: str
    details: list[str] = field(default_factory=list)
    tokens: int = 0


@dataclass
class Agent:
    key: str
    name: str
    model: str
    role: str
    runs: list[Run] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.runs)


def _status(entry: dict) -> str | None:
    if entry.get("stopped"):
        return "stopped"
    if entry.get("skipped"):
        return "skipped"
    if entry.get("error"):
        return "failed"
    return None


def _routes(routes: list[str]) -> str:
    counts: dict[str, int] = {}
    for r in routes:
        counts[r] = counts.get(r, 0) + 1
    return ", ".join(f"{ROUTE_RU.get(r, r)} — {n}" for r, n in counts.items())


def _director(day: str, st: dict) -> Run | None:
    d = st.get("director")
    if not d:
        return None
    if status := _status(d):
        return Run(day, status, "указания не составлены",
                   str(d.get("stopped") or d.get("skipped") or d.get("error")),
                   tokens=d.get("tokens", 0))
    directions = d.get("directions") or {}
    asked = [f"исследователю — {len(directions.get('researcher') or {})} стратегий",
             f"программисту — {len(directions.get('code_requests') or [])} запросов",
             f"в каталог — {len(directions.get('catalogue_requests') or [])} просьб"]
    pending = directions.get("pending_owner_review")
    if pending:  # the night saved it before the owner's review: today's file says more
        from qlab.directions import read_directions

        now = read_directions()
        if now.directions is not None and now.directions.date.isoformat() == directions.get(
                "date") and now.status == "active":
            pending = False
    return Run(day, "ran", f"записка {d.get('memo') or '—'}; " + "; ".join(asked),
               "указания ждут вашего подтверждения" if pending else "указания подтверждены и "
               "действуют",
               details=[*(d.get("problems") or []),
                        *(f"не принято: {r}" for r in d.get("rejected") or [])],
               tokens=d.get("tokens", 0))


def _researcher(day: str, st: dict) -> Run | None:
    entries = st.get("research")
    if not entries:
        return None
    lines, routes, details = [], [], []
    for e in entries:
        if e.get("stopped"):
            details.append(f"{e['strategy']}: остановлено — {e['stopped']}")
            continue
        lines.append(e["strategy"])
        for line in e.get("stand") or []:
            if line.startswith("- "):
                details.append(f"{e['strategy']}: {line[2:].replace('`', '')}")
                if m := _ROUTE_IN_LINE.search(line):
                    routes.append(m.group(1))
            elif line.startswith("  - "):  # the experiment's second stage, under its line
                details.append(f"{e['strategy']}: {line[4:].replace('`', '')}")
        details += [f"{e['strategy']}: не принято — {p}" for p in e.get("problems") or []]
        details += [f"{e['strategy']}: предлагает вывести {d}" for d in e.get("drops") or []]
    experiments = sum(len(e.get("specs") or []) for e in entries)
    return Run(day, "ran" if lines else "stopped",
               f"{len(lines)} стратегий ({', '.join(lines)}); экспериментов {experiments}",
               f"исходы экспериментов: {_routes(routes)}" if routes
               else "экспериментов на стенде нет",
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _assessor(day: str, st: dict) -> Run | None:
    entries = st.get("assessed")
    if not entries:
        return None
    disputes = [e["idea"] for e in entries if e.get("disputes")]
    details = [f"{e['idea']}: остановлено — {e['stopped']}" if e.get("stopped") else
               f"{e['idea']}: {_VIEW_RU.get(e.get('view'), e.get('view'))} — {e.get('summary', '')}"
               for e in entries]
    done = [e for e in entries if not e.get("stopped")]
    return Run(day, "ran" if done else "stopped", f"вторых мнений: {len(done)}",
               f"спорит с вердиктом: {', '.join(disputes)}" if disputes
               else "с вердиктами согласен" if done else "не дал мнения",
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _programmer(day: str, st: dict) -> Run | None:
    p = st.get("programmer")
    if p is None:
        return None
    entries = p.get("entries") or []
    if not entries:
        return Run(day, "stopped" if p.get("stopped") else "skipped", "запросов не брал",
                   p.get("stopped") or "очередь пуста")
    details = []
    for e in entries:
        details += [f"{e['id']}: {line.splitlines()[0]}" for line in e.get("problems") or []]
        details += [f"{e['id']}: проверяющий — {g}" for g in e.get("review") or []]
    return Run(day, "ran", "; ".join(f"{e['id']} ({e['repo']}): {e['what']}" for e in entries),
               "; ".join(_PROGRAMMER_RU.get(e["status"], e["status"]) for e in entries),
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _implementer(day: str, st: dict) -> Run | None:
    entries = st.get("implemented")
    if not entries:
        return None
    details = []
    for e in entries:
        details += [f"{e['card']}: {p}" for p in e.get("problems") or []]
        details += [f"{e['card']}: проверяющий — {r}" for r in e.get("review") or []]
    passed = [e["card"] for e in entries if not e.get("problems") and not e.get("review")]
    return Run(day, "ran", "карточки: " + ", ".join(e["card"] for e in entries),
               f"код написан и прошёл проверки: {len(passed)} из {len(entries)}",
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _scout(day: str, st: dict) -> Run | None:
    s, handoff = st.get("scout"), (st.get("publications") or {}).get("handoff") or {}
    if not s and not handoff:
        return None
    s = s or {}
    papers = handoff.get("papers") or []
    drafts = [x for x in [s.get("draft")] + list((handoff.get("drafts") or {}).values()) if x]
    details = [f"площадки: {s['reply']}"] if s.get("reply") else []
    details += [f"«{p['title']}»: {p['result']}" for p in papers]
    if s.get("stopped") or handoff.get("stopped"):
        details.append(f"остановлено: {s.get('stopped') or handoff.get('stopped')}")
    return Run(day, "stopped" if s.get("stopped") and not papers else "ran",
               f"события площадок; аннотаций статей: {len(papers)}",
               f"черновиков карточек: {len(drafts)}"
               + (f" ({', '.join(drafts)})" if drafts else ""),
               details=details, tokens=s.get("tokens", 0) + handoff.get("tokens", 0))


def _cataloguer(day: str, st: dict) -> Run | None:
    c = st.get("catalogue")
    if c is None:
        return None
    entries = c.get("entries") or []
    if not entries:
        return Run(day, "stopped" if c.get("stopped") else "skipped", "статей не брал",
                   c.get("stopped") or c.get("error") or "статей в очереди нет")
    details = []
    for e in entries:
        details += [f"{e['id']}: {p}" for p in e.get("problems") or []]
        details += [f"{e['id']}: проверяющий — {g}" for g in e.get("review") or []]
        if e.get("reason"):
            details.append(f"{e['id']}: {e['reason']}")
        if e.get("needs_block"):
            details.append(f"{e['id']}: нужен новый блок — {e['needs_block']}")
    return Run(day, "ran", "; ".join(f"«{e.get('title') or e['id']}»" for e in entries),
               "; ".join(_CATALOGUE_RU.get(e["status"], e["status"])
                         + (f" `{e['mechanism']}`" if e.get("mechanism") else "")
                         for e in entries),
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _carder(day: str, st: dict) -> Run | None:
    c = st.get("cards")
    if c is None:
        return None
    entries = c.get("entries") or []
    if not entries:
        if c.get("waiting_for"):
            return Run(day, "skipped", "идей не брал",
                       f"карточка {c['waiting_for']} ещё не взята реализатором")
        return Run(day, "stopped" if c.get("stopped") else "skipped", "идей не брал",
                   c.get("stopped") or c.get("error") or "новых идей во входящих нет")
    details = []
    for e in entries:
        details += [f"{e['id']}: {p}" for p in e.get("problems") or []]
        if e.get("duplicate_of"):
            details.append(f"{e['id']}: повтор — {e['duplicate_of']}")
        if e.get("question"):
            details.append(f"{e['id']}: вопрос владельцу — {e['question']}")
        if e.get("card"):
            details.append(f"{e['id']}: карточка {e['card']}, новый блок {e.get('new_block')}")
    return Run(day, "ran", "; ".join(f"{e['id']}: {' '.join(e['text'].split())[:80]}"
                                     for e in entries),
               "; ".join(_CARDER_RU.get(e.get("status"), str(e.get("status"))) for e in entries),
               details=details, tokens=sum(e.get("tokens", 0) for e in entries))


def _generator(day: str, st: dict) -> Run | None:
    g = st.get("generate")
    if g is None:
        return None
    if g.get("skipped") or (g.get("stopped") and not g.get("accepted")):
        return Run(day, _status(g) or "skipped", "идей не придумывал",
                   str(g.get("skipped") or g.get("stopped")), tokens=g.get("tokens", 0))
    accepted, rejected = g.get("accepted") or [], g.get("rejected") or []
    results = g.get("results") or []
    details = [f"{r['idea_id']} ({r['mechanism']}): "
               + (f"сбой — {r['error']}" if r.get("error") else
                  f"{ROUTE_RU.get(r.get('route'), r.get('route'))} — {r.get('reason') or ''}")
               for r in results]
    details += [f"{Path(r['path']).stem}: отклонён кодом — {'; '.join(r['problems'])[:300]}"
                for r in rejected]
    return Run(day, "ran", f"спеков принято кодом: {len(accepted)}, отклонено: {len(rejected)}",
               f"исходы на стенде: {_routes([r.get('route') or 'error' for r in results])}"
               if results else "на стенд ничего не попало",
               details=details, tokens=g.get("tokens", 0))


AGENTS: tuple[tuple[str, str, str, str, Callable[[str, dict], Run | None]], ...] = (
    ("director", "Руководитель исследований", "Opus",
     "Раз в две ночи смотрит на направления поиска: что продолжать, что приостановить, что "
     "поручить исследователю, генератору и программисту. Указания действуют после вашего "
     "подтверждения.", _director),
    ("researcher", "Исследователь", "Opus",
     "Каждую ночь разбирает просадки живых стратегий на данных до отреза и ставит эксперименты; "
     "исход каждого эксперимента выносит стенд.", _researcher),
    ("assessor", "Оценщик", "Opus",
     "Второе мнение на свежие вердикты стенда и на стратегии с деньгами; вердикт не меняет, "
     "спор решает владелец.", _assessor),
    ("programmer", "Программист", "Sonnet",
     "Пишет код, которого не хватает для проверки направления, в отдельной ветке; в основную "
     "ветку — только по вашей команде.", _programmer),
    ("implementer", "Реализатор", "Sonnet",
     "Пишет код стратегии по карточке, когда её нельзя собрать из блоков.", _implementer),
    ("scout", "Разведчик", "Sonnet",
     "Читает события площадок и аннотации новых статей и набрасывает черновики карточек.",
     _scout),
    ("cataloguer", "Разборщик статей", "Sonnet",
     "Читает статью целиком и записывает механизм в каталог; каждое число код сверяет с текстом "
     "статьи.", _cataloguer),
    ("carder", "Карточник", "Sonnet",
     "Берёт идею из входящих, для которой нужен новый код, и пишет карточку для реализатора; "
     "либо отвечает «повтор» (называя, что уже пробовали), либо задаёт вопрос владельцу.",
     _carder),
    ("generator", "Генератор идей", "Sonnet",
     "Собирает новые стратегии из механизмов каталога и своих чисел; каждую проверяет стенд.",
     _generator),
)


def _by_hand(key: str, seen: set[str]) -> list[Run]:
    """Attempts made by hand (`qlab code run`, `qlab catalogue run`): they are in the agent's
    own ledger, not in a night's state."""
    runs = []
    if key == "programmer":
        from qlab.agents.programmer import load_queue

        for r in load_queue():
            if r.attempts and r.id not in seen:
                runs.append(_programmer(r.created[:10], {"programmer": {"entries": [
                    asdict(r)]}}))
    elif key == "cataloguer":
        from qlab.agents.cataloguer import load_ledger

        for p in load_ledger():
            if p.attempts and p.id not in seen and p.day:
                runs.append(_cataloguer(p.day, {"catalogue": {"entries": [asdict(p)]}}))
    for run in runs:
        run.did = f"вручную: {run.did}"
    return runs


def _ids(states: list[tuple[str, dict]], key: str) -> set[str]:
    stage = {"programmer": "programmer", "cataloguer": "catalogue"}.get(key)
    if stage is None:
        return set()
    return {e["id"] for _, st in states for e in (st.get(stage) or {}).get("entries") or []}


def agent_reports(days: int = 14, state_dir: Path = STATE_DIR) -> dict:
    """{"nights": [...], "agents": [...]}: the last `days` nights, newest first, each agent's run
    on each night it ran, and the attempts made by hand in between."""
    files = sorted(state_dir.glob("????-??-??.json"), reverse=True)[:days]
    states = []
    for f in files:
        try:
            states.append((f.stem, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    agents = []
    for key, name, model, role, read in AGENTS:
        agent = Agent(key, name, model, role)
        for day, st in states:
            try:
                run = read(day, st)
            except Exception as exc:  # noqa: BLE001 - a state of an older format: say so
                run = Run(day, "failed", "запись ночи не прочитана", f"{type(exc).__name__}: {exc}")
            if run is not None:
                agent.runs.append(run)
        try:
            agent.runs += _by_hand(key, _ids(states, key))
        except Exception as exc:  # noqa: BLE001 - the ledger is unreadable: say so
            agent.runs.append(Run("—", "failed", "журнал агента не прочитан",
                                  f"{type(exc).__name__}: {exc}"))
        agent.runs.sort(key=lambda r: r.day, reverse=True)
        agents.append({**asdict(agent), "tokens": agent.tokens})
    return {"nights": [day for day, _ in states], "agents": agents}


__all__ = ["AGENTS", "agent_reports"]
