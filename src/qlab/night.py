"""The nightly run (docs/NIGHT.md).

Owner, 2026-10-03: «делай все 3 одну за одной» -- first of the three, the
nightly loop that spends no tokens. CLAUDE.md fixes the order of cutting,
cheap and mandatory first:

1. **recheck** -- strategies on the watch list (`night/watchlist.yaml`)
   re-evaluated on data extended to the last closed day, so their forward
   tests grow by a day each night; the graveyard swept by market regime once
   a week (owner, 2026-10-03: «если мы кладбище прогоним по режимам — найдем
   много нового»); the production paper books reconciled with the stand;
2. **implement** -- candidates written by the implementer agent (later);
3. **search** -- new candidates (later).

Each night writes a morning report (`reports/night/<date>.md`). The run is
idempotent: a night's state (`data/night/<date>.json`) lists what is done, so
a run that stops -- a crash, a budget stop -- continues where it stopped.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

from qlab.pipeline.spec import StrategySpec, load_spec

WATCHLIST = Path("night/watchlist.yaml")
IMPLEMENT_QUEUE = Path("night/implement_queue.yaml")
STATE_DIR = Path("data/night")
REPORT_DIR = Path("reports/night")
PAPER_DIR = Path("data/paper")
PROD = "dis@10.8.0.5"
PROD_DB = "file:data/frab.db?mode=ro"  # read-only: q-lab never changes frab's state

KEY_METRICS = ("ann_return_net", "sharpe_net", "max_dd", "noise_return_percentile",
               "forward_days", "forward_days_needed", "fit_window_days", "noise_power",
               "fit_ann_return_net", "fit_sharpe_net", "fit_max_dd", "selection_days")


@dataclass
class ItemResult:
    kind: str
    name: str
    trial_id: int | None = None
    route: str | None = None
    previous_route: str | None = None
    reason: str | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None


def last_closed_day(now: datetime | None = None) -> date:
    """The last UTC day whose daily candle has closed."""
    now = now or datetime.now(UTC)
    return (now - timedelta(days=1)).date()


def with_end(spec: StrategySpec, end: date) -> StrategySpec:
    """The spec with its data extended to `end` (never shortened)."""
    if end <= spec.data.end:
        return spec
    return spec.model_copy(update={"data": spec.data.model_copy(update={"end": end})})


def _state_path(day: date) -> Path:
    return STATE_DIR / f"{day.isoformat()}.json"


def load_state(day: date) -> dict:
    path = _state_path(day)
    return json.loads(path.read_text()) if path.is_file() else {"done": {}}


def save_state(day: date, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(day).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(_state_path(day))


def previous_route(session, idea_id: str, before: int) -> str | None:
    from qlab.registry.models import Spec, Trial

    row = (session.query(Trial.route).join(Spec, Spec.id == Trial.spec_id)
           .filter(Spec.idea_id == idea_id, Trial.route.is_not(None), Trial.id < before)
           .order_by(Trial.id.desc()).first())
    return getattr(row[0], "value", row[0]) if row else None


def evaluate_item(kind: str, spec: StrategySpec, *, end: date, noise_trials: int,
                  deployable_capital_usd: float) -> ItemResult:
    """One evaluation in its own transaction, so one failure cannot roll back
    the night's other results."""
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    result = ItemResult(kind=kind, name=spec.idea_id)
    try:
        with session_scope() as session:
            out = evaluate_spec_with_shape_aware_bar(
                with_end(spec, end), session=session, ruleset=load_latest(),
                deployable_capital_usd=deployable_capital_usd, n_trials=noise_trials,
            ).candidate
            result.trial_id = out.trial_id
            result.route = out.routing.route
            result.reason = out.routing.reason
            result.previous_route = previous_route(session, spec.idea_id, out.trial_id)
            metrics = out.metrics or {}
            result.metrics = {k: metrics[k] for k in KEY_METRICS if k in metrics}
            result.metrics.update({k: v for k, v in metrics.items()
                                   if k.startswith(("regime_", "fit_regime_"))
                                   and k.endswith(("_return", "_noise_percentile"))})
    except Exception as exc:  # noqa: BLE001 - recorded in the report, the night goes on
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def unwatched_bench(watch: list[str], spec_dir: Path = Path("specs")) -> list[str]:
    """Bench ideas no spec on the watch list belongs to. Their forward tests
    never grow, so "waiting for a forward test" would be for ever: on
    2026-10-10 seventeen of twenty-eight bench ideas were finished
    experiments and controls nobody followed. The report names them so the
    owner either follows or retires each (`qlab retire`)."""
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, IdeaStatus

    watched = set()
    for path in watch:
        try:
            watched.add(load_spec(Path(path)).idea_id)
        except Exception:  # noqa: BLE001 - an unloadable spec fails loudly in its own run
            continue
    with session_scope() as session:
        bench = [i.id for i in session.query(Idea).filter(Idea.status == IdeaStatus.BENCH)]
    return sorted(i for i in bench if i not in watched)


def graveyard_specs(spec_dir: Path = Path("specs")) -> list[Path]:
    """Spec files of ideas whose status is rejected -- what the weekly
    regime sweep re-runs."""
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, IdeaStatus

    with session_scope() as session:
        rejected = {i.id for i in session.query(Idea).filter(Idea.status == IdeaStatus.REJECTED)}
    out = []
    for path in sorted(spec_dir.glob("*.yaml")):
        try:
            if load_spec(path).idea_id in rejected:
                out.append(path)
        except Exception:  # noqa: BLE001 - an unloadable spec is not the night's business
            continue
    return out


def _ssh_query(sql: str, out: Path, header: bool = True) -> None:
    flags = "-header -csv" if header else ""
    cmd = (f"cd ~/prj/funding-rate-arbitrage && sqlite3 {flags} \"{PROD_DB}\" \"{sql}\"")
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", PROD, cmd],
                          capture_output=True, text=True, check=True, timeout=300)
    out.write_text(done.stdout)


def paper_reconciliation(day: date) -> list[str]:
    """Export the paper books read-only from the production host and replay
    them through frab's live code on q-lab's own data (scripts/replay_*);
    the lines that compare the two go into the report."""
    folder = PAPER_DIR / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        _ssh_query("select strategy_id, ts_ms, coin, price, equity, book_equity, cash, "
                   "spot_value, short_pnl, hedge_on from b2_equity "
                   "order by strategy_id, coin, ts_ms", folder / "b2_equity.csv")
        _ssh_query("select params_json from strategies where id=5", folder / "trend_params.json",
                   header=False)
        _ssh_query("select * from trend_equity where strategy_id=5 order by ts_ms",
                   folder / "trend_equity.csv")
        _ssh_query("select * from trend_events where strategy_id=5 order by ts_ms",
                   folder / "trend_events.csv")
    except (subprocess.SubprocessError, OSError) as exc:
        return [f"export from the production host failed: {exc}"]
    runs = {
        "Bv2 (book 3 = b2, sized as created with carry on; book 4 = b2_cold)": [
            "scripts/replay_bv2_paper.py", "--paper", str(folder / "b2_equity.csv"),
            "--b2-sized-with-carry"],
        "trend": ["scripts/replay_trend_paper.py", "--paper", str(folder / "trend_equity.csv"),
                  "--params", str(folder / "trend_params.json"),
                  "--events", str(folder / "trend_events.csv")],
    }
    for label, args in runs.items():
        done = subprocess.run(["uv", "run", "python", *args], capture_output=True, text=True,
                              timeout=1800)
        keep = [ln for ln in done.stdout.splitlines()
                if any(w in ln for w in ("book ", "correlation", "paper", "replay", "fills matched",
                                         "matched hours", "max |"))]
        status = "" if done.returncode == 0 else f" (exit {done.returncode})"
        lines.append(f"**{label}**{status}")
        lines += [f"    {ln}" for ln in keep] or [f"    {done.stderr.strip()[-300:]}"]
    return lines


def _pct(v) -> str:
    return f"{v:+.1%}" if isinstance(v, int | float) and v == v else "—"


def _report_row(m: dict) -> tuple[str, str, str, str, str, str, str]:
    """Selection period and forward test kept apart (docs/FIT_VS_FORWARD.md).

    With a forward test, the run's own `ann_return_net` is the forward
    test's, and nineteen days annualized read as +400% a year: the forward
    test is shown as its total over its days instead, and the yearly
    figures and the regime split are the selection period's. Without one,
    the whole window is the selection period."""
    split = "fit_ann_return_net" in m
    prefix = "fit_" if split else ""
    days = m.get("forward_days") or 0
    forward_total = "—"
    if days > 0 and isinstance(m.get("ann_return_net"), int | float):
        forward_total = _pct((1 + m["ann_return_net"]) ** (days / 365.25) - 1)
    if days > 0 and not split:
        # The whole judged window is the forward test (a causal selection, or
        # parameters fixed before the data began): no selection period to show.
        return ("—", "—", f"{days:.0f}", forward_total,
                *(_pct(m.get(f"regime_{r}_return")) for r in ("bull", "flat", "bear")))
    sharpe = m.get(f"{prefix}sharpe_net")
    return (_pct(m.get(f"{prefix}ann_return_net")),
            f"{sharpe:.2f}" if isinstance(sharpe, int | float) else "—",
            f"{days:.0f}", forward_total,
            *(_pct(m.get(f"{prefix}regime_{r}_return")) for r in ("bull", "flat", "bear")))


def write_report(day: date, state: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Ночной прогон {day.isoformat()}", ""]
    items = [ItemResult(**r) for r in state.get("results", [])]
    for kind, title in (("watch", "Стратегии на проверке вперёд"),
                        ("graveyard", "Кладбище по режимам")):
        group = [r for r in items if r.kind == kind]
        if not group:
            continue
        lines += [f"## {title}", "",
                  "Годовые числа и режимы — по периоду подбора; проверка вперёд — итогом за её "
                  "дни, без пересчёта в год.", "",
                  "| стратегия | исход | было | подбор: в год | Шарп | вперёд, дней | "
                  "вперёд, итого | рост | боковик | падение |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in group:
            if r.error:
                lines.append(f"| {r.name} | ошибка: {r.error[:120]} | | | | | | | | |")
                continue
            m = r.metrics
            changed = " **(изменился)**" if r.previous_route and r.previous_route != r.route else ""
            lines.append(f"| {r.name} | {r.route}{changed} | {r.previous_route or '—'} | "
                         + " | ".join(_report_row(m)) + " |")
        lines.append("")
    if idle := state.get("unwatched_bench"):
        lines += ["## На скамейке, но не в списке наблюдения", "",
                  "Проверка вперёд у этих идей не растёт: либо добавить в `night/watchlist.yaml`, "
                  "либо вывести (`qlab retire`).", "", *[f"- {i}" for i in idle], ""]
    if built := state.get("detectors"):
        lines += ["## Определители режима (пересобраны до списка наблюдения)", ""]
        lines += [f"- {'ок' if b['ok'] else '**сбой**'} — {b['name']}: {b['note']}" for b in built]
        lines.append("")
    if kind_lines := state.get("paper"):
        lines += ["## Сверка бумаги на проде со стендом", "", *kind_lines, ""]
    if studied := state.get("research"):
        lines += ["## Исследователь (раз в неделю)", ""]
        for e in studied:
            if e.get("stopped"):
                lines.append(f"- {e['strategy']}: остановлено — {e['stopped']}")
                continue
            lines.append(f"- **{e['strategy']}**: записка `{e.get('memo')}`, экспериментов "
                         f"{len(e.get('specs') or [])}, {e.get('tokens', 0):,} токенов")
            lines += [f"    - не принято: {p}" for p in e.get("problems") or []]
            lines += [f"    - {line}" for line in e.get("stand") or [] if line.startswith("- ")]
        lines.append("")
    if assessed := state.get("assessed"):
        views = {"agree": "согласен", "too_harsh": "вердикт слишком строгий",
                 "too_lenient": "вердикт слишком мягкий", "unsure": "не уверен"}
        lines += ["## Второе мнение агента", "",
                  "Вердикты не меняются; где агент спорит — решать владельцу.", ""]
        for e in assessed:
            if e.get("stopped"):
                lines.append(f"- {e['idea']} (прогон {e['trial']}): остановлено — {e['stopped']}")
                continue
            flag = " — **спорит с вердиктом**" if e.get("disputes") else ""
            lines.append(f"- **{e['idea']}** (прогон {e['trial']}): "
                         f"{views.get(e['view'], e['view'])}{flag}; {e.get('tokens', 0):,} "
                         f"токенов. {e.get('summary', '')}")
        lines.append("")
    if implemented := state.get("implemented"):
        lines += ["## Реализация карточек агентом", ""]
        for e in implemented:
            lines.append(f"- **{e['card']}** → {e.get('idea_id', '?')}: "
                         f"{e.get('tokens', 0):,} токенов; ответ агента: {e.get('reply') or '—'}")
            for p in e.get("problems") or []:
                lines.append(f"    - проверка не пройдена: {p}")
            for r in e.get("review") or []:
                lines.append(f"    - проверяющий: {r}")
            if e.get("stand"):
                lines.append(f"    - стенд: {e['stand']}")
            if e.get("stopped"):
                lines.append(f"    - остановлено: {e['stopped']}")
        lines.append("")
    if gen := state.get("generate"):
        from qlab.agents.generator import GeneratedResult, result_line

        lines += ["## Генератор идей (раз в неделю)", "",
                  "Агент берёт опубликованные механизмы из `docs/MECHANISMS.md` и пишет описания "
                  "новых стратегий из готовых блоков; код проверяет каждое число по каталогу, "
                  "стенд прогоняет прошедшие проверку. Вердикты выносит код.", ""]
        if gen.get("skipped"):
            lines.append(f"- не запускался — {gen['skipped']}")
        if gen.get("stopped"):
            lines.append(f"- остановлено: {gen['stopped']}; продолжит в следующую ночь")
        if "accepted" in gen:
            lines.append(f"- принято стратегий: {len(gen['accepted'])}, не принято: "
                         f"{len(gen.get('rejected') or [])}; {gen.get('tokens', 0):,} токенов; "
                         f"ответ агента: {gen.get('reply') or '—'}")
        lines += [f"    - {p}" for p in gen.get("problems") or []]
        for r in gen.get("rejected") or []:
            lines += [f"    - не принято `{r['path']}`: {p}" for p in r["problems"]]
        for r in gen.get("results") or []:
            lines.append(f"- {result_line(GeneratedResult(**r))}")
        lines.append("")
    if (events := state.get("events")) is not None:
        lines += ["## Площадки", ""]
        structural = [e for e in events if e["kind"] in
                      ("listed", "delisted", "relisted", "removed", "changed", "status")]
        if state.get("events_error"):
            lines.append(f"Наблюдатель упал: {state['events_error']}")
        lines.append(f"Событий: {len(events)}, из них листинги, снятия и изменения условий — "
                     f"{len(structural)}.")
        lines += [f"- {e['source']}: {e['kind']} {e['instrument']} {e['detail']}"
                  for e in structural]
        extremes = [e for e in events if e["kind"] == "funding-extreme"]
        if extremes:
            lines.append("- крайности фандинга: " + "; ".join(
                f"{e['instrument']} {e['detail'].split(' a year')[0]}" for e in extremes[:6]))
        lines.append("")
    if sc := state.get("scout"):
        lines += ["## Разведчик", "", f"{sc.get('reply') or sc.get('stopped') or '—'}"
                  + (f" ({sc.get('tokens', 0):,} токенов)" if sc.get("tokens") else "")]
        lines += [f"- проверка не пройдена: {p}" for p in sc.get("problems") or []]
        lines.append("")
    lines += ["## Токены", "", _token_lines(day), ""]
    path = REPORT_DIR / f"{day.isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _token_lines(day: date) -> str:
    from qlab.registry.db import session_scope
    from qlab.registry.models import TokenSpend

    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    with session_scope() as session:
        rows = session.query(TokenSpend).filter(TokenSpend.at >= start).all()
        by: dict[str, int] = {}
        for r in rows:
            by[r.stage] = by.get(r.stage, 0) + (r.tokens_in or 0) + (r.tokens_out or 0) + (
                r.tokens_cache_write or 0) + (r.tokens_cache_read or 0)
        last = max(rows, key=lambda r: r.at, default=None)
    if not by:
        return "Токены за эту ночь не тратились."
    week = (f"; недельное окно после последнего вызова — {last.seven_day_util:.0%}"
            if last is not None and last.seven_day_util is not None else "")
    return "Потрачено: " + ", ".join(f"{k} {v:,}" for k, v in by.items()) + week


def run(day: date | None = None, *, graveyard: bool | None = None, paper: bool = True,
        implement: bool = True, search: bool = True, detectors: bool = True,
        assess: bool = True, research: bool | None = None, generate: bool | None = None,
        noise_trials: int = 200,
        deployable_capital_usd: float = 3000.0) -> Path:
    """The night's recheck stage; returns the report path. `graveyard`
    defaults to Sundays (weekly)."""
    day = day or datetime.now(UTC).date()
    end = last_closed_day(datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                          + timedelta(hours=12))
    state = load_state(day)
    state.setdefault("results", [])
    if detectors and "detectors" not in state:
        # Before the watch list: strategies on it read these files
        # (docs/REGIME_DETECT.md); a failed build keeps yesterday's file.
        from qlab.detector_builds import CONFIG, build_all

        state["detectors"] = build_all(end) if CONFIG.is_file() else []
        save_state(day, state)
    watch = yaml.safe_load(WATCHLIST.read_text(encoding="utf-8"))["specs"]
    state["unwatched_bench"] = unwatched_bench(watch)
    sweep = graveyard if graveyard is not None else day.weekday() == 6
    queue = [("watch", Path(p)) for p in watch]
    if sweep:
        queue += [("graveyard", p) for p in graveyard_specs() if str(p) not in watch]
    for kind, path in queue:
        key = f"{kind}:{path}"
        if key in state["done"]:
            continue
        result = evaluate_item(kind, load_spec(path), end=end, noise_trials=noise_trials,
                               deployable_capital_usd=deployable_capital_usd)
        state["results"] = [r for r in state["results"]
                            if not (r["kind"] == kind and r["name"] == result.name)]
        state["results"].append(asdict(result))
        if result.error is None:  # a failed item is tried again on the next run
            state["done"][key] = datetime.now(UTC).isoformat()
        save_state(day, state)
    if paper and "paper" not in state:
        state["paper"] = paper_reconciliation(day)
        save_state(day, state)
    if assess:
        assess_stage(day, state)
    if research if research is not None else sweep:
        research_stage(day, state, noise_trials=noise_trials)
    if implement:
        implement_stage(day, state, noise_trials=noise_trials,
                        deployable_capital_usd=deployable_capital_usd)
    if search:
        search_stage(day, state)
    if generate is True or (search and generate is None):
        # Last: the search for new ideas is what the budget cuts first, and the
        # generator is the most expensive part of it.
        generate_stage(day, state, noise_trials=noise_trials,
                       deployable_capital_usd=deployable_capital_usd, force=generate)
    return write_report(day, state)


def worth_assessing(session, results: list[dict]) -> list[dict]:
    """Watched runs worth a second opinion (docs/ASSESSOR.md): the verdict
    changed, or the idea is on paper or live -- where a wrong verdict costs
    something. Runs already assessed are skipped."""
    from qlab.registry.models import Assessment, Idea, IdeaStatus

    out = []
    for r in results:
        if r.get("error") or not r.get("trial_id") or r.get("kind") != "watch":
            continue
        idea = session.get(Idea, r["name"])
        changed = bool(r.get("previous_route")) and r["previous_route"] != r.get("route")
        stake = idea is not None and idea.status in (IdeaStatus.PAPER, IdeaStatus.LIVE)
        done = session.query(Assessment.id).filter(
            Assessment.trial_id == r["trial_id"]).first() is not None
        if (changed or stake) and not done:
            out.append(r)
    return out


def assess_stage(day: date, state: dict) -> None:
    """The assessor's second opinions, in the recheck stage (CLAUDE.md order:
    the health of strategies already running comes first). The verdicts stay
    the code's; a stop keeps the rest for the next night."""
    from qlab.agents.assessor import build_brief, run_assessment
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    out = state.setdefault("assessed", [])
    with session_scope() as session:
        todo = [r for r in worth_assessing(session, state.get("results", []))
                if f"assess:{r['trial_id']}" not in state["done"]]
        if not todo:
            return
        night_budget = NightBudget.open(session, probe(session).report.reading)
        stage = night_budget.stage(Stage.RECHECK, candidates=len(todo))
        for r in todo:
            entry = {"idea": r["name"], "trial": r["trial_id"]}
            try:
                budget = stage.candidate(r["name"])
                outcome = run_assessment(brief=build_brief(session, r["trial_id"],
                                                           paper=state.get("paper")),
                                         budget=budget, session=session)
                entry.update(view=outcome.view, disputes=outcome.dispute,
                             summary=outcome.summary, tokens=budget.spent_tokens)
                state["done"][f"assess:{r['trial_id']}"] = datetime.now(UTC).isoformat()
            except NightExhausted as stop:
                entry["stopped"] = f"night budget: {stop}"
                out.append(entry)
                break
            except BudgetExhausted as stop:
                entry["stopped"] = f"candidate budget: {stop}"
            out.append(entry)
            session.commit()
            save_state(day, state)
        session.commit()
    save_state(day, state)


RESEARCH = Path("night/research.yaml")
LIMITS = Path("night/limits.yaml")
ACTIVE_STATUSES = ("candidate", "bench", "validated")


def is_agent_idea(idea_id: str) -> bool:
    """Ideas the agents create: the generator's `gen-*`, the researcher's `*-research-*`."""
    return idea_id.startswith("gen-") or "-research-" in idea_id


def agent_idea_slots(session, limits: Path = LIMITS) -> tuple[int, int, list[str]]:
    """(free places, the cap, the active agent ideas) under night/limits.yaml:
    the owner keeps at most a few agent ideas in active development at once.
    Without the file there is no cap."""
    from qlab.registry.models import Idea

    active = sorted(i.id for i in session.query(Idea).filter(
        Idea.status.in_(ACTIVE_STATUSES)) if is_agent_idea(i.id))
    raw = (yaml.safe_load(limits.read_text(encoding="utf-8")) or {}).get(
        "max_active_agent_ideas") if limits.is_file() else None
    if raw is None:  # no limits file: no cap (the repository ships one)
        return 1_000_000, 0, active
    cap = int(raw)
    return max(0, cap - len(active)), cap, active


def full_house_focus(cap: int, active: list[str]) -> str:
    """The researcher's instruction when no place is free."""
    return (f"The lab keeps at most {cap} agent ideas in active development and all places are "
            f"taken ({', '.join(active)}). Propose NO new experiment spec. Instead, in the memo, "
            "say which of these active ideas you would drop and why, and what you would try in "
            "its place.")


def latest_memo(name: str, memo_dir: Path = Path("docs/research")) -> Path | None:
    """The strategy's newest research memo, handed to its next run."""
    memos = sorted(memo_dir.glob(f"{name}-*.md"))
    return memos[-1] if memos else None


def research_stage(day: date, state: dict, *, noise_trials: int,
                   config: Path = RESEARCH) -> None:
    """The researcher on each strategy of `night/research.yaml`, weekly, in
    the implement stage's budget (CLAUDE.md order: after rechecks, before the
    search for new ideas); the stand then runs its experiments. A stop keeps
    the rest for the next night that runs the stage."""
    from qlab.agents.researcher import research, run_experiments
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    if not config.is_file():
        return
    plan = yaml.safe_load(config.read_text(encoding="utf-8"))
    out = state.setdefault("research", [])
    todo = {name: e for name, e in (plan.get("strategies") or {}).items()
            if f"research:{name}" not in state["done"]}
    if not todo:
        return
    with session_scope() as session:
        free, cap, active = agent_idea_slots(session)
        stage = NightBudget.open(session, probe(session).report.reading).stage(
            Stage.IMPLEMENT, candidates=len(todo))
        for name, e in todo.items():
            entry = {"strategy": name}
            docs = [Path(d) for d in e.get("docs") or [] if Path(d).is_file()]
            if (memo := latest_memo(name)) is not None:
                docs.append(memo)
            try:
                budget = stage.candidate(None)
                outcome = research(name, idea_ids=list(e["ideas"]),
                                   specs=[Path(p) for p in e["specs"]], docs=docs,
                                   budget=budget, session=session,
                                   max_experiments=min(int(plan.get("max_experiments", 2)), free),
                                   focus="\n\n".join(filter(None, [
                                       plan.get("focus"),
                                       full_house_focus(cap, active) if free == 0 else None])))
                free = max(0, free - len(outcome.specs))
                session.commit()
                entry.update(memo=str(outcome.memo or ""), specs=[str(p) for p in outcome.specs],
                             problems=outcome.problems, tokens=budget.spent_tokens)
                state["done"][f"research:{name}"] = datetime.now(UTC).isoformat()
            except NightExhausted as stop:
                entry["stopped"] = f"night budget: {stop}"
                out.append(entry)
                break
            except BudgetExhausted as stop:
                entry["stopped"] = f"candidate budget: {stop}"
            out.append(entry)
            save_state(day, state)
    save_state(day, state)
    # The stand runs the experiments, each in its own transactions; then the
    # finished memo goes into the registry, one note per idea of the strategy.
    for entry in out:
        if entry.get("memo") and entry.get("specs") and "stand" not in entry:
            entry["stand"] = run_experiments(Path(entry["memo"]), [Path(p) for p in entry["specs"]],
                                             name=entry["strategy"], noise_trials=noise_trials)
            save_state(day, state)
        if entry.get("memo") and "noted" not in entry:
            record_research_note(entry, plan)
            save_state(day, state)


def record_research_note(entry: dict, plan: dict) -> None:
    """Store one strategy's finished memo as research notes (registry). A
    failure is written into the entry for the report, not raised: the memo
    file is already safe on disk and the night goes on."""
    from qlab.agents.researcher import MODEL
    from qlab.registry.db import session_scope
    from qlab.registry.research_notes import spec_index, upsert_research_notes

    strategy = (plan.get("strategies") or {}).get(entry["strategy"])
    if strategy is None:
        entry["noted"] = 0
        return
    try:
        with session_scope() as session:
            notes = upsert_research_notes(
                session, name=entry["strategy"], idea_ids=strategy["ideas"],
                memo_path=entry["memo"], problems=entry.get("problems") or [],
                tokens=entry.get("tokens"), model=MODEL,
                specs=spec_index(extra=entry.get("specs") or []))
        entry["noted"] = len(notes)
    except Exception as exc:  # noqa: BLE001 - the report says so, the night goes on
        entry["noted"] = 0
        entry["note_error"] = f"{type(exc).__name__}: {exc}"


def search_stage(day: date, state: dict) -> None:
    """Stage 3, cut first (CLAUDE.md): the venue watcher's events, then the
    scout's draft card (if any) inside what tonight's budget has left."""
    from qlab.agents.scout import scout
    from qlab.budget import BudgetExhausted, NightBudget, Stage, probe
    from qlab.registry.db import session_scope
    from qlab.venue_watch import watch

    if "events" not in state:
        try:
            state["events"] = [e.__dict__ for e in watch(day)]
        except Exception as exc:  # noqa: BLE001 - the report says so, the night goes on
            state["events"] = []
            state["events_error"] = f"{type(exc).__name__}: {exc}"
        save_state(day, state)
    if "scout" in state:
        return
    with session_scope() as session:
        try:
            night_budget = NightBudget.open(session, probe(session).report.reading)
            budget = night_budget.stage(Stage.SEARCH, candidates=1).candidate(None)
            outcome = scout(day, budget=budget, session=session)
            state["scout"] = {"reply": outcome.reply, "draft": str(outcome.draft or ""),
                              "problems": outcome.problems, "tokens": budget.spent_tokens}
        except BudgetExhausted as stop:
            state["scout"] = {"reply": None, "stopped": str(stop)}
        session.commit()
    save_state(day, state)


GENERATE = Path("night/generate.yaml")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _weekday(value) -> int:
    if isinstance(value, str) and value.strip().lower() in _WEEKDAYS:
        return _WEEKDAYS.index(value.strip().lower())
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 6:
        return value
    raise ValueError(f"night/generate.yaml: weekday must be a day name or 0..6, got {value!r}")


def generate_due(day: date, plan: dict) -> bool:
    """Whether the idea generator runs tonight: enabled, and on its cadence
    (daily, or weekly on `weekday`). A run the budget stopped before the agent
    finished is continued the next night -- a stop is the normal mode
    (CLAUDE.md), not a lost week."""
    if not plan.get("enabled", False):
        return False
    if plan.get("cadence", "weekly") == "daily":
        return True
    if day.weekday() == _weekday(plan.get("weekday", "wednesday")):
        return True
    yesterday = load_state(day - timedelta(days=1))
    return bool((yesterday.get("generate") or {}).get("stopped")
                and "generate:agent" not in yesterday.get("done", {}))


def _data_end_for_generation(session, day: date) -> date:
    """The latest Binance snapshot end the registry knows, never later than
    the last closed day."""
    from qlab.agents.generator import latest_binance_end

    closed = last_closed_day(datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                             + timedelta(hours=12))
    latest = latest_binance_end(session)
    return min(latest, closed) if latest else closed


def generate_stage(day: date, state: dict, *, noise_trials: int,
                   deployable_capital_usd: float = 3000.0, config: Path = GENERATE,
                   force: bool | None = None) -> None:
    """The idea generator (`qlab.agents.generator`), in the search stage -- the
    one CLAUDE.md cuts first: one agent call writes up to `max_specs` specs from
    docs/MECHANISMS.md, code guards them, the stand runs those that pass.

    Resumable: the agent's call and each spec's run have their own done-markers
    in the night's state, so a restart goes on from where it stopped. `force`
    overrides the config's cadence and `enabled` (the CLI's --generate)."""
    from qlab.agents.generator import (
        CATALOGUE,
        GeneratedSpec,
        generate,
        run_generated_one,
    )
    from qlab.budget import BudgetExhausted, NightBudget, Stage, probe
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea

    if force is False:
        return
    if not config.is_file() and force is not True:
        return
    plan = yaml.safe_load(config.read_text(encoding="utf-8")) if config.is_file() else {}
    plan = plan or {}
    if force is None and not generate_due(day, plan):
        return
    gen = state.setdefault("generate", {})
    catalogue_path = Path(plan.get("catalogue", CATALOGUE))

    if "generate:agent" not in state["done"]:
        if (state.get("scout") or {}).get("stopped"):
            gen["stopped"] = "the night's budget ended in the search stage before this"
            save_state(day, state)
            return
        if not catalogue_path.is_file():
            gen["skipped"] = f"no catalogue of mechanisms at {catalogue_path}"
            save_state(day, state)
            return
        with session_scope() as session:
            free, cap, active = agent_idea_slots(session)
        if free == 0:
            gen["skipped"] = (f"в работе уже {len(active)} идей агентов при потолке {cap} "
                              f"({', '.join(active)}): новых не завожу")
            save_state(day, state)
            return
        gen.pop("skipped", None)
        gen.pop("stopped", None)
        with session_scope() as session:
            try:
                night_budget = NightBudget.open(session, probe(session).report.reading)
                budget = night_budget.stage(Stage.SEARCH, candidates=1).candidate(None)
                outcome = generate(
                    budget=budget, session=session, catalogue_path=catalogue_path,
                    max_specs=min(int(plan.get("max_specs", 3)), free),
                    data_end=_data_end_for_generation(session, day),
                    registry_idea_ids=[i for (i,) in session.query(Idea.id)])
                gen.update(
                    reply=outcome.reply, problems=outcome.problems, reverted=outcome.reverted,
                    tokens=budget.spent_tokens,
                    accepted=[{"idea_id": a.spec.idea_id, "path": str(a.path),
                               "mechanism": a.mechanism} for a in outcome.accepted],
                    rejected=[{"path": r.path, "problems": r.problems} for r in outcome.rejected])
                state["done"]["generate:agent"] = datetime.now(UTC).isoformat()
            except BudgetExhausted as stop:
                gen["stopped"] = str(stop)
            session.commit()
        save_state(day, state)
        if "generate:agent" not in state["done"]:
            return

    results = gen.setdefault("results", [])
    for entry in gen.get("accepted") or []:
        key = f"generate:run:{entry['idea_id']}"
        if key in state["done"]:
            continue
        item = GeneratedSpec(path=Path(entry["path"]), spec=load_spec(Path(entry["path"])),
                             mechanism=entry["mechanism"])
        result = run_generated_one(item, noise_trials=noise_trials,
                                   deployable_capital_usd=deployable_capital_usd)
        results[:] = [r for r in results if r["idea_id"] != result.idea_id]
        results.append(asdict(result))
        if result.error is None:  # a failed run is tried again on the next restart
            state["done"][key] = datetime.now(UTC).isoformat()
        save_state(day, state)


def implement_stage(day: date, state: dict, *, noise_trials: int,
                    deployable_capital_usd: float) -> None:
    """Stage 2 (CLAUDE.md order): cards from `night/implement_queue.yaml` not
    yet implemented, written by the implementer agent and run through every
    guard, inside tonight's token budget. A night stop or a candidate's cap
    is not a failure: the card stays queued for the next night."""
    from qlab.agents.implementer import implement
    from qlab.agents.reviewer import code_files, run_review
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    if not IMPLEMENT_QUEUE.is_file():
        return
    cards = yaml.safe_load(IMPLEMENT_QUEUE.read_text(encoding="utf-8")).get("cards") or []
    pending = [c for c in cards if f"implement:{c}" not in state["done"]]
    if not pending:
        return
    out = state.setdefault("implemented", [])
    with session_scope() as session:
        night_budget = NightBudget.open(session, probe(session).report.reading)
        stage = night_budget.stage(Stage.IMPLEMENT, candidates=len(pending))
        for card in pending:
            entry = {"card": card}
            try:
                budget = stage.candidate(None)
                outcome = implement(Path(card), budget=budget, session=session)
                entry.update(idea_id=outcome.idea_id, reply=outcome.agent_reply,
                             problems=outcome.problems,
                             not_expressible=outcome.not_expressible)
                if outcome.ok:
                    spec = load_spec(outcome.spec_path)
                    review = run_review(
                        spec=spec, spec_text=outcome.spec_path.read_text(encoding="utf-8"),
                        sources={card: Path(card).read_text(encoding="utf-8")},
                        code=code_files(spec), budget=budget, session=session)
                    entry["review"] = [f"[{f.kind}] {f.summary}" for f in review.accepted]
                entry["tokens"] = budget.spent_tokens
                state["done"][f"implement:{card}"] = datetime.now(UTC).isoformat()
            except NightExhausted as stop:
                entry["stopped"] = f"night budget: {stop}"
                out.append(entry)
                save_state(day, state)
                session.commit()
                return
            except BudgetExhausted as stop:
                entry["stopped"] = f"candidate budget: {stop}"
            out.append(entry)
            save_state(day, state)
            session.commit()
    # The stand runs each implemented spec in its own transaction.
    for entry in out:
        if entry.get("problems") == [] and "stand" not in entry:
            spec = load_spec(Path("specs/agent") / f"{entry['idea_id']}.yaml")
            _ensure_idea(spec, entry["card"])
            result = evaluate_item("implemented", spec, end=spec.data.end,
                                   noise_trials=noise_trials,
                                   deployable_capital_usd=deployable_capital_usd)
            entry["stand"] = result.error or f"{result.route}: {result.reason}"
            save_state(day, state)


def _ensure_idea(spec: StrategySpec, card: str) -> None:
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    with session_scope() as session:
        if session.get(Idea, spec.idea_id) is None:
            repo.upsert_idea(session, id=spec.idea_id, title=spec.title,
                             source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                             profile=Profile.OTHER,
                             notes=f"Implemented by the implementer agent from {card}.")


__all__ = ["ItemResult", "evaluate_item", "generate_due", "generate_stage", "graveyard_specs",
           "last_closed_day",
           "paper_reconciliation", "run", "with_end", "write_report"]
