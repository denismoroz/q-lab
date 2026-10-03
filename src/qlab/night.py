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
               "forward_days", "forward_days_needed")


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
            result.metrics.update({k: v for k, v in metrics.items() if k.startswith("regime_")
                                   and k.endswith(("_return", "_noise_percentile"))})
    except Exception as exc:  # noqa: BLE001 - recorded in the report, the night goes on
        result.error = f"{type(exc).__name__}: {exc}"
    return result


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
        "Bv2 (b2 as created, carry-on sizes)": [
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
                if any(w in ln for w in ("correlation", "paper", "replay", "fills matched",
                                         "matched hours", "max |"))]
        status = "" if done.returncode == 0 else f" (exit {done.returncode})"
        lines.append(f"**{label}**{status}")
        lines += [f"    {ln}" for ln in keep] or [f"    {done.stderr.strip()[-300:]}"]
    return lines


def _pct(v) -> str:
    return f"{v:+.1%}" if isinstance(v, int | float) else "—"


def write_report(day: date, state: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Ночной прогон {day.isoformat()}", ""]
    items = [ItemResult(**r) for r in state.get("results", [])]
    for kind, title in (("watch", "Стратегии на проверке вперёд"),
                        ("graveyard", "Кладбище по режимам")):
        group = [r for r in items if r.kind == kind]
        if not group:
            continue
        lines += [f"## {title}", "", "| стратегия | исход | было | в год | Шарп | вперёд, дней | "
                  "рост | боковик | падение |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in group:
            if r.error:
                lines.append(f"| {r.name} | ошибка: {r.error[:120]} | | | | | | | |")
                continue
            m = r.metrics
            changed = " **(изменился)**" if r.previous_route and r.previous_route != r.route else ""
            lines.append(
                f"| {r.name} | {r.route}{changed} | {r.previous_route or '—'} | "
                f"{_pct(m.get('ann_return_net'))} | {m.get('sharpe_net', float('nan')):.2f} | "
                f"{m.get('forward_days', 0):.0f} | {_pct(m.get('regime_bull_return'))} | "
                f"{_pct(m.get('regime_flat_return'))} | {_pct(m.get('regime_bear_return'))} |")
        lines.append("")
    if kind_lines := state.get("paper"):
        lines += ["## Сверка бумаги на проде со стендом", "", *kind_lines, ""]
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
        implement: bool = True, search: bool = True, noise_trials: int = 200,
        deployable_capital_usd: float = 3000.0) -> Path:
    """The night's recheck stage; returns the report path. `graveyard`
    defaults to Sundays (weekly)."""
    day = day or datetime.now(UTC).date()
    end = last_closed_day(datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                          + timedelta(hours=12))
    state = load_state(day)
    state.setdefault("results", [])
    watch = yaml.safe_load(WATCHLIST.read_text(encoding="utf-8"))["specs"]
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
    if implement:
        implement_stage(day, state, noise_trials=noise_trials,
                        deployable_capital_usd=deployable_capital_usd)
    if search:
        search_stage(day, state)
    return write_report(day, state)


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


__all__ = ["ItemResult", "evaluate_item", "graveyard_specs", "last_closed_day",
           "paper_reconciliation", "run", "with_end", "write_report"]
