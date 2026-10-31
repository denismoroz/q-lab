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

import copy
import json
import subprocess
import threading
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

from qlab.capital_text import CAP_NUMBERS, headline
from qlab.nightflow.runner import StepImpl
from qlab.pipeline.spec import StrategySpec, load_spec
from qlab.year_view import year_keys, years_line

WATCHLIST = Path("night/watchlist.yaml")
IMPLEMENT_QUEUE = Path("night/implement_queue.yaml")
STATE_DIR = Path("data/night")
REPORT_DIR = Path("reports/night")
PAPER_DIR = Path("data/paper")
SCHEDULE = Path("night/schedule.yaml")
PROD = "dis@10.8.0.5"
PROD_DB = "file:data/frab.db?mode=ro"  # read-only: q-lab never changes frab's state

#: Stage names in the morning report's stop line (`past_stop`).
STOP_STAGES = {"detectors": "определители режима",
               "recheck": "перепроверка списка наблюдения", "waiting": "ждущие идеи",
               "twins": "двойники и вторая ступень",
               "second_venue": "вторая площадка для закрытых идей",
               "venue_gap": "почему площадки расходятся",
               "twins-research": "двойники экспериментов исследователя",
               "restamp": "пересчёт старых результатов",
               "anomalies": "аномалии", "anomalies-investigate": "разбор аномалий агентом",
               "overlays": "надстройки", "ladder": "лестница оценки", "report": "утренний отчёт",
               "paper": "сверка бумаги", "decay": "детектор угасания",
               "allocation": "распределение капитала", "assess": "второе мнение",
               "director": "руководитель исследований", "research": "исследователь",
               "experiments": "эксперименты исследователя", "programmer": "программист",
               "implement": "реализация карточек",
               "search": "поиск", "catalogue": "разбор статей в каталог",
               "cards": "карточки для входящих идей", "generate": "генератор идей"}

KEY_METRICS = ("ann_return_net", "sharpe_net", "max_dd", "noise_return_percentile",
               "forward_days", "forward_days_needed", "fit_window_days", "noise_power",
               "fit_ann_return_net", "fit_sharpe_net", "fit_max_dd", "selection_days",
               "ann_return_net_ex_best_1pct", *CAP_NUMBERS)


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


#: Called with the state after every save: the journal's item tracker watches the done-markers
#: this way (`qlab.nightflow.runner.ItemTracker`), so the stages did not have to change.
STATE_OBSERVERS: list = []


_WRITE_LOCK = threading.Lock()


def write_state_text(day: date, text: str) -> None:
    """The night's state file, replaced atomically; two writers never share the temporary file."""
    with _WRITE_LOCK:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _state_path(day).with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(_state_path(day))


def save_state(day: date, state: dict) -> None:
    """Save the night's state. A step's `state` is its view (`qlab.nightflow.statehub`): then the
    step's own keys are merged into the hub's master and the file is written from the master, so
    steps running together never walk a dict another one is changing. A plain dict (a hand run, a
    test) is written as it is."""
    hub = getattr(state, "hub", None)
    if hub is not None:
        hub.save(state)
        return
    write_state_text(day, json.dumps(state, indent=1, default=str))
    for observer in list(STATE_OBSERVERS):
        try:
            observer(state)
        except Exception:  # noqa: BLE001 - the journal must never fail a save of the night
            pass


def stop_time(hhmm: str, now: datetime | None = None) -> datetime:
    """`HH:MM` today on this machine's clock, as an aware time: the night's
    stop time (`qlab night run --stop-at`). Owner, 2026-10-26: the night must
    not run into the working day and must be over before 08:00 Minsk."""
    now = (now or datetime.now(UTC)).astimezone()
    hour, minute = (int(x) for x in hhmm.split(":"))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


class Deadline(datetime):
    """A stop time that cuts only the night's code stages (`code_stages_cut`): past it they start
    nothing new, while the agents' stages go on until the real stop time."""


def code_stages_cut(stop_at: datetime | None, started: datetime | None = None,
                    path: Path = SCHEDULE) -> datetime | None:
    """The time after which the restamp, the venue-gap, the anomaly scan and the overlays start no
    new idea (`night/schedule.yaml`, default 05:00 local), as the `stop_at` those stages are given.

    The real stop time when it comes first, when the file says nothing, or when the run began after
    the deadline (a run by hand in the day); a `Deadline` otherwise. A night without a stop time has
    no deadline either."""
    if stop_at is None:
        return None
    try:
        hhmm = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
            "code_stages_deadline")
    except (OSError, ValueError, yaml.YAMLError):
        hhmm = None
    if not hhmm:
        return stop_at
    started = started or datetime.now(UTC)
    deadline = stop_time(str(hhmm), started)
    if deadline >= stop_at or started >= deadline:
        return stop_at
    return Deadline(deadline.year, deadline.month, deadline.day, deadline.hour, deadline.minute,
                    deadline.second, deadline.microsecond, tzinfo=deadline.tzinfo)


def past_stop(stop_at: datetime | None, state: dict, stage: str) -> bool:
    """Whether the stop time has passed, checked before each piece of work so a
    started one always finishes. The first stage it cuts goes into the state
    (`stopped_at`), with every stage it kept from starting afterwards (`cut`), for
    the morning report; what was cut runs on the next night (the rechecks are
    redone every night anyway; unrun experiments carry over, `carried_research`).

    A `Deadline` (the code stages' own, `code_stages_cut`) is not a stop of the night: it is
    written to `state["deadline"]` with the stages it deferred, and the night goes on."""
    if stop_at is None or datetime.now(UTC) < stop_at:
        return False
    hub = getattr(state, "hub", None)
    if hub is not None:  # steps running together: the stop lines live in the hub's master
        with hub.lock:
            _note_stop(hub.master, stop_at, stage)
        return True
    _note_stop(state, stop_at, stage)
    return True


def _note_stop(state: dict, stop_at: datetime, stage: str) -> None:
    top = stage.split(":")[0]
    if isinstance(stop_at, Deadline):
        block = state.setdefault("deadline", {"at": stop_at.isoformat(), "deferred": []})
        if top not in block["deferred"]:
            block["deferred"].append(top)
        return
    stopped = state.setdefault("stopped_at", {"time": datetime.now(UTC).isoformat(),
                                              "stop_at": stop_at.isoformat(), "before": stage})
    cut = stopped.setdefault("cut", [])
    if top not in cut:
        cut.append(top)


def previous_route(session, idea_id: str, before: int) -> str | None:
    from qlab.registry.models import Spec, Trial

    row = (session.query(Trial.route).join(Spec, Spec.id == Trial.spec_id)
           .filter(Spec.idea_id == idea_id, Trial.route.is_not(None), Trial.id < before)
           .order_by(Trial.id.desc()).first())
    return getattr(row[0], "value", row[0]) if row else None


def rerun_spec(spec: StrategySpec) -> StrategySpec:
    """The spec a night re-run evaluates. A spec with open numbers (qlab.own_numbers) is its
    one-choice form, its cut computed from the window the spec file declares -- before the
    data is extended -- so the forward days grow night by night and the choice stays the one
    made at the first run. Its candidates were recorded as trials by that first run (the
    researcher or the generator); re-running the same choice adds no new candidate. The
    night of 2026-10-27 passed such specs to the stand as they were and both failed."""
    from qlab import own_numbers

    if not own_numbers.open_params(spec):
        return spec
    config = own_numbers.load_config()
    if found := own_numbers.problems(spec, config):
        raise ValueError("; ".join(found))
    return own_numbers.to_retune(spec, config)


def evaluate_item(kind: str, spec: StrategySpec, *, end: date, noise_trials: int) -> ItemResult:
    """One evaluation in its own transaction, so one failure cannot roll back
    the night's other results."""
    from qlab.calibration.shape_aware import evaluate_spec_with_shape_aware_bar
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    result = ItemResult(kind=kind, name=spec.idea_id)
    try:
        spec = rerun_spec(spec)
        with session_scope() as session:
            out = evaluate_spec_with_shape_aware_bar(
                with_end(spec, end), session=session, ruleset=load_latest(), n_trials=noise_trials,
            ).candidate
            result.trial_id = out.trial_id
            result.route = out.routing.route
            result.reason = out.routing.reason
            result.previous_route = previous_route(session, spec.idea_id, out.trial_id)
            metrics = out.metrics or {}
            result.metrics = {k: metrics[k] for k in KEY_METRICS if k in metrics}
            result.metrics.update({k: v for k, v in metrics.items()
                                   if k.startswith(("regime_", "fit_regime_", "cap_regime_",
                                                    "fit_cap_regime_"))
                                   and k.endswith(("_return", "_noise_percentile"))})
            result.metrics.update({k: metrics[k] for k in year_keys(metrics)})
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
    from qlab.registry.stages import is_stage2_child

    watched = set()
    for path in watch:
        try:
            watched.add(load_spec(Path(path)).idea_id)
        except Exception:  # noqa: BLE001 - an unloadable spec fails loudly in its own run
            continue
    with session_scope() as session:
        # The second stage of a generated idea (docs/TWO_STAGE.md) waits with its parent,
        # which is named; naming both would count one idea twice.
        # An overlay variant is judged against its base on one fixed history
        # (qlab.overlay_stage); it needs no place on the watch list.
        bench = [i.id for i in session.query(Idea).filter(Idea.status == IdeaStatus.BENCH)
                 if not is_stage2_child(i) and "-ov-" not in i.id]
    return sorted(i for i in bench if i not in watched)


def graveyard_specs(spec_dir: Path = Path("specs")) -> list[Path]:
    """Spec files of ideas whose status is rejected -- what the weekly regime sweep re-runs.

    The whole `specs/` tree is walked (owner, 2026-10-29, docs/EVALUATION_PATHS.md П2: the sweep
    saw 3 of 35 rejected ideas because only the folder's own files were looked at; the rest sat
    in `specs/research/`, `specs/generated/`). Two kinds of file are left out on purpose: a second
    stage (`<id>-hl`) follows its first stage and has no sweep of its own, and an overlay variant
    (`<id>-ov-...`) is judged against its base on one fixed history (`qlab.overlay_stage`).
    Measured on 2026-10-29: 15 ideas, about half an hour at 110-190 seconds an hourly item with
    its 200 random books -- under the hour the owner's night can spare on a Sunday."""
    from qlab.overlay_variants import is_variant_id
    from qlab.registry.db import session_scope
    from qlab.registry.models import Idea, IdeaStatus
    from qlab.registry.stages import is_stage2_child

    with session_scope() as session:
        rejected = {i.id for i in session.query(Idea).filter(Idea.status == IdeaStatus.REJECTED)
                    if not is_stage2_child(i) and not is_variant_id(i.id)}
    out = []
    for path in sorted(spec_dir.rglob("*.yaml")):
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
        # trend_retune_* books (frab, one strategy row each), night/paper_books.yaml
        _ssh_query("select * from trend_equity where strategy_id in (select id from strategies "
                   "where name like 'trend_retune%') order by strategy_id, ts_ms",
                   folder / "trend_retune_equity.csv")
    except (subprocess.SubprocessError, OSError) as exc:
        return [f"export from the production host failed: {exc}"]
    lines += _export_live_books(folder)
    retune_lines, retune_runs = _retune_replay_runs(folder)
    lines += retune_lines
    runs = {
        "Bv2 (book 3 = b2, sized as created with carry on; book 4 = b2_cold)": [
            "scripts/replay_bv2_paper.py", "--paper", str(folder / "b2_equity.csv"),
            "--b2-sized-with-carry", "--out", str(folder / "replay_b2.csv")],
        "trend": ["scripts/replay_trend_paper.py", "--paper", str(folder / "trend_equity.csv"),
                  "--params", str(folder / "trend_params.json"),
                  "--events", str(folder / "trend_events.csv"),
                  "--out", str(folder / "replay_trend.csv")],
    }
    for label, args in {**runs, **retune_runs}.items():
        try:
            done = subprocess.run(["uv", "run", "python", *args], capture_output=True, text=True,
                                  timeout=1800)
        except subprocess.TimeoutExpired:
            lines += [f"**{label}**", "    the replay did not finish in 30 minutes"]
            continue
        keep = [ln for ln in done.stdout.splitlines()
                if any(w in ln for w in REPLAY_KEYWORDS)]
        status = "" if done.returncode == 0 else f" (exit {done.returncode})"
        lines.append(f"**{label}**{status}")
        lines += [f"    {ln}" for ln in keep] or [f"    {done.stderr.strip()[-300:]}"]
    return lines + _paper_check_lines(day, folder)


# The lines of a replay script's output that go into the report and into the night's state,
# where `qlab.paper_rule` reads them back.
REPLAY_KEYWORDS = ("book ", "correlation", "paper", "replay", "fills matched", "matched hours",
                   "max |", "history", "universes")

IDEA_B2_RETUNE = "bv2-research-headroom-retune"
IDEA_TREND_RETUNE = "trend-retune-weekly"


def _retune_replay_runs(folder: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Exports and replay commands of the two re-tuned paper books (night/paper_books.yaml):
    `b2_retune` (scripts/replay_b2_retune_paper.py, live from the book's `realised_from`) and
    `trend_retune_3k` (scripts/replay_trend_retune_paper.py). Everything is read-only on the
    production host. A book missing from the mapping, or an export that fails, costs only
    that book's replay and is a report line: the other replays and the night go on."""
    from qlab.paper_check import as_utc, load_books

    try:
        books = load_books()
    except (OSError, KeyError, ValueError) as exc:
        return [f"re-tuned books not replayed: {type(exc).__name__}: {exc}"], {}
    lines: list[str] = []
    runs: dict[str, list[str]] = {}
    b2 = books.get(IDEA_B2_RETUNE)
    if b2 is not None:
        sid = b2["strategy_id"]
        try:
            live_from = as_utc(b2["realised_from"]).isoformat()
            _ssh_query(f"select params_json from strategies where id={sid}",
                       folder / "b2_retune_params.json", header=False)
            _ssh_query(f"select ts_ms, details_json from b2_events where strategy_id={sid} "
                       "and kind='retune' order by ts_ms", folder / "b2_retune_events.csv")
            runs[f"Bv2 re-tuned monthly (book {sid} = b2_retune)"] = [
                "scripts/replay_b2_retune_paper.py", "--paper", str(folder / "b2_equity.csv"),
                "--params", str(folder / "b2_retune_params.json"),
                "--events", str(folder / "b2_retune_events.csv"), "--live-from", live_from,
                "--out", str(folder / (b2.get("replay_file") or "replay_b2_retune.csv"))]
        except (subprocess.SubprocessError, OSError, KeyError, AttributeError) as exc:
            lines.append(f"b2_retune not replayed: {type(exc).__name__}: {exc}")
    trend = books.get(IDEA_TREND_RETUNE)
    if trend is not None:
        sid = trend["strategy_id"]
        try:
            _ssh_query(f"select params_json from strategies where id={sid}",
                       folder / "trend_retune_params.json", header=False)
            _ssh_query(f"select * from trend_events where strategy_id={sid} order by ts_ms",
                       folder / "trend_retune_events.csv")
            runs[trend.get("replay_block") or "trend_retune"] = [
                "scripts/replay_trend_retune_paper.py", "--paper",
                str(folder / "trend_retune_equity.csv"), "--strategy-id", str(sid),
                "--params", str(folder / "trend_retune_params.json"),
                "--events", str(folder / "trend_retune_events.csv"),
                "--out", str(folder / (trend.get("replay_file") or "replay_trend_retune.csv"))]
        except (subprocess.SubprocessError, OSError, KeyError) as exc:
            lines.append(f"trend retune book not replayed: {type(exc).__name__}: {exc}")
    return lines, runs


def _export_live_books(folder: Path) -> list[str]:
    """The raw rows of the live-money books (night/paper_books.yaml, `live_books`), read-only,
    and the strategy-equity series built from them (qlab.strategy_pnl: the strategy's budget
    plus its own result, not the account's total), for the decay detector (qlab.decay). A
    failure is a report line: it must not stop the paper books' reconciliation."""
    from qlab.decay import build_live_series, live_exports

    try:
        queries = live_exports()
    except (OSError, KeyError, ValueError) as exc:
        return [f"live books not exported: {type(exc).__name__}: {exc}"]
    lines: list[str] = []
    for filename, sql in queries:
        try:
            _ssh_query(sql, folder / filename)
        except (subprocess.SubprocessError, OSError) as exc:
            lines.append(f"export of {filename} from the production host failed: {exc}")
    if not lines:  # the series is built from the rows just exported, only if all arrived
        lines += build_live_series(folder)
    return lines


def _paper_check_lines(day: date, folder: Path) -> list[str]:
    """Each paper book against the stand's forward test over the SAME window
    (qlab.paper_check, night/paper_books.yaml); one row per idea in the
    registry. A failure here is a report line, not a failed night."""
    from qlab.paper_check import run_paper_checks
    from qlab.registry.db import session_scope

    try:
        with session_scope() as session:
            checks = run_paper_checks(session, day, folder)
    except Exception as exc:  # noqa: BLE001 - the reconciliation above already succeeded
        return [f"**Бумага и стенд за один отрезок** не посчитаны: {type(exc).__name__}: {exc}"]
    return ["**Бумага и стенд за один отрезок**", *[f"    {ln}" for ln in checks]]


def decay_stage(day: date, end: date, state: dict) -> None:
    """Where every running book's realised results fall among the same-length
    windows of its own stand history (qlab.decay). Reads the night's export and
    the stand's cached panels, writes no trial. A failure is a report line, not a
    failed night."""
    from qlab.decay import run_decay, save_results
    from qlab.registry.db import session_scope

    try:
        with session_scope() as session:
            results, lines = run_decay(session, day, end)
            saved = save_results(session, results, day)
        if results and not saved:
            lines.append("В реестр не записано: таблицы `decay_check` ещё нет "
                         "(миграция не применена); результат лежит в состоянии ночи.")
        state["decay"] = {"lines": lines, "results": [r.to_dict() for r in results]}
    except Exception as exc:  # noqa: BLE001
        state["decay"] = {"lines": [f"не посчитано: {type(exc).__name__}: {exc}"], "results": []}


def unwatched_not_followed(state: dict) -> list[str]:
    """Bench ideas no spec on the watch list belongs to and that the waiting stage does not
    follow either (the recheck writes the first list, the waiting stage the second)."""
    followed = set(state.get("waiting_followed") or [])
    return [i for i in state.get("unwatched_bench") or [] if i not in followed]


def ideas_in_all(state: dict) -> list[dict]:
    """The inbox ideas added tonight: by the researcher and by the director (two keys, one owner
    each)."""
    return [*(state.get("ideas_in") or []), *(state.get("ideas_in_director") or [])]


def _pct(v) -> str:
    return f"{v:+.1%}" if isinstance(v, int | float) and v == v else "—"


def _report_row(m: dict) -> tuple[str, str, str, str, str, str, str]:
    """Selection period and forward test kept apart (docs/FIT_VS_FORWARD.md).

    With a forward test, the run's own yearly figure is the forward
    test's, and nineteen days annualized read as +400% a year: the forward
    test is shown as its total over its days instead, and the yearly
    figures and the regime split are the selection period's. Without one,
    the whole window is the selection period.

    All figures are percents of the strategy's own money when the run has them
    (docs/CAPITAL_BASE.md; the stand's share of the book's balance only for a run that has
    none)."""
    split = "fit_ann_return_net" in m
    prefix = "fit_" if split else ""
    days = m.get("forward_days") or 0
    now, part = headline(m), headline(m, prefix)
    regime = "cap_regime_" if now.on_capital else "regime_"
    forward_total = "—"
    if days > 0 and now.on_capital and now.total is not None:
        forward_total = _pct(now.total)
    elif days > 0 and isinstance(m.get("ann_return_net"), int | float):
        forward_total = _pct((1 + m["ann_return_net"]) ** (days / 365.25) - 1)
    if days > 0 and not split:
        # The whole judged window is the forward test (a causal selection, or
        # parameters fixed before the data began): no selection period to show.
        return ("—", "—", f"{days:.0f}", forward_total,
                *(_pct(m.get(f"{regime}{r}_return")) for r in ("bull", "flat", "bear")))
    return (_pct(part.ann),
            f"{part.sharpe:.2f}" if part.sharpe is not None else "—",
            f"{days:.0f}", forward_total,
            *(_pct(m.get(f"{prefix}{regime}{r}_return")) for r in ("bull", "flat", "bear")))


def write_report(day: date, state: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Ночной прогон {day.isoformat()}", ""]
    if stop := state.get("stopped_at"):
        lines += [f"**Ночь остановлена по сроку "
                  f"{datetime.fromisoformat(stop['stop_at']).astimezone():%H:%M}** перед этапом "
                  f"«{STOP_STAGES.get(stop['before'].split(':')[0], stop['before'])}» "
                  f"({stop['before']}). Начатое доделано, остальное — в следующую ночь: "
                  "перепроверки она делает заново, невыполненные эксперименты исследователя "
                  "переносятся.", ""]
        if cut := stop.get("cut"):
            lines += ["Из-за срока не начаты: " + ", ".join(
                f"«{STOP_STAGES.get(name, name)}»" for name in cut) + ".", ""]
    if deadline := state.get("deadline"):
        at = datetime.fromisoformat(deadline["at"]).astimezone()
        lines += [f"**После {at:%H:%M} кодовые этапы новых дел не начинали** (срок "
                  "`night/schedule.yaml`): " + ", ".join(
                      f"«{STOP_STAGES.get(name, name)}»" for name in deadline["deferred"])
                  + ". Так программист, руководитель и исследователь получают своё время; начатое "
                  "доделано, остальное ждёт следующей ночи.", ""]
    account = state.get("account") or {}
    if account.get("ok") is False:
        lines += ["**Агенты этой ночью не работали: Claude CLI вошёл не под тем аккаунтом.** "
                  f"{account.get('message')} Этапы без агентов (перепроверки, бумажные книги, "
                  "распад) прошли как обычно, а шаги агентов остановлены без траты токенов и "
                  "продолжатся в следующую ночь, когда вход будет выполнен.", ""]
    if off := state.get("limits_off"):
        lines += ["**Ограничения гарда токенов этой ночью были выключены владельцем** "
                  f"(night/budget_off.yaml): {off}. Ночь остановилась бы только на лимитах "
                  "самой подписки; сколько она потратила — в разделе «Токены».", ""]
    if errors := state.get("stage_errors"):
        lines += ["## Сбои стадий", "",
                  "Стадия упала, ночь пошла дальше; несделанное повторится при следующем запуске.",
                  "", *(f"- **{STOP_STAGES.get(name, name)}** (`{name}`): {text}"
                        for name, text in errors.items()), ""]
    items = [ItemResult(**r) for r in state.get("results", [])]
    for kind, title in (("watch", "Стратегии на проверке вперёд"),
                        ("graveyard", "Кладбище по режимам")):
        group = [r for r in items if r.kind == kind]
        if not group:
            continue
        lines += [f"## {title}", "",
                  "Годовые числа и режимы — по периоду подбора; проверка вперёд — итогом за её "
                  "дни, без пересчёта в год. Все числа — в процентах от денег стратегии (то, с чем "
                  "она началась; у стратегии, которая прибыль не вкладывает обратно, доходность "
                  "складывается, а не склеивается), не от баланса книги.", "",
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
        # The same runs year by year (docs/YEARS.md): a total that rests on one lucky period
        # is visible at a glance. A description; no verdict reads it.
        lines += [f"- `{r.name}`: {line}" for r in group
                  if not r.error and (line := years_line(r.metrics, signs=True))]
        if any(not r.error and years_line(r.metrics) for r in group):
            lines.append("")
    if state.get("ladder"):
        from qlab.ladder import SECTION_TITLE as LADDER_TITLE

        lines += [f"## {LADDER_TITLE}", "", *state["ladder"], ""]
    if state.get("waiting") or state.get("waiting_notes"):
        from qlab.waiting import SECTION_TITLE as WAITING_TITLE
        from qlab.waiting import report_lines as waiting_lines

        lines += [f"## {WAITING_TITLE}", "",
                  *waiting_lines(state.get("waiting") or [], state.get("waiting_notes")), ""]
    if state.get("twins"):
        from qlab.twin_stage import SECTION_TITLE as TWINS_TITLE
        from qlab.twin_stage import report_lines as twins_lines

        if body := twins_lines(state["twins"]):
            lines += [f"## {TWINS_TITLE}", "", *body, ""]
    if state.get("second_venue"):
        from qlab.second_venue import SECTION_TITLE as SECOND_VENUE_TITLE
        from qlab.second_venue import report_lines as second_venue_lines

        if body := second_venue_lines(state["second_venue"]):
            lines += [f"## {SECOND_VENUE_TITLE}", "", *body, ""]
    if state.get("restamp"):
        from qlab.restamp import SECTION_TITLE as RESTAMP_TITLE
        from qlab.restamp import report_lines as restamp_lines

        if body := restamp_lines(state["restamp"]):
            lines += [f"## {RESTAMP_TITLE}", "", *body, ""]
    if state.get("anomalies"):
        from qlab.anomalies.report import SECTION_TITLE as ANOMALIES_TITLE
        from qlab.anomalies.report import report_lines as anomalies_lines
        from qlab.registry.db import session_scope as anomalies_session

        with anomalies_session() as session:
            body = anomalies_lines(state["anomalies"], session)
        if body:
            lines += [f"## {ANOMALIES_TITLE}", "", *body, ""]
    if state.get("overlays"):
        from qlab.overlay_stage import SECTION_TITLE as OVERLAYS_TITLE
        from qlab.overlay_stage import report_lines as overlays_lines

        if body := overlays_lines(state["overlays"]):
            lines += [f"## {OVERLAYS_TITLE}", "", *body, ""]
    if idle := unwatched_not_followed(state):
        lines += ["## На скамейке, но не в списке наблюдения", "",
                  "Проверка вперёд у этих идей не растёт: либо добавить в `night/watchlist.yaml`, "
                  "либо вывести (`qlab retire`).", "", *[f"- {i}" for i in idle], ""]
    if built := state.get("detectors"):
        lines += ["## Определители режима (пересобраны до списка наблюдения)", ""]
        lines += [f"- {'ок' if b['ok'] else '**сбой**'} — {b['name']}: {b['note']}" for b in built]
        lines.append("")
    if kind_lines := state.get("paper"):
        lines += ["## Сверка бумаги на проде со стендом", "", *kind_lines, "",
                  *_paper_rule_lines(kind_lines), ""]
    if decayed := state.get("decay"):
        from qlab.decay import SECTION_TITLE

        lines += [f"## {SECTION_TITLE}", "", *decayed["lines"], ""]
    if split := state.get("allocation"):
        from qlab.pool_allocation import SECTION_TITLE as ALLOCATION_TITLE

        lines += [f"## {split.get('title') or ALLOCATION_TITLE}", "", *split["lines"], ""]
    if directed := state.get("director"):
        lines += ["## Руководитель исследований", "", *_director_lines(directed), ""]
    if studied := state.get("research"):
        lines += ["## Исследователь", ""]
        for e in studied:
            if e.get("stopped"):
                lines.append(f"- {e['strategy']}: остановлено — {e['stopped']}")
                continue
            if e.get("carried_from"):
                lines.append(f"- **{e['strategy']}**: эксперименты записки `{e.get('memo')}` "
                             f"от {e['carried_from']}, перенесённые остановкой по сроку")
            else:
                lines.append(f"- **{e['strategy']}**: записка `{e.get('memo')}`, экспериментов "
                             f"{len(e.get('specs') or [])}, {e.get('tokens', 0):,} токенов")
            lines += [f"    - не принято: {p}" for p in e.get("problems") or []]
            lines += [f"    - предлагает вывести `{c['idea']}`, но он уже закрыт ({c['status']}): "
                      "делать нечего" for c in e.get("closed") or []]
            # an experiment's line, and under it its second stage (indented, docs/TWO_STAGE.md)
            lines += [f"    {line}" if line.startswith("  ") else f"    - {line[2:]}"
                      for line in e.get("stand") or [] if line.startswith(("- ", "  - "))]
        lines.append("")
        if drops := [(e["strategy"], d) for e in studied for d in e.get("drops") or []]:
            lines += ["## Исследователь предлагает вывести варианты (решает владелец)", "",
                      "Это предложение агента, не решение: код статусов по его слову не меняет. "
                      "Вывести вариант — `qlab retire <идея> --reason '…'`.", ""]
            lines += [f"- {strategy}: `{d['idea']}` — {d['why']}"
                      + (f" (идеи в реестре нет, это спек на скамейке `{d['spec']}`: убрать — "
                         "удалить файл)" if d.get("spec") else "")
                      for strategy, d in drops]
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
    if (coded := state.get("programmer")) is not None:
        from qlab.agents.programmer import load_queue
        from qlab.agents.programmer import report_lines as programmer_lines

        lines += ["## Программист", "", *programmer_lines(coded.get("entries") or [],
                                                           load_queue()), ""]
        if coded.get("stopped"):
            lines.append(f"- остановлено: {coded['stopped']}; продолжит в следующую ночь")
            lines.append("")
    if (read := state.get("catalogue")) is not None:
        from qlab.agents.cataloguer import load_ledger
        from qlab.agents.cataloguer import report_lines as catalogue_lines

        lines += ["## Разбор статей в каталог", "", *catalogue_lines(read.get("entries") or [],
                                                                     load_ledger()), ""]
        if read.get("stopped"):
            lines += [f"- остановлено: {read['stopped']}; продолжит в следующую ночь", ""]
        if read.get("error"):
            lines += [f"- сбой до выбора статьи: {read['error']}", ""]
    if state.get("cards") is not None or ideas_in_all(state):
        from qlab import ideas
        from qlab.agents.carder import report_lines as card_lines

        lines += ["## Входящие идеи и карточки", "",
                  *card_lines(state.get("cards"), ideas.load(), ideas_in_all(state))]
        for what, key in (("остановлено", "stopped"), ("сбой", "error")):
            if (state.get("cards") or {}).get(key):
                lines.append(f"- {what}: {state['cards'][key]}")
        lines.append("")
    if implemented := state.get("implemented"):
        from qlab import implement_run

        lines += ["## Реализация карточек агентом", ""]
        if any(e.get("runs") for e in implemented):
            lines += [
                "Идея из входящих, которой нужен новый код, за одну ночь доходит до исхода: агент "
                "пишет блок и спек, а дальше без агента и без токенов код прогоняет написанную "
                "версию, версию с выбором кода (раз в месяц и только по прошлому он выбирает между "
                "вариантами карточки на истории Binance с 2019 года), базовую стратегию на тех же "
                "данных, что и версия с выбором кода, и второй этап на Hyperliquid для прошедших "
                "(`docs/IMPLEMENTER.md`). Исход каждого прогона выносит стенд.", ""]
        for e in implemented:
            head = (f"- **{e['card']}** → {e.get('idea_id', '?')}: "
                    + ("прогоны продолжены с прошлой ночи" if e.get("resumed") else
                       f"{e.get('tokens', 0):,} токенов; ответ агента: {e.get('reply') or '—'}"))
            lines.append(head)
            for p in e.get("problems") or []:
                lines.append(f"    - проверка не пройдена: {p}")
            if e.get("problems") and e.get("attempts", 0) >= implement_run.MAX_ATTEMPTS:
                lines.append(f"    - попыток {e['attempts']} из {implement_run.MAX_ATTEMPTS}: "
                             "больше не пробуем, карточка ждёт владельца")
            for r in e.get("review") or []:
                lines.append(f"    - проверяющий: {r}")
            if e.get("stand"):
                lines.append(f"    - стенд: {e['stand']}")
            lines += implement_run.entry_lines(e)
            if e.get("stopped"):
                lines.append(f"    - остановлено: {e['stopped']}")
        lines.append("")
    if gen := state.get("generate"):
        from qlab.agents.generator import GeneratedResult, result_line, second_stage_lines

        lines += ["## Генератор идей", "",
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
        lines += [f"    - день фиксации назвал код: {n}" for n in gen.get("named_fixation") or []]
        for r in gen.get("rejected") or []:
            lines += [f"    - не принято `{r['path']}`: {p}" for p in r["problems"]]
        for r in gen.get("results") or []:
            lines.append(f"- {result_line(GeneratedResult(**r))}")
            lines += [f"    - {ln}" for ln in second_stage_lines(GeneratedResult(**r))]
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
    from qlab.publications.stage import report_lines as publication_lines

    lines += publication_lines(state)
    lines += _forward_chart_lines(day)
    lines += _year_signs_lines()
    lines += _protection_lines(day)
    lines += ["## Воронка агентов", "", *_funnel_lines(day), ""]
    tokens = _token_lines(day)
    if account.get("ok") is False:
        tokens += ("\n\nАккаунт Claude CLI не тот, какой назван в night/account.yaml: агенты "
                   "ночью ничего не потратили.")
    lines += ["## Токены", "", tokens, ""]
    path = REPORT_DIR / f"{day.isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _director_lines(entry: dict) -> list[str]:
    """The research director's memo and key decisions for the morning report
    (docs/DIRECTOR.md). What it proposes to close is the owner's to decide."""
    lines = ["Агент следит за направлениями поиска и подсказывает генератору идей и "
             "исследователю, куда смотреть. Код проверяет каждое его слово; закрывать "
             "направления он не может.", ""]
    if entry.get("error"):
        lines.append(f"- сбой этапа: {entry['error']}")
    if entry.get("stopped"):
        lines.append(f"- не запускался: {entry['stopped']}; продолжит в следующую ночь")
    if entry.get("memo"):
        lines.append(f"- записка `{entry['memo']}`, {entry.get('tokens', 0):,} токенов")
    lines += [f"    - {p}" for p in entry.get("problems") or []]
    d = entry.get("directions")
    if d is None:
        lines += [f"    - не принято: {p}" for p in entry.get("rejected") or []]
        return lines
    if d.get("pending_owner_review"):
        lines.append("- **направления ждут проверки владельца**: генератор и исследователь файл "
                     "`night/directions.yaml` пока не читают. Прочитать записку и снять пометку: "
                     "`uv run qlab director --approve`.")
    gen = d.get("generator") or {}
    if gen.get("prefer_mechanisms"):
        lines.append(f"- генератору предпочитать: {', '.join(gen['prefer_mechanisms'])}")
    lines += [f"- генератору приостановить `{p['id']}` — {p['why']}"
              for p in gen.get("pause_mechanisms") or []]
    lines += [f"- исследователь, {name}: важность {s['priority']}"
              + (f"; пометка: {s['note']}" if s.get("note") else "")
              for name, s in (d.get("researcher") or {}).items()]
    lines += [f"- просьба к каталогу ({r['family']}): {r['look_for']} — {r['why']}"
              for r in d.get("catalogue_requests") or []]
    lines += [f"- просьба о данных: {r['what']} — {r['why']}"
              for r in d.get("data_requests") or []]
    lines += [f"- **предлагает закрыть `{c['family']}` — решает владелец**: {c['why']}"
              for c in d.get("propose_close_family") or []]
    lines += [f"    - отброшено кодом: {p}" for p in entry.get("rejected") or []]
    return lines


def _paper_rule_lines(replay_lines: list[str]) -> list[str]:
    """How long each paper book has run against the owner's 4-6 weeks rule
    (qlab.paper_rule). A failure is a report line, not a failed night."""
    from qlab.paper_rule import RULE_TITLE, report_lines
    from qlab.registry.db import session_scope

    try:
        with session_scope() as session:
            return report_lines(session, replay_lines=replay_lines)
    except Exception as exc:  # noqa: BLE001
        return [f"**{RULE_TITLE}** не посчитано: {type(exc).__name__}: {exc}"]


CONSOLE_URL = "http://localhost:5890"
FORWARD_CHARTS_TITLE = "Графики прошедших идей"


def _forward_chart_lines(day: date) -> list[str]:
    """For each idea that passed every rule tonight (a `paper` run written on the report's day),
    where to see how it did on its forward test: the console's idea page draws one chart per
    stage (`qlab.forward_series`, docs/FORWARD_CHART.md). Markdown cannot draw a chart, so this
    is a link and a word on what is ready. Empty when nothing passed; a failure is one line."""
    from qlab import forward_series
    from qlab.registry.db import session_scope

    since = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    try:
        with session_scope() as session:
            passed = forward_series.passed_on(session, since, since + timedelta(days=1))
    except Exception as exc:  # noqa: BLE001
        return [f"## {FORWARD_CHARTS_TITLE}", "",
                f"Не посчитано: {type(exc).__name__}: {exc}", ""]
    if not passed:
        return []
    lines = [f"## {FORWARD_CHARTS_TITLE}", "",
             "Как идея отработала на проверке вперёд, день за днём и по этапам (деньги стратегии, "
             "проценты от её капитала). График рисует консоль на странице идеи; здесь только "
             "ссылка.", ""]
    for entry in passed:
        stages = entry["stages"]
        ready = [s for s in stages if s["has_series"]]
        names = ", ".join(
            (f"этап {s['stage']}" if len(stages) > 1 else "проверка вперёд")
            + (f" ({s['venue']})" if s["venue"] else "") for s in stages)
        state = ("графики готовы" if len(ready) == len(stages) else
                 "ряд по дням ещё не построен: `uv run qlab forward-series rebuild`")
        lines.append(f"- **{entry['idea_id']}**: {CONSOLE_URL}/ideas/{entry['idea_id']} — "
                     f"{names}; {state}")
    lines.append("")
    return lines


def _year_signs_lines() -> list[str]:
    """The four year signs counted over the registry (`qlab.year_signs`, docs/YEARS.md): a few
    lines with the totals and what they rest on; the table is `qlab year-signs`. A recording,
    not a verdict. Empty when no run has a year view yet; a failure is one line."""
    from qlab import year_signs
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    try:
        with session_scope() as session:
            body = year_signs.report_lines(session, load_latest())
    except Exception as exc:  # noqa: BLE001 - a report line, not a failed night
        return [f"## {year_signs.SECTION_TITLE}", "",
                f"Не посчитано: {type(exc).__name__}: {exc}", ""]
    return [f"## {year_signs.SECTION_TITLE}", "", *body, ""] if body else []


def _protection_lines(day: date) -> list[str]:
    """The protection readings of the Bv2 family (`qlab.protection`, docs/PROTECTION.md): the
    totals over the live ideas' latest measured runs and a line for each run of the night. A
    recording, not a verdict. Empty when no run carries a measurement; a failure is one line."""
    from qlab import protection
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    since = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    try:
        with session_scope() as session:
            body = protection.report_lines(session, load_latest(), since=since)
    except Exception as exc:  # noqa: BLE001 - a report line, not a failed night
        return [f"## {protection.SECTION_TITLE}", "",
                f"Не посчитано: {type(exc).__name__}: {exc}", ""]
    return [f"## {protection.SECTION_TITLE}", "", *body, ""] if body else []


def _funnel_lines(day: date) -> list[str]:
    """The agents' funnel and the price of one survivor, counted up to the end
    of the report's day (qlab.funnel)."""
    from qlab.funnel import agent_funnel, report_lines
    from qlab.registry.db import session_scope

    until = min(datetime.now(UTC),
                datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC))
    try:
        with session_scope() as session:
            funnel = agent_funnel(session, until)
    except Exception as exc:  # noqa: BLE001
        return [f"Воронка не посчитана: {type(exc).__name__}: {exc}"]
    lines = ["Что агенты нашли сами и сколько это стоило. «Выживший» — идея, прошедшая все "
             "правила стенда и ждущая решения владельца.", ""]
    for period in funnel.periods:
        lines += [*report_lines(period), ""]
    return lines[:-1]


def account_state() -> dict:
    """Which Claude account the CLI is logged in to, for the report (`claude auth status`
    costs no tokens). The agent stages refuse a wrong one themselves; this only records it."""
    from qlab.budget import WrongAccount
    from qlab.budget.agent import check_account

    try:
        return {"ok": True, "email": check_account().email}
    except WrongAccount as refusal:
        return {"ok": False, "found": refusal.found, "expected": refusal.expected,
                "message": str(refusal)}
    except Exception as exc:  # noqa: BLE001 - the report names it, the night goes on
        return {"ok": False, "found": None, "expected": None,
                "message": f"Аккаунт Claude CLI не проверен: {type(exc).__name__}: {exc}."}


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
        under = sorted({r.account or "не записан" for r in rows})
    if not by:
        return "Токены за эту ночь не тратились."
    week = (f"; недельное окно после последнего вызова — {last.seven_day_util:.0%}"
            if last is not None and last.seven_day_util is not None else "")
    return (f"Аккаунт: {', '.join(under)}. Потрачено: "
            + ", ".join(f"{k} {v:,}" for k, v in by.items()) + week)


@dataclass
class NightContext:
    """What a step of the night works with: the night's day and data end, its saved `state`, the
    stop times, and the flags `run` was called with (docs/PIPELINE.md). One object is shared by
    every step, but each step gets its own `state` through `for_step`: a private copy that the
    hub merges back (`qlab.nightflow.statehub`), so steps running together never share a dict.
    `state` here is the master; only the hub changes it once the steps have started."""

    day: date
    end: date
    state: dict
    noise_trials: int
    stop_at: datetime | None
    code_cut: datetime | None
    sweep: bool
    flags: dict
    run_twins: bool
    research_on: bool | None = None
    prefetcher: object = None
    report_path: Path | None = None
    hub: object = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def save(self) -> None:
        save_state(self.day, self.state)

    def for_step(self, step) -> StepContext | NightContext:
        """The context one step runs with: this one, with the step's own view of the state."""
        if self.hub is None:
            return self
        return StepContext(self, self.hub.view(step.id))

    def snapshot(self) -> dict:
        """The night's state as it stands, for a reader that is not a step."""
        return self.hub.snapshot() if self.hub is not None else self.state

    def finish_prefetch(self) -> None:
        """Stop the background fetch once (the stages that needed its data are done)."""
        with self._lock:
            prefetcher, self.prefetcher = self.prefetcher, None
        if prefetcher is None:
            return
        if self.hub is None:
            stop_prefetch(prefetcher, self.state)
            return
        scratch: dict = {}
        stop_prefetch(prefetcher, scratch)
        self.hub.set_key("prefetch", scratch.get("prefetch"))

    def wants_research(self) -> bool:
        """Whether the researcher's branch of the night is on tonight: asked for, or due."""
        with self._lock:
            if self.research_on is None:
                asked = self.flags["research"]
                self.research_on = bool(asked if asked is not None else research_due(
                    self.day, load_research_plan(), sweep=self.sweep))
            return self.research_on


class StepContext:
    """A `NightContext` as one step sees it: the same day, flags and stop times, and its own
    `state` (the step's view), saved through its own `save`."""

    def __init__(self, root: NightContext, state: dict) -> None:
        self._root = root
        self.state = state

    def save(self) -> None:
        save_state(self._root.day, self.state)

    def snapshot(self) -> dict:
        return self._root.snapshot()

    def release(self) -> None:
        """The step has ended: its view takes no further part in a stop of the night."""
        hub = getattr(self.state, "hub", None)
        if hub is not None:
            hub.close(self.state)

    def __getattr__(self, name: str):
        return getattr(self._root, name)


def _watch_specs() -> list[str]:
    return yaml.safe_load(WATCHLIST.read_text(encoding="utf-8"))["specs"]


def _flag(name: str):
    """`skip_reason` of a step an old command-line flag switches off."""
    return lambda ctx: None if ctx.flags[name] else "выключен флагом команды"


def _flag_and_not_in_state(name: str, key: str, done: str):
    def reason(ctx: NightContext) -> str | None:
        if not ctx.flags[name]:
            return "выключен флагом команды"
        return done if key in ctx.state else None
    return reason


def _cut_by(stop: str, name: str):
    """`is_cut` of a step that the night's stop time (`stop`: stop_at or code_cut) keeps from
    starting."""
    return lambda ctx: past_stop(getattr(ctx, stop), ctx.state, name)


def _step_detectors(ctx: NightContext) -> None:
    # Before the watch list: strategies on it read these files (docs/REGIME_DETECT.md); a failed
    # build keeps yesterday's file.
    from qlab.detector_builds import CONFIG, build_all

    ctx.state["detectors"] = build_all(ctx.end) if CONFIG.is_file() else []


def _detector_items(ctx: NightContext) -> list[tuple[str, str, str]]:
    return [(b["name"], "ok" if b["ok"] else "failed", b["note"])
            for b in ctx.state.get("detectors") or []]


def _detectors_all_failed(ctx: NightContext) -> str | None:
    built = ctx.state.get("detectors") or []
    if built and not any(b["ok"] for b in built):
        return "ни один определитель не собран: " + "; ".join(b["note"] for b in built)[:300]
    return None


def _step_recheck(ctx: NightContext) -> None:
    state, watch = ctx.state, _watch_specs()
    state["unwatched_bench"] = unwatched_bench(watch)
    queue = [("watch", Path(p)) for p in watch]
    if ctx.sweep:
        queue += [("graveyard", p) for p in graveyard_specs() if str(p) not in watch]
    for kind, path in queue:
        key = f"{kind}:{path}"
        if key in state["done"]:
            continue
        if past_stop(ctx.stop_at, state, "recheck"):
            break
        try:
            spec = load_spec(path)
        except Exception as exc:  # noqa: BLE001 - one broken spec file is its own row
            result = ItemResult(kind=kind, name=str(path), error=f"{type(exc).__name__}: {exc}")
        else:
            result = evaluate_item(kind, spec, end=ctx.end, noise_trials=ctx.noise_trials)
        state["results"] = [r for r in state["results"]
                            if not (r["kind"] == kind and r["name"] == result.name)]
        state["results"].append(asdict(result))
        if result.error is None:  # a failed item is tried again on the next run
            state["done"][key] = datetime.now(UTC).isoformat()
        ctx.save()


def _step_waiting(ctx: NightContext) -> None:
    # Recheck too: ideas waiting for data that are not on the watch list (qlab.waiting).
    from qlab.waiting import followed_ideas, waiting_stage

    watch = _watch_specs()
    try:
        waiting_stage(ctx.day, ctx.end, ctx.state, watch=watch, noise_trials=ctx.noise_trials)
    finally:
        # The ideas this stage follows leave the report's «не на списке» line. They are kept apart
        # from `unwatched_bench`, which is the recheck's key: each step writes only its own.
        ctx.state["waiting_followed"] = sorted(followed_ideas(ctx.state.get("waiting")))


def _step_twins(ctx: NightContext) -> None:
    # Ideas that wait for forward data get an answer from the history that exists: their causal
    # twin on Binance, then the second stage (qlab.twin_stage). Code only.
    from qlab.twin_stage import twins_stage

    twins_stage(ctx.day, ctx.end, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.stop_at)


def _step_restamp(ctx: NightContext) -> None:
    # Results measured an older way are measured again (docs/RESTAMP.md). Code only.
    restamp_stage(ctx.day, ctx.end, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.code_cut)


def _step_venue_gap(ctx: NightContext) -> None:
    # Why the two venues of an open idea differ (qlab.venue_gap). Code only.
    venue_gap_stage(ctx.day, ctx.end, ctx.state, stop_at=ctx.code_cut)


def _step_anomalies(ctx: NightContext) -> None:
    # Too-good results (docs/ANOMALIES.md): the code part, flags on the passed runs.
    anomalies_stage(ctx.day, ctx.end, ctx.state, stop_at=ctx.code_cut, agent=False)


def _step_anomalies_investigate(ctx: NightContext) -> None:
    # The investigator agent on the flagged runs, bounded and guarded.
    anomalies_stage(ctx.day, ctx.end, ctx.state, stop_at=ctx.code_cut, code=False)


def _anomaly_failures(part: str):
    def failures(ctx: NightContext) -> list[tuple[str, str, str]]:
        block = ctx.state.get("anomalies") or {}
        return [(str(e.get("idea") or e.get("trial")), "failed", str(e["error"])[:300])
                for e in block.get(part) or [] if e.get("error")]
    return failures


def _step_overlays(ctx: NightContext) -> None:
    # Overlays on the strategies that passed (qlab.overlay_stage). Code only.
    overlays_stage(ctx.day, ctx.end, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.code_cut)


def _step_paper(ctx: NightContext) -> None:
    ctx.state["paper"] = paper_reconciliation(ctx.day)


def _paper_swallowed(ctx: NightContext) -> str | None:
    lines = ctx.state.get("paper") or []
    return lines[0] if lines and lines[0].startswith("export from the production host failed") \
        else None


def _step_decay(ctx: NightContext) -> None:
    # After the watch list: the stand's panels are cached by then.
    decay_stage(ctx.day, ctx.end, ctx.state)


def _decay_swallowed(ctx: NightContext) -> str | None:
    found = ctx.state.get("decay") or {}
    if not found.get("results"):
        return next((ln for ln in found.get("lines") or [] if "не посчитано" in ln), None)
    return None


def _step_allocation(ctx: NightContext) -> None:
    # The capital split is a proposal for the owner (qlab.pool_allocation); it moves nothing.
    from qlab.pool_allocation import night_stage

    night_stage(ctx.day, ctx.end, ctx.state)


def _allocation_swallowed(ctx: NightContext) -> str | None:
    found = ctx.state.get("allocation") or {}
    if found.get("proposal") is None:
        return next((ln for ln in found.get("lines") or [] if "не посчитано" in ln), None)
    return None


def _step_assess(ctx: NightContext) -> None:
    assess_stage(ctx.day, ctx.state)


def _step_programmer(ctx: NightContext) -> None:
    # Before the director and the researcher (owner, 2026-10-30: «переноси токены на
    # программиста»).
    programmer_stage(ctx.day, ctx.state, stop_at=ctx.stop_at)


def _step_director(ctx: NightContext) -> None:
    # Before the researcher and the generator it steers.
    director_stage(ctx.day, ctx.state, force=ctx.flags["director"])


def _step_research(ctx: NightContext) -> None:
    research_stage(ctx.day, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.stop_at)


def _research_skip(ctx: NightContext) -> str | None:
    return None if ctx.wants_research() else "не по расписанию: исследователь сегодня не запрошен"


def _twins_research_skip(ctx: NightContext) -> str | None:
    if not ctx.wants_research():
        return "не по расписанию: исследователь сегодня не работает"
    return None if ctx.run_twins else "выключен флагом команды"


def _step_twins_research(ctx: NightContext) -> None:
    # The experiments the researcher has just run get their twins the same night.
    from qlab.twin_stage import twins_stage

    twins_stage(ctx.day, ctx.end, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.stop_at,
                label="twins-research")


def _step_implement(ctx: NightContext) -> None:
    implement_stage(ctx.day, ctx.state, noise_trials=ctx.noise_trials, stop_at=ctx.stop_at)


def _step_search(ctx: NightContext) -> None:
    search_stage(ctx.day, ctx.state)


def _step_catalogue(ctx: NightContext) -> None:
    # Before the generator: a mechanism catalogued tonight can be built tonight.
    catalogue_stage(ctx.day, ctx.state)


def _step_cards(ctx: NightContext) -> None:
    # After the implement stage on purpose: a card written tonight is implemented on the NEXT
    # night's implement stage, and tonight's search budget pays for the card.
    cards_stage(ctx.day, ctx.state)


def _step_generate(ctx: NightContext) -> None:
    # Last of the agents: the search for new ideas is what the budget cuts first, and the
    # generator is the most expensive part of it.
    generate_stage(ctx.day, ctx.state, noise_trials=ctx.noise_trials, force=ctx.flags["generate"])


def _generate_skip(ctx: NightContext) -> str | None:
    generate, search = ctx.flags["generate"], ctx.flags["search"]
    return None if generate is True or (search and generate is None) else "выключен флагом команды"


def _step_second_venue(ctx: NightContext) -> None:
    # Ideas rejected on Binance before the venues were judged independently get their Hyperliquid
    # side (qlab.second_venue). Code only, AFTER the agents, with its own time limit.
    second_venue_stage(ctx.day, ctx.end, ctx.state, noise_trials=ctx.noise_trials,
                       stop_at=ctx.stop_at)


def _step_ladder(ctx: NightContext) -> None:
    # The ladder of every open idea as one table, read after every stage has written its runs
    # (qlab.ladder, docs/LADDER.md); a reading, so it costs no token.
    ctx.finish_prefetch()
    lines = ladder_stage()
    if lines:
        ctx.state["ladder"] = lines


def _step_report(ctx: NightContext) -> None:
    ctx.finish_prefetch()
    ctx.save()  # with the stop line, if the stop time cut the night
    try:
        ctx.report_path = write_report(ctx.day, ctx.state)
    except Exception as exc:  # noqa: BLE001 - a report must exist in the morning
        path = REPORT_DIR / f"{ctx.day.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Ночной прогон {ctx.day.isoformat()}\n\nОтчёт не собрался: "
                        f"{type(exc).__name__}: {exc}. Состояние ночи цело: "
                        f"`{_state_path(ctx.day)}`.\n", encoding="utf-8")
        ctx.report_path = path
        raise


def _stage_note(name: str, ctx: NightContext) -> str | None:
    """What a code stage left undone tonight and why (too heavy, past its time), for the pipeline
    page. A note that cannot be made is no note: it never fails the step."""
    try:
        if name == "restamp":
            from qlab.restamp import page_note
        else:
            from qlab.second_venue import page_note
        return page_note(ctx.state.get(name))
    except Exception:  # noqa: BLE001 - a note is advice for the page
        return None


def night_steps() -> dict[str, StepImpl]:
    """How each step of `night/pipeline.yaml` is run. The key is the step's id; `error_name` is
    the stage's name in the state's `stage_errors` and in the stop line of the report."""
    flag, cut = _flag, _cut_by
    return {
        "detectors": StepImpl(
            _step_detectors, "detectors",
            skip_reason=_flag_and_not_in_state("detectors", "detectors",
                                               "уже собраны этой ночью (по состоянию ночи)"),
            inspect=_detectors_all_failed, finals=_detector_items),
        "recheck": StepImpl(_step_recheck, "recheck"),
        "waiting": StepImpl(_step_waiting, "waiting", skip_reason=flag("waiting"),
                            is_cut=cut("stop_at", "waiting")),
        "twins": StepImpl(_step_twins, "twins", skip_reason=lambda c: None if c.run_twins
                          else "выключен флагом команды", is_cut=cut("stop_at", "twins")),
        "restamp": StepImpl(_step_restamp, "restamp", skip_reason=flag("restamp"),
                            is_cut=cut("code_cut", "restamp"),
                            note=lambda c: _stage_note("restamp", c)),
        "venue_gap": StepImpl(_step_venue_gap, "venue_gap", skip_reason=flag("venue_gap"),
                              is_cut=cut("code_cut", "venue_gap")),
        "anomalies": StepImpl(_step_anomalies, "anomalies", skip_reason=flag("anomalies"),
                              is_cut=cut("code_cut", "anomalies"),
                              finals=_anomaly_failures("scans")),
        "anomalies-investigate": StepImpl(
            _step_anomalies_investigate, "anomalies-investigate", skip_reason=flag("anomalies"),
            is_cut=cut("code_cut", "anomalies"), finals=_anomaly_failures("investigations")),
        "overlays": StepImpl(_step_overlays, "overlays", skip_reason=lambda c: None if c.flags[
            "overlays"] else "выключен флагом команды", is_cut=cut("code_cut", "overlays")),
        "paper": StepImpl(
            _step_paper, "paper",
            skip_reason=_flag_and_not_in_state("paper", "paper",
                                               "уже сверено этой ночью (по состоянию ночи)"),
            is_cut=cut("stop_at", "paper"), inspect=_paper_swallowed),
        "decay": StepImpl(
            _step_decay, "decay",
            skip_reason=_flag_and_not_in_state("decay", "decay",
                                               "уже посчитано этой ночью (по состоянию ночи)"),
            is_cut=cut("stop_at", "decay"), inspect=_decay_swallowed),
        "allocation": StepImpl(
            _step_allocation, "allocation",
            skip_reason=lambda c: ("нет результата детектора угасания этой ночи"
                                   if "decay" not in c.state else
                                   "уже посчитано этой ночью (по состоянию ночи)"
                                   if "allocation" in c.state else None),
            is_cut=cut("stop_at", "allocation"), inspect=_allocation_swallowed),
        "assess": StepImpl(_step_assess, "assess", skip_reason=flag("assess"),
                           is_cut=cut("stop_at", "assess")),
        "programmer": StepImpl(_step_programmer, "programmer", skip_reason=flag("programmer"),
                               is_cut=cut("stop_at", "programmer")),
        "director": StepImpl(_step_director, "director", skip_reason=flag("director_on"),
                             is_cut=cut("stop_at", "director")),
        "research": StepImpl(_step_research, "research", skip_reason=_research_skip),
        "twins-research": StepImpl(_step_twins_research, "twins-research",
                                   skip_reason=_twins_research_skip,
                                   is_cut=cut("stop_at", "twins-research")),
        "implement": StepImpl(_step_implement, "implement", skip_reason=flag("implement"),
                              is_cut=cut("stop_at", "implement")),
        "search": StepImpl(_step_search, "search", skip_reason=flag("search"),
                           is_cut=cut("stop_at", "search")),
        "catalogue": StepImpl(
            _step_catalogue, "catalogue",
            skip_reason=lambda c: None if c.flags["search"] and c.flags["catalogue"]
            else "выключен флагом команды", is_cut=cut("stop_at", "catalogue")),
        "cards": StepImpl(
            _step_cards, "cards",
            skip_reason=lambda c: None if c.flags["search"] and c.flags["cards"]
            else "выключен флагом команды", is_cut=cut("stop_at", "cards")),
        "generate": StepImpl(_step_generate, "generate", skip_reason=_generate_skip,
                             is_cut=cut("stop_at", "generate")),
        "second_venue": StepImpl(
            _step_second_venue, "second_venue", skip_reason=lambda c: None if c.flags[
                "second_venue"] else "выключен флагом команды",
            is_cut=cut("stop_at", "second_venue"),
            note=lambda c: _stage_note("second_venue", c)),
        "ladder": StepImpl(_step_ladder, "ladder", skip_reason=flag("ladder")),
        "report": StepImpl(_step_report, "report"),
    }


def schedule_skips(day: date) -> dict[str, str]:
    """Steps that have nothing to do on `day` by their own cadence (the researcher, the director,
    the generator), for the console's preview. A reading; any failure to read a config means
    "no answer", never a guess."""
    out: dict[str, str] = {}
    try:
        if not research_due(day, load_research_plan(), sweep=day.weekday() == 6):
            out["research"] = "по расписанию исследователь сегодня не работает"
            out["twins-research"] = out["research"]
    except Exception:  # noqa: BLE001
        pass
    try:
        from qlab.directions import load_config

        cfg = load_config(DIRECTOR)
        if cfg is None or not director_due(day, cfg):
            out["director"] = "по расписанию руководитель сегодня не работает"
    except Exception:  # noqa: BLE001
        pass
    try:
        plan = yaml.safe_load(GENERATE.read_text(encoding="utf-8")) or {}
        if not generate_due(day, plan):
            out["generate"] = "по расписанию генератор сегодня не работает"
    except Exception:  # noqa: BLE001
        pass
    return out


def reset_steps(state: dict, table, step_ids: Collection[str]) -> None:
    """Forget what the night's state says these steps did, so that a restart does the work again
    instead of skipping what is marked done (the markers and the blocks named in the table)."""
    for step_id in step_ids:
        step = table.by_id[step_id]
        for key in step.state_keys:
            state.pop(key, None)
        if step.done_prefixes:
            state["done"] = {k: v for k, v in state.get("done", {}).items()
                             if not k.startswith(step.done_prefixes)}
        if step.id == "recheck":
            state["results"] = []
        state.get("stage_errors", {}).pop(step_id, None)


def run(day: date | None = None, *, graveyard: bool | None = None, paper: bool = True,
        implement: bool = True, search: bool = True, detectors: bool = True,
        assess: bool = True, research: bool | None = None, generate: bool | None = None,
        decay: bool | None = None, director: bool | None = None, waiting: bool = True,
        twins: bool | None = None, overlays: bool | None = None,
        anomalies: bool = True, restamp: bool = True, noise_trials: int = 200,
        stop_at: datetime | None = None, programmer: bool = True,
        catalogue: bool = True, cards: bool = True, ladder: bool | None = None,
        second_venue: bool | None = None, venue_gap: bool = True,
        journal=None, trigger: str = "hand", off: Collection[str] = (),
        only: Collection[str] | None = None, restart: str | None = None,
        concurrency: int = 1, command: str | None = None) -> Path:
    """The night's recheck stage; returns the report path. `graveyard`
    defaults to Sundays (weekly). `decay` (the decay detector, qlab.decay)
    follows `paper` unless given: it reads the paper books' export. `director`
    (the research director, docs/DIRECTOR.md) runs on its own cadence unless
    given, and not at all on a night with `research=False` (it steers the
    researcher and the generator; a night without the researcher is a partial
    one). `twins` (causal twins and the second stage, `qlab.twin_stage`) follows `waiting`
    unless given: it is the cheap, high-value part of the recheck, and runs right after the
    waiting stage and once more after the researcher. `overlays` (every overlay and pair on a
    strategy that passed, `qlab.overlay_stage`) follows `twins`, which it comes right after.
    `second_venue` (default: with the twins; `qlab.second_venue`, docs/VENUE_SPLIT.md) runs the
    Hyperliquid side of the oldest ideas rejected on Binance before the venues were judged
    independently, a few a night and within a time limit (night/second_venue.yaml); it stands
    AFTER the agent stages, so that a night that runs short cuts it, not the agents. The restamp,
    the venue gap, the anomalies and the overlays start nothing new after the deadline of
    night/schedule.yaml (`code_stages_cut`), for the same reason. `venue_gap` (`qlab.venue_gap`)
    measures why the two venues of an open idea differ, for its page and the report: code only.
    `restamp` (`qlab.restamp`: the ideas whose latest run was measured under an older stand or
    rules version, run again, bounded) comes right after the twins and before the anomalies, which
    then scan the new runs. `anomalies` (`qlab.anomalies.stage`: the code detector on the passed
    runs, then the investigator on the flagged ones, bounded and guarded) runs before the
    overlays, because an overlay base must have been scanned with no flag fired
    (`qlab.overlay_stage.anomaly_refusal`).
    `ladder` (default on; `qlab.ladder`) puts one table of every open idea's seven steps into the
    report, read last, after every stage has written its runs: no token, no verdict.
    Nothing new starts after `stop_at` (`past_stop`); the report says where the night stopped.

    The night is a table of steps (`night/pipeline.yaml`, docs/PIPELINE.md) that `nightflow.Runner`
    goes through in the table's order. A step that fails is a journal row and a line of the report;
    only the steps that depend on its output are blocked, the rest run. The flags above switch a
    step off the old way (nothing else is touched). `off` switches steps off the new way: the steps
    that depend on one of them by `data` are blocked unless it was done earlier the same night.
    `only` limits the run to the named steps (a restart), `restart` names the step restarted. The
    `journal` (`nightflow.journal`) records the run; without one nothing is recorded.
    `concurrency` is how many steps may run together (1: the table's order)."""
    import os
    import sys

    from qlab.nightflow.journal import NullJournal
    from qlab.nightflow.plan import make_plan
    from qlab.nightflow.runner import ItemTracker, Runner
    from qlab.nightflow.statehub import NightStateHub
    from qlab.nightflow.table import load_table

    journal = journal if journal is not None else NullJournal()
    day = day or datetime.now(UTC).date()
    end = last_closed_day(datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                          + timedelta(hours=12))
    state = load_state(day)
    code_cut = code_stages_cut(stop_at)  # the code stages' own, earlier, time (schedule.yaml)
    state.setdefault("results", [])
    state.pop("stopped_at", None)  # a resumed night gets its own stop time
    state.pop("deadline", None)  # and its own deadline for the code stages
    table = load_table()
    impls = night_steps()
    done_today = journal.done_today(day)
    plan = make_plan(table, off=off, only=only, done_today=done_today)
    running = [p.step for p in plan if p.action != "outside"]
    if only is None:
        state.pop("stage_errors", None)  # and its own failures: a stage that failed is tried again
    else:
        for step_id in running:  # a partial run forgets only the failures of the steps it redoes
            state.get("stage_errors", {}).pop(impls[step_id].error_name, None)
    if restart:
        reset_steps(state, table, [restart, *table.descendants(restart, ("data",))])
    state["account"] = account_state()  # the report says which account the agents ran under
    from qlab.budget.guard import limits_off

    state["limits_off"] = limits_off()  # the report says so when the guard's ceilings were off
    run_twins = twins if twins is not None else waiting
    director_on = director if director is not None else research is not False
    ctx = NightContext(
        day=day, end=end, state=state, noise_trials=noise_trials, stop_at=stop_at,
        code_cut=code_cut, run_twins=run_twins,
        sweep=graveyard if graveyard is not None else day.weekday() == 6,
        flags={"graveyard": graveyard, "paper": paper, "implement": implement, "search": search,
               "detectors": detectors, "assess": assess, "research": research,
               "generate": generate, "decay": decay if decay is not None else paper,
               "director": director, "director_on": director_on, "waiting": waiting,
               "overlays": overlays if overlays is not None else run_twins,
               "anomalies": anomalies, "restamp": restamp, "programmer": programmer,
               "catalogue": catalogue, "cards": cards, "ladder": ladder is not False,
               "second_venue": second_venue if second_venue is not None else run_twins,
               "venue_gap": venue_gap})
    # The data of the watch list is fetched in the background while the stages in front compute
    # (qlab.prefetch, docs/NIGHT_SPEED.md); the stages that need it wait for it, never fetch twice.
    if "recheck" in running and WATCHLIST.is_file():
        ctx.prefetcher = start_prefetch(end, state)
    run_id = journal.start_run(day, trigger=trigger, steps_off=list(off), only=(
        list(only) if only is not None else None), restart=restart, concurrency=concurrency,
        pid=os.getpid(), command=command)
    tracker = ItemTracker(table, journal)
    # From here on each step works on its own copy of the state and the hub merges it back
    # (qlab.nightflow.statehub); the master `state` is changed by the hub alone.
    hub = NightStateHub(state, table, write=lambda text: write_state_text(day, text),
                        observers=lambda: STATE_OBSERVERS,
                        strict=os.environ.get("QLAB_STATE_STRICT") == "1")
    ctx.hub = hub

    def note_failure(step_id: str, error: str) -> None:
        hub.record_error(impls[step_id].error_name, error)

    STATE_OBSERVERS.append(tracker.observe)
    outcome = "ok"
    try:
        runner = Runner(table, impls, ctx, journal=journal, run_id=run_id, off=off, only=only,
                        done_today=done_today, concurrency=concurrency, tracker=tracker,
                        on_failure=note_failure)
        results = runner.execute()
        if any(v in ("failed", "interrupted") for v in results.values()):
            outcome = "failed"
    except BaseException:
        outcome = "interrupted"
        raise
    finally:
        STATE_OBSERVERS.remove(tracker.observe)
        try:
            ctx.finish_prefetch()
            hub.save_open_views()  # a stopped night keeps what its steps had in hand
            hub.save_master()
            if hub.violations:
                print(f"[state] {len(hub.violations)} write(s) to keys a step does not own",
                      file=sys.stderr, flush=True)
        finally:
            journal.finish_run(run_id, outcome)
    return ctx.report_path or REPORT_DIR / f"{day.isoformat()}.md"


def start_prefetch(end: date, state: dict):
    """Start fetching the data of the watch list's items that have not run tonight, in their
    order (`qlab.prefetch`); None when it is off or the list cannot be read. The spec an item
    runs is `with_end(rerun_spec(spec), end)`, so that is the data it is asked for."""
    from qlab import prefetch

    if not prefetch.enabled() or not WATCHLIST.is_file():
        return None
    specs = []
    try:
        for path in yaml.safe_load(WATCHLIST.read_text(encoding="utf-8"))["specs"]:
            if f"watch:{path}" in state["done"]:
                continue
            try:
                specs.append(with_end(rerun_spec(load_spec(Path(path))), end))
            except Exception:  # noqa: BLE001 - a broken spec is its own row in the watch list run
                continue
    except Exception:  # noqa: BLE001 - no prefetch is no harm
        return None
    return prefetch.start(specs)


def stop_prefetch(prefetcher, state: dict) -> None:
    """Stop the prefetch thread (it finishes the fetch in hand, up to two minutes) and keep what
    it did in the night's state."""
    if prefetcher is None:
        return
    prefetcher.stop(timeout=120)
    state["prefetch"] = prefetcher.report()


def guarded(day: date, state: dict, name: str, fn, *args, **kwargs):
    """Run one stage; its failure is a line of the morning report, never the end of the night.
    The night of 2026-10-27 stopped at 04:51 on one stage's AttributeError and research, the
    generator and the programmer never ran. A failed stage left its done-markers unset, so the
    next run tries it again. A KeyboardInterrupt (not an Exception) still stops the night."""
    import traceback

    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        where = traceback.extract_tb(exc.__traceback__)[-1]
        text = f"{type(exc).__name__}: {exc} ({Path(where.filename).name}:{where.lineno})"
        hub = getattr(state, "hub", None)
        if hub is not None:  # a step's view: the line goes to the hub's master
            hub.record_error(name, text)
            return None
        state.setdefault("stage_errors", {})[name] = text
        save_state(day, state)
        return None


def ladder_stage() -> list[str]:
    """The morning report's table of every open idea's ladder (`qlab.ladder.report_lines`),
    reached through the night so that the suite can keep whole-night tests off the owner's
    registry (conftest.py)."""
    from qlab import ladder
    from qlab.registry.db import session_scope

    with session_scope() as session:
        return ladder.report_lines(session, ladder.load_ledgers())


def second_venue_stage(*args, **kwargs):
    """The queue of closed ideas whose Hyperliquid side was never run (`qlab.second_venue`),
    reached through the night so that the suite can keep whole-night tests off the owner's
    registry (conftest.py)."""
    from qlab import second_venue

    return second_venue.second_venue_stage(*args, **kwargs)


def venue_gap_stage(*args, **kwargs):
    """The measurement behind «почему площадки расходятся» (`qlab.venue_gap.venue_gap_stage`),
    reached through the night so that the suite can keep whole-night tests off the owner's
    registry (conftest.py)."""
    from qlab import venue_gap

    return venue_gap.venue_gap_stage(*args, **kwargs)


def restamp_stage(*args, **kwargs):
    """The restamp stage (`qlab.restamp.restamp_stage`), reached through the night so that the
    suite can keep whole-night tests off the owner's registry (conftest.py); a test of the stage
    calls the function of `qlab.restamp` itself."""
    from qlab import restamp

    return restamp.restamp_stage(*args, **kwargs)


def anomalies_stage(*args, **kwargs):
    """The anomalies stage (`qlab.anomalies.stage.anomalies_stage`), reached through the night so
    that the suite can keep whole-night tests off the owner's registry and the agent (conftest.py);
    a test of the stage calls the function of `qlab.anomalies.stage` itself."""
    from qlab.anomalies import stage

    return stage.anomalies_stage(*args, **kwargs)


def overlays_stage(*args, **kwargs):
    """The overlays stage (`qlab.overlay_stage.overlays_stage`), reached through the night so that
    the suite can keep the tests that run a whole night off the owner's real registry and files
    (conftest.py); a test of the stage calls the function of `qlab.overlay_stage` itself."""
    from qlab import overlay_stage

    return overlay_stage.overlays_stage(*args, **kwargs)


PROGRAMMER = Path("night/programmer.yaml")
CATALOGUER = Path("night/cataloguer.yaml")


def catalogue_stage(day: date, state: dict, *, config: Path = CATALOGUER) -> None:
    """The cataloguer agent (`qlab.agents.cataloguer`, docs/CATALOGUER.md): up to `per_night`
    papers read in full and written into the catalogue when code and the reviewer pass them, in
    the search stage's budget. Resumable: a paper done tonight is not repeated."""
    from dataclasses import asdict

    from qlab.agents import cataloguer as cat
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    if not config.is_file():
        return
    cfg = cat.load_config(config)
    if not cfg.get("enabled", False):
        return
    entry = state.setdefault("catalogue", {"entries": []})
    ledger = cat.load_ledger()
    done = {e["id"] for e in entry["entries"]}
    max_attempts = int(cfg.get("max_attempts", 3))
    for _ in range(int(cfg.get("per_night", 1)) - len(done)):
        paper = None
        try:
            with session_scope() as session:
                paper = cat.next_paper(ledger, session, max_attempts=max_attempts)
                if paper is None or paper.id in done:
                    break
                stage = NightBudget.open(session, probe(session).report.reading).stage(
                    Stage.SEARCH, candidates=1)
                cat.run_paper(paper, cfg, budget=stage.candidate(None), session=session, day=day)
        except (NightExhausted, BudgetExhausted) as stop:
            entry["stopped"] = str(stop)
            cat.save_ledger(ledger)
            save_state(day, state)
            return
        except Exception as exc:  # noqa: BLE001 - a report line, the night goes on
            if paper is None:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                break
            paper.problems.append(f"{type(exc).__name__}: {exc}")
            paper.status = "failed"
        entry["entries"].append(asdict(paper))
        done.add(paper.id)
        cat.save_ledger(ledger)
        save_state(day, state)


CARDER = Path("night/carder.yaml")


def ingest_memo_ideas(day: date, state: dict, memo: str | Path, source: str,
                      key: str = "ideas_in") -> dict:
    """The ideas of an agent's memo (its ```ideas block) into the inbox (`qlab.ideas`), and
    into the night's state for the report. Optional by design: a memo without the block adds
    nothing, and a failure here is one line in the state, never the end of the stage."""
    from qlab import ideas

    try:
        found = ideas.ingest_memo(Path(memo), source=source, day=day)
        if found["added"]:
            state.setdefault(key, []).extend(
                {"id": e["id"], "source": source, "text": e["text"]}
                for e in ideas.load() if e["id"] in found["added"])
        return found
    except Exception as exc:  # noqa: BLE001 - the memo is already safe on disk
        return {"added": [], "known": [], "dropped": [], "error": f"{type(exc).__name__}: {exc}"}


_CARDER_FIELDS = ("attempts", "problems", "review_blocks", "review_notes")


def _restore_idea(idea: dict, kept: dict) -> None:
    for key in _CARDER_FIELDS:
        if key in kept:
            idea[key] = kept[key]
        else:
            idea.pop(key, None)


def cards_stage(day: date, state: dict, *, config: Path = CARDER) -> None:
    """The carder agent (`qlab.agents.carder`, docs/IDEAS.md): up to `per_night` ideas of the
    inbox turned into a card for the implementer (or answered "repeat" / a question to the
    owner), in the search stage's budget. At most `max_waiting` cards of the carder wait for the
    implementer: while that many do, no idea is taken (and no tokens are spent). Resumable: an
    idea done tonight is not repeated."""
    from qlab import ideas
    from qlab import tried as tried_module
    from qlab.agents import carder
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    if not config.is_file():
        return
    cfg = carder.load_config(config)
    if not cfg.get("enabled", False):
        return
    entry = state.setdefault("cards", {"entries": []})
    inbox = ideas.load()
    entry["incoming"] = [{"id": e["id"], "source": e["source"], "text": e["text"]}
                         for e in inbox if str(e["date"]) == day.isoformat()
                         and e["id"] not in {i["id"] for i in ideas_in_all(state)}]
    if moved := carder.sync_implemented(inbox):
        entry["moved"] = moved
        ideas.save(inbox)
    max_waiting = int(cfg.get("max_waiting", 1))
    if len(waiting := carder.pending_cards(inbox)) >= max_waiting:
        entry["waiting_for"] = waiting[0]["card"]
        save_state(day, state)
        return
    registry = tried_module.load_tried()
    done = {e["id"] for e in entry["entries"]}
    max_attempts = int(cfg.get("max_attempts", 3))
    for _ in range(int(cfg.get("per_night", 1)) - len(done)):
        if len(waiting := carder.pending_cards(inbox)) >= max_waiting:
            entry["waiting_for"] = waiting[0]["card"]  # tonight's cards filled the places
            break
        idea = ideas.oldest_new(inbox, max_attempts=max_attempts)
        if idea is None or idea["id"] in done:
            break
        # An attempt cut short keeps the idea as it was: the earlier remarks are what the next
        # attempt edits the draft by, and a half-written draft is not one to edit.
        kept = {k: copy.deepcopy(idea[k]) for k in _CARDER_FIELDS if k in idea}
        draft = carder._snapshot(carder.DRAFTS / f"{idea['id']}.yaml")
        try:
            with session_scope() as session:
                stage = NightBudget.open(session, probe(session).report.reading).stage(
                    Stage.SEARCH, candidates=1)
                result = carder.run_idea(idea, inbox, cfg, budget=stage.candidate(None),
                                         session=session, day=day, registry=registry)
        except (NightExhausted, BudgetExhausted) as stop:
            _restore_idea(idea, kept)
            carder._restore(draft)
            entry["stopped"] = str(stop)
            ideas.save(inbox)
            save_state(day, state)
            return
        except Exception as exc:  # noqa: BLE001 - a report line, the night goes on
            _restore_idea(idea, kept)
            carder._restore(draft)
            idea["attempts"] = int(idea.get("attempts") or 0) + 1
            if idea["attempts"] >= max_attempts:
                ideas.set_status(idea, "needs-owner", "попытки карточника кончились сбоями; "
                                  f"последний: {type(exc).__name__}: {exc}")
            result = {"id": idea["id"], "source": idea["source"], "text": idea["text"],
                      "status": idea["status"], "problems": [f"{type(exc).__name__}: {exc}"],
                      "tokens": 0}
        entry["entries"].append(result)
        done.add(idea["id"])
        ideas.save(inbox)
        save_state(day, state)
    save_state(day, state)


def programmer_stage(day: date, state: dict, *, config: Path = PROGRAMMER,
                     stop_at: datetime | None = None) -> None:
    """The programmer agent (`qlab.agents.programmer`, docs/PROGRAMMER.md): the director's
    approved code requests are queued, then up to `per_night` are attempted one after another,
    each on its own branch, in the implement stage's budget. The stage runs BEFORE the director
    and the researcher (owner, 2026-10-30: «переноси токены на программиста»), so it has the
    night's first agent budget and time; the requests that give the generator a new block go
    first. The stage may spend `night_share` of what the night has left when it starts, divided
    among the night's requests (the planner, the writer and the reviewer of one request share
    its part), and takes no new request after the stop time. A request is a LOOP (owner,
    2026-10-31): the writer, code's checks and the reviewer repeat in one worktree until the
    checks pass and the reviewer accepts, at most `max_iterations` times; the iterations share the
    request's part of the allowance, and when it runs out the request stops where it is, its
    worktree and remarks kept, and the next night goes on from there. Nothing is merged: an
    accepted request waits for the owner's `qlab code merge`; one the planner stopped with an open
    question waits for `qlab code answer`, one that used its iterations for `qlab code retry`.
    Resumable: a request done tonight is not repeated."""
    from dataclasses import asdict

    from qlab.agents import programmer as prog
    from qlab.budget import (
        BudgetExhausted,
        CandidateBudget,
        CandidateExhausted,
        NightBudget,
        NightExhausted,
        Stage,
        probe,
    )
    from qlab.directions import active_directions
    from qlab.registry.db import session_scope

    if not config.is_file():
        return
    cfg = prog.load_config(config)
    if not cfg.get("enabled", False):
        return
    entry = state.setdefault("programmer", {"entries": []})
    queue = prog.load_queue()
    directions = active_directions()
    for r in (directions.code_requests if directions is not None else ()):
        prog.enqueue(queue, repo=r.repo, what=r.what, why=r.why, strategy=r.strategy,
                     source=f"research director, directions of {directions.date}", day=day)
    prog.save_queue(queue)
    done = {e["id"] for e in entry["entries"]}
    max_iterations = int(cfg.get("max_iterations", prog.DEFAULT_MAX_ITERATIONS))
    per_night = int(cfg.get("per_night", 1))
    share = float(cfg.get("night_share") or 1.0)
    for _ in range(per_night - len(done)):
        req = prog.next_request(queue, max_iterations, skip=done)
        if req is None:
            break
        if past_stop(stop_at, state, "programmer"):
            break
        try:
            with session_scope() as session:
                night_budget = NightBudget.open(session, probe(session).report.reading)
                night_budget.check()
                # The stage's allowance is fixed when it starts (a re-run of the stage in the
                # same night keeps it); each request gets an equal part of what is not spent.
                entry.setdefault("allowance", int(night_budget.tokens_left() * share))
                spent = sum(e.get("tokens", 0) for e in entry["entries"])
                cap = max(0, entry["allowance"] - spent) // max(1, per_night - len(done))
                prog.run_request(
                    req, cfg, session=session, budget=CandidateBudget(
                        night=night_budget, stage=Stage.IMPLEMENT, idea_id=None, cap_tokens=cap),
                    checkpoint=lambda _r: prog.save_queue(queue))
        except CandidateExhausted as stop:
            # This request's share of the stage's allowance is spent: the loop stops where it is
            # (its worktree and remarks are kept) and the next night goes on from there. The other
            # request of the night has its own share.
            req.stopped = (f"доля ночного бюджета на этот запрос израсходована после "
                           f"{req.iterations} итераций ({stop}); рабочая копия сохранена, "
                           "продолжит в следующую ночь")
        except (NightExhausted, BudgetExhausted) as stop:
            entry["stopped"] = str(stop)
            prog.save_queue(queue)
            save_state(day, state)
            return
        except Exception as exc:  # noqa: BLE001 - a report line, the night goes on
            prog.environment_stop(req, exc)
        entry["entries"].append(asdict(req))
        done.add(req.id)
        prog.save_queue(queue)
        save_state(day, state)


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
            except Exception as exc:  # noqa: BLE001 - a second opinion is never worth the night
                session.rollback()
                entry["stopped"] = f"сбой: {type(exc).__name__}: {exc}"
            out.append(entry)
            session.commit()
            save_state(day, state)
        session.commit()
    save_state(day, state)


RESEARCH = Path("night/research.yaml")
LIMITS = Path("night/limits.yaml")
NO_CAP = 1_000_000


def is_generator_idea(idea_id: str) -> bool:
    """Ideas the generator creates: `gen-*`."""
    return idea_id.startswith("gen-")


def _limit(limits: Path, key: str) -> int | None:
    """One number from night/limits.yaml; None when the file or the key is missing
    (the repository ships the file, so a missing one means no cap)."""
    if not limits.is_file():
        return None
    raw = (yaml.safe_load(limits.read_text(encoding="utf-8")) or {}).get(key)
    return None if raw is None else int(raw)


def load_research_plan(config: Path = RESEARCH) -> dict:
    """night/research.yaml as a dict; empty when the file is missing."""
    if not config.is_file():
        return {}
    return yaml.safe_load(config.read_text(encoding="utf-8")) or {}


def strategy_slots(session, research_plan: dict,
                   limits: Path = LIMITS) -> tuple[int, int, list[str]]:
    """(free places, the cap, the active strategies) under night/limits.yaml.
    Owner, 2026-10-25: the lab counts strategies, not ideas. The researcher's
    strategies (`research_plan["strategies"]`) always occupy their places; each
    active generator idea that is a root of its own is a new strategy and takes
    a free place. Variants (ideas with a parent) take none. Without the limits
    file there is no cap."""
    from qlab.registry.models import Idea
    from qlab.registry.strategies import ACTIVE_STATUSES, strategy_root

    plan = research_plan.get("strategies") or {}
    taken = {strategy_root(session, e["ideas"][0]) for e in plan.values() if e.get("ideas")}
    generated = sorted(
        i for (i,) in session.query(Idea.id).filter(Idea.status.in_(ACTIVE_STATUSES))
        if is_generator_idea(i) and strategy_root(session, i) == i and i not in taken)
    active = [*plan, *generated]
    cap = _limit(limits, "max_active_strategies")
    if cap is None:
        return NO_CAP, 0, active
    return max(0, cap - len(active)), cap, active


def variant_slots(session, strategy: dict, limits: Path = LIMITS) -> tuple[int, int, list[str]]:
    """(free places, the cap, the active variants) for one strategy of the
    research plan: the variants are the active branches under the root of the
    strategy's first idea -- a variant together with its own variants (the same
    idea on other data) takes one place. Without the limits file there is no cap."""
    from qlab.registry.strategies import active_branches, strategy_root

    variants = active_branches(session, strategy_root(session, strategy["ideas"][0]))
    cap = _limit(limits, "max_active_variants_per_strategy")
    if cap is None:
        return NO_CAP, 0, variants
    return max(0, cap - len(variants)), cap, variants


def plan_focus(plan: dict) -> str | None:
    """The plan's general focus. When `focus_decided` names a date, the quote is a decision
    the owner already took (and the researcher already answered): it is passed as a rule to
    work by, so nightly memos stop spending their first section re-answering it."""
    focus = plan.get("focus")
    if not focus or not plan.get("focus_decided"):
        return focus
    return (f"The owner's standing decision of {plan['focus_decided']}, already answered and in "
            f"effect -- work by it, do not answer or discuss it again in the memo: «{focus}»")


def owner_ideas_focus(ideas: list[str] | None) -> str | None:
    """The owner's own ideas for one strategy (night/research.yaml
    `owner_ideas`), verbatim, for the researcher to work through."""
    if not ideas:
        return None
    lines = "\n".join(f"- «{idea}»" for idea in ideas)
    return ("The owner's own ideas for this strategy, verbatim -- treat each as a hypothesis "
            "to test with an experiment spec (judged on data the fix did not see), or say in "
            f"the memo why it cannot be tested yet:\n{lines}")


def full_strategy_focus(name: str, cap: int, variants: list[str]) -> str:
    """The researcher's instruction when the strategy has no free variant place."""
    return (f"The strategy `{name}` already has {len(variants)} active variants and the lab keeps "
            f"at most {cap} per strategy ({', '.join(variants)}). Propose NO new experiment spec. "
            "Instead, in the memo, say which of these active variants you would drop and why, "
            "and what you would try in its place; write each variant you would drop as a line "
            "`DROP: <idea_id> — <why>` (the owner decides; you cannot drop anything).")


def split_drops(session, drops: list, variants: list[str], root: str,
                spec_dir: Path = Path("specs")) -> tuple[list[dict], list[dict], list[str]]:
    """The researcher's DROP proposals sorted for the owner: (proposals, closed,
    problems). Nothing here changes a status.

    - an active idea of this strategy -> a proposal (`qlab retire`);
    - a spec file on the bench whose idea is not in the registry (the bv2 owner-idea
      measurements, specs/bv2-owner-ideas/) -> a proposal too, carrying the spec path
      (removing the file is the owner's call);
    - an idea the stand or the owner already closed -> `closed`: nothing to do (on
      2026-10-26 four of six proposals were these, all reported as "not accepted");
    - an idea of another strategy, or a name that is neither an idea nor a spec ->
      a problem."""
    from qlab.registry.models import Idea
    from qlab.registry.strategies import ACTIVE_STATUSES, strategy_root

    proposals: list[dict] = []
    closed: list[dict] = []
    problems: list[str] = []
    specs: dict[str, str] | None = None
    for d in drops:
        idea = session.get(Idea, d.idea_id)
        if idea is not None and strategy_root(session, d.idea_id) != root:
            problems.append(f"DROP names `{d.idea_id}`, an idea of strategy "
                            f"{strategy_root(session, d.idea_id)}: ignored")
        elif idea is not None and idea.status not in ACTIVE_STATUSES and d.idea_id != root:
            # before the variants check: a variant the researcher was shown can be closed already
            # (2026-10-29: five of six proposals were ideas the owner retired the day before)
            closed.append({"idea": d.idea_id, "status": str(idea.status.value)})
        elif d.idea_id in variants or (idea is not None and d.idea_id != root):
            proposals.append({"idea": d.idea_id, "why": d.why})
        elif idea is not None:
            problems.append(f"DROP names `{d.idea_id}`, the strategy itself: ignored")
        else:
            specs = spec_idea_paths(spec_dir) if specs is None else specs
            if d.idea_id in specs:
                proposals.append({"idea": d.idea_id, "why": d.why, "spec": specs[d.idea_id]})
            else:
                problems.append(f"DROP names `{d.idea_id}`, which is neither an idea in the "
                                "registry nor a spec: ignored")
    return proposals, closed, problems


def spec_idea_paths(spec_dir: Path = Path("specs")) -> dict[str, str]:
    """idea_id -> spec path for every spec file under `spec_dir` (first one wins)."""
    import re

    found: dict[str, str] = {}
    for path in sorted(spec_dir.rglob("*.yaml")):
        match = re.search(r"^idea_id:\s*['\"]?([\w.-]+)", path.read_text(encoding="utf-8"),
                          re.MULTILINE)
        if match:
            found.setdefault(match.group(1), str(path))
    return found


def drop_focus(variants: list[str]) -> str:
    """What a DROP line may name, told to the researcher on every run."""
    return ("DROP lines may name: an active variant of this strategy ("
            + (", ".join(variants) or "none") + "), or a spec file on the bench whose idea is "
            "not in the registry. Ideas the stand or the owner already closed (rejected, "
            "retired, on the graveyard) need no DROP line.")


def latest_memo(name: str, memo_dir: Path = Path("docs/research")) -> Path | None:
    """The strategy's newest research memo, handed to its next run."""
    memos = sorted(memo_dir.glob(f"{name}-*.md"))
    return memos[-1] if memos else None


def research_slots(plan: dict) -> int:
    """How many strategies the researcher takes a night: `rotation: {per_night: N}` in
    night/research.yaml (owner, 2026-10-30: the researchers spent 8.8M tokens on variants of the
    same five strategies while new mechanisms waited for code), else all of them."""
    strategies = len(plan.get("strategies") or {})
    per_night = (plan.get("rotation") or {}).get("per_night")
    return min(strategies, int(per_night)) if per_night else strategies


def unfinished_research(day: date, plan: dict) -> list[str]:
    """The plan's strategies whose research did not finish on the previous
    night (the stage ran, a budget stop left them). They go first tonight. With a rotation
    only a strategy the night started and a stop cut counts: the ones that simply were not
    their turn are not unfinished."""
    yesterday = load_state(day - timedelta(days=1))
    if "research" not in yesterday:
        return []
    rotating = bool(plan.get("rotation"))
    stopped = {e["strategy"] for e in yesterday["research"] if e.get("stopped")}
    return [name for name in plan.get("strategies") or {}
            if f"research:{name}" not in yesterday.get("done", {})
            and (not rotating or name in stopped)]


def last_researched(day: date, names: list[str], *, lookback: int = 60) -> dict[str, date]:
    """The latest night before `day` on which each strategy's research finished (its done
    marker in the night's state), so a rotation keeps its turn from night to night."""
    found: dict[str, date] = {}
    for back in range(1, lookback + 1):
        night_day = day - timedelta(days=back)
        done = load_state(night_day).get("done") or {}
        for name in names:
            if name not in found and f"research:{name}" in done:
                found[name] = night_day
        if len(found) == len(names):
            break
    return found


def rotation_order(day: date, names: list[str], directions, unfinished: list[str],
                   last: dict[str, date]) -> list[str]:
    """The rotation's order of the waiting strategies: the ones a stop cut first; then those
    that were NOT researched last night (so a strategy the director marks high importance goes
    first, yet takes at most every second night and the others keep their turns); then high
    importance before normal before low; then the one researched longest ago, never first; then
    by name."""
    from qlab.directions import priority_rank

    first, yesterday = set(unfinished), day - timedelta(days=1)
    return sorted(names, key=lambda n: (
        n not in first, last.get(n) == yesterday, priority_rank(directions, n),
        last.get(n, date.min), n))


def carried_research(day: date, out: list[dict]) -> list[dict]:
    """The previous night's research entries the stop time cut after the memo:
    experiments not run on the stand, or the memo not yet noted in the registry.
    They run tonight before tonight's own, once (marked `carried_from`)."""
    yesterday = day - timedelta(days=1)
    have = {(e["strategy"], e.get("memo")) for e in out}
    return [{**e, "carried_from": yesterday.isoformat()}
            for e in load_state(yesterday).get("research", [])
            if e.get("memo") and (e["strategy"], e["memo"]) not in have
            and ("noted" not in e or (e.get("specs") and "stand" not in e))]


def research_due(day: date, plan: dict, *, sweep: bool) -> bool:
    """Whether the researcher runs tonight. `cadence: nightly` in
    night/research.yaml: every night. `weekly` (the default): on the weekly day
    (`sweep`), and also the night after a stop left a strategy unfinished -- a
    stop is the normal mode (CLAUDE.md), not a lost week."""
    if not plan.get("strategies"):
        return False
    cadence = plan.get("cadence", "weekly")
    if cadence == "nightly":
        return True
    if cadence != "weekly":
        raise ValueError(f"night/research.yaml: cadence must be nightly or weekly, got {cadence!r}")
    return sweep or bool(unfinished_research(day, plan))


def research_stage(day: date, state: dict, *, noise_trials: int,
                   config: Path = RESEARCH, stop_at: datetime | None = None) -> None:
    """The researcher on the strategies of `night/research.yaml` (all of them, or with a
    `rotation` the night's one: `rotation_order`), on its cadence (`research_due`), in the
    implement stage's budget (CLAUDE.md order: after rechecks and the programmer, before the
    search for new ideas); the stand then runs its experiments. Each strategy is researched at
    most once per night (the night's `done` marker); one the budget stopped goes first on the
    next night that runs the stage. Nothing new starts after `stop_at`; experiments
    the stop left unrun run first on the next night (`carried_research`)."""
    from qlab.agents.researcher import research, run_experiments
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope
    from qlab.registry.strategies import strategy_root

    if not config.is_file():
        return
    plan = yaml.safe_load(config.read_text(encoding="utf-8"))
    out = state.setdefault("research", [])
    out[:0] = carried_research(day, out)
    from qlab.directions import active_directions, order_strategies, researcher_focus

    strategies = plan.get("strategies") or {}
    # The research director's directions (docs/DIRECTOR.md), once the owner has reviewed
    # them: high-priority strategies go first, so a short budget cuts the low ones.
    directions = active_directions()
    waiting = [name for name in strategies if f"research:{name}" not in state["done"]]
    unfinished = unfinished_research(day, plan)
    if plan.get("rotation"):
        # One strategy a night (night/research.yaml `rotation`): the stage keeps the turns across
        # nights; a strategy already finished tonight takes one of tonight's places.
        places = research_slots(plan) - (len(strategies) - len(waiting))
        picked = rotation_order(day, waiting, directions, unfinished,
                                last_researched(day, waiting))[:max(0, places)]
    else:
        picked = order_strategies(waiting, directions, unfinished)
    todo = {name: strategies[name] for name in picked}
    if todo:
        # A run stopped earlier tonight tries again now: its old stop line goes.
        out[:] = [e for e in out if not (e.get("stopped") and e["strategy"] in todo)]
        with session_scope() as session:
            # `night_share`: with a rotation the one strategy of the night must not take all
            # that is left (the cards and the search come after), so its cap is the share.
            share = float(plan.get("night_share") or 0) or None
            stage = NightBudget.open(session, probe(session).report.reading).stage(
                Stage.IMPLEMENT, candidates=round(len(todo) / share) if share else len(todo))
            for name, e in todo.items():
                if past_stop(stop_at, state, f"research:{name}"):
                    break
                entry = {"strategy": name}
                free, cap, variants = variant_slots(session, e)
                docs = [Path(d) for d in e.get("docs") or [] if Path(d).is_file()]
                if (memo := latest_memo(name)) is not None:
                    docs.append(memo)
                try:
                    budget = stage.candidate(None)
                    outcome = research(name, idea_ids=list(e["ideas"]),
                                       specs=[Path(p) for p in e["specs"]], docs=docs,
                                       budget=budget, session=session,
                                       max_experiments=min(
                                           int(plan.get("max_experiments", 2)), free),
                                       focus="\n\n".join(filter(None, [
                                           plan_focus(plan),
                                           owner_ideas_focus(e.get("owner_ideas")),
                                           researcher_focus(name, directions),
                                           full_strategy_focus(name, cap, variants)
                                           if free == 0 else drop_focus(variants)])))
                    session.commit()
                    kept, closed, stray = split_drops(
                        session, outcome.drops, variants,
                        strategy_root(session, e["ideas"][0]))
                    entry.update(memo=str(outcome.memo or ""),
                                 specs=[str(p) for p in outcome.specs],
                                 problems=[*outcome.problems, *stray],
                                 drops=kept, closed=closed,
                                 tokens=budget.spent_tokens)
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
            if past_stop(stop_at, state, "experiments"):
                break
            entry["stand"] = run_experiments(Path(entry["memo"]), [Path(p) for p in entry["specs"]],
                                             name=entry["strategy"], noise_trials=noise_trials)
            save_state(day, state)
        if entry.get("memo") and "noted" not in entry:
            record_research_note(entry, plan)
            save_state(day, state)
        if entry.get("memo") and "ideas_in" not in entry:
            # An idea that needs new code goes to the inbox (qlab.ideas, docs/IDEAS.md).
            entry["ideas_in"] = ingest_memo_ideas(day, state, entry["memo"], "agent:researcher")
            save_state(day, state)


DIRECTOR = Path("night/director.yaml")


def last_director_day(day: date) -> date | None:
    """The latest night before `day` whose director call finished (its done-marker)."""
    found: date | None = None
    for path in STATE_DIR.glob("*.json"):
        try:
            night_day = date.fromisoformat(path.stem)
            finished = "director:agent" in (json.loads(path.read_text()).get("done") or {})
        except (ValueError, OSError):
            continue
        if finished and night_day < day and (found is None or night_day > found):
            found = night_day
    return found


def director_due(day: date, config) -> bool:
    """Whether the research director runs tonight: enabled, and `cadence_days` have
    passed since its last finished call (never run: due). A call the budget stopped
    is continued the next night -- a stop is the normal mode (CLAUDE.md)."""
    if not config.enabled:
        return False
    yesterday = load_state(day - timedelta(days=1))
    if (yesterday.get("director") or {}).get("stopped") and \
            "director:agent" not in yesterday.get("done", {}):
        return True
    last = last_director_day(day)
    return last is None or (day - last).days >= config.cadence_days


def director_stage(day: date, state: dict, *, config: Path = DIRECTOR,
                   force: bool | None = None) -> None:
    """The research director (`qlab.agents.director`) every `cadence_days` nights,
    before the researcher and the generator it steers, in the implement stage's budget
    (it steers the costliest work). Short of budget it is skipped with a line in the
    report; nothing else is cut for it. Its memo's block becomes
    `night/directions.yaml` (`qlab.directions`), waiting for the owner's review on the
    first run.

    Resumable: the agent's call (`director:agent`) and the writing of the directions
    (`director:directions`, no tokens) have their own done-markers. `force` overrides
    the config's cadence and `enabled` (the CLI's --director)."""
    from qlab.agents import director as director_module
    from qlab.agents.director import apply_memo, direct, known_names, summary
    from qlab.agents.generator import CATALOGUE, catalogue
    from qlab.budget import BudgetExhausted, NightBudget, Stage, probe
    from qlab.directions import load_config
    from qlab.registry.db import session_scope

    if force is False or (not config.is_file() and force is not True):
        return
    cfg = load_config(config)
    if cfg is None:
        return
    if force is None and not director_due(day, cfg):
        return
    if "director:directions" in state["done"]:
        return
    plan = load_research_plan()
    root = director_module.QLAB_ROOT
    entry = state.setdefault("director", {})
    try:
        if "director:agent" not in state["done"]:
            entry.pop("stopped", None)
            with session_scope() as session:
                try:
                    night_budget = NightBudget.open(session, probe(session).report.reading)
                    # One candidate of the implement stage's share among the researcher's
                    # strategies and itself: it must not take the whole of tonight.
                    budget = night_budget.stage(
                        Stage.IMPLEMENT, candidates=1 + research_slots(plan)
                    ).candidate(None)
                    outcome = direct(budget=budget, session=session, config=cfg, plan=plan)
                    entry.update(memo=str(outcome.memo or ""), problems=outcome.problems,
                                 reply=outcome.reply, tokens=budget.spent_tokens)
                    state["done"]["director:agent"] = datetime.now(UTC).isoformat()
                except BudgetExhausted as stop:
                    entry["stopped"] = str(stop)
                session.commit()
            save_state(day, state)
        if "director:agent" in state["done"] and "director:directions" not in state["done"]:
            if entry.get("memo") and "ideas_in" not in entry:
                entry["ideas_in"] = ingest_memo_ideas(day, state, root / entry["memo"],
                                                      "agent:director", key="ideas_in_director")
            if entry.get("memo"):
                mechanism_ids, strategy_names, family_names = known_names(
                    plan, cfg, catalogue(root / CATALOGUE))
                result = apply_memo(Path(entry["memo"]), root=root, mechanism_ids=mechanism_ids,
                                    strategy_names=strategy_names, family_names=family_names,
                                    day=day)
                entry["rejected"] = result.problems
                if result.directions is not None:
                    entry["directions"] = summary(result.directions)
            state["done"]["director:directions"] = datetime.now(UTC).isoformat()
    except Exception as exc:  # noqa: BLE001 - a steering stage must not fail the night
        entry["error"] = f"{type(exc).__name__}: {exc}"
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
    from qlab.publications.stage import fetch_stage, scout_stage
    from qlab.registry.db import session_scope
    from qlab.venue_watch import watch

    fetch_stage(day, state)  # new papers: code only, no tokens (qlab.publications)
    if "events" not in state:
        try:
            state["events"] = [e.__dict__ for e in watch(day)]
        except Exception as exc:  # noqa: BLE001 - the report says so, the night goes on
            state["events"] = []
            state["events_error"] = f"{type(exc).__name__}: {exc}"
        save_state(day, state)
    if "scout" in state:
        scout_stage(day, state)
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
    scout_stage(day, state)  # then a few abstracts of new papers, in the same share


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


def generate_stage(day: date, state: dict, *, noise_trials: int, config: Path = GENERATE,
                   force: bool | None = None) -> None:
    """The idea generator (`qlab.agents.generator`), in the search stage -- the
    one CLAUDE.md cuts first: one agent call writes up to `max_specs` specs from
    docs/MECHANISMS.md, code guards them, the stand runs those that pass.

    Resumable: the agent's call and each spec's run have their own done-markers
    in the night's state, so a restart goes on from where it stopped. `force`
    overrides the config's cadence and `enabled` (the CLI's --generate)."""
    from qlab.agents.generator import (
        CATALOGUE,
        GeneratedResult,
        GeneratedSpec,
        generate,
        run_generated_one,
        run_second_stage,
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
            free, cap, active = strategy_slots(session, load_research_plan())
        if free == 0:
            gen["skipped"] = (f"в работе уже {len(active)} стратегий при потолке {cap} "
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
                    max_own=int(plan.get("max_own", 1)),
                    data_end=_data_end_for_generation(session, day),
                    registry_idea_ids=[i for (i,) in session.query(Idea.id)])
                gen.update(
                    reply=outcome.reply, problems=outcome.problems, reverted=outcome.reverted,
                    named_fixation=outcome.notes, tokens=budget.spent_tokens,
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
        before = next((r for r in results if r["idea_id"] == entry["idea_id"]), None)
        if before and before.get("error") is None and (
                (before.get("stage2") or {}).get("status") == "error"):
            # Stage 1 is on record; only the second stage (code, no tokens) failed:
            # run it again, not the stage-1 trial and its noise a second time.
            result = GeneratedResult(**before)
            run_second_stage(item, result, noise_trials=noise_trials)
        else:
            result = run_generated_one(item, noise_trials=noise_trials)
        results[:] = [r for r in results if r["idea_id"] != result.idea_id]
        results.append(asdict(result))
        if result.error is None and (result.stage2 or {}).get("status") != "error":
            # a failed run is tried again on the next restart
            state["done"][key] = datetime.now(UTC).isoformat()
        save_state(day, state)


def implement_stage(day: date, state: dict, *, noise_trials: int, stop_at: datetime | None = None,
                    queue: Path | None = None, ledger_path: Path | None = None) -> None:
    """Stage 2 (CLAUDE.md order): cards from `night/implement_queue.yaml` written by the
    implementer agent and run through every guard, inside tonight's token budget; then, by code
    and outside the budget, the written spec, its causal version over the card's grid and the
    card's baseline on the same panel, and the second stage of every survivor
    (`qlab.implement_run`, docs/IMPLEMENTER.md). A night stop or a candidate's cap is not a
    failure: a card the agent did not reach stays queued, and a run cut by the stop time is done
    on the next night from the ledger (`data/implement/ledger.json`). A card whose block and
    runs are settled is not taken again; one the agent failed twice waits for the owner."""
    from qlab import implement_run, twin_stage
    from qlab.agents.implementer import implement, register_block
    from qlab.agents.reviewer import BLOCKING_KINDS, code_files, run_review
    from qlab.budget import BudgetExhausted, NightBudget, NightExhausted, Stage, probe
    from qlab.registry.db import session_scope

    ledger_path = ledger_path or implement_run.LEDGER
    ledger = twin_stage.load_ledger(ledger_path)
    ctx = implement_run.RunContext(
        day=day, end=last_closed_day(), noise_trials=noise_trials, state=state, ledger=ledger,
        ledger_path=ledger_path, stop_at=stop_at, stage2_config=implement_run.load_stage2_config())
    out = state.setdefault("implemented", [])
    seen = {e["card"]: e for e in out}

    # Runs left over from an earlier night: the agent is done with them, no tokens are needed.
    for card in implement_run.pending_cards(ledger):
        rec = ledger[card]
        entry = seen.get(card)
        if entry is None:
            entry = {**(rec.get("entry") or {}), "card": card, "resumed": True}
            out.append(entry)
        entry["runs"] = rec.setdefault("runs", {})
        if not implement_run.run_card(card, rec, ctx):
            entry["stopped"] = "время ночи вышло; остальные прогоны продолжатся следующей ночью"
        _stand_line(entry)
        ctx.save()

    queue = queue or IMPLEMENT_QUEUE
    if not queue.is_file():
        return
    cards = yaml.safe_load(queue.read_text(encoding="utf-8")).get("cards") or []
    pending = [c for c in cards if f"implement:{c}" not in state["done"]
               and implement_run.should_take(ledger.get(c))]
    if not pending:
        return
    fresh: list[dict] = []
    with session_scope() as session:
        night_budget = NightBudget.open(session, probe(session).report.reading)
        stage = night_budget.stage(Stage.IMPLEMENT, candidates=len(pending))
        for card in pending:
            if past_stop(stop_at, state, f"implement:{card}"):
                break
            entry = {"card": card}
            try:
                budget = stage.candidate(None)
                outcome = implement(Path(card), budget=budget, session=session)
                rec = ledger.setdefault(card, implement_run.new_record(outcome.idea_id, day))
                rec["attempts"] += 1
                entry.update(idea_id=outcome.idea_id, reply=outcome.agent_reply,
                             problems=outcome.problems, not_expressible=outcome.not_expressible,
                             attempts=rec["attempts"])
                if outcome.fields is not None:
                    entry["idea_text"] = outcome.fields.idea_text
                rec["status"] = "not-expressible" if outcome.not_expressible else "failed"
                if outcome.ok:
                    spec = load_spec(outcome.spec_path)
                    # The review row points at the idea: the idea comes first (2026-10-30: the
                    # row of 'bv2-fast-release-by-own-trend' failed its foreign key and the
                    # whole stage with it).
                    _ensure_idea(spec, card, session)
                    review = run_review(
                        spec=spec, spec_text=outcome.spec_path.read_text(encoding="utf-8"),
                        sources={card: Path(card).read_text(encoding="utf-8")},
                        code=code_files(spec), budget=budget, session=session)
                    entry["review"] = [f"[{f.kind}] {f.summary}" for f in review.accepted]
                    gaps = [f.summary for f in review.accepted
                            if f.kind in BLOCKING_KINDS and f.summary not in spec.review_answers]
                    if gaps:
                        rec["blocked"] = "; ".join(gaps)
                    else:
                        try:
                            registered = register_block(outcome)
                        except Exception as exc:  # noqa: BLE001 - the runs do not depend on it
                            registered = False
                            entry["problems"] = [*entry["problems"], "the block was not "
                                                 f"registered: {type(exc).__name__}: {exc}"]
                        if registered:
                            block = outcome.fields.new_block
                            entry["block"] = {
                                "name": block.name, "rule": block.rule,
                                "file": str(outcome.code_path), "reused": outcome.block_reused,
                                "params": [{"name": p.name, "meaning": p.meaning}
                                           for p in block.params]}
                    rec["idea_id"] = outcome.idea_id
                    rec["status"] = "pending-run"
                    rec["entry"] = dict(entry)
                entry["tokens"] = budget.spent_tokens
                state["done"][f"implement:{card}"] = datetime.now(UTC).isoformat()
            except NightExhausted as stop:
                entry["stopped"] = f"night budget: {stop}"
                out.append(entry)
                save_state(day, state)
                session.commit()
                break
            except BudgetExhausted as stop:
                entry["stopped"] = f"candidate budget: {stop}"
            out.append(entry)
            fresh.append(entry)
            twin_stage.save_ledger(ledger, ledger_path)
            save_state(day, state)
            session.commit()
    # The stand runs in its own transactions, so after the agent's session is closed.
    for entry in fresh:
        rec = ledger.get(entry["card"])
        if rec is None or rec.get("status") != "pending-run":
            continue
        entry["runs"] = rec["runs"]
        if not implement_run.run_card(entry["card"], rec, ctx):
            entry["stopped"] = "время ночи вышло; остальные прогоны продолжатся следующей ночью"
        _stand_line(entry)
        ctx.save()


def _stand_line(entry: dict) -> None:
    """The written spec's own verdict as one line (what the report printed before the extra runs
    existed)."""
    spec_run = (entry.get("runs") or {}).get("spec") or {}
    if spec_run.get("action") == "ran":
        entry["stand"] = spec_run.get("error") or f"{spec_run.get('route')}: " \
            f"{spec_run.get('route_reason')}"
    elif spec_run.get("action") == "error":
        entry["stand"] = str(spec_run.get("error"))


def _ensure_idea(spec: StrategySpec, card: str, session=None) -> None:
    """The idea of a written spec, in the registry. With a `session` it is written in that one
    (the implement stage's agent session): its other writes wait in the same transaction, and a
    second connection would find the database locked."""
    from qlab.registry import repo
    from qlab.registry.db import session_scope
    from qlab.registry.models import AssetClass, Idea, Profile, SourceType

    def make(s) -> None:
        if s.get(Idea, spec.idea_id) is None:
            repo.upsert_idea(s, id=spec.idea_id, title=spec.title,
                             source_type=SourceType.INTERNAL, asset_class=AssetClass.CRYPTO_PERP,
                             profile=Profile.OTHER,
                             notes=f"Implemented by the implementer agent from {card}.")

    if session is not None:
        make(session)
        session.flush()
        return
    with session_scope() as own:
        make(own)


__all__ = ["ItemResult", "decay_stage", "director_due", "director_stage", "evaluate_item",
           "generate_due", "generate_stage",
           "graveyard_specs", "last_closed_day",
           "paper_reconciliation", "run", "with_end", "write_report"]
