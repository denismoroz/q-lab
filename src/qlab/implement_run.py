"""What happens to an implemented card after the guards pass: its verdict by the next morning
(docs/IMPLEMENTER.md, «Из идеи в блок и в проверку за одну ночь»).

Owner, 2026-10-27: the framework itself turns an idea that needs NEW code into a verdict by the
next morning. The implementer agent writes the block and the spec (`qlab.agents.implementer`);
this module is everything after it, and it is CODE only: no agent, no tokens, so it sits outside
the budget guard by construction. Five steps, in this order, each a run that writes its trials
(the deflation counts every one of them):

1. `spec`         -- the spec as written, on the stand, matched noise as configured;
2. `causal`       -- the CAUSAL version over the card's declared grid: `Retune` picks the
                     variant every month from the past only, expanding window, after-cost Sharpe,
                     on Binance's history from 2019-09-08 (or the card's `honest_from`)
                     (`qlab.twin.causal_twin(grid=...)`; a hand-picked coin list is judged only
                     from the day the card says it was known);
3. `baseline`     -- the card's baseline spec re-run on exactly the causal version's panel, costs
                     and judged part (`qlab.twin.same_panel_spec`), so the report compares like
                     with like; with no causal version, on the written spec's panel;
4. `stage2-spec`, `stage2-causal` -- the second stage on Hyperliquid
                     (`qlab.pipeline.stage2.run_stage2`, a consistency check with stage 1, not a
                     measurement) for every survivor of step 1 or 2.

The heavy-data guard of the twins stage applies to every step (`night/twins.yaml`): an hourly
panel of every Binance perpetual is not fetched in a night. The night's stop time is checked
before each step; a step that is cut, or that failed, is kept in the ledger
(`data/implement/ledger.json`) and done on the next night, where the previous steps' results are
reused as they are. A reviewer's verified gap (the code does not express the card) runs step 1
only: the stand answers "not evaluable" and the rest would be about a different strategy.

Nothing here judges: every verdict is the stand's."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from qlab import twin, twin_stage
from qlab.card_fields import CardFieldError, CardFields, read_card_fields
from qlab.pipeline.spec import StrategySpec, load_spec

LEDGER = Path("data/implement/ledger.json")
BASELINE_DIR = Path("data/implement/baselines")
SPEC_DIR = Path("specs/agent")
BASE_SUFFIX = "-base"
MAX_ATTEMPTS = 2
"""Agent attempts at one card; after two failures the card waits for the owner (as the
programmer's requests do, docs/PROGRAMMER.md)."""
MAX_STEP_TRIES = 2
"""A step that failed with an error is tried again on the next night, once."""
STEPS = ("spec", "causal", "baseline", "stage2-spec", "stage2-causal")
FINAL = ("complete", "not-expressible", "blocked")
"""Ledger statuses after which a card is not taken again."""


# --------------------------------------------------------------------------
# The ledger: one record per card


def new_record(idea_id: str, day: date) -> dict:
    return {"idea_id": idea_id, "attempts": 0, "status": "new", "day": day.isoformat(),
            "runs": {}}


def spec_file(idea_id: str) -> Path:
    return SPEC_DIR / f"{idea_id}.yaml"


def pending_cards(ledger: Mapping[str, Mapping]) -> list[str]:
    """Cards whose agent step is done and whose runs are not: they go first, tonight."""
    return [card for card, rec in ledger.items() if rec.get("status") == "pending-run"]


def waits_for_owner(rec: Mapping | None) -> bool:
    """A card the agent failed on `MAX_ATTEMPTS` times."""
    return bool(rec) and rec.get("status") == "failed" and rec.get("attempts", 0) >= MAX_ATTEMPTS


def should_take(rec: Mapping | None) -> bool:
    """Whether the agent should be called on a queued card."""
    if rec is None:
        return True
    return rec.get("status") not in (*FINAL, "pending-run") and not waits_for_owner(rec)


# --------------------------------------------------------------------------
# The context of one night's runs


@dataclass
class RunContext:
    day: date
    end: date
    noise_trials: int
    deployable_capital_usd: float
    state: dict
    ledger: dict
    ledger_path: Path = LEDGER
    stop_at: datetime | None = None
    twins: twin_stage.Config | None = None
    stage2_config: object = None
    allow_heavy: bool = False
    venue_names: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.twins is None:
            self.twins = twin_stage.load_config() or twin_stage.Config(
                enabled=True, max_new_twins_per_night=0, slow_twin_minutes=20.0)

    def save(self) -> None:
        from qlab import night

        twin_stage.save_ledger(self.ledger, self.ledger_path)
        night.save_state(self.day, self.state)

    def listing(self) -> list[str]:
        if self.venue_names is None:
            self.venue_names = twin_stage._binance_listing()
        return self.venue_names


def load_stage2_config():
    """The second stage's config, or None when it cannot be read (the step is then skipped
    with a note, the stage 1 results stand)."""
    from qlab.pipeline import stage2

    try:
        return stage2.load_config()
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _heavy(ctx: RunContext, spec: StrategySpec) -> str | None:
    if ctx.allow_heavy:
        return None
    cfg = ctx.twins
    return twin_stage.heavy_data(spec, cfg.slow_twin_minutes, cfg.max_hourly_instruments,
                                 cfg.seconds_per_instrument)


def _data_text(spec: StrategySpec) -> str:
    d = spec.data
    text = f"{d.source}, {d.interval}, {d.start} … {d.end}"
    return text + (f"; судится с {spec.judged_from}" if spec.judged_from else "")


def _cite(card: str) -> str:
    return Path(card).as_posix()


# --------------------------------------------------------------------------
# The steps. Each returns the step's result and never raises.


def _step_spec(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
               fields: CardFields | None) -> dict:
    from qlab import night

    entry: dict = {"idea": spec.idea_id, "action": "ran", "data": _data_text(spec)}
    heavy = _heavy(ctx, spec)
    if heavy:
        entry.update(action="heavy", reason=heavy)
        return entry
    twin_stage.stage1_on_stand(
        spec, entry, ensure_idea=lambda: night._ensure_idea(spec, card), end=ctx.end,
        noise_trials=ctx.noise_trials, deployable_capital_usd=ctx.deployable_capital_usd,
        slow_minutes=ctx.twins.slow_twin_minutes, kind="implemented")
    return entry


def _causal_spec(spec: StrategySpec, runs: Mapping) -> StrategySpec | None:
    """The causal version as the file holds it, once step 2 made it."""
    if (runs.get("causal") or {}).get("action") != "ran":
        return None
    return load_spec(twin.twin_path(spec_file(spec.idea_id)))


def _step_causal(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
                 fields: CardFields | None) -> dict:
    from qlab import night, own_numbers

    cfg = ctx.twins
    entry: dict = {"idea": f"{spec.idea_id}{twin.TWIN_SUFFIX}", "action": "none"}
    names = ctx.listing() if twin.needs_listing(
        spec, None, coin_params=cfg.coin_list_params) else None
    grid = None
    if fields is not None and fields.grid is not None:
        grid, why = twin.card_grid(
            spec, axes=dict(fields.grid.axes) or None,
            candidates=dict(fields.grid.candidates) or None, cite=_cite(card),
            max_candidates=int(own_numbers.load_config()["max_candidates"]))
        if grid is None:
            entry["reason"] = why
            return entry
    honest = exec_venue = None
    if fields is not None and fields.data is not None:
        if fields.data.honest_from is not None:
            honest = (fields.data.honest_from, _cite(card))
        if fields.data.venue != twin.TWIN_VENUE:
            exec_venue = fields.data.venue
    result = twin.causal_twin(
        spec, spec_path=spec_file(spec.idea_id), venue_names=names,
        coin_params=cfg.coin_list_params, seconds_per_instrument=cfg.seconds_per_instrument,
        grid=grid, honest_from=honest, execution_venue=exec_venue)
    if result.spec is None:
        entry["reason"] = result.reason
        return entry
    path = twin.twin_path(spec_file(spec.idea_id))
    made = load_spec(path) if path.is_file() else result.spec  # a file is never rebuilt
    if not path.is_file():
        twin.write_twin(spec_file(spec.idea_id), made, spec.idea_id)
    entry.update(
        action="ran", kind=result.kind, candidates=list(result.candidates),
        twin_spec=path.as_posix(), data=_data_text(made),
        first_choice=made.params_fixed_at.isoformat() if made.params_fixed_at else None,
        instruments=list(result.instruments), list_source=result.list_source)
    heavy = _heavy(ctx, made)
    if heavy:
        entry.update(action="heavy", reason=heavy)
        return entry

    def ensure() -> None:
        night._ensure_idea(spec, card)
        twin_stage._ensure_twin_idea(made, spec.idea_id)

    twin_stage.stage1_on_stand(
        made, entry, ensure_idea=ensure, end=ctx.end, noise_trials=ctx.noise_trials,
        deployable_capital_usd=ctx.deployable_capital_usd,
        slow_minutes=ctx.twins.slow_twin_minutes, kind="implemented")
    return entry


def _step_baseline(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
                   fields: CardFields | None) -> dict:
    if fields is None or fields.baseline_spec is None:
        return {"action": "skipped", "reason": "у карточки нет базовой стратегии"}
    baseline = load_spec(fields.baseline_spec)
    like = _causal_spec(spec, rec["runs"])
    like_file = twin.twin_path(spec_file(spec.idea_id)) if like is not None \
        else spec_file(spec.idea_id)
    like = like or spec
    names = ctx.listing() if baseline.required_instruments \
        and baseline.data.source != twin.TWIN_VENUE else None
    base = twin.same_panel_spec(baseline, like, idea_id=f"{spec.idea_id}{BASE_SUFFIX}",
                                cite=like_file.as_posix(), venue_names=names)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    (BASELINE_DIR / f"{base.idea_id}.yaml").write_text(
        yaml.safe_dump(base.model_dump(mode="json", exclude_none=True), sort_keys=False,
                       allow_unicode=True, width=100), encoding="utf-8")
    entry: dict = {"idea": base.idea_id, "baseline": baseline.idea_id, "action": "ran",
                   "on_panel_of": like.idea_id, "data": _data_text(base)}
    heavy = _heavy(ctx, base)
    if heavy:
        entry.update(action="heavy", reason=heavy)
        return entry
    note = (f"The baseline {baseline.idea_id} on the panel of {like.idea_id} "
            "(qlab.implement_run, docs/IMPLEMENTER.md).")

    def ensure() -> None:
        twin_stage.ensure_variant_idea(base, spec.idea_id, note)

    twin_stage.stage1_on_stand(
        base, entry, ensure_idea=ensure, end=ctx.end, noise_trials=ctx.noise_trials,
        deployable_capital_usd=ctx.deployable_capital_usd,
        slow_minutes=ctx.twins.slow_twin_minutes, kind="implemented")
    return entry


def _step_stage2(which: str) -> Callable[..., dict]:
    def step(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
             fields: CardFields | None) -> dict:
        source = (rec["runs"].get(which) or {})
        if source.get("action") != "ran":
            if _needs_run(source):  # the run it follows failed and is tried again tomorrow
                return {"action": "waiting", "reason": "эта версия упала и будет прогнана ещё раз"}
            return {"action": "skipped", "reason": "эта версия не прогнана"}
        target = spec if which == "spec" else _causal_spec(spec, rec["runs"])
        if not twin_stage.goes_to_stage2(target):
            return {"action": "skipped",
                    "reason": f"данные уже с {target.data.source}, второму этапу нечего проверять"}
        if ctx.stage2_config is None:
            return {"action": "skipped", "reason": "настройки второго этапа не прочитаны или "
                    "он выключен (`night/stage2.yaml`)"}
        holder = {"route": source.get("route"), "trial": source.get("trial")}
        twin_stage.second_stage(
            target, holder, end=ctx.end, noise_trials=ctx.noise_trials,
            deployable_capital_usd=ctx.deployable_capital_usd, config=ctx.stage2_config)
        if "stage2" not in holder:
            return {"action": "skipped", "route": source.get("route"),
                    "reason": f"на первом этапе исход «{twin_stage.route_ru(source.get('route'))}»"
                              ", на второй идут только прошедшие"}
        out = {"action": "ran", "route": source.get("route"), "stage2": holder["stage2"],
               "seconds": holder.get("stage2_seconds")}
        if holder["stage2"].get("status") == "error":
            out.update(action="error", error=holder["stage2"].get("error"))
        return out
    return step


_STEP: dict[str, Callable[..., dict]] = {
    "spec": _step_spec, "causal": _step_causal, "baseline": _step_baseline,
    "stage2-spec": _step_stage2("spec"), "stage2-causal": _step_stage2("causal"),
}


def _needs_run(done: Mapping | None) -> bool:
    """Whether a step is still to do: never done, waiting for a run that failed, or failed
    itself and not yet tried `MAX_STEP_TRIES` times."""
    if done is None or done.get("action") == "waiting":
        return True
    return done.get("action") == "error" and done.get("tries", 1) < MAX_STEP_TRIES


def read_fields(card: str) -> CardFields | None:
    """The card's inbox fields, None for a card that has none."""
    try:
        fields = read_card_fields(card)
    except (CardFieldError, OSError):
        return None
    return fields if fields.inbox else None


def run_card(card: str, rec: dict, ctx: RunContext) -> bool:
    """Do the steps of one card that are not done. True when every step is settled; False when
    the night's stop time cut it (the rest waits in the ledger). The ledger and the night's
    state are saved after each step."""
    from qlab import night

    spec = load_spec(spec_file(rec["idea_id"]))
    fields = read_fields(card)
    runs = rec.setdefault("runs", {})
    for name in STEPS:
        if not _needs_run(runs.get(name)):
            continue
        blocked = rec.get("blocked")
        if blocked and name != "spec":
            runs[name] = {"action": "skipped", "reason": "проверяющий нашёл пробел между "
                          "карточкой и кодом: прогоны о другой стратегии не нужны"}
            continue
        if night.past_stop(ctx.stop_at, ctx.state, f"implement:{card}:{name}"):
            return False
        tries = (runs.get(name) or {}).get("tries", 0) + 1
        try:
            result = _STEP[name](ctx, card, rec, spec, fields)
        except Exception as exc:  # noqa: BLE001 - one step's failure must not stop the others
            result = {"action": "error", "error": f"{type(exc).__name__}: {exc}"}
        result["tries"] = tries
        runs[name] = result
        ctx.save()
    if any(_needs_run(runs.get(name)) for name in STEPS):
        rec["status"] = "pending-run"  # a step that failed is tried again on the next night
        return True
    rec["status"] = "blocked" if rec.get("blocked") else "complete"
    return True


# --------------------------------------------------------------------------
# The morning report


def _cell(metrics: Mapping, key: str, fmt: Callable[[object], str]) -> str:
    return fmt(metrics[key]) if metrics.get(key) is not None else "—"


def _sharpe(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, int | float) else "—"


LABELS = {"baseline": "база", "spec": "написанная", "causal": "с выбором кода"}


def _label(step: str, r: Mapping) -> str:
    if step == "baseline":
        return f"база `{r.get('baseline')}`"
    return f"{LABELS[step]} `{r.get('idea')}`"


def table_lines(runs: Mapping[str, Mapping]) -> list[str]:
    """The comparison: baseline, written spec, causal version -- per year, Sharpe, drawdown, the
    return without the best 1% of days, the route -- plain Russian."""
    rows = []
    for step in ("baseline", "spec", "causal"):
        r = runs.get(step)
        if not r or r.get("action") == "skipped":
            continue  # a version that was never due (no baseline on the card) has no row
        m = r.get("metrics") or {} if r.get("action") == "ran" else {}
        rows.append(
            f"| {_label(step, r)} | {r.get('data') or '—'} | "
            f"{_cell(m, 'ann_return_net', twin_stage.pct)} | "
            f"{_cell(m, 'sharpe_net', _sharpe)} | {_cell(m, 'max_dd', twin_stage.pct)} | "
            f"{_cell(m, 'ann_return_net_ex_best_1pct', twin_stage.pct)} | "
            f"{twin_stage.route_ru(r.get('route')) if m else '—'} |")
    if not rows:
        return []
    return ["", "    | версия | данные | в год | Шарп | просадка | без 1% лучших дней | исход |",
            "    |---|---|---|---|---|---|---|", *(f"    {row}" for row in rows), ""]


def _why_not(step: str, r: Mapping) -> str | None:
    action = r.get("action")
    name = LABELS.get(step, step)
    if action == "ran":
        return None
    if action == "none":
        return (f"{name}: не сделана — {r.get('reason')} (причина — слова кода)")
    if action == "heavy":
        return (f"{name}: не прогнана, данные слишком тяжёлые для одной ночи — "
                f"{r.get('reason')} (слова кода); вручную: `qlab twins run --allow-heavy`")
    if action == "error":
        return f"{name}: сбой — {str(r.get('error'))[:200]}; повторится в следующую ночь"
    return f"{name}: не прогнана — {r.get('reason')}"


def entry_lines(e: Mapping) -> list[str]:
    """What the implement section adds for one card once the agent's own lines are printed:
    the idea, the block, the comparison table, the years, the second stage."""
    from qlab.pipeline.stage2 import report_lines as stage2_lines
    from qlab.year_view import years_line

    out: list[str] = []
    if e.get("idea_text"):
        out.append(f"    - идея: {e['idea_text']}")
    block = e.get("block")
    if block:
        params = "; ".join(f"{p['name']} — {p['meaning']}" for p in block.get("params") or [])
        rule = str(block.get("rule") or "").strip().rstrip(".")
        out.append(f"    - блок `{block['name']}` ({block.get('file')}): {rule}. "
                   f"Параметры: {params}. "
                   + ("Блок уже был написан раньше, использован снова."
                      if block.get("reused") else "Блок написан впервые и записан в "
                      "`night/agent_blocks.yaml`: его можно брать в другие спеки."))
    runs = e.get("runs") or {}
    if not runs:
        return out
    causal = runs.get("causal") or {}
    if causal.get("action") == "ran":
        out.append(
            f"    - версия с выбором кода: числа выбирает код раз в месяц и только по прошлому, "
            f"между вариантами карточки ({', '.join(causal.get('candidates') or [])}); данные "
            f"Binance, торгует с {causal.get('first_choice')} (до первого выбора "
            "она не торгует)"
            + (f"; монет только {len(causal['instruments'])}: список подобран заранее, поэтому "
               "версия судится только со дня, когда он был известен" if causal.get("instruments")
               else ""))
    out += table_lines(runs)
    spec_run, base_run = runs.get("spec") or {}, runs.get("baseline") or {}
    if spec_run.get("action") == "ran" and (causal.get("action") == "ran"
                                            or base_run.get("action") == "ran") \
            and spec_run.get("data") != (causal or base_run).get("data"):
        out.append("    - строка «написанная» считалась на своих данных, а база и версия с "
                   "выбором кода — на одних и тех же: сравнивать их друг с другом можно, "
                   "с написанной — осторожно.")
    for step in ("baseline", "spec", "causal"):
        r = runs.get(step)
        if not r:
            continue
        if line := _why_not(step, r):
            out.append(f"    - {line}")
        elif years := years_line(r.get("metrics"), signs=True):
            out.append(f"    - {LABELS[step]}: {years}")
        if r.get("action") == "ran" and r.get("route_reason") and step != "baseline" \
                and r.get("route") not in ("paper", "shelf"):
            out.append(f"    - {LABELS[step]}, причина исхода (слова кода): {r['route_reason']}")
    for key, label in (("stage2-spec", "написанной версии"), ("stage2-causal", "версии с "
                                                                              "выбором кода")):
        r = runs.get(key)
        if not r:
            continue
        if r.get("action") == "ran":
            out.append(f"    - второй этап {label}:")
            out += [f"        - {ln}" for ln in stage2_lines({"route": r.get("route")},
                                                             r["stage2"])]
        elif r.get("action") == "error":
            out.append(f"    - второй этап {label}: сбой — {str(r.get('error'))[:200]}; "
                       "повторится в следующую ночь")
        else:
            out.append(f"    - второй этап {label}: не проводился — {r.get('reason')}")
    return out


def summary(e: Mapping) -> str:
    """One short line for the console's agents page: the routes of the three versions."""
    runs = e.get("runs") or {}
    parts = []
    for step in ("spec", "causal", "baseline"):
        r = runs.get(step)
        if not r:
            continue
        parts.append(f"{LABELS[step]}: " + (twin_stage.route_ru(r.get("route"))
                                             if r.get("action") == "ran" else
                                             {"none": "не сделана", "heavy": "тяжёлые данные",
                                              "error": "сбой"}.get(str(r.get("action")),
                                                                    "не прогнана")))
    return "; ".join(parts)


__all__ = ["BASELINE_DIR", "LEDGER", "MAX_ATTEMPTS", "RunContext", "STEPS", "entry_lines",
           "load_stage2_config", "new_record", "pending_cards", "read_fields", "run_card",
           "should_take", "spec_file", "summary", "table_lines", "waits_for_owner"]
