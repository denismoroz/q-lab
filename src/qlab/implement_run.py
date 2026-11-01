"""What happens to an implemented card after the guards pass: its verdict by the next morning
(docs/IMPLEMENTER.md, «Из идеи в блок и в проверку за одну ночь»).

Owner, 2026-10-27: the framework itself turns an idea that needs NEW code into a verdict by the
next morning. The implementer agent writes the block and the spec (`qlab.agents.implementer`);
this module is everything after it, and it is CODE only: no agent, no tokens, so it sits outside
the budget guard by construction. Owner, 2026-10-31: one idea, one check (docs/ONE_CHECK.md) --
before, a card ran as the written spec AND as its causal version AND the second stage of each;
now it is three steps, in this order, each a run that writes its trials (the deflation counts
every one of them):

1. `check`        -- the idea's one check, under the idea's own id: over the card's declared grid
                     (`StrategySpec.choice`, written into the spec by code from the card) `Retune`
                     picks the variant every month from the past only, expanding window,
                     after-cost Sharpe, on Binance's history from 2019-09-08 (or the card's
                     `honest_from`), judged at once (`qlab.one_check`); every variant is a trial
                     of its own. A card whose spec says there is nothing to choose runs as written;
2. `baseline`     -- the card's baseline spec re-run on exactly the check's panel, costs and
                     judged part (`qlab.twin.same_panel_spec`), so the report compares like with
                     like; with nothing to choose, on the written spec's panel;
3. `stage2`       -- the second stage on Hyperliquid (`qlab.pipeline.stage2.run_stage2`, the same
                     idea there, a consistency check with stage 1, not a measurement) for a
                     survivor of step 1.

The heavy-data guard of the twins stage applies to every step (`night/twins.yaml`): an hourly
panel of every Binance perpetual is not fetched in a night. The night's stop time is checked
before each step; a step that is cut, or that failed, is kept in the ledger
(`data/implement/ledger.json`) and done on the next night, where the previous steps' results are
reused as they are. A reviewer's verified gap (the code does not express the card) runs step 1
only: the stand answers "not evaluable" and the rest would be about a different strategy.
Ledger records of cards finished before the rule keep their old steps (`spec`, `causal`,
`stage2-spec`, `stage2-causal`) and are shown as they were.

Nothing here judges: every verdict is the stand's."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from qlab import twin, twin_stage
from qlab.capital_text import headline
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
STEPS = ("check", "baseline", "stage2")
OLD_STEPS = ("spec", "causal", "stage2-spec", "stage2-causal")
"""The steps of a card finished before the one-check rule (shown, never run again)."""
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


def with_choice(spec_path: Path, spec: StrategySpec, fields: CardFields | None, card: str
                ) -> StrategySpec:
    """The spec with the `choice` block of the card's grid, written into the spec file by code.

    The implementer agent writes the plain spec carrying the grid's first variant and no
    `choice`; the lists come from the card (the owner's idea), and code is the one that declares
    them, so that every later night's re-run of the idea (`qlab.waiting`) finds the same check in
    the file. A spec that already has a `choice` (the agent's `nothing_to_choose`) is as it is."""
    from qlab.pipeline.spec import Choice

    if spec.choice is not None or fields is None or fields.grid is None:
        return spec
    grid = fields.grid
    choice = Choice(axes={p: list(v) for p, v in grid.axes.items()},
                    candidates={n: dict(o) for n, o in grid.candidates.items()},
                    source=_cite(card))
    block = {"axes": {p: list(v) for p, v in choice.axes.items()}} if choice.axes else {
        "candidates": {n: dict(o) for n, o in choice.candidates.items()}}
    block["source"] = choice.source
    text = spec_path.read_text(encoding="utf-8").rstrip("\n")
    spec_path.write_text(
        text + "\n\n# Written by code from the card's grid (qlab.implement_run,\n"
        "# docs/ONE_CHECK.md): the lists the idea's one check chooses among, every month from the\n"
        "# past only.\n"
        + yaml.safe_dump({"choice": block}, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    return load_spec(spec_path)


def _check_form(ctx: RunContext, card: str, spec: StrategySpec, fields: CardFields | None
                ) -> twin.TwinResult:
    """The idea's check as the card describes it (`qlab.one_check.check_spec`): the same call for
    the check step, the baseline's panel and the second stage."""
    from qlab import one_check

    cfg = ctx.twins
    names = ctx.listing() if twin.needs_listing(
        spec, None, coin_params=cfg.coin_list_params) else None
    honest = exec_venue = None
    if fields is not None and fields.data is not None:
        if fields.data.honest_from is not None:
            honest = (fields.data.honest_from, _cite(card))
        if fields.data.venue != twin.TWIN_VENUE:
            exec_venue = fields.data.venue
    return one_check.check_spec(
        spec, venue_names=names, coin_params=cfg.coin_list_params,
        seconds_per_instrument=cfg.seconds_per_instrument, peers=[], honest_from=honest,
        execution_venue=exec_venue)


def _checked(ctx: RunContext, card: str, spec: StrategySpec, fields: CardFields | None
             ) -> StrategySpec:
    """The spec the idea's check runs, or the spec itself when there is nothing to choose."""
    from qlab import one_check

    if not one_check.declares_lists(spec):
        return spec
    return _check_form(ctx, card, spec, fields).spec or spec


def _step_check(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
                fields: CardFields | None) -> dict:
    from qlab import night, one_check
    from qlab.registry.db import session_scope
    from qlab.rules.loader import load_latest

    entry: dict = {"idea": spec.idea_id, "action": "ran", "data": _data_text(spec)}
    ran = spec
    if rec.get("blocked"):
        # the reviewer found a gap between the card and the code: the stand answers «проверить
        # нечем» for the spec as written, and no value of a list is worth a trial of its own
        pass
    elif one_check.declares_lists(spec):
        result = _check_form(ctx, card, spec, fields)
        if result.spec is None:
            entry.update(action="none", reason=result.reason)
            return entry
        ran = result.spec
        made = ran.params_fixed_at.isoformat() if ran.params_fixed_at else None
        entry.update(kind=result.kind, candidates=list(result.candidates), data=_data_text(ran),
                     first_choice=made, instruments=list(result.instruments),
                     list_source=result.list_source)
    else:
        entry["nothing_to_choose"] = one_check.nothing_to_choose(spec)
    heavy = _heavy(ctx, ran)
    if heavy:
        entry.update(action="heavy", reason=heavy)
        return entry
    night._ensure_idea(spec, card)
    if ran is not spec:  # every value of the lists is a trial of its own, before the verdict
        with session_scope() as session:
            entry["candidate_trials"] = one_check.record_candidates(ran, session, load_latest())
    twin_stage.stage1_on_stand(
        ran, entry, ensure_idea=lambda: night._ensure_idea(spec, card), end=ctx.end,
        noise_trials=ctx.noise_trials, slow_minutes=ctx.twins.slow_twin_minutes,
        kind="implemented")
    return entry


def _step_baseline(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
                   fields: CardFields | None) -> dict:
    if fields is None or fields.baseline_spec is None:
        return {"action": "skipped", "reason": "у карточки нет базовой стратегии"}
    baseline = load_spec(fields.baseline_spec)
    like = _checked(ctx, card, spec, fields)
    like_file = spec_file(spec.idea_id)
    names = ctx.listing() if baseline.required_instruments \
        and baseline.data.source != twin.TWIN_VENUE else None
    base = twin.same_panel_spec(baseline, like, idea_id=f"{spec.idea_id}{BASE_SUFFIX}",
                                cite=like_file.as_posix(), venue_names=names,
                                own_sources=like is not spec)
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
        slow_minutes=ctx.twins.slow_twin_minutes, kind="implemented")
    return entry


def _step_stage2(ctx: RunContext, card: str, rec: dict, spec: StrategySpec,
                 fields: CardFields | None) -> dict:
    source = (rec["runs"].get("check") or {})
    if source.get("action") != "ran":
        if _needs_run(source):  # the run it follows failed and is tried again tomorrow
            return {"action": "waiting", "reason": "проверка упала и будет прогнана ещё раз"}
        return {"action": "skipped", "reason": "проверка не прогнана"}
    target = _checked(ctx, card, spec, fields)
    if not twin_stage.goes_to_stage2(target):
        return {"action": "skipped",
                "reason": f"данные уже с {target.data.source}, второму этапу нечего проверять"}
    if ctx.stage2_config is None:
        return {"action": "skipped", "reason": "настройки второго этапа не прочитаны или "
                "он выключен (`night/stage2.yaml`)"}
    holder = {"route": source.get("route"), "trial": source.get("trial")}
    twin_stage.second_stage(
        target, holder, end=ctx.end, noise_trials=ctx.noise_trials, config=ctx.stage2_config)
    if "stage2" not in holder:
        return {"action": "skipped", "route": source.get("route"),
                "reason": f"на первом этапе исход «{twin_stage.route_ru(source.get('route'))}»"
                          ", на второй идут только прошедшие"}
    out = {"action": "ran", "route": source.get("route"), "stage2": holder["stage2"],
           "seconds": holder.get("stage2_seconds")}
    if holder["stage2"].get("status") == "error":
        out.update(action="error", error=holder["stage2"].get("error"))
    return out


_STEP: dict[str, Callable[..., dict]] = {
    "check": _step_check, "baseline": _step_baseline, "stage2": _step_stage2,
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
    spec = with_choice(spec_file(rec["idea_id"]), spec, fields, card)
    runs = rec.setdefault("runs", {})
    for name in STEPS:
        if not _needs_run(runs.get(name)):
            continue
        blocked = rec.get("blocked")
        if blocked and name != STEPS[0]:
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


LABELS = {"baseline": "база", "check": "проверка идеи", "spec": "написанная",
          "causal": "с выбором кода"}
SHOWN = ("baseline", "check", "spec", "causal")
"""The runs a card's table shows, in this order; `spec` and `causal` are only in the records of
cards finished before the one-check rule."""


def _label(step: str, r: Mapping) -> str:
    if step == "baseline":
        return f"база `{r.get('baseline')}`"
    return f"{LABELS[step]} `{r.get('idea')}`"


def table_lines(runs: Mapping[str, Mapping]) -> list[str]:
    """The comparison: baseline and the idea's check -- per year, Sharpe, drawdown, the return
    without the best 1% of days, the route -- plain Russian."""
    rows = []
    for step in SHOWN:
        r = runs.get(step)
        if not r or r.get("action") == "skipped":
            continue  # a version that was never due (no baseline on the card) has no row
        m = r.get("metrics") or {} if r.get("action") == "ran" else {}
        h = headline(m)  # on the strategy's own money when the run has it (CAPITAL_BASE.md)
        rows.append(
            f"| {_label(step, r)} | {r.get('data') or '—'} | "
            f"{_cell({'v': h.ann}, 'v', twin_stage.pct)} | "
            f"{_cell({'v': h.sharpe}, 'v', _sharpe)} | "
            f"{_cell({'v': h.max_dd}, 'v', twin_stage.pct)} | "
            f"{_cell({'v': h.ex_best_1pct}, 'v', twin_stage.pct)} | "
            f"{twin_stage.route_ru(r.get('route')) if m else '—'} |")
    if not rows:
        return []
    return ["", "    Числа — в процентах от денег стратегии (docs/CAPITAL_BASE.md).",
            "    | версия | данные | в год | Шарп | просадка | без 1% лучших дней | исход |",
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
    check = runs.get("check") or runs.get("causal") or {}
    if check.get("action") == "ran" and check.get("candidates"):
        out.append(
            f"    - одна проверка: числа выбирает код раз в месяц и только по прошлому, между "
            f"вариантами карточки ({', '.join(check.get('candidates') or [])}), каждый записан "
            f"отдельным испытанием; данные Binance, торгует с {check.get('first_choice')} (до "
            "первого выбора она не торгует)"
            + (f"; монет только {len(check['instruments'])}: список подобран заранее, поэтому "
               "идея судится только со дня, когда он был известен" if check.get("instruments")
               else ""))
    elif check.get("action") == "ran" and check.get("nothing_to_choose"):
        out.append(f"    - выбирать не из чего: {check['nothing_to_choose']}; судится после дня "
                   "фиксации")
    out += table_lines(runs)
    base_run = runs.get("baseline") or {}
    spec_run = runs.get("spec") or {}
    if spec_run.get("action") == "ran" and (check.get("action") == "ran"
                                            or base_run.get("action") == "ran") \
            and spec_run.get("data") != (check or base_run).get("data"):
        out.append("    - строка «написанная» считалась на своих данных, а база и версия с "
                   "выбором кода — на одних и тех же: сравнивать их друг с другом можно, "
                   "с написанной — осторожно.")
    for step in SHOWN:
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
    for key, label in (("stage2", "проверки идеи"), ("stage2-spec", "написанной версии"),
                       ("stage2-causal", "версии с выбором кода")):
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
    """One short line for the console's agents page: the routes of the check and the baseline."""
    runs = e.get("runs") or {}
    parts = []
    for step in ("check", "spec", "causal", "baseline"):
        r = runs.get(step)
        if not r:
            continue
        parts.append(f"{LABELS[step]}: " + (twin_stage.route_ru(r.get("route"))
                                             if r.get("action") == "ran" else
                                             {"none": "не сделана", "heavy": "тяжёлые данные",
                                              "error": "сбой"}.get(str(r.get("action")),
                                                                    "не прогнана")))
    return "; ".join(parts)


__all__ = ["BASELINE_DIR", "LEDGER", "MAX_ATTEMPTS", "OLD_STEPS", "RunContext", "STEPS",
           "entry_lines", "with_choice",
           "load_stage2_config", "new_record", "pending_cards", "read_fields", "run_card",
           "should_take", "spec_file", "summary", "table_lines", "waits_for_owner"]
