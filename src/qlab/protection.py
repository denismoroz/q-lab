"""How well Bv2 protects: the four protection readings P1 - P4 (docs/PROTECTION.md).

Owner, 2026-10-27: Bv2 is insurance, not an income strategy (it holds coins and, in falls, hedges
them with short perpetual positions); the noise bar measures income only. Owner, 2026-10-29:
«да, чини учёт и сделай замер защиты для Bv2». Measure before rule (owner, 2026-10-28: «соберем
по признакам проходи или нет ... когда наберем статистику - решим»): the readings **decide
nothing**; each is recorded for every run of the Bv2 family as numbers plus "holds / does not
hold / not defined", and the owner decides later whether a protection sign becomes a rule.

**What is compared.** The strategy's own money (docs/CAPITAL_BASE.md: the `capital_return` series,
summed for Bv2, which does not reinvest) against its own long side held and not hedged: the
positive weights of the book, as the book declared them over time, times the base, times the coins'
price moves, with no costs and no funding. That is "what the same coins would have done without
the shorts", in the same dollars and on the same base. It is NOT the console's dotted "just hold
the coins" line (equal parts of the whole capital in each coin): Bv2 keeps about half of its money
in coins, so against a basket that puts all of it in coins any half-sized book would "cut the
fall" by half without hedging anything.

**The readings**, in whole calendar weeks (Monday to Sunday; the incomplete first and last weeks
are left out), on the judged part of the run:

* P1 -- the worst fall (peak to trough on the strategy's money) is at most half the long side's;
* P2 -- over the weeks in which the long side fell, the strategy's weekly returns add up to at most
  0.3 of the long side's: how much of a fall the strategy still takes;
* P3 -- the same ratio over the weeks the long side rose: how much of a rise it keeps. Shown,
  no line;
* P4 -- the return over the whole part is not below zero.

The lines (0.5, 0.3, 0) are in the ruleset (`informational`, rules 2026-10-29.1), declared before
the numbers were looked at; nothing here holds a line of its own. What is declared here is only
when a reading is *not defined* (`MIN_WEEKS`, `MIN_FALL`): too short a part says nothing.

The metrics are named `prot_*`; `read_protection` turns them into the readings, like
`qlab.year_signs.read_signs`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlab.rules.engine import read_informational
from qlab.rules.schema import RuleSet

HOLDS, FAILS, UNDEFINED = "holds", "fails", "not defined"
HOLDS_RU, FAILS_RU, UNDEFINED_RU = "держится", "не держится", "не определено"
NUMBER, NUMBER_RU = "number", "число"

MIN_WEEKS = 8
"""Whole calendar weeks a part needs for any reading, and falling (rising) weeks P2 (P3) needs:
fewer say nothing about how the strategy behaves in a fall."""
MIN_FALL = 0.02
"""The long side's worst fall, in shares of the strategy's money, below which P1 is not defined:
if the coins hardly fell there was nothing to be protected from."""

BV2_CODE_REFS = frozenset({"qlab.strategies.live.bv2:LiveBv2", "qlab.strategies.bv2:BStrategyV2"})
WRAPPER_KEYS = (("inner_code_ref", "inner_params"), ("inner_code_ref", "base_params"))
"""How a wrapper (`Retune`, `RegimeGate`, `Overlaid`, `SplitCadence`) names the strategy it wraps:
the key of its code_ref and of its params. The calibration's noise books are not followed: a noise
book is not a strategy to be judged."""
STORE = Path("data/protection")


# --------------------------------------------------------------------------
# Which runs


def is_bv2_family(code_ref: str, params: Mapping[str, Any] | None, _depth: int = 0) -> bool:
    """Whether a spec runs Bv2 itself or a wrapper around it (a re-tuning twin, a gate, an
    overlay, a cadence split). The data venue does not matter: the Binance long history and the
    Hyperliquid second stage both are."""
    if code_ref in BV2_CODE_REFS:
        return True
    if _depth > 4 or not isinstance(params, Mapping):
        return False
    for code_key, params_key in WRAPPER_KEYS:
        inner = params.get(code_key)
        if isinstance(inner, str) and is_bv2_family(
                inner, params.get(params_key), _depth + 1):  # type: ignore[arg-type]
            return True
    return False


# --------------------------------------------------------------------------
# The numbers


def long_side_returns(prices: pd.DataFrame, weights: pd.DataFrame,
                      base: pd.Series | None, index: pd.Index) -> pd.Series:
    """The book's own long side, unhedged, per bar, as a share of the strategy's money.

    At decision time `t` the book holds `weights.loc[t]` (shares of its equity) over `(t, t+1]`;
    the positive part is the coins it owns. Times the base (the book's equity over the strategy's
    money, `qlab.harness.strategy.own_base`; 1 when the book has none) and the coins' price moves
    over the bar, with no costs and no funding. Indexed like the strategy's own series."""
    forward = prices.pct_change().shift(-1).fillna(0.0)
    long_part = weights.clip(lower=0.0)
    columns = [c for c in long_part.columns if c in forward.columns]
    share = (long_part[columns] * forward[columns]).sum(axis=1)
    if base is not None:
        share = share * base.reindex(share.index).ffill().bfill().fillna(1.0)
    return share.reindex(index).fillna(0.0).rename("long_side_return")


def _by_day(returns: pd.Series, reinvested: bool) -> pd.Series:
    days = pd.DatetimeIndex(returns.index).tz_convert("UTC").normalize()
    values = returns.astype(float)
    return (1.0 + values).groupby(days).prod() - 1.0 if reinvested else values.groupby(days).sum()


def _by_whole_week(daily: pd.Series, reinvested: bool) -> pd.Series:
    """One return per whole calendar week (Monday to Sunday, seven days present)."""
    weeks = daily.index.tz_localize(None).to_period("W-SUN")
    grouped = daily.groupby(weeks)
    whole = grouped.size() == 7
    weekly = grouped.apply(lambda w: float(np.prod(1.0 + w.to_numpy()) - 1.0) if reinvested
                           else float(w.sum()))
    return weekly[whole]


def _worst_fall(daily: pd.Series, reinvested: bool) -> float:
    """The deepest fall from a peak of the money, as a negative share of the strategy's money
    (points of it for a book that does not reinvest, a fraction of the peak for one that does)."""
    if reinvested:
        growth = (1.0 + daily).cumprod()
        peak = growth.cummax().clip(lower=1.0)
        return float((growth / peak - 1.0).min())
    curve = daily.cumsum()
    return float((curve - curve.cummax().clip(lower=0.0)).min())


def _total(daily: pd.Series, reinvested: bool) -> float:
    return float((1.0 + daily).prod() - 1.0) if reinvested else float(daily.sum())


def protection_metrics(strategy: pd.Series, long_side: pd.Series, *, reinvested: bool
                       ) -> dict[str, float]:
    """The `prot_*` metrics of one judged part: the strategy's returns per bar on its own money
    and the long side's on the same money. A number that is not defined for this part is left
    out (a missing metric reads as "not defined", never as a failure). Pure."""
    if len(strategy) < 2:
        return {}
    ours = _by_day(strategy, reinvested)
    theirs = _by_day(long_side.reindex(strategy.index).fillna(0.0), reinvested).reindex(
        ours.index).fillna(0.0)
    week_ours = _by_whole_week(ours, reinvested)
    week_theirs = _by_whole_week(theirs, reinvested)
    out: dict[str, float] = {
        "prot_weeks": float(len(week_ours)),
        "prot_days": float(len(ours)),
        "prot_total_return": _total(ours, reinvested),
        "prot_long_total_return": _total(theirs, reinvested),
        "prot_worst_fall": _worst_fall(ours, reinvested),
        "prot_long_worst_fall": _worst_fall(theirs, reinvested),
        "prot_reinvested": 1.0 if reinvested else 0.0,
    }
    falls = week_theirs < 0
    rises = week_theirs > 0
    out["prot_fall_weeks"] = float(falls.sum())
    out["prot_rise_weeks"] = float(rises.sum())
    if len(week_ours) < MIN_WEEKS:
        # too short a part: only the plain facts above stand, no reading is defined
        return out
    if out["prot_long_worst_fall"] <= -MIN_FALL:
        out["prot_fall_ratio"] = out["prot_worst_fall"] / out["prot_long_worst_fall"]
    if falls.sum() >= MIN_WEEKS:
        out["prot_fall_capture"] = float(week_ours[falls].sum() / week_theirs[falls].sum())
    if rises.sum() >= MIN_WEEKS:
        out["prot_rise_capture"] = float(week_ours[rises].sum() / week_theirs[rises].sum())
    return out


def protection_from_run(result: Any, part_panel: Any, part_weights: pd.DataFrame,
                        base: pd.Series | None, reinvested: bool) -> dict[str, float]:
    """`protection_metrics` of a backtest result of the judged part (`RunResult`): its
    `capital_return` against its own long side."""
    own = result.capital_return if result.capital_return is not None else result.net_return
    long_side = long_side_returns(part_panel.prices, part_weights, base, own.index)
    return protection_metrics(own, long_side, reinvested=reinvested)


# --------------------------------------------------------------------------
# The readings


@dataclass(frozen=True)
class ProtectionSpec:
    letter: str
    rule_id: str
    """The informational reading of the ruleset that draws the line."""
    title: str
    """What the reading says, in plain Russian (the line itself is in the ruleset)."""


P_FALL, P_WEEKS, P_RISE, P_LOSS = (
    "protection_fall_cut", "protection_fall_weeks_loss", "protection_rise_kept",
    "protection_no_loss")
READINGS = (
    ProtectionSpec("П1", P_FALL,
                   "худшее падение не больше половины падения тех же монет без защиты"),
    ProtectionSpec("П2", P_WEEKS,
                   "в недели падения монет стратегия теряет не больше 30% их падения"),
    ProtectionSpec("П3", P_RISE, "в недели роста монет стратегия сохраняет столько их роста"),
    ProtectionSpec("П4", P_LOSS, "за весь период стратегия не в минусе"),
)
LETTERS = tuple(r.letter for r in READINGS)
NO_RUN_RU = "у прогона нет замера защиты (он сделан раньше или идея не из семьи Bv2)"
FEW_WEEKS_RU = f"полных недель меньше {MIN_WEEKS}"
FEW_FALLS_RU = f"недель падения монет меньше {MIN_WEEKS}"
FEW_RISES_RU = f"недель роста монет меньше {MIN_WEEKS}"
SMALL_FALL_RU = (f"монеты почти не падали: их худшее падение меньше {MIN_FALL:.0%} денег "
                 "стратегии, защищаться было не от чего")


@dataclass(frozen=True)
class ProtectionReading:
    letter: str
    rule_id: str
    title: str
    metric: str
    value: float | None
    comparator: str | None
    threshold: float | None
    holds: bool | None
    """None: not defined for this run (or, for P3, no line is drawn)."""
    reason: str | None = None

    @property
    def number_only(self) -> bool:
        return self.comparator is None

    @property
    def state(self) -> str:
        """`holds`, `fails`, `not defined`; `number` for a reading with no line (P3) that has
        its number."""
        if self.value is None:
            return UNDEFINED
        return NUMBER if self.number_only else HOLDS if self.holds else FAILS

    @property
    def state_ru(self) -> str:
        return {HOLDS: HOLDS_RU, FAILS: FAILS_RU, UNDEFINED: UNDEFINED_RU,
                NUMBER: NUMBER_RU}[self.state]

    def to_json(self) -> dict[str, Any]:
        return {"letter": self.letter, "id": self.rule_id, "title": self.title,
                "metric": self.metric, "value": self.value, "comparator": self.comparator,
                "threshold": self.threshold, "holds": self.holds, "state": self.state,
                "reason": self.reason}


def has_protection(metrics: Mapping[str, object] | None) -> bool:
    """Whether the run recorded a protection measurement."""
    return any(k.startswith("prot_") for k in (metrics or {}))


def _number(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) and value == value else None


def not_defined_reason(spec: ProtectionSpec, metrics: Mapping[str, object]) -> str:
    """Why reading `spec` has no number in `metrics`, in plain Russian."""
    if not has_protection(metrics):
        return NO_RUN_RU
    weeks = _number(metrics, "prot_weeks")
    if weeks is None or weeks < MIN_WEEKS:
        return FEW_WEEKS_RU
    if spec.rule_id == P_FALL:
        return SMALL_FALL_RU
    if spec.rule_id == P_WEEKS:
        return FEW_FALLS_RU
    if spec.rule_id == P_RISE:
        return FEW_RISES_RU
    return FEW_WEEKS_RU


def read_protection(metrics: Mapping[str, object] | None, ruleset: RuleSet
                    ) -> list[ProtectionReading]:
    """The readings `ruleset` draws a line (or a number) for, against `metrics`, in the order
    П1 - П4. A reading the ruleset does not have (an older version) is left out. Pure."""
    m: Mapping[str, object] = metrics or {}
    found = {r.rule_id: r for r in read_informational(m, ruleset)}  # type: ignore[arg-type]
    out = []
    for spec in READINGS:
        r = found.get(spec.rule_id)
        if r is None:
            continue
        undefined = r.value is None
        out.append(ProtectionReading(
            letter=spec.letter, rule_id=spec.rule_id, title=spec.title, metric=r.metric,
            value=r.value, comparator=r.comparator, threshold=r.threshold, holds=r.holds,
            reason=not_defined_reason(spec, m) if undefined else None))
    return out


def reading_line(reading: ProtectionReading) -> str:
    """One reading in words: «П2: ... — держится (0.12)»."""
    if reading.value is None:
        return f"{reading.letter}: {reading.title} — {UNDEFINED_RU} ({reading.reason})"
    if reading.number_only:
        return f"{reading.letter}: {reading.title} — {_fmt(reading)}"
    return f"{reading.letter}: {reading.title} — {reading.state_ru} ({_fmt(reading)})"


def _fmt(reading: ProtectionReading) -> str:
    if reading.value is None:
        return "—"
    if reading.rule_id == P_LOSS:
        return f"{reading.value:+.1%}".replace("-", "−")
    return f"{reading.value:.2f}".replace("-", "−")


def sentence(metrics: Mapping[str, object] | None, ruleset: RuleSet) -> str | None:
    """The readings as one compact tail for a report line, e.g. «защита: П1 держится, П2 не
    держится, П3 0.41, П4 держится». None when the run has no measurement or the ruleset draws
    no reading."""
    if not has_protection(metrics):
        return None
    readings = read_protection(metrics, ruleset)
    if not readings:
        return None
    parts = []
    for r in readings:
        if r.value is None:
            parts.append(f"{r.letter} не определено")
        elif r.number_only:
            parts.append(f"{r.letter} {_fmt(r)}")
        else:
            parts.append(f"{r.letter} {r.state_ru} ({_fmt(r)})")
    return "защита: " + ", ".join(parts)


def view(metrics: Mapping[str, object] | None, ruleset: RuleSet | None) -> dict[str, Any] | None:
    """The protection measurement as the console's API serves it: the readings, the plain
    numbers behind them and the length of the part. None when the run has no measurement."""
    if not has_protection(metrics) or ruleset is None:
        return None
    m = metrics or {}
    numbers = {k: _number(m, k) for k in m if k.startswith("prot_")}
    return {"readings": [r.to_json() for r in read_protection(m, ruleset)],
            "numbers": numbers, "min_weeks": MIN_WEEKS}


def view_of_trial(trial_id: int, metrics: Mapping[str, object] | None,
                  ruleset: RuleSet | None) -> dict[str, Any] | None:
    """`view` of a run's own metrics, else of the measurement stored for it afterwards
    (`measure_trial`/`save`), else None. Reads at most one small file."""
    return view(metrics if has_protection(metrics) else load(trial_id), ruleset)


# --------------------------------------------------------------------------
# Over the registry


STAGE_BINANCE, STAGE_HYPERLIQUID, STAGE_OTHER = "Binance", "Hyperliquid 2", "другой"
STAGE_HYPERLIQUID_ALONE = "Hyperliquid"
OPEN_STATUSES = ("live", "paper", "validated", "bench", "candidate", "implemented", "speccing")


@dataclass(frozen=True)
class ProtectionRow:
    """One idea of the Bv2 family: its latest routed run and the readings of it."""

    idea_id: str
    status: str
    trial_id: int
    stage: str
    route: str
    source: str
    """Where the numbers came from: `run`, `stored`, `computed` or `none` (no measurement)."""
    metrics: dict[str, object]
    readings: list[ProtectionReading]

    def reading(self, letter: str) -> ProtectionReading | None:
        return next((r for r in self.readings if r.letter == letter), None)

    def to_json(self) -> dict[str, Any]:
        return {"idea_id": self.idea_id, "status": self.status, "trial_id": self.trial_id,
                "stage": self.stage, "route": self.route, "source": self.source,
                "numbers": {k: v for k, v in self.metrics.items() if k.startswith("prot_")},
                "readings": [r.to_json() for r in self.readings]}


def collect(session: Any, ruleset: RuleSet, *, compute: bool = False, live_only: bool = True,
            store: Path | None = None) -> list[ProtectionRow]:
    """The protection readings of each Bv2-family idea's latest routed run. Without `compute`
    only a run that carries a measurement (recorded or stored) has numbers; with it a run
    without one is measured again from its stored spec and snapshot, and the result is stored
    for the next time. `live_only` leaves out retired, rejected and decayed ideas. Calibration
    noise is not a strategy and is never listed."""
    from qlab.registry.models import Idea, Spec, Trial
    from qlab.registry.stages import is_stage2_child

    query = (session.query(Trial, Spec, Idea).join(Spec, Trial.spec_id == Spec.id)
             .join(Idea, Spec.idea_id == Idea.id)
             .filter(Trial.route.isnot(None), Trial.metrics.isnot(None),
                     ~Idea.id.like("noise-%"))
             .order_by(Idea.id, Trial.id))
    latest: dict[str, tuple[Any, Any, Any]] = {}
    for trial, spec, idea in query:
        if not is_bv2_family(spec.code_ref, spec.params):
            continue
        if live_only and idea.status.value not in OPEN_STATUSES:
            continue
        latest[idea.id] = (trial, spec, idea)
    rows = []
    for trial, spec, idea in latest.values():
        metrics, source = metrics_of(session, trial, compute=compute, store=store)
        venue = (spec.data_requirements or {}).get("source")
        stage = (STAGE_HYPERLIQUID if is_stage2_child(idea) else
                 STAGE_BINANCE if venue == "binance" else
                 STAGE_HYPERLIQUID_ALONE if venue == "hyperliquid" else STAGE_OTHER)
        rows.append(ProtectionRow(
            idea_id=idea.id, status=idea.status.value, trial_id=trial.id, stage=stage,
            route=trial.route.value, source=source, metrics=metrics,
            readings=read_protection(metrics, ruleset)))
    return rows


def _cell(reading: ProtectionReading | None) -> str:
    if reading is None or reading.value is None:
        return "—"
    if reading.number_only:
        return _fmt(reading)
    mark = "да" if reading.holds else "нет"
    return f"{mark} {_fmt(reading)}"


def render_table(rows: list[ProtectionRow]) -> list[str]:
    """One line per idea: idea, status, stage ("Hyperliquid 2" is the second stage of an idea
    judged first on Binance, "Hyperliquid" an idea judged on Hyperliquid alone), trial, whole
    weeks, then П1 - П4 as «да / нет» with the number (П3 the number alone), «—» where a reading
    is not defined."""
    width = max([len(r.idea_id) for r in rows] + [4])
    head = (f"{'idea':<{width}} {'status':<11} {'stage':<13} {'trial':>7} {'weeks':>5}  "
            + "  ".join(f"{letter:<9}" for letter in LETTERS))
    lines = [head]
    for r in rows:
        weeks = _number(r.metrics, "prot_weeks")
        cells = "  ".join(f"{_cell(r.reading(letter)):<9}" for letter in LETTERS)
        lines.append(f"{r.idea_id:<{width}} {r.status:<11} {r.stage:<13} {r.trial_id:>7} "
                     f"{'' if weeks is None else int(weeks):>5}  {cells}"
                     + ("" if r.source != "none" else "   (нет замера)"))
    return lines


def summarize(rows: list[ProtectionRow]) -> dict[str, dict[str, int]]:
    """Per reading: how many ideas hold, fail, are not defined (P3 has no line and is only
    counted as defined or not)."""
    out: dict[str, dict[str, int]] = {}
    for letter in LETTERS:
        counts = {HOLDS: 0, FAILS: 0, UNDEFINED: 0, "number": 0}
        for r in rows:
            reading = r.reading(letter)
            if reading is None or reading.value is None:
                counts[UNDEFINED] += 1
            elif reading.number_only:
                counts["number"] += 1
            else:
                counts[HOLDS if reading.holds else FAILS] += 1
        out[letter] = counts
    return out


SECTION_TITLE = "Защита Bv2: замер"


def report_lines(session: Any, ruleset: RuleSet, since: Any | None = None) -> list[str]:
    """The morning report's few lines: the readings of each live Bv2-family idea's latest run
    that carries a measurement; with `since` (a datetime) only runs written after it get a line of
    their own, the totals always cover all. Empty when no run carries a measurement yet."""
    rows = [r for r in collect(session, ruleset) if r.source != "none"]
    if not rows:
        return []
    totals = summarize(rows)
    lines = [
        "Замер защиты ничего не решает: он только записывается (владелец, 2026-10-29: «сделай "
        "замер защиты для Bv2»), а станет ли какой-то признак правилом, решит владелец "
        f"(`docs/PROTECTION.md`). Основа: последний прогон каждой живой идеи семьи Bv2 с замером "
        f"— {len(rows)} идей; таблица и числа: `uv run qlab protection`.", ""]
    for spec in READINGS:
        t = totals[spec.letter]
        if spec.rule_id == P_RISE:
            lines.append(f"- {spec.letter} ({spec.title}): число есть у {t['number']}, "
                         f"не определено у {t[UNDEFINED]}")
        else:
            lines.append(f"- {spec.letter} ({spec.title}): держится {t[HOLDS]}, "
                         f"не держится {t[FAILS]}, не определено {t[UNDEFINED]}")
    if since is not None:
        from qlab.registry.models import Trial

        fresh = [r for r in rows if (t := session.get(Trial, r.trial_id)) is not None
                 and t.started_at is not None and _aware(t.started_at) >= since]
        if fresh:
            lines += ["", f"Прогоны с {since.date().isoformat()}:"]
            lines += [f"- **{r.idea_id}** ({r.stage}, прогон {r.trial_id}): "
                      f"{sentence(r.metrics, ruleset)}" for r in fresh]
    return lines


def _aware(moment: Any) -> Any:
    from datetime import UTC

    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


# --------------------------------------------------------------------------
# Stored measurements of runs made before the measurement existed


def store_dir() -> Path:
    return STORE


def path_for(trial_id: int, store: Path | None = None) -> Path:
    return (store or store_dir()) / f"{trial_id}.json"


def save(trial_id: int, metrics: Mapping[str, float], store: Path | None = None) -> Path:
    path = path_for(trial_id, store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"trial_id": trial_id, "metrics": dict(metrics)}, indent=1),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load(trial_id: int, store: Path | None = None) -> dict[str, float] | None:
    try:
        payload = json.loads(path_for(trial_id, store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    return metrics if isinstance(metrics, dict) else None


def measure_trial(session: Any, trial_id: int, *, spec_root: Path | None = None
                  ) -> tuple[dict[str, float] | None, str | None]:
    """The protection metrics of a stored run, made again from its stored spec and snapshot over
    the part the verdict was read on (`qlab.forward_series.replay_run`). `(None, reason)` when
    the run cannot be made again. Writes nothing; CPU only."""
    from qlab.forward_series import replay_run

    run, reason = replay_run(session, trial_id, spec_root=spec_root, part="judged")
    if run is None:
        return None, reason
    if not is_bv2_family(run.spec.code_ref, run.spec.params):
        return None, "идея не из семьи Bv2"
    drawn_on_capital = run.capital.refused is None
    metrics = protection_from_run(
        run.result, run.part_panel, run.part_weights,
        run.capital.base if drawn_on_capital else None,
        run.result.capital_reinvested if drawn_on_capital else True)
    return metrics, None


def metrics_of(session: Any, trial: Any, *, compute: bool = False, store: Path | None = None
               ) -> tuple[dict[str, object], str]:
    """A run's protection metrics and where they came from: `"run"` (recorded in the trial),
    `"stored"` (measured afterwards, `save`d), `"computed"` (measured now, with `compute`) or
    `"none"` (an empty mapping)."""
    if has_protection(trial.metrics):
        return dict(trial.metrics), "run"
    kept = load(trial.id, store)
    if kept:
        return dict(kept), "stored"
    if compute:
        measured, _ = measure_trial(session, trial.id)
        if measured:
            save(trial.id, measured, store)
            return dict(measured), "computed"
    return {}, "none"


__all__ = [
    "SECTION_TITLE", "ProtectionRow", "collect", "render_table", "report_lines", "summarize",
    "BV2_CODE_REFS", "FAILS", "HOLDS", "LETTERS", "MIN_FALL", "MIN_WEEKS", "READINGS", "UNDEFINED",
    "ProtectionReading", "ProtectionSpec", "has_protection", "is_bv2_family", "long_side_returns",
    "measure_trial", "metrics_of", "not_defined_reason", "protection_from_run",
    "protection_metrics", "read_protection", "reading_line", "sentence", "view", "view_of_trial"]
