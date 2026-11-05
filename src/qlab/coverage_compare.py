"""A book whose coins were listed at different times: two ways to count it, side by side
(docs/COVERAGE.md).

Owner, 2026-11-05: «да, сделай замер обоими способами». The stand judges a book that names its legs
(`StrategySpec.required_instruments`) only from the day ALL of them have history (the complete-book
cut, `qlab.pipeline.evaluate.complete_book_window`, docs/REGIMES.md), and the data a spec asks for
may itself start late. A coin listed later therefore costs the whole book its early years, and the
last third (rules 2026-11-04.2) starts much later than it would on the longer history. Which of
the two is fair is the owner's call; this module only **measures both and decides nothing** (owner's
rule «сначала замер, потом правило»): no rule reads it, no status follows from it.

**«Как есть»** is the stand's own reading: the spec's data, the complete-book cut, the last third of
that window. **«Монеты входят по мере появления»** runs the very same spec over the window from the
first day ANY leg has data, with no cut. A coin joins the book when both its legs exist; until then
its money sits aside as cash (the live Bv2 driver opens a coin's book on the first bar both legs are
tradeable and priced and the startup history is there, and its slot's money waits as cash until
then; a weights strategy holds zero weight in a coin that is not tradeable). The 2/3 - 1/3 cut of
rules 2026-11-04.2 is applied to the longer window, the same costs, the strategy's own money
(`cap_*`, docs/CAPITAL_BASE.md).

**Knowable on each day.** A fixed coin list written by hand may know which coins rose later
(`qlab.knowable_coins`). So in the second form a coin does not join before the first day the venue's
own data made it an obvious pick (listed at least `window_days` days and among the top coins by
turnover); the book's list is knowable from the day the last of its coins is. A coin that was never
an obvious pick keeps the day its legs appeared and is marked. A book whose coins are chosen by the
strategy itself (`params.universe`) needs no such gate.

**What is reported for each form:** the window, the day the last third starts, its annual return
and its worst fall on the strategy's own money, whether the earning can be told from zero (the
bootstrap p of `qlab.earning_test`), and the share of the last third in which the book was not
complete. Noise books are not run here (200 books per reading, the stand's own place among them
stays on the run); the reading says so.

Stored per idea as a small JSON ledger (`STORE`, path overridable by `QLAB_COVERAGE_COMPARE`).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlab.pipeline.spec import StrategySpec

STORE = Path("data/coverage_compare")
ENV = "QLAB_COVERAGE_COMPARE"
AS_IS, AS_LISTED = "as_is", "as_listed"
FORMS = (AS_IS, AS_LISTED)
OPEN_STATUSES = ("candidate", "bench", "paper", "live")
"""Ideas that are not put away (rejected, retired, decayed)."""


def store_dir() -> Path:
    return Path(os.environ.get(ENV) or STORE)


def path_for(idea_id: str, store: Path | None = None) -> Path:
    return (store or store_dir()) / f"{idea_id}.json"


def save(payload: dict[str, Any], store: Path | None = None) -> Path:
    path = path_for(str(payload["idea_id"]), store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load(idea_id: str, store: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads(path_for(idea_id, store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_all(store: Path | None = None) -> dict[str, dict[str, Any]]:
    folder = store or store_dir()
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        found = load(path.stem, folder)
        if found is not None:
            out[path.stem] = found
    return out


# --------------------------------------------------------------------------
# The two forms on one panel


@dataclass(frozen=True)
class Gate:
    """When each coin of a fixed list may join (the venue's own data made it an obvious pick)."""

    days: dict[str, date | None]
    """coin -> first knowable day; None = never an obvious pick in the window."""
    known_from: date | None
    plain: str


def _coins_of(spec: StrategySpec) -> list[str]:
    from qlab.knowable_coins import coin_of

    names = spec.required_instruments or spec.data.instruments or []
    return list(dict.fromkeys(coin_of(c) for c in names))


def knowable_gate(spec: StrategySpec, first: date, last: date) -> Gate | None:
    """Per-coin first knowable day for a fixed list, or None when the gate does not apply (the
    coins are discovered, chosen by the strategy itself, or the venue's volume cache is absent)."""
    if spec.data.instruments is None or "universe" in spec.params:
        return None
    coins = _coins_of(spec)
    if not coins:
        return None
    from qlab import knowable_coins

    try:
        config = knowable_coins.load_config()
        volume = knowable_coins.raw_volume(spec.data.source, config)
        if volume is None or volume.empty:
            return None
        found = knowable_coins.check(volume, coins, venue=spec.data.source, start=first,
                                     end=last, config=config)
    except Exception:  # noqa: BLE001 - a check that cannot be made leaves the gate off
        return None
    return Gate(days=dict(found.coins), known_from=found.known_from, plain=found.plain)


def gate_panel(panel, gate: Gate | None):
    """`panel` with every coin of the gate untradeable before its first knowable day (a coin
    never knowable keeps its own listing: marked, not removed). Returns the panel and the
    first day each coin was allowed in."""
    if gate is None:
        return panel, {}
    from qlab.knowable_coins import SPOT_SUFFIX, coin_of

    tradeable = panel.tradeable.copy()
    opened: dict[str, date] = {}
    index = panel.prices.index
    for column in tradeable.columns:
        day = gate.days.get(coin_of(column))
        if day is None:
            continue
        before = index < pd.Timestamp(day, tz="UTC")
        tradeable.loc[before, column] = False
        if not column.endswith(SPOT_SUFFIX):
            opened[column] = day
    return dataclasses.replace(panel, tradeable=tradeable), opened


def complete_share(panel, required: Sequence[str] | None, lo: int, hi: int) -> float | None:
    """Share of the bars `lo..hi` in which not every leg of the book was there, or None when the
    book names no legs. A bar the book is held through (no price) counts as there, the way the
    complete-book cut counts it."""
    if not required:
        return None
    from qlab.harness.gaps import holdable_gaps

    present = panel.tradeable | holdable_gaps(panel.prices, panel.funding,
                                              panel.meta.get("no_funding_instruments", ()))
    columns = [c for c in required if c in present.columns]
    if len(columns) != len(required):
        return 1.0
    complete = present[columns].iloc[lo : hi + 1].all(axis=1).to_numpy(dtype=bool)
    return float(1.0 - complete.mean()) if len(complete) else None


def first_leg_row(panel, required: Sequence[str] | None) -> int:
    """The first row on which ANY leg of the book is tradeable (0 when it names none)."""
    columns = [c for c in (required or panel.tradeable.columns) if c in panel.tradeable.columns]
    if not columns:
        return 0
    any_leg = panel.tradeable[columns].any(axis=1).to_numpy(dtype=bool)
    return int(np.argmax(any_leg)) if any_leg.any() else 0


def measure_panel(spec: StrategySpec, panel, ruleset, *, listed: bool,
                  gate: Gate | None = None) -> dict[str, Any]:
    """One form of the reading on a ready panel (no registry, no network).

    `listed=False` is the stand's own: the longest complete stretch of the book's legs
    (`complete_book_window`), weights computed on the whole panel so the warm-up is not lost.
    `listed=True` cuts nothing: the panel starts at the first day any leg has data and each coin
    joins when its legs exist (and, with a `gate`, when it was knowable)."""
    from qlab.harness.costs import CostModel
    from qlab.harness.metrics import capital_metrics, compute_metrics
    from qlab.harness.run import run_backtest
    from qlab.harness.strategy import capital_setup, own_costs
    from qlab.pipeline.evaluate import (
        _slice_rows,
        complete_book_window,
        judged_rows,
        resolve_strategy,
        split_day,
        with_split_day,
    )

    required = spec.required_instruments
    opened: dict[str, date] = {}
    if listed:
        panel, opened = gate_panel(panel, gate)
        panel = _slice_rows(panel, first_leg_row(panel, required), len(panel.prices) - 1)
        coverage = None
    else:
        coverage = complete_book_window(panel, required) if required is not None else None
    index = panel.prices.index
    rows = judged_rows(spec, panel, coverage, ruleset)
    spec = with_split_day(spec, index, rows)  # a choice made at the cut
    strategy = resolve_strategy(spec.code_ref)
    weights = strategy.target_weights(panel, spec.params)
    costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                      slippage_bps=spec.costs.slippage_bps)
    extra_cost = own_costs(strategy, panel, spec.params)
    capital = capital_setup(strategy, panel, spec.params)
    lo, hi = rows.judged
    part_panel = _slice_rows(panel, lo, hi)
    result = run_backtest(part_panel, weights.iloc[lo : hi + 1], costs, part_panel.funding,
                          extra_cost=extra_cost, **capital.kwargs)
    on_capital = capital.refused is None
    if on_capital:
        measured = capital_metrics(result)
        ann, drawdown = measured.get("cap_ann_return"), measured.get("cap_max_dd")
        from qlab.earning_test import metrics_of as earning_metrics

        earning = earning_metrics(result.capital_return, additive=not result.capital_reinvested)
    else:
        measured = compute_metrics(result)
        ann, drawdown = measured.get("ann_return_net"), measured.get("max_dd")
        earning = {}
    bar = pd.Timedelta(days=1) if len(index) < 2 else index[1] - index[0]
    cut = split_day(index, rows)
    return {
        "window_first": index[rows.first].date().isoformat(),
        "window_last": index[rows.last].date().isoformat(),
        "window_days": round((index[rows.last] - index[rows.first] + bar) / pd.Timedelta(days=1)),
        "cut_day": None if cut is None else cut.isoformat(),
        "judged_first": index[lo].date().isoformat(),
        "judged_days": round((index[hi] - index[lo] + bar) / pd.Timedelta(days=1)),
        "ann_return": None if ann is None or ann != ann else float(ann),
        "max_dd": None if drawdown is None or drawdown != drawdown else float(drawdown),
        "p": earning.get("cap_mean_positive_p"),
        "on_capital": on_capital,
        "noise": "skipped",
        "incomplete_share": complete_share(panel, required, lo, hi),
        "joins": {c: d.isoformat() for c, d in opened.items()} or None,
    }


# --------------------------------------------------------------------------
# One idea


def earliest_start(session, source: str) -> date | None:
    """The first day any snapshot of `source` ever held: how far back the lab's data goes."""
    from sqlalchemy import func

    from qlab.registry.models import DataSnapshot

    return session.query(func.min(DataSnapshot.range_start)).filter(
        DataSnapshot.source == source).scalar()


def _differs_ru(rows: dict[str, dict[str, Any]], coins_note: str = "",
                judged_from: date | None = None) -> str:
    a, b = rows.get(AS_IS), rows.get(AS_LISTED)
    if not a or not b or "error" in a or "error" in b:
        return "Один из двух способов посчитать не удалось: сравнивать нечего."
    if a["window_first"] == b["window_first"]:
        if judged_from is not None:
            day = _ru_day(judged_from.isoformat())
            return (f"Окна совпадают: в спеке записано судить только с {day} (раньше — разгон), "
                    "поэтому оба способа смотрят на одни и те же дни.")
        return ("Окна совпадают: все монеты книги были с первого дня данных, "
                "способы дают одно и то же.")
    return (f"Первый способ ждёт все монеты сразу и судит с {_ru_day(a['window_first'])}, второй "
            f"берёт историю с {_ru_day(b['window_first'])} и пускает каждую монету, когда она "
            f"появилась{coins_note}; последняя треть начинается {_ru_day(a['cut_day'])} против "
            f"{_ru_day(b['cut_day'])}.")


def _ru_day(day: str | None) -> str:
    return "—" if not day else f"{day[8:10]}.{day[5:7]}.{day[:4]}"


def measure_idea(session, spec: StrategySpec, ruleset=None, *, data_end: date | None = None
                 ) -> dict[str, Any]:
    """Both forms for one idea's spec: the stand's own window and the longer one. Resolves the
    two snapshots (the stand's own if on disk; the longer one is built from the raw cache when
    no snapshot answers it), measures, returns the payload `save` stores."""
    from qlab.night import rerun_spec, with_end
    from qlab.pipeline.evaluate import cut_truncated_history, resolve_panel
    from qlab.rules.loader import load_latest

    ruleset = ruleset or load_latest()
    started = time.monotonic()
    rows: dict[str, dict[str, Any]] = {}
    try:
        # What the stand itself runs: an idea with open numbers or declared lists is its one-check
        # form (qlab.night.rerun_spec), cut from the window the file declares, then extended.
        spec = rerun_spec(spec)
    except Exception as exc:  # noqa: BLE001 - the reading says why there is nothing to compare
        failure = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        return {"idea_id": spec.idea_id, "measured_at": datetime.now(UTC).isoformat(
            timespec="seconds"), "data_end": spec.data.end.isoformat(),
            "rules_version": ruleset.version, "rows": {AS_IS: failure, AS_LISTED: failure},
            "affected": False, "knowable": None, "judged_from": None,
            "differs": _differs_ru({AS_IS: failure, AS_LISTED: failure}),
            "seconds": {"total": round(time.monotonic() - started, 1)}}
    if data_end is not None:
        spec = with_end(spec, data_end)
    gate: Gate | None = None
    seconds: dict[str, float] = {}

    for form in FORMS:
        t0 = time.monotonic()
        try:
            wanted = spec
            if form == AS_LISTED:
                earliest = earliest_start(session, spec.data.source)
                if earliest is not None and earliest < spec.data.start:
                    wanted = spec.model_copy(update={"data": spec.data.model_copy(
                        update={"start": earliest})})
            panel = resolve_panel(session, wanted)
            session.commit()
            if panel.meta.get("history_truncated"):
                panel = cut_truncated_history(panel)
            if form == AS_LISTED:
                gate = knowable_gate(wanted, panel.prices.index[0].date(),
                                     panel.prices.index[-1].date())
            rows[form] = measure_panel(wanted, panel, ruleset, listed=form == AS_LISTED,
                                       gate=gate if form == AS_LISTED else None)
        except Exception as exc:  # noqa: BLE001 - one form failing must not hide the other
            rows[form] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        seconds[form] = round(time.monotonic() - t0, 1)

    affected = (AS_IS in rows and AS_LISTED in rows and "error" not in rows[AS_IS]
                and "error" not in rows[AS_LISTED]
                and rows[AS_IS]["window_first"] != rows[AS_LISTED]["window_first"])
    knowable = None if gate is None else {
        "coins": {c: None if d is None else d.isoformat() for c, d in gate.days.items()},
        "known_from": None if gate.known_from is None else gate.known_from.isoformat(),
        "plain": gate.plain}
    note = ""
    if knowable is not None:
        never = [c for c, d in knowable["coins"].items() if d is None]
        note = (" (не раньше дня, когда монету можно было выбрать, не зная будущего"
                + (f"; {', '.join(never)} такой день не наступил, их берём по дню появления"
                   if never else "") + ")")
    return {
        "idea_id": spec.idea_id,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_end": spec.data.end.isoformat(),
        "rules_version": ruleset.version,
        "rows": rows,
        "affected": bool(affected),
        "knowable": knowable,
        "judged_from": None if spec.judged_from is None else spec.judged_from.isoformat(),
        "differs": _differs_ru(rows, note, spec.judged_from),
        "seconds": {**seconds, "total": round(time.monotonic() - started, 1)},
    }


# --------------------------------------------------------------------------
# Which ideas, and which spec


def spec_for_idea(session, idea_id: str, spec_dir: Path | None = None) -> StrategySpec | None:
    """The spec file of an idea. Several files naming one idea: the one that asks for the data the
    idea's latest registered spec asked for (source, interval, instruments, code)."""
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Spec
    from qlab.waiting import SPEC_DIR, spec_files

    paths = spec_files(spec_dir or SPEC_DIR).get(idea_id) or []
    if not paths:
        return None
    specs = [load_spec(p) for p in paths]
    if len(specs) == 1:
        return specs[0]
    row = session.query(Spec).filter(Spec.idea_id == idea_id).order_by(Spec.version.desc()).first()
    if row is not None:
        want = row.data_requirements or {}
        for spec in specs:
            if (spec.code_ref == row.code_ref and spec.data.source == want.get("source")
                    and spec.data.interval == want.get("interval")
                    and (spec.data.instruments or None) == (want.get("instruments") or None)):
                return spec
    return specs[0]


def last_run_end(session, idea_id: str) -> date | None:
    from qlab.overlay_stage import last_run_end as _end

    return _end(session, idea_id)


def candidates(session) -> list[str]:
    """Open ideas whose latest run's judged window is cut by the complete-book rule: the stored
    `book_coverage` is below 1 (and above 0: a book with a leg missing from the data altogether
    is not evaluable, nothing to compare), or the book names its legs and its data starts later
    than the first day the lab holds any data of its venue."""
    from sqlalchemy import func

    from qlab.registry.models import Idea, Spec, Trial, TrialStatus

    last = (session.query(Spec.idea_id, func.max(Trial.id).label("tid"))
            .join(Trial, Trial.spec_id == Spec.id)
            .filter(Trial.status.in_([TrialStatus.OK, TrialStatus.NOT_EVALUABLE]))
            .group_by(Spec.idea_id).subquery())
    out: list[str] = []
    for idea, trial in (session.query(Idea, Trial).join(last, last.c.idea_id == Idea.id)
                        .join(Trial, Trial.id == last.c.tid)
                        .filter(Idea.status.in_(OPEN_STATUSES)).order_by(Idea.id)):
        if idea.id.startswith("noise-"):
            continue
        coverage = (trial.metrics or {}).get("book_coverage")
        if isinstance(coverage, int | float) and 0.0 < coverage < 1.0:
            out.append(idea.id)
            continue
        if coverage == 1.0:  # a named book that is complete on the window it was given
            spec = spec_for_idea(session, idea.id)
            if spec is None or not spec.required_instruments:
                continue
            earliest = earliest_start(session, spec.data.source)
            if earliest is not None and earliest < spec.data.start:
                out.append(idea.id)
    return out


# --------------------------------------------------------------------------
# Words for the owner (idea page, CLI)


TITLE = "Два способа считать книгу, монеты которой появились в разное время"
ROW_TITLE = {AS_IS: "Как есть: судим с дня, когда есть все монеты сразу",
             AS_LISTED: "Монеты входят по мере появления"}
NOTE = "Ничего не решает: сначала замер, правило владелец делает по статистике."


def _pct(v: object) -> str:
    return "—" if not isinstance(v, int | float) or v != v else f"{v:+.1%}".replace("-", "−")


def _p_ru(p: object) -> str:
    if not isinstance(p, int | float):
        return "не определено"
    from qlab.earning_test import ALPHA, p_text

    return f"{'отличим' if p <= ALPHA else 'не отличим'} от нуля ({p_text(float(p))})"


def row_ru(row: dict[str, Any]) -> dict[str, Any]:
    """One row of the block in plain words."""
    if "error" in row:
        return {"error": row["error"]}
    share = row.get("incomplete_share")
    return {
        "window": f"{_ru_day(row['window_first'])} – {_ru_day(row['window_last'])}",
        "cut": _ru_day(row.get("cut_day")),
        "ann_return": _pct(row.get("ann_return")),
        "max_dd": _pct(row.get("max_dd")),
        "earning": _p_ru(row.get("p")),
        "incomplete": "—" if share is None else f"{share:.0%}",
        "days": row.get("judged_days"),
    }


def view(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The idea page's block, or None when the idea has no reading."""
    if not payload:
        return None
    return {
        "title": TITLE, "note": NOTE, "differs": payload.get("differs"),
        "affected": payload.get("affected"), "measured_at": payload.get("measured_at"),
        "data_end": payload.get("data_end"),
        "knowable": (payload.get("knowable") or {}).get("plain"),
        "rows": [{"id": f, "title": ROW_TITLE[f], **row_ru(payload["rows"][f])}
                 for f in FORMS if f in payload.get("rows", {})],
    }


def lines_ru(payload: dict[str, Any]) -> list[str]:
    out = [f"{TITLE}: {payload['idea_id']} (данные до {payload.get('data_end')})"]
    for f in FORMS:
        row = (payload.get("rows") or {}).get(f)
        if row is None:
            continue
        r = row_ru(row)
        if "error" in r:
            out.append(f"  {ROW_TITLE[f]}: не посчитано ({r['error']})")
            continue
        out.append(
            f"  {ROW_TITLE[f]}: окно {r['window']}, последняя треть с {r['cut']} "
            f"({r['days']} дн.), доход в год {r['ann_return']}, худшее падение {r['max_dd']}, "
            f"заработок {r['earning']}, книга неполная {r['incomplete']} последней трети")
    out.append(f"  {payload.get('differs')}")
    out.append(f"  {NOTE}")
    return out


__all__ = ["AS_IS", "AS_LISTED", "ENV", "FORMS", "Gate", "STORE", "TITLE", "candidates",
           "complete_share", "earliest_start", "first_leg_row", "gate_panel", "knowable_gate",
           "last_run_end", "lines_ru", "load", "load_all", "measure_idea", "measure_panel",
           "path_for", "row_ru", "save", "spec_for_idea", "store_dir", "view"]
