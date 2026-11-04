"""The money of every idea's run over the whole data period, day by day (docs/FORWARD_CHART.md).

The owner (2026-10-28): «когда стратегия прошла - добавить для неё график - как она
проработала на forward test. для каждого из этапов». A trial stores only numbers
(`trial.metrics`); this module keeps the judged part's daily money as a small file beside
them -- `data/forward_series/<trial_id>.json`, a few hundred points -- and reads it back
for the console and the morning report.

* **The window** is everything the run judges, from the window's first row to its last
  (`judged_rows`): the warm-up, the selection part and the forward part, as the rules of the
  run's version split it. The file's top-level figures are the judged part's (what the rules
  read, the forward test when there is one); `whole` holds the money over the whole window and
  each part's own totals (docs/IDEA_STANDARD.md, item 5; owner 2026-11-04).
* **Money** is the strategy's: the run's `capital_return` (a share of the capital the
  strategy started with, never of the account balance or of a book balance that has grown
  with money set aside, CLAUDE.md money rule, docs/CAPITAL_BASE.md) from zero: compounded
  for a strategy that reinvests its profit, summed for one that does not. The old stand
  figure (a share of the grown book, compounded) is kept beside as `stand_total_return`.
* **Two sources, one result.** `evaluate_spec` stores the series of every run, whatever its
  route (a refusal too), from the very backtests the verdict rests on. For runs made before
  this existed, `replay` re-runs the stored spec on the stored snapshot -- CPU only, no
  network, no trial, no verdict -- through the same row-cutting (`judged_rows`), and says
  whether the replayed total equals the trial's recorded one.
* Display only: no rule and no agent reads these numbers; nothing here decides anything.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

STORE_DIR = Path("data/forward_series")
SPEC_ROOT = Path("specs")
FORMAT = 2
"""Format 2 adds `whole`: the money over the whole judged window with its parts. Format 1 (the
forward part only) is still read; the console says so for such a file."""
READABLE_FORMATS = frozenset({1, 2})
WHOLE_REVISION = 2
"""Revision of `whole`: 2 draws the selection part of a run that chooses among declared values as
the chosen value's own money (`qlab.chosen_curve`) instead of a flat warm-up. A file of an older
revision with a warm-up part is built again by `rebuild`."""
PASSED_ROUTE = "paper"
"""The route of a run that passed every rule (`TrialRoute.PAPER`): «прошла все правила,
ждёт решения владельца»."""
MATCH_TOLERANCE = 1e-6
"""How far the replayed total may sit from the trial's recorded one (a fraction) and still
count as the same run."""
VENUE_NAMES = {"binance": "Binance", "hyperliquid": "Hyperliquid"}
CLOSED_STATUSES = frozenset({"retired", "rejected", "decayed"})
NO_CHART_STATUSES = frozenset({"retired"})
"""Ideas the owner put away: nothing is drawn for them. A rejected or decayed idea is drawn, so
that the reason for the refusal can be seen (docs/IDEA_STANDARD.md, item 5)."""
PART_ORDER = ("before", "warmup", "selection", "forward")
"""Parts of the window, as the split names them, in time order."""


def store_dir() -> Path:
    """`QLAB_FORWARD_SERIES` if set (like `QLAB_DB`, `QLAB_CARDS`), else `data/forward_series`."""
    return Path(os.environ.get("QLAB_FORWARD_SERIES") or STORE_DIR)


def series_path(trial_id: int, store: Path | None = None) -> Path:
    return (store if store is not None else store_dir()) / f"{trial_id}.json"


def venue_name(source: str | None) -> str | None:
    return None if not source else VENUE_NAMES.get(source, source)


# --------------------------------------------------------------------------
# The series itself


def _drawdown(growth: pd.Series) -> float:
    peak = growth.cummax().clip(lower=1.0)
    return float((growth / peak - 1.0).min())


def _annual(total: float, days: int, reinvested: bool = True) -> float | None:
    if days <= 0:
        return None
    if not reinvested:
        return total * 365.25 / days
    return -1.0 if 1.0 + total <= 0 else (1.0 + total) ** (365.25 / days) - 1.0


def _by_day(returns: pd.Series, days: pd.DatetimeIndex, reinvested: bool) -> pd.Series:
    """Bars of one UTC day as one day's return: compounded, or summed for a book whose profit
    is not reinvested."""
    values = returns.astype(float)
    if reinvested:
        return (1.0 + values).groupby(days).prod() - 1.0
    return values.groupby(days).sum()


def _curve(daily: pd.Series, reinvested: bool) -> pd.Series:
    """The cumulative fraction from zero, day by day."""
    return (1.0 + daily).cumprod() - 1.0 if reinvested else daily.cumsum()


def _max_dd(curve: pd.Series, reinvested: bool) -> float:
    if reinvested:
        return _drawdown(1.0 + curve)
    return float((curve - curve.cummax().clip(lower=0.0)).min())


def _part_stats(daily: pd.Series, hold_daily: pd.Series | None, reinvested: bool) -> dict[str, Any]:
    """Totals of one stretch of days, counted from zero at its own first day."""
    curve = _curve(daily, reinvested)
    n_days = int((daily.index[-1] - daily.index[0]).days) + 1
    total = float(curve.iloc[-1])
    hold_total = None
    if hold_daily is not None:
        hold_total = float(_curve(hold_daily, reinvested).iloc[-1])
    return {
        "start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat(),
        "days": n_days, "total_return": total, "ann_return": _annual(total, n_days, reinvested),
        "max_dd": _max_dd(curve, reinvested), "hold_total": hold_total,
        "hold_ann": None if hold_total is None else _annual(hold_total, n_days, reinvested)}


def part_bounds(index: pd.DatetimeIndex, rows: Any, split_mode: bool
                ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """The parts of a run's window as `(first bar, last bar)` times, read from the very rows the
    verdict cuts (`qlab.pipeline.evaluate.judged_rows`), so that the chart follows whatever split
    the rules give. A run judged without a split has no parts."""
    if not split_mode:
        return {}
    found = {"before": rows.split.before, "warmup": rows.warmup,
             "selection": rows.split.selection, "forward": rows.split.forward}
    out = {}
    for name in PART_ORDER:
        span = found[name]
        if span is not None:
            out[name] = (index[span[0]], index[span[1]])
    return out


def build_whole(*, net_return: pd.Series, hold_return: pd.Series | None,
                hold_coins: list[str] | None, bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                capital_return: pd.Series | None = None, reinvested: bool = True,
                chosen: dict[str, Any] | None = None) -> dict[str, Any]:
    """The money over the whole judged window, day by day, with its parts.

    The same reading as `build_payload` (the strategy's own capital, compounded or summed per
    `reinvested`), but over every bar from the window's first row to its last, from zero on the
    eve of the first day. `parts` lists each part the split gives (`before`, `warmup`,
    `selection`, `forward`, in time order) with its totals counted from zero at its own first
    day; `split_date` is the first day of the forward part (null when the run has none). The
    day of a part is that of its first bar.

    `chosen` (`qlab.chosen_curve.ChosenCandidate.info`) says the series holds, on the `selection`
    part, the money of the value the run's wrapper chose by looking at exactly that part: the part
    is marked `chosen_in_sample` with the value's name, and the figures of the part are kept in
    `selection_numbers` (informational: no rule reads them)."""
    if len(net_return) == 0:
        raise ValueError("no returns to draw")
    drawn = net_return if capital_return is None else capital_return
    days = pd.DatetimeIndex(drawn.index).tz_convert("UTC").normalize()
    daily = _by_day(drawn, days, reinvested)
    hold_daily = None
    if hold_return is not None:
        hold_daily = _by_day(hold_return.reindex(drawn.index).fillna(0.0), days, reinvested)
    curve = _curve(daily, reinvested)
    hold_curve = None if hold_daily is None else _curve(hold_daily, reinvested)
    first_day = curve.index[0]
    points: list[list[Any]] = [[(first_day - pd.Timedelta(days=1)).date().isoformat(), 0.0,
                                None if hold_curve is None else 0.0]]
    for day in curve.index:
        points.append([day.date().isoformat(), float(curve[day]),
                       None if hold_curve is None else float(hold_curve[day])])

    starts = {name: pd.Timestamp(lo).tz_convert("UTC").normalize()
              for name, (lo, _hi) in bounds.items()}
    label = pd.Series([None] * len(daily), index=daily.index, dtype=object)
    for name in sorted(starts, key=lambda n: starts[n]):
        label.loc[daily.index >= starts[name]] = name
    parts = []
    for name in PART_ORDER:
        mask = (label == name).to_numpy()
        if name not in bounds or not mask.any():
            continue
        entry = {"part": name, **_part_stats(
            daily[mask], None if hold_daily is None else hold_daily[mask], reinvested)}
        if chosen is not None and name == "selection":
            entry.update(chosen_in_sample=True, chosen=chosen["name"])
        parts.append(entry)
    whole = _part_stats(daily, hold_daily, reinvested)
    return {
        **whole,
        "revision": WHOLE_REVISION,
        "selection_numbers": None if chosen is None else chosen["numbers"],
        "points": points,
        "hold_coins": hold_coins,
        "parts": parts,
        "split_date": next((p["start"] for p in parts if p["part"] == "forward"), None),
    }


def build_payload(*, trial_id: int, idea_id: str, venue: str | None, interval: str | None,
                  net_return: pd.Series, hold_return: pd.Series | None,
                  hold_coins: list[str] | None = None, source: str = "evaluate",
                  recorded_total: float | None = None,
                  capital_return: pd.Series | None = None, reinvested: bool = True,
                  recorded_capital_total: float | None = None,
                  whole: dict[str, Any] | None = None,
                  judged_on_forward: bool = True) -> dict[str, Any]:
    """The judged part's money by day, as a plain mapping that `save` writes.

    The line is drawn from `capital_return` (the stand's `net_return` when none is given):
    each bar's share of the capital the strategy started with, indexed by decision time. Bars
    of one UTC day are compounded into one point, or summed when `reinvested` is False (a
    strategy that does not put its profit back into positions, docs/CAPITAL_BASE.md); the
    dotted "just hold the coins" line is read the same way. `points` are `[date, strategy,
    hold]`, cumulative fractions from zero (0.05 is +5% of the strategy's capital), the first
    point being the zero line on the eve of the first day. `hold` is null when the book holds
    nothing. `stand_total_return` is the old reading of the same bars (compounded shares of
    the book's grown equity). `recorded_total` is the trial's own `total_return_net` for the
    same part and `recorded_capital_total` its `cap_total_return`; `matches_trial` says whether
    this series reproduces the capital one when the trial recorded it, else the stand one.

    `whole` (`build_whole`) is the money over the whole window with its parts; the fields above
    stay the judged part's, which the rules read. `judged_on_forward` says whether that part is
    a forward test (a run judged on its selection period has none)."""
    if len(net_return) == 0:
        raise ValueError("no returns to draw")
    drawn = net_return if capital_return is None else capital_return
    days = pd.DatetimeIndex(drawn.index).tz_convert("UTC").normalize()
    curve = _curve(_by_day(drawn, days, reinvested), reinvested)
    stand_total = float((1.0 + net_return.astype(float)).prod()) - 1.0
    hold_curve = None
    if hold_return is not None:
        hold_curve = _curve(
            _by_day(hold_return.reindex(drawn.index).fillna(0.0), days, reinvested), reinvested)

    first_day = curve.index[0]
    points: list[list[Any]] = [[(first_day - pd.Timedelta(days=1)).date().isoformat(), 0.0,
                                None if hold_curve is None else 0.0]]
    for day in curve.index:
        points.append([day.date().isoformat(), float(curve[day]),
                       None if hold_curve is None else float(hold_curve[day])])
    n_days = int((curve.index[-1] - first_day).days) + 1
    total = float(curve.iloc[-1])
    hold_total = None if hold_curve is None else float(hold_curve.iloc[-1])
    max_dd = _max_dd(curve, reinvested)
    if recorded_capital_total is not None:
        matches: bool | None = abs(total - recorded_capital_total) <= MATCH_TOLERANCE
        recorded = recorded_capital_total
    elif recorded_total is not None:
        matches = abs(stand_total - recorded_total) <= MATCH_TOLERANCE
        recorded = recorded_total
    else:
        matches, recorded = None, None
    return {
        "format": FORMAT,
        "trial_id": trial_id,
        "idea_id": idea_id,
        "venue": venue,
        "interval": interval,
        "start": first_day.date().isoformat(),
        "end": curve.index[-1].date().isoformat(),
        "days": n_days,
        "base": "strategy_capital" if capital_return is not None else "stand",
        "reinvested": reinvested,
        "total_return": total,
        "ann_return": _annual(total, n_days, reinvested),
        "max_dd": max_dd,
        "stand_total_return": stand_total,
        "hold_total": hold_total,
        "hold_ann": None if hold_total is None else _annual(hold_total, n_days, reinvested),
        "hold_coins": hold_coins,
        "points": points,
        "built_by": source,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "recorded_total": recorded,
        "matches_trial": matches,
        "judged_on_forward": judged_on_forward,
        "whole": whole,
    }


def save(payload: dict[str, Any], store: Path | None = None) -> Path:
    path = series_path(int(payload["trial_id"]), store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    return path


def load(trial_id: int, store: Path | None = None) -> dict[str, Any] | None:
    """The stored series, or None when there is none (or it cannot be read)."""
    path = series_path(trial_id, store)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = isinstance(payload, dict) and payload.get("format") in READABLE_FORMATS
    return payload if ok else None


def has_whole(payload: dict[str, Any] | None) -> bool:
    """Whether a stored series covers the whole window (format 2); a format 1 file is the
    forward part only."""
    return bool(payload and payload.get("whole"))


def is_current(payload: dict[str, Any] | None) -> bool:
    """Whether a stored series is built the way `rebuild` builds it now: it covers the whole
    window, and a warm-up it shows is not one that a later revision draws as the chosen value's
    own money (`WHOLE_REVISION`)."""
    if not has_whole(payload):
        return False
    whole = payload["whole"]  # type: ignore[index]
    if int(whole.get("revision") or 1) >= WHOLE_REVISION:
        return True
    return not any(p.get("part") == "warmup" for p in whole.get("parts") or [])


def store_from_run(*, trial_id: int, spec: Any, result: Any, part_panel: Any,
                   part_weights: pd.DataFrame, recorded_total: float | None,
                   recorded_capital_total: float | None = None, on_capital: bool = True,
                   whole_result: Any = None, whole_panel: Any = None,
                   whole_weights: pd.DataFrame | None = None,
                   bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
                   judged_on_forward: bool = True, chosen: Any = None,
                   store: Path | None = None) -> Path:
    """Write the series of a run `evaluate_spec` has just made, from the very backtest
    results the verdict rests on: the judged part's (`result`, what the rules read) and, when
    given, the whole window's (`whole_*`) with the parts' `bounds`. `chosen`
    (`qlab.chosen_curve.ChosenCandidate`) puts the chosen value's own money on the selection
    part of the whole window; the judged part stays the run's."""
    from qlab.harness.metrics import held_coins, hold_returns

    capital_of = (lambda r: r.capital_return) if on_capital else (lambda r: None)
    reinvested = result.capital_reinvested if on_capital else True
    whole = None
    if whole_result is not None:
        net, cap, weights = whole_result.net_return, capital_of(whole_result), whole_weights
        if chosen is not None:
            net, stitched_cap, weights = chosen.stitch(whole_result, whole_weights)
            cap = stitched_cap if on_capital else None
            bounds = chosen.chart_bounds(bounds or {})
        whole = build_whole(
            net_return=net, hold_return=hold_returns(whole_panel.prices, weights, net.index),
            hold_coins=held_coins(weights), bounds=bounds or {}, capital_return=cap,
            reinvested=whole_result.capital_reinvested if on_capital else True,
            chosen=None if chosen is None else chosen.info())
    payload = build_payload(
        trial_id=trial_id, idea_id=spec.idea_id, venue=spec.data.source,
        interval=spec.data.interval, net_return=result.net_return,
        hold_return=hold_returns(part_panel.prices, part_weights, result.net_return.index),
        hold_coins=held_coins(part_weights), source="evaluate", recorded_total=recorded_total,
        capital_return=capital_of(result), reinvested=reinvested,
        recorded_capital_total=recorded_capital_total if on_capital else None,
        whole=whole, judged_on_forward=judged_on_forward)
    return save(payload, store)


# --------------------------------------------------------------------------
# Replaying a stored run


@dataclass(frozen=True)
class Replay:
    payload: dict[str, Any] | None
    reason: str | None = None
    """Why there is no payload, in plain Russian (the owner may read it in the console)."""


def spec_file_for(session: Session, spec_id: int, root: Path | None = None) -> Path | None:
    """The spec file whose content is exactly the stored `spec` row: the run is replayed from
    what it was run with, not from a file edited since (the data window aside). Of the files
    that name the idea and match, one stating `params_fixed_at` wins, then the last by name
    (as in `qlab.pool_allocation.spec_path_for_idea`)."""
    from qlab.pipeline.evaluate import _same_content, _spec_row_content
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Spec

    row = session.get(Spec, spec_id)
    if row is None:
        return None
    base = root if root is not None else SPEC_ROOT
    pattern = re.compile(rf"^idea_id:\s*{re.escape(row.idea_id)}\s*$", re.M)
    matching: list[tuple[bool, str, Path]] = []
    for path in sorted(base.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if not pattern.search(text):
            continue
        try:
            spec = load_spec(path)
        except Exception:  # noqa: BLE001 - a spec that no longer loads is simply not this one
            continue
        data_requirements, costs_model = _spec_row_content(spec)
        # The night moves a spec's data window to yesterday when it runs it and writes that into
        # the stored row, not into the file: the window is the one thing allowed to differ.
        for window in ("start", "end"):
            data_requirements[window] = row.data_requirements.get(window)
        if _same_content(row, spec, data_requirements, costs_model):
            matching.append((spec.params_fixed_at is not None, str(path), path))
    return max(matching)[2] if matching else None


def stored_spec(session: Session, spec_id: int, root: Path | None = None) -> Any:
    """The `StrategySpec` the stored `spec` row was run as, or None.

    A spec file whose content is the row (`spec_file_for`) is that spec. An idea that declares
    lists of values (`choice`, docs/ONE_CHECK.md) is run as its check form -- a `Retune` that
    chooses once at the stand's cut -- which no file holds: the file is rebuilt into the form by
    the very code the night uses (`qlab.one_check.check_spec`) and compared with the row the same
    way, the data window aside. The row keeps `choose_at: split`, and so does the form; the day is
    put in only when a run starts (`with_split_day`)."""
    from qlab import one_check
    from qlab.pipeline.evaluate import _same_content, _spec_row_content
    from qlab.pipeline.spec import load_spec
    from qlab.registry.models import Spec

    path = spec_file_for(session, spec_id, root)
    if path is not None:
        return load_spec(path)
    row = session.get(Spec, spec_id)
    if row is None:
        return None
    base = root if root is not None else SPEC_ROOT
    pattern = re.compile(rf"^idea_id:\s*{re.escape(row.idea_id)}\s*$", re.M)
    for file in sorted(base.rglob("*.yaml")):
        if not pattern.search(file.read_text(encoding="utf-8")):
            continue
        try:
            spec = load_spec(file)
            if not one_check.declares_lists(spec):
                continue
            form = one_check.check_spec(spec, spec_path=file).spec
        except Exception:  # noqa: BLE001 - a file that cannot be made into its check is not this
            continue
        if form is None:
            continue
        data_requirements, costs_model = _spec_row_content(form)
        for window in ("start", "end"):
            data_requirements[window] = row.data_requirements.get(window)
        if _same_content(row, form, data_requirements, costs_model):
            return form
    return None


def _rules_version(session: Session, trial_id: int) -> str | None:
    from qlab.registry.models import Verdict

    row = (session.query(Verdict.rules_version).filter(Verdict.trial_id == trial_id)
           .order_by(Verdict.id).first())
    return None if row is None else row[0]


@dataclass(frozen=True)
class ReplayedRun:
    """A stored run backtested again: the pieces a reading of its judged part needs."""

    spec: Any
    trial: Any
    result: Any
    part_panel: Any
    part_weights: pd.DataFrame
    capital: Any
    """`qlab.harness.strategy.CapitalSetup`: the base (None: 1) and the declared reinvest choice."""
    rows: Any = None
    """`JudgedRows` of the run (set for `part="whole"`)."""
    split_mode: bool = False
    whole: tuple[Any, Any, pd.DataFrame] | None = None
    """The whole window's `(result, panel, weights)` (set for `part="whole"`)."""
    index: pd.DatetimeIndex | None = None
    chosen: Any = None
    """`qlab.chosen_curve.ChosenCandidate` of a run that chooses among declared values (set for
    `part="whole"`): the selection part's money, display only."""


def replay_run(session: Session, trial_id: int, *, spec_root: Path | None = None,
               part: str = "forward") -> tuple[ReplayedRun | None, str | None]:
    """Re-run the stored spec of trial `trial_id` on its stored snapshot over one part of the
    window and return the pieces, or `(None, reason)` -- the reason in plain Russian. Writes
    nothing.

    `part` is `"forward"` (the forward test), `"judged"` (the part the verdict was read on: the
    forward test when the run split one off, else the window after the warm-up; the year
    view and the protection reading are read on it) or `"whole"` (the judged part as for
    `"judged"` and, besides, the whole window with the split's parts; the chart draws it). The
    cut rows come from `qlab.pipeline.evaluate.judged_rows`, the rules' version from the run's
    own verdicts."""
    from qlab.data.snapshot import load_snapshot
    from qlab.harness.costs import CostModel
    from qlab.harness.run import run_backtest
    from qlab.harness.strategy import capital_setup, own_costs
    from qlab.pipeline.evaluate import (
        _slice_rows,
        complete_book_window,
        cut_truncated_history,
        judged_rows,
        resolve_strategy,
        with_split_day,
    )
    from qlab.registry.models import Spec, Trial
    from qlab.rules import loader

    trial = session.get(Trial, trial_id)
    if trial is None:
        return None, f"прогона {trial_id} нет в реестре"
    if trial.metrics is None or trial.snapshot_id is None:
        return None, "у прогона нет чисел или снимка данных: считать нечего"
    if part == "forward" and trial.metrics.get("judged_on_forward") != 1.0:
        return None, "у этого прогона нет проверки вперёд: судили период подбора"
    spec_row = session.get(Spec, trial.spec_id)
    spec = stored_spec(session, trial.spec_id, spec_root)
    if spec_row is None or spec is None:
        return None, ("файл спека, с которым шёл прогон, не найден или с тех пор изменён: "
                      "ряд честно не восстановить")
    version = _rules_version(session, trial_id)
    try:
        ruleset = loader.load(version) if version else loader.load_latest()
    except Exception:  # noqa: BLE001 - a retired ruleset file: the latest cuts rows the same way
        ruleset = loader.load_latest()
    try:
        panel = load_snapshot(trial.snapshot_id, session=session)
    except (ValueError, OSError) as exc:
        return None, f"нет данных, на которых шёл прогон: {exc}"
    if panel.meta.get("history_truncated"):
        panel = cut_truncated_history(panel)

    coverage = (complete_book_window(panel, spec.required_instruments)
                if spec.required_instruments is not None else None)
    rows = judged_rows(spec, panel, coverage, ruleset)
    if part == "forward":
        if rows.split.forward is None:
            return None, "в окне прогона нет проверки вперёд"
        lo, hi = rows.split.forward
    else:
        lo, hi = rows.judged
    try:
        spec = with_split_day(spec, panel.prices.index, rows)  # a choice made at the cut
        strategy = resolve_strategy(spec.code_ref)
        weights = strategy.target_weights(panel, spec.params)
        costs = CostModel(taker_fee_bps=spec.costs.taker_fee_bps,
                          slippage_bps=spec.costs.slippage_bps)
        extra_cost = own_costs(strategy, panel, spec.params)
        capital = capital_setup(strategy, panel, spec.params)

        def run_rows(a: int, b: int) -> tuple[Any, Any, pd.DataFrame]:
            sliced = _slice_rows(panel, a, b)
            part_weights_ = weights.iloc[a:b + 1]
            return (run_backtest(sliced, part_weights_, costs, sliced.funding,
                                 extra_cost=extra_cost, **capital.kwargs),
                    sliced, part_weights_)

        result, part_panel, part_weights = run_rows(lo, hi)
        whole = None
        chosen = None
        if part == "whole":
            whole = ((result, part_panel, part_weights) if (lo, hi) == (rows.first, rows.last)
                     else run_rows(rows.first, rows.last))
            if (ruleset.splits_fit_forward and rows.warmup is not None
                    and rows.split.forward is not None):
                chosen = _chosen_of(strategy, panel, spec.params, rows, costs)
    except Exception as exc:  # noqa: BLE001 - strategy code is arbitrary
        return None, f"прогон не удалось повторить: {type(exc).__name__}: {exc}"
    return ReplayedRun(spec, trial, result, part_panel, part_weights, capital, rows=rows,
                       split_mode=bool(ruleset.splits_fit_forward), whole=whole,
                       index=panel.prices.index, chosen=chosen), None


def _chosen_of(strategy: Any, panel: Any, params: Any, rows: Any, costs: Any) -> Any:
    """The chosen value's own money over the selection part of a run whose wrapper chooses among
    declared values, or None (any other run; a run whose value cannot be replayed keeps the flat
    warm-up: this is a picture, it must not fail the chart)."""
    from qlab.chosen_curve import chosen_candidate

    try:
        return chosen_candidate(strategy, panel, params, first=rows.first,
                                cut=rows.split.forward[0], costs=costs)
    except Exception:  # noqa: BLE001
        return None


def replay(session: Session, trial_id: int, *, spec_root: Path | None = None) -> Replay:
    """Re-run the stored spec of trial `trial_id` on its stored snapshot and return the series:
    the judged part's (what the rules read) and the whole window's with its parts. Writes
    nothing: no trial, no verdict, no status; the caller `save`s it.

    When the replayed total differs from the one the trial recorded (the strategy's code
    changed since), the payload says so (`matches_trial` false)."""
    from qlab.harness.metrics import held_coins, hold_returns

    run, reason = replay_run(session, trial_id, spec_root=spec_root, part="whole")
    if run is None:
        return Replay(None, reason)
    spec, trial, result = run.spec, run.trial, run.result
    recorded = trial.metrics.get("total_return_net")
    recorded_capital = trial.metrics.get("cap_total_return")
    drawn_on_capital = run.capital.refused is None
    whole_result, whole_panel, whole_weights = run.whole
    net, cap, drawn_weights = whole_result.net_return, whole_result.capital_return, whole_weights
    bounds = part_bounds(run.index, run.rows, run.split_mode)
    if run.chosen is not None:  # the selection part is the chosen value's own money
        net, cap, drawn_weights = run.chosen.stitch(whole_result, whole_weights)
        bounds = run.chosen.chart_bounds(bounds)
    whole = build_whole(
        net_return=net,
        hold_return=hold_returns(whole_panel.prices, drawn_weights, net.index),
        hold_coins=held_coins(drawn_weights), bounds=bounds,
        capital_return=cap if drawn_on_capital else None,
        reinvested=whole_result.capital_reinvested if drawn_on_capital else True,
        chosen=None if run.chosen is None else run.chosen.info())
    payload = build_payload(
        trial_id=trial_id, idea_id=spec.idea_id, venue=spec.data.source,
        interval=spec.data.interval, net_return=result.net_return,
        hold_return=hold_returns(run.part_panel.prices, run.part_weights, result.net_return.index),
        hold_coins=held_coins(run.part_weights), source="rebuild",
        recorded_total=recorded if isinstance(recorded, int | float) else None,
        capital_return=result.capital_return if drawn_on_capital else None,
        reinvested=result.capital_reinvested if drawn_on_capital else True,
        recorded_capital_total=(recorded_capital if drawn_on_capital
                                and isinstance(recorded_capital, int | float) else None),
        whole=whole, judged_on_forward=run.rows.split.forward is not None and run.split_mode)
    return Replay(payload)


# --------------------------------------------------------------------------
# Which runs the owner looks at


@dataclass(frozen=True)
class StageRun:
    """One stage of an idea: whose run it is and the latest routed trial."""

    number: int
    idea_id: str
    trial_id: int | None
    route: str | None
    source: str | None
    """The data venue the run's spec names (`binance`, `hyperliquid`)."""
    started_at: datetime | None


def stage_runs(session: Session, idea_id: str) -> tuple[bool, list[StageRun]] | None:
    """The stages to draw for `idea_id`, in order, and whether the idea has two. None when
    the idea does not exist. An idea with a second stage (or a second stage itself) lists
    stage 1 (the parent) and stage 2 (the child, `<parent>-hl`); any other idea has one
    stage, the run of its own."""
    from qlab.registry.models import Idea
    from qlab.registry.stages import is_stage2_child, latest_routed, stage2_child

    idea = session.get(Idea, idea_id)
    if idea is None:
        return None
    if is_stage2_child(idea):
        parent_id, child_id = idea.parent_id, idea.id
    else:
        child = stage2_child(session, idea.id)
        parent_id, child_id = idea.id, None if child is None else child.id

    def stage(number: int, stage_idea_id: str) -> StageRun:
        found = latest_routed(session, stage_idea_id)
        if found is None:
            return StageRun(number, stage_idea_id, None, None, None, None)
        trial, spec = found
        return StageRun(number, stage_idea_id, trial.id,
                        None if trial.route is None else str(trial.route.value),
                        (spec.data_requirements or {}).get("source"), trial.started_at)

    stages = [stage(1, parent_id)]
    if child_id is not None:
        stages.append(stage(2, child_id))
    return child_id is not None, stages


NO_SECOND_STAGE_YET_RU = ("На Hyperliquid ещё не прогонялась: второй этап этой идеи ночь пока не "
                          "записала.")
NO_SECOND_STAGE_NEEDED_RU = "Вторая площадка не нужна: прогон идеи и так на Hyperliquid."
NO_RUN_RU = "У этого этапа ещё нет прогона: идея ещё не прогонялась."
NOT_STORED_RU = ("Ряд по дням для этого прогона ещё не сохранён. Он появится в ближайший "
                 "ночной прогон; сейчас построить: `uv run qlab forward-series rebuild --all`.")
FORWARD_ONLY_RU = ("Сохранён только отрезок проверки вперёд (файл построен до того, как ряд стали "
                   "хранить за весь период); весь период появится после пересчёта.")


def forward_view(session: Session, idea_id: str, *, store: Path | None = None
                 ) -> dict[str, Any] | None:
    """What the console draws on an idea's page. None for an unknown idea.

    Every idea with a run gets its stages' series, whatever the outcome: `passed` only says
    whether the idea passed every rule (information, it does not gate the chart). Each stage
    lists `series` (the stored payload; its `whole` is the money over the whole window with the
    split's parts) or, when none is stored, a `message` in plain Russian; `window` is `whole`,
    `forward_only` (a format 1 file) or null. Nothing is drawn for an idea the owner put away
    (`NO_CHART_STATUSES`) or for a calibration book: `stages` is empty and `reason` says why.
    `second_stage_message` explains the missing second stage of a one-stage idea."""
    from qlab.registry.models import Idea

    found = stage_runs(session, idea_id)
    if found is None:
        return None
    two_stage, stages = found
    idea = session.get(Idea, idea_id)
    status = idea.status.value
    own = next((s for s in stages if s.idea_id == idea_id), stages[0])
    passed = status == "validated" or own.route == PASSED_ROUTE
    if status in NO_CHART_STATUSES or idea_id.startswith("noise-"):
        why = ("Идея убрана владельцем: график не рисуется." if status in NO_CHART_STATUSES
               else "Это книга для калибровки шума: графика у неё нет.")
        return {"idea_id": idea_id, "passed": False, "two_stage": two_stage, "stages": [],
                "reason": why, "second_stage_message": None}
    out = []
    for stage in stages:
        entry: dict[str, Any] = {
            "stage": stage.number, "idea_id": stage.idea_id, "trial_id": stage.trial_id,
            "route": stage.route, "venue": venue_name(stage.source), "series": None,
            "window": None, "message": None}
        if stage.trial_id is None:
            entry["message"] = NO_RUN_RU
        else:
            payload = load(stage.trial_id, store)
            if payload is None:
                entry["message"] = NOT_STORED_RU
            else:
                entry["series"] = payload
                entry["window"] = "whole" if has_whole(payload) else "forward_only"
                if entry["window"] == "forward_only":
                    entry["message"] = FORWARD_ONLY_RU
        out.append(entry)
    second = None
    if not two_stage:
        second = (NO_SECOND_STAGE_NEEDED_RU if own.source == "hyperliquid"
                  else NO_SECOND_STAGE_YET_RU)
    return {"idea_id": idea_id, "passed": passed, "two_stage": two_stage, "stages": out,
            "reason": None, "second_stage_message": second}


def passed_ideas(session: Session) -> list[str]:
    """Every open idea that passed: status `validated`, or a latest routed run that passed every
    rule (`PASSED_ROUTE`) -- parents and second stages alike; calibration noise and closed
    ideas are left out."""
    from qlab.registry.models import Idea
    from qlab.registry.stages import latest_routed

    found = []
    for idea in session.query(Idea).order_by(Idea.id):
        if idea.status.value in CLOSED_STATUSES or idea.id.startswith("noise-"):
            continue
        if idea.status.value == "validated":
            found.append(idea.id)
            continue
        latest = latest_routed(session, idea.id)
        if latest is not None and latest[0].route is not None \
                and latest[0].route.value == PASSED_ROUTE:
            found.append(idea.id)
    return found


def chart_ideas(session: Session) -> list[str]:
    """Every idea whose runs get a chart: not put away by the owner, not a calibration book,
    parents and second stages alike."""
    from qlab.registry.models import Idea

    return [idea.id for idea in session.query(Idea).order_by(Idea.id)
            if idea.status.value not in NO_CHART_STATUSES and not idea.id.startswith("noise-")]


def rebuild(session: Session, idea_ids: list[str] | None = None, *, force: bool = False,
            store: Path | None = None, spec_root: Path | None = None, every: bool = False,
            on_done: Any = None) -> list[tuple[str, int, str]]:
    """Build the missing series of the ideas' stages by `replay`: the passed ideas by default,
    `chart_ideas` when `every`. Returns `(idea_id, trial_id, outcome)` per stage run looked at;
    a series that covers the whole window is kept unless `force` (a format 1 file, the forward
    part only, and a file whose warm-up a later revision draws as the chosen value's own money
    are built again). `on_done(idea_id, trial_id, outcome, seconds)` is called after
    each run looked at. Writes only the series files."""
    from time import perf_counter

    done: list[tuple[str, int, str]] = []
    seen: set[int] = set()
    if idea_ids is not None:
        wanted = idea_ids
    else:
        wanted = chart_ideas(session) if every else passed_ideas(session)

    def note(idea_id: str, trial_id: int, outcome: str, started: float) -> None:
        done.append((idea_id, trial_id, outcome))
        if on_done is not None:
            on_done(idea_id, trial_id, outcome, perf_counter() - started)

    for idea_id in wanted:
        found = stage_runs(session, idea_id)
        if found is None:
            done.append((idea_id, 0, "нет такой идеи"))
            continue
        for stage in found[1]:
            if stage.trial_id is None or stage.trial_id in seen:
                continue
            seen.add(stage.trial_id)
            started = perf_counter()
            if not force and is_current(load(stage.trial_id, store)):
                note(stage.idea_id, stage.trial_id, "уже есть", started)
                continue
            result = replay(session, stage.trial_id, spec_root=spec_root)
            if result.payload is None:
                note(stage.idea_id, stage.trial_id, f"нет: {result.reason}", started)
                continue
            save(result.payload, store)
            same = result.payload["matches_trial"]
            note(stage.idea_id, stage.trial_id,
                 "построен" + ("" if same in (True, None)
                                else ", но расходится с записью прогона"), started)
    return done


def passed_on(session: Session, since: datetime, until: datetime) -> list[dict[str, Any]]:
    """The passed ideas whose passing run (route `paper`) was written between `since` and
    `until`, one entry per strategy (a second stage is folded under its parent), for the
    morning report: `{idea_id, stages: [{stage, idea_id, trial_id, venue, has_series}]}`."""
    from qlab.registry.models import Idea, Spec, Trial, TrialRoute
    from qlab.registry.stages import is_stage2_child

    rows = (session.query(Spec.idea_id).join(Trial, Trial.spec_id == Spec.id)
            .filter(Trial.route == TrialRoute.PAPER, Trial.started_at >= since,
                    Trial.started_at < until).distinct().all())
    roots: dict[str, None] = {}
    for (stage_idea_id,) in rows:
        idea = session.get(Idea, stage_idea_id)
        if idea is None or idea.status.value in CLOSED_STATUSES or stage_idea_id.startswith(
                "noise-"):
            continue
        roots[idea.parent_id if is_stage2_child(idea) else idea.id] = None
    out = []
    for root in sorted(roots):
        found = stage_runs(session, root)
        if found is None:
            continue
        out.append({"idea_id": root, "stages": [
            {"stage": s.number, "idea_id": s.idea_id, "trial_id": s.trial_id,
             "venue": venue_name(s.source), "route": s.route,
             "has_series": s.trial_id is not None and load(s.trial_id) is not None}
            for s in found[1]]})
    return out


__all__ = ["PASSED_ROUTE", "STORE_DIR", "Replay", "StageRun", "build_payload", "build_whole",
           "chart_ideas", "forward_view", "has_whole", "load", "part_bounds", "passed_ideas",
           "passed_on", "rebuild", "replay", "save", "series_path",
           "spec_file_for", "stage_runs", "store_from_run", "stored_spec"]
